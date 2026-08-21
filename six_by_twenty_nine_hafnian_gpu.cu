// Exact CUDA residual-hafnian solver for T_4(6,29).

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#include "hafnian_gpu_core.cuh"
#include "sha256.hpp"
#include "six_by_twenty_nine_catalog.hpp"

namespace {

using Clock=std::chrono::steady_clock;
using six_by_twenty_nine::Catalog;
using six_by_twenty_nine::Query;
constexpr const char* ALGORITHM="glynn-trace-hessenberg-residual-cuda-v1";

bool is_prime_u32(uint32_t n) {
    if(n<2)return false;
    for(uint32_t p:{2U,3U,5U,7U,11U,13U,17U,19U,23U,29U,31U,37U}) {
        if(n==p)return true;
        if(n%p==0)return false;
    }
    uint32_t d=n-1,s=0;
    while(!(d&1)){d>>=1;++s;}
    auto mul=[n](uint64_t a,uint64_t b){return uint64_t((unsigned __int128)a*b%n);};
    auto power=[&](uint64_t a,uint64_t exponent){
        uint64_t result=1;
        while(exponent){if(exponent&1)result=mul(result,a);a=mul(a,a);exponent>>=1;}
        return result;
    };
    for(uint32_t base:{2U,3U,5U,7U,11U}) {
        if(base>=n)continue;
        uint64_t value=power(base,d);
        if(value==1||value==n-1)continue;
        bool composite=true;
        for(uint32_t r=1;r<s;++r){value=mul(value,value);if(value==n-1){composite=false;break;}}
        if(composite)return false;
    }
    return true;
}

struct Options {
    uint32_t prime=2147483647U;
    uint64_t begin=0,end=0,chunk_terms=UINT64_C(1)<<20;
    unsigned query=UINT32_MAX,blocks=0,threads=256;
    std::string output;
    bool list=false,self_test=false,run=false;
};

uint64_t number(const char* text) {
    char* end=nullptr;
    auto value=std::strtoull(text,&end,10);
    if(!end||*end)throw std::runtime_error("invalid integer");
    return value;
}

Options parse_options(int argc,char** argv) {
    Options options;
    for(int i=1;i<argc;++i) {
        std::string argument=argv[i];
        if(argument=="--list")options.list=true;
        else if(argument=="--self-test")options.self_test=true;
        else if(argument=="--run")options.run=true;
        else if(argument=="--query"&&i+1<argc)options.query=unsigned(number(argv[++i]));
        else if(argument=="--prime"&&i+1<argc)options.prime=uint32_t(number(argv[++i]));
        else if(argument=="--begin"&&i+1<argc)options.begin=number(argv[++i]);
        else if(argument=="--end"&&i+1<argc)options.end=number(argv[++i]);
        else if(argument=="--chunk-terms"&&i+1<argc)options.chunk_terms=number(argv[++i]);
        else if(argument=="--blocks"&&i+1<argc)options.blocks=unsigned(number(argv[++i]));
        else if(argument=="--threads"&&i+1<argc)options.threads=unsigned(number(argv[++i]));
        else if(argument=="--output"&&i+1<argc)options.output=argv[++i];
        else throw std::runtime_error(
            "usage: six_by_twenty_nine_hafnian_gpu [--list|--self-test|--run --query Q --prime P --begin B --end E --chunk-terms N --output FILE]");
    }
    return options;
}

void write_atomic(const std::string& path,const std::string& contents) {
    if(path.empty())return;
    std::filesystem::path target(path);
    if(!target.parent_path().empty())std::filesystem::create_directories(target.parent_path());
    auto temporary=target;
    temporary+=".tmp."+std::to_string(::getpid());
    {
        std::ofstream output(temporary);
        if(!output)throw std::runtime_error("cannot open result file");
        output<<contents;
        if(!output)throw std::runtime_error("cannot write result file");
    }
    std::filesystem::rename(temporary,target);
}

template<unsigned N>
std::string run_query(
    const Query& query,const Catalog& catalog,const Options& options,const char* executable) {
    static_assert(N==54||N==56||N==58||N==62);
    constexpr unsigned HALF=N/2;
    constexpr uint64_t TOTAL_TERMS=UINT64_C(1)<<(HALF-1);
    if(query.vertices!=N)throw std::runtime_error("query/kernel order mismatch");
    uint64_t end=options.end?options.end:TOTAL_TERMS;
    if(options.begin>=end||end>TOTAL_TERMS||!options.chunk_terms||
            !options.threads||options.threads>1024)
        throw std::runtime_error("invalid work range/configuration");

    HafnianMontgomery mod;
    mod.p=options.prime;
    mod.negative_inverse=0U-hafnian_inverse_mod_2_32(options.prime);
    mod.one=uint32_t((UINT64_C(1)<<32)%options.prime);
    if(hafnian_host_montgomery_mul(mod.one,1,mod)!=1)
        throw std::runtime_error("Montgomery setup failure");
    std::array<uint32_t,HALF+1> inverses{};
    for(unsigned i=1;i<=HALF;++i) {
        uint32_t encoded=uint32_t(uint64_t(i)*mod.one%mod.p);
        inverses[i]=hafnian_host_montgomery_power(encoded,mod.p-2,mod);
    }

    int device=0,multiprocessors=0;
    hafnian_cuda_check(cudaGetDevice(&device),"cudaGetDevice");
    hafnian_cuda_check(cudaDeviceGetAttribute(
        &multiprocessors,cudaDevAttrMultiProcessorCount,device),"multiprocessor count");
    unsigned blocks=options.blocks?options.blocks:unsigned(multiprocessors)*4;
    uint8_t* device_adjacency=nullptr;
    uint32_t* device_inverses=nullptr;
    uint32_t* device_sums=nullptr;
    hafnian_cuda_check(cudaMalloc(&device_adjacency,query.adjacency.size()),"allocate adjacency");
    hafnian_cuda_check(cudaMalloc(&device_inverses,sizeof(inverses)),"allocate inverses");
    hafnian_cuda_check(cudaMalloc(&device_sums,size_t(blocks)*sizeof(uint32_t)),"allocate sums");
    hafnian_cuda_check(cudaMemcpy(device_adjacency,query.adjacency.data(),query.adjacency.size(),
        cudaMemcpyHostToDevice),"copy adjacency");
    hafnian_cuda_check(cudaMemcpy(device_inverses,inverses.data(),sizeof(inverses),
        cudaMemcpyHostToDevice),"copy inverses");
    std::vector<uint32_t> host_sums(blocks);

    constexpr size_t shared_bytes=hafnian_shared_bytes<N>();
    hafnian_cuda_check(cudaFuncSetAttribute(hafnian_terms_kernel<N>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared_bytes)),"set dynamic shared memory");
    uint32_t partial=0;
    auto started=Clock::now();
    std::string binary_digest=sha256_file(executable);
    auto publish=[&](uint64_t covered_end,double elapsed) {
        char buffer[8192];
        int length=std::snprintf(buffer,sizeof(buffer),
            "format six-by-twenty-nine-hafnian-v1\nalgorithm %s\nrows 6\ncolumns 29\n"
            "catalog_sha256 %s\nquery_id %u\nquery_sha256 %s\noccupied_tokens %" PRIu64 "\n"
            "defect_count %u\nexcess %u\nunmatched_tokens %u\ndefect_coefficient %" PRIu64 "\n"
            "vertices %u\nsolver_binary_sha256 %s\nprime %u\nbegin %" PRIu64 "\nend %" PRIu64 "\n"
            "total_terms %" PRIu64 "\npartial_glynn_sum %u\nblocks %u\nthreads %u\n"
            "chunk_terms %" PRIu64 "\nelapsed_seconds %.9f\nstatus complete\n",
            ALGORITHM,catalog.digest.c_str(),query.id,query.digest.c_str(),query.occupied,
            query.defect_count,query.excess,query.unmatched,query.defect_coefficient,N,
            binary_digest.c_str(),options.prime,options.begin,covered_end,TOTAL_TERMS,partial,
            blocks,options.threads,options.chunk_terms,elapsed);
        if(length<0||size_t(length)>=sizeof(buffer))
            throw std::runtime_error("result formatting failed");
        std::string result(buffer,size_t(length));
        result+="result_payload_sha256 "+sha256_string(result)+"\n";
        write_atomic(options.output,result);
        return result;
    };

    std::string final_result;
    for(uint64_t begin=options.begin;begin<end;begin+=options.chunk_terms) {
        uint64_t chunk_end=std::min(end,begin+options.chunk_terms);
        hafnian_terms_kernel<N><<<blocks,options.threads,shared_bytes>>>(
            device_adjacency,begin,chunk_end,mod,device_inverses,device_sums);
        hafnian_cuda_check(cudaGetLastError(),"launch hafnian kernel");
        hafnian_cuda_check(cudaMemcpy(host_sums.data(),device_sums,
            size_t(blocks)*sizeof(uint32_t),cudaMemcpyDeviceToHost),"copy block sums");
        for(uint32_t value:host_sums) {
            partial+=value;
            if(partial>=mod.p)partial-=mod.p;
        }
        double elapsed=std::chrono::duration<double>(Clock::now()-started).count();
        final_result=publish(chunk_end,elapsed);
        std::printf(
            "HAFNIAN_6X29_PROGRESS query=%u vertices=%u prime=%u begin=%" PRIu64
            " end=%" PRIu64 " elapsed=%.6f terms_per_second=%.3f\n",
            query.id,N,options.prime,options.begin,chunk_end,elapsed,
            double(chunk_end-options.begin)/elapsed);
        std::fflush(stdout);
    }
    cudaFree(device_sums);
    cudaFree(device_inverses);
    cudaFree(device_adjacency);
    return final_result;
}

std::string dispatch(
    const Query& query,const Catalog& catalog,const Options& options,const char* executable) {
    switch(query.vertices) {
        case 54:return run_query<54>(query,catalog,options,executable);
        case 56:return run_query<56>(query,catalog,options,executable);
        case 58:return run_query<58>(query,catalog,options,executable);
        case 62:return run_query<62>(query,catalog,options,executable);
        default:throw std::runtime_error("unsupported residual graph order");
    }
}

} // namespace

int main(int argc,char** argv) {
    try {
        Options options=parse_options(argc,argv);
        Catalog catalog=six_by_twenty_nine::build_catalog();
        std::printf("HAFNIAN_6X29_CATALOG queries=%zu digest=%s exact=OK\n",
            catalog.queries.size(),catalog.digest.c_str());
        if(options.list) {
            for(const Query& query:catalog.queries)
                std::printf(
                    "HAFNIAN_6X29_QUERY id=%u occupied=%" PRIu64 " defects=%u excess=%u "
                    "unmatched=%u coefficient=%" PRIu64 " vertices=%u terms=%" PRIu64
                    " digest=%s\n",query.id,query.occupied,query.defect_count,query.excess,
                    query.unmatched,query.defect_coefficient,query.vertices,
                    UINT64_C(1)<<(query.vertices/2-1),query.digest.c_str());
        }
        if(options.self_test&&!options.run) {
            options.run=true;
            options.query=0;
            options.prime=2147483647U;
            options.begin=0;
            options.end=16;
            options.chunk_terms=16;
            options.blocks=1;
            options.threads=128;
        }
        if(!options.run)return 0;
        if(options.query>=catalog.queries.size())throw std::runtime_error("query ID out of range");
        const Query& query=catalog.queries[options.query];
        if(!is_prime_u32(options.prime)||options.prime<=query.vertices/2||options.prime>INT32_MAX)
            throw std::runtime_error("prime outside supported range");
        std::string result=dispatch(query,catalog,options,argv[0]);
        std::fwrite(result.data(),1,result.size(),stdout);
        return 0;
    } catch(const std::exception& error) {
        std::fprintf(stderr,"error: %s\n",error.what());
        return 2;
    }
}
