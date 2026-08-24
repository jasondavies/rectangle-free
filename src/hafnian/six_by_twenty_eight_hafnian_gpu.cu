// Exact persistent CUDA residual-hafnian solver for T_4(6,28).

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
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <unistd.h>

#include "hafnian_gpu_core.cuh"
#include "hafnian_gray_gpu_core.cuh"
#include "../common/sha256.hpp"
#include "six_by_twenty_eight_catalog.hpp"

namespace {

using Clock=std::chrono::steady_clock;
using six_by_twenty_eight::Catalog;
using six_by_twenty_eight::Query;
#if defined(HAFNIAN_RUNTIME_MONTGOMERY_CONTROL)
constexpr const char* ALGORITHM="glynn-trace-hessenberg-residual-runtime-montgomery-control-v1";
constexpr const char* FORMAT="six-by-twenty-eight-hafnian-v1";
#else
constexpr const char* ALGORITHM="glynn-gray-fraction-free-lanczos-fixed-field-cuda-v3";
constexpr const char* FORMAT="six-by-twenty-eight-hafnian-v2";
#endif

bool is_prime_u32(uint32_t n) {
    if(n<2)return false;
    for(uint32_t p:{2U,3U,5U,7U,11U,13U,17U,19U,23U,29U,31U,37U}) {
        if(n==p)return true;
        if(n%p==0)return false;
    }
    uint32_t d=n-1,s=0;
    while(!(d&1)){d>>=1;++s;}
    auto mul=[n](uint64_t a,uint64_t b){
        return uint64_t((static_cast<unsigned __int128>(a)*b)%n);
    };
    auto power=[&](uint64_t a,uint64_t exponent){
        uint64_t result=1;
        while(exponent){
            if(exponent&1)result=mul(result,a);
            a=mul(a,a);exponent>>=1;
        }
        return result;
    };
    for(uint32_t base:{2U,3U,5U,7U,11U}) {
        if(base>=n)continue;
        uint64_t value=power(base,d);
        if(value==1||value==n-1)continue;
        bool composite=true;
        for(uint32_t r=1;r<s;++r) {
            value=mul(value,value);
            if(value==n-1){composite=false;break;}
        }
        if(composite)return false;
    }
    return true;
}

uint64_t number(const std::string& text) {
    char* end=nullptr;
    uint64_t value=std::strtoull(text.c_str(),&end,10);
    if(!end||*end)throw std::runtime_error("invalid integer: "+text);
    return value;
}

struct Task {
    unsigned query=UINT32_MAX;
    uint32_t prime=2147483647U;
    uint64_t begin=0,end=0;
    std::string output;
};

struct Options {
    Task task;
    // N=48 queries contain only 2^23 terms, so the production default writes
    // them once.  At the slow N=64 end this remains a roughly 10--15 second
    // interruption window on the measured RTX PRO 6000 worker.
    uint64_t chunk_terms=UINT64_C(1)<<24;
    // Zero selects the measured order-specific production launch geometry.
    // At Gray-enabled orders, a nonzero value is the number of logical warp
    // chain slots; the exact fallback grid remains occupancy-derived.
    unsigned blocks=0,threads=0;
    std::string batch;
    bool list=false,self_test=false,run=false;
};

Options parse_options(int argc,char** argv) {
    Options options;
    for(int i=1;i<argc;++i) {
        std::string argument=argv[i];
        auto take=[&]() {
            if(++i>=argc)throw std::runtime_error("missing value for "+argument);
            return std::string(argv[i]);
        };
        if(argument=="--list")options.list=true;
        else if(argument=="--self-test")options.self_test=true;
        else if(argument=="--run")options.run=true;
        else if(argument=="--batch")options.batch=take();
        else if(argument=="--query")options.task.query=unsigned(number(take()));
        else if(argument=="--prime")options.task.prime=uint32_t(number(take()));
        else if(argument=="--begin")options.task.begin=number(take());
        else if(argument=="--end")options.task.end=number(take());
        else if(argument=="--chunk-terms")options.chunk_terms=number(take());
        else if(argument=="--blocks")options.blocks=unsigned(number(take()));
        else if(argument=="--threads")options.threads=unsigned(number(take()));
        else if(argument=="--output")options.task.output=take();
        else throw std::runtime_error(
            "usage: six_by_twenty_eight_hafnian_gpu [--list|--self-test|"
            "--run --query Q --prime P --begin B --end E --output FILE|"
            "--batch FILE] [--chunk-terms N --blocks N --threads N]");
    }
    return options;
}

void write_atomic(const std::string& path,const std::string& contents) {
    if(path.empty())return;
    std::filesystem::path target(path);
    if(!target.parent_path().empty())
        std::filesystem::create_directories(target.parent_path());
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

std::vector<Task> read_batch(const std::string& path) {
    std::ifstream input(path);
    if(!input)throw std::runtime_error("cannot open batch file: "+path);
    std::vector<Task> tasks;
    std::string line;
    while(std::getline(input,line)) {
        if(line.empty()||line[0]=='#')continue;
        std::istringstream fields(line);
        Task task;
        uint64_t prime=0;
        if(!(fields>>task.query>>prime>>task.begin>>task.end>>task.output))
            throw std::runtime_error("malformed batch line: "+line);
        std::string extra;
        if(fields>>extra)throw std::runtime_error("extra batch field: "+line);
        task.prime=uint32_t(prime);
        tasks.push_back(std::move(task));
    }
    return tasks;
}

class DeviceWorkspace {
  public:
    DeviceWorkspace() {
        int device=0;
        hafnian_cuda_check(cudaGetDevice(&device),"cudaGetDevice");
        hafnian_cuda_check(cudaDeviceGetAttribute(
            &multiprocessors,cudaDevAttrMultiProcessorCount,device),
            "multiprocessor count");
        hafnian_cuda_check(cudaMalloc(&adjacency,64*64),"allocate adjacency");
        hafnian_cuda_check(cudaMalloc(&inverses,33*sizeof(uint32_t)),"allocate inverses");
        hafnian_cuda_check(cudaMalloc(&gray_failures,sizeof(uint32_t)),
            "allocate Gray failure counter");
    }
    ~DeviceWorkspace() {
        cudaFree(gray_scratch);cudaFree(gray_failures);cudaFree(sums);
        cudaFree(inverses);cudaFree(adjacency);
    }
    DeviceWorkspace(const DeviceWorkspace&)=delete;
    DeviceWorkspace& operator=(const DeviceWorkspace&)=delete;

    void ensure_blocks(unsigned wanted) {
        if(wanted<=block_capacity)return;
        if(sums)hafnian_cuda_check(cudaFree(sums),"free old block sums");
        hafnian_cuda_check(cudaMalloc(&sums,size_t(wanted)*sizeof(uint32_t)),
            "allocate block sums");
        host_sums.resize(wanted);
        block_capacity=wanted;
    }

    void ensure_gray_words(size_t wanted) {
        if(wanted<=gray_word_capacity)return;
        if(gray_scratch)
            hafnian_cuda_check(cudaFree(gray_scratch),"free old Gray scratch");
        hafnian_cuda_check(cudaMalloc(&gray_scratch,wanted*sizeof(uint32_t)),
            "allocate Gray scratch");
        gray_word_capacity=wanted;
    }

    uint8_t* adjacency=nullptr;
    uint32_t* inverses=nullptr;
    uint32_t* sums=nullptr;
    uint32_t* gray_failures=nullptr;
    uint32_t* gray_scratch=nullptr;
    int multiprocessors=0;
    unsigned block_capacity=0;
    size_t gray_word_capacity=0;
    std::vector<uint32_t> host_sums;
};

template<unsigned N,class ActiveMod>
std::string run_query(const Query& query,const Catalog& catalog,const Task& task,
    const Options& options,DeviceWorkspace& workspace,
    const std::string& binary_digest) {
    static_assert(N==48||N==50||N==52||N==54||N==56||N==58||N==60||N==64);
    constexpr unsigned HALF=N/2;
    constexpr unsigned GRAY_CHAIN=N==48?6:(N<=58?7:0);
    constexpr uint64_t TOTAL_TERMS=UINT64_C(1)<<(HALF-1);
    if(query.vertices!=N)throw std::runtime_error("query/kernel order mismatch");
    uint64_t end=task.end?task.end:TOTAL_TERMS;
    if(task.begin>=end||end>TOTAL_TERMS||!options.chunk_terms||
            options.threads>1024)
        throw std::runtime_error("invalid work range/configuration");
    if(!is_prime_u32(task.prime)||task.prime<=HALF||task.prime>INT32_MAX)
        throw std::runtime_error("prime outside supported range");

    ActiveMod mod{};
    if constexpr(std::is_same_v<ActiveMod,HafnianMontgomery>) {
        mod.p=task.prime;
        mod.negative_inverse=0U-hafnian_inverse_mod_2_32(task.prime);
        mod.one=uint32_t((UINT64_C(1)<<32)%task.prime);
    } else if(task.prime!=ActiveMod::p) {
        throw std::runtime_error("prime does not match Montgomery specialization");
    }
    if(hafnian_host_montgomery_mul(mod.one,1,mod)!=1)
        throw std::runtime_error("Montgomery setup failure");
    std::array<uint32_t,HALF+1> inverse_values{};
    for(unsigned i=1;i<=HALF;++i) {
        uint32_t encoded=uint32_t(uint64_t(i)*mod.one%mod.p);
        inverse_values[i]=hafnian_host_montgomery_power(encoded,mod.p-2,mod);
    }
    hafnian_cuda_check(cudaMemcpy(workspace.adjacency,query.adjacency.data(),
        query.adjacency.size(),cudaMemcpyHostToDevice),"copy adjacency");
    hafnian_cuda_check(cudaMemcpy(workspace.inverses,inverse_values.data(),
        sizeof(inverse_values),cudaMemcpyHostToDevice),"copy inverses");

    constexpr size_t independent_shared_bytes=hafnian_shared_bytes<N>();
    unsigned threads=options.threads?options.threads:(N<=50?224U:256U);
    // Use complete residency waves for the independent fallback.  The old
    // fixed 4*SM grid leaves a severe
    // 3+1 tail for N=64, whereas two full waves remain balanced for every N.
    unsigned recommended_blocks=hafnian_recommended_blocks<N,ActiveMod>(
        threads,workspace.multiprocessors);
    unsigned blocks=options.blocks?options.blocks:recommended_blocks;

    bool gray_enabled=false;
    unsigned gray_slots=0,gray_grid_blocks=0,gray_active_blocks=0;
    uint32_t gray_failures_total=0,gray_fallback_chunks=0,gray_chunks=0;
    hafnian_gray::RankFactor gray_factor;
    hafnian_gray::DeviceFactors gray_factors;
#if !defined(HAFNIAN_RUNTIME_MONTGOMERY_CONTROL)
    if constexpr(N>=48&&N<=58) {
        constexpr unsigned CHAIN=GRAY_CHAIN;
        if(options.threads&&options.threads!=hafnian_gray::THREADS)
            throw std::runtime_error(
                "orders 48 and 50 use a fixed 64-thread Gray kernel");
        gray_factor=hafnian_gray::factor_adjacency(query,mod.p);
        if(gray_factor.rank) {
            gray_factors=hafnian_gray::make_device_factors<N>(query,gray_factor,mod);
            int active=0;
            hafnian_cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active,hafnian_gray::terms_kernel<N,CHAIN,ActiveMod>,
                hafnian_gray::THREADS,hafnian_gray::shared_bytes<N>()),
                "compute Gray kernel occupancy");
            if(active<=0)throw std::runtime_error("Gray kernel has zero occupancy");
            gray_active_blocks=unsigned(active);
            uint64_t requested=options.blocks?options.blocks:
                uint64_t(workspace.multiprocessors)*unsigned(active)*
                hafnian_gray::WARPS_PER_BLOCK*24U;
            const uint64_t maximum_groups=(std::min<uint64_t>(
                options.chunk_terms,end-task.begin)+CHAIN-1)/CHAIN;
            requested=std::min(requested,std::max<uint64_t>(2,maximum_groups));
            requested=std::max<uint64_t>(2,(requested+1)&~UINT64_C(1));
            if(requested>UINT32_MAX)
                throw std::runtime_error("too many Gray chain slots");
            gray_slots=unsigned(requested);
            gray_grid_blocks=gray_slots/hafnian_gray::WARPS_PER_BLOCK;
            workspace.ensure_gray_words(
                hafnian_gray::scratch_words<N,CHAIN>(gray_slots));
            gray_enabled=true;
            threads=N<=50?224U:256U;
            blocks=hafnian_recommended_blocks<N,ActiveMod>(
                threads,workspace.multiprocessors);
        }
    }
#endif
    workspace.ensure_blocks(std::max(blocks,gray_slots));
    uint32_t partial=0;
    auto started=Clock::now();
    auto publish=[&](uint64_t covered_end,double elapsed) {
        char buffer[8192];
        int length=std::snprintf(buffer,sizeof(buffer),
            "format %s\nalgorithm %s\nrows 6\ncolumns 28\n"
            "catalog_sha256 %s\nquery_id %u\nquery_sha256 %s\n"
            "occupied_tokens %" PRIu64 "\ndefect_count %u\nexcess %u\n"
            "unmatched_tokens %u\ndefect_coefficient %" PRIu64 "\n"
            "matching_bound_power %u\nvertices %u\nmatrix_stride %u\n"
            "solver_binary_sha256 %s\nprime %u\nbegin %" PRIu64 "\nend %" PRIu64 "\n"
            "total_terms %" PRIu64 "\npartial_glynn_sum %u\nblocks %u\nthreads %u\n"
            "gray_enabled %u\ngray_chain %u\ngray_slots %u\ngray_grid_blocks %u\n"
            "gray_active_blocks_per_sm %u\ngray_chunks %u\ngray_failures %u\n"
            "gray_fallback_chunks %u\nchunk_terms %" PRIu64
            "\nelapsed_seconds %.9f\nstatus complete\n",
            FORMAT,ALGORITHM,catalog.digest.c_str(),query.id,query.digest.c_str(),
            query.occupied,query.defect_count,query.excess,query.unmatched,
            query.defect_coefficient,query.matching_bound_power,N,
            N+1,binary_digest.c_str(),task.prime,task.begin,
            covered_end,TOTAL_TERMS,partial,blocks,threads,
            unsigned(gray_enabled),gray_enabled?GRAY_CHAIN:0U,
            gray_slots,
            gray_grid_blocks,gray_active_blocks,gray_chunks,
            gray_failures_total,gray_fallback_chunks,
            options.chunk_terms,elapsed);
        if(length<0||size_t(length)>=sizeof(buffer))
            throw std::runtime_error("result formatting failed");
        std::string result(buffer,size_t(length));
        result+="result_payload_sha256 "+sha256_string(result)+"\n";
        write_atomic(task.output,result);
        return result;
    };

    std::string final_result;
    for(uint64_t begin=task.begin;begin<end;begin+=options.chunk_terms) {
        uint64_t chunk_end=std::min(end,begin+options.chunk_terms);
        unsigned sum_count=blocks;
        bool fallback=!gray_enabled;
#if !defined(HAFNIAN_RUNTIME_MONTGOMERY_CONTROL)
        if constexpr(N>=48&&N<=58) {
            if(gray_enabled) {
                constexpr unsigned CHAIN=GRAY_CHAIN;
                hafnian_cuda_check(cudaMemset(
                    workspace.gray_failures,0,sizeof(uint32_t)),
                    "clear Gray failure counter");
                hafnian_gray::terms_kernel<N,CHAIN,ActiveMod><<<
                    gray_grid_blocks,hafnian_gray::THREADS,
                    hafnian_gray::shared_bytes<N>()>>>(
                    gray_factors.edge_matrices,gray_factors.update_vectors,
                    gray_factors.metric,gray_factor.rank,begin,chunk_end,mod,
                    workspace.inverses,workspace.gray_scratch,workspace.sums,
                    workspace.gray_failures);
                hafnian_cuda_check(cudaGetLastError(),"launch Gray hafnian kernel");
                uint32_t failures=0;
                hafnian_cuda_check(cudaMemcpy(&failures,workspace.gray_failures,
                    sizeof(uint32_t),cudaMemcpyDeviceToHost),
                    "copy Gray failure counter");
                ++gray_chunks;
                gray_failures_total+=failures;
                fallback=failures!=0;
#if defined(HAFNIAN_FORCE_GRAY_FALLBACK)
                fallback=true;
#endif
                if(fallback)++gray_fallback_chunks;
                else sum_count=gray_slots;
            }
        }
#endif
        if(fallback) {
            hafnian_terms_kernel<N,ActiveMod,true><<<
                blocks,threads,independent_shared_bytes>>>(
                workspace.adjacency,begin,chunk_end,mod,workspace.inverses,
                workspace.sums);
            hafnian_cuda_check(cudaGetLastError(),
                "launch exact Gray-order fallback kernel");
            sum_count=blocks;
        }
        hafnian_cuda_check(cudaMemcpy(workspace.host_sums.data(),workspace.sums,
            size_t(sum_count)*sizeof(uint32_t),cudaMemcpyDeviceToHost),
            "copy block sums");
        for(unsigned block=0;block<sum_count;++block) {
            uint32_t value=workspace.host_sums[block];
            partial+=value;
            if(partial>=mod.p)partial-=mod.p;
        }
        double elapsed=std::chrono::duration<double>(Clock::now()-started).count();
        final_result=publish(chunk_end,elapsed);
        std::printf(
            "HAFNIAN_6X28_PROGRESS query=%u vertices=%u stride=%u prime=%u "
            "begin=%" PRIu64 " end=%" PRIu64
            " gray=%u gray_failures=%u fallback_chunks=%u"
            " elapsed=%.6f terms_per_second=%.3f\n",
            query.id,N,N+1,task.prime,task.begin,chunk_end,
            unsigned(gray_enabled),gray_failures_total,gray_fallback_chunks,
            elapsed,double(chunk_end-task.begin)/elapsed);
        std::fflush(stdout);
    }
    return final_result;
}

template<class Mod>
std::string dispatch_mod(const Query& query,const Catalog& catalog,const Task& task,
    const Options& options,DeviceWorkspace& workspace,
    const std::string& binary_digest) {
    switch(query.vertices) {
        case 48:return run_query<48,Mod>(query,catalog,task,options,workspace,binary_digest);
        case 50:return run_query<50,Mod>(query,catalog,task,options,workspace,binary_digest);
        case 52:return run_query<52,Mod>(query,catalog,task,options,workspace,binary_digest);
        case 54:return run_query<54,Mod>(query,catalog,task,options,workspace,binary_digest);
        case 56:return run_query<56,Mod>(query,catalog,task,options,workspace,binary_digest);
        case 58:return run_query<58,Mod>(query,catalog,task,options,workspace,binary_digest);
        case 60:return run_query<60,Mod>(query,catalog,task,options,workspace,binary_digest);
        case 64:return run_query<64,Mod>(query,catalog,task,options,workspace,binary_digest);
        default:throw std::runtime_error("unsupported residual graph order");
    }
}

template<class Mod>
std::string dispatch_fixed_small(const Query& query,const Catalog& catalog,
    const Task& task,const Options& options,DeviceWorkspace& workspace,
    const std::string& binary_digest) {
    switch(query.vertices) {
        case 48:return run_query<48,Mod>(query,catalog,task,options,workspace,binary_digest);
        case 50:return run_query<50,Mod>(query,catalog,task,options,workspace,binary_digest);
        case 52:return run_query<52,Mod>(query,catalog,task,options,workspace,binary_digest);
        case 54:return run_query<54,Mod>(query,catalog,task,options,workspace,binary_digest);
        case 56:return run_query<56,Mod>(query,catalog,task,options,workspace,binary_digest);
        case 58:return run_query<58,Mod>(query,catalog,task,options,workspace,binary_digest);
        default:throw std::runtime_error(
            "fixed-prime specialization is only used at orders 48 through 58");
    }
}

std::string dispatch(const Query& query,const Catalog& catalog,const Task& task,
    const Options& options,DeviceWorkspace& workspace,
    const std::string& binary_digest) {
#if defined(HAFNIAN_RUNTIME_MONTGOMERY_CONTROL)
    return dispatch_mod<HafnianMontgomery>(
        query,catalog,task,options,workspace,binary_digest);
#else
    if(task.prime==HafnianMersenne31::p)
        return dispatch_mod<HafnianMersenne31>(
            query,catalog,task,options,workspace,binary_digest);
    if(query.vertices>58)
        return dispatch_mod<HafnianMontgomery>(
            query,catalog,task,options,workspace,binary_digest);
    switch(task.prime) {
        case 2147483629U:return dispatch_fixed_small<HafnianMontgomeryConstant<2147483629U>>(
            query,catalog,task,options,workspace,binary_digest);
        case 2147483587U:return dispatch_fixed_small<HafnianMontgomeryConstant<2147483587U>>(
            query,catalog,task,options,workspace,binary_digest);
        case 2147483579U:return dispatch_fixed_small<HafnianMontgomeryConstant<2147483579U>>(
            query,catalog,task,options,workspace,binary_digest);
        default:throw std::runtime_error("prime is outside the production CRT schedule");
    }
#endif
}

} // namespace

int main(int argc,char** argv) try {
    Options options=parse_options(argc,argv);
    auto catalog=six_by_twenty_eight::build_catalog();
    std::printf(
        "HAFNIAN_6X28_CATALOG queries=%zu digest=%s matrix_padding=%u exact=OK\n",
        catalog.queries.size(),catalog.digest.c_str(),1U);
    if(options.list) {
        for(const Query& query:catalog.queries)
            std::printf(
                "HAFNIAN_6X28_QUERY id=%u occupied=%" PRIu64
                " defects=%u excess=%u unmatched=%u coefficient=%" PRIu64
                " vertices=%u terms=%" PRIu64 " matching_bound_power=%u digest=%s\n",
                query.id,query.occupied,query.defect_count,query.excess,
                query.unmatched,query.defect_coefficient,query.vertices,
                UINT64_C(1)<<(query.vertices/2-1),query.matching_bound_power,
                query.digest.c_str());
    }
    if(options.self_test&&!options.run&&!options.batch.size()) {
        options.run=true;
        options.task.query=0;
        options.task.prime=2147483647U;
        options.task.begin=0;
        options.task.end=16;
        options.chunk_terms=16;
        options.blocks=1;
        options.threads=128;
    }
    std::vector<Task> tasks;
    if(!options.batch.empty())tasks=read_batch(options.batch);
    if(options.run)tasks.push_back(options.task);
    if(tasks.empty())return 0;
    DeviceWorkspace workspace;
    const std::string binary_digest=sha256_file(argv[0]);
    for(const Task& task:tasks) {
        if(task.query>=catalog.queries.size())
            throw std::runtime_error("query ID out of range");
        const Query& query=catalog.queries[task.query];
        std::string result=dispatch(
            query,catalog,task,options,workspace,binary_digest);
        if(tasks.size()==1)std::fwrite(result.data(),1,result.size(),stdout);
    }
    return 0;
} catch(const std::exception& error) {
    std::fprintf(stderr,"error: %s\n",error.what());
    return 2;
}
