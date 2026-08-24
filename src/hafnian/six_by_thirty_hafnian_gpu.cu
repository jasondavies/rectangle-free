// Exact CUDA Glynn/trace hafnian solver for T_4(6,30).
// See six_by_thirty_hafnian.cpp for the independent CPU implementation.

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
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

#include "../common/sha256.hpp"
#include "hafnian_gpu_core.cuh"

namespace {

using Clock = std::chrono::steady_clock;
constexpr unsigned ROWS = 6, COLOURS = 4, PAIRS = 15;
constexpr unsigned N = 60, HALF = 30;
constexpr uint64_t TOTAL_TERMS = UINT64_C(1) << 29;
constexpr const char* ALGORITHM = "glynn-trace-hessenberg-cuda-v1";

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

bool is_prime_u32(uint32_t n) {
    if (n < 2) return false;
    for (uint32_t p : {2U,3U,5U,7U,11U,13U,17U,19U,23U,29U,31U,37U}) {
        if (n == p) return true;
        if (n % p == 0) return false;
    }
    uint32_t d = n - 1, s = 0;
    while (!(d & 1)) { d >>= 1; ++s; }
    auto mul = [n](uint64_t a, uint64_t b) { return uint64_t((unsigned __int128)a * b % n); };
    auto power = [&](uint64_t a, uint64_t exponent) {
        uint64_t result = 1;
        while (exponent) { if (exponent & 1) result = mul(result,a); a=mul(a,a); exponent>>=1; }
        return result;
    };
    for (uint32_t base : {2U,3U,5U,7U,11U}) {
        if (base >= n) continue;
        uint64_t x = power(base,d);
        if (x == 1 || x == n-1) continue;
        bool composite = true;
        for (uint32_t r=1;r<s;++r) { x=mul(x,x); if(x==n-1){composite=false;break;} }
        if (composite) return false;
    }
    return true;
}

struct HostGraph {
    std::array<uint8_t,N*N> reordered{};
    std::array<uint8_t,N> order{};
};

HostGraph build_graph() {
    std::array<std::pair<unsigned,unsigned>,PAIRS> pairs{};
    unsigned count=0;
    for(unsigned i=0;i<ROWS;++i) for(unsigned j=i+1;j<ROWS;++j) pairs[count++]={i,j};
    auto disjoint=[&](unsigned p,unsigned q){
        auto [a,b]=pairs[p]; auto [c,d]=pairs[q];
        return a!=c&&a!=d&&b!=c&&b!=d;
    };
    std::array<int,PAIRS> right_match{}; right_match.fill(-1);
    auto augment=[&](auto&& self,unsigned left,std::array<uint8_t,PAIRS>& seen)->bool{
        for(unsigned right=0;right<PAIRS;++right) {
            if(!disjoint(left,right)||seen[right])continue;
            seen[right]=1;
            if(right_match[right]<0||self(self,unsigned(right_match[right]),seen)){
                right_match[right]=int(left); return true;
            }
        }
        return false;
    };
    for(unsigned left=0;left<PAIRS;++left){
        std::array<uint8_t,PAIRS> seen{};
        if(!augment(augment,left,seen))throw std::runtime_error("KG matching construction failed");
    }
    std::array<unsigned,PAIRS> mate{};
    for(unsigned right=0;right<PAIRS;++right)mate[unsigned(right_match[right])]=right;
    HostGraph graph;
    unsigned first=0,second=HALF;
    for(unsigned colour=0;colour<COLOURS;colour+=2)for(unsigned pair=0;pair<PAIRS;++pair){
        graph.order[first++]=uint8_t(colour*PAIRS+pair);
        graph.order[second++]=uint8_t((colour+1)*PAIRS+mate[pair]);
    }
    std::array<uint8_t,N> sorted=graph.order; std::sort(sorted.begin(),sorted.end());
    for(unsigned i=0;i<N;++i)if(sorted[i]!=i)throw std::runtime_error("invalid graph order");
    auto adjacent=[&](unsigned u,unsigned v){
        unsigned colour_u=u/PAIRS,colour_v=v/PAIRS;
        return colour_u!=colour_v&&disjoint(u%PAIRS,v%PAIRS);
    };
    uint64_t degree_sum=0;
    for(unsigned i=0;i<N;++i)for(unsigned j=0;j<N;++j){
        graph.reordered[i*N+j]=uint8_t(adjacent(graph.order[i],graph.order[j]));
        degree_sum+=graph.reordered[i*N+j];
    }
    if(degree_sum!=1080)throw std::runtime_error("target graph census mismatch");
    for(unsigned i=0;i<HALF;++i)if(!graph.reordered[i*N+i+HALF])
        throw std::runtime_error("reference edge missing");
    return graph;
}

std::string graph_sha256(const HostGraph& graph) {
    Sha256 hash;
    const std::string header="token-graph-Kq-cross-KG-r-2-v1\n"; hash.update(header);
    const uint8_t dimensions[]={ROWS,COLOURS,PAIRS,N,HALF}; hash.update(dimensions,sizeof(dimensions));
    // Match the CPU digest: adjacency in original colour/pair order, then the
    // reference-pair ordering.  Reconstruct original adjacency explicitly.
    std::array<uint8_t,N*N> original{};
    for(unsigned i=0;i<N;++i)for(unsigned j=0;j<N;++j)
        original[graph.order[i]*N+graph.order[j]]=graph.reordered[i*N+j];
    hash.update(original.data(),original.size());
    hash.update(graph.order.data(),graph.order.size());
    return hash.finish_hex();
}

struct Options {
    uint32_t prime=2147483647U;
    uint64_t begin=0,end=0,chunk_terms=UINT64_C(1)<<20;
    unsigned blocks=0,threads=256;
    std::string output;
    bool self_test=false,run=false;
};
uint64_t number(const char* text){char* end=nullptr;auto value=std::strtoull(text,&end,10);if(!end||*end)throw std::runtime_error("invalid integer");return value;}
Options options(int argc,char** argv){
    Options o;
    for(int i=1;i<argc;++i){std::string a=argv[i];
        if(a=="--self-test")o.self_test=true; else if(a=="--run")o.run=true;
        else if(a=="--prime"&&i+1<argc)o.prime=uint32_t(number(argv[++i]));
        else if(a=="--begin"&&i+1<argc)o.begin=number(argv[++i]);
        else if(a=="--end"&&i+1<argc)o.end=number(argv[++i]);
        else if(a=="--chunk-terms"&&i+1<argc)o.chunk_terms=number(argv[++i]);
        else if(a=="--blocks"&&i+1<argc)o.blocks=unsigned(number(argv[++i]));
        else if(a=="--threads"&&i+1<argc)o.threads=unsigned(number(argv[++i]));
        else if(a=="--output"&&i+1<argc)o.output=argv[++i];
        else throw std::runtime_error("usage: six_by_thirty_hafnian_gpu [--self-test] [--run --prime P --begin B --end E --chunk-terms N --output FILE]");
    }return o;
}
void write_atomic(const std::string& path,const std::string& contents){
    if(path.empty())return; std::filesystem::path target(path); if(!target.parent_path().empty())std::filesystem::create_directories(target.parent_path());
    auto temporary=target; temporary+=".tmp."+std::to_string(::getpid());
    {std::ofstream out(temporary);if(!out)throw std::runtime_error("cannot open result");out<<contents;if(!out)throw std::runtime_error("cannot write result");}
    std::filesystem::rename(temporary,target);
}

} // namespace

int main(int argc,char** argv){
    try{
        Options o=options(argc,argv);
        bool device_self_test=o.self_test&&!o.run;
        if(device_self_test){
            o.run=true;o.prime=2147483647U;o.begin=0;o.end=16;
            o.blocks=1;o.threads=128;o.chunk_terms=16;o.output.clear();
        }
        if(!is_prime_u32(o.prime)||o.prime<=HALF||o.prime>INT32_MAX)throw std::runtime_error("prime must be in (30,2^31)");
        HostGraph graph=build_graph(); std::string graph_digest=graph_sha256(graph);
        std::printf("HAFNIAN_GPU_GRAPH vertices=60 degree=18 edges=540 terms=%" PRIu64 " graph_sha256=%s exact=OK\n",TOTAL_TERMS,graph_digest.c_str());
        HafnianMontgomery mod; mod.p=o.prime; mod.negative_inverse=0U-hafnian_inverse_mod_2_32(o.prime); mod.one=uint32_t((UINT64_C(1)<<32)%o.prime);
        if(hafnian_host_montgomery_mul(mod.one,1,mod)!=1)throw std::runtime_error("Montgomery setup failure");
        std::array<uint32_t,HALF+1> inverses{};
        for(unsigned i=1;i<=HALF;++i){uint32_t encoded=uint32_t(uint64_t(i)*mod.one%mod.p);inverses[i]=hafnian_host_montgomery_power(encoded,mod.p-2,mod);}
        if(!o.run)return 0;
        if(!o.end)o.end=TOTAL_TERMS;
        if(o.begin>=o.end||o.end>TOTAL_TERMS||!o.chunk_terms||!o.threads||o.threads>1024)throw std::runtime_error("invalid work range/configuration");
        int device=0,multiprocessors=0; cuda_check(cudaGetDevice(&device),"cudaGetDevice");
        cuda_check(cudaDeviceGetAttribute(&multiprocessors,cudaDevAttrMultiProcessorCount,device),"cudaDeviceGetAttribute");
        if(!o.blocks)o.blocks=hafnian_recommended_blocks<N>(o.threads,multiprocessors);
        else (void)hafnian_recommended_blocks<N>(o.threads,multiprocessors);
        uint8_t* device_adjacency=nullptr; uint32_t* device_inverses=nullptr; uint32_t* device_sums=nullptr;
        cuda_check(cudaMalloc(&device_adjacency,graph.reordered.size()),"cudaMalloc adjacency");
        cuda_check(cudaMalloc(&device_inverses,sizeof(inverses)),"cudaMalloc inverses");
        cuda_check(cudaMalloc(&device_sums,size_t(o.blocks)*sizeof(uint32_t)),"cudaMalloc sums");
        cuda_check(cudaMemcpy(device_adjacency,graph.reordered.data(),graph.reordered.size(),cudaMemcpyHostToDevice),"copy adjacency");
        cuda_check(cudaMemcpy(device_inverses,inverses.data(),sizeof(inverses),cudaMemcpyHostToDevice),"copy inverses");
        std::vector<uint32_t> host_sums(o.blocks);
        uint32_t partial=0; uint64_t completed=0;
        const size_t shared_bytes=hafnian_shared_bytes<N>();
        auto started=Clock::now();
        std::string binary_digest=sha256_file(argv[0]);
        auto publish=[&](uint64_t covered_end,double elapsed){
            char buffer[4096];
            int length=std::snprintf(buffer,sizeof(buffer),
                "format six-by-thirty-hafnian-v1\nalgorithm %s\nrows 6\ncolours 4\nvertices 60\nedges 540\n"
                "graph_sha256 %s\nsolver_binary_sha256 %s\nprime %u\nbegin %" PRIu64 "\nend %" PRIu64 "\ntotal_terms %" PRIu64 "\n"
                "partial_glynn_sum %u\nblocks %u\nthreads %u\nchunk_terms %" PRIu64 "\nelapsed_seconds %.9f\nstatus complete\n",
                ALGORITHM,graph_digest.c_str(),binary_digest.c_str(),o.prime,o.begin,covered_end,TOTAL_TERMS,partial,o.blocks,o.threads,o.chunk_terms,elapsed);
            if(length<0||size_t(length)>=sizeof(buffer))throw std::runtime_error("result formatting failed");
            std::string result(buffer,size_t(length));result+="result_payload_sha256 "+sha256_string(result)+"\n";
            write_atomic(o.output,result);
            return result;
        };
        std::string final_result;
        for(uint64_t begin=o.begin;begin<o.end;begin+=o.chunk_terms){
            uint64_t end=std::min(o.end,begin+o.chunk_terms);
            hafnian_terms_kernel<N><<<o.blocks,o.threads,shared_bytes>>>(device_adjacency,begin,end,mod,device_inverses,device_sums);
            cuda_check(cudaGetLastError(),"launch hafnian kernel");
            cuda_check(cudaMemcpy(host_sums.data(),device_sums,size_t(o.blocks)*sizeof(uint32_t),cudaMemcpyDeviceToHost),"copy block sums");
            for(uint32_t value:host_sums){partial+=value;if(partial>=mod.p)partial-=mod.p;}
            completed=end-o.begin;
            double elapsed=std::chrono::duration<double>(Clock::now()-started).count();
            final_result=publish(end,elapsed);
            std::printf("HAFNIAN_GPU_PROGRESS prime=%u begin=%" PRIu64 " end=%" PRIu64 " completed=%" PRIu64 " elapsed=%.6f terms_per_second=%.3f\n",o.prime,o.begin,o.end,completed,elapsed,completed/elapsed);std::fflush(stdout);
        }
        cudaFree(device_sums);cudaFree(device_inverses);cudaFree(device_adjacency);
        if(device_self_test){
            if(partial!=2095133610U)throw std::runtime_error("CUDA target fixture mismatch");
            std::printf("HAFNIAN_GPU_SELF_TEST prime=2147483647 begin=0 end=16 residue=%u montgomery=OK cpu_fixture=OK exact=OK\n",partial);
            return 0;
        }
        std::fwrite(final_result.data(),1,final_result.size(),stdout);return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"error: %s\n",error.what());return 2;}
}
