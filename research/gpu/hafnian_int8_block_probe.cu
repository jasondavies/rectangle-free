// Exact INT8 tensor-core gate for a blocked finite-field Hessenberg update.
//
// This deliberately benchmarks the dense operation that a blocked hafnian
// solver could expose:
//
//     C <- C - A * B (mod p),
//
// with A=Mx16 and B=16xN.  The tensor path uses one exact
// mma.sync.m16n8k16.u8.u8.s32 per 16x8 output tile.  It is compared with the
// same small-prime scalar operation and with a 31-bit Montgomery operation.
// The program also computes the exact 6x28 CRT-image multiplier, so a raw
// microkernel speedup is not mistaken for an end-to-end win.

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/hafnian/six_by_twenty_eight_catalog.hpp"

namespace {

void cuda_check(cudaError_t status,const char* operation) {
    if(status!=cudaSuccess)
        throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}

uint64_t number(const std::string& text) {
    char* end=nullptr;
    uint64_t value=std::strtoull(text.c_str(),&end,10);
    if(!end||*end)throw std::runtime_error("invalid integer: "+text);
    return value;
}

struct Options {
    unsigned order=64;
    unsigned block_k=16;
    unsigned batches=4096;
    unsigned iterations=20;
    unsigned warmup=3;
    unsigned threads=256;
    uint32_t small_prime=251;
};

Options parse_options(int argc,char** argv) {
    Options options;
    for(int index=1;index<argc;++index) {
        std::string argument=argv[index];
        auto take=[&]() {
            if(++index>=argc)throw std::runtime_error("missing value for "+argument);
            return std::string(argv[index]);
        };
        if(argument=="--order")options.order=unsigned(number(take()));
        else if(argument=="--block-k")options.block_k=unsigned(number(take()));
        else if(argument=="--batches")options.batches=unsigned(number(take()));
        else if(argument=="--iterations")options.iterations=unsigned(number(take()));
        else if(argument=="--warmup")options.warmup=unsigned(number(take()));
        else if(argument=="--threads")options.threads=unsigned(number(take()));
        else if(argument=="--prime")options.small_prime=uint32_t(number(take()));
        else throw std::runtime_error(
            "usage: hafnian_int8_block_probe [--order 48|64] [--block-k 8|16|32] [--batches N] "
            "[--iterations N] [--warmup N] [--threads N] [--prime P]");
    }
    if((options.order!=48&&options.order!=64)||
            (options.block_k!=8&&options.block_k!=16&&options.block_k!=32)||!options.batches||
            !options.iterations||!options.threads||options.threads>1024||
            options.threads%32||options.small_prime<=32||options.small_prime>251)
        throw std::runtime_error("invalid probe configuration");
    return options;
}

bool is_prime(uint32_t value) {
    if(value<2)return false;
    for(uint32_t divisor=2;uint64_t(divisor)*divisor<=value;++divisor)
        if(value%divisor==0)return false;
    return true;
}

__device__ __forceinline__ uint32_t reduce_u32(
    uint32_t value,uint32_t prime,uint32_t reciprocal) {
    uint32_t quotient=__umulhi(value,reciprocal);
    uint32_t remainder=value-quotient*prime;
    if(remainder>=prime)remainder-=prime;
    if(remainder>=prime)remainder-=prime;
    return remainder;
}

__device__ __forceinline__ void mma_u8_16x8x16(
    uint32_t a0,uint32_t a1,uint32_t b0,int32_t (&d)[4]) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]),"+r"(d[1]),"+r"(d[2]),"+r"(d[3])
        : "r"(a0),"r"(a1),"r"(b0));
#else
    d[0]=d[1]=d[2]=d[3]=0;
#endif
}

template<unsigned N,unsigned K>
__global__ void tensor_update_u8(
    const uint8_t* __restrict__ left,
    const uint8_t* __restrict__ right_columns,
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    uint64_t batches,uint32_t prime,uint32_t reciprocal) {
    constexpr unsigned ROW_TILES=N/16;
    constexpr unsigned COLUMN_TILES=N/8;
    constexpr unsigned TILES=ROW_TILES*COLUMN_TILES;
    unsigned lane=threadIdx.x&31U;
    unsigned warp_in_block=threadIdx.x>>5;
    uint64_t warp=uint64_t(blockIdx.x)*(blockDim.x/32)+warp_in_block;
    uint64_t warp_stride=uint64_t(gridDim.x)*(blockDim.x/32);
    uint64_t tasks=batches*TILES;
    for(uint64_t task=warp;task<tasks;task+=warp_stride) {
        uint64_t batch=task/TILES;
        unsigned tile=unsigned(task%TILES);
        unsigned row_base=(tile/COLUMN_TILES)*16;
        unsigned column_base=(tile%COLUMN_TILES)*8;
        unsigned group=lane>>2;
        unsigned thread_in_group=lane&3U;
        const uint8_t* left_batch=left+batch*N*K;
        const uint8_t* right_batch=right_columns+batch*N*K;
        int32_t result[4]={0,0,0,0};
#pragma unroll
        for(unsigned k_tile=0;k_tile<K;k_tile+=16) {
            unsigned k_base=k_tile+thread_in_group*4;
            bool valid=k_base<K;
            uint32_t a0=valid?*reinterpret_cast<const uint32_t*>(
                left_batch+(row_base+group)*K+k_base):0;
            uint32_t a1=valid?*reinterpret_cast<const uint32_t*>(
                left_batch+(row_base+group+8)*K+k_base):0;
            uint32_t b0=valid?*reinterpret_cast<const uint32_t*>(
                right_batch+(column_base+group)*K+k_base):0;
            mma_u8_16x8x16(a0,a1,b0,result);
        }
        unsigned output_columns[2]={
            column_base+thread_in_group*2,
            column_base+thread_in_group*2+1};
        unsigned output_rows[2]={row_base+group,row_base+group+8};
#pragma unroll
        for(unsigned row_half=0;row_half<2;++row_half) {
#pragma unroll
            for(unsigned column_half=0;column_half<2;++column_half) {
                unsigned result_index=row_half*2+column_half;
                uint32_t product=reduce_u32(
                    uint32_t(result[result_index]),prime,reciprocal);
                uint64_t offset=batch*N*N+
                    output_rows[row_half]*N+output_columns[column_half];
                uint32_t value=uint32_t(input[offset])+prime-product;
                if(value>=prime)value-=prime;
                output[offset]=uint8_t(value);
            }
        }
    }
}

template<unsigned N,unsigned K>
__global__ void scalar_update_u8(
    const uint8_t* __restrict__ left,
    const uint8_t* __restrict__ right_columns,
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    uint64_t batches,uint32_t prime,uint32_t reciprocal) {
    uint64_t cells=batches*N*N;
    for(uint64_t cell=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
            cell<cells;cell+=uint64_t(gridDim.x)*blockDim.x) {
        uint64_t batch=cell/(N*N);
        unsigned within=unsigned(cell%(N*N));
        unsigned row=within/N,column=within%N;
        const uint8_t* left_row=left+(batch*N+row)*K;
        const uint8_t* right_column=right_columns+(batch*N+column)*K;
        uint32_t product=0;
#pragma unroll
        for(unsigned k=0;k<K;++k)
            product+=uint32_t(left_row[k])*right_column[k];
        product=reduce_u32(product,prime,reciprocal);
        uint32_t value=uint32_t(input[cell])+prime-product;
        if(value>=prime)value-=prime;
        output[cell]=uint8_t(value);
    }
}

struct Montgomery {
    uint32_t prime;
    uint32_t negative_inverse;
};

__host__ __device__ __forceinline__ uint32_t inverse_mod_2_32(uint32_t odd) {
    uint32_t value=odd;
    for(unsigned iteration=0;iteration<5;++iteration)value*=2U-odd*value;
    return value;
}

__host__ __device__ __forceinline__ uint32_t montgomery_mul(
    uint32_t left,uint32_t right,Montgomery mod) {
    uint64_t product=uint64_t(left)*right;
    uint32_t multiplier=uint32_t(product)*mod.negative_inverse;
    uint64_t reduced=(product+uint64_t(multiplier)*mod.prime)>>32;
    if(reduced>=mod.prime)reduced-=mod.prime;
    return uint32_t(reduced);
}

template<unsigned N,unsigned K>
__global__ void scalar_update_montgomery(
    const uint32_t* __restrict__ left,
    const uint32_t* __restrict__ right_columns,
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,uint64_t batches,Montgomery mod) {
    uint64_t cells=batches*N*N;
    for(uint64_t cell=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
            cell<cells;cell+=uint64_t(gridDim.x)*blockDim.x) {
        uint64_t batch=cell/(N*N);
        unsigned within=unsigned(cell%(N*N));
        unsigned row=within/N,column=within%N;
        const uint32_t* left_row=left+(batch*N+row)*K;
        const uint32_t* right_column=right_columns+(batch*N+column)*K;
        uint32_t product=0;
#pragma unroll
        for(unsigned k=0;k<K;++k) {
            product+=montgomery_mul(left_row[k],right_column[k],mod);
            if(product>=mod.prime)product-=mod.prime;
        }
        output[cell]=input[cell]>=product?
            input[cell]-product:uint32_t(uint64_t(input[cell])+mod.prime-product);
    }
}

template<class Launch>
float time_kernel(unsigned warmup,unsigned iterations,Launch&& launch) {
    for(unsigned iteration=0;iteration<warmup;++iteration)launch();
    cuda_check(cudaDeviceSynchronize(),"synchronize warmup");
    cudaEvent_t start=nullptr,stop=nullptr;
    cuda_check(cudaEventCreate(&start),"create start event");
    cuda_check(cudaEventCreate(&stop),"create stop event");
    cuda_check(cudaEventRecord(start),"record start event");
    for(unsigned iteration=0;iteration<iterations;++iteration)launch();
    cuda_check(cudaEventRecord(stop),"record stop event");
    cuda_check(cudaEventSynchronize(stop),"synchronize stop event");
    float elapsed=0;
    cuda_check(cudaEventElapsedTime(&elapsed,start,stop),"measure elapsed time");
    cudaEventDestroy(stop);cudaEventDestroy(start);
    return elapsed/iterations;
}

template<class T>
T* device_allocate(size_t count,const char* operation) {
    T* pointer=nullptr;
    cuda_check(cudaMalloc(&pointer,count*sizeof(T)),operation);
    return pointer;
}

uint64_t checksum(const std::vector<uint8_t>& values) {
    uint64_t hash=UINT64_C(1469598103934665603);
    for(uint8_t value:values)hash=(hash^value)*UINT64_C(1099511628211);
    return hash;
}

struct CrtWork {
    uint64_t current_terms=0;
    uint64_t small_terms=0;
    unsigned maximum_small_images=0;
};

CrtWork crt_work() {
    constexpr std::array<uint32_t,4> current_primes={
        2147483647U,2147483629U,2147483587U,2147483579U};
    constexpr std::array<uint32_t,20> small_primes={
        251,241,239,233,229,227,223,211,199,197,
        193,191,181,179,173,167,163,157,151,149};
    auto required=[](unsigned bound,const auto& primes) {
        unsigned __int128 product=1;
        unsigned __int128 target=static_cast<unsigned __int128>(1)<<bound;
        for(unsigned count=0;count<primes.size();++count) {
            product*=primes[count];
            if(product>target)return count+1;
        }
        throw std::runtime_error("probe prime list does not cover query bound");
    };
    auto catalog=six_by_twenty_eight::build_catalog();
    CrtWork work;
    for(const auto& query:catalog.queries) {
        uint64_t terms=UINT64_C(1)<<(query.vertices/2-1);
        unsigned current=required(query.matching_bound_power,current_primes);
        unsigned small=required(query.matching_bound_power,small_primes);
        work.current_terms+=terms*current;
        work.small_terms+=terms*small;
        work.maximum_small_images=std::max(work.maximum_small_images,small);
    }
    if(work.current_terms!=UINT64_C(1063130234880))
        throw std::runtime_error("unexpected production CRT work census");
    return work;
}

template<unsigned N,unsigned K>
int run(const Options& options) {
    constexpr uint32_t LARGE_PRIME=2147483647U;
    size_t panels=size_t(options.batches)*N*K;
    size_t cells=size_t(options.batches)*N*N;
    std::vector<uint8_t> left(panels),right(panels),input(cells);
    uint64_t state=UINT64_C(0x9e3779b97f4a7c15);
    auto random=[&]() {
        state^=state>>12;state^=state<<25;state^=state>>27;
        return state*UINT64_C(2685821657736338717);
    };
    for(uint8_t& value:left)value=uint8_t(random()%options.small_prime);
    for(uint8_t& value:right)value=uint8_t(random()%options.small_prime);
    for(uint8_t& value:input)value=uint8_t(random()%options.small_prime);

    uint8_t *device_left=device_allocate<uint8_t>(panels,"allocate u8 left");
    uint8_t *device_right=device_allocate<uint8_t>(panels,"allocate u8 right");
    uint8_t *device_input=device_allocate<uint8_t>(cells,"allocate u8 input");
    uint8_t *device_tensor=device_allocate<uint8_t>(cells,"allocate tensor output");
    uint8_t *device_scalar=device_allocate<uint8_t>(cells,"allocate scalar output");
    cuda_check(cudaMemcpy(device_left,left.data(),panels,cudaMemcpyHostToDevice),"copy u8 left");
    cuda_check(cudaMemcpy(device_right,right.data(),panels,cudaMemcpyHostToDevice),"copy u8 right");
    cuda_check(cudaMemcpy(device_input,input.data(),cells,cudaMemcpyHostToDevice),"copy u8 input");

    Montgomery mod{LARGE_PRIME,0U-inverse_mod_2_32(LARGE_PRIME)};
    uint32_t montgomery_one=uint32_t((UINT64_C(1)<<32)%LARGE_PRIME);
    std::vector<uint32_t> left_mont(panels),right_mont(panels),input_mont(cells);
    for(uint32_t& value:left_mont)value=uint32_t(random()%LARGE_PRIME);
    for(uint32_t& value:right_mont)value=uint32_t(random()%LARGE_PRIME);
    for(uint32_t& value:input_mont)value=uint32_t(random()%LARGE_PRIME);
    for(uint32_t& value:left_mont)value=uint32_t(uint64_t(value)*montgomery_one%LARGE_PRIME);
    for(uint32_t& value:right_mont)value=uint32_t(uint64_t(value)*montgomery_one%LARGE_PRIME);
    for(uint32_t& value:input_mont)value=uint32_t(uint64_t(value)*montgomery_one%LARGE_PRIME);
    uint32_t *device_left_mont=device_allocate<uint32_t>(panels,"allocate Montgomery left");
    uint32_t *device_right_mont=device_allocate<uint32_t>(panels,"allocate Montgomery right");
    uint32_t *device_input_mont=device_allocate<uint32_t>(cells,"allocate Montgomery input");
    uint32_t *device_output_mont=device_allocate<uint32_t>(cells,"allocate Montgomery output");
    cuda_check(cudaMemcpy(device_left_mont,left_mont.data(),panels*sizeof(uint32_t),cudaMemcpyHostToDevice),"copy Montgomery left");
    cuda_check(cudaMemcpy(device_right_mont,right_mont.data(),panels*sizeof(uint32_t),cudaMemcpyHostToDevice),"copy Montgomery right");
    cuda_check(cudaMemcpy(device_input_mont,input_mont.data(),cells*sizeof(uint32_t),cudaMemcpyHostToDevice),"copy Montgomery input");

    int device=0,multiprocessors=0,major=0,minor=0;
    cuda_check(cudaGetDevice(&device),"get device");
    cuda_check(cudaDeviceGetAttribute(&multiprocessors,cudaDevAttrMultiProcessorCount,device),"get multiprocessor count");
    cuda_check(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device),"get compute capability major");
    cuda_check(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device),"get compute capability minor");
    if(major<8)throw std::runtime_error("integer m16n8k16 probe requires compute capability 8.0+");
    unsigned blocks=unsigned(multiprocessors)*8;
    uint32_t reciprocal=uint32_t((UINT64_C(1)<<32)/options.small_prime);
    auto tensor_launch=[&]() {
        tensor_update_u8<N,K><<<blocks,options.threads>>>(
            device_left,device_right,device_input,device_tensor,
            options.batches,options.small_prime,reciprocal);
    };
    auto scalar_launch=[&]() {
        scalar_update_u8<N,K><<<blocks,options.threads>>>(
            device_left,device_right,device_input,device_scalar,
            options.batches,options.small_prime,reciprocal);
    };
    auto montgomery_launch=[&]() {
        scalar_update_montgomery<N,K><<<blocks,options.threads>>>(
            device_left_mont,device_right_mont,device_input_mont,
            device_output_mont,options.batches,mod);
    };
    tensor_launch();scalar_launch();montgomery_launch();
    cuda_check(cudaGetLastError(),"launch correctness kernels");
    cuda_check(cudaDeviceSynchronize(),"synchronize correctness kernels");
    std::vector<uint8_t> tensor(cells),scalar(cells);
    std::vector<uint32_t> montgomery_output(cells);
    cuda_check(cudaMemcpy(tensor.data(),device_tensor,cells,cudaMemcpyDeviceToHost),"copy tensor output");
    cuda_check(cudaMemcpy(scalar.data(),device_scalar,cells,cudaMemcpyDeviceToHost),"copy scalar output");
    cuda_check(cudaMemcpy(montgomery_output.data(),device_output_mont,
        cells*sizeof(uint32_t),cudaMemcpyDeviceToHost),"copy Montgomery output");
    if(tensor!=scalar) {
        size_t mismatch=0;
        while(mismatch<cells&&tensor[mismatch]==scalar[mismatch])++mismatch;
        throw std::runtime_error("tensor/scalar mismatch at cell "+std::to_string(mismatch));
    }
    unsigned validation_batches=std::min(options.batches,4U);
    for(unsigned batch=0;batch<validation_batches;++batch)
        for(unsigned row=0;row<N;++row)
            for(unsigned column=0;column<N;++column) {
                uint32_t product=0;
                for(unsigned k=0;k<K;++k)
                    product+=uint32_t(left[(batch*N+row)*K+k])*
                        right[(batch*N+column)*K+k];
                uint8_t expected=uint8_t((uint32_t(input[(batch*N+row)*N+column])+
                    options.small_prime-product%options.small_prime)%options.small_prime);
                if(tensor[(batch*N+row)*N+column]!=expected)
                    throw std::runtime_error("CPU/tensor correctness mismatch");
                uint32_t montgomery_product=0;
                for(unsigned k=0;k<K;++k) {
                    montgomery_product+=montgomery_mul(
                        left_mont[(batch*N+row)*K+k],
                        right_mont[(batch*N+column)*K+k],mod);
                    if(montgomery_product>=LARGE_PRIME)
                        montgomery_product-=LARGE_PRIME;
                }
                uint32_t montgomery_expected=
                    input_mont[(batch*N+row)*N+column]>=montgomery_product?
                    input_mont[(batch*N+row)*N+column]-montgomery_product:
                    uint32_t(uint64_t(input_mont[(batch*N+row)*N+column])+
                        LARGE_PRIME-montgomery_product);
                if(montgomery_output[(batch*N+row)*N+column]!=montgomery_expected)
                    throw std::runtime_error("CPU/Montgomery correctness mismatch");
            }

    float tensor_ms=time_kernel(options.warmup,options.iterations,tensor_launch);
    float scalar_ms=time_kernel(options.warmup,options.iterations,scalar_launch);
    float montgomery_ms=time_kernel(options.warmup,options.iterations,montgomery_launch);
    cuda_check(cudaGetLastError(),"benchmark kernel status");
    CrtWork work=crt_work();
    double crt_multiplier=double(work.small_terms)/work.current_terms;
    double raw_speedup=montgomery_ms/tensor_ms;
    double adjusted_speedup=raw_speedup/crt_multiplier;
    double conservative_gate=2.0*crt_multiplier;
    std::printf(
        "HAFNIAN_INT8_BLOCK device_cc=%d.%d order=%u block_k=%u batches=%u "
        "iterations=%u small_prime=%u blocks=%u threads=%u "
        "tensor_ms=%.6f scalar_u8_ms=%.6f scalar_montgomery_ms=%.6f "
        "tensor_vs_u8=%.6f tensor_vs_montgomery=%.6f "
        "current_crt_terms=%" PRIu64 " small_crt_terms=%" PRIu64 " "
        "crt_work_multiplier=%.6f maximum_small_images=%u "
        "crt_adjusted_speedup=%.6f conservative_required_raw_speedup=%.6f "
        "checksum=%016" PRIx64 " gate=%s exact=OK\n",
        major,minor,N,K,options.batches,options.iterations,
        options.small_prime,blocks,options.threads,tensor_ms,scalar_ms,
        montgomery_ms,scalar_ms/tensor_ms,raw_speedup,
        work.current_terms,work.small_terms,crt_multiplier,
        work.maximum_small_images,adjusted_speedup,conservative_gate,
        checksum(tensor),raw_speedup>=conservative_gate?"PASS":"REJECT");

    cudaFree(device_output_mont);cudaFree(device_input_mont);
    cudaFree(device_right_mont);cudaFree(device_left_mont);
    cudaFree(device_scalar);cudaFree(device_tensor);cudaFree(device_input);
    cudaFree(device_right);cudaFree(device_left);
    return 0;
}

} // namespace

int main(int argc,char** argv) try {
    Options options=parse_options(argc,argv);
    if(!is_prime(options.small_prime))
        throw std::runtime_error("--prime must be prime");
    if(options.order==48) {
        if(options.block_k==8)return run<48,8>(options);
        if(options.block_k==16)return run<48,16>(options);
        return run<48,32>(options);
    }
    if(options.block_k==8)return run<64,8>(options);
    if(options.block_k==16)return run<64,16>(options);
    return run<64,32>(options);
} catch(const std::exception& error) {
    std::fprintf(stderr,"error: %s\n",error.what());
    return 2;
}
