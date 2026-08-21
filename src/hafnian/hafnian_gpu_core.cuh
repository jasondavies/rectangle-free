#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

struct HafnianMontgomery {
    uint32_t p=0,negative_inverse=0,one=0;
};

inline void hafnian_cuda_check(cudaError_t status,const char* operation) {
    if(status!=cudaSuccess)
        throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}

inline uint32_t hafnian_inverse_mod_2_32(uint32_t odd) {
    uint32_t value=odd;
    for(unsigned i=0;i<5;++i)value*=2U-odd*value;
    return value;
}

inline uint32_t hafnian_host_montgomery_mul(
    uint32_t a,uint32_t b,const HafnianMontgomery& mod) {
    uint64_t product=uint64_t(a)*b;
    uint32_t multiplier=uint32_t(product)*mod.negative_inverse;
    uint64_t reduced=(product+uint64_t(multiplier)*mod.p)>>32;
    if(reduced>=mod.p)reduced-=mod.p;
    return uint32_t(reduced);
}

inline uint32_t hafnian_host_montgomery_power(
    uint32_t a,uint64_t exponent,const HafnianMontgomery& mod) {
    uint32_t result=mod.one;
    while(exponent) {
        if(exponent&1)result=hafnian_host_montgomery_mul(result,a,mod);
        a=hafnian_host_montgomery_mul(a,a,mod);
        exponent>>=1;
    }
    return result;
}

__device__ __forceinline__ uint32_t hafnian_add_mod(uint32_t a,uint32_t b,uint32_t p) {
    uint32_t value=a+b;
    if(value>=p||value<a)value-=p;
    return value;
}

__device__ __forceinline__ uint32_t hafnian_sub_mod(uint32_t a,uint32_t b,uint32_t p) {
    return a>=b?a-b:uint32_t(uint64_t(a)+p-b);
}

__device__ __forceinline__ uint32_t hafnian_neg_mod(uint32_t a,uint32_t p) {
    return a?p-a:0;
}

__device__ __forceinline__ uint32_t hafnian_montgomery_mul(
    uint32_t a,uint32_t b,HafnianMontgomery mod) {
    uint64_t product=uint64_t(a)*b;
    uint32_t multiplier=uint32_t(product)*mod.negative_inverse;
    uint64_t reduced=(product+uint64_t(multiplier)*mod.p)>>32;
    if(reduced>=mod.p)reduced-=mod.p;
    return uint32_t(reduced);
}

__device__ inline uint32_t hafnian_montgomery_power(
    uint32_t a,uint32_t exponent,HafnianMontgomery mod) {
    uint32_t result=mod.one;
    while(exponent) {
        if(exponent&1)result=hafnian_montgomery_mul(result,a,mod);
        a=hafnian_montgomery_mul(a,a,mod);
        exponent>>=1;
    }
    return result;
}

template<unsigned N>
__global__ void hafnian_terms_kernel(
    const uint8_t* __restrict__ adjacency,uint64_t begin,uint64_t end,
    HafnianMontgomery mod,const uint32_t* __restrict__ inverse_small,
    uint32_t* __restrict__ block_sums) {
    static_assert(N%2==0&&N<=62);
    constexpr unsigned HALF=N/2;
    constexpr unsigned POLY_STRIDE=HALF+1;
    extern __shared__ uint32_t shared[];
    uint32_t* matrix=shared;
    uint32_t* poly=matrix+N*N;
    uint32_t* factors=poly+(N+1)*POLY_STRIDE;
    uint32_t* char_factors=factors+N;
    uint32_t* scalar=char_factors+N;
    uint32_t local_sum=0;

    for(uint64_t term=begin+blockIdx.x;term<end;term+=gridDim.x) {
        for(unsigned index=threadIdx.x;index<N*N;index+=blockDim.x) {
            unsigned row=index/N;
            unsigned column=index%N;
            unsigned edge=column%HALF;
            unsigned paired=column<HALF?column+HALF:column-HALF;
            bool positive=edge==0||(term&(UINT64_C(1)<<(edge-1)));
            matrix[index]=adjacency[row*N+paired]?(positive?mod.one:mod.p-mod.one):0;
        }
        __syncthreads();

        for(unsigned column=0;column+2<N;++column) {
            if(threadIdx.x==0) {
                unsigned pivot=column+1;
                while(pivot<N&&matrix[pivot*N+column]==0)++pivot;
                scalar[0]=pivot;
            }
            __syncthreads();
            unsigned pivot=scalar[0];
            if(pivot==N)continue;
            if(pivot!=column+1) {
                for(unsigned j=threadIdx.x;j<N;j+=blockDim.x) {
                    uint32_t temporary=matrix[pivot*N+j];
                    matrix[pivot*N+j]=matrix[(column+1)*N+j];
                    matrix[(column+1)*N+j]=temporary;
                }
                __syncthreads();
                for(unsigned i=threadIdx.x;i<N;i+=blockDim.x) {
                    uint32_t temporary=matrix[i*N+pivot];
                    matrix[i*N+pivot]=matrix[i*N+column+1];
                    matrix[i*N+column+1]=temporary;
                }
                __syncthreads();
            }
            if(threadIdx.x==0)
                scalar[1]=hafnian_montgomery_power(matrix[(column+1)*N+column],mod.p-2,mod);
            __syncthreads();
            for(unsigned row=column+2+threadIdx.x;row<N;row+=blockDim.x)
                factors[row]=hafnian_montgomery_mul(matrix[row*N+column],scalar[1],mod);
            __syncthreads();
            unsigned width=N-column;
            unsigned cells=(N-column-2)*width;
            for(unsigned item=threadIdx.x;item<cells;item+=blockDim.x) {
                unsigned row=column+2+item/width;
                unsigned j=column+item%width;
                matrix[row*N+j]=hafnian_sub_mod(matrix[row*N+j],
                    hafnian_montgomery_mul(factors[row],matrix[(column+1)*N+j],mod),mod.p);
            }
            __syncthreads();
            for(unsigned row=threadIdx.x;row<N;row+=blockDim.x) {
                uint32_t sum=0;
                for(unsigned eliminated=column+2;eliminated<N;++eliminated)
                    sum=hafnian_add_mod(sum,hafnian_montgomery_mul(
                        factors[eliminated],matrix[row*N+eliminated],mod),mod.p);
                matrix[row*N+column+1]=hafnian_add_mod(matrix[row*N+column+1],sum,mod.p);
            }
            __syncthreads();
        }

        for(unsigned index=threadIdx.x;index<(N+1)*POLY_STRIDE;index+=blockDim.x)
            poly[index]=0;
        if(threadIdx.x==0)poly[0]=mod.one;
        __syncthreads();
        for(unsigned size=1;size<=N;++size) {
            if(threadIdx.x==0) {
                uint32_t product=mod.one;
                for(unsigned distance=1;distance<size;++distance) {
                    unsigned subrow=size-distance;
                    product=hafnian_montgomery_mul(
                        product,matrix[subrow*N+subrow-1],mod);
                    char_factors[distance]=hafnian_montgomery_mul(
                        product,matrix[(size-distance-1)*N+size-1],mod);
                }
            }
            __syncthreads();
            for(unsigned k=threadIdx.x;k<=min(size,HALF);k+=blockDim.x) {
                uint32_t value=k<=size-1?poly[(size-1)*POLY_STRIDE+k]:0;
                if(k)value=hafnian_sub_mod(value,hafnian_montgomery_mul(
                    matrix[(size-1)*N+size-1],poly[(size-1)*POLY_STRIDE+k-1],mod),mod.p);
                for(unsigned distance=1;distance<size&&distance+1<=k;++distance)
                    value=hafnian_sub_mod(value,hafnian_montgomery_mul(
                        char_factors[distance],poly[(size-distance-1)*POLY_STRIDE+k-distance-1],mod),mod.p);
                poly[size*POLY_STRIDE+k]=value;
            }
            __syncthreads();
        }

        if(threadIdx.x==0) {
            uint32_t traces[HALF+1]{};
            uint32_t coefficients[HALF+1]{};
            coefficients[0]=mod.one;
            for(unsigned k=1;k<=HALF;++k) {
                uint32_t value=hafnian_montgomery_mul(
                    uint32_t(uint64_t(k)*mod.one%mod.p),poly[N*POLY_STRIDE+k],mod);
                for(unsigned j=1;j<k;++j)value=hafnian_add_mod(value,
                    hafnian_montgomery_mul(poly[N*POLY_STRIDE+j],traces[k-j],mod),mod.p);
                traces[k]=hafnian_neg_mod(value,mod.p);
            }
            for(unsigned degree=1;degree<=HALF;++degree) {
                uint32_t sum=0;
                for(unsigned k=1;k<=degree;++k)
                    sum=hafnian_add_mod(sum,hafnian_montgomery_mul(
                        hafnian_montgomery_mul(traces[k],inverse_small[2],mod),
                        coefficients[degree-k],mod),mod.p);
                coefficients[degree]=hafnian_montgomery_mul(sum,inverse_small[degree],mod);
            }
            unsigned negatives=(HALF-1)-__popcll(term);
            uint32_t contribution=negatives&1?
                hafnian_neg_mod(coefficients[HALF],mod.p):coefficients[HALF];
            local_sum=hafnian_add_mod(local_sum,contribution,mod.p);
        }
        __syncthreads();
    }
    if(threadIdx.x==0)
        block_sums[blockIdx.x]=hafnian_montgomery_mul(local_sum,1,mod);
}

template<unsigned N>
constexpr size_t hafnian_shared_bytes() {
    constexpr unsigned HALF=N/2;
    return (N*N+(N+1)*(HALF+1)+2*N+4)*sizeof(uint32_t);
}
