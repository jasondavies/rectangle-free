#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#ifndef HAFNIAN_DIRECT_SQRT_RECURRENCE
#define HAFNIAN_DIRECT_SQRT_RECURRENCE 1
#endif

struct HafnianMontgomery {
    uint32_t p=0,negative_inverse=0,one=0;
};

struct HafnianMersenne31 {
    static constexpr uint32_t p=2147483647U;
    static constexpr uint32_t one=1;
};

constexpr uint32_t hafnian_const_inverse_mod_2_32(uint32_t odd) {
    uint32_t value=odd;
    for(unsigned i=0;i<5;++i)value*=2U-odd*value;
    return value;
}

template<uint32_t P>
struct HafnianMontgomeryConstant {
    static_assert(P&1);
    static constexpr uint32_t p=P;
    static constexpr uint32_t negative_inverse=0U-hafnian_const_inverse_mod_2_32(P);
    static constexpr uint32_t one=uint32_t((UINT64_C(1)<<32)%P);
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

template<uint32_t P>
inline uint32_t hafnian_host_montgomery_mul(
    uint32_t a,uint32_t b,HafnianMontgomeryConstant<P>) {
    uint64_t product=uint64_t(a)*b;
    uint32_t multiplier=uint32_t(product)*HafnianMontgomeryConstant<P>::negative_inverse;
    uint64_t reduced=(product+uint64_t(multiplier)*P)>>32;
    if(reduced>=P)reduced-=P;
    return uint32_t(reduced);
}

template<uint32_t P>
inline uint32_t hafnian_host_montgomery_power(
    uint32_t a,uint64_t exponent,HafnianMontgomeryConstant<P> mod) {
    uint32_t result=mod.one;
    while(exponent) {
        if(exponent&1)result=hafnian_host_montgomery_mul(result,a,mod);
        a=hafnian_host_montgomery_mul(a,a,mod);
        exponent>>=1;
    }
    return result;
}

inline uint32_t hafnian_host_montgomery_mul(
    uint32_t a,uint32_t b,HafnianMersenne31) {
    const uint64_t product=uint64_t(a)*b;
    uint64_t reduced=(product&HafnianMersenne31::p)+(product>>31);
    if(reduced>=HafnianMersenne31::p)reduced-=HafnianMersenne31::p;
    return uint32_t(reduced);
}

inline uint32_t hafnian_host_montgomery_power(
    uint32_t a,uint64_t exponent,HafnianMersenne31 mod) {
    uint32_t result=1;
    while(exponent) {
        if(exponent&1)result=hafnian_host_montgomery_mul(result,a,mod);
        a=hafnian_host_montgomery_mul(a,a,mod);exponent>>=1;
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

template<uint32_t P>
__device__ __forceinline__ uint32_t hafnian_montgomery_mul(
    uint32_t a,uint32_t b,HafnianMontgomeryConstant<P>) {
    uint64_t product=uint64_t(a)*b;
    uint32_t multiplier=uint32_t(product)*HafnianMontgomeryConstant<P>::negative_inverse;
    uint64_t reduced=(product+uint64_t(multiplier)*P)>>32;
    if(reduced>=P)reduced-=P;
    return uint32_t(reduced);
}

__device__ __forceinline__ uint32_t hafnian_mul(
    uint32_t a,uint32_t b,HafnianMontgomery mod) {
    return hafnian_montgomery_mul(a,b,mod);
}

template<uint32_t P>
__device__ __forceinline__ uint32_t hafnian_mul(
    uint32_t a,uint32_t b,HafnianMontgomeryConstant<P> mod) {
    return hafnian_montgomery_mul(a,b,mod);
}

__device__ __forceinline__ uint32_t hafnian_mul(
    uint32_t a,uint32_t b,HafnianMersenne31) {
    const uint64_t product=uint64_t(a)*b;
    uint64_t reduced=(product&HafnianMersenne31::p)+(product>>31);
    if(reduced>=HafnianMersenne31::p)reduced-=HafnianMersenne31::p;
    return uint32_t(reduced);
}

__device__ inline uint32_t hafnian_power(
    uint32_t a,uint32_t exponent,HafnianMersenne31 mod) {
    if(exponent!=HafnianMersenne31::p-2) {
        uint32_t result=1;
        while(exponent) {
            if(exponent&1)result=hafnian_mul(result,a,mod);
            a=hafnian_mul(a,a,mod);exponent>>=1;
        }
        return result;
    }
    auto square_n=[&](uint32_t value,unsigned count) {
#pragma unroll
        for(unsigned i=0;i<count;++i)value=hafnian_mul(value,value,mod);
        return value;
    };
    const uint32_t a4=square_n(a,2);
    const uint32_t a5=hafnian_mul(a4,a,mod);
    const uint32_t a10=hafnian_mul(a5,a5,mod);
    const uint32_t a15=hafnian_mul(a5,a10,mod);
    const uint32_t a120=square_n(a15,3);
    const uint32_t a125=hafnian_mul(a5,a120,mod);
    const uint32_t a250=hafnian_mul(a125,a125,mod);
    const uint32_t a255=hafnian_mul(a5,a250,mod);
    const uint32_t a65535=hafnian_mul(square_n(a255,8),a255,mod);
    const uint32_t a16777215=hafnian_mul(square_n(a65535,8),a255,mod);
    return hafnian_mul(square_n(a16777215,7),a125,mod);
}

// All device inversions use exponent P-2.  The four production exponents have
// fixed 37--38-multiply addition chains; other compile-time primes retain the
// generic radix-4 path below.
template<uint32_t P>
__device__ inline uint32_t hafnian_power(
    uint32_t a,uint32_t exponent,HafnianMontgomeryConstant<P> mod) {
    if(exponent!=P-2) {
        uint32_t result=mod.one;
        while(exponent) {
            if(exponent&1)result=hafnian_mul(result,a,mod);
            a=hafnian_mul(a,a,mod);
            exponent>>=1;
        }
        return result;
    }
    if constexpr(P==2147483647U||P==2147483629U||
            P==2147483587U||P==2147483579U) {
        auto square_n=[&](uint32_t value,unsigned count) {
#pragma unroll
            for(unsigned i=0;i<count;++i)value=hafnian_mul(value,value,mod);
            return value;
        };
        uint32_t a255=0,tail=0;
        if constexpr(P==2147483647U) {
            const uint32_t a4=square_n(a,2);
            const uint32_t a5=hafnian_mul(a4,a,mod);
            const uint32_t a10=hafnian_mul(a5,a5,mod);
            const uint32_t a15=hafnian_mul(a5,a10,mod);
            const uint32_t a120=square_n(a15,3);
            const uint32_t a125=hafnian_mul(a5,a120,mod);
            const uint32_t a250=hafnian_mul(a125,a125,mod);
            a255=hafnian_mul(a5,a250,mod);tail=a125;
        } else if constexpr(P==2147483629U) {
            const uint32_t a2=hafnian_mul(a,a,mod);
            const uint32_t a4=hafnian_mul(a2,a2,mod);
            const uint32_t a6=hafnian_mul(a2,a4,mod);
            const uint32_t a7=hafnian_mul(a,a6,mod);
            const uint32_t a9=hafnian_mul(a2,a7,mod);
            const uint32_t a16=hafnian_mul(a7,a9,mod);
            const uint32_t a25=hafnian_mul(a9,a16,mod);
            const uint32_t a41=hafnian_mul(a16,a25,mod);
            const uint32_t a82=hafnian_mul(a41,a41,mod);
            const uint32_t a107=hafnian_mul(a25,a82,mod);
            const uint32_t a214=hafnian_mul(a107,a107,mod);
            a255=hafnian_mul(a41,a214,mod);tail=a107;
        } else if constexpr(P==2147483587U) {
            const uint32_t a4=square_n(a,2);
            const uint32_t a5=hafnian_mul(a4,a,mod);
            const uint32_t a10=hafnian_mul(a5,a5,mod);
            const uint32_t a15=hafnian_mul(a5,a10,mod);
            const uint32_t a60=square_n(a15,2);
            const uint32_t a65=hafnian_mul(a5,a60,mod);
            const uint32_t a130=hafnian_mul(a65,a65,mod);
            const uint32_t a195=hafnian_mul(a65,a130,mod);
            a255=hafnian_mul(a60,a195,mod);tail=a65;
        } else {
            const uint32_t a2=hafnian_mul(a,a,mod);
            const uint32_t a3=hafnian_mul(a,a2,mod);
            const uint32_t a24=square_n(a3,3);
            const uint32_t a27=hafnian_mul(a3,a24,mod);
            const uint32_t a54=hafnian_mul(a27,a27,mod);
            const uint32_t a57=hafnian_mul(a3,a54,mod);
            const uint32_t a228=square_n(a57,2);
            a255=hafnian_mul(a27,a228,mod);tail=a57;
        }
        const uint32_t a65535=hafnian_mul(square_n(a255,8),a255,mod);
        const uint32_t a16777215=hafnian_mul(square_n(a65535,8),a255,mod);
        return hafnian_mul(square_n(a16777215,7),tail,mod);
    }
    constexpr uint32_t INVERSE_EXPONENT=P-2;
    uint32_t a2=hafnian_mul(a,a,mod);
    uint32_t a3=hafnian_mul(a2,a,mod);
    uint32_t result=a;
#pragma unroll
    for(int shift=28;shift>=0;shift-=2) {
        result=hafnian_mul(result,result,mod);
        result=hafnian_mul(result,result,mod);
        constexpr uint32_t MASK=3;
        uint32_t digit=(INVERSE_EXPONENT>>shift)&MASK;
        if(digit==1)result=hafnian_mul(result,a,mod);
        else if(digit==2)result=hafnian_mul(result,a2,mod);
        else if(digit==3)result=hafnian_mul(result,a3,mod);
    }
    return result;
}

template<class Mod>
__device__ inline uint32_t hafnian_power(
    uint32_t a,uint32_t exponent,Mod mod) {
    uint32_t result=mod.one;
    while(exponent) {
        if(exponent&1)result=hafnian_mul(result,a,mod);
        a=hafnian_mul(a,a,mod);
        exponent>>=1;
    }
    return result;
}

template<unsigned N,class Mod=HafnianMontgomery,bool GRAY_ORDER=false>
__global__ void hafnian_terms_kernel(
    const uint8_t* __restrict__ adjacency,uint64_t begin,uint64_t end,
    Mod mod,const uint32_t* __restrict__ inverse_small,
    uint32_t* __restrict__ block_sums) {
    static_assert(N%2==0&&N<=64);
    constexpr unsigned HALF=N/2;
    constexpr unsigned POLY_STRIDE=HALF+1;
    constexpr unsigned MATRIX_STRIDE=N+1;
    extern __shared__ uint32_t shared[];
    uint32_t* matrix=shared;
    uint32_t* poly=matrix+N*MATRIX_STRIDE;
    uint32_t* factors=poly+(N+1)*POLY_STRIDE;
    // Hessenberg elimination and characteristic-polynomial construction do
    // not overlap, so these two work arrays can share storage.  The saved N
    // words exactly pay for the conflict-free N+1 matrix stride.
    uint32_t* char_factors=factors;
    uint32_t* scalar=factors+N;
    uint32_t local_sum=0;

    for(uint64_t term=begin+blockIdx.x;term<end;term+=gridDim.x) {
        const uint64_t signs=GRAY_ORDER?term^(term>>1):term;
        for(unsigned index=threadIdx.x;index<N*N;index+=blockDim.x) {
            unsigned row=index/N;
            unsigned column=index%N;
            unsigned edge=column%HALF;
            unsigned paired=column<HALF?column+HALF:column-HALF;
            bool positive=edge==0||(signs&(UINT64_C(1)<<(edge-1)));
            matrix[row*MATRIX_STRIDE+column]=
                adjacency[row*N+paired]?(positive?mod.one:mod.p-mod.one):0;
        }
        __syncthreads();

        for(unsigned column=0;column+2<N;++column) {
            if(threadIdx.x==0) {
                unsigned pivot=column+1;
                while(pivot<N&&matrix[pivot*MATRIX_STRIDE+column]==0)++pivot;
                scalar[0]=pivot;
            }
            __syncthreads();
            unsigned pivot=scalar[0];
            if(pivot==N)continue;
            if(pivot!=column+1) {
                for(unsigned j=threadIdx.x;j<N;j+=blockDim.x) {
                    uint32_t temporary=matrix[pivot*MATRIX_STRIDE+j];
                    matrix[pivot*MATRIX_STRIDE+j]=matrix[(column+1)*MATRIX_STRIDE+j];
                    matrix[(column+1)*MATRIX_STRIDE+j]=temporary;
                }
                __syncthreads();
                for(unsigned i=threadIdx.x;i<N;i+=blockDim.x) {
                    uint32_t temporary=matrix[i*MATRIX_STRIDE+pivot];
                    matrix[i*MATRIX_STRIDE+pivot]=matrix[i*MATRIX_STRIDE+column+1];
                    matrix[i*MATRIX_STRIDE+column+1]=temporary;
                }
                __syncthreads();
            }
            if(threadIdx.x==0)
                scalar[1]=hafnian_power(
                    matrix[(column+1)*MATRIX_STRIDE+column],mod.p-2,mod);
            __syncthreads();
            for(unsigned row=column+2+threadIdx.x;row<N;row+=blockDim.x)
                factors[row]=hafnian_mul(
                    matrix[row*MATRIX_STRIDE+column],scalar[1],mod);
            __syncthreads();
            unsigned width=N-column;
            unsigned cells=(N-column-2)*width;
            for(unsigned item=threadIdx.x;item<cells;item+=blockDim.x) {
                unsigned row=column+2+item/width;
                unsigned j=column+item%width;
                matrix[row*MATRIX_STRIDE+j]=hafnian_sub_mod(
                    matrix[row*MATRIX_STRIDE+j],hafnian_mul(
                        factors[row],matrix[(column+1)*MATRIX_STRIDE+j],mod),mod.p);
            }
            __syncthreads();
            for(unsigned row=threadIdx.x;row<N;row+=blockDim.x) {
                uint32_t sum=0;
                for(unsigned eliminated=column+2;eliminated<N;++eliminated)
                    sum=hafnian_add_mod(sum,hafnian_mul(
                        factors[eliminated],matrix[row*MATRIX_STRIDE+eliminated],mod),mod.p);
                matrix[row*MATRIX_STRIDE+column+1]=hafnian_add_mod(
                    matrix[row*MATRIX_STRIDE+column+1],sum,mod.p);
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
                    product=hafnian_mul(
                        product,matrix[subrow*MATRIX_STRIDE+subrow-1],mod);
                    char_factors[distance]=hafnian_mul(
                        product,matrix[(size-distance-1)*MATRIX_STRIDE+size-1],mod);
                }
            }
            __syncthreads();
            for(unsigned k=threadIdx.x;k<=min(size,HALF);k+=blockDim.x) {
                uint32_t value=k<=size-1?poly[(size-1)*POLY_STRIDE+k]:0;
                if(k)value=hafnian_sub_mod(value,hafnian_mul(
                    matrix[(size-1)*MATRIX_STRIDE+size-1],
                    poly[(size-1)*POLY_STRIDE+k-1],mod),mod.p);
                for(unsigned distance=1;distance<size&&distance+1<=k;++distance)
                    value=hafnian_sub_mod(value,hafnian_mul(
                        char_factors[distance],poly[(size-distance-1)*POLY_STRIDE+k-distance-1],mod),mod.p);
                poly[size*POLY_STRIDE+k]=value;
            }
            __syncthreads();
        }

        if(threadIdx.x==0) {
            uint32_t coefficients[HALF+1]{};
            coefficients[0]=mod.one;
#if HAFNIAN_DIRECT_SQRT_RECURRENCE
            // If P(z)=det(I-zK) and Q(z)=P(z)^(-1/2), then
            // 2 P Q' + P' Q = 0.  This computes the required coefficient
            // directly and avoids materialising the traces first.
            for(unsigned degree=1;degree<=HALF;++degree) {
                uint32_t sum=0;
                for(unsigned k=1;k<=degree;++k) {
                    const uint32_t factor=uint32_t(
                        uint64_t(2*degree-k)*mod.one%mod.p);
                    sum=hafnian_add_mod(sum,hafnian_mul(hafnian_mul(
                        poly[N*POLY_STRIDE+k],coefficients[degree-k],mod),
                        factor,mod),mod.p);
                }
                coefficients[degree]=hafnian_neg_mod(hafnian_mul(hafnian_mul(
                    sum,inverse_small[2],mod),inverse_small[degree],mod),mod.p);
            }
#else
            uint32_t traces[HALF+1]{};
            for(unsigned k=1;k<=HALF;++k) {
                uint32_t value=hafnian_mul(
                    uint32_t(uint64_t(k)*mod.one%mod.p),poly[N*POLY_STRIDE+k],mod);
                for(unsigned j=1;j<k;++j)value=hafnian_add_mod(value,
                    hafnian_mul(poly[N*POLY_STRIDE+j],traces[k-j],mod),mod.p);
                traces[k]=hafnian_neg_mod(value,mod.p);
            }
            for(unsigned degree=1;degree<=HALF;++degree) {
                uint32_t sum=0;
                for(unsigned k=1;k<=degree;++k)
                    sum=hafnian_add_mod(sum,hafnian_mul(
                        hafnian_mul(traces[k],inverse_small[2],mod),
                        coefficients[degree-k],mod),mod.p);
                coefficients[degree]=hafnian_mul(sum,inverse_small[degree],mod);
            }
#endif
            unsigned negatives=(HALF-1)-__popcll(signs);
            uint32_t contribution=negatives&1?
                hafnian_neg_mod(coefficients[HALF],mod.p):coefficients[HALF];
            local_sum=hafnian_add_mod(local_sum,contribution,mod.p);
        }
        __syncthreads();
    }
    if(threadIdx.x==0)
        block_sums[blockIdx.x]=hafnian_mul(local_sum,1,mod);
}

template<unsigned N>
constexpr size_t hafnian_shared_bytes() {
    constexpr unsigned HALF=N/2;
    constexpr unsigned MATRIX_STRIDE=N+1;
    return (N*MATRIX_STRIDE+(N+1)*(HALF+1)+N+4)*sizeof(uint32_t);
}

template<unsigned N,class Mod=HafnianMontgomery>
inline unsigned hafnian_recommended_blocks(
    unsigned threads,int multiprocessors,unsigned residency_waves=2) {
    constexpr size_t shared_bytes=hafnian_shared_bytes<N>();
    hafnian_cuda_check(cudaFuncSetAttribute(hafnian_terms_kernel<N,Mod>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared_bytes)),
        "set hafnian dynamic shared memory");
    int active_blocks=0;
    hafnian_cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,hafnian_terms_kernel<N,Mod>,threads,shared_bytes),
        "compute hafnian kernel occupancy");
    if(active_blocks<=0||multiprocessors<=0||!residency_waves)
        throw std::runtime_error("hafnian kernel has zero launch occupancy");
    return unsigned(multiprocessors)*unsigned(active_blocks)*residency_waves;
}
