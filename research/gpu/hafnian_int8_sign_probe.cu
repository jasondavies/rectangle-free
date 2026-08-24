// Complete exact small-prime hafnian sign-term prototype.
//
// A panel of finite-field Gauss similarities is factored sequentially with
// packed DP4A dot products.  Its two trailing updates are then applied with
// exact INT8 tensor MMA.  The kernel pipeline includes matrix generation,
// pivoted Hessenberg reduction, La Budde characteristic-polynomial recovery,
// Newton extraction, and the sign contribution.

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/hafnian/six_by_twenty_eight_catalog.hpp"

namespace {

constexpr unsigned PANEL=32;

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
    unsigned order=64,batches=4096,iterations=10,warmup=2,query=UINT32_MAX;
    unsigned panel_limit=UINT32_MAX;
    unsigned panel_width=PANEL;
    uint32_t prime=251;
    uint64_t begin=0;
    bool scalar_updates=false;
    bool fused=false;
    bool work_census=false;
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
        else if(argument=="--batches")options.batches=unsigned(number(take()));
        else if(argument=="--iterations")options.iterations=unsigned(number(take()));
        else if(argument=="--warmup")options.warmup=unsigned(number(take()));
        else if(argument=="--query")options.query=unsigned(number(take()));
        else if(argument=="--prime")options.prime=uint32_t(number(take()));
        else if(argument=="--begin")options.begin=number(take());
        else if(argument=="--scalar-updates")options.scalar_updates=true;
        else if(argument=="--fused")options.fused=true;
        else if(argument=="--work-census")options.work_census=true;
        else if(argument=="--panel-limit")options.panel_limit=unsigned(number(take()));
        else if(argument=="--panel-width")options.panel_width=unsigned(number(take()));
        else throw std::runtime_error(
            "usage: hafnian_int8_sign_probe [--order 48|54|60|64] [--query Q] "
            "[--batches N] [--iterations N] [--warmup N] [--prime P] [--begin N]");
    }
    if((options.order<48||options.order>64||(options.order&1))||
            !options.batches||!options.iterations||!options.panel_width||
            options.panel_width>PANEL||options.prime<=32||options.prime>251)
        throw std::runtime_error("invalid probe configuration");
    return options;
}

__device__ __forceinline__ uint32_t add_mod(uint32_t,uint32_t,uint32_t);
__device__ __forceinline__ uint32_t sub_mod(uint32_t,uint32_t,uint32_t);
__device__ __forceinline__ uint32_t mul_mod(uint32_t,uint32_t,uint32_t);
__device__ __forceinline__ uint32_t reduce_u32(uint32_t,uint32_t,uint32_t);
__device__ __forceinline__ void mma_u8_16x8x16(
    uint32_t,uint32_t,uint32_t,int32_t (&)[4]);

template<unsigned M,unsigned NOUT,unsigned K,bool SUBTRACT>
__device__ __forceinline__ void warp_tensor_product_u8(
    const uint8_t* left,const uint8_t* right_columns,const uint8_t* input,
    uint8_t* output,uint32_t prime,uint32_t reciprocal) {
    static_assert(M%16==0&&NOUT%8==0&&K%16==0);
    unsigned lane=threadIdx.x&31U,group=lane>>2,thread_in_group=lane&3U;
    for(unsigned row_base=0;row_base<M;row_base+=16)
        for(unsigned column_base=0;column_base<NOUT;column_base+=8) {
            int32_t result[4]={0,0,0,0};
#pragma unroll
            for(unsigned k_tile=0;k_tile<K;k_tile+=16) {
                unsigned k=k_tile+thread_in_group*4;
                uint32_t a0=*reinterpret_cast<const uint32_t*>(
                    left+(row_base+group)*K+k);
                uint32_t a1=*reinterpret_cast<const uint32_t*>(
                    left+(row_base+group+8)*K+k);
                uint32_t b0=*reinterpret_cast<const uint32_t*>(
                    right_columns+(column_base+group)*K+k);
                mma_u8_16x8x16(a0,a1,b0,result);
            }
            unsigned columns[2]={
                column_base+thread_in_group*2,column_base+thread_in_group*2+1};
            unsigned rows[2]={row_base+group,row_base+group+8};
#pragma unroll
            for(unsigned ri=0;ri<2;++ri)for(unsigned ci=0;ci<2;++ci) {
                unsigned offset=rows[ri]*NOUT+columns[ci];
                uint32_t product=reduce_u32(uint32_t(result[ri*2+ci]),prime,reciprocal);
                if constexpr(SUBTRACT)
                    output[offset]=uint8_t(sub_mod(input[offset],product,prime));
                else output[offset]=uint8_t(product);
            }
            __syncwarp();
        }
}

template<unsigned M,unsigned NOUT,unsigned K,bool SUBTRACT>
__device__ __forceinline__ void warp_scalar_product_u8(
    const uint8_t* left,const uint8_t* right_columns,const uint8_t* input,
    uint8_t* output,uint32_t prime) {
    unsigned lane=threadIdx.x&31U;
    for(unsigned cell=lane;cell<M*NOUT;cell+=32) {
        unsigned row=cell/NOUT,column=cell%NOUT;
        uint32_t product=0;
        for(unsigned k=0;k<K;++k)product+=
            uint32_t(left[row*K+k])*right_columns[column*K+k];
        product%=prime;
        if constexpr(SUBTRACT)output[cell]=uint8_t(sub_mod(input[cell],product,prime));
        else output[cell]=uint8_t(product);
    }
    __syncwarp();
}

template<unsigned N,unsigned PN,unsigned WARPS>
__global__ void fused_sign_terms(
    const uint8_t* __restrict__ adjacency,uint8_t* __restrict__ outputs,
    uint8_t* __restrict__ debug_matrices,uint8_t* __restrict__ debug_factors,
    const uint8_t* __restrict__ inverse_table,uint64_t begin,
    unsigned batches,unsigned panel_width,unsigned panel_limit,
    uint32_t prime,uint32_t reciprocal,
    bool scalar_updates,bool debug_outputs) {
    constexpr unsigned HALF=N/2,STRIDE=HALF+1;
    constexpr unsigned MATRIX_ENTRIES=PN*PN;
    constexpr unsigned PANEL_ENTRIES=PN*PANEL;
    constexpr unsigned PER_WARP=MATRIX_ENTRIES+3*PANEL_ENTRIES+PANEL*PANEL+PANEL;
    unsigned lane=threadIdx.x&31U,warp_in_block=threadIdx.x>>5;
    unsigned global_warp=blockIdx.x*WARPS+warp_in_block;
    unsigned warp_stride=gridDim.x*WARPS;
    extern __shared__ uint8_t shared[];
    uint8_t* local=shared+warp_in_block*PER_WARP;
    uint8_t* matrix=local;
    uint8_t* f_row=matrix+MATRIX_ENTRIES;
    uint8_t* f=f_row+PANEL_ENTRIES;
    uint8_t* scratch=f+PANEL_ENTRIES;
    uint8_t* t=scratch+PANEL_ENTRIES;
    uint8_t* s=t+PANEL*PANEL;

    for(unsigned batch=global_warp;batch<batches;batch+=warp_stride) {
        uint64_t term=begin+batch;
        for(unsigned index=lane;index<MATRIX_ENTRIES;index+=32) {
            unsigned row=index/PN,column=index%PN;
            uint8_t value=0;
            if(row<N&&column<N) {
                unsigned edge=column%HALF;
                unsigned paired=column<HALF?column+HALF:column-HALF;
                bool positive=edge==0||(term&(UINT64_C(1)<<(edge-1)));
                if(adjacency[row*N+paired])value=uint8_t(positive?1:prime-1);
            }
            matrix[index]=value;
        }
        __syncwarp();

        unsigned panel_index=0;
        for(unsigned first=0;first+2<N&&panel_index<panel_limit;
                first+=panel_width,++panel_index) {
            for(unsigned index=lane;index<PANEL_ENTRIES;index+=32) {
                f[index]=0;f_row[index]=0;
            }
            for(unsigned index=lane;index<PANEL*PANEL;index+=32)t[index]=0;
            __syncwarp();
            uint8_t* current=scratch;
            uint8_t* z=current+PN;
            uint8_t* q=z+PANEL;
            for(unsigned index=lane;index<PANEL;index+=32)q[index]=0;
            unsigned rank=0,end=min(first+panel_width,N-2U);
            for(unsigned column=first;column<end;++column) {
                int previous=-1;
                if(lane==0)for(unsigned k=0;k<rank;++k)
                    if(s[k]==column)previous=int(k);
                previous=__shfl_sync(0xffffffffU,previous,0);
                for(unsigned row=lane;row<N;row+=32) {
                    uint32_t y=matrix[row*PN+column];
                    if(previous>=0) {
                        uint32_t dot=0;
#pragma unroll
                        for(unsigned k=0;k<PN;k+=4)
                            dot=__dp4a(
                                *reinterpret_cast<const uint32_t*>(matrix+row*PN+k),
                                *reinterpret_cast<const uint32_t*>(f+unsigned(previous)*PN+k),dot);
                        y=add_mod(y,dot%prime,prime);
                    }
                    current[row]=uint8_t(y);
                }
                __syncwarp();
                for(unsigned k=lane;k<rank;k+=32)z[k]=current[s[k]];
                __syncwarp();
                for(unsigned output=lane;output<rank;output+=32) {
                    uint32_t sum=0;
                    for(unsigned k=0;k<=output;++k)
                        sum+=uint32_t(t[k*PANEL+output])*z[k];
                    q[output]=uint8_t(sum%prime);
                }
                __syncwarp();
                for(unsigned row=lane;row<N;row+=32) {
                    uint32_t correction=0;
#pragma unroll
                    for(unsigned k=0;k<PANEL;k+=4)
                        correction=__dp4a(
                            *reinterpret_cast<const uint32_t*>(f_row+row*PANEL+k),
                            *reinterpret_cast<const uint32_t*>(q+k),correction);
                    current[row]=uint8_t(sub_mod(current[row],correction%prime,prime));
                }
                __syncwarp();

                unsigned selector=column+1,pivot=N;
                if(lane==0) {
                    pivot=selector;
                    while(pivot<N&&!current[pivot])++pivot;
                }
                pivot=__shfl_sync(0xffffffffU,pivot,0);
                if(pivot==N)continue;
                if(pivot!=selector) {
                    for(unsigned j=lane;j<N;j+=32) {
                        uint8_t temporary=matrix[pivot*PN+j];
                        matrix[pivot*PN+j]=matrix[selector*PN+j];
                        matrix[selector*PN+j]=temporary;
                    }
                    __syncwarp();
                    for(unsigned row=lane;row<N;row+=32) {
                        uint8_t temporary=matrix[row*PN+pivot];
                        matrix[row*PN+pivot]=matrix[row*PN+selector];
                        matrix[row*PN+selector]=temporary;
                    }
                    for(unsigned k=lane;k<rank;k+=32) {
                        uint8_t temporary=f[k*PN+pivot];
                        f[k*PN+pivot]=f[k*PN+selector];
                        f[k*PN+selector]=temporary;
                    }
                    for(unsigned k=lane;k<PANEL;k+=32) {
                        uint8_t temporary=f_row[pivot*PANEL+k];
                        f_row[pivot*PANEL+k]=f_row[selector*PANEL+k];
                        f_row[selector*PANEL+k]=temporary;
                    }
                    if(lane==0) {
                        uint8_t temporary=current[pivot];
                        current[pivot]=current[selector];current[selector]=temporary;
                    }
                    __syncwarp();
                }

                uint32_t inverse=inverse_table[current[selector]];
                for(unsigned row=lane;row<PN;row+=32) {
                    uint8_t value=uint8_t(row>selector&&row<N?
                        mul_mod(current[row],inverse,prime):0);
                    f[rank*PN+row]=value;f_row[row*PANEL+rank]=value;
                }
                if(lane==0)s[rank]=uint8_t(selector);
                __syncwarp();
                for(unsigned output_column=lane;output_column<rank;output_column+=32) {
                    uint32_t sum=0;
                    for(unsigned k=output_column;k<rank;++k)
                        sum+=uint32_t(f[k*PN+selector])*t[output_column*PANEL+k];
                    sum%=prime;
                    t[output_column*PANEL+rank]=uint8_t(sum?prime-sum:0);
                }
                if(lane==0)t[rank*PANEL+rank]=1;
                ++rank;__syncwarp();
            }

            // scratch=A*F, then B=A+(A*F)S^T.
            if(scalar_updates)warp_scalar_product_u8<PN,PANEL,PN,false>(
                matrix,f,nullptr,scratch,prime);
            else warp_tensor_product_u8<PN,PANEL,PN,false>(
                    matrix,f,nullptr,scratch,prime,reciprocal);
            for(unsigned index=lane;index<PN*rank;index+=32) {
                unsigned row=index/rank,k=index%rank;
                matrix[row*PN+s[k]]=uint8_t(add_mod(
                    matrix[row*PN+s[k]],scratch[row*PANEL+k],prime));
            }
            __syncwarp();
            // f becomes W=F*T.  scratch is then repacked as S^T*B.
            if(scalar_updates)warp_scalar_product_u8<PN,PANEL,PANEL,false>(
                f_row,t,nullptr,f,prime);
            else warp_tensor_product_u8<PN,PANEL,PANEL,false>(
                    f_row,t,nullptr,f,prime,reciprocal);
            if(debug_outputs) {
                for(unsigned index=lane;index<PANEL_ENTRIES;index+=32)
                    debug_factors[uint64_t(batch)*PANEL_ENTRIES+index]=f[index];
                __syncwarp();
            }
            for(unsigned index=lane;index<PN*PANEL;index+=32) {
                unsigned column=index/PANEL,k=index%PANEL;
                scratch[index]=k<rank?matrix[s[k]*PN+column]:0;
            }
            __syncwarp();
            if(scalar_updates)warp_scalar_product_u8<PN,PN,PANEL,true>(
                f,scratch,matrix,matrix,prime);
            else warp_tensor_product_u8<PN,PN,PANEL,true>(
                    f,scratch,matrix,matrix,prime,reciprocal);
        }

        if(debug_outputs) {
            for(unsigned index=lane;index<MATRIX_ENTRIES;index+=32)
                debug_matrices[uint64_t(batch)*MATRIX_ENTRIES+index]=matrix[index];
            __syncwarp();
        }

        // Reuse all storage after the matrix for La Budde/Newton state.
        constexpr unsigned POLY_ENTRIES=(N+1)*STRIDE;
        uint8_t* poly=f_row;
        uint8_t* char_factors=poly+POLY_ENTRIES;
        uint8_t* traces=char_factors+N;
        uint8_t* coefficients=traces+STRIDE;
        for(unsigned index=lane;index<POLY_ENTRIES;index+=32)poly[index]=0;
        if(lane==0)poly[0]=1;
        __syncwarp();
        for(unsigned size=1;size<=N;++size) {
            if(lane==0) {
                uint32_t product=1;
                for(unsigned distance=1;distance<size;++distance) {
                    unsigned subrow=size-distance;
                    product=mul_mod(product,matrix[subrow*PN+subrow-1],prime);
                    char_factors[distance]=uint8_t(mul_mod(
                        product,matrix[(size-distance-1)*PN+size-1],prime));
                }
            }
            __syncwarp();
            for(unsigned k=lane;k<=min(size,HALF);k+=32) {
                uint32_t value=k<=size-1?poly[(size-1)*STRIDE+k]:0;
                if(k)value=sub_mod(value,mul_mod(
                    matrix[(size-1)*PN+size-1],poly[(size-1)*STRIDE+k-1],prime),prime);
                for(unsigned distance=1;distance<size&&distance+1<=k;++distance)
                    value=sub_mod(value,mul_mod(char_factors[distance],
                        poly[(size-distance-1)*STRIDE+k-distance-1],prime),prime);
                poly[size*STRIDE+k]=uint8_t(value);
            }
            __syncwarp();
        }
        if(lane==0) {
            coefficients[0]=1;
            for(unsigned k=1;k<=HALF;++k) {
                uint32_t value=mul_mod(k%prime,poly[N*STRIDE+k],prime);
                for(unsigned j=1;j<k;++j)
                    value=add_mod(value,mul_mod(poly[N*STRIDE+j],traces[k-j],prime),prime);
                traces[k]=uint8_t(value?prime-value:0);
            }
            for(unsigned degree=1;degree<=HALF;++degree) {
                uint32_t sum=0;
                for(unsigned k=1;k<=degree;++k)
                    sum=add_mod(sum,mul_mod(mul_mod(
                        traces[k],inverse_table[2],prime),coefficients[degree-k],prime),prime);
                coefficients[degree]=uint8_t(mul_mod(sum,inverse_table[degree],prime));
            }
            unsigned negatives=(HALF-1)-__popcll(term);
            uint32_t value=coefficients[HALF];
            outputs[batch]=uint8_t((negatives&1)&&value?prime-value:value);
        }
        __syncwarp();
    }
}

bool is_prime(uint32_t value) {
    if(value<2)return false;
    for(uint32_t divisor=2;uint64_t(divisor)*divisor<=value;++divisor)
        if(value%divisor==0)return false;
    return true;
}

__device__ __forceinline__ uint32_t add_mod(uint32_t a,uint32_t b,uint32_t p) {
    uint32_t value=a+b;
    return value>=p?value-p:value;
}

__device__ __forceinline__ uint32_t sub_mod(uint32_t a,uint32_t b,uint32_t p) {
    return a>=b?a-b:a+p-b;
}

__device__ __forceinline__ uint32_t mul_mod(uint32_t a,uint32_t b,uint32_t p) {
    return a*b%p;
}

__device__ __forceinline__ uint32_t power_mod(uint32_t a,uint32_t exponent,uint32_t p) {
    uint32_t result=1;
    while(exponent) {
        if(exponent&1)result=mul_mod(result,a,p);
        a=mul_mod(a,a,p);
        exponent>>=1;
    }
    return result;
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

template<unsigned N,unsigned PN>
__global__ void initialize_matrices(
    const uint8_t* __restrict__ adjacency,uint8_t* __restrict__ matrices,
    uint64_t begin,unsigned batches,uint32_t prime) {
    constexpr unsigned HALF=N/2;
    uint64_t cells=uint64_t(batches)*PN*PN;
    for(uint64_t cell=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
            cell<cells;cell+=uint64_t(gridDim.x)*blockDim.x) {
        unsigned within=unsigned(cell%(PN*PN));
        unsigned row=within/PN,column=within%PN;
        uint8_t value=0;
        if(row<N&&column<N) {
            unsigned edge=column%HALF;
            unsigned paired=column<HALF?column+HALF:column-HALF;
            uint64_t term=begin+cell/(PN*PN);
            bool positive=edge==0||(term&(UINT64_C(1)<<(edge-1)));
            if(adjacency[row*N+paired])value=uint8_t(positive?1:prime-1);
        }
        matrices[cell]=value;
    }
}

template<unsigned N,unsigned PN>
__global__ void factor_panels(
    uint8_t* __restrict__ matrices,uint8_t* __restrict__ factors,
    uint8_t* __restrict__ factors_row,uint8_t* __restrict__ inverse_lower,
    uint8_t* __restrict__ selectors,uint8_t* __restrict__ ranks,
    unsigned batches,unsigned first_column,
    unsigned panel_width,uint32_t prime,uint32_t reciprocal) {
    unsigned lane=threadIdx.x&31U;
    unsigned warp_in_block=threadIdx.x>>5;
    unsigned warps_per_block=blockDim.x/32;
    unsigned batch=blockIdx.x*warps_per_block+warp_in_block;
    if(batch>=batches)return;
    uint8_t* matrix=matrices+uint64_t(batch)*PN*PN;
    constexpr unsigned PER_WARP=2*PN*PANEL+PANEL*PANEL+PN+3*PANEL;
    extern __shared__ uint8_t shared[];
    uint8_t* local=shared+warp_in_block*PER_WARP;
    uint8_t* f=local;                              // column-major
    uint8_t* f_row=f+PANEL*PN;                    // row-major
    uint8_t* t=f_row+PN*PANEL;                    // column-major
    uint8_t* current=t+PANEL*PANEL;
    uint8_t* s=current+PN;
    uint8_t* z=s+PANEL;
    uint8_t* q=z+PANEL;
    for(unsigned index=lane;index<PANEL*PN;index+=32)f[index]=0;
    for(unsigned index=lane;index<PN*PANEL;index+=32)f_row[index]=0;
    for(unsigned index=lane;index<PANEL*PANEL;index+=32)t[index]=0;
    for(unsigned index=lane;index<PANEL;index+=32)q[index]=0;
    if(lane==0)ranks[batch]=0;
    __syncwarp();

    unsigned rank=0;
    unsigned end=min(first_column+panel_width,N-2U);
    for(unsigned column=first_column;column<end;++column) {
        int previous=-1;
        if(lane==0)for(unsigned k=0;k<rank;++k)if(s[k]==column)previous=int(k);
        previous=__shfl_sync(0xffffffffU,previous,0);
        for(unsigned row=lane;row<N;row+=32) {
            uint32_t y=matrix[row*PN+column];
            if(previous>=0) {
                const uint8_t* matrix_row=matrix+row*PN;
                const uint8_t* factor_column=f+unsigned(previous)*PN;
                uint32_t dot=0;
#pragma unroll
                for(unsigned k=0;k<PN;k+=4)
                    dot=__dp4a(
                        *reinterpret_cast<const uint32_t*>(matrix_row+k),
                        *reinterpret_cast<const uint32_t*>(factor_column+k),dot);
                y=add_mod(y,dot%prime,prime);
            }
            current[row]=uint8_t(y);
        }
        __syncwarp();
        for(unsigned k=lane;k<rank;k+=32)z[k]=current[s[k]];
        __syncwarp();
        for(unsigned output=lane;output<rank;output+=32) {
            uint32_t sum=0;
            for(unsigned k=0;k<=output;++k)
                sum+=uint32_t(t[k*PANEL+output])*z[k];
            q[output]=uint8_t(sum%prime);
        }
        __syncwarp();
        for(unsigned row=lane;row<N;row+=32) {
            uint32_t correction=0;
#pragma unroll
            for(unsigned k=0;k<PANEL;k+=4)
                correction=__dp4a(
                    *reinterpret_cast<const uint32_t*>(f_row+row*PANEL+k),
                    *reinterpret_cast<const uint32_t*>(q+k),correction);
            current[row]=uint8_t(sub_mod(
                current[row],correction%prime,prime));
        }
        __syncwarp();

        unsigned selector=column+1,pivot=N;
        if(lane==0) {
            pivot=selector;
            while(pivot<N&&!current[pivot])++pivot;
        }
        pivot=__shfl_sync(0xffffffffU,pivot,0);
        if(pivot==N)continue;
        if(pivot!=selector) {
            for(unsigned j=lane;j<N;j+=32) {
                uint8_t temporary=matrix[pivot*PN+j];
                matrix[pivot*PN+j]=matrix[selector*PN+j];
                matrix[selector*PN+j]=temporary;
            }
            __syncwarp();
            for(unsigned row=lane;row<N;row+=32) {
                uint8_t temporary=matrix[row*PN+pivot];
                matrix[row*PN+pivot]=matrix[row*PN+selector];
                matrix[row*PN+selector]=temporary;
            }
            for(unsigned k=lane;k<rank;k+=32) {
                uint8_t temporary=f[k*PN+pivot];
                f[k*PN+pivot]=f[k*PN+selector];
                f[k*PN+selector]=temporary;
            }
            for(unsigned k=lane;k<PANEL;k+=32) {
                uint8_t temporary=f_row[pivot*PANEL+k];
                f_row[pivot*PANEL+k]=f_row[selector*PANEL+k];
                f_row[selector*PANEL+k]=temporary;
            }
            if(lane==0) {
                uint8_t temporary=current[pivot];
                current[pivot]=current[selector];current[selector]=temporary;
            }
            __syncwarp();
        }

        uint32_t inverse=0;
        if(lane==0)inverse=power_mod(current[selector],prime-2,prime);
        inverse=__shfl_sync(0xffffffffU,inverse,0);
        for(unsigned row=lane;row<PN;row+=32) {
            uint8_t value=uint8_t(row>selector&&row<N?
                mul_mod(current[row],inverse,prime):0);
            f[rank*PN+row]=value;
            f_row[row*PANEL+rank]=value;
        }
        if(lane==0)s[rank]=uint8_t(selector);
        __syncwarp();

        for(unsigned output_column=lane;output_column<rank;output_column+=32) {
            uint32_t sum=0;
            for(unsigned k=output_column;k<rank;++k)
                sum+=uint32_t(f[k*PN+selector])*t[output_column*PANEL+k];
            sum%=prime;
            t[output_column*PANEL+rank]=uint8_t(sum?prime-sum:0);
        }
        if(lane==0)t[rank*PANEL+rank]=1;
        ++rank;
        __syncwarp();
    }
    uint8_t* output_f=factors+uint64_t(batch)*PANEL*PN;
    uint8_t* output_f_row=factors_row+uint64_t(batch)*PN*PANEL;
    uint8_t* output_t=inverse_lower+uint64_t(batch)*PANEL*PANEL;
    uint8_t* output_s=selectors+uint64_t(batch)*PANEL;
    for(unsigned index=lane;index<PANEL*PN;index+=32)output_f[index]=f[index];
    for(unsigned index=lane;index<PN*PANEL;index+=32)output_f_row[index]=f_row[index];
    for(unsigned index=lane;index<PANEL*PANEL;index+=32)output_t[index]=t[index];
    for(unsigned index=lane;index<PANEL;index+=32)output_s[index]=s[index];
    if(lane==0)ranks[batch]=uint8_t(rank);
}

template<unsigned M,unsigned NOUT,unsigned K,bool SUBTRACT>
__global__ void tensor_product_u8(
    const uint8_t* __restrict__ left,const uint8_t* __restrict__ right_columns,
    const uint8_t* __restrict__ input,uint8_t* __restrict__ output,
    unsigned batches,uint32_t prime,uint32_t reciprocal) {
    static_assert(M%16==0&&NOUT%8==0&&K%16==0);
    constexpr unsigned ROW_TILES=M/16,COLUMN_TILES=NOUT/8;
    constexpr unsigned TILES=ROW_TILES*COLUMN_TILES;
    unsigned lane=threadIdx.x&31U,warp_in_block=threadIdx.x>>5;
    uint64_t warp=uint64_t(blockIdx.x)*(blockDim.x/32)+warp_in_block;
    uint64_t stride=uint64_t(gridDim.x)*(blockDim.x/32),tasks=uint64_t(batches)*TILES;
    for(uint64_t task=warp;task<tasks;task+=stride) {
        unsigned batch=unsigned(task/TILES),tile=unsigned(task%TILES);
        unsigned row_base=(tile/COLUMN_TILES)*16,column_base=(tile%COLUMN_TILES)*8;
        unsigned group=lane>>2,thread_in_group=lane&3U;
        const uint8_t* left_batch=left+uint64_t(batch)*M*K;
        const uint8_t* right_batch=right_columns+uint64_t(batch)*NOUT*K;
        int32_t result[4]={0,0,0,0};
#pragma unroll
        for(unsigned k_tile=0;k_tile<K;k_tile+=16) {
            unsigned k=k_tile+thread_in_group*4;
            uint32_t a0=*reinterpret_cast<const uint32_t*>(
                left_batch+(row_base+group)*K+k);
            uint32_t a1=*reinterpret_cast<const uint32_t*>(
                left_batch+(row_base+group+8)*K+k);
            uint32_t b0=*reinterpret_cast<const uint32_t*>(
                right_batch+(column_base+group)*K+k);
            mma_u8_16x8x16(a0,a1,b0,result);
        }
        unsigned columns[2]={column_base+thread_in_group*2,column_base+thread_in_group*2+1};
        unsigned rows[2]={row_base+group,row_base+group+8};
#pragma unroll
        for(unsigned ri=0;ri<2;++ri)for(unsigned ci=0;ci<2;++ci) {
            uint64_t offset=uint64_t(batch)*M*NOUT+rows[ri]*NOUT+columns[ci];
            uint32_t product=reduce_u32(uint32_t(result[ri*2+ci]),prime,reciprocal);
            if constexpr(SUBTRACT)output[offset]=uint8_t(sub_mod(input[offset],product,prime));
            else output[offset]=uint8_t(product);
        }
    }
}

template<unsigned M,unsigned NOUT,unsigned K,bool SUBTRACT>
__global__ void scalar_product_u8(
    const uint8_t* __restrict__ left,const uint8_t* __restrict__ right_columns,
    const uint8_t* __restrict__ input,uint8_t* __restrict__ output,
    unsigned batches,uint32_t prime,uint32_t reciprocal) {
    uint64_t cells=uint64_t(batches)*M*NOUT;
    for(uint64_t cell=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
            cell<cells;cell+=uint64_t(gridDim.x)*blockDim.x) {
        unsigned batch=unsigned(cell/(M*NOUT)),within=unsigned(cell%(M*NOUT));
        unsigned row=within/NOUT,column=within%NOUT;
        const uint8_t* left_row=left+(uint64_t(batch)*M+row)*K;
        const uint8_t* right_column=right_columns+(uint64_t(batch)*NOUT+column)*K;
        uint32_t product=0;
        for(unsigned k=0;k<K;++k)product+=uint32_t(left_row[k])*right_column[k];
        product=reduce_u32(product,prime,reciprocal);
        if constexpr(SUBTRACT)output[cell]=uint8_t(sub_mod(input[cell],product,prime));
        else output[cell]=uint8_t(product);
    }
}

template<unsigned PN>
__global__ void scatter_right_update(
    uint8_t* __restrict__ matrices,const uint8_t* __restrict__ products,
    const uint8_t* __restrict__ selectors,const uint8_t* __restrict__ ranks,
    unsigned batches,uint32_t prime) {
    uint64_t cells=uint64_t(batches)*PN*PANEL;
    for(uint64_t cell=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
            cell<cells;cell+=uint64_t(gridDim.x)*blockDim.x) {
        unsigned batch=unsigned(cell/(PN*PANEL));
        unsigned within=unsigned(cell%(PN*PANEL));
        unsigned row=within/PANEL,k=within%PANEL;
        if(k<ranks[batch]) {
            unsigned column=selectors[uint64_t(batch)*PANEL+k];
            uint64_t matrix_offset=uint64_t(batch)*PN*PN+row*PN+column;
            matrices[matrix_offset]=uint8_t(add_mod(
                matrices[matrix_offset],products[cell],prime));
        }
    }
}

template<unsigned PN>
__global__ void gather_pivot_rows(
    const uint8_t* __restrict__ matrices,uint8_t* __restrict__ right_columns,
    const uint8_t* __restrict__ selectors,const uint8_t* __restrict__ ranks,
    unsigned batches) {
    uint64_t cells=uint64_t(batches)*PN*PANEL;
    for(uint64_t cell=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
            cell<cells;cell+=uint64_t(gridDim.x)*blockDim.x) {
        unsigned batch=unsigned(cell/(PN*PANEL));
        unsigned within=unsigned(cell%(PN*PANEL));
        unsigned column=within/PANEL,k=within%PANEL;
        right_columns[cell]=k<ranks[batch]?
            matrices[uint64_t(batch)*PN*PN+
                selectors[uint64_t(batch)*PANEL+k]*PN+column]:0;
    }
}

template<unsigned N,unsigned PN>
__global__ void extract_sign_terms(
    const uint8_t* __restrict__ matrices,uint8_t* __restrict__ outputs,
    const uint8_t* __restrict__ inverse_small,
    uint64_t begin,unsigned batches,uint32_t prime) {
    constexpr unsigned HALF=N/2,STRIDE=HALF+1;
    unsigned lane=threadIdx.x&31U,warp_in_block=threadIdx.x>>5;
    unsigned warps=blockDim.x/32,batch=blockIdx.x*warps+warp_in_block;
    if(batch>=batches)return;
    const uint8_t* matrix=matrices+uint64_t(batch)*PN*PN;
    extern __shared__ uint8_t shared[];
    constexpr unsigned POLY_ENTRIES=(N+1)*STRIDE;
    constexpr unsigned PER_WARP=POLY_ENTRIES+N+2*STRIDE;
    uint8_t* local_poly=shared+warp_in_block*PER_WARP;
    uint8_t* factors=local_poly+POLY_ENTRIES;
    uint8_t* traces=factors+N;
    uint8_t* coefficients=traces+STRIDE;
    for(unsigned index=lane;index<(N+1)*STRIDE;index+=32)local_poly[index]=0;
    if(lane==0)local_poly[0]=1;
    __syncwarp();
    for(unsigned size=1;size<=N;++size) {
        if(lane==0) {
            uint32_t product=1;
            for(unsigned distance=1;distance<size;++distance) {
                unsigned subrow=size-distance;
                product=mul_mod(product,matrix[subrow*PN+subrow-1],prime);
                factors[distance]=uint8_t(mul_mod(
                    product,matrix[(size-distance-1)*PN+size-1],prime));
            }
        }
        __syncwarp();
        for(unsigned k=lane;k<=min(size,HALF);k+=32) {
            uint32_t value=k<=size-1?local_poly[(size-1)*STRIDE+k]:0;
            if(k)value=sub_mod(value,mul_mod(
                matrix[(size-1)*PN+size-1],local_poly[(size-1)*STRIDE+k-1],prime),prime);
            for(unsigned distance=1;distance<size&&distance+1<=k;++distance)
                value=sub_mod(value,mul_mod(
                    factors[distance],local_poly[(size-distance-1)*STRIDE+k-distance-1],prime),prime);
            local_poly[size*STRIDE+k]=uint8_t(value);
        }
        __syncwarp();
    }
    if(lane==0) {
        coefficients[0]=1;
        for(unsigned k=1;k<=HALF;++k) {
            uint32_t value=mul_mod(k%prime,local_poly[N*STRIDE+k],prime);
            for(unsigned j=1;j<k;++j)value=add_mod(value,mul_mod(
                local_poly[N*STRIDE+j],traces[k-j],prime),prime);
            traces[k]=uint8_t(value?prime-value:0);
        }
        for(unsigned degree=1;degree<=HALF;++degree) {
            uint32_t sum=0;
            for(unsigned k=1;k<=degree;++k)
                sum=add_mod(sum,mul_mod(mul_mod(
                    traces[k],inverse_small[2],prime),coefficients[degree-k],prime),prime);
            coefficients[degree]=uint8_t(mul_mod(sum,inverse_small[degree],prime));
        }
        uint64_t term=begin+batch;
        unsigned negatives=(HALF-1)-__popcll(term);
        uint32_t value=coefficients[HALF];
        outputs[batch]=uint8_t((negatives&1)&&value?prime-value:value);
    }
}

struct HostMod {
    uint32_t p;
    uint32_t add(uint32_t a,uint32_t b)const{uint32_t v=a+b;return v>=p?v-p:v;}
    uint32_t sub(uint32_t a,uint32_t b)const{return a>=b?a-b:a+p-b;}
    uint32_t mul(uint32_t a,uint32_t b)const{return uint32_t(uint64_t(a)*b%p);}
    uint32_t power(uint32_t a,uint32_t e)const{uint32_t r=1;while(e){if(e&1)r=mul(r,a);a=mul(a,a);e>>=1;}return r;}
    uint32_t inverse(uint32_t a)const{return power(a,p-2);}
};

template<unsigned N>
uint8_t cpu_sign_term(
    const six_by_twenty_eight::Query& query,uint64_t term,HostMod mod,
    std::vector<uint8_t>* hessenberg=nullptr,unsigned elimination_columns=N) {
    constexpr unsigned HALF=N/2;
    std::array<uint32_t,N*N> matrix{};
    auto at=[&](unsigned row,unsigned column)->uint32_t&{return matrix[row*N+column];};
    for(unsigned row=0;row<N;++row)for(unsigned column=0;column<N;++column) {
        unsigned edge=column%HALF,paired=column<HALF?column+HALF:column-HALF;
        bool positive=edge==0||(term&(UINT64_C(1)<<(edge-1)));
        if(query.adjacency[row*N+paired])at(row,column)=positive?1:mod.p-1;
    }
    for(unsigned column=0;column+2<N&&column<elimination_columns;++column) {
        unsigned pivot=column+1;
        while(pivot<N&&!at(pivot,column))++pivot;
        if(pivot==N)continue;
        if(pivot!=column+1) {
            for(unsigned j=0;j<N;++j)std::swap(at(pivot,j),at(column+1,j));
            for(unsigned i=0;i<N;++i)std::swap(at(i,pivot),at(i,column+1));
        }
        uint32_t inverse=mod.inverse(at(column+1,column));
        std::array<uint32_t,N> factors{};
        for(unsigned row=column+2;row<N;++row)factors[row]=mod.mul(at(row,column),inverse);
        for(unsigned row=column+2;row<N;++row)for(unsigned j=column;j<N;++j)
            at(row,j)=mod.sub(at(row,j),mod.mul(factors[row],at(column+1,j)));
        for(unsigned row=0;row<N;++row) {
            uint32_t sum=0;
            for(unsigned k=column+2;k<N;++k)sum=mod.add(sum,mod.mul(at(row,k),factors[k]));
            at(row,column+1)=mod.add(at(row,column+1),sum);
        }
    }
    if(hessenberg) {
        hessenberg->resize(N*N);
        for(unsigned row=0;row<N;++row)for(unsigned column=0;column<N;++column)
            (*hessenberg)[row*N+column]=uint8_t(at(row,column));
    }
    std::array<std::array<uint32_t,HALF+1>,N+1> poly{};
    poly[0][0]=1;
    for(unsigned size=1;size<=N;++size) {
        unsigned diagonal=size-1;
        for(unsigned k=0;k<=std::min(size,HALF);++k) {
            uint32_t value=k<=size-1?poly[size-1][k]:0;
            if(k)value=mod.sub(value,mod.mul(at(diagonal,diagonal),poly[size-1][k-1]));
            uint32_t product=1;
            for(unsigned distance=1;distance<size&&distance+1<=k;++distance) {
                unsigned subrow=size-distance;
                product=mod.mul(product,at(subrow,subrow-1));
                value=mod.sub(value,mod.mul(mod.mul(
                    product,at(size-distance-1,size-1)),poly[size-distance-1][k-distance-1]));
            }
            poly[size][k]=value;
        }
    }
    std::array<uint32_t,HALF+1> traces{},coefficients{};
    coefficients[0]=1;
    for(unsigned k=1;k<=HALF;++k) {
        uint32_t value=mod.mul(k%mod.p,poly[N][k]);
        for(unsigned j=1;j<k;++j)value=mod.add(value,mod.mul(poly[N][j],traces[k-j]));
        traces[k]=value?mod.p-value:0;
    }
    for(unsigned degree=1;degree<=HALF;++degree) {
        uint32_t sum=0;
        for(unsigned k=1;k<=degree;++k)sum=mod.add(sum,mod.mul(
            mod.mul(traces[k],mod.inverse(2)),coefficients[degree-k]));
        coefficients[degree]=mod.mul(sum,mod.inverse(degree));
    }
    unsigned negatives=(HALF-1)-__builtin_popcountll(term);
    uint32_t result=coefficients[HALF];
    return uint8_t((negatives&1)&&result?mod.p-result:result);
}

template<class T>T* allocate_device(size_t count,const char* operation) {
    T* pointer=nullptr;cuda_check(cudaMalloc(&pointer,count*sizeof(T)),operation);return pointer;
}

template<unsigned N,unsigned PN>
int run(const Options& options,const six_by_twenty_eight::Query& query) {
    constexpr unsigned HALF=N/2,STRIDE=HALF+1;
    size_t matrix_cells=size_t(options.batches)*PN*PN;
    size_t panel_entries=size_t(options.batches)*PN*PANEL;
    size_t small_entries=size_t(options.batches)*PANEL*PANEL;
    uint8_t* d_adjacency=allocate_device<uint8_t>(N*N,"allocate adjacency");
    uint8_t* matrices=allocate_device<uint8_t>(matrix_cells,"allocate matrices");
    uint8_t* factors=allocate_device<uint8_t>(panel_entries,"allocate factors");
    uint8_t* factors_row=allocate_device<uint8_t>(panel_entries,"allocate row-major factors");
    uint8_t* w=allocate_device<uint8_t>(panel_entries,"allocate W");
    uint8_t* inverse_lower=allocate_device<uint8_t>(small_entries,"allocate inverse lower");
    uint8_t* selectors=allocate_device<uint8_t>(size_t(options.batches)*PANEL,"allocate selectors");
    uint8_t* ranks=allocate_device<uint8_t>(options.batches,"allocate ranks");
    uint8_t* products=allocate_device<uint8_t>(panel_entries,"allocate products");
    uint8_t* pivot_rows=allocate_device<uint8_t>(panel_entries,"allocate pivot rows");
    uint8_t* outputs=allocate_device<uint8_t>(options.batches,"allocate outputs");
    uint8_t* inverse_small=allocate_device<uint8_t>(256,"allocate inverses");
    cuda_check(cudaMemcpy(d_adjacency,query.adjacency.data(),N*N,cudaMemcpyHostToDevice),"copy adjacency");
    std::vector<uint8_t> inverses(256);
    HostMod host_mod{options.prime};
    for(unsigned k=1;k<options.prime;++k)inverses[k]=uint8_t(host_mod.inverse(k));
    cuda_check(cudaMemcpy(inverse_small,inverses.data(),256,cudaMemcpyHostToDevice),"copy inverses");

    int device=0,multiprocessors=0;
    cuda_check(cudaGetDevice(&device),"get device");
    cuda_check(cudaDeviceGetAttribute(&multiprocessors,cudaDevAttrMultiProcessorCount,device),"get SM count");
    unsigned scalar_blocks=(options.batches+7)/8;
    unsigned bulk_blocks=unsigned(multiprocessors)*8;
    uint32_t reciprocal=uint32_t((UINT64_C(1)<<32)/options.prime);
    constexpr unsigned FUSED_WARPS=PN==48?6:4;
    constexpr unsigned FUSED_SHARED=FUSED_WARPS*(
        PN*PN+3*PN*PANEL+PANEL*PANEL+PANEL);
    cuda_check(cudaFuncSetAttribute(fused_sign_terms<N,PN,FUSED_WARPS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,FUSED_SHARED),"set fused shared memory");
    int fused_active=0;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &fused_active,fused_sign_terms<N,PN,FUSED_WARPS>,FUSED_WARPS*32,FUSED_SHARED),
        "compute fused occupancy");
    if(fused_active<=0)throw std::runtime_error("fused kernel has zero occupancy");
    unsigned fused_blocks=unsigned(multiprocessors)*unsigned(fused_active)*2;
    auto launch=[&](std::vector<cudaEvent_t>* events=nullptr,
                    std::vector<std::string>* labels=nullptr) {
        auto mark=[&](const std::string& label) {
            if(!events)return;
            cudaEvent_t event=nullptr;
            cuda_check(cudaEventCreate(&event),"create profile event");
            cuda_check(cudaEventRecord(event),"record profile event");
            events->push_back(event);labels->push_back(label);
        };
        mark("start");
        if(options.fused) {
            fused_sign_terms<N,PN,FUSED_WARPS><<<
                fused_blocks,FUSED_WARPS*32,FUSED_SHARED>>>(
                d_adjacency,outputs,matrices,factors,inverse_small,options.begin,options.batches,
                options.panel_width,options.panel_limit,options.prime,reciprocal,
                options.scalar_updates,options.panel_limit!=UINT32_MAX);
            mark("fused");
            return;
        }
        initialize_matrices<N,PN><<<bulk_blocks,256>>>(
            d_adjacency,matrices,options.begin,options.batches,options.prime);
        mark("initialize");
        unsigned panel_index=0;
        for(unsigned first=0;first+2<N&&panel_index<options.panel_limit;
                first+=options.panel_width,++panel_index) {
            constexpr unsigned FACTOR_SHARED=8*(2*PN*PANEL+PANEL*PANEL+PN+3*PANEL);
            factor_panels<N,PN><<<scalar_blocks,256,FACTOR_SHARED>>>(
                matrices,factors,factors_row,inverse_lower,selectors,ranks,
                options.batches,first,options.panel_width,options.prime,reciprocal);
            mark("factor_"+std::to_string(panel_index));
            if(options.scalar_updates)
                scalar_product_u8<PN,PANEL,PN,false><<<bulk_blocks,256>>>(
                    matrices,factors,nullptr,products,options.batches,options.prime,reciprocal);
            else tensor_product_u8<PN,PANEL,PN,false><<<bulk_blocks,256>>>(
                    matrices,factors,nullptr,products,options.batches,options.prime,reciprocal);
            mark("right_product_"+std::to_string(panel_index));
            scatter_right_update<PN><<<bulk_blocks,256>>>(
                matrices,products,selectors,ranks,options.batches,options.prime);
            mark("scatter_"+std::to_string(panel_index));
            gather_pivot_rows<PN><<<bulk_blocks,256>>>(
                matrices,pivot_rows,selectors,ranks,options.batches);
            mark("gather_"+std::to_string(panel_index));
            tensor_product_u8<PN,PANEL,PANEL,false><<<bulk_blocks,256>>>(
                factors_row,inverse_lower,nullptr,w,options.batches,options.prime,reciprocal);
            mark("compact_w_"+std::to_string(panel_index));
            if(options.scalar_updates)
                scalar_product_u8<PN,PN,PANEL,true><<<bulk_blocks,256>>>(
                    w,pivot_rows,matrices,matrices,options.batches,options.prime,reciprocal);
            else tensor_product_u8<PN,PN,PANEL,true><<<bulk_blocks,256>>>(
                    w,pivot_rows,matrices,matrices,options.batches,options.prime,reciprocal);
            mark("left_product_"+std::to_string(panel_index));
        }
        if(options.panel_limit==UINT32_MAX) {
            constexpr size_t shared=8*((N+1)*STRIDE+N+2*STRIDE);
            extract_sign_terms<N,PN><<<scalar_blocks,256,shared>>>(
                matrices,outputs,inverse_small,options.begin,options.batches,options.prime);
            mark("extract");
        }
    };
    launch();cuda_check(cudaGetLastError(),"launch complete sign pipeline");
    cuda_check(cudaDeviceSynchronize(),"synchronize correctness run");
    std::vector<uint8_t> host_outputs(options.batches);
    cuda_check(cudaMemcpy(host_outputs.data(),outputs,options.batches,cudaMemcpyDeviceToHost),"copy outputs");
    if(!options.fused||options.panel_limit!=UINT32_MAX) {
        std::vector<uint8_t> gpu_matrix(PN*PN),cpu_matrix;
        cuda_check(cudaMemcpy(gpu_matrix.data(),matrices,PN*PN,cudaMemcpyDeviceToHost),"copy first Hessenberg matrix");
        unsigned compared_columns=options.panel_limit==UINT32_MAX?N:
            std::min(N-2,options.panel_limit*options.panel_width);
        cpu_sign_term<N>(query,options.begin,host_mod,&cpu_matrix,compared_columns);
        for(unsigned row=0;row<N;++row)for(unsigned column=0;column<N;++column)
            if(gpu_matrix[row*PN+column]!=cpu_matrix[row*N+column]) {
                std::vector<uint8_t> debug_f(PN*PANEL);
                cuda_check(cudaMemcpy(debug_f.data(),factors,PN*PANEL,cudaMemcpyDeviceToHost),
                    "copy debug factors");
                throw std::runtime_error(
                    "Hessenberg mismatch row="+std::to_string(row)+" column="+
                    std::to_string(column)+" expected="+std::to_string(cpu_matrix[row*N+column])+
                    " actual="+std::to_string(gpu_matrix[row*PN+column])+
                    " w00="+std::to_string(debug_f[0])+" w10="+
                    std::to_string(debug_f[PANEL]));
            }
    }
    if(options.panel_limit!=UINT32_MAX) {
        std::printf("HAFNIAN_INT8_PANEL_DEBUG order=%u panels=%u width=%u scalar=%u matrix_exact=OK\n",
            N,options.panel_limit,options.panel_width,options.scalar_updates?1:0);
        return 0;
    }
    unsigned validation=std::min(options.batches,16U);
    for(unsigned batch=0;batch<validation;++batch) {
        uint8_t expected=cpu_sign_term<N>(query,options.begin+batch,host_mod);
        if(host_outputs[batch]!=expected)throw std::runtime_error(
            "complete sign-term mismatch batch="+std::to_string(batch)+
            " expected="+std::to_string(expected)+" actual="+std::to_string(host_outputs[batch]));
    }

    std::vector<cudaEvent_t> profile_events;
    std::vector<std::string> profile_labels;
    launch(&profile_events,&profile_labels);
    cuda_check(cudaEventSynchronize(profile_events.back()),"synchronize profile");
    std::printf("HAFNIAN_INT8_SIGN_PROFILE");
    for(size_t index=1;index<profile_events.size();++index) {
        float stage_ms=0;
        cuda_check(cudaEventElapsedTime(
            &stage_ms,profile_events[index-1],profile_events[index]),"measure profile stage");
        std::printf(" %s_ms=%.6f",profile_labels[index].c_str(),stage_ms);
    }
    std::printf("\n");
    for(cudaEvent_t event:profile_events)cudaEventDestroy(event);

    for(unsigned iteration=0;iteration<options.warmup;++iteration)launch();
    cuda_check(cudaDeviceSynchronize(),"synchronize warmup");
    cudaEvent_t start=nullptr,stop=nullptr;
    cuda_check(cudaEventCreate(&start),"create start event");
    cuda_check(cudaEventCreate(&stop),"create stop event");
    cuda_check(cudaEventRecord(start),"record start");
    for(unsigned iteration=0;iteration<options.iterations;++iteration)launch();
    cuda_check(cudaEventRecord(stop),"record stop");
    cuda_check(cudaEventSynchronize(stop),"synchronize stop");
    float elapsed=0;cuda_check(cudaEventElapsedTime(&elapsed,start,stop),"measure time");
    float milliseconds=elapsed/options.iterations;
    uint64_t checksum=UINT64_C(1469598103934665603);
    for(uint8_t value:host_outputs)checksum=(checksum^value)*UINT64_C(1099511628211);
    cudaDeviceProp properties{};cuda_check(cudaGetDeviceProperties(&properties,device),"get properties");
    std::printf(
        "HAFNIAN_INT8_SIGN device=%s cc=%d.%d order=%u padded=%u panel=%u query=%u "
        "prime=%u batches=%u iterations=%u fused=%u fused_active_blocks=%d milliseconds=%.6f terms_per_second=%.3f "
        "target_30m=%s checksum=%016" PRIx64 " validation_terms=%u exact=OK\n",
        properties.name,properties.major,properties.minor,N,PN,options.panel_width,query.id,
        options.prime,options.batches,options.iterations,options.fused?1:0,
        fused_active,milliseconds,
        options.batches/(milliseconds*1e-3),
        options.batches/(milliseconds*1e-3)>=3.0e7?"PASS":"REJECT",
        checksum,validation);
    cudaEventDestroy(stop);cudaEventDestroy(start);
    cudaFree(inverse_small);cudaFree(outputs);cudaFree(pivot_rows);
    cudaFree(products);cudaFree(ranks);cudaFree(selectors);cudaFree(inverse_lower);
    cudaFree(w);cudaFree(factors_row);cudaFree(factors);cudaFree(matrices);cudaFree(d_adjacency);
    return 0;
}

} // namespace

int main(int argc,char** argv) try {
    Options options=parse_options(argc,argv);
    if(!is_prime(options.prime))throw std::runtime_error("--prime must be prime");
    auto catalog=six_by_twenty_eight::build_catalog();
    if(options.work_census) {
        constexpr std::array<uint32_t,20> primes={
            251,241,239,233,229,227,223,211,199,197,
            193,191,181,179,173,167,163,157,151,149};
        std::array<uint64_t,65> queries{},sign_terms{},small_terms{};
        auto required=[&](unsigned bound) {
            unsigned __int128 product=1;
            const unsigned __int128 target=static_cast<unsigned __int128>(1)<<bound;
            for(unsigned count=0;count<primes.size();++count) {
                product*=primes[count];
                if(product>target)return count+1;
            }
            throw std::runtime_error("small-prime list does not cover bound");
        };
        for(const auto& item:catalog.queries) {
            uint64_t terms=UINT64_C(1)<<(item.vertices/2-1);
            ++queries[item.vertices];sign_terms[item.vertices]+=terms;
            small_terms[item.vertices]+=terms*required(item.matching_bound_power);
        }
        uint64_t total=0;
        for(unsigned order=48;order<=64;order+=2) {
            total+=small_terms[order];
            std::printf("HAFNIAN_INT8_WORK order=%u queries=%" PRIu64
                " sign_terms=%" PRIu64 " small_prime_terms=%" PRIu64 "\n",
                order,queries[order],sign_terms[order],small_terms[order]);
        }
        std::printf("HAFNIAN_INT8_WORK total_small_prime_terms=%" PRIu64 " exact=OK\n",total);
        return 0;
    }
    const six_by_twenty_eight::Query* query=nullptr;
    if(options.query!=UINT32_MAX) {
        if(options.query>=catalog.queries.size())throw std::runtime_error("invalid query");
        query=&catalog.queries[options.query];
        if(query->vertices!=options.order)throw std::runtime_error("query/order mismatch");
    } else {
        for(const auto& candidate:catalog.queries)if(candidate.vertices==options.order) {
            query=&candidate;break;
        }
    }
    if(!query)throw std::runtime_error("no query with requested order");
    uint64_t total=UINT64_C(1)<<(options.order/2-1);
    if(options.begin+options.batches>total)throw std::runtime_error("term range exceeds query");
    if(options.order==48)return run<48,48>(options,*query);
    if(options.order==50)return run<50,64>(options,*query);
    if(options.order==52)return run<52,64>(options,*query);
    if(options.order==54)return run<54,64>(options,*query);
    if(options.order==56)return run<56,64>(options,*query);
    if(options.order==58)return run<58,64>(options,*query);
    if(options.order==60)return run<60,64>(options,*query);
    if(options.order==62)return run<62,64>(options,*query);
    return run<64,64>(options,*query);
} catch(const std::exception& error) {
    std::fprintf(stderr,"error: %s\n",error.what());
    return 2;
}
