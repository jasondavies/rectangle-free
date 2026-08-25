#pragma once

#include "hafnian_gray_gpu_core.cuh"

#include <cstddef>
#include <cstdint>

// Exact eight-term Gray chains evaluated from one dense tridiagonalization.
// Each group of four terms uses a truncated resolvent determinant; one
// structured rank-eight Lanczos refresh advances between the groups.
namespace hafnian_resolvent {

constexpr unsigned WARPS_PER_BLOCK=2;
constexpr unsigned THREADS=32*WARPS_PER_BLOCK;
template<unsigned N,class Mod>
constexpr unsigned min_blocks_per_sm() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
    // Ada has fewer resident warps than Blackwell and benefits from retaining
    // 72 registers rather than forcing the resolver down to 48--56 registers.
    return 14;
#else
    return std::is_same_v<Mod,HafnianMersenne31> && N==48 ? 20 : 18;
#endif
}
constexpr unsigned CHAIN=8;
constexpr unsigned RESOLVENT_CHAIN=4;
constexpr unsigned UPDATE_COLUMNS=2*(RESOLVENT_CHAIN-1);
constexpr unsigned FUTURE_COLUMNS=2*(CHAIN-1);

template<unsigned N,class Mod>
__device__ __forceinline__ void characteristic_from_tridiagonal(
    const uint32_t* diagonal,const uint32_t* beta,unsigned rank,Mod mod,
    uint32_t* output,uint32_t* first,uint32_t* second,uint32_t* third) {
    constexpr unsigned DEGREE=N/2,S=DEGREE+1;
    const unsigned lane=threadIdx.x&31;
    for(unsigned i=lane;i<S;i+=32)first[i]=second[i]=third[i]=0;
    __syncwarp();
    if(lane==0)second[0]=mod.one;
    __syncwarp();
    uint32_t *older=first,*previous=second,*next=third;
    for(unsigned size=1;size<=rank;++size) {
        const unsigned at=size-1,maximum=min(size,DEGREE);
        if(lane<=maximum) {
            const unsigned defect=lane;
            uint32_t value=defect<size?previous[defect]:0;
            if(defect)value=hafnian_sub_mod(value,hafnian_mul(
                diagonal[at],previous[defect-1],mod),mod.p);
            if(size>1&&defect>=2)value=hafnian_sub_mod(value,hafnian_mul(
                beta[at],older[defect-2],mod),mod.p);
            next[defect]=value;
        }
        __syncwarp();
        uint32_t* temporary=older;older=previous;previous=next;next=temporary;
    }
    for(unsigned i=lane;i<S;i+=32)output[i]=previous[i];
    __syncwarp();
}

template<unsigned N,class Mod>
__device__ __forceinline__ uint32_t term_from_characteristic(
    const uint32_t* characteristic,uint64_t signs,Mod mod,
    const uint32_t* inverse_small,uint32_t* coefficients) {
    constexpr unsigned DEGREE=N/2,S=DEGREE+1;
    const unsigned lane=threadIdx.x&31;
    for(unsigned i=lane;i<S;i+=32)coefficients[i]=0;
    __syncwarp();
    if(lane==0)coefficients[0]=mod.one;
    __syncwarp();
    for(unsigned degree=1;degree<=DEGREE;++degree) {
        uint32_t sum=0;
        for(unsigned k=lane+1;k<=degree;k+=32) {
            const uint32_t factor=uint32_t(uint64_t(2*degree-k)*mod.one%mod.p);
            sum=hafnian_add_mod(sum,hafnian_mul(hafnian_mul(
                characteristic[k],coefficients[degree-k],mod),factor,mod),mod.p);
        }
        for(unsigned offset=16;offset;offset>>=1)
            sum=hafnian_add_mod(sum,__shfl_down_sync(0xffffffff,sum,offset),mod.p);
        if(lane==0)coefficients[degree]=hafnian_neg_mod(hafnian_mul(
            hafnian_mul(sum,inverse_small[2],mod),inverse_small[degree],mod),mod.p);
        __syncwarp();
    }
    uint32_t result=0;
    if(lane==0) {
        const unsigned negatives=(DEGREE-1)-__popcll(signs);
        result=negatives&1?hafnian_neg_mod(coefficients[DEGREE],mod.p):
            coefficients[DEGREE];
    }
    return __shfl_sync(0xffffffff,result,0);
}

template<unsigned N,unsigned LEFT_MIN=0,unsigned RIGHT_MIN=0,class Mod>
__device__ __forceinline__ void multiply_polynomials(
    const uint32_t* left,const uint32_t* right,uint32_t* output,Mod mod) {
    constexpr unsigned S=N/2+1;
    const unsigned lane=threadIdx.x&31;
    // One lane per coefficient is best for these degree-24 polynomials.
    if(lane<S) {
        uint32_t value=0;
        if constexpr(LEFT_MIN+RIGHT_MIN==0) {
            unsigned k=0;
            for(;k+3<=lane;k+=4)value=hafnian_add_mod(value,
                hafnian_sum_products4(left[k],right[lane-k],
                    left[k+1],right[lane-k-1],left[k+2],right[lane-k-2],
                    left[k+3],right[lane-k-3],mod),mod.p);
            for(;k+1<=lane;k+=2)value=hafnian_add_mod(value,
                hafnian_sum_products2(left[k],right[lane-k],
                    left[k+1],right[lane-k-1],mod),mod.p);
            if(k<=lane)value=hafnian_add_mod(value,
                hafnian_mul(left[k],right[lane-k],mod),mod.p);
        } else if(lane>=LEFT_MIN+RIGHT_MIN) {
            unsigned k=LEFT_MIN;
            for(;k+3<=lane-RIGHT_MIN;k+=4)
                value=hafnian_add_mod(value,hafnian_sum_products4(
                    left[k],right[lane-k],left[k+1],right[lane-k-1],
                    left[k+2],right[lane-k-2],left[k+3],right[lane-k-3],mod),mod.p);
            for(;k+1<=lane-RIGHT_MIN;k+=2)
                value=hafnian_add_mod(value,hafnian_sum_products2(
                    left[k],right[lane-k],left[k+1],right[lane-k-1],mod),mod.p);
            if(k<=lane-RIGHT_MIN)value=hafnian_add_mod(value,
                hafnian_mul(left[k],right[lane-k],mod),mod.p);
        }
        output[lane]=value;
    }
    __syncwarp();
}

template<unsigned N,class Mod>
__device__ __forceinline__ void invert_polynomial(
    const uint32_t* input,uint32_t* output,Mod mod) {
    constexpr unsigned DEGREE=N/2;
    const unsigned lane=threadIdx.x&31;
    // The symmetric update metric has diagonal entries +/-1.  Off-diagonal
    // series have valuation one, so Schur products cannot change a pivot's
    // constant term; its inverse constant is the same +/-1.
    const uint32_t inverse_constant=input[0];
    if(lane==0)output[0]=inverse_constant;
    __syncwarp();
    for(unsigned degree=1;degree<=DEGREE;++degree) {
        uint32_t sum=0;
        for(unsigned k=lane+1;k<=degree;k+=32)sum=hafnian_add_mod(sum,
            hafnian_mul(input[k],output[degree-k],mod),mod.p);
        for(unsigned offset=16;offset;offset>>=1)
            sum=hafnian_add_mod(sum,__shfl_down_sync(0xffffffff,sum,offset),mod.p);
        if(lane==0)output[degree]=inverse_constant==mod.one?
            hafnian_neg_mod(sum,mod.p):sum;
        __syncwarp();
    }
}

template<unsigned N,class Mod>
__device__ __forceinline__ void add_four_terms(
    const uint32_t* diagonal,const uint32_t* beta,const uint32_t* metric,
    uint32_t* future,uint64_t base_index,Mod mod,
    const uint32_t* inverse_small,uint32_t* q,uint32_t* dense,
    uint32_t* first,uint32_t* second,uint32_t* third,uint32_t* fourth,
    uint32_t* determinant,uint32_t* inverse,uint32_t& local_sum) {
    constexpr unsigned HALF=N/2,S=HALF+1,M=UPDATE_COLUMNS;
    const unsigned lane=threadIdx.x&31;

    characteristic_from_tridiagonal<N>(
        diagonal,beta,N,mod,first,second,third,fourth);
    uint32_t contribution=term_from_characteristic<N>(
        first,base_index^(base_index>>1),mod,inverse_small,second);
    if(lane==0)local_sum=hafnian_add_mod(local_sum,contribution,mod.p);
    __syncwarp();

    // In the +/- factor basis C is diagonal with entries +/-1.  Use
    // det(I-z C G)=det(C) det(C-zG); the latter matrix is symmetric.
    uint32_t* power0=dense;
    uint32_t* power1=power0+M*N;
    uint32_t* gram=power1+M*N;
    for(unsigned index=lane;index<M*M*S;index+=32)q[index]=0;
    for(unsigned i=lane;i<M;i+=32) {
        const uint64_t index=base_index+i/2+1;
        const unsigned edge=unsigned(__ffsll(index));
        const uint64_t signs=index^(index>>1);
        const uint32_t sign=(signs&(UINT64_C(1)<<(edge-1)))?
            mod.one:hafnian_neg_mod(mod.one,mod.p);
        q[(size_t(i)*M+i)*S]=(i&1)?hafnian_neg_mod(sign,mod.p):sign;
    }
    for(unsigned index=lane;index<M*N;index+=32) {
        const unsigned row=index%N;
        const uint32_t value=future[index];
        power0[index]=value;
        future[index]=hafnian_mul(metric[row],value,mod);
    }
    __syncwarp();
    for(unsigned degree=0;degree<HALF;++degree) {
        constexpr unsigned DOT_GROUP=4,GROUPS=32/DOT_GROUP;
        const unsigned group=lane/DOT_GROUP,within=lane&(DOT_GROUP-1);
        constexpr unsigned PAIRS=M*(M+1)/2;
        for(unsigned batch=0;batch*GROUPS<PAIRS;++batch) {
            const unsigned wanted=batch*GROUPS+group;
            unsigned pair_row=0,remainder=wanted;
            while(pair_row<M&&remainder>=M-pair_row)
                remainder-=M-pair_row++;
            const unsigned pair_column=pair_row+remainder;
            uint32_t value=0;
            if(wanted<PAIRS) {
                unsigned coordinate=within;
                for(;coordinate+3*DOT_GROUP<N;coordinate+=4*DOT_GROUP)
                    value=hafnian_add_mod(value,hafnian_sum_products4(
                        future[size_t(pair_row)*N+coordinate],
                        power0[size_t(pair_column)*N+coordinate],
                        future[size_t(pair_row)*N+coordinate+DOT_GROUP],
                        power0[size_t(pair_column)*N+coordinate+DOT_GROUP],
                        future[size_t(pair_row)*N+coordinate+2*DOT_GROUP],
                        power0[size_t(pair_column)*N+coordinate+2*DOT_GROUP],
                        future[size_t(pair_row)*N+coordinate+3*DOT_GROUP],
                        power0[size_t(pair_column)*N+coordinate+3*DOT_GROUP],mod),mod.p);
                for(;coordinate<N;coordinate+=DOT_GROUP)
                    value=hafnian_add_mod(value,hafnian_mul(
                        future[size_t(pair_row)*N+coordinate],
                        power0[size_t(pair_column)*N+coordinate],mod),mod.p);
            }
            for(unsigned offset=DOT_GROUP/2;offset;offset>>=1)
                value=hafnian_add_mod(value,
                    __shfl_down_sync(0xffffffff,value,offset,DOT_GROUP),mod.p);
            if(!within&&wanted<PAIRS)
                gram[size_t(pair_row)*M+pair_column]=
                    gram[size_t(pair_column)*M+pair_row]=value;
        }
        __syncwarp();
        for(unsigned cell=lane;cell<M*M;cell+=32)
            q[size_t(cell)*S+degree+1]=hafnian_neg_mod(gram[cell],mod.p);
        for(unsigned index=lane;index<M*N;index+=32) {
            const unsigned column=index/N,row=index%N;
            uint32_t value=hafnian_mul(
                diagonal[row],power0[size_t(column)*N+row],mod);
            if(row)value=hafnian_add_mod(value,
                power0[size_t(column)*N+row-1],mod.p);
            if(row+1<N)value=hafnian_add_mod(value,hafnian_mul(
                beta[row+1],power0[size_t(column)*N+row+1],mod),mod.p);
            power1[index]=value;
        }
        __syncwarp();
        uint32_t* temporary=power0;power0=power1;power1=temporary;
    }

    for(unsigned i=lane;i<S;i+=32)determinant[i]=i?0:mod.one;
    __syncwarp();
    for(unsigned pivot=0;pivot<M;++pivot) {
        uint32_t* pivot_poly=q+(size_t(pivot)*M+pivot)*S;
        invert_polynomial<N>(pivot_poly,inverse,mod);
        multiply_polynomials<N>(determinant,pivot_poly,third,mod);
        for(unsigned i=lane;i<S;i+=32)determinant[i]=third[i];
        __syncwarp();
        for(unsigned row=pivot+1;row<M;++row) {
            const uint32_t* source=q+(size_t(row)*M+pivot)*S;
            uint32_t* multiplier=q+(size_t(pivot)*M+row)*S;
            multiply_polynomials<N,1,0>(source,inverse,third,mod);
            for(unsigned i=lane;i<S;i+=32)multiplier[i]=third[i];
            __syncwarp();
        }
        for(unsigned row=pivot+1;row<M;++row)
            for(unsigned column=pivot+1;column<=row;++column) {
                const uint32_t* multiplier=q+(size_t(pivot)*M+row)*S;
                const uint32_t* right=q+(size_t(column)*M+pivot)*S;
                uint32_t* target=q+(size_t(row)*M+column)*S;
                multiply_polynomials<N,1,1>(multiplier,right,third,mod);
                for(unsigned i=lane;i<S;i+=32)
                    target[i]=hafnian_sub_mod(target[i],third[i],mod.p);
                __syncwarp();
            }
        if(pivot&1) {
            multiply_polynomials<N>(first,determinant,third,mod);
            const unsigned stage=(pivot+1)/2;
            if(stage&1)for(unsigned i=lane;i<S;i+=32)
                third[i]=hafnian_neg_mod(third[i],mod.p);
            __syncwarp();
            const uint64_t index=base_index+stage;
            contribution=term_from_characteristic<N>(
                third,index^(index>>1),mod,inverse_small,second);
            if(lane==0)local_sum=hafnian_add_mod(local_sum,contribution,mod.p);
            __syncwarp();
        }
    }
}

template<unsigned N,class Mod>
__global__ __launch_bounds__(THREADS,min_blocks_per_sm<N,Mod>()) void terms_kernel(
    const uint32_t* __restrict__ edge_matrices,
    const uint32_t* __restrict__ update_vectors,
    const uint32_t* __restrict__ fixed_metric,
    uint64_t begin,uint64_t end,Mod mod,const uint32_t* __restrict__ inverse_small,
    uint32_t* __restrict__ scratch,uint32_t* __restrict__ chain_sums,
    uint32_t* __restrict__ failures) {
    static_assert(N==48||N==50||N==52,
        "the measured resolver backend is specialized for orders 48, 50, and 52");
    constexpr unsigned HALF=N/2,S=HALF+1,M=UPDATE_COLUMNS;
    static_assert(M*M*S<=N*N);
    const unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const size_t slot=size_t(blockIdx.x)*WARPS_PER_BLOCK+warp;
    const size_t slots=size_t(gridDim.x)*WARPS_PER_BLOCK;
    extern __shared__ uint32_t shared[];
    uint32_t* work=shared+warp*6*N;
    uint32_t* vector0=work;
    uint32_t* vector1=vector0+N;
    uint32_t* vector2=vector1+N;
    uint32_t* vector3=vector2+N;
    uint32_t* first=work;
    uint32_t* second=first+S;
    uint32_t* third=second+S;
    uint32_t* fourth=third+S;
    uint32_t* determinant=fourth+S;
    uint32_t* inverse=determinant+S;

    uint32_t* dense=scratch+slot*N*N;
    uint32_t* basis=scratch+(slots+slot)*N*N;
    uint32_t* future=scratch+2*slots*N*N+slot*FUTURE_COLUMNS*N;
    uint32_t* refresh_factors=scratch+2*slots*N*N+
        slots*FUTURE_COLUMNS*N+slot*2*RESOLVENT_CHAIN*N;
    uint32_t* state=scratch+2*slots*N*N+
        slots*(FUTURE_COLUMNS+2*RESOLVENT_CHAIN)*N+slot*8*N;
    uint32_t* metric0=state;
    uint32_t* metric1=metric0+N;
    uint32_t* inverse_metric0=metric1+N;
    uint32_t* inverse_metric1=inverse_metric0+N;
    uint32_t* diagonal0=inverse_metric1+N;
    uint32_t* diagonal1=diagonal0+N;
    uint32_t* beta0=diagonal1+N;
    uint32_t* beta1=beta0+N;
    uint32_t* q=dense+2*M*N+M*M;
    static_assert(2*M*N+M*M+M*M*S<=N*N);

    uint32_t local_sum=0,local_failures=0;
    const uint64_t stride=uint64_t(slots)*CHAIN;
    for(uint64_t chain_begin=begin+uint64_t(slot)*CHAIN;
            chain_begin<end;chain_begin+=stride) {
        const uint64_t signs0=chain_begin^(chain_begin>>1);
        for(unsigned stage=1;stage<CHAIN;++stage) {
            const uint64_t index=chain_begin+stage;
            const unsigned edge=unsigned(__ffsll(index));
            const uint32_t* source=update_vectors+size_t(edge)*2*N;
            for(unsigned row=lane;row<N;row+=32) {
                future[(size_t(stage-1)*2+0)*N+row]=source[row];
                future[(size_t(stage-1)*2+1)*N+row]=source[N+row];
            }
        }
        constexpr size_t CELLS=size_t(N)*N;
        for(size_t cell=lane;cell<CELLS;cell+=32) {
            uint32_t value=0;
            for(unsigned edge=0;edge<HALF;++edge) {
                const uint32_t addend=edge_matrices[size_t(edge)*CELLS+cell];
                const bool positive=edge==0||(signs0&(UINT64_C(1)<<(edge-1)));
                value=positive?hafnian_add_mod(value,addend,mod.p):
                    hafnian_sub_mod(value,addend,mod.p);
            }
            dense[cell]=value;
        }
        __syncwarp();
        bool ok=false;
        for(unsigned attempt=0;attempt<4&&!ok;++attempt)
            ok=hafnian_gray::generalized_lanczos(
                dense,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0,
                fixed_metric,N,hafnian_gray::splitmix64(chain_begin)^attempt,
                mod,basis,metric0,inverse_metric0,diagonal0,beta0,
                vector0,vector1,vector2,vector3);
        if(!ok) {++local_failures;continue;}
        for(unsigned stage=1;stage<CHAIN;++stage) {
            uint32_t* z0=future+(size_t(stage-1)*2+0)*N;
            uint32_t* z1=future+(size_t(stage-1)*2+1)*N;
            hafnian_gray::inverse_basis_apply_pair(
                basis,fixed_metric,inverse_metric0,z0,z1,
                vector0,vector1,vector2,vector3,N,mod);
            for(unsigned row=lane;row<N;row+=32) {
                if(stage<RESOLVENT_CHAIN) {
                    z0[row]=hafnian_add_mod(vector0[row],vector1[row],mod.p);
                    z1[row]=hafnian_sub_mod(vector0[row],vector1[row],mod.p);
                } else {
                    z0[row]=vector0[row];z1[row]=vector1[row];
                }
                if(stage<=RESOLVENT_CHAIN) {
                    refresh_factors[(size_t(stage-1)*2+0)*N+row]=
                        hafnian_add_mod(vector0[row],vector1[row],mod.p);
                    refresh_factors[(size_t(stage-1)*2+1)*N+row]=
                        hafnian_sub_mod(vector0[row],vector1[row],mod.p);
                }
            }
            __syncwarp();
        }

        add_four_terms<N>(diagonal0,beta0,metric0,future,chain_begin,
            mod,inverse_small,q,dense,first,second,third,fourth,
            determinant,inverse,local_sum);
        const uint64_t refresh_index=chain_begin+RESOLVENT_CHAIN;
        // The resolver evaluated terms 1--3 without advancing the Lanczos
        // state.  Apply all four accumulated Gray updates as one exact signed
        // rank-eight correction to the term-zero tridiagonal matrix.
        ok=false;
        for(unsigned attempt=0;attempt<4&&!ok;++attempt)
            ok=hafnian_gray::generalized_lanczos<2*RESOLVENT_CHAIN>(
                nullptr,diagonal0,beta0,refresh_factors,nullptr,
                nullptr,nullptr,uint32_t(chain_begin),metric0,N,
                hafnian_gray::splitmix64(refresh_index)^attempt,mod,
                basis,metric1,inverse_metric1,diagonal1,beta1,
                vector0,vector1,vector2,vector3);
        if(!ok) {++local_failures;continue;}
        for(unsigned stage=RESOLVENT_CHAIN+1;stage<CHAIN;++stage) {
            uint32_t* z0=future+(size_t(stage-1)*2+0)*N;
            uint32_t* z1=future+(size_t(stage-1)*2+1)*N;
            hafnian_gray::inverse_basis_apply_pair(
                basis,metric0,inverse_metric1,z0,z1,
                vector0,vector1,vector2,vector3,N,mod);
            for(unsigned row=lane;row<N;row+=32) {
                z0[row]=hafnian_add_mod(vector0[row],vector1[row],mod.p);
                z1[row]=hafnian_sub_mod(vector0[row],vector1[row],mod.p);
            }
            __syncwarp();
        }
        add_four_terms<N>(diagonal1,beta1,metric1,
            future+2*RESOLVENT_CHAIN*N,refresh_index,
            mod,inverse_small,q,dense,first,second,third,fourth,
            determinant,inverse,local_sum);
    }
    if(lane==0) {
        chain_sums[slot]=hafnian_mul(local_sum,1,mod);
        if(local_failures)atomicAdd(failures,local_failures);
    }
}

template<unsigned N>
constexpr size_t shared_bytes() {
    return size_t(WARPS_PER_BLOCK)*6*N*sizeof(uint32_t);
}

template<unsigned N>
constexpr size_t scratch_words(size_t slots) {
    return size_t(2)*slots*N*N+
        slots*(FUTURE_COLUMNS+2*RESOLVENT_CHAIN)*N+slots*8*N;
}

} // namespace hafnian_resolvent
