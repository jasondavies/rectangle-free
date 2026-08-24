// Exact CUDA gate for Gray-ordered rank-two hafnian updates.
//
// A symmetric residual adjacency matrix has an exact factorisation A=X D X^T.
// The nonzero characteristic factor of the signed Glynn matrix is that of
//
//     K_s = D X^T R_s X.
//
// Consecutive Gray signs change K_s by rank two.  A two-warp CTA keeps a short
// chain in shared memory, periodically rebuilds K_s, and eagerly carries all
// future rank-two factors through each disposable generalized-Lanczos basis.
// The intervening changes require O(L r^2) work per term.  This is
// a probe, not a production backend: any Lanczos breakdown is reported and
// invalidates the comparison rather than falling back silently.

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
#include <type_traits>
#include <vector>

#include "../../src/hafnian/hafnian_gpu_core.cuh"
#include "../../src/hafnian/six_by_twenty_eight_catalog.hpp"

namespace {

using six_by_twenty_eight::Query;

uint64_t number(const std::string& text) {
    char* end=nullptr;
    const uint64_t value=std::strtoull(text.c_str(),&end,10);
    if(!end||*end)throw std::runtime_error("invalid integer: "+text);
    return value;
}

struct HostMod {
    uint32_t p;
    uint32_t add(uint32_t a,uint32_t b)const {
        const uint32_t value=a+b;
        return value>=p?value-p:value;
    }
    uint32_t sub(uint32_t a,uint32_t b)const {
        return a>=b?a-b:uint32_t(uint64_t(a)+p-b);
    }
    uint32_t mul(uint32_t a,uint32_t b)const{return uint32_t(uint64_t(a)*b%p);}
    uint32_t power(uint32_t a,uint64_t exponent)const {
        uint32_t result=1;
        while(exponent) {
            if(exponent&1)result=mul(result,a);
            a=mul(a,a);exponent>>=1;
        }
        return result;
    }
    uint32_t inverse(uint32_t a)const {
        if(!a)throw std::runtime_error("zero modular inverse");
        return power(a,p-2);
    }
};

struct HostMatrix {
    unsigned n=0;
    std::vector<uint32_t> values;
    explicit HostMatrix(unsigned order=0):n(order),values(size_t(order)*order){}
    uint32_t& at(unsigned row,unsigned column){return values[size_t(row)*n+column];}
};

struct RankFactor {
    unsigned order=0,rank=0;
    std::vector<uint32_t> vectors; // ordinary residues, order x rank
    std::vector<uint32_t> diagonal,metric;
    uint32_t at(unsigned row,unsigned column)const {
        return vectors[size_t(row)*rank+column];
    }
};

RankFactor factor_adjacency(const Query& query,const HostMod& mod) {
    const unsigned n=query.vertices;
    HostMatrix residual(n);
    for(unsigned i=0;i<n;++i)for(unsigned j=0;j<n;++j)
        residual.at(i,j)=query.adjacency[size_t(i)*n+j];
    std::vector<std::vector<uint32_t>> columns;
    std::vector<uint32_t> diagonal;
    for(;;) {
        unsigned first=n,second=n;
        for(unsigned i=0;i<n;++i)if(residual.at(i,i)){first=i;break;}
        std::vector<uint32_t> image(n);
        uint32_t pivot=0;
        if(first<n) {
            pivot=residual.at(first,first);
            for(unsigned row=0;row<n;++row)image[row]=residual.at(row,first);
        } else {
            for(unsigned i=0;i<n&&first==n;++i)for(unsigned j=i+1;j<n;++j)
                if(residual.at(i,j)){first=i;second=j;break;}
            if(first==n)break;
            pivot=mod.add(residual.at(first,second),residual.at(first,second));
            for(unsigned row=0;row<n;++row)
                image[row]=mod.add(residual.at(row,first),residual.at(row,second));
        }
        const uint32_t inverse=mod.inverse(pivot);
        std::vector<uint32_t> vector(n);
        for(unsigned row=0;row<n;++row)vector[row]=mod.mul(image[row],inverse);
        for(unsigned row=0;row<n;++row)for(unsigned column=0;column<n;++column)
            residual.at(row,column)=mod.sub(residual.at(row,column),
                mod.mul(pivot,mod.mul(vector[row],vector[column])));
        columns.push_back(std::move(vector));
        diagonal.push_back(pivot);
    }
    RankFactor factor;
    factor.order=n;factor.rank=unsigned(columns.size());
    factor.diagonal=diagonal;factor.metric.resize(factor.rank);
    factor.vectors.resize(size_t(n)*factor.rank);
    for(unsigned column=0;column<factor.rank;++column) {
        factor.metric[column]=mod.inverse(diagonal[column]);
        for(unsigned row=0;row<n;++row)
            factor.vectors[size_t(row)*factor.rank+column]=columns[column][row];
    }
    for(unsigned row=0;row<n;++row)for(unsigned column=0;column<n;++column) {
        uint32_t value=0;
        for(unsigned k=0;k<factor.rank;++k)value=mod.add(value,mod.mul(
            factor.diagonal[k],mod.mul(factor.at(row,k),factor.at(column,k))));
        if(value!=uint32_t(query.adjacency[size_t(row)*n+column]))
            throw std::runtime_error("rank-factor reconstruction failed");
    }
    return factor;
}

template<class Mod>
uint32_t encode(uint32_t value,Mod mod) {
    return uint32_t(uint64_t(value)*mod.one%mod.p);
}

struct DeviceInput {
    uint32_t *edge_matrices=nullptr,*update_vectors=nullptr,*metric=nullptr;
    uint8_t* adjacency=nullptr;
    uint32_t* inverse_small=nullptr;
    DeviceInput()=default;
    DeviceInput(const DeviceInput&)=delete;
    DeviceInput& operator=(const DeviceInput&)=delete;
    DeviceInput(DeviceInput&& other)noexcept:
        edge_matrices(other.edge_matrices),update_vectors(other.update_vectors),
        metric(other.metric),adjacency(other.adjacency),inverse_small(other.inverse_small) {
        other.edge_matrices=nullptr;other.update_vectors=nullptr;other.metric=nullptr;
        other.adjacency=nullptr;other.inverse_small=nullptr;
    }
    ~DeviceInput() {
        cudaFree(inverse_small);cudaFree(adjacency);cudaFree(metric);
        cudaFree(update_vectors);cudaFree(edge_matrices);
    }
};

template<unsigned N,class Mod>
DeviceInput make_device_input(const Query& query,const RankFactor& factor,Mod mod) {
    constexpr unsigned HALF=N/2;
    if(query.vertices!=N||factor.order!=N)throw std::runtime_error("order mismatch");
    const unsigned r=factor.rank;
    std::vector<uint32_t> edge(size_t(HALF)*r*r);
    std::vector<uint32_t> updates(size_t(HALF)*2*r);
    std::vector<uint32_t> metric(r),inverses(HALF+1);
    const HostMod ordinary{mod.p};
    for(unsigned e=0;e<HALF;++e)for(unsigned i=0;i<r;++i) {
        updates[(size_t(e)*2+0)*r+i]=encode(
            ordinary.mul(factor.diagonal[i],factor.at(e,i)),mod);
        updates[(size_t(e)*2+1)*r+i]=encode(
            ordinary.mul(factor.diagonal[i],factor.at(e+HALF,i)),mod);
        for(unsigned j=0;j<r;++j) {
            uint32_t value=ordinary.add(
                ordinary.mul(factor.at(e,i),factor.at(e+HALF,j)),
                ordinary.mul(factor.at(e+HALF,i),factor.at(e,j)));
            value=ordinary.mul(factor.diagonal[i],value);
            edge[(size_t(e)*r+i)*r+j]=encode(value,mod);
        }
    }
    for(unsigned i=0;i<r;++i)metric[i]=encode(factor.metric[i],mod);
    for(unsigned i=1;i<=HALF;++i)inverses[i]=encode(ordinary.inverse(i),mod);
    DeviceInput result;
    hafnian_cuda_check(cudaMalloc(&result.edge_matrices,edge.size()*sizeof(uint32_t)),
        "allocate Gray edge matrices");
    hafnian_cuda_check(cudaMalloc(&result.update_vectors,updates.size()*sizeof(uint32_t)),
        "allocate Gray update vectors");
    hafnian_cuda_check(cudaMalloc(&result.metric,metric.size()*sizeof(uint32_t)),
        "allocate Gray metric");
    hafnian_cuda_check(cudaMalloc(&result.adjacency,query.adjacency.size()),
        "allocate baseline adjacency");
    hafnian_cuda_check(cudaMalloc(&result.inverse_small,inverses.size()*sizeof(uint32_t)),
        "allocate inverse table");
    hafnian_cuda_check(cudaMemcpy(result.edge_matrices,edge.data(),edge.size()*sizeof(uint32_t),
        cudaMemcpyHostToDevice),"copy Gray edge matrices");
    hafnian_cuda_check(cudaMemcpy(result.update_vectors,updates.data(),updates.size()*sizeof(uint32_t),
        cudaMemcpyHostToDevice),"copy Gray update vectors");
    hafnian_cuda_check(cudaMemcpy(result.metric,metric.data(),metric.size()*sizeof(uint32_t),
        cudaMemcpyHostToDevice),"copy Gray metric");
    hafnian_cuda_check(cudaMemcpy(result.adjacency,query.adjacency.data(),query.adjacency.size(),
        cudaMemcpyHostToDevice),"copy baseline adjacency");
    hafnian_cuda_check(cudaMemcpy(result.inverse_small,inverses.data(),inverses.size()*sizeof(uint32_t),
        cudaMemcpyHostToDevice),"copy inverse table");
    return result;
}

__device__ __forceinline__ uint64_t splitmix64(uint64_t value) {
    value+=UINT64_C(0x9e3779b97f4a7c15);
    value=(value^(value>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    value=(value^(value>>27))*UINT64_C(0x94d049bb133111eb);
    return value^(value>>31);
}

template<class Mod>
__device__ uint32_t block_dot_metric(const uint32_t* left,const uint32_t* right,
    const uint32_t* metric,unsigned rank,Mod mod,uint32_t* warp_sums,uint32_t* scalar) {
    uint32_t value=0;
    if(threadIdx.x<rank)value=hafnian_mul(
        left[threadIdx.x],hafnian_mul(metric[threadIdx.x],right[threadIdx.x],mod),mod);
    for(unsigned offset=16;offset;offset>>=1)
        value=hafnian_add_mod(value,__shfl_down_sync(0xffffffff,value,offset),mod.p);
    if((threadIdx.x&31)==0)warp_sums[threadIdx.x>>5]=value;
    __syncthreads();
    if(threadIdx.x==0)
        scalar[0]=hafnian_add_mod(warp_sums[0],warp_sums[1],mod.p);
    __syncthreads();
    return scalar[0];
}

template<class Mod>
__device__ void block_dot_metric_pair(
    const uint32_t* left0,const uint32_t* left1,const uint32_t* right,
    const uint32_t* metric,unsigned rank,Mod mod,uint32_t* warp_sums,
    uint32_t* scalar) {
    uint32_t value0=0,value1=0;
    if(threadIdx.x<rank) {
        const uint32_t weighted=hafnian_mul(metric[threadIdx.x],right[threadIdx.x],mod);
        value0=hafnian_mul(left0[threadIdx.x],weighted,mod);
        value1=hafnian_mul(left1[threadIdx.x],weighted,mod);
    }
    for(unsigned offset=16;offset;offset>>=1) {
        value0=hafnian_add_mod(value0,__shfl_down_sync(0xffffffff,value0,offset),mod.p);
        value1=hafnian_add_mod(value1,__shfl_down_sync(0xffffffff,value1,offset),mod.p);
    }
    if((threadIdx.x&31)==0) {
        const unsigned warp=threadIdx.x>>5;
        warp_sums[warp]=value0;
        warp_sums[2+warp]=value1;
    }
    __syncthreads();
    if(threadIdx.x==0) {
        scalar[0]=hafnian_add_mod(warp_sums[0],warp_sums[1],mod.p);
        scalar[1]=hafnian_add_mod(warp_sums[2],warp_sums[3],mod.p);
    }
    __syncthreads();
}

template<class Mod>
__device__ void apply_dense(const uint32_t* matrix,const uint32_t* input,
    uint32_t* output,unsigned rank,Mod mod) {
    if(threadIdx.x<rank) {
        uint32_t sum=0;
        for(unsigned column=0;column<rank;++column)
            sum=hafnian_add_mod(sum,hafnian_mul(
                matrix[size_t(threadIdx.x)*rank+column],input[column],mod),mod.p);
        output[threadIdx.x]=sum;
    }
    __syncthreads();
}

template<class Mod>
__device__ void apply_tridiagonal_update(
    const uint32_t* diagonal,const uint32_t* beta,const uint32_t* metric,
    const uint32_t* z0,const uint32_t* z1,uint32_t delta,
    const uint32_t* input,uint32_t* output,unsigned rank,Mod mod,
    uint32_t* warp_sums,uint32_t* scalar) {
    block_dot_metric_pair(z0,z1,input,metric,rank,mod,warp_sums,scalar);
    const uint32_t projection0=scalar[0],projection1=scalar[1];
    if(threadIdx.x<rank) {
        const unsigned row=threadIdx.x;
        uint32_t value=hafnian_mul(diagonal[row],input[row],mod);
        if(row)value=hafnian_add_mod(value,input[row-1],mod.p);
        if(row+1<rank)value=hafnian_add_mod(
            value,hafnian_mul(beta[row+1],input[row+1],mod),mod.p);
        value=hafnian_add_mod(value,hafnian_mul(
            z0[row],hafnian_mul(delta,projection1,mod),mod),mod.p);
        value=hafnian_add_mod(value,hafnian_mul(
            z1[row],hafnian_mul(delta,projection0,mod),mod),mod.p);
        output[row]=value;
    }
    __syncthreads();
}

template<class Mod>
__device__ bool generalized_lanczos(
    const uint32_t* dense,const uint32_t* old_diagonal,const uint32_t* old_beta,
    const uint32_t* low0,const uint32_t* low1,uint32_t delta,
    const uint32_t* old_metric,unsigned rank,uint64_t seed,Mod mod,
    uint32_t* basis,uint32_t* new_metric,uint32_t* inverse_new_metric,
    uint32_t* new_diagonal,uint32_t* new_beta,
    uint32_t* previous,uint32_t* current,uint32_t* next,uint32_t* applied,
    uint32_t* warp_sums,uint32_t* scalar) {
    if(threadIdx.x<rank) {
        const uint32_t raw=uint32_t(splitmix64(seed+threadIdx.x)%(mod.p-1))+1;
        current[threadIdx.x]=uint32_t(uint64_t(raw)*mod.one%mod.p);
        previous[threadIdx.x]=0;
    }
    __syncthreads();
    uint32_t norm=block_dot_metric(current,current,old_metric,rank,mod,warp_sums,scalar);
    if(!norm)return false;
    if(threadIdx.x==0)new_beta[0]=0;
    __syncthreads();
    for(unsigned column=0;column<rank;++column) {
        if(threadIdx.x<rank)
            basis[size_t(threadIdx.x)*rank+column]=current[threadIdx.x];
        if(dense)apply_dense(dense,current,applied,rank,mod);
        else apply_tridiagonal_update(old_diagonal,old_beta,old_metric,
            low0,low1,delta,current,applied,rank,mod,warp_sums,scalar);
        const uint32_t numerator=block_dot_metric(
            current,applied,old_metric,rank,mod,warp_sums,scalar);
        if(threadIdx.x==0) {
            inverse_new_metric[column]=hafnian_power(norm,mod.p-2,mod);
            new_metric[column]=norm;
            new_diagonal[column]=hafnian_mul(
                numerator,inverse_new_metric[column],mod);
        }
        __syncthreads();
        if(threadIdx.x<rank) {
            const unsigned row=threadIdx.x;
            uint32_t value=hafnian_sub_mod(applied[row],
                hafnian_mul(new_diagonal[column],current[row],mod),mod.p);
            if(column)value=hafnian_sub_mod(value,
                hafnian_mul(new_beta[column],previous[row],mod),mod.p);
            next[row]=value;
        }
        __syncthreads();
        if(column+1==rank)break;
        const uint32_t next_norm=block_dot_metric(next,next,old_metric,rank,mod,warp_sums,scalar);
        if(!next_norm)return false;
        if(threadIdx.x==0)new_beta[column+1]=hafnian_mul(
            next_norm,inverse_new_metric[column],mod);
        __syncthreads();
        if(threadIdx.x<rank) {
            previous[threadIdx.x]=current[threadIdx.x];
            current[threadIdx.x]=next[threadIdx.x];
        }
        __syncthreads();
        norm=next_norm;
    }
    return true;
}

template<class Mod>
__device__ void inverse_basis_apply_pair(
    const uint32_t* basis,const uint32_t* old_metric,const uint32_t* inverse_new_metric,
    const uint32_t* input0,const uint32_t* input1,uint32_t* output0,uint32_t* output1,
    unsigned rank,Mod mod) {
    if(threadIdx.x<rank) {
        const unsigned target=threadIdx.x;
        uint32_t sum0=0,sum1=0;
        for(unsigned source=0;source<rank;++source) {
            const uint32_t factor=hafnian_mul(
                basis[size_t(source)*rank+target],old_metric[source],mod);
            sum0=hafnian_add_mod(sum0,hafnian_mul(factor,input0[source],mod),mod.p);
            sum1=hafnian_add_mod(sum1,hafnian_mul(factor,input1[source],mod),mod.p);
        }
        output0[target]=hafnian_mul(sum0,inverse_new_metric[target],mod);
        output1[target]=hafnian_mul(sum1,inverse_new_metric[target],mod);
    }
    __syncthreads();
}

template<unsigned N,class Mod>
__device__ uint32_t term_from_tridiagonal(
    const uint32_t* diagonal,const uint32_t* beta,uint64_t signs,unsigned rank,
    Mod mod,const uint32_t* inverse_small,uint32_t* poly) {
    constexpr unsigned HALF=N/2,STRIDE=HALF+1;
    uint32_t* older=poly;
    uint32_t* previous=older+STRIDE;
    uint32_t* next=previous+STRIDE;
    if(threadIdx.x==0) {
        for(unsigned i=0;i<3*STRIDE;++i)poly[i]=0;
        previous[0]=mod.one;
        for(unsigned size=1;size<=rank;++size) {
            for(unsigned defect=0;defect<=HALF;++defect)next[defect]=0;
            const unsigned at=size-1,maximum=min(size,HALF);
            for(unsigned defect=0;defect<=maximum;++defect) {
                uint32_t value=defect<=size-1?previous[defect]:0;
                if(defect)value=hafnian_sub_mod(value,hafnian_mul(
                    diagonal[at],previous[defect-1],mod),mod.p);
                if(size>1&&defect>=2)value=hafnian_sub_mod(value,hafnian_mul(
                    beta[at],older[defect-2],mod),mod.p);
                next[defect]=value;
            }
            uint32_t* temporary=older;older=previous;previous=next;next=temporary;
        }
        uint32_t traces[HALF+1]{},coefficients[HALF+1]{};
        coefficients[0]=mod.one;
        for(unsigned k=1;k<=HALF;++k) {
            uint32_t value=hafnian_mul(
                uint32_t(uint64_t(k)*mod.one%mod.p),previous[k],mod);
            for(unsigned j=1;j<k;++j)value=hafnian_add_mod(value,
                hafnian_mul(previous[j],traces[k-j],mod),mod.p);
            traces[k]=hafnian_neg_mod(value,mod.p);
        }
        for(unsigned degree=1;degree<=HALF;++degree) {
            uint32_t sum=0;
            for(unsigned k=1;k<=degree;++k)sum=hafnian_add_mod(sum,hafnian_mul(
                hafnian_mul(traces[k],inverse_small[2],mod),
                coefficients[degree-k],mod),mod.p);
            coefficients[degree]=hafnian_mul(sum,inverse_small[degree],mod);
        }
        const unsigned negatives=(HALF-1)-__popcll(signs);
        poly[0]=negatives&1?hafnian_neg_mod(coefficients[HALF],mod.p):coefficients[HALF];
    }
    __syncthreads();
    return poly[0];
}

template<unsigned N,unsigned CHAIN,class Mod>
__global__ void gray_update_kernel(
    const uint32_t* __restrict__ edge_matrices,
    const uint32_t* __restrict__ update_vectors,
    const uint32_t* __restrict__ fixed_metric,unsigned rank,
    uint64_t begin,uint64_t end,Mod mod,const uint32_t* __restrict__ inverse_small,
    uint32_t* __restrict__ dense_scratch,uint32_t* __restrict__ block_sums,
    uint32_t* __restrict__ failures) {
    static_assert(CHAIN>=1&&CHAIN<=8);
    constexpr unsigned HALF=N/2,STRIDE=HALF+1;
    extern __shared__ uint32_t shared[];
    // Future rank-two factors are carried eagerly through each new basis.
    // This performs the same matrix-vector products as a lazy basis chain,
    // but lets every basis be discarded immediately and makes shared storage
    // independent of CHAIN at the dominant N^2 scale.
    uint32_t* matrices=shared;                         // one disposable basis: N*N
    uint32_t* metrics=matrices+N*N;                   // ping-pong: 2*N
    uint32_t* inverse_metrics=metrics+2*N;            // ping-pong: 2*N
    uint32_t* diagonals=inverse_metrics+2*N;          // 2*N
    uint32_t* betas=diagonals+2*N;                    // 2*N
    uint32_t* vectors=betas+2*N;
    uint32_t* previous=vectors+0*N;
    uint32_t* current=vectors+1*N;
    uint32_t* next=vectors+2*N;
    uint32_t* applied=vectors+3*N;
    uint32_t* future=vectors+4*N;                     // 2*(CHAIN-1)*N
    uint32_t* temporary0=future+2*(CHAIN-1)*N;
    uint32_t* temporary1=temporary0+N;
    uint32_t* poly=temporary1+N;                      // 3*(HALF+1)
    uint32_t* warp_sums=poly+3*STRIDE;                // 4
    uint32_t* scalar=warp_sums+4;                     // 4
    uint32_t local_sum=0,local_failures=0;

    const uint64_t chain_stride=uint64_t(gridDim.x)*CHAIN;
    uint32_t* dense=dense_scratch+size_t(blockIdx.x)*N*N;
    for(uint64_t chain_begin=begin+uint64_t(blockIdx.x)*CHAIN;
            chain_begin<end;chain_begin+=chain_stride) {
        const uint64_t signs0=chain_begin^(chain_begin>>1);
        const uint64_t chain_end=min(end,chain_begin+CHAIN);
        for(unsigned step=1;step<chain_end-chain_begin;++step) {
            const unsigned edge=unsigned(__ffsll(chain_begin+step));
            const uint32_t* source=update_vectors+size_t(edge)*2*rank;
            if(threadIdx.x<rank) {
                future[(size_t(step-1)*2+0)*N+threadIdx.x]=source[threadIdx.x];
                future[(size_t(step-1)*2+1)*N+threadIdx.x]=source[rank+threadIdx.x];
            }
        }
        const size_t cells=size_t(rank)*rank;
        for(size_t cell=threadIdx.x;cell<cells;cell+=blockDim.x) {
            uint32_t value=0;
            for(unsigned edge=0;edge<HALF;++edge) {
                uint32_t addend=edge_matrices[size_t(edge)*cells+cell];
                const bool positive=edge==0||(signs0&(UINT64_C(1)<<(edge-1)));
                value=positive?hafnian_add_mod(value,addend,mod.p):
                    hafnian_sub_mod(value,addend,mod.p);
            }
            dense[cell]=value;
        }
        __syncthreads();
        bool ok=false;
        for(unsigned attempt=0;attempt<4&&!ok;++attempt)
            ok=generalized_lanczos(
                dense,nullptr,nullptr,nullptr,nullptr,0,
                fixed_metric,rank,splitmix64(chain_begin)^attempt,mod,
                matrices,metrics,inverse_metrics,diagonals,betas,
                previous,current,next,applied,warp_sums,scalar);
        if(!ok) {
            ++local_failures;
            __syncthreads();
            continue;
        }
        // Carry every future update into the base tridiagonal coordinates,
        // allowing the base basis to be overwritten by the first update.
        for(unsigned future_step=1;future_step<chain_end-chain_begin;++future_step) {
            uint32_t* first=future+(size_t(future_step-1)*2+0)*N;
            uint32_t* second=future+(size_t(future_step-1)*2+1)*N;
            inverse_basis_apply_pair(matrices,fixed_metric,inverse_metrics,
                first,second,temporary0,temporary1,rank,mod);
            if(threadIdx.x<rank) {
                first[threadIdx.x]=temporary0[threadIdx.x];
                second[threadIdx.x]=temporary1[threadIdx.x];
            }
            __syncthreads();
        }
        uint32_t contribution=term_from_tridiagonal<N>(
            diagonals,betas,signs0,rank,mod,inverse_small,poly);
        if(threadIdx.x==0)local_sum=hafnian_add_mod(local_sum,contribution,mod.p);
        __syncthreads();

        unsigned current_buffer=0;
        for(uint64_t index=chain_begin+1;index<chain_end;++index) {
            const uint64_t signs=index^(index>>1);
            const unsigned edge=unsigned(__ffsll(index)); // ctz(index)+1
            const unsigned stage=unsigned(index-chain_begin);
            uint32_t* z0=future+(size_t(stage-1)*2+0)*N;
            uint32_t* z1=future+(size_t(stage-1)*2+1)*N;
            const uint32_t* old_metric=metrics+current_buffer*N;
            const uint32_t delta=(signs&(UINT64_C(1)<<(edge-1)))?
                hafnian_add_mod(mod.one,mod.one,mod.p):
                hafnian_neg_mod(hafnian_add_mod(mod.one,mod.one,mod.p),mod.p);
            const unsigned next_buffer=current_buffer^1;
            ok=false;
            for(unsigned attempt=0;attempt<4&&!ok;++attempt)
                ok=generalized_lanczos(nullptr,diagonals+current_buffer*N,
                    betas+current_buffer*N,z0,z1,delta,old_metric,rank,
                    splitmix64(index)^attempt,mod,matrices,
                    metrics+next_buffer*N,inverse_metrics+next_buffer*N,
                    diagonals+next_buffer*N,betas+next_buffer*N,
                    previous,current,next,applied,warp_sums,scalar);
            if(!ok)break;
            for(unsigned future_step=stage+1;
                    future_step<chain_end-chain_begin;++future_step) {
                uint32_t* first=future+(size_t(future_step-1)*2+0)*N;
                uint32_t* second=future+(size_t(future_step-1)*2+1)*N;
                inverse_basis_apply_pair(matrices,old_metric,
                    inverse_metrics+next_buffer*N,first,second,
                    temporary0,temporary1,rank,mod);
                if(threadIdx.x<rank) {
                    first[threadIdx.x]=temporary0[threadIdx.x];
                    second[threadIdx.x]=temporary1[threadIdx.x];
                }
                __syncthreads();
            }
            current_buffer=next_buffer;
            contribution=term_from_tridiagonal<N>(diagonals+current_buffer*N,
                betas+current_buffer*N,signs,rank,mod,inverse_small,poly);
            if(threadIdx.x==0)local_sum=hafnian_add_mod(local_sum,contribution,mod.p);
            __syncthreads();
        }
        if(!ok)++local_failures;
        __syncthreads();
    }
    if(threadIdx.x==0) {
        block_sums[blockIdx.x]=hafnian_mul(local_sum,1,mod);
        if(local_failures)atomicAdd(failures,local_failures);
    }
}

template<unsigned N,unsigned CHAIN>
constexpr size_t gray_shared_bytes() {
    constexpr unsigned HALF=N/2;
    return (N*N+4*N+4*N+4*N+2*(CHAIN-1)*N+2*N+
        3*(HALF+1)+4+4)*sizeof(uint32_t);
}

template<unsigned N,unsigned CHAIN,class Mod>
void run_probe(const Query& query,const RankFactor& factor,uint64_t terms,
    unsigned requested_blocks,unsigned iterations,Mod mod) {
    constexpr unsigned THREADS=64;
    DeviceInput input=make_device_input<N>(query,factor,mod);
    int device=0,multiprocessors=0;
    hafnian_cuda_check(cudaGetDevice(&device),"get CUDA device");
    hafnian_cuda_check(cudaDeviceGetAttribute(&multiprocessors,
        cudaDevAttrMultiProcessorCount,device),"get SM count");
    constexpr size_t dynamic_shared=gray_shared_bytes<N,CHAIN>();
    hafnian_cuda_check(cudaFuncSetAttribute(gray_update_kernel<N,CHAIN,Mod>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,int(dynamic_shared)),
        "set Gray dynamic shared memory");
    int active=0;
    hafnian_cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,
        gray_update_kernel<N,CHAIN,Mod>,THREADS,dynamic_shared),
        "query Gray occupancy");
    if(!active)throw std::runtime_error("Gray kernel has zero occupancy");
    const unsigned gray_blocks=requested_blocks?requested_blocks:
        unsigned(multiprocessors*active*2);
    const unsigned baseline_threads=N<=50?224:256;
    const unsigned baseline_blocks=hafnian_recommended_blocks<N,Mod>(
        baseline_threads,multiprocessors);
    const unsigned maximum_blocks=std::max(gray_blocks,baseline_blocks);
    uint32_t *device_sums=nullptr,*device_failures=nullptr,*device_dense=nullptr;
    hafnian_cuda_check(cudaMalloc(&device_sums,size_t(maximum_blocks)*sizeof(uint32_t)),
        "allocate probe sums");
    hafnian_cuda_check(cudaMalloc(&device_failures,sizeof(uint32_t)),
        "allocate failure count");
    hafnian_cuda_check(cudaMalloc(&device_dense,
        size_t(gray_blocks)*N*N*sizeof(uint32_t)),"allocate dense Gray scratch");
    std::vector<uint32_t> sums(maximum_blocks);
    cudaEvent_t started,finished;
    hafnian_cuda_check(cudaEventCreate(&started),"create start event");
    hafnian_cuda_check(cudaEventCreate(&finished),"create finish event");
    auto time_launch=[&](auto launch,unsigned blocks,uint32_t& residue,uint32_t& failures) {
        float best=1e30f,total=0;
        residue=failures=0;
        for(unsigned iteration=0;iteration<=iterations;++iteration) {
            hafnian_cuda_check(cudaMemset(device_failures,0,sizeof(uint32_t)),"zero failures");
            hafnian_cuda_check(cudaEventRecord(started),"record start");
            launch();
            hafnian_cuda_check(cudaGetLastError(),"launch probe kernel");
            hafnian_cuda_check(cudaEventRecord(finished),"record finish");
            hafnian_cuda_check(cudaEventSynchronize(finished),"synchronize probe");
            float milliseconds=0;
            hafnian_cuda_check(cudaEventElapsedTime(&milliseconds,started,finished),"time probe");
            hafnian_cuda_check(cudaMemcpy(sums.data(),device_sums,
                size_t(blocks)*sizeof(uint32_t),cudaMemcpyDeviceToHost),"copy probe sums");
            hafnian_cuda_check(cudaMemcpy(&failures,device_failures,sizeof(uint32_t),
                cudaMemcpyDeviceToHost),"copy failure count");
            uint32_t current=0;
            for(unsigned block=0;block<blocks;++block) {
                current+=sums[block];if(current>=mod.p)current-=mod.p;
            }
            if(iteration) {best=std::min(best,milliseconds);total+=milliseconds;}
            residue=current;
        }
        return std::array<float,2>{best,total/iterations};
    };
    uint32_t baseline_residue=0,baseline_failures=0;
    const auto baseline=time_launch([&] {
        hafnian_terms_kernel<N,Mod><<<baseline_blocks,baseline_threads,
            hafnian_shared_bytes<N>()>>>(input.adjacency,0,terms,mod,
                input.inverse_small,device_sums);
    },baseline_blocks,baseline_residue,baseline_failures);
    uint32_t gray_residue=0,gray_failures=0;
    const auto gray=time_launch([&] {
        gray_update_kernel<N,CHAIN,Mod><<<gray_blocks,THREADS,dynamic_shared>>>(
            input.edge_matrices,input.update_vectors,input.metric,factor.rank,
            0,terms,mod,input.inverse_small,device_dense,device_sums,device_failures);
    },gray_blocks,gray_residue,gray_failures);
    const bool exact=!gray_failures&&gray_residue==baseline_residue;
    std::printf(
        "GRAY_UPDATE_GPU_RESULT vertices=%u rank=%u chain=%u terms=%" PRIu64
        " prime=%u gray_blocks=%u active_blocks_per_sm=%d shared_bytes=%zu "
        "baseline_ms_best=%.6f baseline_ms_mean=%.6f gray_ms_best=%.6f "
        "gray_ms_mean=%.6f speedup_best=%.6f speedup_mean=%.6f "
        "baseline_residue=%u gray_residue=%u failures=%u exact=%s\n",
        N,factor.rank,CHAIN,terms,mod.p,gray_blocks,active,dynamic_shared,
        baseline[0],baseline[1],gray[0],gray[1],baseline[0]/gray[0],
        baseline[1]/gray[1],baseline_residue,gray_residue,gray_failures,
        exact?"OK":"FAIL");
    cudaEventDestroy(finished);cudaEventDestroy(started);
    cudaFree(device_dense);cudaFree(device_failures);cudaFree(device_sums);
    if(!exact)throw std::runtime_error("Gray GPU result failed exact comparison");
}

template<unsigned N,class Mod>
void dispatch_chain(const Query& query,const RankFactor& factor,uint64_t terms,
    unsigned chain,unsigned blocks,unsigned iterations,Mod mod) {
    switch(chain) {
        case 1:return run_probe<N,1>(query,factor,terms,blocks,iterations,mod);
        case 2:return run_probe<N,2>(query,factor,terms,blocks,iterations,mod);
        case 4:return run_probe<N,4>(query,factor,terms,blocks,iterations,mod);
        case 6:return run_probe<N,6>(query,factor,terms,blocks,iterations,mod);
        case 8:return run_probe<N,8>(query,factor,terms,blocks,iterations,mod);
        default:throw std::runtime_error("chain must be 1, 2, 4, 6, or 8");
    }
}

template<class Mod>
void dispatch_order(const Query& query,const RankFactor& factor,uint64_t terms,
    unsigned chain,unsigned blocks,unsigned iterations,Mod mod) {
    switch(query.vertices) {
        case 48:return dispatch_chain<48>(query,factor,terms,chain,blocks,iterations,mod);
        case 50:return dispatch_chain<50>(query,factor,terms,chain,blocks,iterations,mod);
        default:throw std::runtime_error("GPU Gray probe currently gates orders 48 and 50");
    }
}

template<class Mod>
void run_fixed(const Query& query,uint64_t terms,unsigned chain,unsigned blocks,
    unsigned iterations,Mod mod) {
    const RankFactor factor=factor_adjacency(query,HostMod{mod.p});
    if(factor.rank<query.vertices/2)
        throw std::runtime_error("reduced rank is below required coefficient degree");
    dispatch_order(query,factor,terms,chain,blocks,iterations,mod);
}

} // namespace

int main(int argc,char** argv) {
    try {
        unsigned query_id=3321,chain=4,blocks=0,iterations=3;
        uint64_t terms=UINT64_C(1)<<20;
        uint32_t prime=2147483647U;
        for(int i=1;i<argc;++i) {
            const std::string argument=argv[i];
            auto take=[&]() {
                if(++i>=argc)throw std::runtime_error("missing value for "+argument);
                return std::string(argv[i]);
            };
            if(argument=="--query")query_id=unsigned(number(take()));
            else if(argument=="--terms")terms=number(take());
            else if(argument=="--chain")chain=unsigned(number(take()));
            else if(argument=="--blocks")blocks=unsigned(number(take()));
            else if(argument=="--iterations")iterations=unsigned(number(take()));
            else if(argument=="--prime")prime=uint32_t(number(take()));
            else throw std::runtime_error(
                "usage: hafnian_gray_update_gpu_probe [--query Q] [--terms 2^k] "
                "[--chain 1|2|4|6|8] [--blocks N] [--iterations N] [--prime P]");
        }
        if(!terms||(terms&(terms-1))||!iterations)
            throw std::runtime_error("terms must be a nonzero power of two");
        auto catalog=six_by_twenty_eight::build_catalog();
        if(query_id>=catalog.queries.size())throw std::runtime_error("invalid query");
        const Query& query=catalog.queries[query_id];
        if(terms>(UINT64_C(1)<<(query.vertices/2-1)))
            throw std::runtime_error("term count exceeds sign domain");
        switch(prime) {
            case 2147483647U:return run_fixed(query,terms,chain,blocks,iterations,
                HafnianMontgomeryConstant<2147483647U>{}),0;
            case 2147483629U:return run_fixed(query,terms,chain,blocks,iterations,
                HafnianMontgomeryConstant<2147483629U>{}),0;
            case 2147483587U:return run_fixed(query,terms,chain,blocks,iterations,
                HafnianMontgomeryConstant<2147483587U>{}),0;
            case 2147483579U:return run_fixed(query,terms,chain,blocks,iterations,
                HafnianMontgomeryConstant<2147483579U>{}),0;
            default:throw std::runtime_error("prime is outside the production CRT schedule");
        }
    } catch(const std::exception& error) {
        std::fprintf(stderr,"error: %s\n",error.what());
        return 1;
    }
}
