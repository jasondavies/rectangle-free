#pragma once

#include "hafnian_gpu_core.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hafnian_gray {

constexpr unsigned WARPS_PER_BLOCK=2;
constexpr unsigned THREADS=32*WARPS_PER_BLOCK;
constexpr unsigned MIN_BLOCKS_PER_SM=16;

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
    std::vector<uint32_t> vectors;
    std::vector<uint32_t> diagonal,metric;
    uint32_t at(unsigned row,unsigned column)const {
        return vectors[size_t(row)*rank+column];
    }
};

template<class Query>
RankFactor factor_adjacency(const Query& query,uint32_t prime) {
    const HostMod mod{prime};
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
    factor.order=n;factor.rank=unsigned(columns.size());factor.diagonal=diagonal;
    factor.metric.resize(factor.rank);
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

struct DeviceFactors {
    uint32_t *edge_matrices=nullptr,*update_vectors=nullptr,*metric=nullptr;
    DeviceFactors()=default;
    DeviceFactors(const DeviceFactors&)=delete;
    DeviceFactors& operator=(const DeviceFactors&)=delete;
    DeviceFactors(DeviceFactors&& other)noexcept:
        edge_matrices(other.edge_matrices),update_vectors(other.update_vectors),
        metric(other.metric) {
        other.edge_matrices=nullptr;other.update_vectors=nullptr;other.metric=nullptr;
    }
    DeviceFactors& operator=(DeviceFactors&& other)noexcept {
        if(this==&other)return *this;
        cudaFree(metric);cudaFree(update_vectors);cudaFree(edge_matrices);
        edge_matrices=other.edge_matrices;update_vectors=other.update_vectors;
        metric=other.metric;
        other.edge_matrices=nullptr;other.update_vectors=nullptr;other.metric=nullptr;
        return *this;
    }
    ~DeviceFactors() {
        cudaFree(metric);cudaFree(update_vectors);cudaFree(edge_matrices);
    }
};

template<unsigned N,class Query,class Mod>
DeviceFactors make_device_factors(const Query& query,const RankFactor& factor,Mod mod) {
    constexpr unsigned HALF=N/2;
    if(query.vertices!=N||factor.order!=N)throw std::runtime_error("Gray order mismatch");
    const unsigned rank=factor.rank;
    std::vector<uint32_t> edge(size_t(HALF)*rank*rank);
    std::vector<uint32_t> updates(size_t(HALF)*2*rank);
    std::vector<uint32_t> metric(rank);
    const HostMod ordinary{mod.p};
    for(unsigned e=0;e<HALF;++e)for(unsigned i=0;i<rank;++i) {
        updates[(size_t(e)*2+0)*rank+i]=encode(
            ordinary.mul(factor.diagonal[i],factor.at(e,i)),mod);
        updates[(size_t(e)*2+1)*rank+i]=encode(
            ordinary.mul(factor.diagonal[i],factor.at(e+HALF,i)),mod);
        for(unsigned j=0;j<rank;++j) {
            uint32_t value=ordinary.add(
                ordinary.mul(factor.at(e,i),factor.at(e+HALF,j)),
                ordinary.mul(factor.at(e+HALF,i),factor.at(e,j)));
            value=ordinary.mul(factor.diagonal[i],value);
            edge[(size_t(e)*rank+i)*rank+j]=encode(value,mod);
        }
    }
    for(unsigned i=0;i<rank;++i)metric[i]=encode(factor.metric[i],mod);
    DeviceFactors result;
    hafnian_cuda_check(cudaMalloc(&result.edge_matrices,edge.size()*sizeof(uint32_t)),
        "allocate Gray edge matrices");
    hafnian_cuda_check(cudaMalloc(&result.update_vectors,updates.size()*sizeof(uint32_t)),
        "allocate Gray update vectors");
    hafnian_cuda_check(cudaMalloc(&result.metric,metric.size()*sizeof(uint32_t)),
        "allocate Gray metric");
    hafnian_cuda_check(cudaMemcpy(result.edge_matrices,edge.data(),edge.size()*sizeof(uint32_t),
        cudaMemcpyHostToDevice),"copy Gray edge matrices");
    hafnian_cuda_check(cudaMemcpy(result.update_vectors,updates.data(),updates.size()*sizeof(uint32_t),
        cudaMemcpyHostToDevice),"copy Gray update vectors");
    hafnian_cuda_check(cudaMemcpy(result.metric,metric.data(),metric.size()*sizeof(uint32_t),
        cudaMemcpyHostToDevice),"copy Gray metric");
    return result;
}

__device__ __forceinline__ uint64_t splitmix64(uint64_t value) {
    value+=UINT64_C(0x9e3779b97f4a7c15);
    value=(value^(value>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    value=(value^(value>>27))*UINT64_C(0x94d049bb133111eb);
    return value^(value>>31);
}

template<class Mod>
__device__ uint32_t dot_metric(const uint32_t* left,const uint32_t* right,
    const uint32_t* metric,unsigned rank,Mod mod) {
    const unsigned lane=threadIdx.x&31;
    uint32_t value=0;
    for(unsigned row=lane;row<rank;row+=32)value=hafnian_add_mod(value,
        hafnian_mul(left[row],hafnian_mul(metric[row],right[row],mod),mod),mod.p);
    for(unsigned offset=16;offset;offset>>=1)
        value=hafnian_add_mod(value,__shfl_down_sync(0xffffffff,value,offset),mod.p);
    return __shfl_sync(0xffffffff,value,0);
}

template<class Mod>
__device__ void dot_metric_pair(const uint32_t* left0,const uint32_t* left1,
    const uint32_t* right,const uint32_t* metric,unsigned rank,Mod mod,
    uint32_t& result0,uint32_t& result1) {
    const unsigned lane=threadIdx.x&31;
    uint32_t value0=0,value1=0;
    for(unsigned row=lane;row<rank;row+=32) {
        const uint32_t weighted=hafnian_mul(metric[row],right[row],mod);
        value0=hafnian_add_mod(value0,hafnian_mul(left0[row],weighted,mod),mod.p);
        value1=hafnian_add_mod(value1,hafnian_mul(left1[row],weighted,mod),mod.p);
    }
    for(unsigned offset=16;offset;offset>>=1) {
        value0=hafnian_add_mod(value0,__shfl_down_sync(0xffffffff,value0,offset),mod.p);
        value1=hafnian_add_mod(value1,__shfl_down_sync(0xffffffff,value1,offset),mod.p);
    }
    result0=__shfl_sync(0xffffffff,value0,0);
    result1=__shfl_sync(0xffffffff,value1,0);
}

template<class Mod>
__device__ void apply_dense(const uint32_t* matrix,const uint32_t* input,
    uint32_t* output,unsigned rank,Mod mod) {
    const unsigned lane=threadIdx.x&31;
    for(unsigned row=lane;row<rank;row+=32) {
        uint32_t sum=0;
        for(unsigned column=0;column<rank;++column)
            sum=hafnian_add_mod(sum,hafnian_mul(
                matrix[size_t(column)*rank+row],input[column],mod),mod.p);
        output[row]=sum;
    }
    __syncwarp();
}

template<class Mod>
__device__ void apply_tridiagonal_update(
    const uint32_t* diagonal,const uint32_t* beta,const uint32_t* metric,
    const uint32_t* z0,const uint32_t* z1,uint32_t delta,
    const uint32_t* input,uint32_t* output,unsigned rank,Mod mod) {
    uint32_t projection0=0,projection1=0;
    dot_metric_pair(z0,z1,input,metric,rank,mod,projection0,projection1);
    const unsigned lane=threadIdx.x&31;
    for(unsigned row=lane;row<rank;row+=32) {
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
    __syncwarp();
}

template<class Mod>
__device__ bool generalized_lanczos(
    const uint32_t* dense,const uint32_t* old_diagonal,const uint32_t* old_beta,
    const uint32_t* low0,const uint32_t* low1,uint32_t delta,
    const uint32_t* old_metric,unsigned rank,uint64_t seed,Mod mod,
    uint32_t* basis,uint32_t* new_metric,uint32_t* inverse_new_metric,
    uint32_t* new_diagonal,uint32_t* new_beta,
    uint32_t* previous,uint32_t* current,uint32_t* next,uint32_t* applied) {
    const unsigned lane=threadIdx.x&31;
    for(unsigned row=lane;row<rank;row+=32) {
        const uint32_t raw=uint32_t(splitmix64(seed+row)%(mod.p-1))+1;
        current[row]=uint32_t(uint64_t(raw)*mod.one%mod.p);
        previous[row]=0;
    }
    __syncwarp();
    uint32_t norm=dot_metric(current,current,old_metric,rank,mod);
    if(!norm)return false;
    if(lane==0)new_beta[0]=0;
    __syncwarp();
    for(unsigned column=0;column<rank;++column) {
        for(unsigned row=lane;row<rank;row+=32)
            basis[size_t(row)*rank+column]=current[row];
        if(dense)apply_dense(dense,current,applied,rank,mod);
        else apply_tridiagonal_update(old_diagonal,old_beta,old_metric,
            low0,low1,delta,current,applied,rank,mod);
        const uint32_t numerator=dot_metric(current,applied,old_metric,rank,mod);
        if(lane==0) {
            inverse_new_metric[column]=hafnian_power(norm,mod.p-2,mod);
            new_metric[column]=norm;
            new_diagonal[column]=hafnian_mul(
                numerator,inverse_new_metric[column],mod);
        }
        __syncwarp();
        for(unsigned row=lane;row<rank;row+=32) {
            uint32_t value=hafnian_sub_mod(applied[row],
                hafnian_mul(new_diagonal[column],current[row],mod),mod.p);
            if(column)value=hafnian_sub_mod(value,
                hafnian_mul(new_beta[column],previous[row],mod),mod.p);
            next[row]=value;
        }
        __syncwarp();
        if(column+1==rank)break;
        const uint32_t next_norm=dot_metric(next,next,old_metric,rank,mod);
        if(!next_norm)return false;
        if(lane==0)new_beta[column+1]=hafnian_mul(
            next_norm,inverse_new_metric[column],mod);
        __syncwarp();
        for(unsigned row=lane;row<rank;row+=32) {
            previous[row]=current[row];current[row]=next[row];
        }
        __syncwarp();
        norm=next_norm;
    }
    return true;
}

template<class Mod>
__device__ void inverse_basis_apply_pair(
    const uint32_t* basis,const uint32_t* old_metric,const uint32_t* inverse_new_metric,
    const uint32_t* input0,const uint32_t* input1,uint32_t* output0,uint32_t* output1,
    unsigned rank,Mod mod) {
    const unsigned lane=threadIdx.x&31;
    for(unsigned target=lane;target<rank;target+=32) {
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
    __syncwarp();
}

template<unsigned N,class Mod>
__device__ uint32_t term_from_tridiagonal(
    const uint32_t* diagonal,const uint32_t* beta,uint64_t signs,unsigned rank,
    Mod mod,const uint32_t* inverse_small,uint32_t* poly) {
    constexpr unsigned HALF=N/2,STRIDE=HALF+1;
    const unsigned lane=threadIdx.x&31;
    uint32_t* older=poly;
    uint32_t* previous=older+STRIDE;
    uint32_t* next=previous+STRIDE;
    for(unsigned index=lane;index<3*STRIDE;index+=32)poly[index]=0;
    if(lane==0)previous[0]=mod.one;
    __syncwarp();
    for(unsigned size=1;size<=rank;++size) {
        const unsigned at=size-1,maximum=min(size,HALF);
        if(lane<=maximum) {
            const unsigned defect=lane;
            uint32_t value=defect<=size-1?previous[defect]:0;
            if(defect)value=hafnian_sub_mod(value,hafnian_mul(
                diagonal[at],previous[defect-1],mod),mod.p);
            if(size>1&&defect>=2)value=hafnian_sub_mod(value,hafnian_mul(
                beta[at],older[defect-2],mod),mod.p);
            next[defect]=value;
        }
        __syncwarp();
        uint32_t* temporary=older;older=previous;previous=next;next=temporary;
    }
    uint32_t* coefficients=older;
    if(lane<=HALF)coefficients[lane]=0;
    if(lane==0)coefficients[0]=mod.one;
    __syncwarp();
    for(unsigned degree=1;degree<=HALF;++degree) {
        uint32_t sum=0;
        for(unsigned k=lane+1;k<=degree;k+=32) {
            const uint32_t factor=uint32_t(uint64_t(2*degree-k)*mod.one%mod.p);
            sum=hafnian_add_mod(sum,hafnian_mul(
                hafnian_mul(previous[k],coefficients[degree-k],mod),factor,mod),mod.p);
        }
        for(unsigned offset=16;offset;offset>>=1)
            sum=hafnian_add_mod(sum,__shfl_down_sync(0xffffffff,sum,offset),mod.p);
        if(lane==0)coefficients[degree]=hafnian_neg_mod(hafnian_mul(
            hafnian_mul(sum,inverse_small[2],mod),inverse_small[degree],mod),mod.p);
        __syncwarp();
    }
    uint32_t result=0;
    if(lane==0) {
        const unsigned negatives=(HALF-1)-__popcll(signs);
        result=negatives&1?hafnian_neg_mod(coefficients[HALF],mod.p):coefficients[HALF];
    }
    return __shfl_sync(0xffffffff,result,0);
}

template<unsigned N,unsigned CHAIN,class Mod>
__global__ __launch_bounds__(THREADS,MIN_BLOCKS_PER_SM) void terms_kernel(
    const uint32_t* __restrict__ edge_matrices,
    const uint32_t* __restrict__ update_vectors,
    const uint32_t* __restrict__ fixed_metric,unsigned rank,
    uint64_t begin,uint64_t end,Mod mod,const uint32_t* __restrict__ inverse_small,
    uint32_t* __restrict__ scratch,uint32_t* __restrict__ chain_sums,
    uint32_t* __restrict__ failures) {
    static_assert(CHAIN>=1&&CHAIN<=8);
    constexpr unsigned HALF=N/2;
    const unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const size_t slot=size_t(blockIdx.x)*WARPS_PER_BLOCK+warp;
    const size_t slots=size_t(gridDim.x)*WARPS_PER_BLOCK;
    extern __shared__ uint32_t shared[];
    uint32_t* vectors=shared+warp*4*N;
    uint32_t* previous=vectors;
    uint32_t* current=previous+N;
    uint32_t* next=current+N;
    uint32_t* applied=next+N;
    uint32_t* temporary0=previous;
    uint32_t* temporary1=current;
    uint32_t* poly=vectors;

    uint32_t* dense=scratch+slot*N*N;
    uint32_t* basis=scratch+(slots+slot)*N*N;
    uint32_t* future=scratch+2*slots*N*N+slot*2*(CHAIN-1)*N;
    uint32_t* state=scratch+2*slots*N*N+slots*2*(CHAIN-1)*N+slot*8*N;
    uint32_t* metrics=state;
    uint32_t* inverse_metrics=metrics+2*N;
    uint32_t* diagonals=inverse_metrics+2*N;
    uint32_t* betas=diagonals+2*N;

    uint32_t local_sum=0,local_failures=0;
    const uint64_t chain_stride=uint64_t(slots)*CHAIN;
    for(uint64_t chain_begin=begin+uint64_t(slot)*CHAIN;
            chain_begin<end;chain_begin+=chain_stride) {
        const uint64_t signs0=chain_begin^(chain_begin>>1);
        const uint64_t chain_end=min(end,chain_begin+CHAIN);
        for(unsigned step=1;step<chain_end-chain_begin;++step) {
            const unsigned edge=unsigned(__ffsll(chain_begin+step));
            const uint32_t* source=update_vectors+size_t(edge)*2*rank;
            for(unsigned row=lane;row<rank;row+=32) {
                future[(size_t(step-1)*2+0)*N+row]=source[row];
                future[(size_t(step-1)*2+1)*N+row]=source[rank+row];
            }
        }
        const size_t cells=size_t(rank)*rank;
        for(size_t cell=lane;cell<cells;cell+=32) {
            uint32_t value=0;
            for(unsigned edge=0;edge<HALF;++edge) {
                const uint32_t addend=edge_matrices[size_t(edge)*cells+cell];
                const bool positive=edge==0||(signs0&(UINT64_C(1)<<(edge-1)));
                value=positive?hafnian_add_mod(value,addend,mod.p):
                    hafnian_sub_mod(value,addend,mod.p);
            }
            dense[size_t(cell%rank)*rank+cell/rank]=value;
        }
        __syncwarp();
        bool ok=false;
        for(unsigned attempt=0;attempt<4&&!ok;++attempt)
            ok=generalized_lanczos(dense,nullptr,nullptr,nullptr,nullptr,0,
                fixed_metric,rank,splitmix64(chain_begin)^attempt,mod,basis,
                metrics,inverse_metrics,diagonals,betas,
                previous,current,next,applied);
        if(!ok) {++local_failures;continue;}
        for(unsigned future_step=1;future_step<chain_end-chain_begin;++future_step) {
            uint32_t* first=future+(size_t(future_step-1)*2+0)*N;
            uint32_t* second=future+(size_t(future_step-1)*2+1)*N;
            inverse_basis_apply_pair(basis,fixed_metric,inverse_metrics,
                first,second,temporary0,temporary1,rank,mod);
            for(unsigned row=lane;row<rank;row+=32) {
                first[row]=temporary0[row];second[row]=temporary1[row];
            }
            __syncwarp();
        }
        uint32_t contribution=term_from_tridiagonal<N>(
            diagonals,betas,signs0,rank,mod,inverse_small,poly);
        if(lane==0)local_sum=hafnian_add_mod(local_sum,contribution,mod.p);
        __syncwarp();

        unsigned current_buffer=0;
        for(uint64_t index=chain_begin+1;index<chain_end;++index) {
            const uint64_t signs=index^(index>>1);
            const unsigned edge=unsigned(__ffsll(index));
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
                    splitmix64(index)^attempt,mod,basis,metrics+next_buffer*N,
                    inverse_metrics+next_buffer*N,diagonals+next_buffer*N,
                    betas+next_buffer*N,previous,current,next,applied);
            if(!ok)break;
            for(unsigned future_step=stage+1;
                    future_step<chain_end-chain_begin;++future_step) {
                uint32_t* first=future+(size_t(future_step-1)*2+0)*N;
                uint32_t* second=future+(size_t(future_step-1)*2+1)*N;
                inverse_basis_apply_pair(basis,old_metric,
                    inverse_metrics+next_buffer*N,first,second,
                    temporary0,temporary1,rank,mod);
                for(unsigned row=lane;row<rank;row+=32) {
                    first[row]=temporary0[row];second[row]=temporary1[row];
                }
                __syncwarp();
            }
            current_buffer=next_buffer;
            contribution=term_from_tridiagonal<N>(diagonals+current_buffer*N,
                betas+current_buffer*N,signs,rank,mod,inverse_small,poly);
            if(lane==0)local_sum=hafnian_add_mod(local_sum,contribution,mod.p);
            __syncwarp();
        }
        if(!ok)++local_failures;
        __syncwarp();
    }
    if(lane==0) {
        chain_sums[slot]=hafnian_mul(local_sum,1,mod);
        if(local_failures)atomicAdd(failures,local_failures);
    }
}

template<unsigned N>
constexpr size_t shared_bytes() {return size_t(WARPS_PER_BLOCK)*4*N*sizeof(uint32_t);}

template<unsigned N,unsigned CHAIN>
constexpr size_t scratch_words(size_t slots) {
    return size_t(2)*slots*N*N+slots*2*(CHAIN-1)*N+slots*8*N;
}

} // namespace hafnian_gray
