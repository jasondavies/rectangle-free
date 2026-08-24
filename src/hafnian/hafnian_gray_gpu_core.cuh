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
__device__ void dot_pair(const uint32_t* left0,const uint32_t* left1,
    const uint32_t* right,unsigned rank,Mod mod,
    uint32_t& result0,uint32_t& result1) {
    const unsigned lane=threadIdx.x&31;
    uint32_t value0=0,value1=0;
    for(unsigned row=lane;row<rank;row+=32) {
        value0=hafnian_add_mod(value0,
            hafnian_mul(left0[row],right[row],mod),mod.p);
        value1=hafnian_add_mod(value1,
            hafnian_mul(left1[row],right[row],mod),mod.p);
    }
    for(unsigned offset=16;offset;offset>>=1) {
        value0=hafnian_add_mod(value0,
            __shfl_down_sync(0xffffffff,value0,offset),mod.p);
        value1=hafnian_add_mod(value1,
            __shfl_down_sync(0xffffffff,value1,offset),mod.p);
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
        unsigned column=0;
        for(;column+3<rank;column+=4)
            sum=hafnian_add_mod(sum,hafnian_sum_products4(
                matrix[size_t(column)*rank+row],input[column],
                matrix[size_t(column+1)*rank+row],input[column+1],
                matrix[size_t(column+2)*rank+row],input[column+2],
                matrix[size_t(column+3)*rank+row],input[column+3],mod),mod.p);
        for(;column+1<rank;column+=2)
            sum=hafnian_add_mod(sum,hafnian_sum_products2(
                matrix[size_t(column)*rank+row],input[column],
                matrix[size_t(column+1)*rank+row],input[column+1],mod),mod.p);
        if(column<rank)sum=hafnian_add_mod(sum,hafnian_mul(
            matrix[size_t(column)*rank+row],input[column],mod),mod.p);
        output[row]=sum;
    }
    __syncwarp();
}

template<class Mod>
__device__ void apply_tridiagonal_update(
    const uint32_t* diagonal,const uint32_t* beta,const uint32_t* metric,
    const uint32_t* z0,const uint32_t* z1,
    const uint32_t* weighted_z0,const uint32_t* weighted_z1,uint32_t delta,
    const uint32_t* input,uint32_t* output,unsigned rank,Mod mod) {
    uint32_t projection0=0,projection1=0;
    dot_pair(weighted_z0,weighted_z1,input,rank,mod,projection0,projection1);
    uint32_t correction0=hafnian_add_mod(projection1,projection1,mod.p);
    uint32_t correction1=hafnian_add_mod(projection0,projection0,mod.p);
    if(delta!=hafnian_add_mod(mod.one,mod.one,mod.p)) {
        correction0=hafnian_neg_mod(correction0,mod.p);
        correction1=hafnian_neg_mod(correction1,mod.p);
    }
    const unsigned lane=threadIdx.x&31;
    for(unsigned row=lane;row<rank;row+=32) {
        uint32_t value=hafnian_mul(diagonal[row],input[row],mod);
        if(row)value=hafnian_add_mod(value,input[row-1],mod.p);
        if(row+1<rank)value=hafnian_add_mod(
            value,hafnian_mul(beta[row+1],input[row+1],mod),mod.p);
        value=hafnian_add_mod(value,
            hafnian_mul(z0[row],correction0,mod),mod.p);
        value=hafnian_add_mod(value,
            hafnian_mul(z1[row],correction1,mod),mod.p);
        output[row]=value;
    }
    __syncwarp();
}

template<class Mod>
__device__ __forceinline__ uint32_t warp_inclusive_product(
    uint32_t value,Mod mod) {
    const unsigned lane=threadIdx.x&31;
    for(unsigned offset=1;offset<32;offset<<=1) {
        const uint32_t other=__shfl_up_sync(0xffffffff,value,offset);
        if(lane>=offset)value=hafnian_mul(value,other,mod);
    }
    return value;
}

template<class Mod>
__device__ __forceinline__ uint32_t warp_inclusive_reverse_product(
    uint32_t value,Mod mod) {
    const unsigned lane=threadIdx.x&31;
    for(unsigned offset=1;offset<32;offset<<=1) {
        const uint32_t other=__shfl_down_sync(0xffffffff,value,offset);
        if(lane+offset<32)value=hafnian_mul(value,other,mod);
    }
    return value;
}

template<class Mod>
__device__ bool generalized_lanczos(
    const uint32_t* dense,const uint32_t* old_diagonal,const uint32_t* old_beta,
    const uint32_t* low0,const uint32_t* low1,
    uint32_t* weighted_low0,uint32_t* weighted_low1,uint32_t delta,
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
    if(!dense) {
        for(unsigned row=lane;row<rank;row+=32) {
            weighted_low0[row]=hafnian_mul(old_metric[row],low0[row],mod);
            weighted_low1[row]=hafnian_mul(old_metric[row],low1[row],mod);
        }
        __syncwarp();
    }
    uint32_t previous_norm=0,previous_scale=0;
    for(unsigned column=0;column<rank;++column) {
        for(unsigned row=lane;row<rank;row+=32)
            basis[size_t(row)*rank+column]=current[row];
        if(dense)apply_dense(dense,current,applied,rank,mod);
        else apply_tridiagonal_update(old_diagonal,old_beta,old_metric,
            low0,low1,weighted_low0,weighted_low1,delta,
            current,applied,rank,mod);
        uint32_t norm=0,numerator=0;
        dot_metric_pair(current,applied,current,old_metric,rank,mod,
            norm,numerator);
        if(!norm)return false;
        uint32_t scale=0,middle=0,norm_squared=0;
        if(lane==0) {
            new_metric[column]=norm;
            new_diagonal[column]=numerator;
            if(column==0)scale=norm;
            else {
                const uint32_t common=hafnian_mul(
                    previous_norm,previous_scale,mod);
                scale=hafnian_mul(norm,common,mod);
                middle=hafnian_mul(common,numerator,mod);
            }
            if(column==0)middle=numerator;
            norm_squared=hafnian_mul(norm,norm,mod);
        }
        scale=__shfl_sync(0xffffffff,scale,0);
        middle=__shfl_sync(0xffffffff,middle,0);
        norm_squared=__shfl_sync(0xffffffff,norm_squared,0);
        if(column+1<rank) {
            for(unsigned row=lane;row<rank;row+=32) {
                uint32_t value=hafnian_sub_mod(
                    hafnian_mul(scale,applied[row],mod),
                    hafnian_mul(middle,current[row],mod),mod.p);
                if(column)value=hafnian_sub_mod(value,
                    hafnian_mul(norm_squared,previous[row],mod),mod.p);
                next[row]=value;
            }
            __syncwarp();
            for(unsigned row=lane;row<rank;row+=32) {
                previous[row]=current[row];current[row]=next[row];
            }
            __syncwarp();
        }
        previous_norm=norm;previous_scale=scale;
    }

    // Batch-invert every norm and derive all normalisation coefficients with
    // warp product scans.  The upper lane segment is padded by multiplicative
    // identities, so this covers every maintained rank (at most 64).
    const unsigned upper_column=lane+32;
    const uint32_t norm0=lane<rank?new_metric[lane]:mod.one;
    const uint32_t norm1=upper_column<rank?new_metric[upper_column]:mod.one;
    const uint32_t prefix0=warp_inclusive_product(norm0,mod);
    const uint32_t local_prefix1=warp_inclusive_product(norm1,mod);
    const uint32_t total0=__shfl_sync(0xffffffff,prefix0,31);
    const uint32_t total1=__shfl_sync(0xffffffff,local_prefix1,31);
    const uint32_t prefix1=hafnian_mul(total0,local_prefix1,mod);
    uint32_t inverse_total=lane==0?
        hafnian_power(hafnian_mul(total0,total1,mod),mod.p-2,mod):0;
    inverse_total=__shfl_sync(0xffffffff,inverse_total,0);

    const uint32_t suffix0=warp_inclusive_reverse_product(norm0,mod);
    const uint32_t suffix1=warp_inclusive_reverse_product(norm1,mod);
    const uint32_t shifted_prefix0=__shfl_up_sync(0xffffffff,prefix0,1);
    const uint32_t shifted_prefix1=__shfl_up_sync(0xffffffff,prefix1,1);
    const uint32_t shifted_suffix0=__shfl_down_sync(0xffffffff,suffix0,1);
    const uint32_t shifted_suffix1=__shfl_down_sync(0xffffffff,suffix1,1);
    const uint32_t before0=lane?shifted_prefix0:mod.one;
    const uint32_t before1=lane?shifted_prefix1:total0;
    const uint32_t after0=hafnian_mul(
        lane+1<32?shifted_suffix0:mod.one,total1,mod);
    const uint32_t after1=lane+1<32?shifted_suffix1:mod.one;
    const uint32_t inverse_norm0=hafnian_mul(
        hafnian_mul(before0,after0,mod),inverse_total,mod);
    const uint32_t inverse_norm1=hafnian_mul(
        hafnian_mul(before1,after1,mod),inverse_total,mod);

    // s_j = h_j (product_{i<j} h_i)^2 and c_j = product_{i<j} s_i.
    // Both c_j and its inverse are therefore another pair of product scans.
    const uint32_t inverse_prefix0=hafnian_mul(after0,inverse_total,mod);
    const uint32_t inverse_prefix1=hafnian_mul(after1,inverse_total,mod);
    const uint32_t inverse_total0=__shfl_sync(0xffffffff,inverse_prefix0,31);
    const uint32_t shifted_inverse_prefix0=
        __shfl_up_sync(0xffffffff,inverse_prefix0,1);
    const uint32_t shifted_inverse_prefix1=
        __shfl_up_sync(0xffffffff,inverse_prefix1,1);
    const uint32_t prior_prefix0=lane?shifted_prefix0:mod.one;
    const uint32_t prior_prefix1=lane?shifted_prefix1:total0;
    const uint32_t prior_inverse_prefix0=lane?
        shifted_inverse_prefix0:mod.one;
    const uint32_t prior_inverse_prefix1=lane?
        shifted_inverse_prefix1:inverse_total0;
    const uint32_t scale0=hafnian_mul(norm0,
        hafnian_mul(prior_prefix0,prior_prefix0,mod),mod);
    const uint32_t scale1=hafnian_mul(norm1,
        hafnian_mul(prior_prefix1,prior_prefix1,mod),mod);
    const uint32_t inverse_scale0=hafnian_mul(inverse_norm0,
        hafnian_mul(prior_inverse_prefix0,prior_inverse_prefix0,mod),mod);
    const uint32_t inverse_scale1=hafnian_mul(inverse_norm1,
        hafnian_mul(prior_inverse_prefix1,prior_inverse_prefix1,mod),mod);
    const uint32_t scale_prefix0=warp_inclusive_product(scale0,mod);
    const uint32_t scale_local_prefix1=warp_inclusive_product(scale1,mod);
    const uint32_t scale_total0=__shfl_sync(0xffffffff,scale_prefix0,31);
    const uint32_t scale_prefix1=hafnian_mul(scale_total0,scale_local_prefix1,mod);
    const uint32_t inverse_scale_prefix0=
        warp_inclusive_product(inverse_scale0,mod);
    const uint32_t inverse_scale_local_prefix1=
        warp_inclusive_product(inverse_scale1,mod);
    const uint32_t inverse_scale_total0=
        __shfl_sync(0xffffffff,inverse_scale_prefix0,31);
    const uint32_t inverse_scale_prefix1=hafnian_mul(
        inverse_scale_total0,inverse_scale_local_prefix1,mod);
    const uint32_t shifted_scale_prefix0=
        __shfl_up_sync(0xffffffff,scale_prefix0,1);
    const uint32_t shifted_scale_prefix1=
        __shfl_up_sync(0xffffffff,scale_prefix1,1);
    const uint32_t shifted_inverse_scale_prefix0=
        __shfl_up_sync(0xffffffff,inverse_scale_prefix0,1);
    const uint32_t shifted_inverse_scale_prefix1=
        __shfl_up_sync(0xffffffff,inverse_scale_prefix1,1);
    const uint32_t basis_scale0=lane?shifted_scale_prefix0:mod.one;
    const uint32_t basis_scale1=lane?shifted_scale_prefix1:scale_total0;
    const uint32_t inverse_basis_scale0=lane?
        shifted_inverse_scale_prefix0:mod.one;
    const uint32_t inverse_basis_scale1=lane?
        shifted_inverse_scale_prefix1:inverse_scale_total0;

    const uint32_t shifted_inverse_norm0=
        __shfl_up_sync(0xffffffff,inverse_norm0,1);
    const uint32_t shifted_inverse_norm1=
        __shfl_up_sync(0xffffffff,inverse_norm1,1);
    const uint32_t shifted_inverse_scale0=
        __shfl_up_sync(0xffffffff,inverse_scale0,1);
    const uint32_t shifted_inverse_scale1=
        __shfl_up_sync(0xffffffff,inverse_scale1,1);
    const uint32_t last_inverse_norm0=
        __shfl_sync(0xffffffff,inverse_norm0,31);
    const uint32_t last_inverse_scale0=
        __shfl_sync(0xffffffff,inverse_scale0,31);
    const uint32_t previous_inverse_norm0=lane?shifted_inverse_norm0:0;
    const uint32_t previous_inverse_scale0=lane?shifted_inverse_scale0:0;
    const uint32_t previous_inverse_norm1=lane?
        shifted_inverse_norm1:last_inverse_norm0;
    const uint32_t previous_inverse_scale1=lane?
        shifted_inverse_scale1:last_inverse_scale0;
    if(lane<rank) {
        new_beta[lane]=lane?hafnian_mul(hafnian_mul(
            norm0,previous_inverse_norm0,mod),hafnian_mul(
            previous_inverse_scale0,previous_inverse_scale0,mod),mod):0;
        new_diagonal[lane]=hafnian_mul(
            new_diagonal[lane],inverse_norm0,mod);
        new_metric[lane]=hafnian_mul(norm0,hafnian_mul(
            inverse_basis_scale0,inverse_basis_scale0,mod),mod);
        inverse_new_metric[lane]=hafnian_mul(
            inverse_norm0,basis_scale0,mod);
    }
    if(upper_column<rank) {
        new_beta[upper_column]=hafnian_mul(hafnian_mul(
            norm1,previous_inverse_norm1,mod),hafnian_mul(
            previous_inverse_scale1,previous_inverse_scale1,mod),mod);
        new_diagonal[upper_column]=hafnian_mul(
            new_diagonal[upper_column],inverse_norm1,mod);
        new_metric[upper_column]=hafnian_mul(norm1,hafnian_mul(
            inverse_basis_scale1,inverse_basis_scale1,mod),mod);
        inverse_new_metric[upper_column]=hafnian_mul(
            inverse_norm1,basis_scale1,mod);
    }
    __syncwarp();
    return true;
}

template<class Mod>
__device__ void inverse_basis_apply_pair(
    const uint32_t* basis,const uint32_t* old_metric,const uint32_t* inverse_new_metric,
    const uint32_t* input0,const uint32_t* input1,uint32_t* output0,uint32_t* output1,
    uint32_t* weighted0,uint32_t* weighted1,unsigned rank,Mod mod) {
    const unsigned lane=threadIdx.x&31;
    for(unsigned source=lane;source<rank;source+=32) {
        weighted0[source]=hafnian_mul(old_metric[source],input0[source],mod);
        weighted1[source]=hafnian_mul(old_metric[source],input1[source],mod);
    }
    __syncwarp();
    for(unsigned target=lane;target<rank;target+=32) {
        uint32_t sum0=0,sum1=0;
        for(unsigned source=0;source<rank;++source) {
            const uint32_t factor=basis[size_t(source)*rank+target];
            sum0=hafnian_add_mod(sum0,
                hafnian_mul(factor,weighted0[source],mod),mod.p);
            sum1=hafnian_add_mod(sum1,
                hafnian_mul(factor,weighted1[source],mod),mod.p);
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
    __syncwarp();
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

template<unsigned N,unsigned CHAIN,class Mod,bool FULL_RANK>
__global__ __launch_bounds__(THREADS,MIN_BLOCKS_PER_SM) void terms_kernel(
    const uint32_t* __restrict__ edge_matrices,
    const uint32_t* __restrict__ update_vectors,
    const uint32_t* __restrict__ fixed_metric,unsigned rank,
    uint64_t begin,uint64_t end,Mod mod,const uint32_t* __restrict__ inverse_small,
    uint32_t* __restrict__ scratch,uint32_t* __restrict__ chain_sums,
    uint32_t* __restrict__ failures) {
    static_assert(CHAIN>=1&&CHAIN<=8);
    if constexpr(FULL_RANK)rank=N;
    constexpr unsigned HALF=N/2;
    const unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const size_t slot=size_t(blockIdx.x)*WARPS_PER_BLOCK+warp;
    const size_t slots=size_t(gridDim.x)*WARPS_PER_BLOCK;
    extern __shared__ uint32_t shared[];
    uint32_t* vectors=shared+warp*6*N;
    uint32_t* previous=vectors;
    uint32_t* current=previous+N;
    uint32_t* next=current+N;
    uint32_t* applied=next+N;
    uint32_t* weighted_low0=applied+N;
    uint32_t* weighted_low1=weighted_low0+N;
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
                const bool positive=edge==0||
                    (signs0&(UINT64_C(1)<<(edge-1)));
                value=positive?hafnian_add_mod(value,addend,mod.p):
                    hafnian_sub_mod(value,addend,mod.p);
            }
            dense[size_t(cell%rank)*rank+cell/rank]=value;
        }
        __syncwarp();
        bool ok=false;
        for(unsigned attempt=0;attempt<4&&!ok;++attempt)
            ok=generalized_lanczos(dense,nullptr,nullptr,nullptr,nullptr,
                nullptr,nullptr,0,fixed_metric,rank,
                splitmix64(chain_begin)^attempt,mod,basis,
                metrics,inverse_metrics,diagonals,betas,
                previous,current,next,applied);
        if(!ok) {++local_failures;continue;}
        for(unsigned future_step=1;future_step<chain_end-chain_begin;++future_step) {
            uint32_t* first=future+(size_t(future_step-1)*2+0)*N;
            uint32_t* second=future+(size_t(future_step-1)*2+1)*N;
            inverse_basis_apply_pair(basis,fixed_metric,inverse_metrics,
                first,second,temporary0,temporary1,next,applied,rank,mod);
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
                    betas+current_buffer*N,z0,z1,
                    weighted_low0,weighted_low1,delta,
                    old_metric,rank,
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
                    temporary0,temporary1,next,applied,rank,mod);
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
constexpr size_t shared_bytes() {return size_t(WARPS_PER_BLOCK)*6*N*sizeof(uint32_t);}

template<unsigned N,unsigned CHAIN>
constexpr size_t scratch_words(size_t slots) {
    return size_t(2)*slots*N*N+slots*2*(CHAIN-1)*N+slots*8*N;
}

} // namespace hafnian_gray
