#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <omp.h>
#include "hafnian_boundary_plan.hpp"
#include "hafnian_boundary_order.hpp"
#include "hafnian_inverse_chain.hpp"
#ifndef CORE_HOST_EMULATION
#include <cuda_runtime.h>
#define CORE_DEVICE __device__
#define CORE_SYNC() __syncthreads()
#else
#define CORE_DEVICE
#define CORE_SYNC() _Pragma("omp barrier")
#endif

// Experiment 479: accepted research default. Set all three OPT switches to
// zero for the exact control. No switch changes sign domains or query IDs.
#ifndef CORE_OPT_HESS
#define CORE_OPT_HESS 1
#endif
#ifndef CORE_OPT_BOUNDARY
#define CORE_OPT_BOUNDARY 1
#endif
#ifndef CORE_OPT_SCRATCH
#define CORE_OPT_SCRATCH 1
#endif
#ifndef CORE_PROFILE
#define CORE_PROFILE 0
#endif
// Bounded research candidates; retain the accepted kernel as the default.
#ifndef CORE_OPT_WARP_POLY
#define CORE_OPT_WARP_POLY 0
#endif
#ifndef CORE_OPT_SPARSE_MOMENTS
#define CORE_OPT_SPARSE_MOMENTS 0
#endif
#ifndef CORE_BOUNDARY_ORDER
#define CORE_BOUNDARY_ORDER 0
#endif
#ifndef CORE_MAX_POOL
#define CORE_MAX_POOL 11
#endif
#ifndef CORE_OPT_INVERSE_CHAIN
#define CORE_OPT_INVERSE_CHAIN 0
#endif
#ifndef CORE_OPT_LIVE_MOMENTS
#define CORE_OPT_LIVE_MOMENTS 0
#endif
#ifndef CORE_OPT_SYNC_CLEAR
#define CORE_OPT_SYNC_CLEAR 0
#endif
#if (CORE_OPT_INVERSE_CHAIN != 0 && CORE_OPT_INVERSE_CHAIN != 1) || (CORE_OPT_LIVE_MOMENTS != 0 && CORE_OPT_LIVE_MOMENTS != 1) || (CORE_OPT_SYNC_CLEAR != 0 && CORE_OPT_SYNC_CLEAR != 1)
#error "Kernel candidate switches must be 0 or 1"
#endif
#if (CORE_OPT_WARP_POLY != 0 && CORE_OPT_WARP_POLY != 1) || (CORE_OPT_SPARSE_MOMENTS != 0 && CORE_OPT_SPARSE_MOMENTS != 1)
#error "Kernel candidate switches must be 0 or 1"
#endif
#if CORE_BOUNDARY_ORDER < 0 || CORE_BOUNDARY_ORDER > 64
#error "Boundary ordering is a bounded 0..64-trial offline search"
#endif
#if CORE_MAX_POOL != 11 && CORE_MAX_POOL != 13
#error "Only bounded pool 11/13 experiments are supported"
#endif
#if CORE_OPT_WARP_POLY && !defined(CORE_HOST_EMULATION)
#define CORE_POLY_SYNC() __syncwarp()
#else
#define CORE_POLY_SYNC() CORE_SYNC()
#endif
#if CORE_PROFILE && !defined(CORE_HOST_EMULATION)
#define CORE_STAMP(i) do { CORE_SYNC(); if(tid==0)phase[i]=clock64(); } while(0)
#else
#define CORE_STAMP(i) do {} while(0)
#endif

namespace core_gpu {
constexpr unsigned MAX_C=48, MAX_Q=CORE_MAX_POOL, MAX_MASK=1u<<MAX_Q;
struct Input {
    unsigned c,q,n,degree,stride,states,queries,words,matrix_stride,pairs;
    unsigned starts[7]{}; // even boundary subset sizes 0,2,...,10
    unsigned masks[MAX_MASK]{},slot[MAX_MASK]{},answers[256]{};
    unsigned pair_i[MAX_Q*MAX_Q]{},pair_j[MAX_Q*MAX_Q]{};
    unsigned moment_count=0,moment_source[MAX_Q]{},moment_slot[MAX_Q]{};
    unsigned transition_begin[MAX_MASK+1]{};
    uint16_t child_slot[MAX_MASK*MAX_Q]{},edge_slot[MAX_MASK*MAX_Q]{};
    uint32_t inverse[49]{};
    uint8_t adjacency[64*64]{};
    uint8_t core_degree[MAX_C]{},boundary_degree[MAX_Q]{};
    uint8_t core_neighbors[MAX_C*MAX_C]{},boundary_neighbors[MAX_Q*MAX_C]{};
};

template<class Problem> Input prepare(const Problem& original) {
    if(original.core>MAX_C||(original.core&1)||original.q>MAX_Q||original.adjacency.n>64||
       original.adjacency.n!=original.core+original.q||original.masks.empty()||original.masks.size()>256)
        throw std::runtime_error("CUDA gate dimensions exceeded");
    for(unsigned mask:original.masks)
        if(mask>=(1u<<original.q)||(__builtin_popcount(mask)&1)||__builtin_popcount(mask)>10)
            throw std::runtime_error("CUDA gate requires even boundary masks of at most ten vertices");
    auto p=original;
#if CORE_BOUNDARY_ORDER
    auto order=common_boundary::choose(p.q,p.masks,CORE_BOUNDARY_ORDER);
    for(unsigned i=0;i<p.adjacency.n;++i)for(unsigned j=0;j<p.adjacency.n;++j){
        unsigned old_i=i<p.core?i:p.core+order[i-p.core];
        unsigned old_j=j<p.core?j:p.core+order[j-p.core];
        p.adjacency.at(i,j)=original.adjacency.at(old_i,old_j);
    }
    for(auto& mask:p.masks){unsigned relabelled=0;
        for(unsigned j=0;j<p.q;++j)if(mask&(1u<<order[j]))relabelled|=1u<<j;
        mask=relabelled;}
#endif
    Input in{};in.c=p.core;in.q=p.q;in.n=p.adjacency.n;
    in.degree=p.core/2;in.stride=in.degree+1;in.queries=p.masks.size();
    auto plan=hafnian_boundary_plan(p.q,p.masks);
    std::sort(plan.begin(),plan.end(),[](unsigned a,unsigned b){
        return std::make_pair(__builtin_popcount(a),a)<std::make_pair(__builtin_popcount(b),b);});
    in.states=plan.size();
    for(unsigned i=0;i<plan.size();++i){in.masks[i]=plan[i];in.slot[plan[i]]=i;}
    for(unsigned level=0;level<=6;++level) {
        unsigned start=0;while(start<plan.size()&&unsigned(__builtin_popcount(plan[start]))<2*level)++start;
        in.starts[level]=start;
    }
    for(unsigned j=0;j<p.masks.size();++j)in.answers[j]=in.slot[p.masks[j]];
    unsigned pair_slot[MAX_Q*MAX_Q];
    std::fill(std::begin(pair_slot),std::end(pair_slot),UINT32_MAX);
    unsigned transitions=0;
    for(unsigned idx=0;idx<plan.size();++idx){
        in.transition_begin[idx]=transitions;
        unsigned mask=plan[idx];if(!mask)continue;
        unsigned first=unsigned(__builtin_ctz(mask)),rest=mask^(1u<<first);
        for(unsigned j=first+1;j<in.q;++j)if(rest&(1u<<j)){
            unsigned& edge=pair_slot[first*in.q+j];
            if(edge==UINT32_MAX){edge=in.pairs++;in.pair_i[edge]=first;in.pair_j[edge]=j;}
            in.child_slot[transitions]=uint16_t(in.slot[rest^(1u<<j)]);
            in.edge_slot[transitions++]=uint16_t(edge);
        }
    }
    in.transition_begin[plan.size()]=transitions;
    // Only the second endpoint of a stored upper-triangular pair is queried.
    // Source columns evolve independently, so absent columns need no powers.
    for(unsigned j=0;j<in.q;++j){
        bool used=!CORE_OPT_BOUNDARY;
        for(unsigned ij=0;ij<in.pairs;++ij)used|=in.pair_j[ij]==j;
        if(used || !CORE_OPT_LIVE_MOMENTS){
            in.moment_slot[j]=in.moment_count;
            in.moment_source[in.moment_count++]=j;
        }
    }
    for(unsigned i=0;i<in.n;++i)for(unsigned j=0;j<in.n;++j)
        in.adjacency[i*in.n+j]=p.adjacency.at(i,j);
    for(unsigned i=0;i<in.c;++i)for(unsigned j=0;j<in.c;++j)
        if(in.adjacency[(i^1)*in.n+j])in.core_neighbors[i*MAX_C+in.core_degree[i]++]=j;
    for(unsigned i=0;i<in.q;++i)for(unsigned j=0;j<in.c;++j)
        if(in.adjacency[(in.c+i)*in.n+j])in.boundary_neighbors[i*MAX_C+in.boundary_degree[i]++]=j;
    in.matrix_stride=in.c+(CORE_OPT_HESS?1:0);
    unsigned core_words=in.c*in.matrix_stride+(in.c+1)*in.stride+
        (CORE_OPT_WARP_POLY?std::max(in.c,32u):in.c)+2;
    unsigned k_words=(CORE_OPT_BOUNDARY?in.pairs:in.q*in.q)*in.stride;
    in.words=CORE_OPT_SCRATCH ? in.stride+k_words+
        std::max({core_words,2*in.c*in.moment_count,in.states*in.stride}) :
        core_words+in.stride+k_words+2*in.c*in.moment_count+in.states*in.stride;
    return in;
}

inline uint32_t host_power(uint32_t x,uint32_t exponent,uint32_t prime) {
    uint32_t value=1;
    for(;exponent;exponent>>=1,x=uint64_t(x)*x%prime)
        if(exponent&1)value=uint64_t(value)*x%prime;
    return value;
}
inline void set_field(Input& in,uint32_t prime) {
    // A queue alternates all four production fields between groups. Retain
    // their tables instead of recomputing 48 inverses on every field switch.
    // Bounded round-robin replacement also supports research prime sets.
    struct Table { uint32_t prime=0; std::array<uint32_t,49> inverse{}; };
    thread_local std::array<Table,4> tables{};
    thread_local unsigned next=0;
    Table* found=nullptr;
    for(auto& table:tables)if(table.prime==prime){found=&table;break;}
    if(!found){
        found=&tables[next];next=(next+1)%tables.size();
        for(unsigned j=1;j<found->inverse.size();++j)
            found->inverse[j]=host_power(j,prime-2,prime);
        found->prime=prime;
    }
    std::copy(found->inverse.begin(),found->inverse.end(),in.inverse);
}

template<class Problem> Input pack(const Problem& p,uint32_t prime) {
    auto in=prepare(p);set_field(in,prime);return in;
}

template<uint32_t P> struct Field {
    CORE_DEVICE static uint32_t add(uint32_t a,uint32_t b){uint32_t s=a+b;return s>=P?s-P:s;}
    CORE_DEVICE static uint32_t neg(uint32_t a){return a?P-a:0;}
    CORE_DEVICE static uint32_t sub(uint32_t a,uint32_t b){return a>=b?a-b:P-(b-a);}
    CORE_DEVICE static uint32_t reduce(uint64_t t){
        constexpr uint64_t mask=UINT64_C(2147483647),delta=UINT64_C(2147483648)-P;
        // Also valid for a sum of four products: the first fold is below
        // 2^31+69*(2^33-1), and the second remains below 2p.
        t=(t&mask)+(t>>31)*delta;t=(t&mask)+(t>>31)*delta;
        return uint32_t(t>=P?t-P:t);
    }
    CORE_DEVICE static uint32_t mul(uint32_t a,uint32_t b){return reduce(uint64_t(a)*b);}
    CORE_DEVICE static uint32_t inverse(uint32_t a){
#if CORE_OPT_INVERSE_CHAIN
        return hafnian_inverse_chain<P>(a,[](uint32_t x,uint32_t y){return mul(x,y);});
#else
        uint32_t out=1;for(uint32_t e=P-2;e;e>>=1){if(e&1)out=mul(out,a);a=mul(a,a);}return out;
#endif
    }
};

template<uint32_t P>
CORE_DEVICE void term(const Input& in,uint64_t signs,uint32_t* scratch,
                      uint32_t* output,unsigned tid,unsigned nt,uint64_t* phase=nullptr) {
    using F=Field<P>;
    CORE_STAMP(0);
    unsigned c=in.c,q=in.q,m=in.degree,s=in.stride,n=in.n,hs=in.matrix_stride;
    unsigned nk=CORE_OPT_BOUNDARY?in.pairs:q*q;
    unsigned nq=CORE_OPT_LIVE_MOMENTS?in.moment_count:q;
#if CORE_OPT_SCRATCH
    uint32_t* f=scratch;
    uint32_t* k=f+s;
    uint32_t* h=k+nk*s;
    uint32_t* poly=h+c*hs;
    uint32_t* factors=poly+(c+1)*s;
    uint32_t* pivot=factors+(CORE_OPT_WARP_POLY&&c<32?32:c);
    uint32_t* power=h;
    uint32_t* next=power+c*nq;
    uint32_t* memo=h;
#else
    uint32_t* h=scratch;
    uint32_t* poly=h+c*hs;
    uint32_t* f=poly+(c+1)*s;
    uint32_t* factors=f+s;
    uint32_t* pivot=factors+(CORE_OPT_WARP_POLY&&c<32?32:c);
    uint32_t* k=pivot+2;
    uint32_t* power=k+nk*s;
    uint32_t* next=power+c*nq;
    uint32_t* memo=next+c*nq;
#endif
    for(unsigned i=tid;i<in.words;i+=nt)scratch[i]=0;
    CORE_SYNC();
    for(unsigned ij=tid;ij<c*c;ij+=nt){unsigned i=ij/c,j=ij%c;
        bool positive=i/2==0||(signs&(UINT64_C(1)<<(i/2-1)));
        h[i*hs+j]=in.adjacency[(i^1)*n+j]?(positive?1:P-1):0;
    }
    CORE_SYNC();
    for(unsigned col=0;col+2<c;++col) {
        if(tid==0){unsigned r=col+1;while(r<c&&!h[r*hs+col])++r;*pivot=r;}
        CORE_SYNC();
        unsigned r=*pivot;
        // All threads must copy the pivot before a zero-column skip lets
        // thread zero publish the next column's pivot.
        CORE_SYNC();
        if(r==c)continue; // identical branch across CTA
#if CORE_OPT_HESS
        if(r!=col+1){
#endif
        for(unsigned j=tid;j<c;j+=nt){uint32_t t=h[r*hs+j];h[r*hs+j]=h[(col+1)*hs+j];h[(col+1)*hs+j]=t;}
        CORE_SYNC();
        for(unsigned i=tid;i<c;i+=nt){uint32_t t=h[i*hs+r];h[i*hs+r]=h[i*hs+col+1];h[i*hs+col+1]=t;}
        CORE_SYNC();
#if CORE_OPT_HESS
        }
#endif
        if(tid==0)pivot[1]=F::inverse(h[(col+1)*hs+col]);
        CORE_SYNC();
        for(unsigned i=col+2+tid;i<c;i+=nt)factors[i]=F::mul(h[i*hs+col],pivot[1]);
        CORE_SYNC();
        // Commuting row eliminations followed by their joint inverse-column
        // update. The pivot row and every factor are stable during each pass.
#if CORE_OPT_HESS
        unsigned width=c-col,cells=(c-col-2)*width;
        for(unsigned ij=tid;ij<cells;ij+=nt){unsigned i=col+2+ij/width,j=col+ij%width;
            h[i*hs+j]=F::sub(h[i*hs+j],F::mul(factors[i],h[(col+1)*hs+j]));}
#else
        for(unsigned ij=tid;ij<c*c;ij+=nt){unsigned i=ij/c,j=ij%c;
            if(i>col+1&&j>=col)h[i*hs+j]=F::sub(h[i*hs+j],F::mul(factors[i],h[(col+1)*hs+j]));}
#endif
        CORE_SYNC();
        for(unsigned i=tid;i<c;i+=nt){uint32_t value=h[i*hs+col+1];
            for(unsigned j=col+2;j<c;++j)value=F::add(value,F::mul(factors[j],h[i*hs+j]));
            h[i*hs+col+1]=value;}
        CORE_SYNC();
    }
    CORE_STAMP(1);
#if CORE_OPT_WARP_POLY
    if(tid<32){
    const unsigned poly_nt=nt<32?nt:32;
#else
    const unsigned poly_nt=nt;
#endif
    if(tid==0)poly[0]=1;
    CORE_POLY_SYNC();
    for(unsigned size=1;size<=c;++size) {
#if CORE_OPT_HESS
        if(tid==0){uint32_t product=1;
            for(unsigned dist=1;dist<size&&dist<m;++dist){
                product=F::mul(product,h[(size-dist)*hs+size-dist-1]);
                factors[dist]=F::mul(product,h[(size-dist-1)*hs+size-1]);}}
        CORE_POLY_SYNC();
#endif
        for(unsigned d=tid;d<=m&&d<=size;d+=poly_nt) {
            uint32_t value=d<size?poly[(size-1)*s+d]:0;
            if(d)value=F::sub(value,F::mul(h[(size-1)*hs+size-1],poly[(size-1)*s+d-1]));
#if !CORE_OPT_HESS
            uint32_t product=1;
#endif
            for(unsigned dist=1;dist<size&&dist+1<=d;++dist){
#if CORE_OPT_HESS
                value=F::sub(value,F::mul(factors[dist],poly[(size-dist-1)*s+d-dist-1]));
#else
                product=F::mul(product,h[(size-dist)*hs+size-dist-1]);
                value=F::sub(value,F::mul(F::mul(product,h[(size-dist-1)*hs+size-1]),poly[(size-dist-1)*s+d-dist-1]));
#endif
            }
            poly[size*s+d]=value;
        }
        CORE_POLY_SYNC();
    }
#if CORE_OPT_WARP_POLY
    if(tid==0)f[0]=1;
    CORE_POLY_SYNC();
    for(unsigned d=1;d<=m;++d){uint32_t value=0;
        for(unsigned j=tid+1;j<=d;j+=poly_nt)
            value=F::add(value,F::mul(2*d-j,F::mul(poly[c*s+j],f[d-j])));
#ifndef CORE_HOST_EMULATION
        for(unsigned delta=16;delta;delta>>=1)
            value=F::add(value,__shfl_down_sync(0xffffffffu,value,delta));
#else
        factors[tid]=value;CORE_POLY_SYNC();
        if(tid==0)for(unsigned j=1;j<poly_nt;++j)value=F::add(value,factors[j]);
#endif
        if(tid==0)f[d]=F::neg(F::mul(value,in.inverse[2*d]));
        CORE_POLY_SYNC();
    }
    } // Other warps wait at the phase boundary, not at every coefficient row.
    CORE_SYNC();
#else
    if(tid==0){f[0]=1;for(unsigned d=1;d<=m;++d){uint32_t value=0;
        for(unsigned j=1;j<=d;++j)value=F::add(value,F::mul(2*d-j,F::mul(poly[c*s+j],f[d-j])));
        f[d]=F::neg(F::mul(value,in.inverse[2*d]));}}
#endif
#if CORE_OPT_SCRATCH && !(CORE_OPT_SYNC_CLEAR && CORE_OPT_WARP_POLY)
    // f must finish reading poly before moment buffers overwrite its arena.
    CORE_SYNC();
#endif
    CORE_STAMP(2);
    for(unsigned ij=tid;ij<nk;ij+=nt){
        unsigned i=CORE_OPT_BOUNDARY?in.pair_i[ij]:ij/q,j=CORE_OPT_BOUNDARY?in.pair_j[ij]:ij%q;
        k[ij*s]=in.adjacency[(c+i)*n+c+j];}
    for(unsigned vj=tid;vj<c*nq;vj+=nt){unsigned v=vj/nq,j=CORE_OPT_LIVE_MOMENTS?in.moment_source[vj%nq]:vj%nq;
        bool positive=v/2==0||(signs&(UINT64_C(1)<<(v/2-1)));
        power[vj]=in.adjacency[(v^1)*n+c+j]?(positive?1:P-1):0;}
    CORE_SYNC();
    for(unsigned d=1;d<=m;++d) {
        for(unsigned ij=tid;ij<nk;ij+=nt){
            unsigned i=CORE_OPT_BOUNDARY?in.pair_i[ij]:ij/q,j=CORE_OPT_BOUNDARY?in.pair_j[ij]:ij%q;uint32_t value=0;
            unsigned source=CORE_OPT_LIVE_MOMENTS?in.moment_slot[j]:j;
#if CORE_OPT_SPARSE_MOMENTS
            uint64_t pending=0;
            for(unsigned a=0;a<in.boundary_degree[i];++a)
                pending+=power[unsigned(in.boundary_neighbors[i*MAX_C+a])*nq+source];
            value=F::reduce(pending); // <=48*(P-1), comfortably within uint64_t.
#else
            for(unsigned v=0;v<c;++v)if(in.adjacency[(c+i)*n+v])value=F::add(value,power[v*nq+source]);
#endif
            k[ij*s+d]=value;}
        if(d<m){
            for(unsigned vj=tid;vj<c*nq;vj+=nt){unsigned v=vj/nq,j=vj%nq;uint32_t value=0;
#if CORE_OPT_SPARSE_MOMENTS
                uint64_t pending=0;
                for(unsigned a=0;a<in.core_degree[v];++a)
                    pending+=power[unsigned(in.core_neighbors[v*MAX_C+a])*nq+j];
                value=F::reduce(pending);
#else
                for(unsigned u=0;u<c;++u)if(in.adjacency[(v^1)*n+u])value=F::add(value,power[u*nq+j]);
#endif
                bool positive=v/2==0||(signs&(UINT64_C(1)<<(v/2-1)));
                next[vj]=positive?value:F::neg(value);}
#if !CORE_OPT_SYNC_CLEAR
            CORE_SYNC();
#endif
            uint32_t* temporary=power;power=next;next=temporary;
        }
        CORE_SYNC();
    }
    CORE_STAMP(3);
#if CORE_OPT_SCRATCH
    // moment powers are dead; reset their arena before using it as memo.
    // All nonempty states are overwritten coefficient-by-coefficient before
    // any later level reads them. Only the empty polynomial needs clearing.
    for(unsigned i=tid;i<(CORE_OPT_SYNC_CLEAR?s:in.states*s);i+=nt)memo[i]=0;
    CORE_SYNC();
#endif
    if(tid==0)memo[in.slot[0]*s]=1;
    CORE_SYNC();
    for(unsigned level=1;level<=5;++level) {
        for(unsigned idx=in.starts[level]*s+tid;idx<in.starts[level+1]*s;idx+=nt){
            unsigned slot=idx/s,d=idx%s;
#if CORE_OPT_BOUNDARY
            unsigned begin=in.transition_begin[slot],end=in.transition_begin[slot+1];
            // hafnian of a two-vertex matrix is its sole off-diagonal entry.
            if(level==1){memo[idx]=k[in.edge_slot[begin]*s+d];continue;}
            uint32_t value=0;
            for(unsigned t=begin;t<end;++t){
                const uint32_t* child=memo+in.child_slot[t]*s;
                const uint32_t* edge=k+in.edge_slot[t]*s;
                unsigned a=0;
                for(;a+3<=d;a+=4){uint64_t pending=uint64_t(edge[a])*child[d-a];
                    pending+=uint64_t(edge[a+1])*child[d-a-1];
                    pending+=uint64_t(edge[a+2])*child[d-a-2];
                    pending+=uint64_t(edge[a+3])*child[d-a-3];
                    value=F::add(value,F::reduce(pending));}
                uint64_t pending=0;for(;a<=d;++a)pending+=uint64_t(edge[a])*child[d-a];
                value=F::add(value,F::reduce(pending));
            }
#else
            unsigned mask=in.masks[slot];
            unsigned first=0;while(!(mask&(1u<<first)))++first;
            unsigned rest=mask^(1u<<first);uint32_t value=0;
            for(unsigned j=first+1;j<q;++j)if(rest&(1u<<j)) {
                const uint32_t* child=memo+in.slot[rest^(1u<<j)]*s;
                const uint32_t* edge=k+(first*q+j)*s;
                for(unsigned a=0;a<=d;++a)value=F::add(value,F::mul(edge[a],child[d-a]));
            }
#endif
            memo[idx]=value;
        }
        CORE_SYNC();
    }
    CORE_STAMP(4);
    unsigned ones=0;for(uint64_t b=signs;b;b&=b-1)++ones;
    bool negative=m&&((m-1-ones)&1);
    for(unsigned j=tid;j<in.queries;j+=nt){uint32_t value=0;
        for(unsigned d=0;d<=m;++d)value=F::add(value,F::mul(f[d],memo[in.answers[j]*s+m-d]));
        output[j]=negative?F::neg(value):value;}
    CORE_SYNC();
    CORE_STAMP(5);
}
} // namespace core_gpu

namespace core_gpu {
#ifndef CORE_HOST_EMULATION
void checked(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<uint32_t P> __global__ void kernel(const Input* in,uint64_t begin,uint32_t* output,uint64_t* phases) {
    extern __shared__ uint32_t scratch[];
    uint64_t index=begin+blockIdx.x;
    term<P>(*in,index^(index>>1),scratch,output+size_t(blockIdx.x)*in->queries,threadIdx.x,blockDim.x,
        CORE_PROFILE?phases+size_t(blockIdx.x)*6:nullptr);
}
#endif


} // namespace core_gpu
