#pragma once
// Independent CPU sign-term check: Hessenberg -> full characteristic
// polynomial -> Newton traces -> exponential series. No CUDA arithmetic,
// Gray updates, or common-core boundary recurrence is used here.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace hafnian_reference {
struct Mod {
    uint32_t p;
    uint32_t add(uint32_t a,uint32_t b)const {return uint64_t(a)+b>=p?a-(p-b):a+b;}
    uint32_t sub(uint32_t a,uint32_t b)const {return a>=b?a-b:p-(b-a);}
    uint32_t neg(uint32_t a)const {return a?p-a:0;}
    uint32_t mul(uint32_t a,uint32_t b)const {return uint64_t(a)*b%p;}
    uint32_t power(uint32_t a,uint32_t e)const {uint32_t x=1;
        for(;e;e>>=1,a=mul(a,a))if(e&1)x=mul(x,a);return x;}
    uint32_t inverse(uint32_t a)const {if(!a)throw std::runtime_error("zero inverse");return power(a,p-2);}
};
template<class Query> uint32_t term(const Query& query,uint64_t index,uint32_t prime) {
    unsigned n=query.vertices,h=n/2;Mod mod{prime};
    if(n<2||n>66||(n&1)||prime<=h||query.adjacency.size()!=size_t(n)*n)
        throw std::runtime_error("invalid reference dimensions");
    uint64_t signs=index^(index>>1);
    std::vector<uint32_t> matrix(n*n);
    auto at=[&](unsigned i,unsigned j)->uint32_t& {return matrix[i*n+j];};
    for(unsigned i=0;i<n;++i)for(unsigned j=0;j<n;++j){
        unsigned pair=j%h,opposite=j<h?j+h:j-h;
        if(query.adjacency[i*n+opposite])at(i,j)=!pair||(signs>>(pair-1)&1)?1:prime-1;
    }
    for(unsigned col=0;col+2<n;++col){
        unsigned pivot=col+1;while(pivot<n&&!at(pivot,col))++pivot;
        if(pivot==n)continue;
        if(pivot!=col+1){
            for(unsigned j=0;j<n;++j)std::swap(at(pivot,j),at(col+1,j));
            for(unsigned i=0;i<n;++i)std::swap(at(i,pivot),at(i,col+1));
        }
        uint32_t inverse=mod.inverse(at(col+1,col));
        for(unsigned i=col+2;i<n;++i){
            uint32_t factor=mod.mul(at(i,col),inverse);if(!factor)continue;
            for(unsigned j=col;j<n;++j)at(i,j)=mod.sub(at(i,j),mod.mul(factor,at(col+1,j)));
            for(unsigned j=0;j<n;++j)at(j,col+1)=mod.add(at(j,col+1),mod.mul(factor,at(j,i)));
        }
    }
    std::vector<std::vector<uint32_t>> poly(n+1,std::vector<uint32_t>(n+1));poly[0][0]=1;
    for(unsigned size=1;size<=n;++size){
        for(unsigned d=0;d<size;++d){
            poly[size][d+1]=mod.add(poly[size][d+1],poly[size-1][d]);
            poly[size][d]=mod.sub(poly[size][d],mod.mul(at(size-1,size-1),poly[size-1][d]));
        }
        uint32_t product=1;
        for(unsigned distance=1;distance<size;++distance){
            unsigned row=size-distance;product=mod.mul(product,at(row,row-1));
            uint32_t factor=mod.mul(product,at(row-1,size-1));
            for(unsigned d=0;d<row;++d)poly[size][d]=mod.sub(poly[size][d],mod.mul(factor,poly[row-1][d]));
        }
    }
    std::vector<uint32_t> traces(h+1),coefficients(h+1);coefficients[0]=1;
    for(unsigned k=1;k<=h;++k){uint32_t sum=mod.mul(k,poly[n][n-k]);
        for(unsigned j=1;j<k;++j)sum=mod.add(sum,mod.mul(poly[n][n-j],traces[k-j]));
        traces[k]=mod.neg(sum);}
    uint32_t half=mod.inverse(2);
    for(unsigned d=1;d<=h;++d){uint32_t sum=0;
        for(unsigned k=1;k<=d;++k)sum=mod.add(sum,mod.mul(mod.mul(traces[k],half),coefficients[d-k]));
        coefficients[d]=mod.mul(sum,mod.inverse(d));}
    return ((h-1-__builtin_popcountll(signs))&1)?mod.neg(coefficients[h]):coefficients[h];
}
}
