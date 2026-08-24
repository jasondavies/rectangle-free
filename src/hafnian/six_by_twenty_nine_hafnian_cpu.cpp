// Independent CPU range evaluator for the exact T_4(6,29) residual hafnians.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SIX_BY_TWENTY_EIGHT
#include "six_by_twenty_eight_catalog.hpp"
#else
#include "six_by_twenty_nine_catalog.hpp"
#endif

namespace {

using Clock=std::chrono::steady_clock;
constexpr unsigned MAX_N=64,MAX_HALF=32;
#ifdef SIX_BY_TWENTY_EIGHT
using DefectQuery=six_by_twenty_eight::Query;
#else
using DefectQuery=six_by_twenty_nine::Query;
#endif

struct Mod {
    uint32_t p;
    uint64_t reciprocal;
    explicit Mod(uint32_t prime):p(prime),
        reciprocal(uint64_t((static_cast<unsigned __int128>(1)<<64)/prime)){}
    uint32_t add(uint32_t a,uint32_t b)const {
        uint32_t value=a+b;
        if(value>=p||value<a)value-=p;
        return value;
    }
    uint32_t sub(uint32_t a,uint32_t b)const{return a>=b?a-b:uint32_t(uint64_t(a)+p-b);}
    uint32_t neg(uint32_t a)const{return a?p-a:0;}
    uint32_t mul(uint32_t a,uint32_t b)const {
        uint64_t product=uint64_t(a)*b;
        uint64_t quotient=uint64_t(static_cast<unsigned __int128>(product)*reciprocal>>64);
        uint64_t remainder=product-quotient*p;
        if(remainder>=p)remainder-=p;
        if(remainder>=p)remainder-=p;
        return uint32_t(remainder);
    }
    uint32_t power(uint32_t a,uint64_t exponent)const {
        uint32_t result=1;
        while(exponent){if(exponent&1)result=mul(result,a);a=mul(a,a);exponent>>=1;}
        return result;
    }
    uint32_t inverse(uint32_t a)const {
        if(!a)throw std::runtime_error("zero inverse");
        return power(a,p-2);
    }
};

struct Matrix {
    unsigned n=0;
    std::array<uint32_t,MAX_N*MAX_N> values{};
    uint32_t& at(unsigned row,unsigned column){return values[row*MAX_N+column];}
    uint32_t at(unsigned row,unsigned column)const{return values[row*MAX_N+column];}
};

void upper_hessenberg(Matrix& matrix,const Mod& mod) {
    unsigned n=matrix.n;
    for(unsigned column=0;column+2<n;++column) {
        unsigned pivot=column+1;
        while(pivot<n&&!matrix.at(pivot,column))++pivot;
        if(pivot==n)continue;
        if(pivot!=column+1) {
            for(unsigned j=0;j<n;++j)std::swap(matrix.at(pivot,j),matrix.at(column+1,j));
            for(unsigned i=0;i<n;++i)std::swap(matrix.at(i,pivot),matrix.at(i,column+1));
        }
        uint32_t inverse=mod.inverse(matrix.at(column+1,column));
        for(unsigned row=column+2;row<n;++row) {
            uint32_t below=matrix.at(row,column);
            if(!below)continue;
            uint32_t factor=mod.mul(below,inverse);
            for(unsigned j=column;j<n;++j)
                matrix.at(row,j)=mod.sub(matrix.at(row,j),mod.mul(factor,matrix.at(column+1,j)));
            for(unsigned i=0;i<n;++i)
                matrix.at(i,column+1)=mod.add(matrix.at(i,column+1),mod.mul(factor,matrix.at(i,row)));
        }
    }
}

std::array<uint32_t,MAX_HALF+1> power_traces(Matrix matrix,unsigned degree,const Mod& mod) {
    upper_hessenberg(matrix,mod);
    unsigned n=matrix.n;
    std::array<std::array<uint32_t,MAX_N+1>,MAX_N+1> poly{};
    poly[0][0]=1;
    for(unsigned size=1;size<=n;++size) {
        unsigned diagonal=size-1;
        for(unsigned d=0;d<size;++d) {
            poly[size][d+1]=mod.add(poly[size][d+1],poly[size-1][d]);
            poly[size][d]=mod.sub(poly[size][d],mod.mul(matrix.at(diagonal,diagonal),poly[size-1][d]));
        }
        uint32_t product=1;
        for(unsigned distance=1;distance<size;++distance) {
            unsigned subrow=size-distance;
            product=mod.mul(product,matrix.at(subrow,subrow-1));
            uint32_t factor=mod.mul(product,matrix.at(size-distance-1,size-1));
            for(unsigned d=0;d<=size-distance-1;++d)
                poly[size][d]=mod.sub(poly[size][d],mod.mul(factor,poly[size-distance-1][d]));
        }
    }
    std::array<uint32_t,MAX_HALF+1> traces{};
    for(unsigned k=1;k<=degree;++k) {
        uint32_t value=0;
        for(unsigned j=1;j<k;++j)
            value=mod.add(value,mod.mul(poly[n][n-j],traces[k-j]));
        value=mod.add(value,mod.mul(k,poly[n][n-k]));
        traces[k]=mod.neg(value);
    }
    return traces;
}

uint32_t trace_term(
    const DefectQuery& query,uint64_t signs,const Mod& mod) {
    unsigned n=query.vertices,half=n/2;
    Matrix matrix;
    matrix.n=n;
    auto sign=[&](unsigned edge){
        if(!edge)return 1U;
        return signs&(UINT64_C(1)<<(edge-1))?1U:mod.p-1;
    };
    unsigned negatives=0;
    for(unsigned edge=1;edge<half;++edge)
        negatives+=!(signs&(UINT64_C(1)<<(edge-1)));
    for(unsigned row=0;row<n;++row)for(unsigned column=0;column<n;++column) {
        unsigned paired=column<half?column+half:column-half;
        if(query.adjacency[row*n+paired])matrix.at(row,column)=sign(column%half);
    }
    auto traces=power_traces(matrix,half,mod);
    std::array<uint32_t,MAX_HALF+1> coefficients{};
    coefficients[0]=1;
    uint32_t inverse_two=mod.inverse(2);
    for(unsigned degree=1;degree<=half;++degree) {
        uint32_t sum=0;
        for(unsigned k=1;k<=degree;++k)
            sum=mod.add(sum,mod.mul(mod.mul(traces[k],inverse_two),coefficients[degree-k]));
        coefficients[degree]=mod.mul(sum,mod.inverse(degree));
    }
    return negatives&1?mod.neg(coefficients[half]):coefficients[half];
}

uint64_t number(const char* text) {
    char* end=nullptr;
    uint64_t value=std::strtoull(text,&end,10);
    if(!end||*end)throw std::runtime_error("invalid integer");
    return value;
}

} // namespace

int main(int argc,char** argv) {
    try {
        unsigned query_id=UINT32_MAX,threads=1;
        uint32_t prime=2147483647U;
        uint64_t begin=0,end=0;
        for(int i=1;i<argc;++i) {
            std::string argument=argv[i];
            if(argument=="--query"&&i+1<argc)query_id=unsigned(number(argv[++i]));
            else if(argument=="--prime"&&i+1<argc)prime=uint32_t(number(argv[++i]));
            else if(argument=="--begin"&&i+1<argc)begin=number(argv[++i]);
            else if(argument=="--end"&&i+1<argc)end=number(argv[++i]);
            else if(argument=="--threads"&&i+1<argc)threads=unsigned(number(argv[++i]));
            else throw std::runtime_error(
                "usage: six_by_twenty_nine_hafnian_cpu --query Q --prime P --begin B --end E --threads N");
        }
#ifdef SIX_BY_TWENTY_EIGHT
        auto catalog=six_by_twenty_eight::build_catalog();
#else
        auto catalog=six_by_twenty_nine::build_catalog();
#endif
        if(query_id>=catalog.queries.size()||!threads)throw std::runtime_error("invalid query/configuration");
        const auto& query=catalog.queries[query_id];
        uint64_t total=UINT64_C(1)<<(query.vertices/2-1);
        if(!end)end=total;
        if(begin>=end||end>total)throw std::runtime_error("invalid term range");
#ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(int(threads));
#else
        if(threads!=1)throw std::runtime_error("binary lacks OpenMP");
#endif
        Mod mod(prime);
        std::vector<uint64_t> sums(threads);
        auto started=Clock::now();
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            unsigned thread=0;
#ifdef _OPENMP
            thread=unsigned(omp_get_thread_num());
#pragma omp for schedule(dynamic,1)
#endif
            for(uint64_t term=begin;term<end;++term) {
                sums[thread]+=trace_term(query,term,mod);
                if(sums[thread]>=(UINT64_C(1)<<62))sums[thread]%=prime;
            }
        }
        uint32_t result=0;
        for(uint64_t sum:sums)result=mod.add(result,uint32_t(sum%prime));
        double seconds=std::chrono::duration<double>(Clock::now()-started).count();
        std::printf(
#ifdef SIX_BY_TWENTY_EIGHT
            "HAFNIAN_6X28_CPU catalog=%s query=%u query_sha=%s vertices=%u prime=%u "
#else
            "HAFNIAN_6X29_CPU catalog=%s query=%u query_sha=%s vertices=%u prime=%u "
#endif
            "begin=%" PRIu64 " end=%" PRIu64 " residue=%u elapsed=%.9f exact=OK\n",
            catalog.digest.c_str(),query.id,query.digest.c_str(),query.vertices,prime,
            begin,end,result,seconds);
        return 0;
    } catch(const std::exception& error) {
        std::fprintf(stderr,"error: %s\n",error.what());
        return 2;
    }
}
