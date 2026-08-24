// Exact CPU gate for Gray-ordered low-rank updates in the hafnian sign sum.
//
// For a symmetric adjacency matrix A, let R_s exchange the two vertices in
// each hafnian pair and attach the repeated sign of that pair.  The production
// matrix is M_s=A*R_s.  Compute once an exact symmetric rank factorization
//
//     A = X*D*X^T.
//
// Sylvester's determinant identity says that the nonzero characteristic factor
// of M_s is the characteristic polynomial of K_s=D*X^T*R_s*X.  Omitted roots
// are zero and do not change the leading N/2 coefficients used by the hafnian
// formula.  K_s is self-adjoint under the fixed diagonal metric D^-1, and one
// Gray flip changes K_s by rank two.  This probe tests an exact generalized-
// Lanczos update between periodic full rebuilds.  The dense basis is kept as a
// short lazy product; block length L costs O(N^3/L + L*N^2) per sign term.

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/hafnian/six_by_twenty_eight_catalog.hpp"

namespace {

using Clock=std::chrono::steady_clock;

struct Mod {
    uint32_t p;
    uint32_t add(uint32_t a,uint32_t b)const {
        const uint32_t value=a+b;
        return value>=p?value-p:value;
    }
    uint32_t sub(uint32_t a,uint32_t b)const {
        return a>=b?a-b:uint32_t(uint64_t(a)+p-b);
    }
    uint32_t neg(uint32_t a)const{return a?p-a:0;}
    uint32_t mul(uint32_t a,uint32_t b)const {
        return uint32_t(uint64_t(a)*b%p);
    }
    uint32_t power(uint32_t a,uint64_t exponent)const {
        uint32_t result=1;
        while(exponent) {
            if(exponent&1)result=mul(result,a);
            a=mul(a,a);
            exponent>>=1;
        }
        return result;
    }
    uint32_t inverse(uint32_t a)const {
        if(!a)throw std::runtime_error("zero inverse");
        return power(a,p-2);
    }
};

struct Matrix {
    unsigned n=0;
    std::vector<uint32_t> values;
    Matrix()=default;
    explicit Matrix(unsigned order):n(order),values(size_t(order)*order){}
    uint32_t& at(unsigned row,unsigned column){return values[size_t(row)*n+column];}
    uint32_t at(unsigned row,unsigned column)const{return values[size_t(row)*n+column];}
};

struct RankMatrix {
    unsigned rows=0,columns=0;
    std::vector<uint32_t> values;
    RankMatrix()=default;
    RankMatrix(unsigned row_count,unsigned column_count):
        rows(row_count),columns(column_count),values(size_t(row_count)*column_count){}
    uint32_t& at(unsigned row,unsigned column){return values[size_t(row)*columns+column];}
    uint32_t at(unsigned row,unsigned column)const{return values[size_t(row)*columns+column];}
};

uint32_t dot_metric(
    const std::vector<uint32_t>& left,const std::vector<uint32_t>& right,
    const std::vector<uint32_t>& metric,const Mod& mod) {
    uint32_t result=0;
    for(size_t i=0;i<left.size();++i)
        result=mod.add(result,mod.mul(metric[i],mod.mul(left[i],right[i])));
    return result;
}

struct Tridiagonal {
    std::vector<uint32_t> diagonal;
    // beta[j] is entry (j-1,j); entry (j,j-1) is one.
    std::vector<uint32_t> beta;
};

struct LanczosResult {
    Tridiagonal tridiagonal;
    Matrix basis;
    std::vector<uint32_t> metric;
    std::vector<uint32_t> inverse_metric;
    unsigned attempts=0;
};

LanczosResult generalized_lanczos(
    unsigned n,const std::vector<uint32_t>& old_metric,
    const std::function<void(const std::vector<uint32_t>&,std::vector<uint32_t>&)>& apply,
    const Mod& mod,std::mt19937_64& random,unsigned maximum_attempts=32) {
    unsigned best_dimension=0,best_zero_termination=0;
    for(unsigned attempt=1;attempt<=maximum_attempts;++attempt) {
        LanczosResult result;
        result.tridiagonal.diagonal.resize(n);
        result.tridiagonal.beta.assign(n,0);
        result.basis=Matrix(n);
        result.metric.resize(n);
        result.inverse_metric.resize(n);
        result.attempts=attempt;
        std::vector<uint32_t> previous(n),current(n),next(n);
        for(uint32_t& value:current)value=uint32_t(random()%mod.p);
        uint32_t current_norm=dot_metric(current,current,old_metric,mod);
        if(!current_norm)continue;
        bool failed=false;
        for(unsigned column=0;column<n;++column) {
            for(unsigned row=0;row<n;++row)result.basis.at(row,column)=current[row];
            result.metric[column]=current_norm;
            const uint32_t inverse_norm=mod.inverse(current_norm);
            result.inverse_metric[column]=inverse_norm;
            apply(current,next);
            const uint32_t alpha=mod.mul(
                dot_metric(current,next,old_metric,mod),inverse_norm);
            result.tridiagonal.diagonal[column]=alpha;
            for(unsigned row=0;row<n;++row) {
                next[row]=mod.sub(next[row],mod.mul(alpha,current[row]));
                if(column)next[row]=mod.sub(
                    next[row],mod.mul(result.tridiagonal.beta[column],previous[row]));
            }
            if(column+1==n)break;
            const uint32_t next_norm=dot_metric(next,next,old_metric,mod);
            if(!next_norm) {
                best_dimension=std::max(best_dimension,column+1);
                if(std::all_of(next.begin(),next.end(),[](uint32_t value){return !value;}))
                    best_zero_termination=std::max(best_zero_termination,column+1);
                failed=true;
                break;
            }
            result.tridiagonal.beta[column+1]=mod.mul(
                next_norm,inverse_norm);
            previous.swap(current);
            current.swap(next);
            std::fill(next.begin(),next.end(),0);
            current_norm=next_norm;
        }
        if(!failed)return result;
    }
    throw std::runtime_error(
        "generalized Lanczos breakdown after all retries; best dimension="+
        std::to_string(best_dimension)+" zero termination="+
        std::to_string(best_zero_termination));
}

void verify_lanczos_relation(
    const LanczosResult& result,
    const std::function<void(const std::vector<uint32_t>&,std::vector<uint32_t>&)>& apply,
    const Mod& mod) {
    const unsigned n=result.basis.n;
    std::vector<uint32_t> input(n),actual(n),expected(n);
    for(unsigned column=0;column<n;++column) {
        for(unsigned row=0;row<n;++row)input[row]=result.basis.at(row,column);
        apply(input,actual);
        for(unsigned row=0;row<n;++row) {
            uint32_t value=mod.mul(result.tridiagonal.diagonal[column],input[row]);
            if(column)value=mod.add(value,mod.mul(
                result.tridiagonal.beta[column],result.basis.at(row,column-1)));
            if(column+1<n)value=mod.add(value,result.basis.at(row,column+1));
            expected[row]=value;
            if(actual[row]!=expected[row])
                throw std::runtime_error(
                    "Lanczos relation failed at "+std::to_string(row)+","+
                    std::to_string(column));
        }
    }
}

void apply_dense(
    const Matrix& matrix,const std::vector<uint32_t>& input,
    std::vector<uint32_t>& output,const Mod& mod) {
    std::fill(output.begin(),output.end(),0);
    for(unsigned row=0;row<matrix.n;++row) {
        uint32_t sum=0;
        for(unsigned column=0;column<matrix.n;++column)
            sum=mod.add(sum,mod.mul(matrix.at(row,column),input[column]));
        output[row]=sum;
    }
}

void apply_tridiagonal_low_rank(
    const Tridiagonal& tridiagonal,const std::vector<uint32_t>& metric,
    const RankMatrix& low_rank,const std::array<uint32_t,16>& kernel,
    const std::vector<uint32_t>& input,std::vector<uint32_t>& output,
    const Mod& mod) {
    const unsigned n=unsigned(input.size());
    std::fill(output.begin(),output.end(),0);
    for(unsigned column=0;column<n;++column) {
        output[column]=mod.add(
            output[column],mod.mul(tridiagonal.diagonal[column],input[column]));
        if(column)output[column-1]=mod.add(
            output[column-1],mod.mul(tridiagonal.beta[column],input[column]));
        if(column+1<n)output[column+1]=mod.add(output[column+1],input[column]);
    }
    std::array<uint32_t,4> projection{},mixed{};
    for(unsigned rank=0;rank<low_rank.columns;++rank)
        for(unsigned row=0;row<n;++row)
            projection[rank]=mod.add(projection[rank],mod.mul(
                low_rank.at(row,rank),mod.mul(metric[row],input[row])));
    for(unsigned row=0;row<low_rank.columns;++row)
        for(unsigned column=0;column<low_rank.columns;++column)
            mixed[row]=mod.add(mixed[row],mod.mul(kernel[row*4+column],projection[column]));
    for(unsigned row=0;row<n;++row)
        for(unsigned rank=0;rank<low_rank.columns;++rank)
            output[row]=mod.add(output[row],mod.mul(low_rank.at(row,rank),mixed[rank]));
}

Matrix build_signed_matrix(
    const six_by_twenty_eight::Query& query,uint64_t signs,const Mod& mod) {
    const unsigned n=query.vertices,half=n/2;
    Matrix matrix(n);
    for(unsigned row=0;row<n;++row)for(unsigned column=0;column<n;++column) {
        const unsigned edge=column%half;
        const unsigned paired=column<half?column+half:column-half;
        const bool positive=edge==0||(signs&(UINT64_C(1)<<(edge-1)));
        if(query.adjacency[size_t(row)*n+paired])
            matrix.at(row,column)=positive?1:mod.p-1;
    }
    return matrix;
}

struct SymmetricRankFactor {
    unsigned original_n=0,rank=0;
    std::vector<uint32_t> vectors; // row-major original_n by rank
    std::vector<uint32_t> diagonal,metric;
    uint32_t at(unsigned row,unsigned column)const {
        return vectors[size_t(row)*rank+column];
    }
};

SymmetricRankFactor factor_adjacency(
    const six_by_twenty_eight::Query& query,const Mod& mod) {
    const unsigned n=query.vertices;
    Matrix residual(n);
    for(unsigned row=0;row<n;++row)for(unsigned column=0;column<n;++column)
        residual.at(row,column)=query.adjacency[size_t(row)*n+column];
    std::vector<std::vector<uint32_t>> columns;
    std::vector<uint32_t> diagonal;
    while(true) {
        unsigned first=n,second=n;
        for(unsigned i=0;i<n;++i)if(residual.at(i,i)) { first=i; break; }
        std::vector<uint32_t> image(n);
        uint32_t pivot=0;
        if(first<n) {
            pivot=residual.at(first,first);
            for(unsigned row=0;row<n;++row)image[row]=residual.at(row,first);
        } else {
            for(unsigned i=0;i<n&&first==n;++i)for(unsigned j=i+1;j<n;++j)
                if(residual.at(i,j)) { first=i; second=j; break; }
            if(first==n)break;
            pivot=mod.add(residual.at(first,second),residual.at(first,second));
            for(unsigned row=0;row<n;++row)
                image[row]=mod.add(residual.at(row,first),residual.at(row,second));
        }
        const uint32_t inverse=mod.inverse(pivot);
        std::vector<uint32_t> vector(n);
        for(unsigned row=0;row<n;++row)vector[row]=mod.mul(image[row],inverse);
        for(unsigned row=0;row<n;++row)for(unsigned column=0;column<n;++column)
            residual.at(row,column)=mod.sub(
                residual.at(row,column),mod.mul(pivot,mod.mul(vector[row],vector[column])));
        columns.push_back(std::move(vector));
        diagonal.push_back(pivot);
    }
    SymmetricRankFactor factor;
    factor.original_n=n;
    factor.rank=unsigned(columns.size());
    factor.diagonal=diagonal;
    factor.metric.resize(factor.rank);
    factor.vectors.resize(size_t(n)*factor.rank);
    for(unsigned column=0;column<factor.rank;++column) {
        factor.metric[column]=mod.inverse(diagonal[column]);
        for(unsigned row=0;row<n;++row)
            factor.vectors[size_t(row)*factor.rank+column]=columns[column][row];
    }
    // Exact reconstruction is cheap at these dimensions and protects the
    // determinant reduction from a silent factorization error.
    for(unsigned row=0;row<n;++row)for(unsigned column=0;column<n;++column) {
        uint32_t value=0;
        for(unsigned k=0;k<factor.rank;++k)
            value=mod.add(value,mod.mul(
                factor.diagonal[k],mod.mul(factor.at(row,k),factor.at(column,k))));
        if(value!=uint32_t(query.adjacency[size_t(row)*n+column]))
            throw std::runtime_error("symmetric rank factor reconstruction failed");
    }
    return factor;
}

Matrix build_reduced_matrix(
    const SymmetricRankFactor& factor,uint64_t signs,const Mod& mod) {
    const unsigned n=factor.original_n,half=n/2,rank=factor.rank;
    Matrix matrix(rank);
    for(unsigned row=0;row<rank;++row)for(unsigned column=0;column<rank;++column) {
        uint32_t gram=0;
        for(unsigned edge=0;edge<half;++edge) {
            const bool positive=edge==0||(signs&(UINT64_C(1)<<(edge-1)));
            uint32_t value=mod.add(
                mod.mul(factor.at(edge,row),factor.at(edge+half,column)),
                mod.mul(factor.at(edge+half,row),factor.at(edge,column)));
            if(!positive)value=mod.neg(value);
            gram=mod.add(gram,value);
        }
        matrix.at(row,column)=mod.mul(factor.diagonal[row],gram);
    }
    return matrix;
}

RankMatrix make_reduced_update_factor(
    const SymmetricRankFactor& factor,unsigned edge,const Mod& mod,
    std::array<uint32_t,16>& kernel,bool new_positive) {
    RankMatrix result(factor.rank,2);
    for(unsigned row=0;row<factor.rank;++row) {
        result.at(row,0)=mod.mul(factor.diagonal[row],factor.at(edge,row));
        result.at(row,1)=mod.mul(
            factor.diagonal[row],factor.at(edge+factor.original_n/2,row));
    }
    kernel.fill(0);
    const uint32_t twice=2;
    const uint32_t delta=new_positive?twice:mod.neg(twice);
    kernel[1]=kernel[4]=delta;
    return result;
}

void upper_hessenberg(Matrix& matrix,const Mod& mod) {
    const unsigned n=matrix.n;
    for(unsigned column=0;column+2<n;++column) {
        unsigned pivot=column+1;
        while(pivot<n&&!matrix.at(pivot,column))++pivot;
        if(pivot==n)continue;
        if(pivot!=column+1) {
            for(unsigned j=0;j<n;++j)std::swap(matrix.at(pivot,j),matrix.at(column+1,j));
            for(unsigned i=0;i<n;++i)std::swap(matrix.at(i,pivot),matrix.at(i,column+1));
        }
        const uint32_t inverse=mod.inverse(matrix.at(column+1,column));
        for(unsigned row=column+2;row<n;++row) {
            const uint32_t factor=mod.mul(matrix.at(row,column),inverse);
            if(!factor)continue;
            for(unsigned j=column;j<n;++j)
                matrix.at(row,j)=mod.sub(
                    matrix.at(row,j),mod.mul(factor,matrix.at(column+1,j)));
            for(unsigned i=0;i<n;++i)
                matrix.at(i,column+1)=mod.add(
                    matrix.at(i,column+1),mod.mul(factor,matrix.at(i,row)));
        }
    }
}

std::vector<uint32_t> hessenberg_coefficients(Matrix matrix,unsigned degree,const Mod& mod) {
    upper_hessenberg(matrix,mod);
    const unsigned n=matrix.n;
    std::vector<std::vector<uint32_t>> poly(n+1,std::vector<uint32_t>(degree+1));
    poly[0][0]=1;
    // poly[size][defect] is the coefficient of lambda^(size-defect).
    for(unsigned size=1;size<=n;++size) {
        const unsigned diagonal=size-1;
        for(unsigned defect=0;defect<=std::min(size,degree);++defect) {
            uint32_t value=defect<=size-1?poly[size-1][defect]:0;
            if(defect)value=mod.sub(
                value,mod.mul(matrix.at(diagonal,diagonal),poly[size-1][defect-1]));
            uint32_t product=1;
            for(unsigned distance=1;distance<size&&distance+1<=defect;++distance) {
                const unsigned subrow=size-distance;
                product=mod.mul(product,matrix.at(subrow,subrow-1));
                const uint32_t factor=mod.mul(
                    product,matrix.at(size-distance-1,size-1));
                value=mod.sub(
                    value,mod.mul(factor,poly[size-distance-1][defect-distance-1]));
            }
            poly[size][defect]=value;
        }
    }
    return poly[n];
}

std::vector<uint32_t> tridiagonal_coefficients(
    const Tridiagonal& matrix,unsigned degree,const Mod& mod) {
    const unsigned n=unsigned(matrix.diagonal.size());
    std::vector<uint32_t> previous_previous(degree+1),previous(degree+1),next(degree+1);
    previous[0]=1;
    for(unsigned size=1;size<=n;++size) {
        std::fill(next.begin(),next.end(),0);
        const unsigned diagonal=size-1;
        for(unsigned defect=0;defect<=std::min(size,degree);++defect) {
            uint32_t value=defect<=size-1?previous[defect]:0;
            if(defect)value=mod.sub(
                value,mod.mul(matrix.diagonal[diagonal],previous[defect-1]));
            if(size>1&&defect>=2)value=mod.sub(
                value,mod.mul(matrix.beta[diagonal],previous_previous[defect-2]));
            next[defect]=value;
        }
        previous_previous=previous;
        previous=next;
    }
    return previous;
}

uint32_t term_from_coefficients(
    const std::vector<uint32_t>& characteristic,uint64_t signs,
    unsigned half,const Mod& mod) {
    std::vector<uint32_t> traces(half+1),coefficients(half+1);
    coefficients[0]=1;
    for(unsigned k=1;k<=half;++k) {
        uint32_t value=mod.mul(k,characteristic[k]);
        for(unsigned j=1;j<k;++j)
            value=mod.add(value,mod.mul(characteristic[j],traces[k-j]));
        traces[k]=mod.neg(value);
    }
    const uint32_t inverse_two=mod.inverse(2);
    for(unsigned degree=1;degree<=half;++degree) {
        uint32_t sum=0;
        for(unsigned k=1;k<=degree;++k)
            sum=mod.add(sum,mod.mul(
                mod.mul(traces[k],inverse_two),coefficients[degree-k]));
        coefficients[degree]=mod.mul(sum,mod.inverse(degree));
    }
    const unsigned negatives=(half-1)-unsigned(__builtin_popcountll(signs));
    return negatives&1?mod.neg(coefficients[half]):coefficients[half];
}

std::vector<uint32_t> polynomial_remainder(
    std::vector<uint32_t> dividend,const std::vector<uint32_t>& divisor,const Mod& mod) {
    while(!dividend.empty()&&!dividend.back())dividend.pop_back();
    if(divisor.empty())throw std::runtime_error("zero polynomial divisor");
    const uint32_t inverse_leading=mod.inverse(divisor.back());
    while(dividend.size()>=divisor.size()) {
        const size_t shift=dividend.size()-divisor.size();
        const uint32_t factor=mod.mul(dividend.back(),inverse_leading);
        for(size_t i=0;i<divisor.size();++i)
            dividend[shift+i]=mod.sub(dividend[shift+i],mod.mul(factor,divisor[i]));
        while(!dividend.empty()&&!dividend.back())dividend.pop_back();
    }
    return dividend;
}

std::vector<uint32_t> polynomial_gcd(
    std::vector<uint32_t> left,std::vector<uint32_t> right,const Mod& mod) {
    while(!right.empty()) {
        auto remainder=polynomial_remainder(std::move(left),right,mod);
        left=std::move(right);
        right=std::move(remainder);
    }
    if(left.empty())return left;
    const uint32_t inverse=mod.inverse(left.back());
    for(uint32_t& value:left)value=mod.mul(value,inverse);
    return left;
}

void report_repeated_factor(
    const six_by_twenty_eight::Query& query,uint64_t signs,const Mod& mod) {
    const unsigned n=query.vertices;
    // hessenberg_coefficients uses descending defect order; polynomial GCD
    // uses ascending powers of lambda.
    auto descending=hessenberg_coefficients(build_signed_matrix(query,signs,mod),n,mod);
    std::vector<uint32_t> polynomial(n+1),derivative(n);
    for(unsigned defect=0;defect<=n;++defect)
        polynomial[n-defect]=descending[defect];
    for(unsigned power=1;power<=n;++power)
        derivative[power-1]=mod.mul(power,polynomial[power]);
    auto repeated=polynomial_gcd(polynomial,derivative,mod);
    unsigned nonzero_terms=0,valuation=0;
    while(valuation<repeated.size()&&!repeated[valuation])++valuation;
    for(uint32_t value:repeated)nonzero_terms+=value!=0;
    std::printf(
        "GRAY_UPDATE_SPECTRUM repeated_degree=%zu repeated_valuation=%u "
        "repeated_nonzero_terms=%u\n",
        repeated.empty()?0:repeated.size()-1,valuation,nonzero_terms);
}

struct Transition {
    Matrix basis;
    std::vector<uint32_t> old_metric,new_metric,inverse_new_metric;
};

RankMatrix inverse_basis_apply(
    const Matrix& basis,const std::vector<uint32_t>& old_metric,
    const std::vector<uint32_t>& inverse_new_metric,const RankMatrix& input,const Mod& mod) {
    RankMatrix output(input.rows,input.columns);
    for(unsigned column=0;column<input.columns;++column)
        for(unsigned target=0;target<input.rows;++target) {
        uint32_t sum=0;
        for(unsigned source=0;source<input.rows;++source)
            sum=mod.add(sum,mod.mul(
                basis.at(source,target),mod.mul(old_metric[source],input.at(source,column))));
        output.at(target,column)=mod.mul(sum,inverse_new_metric[target]);
    }
    return output;
}

uint64_t number(const char* text) {
    char* end=nullptr;
    const uint64_t value=std::strtoull(text,&end,10);
    if(!end||*end)throw std::runtime_error("invalid integer");
    return value;
}

std::vector<unsigned> block_list(const std::string& text) {
    std::vector<unsigned> result;
    size_t start=0;
    while(start<text.size()) {
        const size_t comma=text.find(',',start);
        result.push_back(unsigned(number(text.substr(start,comma-start).c_str())));
        start=comma==std::string::npos?text.size():comma+1;
    }
    if(result.empty()||std::any_of(result.begin(),result.end(),[](unsigned x){return !x;}))
        throw std::runtime_error("invalid block list");
    return result;
}

} // namespace

int main(int argc,char** argv) {
    try {
        unsigned query_id=0,steps=64;
        bool deep_verify=false,catalog_census=false;
        uint64_t start=0;
        uint32_t prime=2147483647U;
        uint64_t seed=UINT64_C(0x677261792d686166);
        std::vector<unsigned> blocks{1,2,4,8,12,16};
        for(int i=1;i<argc;++i) {
            const std::string argument=argv[i];
            if(argument=="--query"&&i+1<argc)query_id=unsigned(number(argv[++i]));
            else if(argument=="--steps"&&i+1<argc)steps=unsigned(number(argv[++i]));
            else if(argument=="--start"&&i+1<argc)start=number(argv[++i]);
            else if(argument=="--prime"&&i+1<argc)prime=uint32_t(number(argv[++i]));
            else if(argument=="--seed"&&i+1<argc)seed=number(argv[++i]);
            else if(argument=="--blocks"&&i+1<argc)blocks=block_list(argv[++i]);
            else if(argument=="--deep-verify")deep_verify=true;
            else if(argument=="--catalog-census")catalog_census=true;
            else throw std::runtime_error(
                "usage: hafnian_gray_update_probe [--query Q] [--start N] [--steps N] "
                "[--prime P] [--blocks 1,2,4,8] [--seed N] [--deep-verify] "
                "[--catalog-census]");
        }
        const Mod mod{prime};
        auto catalog=six_by_twenty_eight::build_catalog();
        if(catalog_census) {
            struct Sector { uint64_t queries=0,accepted=0,terms=0,accepted_terms=0; };
            std::array<Sector,65> sectors{};
            unsigned __int128 all_terms=0,accepted_terms=0;
            for(const auto& census_query:catalog.queries) {
                unsigned prime_count=0;
                unsigned __int128 modulus=1,target=
                    static_cast<unsigned __int128>(1)<<census_query.matching_bound_power;
                for(uint32_t scheduled:
                        std::array<uint32_t,4>{2147483647U,2147483629U,2147483587U,2147483579U}) {
                    modulus*=scheduled;
                    ++prime_count;
                    if(modulus>target)break;
                }
                const uint64_t term_images=(UINT64_C(1)<<(census_query.vertices/2-1))*prime_count;
                Sector& sector=sectors[census_query.vertices];
                ++sector.queries;
                sector.terms+=term_images;
                all_terms+=term_images;
                bool accepted=false;
                try {
                    const SymmetricRankFactor candidate=factor_adjacency(census_query,mod);
                    uint64_t state=seed^uint64_t(census_query.id)*UINT64_C(0x9e3779b97f4a7c15);
                    state^=state>>30;state*=UINT64_C(0xbf58476d1ce4e5b9);
                    state^=state>>27;state*=UINT64_C(0x94d049bb133111eb);
                    state^=state>>31;
                    state&=(UINT64_C(1)<<(census_query.vertices/2-1))-1;
                    Matrix reduced=build_reduced_matrix(candidate,state,mod);
                    std::mt19937_64 local_random(seed^census_query.id);
                    (void)generalized_lanczos(
                        candidate.rank,candidate.metric,
                        [&](const auto& input,auto& output){
                            apply_dense(reduced,input,output,mod);
                        },mod,local_random,4);
                    accepted=true;
                } catch(const std::exception&) {}
                if(accepted) {
                    ++sector.accepted;
                    sector.accepted_terms+=term_images;
                    accepted_terms+=term_images;
                }
            }
            for(unsigned order=0;order<sectors.size();++order)if(sectors[order].queries)
                std::printf(
                    "GRAY_UPDATE_CENSUS order=%u queries=%" PRIu64 " accepted=%" PRIu64
                    " term_images=%" PRIu64 " accepted_term_images=%" PRIu64 "\n",
                    order,sectors[order].queries,sectors[order].accepted,
                    sectors[order].terms,sectors[order].accepted_terms);
            auto print_u128=[](unsigned __int128 value) {
                char digits[64];unsigned count=0;
                do {digits[count++]=char('0'+value%10);value/=10;}while(value);
                while(count)std::putchar(digits[--count]);
            };
            std::printf("GRAY_UPDATE_CENSUS_TOTAL term_images=");print_u128(all_terms);
            std::printf(" accepted_term_images=");print_u128(accepted_terms);
            const double fraction=double(accepted_terms)/double(all_terms);
            std::printf(" accepted_fraction=%.9f\n",fraction);
            return 0;
        }
        if(query_id>=catalog.queries.size())throw std::runtime_error("invalid query");
        const auto& query=catalog.queries[query_id];
        const unsigned n=query.vertices,half=n/2;
        const SymmetricRankFactor factorization=factor_adjacency(query,mod);
        const uint64_t maximum=UINT64_C(1)<<(half-1);
        if(start>=maximum)throw std::runtime_error("start exceeds sign range");
        steps=unsigned(std::min<uint64_t>(steps,maximum-start));
        std::printf(
            "GRAY_UPDATE_CONFIG query=%u vertices=%u reduced_rank=%u prime=%u "
            "start=%" PRIu64 " steps=%u\n",
            query_id,n,factorization.rank,prime,start,steps);
        report_repeated_factor(query,start^(start>>1),mod);

        for(unsigned block:blocks) {
            std::mt19937_64 random(seed^uint64_t(block)*UINT64_C(0x9e3779b97f4a7c15));
            Tridiagonal current;
            Matrix base_basis;
            std::vector<uint32_t> base_metric,base_inverse_metric,current_metric;
            std::vector<Transition> transitions;
            double reset_seconds=0,update_seconds=0,term_seconds=0,reference_seconds=0;
            uint64_t attempts=0,updates=0;
            for(unsigned index=0;index<steps;++index) {
                const uint64_t gray_index=start+index;
                const uint64_t signs=gray_index^(gray_index>>1);
                if(index%block==0) {
                    const auto started=Clock::now();
                    const Matrix reduced=build_reduced_matrix(factorization,signs,mod);
                    auto rebuilt=generalized_lanczos(
                        factorization.rank,factorization.metric,
                        [&](const auto& input,auto& output){
                            apply_dense(reduced,input,output,mod);
                        },mod,random);
                    current=std::move(rebuilt.tridiagonal);
                    base_basis=std::move(rebuilt.basis);
                    base_metric=std::move(rebuilt.metric);
                    base_inverse_metric=std::move(rebuilt.inverse_metric);
                    current_metric=base_metric;
                    transitions.clear();
                    attempts+=rebuilt.attempts;
                    reset_seconds+=std::chrono::duration<double>(Clock::now()-started).count();
                } else {
                    const auto started=Clock::now();
                    const unsigned flipped=unsigned(__builtin_ctzll(gray_index))+1;
                    std::array<uint32_t,16> kernel{};
                    const bool new_positive=signs&(UINT64_C(1)<<(flipped-1));
                    RankMatrix factor=make_reduced_update_factor(
                        factorization,flipped,mod,kernel,new_positive);
                    Matrix next_reduced;
                    if(deep_verify) {
                        const uint64_t previous_index=gray_index-1;
                        const uint64_t previous_signs=previous_index^(previous_index>>1);
                        const Matrix previous_reduced=build_reduced_matrix(
                            factorization,previous_signs,mod);
                        next_reduced=build_reduced_matrix(factorization,signs,mod);
                        for(unsigned row=0;row<factorization.rank;++row)
                            for(unsigned column=0;column<factorization.rank;++column) {
                                uint32_t delta=0;
                                for(unsigned a=0;a<factor.columns;++a)
                                    for(unsigned b=0;b<factor.columns;++b)
                                        delta=mod.add(delta,mod.mul(
                                            factor.at(row,a),mod.mul(kernel[a*4+b],
                                            mod.mul(factor.at(column,b),factorization.metric[column]))));
                                if(mod.add(previous_reduced.at(row,column),delta)!=
                                        next_reduced.at(row,column))
                                    throw std::runtime_error("fixed-coordinate rank-two identity failed");
                            }
                    }
                    factor=inverse_basis_apply(
                        base_basis,factorization.metric,base_inverse_metric,factor,mod);
                    for(const Transition& transition:transitions)
                        factor=inverse_basis_apply(
                            transition.basis,transition.old_metric,
                            transition.inverse_new_metric,factor,mod);
                    if(deep_verify&&transitions.empty()) {
                        std::vector<uint32_t> unit(factorization.rank),actual_vector(factorization.rank);
                        RankMatrix fixed_image(factorization.rank,1);
                        for(unsigned column=0;column<factorization.rank;++column) {
                            std::fill(unit.begin(),unit.end(),0);
                            unit[column]=1;
                            apply_tridiagonal_low_rank(
                                current,current_metric,factor,kernel,unit,actual_vector,mod);
                            for(unsigned row=0;row<factorization.rank;++row) {
                                uint32_t value=0;
                                for(unsigned inner=0;inner<factorization.rank;++inner)
                                    value=mod.add(value,mod.mul(
                                        next_reduced.at(row,inner),base_basis.at(inner,column)));
                                fixed_image.at(row,0)=value;
                            }
                            RankMatrix expected_vector=inverse_basis_apply(
                                base_basis,factorization.metric,base_inverse_metric,
                                fixed_image,mod);
                            for(unsigned row=0;row<factorization.rank;++row)
                                if(actual_vector[row]!=expected_vector.at(row,0))
                                    throw std::runtime_error(
                                        "transformed rank-two operator identity failed at "+
                                        std::to_string(row)+","+std::to_string(column));
                        }
                    }
                    auto updated=generalized_lanczos(
                        factorization.rank,current_metric,
                        [&](const auto& input,auto& output){
                            apply_tridiagonal_low_rank(
                                current,current_metric,factor,kernel,input,output,mod);
                        },mod,random);
                    if(deep_verify)verify_lanczos_relation(
                            updated,
                            [&](const auto& input,auto& output){
                                apply_tridiagonal_low_rank(
                                    current,current_metric,factor,kernel,input,output,mod);
                            },mod);
                    Transition transition{
                        std::move(updated.basis),current_metric,updated.metric,
                        updated.inverse_metric};
                    current=std::move(updated.tridiagonal);
                    current_metric=transition.new_metric;
                    transitions.push_back(std::move(transition));
                    attempts+=updated.attempts;
                    ++updates;
                    update_seconds+=std::chrono::duration<double>(Clock::now()-started).count();
                }

                auto started=Clock::now();
                const uint32_t actual=term_from_coefficients(
                    tridiagonal_coefficients(current,half,mod),signs,half,mod);
                term_seconds+=std::chrono::duration<double>(Clock::now()-started).count();
                started=Clock::now();
                const uint32_t expected=term_from_coefficients(
                    hessenberg_coefficients(build_signed_matrix(query,signs,mod),half,mod),
                    signs,half,mod);
                reference_seconds+=std::chrono::duration<double>(Clock::now()-started).count();
                if(actual!=expected) {
                    const uint32_t reduced_expected=term_from_coefficients(
                        hessenberg_coefficients(
                            build_reduced_matrix(factorization,signs,mod),half,mod),
                        signs,half,mod);
                    std::fprintf(stderr,
                        "mismatch block=%u index=%u signs=%" PRIu64
                        " expected=%u reduced_expected=%u actual=%u\n",
                        block,index,signs,expected,reduced_expected,actual);
                    return 1;
                }
            }
            const double dynamic_seconds=reset_seconds+update_seconds+term_seconds;
            std::printf(
                "GRAY_UPDATE_RESULT block=%u exact=OK attempts=%" PRIu64
                " updates=%" PRIu64 " reset_seconds=%.9f update_seconds=%.9f "
                "term_seconds=%.9f dynamic_seconds=%.9f reference_seconds=%.9f "
                "speedup=%.6f\n",
                block,attempts,updates,reset_seconds,update_seconds,term_seconds,
                dynamic_seconds,reference_seconds,reference_seconds/dynamic_seconds);
        }
        return 0;
    } catch(const std::exception& error) {
        std::fprintf(stderr,"error: %s\n",error.what());
        return 2;
    }
}
