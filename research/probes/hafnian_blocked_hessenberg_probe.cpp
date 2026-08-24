// Exact CPU gate for the compact blocked Gauss-Hessenberg transformation used
// by the small-prime hafnian research path.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr unsigned MAX_N=64;

struct Mod {
    uint32_t p;
    uint32_t add(uint32_t a,uint32_t b)const {
        uint32_t value=a+b;
        return value>=p?value-p:value;
    }
    uint32_t sub(uint32_t a,uint32_t b)const {
        return a>=b?a-b:a+p-b;
    }
    uint32_t mul(uint32_t a,uint32_t b)const {
        return uint32_t(uint64_t(a)*b%p);
    }
    uint32_t power(uint32_t a,uint32_t exponent)const {
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
    std::array<uint32_t,MAX_N*MAX_N> values{};
    uint32_t& at(unsigned row,unsigned column){return values[row*MAX_N+column];}
    uint32_t at(unsigned row,unsigned column)const{return values[row*MAX_N+column];}
};

void symmetric_swap(Matrix& matrix,unsigned a,unsigned b) {
    if(a==b)return;
    for(unsigned column=0;column<matrix.n;++column)
        std::swap(matrix.at(a,column),matrix.at(b,column));
    for(unsigned row=0;row<matrix.n;++row)
        std::swap(matrix.at(row,a),matrix.at(row,b));
}

void scalar_hessenberg(Matrix& matrix,const Mod& mod) {
    const unsigned n=matrix.n;
    std::array<uint32_t,MAX_N> factors{};
    for(unsigned column=0;column+2<n;++column) {
        unsigned pivot=column+1;
        while(pivot<n&&!matrix.at(pivot,column))++pivot;
        if(pivot==n)continue;
        symmetric_swap(matrix,pivot,column+1);
        const uint32_t inverse=mod.inverse(matrix.at(column+1,column));
        for(unsigned row=column+2;row<n;++row)
            factors[row]=mod.mul(matrix.at(row,column),inverse);
        for(unsigned row=column+2;row<n;++row)
            for(unsigned j=column;j<n;++j)
                matrix.at(row,j)=mod.sub(
                    matrix.at(row,j),mod.mul(factors[row],matrix.at(column+1,j)));
        for(unsigned row=0;row<n;++row) {
            uint32_t sum=0;
            for(unsigned eliminated=column+2;eliminated<n;++eliminated)
                sum=mod.add(sum,mod.mul(matrix.at(row,eliminated),factors[eliminated]));
            matrix.at(row,column+1)=mod.add(matrix.at(row,column+1),sum);
        }
    }
}

// Reduce one panel implicitly.  Successful scalar eliminations have
// E_j=I-f_j e_{s_j}^T.  Since e_{s_i}^T f_j=0 for i<j,
//
//   R=E_0^-1 ... E_{b-1}^-1 = I+F S^T,
//   Q=R^-1 = I-F (I+S^T F)^-1 S^T = I-W S^T.
//
// Individual panel columns are obtained from Q*A*R without updating the
// trailing matrix.  At panel completion the two dense products A*F and
// W*(S^T B) apply the complete similarity transformation exactly.
void blocked_panel(
    Matrix& matrix,unsigned first_column,unsigned column_count,const Mod& mod) {
    const unsigned n=matrix.n;
    Matrix base=matrix;
    std::array<unsigned,MAX_N> selectors{};
    std::array<uint32_t,MAX_N*MAX_N> factors{},inverse_lower{},w{};
    unsigned rank=0;
    auto entry=[](auto& values,unsigned row,unsigned column)->auto& {
        return values[row*MAX_N+column];
    };

    const unsigned end=std::min(first_column+column_count,n-2);
    for(unsigned column=first_column;column<end;++column) {
        // x=R*e_column.  Usually only the immediately preceding factor is
        // selected, but the general loop also covers a skipped elimination.
        std::array<uint32_t,MAX_N> x{},y{},current_column{};
        x[column]=1;
        for(unsigned k=0;k<rank;++k)if(selectors[k]==column)
            for(unsigned row=0;row<n;++row)
                x[row]=mod.add(x[row],entry(factors,row,k));
        for(unsigned row=0;row<n;++row) {
            uint32_t sum=0;
            for(unsigned k=0;k<n;++k)
                sum=mod.add(sum,mod.mul(base.at(row,k),x[k]));
            y[row]=sum;
        }
        for(unsigned row=0;row<n;++row) {
            uint32_t correction=0;
            for(unsigned k=0;k<rank;++k)
                correction=mod.add(
                    correction,mod.mul(entry(w,row,k),y[selectors[k]]));
            current_column[row]=mod.sub(y[row],correction);
        }

        const unsigned selector=column+1;
        unsigned pivot=selector;
        while(pivot<n&&!current_column[pivot])++pivot;
        if(pivot==n)continue;
        if(pivot!=selector) {
            symmetric_swap(base,pivot,selector);
            std::swap(current_column[pivot],current_column[selector]);
            for(unsigned k=0;k<rank;++k) {
                std::swap(entry(factors,pivot,k),entry(factors,selector,k));
                std::swap(entry(w,pivot,k),entry(w,selector,k));
            }
        }

        const uint32_t inverse=mod.inverse(current_column[selector]);
        for(unsigned row=selector+1;row<n;++row)
            entry(factors,row,rank)=mod.mul(current_column[row],inverse);
        selectors[rank]=selector;

        // Append the last row of T=(I+S^T F)^-1.  Its diagonal is one and
        // r^T=-l^T*T_old for l_i=F[s_new,i].
        for(unsigned column_index=0;column_index<rank;++column_index) {
            uint32_t sum=0;
            for(unsigned k=column_index;k<rank;++k)
                sum=mod.add(sum,mod.mul(
                    entry(factors,selector,k),entry(inverse_lower,k,column_index)));
            entry(inverse_lower,rank,column_index)=sum?mod.p-sum:0;
        }
        entry(inverse_lower,rank,rank)=1;

        // W_new=[F_old*T_old+f_new*r^T, f_new].
        for(unsigned old_column=0;old_column<rank;++old_column) {
            const uint32_t multiplier=entry(inverse_lower,rank,old_column);
            for(unsigned row=0;row<n;++row)
                entry(w,row,old_column)=mod.add(
                    entry(w,row,old_column),
                    mod.mul(entry(factors,row,rank),multiplier));
        }
        for(unsigned row=0;row<n;++row)
            entry(w,row,rank)=entry(factors,row,rank);
        ++rank;
    }

    // B=A*(I+F*S^T).  Only the selector columns change.
    Matrix updated=base;
    for(unsigned row=0;row<n;++row)for(unsigned k=0;k<rank;++k) {
        uint32_t product=0;
        for(unsigned inner=0;inner<n;++inner)
            product=mod.add(product,mod.mul(
                base.at(row,inner),entry(factors,inner,k)));
        updated.at(row,selectors[k])=mod.add(updated.at(row,selectors[k]),product);
    }

    // A'=Q*B=B-W*(S^T*B).
    matrix=updated;
    for(unsigned row=0;row<n;++row)for(unsigned column=0;column<n;++column) {
        uint32_t product=0;
        for(unsigned k=0;k<rank;++k)
            product=mod.add(product,mod.mul(
                entry(w,row,k),updated.at(selectors[k],column)));
        matrix.at(row,column)=mod.sub(matrix.at(row,column),product);
    }
}

void blocked_hessenberg(Matrix& matrix,unsigned panel_width,const Mod& mod) {
    for(unsigned first=0;first+2<matrix.n;first+=panel_width)
        blocked_panel(matrix,first,panel_width,mod);
}

bool equal(const Matrix& left,const Matrix& right) {
    if(left.n!=right.n)return false;
    for(unsigned row=0;row<left.n;++row)
        for(unsigned column=0;column<left.n;++column)
            if(left.at(row,column)!=right.at(row,column))return false;
    return true;
}

uint64_t number(const char* text) {
    char* end=nullptr;
    const uint64_t value=std::strtoull(text,&end,10);
    if(!end||*end)throw std::runtime_error("invalid integer");
    return value;
}

} // namespace

int main(int argc,char** argv) {
    try {
        unsigned samples=20;
        uint64_t seed=UINT64_C(0x6b6f636b65646865);
        for(int i=1;i<argc;++i) {
            const std::string argument=argv[i];
            if(argument=="--samples"&&i+1<argc)samples=unsigned(number(argv[++i]));
            else if(argument=="--seed"&&i+1<argc)seed=number(argv[++i]);
            else throw std::runtime_error(
                "usage: hafnian_blocked_hessenberg_probe [--samples N] [--seed N]");
        }
        std::mt19937_64 random(seed);
        const std::array<unsigned,4> sizes{48,54,60,64};
        const std::array<unsigned,4> widths{8,16,24,32};
        const std::array<uint32_t,3> primes{61,127,251};
        uint64_t cases=0,pivot_stress_cases=0;
        for(uint32_t prime:primes)for(unsigned n:sizes)for(unsigned sample=0;sample<samples;++sample) {
            Matrix original;
            original.n=n;
            const bool sparse=sample%2==0;
            for(unsigned row=0;row<n;++row)for(unsigned column=0;column<n;++column) {
                uint32_t value=uint32_t(random()%prime);
                if(sparse&&(random()%5))value=0;
                original.at(row,column)=value;
            }
            // Force nontrivial row/column pivots and occasional skipped
            // columns without making the whole matrix structurally special.
            if(sample%4==0) {
                for(unsigned row=1;row<std::min(n,8U);++row)original.at(row,0)=0;
                original.at(std::min(n-1,8U),0)=1;
                ++pivot_stress_cases;
            }
            Matrix expected=original;
            scalar_hessenberg(expected,Mod{prime});
            for(unsigned width:widths) {
                Matrix actual=original;
                blocked_hessenberg(actual,width,Mod{prime});
                ++cases;
                if(!equal(expected,actual)) {
                    for(unsigned row=0;row<n;++row)for(unsigned column=0;column<n;++column)
                        if(expected.at(row,column)!=actual.at(row,column)) {
                            std::fprintf(stderr,
                                "mismatch p=%u n=%u sample=%u panel=%u row=%u col=%u expected=%u actual=%u\n",
                                prime,n,sample,width,row,column,
                                expected.at(row,column),actual.at(row,column));
                            return 1;
                        }
                }
            }
        }
        std::printf(
            "BLOCKED_HESSENBERG cases=%llu pivot_stress=%llu primes=3 sizes=4 panels=4 exact=OK\n",
            static_cast<unsigned long long>(cases),
            static_cast<unsigned long long>(pivot_stress_cases));
        return 0;
    } catch(const std::exception& error) {
        std::fprintf(stderr,"error: %s\n",error.what());
        return 1;
    }
}
