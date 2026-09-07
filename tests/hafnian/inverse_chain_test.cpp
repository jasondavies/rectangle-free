#include "../../src/hafnian/hafnian_inverse_chain.hpp"
#include <cstdio>
#include <random>
#include <stdexcept>

template<uint32_t P> void check() {
    unsigned operations=0;
    auto exponent=hafnian_inverse_chain<P>(1,[&](uint32_t a,uint32_t b){
        ++operations;return uint32_t((uint64_t(a)+b)%(P-1));
    });
    if(exponent!=P-2||operations>38)throw std::runtime_error("invalid addition chain");
    auto mul=[](uint32_t a,uint32_t b){return uint32_t(uint64_t(a)*b%P);};
    const uint32_t r=uint32_t((UINT64_C(1)<<32)%P);
    const uint32_t rinv=hafnian_inverse_chain<P>(r,mul);
    auto mont=[&](uint32_t a,uint32_t b){return mul(mul(a,b),rinv);};
    std::mt19937_64 random(484);
    for(unsigned i=0;i<40000;++i){
        uint32_t a=i<8?i:uint32_t(random()%P);
        if(i>=8&&i<16)a=P-(i-7);
        const auto inverse=hafnian_inverse_chain<P>(a,mul);
        if(a?mul(a,inverse)!=1:inverse!=0)throw std::runtime_error("ordinary inverse mismatch");
        const auto encoded=hafnian_inverse_chain<P>(mul(a,r),mont);
        if(mul(encoded,rinv)!=inverse)throw std::runtime_error("Montgomery inverse mismatch");
    }
    std::printf("INVERSE_CHAIN_TEST prime=%u multiplications=%u cases=40000 ordinary=OK montgomery=OK\n",P,operations);
}
int main(){
    check<2147483647>();check<2147483629>();check<2147483587>();check<2147483579>();
}
