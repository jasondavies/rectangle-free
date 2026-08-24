#include "../../src/hafnian/hafnian_gpu_core.cuh"

#include <array>
#include <cinttypes>
#include <cstdio>
#include <stdexcept>

namespace {

uint64_t next_random(uint64_t& state) {
    state^=state<<13;
    state^=state>>7;
    state^=state<<17;
    return state;
}

template<uint32_t P>
uint64_t check_prime(uint64_t& state) {
    constexpr unsigned RANDOM_CASES=2000000;
    HafnianMontgomeryConstant<P> fixed;
    HafnianMontgomery runtime{
        P,0U-hafnian_inverse_mod_2_32(P),uint32_t((UINT64_C(1)<<32)%P)};
    std::array<uint32_t,8> boundary={0U,1U,2U,P/2,P/2+1,P-3,P-2,P-1};
    uint64_t checked=0;
    auto check=[&](uint32_t a,uint32_t b) {
        uint32_t encoded_a=uint32_t(uint64_t(a)*fixed.one%P);
        uint32_t encoded_b=uint32_t(uint64_t(b)*fixed.one%P);
        uint32_t fixed_product=hafnian_host_montgomery_mul(encoded_a,encoded_b,fixed);
        uint32_t runtime_product=hafnian_host_montgomery_mul(encoded_a,encoded_b,runtime);
        uint32_t decoded=hafnian_host_montgomery_mul(fixed_product,1,fixed);
        if(fixed_product!=runtime_product||decoded!=uint32_t(uint64_t(a)*b%P))
            throw std::runtime_error("fixed Montgomery product mismatch");
        ++checked;
    };
    auto check_sum=[&](uint32_t a,uint32_t b,uint32_t c,uint32_t d) {
        const uint32_t encoded_a=uint32_t(uint64_t(a)*fixed.one%P);
        const uint32_t encoded_b=uint32_t(uint64_t(b)*fixed.one%P);
        const uint32_t encoded_c=uint32_t(uint64_t(c)*fixed.one%P);
        const uint32_t encoded_d=uint32_t(uint64_t(d)*fixed.one%P);
        const uint32_t fixed_sum=hafnian_sum_products2(
            encoded_a,encoded_b,encoded_c,encoded_d,fixed);
        const uint32_t runtime_sum=hafnian_sum_products2(
            encoded_a,encoded_b,encoded_c,encoded_d,runtime);
        const uint32_t decoded=hafnian_host_montgomery_mul(fixed_sum,1,fixed);
        const uint32_t expected=uint32_t(
            (uint64_t(a)*b+uint64_t(c)*d)%P);
        if(fixed_sum!=runtime_sum||decoded!=expected)
            throw std::runtime_error("fused Montgomery sum mismatch");
        ++checked;
    };
    auto check_sum4=[&](uint32_t a,uint32_t b,uint32_t c,uint32_t d,
            uint32_t e,uint32_t f,uint32_t g,uint32_t h) {
        const uint32_t encoded_a=uint32_t(uint64_t(a)*fixed.one%P);
        const uint32_t encoded_b=uint32_t(uint64_t(b)*fixed.one%P);
        const uint32_t encoded_c=uint32_t(uint64_t(c)*fixed.one%P);
        const uint32_t encoded_d=uint32_t(uint64_t(d)*fixed.one%P);
        const uint32_t encoded_e=uint32_t(uint64_t(e)*fixed.one%P);
        const uint32_t encoded_f=uint32_t(uint64_t(f)*fixed.one%P);
        const uint32_t encoded_g=uint32_t(uint64_t(g)*fixed.one%P);
        const uint32_t encoded_h=uint32_t(uint64_t(h)*fixed.one%P);
        const uint32_t sum=hafnian_sum_products4(encoded_a,encoded_b,
            encoded_c,encoded_d,encoded_e,encoded_f,encoded_g,encoded_h,fixed);
        const uint32_t decoded=hafnian_host_montgomery_mul(sum,1,fixed);
        const uint32_t expected=uint32_t((uint64_t(a)*b+uint64_t(c)*d+
            uint64_t(e)*f+uint64_t(g)*h)%P);
        if(decoded!=expected)
            throw std::runtime_error("four-product Montgomery sum mismatch");
        ++checked;
    };
    for(uint32_t a:boundary)for(uint32_t b:boundary)check(a,b);
    for(uint32_t a:boundary)for(uint32_t b:boundary)
        for(uint32_t c:boundary)for(uint32_t d:boundary) {
            check_sum(a,b,c,d);
            check_sum4(a,b,c,d,a,d,c,b);
        }
    for(unsigned sample=0;sample<RANDOM_CASES;++sample) {
        check(uint32_t(next_random(state)%P),uint32_t(next_random(state)%P));
        check_sum(uint32_t(next_random(state)%P),uint32_t(next_random(state)%P),
            uint32_t(next_random(state)%P),uint32_t(next_random(state)%P));
        check_sum4(uint32_t(next_random(state)%P),uint32_t(next_random(state)%P),
            uint32_t(next_random(state)%P),uint32_t(next_random(state)%P),
            uint32_t(next_random(state)%P),uint32_t(next_random(state)%P),
            uint32_t(next_random(state)%P),uint32_t(next_random(state)%P));
    }
    for(uint32_t value:boundary) {
        if(!value)continue;
        uint32_t encoded=uint32_t(uint64_t(value)*fixed.one%P);
        uint32_t inverse=hafnian_host_montgomery_power(encoded,P-2,fixed);
        if(hafnian_host_montgomery_mul(encoded,inverse,fixed)!=fixed.one)
            throw std::runtime_error("fixed Montgomery inverse mismatch");
    }
    return checked;
}

uint64_t check_mersenne(uint64_t& state) {
    constexpr unsigned RANDOM_CASES=2000000;
    constexpr uint32_t P=HafnianMersenne31::p;
    HafnianMersenne31 mod;
    std::array<uint32_t,8> boundary={0U,1U,2U,P/2,P/2+1,P-3,P-2,P-1};
    uint64_t checked=0;
    auto check=[&](uint32_t a,uint32_t b) {
        const uint32_t product=hafnian_host_montgomery_mul(a,b,mod);
        if(product!=uint32_t(uint64_t(a)*b%P))
            throw std::runtime_error("Mersenne product mismatch");
        ++checked;
    };
    auto check_sum=[&](uint32_t a,uint32_t b,uint32_t c,uint32_t d) {
        const uint32_t sum=hafnian_sum_products2(a,b,c,d,mod);
        const uint32_t expected=uint32_t(
            (uint64_t(a)*b+uint64_t(c)*d)%P);
        if(sum!=expected)throw std::runtime_error("fused Mersenne sum mismatch");
        ++checked;
    };
    auto check_sum4=[&](uint32_t a,uint32_t b,uint32_t c,uint32_t d,
            uint32_t e,uint32_t f,uint32_t g,uint32_t h) {
        const uint32_t sum=hafnian_sum_products4(a,b,c,d,e,f,g,h,mod);
        const uint32_t expected=uint32_t((uint64_t(a)*b+uint64_t(c)*d+
            uint64_t(e)*f+uint64_t(g)*h)%P);
        if(sum!=expected)
            throw std::runtime_error("four-product Mersenne sum mismatch");
        ++checked;
    };
    for(uint32_t a:boundary)for(uint32_t b:boundary)check(a,b);
    for(uint32_t a:boundary)for(uint32_t b:boundary)
        for(uint32_t c:boundary)for(uint32_t d:boundary) {
            check_sum(a,b,c,d);
            check_sum4(a,b,c,d,a,d,c,b);
        }
    for(unsigned sample=0;sample<RANDOM_CASES;++sample) {
        check(uint32_t(next_random(state)%P),uint32_t(next_random(state)%P));
        check_sum(uint32_t(next_random(state)%P),uint32_t(next_random(state)%P),
            uint32_t(next_random(state)%P),uint32_t(next_random(state)%P));
        check_sum4(uint32_t(next_random(state)%P),uint32_t(next_random(state)%P),
            uint32_t(next_random(state)%P),uint32_t(next_random(state)%P),
            uint32_t(next_random(state)%P),uint32_t(next_random(state)%P),
            uint32_t(next_random(state)%P),uint32_t(next_random(state)%P));
    }
    for(uint32_t value:boundary) {
        if(!value)continue;
        const uint32_t inverse=hafnian_host_montgomery_power(value,P-2,mod);
        if(hafnian_host_montgomery_mul(value,inverse,mod)!=1)
            throw std::runtime_error("Mersenne inverse mismatch");
    }
    return checked;
}

} // namespace

int main() try {
    uint64_t state=UINT64_C(0xd1b54a32d192ed03),checked=0;
    checked+=check_prime<2147483647U>(state);
    checked+=check_prime<2147483629U>(state);
    checked+=check_prime<2147483587U>(state);
    checked+=check_prime<2147483579U>(state);
    checked+=check_mersenne(state);
    std::printf("FIXED_MONTGOMERY_ARITHMETIC primes=4 mersenne=1 products=%" PRIu64
        " exact=OK\n",checked);
    return 0;
} catch(const std::exception& error) {
    std::fprintf(stderr,"error: %s\n",error.what());
    return 1;
}
