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
    for(uint32_t a:boundary)for(uint32_t b:boundary)check(a,b);
    for(unsigned sample=0;sample<RANDOM_CASES;++sample)
        check(uint32_t(next_random(state)%P),uint32_t(next_random(state)%P));
    for(uint32_t value:boundary) {
        if(!value)continue;
        uint32_t encoded=uint32_t(uint64_t(value)*fixed.one%P);
        uint32_t inverse=hafnian_host_montgomery_power(encoded,P-2,fixed);
        if(hafnian_host_montgomery_mul(encoded,inverse,fixed)!=fixed.one)
            throw std::runtime_error("fixed Montgomery inverse mismatch");
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
    std::printf("FIXED_MONTGOMERY_ARITHMETIC primes=4 products=%" PRIu64
        " exact=OK\n",checked);
    return 0;
} catch(const std::exception& error) {
    std::fprintf(stderr,"error: %s\n",error.what());
    return 1;
}
