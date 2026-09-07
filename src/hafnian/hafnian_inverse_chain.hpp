#pragma once
#include <cstdint>

// Exact 37--38-multiply chains for the four certified hafnian fields.
// mul operates in the caller's representation (ordinary or Montgomery).
// Keeping the chain independent of reduction avoids representation conversions.
template<uint32_t P, class Multiply>
#ifdef __CUDACC__
__device__
#endif
inline uint32_t hafnian_inverse_chain(uint32_t a, Multiply mul) {
    static_assert(P==2147483647U || P==2147483629U ||
                  P==2147483587U || P==2147483579U, "uncertified inverse chain");
    auto square_n=[&](uint32_t value,unsigned count) {
#pragma unroll
        for(unsigned i=0;i<count;++i)value=mul(value,value);
        return value;
    };
    uint32_t a255=0,tail=0;
    if constexpr(P==2147483647U) {
        const uint32_t a4=square_n(a,2);
        const uint32_t a5=mul(a4,a);
        const uint32_t a10=mul(a5,a5);
        const uint32_t a15=mul(a5,a10);
        const uint32_t a120=square_n(a15,3);
        const uint32_t a125=mul(a5,a120);
        const uint32_t a250=mul(a125,a125);
        a255=mul(a5,a250);tail=a125;
    } else if constexpr(P==2147483629U) {
        const uint32_t a2=mul(a,a);
        const uint32_t a4=mul(a2,a2);
        const uint32_t a6=mul(a2,a4);
        const uint32_t a7=mul(a,a6);
        const uint32_t a9=mul(a2,a7);
        const uint32_t a16=mul(a7,a9);
        const uint32_t a25=mul(a9,a16);
        const uint32_t a41=mul(a16,a25);
        const uint32_t a82=mul(a41,a41);
        const uint32_t a107=mul(a25,a82);
        const uint32_t a214=mul(a107,a107);
        a255=mul(a41,a214);tail=a107;
    } else if constexpr(P==2147483587U) {
        const uint32_t a4=square_n(a,2);
        const uint32_t a5=mul(a4,a);
        const uint32_t a10=mul(a5,a5);
        const uint32_t a15=mul(a5,a10);
        const uint32_t a60=square_n(a15,2);
        const uint32_t a65=mul(a5,a60);
        const uint32_t a130=mul(a65,a65);
        const uint32_t a195=mul(a65,a130);
        a255=mul(a60,a195);tail=a65;
    } else {
        const uint32_t a2=mul(a,a);
        const uint32_t a3=mul(a,a2);
        const uint32_t a24=square_n(a3,3);
        const uint32_t a27=mul(a3,a24);
        const uint32_t a54=mul(a27,a27);
        const uint32_t a57=mul(a3,a54);
        const uint32_t a228=square_n(a57,2);
        a255=mul(a27,a228);tail=a57;
    }
    const uint32_t a65535=mul(square_n(a255,8),a255);
    const uint32_t a16777215=mul(square_n(a65535,8),a255);
    return mul(square_n(a16777215,7),tail);
}
