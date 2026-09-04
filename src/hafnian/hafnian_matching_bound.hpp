#pragma once

#include "six_by_twenty_nine_catalog.hpp"

namespace hafnian_matching_bound {
using six_by_twenty_nine::Geometry;
using six_by_twenty_nine::PAIRS;
using six_by_twenty_nine::TOKENS;

inline uint64_t factorial(unsigned value) {
    uint64_t result=1;
    for(unsigned factor=2;factor<=value;++factor)result*=factor;
    return result;
}

inline uint64_t binomial(unsigned n,unsigned k) {
    if(k>n)return 0;
    k=std::min(k,n-k);
    uint64_t result=1;
    for(unsigned i=1;i<=k;++i)result=result*(n-k+i)/i;
    return result;
}

inline unsigned ceil_log2_u64(uint64_t value) {
    if(value<=1)return 0;
    unsigned floor=63U-unsigned(__builtin_clzll(value));
    return (value&(value-1))?floor+1:floor;
}

inline unsigned ceil_log2_factorial(unsigned degree) {
    return ceil_log2_u64(factorial(degree));
}

inline uint16_t matching_bound_power(
    const Geometry& geometry,uint64_t occupied,unsigned unmatched) {
    // Each k-matching chooses its unmatched vertices and then a perfect
    // matching of the induced residual graph.  Friedland's degree bound is
    // rounded upward using a common exact denominator, so this deliberately
    // returns a conservative integral power-of-two bound.
    constexpr uint64_t DENOMINATOR=24504480; // lcm(2,4,...,36)
    std::array<uint64_t,TOKENS> neighbours{};
    for(unsigned left=0;left<TOKENS;++left) {
        unsigned left_colour=left/PAIRS;
        auto [a,b]=geometry.pairs[left%PAIRS];
        for(unsigned right=0;right<TOKENS;++right) {
            unsigned right_colour=right/PAIRS;
            auto [c,d]=geometry.pairs[right%PAIRS];
            if(left_colour!=right_colour&&a!=c&&a!=d&&b!=c&&b!=d)
                neighbours[left]|=UINT64_C(1)<<right;
        }
    }
    const uint64_t remaining=((UINT64_C(1)<<TOKENS)-1)&~occupied;
    uint64_t numerator=0,scan=remaining;
    while(scan) {
        unsigned vertex=unsigned(__builtin_ctzll(scan));
        scan&=scan-1;
        unsigned degree=unsigned(__builtin_popcountll(neighbours[vertex]&remaining));
        if(degree)
            numerator+=uint64_t(ceil_log2_factorial(degree))*
                (DENOMINATOR/(2*degree));
    }
    unsigned perfect_matching_power=unsigned(
        (numerator+DENOMINATOR-1)/DENOMINATOR);
    unsigned original_vertices=unsigned(__builtin_popcountll(remaining));
    return uint16_t(perfect_matching_power+
        ceil_log2_u64(binomial(original_vertices,unmatched)));
}

} // namespace hafnian_matching_bound
