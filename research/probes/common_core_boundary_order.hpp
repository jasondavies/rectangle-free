#pragma once
// Offline exact coordinate permutation. Only operation counts guide this
// bounded search; no claimed GPU speedup and no change to query ownership.
#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>
#include <cstdint>

namespace common_boundary {
using Score=std::tuple<unsigned,unsigned,unsigned>; // convolutions, states, edges
inline Score score(unsigned q,const std::vector<unsigned>& roots,
                   const std::vector<unsigned>& order) {
    std::vector<uint8_t> seen(1u<<q),edges(q*q);unsigned states=0,transitions=0,pairs=0;
    auto visit=[&](auto&& self,unsigned mask)->void {
        if(seen[mask])return;seen[mask]=1;++states;if(!mask)return;
        unsigned first=0;for(auto i:order)if(mask&(1u<<i)){first=i;break;}
        unsigned rest=mask^(1u<<first);
        for(unsigned bits=rest;bits;bits&=bits-1){unsigned bit=bits&-bits,j=__builtin_ctz(bit);
            unsigned edge=std::min(first,j)*q+std::max(first,j);
            if(!edges[edge]){edges[edge]=1;++pairs;}
            if(__builtin_popcount(mask)>2)++transitions;
            self(self,rest^bit);
        }
    };
    for(auto mask:roots)visit(visit,mask);
    return {transitions,states,pairs};
}
inline std::vector<unsigned> choose(unsigned q,const std::vector<unsigned>& roots,unsigned trials) {
    std::vector<unsigned> best(q);std::iota(best.begin(),best.end(),0);
    auto cost=score(q,roots,best);uint64_t state=481;
    for(auto mask:roots)state=state*UINT64_C(6364136223846793005)+mask+1;
    for(unsigned trial=0;trial<trials;++trial){std::vector<unsigned> order(q);std::iota(order.begin(),order.end(),0);
        for(unsigned i=q;i>1;--i){state=state*UINT64_C(6364136223846793005)+1442695040888963407ULL;
            std::swap(order[i-1],order[(state>>32)%i]);}
        auto candidate=score(q,roots,order);if(candidate<cost){cost=candidate;best=std::move(order);}
    }
    return best;
}
}
