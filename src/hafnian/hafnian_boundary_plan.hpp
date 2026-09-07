#pragma once
#include <vector>
#include <stdexcept>

// Prime-independent dependency order for exact even-subset hafnians.
// No polynomial workspace or modular inverses are needed to prepare it.
inline std::vector<unsigned> hafnian_boundary_plan(
        unsigned q,const std::vector<unsigned>& roots) {
    if(q>13)throw std::runtime_error("boundary pool exceeds bounded gate");
    std::vector<bool> seen(1u<<q);
    std::vector<unsigned> plan;
    auto visit=[&](auto&& self,unsigned mask)->void {
        if(seen[mask])return;
        seen[mask]=true;
        if(mask){
            unsigned rest=mask^(mask&-mask);
            for(unsigned candidates=rest;candidates;candidates&=candidates-1)
                self(self,rest^(candidates&-candidates));
        }
        plan.push_back(mask);
    };
    for(unsigned mask:roots){
        if(mask>=(1u<<q)||(__builtin_popcount(mask)&1))
            throw std::runtime_error("invalid even boundary mask");
        visit(visit,mask);
    }
    return plan;
}
