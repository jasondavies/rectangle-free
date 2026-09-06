#include "../../research/probes/common_core_boundary_order.hpp"
#include <random>
#include <iostream>
#include <stdexcept>
int main(){
    std::mt19937 rng(481);unsigned checks=0;
    for(unsigned q:{5u,7u,9u,11u,13u})for(unsigned trial=0;trial<20;++trial){
        std::vector<unsigned> roots;
        for(unsigned mask=0;mask<(1u<<q);++mask)
            if(unsigned(__builtin_popcount(mask))==q-3&&rng()%5==0)roots.push_back(mask);
        if(roots.empty())roots.push_back((1u<<(q-3))-1);
        std::vector<unsigned> identity(q);std::iota(identity.begin(),identity.end(),0);
        auto order=common_boundary::choose(q,roots,16),sorted=order;
        std::sort(sorted.begin(),sorted.end());
        if(sorted!=identity||order!=common_boundary::choose(q,roots,16))throw std::runtime_error("non-deterministic/non-permutation order");
        auto baseline=common_boundary::score(q,roots,identity),candidate=common_boundary::score(q,roots,order);
        if(candidate>baseline)throw std::runtime_error("ordering worsened structural objective");
        auto relabelled=roots;
        for(auto& mask:relabelled){unsigned changed=0;
            for(unsigned j=0;j<q;++j)if(mask&(1u<<order[j]))changed|=1u<<j;mask=changed;}
        if(common_boundary::score(q,relabelled,identity)!=candidate)throw std::runtime_error("relabelled dependency graph differs");
        ++checks;
    }
    std::cout<<"CORE_BOUNDARY_ORDER_TEST cases="<<checks<<" deterministic=OK relabel=OK nonworsening=OK\n";
}
