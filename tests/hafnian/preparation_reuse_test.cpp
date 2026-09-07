#define CORE_HOST_EMULATION
#include "../../src/hafnian/hafnian_common_core.cuh"
#include "../../src/hafnian/hafnian_matching_bound.hpp"
#include "../../src/hafnian/hafnian_common_catalog.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
#include <random>

// Independent copy of the previous per-query construction, including its
// integer rounding. Compare bounds, not floating approximations to them.
unsigned original_bound(const six_by_twenty_nine::Geometry& g,
                        uint64_t occupied, unsigned unmatched) {
    using namespace hafnian_matching_bound;
    uint64_t neighbours[60]{};
    for(unsigned i=0;i<60;++i)for(unsigned j=0;j<60;++j) {
        auto [a,b]=g.pairs[i%15]; auto [c,d]=g.pairs[j%15];
        if(i/15!=j/15&&a!=c&&a!=d&&b!=c&&b!=d)neighbours[i]|=UINT64_C(1)<<j;
    }
    uint64_t remaining=((UINT64_C(1)<<60)-1)&~occupied,numerator=0;
    for(unsigned i=0;i<60;++i)if(remaining&(UINT64_C(1)<<i)) {
        unsigned degree=__builtin_popcountll(neighbours[i]&remaining);
        if(degree)numerator+=uint64_t(ceil_log2_factorial(degree))*(24504480/(2*degree));
    }
    return (numerator+24504479)/24504480+
        ceil_log2_u64(binomial(__builtin_popcountll(remaining),unmatched));
}

int main(int argc, char** argv) {
    assert(argc<=2);
    six_by_twenty_nine::Geometry geometry;
    for(unsigned i=0;i<60;++i) {
        assert(__builtin_popcountll(geometry.matching_neighbours[i])==18);
        assert(!(geometry.matching_neighbours[i]&(UINT64_C(1)<<i)));
        for(unsigned j=0;j<60;++j)
            assert(((geometry.matching_neighbours[i]>>j)&1)==
                   ((geometry.matching_neighbours[j]>>i)&1));
    }
    std::mt19937_64 rng(494);
    std::vector<std::pair<uint64_t,unsigned>> inputs{{0,0},{0,6},
        {(UINT64_C(1)<<60)-1,0},{(UINT64_C(1)<<60)-2,1}};
    for(unsigned i=0;i<10000;++i)inputs.emplace_back(rng()&((UINT64_C(1)<<60)-1),i%7);
    if(argc==2) {
        unsigned slack;std::string digest;
        auto rows=common_catalog::read(argv[1],slack,digest);
        for(const auto& row:rows) {
            uint64_t mask=row.key&((UINT64_C(1)<<60)-1);
            unsigned excess=__builtin_popcountll(mask)-2*(row.key>>60);
            inputs.emplace_back(mask,2*slack-excess);
        }
    }
    auto start=std::chrono::steady_clock::now();
    std::vector<unsigned> expected;
    for(auto [mask,monomers]:inputs)expected.push_back(original_bound(geometry,mask,monomers));
    double baseline=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    start=std::chrono::steady_clock::now();
    for(size_t i=0;i<inputs.size();++i)
        assert(hafnian_matching_bound::matching_bound_power(
            geometry,inputs[i].first,inputs[i].second)==expected[i]);
    double cached=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    // Alternate all production fields, then exceed cache capacity and revisit
    // evicted fields. Every returned entry is checked, including slot zero.
    core_gpu::Input in{};
    unsigned checks=0;
    for(unsigned repeat=0;repeat<10;++repeat)
        for(uint32_t p:{2147483647u,2147483629u,2147483587u,2147483579u,1000003u,1000033u}) {
            for(unsigned hit=0;hit<2;++hit) {
                core_gpu::set_field(in,p);
                assert(in.inverse[0]==0);
                for(unsigned j=1;j<49;++j) {
                    assert(uint64_t(j)*in.inverse[j]%p==1);
                    assert(in.inverse[j]==core_gpu::host_power(j,p-2,p));
                    ++checks;
                }
            }
        }
    std::cout<<"PREPARATION_REUSE exact=OK bounds="<<inputs.size()
             <<" inverse_checks="<<checks<<" baseline_seconds="<<baseline
             <<" cached_seconds="<<cached<<'\n';
}
