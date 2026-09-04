#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "six_by_twenty_nine_catalog.hpp"
#include "hafnian_matching_bound.hpp"

namespace six_by_twenty_eight {

using U128=unsigned __int128;
using six_by_twenty_nine::Geometry;
using six_by_twenty_nine::PAIRS;
using six_by_twenty_nine::TOKENS;
using six_by_twenty_nine::WeightedSupport;
constexpr unsigned WIDTH=28,SLACK=2,BUDGET=2*SLACK,QUERY_COUNT=36398;

struct Query : six_by_twenty_nine::Query {
    // The exact matching count is at most 2^matching_bound_power.  Requiring
    // the CRT modulus to exceed that power of two gives a simple certified
    // per-query stopping rule.
    uint16_t matching_bound_power=0;
};

struct Catalog {
    std::vector<Query> queries;
    std::string digest;
};

using hafnian_matching_bound::factorial;
using hafnian_matching_bound::matching_bound_power;

inline Catalog build_catalog() {
    Geometry geometry;
    std::vector<WeightedSupport> defects;
    for(const WeightedSupport& support:
            six_by_twenty_nine::weighted_supports(geometry))
        if(support.excess&&support.excess<=BUDGET)defects.push_back(support);
    if(defects.size()!=1340)
        throw std::runtime_error("6x28 defect-support census mismatch");

    using Coefficients=std::unordered_map<uint64_t,U128>;
    std::array<std::array<Coefficients,BUDGET+1>,BUDGET+1> layers;
    layers[0][0][0]=1;
    std::unordered_map<uint64_t,uint64_t> canonical_cache;
    canonical_cache.reserve(300000);
    for(unsigned count=0;count<BUDGET;++count) {
        for(unsigned excess=0;excess<BUDGET;++excess) {
            for(const auto& [occupied,coefficient]:layers[count][excess]) {
                for(const WeightedSupport& support:defects) {
                    unsigned child_excess=excess+support.excess;
                    if(child_excess>BUDGET||(occupied&support.mask))continue;
                    uint64_t raw=occupied|support.mask;
                    auto inserted=canonical_cache.emplace(raw,0);
                    if(inserted.second)
                        inserted.first->second=
                            six_by_twenty_nine::canonicalize(geometry,raw);
                    layers[count+1][child_excess][inserted.first->second]+=
                        coefficient*support.weight;
                }
            }
        }
    }

    Catalog catalog;
    catalog.queries.reserve(QUERY_COUNT);
    for(unsigned count=0;count<=BUDGET;++count) {
        uint64_t divisor=factorial(count);
        for(unsigned excess=0;excess<=BUDGET;++excess) {
            for(const auto& [occupied,ordered_coefficient]:layers[count][excess]) {
                if(ordered_coefficient%divisor)
                    throw std::runtime_error("ordered coefficient is not divisible by d!");
                U128 coefficient=ordered_coefficient/divisor;
                if(coefficient>std::numeric_limits<uint64_t>::max())
                    throw std::overflow_error("6x28 defect coefficient exceeds uint64_t");
                Query query;
                query.occupied=occupied;
                query.defect_count=uint8_t(count);
                query.excess=uint8_t(excess);
                query.unmatched=uint8_t(BUDGET-excess);
                query.defect_coefficient=uint64_t(coefficient);
                query.matching_bound_power=matching_bound_power(
                    geometry,occupied,query.unmatched);
                six_by_twenty_nine::build_query_graph(geometry,query,WIDTH);
                catalog.queries.push_back(std::move(query));
            }
        }
    }
    std::sort(catalog.queries.begin(),catalog.queries.end(),
        [](const Query& a,const Query& b) {
            if(a.excess!=b.excess)return a.excess<b.excess;
            if(a.defect_count!=b.defect_count)return a.defect_count<b.defect_count;
            return a.occupied<b.occupied;
        });
    if(catalog.queries.size()!=QUERY_COUNT)
        throw std::runtime_error("6x28 canonical query census mismatch");

    std::map<std::pair<unsigned,unsigned>,std::array<uint64_t,2>> sectors;
    Sha256 hash;
    const std::string header="six-by-twenty-eight-residual-catalog-v1\n";
    hash.update(header);
    for(unsigned id=0;id<catalog.queries.size();++id) {
        Query& query=catalog.queries[id];
        query.id=id;
        auto& sector=sectors[{query.excess,query.defect_count}];
        ++sector[0];
        sector[1]+=query.defect_coefficient;
        hash.update(query.digest);
        hash.update(&query.matching_bound_power,sizeof(query.matching_bound_power));
    }
    const std::map<std::pair<unsigned,unsigned>,std::array<uint64_t,2>> expected={
        {{0,0},{1,1}},{{1,1},{2,840}},{{2,1},{1,1440}},
        {{2,2},{25,303660}},{{3,2},{36,993600}},
        {{3,3},{664,62422320}},{{4,1},{2,480}},
        {{4,2},{42,800640}},{{4,3},{2548,291375360}},
        {{4,4},{33077,8126516160}}};
    if(sectors!=expected)
        throw std::runtime_error("6x28 canonical defect sector mismatch");
    catalog.digest=hash.finish_hex();
    return catalog;
}

} // namespace six_by_twenty_eight
