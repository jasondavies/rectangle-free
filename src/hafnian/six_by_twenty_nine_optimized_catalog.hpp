#pragma once

#include "hafnian_matching_bound.hpp"

namespace six_by_twenty_nine_optimized {

constexpr unsigned WIDTH=29,QUERY_COUNT=33;
struct Query : six_by_twenty_nine::Query {
    uint16_t matching_bound_power=0;
};
struct Catalog {
    std::vector<Query> queries;
    std::string digest;
};

inline Catalog build_catalog() {
    const auto old=six_by_twenty_nine::build_catalog();
    six_by_twenty_nine::Geometry geometry;
    std::map<uint64_t,uint64_t> unused_pairs;
    for(unsigned i=0;i<60;++i)for(unsigned j=i+1;j<60;++j)
        ++unused_pairs[six_by_twenty_nine::canonicalize(
            geometry,(UINT64_C(1)<<i)|(UINT64_C(1)<<j))];
    if(unused_pairs.size()!=5)throw std::runtime_error("unused-pair orbit census mismatch");
    Catalog catalog;
    std::vector<uint64_t> coefficients;
    for(const auto& [occupied,coefficient]:unused_pairs) {
        Query query;
        // These two removed tokens are monomers, NOT defect columns. Their
        // choice is now in coefficient, so no dummy vertices or r! remain.
        query.occupied=occupied;
        query.defect_coefficient=coefficient;
        six_by_twenty_nine::build_query_graph(geometry,query,WIDTH);
        if(query.vertices!=58)throw std::runtime_error("monomer minor order mismatch");
        catalog.queries.push_back(std::move(query));
        coefficients.push_back(coefficient);
    }
    std::sort(coefficients.begin(),coefficients.end());
    if(coefficients!=std::vector<uint64_t>{90,180,240,540,720})
        throw std::runtime_error("unused-pair orbit weights mismatch");
    for(size_t i=1;i<old.queries.size();++i) {
        Query query;
        static_cast<six_by_twenty_nine::Query&>(query)=old.queries[i];
        catalog.queries.push_back(std::move(query));
    }
    if(catalog.queries.size()!=QUERY_COUNT)
        throw std::runtime_error("optimized 6x29 catalog size mismatch");
    Sha256 hash;
    hash.update("six-by-twenty-nine-monomer-catalog-v2\n");
    for(unsigned i=0;i<catalog.queries.size();++i) {
        auto& query=catalog.queries[i];
        query.id=i;
        query.matching_bound_power=hafnian_matching_bound::matching_bound_power(
            geometry,query.occupied,query.unmatched);
        if(query.matching_bound_power>=93)
            throw std::runtime_error("three-prime 6x29 bound exceeded");
        hash.update(query.digest);
        hash.update(&query.matching_bound_power,sizeof(query.matching_bound_power));
    }
    catalog.digest=hash.finish_hex();
    return catalog;
}

} // namespace six_by_twenty_nine_optimized
