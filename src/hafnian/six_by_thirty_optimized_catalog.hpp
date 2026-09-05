#pragma once

#include "hafnian_matching_bound.hpp"

namespace six_by_thirty_optimized {

constexpr unsigned WIDTH=30,QUERY_COUNT=1;
struct Query : six_by_twenty_nine::Query {
    uint16_t matching_bound_power=0;
};
struct Catalog {
    std::vector<Query> queries;
    std::string digest;
};

inline Catalog build_catalog() {
    // Expand a perfect matching at token (colour 0, row pair {0,1}). Its
    // 18 neighbours are equivalent under the stabilizer S_3 x S_2 x S_4:
    // permute the other colours and the four rows outside {0,1}.
    // Consequently pm(H) = 18 * pm(H - {u,v}) for any one neighbour v.
    six_by_twenty_nine::Geometry geometry;
    uint64_t occupied=0;
    unsigned degree=0;
    for(unsigned token=six_by_twenty_nine::PAIRS;token<60;++token) {
        auto [a,b]=geometry.pairs[token%six_by_twenty_nine::PAIRS];
        if(a<2||b<2)continue;
        uint64_t candidate=six_by_twenty_nine::canonicalize(
            geometry,UINT64_C(1)|(UINT64_C(1)<<token));
        if(degree&&candidate!=occupied)
            throw std::runtime_error("endpoint neighbour orbits differ");
        occupied=candidate;
        ++degree;
    }
    if(degree!=18)throw std::runtime_error("endpoint degree is not 18");
    Query query;
    query.occupied=occupied;
    // This is the Laplace coefficient, not a defect count. Restore ALL 30
    // factors of two and 30! only after reconstructing the minor by CRT.
    query.defect_coefficient=degree;
    six_by_twenty_nine::build_query_graph(geometry,query,WIDTH);
    query.matching_bound_power=hafnian_matching_bound::matching_bound_power(
        geometry,occupied,0);
    if(query.vertices!=58||query.matching_bound_power!=85)
        throw std::runtime_error("endpoint minor census/bound mismatch");
    Catalog catalog;
    catalog.queries.push_back(std::move(query));
    Sha256 hash;
    hash.update("six-by-thirty-edge-minor-catalog-v2\n");
    hash.update(catalog.queries[0].digest);
    hash.update(&catalog.queries[0].matching_bound_power,sizeof(uint16_t));
    catalog.digest=hash.finish_hex();
    return catalog;
}

} // namespace six_by_thirty_optimized
