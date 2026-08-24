#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../common/sha256.hpp"

namespace six_by_twenty_nine {

constexpr unsigned ROWS=6,COLOURS=4,PAIRS=15,TOKENS=60,WIDTH=29;
constexpr uint16_t FULL_PAIR_MASK=(UINT16_C(1)<<PAIRS)-1;

struct Geometry {
    std::array<std::pair<uint8_t,uint8_t>,PAIRS> pairs{};
    std::vector<std::array<std::array<uint16_t,256>,2>> row_tables;

    Geometry() {
        unsigned next=0;
        for(unsigned i=0;i<ROWS;++i)for(unsigned j=i+1;j<ROWS;++j)
            pairs[next++]={uint8_t(i),uint8_t(j)};
        auto pair_index=[&](unsigned a,unsigned b) {
            if(a>b)std::swap(a,b);
            for(unsigned p=0;p<PAIRS;++p)
                if(pairs[p].first==a&&pairs[p].second==b)return p;
            throw std::runtime_error("row-pair lookup failed");
        };
        std::array<unsigned,ROWS> permutation{};
        std::iota(permutation.begin(),permutation.end(),0);
        do {
            std::array<unsigned,PAIRS> image{};
            for(unsigned p=0;p<PAIRS;++p)
                image[p]=pair_index(permutation[pairs[p].first],permutation[pairs[p].second]);
            std::array<std::array<uint16_t,256>,2> table{};
            for(unsigned chunk=0;chunk<2;++chunk)for(unsigned byte=0;byte<256;++byte)
                for(unsigned bit=0;bit<8;++bit) {
                    unsigned source=8*chunk+bit;
                    if(source<PAIRS&&(byte&(1U<<bit)))
                        table[chunk][byte]|=uint16_t(1U<<image[source]);
                }
            row_tables.push_back(table);
        } while(std::next_permutation(permutation.begin(),permutation.end()));
        if(row_tables.size()!=720)throw std::runtime_error("S6 construction failed");
    }
};

inline uint64_t pack_planes(std::array<uint16_t,COLOURS> planes) {
    std::sort(planes.begin(),planes.end());
    uint64_t result=0;
    for(unsigned colour=0;colour<COLOURS;++colour)
        result|=uint64_t(planes[colour])<<(colour*PAIRS);
    return result;
}

inline std::array<uint16_t,COLOURS> unpack_planes(uint64_t mask) {
    std::array<uint16_t,COLOURS> planes{};
    for(unsigned colour=0;colour<COLOURS;++colour)
        planes[colour]=uint16_t((mask>>(colour*PAIRS))&FULL_PAIR_MASK);
    return planes;
}

inline uint64_t canonicalize(const Geometry& geometry,uint64_t mask) {
    auto planes=unpack_planes(mask);
    uint64_t best=UINT64_MAX;
    for(const auto& table:geometry.row_tables) {
        std::array<uint16_t,COLOURS> transformed{};
        for(unsigned colour=0;colour<COLOURS;++colour)
            transformed[colour]=table[0][planes[colour]&255]|table[1][planes[colour]>>8];
        best=std::min(best,pack_planes(transformed));
    }
    return best;
}

struct WeightedSupport {
    uint64_t mask=0;
    uint16_t weight=0;
    uint8_t excess=0;
};

inline std::vector<WeightedSupport> weighted_supports(const Geometry& geometry) {
    std::unordered_map<uint64_t,uint16_t> weights;
    weights.reserve(4096);
    for(unsigned encoded=0;encoded<(1U<<(2*ROWS));++encoded) {
        unsigned value=encoded;
        std::array<unsigned,ROWS> row_colour{};
        for(unsigned row=0;row<ROWS;++row) {
            row_colour[row]=value&3U;
            value>>=2;
        }
        uint64_t mask=0;
        for(unsigned colour=0;colour<COLOURS;++colour)
            for(unsigned pair=0;pair<PAIRS;++pair) {
                auto [a,b]=geometry.pairs[pair];
                if(row_colour[a]==colour&&row_colour[b]==colour)
                    mask|=UINT64_C(1)<<(colour*PAIRS+pair);
            }
        ++weights[mask];
    }
    std::vector<WeightedSupport> result;
    result.reserve(weights.size());
    uint64_t physical=0;
    for(auto [mask,weight]:weights) {
        unsigned size=unsigned(__builtin_popcountll(mask));
        if(size<2)throw std::runtime_error("six-row support below size two");
        result.push_back({mask,weight,uint8_t(size-2)});
        physical+=weight;
    }
    std::sort(result.begin(),result.end(),[](const auto& a,const auto& b) {
        if(a.excess!=b.excess)return a.excess<b.excess;
        return a.mask<b.mask;
    });
    if(result.size()!=2088||physical!=4096)
        throw std::runtime_error("weighted column-support census mismatch");
    return result;
}

struct DefectKey {
    uint64_t occupied=0;
    uint8_t count=0;
    bool operator==(const DefectKey& other)const {
        return occupied==other.occupied&&count==other.count;
    }
};

struct DefectKeyHash {
    size_t operator()(const DefectKey& key)const {
        uint64_t value=key.occupied^(uint64_t(key.count)*UINT64_C(0x9e3779b97f4a7c15));
        value^=value>>30;value*=UINT64_C(0xbf58476d1ce4e5b9);
        value^=value>>27;value*=UINT64_C(0x94d049bb133111eb);
        return size_t(value^(value>>31));
    }
};

struct Query {
    unsigned id=0;
    uint64_t occupied=0;
    uint64_t defect_coefficient=0;
    uint8_t defect_count=0;
    uint8_t excess=0;
    uint8_t unmatched=0;
    uint8_t vertices=0;
    std::vector<uint8_t> adjacency;
    std::vector<uint8_t> order;
    std::string digest;
};

inline bool find_perfect_matching(
    const std::vector<uint64_t>& neighbours,uint64_t remaining,
    std::vector<std::pair<unsigned,unsigned>>& matching) {
    if(!remaining)return true;
    unsigned pivot=0,best_degree=65;
    uint64_t scan=remaining;
    while(scan) {
        unsigned vertex=unsigned(__builtin_ctzll(scan));
        scan&=scan-1;
        unsigned degree=unsigned(__builtin_popcountll(neighbours[vertex]&remaining));
        if(degree<best_degree) {
            best_degree=degree;
            pivot=vertex;
            if(degree<=1)break;
        }
    }
    if(!best_degree)return false;
    uint64_t candidates=neighbours[pivot]&remaining;
    uint64_t without_pivot=remaining&~(UINT64_C(1)<<pivot);
    while(candidates) {
        unsigned mate=unsigned(__builtin_ctzll(candidates));
        candidates&=candidates-1;
        matching.push_back({pivot,mate});
        if(find_perfect_matching(neighbours,
                without_pivot&~(UINT64_C(1)<<mate),matching))return true;
        matching.pop_back();
    }
    return false;
}

inline void build_query_graph(const Geometry& geometry,Query& query,unsigned width=WIDTH) {
    std::vector<unsigned> originals;
    for(unsigned token=0;token<TOKENS;++token)
        if(!(query.occupied&(UINT64_C(1)<<token)))originals.push_back(token);
    unsigned original_count=unsigned(originals.size());
    query.vertices=uint8_t(original_count+query.unmatched);
    if(query.vertices%2||query.vertices<48||query.vertices>64)
        throw std::runtime_error("unexpected augmented graph order");
    unsigned n=query.vertices;
    std::vector<uint8_t> natural(size_t(n)*n);
    auto token_adjacent=[&](unsigned left,unsigned right) {
        unsigned left_colour=left/PAIRS,right_colour=right/PAIRS;
        auto [a,b]=geometry.pairs[left%PAIRS];
        auto [c,d]=geometry.pairs[right%PAIRS];
        return left_colour!=right_colour&&a!=c&&a!=d&&b!=c&&b!=d;
    };
    for(unsigned i=0;i<original_count;++i)
        for(unsigned j=0;j<original_count;++j)
            natural[i*n+j]=uint8_t(token_adjacent(originals[i],originals[j]));
    for(unsigned dummy=original_count;dummy<n;++dummy)
        for(unsigned original=0;original<original_count;++original)
            natural[dummy*n+original]=natural[original*n+dummy]=1;

    std::vector<uint64_t> neighbours(n);
    for(unsigned i=0;i<n;++i)for(unsigned j=0;j<n;++j)
        if(natural[i*n+j])neighbours[i]|=UINT64_C(1)<<j;
    std::vector<std::pair<unsigned,unsigned>> matching;
    uint64_t full=n==64?UINT64_MAX:((UINT64_C(1)<<n)-1);
    if(!find_perfect_matching(neighbours,full,matching)||matching.size()!=n/2)
        throw std::runtime_error("augmented residual graph has no perfect matching");
    std::sort(matching.begin(),matching.end());
    query.order.reserve(n);
    for(auto [first,second]:matching)query.order.push_back(uint8_t(first));
    for(auto [first,second]:matching)query.order.push_back(uint8_t(second));
    query.adjacency.resize(size_t(n)*n);
    for(unsigned i=0;i<n;++i)for(unsigned j=0;j<n;++j)
        query.adjacency[i*n+j]=natural[query.order[i]*n+query.order[j]];
    for(unsigned pair=0;pair<n/2;++pair)
        if(!query.adjacency[pair*n+pair+n/2])
            throw std::runtime_error("reference matching reorder failed");

    Sha256 hash;
    const std::string header=width==29
        ? "six-by-twenty-nine-residual-query-v1\n"
        : "six-by-twenty-eight-residual-query-v1\n";
    hash.update(header);
    uint8_t metadata[4]={query.defect_count,query.excess,query.unmatched,query.vertices};
    hash.update(metadata,sizeof(metadata));
    hash.update(&query.occupied,sizeof(query.occupied));
    hash.update(&query.defect_coefficient,sizeof(query.defect_coefficient));
    hash.update(natural.data(),natural.size());
    hash.update(query.order.data(),query.order.size());
    query.digest=hash.finish_hex();
}

struct Catalog {
    std::vector<Query> queries;
    std::string digest;
};

inline Catalog build_catalog() {
    Geometry geometry;
    auto supports=weighted_supports(geometry);
    std::vector<WeightedSupport> excess_one,excess_two;
    for(const auto& support:supports) {
        if(support.excess==1)excess_one.push_back(support);
        else if(support.excess==2)excess_two.push_back(support);
    }
    if(excess_one.size()!=440||excess_two.size()!=720)
        throw std::runtime_error("defect support census mismatch");

    std::unordered_map<DefectKey,uint64_t,DefectKeyHash> raw;
    raw.reserve(84000);
    raw[{0,0}]=1;
    for(const auto& support:excess_one)
        raw[{support.mask,1}]+=support.weight;
    for(const auto& support:excess_two)
        raw[{support.mask,1}]+=support.weight;
    for(size_t i=0;i<excess_one.size();++i)
        for(size_t j=i+1;j<excess_one.size();++j) {
            if(excess_one[i].mask&excess_one[j].mask)continue;
            raw[{excess_one[i].mask|excess_one[j].mask,2}]+=
                uint64_t(excess_one[i].weight)*excess_one[j].weight;
        }
    if(raw.size()!=83071)throw std::runtime_error("raw defect union census mismatch");

    std::unordered_map<DefectKey,uint64_t,DefectKeyHash> orbit;
    orbit.reserve(raw.size());
    for(const auto& [key,coefficient]:raw)
        orbit[{canonicalize(geometry,key.occupied),key.count}]+=coefficient;
    if(orbit.size()!=29)throw std::runtime_error("canonical defect query census mismatch");

    Catalog catalog;
    for(const auto& [key,coefficient]:orbit) {
        Query query;
        query.occupied=key.occupied;
        query.defect_count=key.count;
        query.excess=uint8_t(__builtin_popcountll(key.occupied)-2*key.count);
        query.unmatched=uint8_t(2-query.excess);
        query.defect_coefficient=coefficient;
        build_query_graph(geometry,query);
        catalog.queries.push_back(std::move(query));
    }
    std::sort(catalog.queries.begin(),catalog.queries.end(),[](const Query& a,const Query& b) {
        if(a.excess!=b.excess)return a.excess<b.excess;
        if(a.defect_count!=b.defect_count)return a.defect_count<b.defect_count;
        return a.occupied<b.occupied;
    });

    std::map<std::pair<unsigned,unsigned>,std::array<uint64_t,2>> sectors;
    for(unsigned id=0;id<catalog.queries.size();++id) {
        auto& query=catalog.queries[id];
        query.id=id;
        auto& sector=sectors[{query.excess,query.defect_count}];
        ++sector[0];
        sector[1]+=query.defect_coefficient;
    }
    const std::map<std::pair<unsigned,unsigned>,std::array<uint64_t,2>> expected={
        {{0,0},{1,1}},{{1,1},{2,840}},{{2,1},{1,1440}},{{2,2},{25,303660}}};
    if(sectors!=expected)throw std::runtime_error("canonical defect sector mismatch");

    Sha256 hash;
    const std::string header="six-by-twenty-nine-residual-catalog-v1\n";
    hash.update(header);
    for(const auto& query:catalog.queries)hash.update(query.digest);
    catalog.digest=hash.finish_hex();
    return catalog;
}

} // namespace six_by_twenty_nine
