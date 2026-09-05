#pragma once

// Structural gate only: no claim that the boundary correction is free.
// A family consists of genuine residuals obtained by deleting one size-three
// defect from a common parent. All roots retain the explicitly listed core.
#include <cmath>
#include <cstdio>
#include <functional>
#include <queue>
#include <set>
#include <unordered_map>
#include "../../src/hafnian/six_by_twenty_nine_catalog.hpp"

namespace six_by_common_core {
constexpr uint64_t full = (UINT64_C(1) << 60) - 1;
inline unsigned bits(uint64_t x) { return unsigned(__builtin_popcountll(x)); }
inline uint64_t hash(uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

// Bottom-k hash sample is independent of unordered-map iteration order and
// defect coefficients. Sampling unit is a canonical residual query, not a
// colouring, labelled defect sequence, or hand-picked sibling family.
struct Sample {
    unsigned limit;
    uint64_t seed, population = 0;
    std::priority_queue<std::pair<uint64_t,uint64_t>> heap;
    void add(uint64_t occupied) {
        ++population;
        const auto item = std::make_pair(hash(occupied ^ seed), occupied);
        if (heap.size() < limit) heap.push(item);
        else if (item < heap.top()) { heap.pop(); heap.push(item); }
    }
    std::vector<uint64_t> values() const {
        auto copy = heap;
        std::vector<uint64_t> out;
        while (!copy.empty()) { out.push_back(copy.top().second); copy.pop(); }
        std::sort(out.begin(), out.end());
        return out;
    }
};

struct Child { uint64_t removed, canonical; unsigned id; };
struct Family {
    uint64_t parent = 0;
    unsigned distinct = 0;
    std::vector<Child> children;
};
struct Group {
    uint64_t parent = 0, boundary = 0;
    std::vector<Child> children;
    unsigned size() const { return unsigned(children.size()); }
};

inline Family build_family(const six_by_twenty_nine::Geometry& geometry,
        uint64_t parent, const std::vector<uint64_t>& triples) {
    Family f; f.parent = parent;
    std::map<uint64_t,unsigned> ids;
    for (uint64_t removed : triples) {
        if (removed & parent) continue;
        uint64_t key = six_by_twenty_nine::canonicalize(geometry, parent | removed);
        auto inserted = ids.emplace(key, unsigned(ids.size()));
        f.children.push_back({removed, key, inserted.first->second});
    }
    f.distinct = unsigned(ids.size());
    return f;
}

inline unsigned contained(const Family& f, uint64_t boundary) {
    // At most 440 size-three supports in the six-row universe.
    bool seen[440] = {};
    unsigned count = 0;
    for (const Child& c : f.children) {
        if (!(c.removed & ~boundary) && !seen[c.id]) {
            seen[c.id] = true; ++count;
        }
    }
    return count;
}

inline Group grow(const Family& f, uint64_t anchor, unsigned cap) {
    if (cap < 3 || !(cap & 1) || (anchor & f.parent) || bits(anchor) != 3)
        throw std::runtime_error("invalid common-core anchor/cap");
    uint64_t boundary = anchor;
    unsigned count = contained(f, boundary);
    if (!count) throw std::runtime_error("anchor absent from family");
    for (;;) {
        uint64_t best = boundary;
        unsigned best_count = count, best_added = 1;
        for (const Child& c : f.children) {
            uint64_t next = boundary | c.removed;
            if (next == boundary || bits(next) > cap) continue;
            // Parent augmented order is odd; an odd boundary leaves an even
            // common core, so both core and per-root boundary can be paired.
            if (!(bits(next) & 1)) {
                uint64_t spare = full & ~(f.parent | next);
                if (!spare) continue;
                next |= spare & -spare;
            }
            if (bits(next) > cap) continue;
            unsigned n = contained(f, next), added = bits(next ^ boundary);
            if (n > count &&
                ((n-count)*best_added > (best_count-count)*added ||
                 ((n-count)*best_added == (best_count-count)*added && next < best))) {
                best = next; best_count = n; best_added = added;
            }
        }
        if (best == boundary) break;
        boundary = best; count = best_count;
    }
    Group g; g.parent = f.parent; g.boundary = boundary;
    bool seen[440] = {};
    for (const Child& c : f.children)
        if (!(c.removed & ~boundary) && !seen[c.id]) {
            seen[c.id] = true; g.children.push_back(c);
        }
    if (g.size() != count) throw std::runtime_error("group count mismatch");
    return g;
}

inline unsigned core_order(const Group& g, unsigned unmatched) {
    return 60 - bits(g.parent | g.boundary) + unmatched;
}

inline void validate(const Family& f, const Group& g, uint64_t root,
                     unsigned order, unsigned unmatched) {
    const uint64_t core = full & ~(g.parent | g.boundary);
    bool found = false;
    std::set<uint64_t> keys;
    if ((g.parent & g.boundary) || (core_order(g, unmatched) & 1))
        throw std::runtime_error("invalid common core");
    for (const Child& c : g.children) {
        if ((c.removed & ~g.boundary) || (c.removed & g.parent) ||
            (core & (g.parent | c.removed)) || bits(c.removed) != 3 ||
            60 - bits(g.parent | c.removed) + unmatched != order ||
            !keys.insert(c.canonical).second)
            throw std::runtime_error("invalid/doubled family member");
        found |= c.canonical == root;
        if (std::none_of(f.children.begin(), f.children.end(), [&](const Child& v) {
            return v.removed == c.removed && v.canonical == c.canonical;
        })) throw std::runtime_error("non-family group member");
    }
    if (!found) throw std::runtime_error("group lost sampled root");
    // Pair sorted core tokens (including fixed dummy IDs) first, then sorted
    // boundary tokens. Equal even core order proves a common reference X.
    if ((order - core_order(g, unmatched)) & 1)
        throw std::runtime_error("odd boundary order");
}

template<class ParentExists>
void census(const six_by_twenty_nine::Geometry& geometry,
        const Sample& sample, unsigned excess, unsigned defects, unsigned slack,
        unsigned max_parents, ParentExists parent_exists) {
    std::vector<uint64_t> triples;
    for (const auto& s : six_by_twenty_nine::weighted_supports(geometry))
        if (s.excess == 1) triples.push_back(s.mask);
    std::sort(triples.begin(), triples.end());
    if (triples.size() != 440) throw std::runtime_error("triple census mismatch");
    const unsigned unmatched = 2*slack-excess;
    const unsigned order = 60-2*defects-excess+unmatched;
    std::unordered_map<uint64_t,Family> cache;
    std::array<unsigned,4> caps{5,7,9,11};
    std::array<std::vector<unsigned>,4> sizes;
    std::array<double,4> rebuild_ratio_sum{};
    unsigned missing = 0, examined_parents = 0;
    for (uint64_t root : sample.values()) {
        std::set<uint64_t> parents;
        for (uint64_t s : triples) if (!(s & ~root)) {
            const uint64_t parent = six_by_twenty_nine::canonicalize(geometry, root ^ s);
            if (parent_exists(parent)) parents.insert(parent);
        }
        std::vector<uint64_t> choices(parents.begin(), parents.end());
        std::sort(choices.begin(), choices.end(), [&](uint64_t a,uint64_t b) {
            return std::make_pair(hash(a ^ sample.seed),a) <
                   std::make_pair(hash(b ^ sample.seed),b);
        });
        if (choices.size() > max_parents) choices.resize(max_parents);
        if (choices.empty()) ++missing;
        std::array<Group,4> best;
        for (uint64_t parent : choices) {
            ++examined_parents;
            auto it = cache.find(parent);
            if (it == cache.end()) it = cache.emplace(parent,
                build_family(geometry,parent,triples)).first;
            const Family& f = it->second;
            for (const Child& anchor : f.children) if (anchor.canonical == root)
                for (unsigned k = 0; k < caps.size(); ++k) {
                    Group g = grow(f, anchor.removed, caps[k]);
                    validate(f,g,root,order,unmatched);
                    // Prefer more distinct real queries, then a larger core.
                    if (g.size() > best[k].size() ||
                        (g.size() == best[k].size() && bits(g.boundary) < bits(best[k].boundary)))
                        best[k] = std::move(g);
                }
        }
        for (unsigned k = 0; k < caps.size(); ++k) {
            const Group& g = best[k];
            sizes[k].push_back(std::max(1u,g.size()));
            if (!g.size()) { rebuild_ratio_sum[k] += 1; continue; }
            const unsigned core = core_order(g,unmatched), boundary = order-core;
            // Pure count of common-core sign assignments versus an existing
            // eight-term rebuild cadence. Excludes all boundary/moment costs!
            double ratio = 8.0 / (g.size()*std::ldexp(1.0,int(boundary/2)));
            rebuild_ratio_sum[k] += std::min(1.0,ratio);
            std::printf("CORE27_GROUP e=%u d=%u root=%llu cap=%u parent=%llu boundary=%llu"
                        " core=%u tail=%u queries=%u rebuild_ratio8=%.9f members=",
                excess,defects,(unsigned long long)root,caps[k],
                (unsigned long long)g.parent,(unsigned long long)g.boundary,
                core,boundary,g.size(),ratio);
            for (const Child& c : g.children)
                std::printf("%llu:%llu,",(unsigned long long)c.canonical,
                            (unsigned long long)c.removed);
            std::printf("\n");
        }
    }
    for (unsigned k=0;k<caps.size();++k) {
        auto v=sizes[k]; std::sort(v.begin(),v.end());
        if(v.empty())continue;
        uint64_t sum=0;unsigned multiple=0;
        for(unsigned n:v){sum+=n;multiple+=n>1;}
        std::printf("CORE27_SUMMARY e=%u d=%u order=%u population=%llu sampled=%zu"
                    " missing=%u cap=%u mean_group=%.6f median=%u p90=%u max=%u"
                    " multi_fraction=%.6f optimistic_rebuild_ratio8=%.9f"
                    " parents_examined=%u unique_parents=%zu exact_structure=OK\n",
            excess,defects,order,(unsigned long long)sample.population,v.size(),missing,caps[k],
            double(sum)/v.size(),v[v.size()/2],v[(v.size()-1)*9/10],v.back(),
            double(multiple)/v.size(),rebuild_ratio_sum[k]/v.size(),examined_parents,cache.size());
    }
    std::fflush(stdout);
}
} // namespace six_by_common_core
