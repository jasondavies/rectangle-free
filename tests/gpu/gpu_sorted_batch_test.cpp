#include "../../src/gpu/gpu_sorted_batch.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

struct Edge { uint64_t right; };
int main() {
    std::mt19937 random(494);
    for (unsigned trial = 0; trial < 1000; ++trial) {
        std::vector<Edge> edges;
        std::vector<uint64_t> keys;
        for (unsigned k = 0; k < 200; ++k) {
            if (random() % 3 == 0) continue;
            keys.push_back(k);
            for (unsigned n = random() % 5; n; --n) edges.push_back({k});
        }
        // Subsets model batch assignment and preserve the original edge order.
        std::vector<uint32_t> indices;
        for (uint32_t i = 0; i < edges.size(); ++i)
            if (random() % 2) indices.push_back(i);
        size_t visits = 0;
        gpu_batch::visit_sorted_rights(edges, indices, keys,
            [&](size_t local, uint32_t edge, size_t right) {
                assert(local == visits++ && indices[local] == edge);
                assert(right == size_t(std::lower_bound(keys.begin(), keys.end(),
                                                       edges[edge].right) - keys.begin()));
            });
        assert(visits == indices.size());
    }
    const std::vector<Edge> edges{{1}, {2}, {2}, {5}};
    for (auto indices : {std::vector<uint32_t>{3, 0}, {0, 4}, {0, 1, 3}}) {
        bool rejected = false;
        try { gpu_batch::visit_sorted_rights(edges, indices,
            indices==std::vector<uint32_t>{3,0} ? std::vector<uint64_t>{1,2,5} :
                std::vector<uint64_t>{1, 2}, [](size_t, uint32_t, size_t) {}); }
        catch (const std::exception&) { rejected = true; }
        assert(rejected);
    }
    std::cout << "SORTED_BATCH_TEST exact=OK random_batches=1000\n";
}
