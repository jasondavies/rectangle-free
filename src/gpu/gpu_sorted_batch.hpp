#pragma once

#include <cstddef>
#include <stdexcept>

namespace gpu_batch {
// Both drivers preserve right-key order when assigning sorted edges to a
// batch. Merge those edges with the batch's sorted, unique right dictionary.
// Repeated edges for one right and unused dictionary entries are intentional.
template<class Edges, class Indices, class Keys, class Visitor>
void visit_sorted_rights(const Edges& edges, const Indices& indices,
                         const Keys& keys, Visitor&& visit) {
    size_t right = 0;
    for (size_t local = 0; local < indices.size(); ++local) {
        const auto index = indices[local];
        const auto& edge = edges.at(index);
        while (right < keys.size() && keys[right] < edge.right) ++right;
        if (right == keys.size() || keys[right] != edge.right)
            throw std::runtime_error("packed batch right ownership/order mismatch");
        visit(local, index, right);
    }
}
} // namespace gpu_batch
