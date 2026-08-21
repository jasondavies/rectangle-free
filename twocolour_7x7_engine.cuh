#pragma once

// Scalar full-mask layout, expansion, and join engine used only by the exact
// 7x7 regression solver.  Prefix/BMMA solvers do not include this surface.

#include <future>

#include "twocolour_gpu_common.cuh"

struct PackedPair {
    uint64_t selected_offset;
    uint64_t complement_offset;
    uint32_t selected_count;
    uint32_t complement_count;
};

struct JoinDesc {
    uint64_t lhs_offset;
    uint64_t rhs_offset;
    uint32_t lhs_count;
    uint32_t rhs_count;
};

static uint64_t join_comparisons(const JoinDesc& join) {
    constexpr uint32_t swapped_flag = UINT32_C(1) << 31;
    return uint64_t(join.lhs_count & ~swapped_flag) * join.rhs_count;
}

static std::vector<uint32_t> schedule_joins_heavy_first(std::vector<JoinDesc>& joins) {
    if (joins.size() > size_t(UINT32_MAX)) {
        throw std::runtime_error("too many joins for heavy-first result slots");
    }
    std::vector<uint32_t> order(joins.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        uint64_t a_work = join_comparisons(joins[a]);
        uint64_t b_work = join_comparisons(joins[b]);
        return a_work != b_work ? a_work > b_work : a < b;
    });
    std::vector<JoinDesc> scheduled;
    scheduled.reserve(joins.size());
    for (uint32_t logical_index : order) scheduled.push_back(joins[logical_index]);
    joins.swap(scheduled);
    return order;
}

static JoinDesc make_join(uint64_t left_offset, uint64_t right_offset,
                          uint32_t left_count, uint32_t right_count) {
    constexpr uint32_t swapped_flag = UINT32_C(1) << 31;
    if (right_count < left_count) {
        return JoinDesc{right_offset, left_offset, right_count | swapped_flag, left_count};
    }
    return JoinDesc{left_offset, right_offset, left_count, right_count};
}

struct alignas(16) ExpansionDesc {
    uint64_t source_offset;
    uint64_t destination_offset;
    uint64_t row_map;
    uint32_t count;
    uint32_t reserved;
};

struct CanonicalLayout {
    std::vector<PackedPair> descriptors;
    std::vector<ExpansionDesc> expansions;
    size_t entry_count = 0;
};

struct CanonicalHostBatch {
    size_t begin;
    size_t end;
    std::vector<PrefixKey> right_keys;
    std::vector<PackedPair> right_descriptors;
    std::vector<ExpansionDesc> expansions;
    std::vector<JoinDesc> joins;
    std::vector<uint32_t> result_slots;
    size_t right_entry_count;
    U128 comparisons;
    double build_seconds;
};

static CanonicalLayout build_canonical_layout(const std::vector<PrefixKey>& keys,
                                              const CanonicalFactory& factory) {
    CanonicalLayout layout;
    layout.descriptors.resize(keys.size());
    layout.expansions.reserve(keys.size() * 2);
    for (size_t i = 0; i < keys.size(); i++) {
        const RawCanonicalPair& raw = lookup_raw(factory, keys[i]);
        PackedPair pair{};
        const CanonicalRef references[2] = {raw.selected, raw.complement};
        for (int complement = 0; complement < 2; complement++) {
            const CanonicalRef& reference = references[complement];
            const CanonicalDescriptor& source = factory.descriptors[reference.distribution];
            uint64_t destination = layout.entry_count;
            layout.expansions.push_back(ExpansionDesc{source.offset, destination,
                                                       reference.row_map, source.count, 0});
            if (complement) {
                pair.complement_offset = destination;
                pair.complement_count = source.count;
            } else {
                pair.selected_offset = destination;
                pair.selected_count = source.count;
            }
            layout.entry_count += source.count;
        }
        layout.descriptors[i] = pair;
    }
    return layout;
}

static void upload_entry_soa(const std::vector<Entry>& entries, uint64_t** device_masks,
                             uint32_t** device_weights) {
    std::vector<uint64_t> masks(entries.size());
    std::vector<uint32_t> weights(entries.size());
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)entries.size(); i++) {
        masks[size_t(i)] = entries[size_t(i)].mask;
        weights[size_t(i)] = uint32_t(entries[size_t(i)].weight);
    }
    CUDA_CHECK(cudaMalloc(device_masks, masks.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(device_weights, weights.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(*device_masks, masks.data(), masks.size() * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*device_weights, weights.data(), weights.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
}

__global__ void expand_canonical_distributions_soa(
    const uint64_t* __restrict__ canonical_masks,
    const uint32_t* __restrict__ canonical_weights,
    uint64_t* __restrict__ expanded_masks,
    uint32_t* __restrict__ expanded_weights,
    const ExpansionDesc* __restrict__ descriptions) {
    const ExpansionDesc description = descriptions[blockIdx.x];
    for (uint32_t index = threadIdx.x; index < description.count; index += blockDim.x) {
        size_t source = description.source_offset + index;
        size_t destination = description.destination_offset + index;
        expanded_masks[destination] =
            transform_pair_mask(canonical_masks[source], description.row_map);
        expanded_weights[destination] = canonical_weights[source];
    }
}

static CanonicalHostBatch build_canonical_host_batch(
    const std::vector<Edge>& edges, const std::vector<PackedPair>& left_descriptors,
    const std::unordered_map<PrefixKey, uint32_t>& left_index,
    const CanonicalFactory& right_factory, size_t begin, size_t end) {
    double start = seconds_now();
    CanonicalHostBatch batch{};
    batch.begin = begin;
    batch.end = end;
    for (size_t i = begin; i < end;) {
        batch.right_keys.push_back(edges[i].right);
        PrefixKey right = edges[i].right;
        while (i < end && edges[i].right == right) i++;
    }
    CanonicalLayout right_layout = build_canonical_layout(batch.right_keys, right_factory);
    batch.right_descriptors = std::move(right_layout.descriptors);
    batch.expansions = std::move(right_layout.expansions);
    batch.right_entry_count = right_layout.entry_count;
    batch.joins.reserve((end - begin) * 2);
    size_t edge_index = begin;
    for (size_t group = 0; group < batch.right_keys.size(); group++) {
        const PackedPair& right = batch.right_descriptors[group];
        while (edge_index < end && edges[edge_index].right == batch.right_keys[group]) {
            const PackedPair& left = left_descriptors[left_index.at(edges[edge_index].left)];
            batch.joins.push_back(make_join(left.selected_offset, right.selected_offset,
                                            left.selected_count, right.selected_count));
            batch.joins.push_back(make_join(left.complement_offset, right.complement_offset,
                                            left.complement_count, right.complement_count));
            batch.comparisons += U128(left.selected_count) * right.selected_count;
            batch.comparisons += U128(left.complement_count) * right.complement_count;
            edge_index++;
        }
    }
    batch.result_slots = schedule_joins_heavy_first(batch.joins);
    batch.build_seconds = seconds_now() - start;
    return batch;
}

__global__ void fused_disjoint_joins_soa(
    const uint64_t* __restrict__ left_masks,
    const uint32_t* __restrict__ left_weights,
    const uint64_t* __restrict__ right_masks,
    const uint32_t* __restrict__ right_weights,
    const JoinDesc* __restrict__ joins,
    unsigned long long* __restrict__ results
    ) {
    __shared__ uint64_t right_mask_tile[THREADS];
    __shared__ uint64_t right_swapped_mask_tile[THREADS];
    __shared__ uint32_t right_weight_tile[THREADS];
    __shared__ unsigned long long warp_partial[THREADS / 32];
    const JoinDesc join = joins[blockIdx.x];
    constexpr uint32_t swapped_flag = UINT32_C(1) << 31;
    bool swapped = join.lhs_count & swapped_flag;
    uint32_t lhs_count = join.lhs_count & ~swapped_flag;
    const uint64_t* lhs_masks = swapped ? right_masks : left_masks;
    const uint32_t* lhs_weights = swapped ? right_weights : left_weights;
    const uint64_t* rhs_masks = swapped ? left_masks : right_masks;
    const uint32_t* rhs_weights = swapped ? left_weights : right_weights;
    unsigned long long sum = 0;
    for (uint32_t left_base = 0; left_base < lhs_count; left_base += THREADS) {
        uint32_t left_index = left_base + threadIdx.x;
        uint64_t left_mask = 0;
        uint32_t left_weight = 0;
        uint32_t left_orbit_size = 0;
        if (left_index < lhs_count) {
            left_mask = lhs_masks[join.lhs_offset + left_index];
            left_weight = lhs_weights[join.lhs_offset + left_index];
            left_orbit_size = token_plane_orbit_size(left_mask);
        }
        for (uint32_t right_base = 0; right_base < join.rhs_count; right_base += THREADS) {
            uint32_t right_index = right_base + threadIdx.x;
            if (right_index < join.rhs_count) {
                right_mask_tile[threadIdx.x] = rhs_masks[join.rhs_offset + right_index];
                right_weight_tile[threadIdx.x] = rhs_weights[join.rhs_offset + right_index];
                right_swapped_mask_tile[threadIdx.x] =
                    swap_token_planes(right_mask_tile[threadIdx.x]);
            } else {
                right_mask_tile[threadIdx.x] = 0;
                right_weight_tile[threadIdx.x] = 0;
                right_swapped_mask_tile[threadIdx.x] = 0;
            }
            __syncthreads();
            if (left_index < lhs_count) {
                uint32_t count = min(uint32_t(THREADS), join.rhs_count - right_base);
#pragma unroll 4
                for (uint32_t j = 0; j < count; j++) {
                    uint64_t right_mask = right_mask_tile[j];
                    uint64_t compatible = !(left_mask & right_mask);
                    uint64_t swapped_right = right_swapped_mask_tile[j];
                    if (right_mask != swapped_right) {
                        compatible += !(left_mask & swapped_right);
                    }
                    sum += compatible * uint64_t(left_orbit_size) *
                           uint64_t(left_weight) *
                           uint64_t(right_weight_tile[j]);
                }
            }
            __syncthreads();
        }
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        sum += __shfl_down_sync(UINT32_MAX, sum, offset);
    }
    if ((threadIdx.x & 31) == 0) warp_partial[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < THREADS / 32 ? warp_partial[threadIdx.x] : 0;
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) {
            sum += __shfl_down_sync(UINT32_MAX, sum, offset);
        }
        if (!threadIdx.x) {
            results[blockIdx.x] = sum;
        }
    }
}
