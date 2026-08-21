#pragma once

// Geometry-generic prefix/suffix algebra shared by the 7x9 and 8x8 solvers.
// The coordinate choices and token-plane quotient representation are fixed.

#include <atomic>

#include "twocolour_gpu_common.cuh"

#include "gpu_cuda_utils.cuh"

#if (GRID_ROWS != 7 && GRID_ROWS != 8) || LEFT_COLUMNS + RIGHT_COLUMNS != GRID_COLUMNS || LEFT_COLUMNS > 5 || RIGHT_COLUMNS > 5
#error "the prefix core requires seven or eight rows and halves at most five columns"
#endif
constexpr int PREFIX_PAIR_COUNT = GRID_ROWS == 7 ? 5 : 7;
constexpr uint32_t PREFIX_TASK_CHUNK = 16;

#if GRID_ROWS == 7
using PrefixSuffix = uint32_t;
#else
using PrefixSuffix = uint64_t;
#endif

struct PrefixBucket {
    uint32_t entry_offset;
    uint32_t count;
    uint16_t prefix;
    uint16_t reserved;
};

struct PrefixDistribution {
    uint64_t direct_offset;
    uint32_t bucket_offset;
    uint32_t entry_count;
    uint32_t bucket_count;
};

struct PrefixPair {
    PrefixDistribution selected;
    PrefixDistribution complement;
};

struct PrefixJoinDesc {
    uint32_t left_bucket_offset;
    uint32_t right_bucket_offset;
    uint32_t left_bucket_count;
    uint32_t right_bucket_count;
};



static_assert(sizeof(PrefixBucket) == 12, "bucket descriptors must be compact");
static_assert(sizeof(PrefixJoinDesc) == 16, "join descriptors must be compact");

static __host__ __device__ int prefix_pair_rank(int pair) {
    int rank = 99;
#if GRID_ROWS == 8
    if (pair == 0) rank = 0;
    if (pair == 1) rank = 1;
    if (pair == 7) rank = 2;
    if (pair == 2) rank = 3;
    if (pair == 8) rank = 4;
    if (pair == 13) rank = 5;
    // BMMA-aware census: K4 plus a disjoint K2 traverses substantially fewer
    // physical buckets and weight classes than the older attached edge.
    if (pair == 27) rank = 6;
    if (pair == 3) rank = 7;
    if (pair == 14) rank = 8;
    if (pair == 18) rank = 9;
    if (pair == 9) rank = 10;
    if (pair == 24) rank = 11;
    if (pair == 26) rank = 12;
    if (pair == 25) rank = 13;
    if (pair == 23) rank = 14;
    if (pair == 22) rank = 15;
    if (pair == 4) rank = 16;
    if (pair == 10) rank = 17;
    if (pair == 15) rank = 18;
    if (pair == 19) rank = 19;
    if (pair == 5) rank = 20;
    if (pair == 11) rank = 21;
    if (pair == 16) rank = 22;
    if (pair == 20) rank = 23;
    if (pair == 6) rank = 24;
    if (pair == 12) rank = 25;
    if (pair == 17) rank = 26;
    if (pair == 21) rank = 27;
#else
    if (pair == 0) rank = 0;
    if (pair == 1) rank = 1;
    if (pair == 2) rank = 2;
    if (pair == 6) rank = 3;
    // For the 7-row BMMA join, K4 minus one edge leaves materially fewer
    // physical suffix tiles than the older K3-plus-two-spokes prefix.
    if (pair == 7) rank = 4;
    if (pair == 3) rank = 5;
    if (pair == 11) rank = 6;
    if (pair == 8) rank = 7;
#endif
    return rank;
}

static __host__ __device__ bool prefix_pair_selected(int pair) {
    return prefix_pair_rank(pair) < PREFIX_PAIR_COUNT;
}

static __host__ __device__ void split_pair_mask(uint64_t mask, uint16_t& prefix,
                                                PrefixSuffix& suffix) {
    prefix = 0;
    suffix = 0;
    int prefix_bit = 0;
    int suffix_bit = 0;
    for (int colour = 0; colour < 2; colour++) {
        for (int pair = 0; pair < PAIRS; pair++) {
            PrefixSuffix bit =
                PrefixSuffix((mask >> (colour * PAIRS + pair)) & 1U);
            if (prefix_pair_selected(pair)) {
                prefix |= uint16_t(bit << prefix_bit++);
            } else {
                suffix |= bit << suffix_bit++;
            }
        }
    }
}

static __host__ __device__ uint64_t join_pair_mask(uint16_t prefix,
                                                   PrefixSuffix suffix) {
    uint64_t mask = 0;
    int prefix_bit = 0;
    int suffix_bit = 0;
    for (int colour = 0; colour < 2; colour++) {
        for (int pair = 0; pair < PAIRS; pair++) {
            uint64_t bit = prefix_pair_selected(pair)
                               ? uint64_t((prefix >> prefix_bit++) & 1U)
                               : uint64_t((suffix >> suffix_bit++) & 1U);
            mask |= bit << (colour * PAIRS + pair);
        }
    }
    return mask;
}

static __host__ __device__ uint16_t swap_prefix_token_planes(
    uint16_t prefix) {
    constexpr uint16_t plane_mask =
        uint16_t((uint32_t(1) << PREFIX_PAIR_COUNT) - 1U);
    return uint16_t(((prefix & plane_mask) << PREFIX_PAIR_COUNT) |
                    ((prefix >> PREFIX_PAIR_COUNT) & plane_mask));
}

static __host__ __device__ PrefixSuffix swap_suffix_token_planes(
    PrefixSuffix suffix) {
    constexpr unsigned plane_bits = PAIRS - PREFIX_PAIR_COUNT;
    constexpr PrefixSuffix plane_mask =
        (PrefixSuffix(1) << plane_bits) - PrefixSuffix(1);
    return PrefixSuffix(((suffix & plane_mask) << plane_bits) |
                        ((suffix >> plane_bits) & plane_mask));
}

static __host__ __device__ uint32_t inverse_row_map(uint32_t row_map) {
    uint32_t inverse = 0;
    for (int source = 0; source < ROWS; source++) {
        int destination = int((row_map >> (4 * source)) & 15U);
        inverse |= uint32_t(source) << (4 * destination);
    }
    return inverse;
}

// transform_pair_mask(transform_pair_mask(mask, first), second) is the same
// as applying this composed map once.
static __host__ __device__ uint32_t compose_row_maps(uint32_t first,
                                                    uint32_t second) {
    uint32_t composed = 0;
    for (int source = 0; source < ROWS; source++) {
        int middle = int((first >> (4 * source)) & 15U);
        int destination = int((second >> (4 * middle)) & 15U);
        composed |= uint32_t(destination) << (4 * source);
    }
    return composed;
}

static void validate_mask_split() {
    const uint64_t full_mask = (UINT64_C(1) << (2 * PAIRS)) - 1;
    std::array<uint64_t, 2 * PAIRS + 3> masks{};
    masks[0] = 0;
    masks[1] = full_mask;
    masks[2] = UINT64_C(0x2a955555555) & full_mask;
    for (int bit = 0; bit < 2 * PAIRS; bit++)
        masks[size_t(bit) + 3] = UINT64_C(1) << bit;
    for (uint64_t mask : masks) {
        uint16_t prefix;
        PrefixSuffix suffix;
        split_pair_mask(mask, prefix, suffix);
        if (join_pair_mask(prefix, suffix) != mask) {
            throw std::runtime_error("10/32 mask join validation failed");
        }
        uint64_t swapped = swap_token_planes(mask);
        uint16_t swapped_prefix;
        PrefixSuffix swapped_suffix;
        split_pair_mask(swapped, swapped_prefix, swapped_suffix);
        if (swap_token_planes(swapped) != mask ||
            swapped_prefix != swap_prefix_token_planes(prefix) ||
            swapped_suffix != swap_suffix_token_planes(suffix)) {
            throw std::runtime_error(
                "token-plane prefix/suffix validation failed");
        }
    }
    for (int first = 0; first < 2 * PAIRS; first++) {
        for (int second = 0; second < 2 * PAIRS; second++) {
            uint16_t first_prefix, second_prefix;
            PrefixSuffix first_suffix, second_suffix;
            split_pair_mask(UINT64_C(1) << first, first_prefix, first_suffix);
            split_pair_mask(UINT64_C(1) << second, second_prefix, second_suffix);
            bool original_disjoint = first != second;
            bool split_disjoint = !(first_prefix & second_prefix) &&
                                  !(first_suffix & second_suffix);
            if (original_disjoint != split_disjoint) {
                throw std::runtime_error("10/32 mask split validation failed");
            }
        }
    }
}

static void validate_row_map_algebra() {
    std::array<uint8_t, ROWS> permutation{};
    std::iota(permutation.begin(), permutation.end(), uint8_t(0));
    const uint32_t identity = []() {
        uint32_t map = 0;
        for (int row = 0; row < ROWS; row++) map |= uint32_t(row) << (4 * row);
        return map;
    }();
    do {
        uint32_t map = 0;
        for (int row = 0; row < ROWS; row++) {
            map |= uint32_t(permutation[size_t(row)]) << (4 * row);
        }
        uint32_t inverse = inverse_row_map(map);
        if (compose_row_maps(map, inverse) != identity ||
            compose_row_maps(inverse, map) != identity) {
            throw std::runtime_error("row-map inverse validation failed");
        }
        for (int bit = 0; bit < 2 * PAIRS; bit++) {
            uint64_t mask = UINT64_C(1) << bit;
            if (transform_pair_mask(transform_pair_mask(mask, map), inverse) !=
                mask) {
                throw std::runtime_error("row-map mask round trip failed");
            }
        }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
}

constexpr uint32_t PREFIX_BUCKET_COUNT = uint32_t(1) << (2 * PREFIX_PAIR_COUNT);

static double elapsed_kernel(cudaEvent_t start, cudaEvent_t end) {
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
    return milliseconds / 1000.0;
}
