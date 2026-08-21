#define PREFIX_PRODUCTION_NO_MAIN
#include "../../src/gpu/twocolour_8x8_prefix_solve.cu"

#include <array>
#include <climits>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

struct PtxFragmentA {
    uint32_t bits0;
    uint32_t bits1;
    uint32_t weight0;
    uint32_t weight1;
    bool valid0;
    bool valid1;
};

struct PtxFragmentB {
    uint32_t bits;
    uint32_t weight0;
    uint32_t weight1;
    bool valid0;
    bool valid1;
};

static __device__ __forceinline__ uint32_t suffix_word(uint64_t suffix,
                                                       unsigned word) {
    return word == 0 ? uint32_t(suffix)
                     : word == 1 ? uint32_t(suffix >> 32) : 0;
}

// Native PTX fragment ownership is documented for m16n8k128 as follows:
//   group = lane / 4, word = lane % 4
//   A registers: rows group and group + 8, K word "word"
//   B register: column group, K word "word"
//   D registers: (group, 2*word), (group, 2*word+1),
//                (group+8, 2*word), (group+8, 2*word+1).
static __device__ __forceinline__ void inline_bmma_16x8(
    uint32_t a0, uint32_t a1, uint32_t b, uint32_t& d0, uint32_t& d1,
    uint32_t& d2, uint32_t& d3) {
    uint32_t zero = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(b), "r"(zero), "r"(zero),
          "r"(zero), "r"(zero));
}

template <bool SHUFFLE>
static __device__ __forceinline__ PtxFragmentA load_ptx_a(
    const PrefixEntry* __restrict__ entries, uint32_t offset,
    uint32_t count, uint32_t base, unsigned lane) {
    const unsigned group = lane >> 2;
    const unsigned word = lane & 3U;
    PtxFragmentA fragment{};
    if constexpr (!SHUFFLE) {
        const uint32_t row0 = group;
        const uint32_t row1 = group + 8;
        fragment.valid0 = row0 < count;
        fragment.valid1 = row1 < count;
        uint64_t suffix0 = 0;
        uint64_t suffix1 = 0;
        if (fragment.valid0) {
            const PrefixEntry& entry = entries[offset + base + row0];
            suffix0 = entry.suffix;
            fragment.weight0 = entry.weight;
        }
        if (fragment.valid1) {
            const PrefixEntry& entry = entries[offset + base + row1];
            suffix1 = entry.suffix;
            fragment.weight1 = entry.weight;
        }
        fragment.bits0 = suffix_word(suffix0, word);
        fragment.bits1 = suffix_word(suffix1, word);
        return fragment;
    }

    const uint32_t loaded_row = lane;
    bool loaded_valid = loaded_row < 16 && loaded_row < count;
    uint64_t loaded_suffix = 0;
    uint32_t loaded_weight = 0;
    if (loaded_valid) {
        const PrefixEntry& entry = entries[offset + base + loaded_row];
        loaded_suffix = entry.suffix;
        loaded_weight = entry.weight;
    }
    uint32_t loaded_low = uint32_t(loaded_suffix);
    uint32_t loaded_high = uint32_t(loaded_suffix >> 32);
    uint32_t low0 = __shfl_sync(UINT32_MAX, loaded_low, group);
    uint32_t high0 = __shfl_sync(UINT32_MAX, loaded_high, group);
    uint32_t low1 = __shfl_sync(UINT32_MAX, loaded_low, group + 8);
    uint32_t high1 = __shfl_sync(UINT32_MAX, loaded_high, group + 8);
    fragment.bits0 = word == 0 ? low0 : word == 1 ? high0 : 0;
    fragment.bits1 = word == 0 ? low1 : word == 1 ? high1 : 0;
    fragment.weight0 =
        __shfl_sync(UINT32_MAX, loaded_weight, group);
    fragment.weight1 =
        __shfl_sync(UINT32_MAX, loaded_weight, group + 8);
    fragment.valid0 = group < count;
    fragment.valid1 = group + 8 < count;
    return fragment;
}

template <bool SHUFFLE>
static __device__ __forceinline__ PtxFragmentB load_ptx_b(
    const PrefixEntry* __restrict__ entries, uint32_t offset,
    uint32_t count, uint32_t base, unsigned lane) {
    const unsigned group = lane >> 2;
    const unsigned word = lane & 3U;
    const uint32_t output0 = 2 * word;
    const uint32_t output1 = output0 + 1;
    PtxFragmentB fragment{};
    if constexpr (!SHUFFLE) {
        uint64_t input_suffix = 0;
        if (group < count) {
            input_suffix = entries[offset + base + group].suffix;
        }
        fragment.bits = suffix_word(input_suffix, word);
        fragment.valid0 = output0 < count;
        fragment.valid1 = output1 < count;
        if (fragment.valid0) {
            fragment.weight0 = entries[offset + base + output0].weight;
        }
        if (fragment.valid1) {
            fragment.weight1 = entries[offset + base + output1].weight;
        }
        return fragment;
    }

    const uint32_t loaded_column = lane;
    bool loaded_valid = loaded_column < 8 && loaded_column < count;
    uint64_t loaded_suffix = 0;
    uint32_t loaded_weight = 0;
    if (loaded_valid) {
        const PrefixEntry& entry = entries[offset + base + loaded_column];
        loaded_suffix = entry.suffix;
        loaded_weight = entry.weight;
    }
    uint32_t loaded_low = uint32_t(loaded_suffix);
    uint32_t loaded_high = uint32_t(loaded_suffix >> 32);
    uint32_t low = __shfl_sync(UINT32_MAX, loaded_low, group);
    uint32_t high = __shfl_sync(UINT32_MAX, loaded_high, group);
    fragment.bits = word == 0 ? low : word == 1 ? high : 0;
    fragment.weight0 =
        __shfl_sync(UINT32_MAX, loaded_weight, output0);
    fragment.weight1 =
        __shfl_sync(UINT32_MAX, loaded_weight, output1);
    fragment.valid0 = output0 < count;
    fragment.valid1 = output1 < count;
    return fragment;
}

template <bool SHUFFLE, bool WEIGHTED, bool UNIT_FAST = false>
static __device__ __forceinline__ unsigned long long ptx_bucket_join(
    const PrefixEntry* __restrict__ left_entries, PrefixBucket left,
    const PrefixEntry* __restrict__ right_entries, PrefixBucket right,
    unsigned lane) {
    // A is retained across B tiles.  Pick the orientation with fewer padded
    // 16x8 tiles; break ties by retaining the smaller side.
    uint64_t forward_tiles = uint64_t((left.count + 15) / 16) *
                             ((right.count + 7) / 8);
    uint64_t reverse_tiles = uint64_t((right.count + 15) / 16) *
                             ((left.count + 7) / 8);
    if (reverse_tiles < forward_tiles ||
        (reverse_tiles == forward_tiles && right.count < left.count)) {
        const PrefixEntry* temporary_entries = left_entries;
        left_entries = right_entries;
        right_entries = temporary_entries;
        PrefixBucket temporary_bucket = left;
        left = right;
        right = temporary_bucket;
    }

    unsigned long long sum = 0;
    for (uint32_t left_base = 0; left_base < left.count; left_base += 16) {
        uint32_t left_count = min(uint32_t(16), left.count - left_base);
        PtxFragmentA a = load_ptx_a<SHUFFLE>(
            left_entries, left.entry_offset, left_count, left_base, lane);
        for (uint32_t right_base = 0; right_base < right.count;
             right_base += 8) {
            uint32_t right_count =
                min(uint32_t(8), right.count - right_base);
            PtxFragmentB b = load_ptx_b<SHUFFLE>(
                right_entries, right.entry_offset, right_count, right_base,
                lane);
            uint32_t d0, d1, d2, d3;
            inline_bmma_16x8(a.bits0, a.bits1, b.bits, d0, d1, d2, d3);
            if constexpr (WEIGHTED) {
                bool compatible0 = a.valid0 && b.valid0 && d0 == 0;
                bool compatible1 = a.valid0 && b.valid1 && d1 == 0;
                bool compatible2 = a.valid1 && b.valid0 && d2 == 0;
                bool compatible3 = a.valid1 && b.valid1 && d3 == 0;
                bool all_one = false;
                if constexpr (UNIT_FAST) {
                    bool lane_all_one =
                        (!a.valid0 || a.weight0 == 1) &&
                        (!a.valid1 || a.weight1 == 1) &&
                        (!b.valid0 || b.weight0 == 1) &&
                        (!b.valid1 || b.weight1 == 1);
                    all_one = __all_sync(UINT32_MAX, lane_all_one);
                }
                if (all_one) {
                    sum += compatible0;
                    sum += compatible1;
                    sum += compatible2;
                    sum += compatible3;
                } else {
                    if (compatible0) {
                        sum += uint64_t(a.weight0) * uint64_t(b.weight0);
                    }
                    if (compatible1) {
                        sum += uint64_t(a.weight0) * uint64_t(b.weight1);
                    }
                    if (compatible2) {
                        sum += uint64_t(a.weight1) * uint64_t(b.weight0);
                    }
                    if (compatible3) {
                        sum += uint64_t(a.weight1) * uint64_t(b.weight1);
                    }
                }
            } else {
                sum += a.valid0 && b.valid0 && d0 == 0;
                sum += a.valid0 && b.valid1 && d1 == 0;
                sum += a.valid1 && b.valid0 && d2 == 0;
                sum += a.valid1 && b.valid1 && d3 == 0;
            }
        }
    }
    return sum;
}

// BMMA supplies the complete 16x8 compatibility predicate, but the weights
// form an outer product.  Accumulate the compatible right-weight sum for
// each retained A row across every B tile, then multiply by that row's left
// weight once:
//
//   sum_ij compatible(i,j) * a_i * b_j
//     = sum_i a_i * (sum_j compatible(i,j) * b_j).
//
// Native m16n8k128 output ownership places one row's eight columns in a
// consecutive four-lane subgroup, with two columns per lane.  The subgroup
// reduction therefore needs no shared-memory compatibility matrix.
template <bool SHUFFLE>
static __device__ __forceinline__ unsigned long long
ptx_bucket_join_row_factorized(
    const PrefixEntry* __restrict__ left_entries, PrefixBucket left,
    const PrefixEntry* __restrict__ right_entries, PrefixBucket right,
    unsigned lane) {
    uint64_t forward_tiles = uint64_t((left.count + 15) / 16) *
                             ((right.count + 7) / 8);
    uint64_t reverse_tiles = uint64_t((right.count + 15) / 16) *
                             ((left.count + 7) / 8);
    if (reverse_tiles < forward_tiles ||
        (reverse_tiles == forward_tiles && right.count < left.count)) {
        const PrefixEntry* temporary_entries = left_entries;
        left_entries = right_entries;
        right_entries = temporary_entries;
        PrefixBucket temporary_bucket = left;
        left = right;
        right = temporary_bucket;
    }

    const unsigned word = lane & 3U;
    unsigned long long sum = 0;
    for (uint32_t left_base = 0; left_base < left.count; left_base += 16) {
        uint32_t left_count = min(uint32_t(16), left.count - left_base);
        PtxFragmentA a = load_ptx_a<SHUFFLE>(
            left_entries, left.entry_offset, left_count, left_base, lane);
        // A bucket is a subset of one checked half-distribution, whose total
        // assignment weight is strictly below 2^32.  Therefore each row sum
        // is exactly representable in uint32_t; the final product is still
        // promoted to uint64_t.
        uint32_t row_sum0 = 0;
        uint32_t row_sum1 = 0;
        for (uint32_t right_base = 0; right_base < right.count;
             right_base += 8) {
            uint32_t right_count =
                min(uint32_t(8), right.count - right_base);
            PtxFragmentB b = load_ptx_b<SHUFFLE>(
                right_entries, right.entry_offset, right_count, right_base,
                lane);
            uint32_t d0, d1, d2, d3;
            inline_bmma_16x8(a.bits0, a.bits1, b.bits, d0, d1, d2, d3);

            uint32_t partial0 =
                (a.valid0 && b.valid0 && d0 == 0
                     ? b.weight0
                     : uint32_t(0)) +
                (a.valid0 && b.valid1 && d1 == 0
                     ? b.weight1
                     : uint32_t(0));
            uint32_t partial1 =
                (a.valid1 && b.valid0 && d2 == 0
                     ? b.weight0
                     : uint32_t(0)) +
                (a.valid1 && b.valid1 && d3 == 0
                     ? b.weight1
                     : uint32_t(0));
            partial0 += __shfl_down_sync(UINT32_MAX, partial0, 2, 4);
            partial1 += __shfl_down_sync(UINT32_MAX, partial1, 2, 4);
            partial0 += __shfl_down_sync(UINT32_MAX, partial0, 1, 4);
            partial1 += __shfl_down_sync(UINT32_MAX, partial1, 1, 4);
            if (!word) {
                row_sum0 += partial0;
                row_sum1 += partial1;
            }
        }
        if (!word) {
            if (a.valid0)
                sum += uint64_t(a.weight0) * row_sum0;
            if (a.valid1)
                sum += uint64_t(a.weight1) * row_sum1;
        }
    }
    return sum;
}

static __device__ __forceinline__ unsigned long long scalar_bucket_join(
    const PrefixEntry* __restrict__ left_entries, PrefixBucket left,
    const PrefixEntry* __restrict__ right_entries, PrefixBucket right,
    PrefixEntry* __restrict__ warp_tile, unsigned lane) {
    if (right.count < left.count) {
        const PrefixEntry* temporary_entries = left_entries;
        left_entries = right_entries;
        right_entries = temporary_entries;
        PrefixBucket temporary_bucket = left;
        left = right;
        right = temporary_bucket;
    }
    unsigned long long sum = 0;
    for (uint32_t left_base = 0; left_base < left.count; left_base += 32) {
        uint32_t left_index = left_base + lane;
        bool left_valid = left_index < left.count;
        PrefixEntry left_entry = left_valid
            ? left_entries[left.entry_offset + left_index]
            : PrefixEntry{0, 0};
        for (uint32_t right_base = 0; right_base < right.count;
             right_base += 32) {
            uint32_t right_index = right_base + lane;
            warp_tile[lane] = right_index < right.count
                ? right_entries[right.entry_offset + right_index]
                : PrefixEntry{0, 0};
            __syncwarp();
            uint32_t count = min(uint32_t(32), right.count - right_base);
#pragma unroll
            for (uint32_t offset = 0; offset < count; offset++) {
                PrefixEntry right_entry = warp_tile[offset];
                if (left_valid &&
                    !(left_entry.suffix & right_entry.suffix)) {
                    sum += uint64_t(left_entry.weight) *
                           uint64_t(right_entry.weight);
                }
            }
            __syncwarp();
        }
    }
    return sum;
}

template <bool SHUFFLE, bool WEIGHTED, bool HYBRID,
          bool UNIT_FAST = false, bool ROW_FACTORIZED = false>
__global__ void inline_ptx_prefix_joins(
    const PrefixEntry* __restrict__ left_entries,
    const PrefixEntry* __restrict__ right_entries,
    const PrefixBucket* __restrict__ left_buckets,
    const PrefixBucket* __restrict__ right_buckets,
    const PrefixJoinDesc* __restrict__ joins,
    unsigned long long* __restrict__ results, uint64_t ptx_min_work) {
    __shared__ unsigned long long warp_partial[8];
    __shared__ PrefixEntry warp_tiles[THREADS];
    __shared__ uint32_t next_task;
    const PrefixJoinDesc join = joins[blockIdx.x];
    const unsigned lane = threadIdx.x & 31U;
    const unsigned warp = threadIdx.x >> 5;
    const uint32_t task_count =
        join.left_bucket_count * join.right_bucket_count;
    unsigned long long sum = 0;
    if (!threadIdx.x) next_task = 0;
    __syncthreads();

    uint32_t task_base = 0;
    if (!lane) task_base = atomicAdd(&next_task, uint32_t(PREFIX_TASK_CHUNK));
    task_base = __shfl_sync(UINT32_MAX, task_base, 0);
    while (task_base < task_count) {
        uint32_t task_end =
            min(task_count, task_base + uint32_t(PREFIX_TASK_CHUNK));
        uint32_t left_bucket_index = task_base / join.right_bucket_count;
        uint32_t right_bucket_index =
            task_base - left_bucket_index * join.right_bucket_count;
        uint32_t cached_left_bucket_index = UINT32_MAX;
        PrefixBucket left_bucket{};
        for (uint32_t task = task_base; task < task_end; task++) {
            if (left_bucket_index != cached_left_bucket_index) {
                left_bucket = left_buckets[
                    join.left_bucket_offset + left_bucket_index];
                cached_left_bucket_index = left_bucket_index;
            }
            PrefixBucket right_bucket = right_buckets[
                join.right_bucket_offset + right_bucket_index];
            if (!(left_bucket.prefix & right_bucket.prefix)) {
                uint64_t work =
                    uint64_t(left_bucket.count) * right_bucket.count;
                if constexpr (HYBRID) {
                    if (work < ptx_min_work) {
                        sum += scalar_bucket_join(
                            left_entries, left_bucket, right_entries,
                            right_bucket, &warp_tiles[warp * 32], lane);
                    } else {
                        sum += ptx_bucket_join<SHUFFLE, true, UNIT_FAST>(
                            left_entries, left_bucket, right_entries,
                            right_bucket, lane);
                    }
                } else {
                    if constexpr (ROW_FACTORIZED) {
                        sum += ptx_bucket_join_row_factorized<SHUFFLE>(
                            left_entries, left_bucket, right_entries,
                            right_bucket, lane);
                    } else {
                        sum += ptx_bucket_join<SHUFFLE, WEIGHTED, UNIT_FAST>(
                            left_entries, left_bucket, right_entries,
                            right_bucket, lane);
                    }
                }
            }
            right_bucket_index++;
            if (right_bucket_index == join.right_bucket_count) {
                right_bucket_index = 0;
                left_bucket_index++;
            }
        }
        if (!lane) {
            task_base =
                atomicAdd(&next_task, uint32_t(PREFIX_TASK_CHUNK));
        }
        task_base = __shfl_sync(UINT32_MAX, task_base, 0);
    }

#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        sum += __shfl_down_sync(UINT32_MAX, sum, offset);
    }
    if (!lane) warp_partial[warp] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = lane < 8 ? warp_partial[lane] : 0;
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) {
            sum += __shfl_down_sync(UINT32_MAX, sum, offset);
        }
        if (!lane) results[blockIdx.x] = sum;
    }
}

__global__ void count_unit_weights(const PrefixEntry* entries, size_t count,
                                   unsigned long long* result) {
    __shared__ unsigned long long warp_sums[8];
    unsigned long long local = 0;
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += size_t(blockDim.x) * gridDim.x) {
        local += entries[index].weight == 1;
    }
    unsigned lane = threadIdx.x & 31U;
    unsigned warp = threadIdx.x >> 5;
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        local += __shfl_down_sync(UINT32_MAX, local, offset);
    }
    if (!lane) warp_sums[warp] = local;
    __syncthreads();
    if (threadIdx.x < 32) {
        local = lane < 8 ? warp_sums[lane] : 0;
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) {
            local += __shfl_down_sync(UINT32_MAX, local, offset);
        }
        if (!lane) atomicAdd(result, local);
    }
}

static uint64_t device_unit_weight_count(const PrefixEntry* entries,
                                         size_t count) {
    unsigned long long* device_result = nullptr;
    CUDA_CHECK(cudaMalloc(&device_result, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(device_result, 0, sizeof(unsigned long long)));
    count_unit_weights<<<4096, THREADS>>>(entries, count, device_result);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long result = 0;
    CUDA_CHECK(cudaMemcpy(&result, device_result, sizeof(result),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_result));
    return result;
}

struct DeviceNonunitLayout {
    PrefixEntry* entries = nullptr;
    PrefixBucket* buckets = nullptr;
    size_t entry_count = 0;
    size_t bucket_count = 0;
    double build_seconds = 0;
};

// Exact weight-class representation.  Every original prefix bucket is
// partitioned into contiguous equal-weight runs.  The original bucket order
// and prefix are retained, so existing join descriptors remain valid.
struct WeightClassMeta {
    uint32_t entry_offset;
    uint32_t count;
    uint32_t weight;
    uint32_t reserved;
};
static_assert(sizeof(WeightClassMeta) == 16,
              "weight-class metadata must remain aligned");

struct DeviceWeightClassLayout {
    PrefixEntry* entries = nullptr;
    PrefixBucket* buckets = nullptr;
    WeightClassMeta* classes = nullptr;
    size_t entry_count = 0;
    size_t bucket_count = 0;
    size_t class_count = 0;
    size_t maximum_classes = 0;
    double build_seconds = 0;
};

constexpr unsigned WEIGHT_CLASS_HASH_SLOTS = 32;

static __device__ __forceinline__ unsigned weight_class_hash_slot(
    uint32_t weight) {
    return (weight * UINT32_C(2654435761)) &
           (WEIGHT_CLASS_HASH_SLOTS - 1);
}

static __device__ __forceinline__ unsigned weight_class_find_or_insert(
    uint32_t* weights, uint32_t weight) {
    unsigned slot = weight_class_hash_slot(weight);
#pragma unroll
    for (unsigned probe = 0; probe < WEIGHT_CLASS_HASH_SLOTS; probe++) {
        uint32_t previous = atomicCAS(&weights[slot], 0U, weight);
        if (!previous || previous == weight) return slot;
        slot = (slot + 1) & (WEIGHT_CLASS_HASH_SLOTS - 1);
    }
    return UINT32_MAX;
}

__global__ void count_weight_classes_per_bucket(
    const PrefixEntry* __restrict__ entries,
    const PrefixBucket* __restrict__ buckets, size_t bucket_count,
    uint32_t* __restrict__ counts, uint32_t* __restrict__ overflow) {
    __shared__ uint32_t weights[8 * WEIGHT_CLASS_HASH_SLOTS];
    unsigned warp = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 31U;
    uint32_t* warp_weights =
        &weights[warp * WEIGHT_CLASS_HASH_SLOTS];
    warp_weights[lane] = 0;
    __syncwarp();
    size_t bucket_index = size_t(blockIdx.x) * 8 + warp;
    if (bucket_index >= bucket_count) return;
    PrefixBucket bucket = buckets[bucket_index];
    for (uint32_t base = 0; base < bucket.count; base += 32) {
        uint32_t index = base + lane;
        if (index < bucket.count) {
            uint32_t weight =
                entries[bucket.entry_offset + index].weight;
            unsigned slot =
                weight_class_find_or_insert(warp_weights, weight);
            if (slot == UINT32_MAX) atomicExch(overflow, 1U);
        }
    }
    __syncwarp();
    unsigned occupied = __ballot_sync(
        UINT32_MAX, warp_weights[lane] != 0);
    if (!lane) counts[bucket_index] = __popc(occupied);
}

__global__ void group_weight_classes_per_bucket(
    const PrefixEntry* __restrict__ entries,
    const PrefixBucket* __restrict__ buckets, size_t bucket_count,
    const uint32_t* __restrict__ class_offsets,
    PrefixEntry* __restrict__ grouped_entries,
    PrefixBucket* __restrict__ class_buckets,
    WeightClassMeta* __restrict__ classes,
    uint32_t* __restrict__ overflow) {
    __shared__ uint32_t weights[8 * WEIGHT_CLASS_HASH_SLOTS];
    __shared__ uint32_t counts[8 * WEIGHT_CLASS_HASH_SLOTS];
    __shared__ uint32_t positions[8 * WEIGHT_CLASS_HASH_SLOTS];
    unsigned warp = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 31U;
    uint32_t* warp_weights =
        &weights[warp * WEIGHT_CLASS_HASH_SLOTS];
    uint32_t* warp_counts = &counts[warp * WEIGHT_CLASS_HASH_SLOTS];
    uint32_t* warp_positions =
        &positions[warp * WEIGHT_CLASS_HASH_SLOTS];
    warp_weights[lane] = 0;
    warp_counts[lane] = 0;
    __syncwarp();
    size_t bucket_index = size_t(blockIdx.x) * 8 + warp;
    if (bucket_index >= bucket_count) return;
    PrefixBucket bucket = buckets[bucket_index];
    for (uint32_t base = 0; base < bucket.count; base += 32) {
        uint32_t index = base + lane;
        if (index < bucket.count) {
            PrefixEntry entry = entries[bucket.entry_offset + index];
            unsigned slot =
                weight_class_find_or_insert(warp_weights, entry.weight);
            if (slot == UINT32_MAX) {
                atomicExch(overflow, 1U);
            } else {
                atomicAdd(&warp_counts[slot], 1U);
            }
        }
    }
    __syncwarp();
    unsigned occupied = __ballot_sync(
        UINT32_MAX, warp_weights[lane] != 0);
    uint32_t count = warp_counts[lane];
    uint32_t inclusive = count;
#pragma unroll
    for (unsigned offset = 1; offset < 32; offset <<= 1) {
        uint32_t previous = __shfl_up_sync(UINT32_MAX, inclusive, offset);
        if (lane >= offset) inclusive += previous;
    }
    uint32_t local_base = inclusive - count;
    warp_positions[lane] = local_base;
    if (warp_weights[lane]) {
        unsigned ordinal = __popc(
            occupied & ((UINT32_C(1) << lane) - 1U));
        classes[class_offsets[bucket_index] + ordinal] = WeightClassMeta{
            bucket.entry_offset + local_base, count, warp_weights[lane], 0};
    }
    if (!lane) {
        class_buckets[bucket_index] = PrefixBucket{
            class_offsets[bucket_index], uint32_t(__popc(occupied)),
            bucket.prefix, 0};
    }
    __syncwarp();
    for (uint32_t base = 0; base < bucket.count; base += 32) {
        uint32_t index = base + lane;
        if (index < bucket.count) {
            PrefixEntry entry = entries[bucket.entry_offset + index];
            unsigned slot = weight_class_hash_slot(entry.weight);
#pragma unroll
            for (unsigned probe = 0; probe < WEIGHT_CLASS_HASH_SLOTS;
                 probe++) {
                if (warp_weights[slot] == entry.weight) break;
                slot = (slot + 1) & (WEIGHT_CLASS_HASH_SLOTS - 1);
            }
            uint32_t output = atomicAdd(&warp_positions[slot], 1U);
            grouped_entries[bucket.entry_offset + output] = entry;
        }
    }
}

__global__ void count_nonunit_per_bucket(
    const PrefixEntry* __restrict__ entries,
    const PrefixBucket* __restrict__ buckets, size_t bucket_count,
    uint32_t* __restrict__ counts) {
    unsigned lane = threadIdx.x & 31U;
    unsigned warp = threadIdx.x >> 5;
    size_t bucket_index = size_t(blockIdx.x) * 8 + warp;
    if (bucket_index >= bucket_count) return;
    PrefixBucket bucket = buckets[bucket_index];
    uint32_t local = 0;
    for (uint32_t index = lane; index < bucket.count; index += 32) {
        local += entries[bucket.entry_offset + index].weight != 1;
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        local += __shfl_down_sync(UINT32_MAX, local, offset);
    }
    if (!lane) counts[bucket_index] = local;
}

__global__ void count_unit_per_bucket(
    const PrefixEntry* __restrict__ entries,
    const PrefixBucket* __restrict__ buckets, size_t bucket_count,
    uint32_t* __restrict__ counts) {
    unsigned lane = threadIdx.x & 31U;
    unsigned warp = threadIdx.x >> 5;
    size_t bucket_index = size_t(blockIdx.x) * 8 + warp;
    if (bucket_index >= bucket_count) return;
    PrefixBucket bucket = buckets[bucket_index];
    uint32_t local = 0;
    for (uint32_t index = lane; index < bucket.count; index += 32)
        local += entries[bucket.entry_offset + index].weight == 1;
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        local += __shfl_down_sync(UINT32_MAX, local, offset);
    if (!lane) counts[bucket_index] = local;
}

__global__ void scatter_nonunit_per_bucket(
    const PrefixEntry* __restrict__ entries,
    const PrefixBucket* __restrict__ buckets, size_t bucket_count,
    const uint32_t* __restrict__ offsets,
    PrefixEntry* __restrict__ nonunit_entries) {
    unsigned lane = threadIdx.x & 31U;
    unsigned warp = threadIdx.x >> 5;
    size_t bucket_index = size_t(blockIdx.x) * 8 + warp;
    if (bucket_index >= bucket_count) return;
    PrefixBucket bucket = buckets[bucket_index];
    uint32_t output_base = offsets[bucket_index];
    uint32_t emitted = 0;
    for (uint32_t base = 0; base < bucket.count; base += 32) {
        uint32_t index = base + lane;
        PrefixEntry entry = index < bucket.count
            ? entries[bucket.entry_offset + index]
            : PrefixEntry{0, 1};
        unsigned selected = __ballot_sync(
            UINT32_MAX, index < bucket.count && entry.weight != 1);
        if (selected & (1U << lane)) {
            unsigned lower = selected & ((1U << lane) - 1U);
            nonunit_entries[output_base + emitted + __popc(lower)] = entry;
        }
        emitted += __popc(selected);
    }
}

__global__ void scatter_unit_per_bucket(
    const PrefixEntry* __restrict__ entries,
    const PrefixBucket* __restrict__ buckets, size_t bucket_count,
    const uint32_t* __restrict__ offsets,
    PrefixEntry* __restrict__ unit_entries) {
    unsigned lane = threadIdx.x & 31U;
    unsigned warp = threadIdx.x >> 5;
    size_t bucket_index = size_t(blockIdx.x) * 8 + warp;
    if (bucket_index >= bucket_count) return;
    PrefixBucket bucket = buckets[bucket_index];
    uint32_t output_base = offsets[bucket_index];
    uint32_t emitted = 0;
    for (uint32_t base = 0; base < bucket.count; base += 32) {
        uint32_t index = base + lane;
        PrefixEntry entry = index < bucket.count
            ? entries[bucket.entry_offset + index] : PrefixEntry{0, 1};
        unsigned selected = __ballot_sync(
            UINT32_MAX, index < bucket.count && entry.weight == 1);
        if (selected & (1U << lane)) {
            unsigned lower = selected & ((1U << lane) - 1U);
            unit_entries[output_base + emitted + __popc(lower)] = entry;
        }
        emitted += __popc(selected);
    }
}

__global__ void build_nonunit_buckets(
    const PrefixBucket* __restrict__ source, size_t bucket_count,
    const uint32_t* __restrict__ offsets,
    PrefixBucket* __restrict__ destination) {
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < bucket_count; index += size_t(blockDim.x) * gridDim.x) {
        PrefixBucket bucket = source[index];
        bucket.entry_offset = offsets[index];
        bucket.count = offsets[index + 1] - offsets[index];
        destination[index] = bucket;
    }
}

static DeviceNonunitLayout build_nonunit_layout(
    const DevicePrefixLayout& source) {
    DeviceNonunitLayout result;
    result.bucket_count = source.bucket_count;
    double start = seconds_now();
    uint32_t* offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&offsets,
                          (source.bucket_count + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(offsets + source.bucket_count, 0,
                          sizeof(uint32_t)));
    unsigned bucket_blocks =
        unsigned((source.bucket_count + 7) / 8);
    count_nonunit_per_bucket<<<bucket_blocks, THREADS>>>(
        source.entries, source.buckets, source.bucket_count, offsets);
    CUDA_CHECK(cudaGetLastError());
    thrust::device_ptr<uint32_t> offsets_ptr(offsets);
    thrust::exclusive_scan(offsets_ptr,
                           offsets_ptr + source.bucket_count + 1,
                           offsets_ptr);
    uint32_t total_entries = 0;
    CUDA_CHECK(cudaMemcpy(&total_entries, offsets + source.bucket_count,
                          sizeof(total_entries), cudaMemcpyDeviceToHost));
    result.entry_count = total_entries;
    CUDA_CHECK(cudaMalloc(&result.entries,
                          std::max<size_t>(1, result.entry_count) *
                              sizeof(PrefixEntry)));
    CUDA_CHECK(cudaMalloc(&result.buckets,
                          result.bucket_count * sizeof(PrefixBucket)));
    scatter_nonunit_per_bucket<<<bucket_blocks, THREADS>>>(
        source.entries, source.buckets, source.bucket_count, offsets,
        result.entries);
    CUDA_CHECK(cudaGetLastError());
    unsigned metadata_blocks = unsigned(std::min<size_t>(
        65535, (result.bucket_count + THREADS - 1) / THREADS));
    build_nonunit_buckets<<<metadata_blocks, THREADS>>>(
        source.buckets, result.bucket_count, offsets, result.buckets);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(offsets));
    result.build_seconds = seconds_now() - start;
    return result;
}

static DeviceNonunitLayout build_unit_layout(const DevicePrefixLayout& source) {
    DeviceNonunitLayout result;
    result.bucket_count = source.bucket_count;
    double start = seconds_now();
    uint32_t* offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&offsets,
                          (source.bucket_count + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(offsets + source.bucket_count, 0, sizeof(uint32_t)));
    unsigned bucket_blocks = unsigned((source.bucket_count + 7) / 8);
    count_unit_per_bucket<<<bucket_blocks, THREADS>>>(
        source.entries, source.buckets, source.bucket_count, offsets);
    CUDA_CHECK(cudaGetLastError());
    thrust::device_ptr<uint32_t> offsets_ptr(offsets);
    thrust::exclusive_scan(offsets_ptr,
                           offsets_ptr + source.bucket_count + 1, offsets_ptr);
    uint32_t total_entries = 0;
    CUDA_CHECK(cudaMemcpy(&total_entries, offsets + source.bucket_count,
                          sizeof(total_entries), cudaMemcpyDeviceToHost));
    result.entry_count = total_entries;
    CUDA_CHECK(cudaMalloc(&result.entries,
                          std::max<size_t>(1, result.entry_count) *
                              sizeof(PrefixEntry)));
    CUDA_CHECK(cudaMalloc(&result.buckets,
                          result.bucket_count * sizeof(PrefixBucket)));
    scatter_unit_per_bucket<<<bucket_blocks, THREADS>>>(
        source.entries, source.buckets, source.bucket_count, offsets,
        result.entries);
    CUDA_CHECK(cudaGetLastError());
    unsigned metadata_blocks = unsigned(std::min<size_t>(
        65535, (result.bucket_count + THREADS - 1) / THREADS));
    build_nonunit_buckets<<<metadata_blocks, THREADS>>>(
        source.buckets, result.bucket_count, offsets, result.buckets);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(offsets));
    result.build_seconds = seconds_now() - start;
    return result;
}

static void free_nonunit_layout(DeviceNonunitLayout& layout) {
    CUDA_CHECK(cudaFree(layout.buckets));
    CUDA_CHECK(cudaFree(layout.entries));
    layout = DeviceNonunitLayout{};
}

static DeviceWeightClassLayout build_weight_class_layout(
    const DevicePrefixLayout& source) {
    DeviceWeightClassLayout result;
    result.entry_count = source.entry_count;
    result.bucket_count = source.bucket_count;
    double start = seconds_now();
    uint32_t* offsets = nullptr;
    uint32_t* overflow = nullptr;
    CUDA_CHECK(cudaMalloc(&offsets,
                          (source.bucket_count + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(offsets + source.bucket_count, 0,
                          sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&overflow, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(overflow, 0, sizeof(uint32_t)));
    unsigned bucket_blocks = unsigned((source.bucket_count + 7) / 8);
    count_weight_classes_per_bucket<<<bucket_blocks, THREADS>>>(
        source.entries, source.buckets, source.bucket_count, offsets,
        overflow);
    CUDA_CHECK(cudaGetLastError());
    thrust::device_ptr<uint32_t> offsets_ptr(offsets);
    thrust::exclusive_scan(offsets_ptr,
                           offsets_ptr + source.bucket_count + 1,
                           offsets_ptr);
    uint32_t class_count = 0;
    uint32_t failed = 0;
    CUDA_CHECK(cudaMemcpy(&class_count, offsets + source.bucket_count,
                          sizeof(class_count), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&failed, overflow, sizeof(failed),
                          cudaMemcpyDeviceToHost));
    if (failed)
        throw std::runtime_error(
            "a prefix bucket exceeds 32 exact weight classes");
    result.class_count = class_count;
    CUDA_CHECK(cudaMalloc(&result.entries,
                          std::max<size_t>(1, result.entry_count) *
                              sizeof(PrefixEntry)));
    CUDA_CHECK(cudaMalloc(&result.buckets,
                          result.bucket_count * sizeof(PrefixBucket)));
    CUDA_CHECK(cudaMalloc(&result.classes,
                          std::max<size_t>(1, result.class_count) *
                              sizeof(WeightClassMeta)));
    group_weight_classes_per_bucket<<<bucket_blocks, THREADS>>>(
        source.entries, source.buckets, source.bucket_count, offsets,
        result.entries, result.buckets, result.classes, overflow);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&failed, overflow, sizeof(failed),
                          cudaMemcpyDeviceToHost));
    if (failed)
        throw std::runtime_error("weight-class grouping overflow");
    std::vector<uint32_t> host_offsets(source.bucket_count + 1);
    CUDA_CHECK(cudaMemcpy(host_offsets.data(), offsets,
                          host_offsets.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    for (size_t index = 0; index < source.bucket_count; index++)
        result.maximum_classes = std::max<size_t>(
            result.maximum_classes,
            host_offsets[index + 1] - host_offsets[index]);
    CUDA_CHECK(cudaFree(overflow));
    CUDA_CHECK(cudaFree(offsets));
    result.build_seconds = seconds_now() - start;
    return result;
}

static void free_weight_class_layout(DeviceWeightClassLayout& layout) {
    CUDA_CHECK(cudaFree(layout.classes));
    CUDA_CHECK(cudaFree(layout.buckets));
    CUDA_CHECK(cudaFree(layout.entries));
    layout = DeviceWeightClassLayout{};
}

template <bool OTHER_WEIGHTED>
static __device__ __forceinline__ unsigned long long delta_bucket_join(
    const PrefixEntry* __restrict__ special_entries, PrefixBucket special,
    const PrefixEntry* __restrict__ other_entries, PrefixBucket other,
    PrefixEntry* __restrict__ warp_tile, unsigned lane) {
    unsigned long long sum = 0;
    for (uint32_t special_base = 0; special_base < special.count;
         special_base += 32) {
        uint32_t special_index = special_base + lane;
        bool special_valid = special_index < special.count;
        PrefixEntry special_entry = special_valid
            ? special_entries[special.entry_offset + special_index]
            : PrefixEntry{0, 1};
        for (uint32_t other_base = 0; other_base < other.count;
             other_base += 32) {
            uint32_t other_index = other_base + lane;
            warp_tile[lane] = other_index < other.count
                ? other_entries[other.entry_offset + other_index]
                : PrefixEntry{0, 1};
            __syncwarp();
            uint32_t count = min(uint32_t(32), other.count - other_base);
#pragma unroll
            for (uint32_t offset = 0; offset < count; offset++) {
                PrefixEntry other_entry = warp_tile[offset];
                if (special_valid &&
                    !(special_entry.suffix & other_entry.suffix)) {
                    uint64_t delta = uint64_t(special_entry.weight - 1);
                    sum += OTHER_WEIGHTED
                        ? delta * uint64_t(other_entry.weight)
                        : delta;
                }
            }
            __syncwarp();
        }
    }
    return sum;
}

template <bool INCLUDE_BASE>
__global__ void nonunit_correction_joins(
    const PrefixEntry* __restrict__ left_entries,
    const PrefixEntry* __restrict__ right_entries,
    const PrefixEntry* __restrict__ left_nonunit_entries,
    const PrefixEntry* __restrict__ right_nonunit_entries,
    const PrefixBucket* __restrict__ left_buckets,
    const PrefixBucket* __restrict__ right_buckets,
    const PrefixBucket* __restrict__ left_nonunit_buckets,
    const PrefixBucket* __restrict__ right_nonunit_buckets,
    const PrefixJoinDesc* __restrict__ joins,
    unsigned long long* __restrict__ results) {
    __shared__ unsigned long long warp_partial[8];
    __shared__ PrefixEntry warp_tiles[THREADS];
    __shared__ uint32_t next_task;
    const PrefixJoinDesc join = joins[blockIdx.x];
    const unsigned lane = threadIdx.x & 31U;
    const unsigned warp = threadIdx.x >> 5;
    const uint32_t task_count =
        join.left_bucket_count * join.right_bucket_count;
    unsigned long long sum = 0;
    if (!threadIdx.x) next_task = 0;
    __syncthreads();
    uint32_t task_base = 0;
    if (!lane) task_base = atomicAdd(&next_task, uint32_t(PREFIX_TASK_CHUNK));
    task_base = __shfl_sync(UINT32_MAX, task_base, 0);
    while (task_base < task_count) {
        uint32_t task_end =
            min(task_count, task_base + uint32_t(PREFIX_TASK_CHUNK));
        uint32_t left_bucket_index = task_base / join.right_bucket_count;
        uint32_t right_bucket_index =
            task_base - left_bucket_index * join.right_bucket_count;
        for (uint32_t task = task_base; task < task_end; task++) {
            uint32_t left_slot = join.left_bucket_offset + left_bucket_index;
            uint32_t right_slot =
                join.right_bucket_offset + right_bucket_index;
            PrefixBucket left = left_buckets[left_slot];
            PrefixBucket right = right_buckets[right_slot];
            if (!(left.prefix & right.prefix)) {
                if constexpr (INCLUDE_BASE) {
                    sum += ptx_bucket_join<false, false>(
                        left_entries, left, right_entries, right, lane);
                }
                PrefixBucket left_nonunit =
                    left_nonunit_buckets[left_slot];
                PrefixBucket right_nonunit =
                    right_nonunit_buckets[right_slot];
                sum += delta_bucket_join<true>(
                    left_nonunit_entries, left_nonunit, right_entries,
                    right, &warp_tiles[warp * 32], lane);
                sum += delta_bucket_join<false>(
                    right_nonunit_entries, right_nonunit, left_entries,
                    left, &warp_tiles[warp * 32], lane);
            }
            right_bucket_index++;
            if (right_bucket_index == join.right_bucket_count) {
                right_bucket_index = 0;
                left_bucket_index++;
            }
        }
        if (!lane) {
            task_base =
                atomicAdd(&next_task, uint32_t(PREFIX_TASK_CHUNK));
        }
        task_base = __shfl_sync(UINT32_MAX, task_base, 0);
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        sum += __shfl_down_sync(UINT32_MAX, sum, offset);
    }
    if (!lane) warp_partial[warp] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = lane < 8 ? warp_partial[lane] : 0;
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) {
            sum += __shfl_down_sync(UINT32_MAX, sum, offset);
        }
        if (!lane) results[blockIdx.x] = sum;
    }
}

// Exact disjoint partition of the weighted pair matrix.  Unit x unit is
// handled by the separate predicate-only BMMA launch; this kernel owns the
// other three quadrants and therefore repeats no suffix comparison.
__global__ void unit_partition_residual_joins(
    const PrefixEntry* __restrict__ left_unit_entries,
    const PrefixEntry* __restrict__ right_unit_entries,
    const PrefixEntry* __restrict__ left_nonunit_entries,
    const PrefixEntry* __restrict__ right_nonunit_entries,
    const PrefixBucket* __restrict__ left_unit_buckets,
    const PrefixBucket* __restrict__ right_unit_buckets,
    const PrefixBucket* __restrict__ left_nonunit_buckets,
    const PrefixBucket* __restrict__ right_nonunit_buckets,
    const PrefixJoinDesc* __restrict__ joins,
    unsigned long long* __restrict__ results) {
    __shared__ unsigned long long warp_partial[8];
    __shared__ PrefixEntry warp_tiles[THREADS];
    __shared__ uint32_t next_task;
    const PrefixJoinDesc join = joins[blockIdx.x];
    const unsigned lane = threadIdx.x & 31U;
    const unsigned warp = threadIdx.x >> 5;
    const uint32_t task_count =
        join.left_bucket_count * join.right_bucket_count;
    unsigned long long sum = 0;
    if (!threadIdx.x) next_task = 0;
    __syncthreads();
    uint32_t task_base = 0;
    if (!lane) task_base = atomicAdd(&next_task, uint32_t(PREFIX_TASK_CHUNK));
    task_base = __shfl_sync(UINT32_MAX, task_base, 0);
    while (task_base < task_count) {
        uint32_t task_end = min(task_count,
                                task_base + uint32_t(PREFIX_TASK_CHUNK));
        uint32_t li = task_base / join.right_bucket_count;
        uint32_t ri = task_base - li * join.right_bucket_count;
        for (uint32_t task = task_base; task < task_end; task++) {
            uint32_t ls = join.left_bucket_offset + li;
            uint32_t rs = join.right_bucket_offset + ri;
            PrefixBucket lu = left_unit_buckets[ls];
            PrefixBucket ru = right_unit_buckets[rs];
            if (!(lu.prefix & ru.prefix)) {
                PrefixBucket ln = left_nonunit_buckets[ls];
                PrefixBucket rn = right_nonunit_buckets[rs];
                sum += scalar_bucket_join(
                    left_unit_entries, lu, right_nonunit_entries, rn,
                    &warp_tiles[warp * 32], lane);
                sum += scalar_bucket_join(
                    left_nonunit_entries, ln, right_unit_entries, ru,
                    &warp_tiles[warp * 32], lane);
                sum += scalar_bucket_join(
                    left_nonunit_entries, ln, right_nonunit_entries, rn,
                    &warp_tiles[warp * 32], lane);
            }
            if (++ri == join.right_bucket_count) { ri = 0; li++; }
        }
        if (!lane) task_base = atomicAdd(&next_task,
                                         uint32_t(PREFIX_TASK_CHUNK));
        task_base = __shfl_sync(UINT32_MAX, task_base, 0);
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        sum += __shfl_down_sync(UINT32_MAX, sum, offset);
    if (!lane) warp_partial[warp] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = lane < 8 ? warp_partial[lane] : 0;
#pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(UINT32_MAX, sum, offset);
        if (!lane) results[blockIdx.x] += sum;
    }
}

__global__ void unit_partition_work_census(
    const PrefixBucket* __restrict__ lu,
    const PrefixBucket* __restrict__ ru,
    const PrefixBucket* __restrict__ ln,
    const PrefixBucket* __restrict__ rn,
    const PrefixJoinDesc* __restrict__ joins,
    unsigned long long* __restrict__ totals) {
    const PrefixJoinDesc join = joins[blockIdx.x];
    unsigned long long local[4] = {0, 0, 0, 0};
    uint64_t tasks = uint64_t(join.left_bucket_count) * join.right_bucket_count;
    for (uint64_t task = threadIdx.x; task < tasks; task += blockDim.x) {
        uint32_t li = uint32_t(task / join.right_bucket_count);
        uint32_t ri = uint32_t(task - uint64_t(li) * join.right_bucket_count);
        uint32_t ls = join.left_bucket_offset + li;
        uint32_t rs = join.right_bucket_offset + ri;
        PrefixBucket a = lu[ls], b = ru[rs];
        if (a.prefix & b.prefix) continue;
        uint64_t ac = a.count, bc = b.count;
        uint64_t nc = ln[ls].count, nd = rn[rs].count;
        local[0] += ac * bc;
        local[1] += ac * nd;
        local[2] += nc * bc;
        local[3] += nc * nd;
    }
    for (int kind = 0; kind < 4; kind++) {
        unsigned long long value = local[kind];
#pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            value += __shfl_down_sync(UINT32_MAX, value, offset);
        if (!(threadIdx.x & 31U)) atomicAdd(&totals[kind], value);
    }
}

static __device__ __forceinline__ uint64_t bmma_tile_count(uint32_t left,
                                                           uint32_t right) {
    uint64_t forward = uint64_t((left + 15) / 16) * ((right + 7) / 8);
    uint64_t reverse = uint64_t((right + 15) / 16) * ((left + 7) / 8);
    return min(forward, reverse);
}

__global__ void weight_class_tile_census(
    const PrefixBucket* __restrict__ original_left,
    const PrefixBucket* __restrict__ original_right,
    const PrefixBucket* __restrict__ class_left,
    const PrefixBucket* __restrict__ class_right,
    const WeightClassMeta* __restrict__ left_classes,
    const WeightClassMeta* __restrict__ right_classes,
    const PrefixJoinDesc* __restrict__ joins,
    unsigned long long* __restrict__ totals) {
    const PrefixJoinDesc join = joins[blockIdx.x];
    unsigned long long local[4] = {0, 0, 0, 0};
    uint64_t tasks = uint64_t(join.left_bucket_count) *
                     join.right_bucket_count;
    for (uint64_t task = threadIdx.x; task < tasks; task += blockDim.x) {
        uint32_t li = uint32_t(task / join.right_bucket_count);
        uint32_t ri = uint32_t(task - uint64_t(li) *
                                      join.right_bucket_count);
        uint32_t ls = join.left_bucket_offset + li;
        uint32_t rs = join.right_bucket_offset + ri;
        PrefixBucket left_bucket = original_left[ls];
        PrefixBucket right_bucket = original_right[rs];
        if (left_bucket.prefix & right_bucket.prefix) continue;
        local[0] += bmma_tile_count(left_bucket.count,
                                    right_bucket.count);
        PrefixBucket left_class_bucket = class_left[ls];
        PrefixBucket right_class_bucket = class_right[rs];
        local[2] += uint64_t(left_class_bucket.count) *
                    right_class_bucket.count;
        local[3]++;
        for (uint32_t a = 0; a < left_class_bucket.count; a++) {
            WeightClassMeta left_class =
                left_classes[left_class_bucket.entry_offset + a];
            for (uint32_t b = 0; b < right_class_bucket.count; b++) {
                WeightClassMeta right_class =
                    right_classes[right_class_bucket.entry_offset + b];
                local[1] += bmma_tile_count(left_class.count,
                                            right_class.count);
            }
        }
    }
    for (unsigned kind = 0; kind < 4; kind++) {
        unsigned long long value = local[kind];
#pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            value += __shfl_down_sync(UINT32_MAX, value, offset);
        if (!(threadIdx.x & 31U)) atomicAdd(&totals[kind], value);
    }
}

__global__ void weight_class_prefix_joins(
    const PrefixEntry* __restrict__ left_entries,
    const PrefixEntry* __restrict__ right_entries,
    const PrefixBucket* __restrict__ left_buckets,
    const PrefixBucket* __restrict__ right_buckets,
    const WeightClassMeta* __restrict__ left_classes,
    const WeightClassMeta* __restrict__ right_classes,
    const PrefixJoinDesc* __restrict__ joins,
    unsigned long long* __restrict__ results) {
    __shared__ unsigned long long warp_partial[8];
    __shared__ uint32_t next_task;
    const PrefixJoinDesc join = joins[blockIdx.x];
    const unsigned lane = threadIdx.x & 31U;
    const unsigned warp = threadIdx.x >> 5;
    const uint32_t task_count =
        join.left_bucket_count * join.right_bucket_count;
    unsigned long long sum = 0;
    if (!threadIdx.x) next_task = 0;
    __syncthreads();
    uint32_t task_base = 0;
    if (!lane) task_base = atomicAdd(&next_task,
                                     uint32_t(PREFIX_TASK_CHUNK));
    task_base = __shfl_sync(UINT32_MAX, task_base, 0);
    while (task_base < task_count) {
        uint32_t task_end = min(task_count,
                                task_base + uint32_t(PREFIX_TASK_CHUNK));
        uint32_t li = task_base / join.right_bucket_count;
        uint32_t ri = task_base - li * join.right_bucket_count;
        for (uint32_t task = task_base; task < task_end; task++) {
            PrefixBucket left_bucket =
                left_buckets[join.left_bucket_offset + li];
            PrefixBucket right_bucket =
                right_buckets[join.right_bucket_offset + ri];
            if (!(left_bucket.prefix & right_bucket.prefix)) {
                for (uint32_t a = 0; a < left_bucket.count; a++) {
                    WeightClassMeta left_class =
                        left_classes[left_bucket.entry_offset + a];
                    PrefixBucket left_entries_bucket{
                        left_class.entry_offset, left_class.count, 0, 0};
                    for (uint32_t b = 0; b < right_bucket.count; b++) {
                        WeightClassMeta right_class =
                            right_classes[right_bucket.entry_offset + b];
                        PrefixBucket right_entries_bucket{
                            right_class.entry_offset, right_class.count,
                            0, 0};
                        unsigned long long compatible =
                            ptx_bucket_join<false, false>(
                                left_entries, left_entries_bucket,
                                right_entries, right_entries_bucket, lane);
                        sum += compatible *
                               uint64_t(left_class.weight) *
                               uint64_t(right_class.weight);
                    }
                }
            }
            if (++ri == join.right_bucket_count) {
                ri = 0;
                li++;
            }
        }
        if (!lane) task_base = atomicAdd(&next_task,
                                         uint32_t(PREFIX_TASK_CHUNK));
        task_base = __shfl_sync(UINT32_MAX, task_base, 0);
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        sum += __shfl_down_sync(UINT32_MAX, sum, offset);
    if (!lane) warp_partial[warp] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = lane < 8 ? warp_partial[lane] : 0;
#pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(UINT32_MAX, sum, offset);
        if (!lane) results[blockIdx.x] = sum;
    }
}

template <bool SHUFFLE>
__global__ void synthetic_inline_bmma(
    const PrefixEntry* __restrict__ left,
    const PrefixEntry* __restrict__ right, uint32_t* __restrict__ output,
    uint32_t tests) {
    uint32_t test = blockIdx.x;
    if (test >= tests || threadIdx.x >= 32) return;
    unsigned lane = threadIdx.x;
    PtxFragmentA a = load_ptx_a<SHUFFLE>(
        left, test * 16, 16, 0, lane);
    PtxFragmentB b = load_ptx_b<SHUFFLE>(
        right, test * 8, 8, 0, lane);
    uint32_t d0, d1, d2, d3;
    inline_bmma_16x8(a.bits0, a.bits1, b.bits, d0, d1, d2, d3);
    unsigned group = lane >> 2;
    unsigned word = lane & 3U;
    unsigned column0 = 2 * word;
    unsigned column1 = column0 + 1;
    output[test * 128 + group * 8 + column0] = d0;
    output[test * 128 + group * 8 + column1] = d1;
    output[test * 128 + (group + 8) * 8 + column0] = d2;
    output[test * 128 + (group + 8) * 8 + column1] = d3;
}

static uint64_t probe_mix64(uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

static void validate_synthetic_mapping() {
    constexpr uint32_t tests = 64;
    constexpr uint64_t suffix_mask = (uint64_t(1) << 42) - 1;
    std::vector<PrefixEntry> left(tests * 16);
    std::vector<PrefixEntry> right(tests * 8);
    for (uint32_t test = 0; test < tests; test++) {
        for (uint32_t row = 0; row < 16; row++) {
            uint64_t value = probe_mix64(0x41c64e6dULL * (test + 1) + row);
            left[test * 16 + row] = PrefixEntry{
                value & suffix_mask, uint32_t(1 + value % 7)};
        }
        for (uint32_t column = 0; column < 8; column++) {
            uint64_t value =
                probe_mix64(0x9e3779b97f4a7c15ULL * (test + 1) + column);
            right[test * 8 + column] = PrefixEntry{
                value & suffix_mask, uint32_t(1 + value % 7)};
        }
    }
    PrefixEntry* device_left = upload_vector(left);
    PrefixEntry* device_right = upload_vector(right);
    uint32_t* device_direct = nullptr;
    uint32_t* device_shuffle = nullptr;
    CUDA_CHECK(cudaMalloc(&device_direct, tests * 128 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&device_shuffle, tests * 128 * sizeof(uint32_t)));
    synthetic_inline_bmma<false><<<tests, 32>>>(
        device_left, device_right, device_direct, tests);
    CUDA_CHECK(cudaGetLastError());
    synthetic_inline_bmma<true><<<tests, 32>>>(
        device_left, device_right, device_shuffle, tests);
    CUDA_CHECK(cudaGetLastError());
    std::vector<uint32_t> direct(tests * 128);
    std::vector<uint32_t> shuffle(tests * 128);
    CUDA_CHECK(cudaMemcpy(direct.data(), device_direct,
                          direct.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(shuffle.data(), device_shuffle,
                          shuffle.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    for (uint32_t test = 0; test < tests; test++) {
        for (uint32_t row = 0; row < 16; row++) {
            for (uint32_t column = 0; column < 8; column++) {
                size_t index = test * 128 + row * 8 + column;
                uint32_t expected = uint32_t(__builtin_popcountll(
                    left[test * 16 + row].suffix &
                    right[test * 8 + column].suffix));
                if (direct[index] != expected ||
                    shuffle[index] != expected) {
                    std::fprintf(
                        stderr,
                        "synthetic mapping mismatch test=%u row=%u col=%u "
                        "expected=%u direct=%u shuffle=%u\n",
                        test, row, column, expected, direct[index],
                        shuffle[index]);
                    std::exit(1);
                }
            }
        }
    }
    CUDA_CHECK(cudaFree(device_shuffle));
    CUDA_CHECK(cudaFree(device_direct));
    CUDA_CHECK(cudaFree(device_right));
    CUDA_CHECK(cudaFree(device_left));
    std::printf("PTX_MAPPING tests=%u outputs=%u exact=OK\n", tests,
                tests * 128);
}

static double time_scalar_inline_control(
    const DevicePrefixLayout& left, const DevicePrefixLayout& right,
    const PrefixJoinDesc* joins, unsigned long long* results,
    size_t join_count) {
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    prefix_disjoint_joins<<<unsigned(join_count), THREADS>>>(
        left.entries, right.entries, left.buckets, right.buckets, joins,
        results);
    CUDA_CHECK(cudaGetLastError());
    double seconds = elapsed_kernel(start, end);
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaEventDestroy(start));
    return seconds;
}

template <bool SHUFFLE, bool WEIGHTED, bool HYBRID,
          bool UNIT_FAST = false, bool ROW_FACTORIZED = false>
static double time_inline_ptx(
    const DevicePrefixLayout& left, const DevicePrefixLayout& right,
    const PrefixJoinDesc* joins, unsigned long long* results,
    size_t join_count, uint64_t threshold) {
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    inline_ptx_prefix_joins<SHUFFLE, WEIGHTED, HYBRID, UNIT_FAST,
                            ROW_FACTORIZED>
        <<<unsigned(join_count), THREADS>>>(
            left.entries, right.entries, left.buckets, right.buckets, joins,
            results, threshold);
    CUDA_CHECK(cudaGetLastError());
    double seconds = elapsed_kernel(start, end);
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaEventDestroy(start));
    return seconds;
}

static double time_unit_partitioned(
    const DeviceNonunitLayout& left_unit,
    const DeviceNonunitLayout& right_unit,
    const DeviceNonunitLayout& left_nonunit,
    const DeviceNonunitLayout& right_nonunit,
    const PrefixJoinDesc* joins, unsigned long long* results,
    size_t join_count) {
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    inline_ptx_prefix_joins<false, false, false>
        <<<unsigned(join_count), THREADS>>>(
            left_unit.entries, right_unit.entries, left_unit.buckets,
            right_unit.buckets, joins, results, 0);
    CUDA_CHECK(cudaGetLastError());
    unit_partition_residual_joins<<<unsigned(join_count), THREADS>>>(
        left_unit.entries, right_unit.entries, left_nonunit.entries,
        right_nonunit.entries, left_unit.buckets, right_unit.buckets,
        left_nonunit.buckets, right_nonunit.buckets, joins, results);
    CUDA_CHECK(cudaGetLastError());
    double seconds = elapsed_kernel(start, end);
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaEventDestroy(start));
    return seconds;
}

static std::array<uint64_t, 4> unit_partition_census(
    const DeviceNonunitLayout& left_unit,
    const DeviceNonunitLayout& right_unit,
    const DeviceNonunitLayout& left_nonunit,
    const DeviceNonunitLayout& right_nonunit,
    const PrefixJoinDesc* joins, size_t join_count) {
    unsigned long long* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(device, 0, 4 * sizeof(unsigned long long)));
    unit_partition_work_census<<<unsigned(join_count), THREADS>>>(
        left_unit.buckets, right_unit.buckets, left_nonunit.buckets,
        right_nonunit.buckets, joins, device);
    CUDA_CHECK(cudaGetLastError());
    std::array<uint64_t, 4> result{};
    CUDA_CHECK(cudaMemcpy(result.data(), device,
                          4 * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device));
    return result;
}

static std::array<uint64_t, 4> weight_class_census(
    const DevicePrefixLayout& original_left,
    const DevicePrefixLayout& original_right,
    const DeviceWeightClassLayout& class_left,
    const DeviceWeightClassLayout& class_right,
    const PrefixJoinDesc* joins, size_t join_count) {
    unsigned long long* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(device, 0, 4 * sizeof(unsigned long long)));
    weight_class_tile_census<<<unsigned(join_count), THREADS>>>(
        original_left.buckets, original_right.buckets,
        class_left.buckets, class_right.buckets, class_left.classes,
        class_right.classes, joins, device);
    CUDA_CHECK(cudaGetLastError());
    std::array<uint64_t, 4> result{};
    CUDA_CHECK(cudaMemcpy(result.data(), device,
                          4 * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device));
    return result;
}

static double time_weight_class_bmma(
    const DeviceWeightClassLayout& left,
    const DeviceWeightClassLayout& right,
    const PrefixJoinDesc* joins, unsigned long long* results,
    size_t join_count) {
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    weight_class_prefix_joins<<<unsigned(join_count), THREADS>>>(
        left.entries, right.entries, left.buckets, right.buckets,
        left.classes, right.classes, joins, results);
    CUDA_CHECK(cudaGetLastError());
    double seconds = elapsed_kernel(start, end);
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaEventDestroy(start));
    return seconds;
}

static double time_nonunit_decomposed(
    const DevicePrefixLayout& left, const DevicePrefixLayout& right,
    const DeviceNonunitLayout& left_nonunit,
    const DeviceNonunitLayout& right_nonunit,
    const PrefixJoinDesc* joins, unsigned long long* combined_results,
    size_t join_count) {
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    nonunit_correction_joins<true><<<unsigned(join_count), THREADS>>>(
        left.entries, right.entries, left_nonunit.entries,
        right_nonunit.entries, left.buckets, right.buckets,
        left_nonunit.buckets, right_nonunit.buckets, joins,
        combined_results);
    CUDA_CHECK(cudaGetLastError());
    double seconds = elapsed_kernel(start, end);
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaEventDestroy(start));
    return seconds;
}

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

static void require_results(
    const std::vector<unsigned long long>& expected,
    unsigned long long* device_results, const char* label) {
    std::vector<unsigned long long> actual(expected.size());
    CUDA_CHECK(cudaMemcpy(actual.data(), device_results,
                          actual.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    if (actual == expected) return;
    size_t mismatch = 0;
    while (mismatch < actual.size() && actual[mismatch] == expected[mismatch]) {
        mismatch++;
    }
    std::fprintf(stderr,
                 "%s mismatch join=%zu expected=%llu actual=%llu\n", label,
                 mismatch, expected[mismatch], actual[mismatch]);
    std::exit(1);
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 6) {
        std::fprintf(stderr,
                     "Usage: %s ORBITS [START=0] [END=8192] [REPEATS=5] "
                     "[MODE=full|weight-class]\n",
                     argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    uint64_t start_record = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 0;
    uint64_t end_record = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 8192;
    int repeats = argc > 4 ? std::atoi(argv[4]) : 5;
    std::string mode = argc > 5 ? argv[5] : "full";
    if (end_record <= start_record || repeats < 1 ||
        (mode != "full" && mode != "weight-class")) return 2;

    validate_synthetic_mapping();
    initialise_tables();
    U128 labelled_weight = 0;
    uint64_t records = 0;
    std::vector<Edge> edges = read_edges(path, start_record, end_record, 0, 0,
                                         labelled_weight, records);
    std::vector<PrefixKey> left_keys = unique_lefts(edges);
    std::vector<PrefixKey> right_keys = unique_rights(edges);
    std::vector<PrefixKey> all_keys = left_keys;
    all_keys.insert(all_keys.end(), right_keys.begin(), right_keys.end());
    CanonicalFactory factory =
        build_canonical_factory(std::move(all_keys), LEFT_COLUMNS);
    ProductionCanonicalDevice canonical = upload_production_canonical(factory);
    DevicePrefixLayout left =
        build_sparse_device_prefix_layout(left_keys, factory, canonical);
    DevicePrefixLayout right =
        build_sparse_device_prefix_layout(right_keys, factory, canonical);

    std::unordered_map<PrefixKey, uint32_t> left_index;
    std::unordered_map<PrefixKey, uint32_t> right_index;
    left_index.reserve(left_keys.size() * 2);
    right_index.reserve(right_keys.size() * 2);
    for (size_t index = 0; index < left_keys.size(); index++) {
        left_index.emplace(left_keys[index], uint32_t(index));
    }
    for (size_t index = 0; index < right_keys.size(); index++) {
        right_index.emplace(right_keys[index], uint32_t(index));
    }
    std::vector<PrefixJoinDesc> joins;
    std::vector<uint64_t> direct_work;
    U128 comparisons = 0;
    joins.reserve(edges.size() * 2);
    direct_work.reserve(edges.size() * 2);
    for (const Edge& edge : edges) {
        const PrefixPair& left_pair = left.pairs[left_index.at(edge.left)];
        const PrefixPair& right_pair = right.pairs[right_index.at(edge.right)];
        const PrefixDistribution lhs[2] = {
            left_pair.selected, left_pair.complement};
        const PrefixDistribution rhs[2] = {
            right_pair.selected, right_pair.complement};
        for (int component = 0; component < 2; component++) {
            joins.push_back(PrefixJoinDesc{
                lhs[component].bucket_offset, rhs[component].bucket_offset,
                lhs[component].bucket_count, rhs[component].bucket_count});
            uint64_t work = uint64_t(lhs[component].entry_count) *
                            rhs[component].entry_count;
            direct_work.push_back(work);
            comparisons += work;
        }
    }
    schedule_prefix_heavy_first(joins, direct_work);
    PrefixJoinDesc* device_joins = upload_vector(joins);
    unsigned long long* scalar_results = nullptr;
    unsigned long long* candidate_results = nullptr;
    CUDA_CHECK(cudaMalloc(&scalar_results,
                          joins.size() * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&candidate_results,
                          joins.size() * sizeof(unsigned long long)));
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::printf(
        "PTX_INPUT records=%llu joins=%zu left_entries=%zu "
        "right_entries=%zu comparisons=%s free_bytes=%zu\n",
        (unsigned long long)records, joins.size(), left.entry_count,
        right.entry_count, u128_string(comparisons).c_str(), free_bytes);
    uint64_t left_unit_count =
        device_unit_weight_count(left.entries, left.entry_count);
    uint64_t right_unit_count =
        device_unit_weight_count(right.entries, right.entry_count);
    std::printf(
        "PTX_WEIGHTS left_unit=%llu left_fraction=%.9f "
        "right_unit=%llu right_fraction=%.9f combined_fraction=%.9f\n",
        (unsigned long long)left_unit_count,
        left.entry_count ? double(left_unit_count) / left.entry_count : 0,
        (unsigned long long)right_unit_count,
        right.entry_count ? double(right_unit_count) / right.entry_count : 0,
        left.entry_count + right.entry_count
            ? double(left_unit_count + right_unit_count) /
                  (left.entry_count + right.entry_count)
            : 0);
    if (mode == "weight-class") {
        DeviceWeightClassLayout class_left =
            build_weight_class_layout(left);
        DeviceWeightClassLayout class_right =
            build_weight_class_layout(right);
        std::printf(
            "PTX_WEIGHT_CLASS_LAYOUT left_classes=%zu right_classes=%zu "
            "left_max_classes=%zu right_max_classes=%zu "
            "left_build=%.6f right_build=%.6f\n",
            class_left.class_count, class_right.class_count,
            class_left.maximum_classes, class_right.maximum_classes,
            class_left.build_seconds, class_right.build_seconds);
        std::array<uint64_t, 4> class_census = weight_class_census(
            left, right, class_left, class_right, device_joins,
            joins.size());
        std::printf(
            "PTX_WEIGHT_CLASS_CENSUS compatible_bucket_pairs=%llu "
            "class_pairs=%llu predicate_tiles=%llu class_tiles=%llu "
            "class_pair_inflation=%.9f tile_inflation=%.9f\n",
            (unsigned long long)class_census[3],
            (unsigned long long)class_census[2],
            (unsigned long long)class_census[0],
            (unsigned long long)class_census[1],
            class_census[3]
                ? double(class_census[2]) / class_census[3] : 0,
            class_census[0]
                ? double(class_census[1]) / class_census[0] : 0);
        time_scalar_inline_control(left, right, device_joins, scalar_results,
                                   joins.size());
        std::vector<unsigned long long> expected(joins.size());
        CUDA_CHECK(cudaMemcpy(expected.data(), scalar_results,
                              expected.size() * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
        time_weight_class_bmma(class_left, class_right, device_joins,
                               candidate_results, joins.size());
        require_results(expected, candidate_results, "weight-class-bmma");
        std::vector<double> scalar_times;
        std::vector<double> predicate_times;
        std::vector<double> class_times;
        for (int repeat = 0; repeat < repeats; repeat++) {
            scalar_times.push_back(time_scalar_inline_control(
                left, right, device_joins, scalar_results, joins.size()));
            predicate_times.push_back(time_inline_ptx<false, false, false>(
                left, right, device_joins, candidate_results, joins.size(),
                0));
            class_times.push_back(time_weight_class_bmma(
                class_left, class_right, device_joins, candidate_results,
                joins.size()));
        }
        double scalar = median(scalar_times);
        double predicate = median(predicate_times);
        double exact_class = median(class_times);
        require_results(expected, candidate_results,
                        "weight-class-bmma-final");
        std::printf(
            "PTX_WEIGHT_CLASS exact=OK scalar=%.6f predicate=%.6f "
            "weight_class=%.6f speedup=%.6f "
            "predicate_ceiling=%.6f repeats=%d\n",
            scalar, predicate, exact_class, scalar / exact_class,
            scalar / predicate, repeats);
        free_weight_class_layout(class_right);
        free_weight_class_layout(class_left);
        CUDA_CHECK(cudaFree(candidate_results));
        CUDA_CHECK(cudaFree(scalar_results));
        CUDA_CHECK(cudaFree(device_joins));
        free_device_prefix_layout(right);
        free_device_prefix_layout(left);
        free_production_canonical(canonical);
        return 0;
    }
    DeviceNonunitLayout left_nonunit = build_nonunit_layout(left);
    DeviceNonunitLayout right_nonunit = build_nonunit_layout(right);
    DeviceNonunitLayout left_unit = build_unit_layout(left);
    DeviceNonunitLayout right_unit = build_unit_layout(right);
    std::printf(
        "PTX_NONUNIT left_entries=%zu left_fraction=%.9f "
        "right_entries=%zu right_fraction=%.9f left_build=%.6f "
        "right_build=%.6f\n",
        left_nonunit.entry_count,
        left.entry_count
            ? double(left_nonunit.entry_count) / left.entry_count
            : 0,
        right_nonunit.entry_count,
        right.entry_count
            ? double(right_nonunit.entry_count) / right.entry_count
            : 0,
        left_nonunit.build_seconds, right_nonunit.build_seconds);
    std::printf(
        "PTX_UNIT_LAYOUT left_entries=%zu right_entries=%zu "
        "left_build=%.6f right_build=%.6f\n",
        left_unit.entry_count, right_unit.entry_count,
        left_unit.build_seconds, right_unit.build_seconds);
    std::array<uint64_t, 4> partition_work = unit_partition_census(
        left_unit, right_unit, left_nonunit, right_nonunit, device_joins,
        joins.size());
    uint64_t partition_total = 0;
    for (uint64_t work : partition_work) partition_total += work;
    std::printf(
        "PTX_PARTITION_WORK total=%llu unit_unit=%llu unit_nonunit=%llu "
        "nonunit_unit=%llu nonunit_nonunit=%llu unit_unit_fraction=%.9f "
        "residual_fraction=%.9f\n",
        (unsigned long long)partition_total,
        (unsigned long long)partition_work[0],
        (unsigned long long)partition_work[1],
        (unsigned long long)partition_work[2],
        (unsigned long long)partition_work[3],
        partition_total ? double(partition_work[0]) / partition_total : 0,
        partition_total
            ? double(partition_work[1] + partition_work[2] +
                     partition_work[3]) /
                  partition_total
            : 0);

    time_scalar_inline_control(left, right, device_joins, scalar_results,
                               joins.size());
    std::vector<unsigned long long> expected(joins.size());
    CUDA_CHECK(cudaMemcpy(expected.data(), scalar_results,
                          expected.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    time_inline_ptx<false, true, false>(
        left, right, device_joins, candidate_results, joins.size(), 0);
    require_results(expected, candidate_results, "ptx-direct");
    time_inline_ptx<false, true, false, false, true>(
        left, right, device_joins, candidate_results, joins.size(), 0);
    require_results(expected, candidate_results, "ptx-row-factor-direct");
    time_inline_ptx<true, true, false, false, true>(
        left, right, device_joins, candidate_results, joins.size(), 0);
    require_results(expected, candidate_results, "ptx-row-factor-shuffle");
    time_inline_ptx<true, true, false>(
        left, right, device_joins, candidate_results, joins.size(), 0);
    require_results(expected, candidate_results, "ptx-shuffle");
    time_inline_ptx<false, true, false, true>(
        left, right, device_joins, candidate_results, joins.size(), 0);
    require_results(expected, candidate_results, "ptx-unit-fast");
    time_nonunit_decomposed(
        left, right, left_nonunit, right_nonunit, device_joins,
        candidate_results, joins.size());
    require_results(expected, candidate_results, "ptx-nonunit-decomposed");
    time_unit_partitioned(
        left_unit, right_unit, left_nonunit, right_nonunit, device_joins,
        candidate_results, joins.size());
    require_results(expected, candidate_results, "ptx-unit-partitioned");

    std::vector<double> scalar_times;
    std::vector<double> direct_times;
    std::vector<double> shuffle_times;
    std::vector<double> direct_predicate_times;
    std::vector<double> shuffle_predicate_times;
    std::vector<double> row_factor_direct_times;
    std::vector<double> row_factor_shuffle_times;
    std::vector<double> unit_fast_times;
    std::vector<double> decomposed_times;
    std::vector<double> partitioned_times;
    for (int repeat = 0; repeat < repeats; repeat++) {
        scalar_times.push_back(time_scalar_inline_control(
            left, right, device_joins, scalar_results, joins.size()));
        direct_times.push_back(time_inline_ptx<false, true, false>(
            left, right, device_joins, candidate_results, joins.size(), 0));
        shuffle_times.push_back(time_inline_ptx<true, true, false>(
            left, right, device_joins, candidate_results, joins.size(), 0));
        direct_predicate_times.push_back(
            time_inline_ptx<false, false, false>(
                left, right, device_joins, candidate_results, joins.size(),
                0));
        shuffle_predicate_times.push_back(
            time_inline_ptx<true, false, false>(
                left, right, device_joins, candidate_results, joins.size(),
                0));
        row_factor_direct_times.push_back(
            time_inline_ptx<false, true, false, false, true>(
                left, right, device_joins, candidate_results, joins.size(),
                0));
        row_factor_shuffle_times.push_back(
            time_inline_ptx<true, true, false, false, true>(
                left, right, device_joins, candidate_results, joins.size(),
                0));
        unit_fast_times.push_back(
            time_inline_ptx<false, true, false, true>(
                left, right, device_joins, candidate_results, joins.size(),
                0));
        decomposed_times.push_back(time_nonunit_decomposed(
            left, right, left_nonunit, right_nonunit, device_joins,
            candidate_results, joins.size()));
        partitioned_times.push_back(time_unit_partitioned(
            left_unit, right_unit, left_nonunit, right_nonunit, device_joins,
            candidate_results, joins.size()));
    }
    double scalar = median(scalar_times);
    double direct = median(direct_times);
    double shuffle = median(shuffle_times);
    double direct_predicate = median(direct_predicate_times);
    double shuffle_predicate = median(shuffle_predicate_times);
    double row_factor_direct = median(row_factor_direct_times);
    double row_factor_shuffle = median(row_factor_shuffle_times);
    double unit_fast = median(unit_fast_times);
    double decomposed = median(decomposed_times);
    double partitioned = median(partitioned_times);
    std::printf(
        "PTX_BASE exact=OK scalar=%.6f direct=%.6f shuffle=%.6f "
        "direct_predicate=%.6f shuffle_predicate=%.6f "
        "row_factor_direct=%.6f row_factor_shuffle=%.6f "
        "unit_fast=%.6f decomposed=%.6f partitioned=%.6f "
        "direct_speedup=%.6f "
        "shuffle_speedup=%.6f row_factor_direct_speedup=%.6f "
        "row_factor_shuffle_speedup=%.6f unit_fast_speedup=%.6f "
        "decomposed_speedup=%.6f partitioned_speedup=%.6f\n",
        scalar, direct, shuffle, direct_predicate, shuffle_predicate,
        row_factor_direct, row_factor_shuffle, unit_fast, decomposed,
        partitioned, scalar / direct, scalar / shuffle,
        scalar / row_factor_direct, scalar / row_factor_shuffle,
        scalar / unit_fast, scalar / decomposed, scalar / partitioned);

    constexpr std::array<uint64_t, 9> thresholds = {
        0, 128, 512, 2048, 8192, 32768, 131072, 524288, UINT64_MAX};
    for (uint64_t threshold : thresholds) {
        std::vector<double> times;
        time_inline_ptx<false, true, true>(
            left, right, device_joins, candidate_results, joins.size(),
            threshold);
        require_results(expected, candidate_results, "hybrid-direct");
        for (int repeat = 0; repeat < repeats; repeat++) {
            times.push_back(time_inline_ptx<false, true, true>(
                left, right, device_joins, candidate_results, joins.size(),
                threshold));
        }
        double elapsed = median(times);
        std::printf(
            "PTX_HYBRID prep=direct threshold=%llu exact=OK time=%.6f "
            "speedup=%.6f\n",
            (unsigned long long)threshold, elapsed, scalar / elapsed);
    }
    for (uint64_t threshold : thresholds) {
        std::vector<double> times;
        time_inline_ptx<true, true, true>(
            left, right, device_joins, candidate_results, joins.size(),
            threshold);
        require_results(expected, candidate_results, "hybrid-shuffle");
        for (int repeat = 0; repeat < repeats; repeat++) {
            times.push_back(time_inline_ptx<true, true, true>(
                left, right, device_joins, candidate_results, joins.size(),
                threshold));
        }
        double elapsed = median(times);
        std::printf(
            "PTX_HYBRID prep=shuffle threshold=%llu exact=OK time=%.6f "
            "speedup=%.6f\n",
            (unsigned long long)threshold, elapsed, scalar / elapsed);
    }

    free_nonunit_layout(right_unit);
    free_nonunit_layout(left_unit);
    free_nonunit_layout(right_nonunit);
    free_nonunit_layout(left_nonunit);
    CUDA_CHECK(cudaFree(candidate_results));
    CUDA_CHECK(cudaFree(scalar_results));
    CUDA_CHECK(cudaFree(device_joins));
    free_device_prefix_layout(right);
    free_device_prefix_layout(left);
    free_production_canonical(canonical);
    return 0;
}
