#pragma once

// Rejected exact suffix-bitplane join prototype retained for reproducibility.
#ifndef SUFFIX_BITPLANE_MIN_COUNT
#define SUFFIX_BITPLANE_MIN_COUNT 257
#endif

__global__ void build_prefix_bucket_bitplanes(
    const PrefixEntry* __restrict__ entries,
    const PrefixBucket* __restrict__ buckets,
    const uint32_t* __restrict__ plane_offsets, uint32_t bucket_count,
    uint64_t* __restrict__ planes,
    unsigned long long* __restrict__ bucket_weight_sums) {
    uint32_t bucket_index = blockIdx.x;
    if (bucket_index >= bucket_count) return;
    PrefixBucket bucket = buckets[bucket_index];
    unsigned long long weight_sum = 0;
    for (uint32_t index = threadIdx.x; index < bucket.count;
         index += blockDim.x) {
        weight_sum += entries[bucket.entry_offset + index].weight;
    }
    __shared__ unsigned long long partial[THREADS];
    partial[threadIdx.x] = weight_sum;
    __syncthreads();
    for (unsigned stride = THREADS / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (!threadIdx.x) bucket_weight_sums[bucket_index] = partial[0];
    uint32_t plane_offset = plane_offsets[bucket_index];
    if (plane_offset == UINT32_MAX) return;
    unsigned lane = threadIdx.x & 31U;
    unsigned warp = threadIdx.x >> 5;
    uint32_t words = (bucket.count + 63) / 64;
    for (uint32_t word = warp; word < words; word += THREADS / 32) {
        uint32_t first_index = word * 64 + lane;
        uint32_t second_index = first_index + 32;
        uint32_t first_suffix =
            first_index < bucket.count
                ? entries[bucket.entry_offset + first_index].suffix
                : 0;
        uint32_t second_suffix =
            second_index < bucket.count
                ? entries[bucket.entry_offset + second_index].suffix
                : 0;
#pragma unroll
        for (unsigned bit = 0; bit < 32; bit++) {
            uint32_t low = __ballot_sync(UINT32_MAX,
                                         first_suffix & (UINT32_C(1) << bit));
            uint32_t high = __ballot_sync(UINT32_MAX,
                                          second_suffix & (UINT32_C(1) << bit));
            if (!lane) {
                planes[size_t(plane_offset) + size_t(bit) * words + word] =
                    uint64_t(low) | (uint64_t(high) << 32);
            }
        }
    }
}

__global__ void prefix_disjoint_joins_bitplanes(
    const PrefixEntry* __restrict__ left_entries,
    const PrefixEntry* __restrict__ right_entries,
    const PrefixBucket* __restrict__ left_buckets,
    const PrefixBucket* __restrict__ right_buckets,
    const PrefixJoinDesc* __restrict__ joins,
    const uint32_t* __restrict__ right_plane_offsets,
    const uint64_t* __restrict__ right_planes,
    const unsigned long long* __restrict__ right_weight_sums,
    unsigned long long* __restrict__ results) {
    __shared__ unsigned long long warp_partial[8];
    __shared__ PrefixEntry warp_tiles[THREADS];
    __shared__ uint32_t next_task;
    const PrefixJoinDesc join = joins[blockIdx.x];
    const unsigned lane = threadIdx.x & 31U;
    const unsigned warp = threadIdx.x >> 5;
    const uint32_t task_count = join.left_bucket_count * join.right_bucket_count;
    unsigned long long sum = 0;

    if (!threadIdx.x) next_task = 0;
    __syncthreads();
    uint32_t task_base = 0;
    if (!lane) task_base = atomicAdd(&next_task, uint32_t(PREFIX_TASK_CHUNK));
    task_base = __shfl_sync(UINT32_MAX, task_base, 0);
    while (task_base < task_count) {
        uint32_t task_end = min(task_count, task_base + uint32_t(PREFIX_TASK_CHUNK));
        uint32_t left_bucket_index = task_base / join.right_bucket_count;
        uint32_t right_bucket_index =
            task_base - left_bucket_index * join.right_bucket_count;
        uint32_t cached_left_bucket_index = UINT32_MAX;
        PrefixBucket left_bucket{};
        for (uint32_t task = task_base; task < task_end; task++) {
            if (left_bucket_index != cached_left_bucket_index) {
                left_bucket =
                    left_buckets[join.left_bucket_offset + left_bucket_index];
                cached_left_bucket_index = left_bucket_index;
            }
            uint32_t global_right = join.right_bucket_offset + right_bucket_index;
            PrefixBucket right_bucket = right_buckets[global_right];
            if (!(left_bucket.prefix & right_bucket.prefix)) {
                uint32_t plane_offset = right_plane_offsets[global_right];
                if (plane_offset != UINT32_MAX) {
                    uint32_t words = (right_bucket.count + 63) / 64;
                    for (uint32_t left_base = 0; left_base < left_bucket.count;
                         left_base += 32) {
                        uint32_t left_index = left_base + lane;
                        if (left_index < left_bucket.count) {
                            PrefixEntry left =
                                left_entries[left_bucket.entry_offset + left_index];
                            unsigned long long right_weight = 0;
                            if (!left.suffix) {
                                right_weight = right_weight_sums[global_right];
                            } else {
                                for (uint32_t word = 0; word < words; word++) {
                                    uint64_t blocked = 0;
                                    uint32_t bits = left.suffix;
                                    while (bits) {
                                        unsigned bit = __ffs(bits) - 1;
                                        blocked |= right_planes[
                                            size_t(plane_offset) +
                                            size_t(bit) * words + word];
                                        bits &= bits - 1;
                                    }
                                    uint64_t compatible = ~blocked;
                                    if (word + 1 == words &&
                                        (right_bucket.count & 63U)) {
                                        compatible &=
                                            (UINT64_C(1) <<
                                             (right_bucket.count & 63U)) -
                                            1;
                                    }
                                    while (compatible) {
                                        unsigned offset = __ffsll(compatible) - 1;
                                        right_weight +=
                                            right_entries[
                                                right_bucket.entry_offset +
                                                word * 64 + offset]
                                                .weight;
                                        compatible &= compatible - 1;
                                    }
                                }
                            }
                            sum += uint64_t(left.weight) * right_weight;
                        }
                    }
                } else {
                    const PrefixEntry* lhs_entries = left_entries;
                    const PrefixEntry* rhs_entries = right_entries;
                    PrefixBucket lhs = left_bucket;
                    PrefixBucket rhs = right_bucket;
                    if (rhs.count < lhs.count) {
                        lhs_entries = right_entries;
                        rhs_entries = left_entries;
                        PrefixBucket temporary = lhs;
                        lhs = rhs;
                        rhs = temporary;
                    }
                    for (uint32_t left_base = 0; left_base < lhs.count;
                         left_base += 32) {
                        uint32_t left_index = left_base + lane;
                        PrefixEntry left =
                            left_index < lhs.count
                                ? lhs_entries[lhs.entry_offset + left_index]
                                : PrefixEntry{0, 0};
                        for (uint32_t right_base = 0; right_base < rhs.count;
                             right_base += 32) {
                            uint32_t right_index = right_base + lane;
                            warp_tiles[warp * 32 + lane] =
                                right_index < rhs.count
                                    ? rhs_entries[rhs.entry_offset + right_index]
                                    : PrefixEntry{0, 0};
                            __syncwarp();
                            uint32_t count =
                                min(uint32_t(32), rhs.count - right_base);
#pragma unroll
                            for (uint32_t offset = 0; offset < count; offset++) {
                                PrefixEntry right =
                                    warp_tiles[warp * 32 + offset];
                                if (left_index < lhs.count &&
                                    !(left.suffix & right.suffix)) {
                                    sum += uint64_t(left.weight) *
                                           uint64_t(right.weight);
                                }
                            }
                            __syncwarp();
                        }
                    }
                }
            }
            right_bucket_index++;
            if (right_bucket_index == join.right_bucket_count) {
                right_bucket_index = 0;
                left_bucket_index++;
            }
        }
        if (!lane) task_base = atomicAdd(&next_task, uint32_t(PREFIX_TASK_CHUNK));
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
