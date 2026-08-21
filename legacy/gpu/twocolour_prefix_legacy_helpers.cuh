#pragma once

// Historical scalar prefix-join benchmark support. Production solvers use the
// grouped BMMA implementation and must not depend on this header.
#include <unordered_set>

#ifndef PREFIX_LHS_PER_THREAD
#define PREFIX_LHS_PER_THREAD 1
#endif
#if PREFIX_LHS_PER_THREAD != 1
#error "token-plane quotient requires PREFIX_LHS_PER_THREAD == 1"
#endif
#ifndef STREAM_RIGHT_GROUPS
#define STREAM_RIGHT_GROUPS 4
#endif

__global__ void prefix_disjoint_joins(
    const PrefixEntry* __restrict__ left_entries,
    const PrefixEntry* __restrict__ right_entries,
    const PrefixBucket* __restrict__ left_buckets,
    const PrefixBucket* __restrict__ right_buckets,
    const PrefixJoinDesc* __restrict__ joins,
    unsigned long long* __restrict__ results
    ) {
    __shared__ unsigned long long warp_partial[8];
    __shared__ PrefixEntry warp_tiles[THREADS];
#if PREFIX_TASK_CHUNK > 0
    __shared__ uint32_t next_task;
#endif
    const PrefixJoinDesc join = joins[blockIdx.x];
    const unsigned lane = threadIdx.x & 31U;
    const unsigned warp = threadIdx.x >> 5;
    const uint32_t task_count = join.left_bucket_count * join.right_bucket_count;
    unsigned long long sums[PREFIX_LHS_PER_THREAD]{};

#if PREFIX_TASK_CHUNK > 0
    if (!threadIdx.x) next_task = 0;
    __syncthreads();
    uint32_t task_base = 0;
    if (!lane) task_base = atomicAdd(&next_task, uint32_t(PREFIX_TASK_CHUNK));
    task_base = __shfl_sync(UINT32_MAX, task_base, 0);
#else
    uint32_t task_base = warp;
#endif
    while (task_base < task_count) {
#if PREFIX_TASK_CHUNK > 0
        uint32_t task_end = min(task_count, task_base + uint32_t(PREFIX_TASK_CHUNK));
        uint32_t left_bucket_index = task_base / join.right_bucket_count;
        uint32_t right_bucket_index =
            task_base - left_bucket_index * join.right_bucket_count;
        uint32_t cached_left_bucket_index = UINT32_MAX;
        PrefixBucket left_bucket{};
#else
        uint32_t task_end = task_base + 1;
#endif
        for (uint32_t task = task_base; task < task_end; task++) {
#if PREFIX_TASK_CHUNK == 0
        uint32_t left_bucket_index = task / join.right_bucket_count;
        uint32_t right_bucket_index = task - left_bucket_index * join.right_bucket_count;
#endif
#if PREFIX_TASK_CHUNK > 0
        if (left_bucket_index != cached_left_bucket_index) {
            left_bucket = left_buckets[join.left_bucket_offset + left_bucket_index];
            cached_left_bucket_index = left_bucket_index;
        }
#else
        PrefixBucket left_bucket = left_buckets[join.left_bucket_offset + left_bucket_index];
#endif
        PrefixBucket right_bucket =
            right_buckets[join.right_bucket_offset + right_bucket_index];
        bool forward_prefix = !(left_bucket.prefix & right_bucket.prefix);
        bool swapped_prefix =
            !(left_bucket.prefix &
              swap_prefix_token_planes(right_bucket.prefix));

        if (forward_prefix || swapped_prefix) {
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
                 left_base += 32 * PREFIX_LHS_PER_THREAD) {
            PrefixEntry left[PREFIX_LHS_PER_THREAD];
            bool left_valid[PREFIX_LHS_PER_THREAD];
#pragma unroll
            for (int item = 0; item < PREFIX_LHS_PER_THREAD; item++) {
                uint32_t left_index = left_base + lane + 32 * item;
                left_valid[item] = left_index < lhs.count;
                left[item] = left_valid[item]
                                 ? lhs_entries[lhs.entry_offset + left_index]
                                 : PrefixEntry{0, 0};
            }
                for (uint32_t right_base = 0; right_base < rhs.count; right_base += 32) {
                uint32_t right_index = right_base + lane;
                warp_tiles[warp * 32 + lane] =
                    right_index < rhs.count
                        ? rhs_entries[rhs.entry_offset + right_index]
                        : PrefixEntry{0, 0};
                __syncwarp();
                uint32_t count = min(uint32_t(32), rhs.count - right_base);
#pragma unroll
                for (uint32_t offset = 0; offset < count; offset++) {
                    PrefixEntry right = warp_tiles[warp * 32 + offset];
#pragma unroll
                    for (int item = 0; item < PREFIX_LHS_PER_THREAD; item++) {
                        if (left_valid[item]) {
                            const uint32_t left_orbit_size =
                                left[item].weight & PREFIX_ENTRY_ORBIT_TWO ? 2U : 1U;
                            const uint32_t right_orbit_size =
                                right.weight & PREFIX_ENTRY_ORBIT_TWO ? 2U : 1U;
                            const uint32_t left_weight =
                                left[item].weight & ~PREFIX_ENTRY_ORBIT_TWO;
                            const uint32_t right_weight =
                                right.weight & ~PREFIX_ENTRY_ORBIT_TWO;
                            uint64_t compatible =
                                forward_prefix &&
                                !(left[item].suffix & right.suffix);
                            if (right_orbit_size == 2 && swapped_prefix) {
                                compatible +=
                                    !(left[item].suffix &
                                      swap_suffix_token_planes(right.suffix));
                            }
                            sums[item] += compatible * left_orbit_size *
                                          uint64_t(left_weight) * right_weight;
                        }
                    }
                }
                    __syncwarp();
                }
            }
        }
#if PREFIX_TASK_CHUNK > 0
        right_bucket_index++;
        if (right_bucket_index == join.right_bucket_count) {
            right_bucket_index = 0;
            left_bucket_index++;
        }
#endif
#if PREFIX_TASK_CHUNK > 0
        }
        if (!lane) task_base = atomicAdd(&next_task, uint32_t(PREFIX_TASK_CHUNK));
        task_base = __shfl_sync(UINT32_MAX, task_base, 0);
#else
        }
        task_base += 8;
#endif
    }

    unsigned long long sum = 0;
#pragma unroll
    for (int item = 0; item < PREFIX_LHS_PER_THREAD; item++) sum += sums[item];
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

static std::vector<Edge> read_random_edges(const std::string& path, size_t wanted,
                                           U128& labelled_weight, uint64_t& records) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path);
    char magic[8];
    uint32_t columns;
    uint64_t count;
    input.read(magic, 8);
    input.read(reinterpret_cast<char*>(&columns), sizeof(columns));
    input.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!input || std::memcmp(magic, ORBIT_MAGIC, 7) || columns != COLUMNS) {
        throw std::runtime_error("invalid orbit file");
    }
    std::vector<Edge> edges;
    edges.reserve(wanted);
    std::unordered_set<uint64_t> used;
    used.reserve(wanted * 2);
    uint64_t state = UINT64_C(0x7265637437783931);
    while (edges.size() < wanted) {
        state = mix64(state + UINT64_C(0x9e3779b97f4a7c15));
        uint64_t index = state % count;
        if (!used.insert(index).second) continue;
        input.seekg(std::streamoff(20 + index * sizeof(OrbitRecord)));
        OrbitRecord record{};
        input.read(reinterpret_cast<char*>(&record), sizeof(record));
        if (!input) throw std::runtime_error("random orbit read failed");
        int cells = cell_count(record.key);
        if (cells * 2 > CELLS) continue;
        uint8_t factor = cells * 2 < CELLS ? 2 : 1;
        edges.push_back(Edge{left_prefix(record.key), right_prefix(record.key),
                             record.weight, factor});
        labelled_weight += record.weight;
        records++;
    }
    std::sort(edges.begin(), edges.end(), [](const Edge& lhs, const Edge& rhs) {
        if (lhs.right != rhs.right) return lhs.right < rhs.right;
        return lhs.left < rhs.left;
    });
    return edges;
}
