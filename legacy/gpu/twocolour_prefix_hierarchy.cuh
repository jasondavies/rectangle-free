#pragma once

#ifndef HIERARCHICAL_PREFIX
#error "the historical hierarchy module requires HIERARCHICAL_PREFIX"
#endif

// Historical two-level prefix layout and join prototype.  Production solvers
// use the single-level prefix implementation in twocolour_prefix_core.cuh.
#ifndef HIERARCHY_SECOND_PAIR_COUNT
#define HIERARCHY_SECOND_PAIR_COUNT 4
#endif
#if GRID_ROWS != 8
#error "the hierarchical prefix prototype currently targets the 8x8 kernel"
#endif
#if PREFIX_PAIR_COUNT + HIERARCHY_SECOND_PAIR_COUNT > \
    (GRID_ROWS * (GRID_ROWS - 1) / 2)
#error "hierarchical prefix selects more row-pair coordinates than exist"
#endif

struct HierarchyBucket {
    uint32_t leaf_offset;
    uint32_t leaf_count;
    uint16_t prefix;
    uint16_t reserved;
};
static_assert(sizeof(HierarchyBucket) == 12,
              "hierarchy bucket descriptors must be compact");
#ifdef HIERARCHY_COMPATIBLE_CHILDREN
constexpr uint32_t HIERARCHY_CHILD_PREFIX_COUNT =
    uint32_t(1) << (2 * HIERARCHY_SECOND_PAIR_COUNT);
static_assert(HIERARCHY_CHILD_PREFIX_COUNT <= 256,
              "warp-local child lookup uses 16-bit ordinals");
#endif

static __host__ __device__ void split_hierarchy_mask(
    uint64_t mask, uint16_t& first, uint16_t& second, PrefixSuffix& suffix) {
    first = 0;
    second = 0;
    suffix = 0;
    int first_bit = 0;
    int second_bit = 0;
    int suffix_bit = 0;
    for (int colour = 0; colour < 2; colour++) {
        for (int pair = 0; pair < PAIRS; pair++) {
            PrefixSuffix bit =
                PrefixSuffix((mask >> (colour * PAIRS + pair)) & 1U);
            int rank = prefix_pair_rank(pair);
            if (rank < PREFIX_PAIR_COUNT) {
                first |= uint16_t(bit << first_bit++);
            } else if (rank < PREFIX_PAIR_COUNT + HIERARCHY_SECOND_PAIR_COUNT) {
                second |= uint16_t(bit << second_bit++);
            } else {
                suffix |= bit << suffix_bit++;
            }
        }
    }
}

struct HierarchyPackedLayouts {
    std::vector<uint64_t> direct_masks;
    std::vector<uint32_t> direct_weights;
    std::vector<PrefixEntry> prefix_entries;
    std::vector<PrefixBucket> buckets;
    std::vector<PrefixBucket> leaf_buckets;
};

static std::vector<PrefixPair> build_hierarchy_prefix_layout(
    const std::vector<PrefixKey>& keys, const CanonicalFactory& factory,
    HierarchyPackedLayouts& layouts) {
    struct WorkItem {
        CanonicalRef reference;
        uint64_t entry_offset;
        uint32_t entry_count;
        std::vector<HierarchyBucket> buckets;
        std::vector<PrefixBucket> leaf_buckets;
    };
    struct UnorderedEntry {
        uint16_t prefix;
        uint16_t second;
        PrefixSuffix suffix;
        uint32_t weight;
    };
    constexpr size_t prefix_count = size_t(1) << (2 * PREFIX_PAIR_COUNT);
    constexpr size_t second_count =
        size_t(1) << (2 * HIERARCHY_SECOND_PAIR_COUNT);

    std::vector<WorkItem> work(keys.size() * 2);
    uint64_t entry_count = 0;
    for (size_t index = 0; index < keys.size(); index++) {
        const RawCanonicalPair& raw = lookup_raw(factory, keys[index]);
        const CanonicalRef references[2] = {raw.selected, raw.complement};
        for (int complement = 0; complement < 2; complement++) {
            WorkItem& item = work[index * 2 + complement];
            item.reference = references[complement];
            item.entry_offset = entry_count;
            item.entry_count = factory.descriptors[item.reference.distribution].count;
            entry_count += item.entry_count;
        }
    }
    if (entry_count > uint64_t(UINT32_MAX) + 1) {
        throw std::overflow_error("prefix entries exceed 32-bit address space");
    }
    layouts.direct_masks.resize(entry_count);
    layouts.direct_weights.resize(entry_count);
    layouts.prefix_entries.resize(entry_count);

#pragma omp parallel for schedule(dynamic, 8)
    for (long long work_index = 0; work_index < (long long)work.size(); work_index++) {
        WorkItem& item = work[size_t(work_index)];
        const CanonicalDescriptor& source =
            factory.descriptors[item.reference.distribution];
        std::vector<UnorderedEntry> unordered(source.count);
        std::vector<uint32_t> counts(prefix_count);
        for (uint32_t index = 0; index < source.count; index++) {
            const Entry& canonical = factory.entries[source.offset + index];
            uint64_t mask = transform_pair_mask(canonical.mask, item.reference.row_map);
            uint32_t weight = uint32_t(canonical.weight);
            uint16_t prefix;
            PrefixSuffix suffix;
            uint16_t second;
            split_hierarchy_mask(mask, prefix, second, suffix);
            size_t destination = size_t(item.entry_offset) + index;
            layouts.direct_masks[destination] = mask;
            layouts.direct_weights[destination] = weight;
            unordered[index] = UnorderedEntry{prefix, second, suffix, weight};
            counts[prefix]++;
        }
        std::vector<uint32_t> positions(prefix_count);
        uint32_t running = 0;
        for (size_t prefix = 0; prefix < prefix_count; prefix++) {
            positions[prefix] = running;
            if (!counts[prefix]) continue;
            running += counts[prefix];
        }
        std::vector<UnorderedEntry> first_grouped(source.count);
        for (const UnorderedEntry& entry : unordered) {
            first_grouped[positions[entry.prefix]++] = entry;
        }
        running = 0;
        std::vector<uint32_t> second_counts(second_count);
        std::vector<uint32_t> second_positions(second_count);
        for (size_t prefix = 0; prefix < prefix_count; prefix++) {
            uint32_t first_count = counts[prefix];
            if (!first_count) continue;
            std::fill(second_counts.begin(), second_counts.end(), 0);
            for (uint32_t index = 0; index < first_count; index++) {
                second_counts[first_grouped[running + index].second]++;
            }
            uint32_t second_running = 0;
            uint32_t leaf_offset = uint32_t(item.leaf_buckets.size());
            for (size_t second = 0; second < second_count; second++) {
                second_positions[second] = second_running;
                if (!second_counts[second]) continue;
                item.leaf_buckets.push_back(PrefixBucket{
                    uint32_t(item.entry_offset + running + second_running),
                    second_counts[second], uint16_t(second), 0});
                second_running += second_counts[second];
            }
            item.buckets.push_back(HierarchyBucket{
                leaf_offset,
                uint32_t(item.leaf_buckets.size()) - leaf_offset,
                uint16_t(prefix), 0});
            for (uint32_t index = 0; index < first_count; index++) {
                const UnorderedEntry& entry = first_grouped[running + index];
                uint32_t destination =
                    running + second_positions[entry.second]++;
                layouts.prefix_entries[size_t(item.entry_offset) + destination] =
                    PrefixEntry{entry.suffix, entry.weight};
            }
            running += first_count;
        }
    }

    std::vector<PrefixPair> result(keys.size());
    for (size_t index = 0; index < keys.size(); index++) {
        PrefixDistribution* distributions[2] = {&result[index].selected,
                                                &result[index].complement};
        for (int complement = 0; complement < 2; complement++) {
            WorkItem& item = work[index * 2 + complement];
            if (layouts.buckets.size() + item.buckets.size() >
                    size_t(UINT32_MAX) + 1 ||
                layouts.leaf_buckets.size() + item.leaf_buckets.size() >
                    size_t(UINT32_MAX) + 1) {
                throw std::overflow_error(
                    "hierarchical prefix buckets exceed 32-bit address space");
            }
            uint32_t leaf_base = uint32_t(layouts.leaf_buckets.size());
            for (HierarchyBucket bucket : item.buckets) {
                bucket.leaf_offset += leaf_base;
                layouts.buckets.push_back(PrefixBucket{
                    bucket.leaf_offset, bucket.leaf_count, bucket.prefix, 0});
            }
            layouts.leaf_buckets.insert(layouts.leaf_buckets.end(),
                                        item.leaf_buckets.begin(),
                                        item.leaf_buckets.end());
            PrefixDistribution& distribution = *distributions[complement];
            distribution.direct_offset = item.entry_offset;
            distribution.entry_count = item.entry_count;
            distribution.bucket_offset = uint32_t(layouts.buckets.size());
            distribution.bucket_offset -= uint32_t(item.buckets.size());
            distribution.bucket_count = uint32_t(item.buckets.size());
        }
    }
    return result;
}

// Return one lane's partial weighted sum for an exact leaf-bucket join.  The
// caller owns warp-level scheduling and the final block reduction.
#ifdef HIERARCHY_COMPATIBLE_CHILDREN
static __device__ __forceinline__ unsigned long long
hierarchy_leaf_bucket_join(
    const PrefixEntry* __restrict__ left_entries, PrefixBucket left_bucket,
    const PrefixEntry* __restrict__ right_entries, PrefixBucket right_bucket,
    PrefixEntry* __restrict__ warp_tile, unsigned lane) {
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
    unsigned long long sums[PREFIX_LHS_PER_THREAD]{};
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
        for (uint32_t right_base = 0; right_base < rhs.count;
             right_base += 32) {
            uint32_t right_index = right_base + lane;
            warp_tile[lane] = right_index < rhs.count
                                  ? rhs_entries[rhs.entry_offset + right_index]
                                  : PrefixEntry{0, 0};
            __syncwarp();
            uint32_t count = min(uint32_t(32), rhs.count - right_base);
#pragma unroll
            for (uint32_t offset = 0; offset < count; offset++) {
                PrefixEntry right = warp_tile[offset];
#pragma unroll
                for (int item = 0; item < PREFIX_LHS_PER_THREAD; item++) {
                    if (left_valid[item] &&
                        !(left[item].suffix & right.suffix)) {
                        sums[item] += uint64_t(left[item].weight) *
                                      uint64_t(right.weight);
                    }
                }
            }
            __syncwarp();
        }
    }
    unsigned long long sum = 0;
#pragma unroll
    for (int item = 0; item < PREFIX_LHS_PER_THREAD; item++) sum += sums[item];
    return sum;
}
#endif

// A sparse two-level traversal: reject incompatible first prefixes, then join
// compatible occupied children on the remaining suffix bits.
__global__ void hierarchy_disjoint_joins(
    const PrefixEntry* __restrict__ left_entries,
    const PrefixEntry* __restrict__ right_entries,
    const PrefixBucket* __restrict__ left_first,
    const PrefixBucket* __restrict__ right_first,
    const PrefixBucket* __restrict__ left_leaves,
    const PrefixBucket* __restrict__ right_leaves,
    const PrefixJoinDesc* __restrict__ joins,
    unsigned long long* __restrict__ results) {
    __shared__ unsigned long long warp_partial[8];
    __shared__ PrefixEntry warp_tiles[THREADS];
    __shared__ uint32_t next_task;
#ifdef HIERARCHY_COMPATIBLE_CHILDREN
    __shared__ uint16_t
        child_lookup[8 * HIERARCHY_CHILD_PREFIX_COUNT];
#endif
    const PrefixJoinDesc join = joins[blockIdx.x];
    const unsigned lane = threadIdx.x & 31U;
    const unsigned warp = threadIdx.x >> 5;
    const uint32_t task_count =
        join.left_bucket_count * join.right_bucket_count;
    unsigned long long sums[PREFIX_LHS_PER_THREAD]{};

    if (!threadIdx.x) next_task = 0;
    __syncthreads();
    uint32_t task_base = 0;
    if (!lane) task_base = atomicAdd(&next_task, uint32_t(PREFIX_TASK_CHUNK));
    task_base = __shfl_sync(UINT32_MAX, task_base, 0);
    while (task_base < task_count) {
        uint32_t task_end = min(task_count,
                                task_base + uint32_t(PREFIX_TASK_CHUNK));
        for (uint32_t task = task_base; task < task_end; task++) {
            uint32_t left_first_index = task / join.right_bucket_count;
            uint32_t right_first_index =
                task - left_first_index * join.right_bucket_count;
            PrefixBucket left_parent =
                left_first[join.left_bucket_offset + left_first_index];
            PrefixBucket right_parent =
                right_first[join.right_bucket_offset + right_first_index];
            if (left_parent.prefix & right_parent.prefix) continue;

#ifdef HIERARCHY_COMPATIBLE_CHILDREN
            uint16_t* warp_lookup =
                &child_lookup[warp * HIERARCHY_CHILD_PREFIX_COUNT];
            for (uint32_t child = lane;
                 child < HIERARCHY_CHILD_PREFIX_COUNT; child += 32) {
                warp_lookup[child] = UINT16_MAX;
            }
            __syncwarp();
            for (uint32_t child = lane; child < right_parent.count;
                 child += 32) {
                PrefixBucket right_bucket =
                    right_leaves[right_parent.entry_offset + child];
                warp_lookup[right_bucket.prefix] = uint16_t(child);
            }
            __syncwarp();
            for (uint32_t left_leaf_index = 0;
                 left_leaf_index < left_parent.count; left_leaf_index++) {
                PrefixBucket left_bucket =
                    left_leaves[left_parent.entry_offset + left_leaf_index];
                uint32_t available =
                    (HIERARCHY_CHILD_PREFIX_COUNT - 1U) ^ left_bucket.prefix;
                uint32_t compatible = available;
                while (true) {
                    uint16_t right_leaf_index = warp_lookup[compatible];
                    if (right_leaf_index != UINT16_MAX) {
                        PrefixBucket right_bucket = right_leaves[
                            right_parent.entry_offset + right_leaf_index];
                        sums[0] += hierarchy_leaf_bucket_join(
                            left_entries, left_bucket, right_entries,
                            right_bucket, &warp_tiles[warp * 32], lane);
                    }
                    if (!compatible) break;
                    compatible = (compatible - 1U) & available;
                }
            }
            __syncwarp();
#else
            uint32_t leaf_task_count = left_parent.count * right_parent.count;
            for (uint32_t leaf_task = 0; leaf_task < leaf_task_count;
                 leaf_task++) {
                uint32_t left_leaf_index = leaf_task / right_parent.count;
                uint32_t right_leaf_index =
                    leaf_task - left_leaf_index * right_parent.count;
                PrefixBucket left_bucket =
                    left_leaves[left_parent.entry_offset + left_leaf_index];
                PrefixBucket right_bucket =
                    right_leaves[right_parent.entry_offset + right_leaf_index];
                if (left_bucket.prefix & right_bucket.prefix) continue;

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
                            PrefixEntry right = warp_tiles[warp * 32 + offset];
#pragma unroll
                            for (int item = 0; item < PREFIX_LHS_PER_THREAD;
                                 item++) {
                                if (left_valid[item] &&
                                    !(left[item].suffix & right.suffix)) {
                                    sums[item] += uint64_t(left[item].weight) *
                                                  uint64_t(right.weight);
                                }
                            }
                        }
                        __syncwarp();
                    }
                }
            }
#endif
        }
        if (!lane) {
            task_base = atomicAdd(&next_task, uint32_t(PREFIX_TASK_CHUNK));
        }
        task_base = __shfl_sync(UINT32_MAX, task_base, 0);
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
