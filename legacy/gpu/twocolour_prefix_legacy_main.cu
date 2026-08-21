#ifndef GRID_ROWS
#define GRID_ROWS 7
#define GRID_COLUMNS 7
#define LEFT_COLUMNS 3
#define RIGHT_COLUMNS 4
#define ORBIT_ROW_BITS 8
#define ORBIT_MAGIC "R7SQT01"
#endif

#include "../../twocolour_prefix_core.cuh"
#include "twocolour_prefix_legacy_helpers.cuh"

#ifdef HIERARCHICAL_PREFIX
#include "twocolour_prefix_hierarchy.cuh"
using LegacyPackedLayouts = HierarchyPackedLayouts;
static std::vector<PrefixPair> build_legacy_prefix_layout(
    const std::vector<PrefixKey>& keys, const CanonicalFactory& factory,
    LegacyPackedLayouts& layouts) {
    return build_hierarchy_prefix_layout(keys, factory, layouts);
}
#else
#include "twocolour_prefix_legacy_layout.cuh"
using LegacyPackedLayouts = PackedLayouts;
static std::vector<PrefixPair> build_legacy_prefix_layout(
    const std::vector<PrefixKey>& keys, const CanonicalFactory& factory,
    LegacyPackedLayouts& layouts) {
    return build_prefix_layout(keys, factory, layouts);
}
#endif

#ifndef PREFIX_PRODUCTION_LIBRARY
int main(int argc, char** argv) {
    if (argc < 2 || argc > 8) {
        std::fprintf(stderr,
                     "Usage: %s ORBITS [START=0] [END=100000] [REPEATS=3] "
                     "[VERIFY_JOINS=32] [JOIN_COPIES=1] [RANDOM_KERNELS=0]\n",
                     argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    uint64_t start_record = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 0;
    uint64_t end_record = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 100000;
    unsigned repeats = argc > 4 ? unsigned(std::strtoul(argv[4], nullptr, 10)) : 3;
    size_t verify_joins = argc > 5 ? std::strtoull(argv[5], nullptr, 10) : 32;
    unsigned join_copies = argc > 6 ? unsigned(std::strtoul(argv[6], nullptr, 10)) : 1;
    size_t random_kernels = argc > 7 ? std::strtoull(argv[7], nullptr, 10) : 0;
    if (!repeats || !join_copies) return 2;

    double total_start = seconds_now();
    initialise_tables();
    validate_mask_split();
    U128 labelled_weight = 0;
    uint64_t records = 0;
    std::vector<Edge> edges = random_kernels
                                  ? read_random_edges(path, random_kernels,
                                                      labelled_weight, records)
                                  : read_edges(path, start_record, end_record, 0, 0,
                                               labelled_weight, records);
    if (edges.empty()) throw std::runtime_error("record range produced no joins");
    std::vector<PrefixKey> left_keys = unique_lefts(edges);
    std::vector<PrefixKey> right_keys = unique_rights(edges);
    double build_start = seconds_now();
    std::unique_ptr<CanonicalFactory> shared_factory;
    std::unique_ptr<CanonicalFactory> left_owned_factory;
    std::unique_ptr<CanonicalFactory> right_owned_factory;
    CanonicalFactory* left_factory = nullptr;
    CanonicalFactory* right_factory = nullptr;
    if (LEFT_COLUMNS == RIGHT_COLUMNS) {
        std::vector<PrefixKey> all_keys = left_keys;
        all_keys.insert(all_keys.end(), right_keys.begin(), right_keys.end());
        shared_factory = std::make_unique<CanonicalFactory>(
            build_canonical_factory(std::move(all_keys), LEFT_COLUMNS));
        left_factory = shared_factory.get();
        right_factory = shared_factory.get();
    } else {
        left_owned_factory = std::make_unique<CanonicalFactory>(
            build_canonical_factory(left_keys, LEFT_COLUMNS));
        right_owned_factory = std::make_unique<CanonicalFactory>(
            build_canonical_factory(right_keys, RIGHT_COLUMNS));
        left_factory = left_owned_factory.get();
        right_factory = right_owned_factory.get();
    }
    double factory_seconds = seconds_now() - build_start;
    double layout_start = seconds_now();
    LegacyPackedLayouts left_layouts;
    LegacyPackedLayouts right_layouts;
    std::vector<PrefixPair> left_pairs =
        build_legacy_prefix_layout(left_keys, *left_factory, left_layouts);
    std::vector<PrefixPair> right_pairs =
        build_legacy_prefix_layout(right_keys, *right_factory, right_layouts);
    double layout_seconds = seconds_now() - layout_start;
    double build_seconds = seconds_now() - build_start;

    std::unordered_map<PrefixKey, uint32_t> left_index;
    std::unordered_map<PrefixKey, uint32_t> right_index;
    for (size_t index = 0; index < left_keys.size(); index++) {
        left_index[left_keys[index]] = uint32_t(index);
    }
    for (size_t index = 0; index < right_keys.size(); index++) {
        right_index[right_keys[index]] = uint32_t(index);
    }

    std::vector<JoinDesc> direct_joins;
    std::vector<PrefixJoinDesc> prefix_joins;
    std::vector<uint64_t> direct_join_work;
    std::vector<uint64_t> prefix_join_work;
    direct_joins.reserve(edges.size() * 2);
    prefix_joins.reserve(edges.size() * 2);
    direct_join_work.reserve(edges.size() * 2);
    prefix_join_work.reserve(edges.size() * 2);
    U128 direct_comparisons = 0;
    U128 suffix_comparisons = 0;
    U128 bucket_pairs = 0;
    U128 compatible_bucket_pairs = 0;
    std::vector<uint64_t> right_comparison_work(right_keys.size());
    for (const Edge& edge : edges) {
        const PrefixPair& left = left_pairs[left_index.at(edge.left)];
        size_t right_id = right_index.at(edge.right);
        const PrefixPair& right = right_pairs[right_id];
        const PrefixDistribution left_distributions[2] = {left.selected, left.complement};
        const PrefixDistribution right_distributions[2] = {right.selected, right.complement};
        for (int complement = 0; complement < 2; complement++) {
            const PrefixDistribution& lhs = left_distributions[complement];
            const PrefixDistribution& rhs = right_distributions[complement];
            direct_joins.push_back(make_join(lhs.direct_offset, rhs.direct_offset,
                                             lhs.entry_count, rhs.entry_count));
            prefix_joins.push_back(PrefixJoinDesc{lhs.bucket_offset, rhs.bucket_offset,
                                                  lhs.bucket_count, rhs.bucket_count});
            uint64_t this_direct_work = uint64_t(lhs.entry_count) * rhs.entry_count;
            uint64_t this_prefix_work = 0;
            direct_join_work.push_back(this_direct_work);
            direct_comparisons += this_direct_work;
            bucket_pairs += U128(lhs.bucket_count) * rhs.bucket_count;
            for (uint32_t li = 0; li < lhs.bucket_count; li++) {
                const PrefixBucket& lb = left_layouts.buckets[lhs.bucket_offset + li];
                for (uint32_t ri = 0; ri < rhs.bucket_count; ri++) {
                    const PrefixBucket& rb = right_layouts.buckets[rhs.bucket_offset + ri];
                    if (lb.prefix & rb.prefix) continue;
#ifdef HIERARCHICAL_PREFIX
                    for (uint32_t lli = 0; lli < lb.count; lli++) {
                        const PrefixBucket& llb =
                            left_layouts.leaf_buckets[lb.entry_offset + lli];
                        for (uint32_t rri = 0; rri < rb.count; rri++) {
                            const PrefixBucket& rrb =
                                right_layouts.leaf_buckets[rb.entry_offset + rri];
                            if (llb.prefix & rrb.prefix) continue;
                            compatible_bucket_pairs++;
                            uint64_t comparison_work =
                                uint64_t(llb.count) * rrb.count;
                            this_prefix_work += comparison_work;
                            suffix_comparisons += comparison_work;
                            if (UINT64_MAX - right_comparison_work[right_id] <
                                comparison_work) {
                                throw std::overflow_error(
                                    "per-prefix comparison work exceeds uint64_t");
                            }
                            right_comparison_work[right_id] += comparison_work;
                        }
                    }
#else
                    compatible_bucket_pairs++;
                    uint64_t comparison_work = uint64_t(lb.count) * rb.count;
                    this_prefix_work += comparison_work;
                    suffix_comparisons += comparison_work;
                    if (UINT64_MAX - right_comparison_work[right_id] <
                        comparison_work) {
                        throw std::overflow_error(
                            "per-prefix comparison work exceeds uint64_t");
                    }
                    right_comparison_work[right_id] += comparison_work;
#endif
                }
            }
            prefix_join_work.push_back(this_prefix_work);
        }
    }
    const size_t original_join_count = direct_joins.size();
#ifdef HEAVY_FIRST_JOINS
    std::vector<uint32_t> direct_schedule(original_join_count);
    std::vector<uint32_t> prefix_schedule(original_join_count);
    std::iota(direct_schedule.begin(), direct_schedule.end(), 0U);
    std::iota(prefix_schedule.begin(), prefix_schedule.end(), 0U);
    auto heavier = [](const std::vector<uint64_t>& work, uint32_t lhs,
                      uint32_t rhs) {
        return work[lhs] != work[rhs] ? work[lhs] > work[rhs] : lhs < rhs;
    };
    std::sort(direct_schedule.begin(), direct_schedule.end(),
              [&](uint32_t lhs, uint32_t rhs) {
                  return heavier(direct_join_work, lhs, rhs);
              });
    std::sort(prefix_schedule.begin(), prefix_schedule.end(),
              [&](uint32_t lhs, uint32_t rhs) {
                  return heavier(prefix_join_work, lhs, rhs);
              });
    std::vector<JoinDesc> scheduled_direct(original_join_count);
    std::vector<PrefixJoinDesc> scheduled_prefix(original_join_count);
    for (size_t index = 0; index < original_join_count; index++) {
        scheduled_direct[index] = direct_joins[direct_schedule[index]];
        scheduled_prefix[index] = prefix_joins[prefix_schedule[index]];
    }
    direct_joins.swap(scheduled_direct);
    prefix_joins.swap(scheduled_prefix);
#else
    std::vector<uint32_t> direct_schedule(original_join_count);
    std::vector<uint32_t> prefix_schedule(original_join_count);
    std::iota(direct_schedule.begin(), direct_schedule.end(), 0U);
    std::iota(prefix_schedule.begin(), prefix_schedule.end(), 0U);
#endif
    const std::vector<JoinDesc> original_direct_joins = direct_joins;
    const std::vector<PrefixJoinDesc> original_prefix_joins = prefix_joins;
    direct_joins.reserve(original_join_count * join_copies);
    prefix_joins.reserve(original_join_count * join_copies);
    for (unsigned copy = 1; copy < join_copies; copy++) {
        direct_joins.insert(direct_joins.end(), original_direct_joins.begin(),
                            original_direct_joins.end());
        prefix_joins.insert(prefix_joins.end(), original_prefix_joins.begin(),
                            original_prefix_joins.end());
    }
    direct_comparisons *= join_copies;
    suffix_comparisons *= join_copies;
    bucket_pairs *= join_copies;
    compatible_bucket_pairs *= join_copies;

    std::printf("INPUT records=%llu kernels=%zu unique_left=%zu unique_right=%zu\n",
                (unsigned long long)records, edges.size(), left_keys.size(), right_keys.size());
#ifdef HIERARCHICAL_PREFIX
    std::printf("LAYOUT left_entries=%zu right_entries=%zu left_buckets=%zu "
                "right_buckets=%zu left_leaf_buckets=%zu "
                "right_leaf_buckets=%zu left_canonical_entries=%zu "
                "right_canonical_entries=%zu factory_seconds=%.6f "
                "layout_seconds=%.6f build_seconds=%.6f\n",
                left_layouts.direct_masks.size(), right_layouts.direct_masks.size(),
                left_layouts.buckets.size(), right_layouts.buckets.size(),
                left_layouts.leaf_buckets.size(),
                right_layouts.leaf_buckets.size(), left_factory->entries.size(),
                right_factory->entries.size(), factory_seconds, layout_seconds,
                build_seconds);
#else
    std::printf("LAYOUT left_entries=%zu right_entries=%zu left_buckets=%zu "
                "right_buckets=%zu left_canonical_entries=%zu "
                "right_canonical_entries=%zu factory_seconds=%.6f "
                "layout_seconds=%.6f build_seconds=%.6f\n",
                left_layouts.direct_masks.size(), right_layouts.direct_masks.size(),
                left_layouts.buckets.size(), right_layouts.buckets.size(),
                left_factory->entries.size(), right_factory->entries.size(),
                factory_seconds, layout_seconds, build_seconds);
#endif
    std::printf("WORK direct=%s suffix=%s retained=%.9f bucket_pairs=%s "
                "compatible_bucket_pairs=%s\n",
                u128_string(direct_comparisons).c_str(),
                u128_string(suffix_comparisons).c_str(),
                double(suffix_comparisons) / double(direct_comparisons),
                u128_string(bucket_pairs).c_str(),
                u128_string(compatible_bucket_pairs).c_str());

    uint64_t* device_left_masks = upload_vector(left_layouts.direct_masks);
    uint32_t* device_left_weights = upload_vector(left_layouts.direct_weights);
    uint64_t* device_right_masks = upload_vector(right_layouts.direct_masks);
    uint32_t* device_right_weights = upload_vector(right_layouts.direct_weights);
    PrefixEntry* device_left_prefix_entries = upload_vector(left_layouts.prefix_entries);
    PrefixEntry* device_right_prefix_entries = upload_vector(right_layouts.prefix_entries);
    PrefixBucket* device_left_buckets = upload_vector(left_layouts.buckets);
    PrefixBucket* device_right_buckets = upload_vector(right_layouts.buckets);
#ifdef HIERARCHICAL_PREFIX
    PrefixBucket* device_left_leaf_buckets =
        upload_vector(left_layouts.leaf_buckets);
    PrefixBucket* device_right_leaf_buckets =
        upload_vector(right_layouts.leaf_buckets);
#endif
    JoinDesc* device_direct_joins = upload_vector(direct_joins);
    PrefixJoinDesc* device_prefix_joins = upload_vector(prefix_joins);
    unsigned long long* device_direct_results = nullptr;
    unsigned long long* device_prefix_results = nullptr;
    CUDA_CHECK(cudaMalloc(&device_direct_results,
                          direct_joins.size() * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&device_prefix_results,
                          prefix_joins.size() * sizeof(unsigned long long)));

    cudaEvent_t event_start, event_end;
    CUDA_CHECK(cudaEventCreate(&event_start));
    CUDA_CHECK(cudaEventCreate(&event_end));
    double direct_seconds = 0;
    double prefix_seconds = 0;
    for (unsigned repeat = 0; repeat <= repeats; repeat++) {
        CUDA_CHECK(cudaEventRecord(event_start));
        fused_disjoint_joins_soa<<<unsigned(direct_joins.size()), THREADS>>>(
            device_left_masks, device_left_weights, device_right_masks,
            device_right_weights, device_direct_joins, device_direct_results);
        CUDA_CHECK(cudaGetLastError());
        double seconds = elapsed_kernel(event_start, event_end);
        if (repeat) direct_seconds += seconds;

        CUDA_CHECK(cudaEventRecord(event_start));
#ifdef HIERARCHICAL_PREFIX
        hierarchy_disjoint_joins<<<unsigned(prefix_joins.size()), THREADS>>>(
            device_left_prefix_entries, device_right_prefix_entries,
            device_left_buckets, device_right_buckets,
            device_left_leaf_buckets, device_right_leaf_buckets,
            device_prefix_joins, device_prefix_results);
#else
        prefix_disjoint_joins<<<unsigned(prefix_joins.size()), THREADS>>>(
            device_left_prefix_entries, device_right_prefix_entries,
            device_left_buckets, device_right_buckets, device_prefix_joins,
            device_prefix_results);
#endif
        CUDA_CHECK(cudaGetLastError());
        seconds = elapsed_kernel(event_start, event_end);
        if (repeat) prefix_seconds += seconds;
    }
    direct_seconds /= repeats;
    prefix_seconds /= repeats;

    std::vector<unsigned long long> direct_results(direct_joins.size());
    std::vector<unsigned long long> prefix_results(prefix_joins.size());
    CUDA_CHECK(cudaMemcpy(direct_results.data(), device_direct_results,
                          direct_results.size() * sizeof(direct_results[0]),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(prefix_results.data(), device_prefix_results,
                          prefix_results.size() * sizeof(prefix_results[0]),
                          cudaMemcpyDeviceToHost));
    std::vector<unsigned long long> logical_direct_results(direct_results.size());
    std::vector<unsigned long long> logical_prefix_results(prefix_results.size());
    for (unsigned copy = 0; copy < join_copies; copy++) {
        size_t base = size_t(copy) * original_join_count;
        for (size_t index = 0; index < original_join_count; index++) {
            logical_direct_results[base + direct_schedule[index]] =
                direct_results[base + index];
            logical_prefix_results[base + prefix_schedule[index]] =
                prefix_results[base + index];
        }
    }
    direct_results.swap(logical_direct_results);
    prefix_results.swap(logical_prefix_results);
    if (direct_results != prefix_results) {
        for (size_t index = 0; index < direct_results.size(); index++) {
            if (direct_results[index] != prefix_results[index]) {
                std::fprintf(stderr, "GPU mismatch join=%zu direct=%llu prefix=%llu\n",
                             index, direct_results[index], prefix_results[index]);
                break;
            }
        }
        return 1;
    }
    verify_joins = std::min(verify_joins, original_join_count);
    for (size_t index = 0; index < verify_joins; index++) {
        const Edge& edge = edges[index / 2];
        DistributionPair left = build_pair(edge.left, LEFT_COLUMNS);
        DistributionPair right = build_pair(edge.right, RIGHT_COLUMNS);
        const Distribution& lhs = index & 1 ? left.complement : left.selected;
        const Distribution& rhs = index & 1 ? right.complement : right.selected;
        uint64_t expected = cpu_join(lhs.entries.data(), lhs.entries.size(),
                                     rhs.entries.data(), rhs.entries.size());
        if (direct_results[index] != expected) {
            std::fprintf(stderr, "CPU mismatch join=%zu gpu=%llu cpu=%llu\n", index,
                         direct_results[index], (unsigned long long)expected);
            return 1;
        }
    }

#if defined(GPU_PREFIX_BUILDER) && !defined(STREAMED_RIGHT_PREFIX_PROBE)
    DevicePrefixLayout gpu_left_layout =
        build_device_prefix_layout(left_keys, *left_factory);
    DevicePrefixLayout gpu_right_layout =
        build_device_prefix_layout(right_keys, *right_factory);
    std::vector<PrefixJoinDesc> gpu_joins;
    gpu_joins.reserve(original_join_count);
    for (const Edge& edge : edges) {
        const PrefixPair& left = gpu_left_layout.pairs[left_index.at(edge.left)];
        const PrefixPair& right = gpu_right_layout.pairs[right_index.at(edge.right)];
        const PrefixDistribution left_distributions[2] = {left.selected,
                                                          left.complement};
        const PrefixDistribution right_distributions[2] = {right.selected,
                                                           right.complement};
        for (int complement = 0; complement < 2; complement++) {
            const PrefixDistribution& lhs = left_distributions[complement];
            const PrefixDistribution& rhs = right_distributions[complement];
            gpu_joins.push_back(PrefixJoinDesc{lhs.bucket_offset, rhs.bucket_offset,
                                               lhs.bucket_count, rhs.bucket_count});
        }
    }
    const std::vector<PrefixJoinDesc> original_gpu_joins = gpu_joins;
    gpu_joins.reserve(original_join_count * join_copies);
    for (unsigned copy = 1; copy < join_copies; copy++) {
        gpu_joins.insert(gpu_joins.end(), original_gpu_joins.begin(),
                         original_gpu_joins.end());
    }
    PrefixJoinDesc* device_gpu_joins = upload_vector(gpu_joins);
    unsigned long long* device_gpu_results = nullptr;
    CUDA_CHECK(cudaMalloc(&device_gpu_results,
                          gpu_joins.size() * sizeof(unsigned long long)));
    CUDA_CHECK(cudaEventRecord(event_start));
    prefix_disjoint_joins<<<unsigned(gpu_joins.size()), THREADS>>>(
        gpu_left_layout.entries, gpu_right_layout.entries,
        gpu_left_layout.buckets, gpu_right_layout.buckets,
        device_gpu_joins, device_gpu_results);
    CUDA_CHECK(cudaGetLastError());
    double gpu_built_join_seconds = elapsed_kernel(event_start, event_end);
    std::vector<unsigned long long> gpu_built_results(gpu_joins.size());
    CUDA_CHECK(cudaMemcpy(gpu_built_results.data(), device_gpu_results,
                          gpu_built_results.size() * sizeof(gpu_built_results[0]),
                          cudaMemcpyDeviceToHost));
    if (gpu_built_results != direct_results) {
        throw std::runtime_error("GPU-built prefix layout result mismatch");
    }
    std::printf("GPU_BUILD left_entries=%zu right_entries=%zu left_buckets=%zu "
                "right_buckets=%zu histogram_seconds=%.6f scatter_seconds=%.6f "
                "metadata_seconds=%.6f plan_seconds=%.6f upload_seconds=%.6f "
                "left_total_seconds=%.6f right_total_seconds=%.6f "
                "total_seconds=%.6f join_seconds=%.6f exact=OK\n",
                gpu_left_layout.entry_count, gpu_right_layout.entry_count,
                gpu_left_layout.bucket_count, gpu_right_layout.bucket_count,
                gpu_left_layout.histogram_seconds + gpu_right_layout.histogram_seconds,
                gpu_left_layout.scatter_seconds + gpu_right_layout.scatter_seconds,
                gpu_left_layout.metadata_seconds + gpu_right_layout.metadata_seconds,
                gpu_left_layout.plan_seconds + gpu_right_layout.plan_seconds,
                gpu_left_layout.upload_seconds + gpu_right_layout.upload_seconds,
                gpu_left_layout.total_seconds, gpu_right_layout.total_seconds,
                gpu_left_layout.total_seconds + gpu_right_layout.total_seconds,
                gpu_built_join_seconds);
    CUDA_CHECK(cudaFree(device_gpu_results));
    CUDA_CHECK(cudaFree(device_gpu_joins));
    free_device_prefix_layout(gpu_right_layout);
    free_device_prefix_layout(gpu_left_layout);
#endif

#ifdef STREAMED_RIGHT_PREFIX_PROBE
    if (join_copies != 1) {
        throw std::runtime_error("streamed-right probe requires JOIN_COPIES=1");
    }
    if (STREAM_RIGHT_GROUPS < 1) {
        throw std::runtime_error("STREAM_RIGHT_GROUPS must be positive");
    }

    double stream_total_start = seconds_now();
    DevicePrefixLayout streamed_left =
        build_device_prefix_layout(left_keys, *left_factory);

    std::vector<uint64_t> right_entry_work(right_keys.size());
    uint64_t remaining_work = 0;
    for (size_t index = 0; index < right_keys.size(); index++) {
        const PrefixPair& pair = right_pairs[index];
        right_entry_work[index] = uint64_t(pair.selected.entry_count) +
                                  pair.complement.entry_count;
        if (UINT64_MAX - remaining_work < right_comparison_work[index]) {
            throw std::overflow_error("stream comparison work exceeds uint64_t");
        }
        remaining_work += right_comparison_work[index];
    }

    const size_t wanted_groups =
        std::min<size_t>(STREAM_RIGHT_GROUPS, right_keys.size());
    size_t right_begin = 0;
    size_t edge_cursor = 0;
    size_t groups_built = 0;
    uint64_t minimum_group_work = UINT64_MAX;
    uint64_t maximum_group_work = 0;
    uint64_t minimum_group_entries = UINT64_MAX;
    uint64_t maximum_group_entries = 0;
    size_t maximum_group_keys = 0;
    size_t maximum_right_entries = 0;
    size_t maximum_right_buckets = 0;
    size_t maximum_right_canonical_entries = 0;
    size_t summed_right_canonical_entries = 0;
    double right_factory_seconds = 0;
    double right_build_seconds = 0;
    double histogram_seconds = streamed_left.histogram_seconds;
    double scatter_seconds = streamed_left.scatter_seconds;
    double metadata_seconds = streamed_left.metadata_seconds;
    double plan_seconds = streamed_left.plan_seconds;
    double upload_seconds = streamed_left.upload_seconds;
    double streamed_join_seconds = 0;

    while (right_begin < right_keys.size()) {
        size_t groups_left = wanted_groups - groups_built;
        size_t keys_that_must_remain = groups_left - 1;
        uint64_t target = (remaining_work + groups_left - 1) / groups_left;
        size_t right_end = right_begin;
        uint64_t group_work = 0;
        uint64_t group_entries = 0;
        while (right_end < right_keys.size() - keys_that_must_remain &&
               (right_end == right_begin || group_work < target)) {
            group_work += right_comparison_work[right_end];
            group_entries += right_entry_work[right_end++];
        }
        std::vector<PrefixKey> group_keys(right_keys.begin() + right_begin,
                                          right_keys.begin() + right_end);

        double factory_start = seconds_now();
        CanonicalFactory group_factory =
            build_canonical_factory(group_keys, RIGHT_COLUMNS);
        right_factory_seconds += seconds_now() - factory_start;
        summed_right_canonical_entries += group_factory.entries.size();
        maximum_right_canonical_entries =
            std::max(maximum_right_canonical_entries, group_factory.entries.size());

        DevicePrefixLayout streamed_right =
            build_device_prefix_layout(group_keys, group_factory);
        right_build_seconds += streamed_right.total_seconds;
        histogram_seconds += streamed_right.histogram_seconds;
        scatter_seconds += streamed_right.scatter_seconds;
        metadata_seconds += streamed_right.metadata_seconds;
        plan_seconds += streamed_right.plan_seconds;
        upload_seconds += streamed_right.upload_seconds;
        maximum_right_entries =
            std::max(maximum_right_entries, streamed_right.entry_count);
        maximum_right_buckets =
            std::max(maximum_right_buckets, streamed_right.bucket_count);
        maximum_group_keys = std::max(maximum_group_keys, group_keys.size());
        minimum_group_work = std::min(minimum_group_work, group_work);
        maximum_group_work = std::max(maximum_group_work, group_work);
        minimum_group_entries = std::min(minimum_group_entries, group_entries);
        maximum_group_entries = std::max(maximum_group_entries, group_entries);

        size_t edge_begin = edge_cursor;
        PrefixKey final_right = group_keys.back();
        while (edge_cursor < edges.size() && edges[edge_cursor].right <= final_right) {
            edge_cursor++;
        }
        std::vector<PrefixJoinDesc> streamed_joins;
        streamed_joins.reserve((edge_cursor - edge_begin) * 2);
        for (size_t edge_index = edge_begin; edge_index < edge_cursor; edge_index++) {
            const Edge& edge = edges[edge_index];
            auto right_found = std::lower_bound(group_keys.begin(), group_keys.end(),
                                                edge.right);
            if (right_found == group_keys.end() || *right_found != edge.right) {
                throw std::runtime_error("streamed right-key ownership mismatch");
            }
            size_t right_index_in_group = size_t(right_found - group_keys.begin());
            const PrefixPair& left = streamed_left.pairs[left_index.at(edge.left)];
            const PrefixPair& right = streamed_right.pairs[right_index_in_group];
            const PrefixDistribution left_distributions[2] = {left.selected,
                                                              left.complement};
            const PrefixDistribution right_distributions[2] = {right.selected,
                                                               right.complement};
            for (int complement = 0; complement < 2; complement++) {
                const PrefixDistribution& lhs = left_distributions[complement];
                const PrefixDistribution& rhs = right_distributions[complement];
                streamed_joins.push_back(PrefixJoinDesc{
                    lhs.bucket_offset, rhs.bucket_offset, lhs.bucket_count,
                    rhs.bucket_count});
            }
        }

        PrefixJoinDesc* device_streamed_joins = upload_vector(streamed_joins);
        unsigned long long* device_streamed_results = nullptr;
        CUDA_CHECK(cudaMalloc(&device_streamed_results,
                              streamed_joins.size() * sizeof(unsigned long long)));
        CUDA_CHECK(cudaEventRecord(event_start));
        prefix_disjoint_joins<<<unsigned(streamed_joins.size()), THREADS>>>(
            streamed_left.entries, streamed_right.entries, streamed_left.buckets,
            streamed_right.buckets, device_streamed_joins, device_streamed_results);
        CUDA_CHECK(cudaGetLastError());
        streamed_join_seconds += elapsed_kernel(event_start, event_end);
        std::vector<unsigned long long> streamed_results(streamed_joins.size());
        CUDA_CHECK(cudaMemcpy(streamed_results.data(), device_streamed_results,
                              streamed_results.size() * sizeof(streamed_results[0]),
                              cudaMemcpyDeviceToHost));
        for (size_t index = 0; index < streamed_results.size(); index++) {
            size_t expected_index = edge_begin * 2 + index;
            if (streamed_results[index] != direct_results[expected_index]) {
                throw std::runtime_error("streamed right-prefix result mismatch");
            }
        }

        CUDA_CHECK(cudaFree(device_streamed_results));
        CUDA_CHECK(cudaFree(device_streamed_joins));
        free_device_prefix_layout(streamed_right);
        remaining_work -= group_work;
        right_begin = right_end;
        groups_built++;
    }
    if (edge_cursor != edges.size() || groups_built != wanted_groups ||
        remaining_work != 0) {
        throw std::runtime_error("streamed right-prefix partition incomplete");
    }
    free_device_prefix_layout(streamed_left);
    std::printf(
        "STREAM_BUILD groups=%zu comparison_work_min=%llu "
        "comparison_work_max=%llu expanded_entries_min=%llu "
        "expanded_entries_max=%llu max_group_keys=%zu left_entries=%zu "
        "left_buckets=%zu "
        "max_right_entries=%zu max_right_buckets=%zu "
        "sum_right_canonical_entries=%zu max_right_canonical_entries=%zu "
        "left_total_seconds=%.6f right_factory_seconds=%.6f "
        "right_build_seconds=%.6f histogram_seconds=%.6f "
        "scatter_seconds=%.6f metadata_seconds=%.6f plan_seconds=%.6f "
        "upload_seconds=%.6f join_seconds=%.6f total_seconds=%.6f exact=OK\n",
        groups_built, (unsigned long long)minimum_group_work,
        (unsigned long long)maximum_group_work,
        (unsigned long long)minimum_group_entries,
        (unsigned long long)maximum_group_entries, maximum_group_keys,
        streamed_left.entry_count, streamed_left.bucket_count, maximum_right_entries,
        maximum_right_buckets, summed_right_canonical_entries,
        maximum_right_canonical_entries, streamed_left.total_seconds,
        right_factory_seconds, right_build_seconds, histogram_seconds,
        scatter_seconds, metadata_seconds, plan_seconds, upload_seconds,
        streamed_join_seconds, seconds_now() - stream_total_start);
#endif

    std::printf("RESULT joins=%zu verified=%zu exact=OK direct_seconds=%.6f "
                "prefix_seconds=%.6f speedup=%.6f direct_Tcomparisons=%.6f "
                "suffix_Tcomparisons=%.6f total_seconds=%.6f\n",
                direct_results.size(), verify_joins, direct_seconds, prefix_seconds,
                direct_seconds / prefix_seconds,
                double(direct_comparisons) / direct_seconds / 1e12,
                double(suffix_comparisons) / prefix_seconds / 1e12,
                seconds_now() - total_start);

    CUDA_CHECK(cudaEventDestroy(event_end));
    CUDA_CHECK(cudaEventDestroy(event_start));
    CUDA_CHECK(cudaFree(device_prefix_results));
    CUDA_CHECK(cudaFree(device_direct_results));
    CUDA_CHECK(cudaFree(device_prefix_joins));
    CUDA_CHECK(cudaFree(device_direct_joins));
#ifdef HIERARCHICAL_PREFIX
    CUDA_CHECK(cudaFree(device_right_leaf_buckets));
    CUDA_CHECK(cudaFree(device_left_leaf_buckets));
#endif
    CUDA_CHECK(cudaFree(device_right_buckets));
    CUDA_CHECK(cudaFree(device_left_buckets));
    CUDA_CHECK(cudaFree(device_right_prefix_entries));
    CUDA_CHECK(cudaFree(device_left_prefix_entries));
    CUDA_CHECK(cudaFree(device_right_weights));
    CUDA_CHECK(cudaFree(device_right_masks));
    CUDA_CHECK(cudaFree(device_left_weights));
    CUDA_CHECK(cudaFree(device_left_masks));
    return 0;
}
#endif
