// Production four-owner 7x9 solver.  Four left-owner layouts remain resident
// while each union right batch is constructed once and joined independently
// against every owner that uses it.  The included production library selects
// the same direct grouped suffix layout and weight-class BMMA kernel as the
// single-owner path.
#include "twocolour_7x9_engine.cuh"

constexpr size_t FOUR_OWNER_COUNT = 4;

struct FourOwnerState {
    PackedWorkItem item;
    PackedWorkResult result;
    std::vector<Edge> edges;
    std::vector<PrefixKey> left_keys;
    std::vector<PrefixKey> right_keys;
    std::vector<uint32_t> edge_left_ids;
    PackedSolveLayout left_layout;
};

struct FourOwnerSharedStats {
    uint64_t union_right_prefixes = 0;
    uint64_t right_batches = 0;
    uint64_t maximum_right_entries = 0;
    uint64_t maximum_right_buckets = 0;
    uint64_t effective_right_entry_cap = 0;
    uint64_t completed_kernels = 0;
    uint64_t right_source_entries = 0;
    uint64_t right_source_chunks = 0;
    uint64_t pipeline_max_source_entries = 0;
    size_t minimum_free_bytes = SIZE_MAX;
    double right_layout_seconds = 0;
    double right_source_plan_seconds = 0;
    double right_source_sort_seconds = 0;
    double right_schedule_seconds = 0;
    double right_plan_seconds = 0;
    double right_upload_seconds = 0;
    double right_source_gather_seconds = 0;
    double right_source_upload_seconds = 0;
    double right_histogram_seconds = 0;
    double right_scatter_seconds = 0;
    double right_metadata_seconds = 0;
    double pipeline_wait_seconds = 0;
    double gpu_seconds = 0;
};

static void release_four_owner_layouts(std::vector<FourOwnerState>& owners) {
    for (FourOwnerState& owner : owners) {
        free_packed_solve_layout(owner.left_layout);
    }
}

static void load_four_owner(FourOwnerState& owner, size_t reserve_bytes) {
    double load_start = seconds_now();
    owner.edges = read_edges(
        owner.item.path, owner.item.start, owner.item.end,
        owner.item.filter_mod, owner.item.filter_id,
        owner.result.labelled_weight, owner.result.records);
    owner.result.load_seconds = seconds_now() - load_start;
    owner.result.kernels = owner.edges.size();
    if (owner.edges.empty()) {
        throw std::runtime_error("four-owner tile does not accept empty work items");
    }
    owner.left_keys = unique_lefts(owner.edges);
    owner.right_keys = unique_rights(owner.edges);
    owner.edge_left_ids = resolve_edge_left_ids(owner.edges, owner.left_keys);
    owner.result.left_prefixes = owner.left_keys.size();
    owner.result.right_prefixes = owner.right_keys.size();

    double factory_start = seconds_now();
    CanonicalFactory factory =
        build_canonical_factory(owner.left_keys, LEFT_COLUMNS);
    owner.result.left_factory_seconds = seconds_now() - factory_start;

    uint64_t estimated_entries = 0;
    for (PrefixKey key : owner.left_keys) {
        const RawCanonicalPair& raw = lookup_raw(factory, key);
        estimated_entries += factory.descriptors[raw.selected.distribution].count;
        estimated_entries += factory.descriptors[raw.complement.distribution].count;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    uint64_t estimated_peak =
        estimated_entries * PACKED_SOLVE_ENTRY_BYTES +
        factory.entries.size() * sizeof(Entry);
    if (free_bytes <= reserve_bytes ||
        estimated_peak > free_bytes - reserve_bytes) {
        throw std::runtime_error("four-owner left layout exceeds memory budget");
    }

    owner.left_layout = build_direct_left_layout(owner.left_keys, factory);
    owner.result.left_layout_seconds = owner.left_layout.total_seconds;
    owner.result.left_entries = owner.left_layout.entry_count;
    owner.result.left_buckets = packed_solve_bucket_count(owner.left_layout);
}

static std::vector<PrefixKey> union_right_keys(
    const std::vector<FourOwnerState>& owners) {
    size_t total = 0;
    for (const FourOwnerState& owner : owners) total += owner.right_keys.size();
    std::vector<PrefixKey> result;
    result.reserve(total);
    for (const FourOwnerState& owner : owners) {
        result.insert(result.end(), owner.right_keys.begin(),
                      owner.right_keys.end());
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

static FourOwnerSharedStats solve_four_owner_tile(
    std::vector<FourOwnerState>& owners, const PackedUniversalCache& cache,
    uint64_t requested_right_entries, uint64_t verify_joins) {
    if (owners.size() != FOUR_OWNER_COUNT) {
        throw std::runtime_error("four-owner solver requires exactly four items");
    }
    FourOwnerSharedStats shared;
    for (FourOwnerState& owner : owners) {
        load_four_owner(owner, cache.memory_reserve_bytes);
        mark_validation_edges(owner.edges, verify_joins);
    }

    size_t free_after_left = 0;
    size_t total_device_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after_left, &total_device_bytes));
    shared.minimum_free_bytes = free_after_left;
    if (free_after_left <= cache.memory_reserve_bytes) {
        release_four_owner_layouts(owners);
        throw std::runtime_error("no right-layout headroom after four left layouts");
    }
    // The persistent packed-source prefetch buffer is already charged in
    // free_after_left and remains live while the right output is materialised.
    uint64_t memory_entry_cap =
        (free_after_left - cache.memory_reserve_bytes) /
        PACKED_SOLVE_ENTRY_BYTES;
    uint64_t right_entry_cap = std::min(requested_right_entries, memory_entry_cap);
    right_entry_cap = std::min<uint64_t>(right_entry_cap, UINT32_MAX);
    if (!right_entry_cap) {
        release_four_owner_layouts(owners);
        throw std::runtime_error("four-owner right-layout entry cap is zero");
    }
    shared.effective_right_entry_cap = right_entry_cap;

    std::vector<PrefixKey> right_keys = union_right_keys(owners);
    shared.union_right_prefixes = right_keys.size();
    if (right_keys.size() > UINT32_MAX) {
        release_four_owner_layouts(owners);
        throw std::overflow_error("four-owner union right index exceeds uint32_t");
    }
    double source_plan_start = seconds_now();
    std::vector<PackedRawSourcePlan> right_plans(right_keys.size());
#pragma omp parallel for schedule(static)
    for (long long index = 0; index < (long long)right_keys.size(); index++) {
        right_plans[size_t(index)] =
            packed_raw_source_plan(cache, right_keys[size_t(index)]);
    }
    shared.right_source_plan_seconds = seconds_now() - source_plan_start;
    for (size_t index = 0; index < right_keys.size(); index++) {
        if (right_plans[index].entries > right_entry_cap) {
            release_four_owner_layouts(owners);
            throw std::runtime_error("one union right prefix exceeds memory cap");
        }
    }

    double schedule_start = seconds_now();
    std::vector<uint32_t> right_order(right_keys.size());
    std::iota(right_order.begin(), right_order.end(), uint32_t(0));
    double source_sort_start = seconds_now();
    std::sort(right_order.begin(), right_order.end(), [&](uint32_t lhs,
                                                           uint32_t rhs) {
        std::array<size_t, 2> left = right_plans[lhs].distributions;
        std::array<size_t, 2> right = right_plans[rhs].distributions;
        if (left[0] > left[1]) std::swap(left[0], left[1]);
        if (right[0] > right[1]) std::swap(right[0], right[1]);
        if (left != right) return left < right;
        return lhs < rhs;
    });
    shared.right_source_sort_seconds = seconds_now() - source_sort_start;

    std::vector<std::vector<uint32_t>> batch_right_indices;
    uint64_t batch_entries = 0;
    uint64_t batch_source_entries = 0;
    const uint64_t source_entry_cap = cache.device_resident
        ? UINT64_MAX
        : cache.prefetch_bytes / sizeof(uint64_t);
    std::unordered_set<size_t> batch_sources;
    batch_sources.reserve(1 << 16);
    for (uint32_t index : right_order) {
        uint64_t entries = right_plans[index].entries;
        auto additional_sources = [&]() {
            uint64_t additional = 0;
            for (size_t complement = 0; complement < 2; complement++) {
                size_t distribution =
                    right_plans[index].distributions[complement];
                if (complement && distribution ==
                                      right_plans[index].distributions[0]) {
                    continue;
                }
                if (!batch_sources.count(distribution)) {
                    additional += cache.counts[distribution];
                }
            }
            return additional;
        };
        uint64_t additional_source_entries = additional_sources();
        bool new_batch = batch_right_indices.empty() ||
                         (batch_entries &&
                          batch_entries + entries > right_entry_cap) ||
                         (batch_source_entries &&
                          batch_source_entries + additional_source_entries >
                              source_entry_cap);
        if (new_batch) {
            batch_right_indices.emplace_back();
            batch_entries = 0;
            batch_source_entries = 0;
            batch_sources.clear();
            additional_source_entries = additional_sources();
        }
        for (size_t distribution : right_plans[index].distributions) {
            batch_sources.insert(distribution);
        }
        batch_right_indices.back().push_back(index);
        batch_entries += entries;
        batch_source_entries += additional_source_entries;
        if (batch_source_entries > source_entry_cap) {
            release_four_owner_layouts(owners);
            throw std::runtime_error(
                "one union source pair exceeds prefetch capacity");
        }
    }
    if (batch_right_indices.size() >= UINT16_MAX) {
        release_four_owner_layouts(owners);
        throw std::overflow_error("too many four-owner right batches");
    }
    std::vector<uint16_t> right_batch(right_keys.size(), UINT16_MAX);
    for (size_t batch = 0; batch < batch_right_indices.size(); batch++) {
        std::sort(batch_right_indices[batch].begin(),
                  batch_right_indices[batch].end());
        for (uint32_t index : batch_right_indices[batch]) {
            if (right_batch[index] != UINT16_MAX) {
                release_four_owner_layouts(owners);
                throw std::runtime_error("duplicate union right ownership");
            }
            right_batch[index] = uint16_t(batch);
        }
    }

    std::vector<std::vector<std::vector<uint32_t>>> batch_edges(
        owners.size(), std::vector<std::vector<uint32_t>>(
                           batch_right_indices.size()));
    for (size_t owner_index = 0; owner_index < owners.size(); owner_index++) {
        const std::vector<Edge>& edges = owners[owner_index].edges;
        if (edges.size() > UINT32_MAX) {
            release_four_owner_layouts(owners);
            throw std::overflow_error("four-owner edge index exceeds uint32_t");
        }
        size_t right_cursor = 0;
        for (size_t edge_index = 0; edge_index < edges.size(); edge_index++) {
            while (right_cursor < right_keys.size() &&
                   right_keys[right_cursor] < edges[edge_index].right) {
                right_cursor++;
            }
            if (right_cursor == right_keys.size() ||
                right_keys[right_cursor] != edges[edge_index].right) {
                release_four_owner_layouts(owners);
                throw std::runtime_error("four-owner edge/right mismatch");
            }
            batch_edges[owner_index][right_batch[right_cursor]].push_back(
                uint32_t(edge_index));
        }
    }
    shared.right_schedule_seconds = seconds_now() - schedule_start;

    auto batch_references = [&](size_t batch_index) {
        std::vector<std::array<PackedLayoutSourceRef, 2>> references;
        references.reserve(batch_right_indices[batch_index].size());
        for (uint32_t index : batch_right_indices[batch_index]) {
            references.push_back(std::array<PackedLayoutSourceRef, 2>{
                PackedLayoutSourceRef{right_plans[index].distributions[0],
                                      right_plans[index].row_maps[0]},
                PackedLayoutSourceRef{right_plans[index].distributions[1],
                                      right_plans[index].row_maps[1]}});
        }
        return references;
    };
    PrefetchedPackedLayout prefetched;
    if (!cache.device_resident) {
        prefetched = prefetch_host_packed_layout(
            batch_references(0), cache);
        shared.pipeline_max_source_entries = prefetched.source_entries;
    }

    cudaEvent_t event_start, event_end;
    CUDA_CHECK(cudaEventCreate(&event_start));
    CUDA_CHECK(cudaEventCreate(&event_end));
    DirectWeightClassWorkspace weight_workspace;

    for (size_t batch_index = 0; batch_index < batch_right_indices.size();
         batch_index++) {
        const std::vector<uint32_t>& group_indices =
            batch_right_indices[batch_index];
        std::vector<PrefixKey> group_keys;
        group_keys.reserve(group_indices.size());
        uint64_t group_entries = 0;
        for (uint32_t index : group_indices) {
            group_keys.push_back(right_keys[index]);
            group_entries += right_plans[index].entries;
        }
        std::vector<std::array<PackedLayoutSourceRef, 2>> references =
            batch_references(batch_index);
        PackedSolveLayout right_layout = cache.device_resident
            ? build_direct_packed_weight_layout_from_resident(
                  references, cache, weight_workspace)
            : build_direct_packed_weight_layout_from_prefetched(
                  prefetched, cache, weight_workspace,
                  shared.pipeline_wait_seconds);
        shared.right_layout_seconds += right_layout.total_seconds;
        shared.right_plan_seconds += right_layout.plan_seconds;
        shared.right_upload_seconds += right_layout.upload_seconds;
        shared.right_source_gather_seconds += right_layout.source_gather_seconds;
        shared.right_source_upload_seconds += right_layout.source_upload_seconds;
        shared.right_histogram_seconds += right_layout.histogram_seconds;
        shared.right_scatter_seconds += right_layout.scatter_seconds;
        shared.right_metadata_seconds += right_layout.metadata_seconds;
        shared.right_source_entries += right_layout.source_entries;
        shared.right_source_chunks += right_layout.source_chunks;
        shared.maximum_right_entries =
            std::max<uint64_t>(shared.maximum_right_entries,
                               right_layout.entry_count);
        shared.maximum_right_buckets =
            std::max<uint64_t>(shared.maximum_right_buckets,
                               packed_solve_bucket_count(right_layout));
        size_t free_with_right = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_with_right, &total_device_bytes));
        shared.minimum_free_bytes =
            std::min(shared.minimum_free_bytes, free_with_right);

        bool next_prefetch_started = false;
        for (size_t owner_index = 0; owner_index < owners.size(); owner_index++) {
            FourOwnerState& owner = owners[owner_index];
            const std::vector<uint32_t>& edge_indices =
                batch_edges[owner_index][batch_index];
            if (edge_indices.empty()) continue;
            double join_plan_start = seconds_now();
            std::vector<PrefixJoinDesc> joins;
            joins.reserve(edge_indices.size() * 2);
            size_t group_cursor = 0;
            for (uint32_t edge_index : edge_indices) {
                const Edge& edge = owner.edges[edge_index];
                while (group_cursor < group_keys.size() &&
                       group_keys[group_cursor] < edge.right) {
                    group_cursor++;
                }
                if (group_cursor == group_keys.size() ||
                    group_keys[group_cursor] != edge.right) {
                    throw std::runtime_error("four-owner batch ownership mismatch");
                }
                const PrefixPair& left = owner.left_layout.pairs[
                    owner.edge_left_ids[edge_index]];
                const PrefixPair& right =
                    right_layout.pairs[group_cursor];
                const PrefixDistribution lhs[2] = {left.selected,
                                                   left.complement};
                const PrefixDistribution rhs[2] = {right.selected,
                                                   right.complement};
                for (int complement = 0; complement < 2; complement++) {
                    joins.push_back(PrefixJoinDesc{
                        lhs[complement].bucket_offset,
                        rhs[complement].bucket_offset,
                        lhs[complement].bucket_count,
                        rhs[complement].bucket_count});
                    owner.result.direct_comparisons +=
                        U128(lhs[complement].entry_count) *
                        rhs[complement].entry_count;
                }
            }

            double join_plan_end = seconds_now();
            owner.result.join_plan_seconds += join_plan_end - join_plan_start;
            double join_upload_start = seconds_now();
            PrefixJoinDesc* device_joins = upload_vector(joins);
            owner.result.join_upload_seconds +=
                seconds_now() - join_upload_start;
            unsigned long long* device_results = nullptr;
            double allocation_start = seconds_now();
            CUDA_CHECK(cudaMalloc(&device_results,
                                  joins.size() * sizeof(unsigned long long)));
            owner.result.result_allocation_seconds +=
                seconds_now() - allocation_start;
            CUDA_CHECK(cudaEventRecord(event_start));
            weight_class_prefix_joins<<<unsigned(joins.size()), THREADS>>>(
                owner.left_layout.suffixes.get(), right_layout.suffixes.get(),
                owner.left_layout.buckets.get(), right_layout.buckets.get(),
                owner.left_layout.classes.get(), right_layout.classes.get(),
                device_joins, device_results);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(event_end));
            if (!cache.device_resident && !next_prefetch_started &&
                batch_index + 1 < batch_right_indices.size()) {
                prefetched = prefetch_host_packed_layout(
                    batch_references(batch_index + 1), cache);
                shared.pipeline_max_source_entries = std::max(
                    shared.pipeline_max_source_entries,
                    prefetched.source_entries);
                next_prefetch_started = true;
            }
            CUDA_CHECK(cudaEventSynchronize(event_end));
            float kernel_milliseconds = 0;
            CUDA_CHECK(cudaEventElapsedTime(&kernel_milliseconds, event_start,
                                            event_end));
            owner.result.gpu_seconds += kernel_milliseconds / 1000.0;
            shared.gpu_seconds += kernel_milliseconds / 1000.0;

            std::vector<unsigned long long> results(joins.size());
            double download_start = seconds_now();
            CUDA_CHECK(cudaMemcpy(results.data(), device_results,
                                  results.size() * sizeof(results[0]),
                                  cudaMemcpyDeviceToHost));
            owner.result.result_download_seconds +=
                seconds_now() - download_start;
            for (size_t local_edge = 0; local_edge < edge_indices.size();
                 local_edge++) {
                const Edge& edge = owner.edges[edge_indices[local_edge]];
                uint64_t selected = results[local_edge * 2];
                uint64_t complement = results[local_edge * 2 + 1];
                if (edge.validate) {
                    double validation_start = seconds_now();
                    DistributionPair validation_left =
                        build_pair(edge.left, LEFT_COLUMNS);
                    DistributionPair validation_right =
                        build_pair(edge.right, RIGHT_COLUMNS);
                    uint64_t expected_selected = cpu_join(
                        validation_left.selected.entries.data(),
                        validation_left.selected.entries.size(),
                        validation_right.selected.entries.data(),
                        validation_right.selected.entries.size());
                    uint64_t expected_complement = cpu_join(
                        validation_left.complement.entries.data(),
                        validation_left.complement.entries.size(),
                        validation_right.complement.entries.data(),
                        validation_right.complement.entries.size());
                    owner.result.validation_seconds +=
                        seconds_now() - validation_start;
                    if (selected != expected_selected ||
                        complement != expected_complement) {
                        throw std::runtime_error(
                            "four-owner direct validation failed");
                    }
                    owner.result.verified++;
                }
                owner.result.covered_weight += U128(edge.factor) * edge.weight;
                owner.result.contribution += U128(edge.factor) * edge.weight *
                                             U128(selected) * complement;
            }
            shared.completed_kernels += edge_indices.size();
            owner.result.right_batches++;
            double free_start = seconds_now();
            CUDA_CHECK(cudaFree(device_results));
            CUDA_CHECK(cudaFree(device_joins));
            owner.result.batch_buffer_free_seconds +=
                seconds_now() - free_start;
        }
        if (!cache.device_resident && !next_prefetch_started &&
            batch_index + 1 < batch_right_indices.size()) {
            prefetched = prefetch_host_packed_layout(
                batch_references(batch_index + 1), cache);
            shared.pipeline_max_source_entries = std::max(
                shared.pipeline_max_source_entries,
                prefetched.source_entries);
        }
        double layout_free_start = seconds_now();
        free_packed_solve_layout(right_layout);
        double layout_free_seconds = seconds_now() - layout_free_start;
        for (FourOwnerState& owner : owners) {
            owner.result.right_layout_free_seconds +=
                layout_free_seconds / owners.size();
        }
        shared.right_batches++;
        if (shared.right_batches == 1 || shared.right_batches % 25 == 0 ||
            batch_index + 1 == batch_right_indices.size()) {
            std::printf(
                "FOUR_OWNER_BATCH batch=%llu right_prefixes=%zu "
                "entries=%llu completed=%llu gpu_seconds=%.6f\n",
                (unsigned long long)shared.right_batches, group_keys.size(),
                (unsigned long long)group_entries,
                (unsigned long long)shared.completed_kernels,
                shared.gpu_seconds);
        }
    }

    CUDA_CHECK(cudaEventDestroy(event_end));
    CUDA_CHECK(cudaEventDestroy(event_start));
    size_t expected_kernels = 0;
    for (FourOwnerState& owner : owners) {
        expected_kernels += owner.edges.size();
        owner.result.effective_right_entry_cap = right_entry_cap;
        owner.result.maximum_right_entries = shared.maximum_right_entries;
        owner.result.maximum_right_buckets = shared.maximum_right_buckets;
        owner.result.minimum_free_bytes = shared.minimum_free_bytes;
        owner.result.right_source_plan_seconds =
            shared.right_source_plan_seconds / owners.size();
        owner.result.right_source_sort_seconds =
            shared.right_source_sort_seconds / owners.size();
        owner.result.right_schedule_seconds =
            shared.right_schedule_seconds / owners.size();
        owner.result.right_layout_seconds =
            shared.right_layout_seconds / owners.size();
        owner.result.right_plan_seconds =
            shared.right_plan_seconds / owners.size();
        owner.result.right_upload_seconds =
            shared.right_upload_seconds / owners.size();
        owner.result.right_source_gather_seconds =
            shared.right_source_gather_seconds / owners.size();
        owner.result.right_source_upload_seconds =
            shared.right_source_upload_seconds / owners.size();
        owner.result.right_histogram_seconds =
            shared.right_histogram_seconds / owners.size();
        owner.result.right_scatter_seconds =
            shared.right_scatter_seconds / owners.size();
        owner.result.right_metadata_seconds =
            shared.right_metadata_seconds / owners.size();
        owner.result.right_source_entries =
            shared.right_source_entries / owners.size();
        owner.result.right_source_chunks =
            shared.right_source_chunks / owners.size();
    }
    if (shared.completed_kernels != expected_kernels) {
        release_four_owner_layouts(owners);
        throw std::runtime_error("four-owner tile left kernels unprocessed");
    }
    release_four_owner_layouts(owners);
    return shared;
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        std::fprintf(
            stderr,
            "Usage: %s PACKED_7X5.cache|CANONICAL_7X5.orbits FOUR_WORK.tsv RESULTS_DIR "
            "[MAX_RIGHT_ENTRIES=4294967295] [VERIFY_JOINS=4]\n\n"
            "FOUR_WORK.tsv must contain exactly four ordinary work items.\n",
            argv[0]);
        return 2;
    }
    uint64_t max_right_entries =
        argc > 4 ? std::strtoull(argv[4], nullptr, 10)
                 : uint64_t(UINT32_MAX);
    uint64_t verify_joins =
        argc > 5 ? std::strtoull(argv[5], nullptr, 10) : 4;
    if (!max_right_entries || max_right_entries > UINT32_MAX) return 2;

    std::vector<PackedWorkItem> items = read_work_manifest(argv[2]);
    if (items.size() != FOUR_OWNER_COUNT) {
        throw std::runtime_error("four-owner manifest must contain four items");
    }
    fs::path results_directory = argv[3];
    fs::create_directories(results_directory);
    PackedRunProvenance run_provenance = packed_run_provenance(
        "/proc/self/exe", packed_cache_identity_sha256(argv[1]));
    std::vector<PackedWorkProvenance> provenance;
    std::vector<PackedWorkClaim> claims;
    provenance.reserve(items.size());
    claims.reserve(items.size());
    for (const PackedWorkItem& item : items) {
        provenance.push_back(packed_work_provenance(run_provenance, item));
        claims.emplace_back(result_path(results_directory, item));
    }
    std::vector<uint8_t> already_complete(items.size());
    size_t complete_count = 0;
    for (size_t index = 0; index < items.size(); index++) {
        already_complete[index] = validated_result_exists(
            results_directory, items[index], provenance[index]);
        complete_count += already_complete[index];
    }
    if (complete_count == items.size()) {
        std::printf("ALL_COMPLETE items=%zu\n", items.size());
        return 0;
    }

    initialise_tables();
    validate_mask_split();
    validate_row_map_algebra();
    std::printf("CONFIG owners=%zu prefix_pairs=%d prefix_bits=%d "
                "suffix_bits=%d entry_bytes=%zu\n",
                FOUR_OWNER_COUNT, PREFIX_PAIR_COUNT, 2 * PREFIX_PAIR_COUNT,
                2 * (PAIRS - PREFIX_PAIR_COUNT), sizeof(PrefixSuffix));
    PackedUniversalCache cache = load_or_build_packed_universal_cache(argv[1]);
    std::vector<FourOwnerState> owners(FOUR_OWNER_COUNT);
    for (size_t index = 0; index < owners.size(); index++) {
        owners[index].item = items[index];
    }
    double tile_start = seconds_now();
    FourOwnerSharedStats shared = solve_four_owner_tile(
        owners, cache, max_right_entries, verify_joins);
    double tile_seconds = seconds_now() - tile_start;

    U128 contribution = 0;
    U128 direct_comparisons = 0;
    uint64_t input_right_prefixes = 0;
    uint64_t kernels = 0;
    double left_factory_seconds = 0;
    double left_layout_seconds = 0;
    for (size_t owner_index = 0; owner_index < owners.size(); owner_index++) {
        FourOwnerState& owner = owners[owner_index];
        // total_seconds is amortised resource time so summing the four atomic
        // result files recovers the actual one-GPU tile wall time.  The full
        // latency remains explicit in FOUR_OWNER_TILE below.
        owner.result.total_seconds = tile_seconds / owners.size();
        if (!already_complete[owner_index]) {
            write_work_result(results_directory, owner.item, owner.result,
                              provenance[owner_index]);
        } else {
            std::printf("RECOMPUTED_EXISTING id=%s publication=skipped\n",
                        owner.item.id.c_str());
        }
        contribution += owner.result.contribution;
        direct_comparisons += owner.result.direct_comparisons;
        input_right_prefixes += owner.result.right_prefixes;
        kernels += owner.result.kernels;
        left_factory_seconds += owner.result.left_factory_seconds;
        left_layout_seconds += owner.result.left_layout_seconds;
        std::printf(
            "FOUR_OWNER_RESULT id=%s records=%llu kernels=%llu "
            "right_prefixes=%llu contribution=%s gpu_seconds=%.6f "
            "verified=%llu exact=OK\n",
            owner.item.id.c_str(),
            (unsigned long long)owner.result.records,
            (unsigned long long)owner.result.kernels,
            (unsigned long long)owner.result.right_prefixes,
            u128_string(owner.result.contribution).c_str(),
            owner.result.gpu_seconds,
            (unsigned long long)owner.result.verified);
    }
    double reuse = input_right_prefixes
                       ? 1.0 - double(shared.union_right_prefixes) /
                                   double(input_right_prefixes)
                       : 0.0;
    std::printf(
        "FOUR_OWNER_TILE owners=%zu kernels=%llu input_right_prefixes=%llu "
        "union_right_prefixes=%llu raw_reuse=%.9f right_batches=%llu "
        "effective_right_entry_cap=%llu maximum_right_entries=%llu "
        "maximum_right_buckets=%llu minimum_free_bytes=%zu "
        "right_source_entries=%llu right_source_chunks=%llu "
        "pipeline_max_source_entries=%llu pipeline_wait_seconds=%.6f "
        "right_source_plan_seconds=%.6f right_source_sort_seconds=%.6f "
        "right_schedule_seconds=%.6f "
        "left_factory_seconds=%.6f left_layout_seconds=%.6f "
        "right_layout_seconds=%.6f right_plan_seconds=%.6f "
        "right_upload_seconds=%.6f "
        "right_source_gather_seconds=%.6f right_source_upload_seconds=%.6f "
        "right_histogram_seconds=%.6f right_scatter_seconds=%.6f "
        "right_metadata_seconds=%.6f gpu_seconds=%.6f "
        "direct_comparisons=%s contribution=%s tile_seconds=%.6f exact=OK\n",
        owners.size(), (unsigned long long)kernels,
        (unsigned long long)input_right_prefixes,
        (unsigned long long)shared.union_right_prefixes, reuse,
        (unsigned long long)shared.right_batches,
        (unsigned long long)shared.effective_right_entry_cap,
        (unsigned long long)shared.maximum_right_entries,
        (unsigned long long)shared.maximum_right_buckets,
        shared.minimum_free_bytes,
        (unsigned long long)shared.right_source_entries,
        (unsigned long long)shared.right_source_chunks,
        (unsigned long long)shared.pipeline_max_source_entries,
        shared.pipeline_wait_seconds, shared.right_source_plan_seconds,
        shared.right_source_sort_seconds, shared.right_schedule_seconds,
        left_factory_seconds,
        left_layout_seconds, shared.right_layout_seconds,
        shared.right_plan_seconds, shared.right_upload_seconds,
        shared.right_source_gather_seconds,
        shared.right_source_upload_seconds, shared.right_histogram_seconds,
        shared.right_scatter_seconds, shared.right_metadata_seconds,
        shared.gpu_seconds, u128_string(direct_comparisons).c_str(),
        u128_string(contribution).c_str(), tile_seconds);

    return 0;
}
