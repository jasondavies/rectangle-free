#ifndef GRID_ROWS
#define GRID_ROWS 7
#endif
#ifndef GRID_COLUMNS
#define GRID_COLUMNS 7
#endif
#ifndef LEFT_COLUMNS
#define LEFT_COLUMNS 3
#endif
#ifndef RIGHT_COLUMNS
#define RIGHT_COLUMNS 4
#endif
#ifndef ORBIT_ROW_BITS
#define ORBIT_ROW_BITS 8
#endif
#ifndef ORBIT_MAGIC
#define ORBIT_MAGIC "R7SQT01"
#endif

#include "twocolour_7x7_engine.cuh"

int main(int argc, char** argv) {
    if (argc < 2 || argc > 8) {
        std::fprintf(stderr,
                     "Usage: %s ORBITS [START=0] [END=0] [BATCH_KERNELS=16384] "
                     "[VERIFY_JOINS=16] [FILTER_MOD=0] [FILTER_ID=0]\n",
                     argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    uint64_t start_record = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 0;
    uint64_t end_record = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 0;
    size_t batch_kernels = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 16384;
    uint64_t verify_joins = argc > 5 ? std::strtoull(argv[5], nullptr, 10) : 16;
    uint64_t filter_mod = argc > 6 ? std::strtoull(argv[6], nullptr, 10) : 0;
    uint64_t filter_id = argc > 7 ? std::strtoull(argv[7], nullptr, 10) : 0;
    if (!batch_kernels || (filter_mod && filter_id >= filter_mod)) return 2;

    double total_start = seconds_now();
    initialise_tables();
    U128 labelled_weight = 0;
    uint64_t records = 0;
    double load_start = seconds_now();
    std::vector<Edge> edges =
        read_edges(path, start_record, end_record, filter_mod, filter_id,
                   labelled_weight, records);
    mark_validation_edges(edges, verify_joins);
    double load_seconds = seconds_now() - load_start;
    std::vector<PrefixKey> left_keys = unique_lefts(edges);
    std::printf("INPUT records=%llu kernels=%zu unique_left=%zu load_seconds=%.6f\n",
                (unsigned long long)records, edges.size(), left_keys.size(), load_seconds);

    double cache_start = seconds_now();
    std::vector<PrefixKey> all_right_keys = unique_rights(edges);
    std::unique_ptr<CanonicalFactory> shared_factory;
    std::unique_ptr<CanonicalFactory> separate_left_factory;
    std::unique_ptr<CanonicalFactory> separate_right_factory;
    CanonicalFactory* left_factory = nullptr;
    CanonicalFactory* right_factory = nullptr;
    if (LEFT_COLUMNS == RIGHT_COLUMNS) {
        std::vector<PrefixKey> all_keys = left_keys;
        all_keys.insert(all_keys.end(), all_right_keys.begin(), all_right_keys.end());
        shared_factory =
            std::make_unique<CanonicalFactory>(build_canonical_factory(std::move(all_keys),
                                                                       LEFT_COLUMNS));
        left_factory = shared_factory.get();
        right_factory = shared_factory.get();
    } else {
        separate_left_factory =
            std::make_unique<CanonicalFactory>(build_canonical_factory(left_keys, LEFT_COLUMNS));
        separate_right_factory = std::make_unique<CanonicalFactory>(
            build_canonical_factory(std::move(all_right_keys), RIGHT_COLUMNS));
        left_factory = separate_left_factory.get();
        right_factory = separate_right_factory.get();
    }
    double cache_build_seconds = seconds_now() - cache_start;
    CanonicalLayout left_layout = build_canonical_layout(left_keys, *left_factory);
    std::vector<PackedPair> left_descriptors = std::move(left_layout.descriptors);
    size_t left_entry_count = left_layout.entry_count;
    double left_seconds = cache_build_seconds;

    std::unordered_map<PrefixKey, uint32_t> left_index;
    left_index.reserve(left_keys.size() * 2);
    for (size_t i = 0; i < left_keys.size(); i++) left_index[left_keys[i]] = uint32_t(i);

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    std::printf("DEVICE name=%s cc=%d.%d sm=%d memory_gib=%.2f\n", properties.name,
                properties.major, properties.minor, properties.multiProcessorCount,
                properties.totalGlobalMem / 1073741824.0);
    size_t left_device_bytes =
        left_entry_count * (sizeof(uint64_t) + sizeof(uint32_t));
    uint64_t* device_left_masks = nullptr;
    uint32_t* device_left_weights = nullptr;
    std::printf("LEFT prefixes=%zu entries=%zu bytes=%zu build_seconds=%.6f\n",
                left_keys.size(), left_entry_count, left_device_bytes, left_seconds);

    uint64_t* device_left_canonical_masks = nullptr;
    uint32_t* device_left_canonical_weights = nullptr;
    uint64_t* device_right_canonical_masks = nullptr;
    uint32_t* device_right_canonical_weights = nullptr;
    upload_entry_soa(left_factory->entries, &device_left_canonical_masks,
                     &device_left_canonical_weights);
    if (right_factory == left_factory) {
        device_right_canonical_masks = device_left_canonical_masks;
        device_right_canonical_weights = device_left_canonical_weights;
    } else {
        upload_entry_soa(right_factory->entries, &device_right_canonical_masks,
                         &device_right_canonical_weights);
    }
    std::printf("CANONICAL shared=%d left_raw=%zu left_distributions=%zu left_entries=%zu "
                "right_raw=%zu right_distributions=%zu right_entries=%zu "
                "build_seconds=%.6f\n",
                right_factory == left_factory, left_factory->raw.size(),
                left_factory->descriptors.size(), left_factory->entries.size(),
                right_factory->raw.size(), right_factory->descriptors.size(),
                right_factory->entries.size(), cache_build_seconds);
    CUDA_CHECK(cudaMalloc(&device_left_masks, left_entry_count * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&device_left_weights, left_entry_count * sizeof(uint32_t)));
    ExpansionDesc* device_left_expansions = nullptr;
    CUDA_CHECK(cudaMalloc(&device_left_expansions,
                          left_layout.expansions.size() * sizeof(ExpansionDesc)));
    CUDA_CHECK(cudaMemcpy(device_left_expansions, left_layout.expansions.data(),
                          left_layout.expansions.size() * sizeof(ExpansionDesc),
                          cudaMemcpyHostToDevice));
    double left_expand_start = seconds_now();
    expand_canonical_distributions_soa<<<unsigned(left_layout.expansions.size()), THREADS>>>(
        device_left_canonical_masks, device_left_canonical_weights, device_left_masks,
        device_left_weights, device_left_expansions);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double left_expand_seconds = seconds_now() - left_expand_start;
    CUDA_CHECK(cudaFree(device_left_expansions));
    left_factory->entries.clear();
    left_factory->entries.shrink_to_fit();
    if (right_factory != left_factory) {
        right_factory->entries.clear();
        right_factory->entries.shrink_to_fit();
    }

    U128 contribution = 0;
    U128 covered_weight = 0;
    U128 comparisons = 0;
    uint64_t right_groups = 0;
    uint64_t checked = 0;
    double right_build_seconds = 0;
    double right_expand_seconds = 0;
    double upload_seconds = 0;
    double gpu_seconds = 0;
    double validation_seconds = 0;
    size_t batch_number = 0;

    std::vector<std::pair<size_t, size_t>> batch_ranges;
    for (size_t begin = 0; begin < edges.size();) {
        size_t end = begin;
        do {
            PrefixKey right = edges[end].right;
            while (end < edges.size() && edges[end].right == right) end++;
        } while (end < edges.size() && end - begin < batch_kernels);
        batch_ranges.emplace_back(begin, end);
        begin = end;
    }

    std::future<CanonicalHostBatch> pending;
    if (!batch_ranges.empty()) {
        pending = std::async(std::launch::async, build_canonical_host_batch,
                             std::cref(edges), std::cref(left_descriptors),
                             std::cref(left_index), std::cref(*right_factory),
                             batch_ranges[0].first, batch_ranges[0].second);
    }
    for (size_t range_index = 0; range_index < batch_ranges.size(); range_index++) {
        CanonicalHostBatch batch = pending.get();
        if (range_index + 1 < batch_ranges.size()) {
            auto next = batch_ranges[range_index + 1];
            pending = std::async(std::launch::async, build_canonical_host_batch,
                                 std::cref(edges), std::cref(left_descriptors),
                                 std::cref(left_index), std::cref(*right_factory), next.first,
                                 next.second);
        }
        right_groups += batch.right_keys.size();
        right_build_seconds += batch.build_seconds;
        comparisons += batch.comparisons;

        uint64_t* device_right_masks = nullptr;
        uint32_t* device_right_weights = nullptr;
        JoinDesc* device_joins = nullptr;
        unsigned long long* device_results = nullptr;
        ExpansionDesc* device_expansions = nullptr;
        size_t right_entry_count = batch.right_entry_count;
        std::vector<unsigned long long> results(batch.joins.size());
        double upload_start = seconds_now();
        CUDA_CHECK(cudaMalloc(&device_right_masks, right_entry_count * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&device_right_weights, right_entry_count * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&device_joins, batch.joins.size() * sizeof(JoinDesc)));
        CUDA_CHECK(cudaMalloc(&device_results, results.size() * sizeof(results[0])));
        CUDA_CHECK(cudaMalloc(&device_expansions,
                              batch.expansions.size() * sizeof(ExpansionDesc)));
        CUDA_CHECK(cudaMemcpy(device_expansions, batch.expansions.data(),
                              batch.expansions.size() * sizeof(ExpansionDesc),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_joins, batch.joins.data(),
                              batch.joins.size() * sizeof(JoinDesc),
                              cudaMemcpyHostToDevice));
        upload_seconds += seconds_now() - upload_start;

        cudaEvent_t event_start, event_end;
        CUDA_CHECK(cudaEventCreate(&event_start));
        CUDA_CHECK(cudaEventCreate(&event_end));
        CUDA_CHECK(cudaEventRecord(event_start));
        expand_canonical_distributions_soa<<<unsigned(batch.expansions.size()), THREADS>>>(
            device_right_canonical_masks, device_right_canonical_weights,
            device_right_masks, device_right_weights, device_expansions);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(event_end));
        CUDA_CHECK(cudaEventSynchronize(event_end));
        float batch_expand_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&batch_expand_ms, event_start, event_end));
        right_expand_seconds += batch_expand_ms / 1000.0;
        CUDA_CHECK(cudaEventRecord(event_start));
        fused_disjoint_joins_soa<<<unsigned(batch.joins.size()), THREADS>>>(
            device_left_masks, device_left_weights, device_right_masks,
            device_right_weights, device_joins, device_results
            );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(event_end));
        CUDA_CHECK(cudaEventSynchronize(event_end));
        float batch_gpu_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&batch_gpu_ms, event_start, event_end));
        gpu_seconds += batch_gpu_ms / 1000.0;
        CUDA_CHECK(cudaMemcpy(results.data(), device_results,
                              results.size() * sizeof(results[0]), cudaMemcpyDeviceToHost));
        std::vector<unsigned long long> logical_results(results.size());
        for (size_t scheduled_index = 0; scheduled_index < results.size(); scheduled_index++) {
            logical_results[batch.result_slots[scheduled_index]] = results[scheduled_index];
        }
        results.swap(logical_results);

        size_t edge_index = batch.begin;
        size_t join_index = 0;
        for (size_t group = 0; group < batch.right_keys.size(); group++) {
            const PackedPair& right = batch.right_descriptors[group];
            while (edge_index < batch.end &&
                   edges[edge_index].right == batch.right_keys[group]) {
                const Edge& edge = edges[edge_index];
                uint64_t selected = results[join_index++];
                uint64_t complement = results[join_index++];
                if (edge.validate) {
                    double validation_start = seconds_now();
                    DistributionPair validation_left = build_pair(edge.left, LEFT_COLUMNS);
                    DistributionPair validation_right = build_pair(edge.right, RIGHT_COLUMNS);
                    uint64_t expected_selected =
                        cpu_join(validation_left.selected.entries.data(),
                                 validation_left.selected.entries.size(),
                                 validation_right.selected.entries.data(),
                                 validation_right.selected.entries.size());
                    uint64_t expected_complement =
                        cpu_join(validation_left.complement.entries.data(),
                                 validation_left.complement.entries.size(),
                                 validation_right.complement.entries.data(),
                                 validation_right.complement.entries.size());
                    validation_seconds += seconds_now() - validation_start;
                    if (selected != expected_selected || complement != expected_complement) {
                        std::fprintf(stderr,
                                     "validation failed kernel=%llu selected=%llu/%llu "
                                     "complement=%llu/%llu\n",
                                     (unsigned long long)checked,
                                     (unsigned long long)selected,
                                     (unsigned long long)expected_selected,
                                     (unsigned long long)complement,
                                     (unsigned long long)expected_complement);
                        return 1;
                    }
                    checked++;
                }
                covered_weight += U128(edge.factor) * edge.weight;
                contribution +=
                    U128(edge.factor) * edge.weight * U128(selected) * complement;
                edge_index++;
            }
        }

        CUDA_CHECK(cudaEventDestroy(event_end));
        CUDA_CHECK(cudaEventDestroy(event_start));
        CUDA_CHECK(cudaFree(device_expansions));
        CUDA_CHECK(cudaFree(device_results));
        CUDA_CHECK(cudaFree(device_joins));
        CUDA_CHECK(cudaFree(device_right_weights));
        CUDA_CHECK(cudaFree(device_right_masks));
        batch_number++;
        if (batch_number == 1 || batch_number % 25 == 0 || batch.end == edges.size()) {
            std::printf("BATCH number=%zu kernels=%zu right_groups=%zu right_entries=%zu "
                        "gpu_seconds=%.6f completed=%zu/%zu\n",
                        batch_number, batch.end - batch.begin, batch.right_keys.size(),
                        right_entry_count, batch_gpu_ms / 1000.0, batch.end,
                        edges.size());
        }
    }

    CUDA_CHECK(cudaFree(device_left_weights));
    CUDA_CHECK(cudaFree(device_left_masks));
    if (device_right_canonical_masks != device_left_canonical_masks) {
        CUDA_CHECK(cudaFree(device_right_canonical_weights));
        CUDA_CHECK(cudaFree(device_right_canonical_masks));
    }
    CUDA_CHECK(cudaFree(device_left_canonical_weights));
    CUDA_CHECK(cudaFree(device_left_canonical_masks));
    double total_seconds = seconds_now() - total_start;
    double comparison_rate = gpu_seconds ? double(comparisons) / gpu_seconds : 0;
    std::printf("RESULT records=%llu labelled_weight=%s kernels=%zu covered_weight=%s "
                "right_groups=%llu verified=%llu comparisons=%s contribution=%s\n",
                (unsigned long long)records, u128_string(labelled_weight).c_str(), edges.size(),
                u128_string(covered_weight).c_str(), (unsigned long long)right_groups,
                (unsigned long long)checked, u128_string(comparisons).c_str(),
                u128_string(contribution).c_str());
    std::printf("TIMING load=%.6f left_build=%.6f right_build=%.6f upload=%.6f "
                "left_expand=%.6f right_expand=%.6f gpu=%.6f validation=%.6f "
                "total=%.6f comparisons_per_second=%.3f\n",
                load_seconds, left_seconds, right_build_seconds, upload_seconds,
                left_expand_seconds, right_expand_seconds, gpu_seconds, validation_seconds,
                total_seconds, comparison_rate);

#if GRID_ROWS == 7 && GRID_COLUMNS == 7
    const U128 expected = U128(UINT64_C(7016720048108792558)) * 100000000U +
                          UINT64_C(76925440);
    if (!filter_mod && start_record == 0 && records == UINT64_C(16853750)) {
        bool valid = labelled_weight == (U128(1) << CELLS) &&
                     edges.size() == UINT64_C(8426875) &&
                     covered_weight == (U128(1) << CELLS) && contribution == expected;
        std::printf("FULL_CHECK expected=%s %s\n", u128_string(expected).c_str(),
                    valid ? "OK" : "FAIL");
        if (!valid) return 1;
    }
#elif GRID_ROWS == 6 && GRID_COLUMNS == 9
    if (!filter_mod && start_record == 0 && records == UINT64_C(130237768)) {
        bool valid = labelled_weight == (U128(1) << CELLS) &&
                     edges.size() == UINT64_C(71696841) &&
                     covered_weight == (U128(1) << CELLS);
        std::printf("FULL_CHECK expected_records=130237768 expected_kernels=71696841 "
                    "expected_weight=%s %s\n",
                    u128_string(U128(1) << CELLS).c_str(), valid ? "OK" : "FAIL");
        if (!valid) return 1;
    }
#elif GRID_ROWS == 7 && GRID_COLUMNS == 8
    if (!filter_mod && start_record == 0 && records == UINT64_C(508147108)) {
        bool valid = labelled_weight == (U128(1) << CELLS) &&
                     edges.size() == UINT64_C(279892401) &&
                     covered_weight == (U128(1) << CELLS);
        std::printf("FULL_CHECK expected_records=508147108 expected_kernels=279892401 "
                    "expected_weight=%s %s\n",
                    u128_string(U128(1) << CELLS).c_str(), valid ? "OK" : "FAIL");
        if (!valid) return 1;
    }
#endif
    return 0;
}
