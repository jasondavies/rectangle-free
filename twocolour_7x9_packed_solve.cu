#include "twocolour_7x9_engine.cuh"

int main(int argc, char** argv) {
    constexpr uint64_t default_max_right_entries = UINT32_MAX;
    if (argc < 4 || argc > 6) {
        std::fprintf(
            stderr,
            "Usage: %s PACKED_7X5.cache|CANONICAL_7X5.orbits WORK.tsv RESULTS_DIR "
            "[MAX_RIGHT_ENTRIES=%llu] [VERIFY_JOINS=4]\n\n"
            "WORK.tsv columns: ID ORBITS START END [FILTER_MOD FILTER_ID]\n",
            argv[0], (unsigned long long)default_max_right_entries);
        return 2;
    }
    uint64_t max_right_entries =
        argc > 4 ? std::strtoull(argv[4], nullptr, 10)
                 : default_max_right_entries;
    uint64_t verify_joins =
        argc > 5 ? std::strtoull(argv[5], nullptr, 10) : 4;
    if (!max_right_entries || max_right_entries > UINT32_MAX) return 2;

    std::vector<PackedWorkItem> items = read_work_manifest(argv[2]);
    fs::path results_directory = argv[3];
    fs::create_directories(results_directory);
    PackedRunProvenance run_provenance = packed_run_provenance(
        "/proc/self/exe", packed_cache_identity_sha256(argv[1]));
    std::vector<std::pair<PackedWorkItem, PackedWorkProvenance>> pending;
    for (const PackedWorkItem& item : items) {
        PackedWorkProvenance provenance =
            packed_work_provenance(run_provenance, item);
        if (validated_result_exists(results_directory, item, provenance)) {
            std::printf("SKIP id=%s result=%s\n", item.id.c_str(),
                        result_path(results_directory, item).c_str());
        } else {
            pending.emplace_back(item, std::move(provenance));
        }
    }
    if (pending.empty()) {
        std::printf("ALL_COMPLETE items=%zu\n", items.size());
        return 0;
    }

    initialise_tables();
    validate_mask_split();
    validate_row_map_algebra();
    std::printf("CONFIG prefix_pairs=%d prefix_bits=%d suffix_bits=%d "
                "entry_bytes=%zu\n",
                PREFIX_PAIR_COUNT, 2 * PREFIX_PAIR_COUNT,
                2 * (PAIRS - PREFIX_PAIR_COUNT), sizeof(PrefixSuffix));
    PackedUniversalCache cache = load_or_build_packed_universal_cache(argv[1]);
    for (size_t index = 0; index < pending.size(); index++) {
        const PackedWorkItem& item = pending[index].first;
        const PackedWorkProvenance& provenance = pending[index].second;
        PackedWorkClaim claim(result_path(results_directory, item));
        if (validated_result_exists(results_directory, item, provenance)) {
            std::printf("SKIP_AFTER_CLAIM id=%s\n", item.id.c_str());
            continue;
        }
        std::printf("WORK_START id=%s index=%zu/%zu path=%s start=%llu end=%llu\n",
                    item.id.c_str(), index + 1, pending.size(), item.path.c_str(),
                    (unsigned long long)item.start,
                    (unsigned long long)item.end);
        PackedWorkResult result = solve_packed_work_item(
            item, cache, max_right_entries, verify_joins);
        write_work_result(results_directory, item, result, provenance);
        std::printf(
            "WORK_RESULT id=%s records=%llu labelled_weight=%s kernels=%llu "
            "covered_weight=%s left_prefixes=%llu left_entries=%llu "
            "left_buckets=%llu right_prefixes=%llu right_batches=%llu "
            "max_right_entries=%llu max_right_buckets=%llu "
            "verified=%llu direct_comparisons=%s contribution=%s "
            "minimum_free=%zu gpu_seconds=%.6f total_seconds=%.6f exact=OK\n",
            item.id.c_str(), (unsigned long long)result.records,
            u128_string(result.labelled_weight).c_str(),
            (unsigned long long)result.kernels,
            u128_string(result.covered_weight).c_str(),
            (unsigned long long)result.left_prefixes,
            (unsigned long long)result.left_entries,
            (unsigned long long)result.left_buckets,
            (unsigned long long)result.right_prefixes,
            (unsigned long long)result.right_batches,
            (unsigned long long)result.maximum_right_entries,
            (unsigned long long)result.maximum_right_buckets,
            (unsigned long long)result.verified,
            u128_string(result.direct_comparisons).c_str(),
            u128_string(result.contribution).c_str(), result.minimum_free_bytes,
            result.gpu_seconds, result.total_seconds);
    }
    std::printf("RUN_COMPLETE completed=%zu cache_seconds=%.6f\n", pending.size(),
                cache.total_seconds);
    return 0;
}
