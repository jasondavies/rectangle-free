#ifndef GRID_ROWS
#define GRID_ROWS 8
#define GRID_COLUMNS 8
#define LEFT_COLUMNS 4
#define RIGHT_COLUMNS 4
#define ORBIT_ROW_BITS 8
#define ORBIT_MAGIC "R8SQT01"
#define TWCOLOUR_GEOMETRY "8x8"
#define TWCOLOUR_RESULT_MAGIC "RECT8X8_PREFIX_RESULT"
#define TWCOLOUR_TRANSPOSE_QUOTIENT 1
#endif
#ifndef TWCOLOUR_GEOMETRY
#error "the production prefix solver requires a geometry name"
#endif
#ifndef TWCOLOUR_RESULT_MAGIC
#error "the production prefix solver requires a result-format magic"
#endif
#ifndef TWCOLOUR_TRANSPOSE_QUOTIENT
#define TWCOLOUR_TRANSPOSE_QUOTIENT 0
#endif
#include "twocolour_prefix_algebra.cuh"
#include "gpu_memory_policy.hpp"

#include <filesystem>
#include <sstream>

#include "gpu_result_checkpoint.hpp"

#if defined(__GLIBC__)
#include <malloc.h>
#endif

static constexpr uint64_t PREFIX_DEFAULT_DEVICE_RESERVE_BYTES = UINT64_C(2) << 30;
#ifndef TWCOLOUR_ESTIMATED_RIGHT_BYTES_PER_ENTRY
#define TWCOLOUR_ESTIMATED_RIGHT_BYTES_PER_ENTRY 28
#endif
static constexpr uint64_t PREFIX_ESTIMATED_RIGHT_BYTES_PER_ENTRY =
    TWCOLOUR_ESTIMATED_RIGHT_BYTES_PER_ENTRY;

namespace fs = std::filesystem;

static std::vector<uint32_t> schedule_prefix_heavy_first(
    std::vector<PrefixJoinDesc>& joins,
    const std::vector<uint64_t>& direct_work) {
    if (joins.size() != direct_work.size() || joins.size() > UINT32_MAX) {
        throw std::runtime_error("invalid production prefix schedule");
    }
    std::vector<uint32_t> order(joins.size());
    std::iota(order.begin(), order.end(), 0U);
    std::sort(order.begin(), order.end(), [&](uint32_t lhs, uint32_t rhs) {
        return direct_work[lhs] != direct_work[rhs]
            ? direct_work[lhs] > direct_work[rhs]
            : lhs < rhs;
    });
    std::vector<PrefixJoinDesc> scheduled;
    scheduled.reserve(joins.size());
    for (uint32_t logical : order) scheduled.push_back(joins[logical]);
    joins.swap(scheduled);
    return order;
}

static std::vector<std::array<CanonicalRef, 2>> resolve_canonical_refs(
    const std::vector<PrefixKey>& keys, const CanonicalFactory& factory) {
    std::vector<std::array<CanonicalRef, 2>> result(keys.size());
    std::vector<size_t> unresolved;
    unresolved.reserve(keys.size());
    size_t raw_index = 0;
    for (size_t index = 0; index < keys.size(); index++) {
        while (raw_index < factory.raw.size() &&
               factory.raw[raw_index].raw < keys[index]) {
            raw_index++;
        }
        if (raw_index < factory.raw.size() &&
            factory.raw[raw_index].raw == keys[index]) {
            const RawCanonicalPair& raw = factory.raw[raw_index];
            result[index] = {raw.selected, raw.complement};
        } else {
            unresolved.push_back(index);
        }
    }
    std::atomic<bool> missing{false};
#pragma omp parallel for schedule(static)
    for (long long item = 0; item < (long long)unresolved.size(); item++) {
        size_t index = unresolved[size_t(item)];
        if (!find_factory_canonical_refs(factory, keys[index], result[index])) {
            missing.store(true, std::memory_order_relaxed);
        }
    }
    if (missing.load(std::memory_order_relaxed)) {
        throw std::runtime_error(
            "solve shard requires a canonical distribution absent from the explicit canonical seed");
    }
    return result;
}

#include "twocolour_weight_class_join.cuh"
#include "twocolour_canonical_device.cuh"

static gpu_checkpoint::RunProvenance prefix_run_provenance(
    const std::string& executable, const std::string& canonical_seed) {
    std::ostringstream configuration;
    configuration << "geometry=" << TWCOLOUR_GEOMETRY
                  << ";left_columns=" << LEFT_COLUMNS
                  << ";right_columns=" << RIGHT_COLUMNS
                  << ";prefix_pairs=" << PREFIX_PAIR_COUNT
                  << ";task_chunk=" << PREFIX_TASK_CHUNK
                  << ";threads=" << THREADS
                  << ";orbit_magic=" << ORBIT_MAGIC
                  << ";transpose_quotient=" << TWCOLOUR_TRANSPOSE_QUOTIENT
                  << ";token_plane_quotient=1"
                  << ";join=" << WEIGHT_CLASS_JOIN_FINGERPRINT;
    return gpu_checkpoint::run_provenance(
        TWCOLOUR_RESULT_MAGIC, TWCOLOUR_GEOMETRY, executable,
        configuration.str(),
        canonical_seed);
}

static bool validated_prefix_result_exists(
    const fs::path& directory, const gpu_checkpoint::WorkItem& item,
    const gpu_checkpoint::WorkProvenance& provenance) {
    static const std::vector<std::string> required = {
        "records", "labelled_weight", "kernels", "covered_weight",
        "direct_comparisons", "contribution", "total_seconds",
        "transpose_quotient"};
    return gpu_checkpoint::validated_result_exists(
        directory, item, provenance, required);
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        std::fprintf(
            stderr,
            "Usage: %s CANONICAL_SEED.orbits WORK.tsv RESULTS_DIR "
            "[BATCH_EDGES=auto] [VERIFY_JOINS=4]\n\n"
            "WORK.tsv columns: ID ORBITS START END [FILTER_MOD FILTER_ID]\n",
            argv[0]);
        return 2;
    }
    const std::string seed_path = argv[1];
    std::vector<gpu_checkpoint::WorkItem> items =
        gpu_checkpoint::read_work_manifest(argv[2]);
    fs::path results_directory = argv[3];
    fs::create_directories(results_directory);
    size_t batch_edges = std::numeric_limits<size_t>::max();
    if (argc > 4 && std::strcmp(argv[4], "auto")) {
        batch_edges = std::strtoull(argv[4], nullptr, 10);
    }
    uint64_t verify_joins =
        argc > 5 ? std::strtoull(argv[5], nullptr, 10) : 4;
    if (!batch_edges) return 2;
    const size_t device_reserve_bytes = gpu_memory_policy::reserve_bytes(
        PREFIX_DEFAULT_DEVICE_RESERVE_BYTES);

    gpu_checkpoint::RunProvenance run_provenance =
        prefix_run_provenance("/proc/self/exe", seed_path);
    std::vector<std::pair<gpu_checkpoint::WorkItem,
                          gpu_checkpoint::WorkProvenance>> pending;
    for (const gpu_checkpoint::WorkItem& item : items) {
        gpu_checkpoint::WorkProvenance provenance =
            gpu_checkpoint::work_provenance(run_provenance, item);
        if (validated_prefix_result_exists(results_directory, item, provenance)) {
            std::printf("SKIP id=%s result=%s\n", item.id.c_str(),
                        gpu_checkpoint::result_path(results_directory, item)
                            .c_str());
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

    // Build and upload the canonical source cache once, then retain it while
    // solving every pending manifest item. Equal-width geometries share one
    // cache; asymmetric splits retain independent left and right caches.
    double session_start = seconds_now();
    U128 seed_labelled_weight = 0;
    uint64_t seed_records = 0;
    double seed_load_start = seconds_now();
    std::vector<Edge> seed_edges = read_edges(seed_path, 0, 0, 0, 0,
                                              seed_labelled_weight,
                                              seed_records);
    double seed_load_seconds = seconds_now() - seed_load_start;
    std::vector<PrefixKey> seed_keys = unique_lefts(seed_edges);
    std::vector<PrefixKey> seed_rights = unique_rights(seed_edges);
#if LEFT_COLUMNS == RIGHT_COLUMNS
    seed_keys.insert(seed_keys.end(), seed_rights.begin(), seed_rights.end());
#endif
    double factory_start = seconds_now();
    CanonicalFactory left_factory =
        build_canonical_factory(std::move(seed_keys), LEFT_COLUMNS);
    CanonicalFactory right_factory_storage;
    const CanonicalFactory* right_factory = &left_factory;
#if LEFT_COLUMNS != RIGHT_COLUMNS
    right_factory_storage =
        build_canonical_factory(std::move(seed_rights), RIGHT_COLUMNS);
    right_factory = &right_factory_storage;
#endif
    double cache_factory_seconds = seconds_now() - factory_start;
    double canonical_upload_start = seconds_now();
    ProductionCanonicalDevice left_canonical =
        upload_production_canonical(left_factory);
    ProductionCanonicalDevice right_canonical_storage;
    const ProductionCanonicalDevice* right_canonical = &left_canonical;
#if LEFT_COLUMNS != RIGHT_COLUMNS
    right_canonical_storage =
        upload_production_canonical(right_factory_storage);
    right_canonical = &right_canonical_storage;
#endif
    double cache_upload_seconds = seconds_now() - canonical_upload_start;
    left_factory.entries.clear();
    left_factory.entries.shrink_to_fit();
#if LEFT_COLUMNS != RIGHT_COLUMNS
    right_factory_storage.entries.clear();
    right_factory_storage.entries.shrink_to_fit();
#endif
#if defined(__GLIBC__)
    // Large asymmetric canonical factories can leave tens of GiB in glibc's
    // arenas after their entries have been uploaded.  Return those unused
    // pages before constructing per-work-item host layouts so several GPU
    // workers can safely share one host.
    malloc_trim(0);
#endif
    std::printf(
        "PREFIX_SHARED_CACHE seed=%s paths=%zu left_canonical_sources=%zu "
        "right_canonical_sources=%zu left_canonical_entries=%zu "
        "right_canonical_entries=%zu seed_load_seconds=%.6f "
        "factory_seconds=%.6f upload_seconds=%.6f token_plane_quotient=%d\n",
        seed_path.c_str(), pending.size(), left_factory.canonical_keys.size(),
        right_factory->canonical_keys.size(), left_canonical.entry_count,
        right_canonical->entry_count, seed_load_seconds,
        cache_factory_seconds, cache_upload_seconds, 1);

    size_t solved_items = 0;
    for (const auto& pending_item : pending) {
    const gpu_checkpoint::WorkItem& item = pending_item.first;
    const gpu_checkpoint::WorkProvenance& provenance = pending_item.second;
    gpu_checkpoint::WorkClaim claim(
        gpu_checkpoint::result_path(results_directory, item));
    if (validated_prefix_result_exists(results_directory, item, provenance)) {
        std::printf("SKIP_AFTER_CLAIM id=%s\n", item.id.c_str());
        continue;
    }
    const bool first_solve = solved_items == 0;
    const std::string& path = item.path;
    const uint64_t start_record = item.start;
    const uint64_t end_record = item.end;
    const uint64_t filter_mod = item.filter_mod;
    const uint64_t filter_id = item.filter_id;
    double total_start = first_solve ? session_start : seconds_now();
    U128 labelled_weight = 0;
    uint64_t records = 0;
    double load_start = seconds_now();
    const bool reuse_seed_edges = first_solve && path == seed_path &&
        start_record == 0 && end_record == 0 && filter_mod == 0;
    std::vector<Edge> edges;
    if (reuse_seed_edges) {
        edges = std::move(seed_edges);
        labelled_weight = seed_labelled_weight;
        records = seed_records;
    } else {
        edges = read_edges(path, start_record, end_record,
                           filter_mod, filter_id,
                           labelled_weight, records);
    }
    double load_seconds = reuse_seed_edges
        ? seed_load_seconds : seconds_now() - load_start;
    if (first_solve) {
        seed_edges.clear();
        seed_edges.shrink_to_fit();
    }
    mark_validation_edges(edges, verify_joins);
    std::vector<PrefixKey> left_keys = unique_lefts(edges);
    std::vector<PrefixKey> right_keys = unique_rights(edges);
    std::vector<uint32_t> edge_left_ids =
        resolve_edge_left_ids(edges, left_keys);
    double factory_seconds = first_solve ? cache_factory_seconds : 0;
    double canonical_resolve_start = seconds_now();
    std::vector<std::array<CanonicalRef, 2>> right_references =
        resolve_canonical_refs(right_keys, *right_factory);
    std::vector<std::array<CanonicalRef, 2>> left_references =
        resolve_canonical_refs(left_keys, left_factory);
    double canonical_resolve_seconds = seconds_now() - canonical_resolve_start;
    std::printf("INPUT path=%s records=%llu kernels=%zu unique_left=%zu "
                "unique_right=%zu prefix_pairs=%d prefix_bits=%d "
                "load_seconds=%.6f\n",
                path.c_str(), (unsigned long long)records, edges.size(),
                left_keys.size(), right_keys.size(), PREFIX_PAIR_COUNT,
                2 * PREFIX_PAIR_COUNT, load_seconds);
    double canonical_upload_seconds = first_solve
        ? cache_upload_seconds : 0;
    DeviceWeightClassLayout left_class_layout;
    DirectWeightClassWorkspace direct_workspace;
    left_class_layout = build_direct_weight_class_layout_from_refs(
        left_references, left_factory, left_canonical, direct_workspace);
    double left_layout_seconds = left_class_layout.build_seconds;
    // The persistent right-batch workspace should start at the first recurring
    // batch's true high-water mark rather than retaining one-off left scratch.
    direct_workspace.reset();

    size_t free_bytes = 0;
    size_t total_device_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_device_bytes));
    size_t minimum_free_bytes = free_bytes;
    std::printf(
        "PREFIX_LEFT prefixes=%zu entries=%zu buckets=%zu "
        "canonical_entries=%zu free_bytes=%zu factory_seconds=%.6f "
        "canonical_upload_seconds=%.6f layout_seconds=%.6f "
        "kernel=direct-weight-class class_count=%zu max_classes=%zu "
        "candidate_slots=%zu fixed_candidate_slots=%zu "
        "canonical_weight_values=%zu max_distribution_weights=%zu "
        "weight_table_seconds=%.6f\n",
        left_keys.size(), left_class_layout.entry_count,
        left_class_layout.bucket_count,
        left_canonical.entry_count, free_bytes, factory_seconds,
        canonical_upload_seconds, left_layout_seconds,
        left_class_layout.class_count, left_class_layout.maximum_classes,
        left_class_layout.candidate_slots, left_class_layout.fixed_candidate_slots,
        left_canonical.class_weight_count,
        left_canonical.maximum_distribution_weights,
        left_canonical.weight_table_seconds);

    // The fixed reserve covers builder scratch, joins, results, and allocator
    // variation around the direct grouped layout.
    constexpr uint64_t estimated_right_bytes_per_entry =
        PREFIX_ESTIMATED_RIGHT_BYTES_PER_ENTRY;
    uint64_t right_entry_budget = free_bytes > device_reserve_bytes
        ? uint64_t(free_bytes - device_reserve_bytes) /
              estimated_right_bytes_per_entry
        : 1;
    // DirectWeightBuildDesc stores destination offsets in uint32_t.  A single
    // layout may contain exactly 2^32 entries but cannot address beyond it.
    right_entry_budget = std::min<uint64_t>(
        right_entry_budget, uint64_t(UINT32_MAX) + 1);
    std::vector<std::pair<size_t, size_t>> batch_ranges;
    std::vector<std::pair<size_t, size_t>> batch_right_ranges;
    size_t right_cursor = 0;
    for (size_t begin = 0; begin < edges.size();) {
        size_t end = begin;
        size_t right_begin = right_cursor;
        uint64_t batch_right_entries = 0;
        while (end < edges.size()) {
            PrefixKey right = edges[end].right;
            size_t group_end = end + 1;
            while (group_end < edges.size() && edges[group_end].right == right) {
                group_end++;
            }
            if (right_cursor >= right_keys.size() ||
                right_keys[right_cursor] != right) {
                throw std::runtime_error("prepared right ownership mismatch");
            }
            const auto& references = right_references[right_cursor];
            uint64_t group_entries =
                right_factory->descriptors[references[0].distribution].count +
                uint64_t(right_factory->descriptors[
                    references[1].distribution].count);
            bool exceeds_kernels = end != begin &&
                group_end - begin > batch_edges;
            bool exceeds_entries = end != begin &&
                batch_right_entries + group_entries > right_entry_budget;
            if (exceeds_kernels || exceeds_entries) break;
            batch_right_entries += group_entries;
            end = group_end;
            right_cursor++;
        }
        batch_ranges.emplace_back(begin, end);
        batch_right_ranges.emplace_back(right_begin, right_cursor);
        begin = end;
    }
    if (right_cursor != right_keys.size()) {
        throw std::runtime_error("incomplete prepared right ownership");
    }
    std::printf(
        "PREFIX_BATCH_PLAN batches=%zu batch_edges=%zu blocks_per_edge=2 "
        "right_entry_budget=%llu reserve_bytes=%llu "
        "estimated_bytes_per_entry=%llu\n",
        batch_ranges.size(), batch_edges,
        (unsigned long long)right_entry_budget,
        (unsigned long long)device_reserve_bytes,
        (unsigned long long)estimated_right_bytes_per_entry);

    U128 contribution = 0;
    U128 covered_weight = 0;
    U128 comparisons = 0;
    uint64_t right_groups = 0;
    uint64_t checked = 0;
    double right_layout_seconds = 0;
    uint64_t right_classes = 0;
    uint64_t right_candidate_slots = 0;
    uint64_t right_fixed_candidate_slots = 0;
    size_t maximum_right_classes = 0;
    size_t maximum_right_entries = 0;
    size_t maximum_right_buckets = 0;
    double gpu_seconds = 0;
    double validation_seconds = 0;
    DeviceBuffer<PrefixJoinDesc> device_joins;
    DeviceBuffer<unsigned long long> device_results;
    CudaEvent event_start;
    CudaEvent event_end;
    for (size_t batch_number = 0; batch_number < batch_ranges.size();
         batch_number++) {
        size_t begin = batch_ranges[batch_number].first;
        size_t end = batch_ranges[batch_number].second;
        size_t right_begin = batch_right_ranges[batch_number].first;
        size_t right_end = batch_right_ranges[batch_number].second;
        std::vector<PrefixKey> group_keys(
            right_keys.begin() + right_begin, right_keys.begin() + right_end);
        right_groups += group_keys.size();
        DeviceWeightClassLayout right_class_layout;
        std::vector<std::array<CanonicalRef, 2>> group_references(
            right_references.begin() + right_begin,
            right_references.begin() + right_end);
        right_class_layout = build_direct_weight_class_layout_from_refs(
            group_references, *right_factory, *right_canonical,
            direct_workspace);
        right_layout_seconds += right_class_layout.build_seconds;
        right_classes += right_class_layout.class_count;
        right_candidate_slots += right_class_layout.candidate_slots;
        right_fixed_candidate_slots += right_class_layout.fixed_candidate_slots;
        maximum_right_classes = std::max(
            maximum_right_classes, right_class_layout.maximum_classes);
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_device_bytes));
        minimum_free_bytes = std::min(minimum_free_bytes, free_bytes);
        size_t batch_right_entries = right_class_layout.entry_count;
        size_t batch_right_buckets = right_class_layout.bucket_count;
        maximum_right_entries =
            std::max(maximum_right_entries, batch_right_entries);
        maximum_right_buckets =
            std::max(maximum_right_buckets, batch_right_buckets);

        std::vector<PrefixJoinDesc> joins;
#ifndef TWCOLOUR_PRESERVE_JOIN_ORDER
        std::vector<uint64_t> join_work;
#endif
        joins.reserve((end - begin) * 2);
#ifndef TWCOLOUR_PRESERVE_JOIN_ORDER
        join_work.reserve((end - begin) * 2);
#endif
        size_t edge_index = begin;
        for (size_t group = 0; group < group_keys.size(); group++) {
            const PrefixPair& right = right_class_layout.pairs[group];
            while (edge_index < end && edges[edge_index].right == group_keys[group]) {
                uint32_t left_id = edge_left_ids[edge_index];
                const PrefixPair& left = left_class_layout.pairs[left_id];
                const PrefixDistribution left_distributions[2] = {
                    left.selected, left.complement};
                const PrefixDistribution right_distributions[2] = {
                    right.selected, right.complement};
                for (int complement = 0; complement < 2; complement++) {
                    const PrefixDistribution& lhs =
                        left_distributions[complement];
                    const PrefixDistribution& rhs =
                        right_distributions[complement];
                    joins.push_back(PrefixJoinDesc{
                        lhs.bucket_offset, rhs.bucket_offset,
                        lhs.bucket_count, rhs.bucket_count});
                    uint64_t work = uint64_t(lhs.entry_count) * rhs.entry_count;
#ifndef TWCOLOUR_PRESERVE_JOIN_ORDER
                    join_work.push_back(work);
#endif
                    comparisons += work;
                }
                edge_index++;
            }
        }
#ifndef TWCOLOUR_PRESERVE_JOIN_ORDER
        std::vector<uint32_t> result_slots =
            schedule_prefix_heavy_first(joins, join_work);
#endif
        device_joins.reserve(joins.size());
        device_results.reserve(joins.size());
        CUDA_CHECK(cudaMemcpy(device_joins.get(), joins.data(),
                              joins.size() * sizeof(PrefixJoinDesc),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(event_start));
        weight_class_prefix_joins<<<unsigned(joins.size()), THREADS>>>(
            left_class_layout.suffixes.get(),
            right_class_layout.suffixes.get(),
            left_class_layout.buckets.get(),
            right_class_layout.buckets.get(),
            left_class_layout.classes.get(),
            right_class_layout.classes.get(),
            device_joins.get(), device_results.get());
        CUDA_CHECK(cudaGetLastError());
        gpu_seconds += elapsed_kernel(event_start, event_end);
        std::vector<unsigned long long> scheduled_results(joins.size());
        CUDA_CHECK(cudaMemcpy(scheduled_results.data(), device_results.get(),
                              scheduled_results.size() *
                                  sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
        std::vector<unsigned long long> results;
#ifdef TWCOLOUR_PRESERVE_JOIN_ORDER
        results = std::move(scheduled_results);
#else
        results.resize(joins.size());
        for (size_t scheduled = 0; scheduled < joins.size(); scheduled++) {
            results[result_slots[scheduled]] = scheduled_results[scheduled];
        }
#endif

        edge_index = begin;
        size_t result_index = 0;
        while (edge_index < end) {
            const Edge& edge = edges[edge_index];
            uint64_t selected = results[result_index++];
            uint64_t complement = results[result_index++];
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
                validation_seconds += seconds_now() - validation_start;
                if (selected != expected_selected ||
                    complement != expected_complement) {
                    throw std::runtime_error(
                        "production prefix validation failed");
                }
                checked++;
            }
            covered_weight += U128(edge.factor) * edge.weight;
            U128 term = U128(edge.factor) * edge.weight * U128(selected) *
                        complement;
            if (~U128(0) - contribution < term) {
                throw std::overflow_error(
                    "per-work-item contribution exceeds unsigned 128-bit");
            }
            contribution += term;
            edge_index++;
        }
        free_weight_class_layout(right_class_layout);
        if (batch_number == 0 || (batch_number + 1) % 25 == 0 ||
            batch_number + 1 == batch_ranges.size()) {
            std::printf(
                "PREFIX_BATCH number=%zu kernels=%zu right_groups=%zu "
                "right_entries=%zu right_buckets=%zu gpu_seconds=%.6f "
                "completed=%zu/%zu\n",
                batch_number + 1, end - begin, group_keys.size(),
                batch_right_entries, batch_right_buckets,
                gpu_seconds, end, edges.size());
        }
    }

    double total_seconds = seconds_now() - total_start;
#if GRID_ROWS == 6 && GRID_COLUMNS == 9
    if (!filter_mod && start_record == 0 && end_record == 0 &&
        records == UINT64_C(130237768)) {
        const bool valid =
            labelled_weight == (U128(1) << CELLS) &&
            edges.size() == UINT64_C(71696841) &&
            covered_weight == (U128(1) << CELLS) &&
            u128_string(contribution) ==
                "197810562116614403484457574400";
        std::printf(
            "FULL_CHECK expected_records=130237768 "
            "expected_kernels=71696841 expected_weight=%s "
            "expected_contribution=197810562116614403484457574400 %s\n",
            u128_string(U128(1) << CELLS).c_str(),
            valid ? "OK" : "FAIL");
        if (!valid) return 1;
    }
#endif
    std::printf(
        "RESULT path=%s records=%llu labelled_weight=%s kernels=%zu "
        "covered_weight=%s right_groups=%llu verified=%llu comparisons=%s "
        "contribution=%s\n",
        path.c_str(), (unsigned long long)records,
        u128_string(labelled_weight).c_str(),
        edges.size(), u128_string(covered_weight).c_str(),
        (unsigned long long)right_groups, (unsigned long long)checked,
        u128_string(comparisons).c_str(), u128_string(contribution).c_str());
    std::printf(
        "TIMING prefix=1 load=%.6f factory=%.6f canonical_upload=%.6f "
        "canonical_resolve=%.6f "
        "left_layout=%.6f right_layout=%.6f "
        "right_classes=%llu max_right_classes=%zu "
        "right_candidate_slots=%llu right_fixed_candidate_slots=%llu "
        "kernel=direct-weight-class "
        "gpu=%.6f validation=%.6f total=%.6f "
        "comparisons_per_second=%.3f minimum_free_bytes=%zu\n",
        load_seconds, factory_seconds, canonical_upload_seconds,
        canonical_resolve_seconds,
        left_layout_seconds, right_layout_seconds,
        (unsigned long long)right_classes, maximum_right_classes,
        (unsigned long long)right_candidate_slots,
        (unsigned long long)right_fixed_candidate_slots,
        gpu_seconds, validation_seconds, total_seconds,
        gpu_seconds ? double(comparisons) / gpu_seconds : 0,
        minimum_free_bytes);
    std::ostringstream result_payload;
    result_payload
        << "transpose_quotient " << TWCOLOUR_TRANSPOSE_QUOTIENT << "\n"
        << "records " << records << "\n"
        << "labelled_weight " << u128_string(labelled_weight) << "\n"
        << "kernels " << edges.size() << "\n"
        << "covered_weight " << u128_string(covered_weight) << "\n"
        << "left_prefixes " << left_keys.size() << "\n"
        << "left_entries " << left_class_layout.entry_count << "\n"
        << "left_buckets " << left_class_layout.bucket_count << "\n"
        << "right_prefixes " << right_keys.size() << "\n"
        << "right_batches " << batch_ranges.size() << "\n"
        << "right_groups " << right_groups << "\n"
        << "maximum_right_entries " << maximum_right_entries << "\n"
        << "maximum_right_buckets " << maximum_right_buckets << "\n"
        << "effective_right_entry_cap " << right_entry_budget << "\n"
        << "verified " << checked << "\n"
        << "direct_comparisons " << u128_string(comparisons) << "\n"
        << "contribution " << u128_string(contribution) << "\n"
        << "minimum_free_bytes " << minimum_free_bytes << "\n"
        << "left_canonical_sources " << left_factory.canonical_keys.size()
        << "\n"
        << "right_canonical_sources " << right_factory->canonical_keys.size()
        << "\n"
        << "left_canonical_entries " << left_canonical.entry_count << "\n"
        << "right_canonical_entries " << right_canonical->entry_count << "\n"
        << "right_classes " << right_classes << "\n"
        << "maximum_right_classes " << maximum_right_classes << "\n"
        << "right_candidate_slots " << right_candidate_slots << "\n"
        << "right_fixed_candidate_slots " << right_fixed_candidate_slots
        << "\n"
        << "load_seconds " << load_seconds << "\n"
        << "cache_factory_seconds " << factory_seconds << "\n"
        << "canonical_upload_seconds " << canonical_upload_seconds << "\n"
        << "canonical_resolve_seconds " << canonical_resolve_seconds << "\n"
        << "left_layout_seconds " << left_layout_seconds << "\n"
        << "right_layout_seconds " << right_layout_seconds << "\n"
        << "gpu_seconds " << gpu_seconds << "\n"
        << "validation_seconds " << validation_seconds << "\n"
        << "total_seconds " << total_seconds << "\n";
    gpu_checkpoint::write_result(results_directory, item, provenance,
                                 result_payload.str());
    std::printf("CHECKPOINT id=%s result=%s\n", item.id.c_str(),
                gpu_checkpoint::result_path(results_directory, item).c_str());
    free_weight_class_layout(left_class_layout);
    solved_items++;
    }
    return 0;
}
