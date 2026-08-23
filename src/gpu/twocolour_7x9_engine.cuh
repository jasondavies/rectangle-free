#define GRID_ROWS 7
#define GRID_COLUMNS 9
#define LEFT_COLUMNS 4
#define RIGHT_COLUMNS 5
#define ORBIT_ROW_BITS 9
#define ORBIT_MAGIC "R7ORB09"


#include "twocolour_prefix_core.cuh"

#include "twocolour_weight_class_bmma.cuh"
#include "twocolour_canonical_device.cuh"

#include <filesystem>
#include <sstream>
#include <unordered_set>

#include "gpu_result_checkpoint.hpp"

namespace fs = std::filesystem;

struct DirectPackedSourceRef {
    uint64_t source_offset;
    uint32_t row_map;
    uint32_t destination_offset;
    uint32_t count;
    PackedWeightSpan weights;
};

static DeviceWeightClassLayout build_direct_packed_weight_layout_from_sources(
    const std::vector<DirectPackedSourceRef>& sources, size_t pair_count,
    const uint64_t* device_entries,
    const PackedUniversalCache& cache,
    DirectWeightClassWorkspace& workspace) {
    if (!device_entries || !cache.device_class_weights ||
        !cache.device_class_orbit_sizes ||
        cache.class_orbit_sizes.size() != cache.class_weights.size() ||
        cache.weight_spans.size() != cache.keys.size()) {
        throw std::runtime_error("direct packed weight cache is unavailable");
    }
    if (sources.size() != pair_count * 2) {
        throw std::runtime_error("direct packed source count mismatch");
    }
    double start = seconds_now();
    std::vector<DirectWeightBuildDesc> descriptions(sources.size());
    uint64_t total_entries = 0;
    uint64_t maximum_destination = 0;
    for (size_t logical = 0; logical < sources.size(); logical++) {
        const DirectPackedSourceRef& source = sources[logical];
        // Empty exact distributions have neither entries nor weight classes.
        // They are valid and their zero-width descriptor is skipped by every
        // entry pass; only a disagreement between the two counts is corrupt.
        if ((source.count == 0) != (source.weights.count == 0) ||
            source.weights.count > WEIGHT_CLASS_HASH_SLOTS ||
            uint64_t(source.destination_offset) + source.count >
                uint64_t(UINT32_MAX) + 1) {
            throw std::overflow_error(
                "direct packed grouped layout exceeds exact bounds");
        }
        descriptions[logical] = DirectWeightBuildDesc{
            source.source_offset, source.row_map, source.destination_offset,
            source.count, 0, 0, source.weights.offset,
            source.weights.count, 0};
        total_entries += source.count;
        maximum_destination = std::max<uint64_t>(
            maximum_destination,
            uint64_t(source.destination_offset) + source.count);
    }
    if (total_entries != maximum_destination ||
        total_entries > uint64_t(UINT32_MAX) + 1) {
        throw std::overflow_error(
            "direct packed grouped destinations are not contiguous");
    }
    double plan_seconds = seconds_now() - start;
    DeviceWeightClassLayout result =
        build_direct_weight_class_layout_from_descriptions(
            std::move(descriptions), pair_count, total_entries,
            cache.device_class_weights,
            cache.device_class_orbit_sizes,
            workspace,
            start,
            [&](DirectWeightBuildDesc* device_descriptions, uint32_t* dense) {
                histogram_direct_weight_prefixes_packed<<<
                    unsigned(sources.size()), THREADS>>>(
                    device_entries, device_descriptions, dense);
            },
            [&](DirectWeightBuildDesc* device_descriptions, uint32_t* dense,
                DirectBucketAux* bucket_aux, uint32_t* candidates,
                uint32_t* failure) {
                histogram_direct_weight_classes_packed<<<
                    unsigned(sources.size()), THREADS>>>(
                    device_entries, device_descriptions, dense, bucket_aux,
                    candidates, failure);
            },
            [&](DirectWeightBuildDesc* device_descriptions, uint32_t* dense,
                DirectBucketAux* bucket_aux, uint32_t* candidates,
                PrefixSuffix* suffixes, uint32_t* failure) {
                scatter_direct_weight_classes_packed<<<
                    unsigned(sources.size()), THREADS>>>(
                    device_entries, device_descriptions, dense, bucket_aux,
                    candidates, suffixes, failure);
            });
    result.plan_seconds = plan_seconds;
    return result;
}


static DeviceWeightClassLayout
build_direct_packed_weight_layout_from_prefetched(
    PrefetchedPackedLayout& prepared, const PackedUniversalCache& cache,
    DirectWeightClassWorkspace& workspace, double& wait_seconds) {
    double wait_start = seconds_now();
    CUDA_CHECK(cudaEventSynchronize(prepared.upload_end));
    wait_seconds += seconds_now() - wait_start;
    float upload_milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&upload_milliseconds,
                                    prepared.upload_start,
                                    prepared.upload_end));
    CUDA_CHECK(cudaEventDestroy(prepared.upload_end));
    CUDA_CHECK(cudaEventDestroy(prepared.upload_start));
    prepared.upload_end = nullptr;
    prepared.upload_start = nullptr;

    double source_plan_start = seconds_now();
    if (prepared.references.size() != prepared.pair_count) {
        throw std::runtime_error("direct prefetched reference mismatch");
    }
    std::vector<DirectPackedSourceRef> sources(prepared.pair_count * 2);
    std::vector<uint8_t> populated(sources.size());
    for (const PrefetchedPackedChunk& chunk : prepared.chunks) {
        if (chunk.descriptions.size() != chunk.logical_indices.size()) {
            throw std::runtime_error("direct prefetched chunk mismatch");
        }
        for (size_t index = 0; index < chunk.descriptions.size(); index++) {
            uint32_t logical = chunk.logical_indices[index];
            if (logical >= sources.size() || populated[logical]) {
                throw std::runtime_error(
                    "direct prefetched logical source mismatch");
            }
            const GpuBucketBuildDesc& description =
                chunk.descriptions[index];
            const PackedLayoutSourceRef& reference =
                prepared.references[logical / 2][logical & 1U];
            sources[logical] = DirectPackedSourceRef{
                description.source_offset, uint32_t(description.row_map),
                description.destination_offset, description.count,
                cache.weight_spans[reference.distribution]};
            populated[logical] = 1;
        }
    }
    if (std::find(populated.begin(), populated.end(), uint8_t(0)) !=
        populated.end()) {
        throw std::runtime_error("direct prefetched source is missing");
    }
    double source_plan_seconds = seconds_now() - source_plan_start;

    DeviceWeightClassLayout result =
        build_direct_packed_weight_layout_from_sources(
            sources, prepared.pair_count, cache.device_prefetch.get(), cache,
            workspace);
    result.plan_seconds += prepared.plan_seconds + source_plan_seconds;
    result.source_gather_seconds = prepared.gather_seconds;
    result.source_upload_seconds = upload_milliseconds / 1000.0;
    result.source_entries = prepared.source_entries;
    result.source_chunks = prepared.chunks.size();
    result.total_seconds += prepared.plan_seconds + source_plan_seconds +
                            prepared.gather_seconds +
                            result.source_upload_seconds;
    return result;
}

static DeviceWeightClassLayout
build_direct_packed_weight_layout_from_resident(
    const std::vector<std::array<PackedLayoutSourceRef, 2>>& references,
    const PackedUniversalCache& cache,
    DirectWeightClassWorkspace& workspace) {
    if (!cache.device_resident || !cache.device_entries) {
        throw std::runtime_error("resident packed cache is unavailable");
    }
    double plan_start = seconds_now();
    std::vector<DirectPackedSourceRef> sources;
    sources.reserve(references.size() * 2);
    uint64_t destination = 0;
    for (const auto& pair : references) {
        for (const PackedLayoutSourceRef& reference : pair) {
            if (reference.distribution >= cache.counts.size()) {
                throw std::runtime_error(
                    "resident packed source reference is out of range");
            }
            uint32_t count = cache.counts[reference.distribution];
            if (destination + count > uint64_t(UINT32_MAX) + 1) {
                throw std::overflow_error(
                    "resident packed layout exceeds 32-bit offsets");
            }
            sources.push_back(DirectPackedSourceRef{
                cache.offsets[reference.distribution], reference.row_map,
                uint32_t(destination), count,
                cache.weight_spans[reference.distribution]});
            destination += count;
        }
    }
    double reference_plan_seconds = seconds_now() - plan_start;
    DeviceWeightClassLayout result =
        build_direct_packed_weight_layout_from_sources(
            sources, references.size(), cache.device_entries.get(), cache,
            workspace);
    result.plan_seconds += reference_plan_seconds;
    result.total_seconds += reference_plan_seconds;
    return result;
}


using PackedSolveLayout = DeviceWeightClassLayout;
// Direct construction retains a four-byte suffix per entry plus sparse bucket
// and class metadata.  Six bytes per logical entry retained more than five GiB
// free in the production-shaped L40S gate; the separate four-GiB reserve covers
// transient builder scratch and workload variation.
constexpr size_t PACKED_SOLVE_ENTRY_BYTES = 6;
static_assert(PACKED_SOLVE_ENTRY_BYTES >= sizeof(PrefixSuffix),
              "direct packed estimate must include every suffix");

static PackedSolveLayout build_direct_left_layout(
    const std::vector<PrefixKey>& keys, const CanonicalFactory& factory) {
    std::vector<std::array<CanonicalRef, 2>> references(keys.size());
    for (size_t index = 0; index < keys.size(); index++) {
        const RawCanonicalPair& raw = lookup_raw(factory, keys[index]);
        references[index] = {raw.selected, raw.complement};
    }
    ProductionCanonicalDevice canonical =
        upload_production_canonical(factory);
    DirectWeightClassWorkspace workspace;
    PackedSolveLayout result = build_direct_weight_class_layout_from_refs(
        references, factory, canonical, workspace);
    result.upload_seconds = canonical.weight_table_seconds;
    result.total_seconds += canonical.weight_table_seconds;
    return result;
}


static void free_packed_solve_layout(PackedSolveLayout& layout) {
    free_weight_class_layout(layout);
}

static uint64_t packed_solve_bucket_count(const PackedSolveLayout& layout) {
    return layout.bucket_count;
}

using PackedWorkItem = gpu_checkpoint::WorkItem;
using PackedRunProvenance = gpu_checkpoint::RunProvenance;
using PackedWorkProvenance = gpu_checkpoint::WorkProvenance;

static PackedRunProvenance packed_run_provenance(
    const std::string& executable, const std::string& canonical_cache_sha256) {
    std::ostringstream configuration;
    configuration << "geometry=7x9;left_columns=" << LEFT_COLUMNS
                  << ";right_columns=" << RIGHT_COLUMNS
                  << ";prefix_pairs=" << PREFIX_PAIR_COUNT
                  << ";task_chunk=" << PREFIX_TASK_CHUNK
                  << ";threads=" << THREADS
                  << ";token_plane_quotient=1"
                  << ";join=weight_class_bmma_dual_plane";
    return PackedRunProvenance{
        "RECT7X9_PACKED_RESULT", "7x9", sha256_file(executable),
        sha256_string(configuration.str()), canonical_cache_sha256};
}

static PackedWorkProvenance packed_work_provenance(
    const PackedRunProvenance& run, const PackedWorkItem& item) {
    return gpu_checkpoint::work_provenance(run, item);
}

struct PackedWorkResult {
    uint64_t records = 0;
    uint64_t kernels = 0;
    uint64_t left_prefixes = 0;
    uint64_t left_entries = 0;
    uint64_t left_buckets = 0;
    uint64_t right_prefixes = 0;
    uint64_t right_batches = 0;
    uint64_t maximum_right_entries = 0;
    uint64_t maximum_right_buckets = 0;
    uint64_t effective_right_entry_cap = 0;
    uint64_t verified = 0;
    U128 labelled_weight = 0;
    U128 covered_weight = 0;
    U128 direct_comparisons = 0;
    U128 contribution = 0;
    double load_seconds = 0;
    double left_factory_seconds = 0;
    double left_layout_seconds = 0;
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
    double join_plan_seconds = 0;
    double join_upload_seconds = 0;
    double result_allocation_seconds = 0;
    double result_download_seconds = 0;
    double batch_buffer_free_seconds = 0;
    double right_layout_free_seconds = 0;
    uint64_t right_source_entries = 0;
    uint64_t right_source_chunks = 0;
    double gpu_seconds = 0;
    double validation_seconds = 0;
    double total_seconds = 0;
    size_t minimum_free_bytes = SIZE_MAX;
};

static std::vector<PackedWorkItem> read_work_manifest(
    const std::string& path) {
    return gpu_checkpoint::read_work_manifest(path);
}

using PackedWorkClaim = gpu_checkpoint::WorkClaim;

static bool validated_result_exists(
    const fs::path& directory, const PackedWorkItem& item,
    const PackedWorkProvenance& provenance) {
    static const std::vector<std::string> required = {
        "records", "labelled_weight", "kernels", "covered_weight",
        "direct_comparisons", "contribution", "total_seconds"};
    return gpu_checkpoint::validated_result_exists(
        directory, item, provenance, required);
}

static void write_work_result(const fs::path& directory,
                              const PackedWorkItem& item,
                              const PackedWorkResult& result,
                              const PackedWorkProvenance& provenance) {
    std::ostringstream payload;
    payload << "records " << result.records << "\n"
            << "labelled_weight " << u128_string(result.labelled_weight) << "\n"
            << "kernels " << result.kernels << "\n"
            << "covered_weight " << u128_string(result.covered_weight) << "\n"
            << "left_prefixes " << result.left_prefixes << "\n"
            << "left_entries " << result.left_entries << "\n"
            << "left_buckets " << result.left_buckets << "\n"
            << "right_prefixes " << result.right_prefixes << "\n"
            << "right_batches " << result.right_batches << "\n"
            << "maximum_right_entries " << result.maximum_right_entries << "\n"
            << "maximum_right_buckets " << result.maximum_right_buckets << "\n"
            << "effective_right_entry_cap "
            << result.effective_right_entry_cap << "\n"
            << "verified " << result.verified << "\n"
            << "direct_comparisons " << u128_string(result.direct_comparisons)
            << "\n"
            << "contribution " << u128_string(result.contribution) << "\n"
            << "minimum_free_bytes " << result.minimum_free_bytes << "\n"
            << "load_seconds " << result.load_seconds << "\n"
            << "left_factory_seconds " << result.left_factory_seconds << "\n"
            << "left_layout_seconds " << result.left_layout_seconds << "\n"
            << "right_layout_seconds " << result.right_layout_seconds << "\n"
            << "right_source_plan_seconds "
            << result.right_source_plan_seconds << "\n"
            << "right_source_sort_seconds "
            << result.right_source_sort_seconds << "\n"
            << "right_schedule_seconds " << result.right_schedule_seconds << "\n"
            << "right_plan_seconds " << result.right_plan_seconds << "\n"
            << "right_upload_seconds " << result.right_upload_seconds << "\n"
            << "right_source_gather_seconds "
            << result.right_source_gather_seconds << "\n"
            << "right_source_upload_seconds "
            << result.right_source_upload_seconds << "\n"
            << "right_histogram_seconds " << result.right_histogram_seconds
            << "\n"
            << "right_scatter_seconds " << result.right_scatter_seconds << "\n"
            << "right_metadata_seconds " << result.right_metadata_seconds << "\n"
            << "join_plan_seconds " << result.join_plan_seconds << "\n"
            << "join_upload_seconds " << result.join_upload_seconds << "\n"
            << "result_allocation_seconds "
            << result.result_allocation_seconds << "\n"
            << "result_download_seconds " << result.result_download_seconds
            << "\n"
            << "batch_buffer_free_seconds "
            << result.batch_buffer_free_seconds << "\n"
            << "right_layout_free_seconds "
            << result.right_layout_free_seconds << "\n"
            << "right_source_entries " << result.right_source_entries << "\n"
            << "right_source_chunks " << result.right_source_chunks << "\n"
            << "gpu_seconds " << result.gpu_seconds << "\n"
            << "validation_seconds " << result.validation_seconds << "\n"
            << "total_seconds " << result.total_seconds << "\n";
    gpu_checkpoint::write_result(directory, item, provenance, payload.str());
}

struct PackedRawSourcePlan {
    uint64_t entries;
    std::array<size_t, 2> distributions;
    std::array<uint32_t, 2> row_maps;
};



static PackedRawSourcePlan packed_raw_source_plan(
    const PackedUniversalCache& cache, PrefixKey raw) {
    const PrefixKey full_mask = (PrefixKey(1) << (RIGHT_COLUMNS * ROWS)) - 1;
    CanonicalForm selected = canonical_prefix(raw, RIGHT_COLUMNS);
    CanonicalForm complement = canonical_prefix(raw ^ full_mask, RIGHT_COLUMNS);
    size_t selected_id = packed_distribution_index(cache, selected.key);
    size_t complement_id = packed_distribution_index(cache, complement.key);
    return PackedRawSourcePlan{
        uint64_t(cache.counts[selected_id]) + cache.counts[complement_id],
        {selected_id, complement_id},
        {selected.row_map, complement.row_map}};
}

static PackedWorkResult solve_packed_work_item(
    const PackedWorkItem& item, const PackedUniversalCache& cache,
    uint64_t requested_right_entries, uint64_t verify_joins) {
    const size_t reserve_bytes = cache.memory_reserve_bytes;
    double total_start = seconds_now();
    PackedWorkResult result;
    double load_start = seconds_now();
    std::vector<Edge> edges =
        read_edges(item.path, item.start, item.end, item.filter_mod, item.filter_id,
                   result.labelled_weight, result.records);
    mark_validation_edges(edges, verify_joins);
    result.load_seconds = seconds_now() - load_start;
    result.kernels = edges.size();
    if (edges.empty()) {
        result.minimum_free_bytes = cache.free_bytes_after;
        result.total_seconds = seconds_now() - total_start;
        return result;
    }

    std::vector<PrefixKey> left_keys = unique_lefts(edges);
    std::vector<PrefixKey> right_keys = unique_rights(edges);
    std::vector<uint32_t> edge_left_ids =
        resolve_edge_left_ids(edges, left_keys);
    result.left_prefixes = left_keys.size();
    result.right_prefixes = right_keys.size();
    double left_factory_start = seconds_now();
    CanonicalFactory left_factory =
        build_canonical_factory(left_keys, LEFT_COLUMNS);
    result.left_factory_seconds = seconds_now() - left_factory_start;

    uint64_t left_entries = 0;
    for (PrefixKey key : left_keys) {
        const RawCanonicalPair& raw = lookup_raw(left_factory, key);
        left_entries += left_factory.descriptors[raw.selected.distribution].count;
        left_entries += left_factory.descriptors[raw.complement.distribution].count;
    }
    size_t free_before_left = 0;
    size_t total_device_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before_left, &total_device_bytes));
    uint64_t estimated_left_peak =
        left_entries * PACKED_SOLVE_ENTRY_BYTES +
        left_factory.entries.size() * sizeof(Entry);
    if (free_before_left <= reserve_bytes ||
        estimated_left_peak > free_before_left - reserve_bytes) {
        throw std::runtime_error("left layout exceeds packed-worker memory budget");
    }
    PackedSolveLayout left_layout =
        build_direct_left_layout(left_keys, left_factory);
    result.left_layout_seconds = left_layout.total_seconds;
    result.left_entries = left_layout.entry_count;
    result.left_buckets = packed_solve_bucket_count(left_layout);
    left_factory = CanonicalFactory{};


    size_t free_after_left = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after_left, &total_device_bytes));
    result.minimum_free_bytes = free_after_left;
    if (free_after_left <= reserve_bytes) {
        free_packed_solve_layout(left_layout);
        throw std::runtime_error("no right-layout headroom after persistent left");
    }
    uint64_t memory_entry_cap =
        (free_after_left - reserve_bytes) / PACKED_SOLVE_ENTRY_BYTES;
    // The prefetched source stream is eight bytes per canonical entry.  It can
    // never exceed the logical labelled stream, so this is a safe batch bound
    // even before source deduplication.
    if (!cache.device_resident) {
        memory_entry_cap = std::min<uint64_t>(
            memory_entry_cap, cache.prefetch_bytes / sizeof(uint64_t));
    }
    uint64_t right_entry_cap =
        std::min<uint64_t>(requested_right_entries, memory_entry_cap);
    right_entry_cap = std::min<uint64_t>(right_entry_cap, UINT32_MAX);
    result.effective_right_entry_cap = right_entry_cap;
    if (!right_entry_cap) {
        free_packed_solve_layout(left_layout);
        throw std::runtime_error("right-layout entry cap is zero");
    }

    double right_source_plan_start = seconds_now();
    std::vector<PackedRawSourcePlan> right_plans(right_keys.size());
#pragma omp parallel for schedule(static)
    for (long long index = 0; index < (long long)right_keys.size(); index++) {
        right_plans[size_t(index)] =
            packed_raw_source_plan(cache, right_keys[size_t(index)]);
    }
    result.right_source_plan_seconds =
        seconds_now() - right_source_plan_start;
    for (size_t index = 0; index < right_keys.size(); index++) {
        if (right_plans[index].entries > right_entry_cap) {
            free_packed_solve_layout(left_layout);
            throw std::runtime_error("one right prefix exceeds the memory cap");
        }
    }

    if (right_keys.size() > UINT32_MAX || edges.size() > UINT32_MAX) {
        throw std::overflow_error("packed scheduling index width exceeded");
    }
    double right_schedule_start = seconds_now();
    std::vector<uint32_t> right_order(right_keys.size());
    std::iota(right_order.begin(), right_order.end(), uint32_t(0));
    double right_source_sort_start = seconds_now();
    std::sort(right_order.begin(), right_order.end(), [&](uint32_t lhs,
                                                           uint32_t rhs) {
        std::array<size_t, 2> left = right_plans[lhs].distributions;
        std::array<size_t, 2> right = right_plans[rhs].distributions;
        if (left[0] > left[1]) std::swap(left[0], left[1]);
        if (right[0] > right[1]) std::swap(right[0], right[1]);
        if (left != right) return left < right;
        return lhs < rhs;
    });
    result.right_source_sort_seconds =
        seconds_now() - right_source_sort_start;
    std::vector<std::vector<uint32_t>> batch_right_indices;
    uint64_t planned_batch_entries = 0;
    for (uint32_t index : right_order) {
        uint64_t entries = right_plans[index].entries;
        if (batch_right_indices.empty() ||
            (planned_batch_entries &&
             planned_batch_entries + entries > right_entry_cap)) {
            batch_right_indices.emplace_back();
            planned_batch_entries = 0;
        }
        batch_right_indices.back().push_back(index);
        planned_batch_entries += entries;
    }
    std::vector<uint16_t> right_batch(right_keys.size(), UINT16_MAX);
    for (size_t batch = 0; batch < batch_right_indices.size(); batch++) {
        if (batch >= UINT16_MAX) {
            throw std::overflow_error("too many packed right batches");
        }
        std::sort(batch_right_indices[batch].begin(),
                  batch_right_indices[batch].end());
        for (uint32_t index : batch_right_indices[batch]) {
            if (right_batch[index] != UINT16_MAX) {
                throw std::runtime_error("duplicate packed right ownership");
            }
            right_batch[index] = uint16_t(batch);
        }
    }
    std::vector<std::vector<uint32_t>> batch_edge_indices(
        batch_right_indices.size());
    size_t right_cursor = 0;
    for (size_t edge_index = 0; edge_index < edges.size(); edge_index++) {
        while (right_cursor < right_keys.size() &&
               right_keys[right_cursor] < edges[edge_index].right) {
            right_cursor++;
        }
        if (right_cursor == right_keys.size() ||
            right_keys[right_cursor] != edges[edge_index].right ||
            right_batch[right_cursor] == UINT16_MAX) {
            throw std::runtime_error("packed edge/right ownership mismatch");
        }
        batch_edge_indices[right_batch[right_cursor]].push_back(
            uint32_t(edge_index));
    }
    result.right_schedule_seconds = seconds_now() - right_schedule_start;

    cudaEvent_t event_start, event_end;
    CUDA_CHECK(cudaEventCreate(&event_start));
    CUDA_CHECK(cudaEventCreate(&event_end));
    double pipeline_wait_seconds = 0;
    DirectWeightClassWorkspace weight_workspace;
    auto batch_references = [&](size_t batch_index) {
        const std::vector<uint32_t>& indices =
            batch_right_indices[batch_index];
        std::vector<std::array<PackedLayoutSourceRef, 2>> references;
        references.reserve(indices.size());
        for (uint32_t index : indices) {
            references.push_back(std::array<PackedLayoutSourceRef, 2>{
                PackedLayoutSourceRef{right_plans[index].distributions[0],
                                      right_plans[index].row_maps[0]},
                PackedLayoutSourceRef{right_plans[index].distributions[1],
                                      right_plans[index].row_maps[1]}});
        }
        return references;
    };
    PrefetchedPackedLayout prefetched;
    uint64_t pipeline_max_source_entries = 0;
    if (!cache.device_resident) {
        prefetched = prefetch_host_packed_layout(batch_references(0), cache);
        pipeline_max_source_entries = prefetched.source_entries;
    }
    size_t completed_edges = 0;
    for (size_t batch_index = 0; batch_index < batch_right_indices.size();
         batch_index++) {
        const std::vector<uint32_t>& group_indices =
            batch_right_indices[batch_index];
        const std::vector<uint32_t>& group_edges =
            batch_edge_indices[batch_index];
        uint64_t group_entries = 0;
        std::vector<PrefixKey> group_keys;
        group_keys.reserve(group_indices.size());
        for (uint32_t index : group_indices) {
            group_entries += right_plans[index].entries;
            group_keys.push_back(right_keys[index]);
        }
        if (group_keys.empty() || group_entries > right_entry_cap) {
            throw std::runtime_error("invalid planned packed right batch");
        }
        std::vector<std::array<PackedLayoutSourceRef, 2>> references =
            batch_references(batch_index);
        PackedSolveLayout right_layout = cache.device_resident
            ? build_direct_packed_weight_layout_from_resident(
                  references, cache, weight_workspace)
            : build_direct_packed_weight_layout_from_prefetched(
                  prefetched, cache, weight_workspace,
                  pipeline_wait_seconds);
        result.right_layout_seconds += right_layout.total_seconds;
        result.right_plan_seconds += right_layout.plan_seconds;
        result.right_upload_seconds += right_layout.upload_seconds;
        result.right_source_gather_seconds +=
            right_layout.source_gather_seconds;
        result.right_source_upload_seconds +=
            right_layout.source_upload_seconds;
        result.right_histogram_seconds += right_layout.histogram_seconds;
        result.right_scatter_seconds += right_layout.scatter_seconds;
        result.right_metadata_seconds += right_layout.metadata_seconds;
        result.right_source_entries += right_layout.source_entries;
        result.right_source_chunks += right_layout.source_chunks;
        result.maximum_right_entries =
            std::max<uint64_t>(result.maximum_right_entries,
                               right_layout.entry_count);
        result.maximum_right_buckets =
            std::max<uint64_t>(result.maximum_right_buckets,
                               packed_solve_bucket_count(right_layout));
        size_t free_with_right = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_with_right, &total_device_bytes));
        result.minimum_free_bytes =
            std::min(result.minimum_free_bytes, free_with_right);



        std::vector<PrefixJoinDesc> joins;
        double join_plan_start = seconds_now();
        joins.reserve(group_edges.size() * 2);
        for (uint32_t edge_index : group_edges) {
            const Edge& edge = edges[edge_index];
            auto found = std::lower_bound(group_keys.begin(), group_keys.end(),
                                          edge.right);
            if (found == group_keys.end() || *found != edge.right) {
                throw std::runtime_error("production right ownership mismatch");
            }
            const PrefixPair& left = left_layout.pairs[edge_left_ids[edge_index]];
            size_t right_pair_index = size_t(found - group_keys.begin());
            const PrefixPair& right = right_layout.pairs[right_pair_index];
            const PrefixDistribution lhs[2] = {left.selected, left.complement};
            const PrefixDistribution rhs[2] = {right.selected, right.complement};
            for (int complement = 0; complement < 2; complement++) {
                joins.push_back(PrefixJoinDesc{
                    lhs[complement].bucket_offset, rhs[complement].bucket_offset,
                    lhs[complement].bucket_count, rhs[complement].bucket_count});
                result.direct_comparisons +=
                    U128(lhs[complement].entry_count) *
                    rhs[complement].entry_count;
            }
        }

        result.join_plan_seconds += seconds_now() - join_plan_start;

        double join_upload_start = seconds_now();
        auto* device_joins = upload_vector(joins);
        result.join_upload_seconds += seconds_now() - join_upload_start;
        double result_allocation_start = seconds_now();
        unsigned long long* device_results = nullptr;
        CUDA_CHECK(cudaMalloc(&device_results,
                              joins.size() * sizeof(unsigned long long)));
        result.result_allocation_seconds +=
            seconds_now() - result_allocation_start;
        CUDA_CHECK(cudaEventRecord(event_start));
        weight_class_prefix_joins<<<unsigned(joins.size()), THREADS>>>(
            left_layout.suffixes.get(), right_layout.suffixes.get(),
            left_layout.buckets.get(), right_layout.buckets.get(),
            left_layout.classes.get(), right_layout.classes.get(),
            device_joins, device_results);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(event_end));
        if (!cache.device_resident &&
            batch_index + 1 < batch_right_indices.size()) {
            prefetched = prefetch_host_packed_layout(
                batch_references(batch_index + 1), cache);
            pipeline_max_source_entries = std::max(
                pipeline_max_source_entries, prefetched.source_entries);
        }
        CUDA_CHECK(cudaEventSynchronize(event_end));
        float kernel_milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_milliseconds, event_start,
                                        event_end));
        result.gpu_seconds += kernel_milliseconds / 1000.0;
        std::vector<unsigned long long> results(joins.size());
        double result_download_start = seconds_now();
        CUDA_CHECK(cudaMemcpy(results.data(), device_results,
                              results.size() * sizeof(results[0]),
                              cudaMemcpyDeviceToHost));
        result.result_download_seconds += seconds_now() - result_download_start;

        for (size_t local_edge = 0; local_edge < group_edges.size();
             local_edge++) {
            const Edge& edge = edges[group_edges[local_edge]];
            uint64_t selected = results[local_edge * 2];
            uint64_t complement = results[local_edge * 2 + 1];
            if (edge.validate) {
                double validation_start = seconds_now();
                DistributionPair validation_left =
                    build_pair(edge.left, LEFT_COLUMNS);
                DistributionPair validation_right =
                    build_pair(edge.right, RIGHT_COLUMNS);
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
                result.validation_seconds += seconds_now() - validation_start;
                if (selected != expected_selected || complement != expected_complement) {
                    throw std::runtime_error("packed production validation failed");
                }
                result.verified++;
            }
            result.covered_weight += U128(edge.factor) * edge.weight;
            result.contribution += U128(edge.factor) * edge.weight *
                                   U128(selected) * complement;
        }

        double batch_buffer_free_start = seconds_now();
        CUDA_CHECK(cudaFree(device_results));
        CUDA_CHECK(cudaFree(device_joins));
        result.batch_buffer_free_seconds +=
            seconds_now() - batch_buffer_free_start;
        double right_layout_free_start = seconds_now();
        free_packed_solve_layout(right_layout);
        result.right_layout_free_seconds += seconds_now() - right_layout_free_start;
        completed_edges += group_edges.size();
        result.right_batches++;
        if (result.right_batches == 1 || result.right_batches % 25 == 0 ||
            batch_index + 1 == batch_right_indices.size()) {
            std::printf(
                "PACKED_BATCH id=%s batch=%llu right_prefixes=%zu entries=%llu "
                "kernels=%zu completed=%zu/%zu gpu_seconds=%.6f\n",
                item.id.c_str(), (unsigned long long)result.right_batches,
                group_keys.size(), (unsigned long long)group_entries,
                group_edges.size(), completed_edges, edges.size(),
                result.gpu_seconds);
        }
    }
    if (completed_edges != edges.size()) {
        throw std::runtime_error("production packed solve left edges unprocessed");
    }
    std::printf(
        "PACKED_PIPELINE batches=%llu prefetch_wait_seconds=%.6f "
        "logical_plan_seconds=%.6f logical_gather_seconds=%.6f "
        "logical_upload_seconds=%.6f max_source_entries=%llu "
        "max_source_gib=%.6f exact=OK\n",
        (unsigned long long)result.right_batches, pipeline_wait_seconds,
        result.right_plan_seconds, result.right_source_gather_seconds,
        result.right_source_upload_seconds,
        (unsigned long long)pipeline_max_source_entries,
        double(pipeline_max_source_entries * sizeof(uint64_t)) /
            (1024.0 * 1024.0 * 1024.0));
    CUDA_CHECK(cudaEventDestroy(event_end));
    CUDA_CHECK(cudaEventDestroy(event_start));
    free_packed_solve_layout(left_layout);
    result.total_seconds = seconds_now() - total_start;
    return result;
}
