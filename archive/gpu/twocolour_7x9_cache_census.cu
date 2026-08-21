#include "../../src/gpu/twocolour_7x9_engine.cuh"

struct StagingCensus {
    uint64_t chunks = 0;
    uint64_t transferred_entries = 0;
    uint64_t maximum_source_entries = 0;
    uint64_t maximum_descriptions = 0;
};

struct SourceScheduleCensus {
    uint64_t batches = 0;
    uint64_t source_entries = 0;
    uint64_t minimum_batch_entries = UINT64_MAX;
    uint64_t maximum_batch_entries = 0;
    uint64_t raw_intervals = 0;
    uint64_t maximum_raw_intervals = 0;
    StagingCensus staging;
};

struct PackedSourceRef {
    size_t distribution;
    uint32_t row_map;
};

static uint64_t packed_transform_key(const PackedSourceRef& reference) {
    if (reference.distribution > UINT32_MAX) {
        throw std::overflow_error("packed distribution id exceeds uint32_t");
    }
    return (uint64_t(reference.distribution) << 32) | reference.row_map;
}

static PackedUniversalCache build_packed_universal_metadata(const char* path) {
    constexpr size_t expected_records = 136758;
    constexpr size_t expected_entries = UINT64_C(4740574641);
    double total_start = seconds_now();
    PackedUniversalCache cache;

    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open universal 7x5 orbit file");
    char magic[8];
    uint32_t columns = 0;
    uint64_t records = 0;
    input.read(magic, sizeof(magic));
    input.read(reinterpret_cast<char*>(&columns), sizeof(columns));
    input.read(reinterpret_cast<char*>(&records), sizeof(records));
    if (!input || std::memcmp(magic, "R7ORB01", 7) || columns != 5 ||
        records != expected_records) {
        throw std::runtime_error("invalid complete 7x5 orbit file");
    }
    cache.keys.reserve(records);
    U128 labelled_weight = 0;
    for (uint64_t index = 0; index < records; index++) {
        OrbitRecord record{};
        input.read(reinterpret_cast<char*>(&record), sizeof(record));
        if (!input) throw std::runtime_error("truncated complete 7x5 orbit file");
        cache.keys.push_back(
            canonical_prefix(compact_orbit_7x5_key(record.key), 5).key);
        labelled_weight += record.weight;
    }
    char trailing;
    if (input.read(&trailing, 1)) {
        throw std::runtime_error("trailing complete 7x5 orbit data");
    }
    std::sort(cache.keys.begin(), cache.keys.end());
    if (std::adjacent_find(cache.keys.begin(), cache.keys.end()) !=
            cache.keys.end() ||
        labelled_weight != (U128(1) << 35)) {
        throw std::runtime_error("universal 7x5 orbit validation failed");
    }

    cache.counts.resize(cache.keys.size());
    double count_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
    for (long long index = 0; index < (long long)cache.keys.size(); index++) {
        Distribution distribution =
            build_distribution(cache.keys[size_t(index)], 5, false);
        cache.counts[size_t(index)] = uint32_t(distribution.entries.size());
    }
    cache.count_seconds = seconds_now() - count_start;
    cache.offsets.resize(cache.keys.size());
    uint64_t total_entries = 0;
    for (size_t index = 0; index < cache.keys.size(); index++) {
        cache.offsets[index] = total_entries;
        total_entries += cache.counts[index];
    }
    if (total_entries != expected_entries) {
        throw std::runtime_error("universal packed entry census changed");
    }
    cache.entry_count = size_t(total_entries);
    cache.bytes = cache.entry_count * sizeof(uint64_t);
    cache.total_seconds = seconds_now() - total_start;
    std::printf(
        "CACHE_METADATA distributions=%zu entries=%zu bytes=%zu "
        "count_seconds=%.6f total_seconds=%.6f exact=OK\n",
        cache.keys.size(), cache.entry_count, cache.bytes, cache.count_seconds,
        cache.total_seconds);
    return cache;
}

static std::array<PackedSourceRef, 2> packed_source_refs(
    const PackedUniversalCache& cache, PrefixKey raw) {
    const PrefixKey full_mask = (PrefixKey(1) << (RIGHT_COLUMNS * ROWS)) - 1;
    CanonicalForm selected = canonical_prefix(raw, RIGHT_COLUMNS);
    CanonicalForm complement =
        canonical_prefix(raw ^ full_mask, RIGHT_COLUMNS);
    return {PackedSourceRef{
                packed_distribution_index(cache, selected.key), selected.row_map},
            PackedSourceRef{packed_distribution_index(cache, complement.key),
                            complement.row_map}};
}

static StagingCensus simulate_staging(
    const std::vector<size_t>& source_ids, const PackedUniversalCache& cache,
    uint64_t source_entry_cap, uint64_t description_cap) {
    if (!source_entry_cap || !description_cap) {
        throw std::invalid_argument("staging limits must be nonzero");
    }
    std::vector<size_t> sorted = source_ids;
    std::sort(sorted.begin(), sorted.end());
    StagingCensus result;
    uint64_t chunk_entries = 0;
    uint64_t chunk_descriptions = 0;
    auto flush = [&]() {
        if (!chunk_descriptions) return;
        result.chunks++;
        result.maximum_source_entries =
            std::max(result.maximum_source_entries, chunk_entries);
        result.maximum_descriptions =
            std::max(result.maximum_descriptions, chunk_descriptions);
        chunk_entries = 0;
        chunk_descriptions = 0;
    };
    for (size_t begin = 0; begin < sorted.size();) {
        size_t end = begin + 1;
        while (end < sorted.size() && sorted[end] == sorted[begin]) end++;
        uint64_t remaining = end - begin;
        uint64_t count = cache.counts[sorted[begin]];
        if (count > source_entry_cap) {
            throw std::runtime_error("one canonical distribution exceeds staging cap");
        }
        while (remaining) {
            if (chunk_descriptions == description_cap ||
                (chunk_descriptions && chunk_entries + count > source_entry_cap)) {
                flush();
            }
            chunk_entries += count;
            result.transferred_entries += count;
            uint64_t take =
                std::min<uint64_t>(remaining,
                                   description_cap - chunk_descriptions);
            chunk_descriptions += take;
            remaining -= take;
            if (remaining) flush();
        }
        begin = end;
    }
    flush();
    return result;
}

static SourceScheduleCensus evaluate_source_schedule(
    const char* name, const std::vector<uint16_t>& assignment,
    const std::vector<uint64_t>& expanded_counts,
    const std::vector<std::array<PackedSourceRef, 2>>& source_pairs,
    const PackedUniversalCache& cache, uint64_t right_entry_cap) {
    constexpr uint64_t staging_entries = (UINT64_C(1024) << 20) / sizeof(uint64_t);
    constexpr uint64_t description_cap = 8192;
    if (assignment.size() != source_pairs.size() ||
        assignment.size() != expanded_counts.size()) {
        throw std::invalid_argument("source schedule size mismatch");
    }
    SourceScheduleCensus result;
    for (uint16_t batch : assignment) {
        result.batches = std::max<uint64_t>(result.batches, uint64_t(batch) + 1);
    }
    if (!result.batches || result.batches > 64) {
        throw std::runtime_error("source schedule batch count is unsupported");
    }
    std::vector<uint64_t> batch_entries(result.batches);
    std::vector<uint64_t> batch_intervals(result.batches);
    std::vector<std::vector<size_t>> batch_sources(result.batches);
    std::vector<uint64_t> source_batch_masks(cache.keys.size());
    for (size_t index = 0; index < assignment.size(); index++) {
        uint16_t batch = assignment[index];
        batch_entries[batch] += expanded_counts[index];
        if (!index || assignment[index - 1] != batch) {
            batch_intervals[batch]++;
        }
        for (const PackedSourceRef& reference : source_pairs[index]) {
            batch_sources[batch].push_back(reference.distribution);
            source_batch_masks[reference.distribution] |= UINT64_C(1) << batch;
        }
    }
    for (size_t id = 0; id < source_batch_masks.size(); id++) {
        result.source_entries +=
            uint64_t(cache.counts[id]) * __builtin_popcountll(source_batch_masks[id]);
    }
    for (size_t batch = 0; batch < result.batches; batch++) {
        if (!batch_entries[batch] || batch_entries[batch] > right_entry_cap) {
            throw std::runtime_error("invalid source-aware batch capacity");
        }
        result.minimum_batch_entries =
            std::min(result.minimum_batch_entries, batch_entries[batch]);
        result.maximum_batch_entries =
            std::max(result.maximum_batch_entries, batch_entries[batch]);
        result.raw_intervals += batch_intervals[batch];
        result.maximum_raw_intervals =
            std::max(result.maximum_raw_intervals, batch_intervals[batch]);
        StagingCensus batch_staging =
            simulate_staging(batch_sources[batch], cache, staging_entries,
                             description_cap);
        result.staging.chunks += batch_staging.chunks;
        result.staging.transferred_entries += batch_staging.transferred_entries;
        result.staging.maximum_source_entries = std::max(
            result.staging.maximum_source_entries,
            batch_staging.maximum_source_entries);
        result.staging.maximum_descriptions = std::max(
            result.staging.maximum_descriptions,
            batch_staging.maximum_descriptions);
    }
    constexpr double gib = 1024.0 * 1024.0 * 1024.0;
    std::printf(
        "SOURCE_SCHEDULE name=%s cap=%llu batches=%llu source_entries=%llu "
        "source_gib=%.6f staging_entries=%llu staging_gib=%.6f chunks=%llu "
        "batch_entries_min=%llu batch_entries_max=%llu raw_intervals=%llu "
        "max_raw_intervals=%llu exact=OK\n",
        name, (unsigned long long)right_entry_cap,
        (unsigned long long)result.batches,
        (unsigned long long)result.source_entries,
        double(result.source_entries * sizeof(uint64_t)) / gib,
        (unsigned long long)result.staging.transferred_entries,
        double(result.staging.transferred_entries * sizeof(uint64_t)) / gib,
        (unsigned long long)result.staging.chunks,
        (unsigned long long)result.minimum_batch_entries,
        (unsigned long long)result.maximum_batch_entries,
        (unsigned long long)result.raw_intervals,
        (unsigned long long)result.maximum_raw_intervals);
    return result;
}

static std::vector<uint16_t> next_fit_schedule(
    const std::vector<size_t>& order, const std::vector<uint64_t>& expanded_counts,
    uint64_t right_entry_cap) {
    std::vector<uint16_t> assignment(order.size(), UINT16_MAX);
    uint16_t batch = 0;
    uint64_t batch_entries = 0;
    for (size_t index : order) {
        uint64_t count = expanded_counts[index];
        if (count > right_entry_cap) {
            throw std::runtime_error("one scheduled prefix exceeds capacity");
        }
        if (batch_entries && batch_entries + count > right_entry_cap) {
            if (batch == UINT16_MAX - 1) {
                throw std::overflow_error("too many source-aware batches");
            }
            batch++;
            batch_entries = 0;
        }
        assignment[index] = batch;
        batch_entries += count;
    }
    return assignment;
}

static std::vector<uint16_t> source_pair_schedule(
    const std::vector<std::array<PackedSourceRef, 2>>& source_pairs,
    const std::vector<uint64_t>& expanded_counts, uint64_t right_entry_cap,
    const std::vector<uint64_t>& source_priority) {
    std::vector<size_t> order(source_pairs.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        const auto normalized = [&](size_t index) {
            size_t first = source_pairs[index][0].distribution;
            size_t second = source_pairs[index][1].distribution;
            if (source_priority[first] < source_priority[second] ||
                (source_priority[first] == source_priority[second] &&
                 first > second)) {
                std::swap(first, second);
            }
            return std::pair<size_t, size_t>{first, second};
        };
        auto left = normalized(lhs);
        auto right = normalized(rhs);
        if (left != right) return left < right;
        return lhs < rhs;
    });
    return next_fit_schedule(order, expanded_counts, right_entry_cap);
}

static std::vector<uint16_t> source_window_schedule(
    size_t window_size,
    const std::vector<std::array<PackedSourceRef, 2>>& source_pairs,
    const std::vector<uint64_t>& expanded_counts,
    const PackedUniversalCache& cache, uint64_t right_entry_cap) {
    if (!window_size) throw std::invalid_argument("zero source window size");
    uint64_t total_entries =
        std::accumulate(expanded_counts.begin(), expanded_counts.end(), UINT64_C(0));
    size_t wanted_batches = size_t((total_entries + right_entry_cap - 1) /
                                   right_entry_cap);
    if (!wanted_batches || wanted_batches >= 64) {
        throw std::runtime_error("unsupported source-window batch count");
    }
    std::vector<uint16_t> assignment(source_pairs.size(), UINT16_MAX);
    std::vector<uint64_t> batch_entries(wanted_batches);
    std::vector<uint64_t> source_batch_masks(cache.keys.size());
    for (size_t begin = 0; begin < source_pairs.size(); begin += window_size) {
        size_t end = std::min(source_pairs.size(), begin + window_size);
        uint64_t window_entries = 0;
        std::vector<size_t> sources;
        sources.reserve((end - begin) * 2);
        for (size_t index = begin; index < end; index++) {
            window_entries += expanded_counts[index];
            sources.push_back(source_pairs[index][0].distribution);
            sources.push_back(source_pairs[index][1].distribution);
        }
        std::sort(sources.begin(), sources.end());
        sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
        if (window_entries > right_entry_cap) {
            throw std::runtime_error("source window exceeds batch capacity");
        }
        size_t best = SIZE_MAX;
        uint64_t best_reuse = 0;
        for (size_t batch = 0; batch < batch_entries.size(); batch++) {
            if (batch_entries[batch] + window_entries > right_entry_cap) continue;
            uint64_t reuse = 0;
            uint64_t bit = UINT64_C(1) << batch;
            for (size_t source : sources) {
                if (source_batch_masks[source] & bit) reuse += cache.counts[source];
            }
            if (best == SIZE_MAX || reuse > best_reuse ||
                (reuse == best_reuse &&
                 batch_entries[batch] < batch_entries[best])) {
                best = batch;
                best_reuse = reuse;
            }
        }
        if (best == SIZE_MAX) {
            if (batch_entries.size() >= 64) {
                throw std::runtime_error("too many source-window batches");
            }
            best = batch_entries.size();
            batch_entries.push_back(0);
        }
        for (size_t index = begin; index < end; index++) {
            assignment[index] = uint16_t(best);
        }
        batch_entries[best] += window_entries;
        uint64_t bit = UINT64_C(1) << best;
        for (size_t source : sources) source_batch_masks[source] |= bit;
    }
    return assignment;
}

static void census_capacity(const std::vector<PrefixKey>& right_keys,
                            const PackedUniversalCache& cache,
                            uint64_t right_entry_cap) {
    constexpr uint64_t description_cap = 8192;
    constexpr std::array<uint64_t, 3> staging_bytes = {
        UINT64_C(256) << 20, UINT64_C(512) << 20, UINT64_C(1024) << 20};
    std::vector<uint64_t> expanded_counts(right_keys.size());
    std::vector<std::array<PackedSourceRef, 2>> source_pairs(right_keys.size());
    for (size_t index = 0; index < right_keys.size(); index++) {
        source_pairs[index] = packed_source_refs(cache, right_keys[index]);
        expanded_counts[index] =
            uint64_t(cache.counts[source_pairs[index][0].distribution]) +
            cache.counts[source_pairs[index][1].distribution];
        if (expanded_counts[index] > right_entry_cap) {
            throw std::runtime_error("one right prefix exceeds census cap");
        }
    }

    uint64_t batches = 0;
    uint64_t total_expanded_entries = 0;
    uint64_t total_unique_source_entries = 0;
    uint64_t maximum_unique_source_entries = 0;
    uint64_t total_intervals = 0;
    uint64_t maximum_intervals = 0;
    uint64_t total_unique_transform_entries = 0;
    uint64_t maximum_unique_transform_entries = 0;
    std::array<StagingCensus, staging_bytes.size()> staging{};
    std::vector<uint8_t> shard_used(cache.keys.size());
    std::unordered_set<uint64_t> shard_transforms;
    shard_transforms.reserve(right_keys.size());
    size_t begin = 0;
    while (begin < right_keys.size()) {
        size_t end = begin;
        uint64_t expanded_entries = 0;
        while (end < right_keys.size() &&
               expanded_entries + expanded_counts[end] <= right_entry_cap) {
            expanded_entries += expanded_counts[end++];
        }
        if (end == begin) throw std::runtime_error("empty census batch");
        std::vector<size_t> ids;
        ids.reserve((end - begin) * 2);
        std::vector<uint64_t> transforms;
        transforms.reserve((end - begin) * 2);
        for (size_t index = begin; index < end; index++) {
            for (const PackedSourceRef& reference : source_pairs[index]) {
                ids.push_back(reference.distribution);
                uint64_t transform = packed_transform_key(reference);
                transforms.push_back(transform);
                shard_transforms.insert(transform);
            }
        }
        std::vector<size_t> unique = ids;
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        uint64_t unique_entries = 0;
        uint64_t intervals = 0;
        size_t previous = SIZE_MAX;
        for (size_t id : unique) {
            unique_entries += cache.counts[id];
            shard_used[id] = 1;
            if (previous == SIZE_MAX || id != previous + 1) intervals++;
            previous = id;
        }
        std::sort(transforms.begin(), transforms.end());
        transforms.erase(std::unique(transforms.begin(), transforms.end()),
                         transforms.end());
        uint64_t transform_entries = 0;
        for (uint64_t transform : transforms) {
            transform_entries += cache.counts[size_t(transform >> 32)];
        }
        total_expanded_entries += expanded_entries;
        total_unique_source_entries += unique_entries;
        maximum_unique_source_entries =
            std::max(maximum_unique_source_entries, unique_entries);
        total_intervals += intervals;
        maximum_intervals = std::max(maximum_intervals, intervals);
        total_unique_transform_entries += transform_entries;
        maximum_unique_transform_entries =
            std::max(maximum_unique_transform_entries, transform_entries);
        for (size_t index = 0; index < staging_bytes.size(); index++) {
            StagingCensus batch = simulate_staging(
                ids, cache, staging_bytes[index] / sizeof(uint64_t),
                description_cap);
            staging[index].chunks += batch.chunks;
            staging[index].transferred_entries += batch.transferred_entries;
            staging[index].maximum_source_entries =
                std::max(staging[index].maximum_source_entries,
                         batch.maximum_source_entries);
            staging[index].maximum_descriptions =
                std::max(staging[index].maximum_descriptions,
                         batch.maximum_descriptions);
        }
        batches++;
        begin = end;
    }

    uint64_t shard_unique_distributions = 0;
    uint64_t shard_unique_entries = 0;
    for (size_t id = 0; id < shard_used.size(); id++) {
        if (!shard_used[id]) continue;
        shard_unique_distributions++;
        shard_unique_entries += cache.counts[id];
    }
    uint64_t shard_transform_entries = 0;
    for (uint64_t transform : shard_transforms) {
        shard_transform_entries += cache.counts[size_t(transform >> 32)];
    }
    constexpr double gib = 1024.0 * 1024.0 * 1024.0;
    std::printf(
        "TRAFFIC cap=%llu batches=%llu right_prefixes=%zu "
        "expanded_entries=%llu expanded_gib=%.6f "
        "batch_unique_entries=%llu batch_unique_gib=%.6f "
        "max_batch_unique_gib=%.6f intervals=%llu max_intervals=%llu "
        "batch_transform_entries=%llu batch_transform_gib=%.6f "
        "max_batch_transform_gib=%.6f transform_reuse=%.6fx "
        "shard_unique_distributions=%llu shard_unique_entries=%llu "
        "shard_unique_gib=%.6f source_reuse=%.6fx "
        "shard_transforms=%zu shard_transform_entries=%llu "
        "shard_transform_gib=%.6f exact=OK\n",
        (unsigned long long)right_entry_cap, (unsigned long long)batches,
        right_keys.size(), (unsigned long long)total_expanded_entries,
        double(total_expanded_entries * sizeof(uint64_t)) / gib,
        (unsigned long long)total_unique_source_entries,
        double(total_unique_source_entries * sizeof(uint64_t)) / gib,
        double(maximum_unique_source_entries * sizeof(uint64_t)) / gib,
        (unsigned long long)total_intervals,
        (unsigned long long)maximum_intervals,
        (unsigned long long)total_unique_transform_entries,
        double(total_unique_transform_entries * sizeof(PrefixEntry)) / gib,
        double(maximum_unique_transform_entries * sizeof(PrefixEntry)) / gib,
        double(total_expanded_entries) /
            double(total_unique_transform_entries),
        (unsigned long long)shard_unique_distributions,
        (unsigned long long)shard_unique_entries,
        double(shard_unique_entries * sizeof(uint64_t)) / gib,
        double(total_expanded_entries) / double(total_unique_source_entries),
        shard_transforms.size(),
        (unsigned long long)shard_transform_entries,
        double(shard_transform_entries * sizeof(PrefixEntry)) / gib);
    for (size_t index = 0; index < staging_bytes.size(); index++) {
        std::printf(
            "STAGING cap=%llu staging_mib=%llu description_cap=%llu "
            "chunks=%llu transfer_entries=%llu transfer_gib=%.6f "
            "max_source_mib=%.6f max_descriptions=%llu overhead=%.6fx\n",
            (unsigned long long)right_entry_cap,
            (unsigned long long)(staging_bytes[index] >> 20),
            (unsigned long long)description_cap,
            (unsigned long long)staging[index].chunks,
            (unsigned long long)staging[index].transferred_entries,
            double(staging[index].transferred_entries * sizeof(uint64_t)) / gib,
            double(staging[index].maximum_source_entries * sizeof(uint64_t)) /
                (1024.0 * 1024.0),
            (unsigned long long)staging[index].maximum_descriptions,
            double(staging[index].transferred_entries) /
                double(total_unique_source_entries));
    }

    uint64_t dedup_batches = 0;
    uint64_t dedup_layout_entries = 0;
    uint64_t dedup_source_entries = 0;
    uint64_t dedup_max_layout_entries = 0;
    uint64_t dedup_max_source_entries = 0;
    uint64_t dedup_max_prefixes = 0;
    std::array<StagingCensus, staging_bytes.size()> dedup_staging{};
    begin = 0;
    while (begin < right_keys.size()) {
        size_t end = begin;
        uint64_t layout_entries = 0;
        std::unordered_set<uint64_t> transforms;
        transforms.reserve(65536);
        while (end < right_keys.size()) {
            uint64_t added = 0;
            uint64_t keys_to_add[2] = {
                packed_transform_key(source_pairs[end][0]),
                packed_transform_key(source_pairs[end][1])};
            if (!transforms.count(keys_to_add[0])) {
                added += cache.counts[source_pairs[end][0].distribution];
            }
            if (keys_to_add[1] != keys_to_add[0] &&
                !transforms.count(keys_to_add[1])) {
                added += cache.counts[source_pairs[end][1].distribution];
            }
            if (end != begin && layout_entries + added > right_entry_cap) break;
            if (added > right_entry_cap) {
                throw std::runtime_error(
                    "one deduplicated right prefix exceeds census cap");
            }
            transforms.insert(keys_to_add[0]);
            transforms.insert(keys_to_add[1]);
            layout_entries += added;
            end++;
        }
        if (end == begin) throw std::runtime_error("empty deduplicated batch");
        std::vector<size_t> ids;
        ids.reserve(transforms.size());
        for (uint64_t transform : transforms) {
            ids.push_back(size_t(transform >> 32));
        }
        std::vector<size_t> unique_ids = ids;
        std::sort(unique_ids.begin(), unique_ids.end());
        unique_ids.erase(std::unique(unique_ids.begin(), unique_ids.end()),
                         unique_ids.end());
        uint64_t source_entries = 0;
        for (size_t id : unique_ids) source_entries += cache.counts[id];
        dedup_layout_entries += layout_entries;
        dedup_source_entries += source_entries;
        dedup_max_layout_entries =
            std::max(dedup_max_layout_entries, layout_entries);
        dedup_max_source_entries =
            std::max(dedup_max_source_entries, source_entries);
        dedup_max_prefixes = std::max<uint64_t>(dedup_max_prefixes, end - begin);
        for (size_t index = 0; index < staging_bytes.size(); index++) {
            StagingCensus batch = simulate_staging(
                ids, cache, staging_bytes[index] / sizeof(uint64_t),
                description_cap);
            dedup_staging[index].chunks += batch.chunks;
            dedup_staging[index].transferred_entries +=
                batch.transferred_entries;
            dedup_staging[index].maximum_source_entries = std::max(
                dedup_staging[index].maximum_source_entries,
                batch.maximum_source_entries);
            dedup_staging[index].maximum_descriptions = std::max(
                dedup_staging[index].maximum_descriptions,
                batch.maximum_descriptions);
        }
        dedup_batches++;
        begin = end;
    }
    std::printf(
        "DEDUP_TRAFFIC cap=%llu batches=%llu right_prefixes=%zu "
        "raw_entries=%llu layout_entries=%llu layout_gib=%.6f "
        "layout_reuse=%.6fx max_layout_entries=%llu "
        "source_entries=%llu source_gib=%.6f max_source_gib=%.6f "
        "max_prefixes=%llu exact=OK\n",
        (unsigned long long)right_entry_cap,
        (unsigned long long)dedup_batches, right_keys.size(),
        (unsigned long long)total_expanded_entries,
        (unsigned long long)dedup_layout_entries,
        double(dedup_layout_entries * sizeof(PrefixEntry)) / gib,
        double(total_expanded_entries) / double(dedup_layout_entries),
        (unsigned long long)dedup_max_layout_entries,
        (unsigned long long)dedup_source_entries,
        double(dedup_source_entries * sizeof(uint64_t)) / gib,
        double(dedup_max_source_entries * sizeof(uint64_t)) / gib,
        (unsigned long long)dedup_max_prefixes);
    for (size_t index = 0; index < staging_bytes.size(); index++) {
        std::printf(
            "DEDUP_STAGING cap=%llu staging_mib=%llu chunks=%llu "
            "transfer_entries=%llu transfer_gib=%.6f overhead=%.6fx\n",
            (unsigned long long)right_entry_cap,
            (unsigned long long)(staging_bytes[index] >> 20),
            (unsigned long long)dedup_staging[index].chunks,
            (unsigned long long)dedup_staging[index].transferred_entries,
            double(dedup_staging[index].transferred_entries * sizeof(uint64_t)) /
                gib,
            double(dedup_staging[index].transferred_entries) /
                double(dedup_source_entries));
    }

    std::vector<size_t> natural_order(right_keys.size());
    std::iota(natural_order.begin(), natural_order.end(), 0);
    evaluate_source_schedule(
        "contiguous",
        next_fit_schedule(natural_order, expanded_counts, right_entry_cap),
        expanded_counts, source_pairs, cache, right_entry_cap);

    std::vector<uint64_t> neutral_priority(cache.keys.size());
    evaluate_source_schedule(
        "source-pair",
        source_pair_schedule(source_pairs, expanded_counts, right_entry_cap,
                             neutral_priority),
        expanded_counts, source_pairs, cache, right_entry_cap);

    std::vector<uint64_t> degree(cache.keys.size());
    for (const auto& pair : source_pairs) {
        degree[pair[0].distribution]++;
        degree[pair[1].distribution]++;
    }
    std::vector<uint64_t> anchor_priority(cache.keys.size());
    for (size_t id = 0; id < anchor_priority.size(); id++) {
        anchor_priority[id] = uint64_t(cache.counts[id]) * degree[id];
    }
    evaluate_source_schedule(
        "source-anchor",
        source_pair_schedule(source_pairs, expanded_counts, right_entry_cap,
                             anchor_priority),
        expanded_counts, source_pairs, cache, right_entry_cap);

    constexpr std::array<size_t, 4> windows = {64, 256, 1024, 4096};
    for (size_t window : windows) {
        std::vector<uint16_t> assignment = source_window_schedule(
            window, source_pairs, expanded_counts, cache, right_entry_cap);
        std::string name = "window-" + std::to_string(window);
        evaluate_source_schedule(name.c_str(), assignment, expanded_counts,
                                 source_pairs, cache, right_entry_cap);
    }
}

static void census_canonical_right(
    const std::vector<Edge>& edges, const std::vector<PrefixKey>& right_keys,
    const PackedUniversalCache& cache, uint64_t right_entry_cap) {
    std::vector<PrefixKey> left_keys = unique_lefts(edges);
    CanonicalFactory left_factory =
        build_canonical_factory(left_keys, LEFT_COLUMNS);

    std::vector<uint64_t> expanded_counts(right_keys.size());
    std::vector<std::array<PackedSourceRef, 2>> source_pairs(right_keys.size());
    for (size_t index = 0; index < right_keys.size(); index++) {
        source_pairs[index] = packed_source_refs(cache, right_keys[index]);
        expanded_counts[index] =
            uint64_t(cache.counts[source_pairs[index][0].distribution]) +
            cache.counts[source_pairs[index][1].distribution];
    }
    std::vector<uint64_t> neutral_priority(cache.keys.size());
    std::vector<uint16_t> assignment = source_pair_schedule(
        source_pairs, expanded_counts, right_entry_cap, neutral_priority);
    uint64_t batch_count = 0;
    for (uint16_t batch : assignment) {
        batch_count = std::max<uint64_t>(batch_count, uint64_t(batch) + 1);
    }
    std::vector<std::unordered_set<uint64_t>> left_transforms(batch_count);
    std::vector<std::unordered_set<size_t>> right_sources(batch_count);
    std::vector<uint64_t> raw_left_entries(batch_count);
    std::vector<uint64_t> current_right_entries(batch_count);
    std::vector<uint64_t> edge_counts(batch_count);
    for (size_t index = 0; index < right_keys.size(); index++) {
        uint16_t batch = assignment[index];
        current_right_entries[batch] += expanded_counts[index];
        right_sources[batch].insert(source_pairs[index][0].distribution);
        right_sources[batch].insert(source_pairs[index][1].distribution);
    }

    const PrefixKey right_full_mask =
        (PrefixKey(1) << (RIGHT_COLUMNS * ROWS)) - 1;
    for (const Edge& edge : edges) {
        auto right_found =
            std::lower_bound(right_keys.begin(), right_keys.end(), edge.right);
        if (right_found == right_keys.end() || *right_found != edge.right) {
            throw std::runtime_error("canonical-right edge lookup failed");
        }
        uint16_t batch = assignment[size_t(right_found - right_keys.begin())];
        edge_counts[batch]++;
        const RawCanonicalPair& left = lookup_raw(left_factory, edge.left);
        const CanonicalRef left_refs[2] = {left.selected, left.complement};
        const CanonicalForm right_forms[2] = {
            canonical_prefix(edge.right, RIGHT_COLUMNS),
            canonical_prefix(edge.right ^ right_full_mask, RIGHT_COLUMNS)};
        for (int complement = 0; complement < 2; complement++) {
            const CanonicalRef& source = left_refs[complement];
            uint32_t query_map = compose_row_maps(
                source.row_map,
                inverse_row_map(right_forms[complement].row_map));
            uint64_t key = (uint64_t(source.distribution) << 32) | query_map;
            left_transforms[batch].insert(key);
            raw_left_entries[batch] +=
                left_factory.descriptors[source.distribution].count;
        }
    }

    uint64_t total_raw_left_entries = 0;
    uint64_t total_left_transform_entries = 0;
    uint64_t total_right_source_entries = 0;
    uint64_t total_current_right_entries = 0;
    uint64_t maximum_proposed_entries = 0;
    uint64_t maximum_left_entries = 0;
    uint64_t maximum_right_entries = 0;
    uint64_t total_transforms = 0;
    for (size_t batch = 0; batch < batch_count; batch++) {
        uint64_t left_entries = 0;
        for (uint64_t key : left_transforms[batch]) {
            left_entries +=
                left_factory.descriptors[size_t(key >> 32)].count;
        }
        uint64_t right_entries = 0;
        for (size_t source : right_sources[batch]) {
            right_entries += cache.counts[source];
        }
        total_raw_left_entries += raw_left_entries[batch];
        total_left_transform_entries += left_entries;
        total_right_source_entries += right_entries;
        total_current_right_entries += current_right_entries[batch];
        total_transforms += left_transforms[batch].size();
        maximum_left_entries = std::max(maximum_left_entries, left_entries);
        maximum_right_entries = std::max(maximum_right_entries, right_entries);
        maximum_proposed_entries =
            std::max(maximum_proposed_entries, left_entries + right_entries);
        std::printf(
            "CANONICAL_RIGHT_BATCH batch=%zu edges=%llu left_transforms=%zu "
            "raw_left_entries=%llu left_entries=%llu right_sources=%zu "
            "right_entries=%llu current_right_entries=%llu exact=OK\n",
            batch, (unsigned long long)edge_counts[batch],
            left_transforms[batch].size(),
            (unsigned long long)raw_left_entries[batch],
            (unsigned long long)left_entries, right_sources[batch].size(),
            (unsigned long long)right_entries,
            (unsigned long long)current_right_entries[batch]);
    }
    constexpr double gib = 1024.0 * 1024.0 * 1024.0;
    std::printf(
        "CANONICAL_RIGHT_TOTAL cap=%llu batches=%llu edges=%zu "
        "left_prefixes=%zu left_canonical_distributions=%zu "
        "left_canonical_entries=%zu raw_left_entries=%llu "
        "left_transforms=%llu left_transform_entries=%llu "
        "left_transform_gib=%.6f left_transform_reuse=%.6fx "
        "right_source_entries=%llu right_source_gib=%.6f "
        "proposed_entries=%llu proposed_gib=%.6f "
        "current_right_entries=%llu current_right_gib=%.6f "
        "layout_reduction=%.6fx max_left_gib=%.6f max_right_gib=%.6f "
        "max_proposed_gib=%.6f exact=OK\n",
        (unsigned long long)right_entry_cap,
        (unsigned long long)batch_count, edges.size(), left_keys.size(),
        left_factory.descriptors.size(), left_factory.entries.size(),
        (unsigned long long)total_raw_left_entries,
        (unsigned long long)total_transforms,
        (unsigned long long)total_left_transform_entries,
        double(total_left_transform_entries * sizeof(PrefixEntry)) / gib,
        double(total_raw_left_entries) / double(total_left_transform_entries),
        (unsigned long long)total_right_source_entries,
        double(total_right_source_entries * sizeof(PrefixEntry)) / gib,
        (unsigned long long)(total_left_transform_entries +
                             total_right_source_entries),
        double((total_left_transform_entries + total_right_source_entries) *
               sizeof(PrefixEntry)) /
            gib,
        (unsigned long long)total_current_right_entries,
        double(total_current_right_entries * sizeof(PrefixEntry)) / gib,
        double(total_current_right_entries) /
            double(total_left_transform_entries + total_right_source_entries),
        double(maximum_left_entries * sizeof(PrefixEntry)) / gib,
        double(maximum_right_entries * sizeof(PrefixEntry)) / gib,
        double(maximum_proposed_entries * sizeof(PrefixEntry)) / gib);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "Usage: %s CANONICAL_7X5.orbits RECT7X9.orbits "
                     "[RIGHT_ENTRY_CAP ...]\n",
                     argv[0]);
        return 2;
    }
    initialise_tables();
    validate_mask_split();
    validate_row_map_algebra();
    PackedUniversalCache cache = build_packed_universal_metadata(argv[1]);
    U128 labelled_weight = 0;
    uint64_t records = 0;
    std::vector<Edge> edges =
        read_edges(argv[2], 0, 0, 0, 0, labelled_weight, records);
    std::vector<PrefixKey> right_keys = unique_rights(edges);
    std::printf("WORK records=%llu kernels=%zu right_prefixes=%zu "
                "labelled_weight=%s exact=OK\n",
                (unsigned long long)records, edges.size(), right_keys.size(),
                u128_string(labelled_weight).c_str());
    if (argc == 3) {
        constexpr std::array<uint64_t, 5> caps = {
            UINT64_C(457654272), UINT64_C(1500000000),
            UINT64_C(2000000000), UINT64_C(2500000000),
            UINT64_C(3000000000)};
        for (uint64_t cap : caps) census_capacity(right_keys, cache, cap);
    } else {
        for (int index = 3; index < argc; index++) {
            uint64_t cap = std::strtoull(argv[index], nullptr, 10);
            if (!cap || cap > UINT32_MAX) return 2;
            census_capacity(right_keys, cache, cap);
            census_canonical_right(edges, right_keys, cache, cap);
        }
    }
    return 0;
}
