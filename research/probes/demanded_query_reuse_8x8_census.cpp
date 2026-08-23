#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <unordered_map>

#define PREFIX_BUCKET_TT_RANK_CENSUS_NO_MAIN
#include "prefix_bucket_tt_rank_census.cpp"

namespace {

struct CanonicalForm {
    uint32_t key = 0;
    uint32_t row_map = 0;
};

struct JoinRef {
    uint32_t left_source = 0;
    uint32_t right_source = 0;
    uint32_t relative_row_map = 0;
};

struct QueryKey {
    uint64_t mask = 0;
    uint32_t source = 0;

    bool operator<(const QueryKey& other) const {
        return source != other.source ? source < other.source
                                      : mask < other.mask;
    }
    bool operator==(const QueryKey& other) const {
        return source == other.source && mask == other.mask;
    }
};

struct QueryGroup {
    QueryKey key;
    uint64_t count = 0;
};

struct Source {
    uint32_t key = 0;
    uint32_t full_support = 0;
    bool needed_left = false;
    bool needed_right = false;
    std::vector<uint64_t> quotient_masks;
    std::vector<uint32_t> automorphisms;
};

struct SourceRequest {
    uint32_t key = 0;
    bool needed_left = false;
    bool needed_right = false;
};

struct CacheStats {
    uint64_t occurrences = 0;
    uint64_t unique = 0;
    uint64_t singleton_keys = 0;
    uint64_t repeated_keys = 0;
    U128 scan_baseline = 0;
    U128 scan_once = 0;
    std::array<uint64_t, 7> keys_at_least{};
    std::array<uint64_t, 7> occurrences_at_least{};
};

constexpr std::array<uint64_t, 7> REUSE_THRESHOLDS = {2, 4, 8, 16, 64, 256,
                                                       1024};

static uint64_t swap_token_planes_local(uint64_t mask) {
    constexpr uint64_t plane_mask = (UINT64_C(1) << PAIRS) - 1;
    return ((mask & plane_mask) << PAIRS) |
           ((mask >> PAIRS) & plane_mask);
}

static uint32_t identity_row_map() {
    uint32_t result = 0;
    for (unsigned row = 0; row < ROWS; row++) result |= row << (4 * row);
    return result;
}

static uint32_t inverse_row_map_local(uint32_t row_map) {
    uint32_t inverse = 0;
    for (unsigned source = 0; source < ROWS; source++) {
        unsigned destination = (row_map >> (4 * source)) & 15U;
        inverse |= source << (4 * destination);
    }
    return inverse;
}

static uint32_t compose_row_maps_local(uint32_t first, uint32_t second) {
    uint32_t result = 0;
    for (unsigned source = 0; source < ROWS; source++) {
        unsigned middle = (first >> (4 * source)) & 15U;
        unsigned destination = (second >> (4 * middle)) & 15U;
        result |= destination << (4 * source);
    }
    return result;
}

static uint64_t transform_pair_mask_local(uint64_t mask, uint32_t row_map) {
    uint64_t result = 0;
    while (mask) {
        unsigned bit = unsigned(__builtin_ctzll(mask));
        mask &= mask - 1;
        unsigned colour = bit / PAIRS;
        unsigned pair = bit % PAIRS;
        unsigned first = 0;
        while (pair >= unsigned(ROWS - first - 1))
            pair -= unsigned(ROWS - first++ - 1);
        unsigned second = first + 1 + pair;
        unsigned image_first = (row_map >> (4 * first)) & 15U;
        unsigned image_second = (row_map >> (4 * second)) & 15U;
        if (image_first > image_second) std::swap(image_first, image_second);
        unsigned image_pair =
            image_first * (2 * ROWS - image_first - 1) / 2 +
            image_second - image_first - 1;
        result |= UINT64_C(1) << (colour * PAIRS + image_pair);
    }
    return result;
}

static std::array<uint8_t, ROWS> unpack_rows(uint32_t key) {
    std::array<uint8_t, ROWS> rows{};
    for (int row = ROWS - 1; row >= 0; row--) {
        rows[size_t(row)] = uint8_t(key & 15U);
        key >>= HALF_COLUMNS;
    }
    return rows;
}

static uint32_t pack_rows(const std::array<uint8_t, ROWS>& rows) {
    uint32_t result = 0;
    for (uint8_t row : rows) result = (result << HALF_COLUMNS) | row;
    return result;
}

static uint8_t permute_pattern(uint8_t pattern,
                               const std::array<int, HALF_COLUMNS>& p) {
    uint8_t result = 0;
    for (unsigned column = 0; column < HALF_COLUMNS; column++)
        if (pattern & (1U << column)) result |= uint8_t(1U << p[column]);
    return result;
}

static CanonicalForm canonical_prefix_local(uint32_t key) {
    const auto patterns = unpack_rows(key);
    std::array<int, HALF_COLUMNS> permutation{};
    std::iota(permutation.begin(), permutation.end(), 0);
    CanonicalForm best{UINT32_MAX, 0};
    do {
        std::array<std::pair<uint8_t, uint8_t>, ROWS> rows{};
        for (unsigned row = 0; row < ROWS; row++)
            rows[row] = {permute_pattern(patterns[row], permutation),
                         uint8_t(row)};
        std::sort(rows.begin(), rows.end());
        std::array<uint8_t, ROWS> sorted{};
        uint32_t row_map = 0;
        for (unsigned row = 0; row < ROWS; row++) {
            sorted[row] = rows[row].first;
            // Maps the canonical row coordinate back to the labelled input.
            row_map |= uint32_t(rows[row].second) << (4 * row);
        }
        uint32_t candidate = pack_rows(sorted);
        if (candidate < best.key) best = CanonicalForm{candidate, row_map};
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    return best;
}

static void generate_row_bijections(
    const std::array<uint8_t, ROWS>& source_patterns,
    const std::array<uint8_t, ROWS>& target_patterns, unsigned source,
    uint16_t used, uint32_t row_map, std::vector<uint32_t>& output) {
    if (source == ROWS) {
        output.push_back(row_map);
        return;
    }
    for (unsigned destination = 0; destination < ROWS; destination++) {
        if ((used & (1U << destination)) ||
            source_patterns[source] != target_patterns[destination])
            continue;
        generate_row_bijections(source_patterns, target_patterns, source + 1,
                                uint16_t(used | (1U << destination)),
                                row_map | (destination << (4 * source)),
                                output);
    }
}

// Every returned row map preserves the distribution: applying the row map to
// the canonical half mask is the same as applying some column permutation.
static std::vector<uint32_t> half_mask_automorphisms(uint32_t key) {
    const auto target = unpack_rows(key);
    std::array<int, HALF_COLUMNS> permutation{};
    std::iota(permutation.begin(), permutation.end(), 0);
    std::vector<uint32_t> result;
    do {
        std::array<uint8_t, ROWS> transformed{};
        for (unsigned row = 0; row < ROWS; row++)
            transformed[row] = permute_pattern(target[row], permutation);
        generate_row_bijections(transformed, target, 0, 0, 0, result);
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    if (result.empty() || !std::binary_search(result.begin(), result.end(),
                                               identity_row_map()))
        throw std::runtime_error("half-mask automorphism census lost identity");
    return result;
}

static std::vector<std::string> split_paths(const std::string& value) {
    std::vector<std::string> result;
    size_t begin = 0;
    for (;;) {
        size_t end = value.find(',', begin);
        result.push_back(value.substr(begin, end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return result;
}

static std::vector<QueryGroup> group_queries(std::vector<QueryKey> queries) {
    std::sort(queries.begin(), queries.end());
    std::vector<QueryGroup> result;
    result.reserve(queries.size());
    for (size_t begin = 0; begin < queries.size();) {
        size_t end = begin + 1;
        while (end < queries.size() && queries[end] == queries[begin]) end++;
        result.push_back(QueryGroup{queries[begin], uint64_t(end - begin)});
        begin = end;
    }
    return result;
}

static std::vector<QueryGroup> group_counted_queries(
    std::vector<QueryGroup> queries) {
    std::sort(queries.begin(), queries.end(),
              [](const QueryGroup& a, const QueryGroup& b) {
                  return a.key < b.key;
              });
    std::vector<QueryGroup> result;
    result.reserve(queries.size());
    for (size_t begin = 0; begin < queries.size();) {
        size_t end = begin + 1;
        uint64_t count = queries[begin].count;
        while (end < queries.size() && queries[end].key == queries[begin].key) {
            count += queries[end].count;
            end++;
        }
        result.push_back(QueryGroup{queries[begin].key, count});
        begin = end;
    }
    return result;
}

static CacheStats cache_stats(const std::vector<QueryGroup>& groups,
                              const std::vector<Source>& sources,
                              const char* label) {
    CacheStats stats;
    std::vector<std::pair<U128, uint32_t>> savings;
    savings.reserve(groups.size());
    for (const QueryGroup& group : groups) {
        uint32_t support = sources[group.key.source].full_support;
        stats.occurrences += group.count;
        stats.unique++;
        stats.scan_baseline += U128(group.count) * support;
        stats.scan_once += support;
        stats.singleton_keys += group.count == 1;
        stats.repeated_keys += group.count > 1;
        for (size_t i = 0; i < REUSE_THRESHOLDS.size(); i++) {
            if (group.count >= REUSE_THRESHOLDS[i]) {
                stats.keys_at_least[i]++;
                stats.occurrences_at_least[i] += group.count;
            }
        }
        if (group.count > 1)
            savings.emplace_back(U128(group.count - 1) * support,
                                 group.key.source);
    }
    std::sort(savings.begin(), savings.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    const U128 avoidable = stats.scan_baseline - stats.scan_once;
    std::cout << std::setprecision(12) << "QUERY_REUSE label=" << label
              << " occurrences=" << stats.occurrences
              << " unique=" << stats.unique
              << " occurrence_reuse="
              << (stats.unique ? double(stats.occurrences) / stats.unique : 0)
              << " cache_hit_ratio="
              << (stats.occurrences
                      ? double(stats.occurrences - stats.unique) /
                            stats.occurrences
                      : 0)
              << " repeated_keys=" << stats.repeated_keys
              << " singleton_keys=" << stats.singleton_keys
              << " scan_baseline=" << u128_string(stats.scan_baseline)
              << " scan_once=" << u128_string(stats.scan_once)
              << " scan_reuse="
              << (stats.scan_once
                      ? double((long double)stats.scan_baseline /
                               (long double)stats.scan_once)
                      : 0)
              << " scan_saving_ratio="
              << (stats.scan_baseline
                      ? double((long double)avoidable /
                               (long double)stats.scan_baseline)
                      : 0)
              << " result_cache_gib_24b="
              << double((long double)stats.unique * 24.0L /
                        (1024.0L * 1024.0L * 1024.0L))
              << '\n';
    for (size_t i = 0; i < REUSE_THRESHOLDS.size(); i++) {
        std::cout << "QUERY_REUSE_THRESHOLD label=" << label
                  << " count=" << REUSE_THRESHOLDS[i]
                  << " keys=" << stats.keys_at_least[i]
                  << " occurrence_ratio="
                  << (stats.occurrences
                          ? double(stats.occurrences_at_least[i]) /
                                stats.occurrences
                          : 0)
                  << '\n';
    }
    constexpr std::array<uint64_t, 5> capacities = {
        1000, 10000, 100000, 1000000, 10000000};
    U128 saved = 0;
    size_t next = 0;
    for (uint64_t capacity : capacities) {
        while (next < savings.size() && next < capacity)
            saved += savings[next++].first;
        std::cout << "QUERY_REUSE_CAPACITY label=" << label
                  << " keys=" << std::min<uint64_t>(capacity, savings.size())
                  << " cache_mib_24b="
                  << double((long double)std::min<uint64_t>(capacity,
                                                            savings.size()) *
                            24.0L / (1024.0L * 1024.0L))
                  << " baseline_scan_saved_ratio="
                  << (stats.scan_baseline
                          ? double((long double)saved /
                                   (long double)stats.scan_baseline)
                          : 0)
                  << " avoidable_scan_coverage="
                  << (avoidable ? double((long double)saved /
                                              (long double)avoidable)
                                : 0)
                  << '\n';
    }
    return stats;
}

static uint64_t canonical_query_mask(uint64_t mask, const Source& source) {
    uint64_t best = std::min(mask, swap_token_planes_local(mask));
    for (uint32_t row_map : source.automorphisms) {
        uint64_t transformed = transform_pair_mask_local(mask, row_map);
        best = std::min(best, transformed);
        best = std::min(best, swap_token_planes_local(transformed));
    }
    return best;
}

static U128 direct_query(const std::vector<FullWeightedEntry>& distribution,
                         uint64_t query) {
    U128 result = 0;
    for (const FullWeightedEntry& entry : distribution)
        if (!(entry.mask & query)) result += entry.weight;
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) {
            std::cerr << "usage: " << argv[0]
                      << " SHARD[,SHARD...] [SAMPLES_PER_SHARD=64]\n";
            return 2;
        }
        const uint64_t samples_per_shard =
            argc >= 3 ? std::stoull(argv[2]) : 64;
        const double start = seconds_now();
        initialise_tables();
        initialise_weighted_increments();

        std::vector<SampleRecord> records;
        for (const std::string& path : split_paths(argv[1])) {
            std::vector<SampleRecord> part =
                read_stride_sample(path, samples_per_shard);
            records.insert(records.end(), part.begin(), part.end());
        }
        if (records.empty()) throw std::runtime_error("empty record sample");

        struct PendingJoin {
            CanonicalForm left;
            CanonicalForm right;
        };
        std::vector<PendingJoin> pending;
        pending.reserve(2 * records.size());
        std::unordered_map<uint32_t, SourceRequest> requests;
        constexpr uint32_t half_full = UINT32_MAX;
        for (const SampleRecord& record : records) {
            uint32_t left = half_prefix(record.key, 0);
            uint32_t right = half_prefix(record.key, HALF_COLUMNS);
            for (unsigned complement = 0; complement < 2; complement++) {
                CanonicalForm l =
                    canonical_prefix_local(left ^ (complement ? half_full : 0));
                CanonicalForm r = canonical_prefix_local(
                    right ^ (complement ? half_full : 0));
                pending.push_back(PendingJoin{l, r});
                auto& lr = requests[l.key];
                lr.key = l.key;
                lr.needed_left = true;
                auto& rr = requests[r.key];
                rr.key = r.key;
                rr.needed_right = true;
            }
        }
        std::vector<SourceRequest> source_requests;
        source_requests.reserve(requests.size());
        for (const auto& item : requests) source_requests.push_back(item.second);
        std::sort(source_requests.begin(), source_requests.end(),
                  [](const SourceRequest& a, const SourceRequest& b) {
                      return a.key < b.key;
                  });
        std::unordered_map<uint32_t, uint32_t> source_ids;
        for (size_t i = 0; i < source_requests.size(); i++)
            source_ids.emplace(source_requests[i].key, uint32_t(i));

        std::vector<Source> sources(source_requests.size());
#pragma omp parallel for schedule(dynamic, 1)
        for (long long i = 0; i < (long long)source_requests.size(); i++) {
            const SourceRequest request = source_requests[size_t(i)];
            std::vector<FullWeightedEntry> full =
                build_full_weighted_distribution(request.key, false);
            Source source;
            source.key = request.key;
            source.full_support = uint32_t(full.size());
            source.needed_left = request.needed_left;
            source.needed_right = request.needed_right;
            if (request.needed_left) {
                source.quotient_masks.reserve((full.size() + 1) / 2);
                uint64_t expanded = 0;
                for (const FullWeightedEntry& entry : full) {
                    uint64_t swapped = swap_token_planes_local(entry.mask);
                    if (entry.mask > swapped) continue;
                    source.quotient_masks.push_back(entry.mask);
                    expanded += entry.mask == swapped ? 1 : 2;
                }
                if (expanded != full.size())
                    throw std::runtime_error(
                        "token-plane quotient support invariant failed");
            }
            if (request.needed_right)
                source.automorphisms = half_mask_automorphisms(request.key);
            sources[size_t(i)] = std::move(source);
        }

        std::vector<JoinRef> joins;
        joins.reserve(pending.size());
        for (const PendingJoin& item : pending) {
            uint32_t left_id = source_ids.at(item.left.key);
            uint32_t right_id = source_ids.at(item.right.key);
            uint32_t relative = compose_row_maps_local(
                item.left.row_map, inverse_row_map_local(item.right.row_map));
            joins.push_back(JoinRef{left_id, right_id, relative});
        }

        // Validate the canonical row-map convention against independently
        // rebuilt labelled supports before measuring reuse.
        const size_t checks = std::min<size_t>(16, pending.size());
        for (size_t check = 0; check < checks; check++) {
            size_t index = check * pending.size() / checks;
            const PendingJoin& item = pending[index];
            uint32_t raw = half_prefix(records[index / 2].key, 0);
            if (index & 1) raw ^= half_full;
            std::vector<FullWeightedEntry> direct =
                build_full_weighted_distribution(raw, false);
            std::vector<uint64_t> direct_masks;
            for (const auto& entry : direct) {
                uint64_t swapped = swap_token_planes_local(entry.mask);
                if (entry.mask <= swapped) direct_masks.push_back(entry.mask);
            }
            const Source& canonical = sources[source_ids.at(item.left.key)];
            std::vector<uint64_t> transformed;
            transformed.reserve(canonical.quotient_masks.size());
            for (uint64_t mask : canonical.quotient_masks) {
                mask = transform_pair_mask_local(mask, item.left.row_map);
                transformed.push_back(
                    std::min(mask, swap_token_planes_local(mask)));
            }
            std::sort(direct_masks.begin(), direct_masks.end());
            std::sort(transformed.begin(), transformed.end());
            if (direct_masks != transformed)
                throw std::runtime_error("canonical query row-map validation failed");
        }

        std::vector<uint64_t> offsets(joins.size() + 1, 0);
        for (size_t i = 0; i < joins.size(); i++)
            offsets[i + 1] = offsets[i] +
                             sources[joins[i].left_source].quotient_masks.size();
        std::vector<QueryKey> occurrences(offsets.back());
#pragma omp parallel for schedule(dynamic, 1)
        for (long long i = 0; i < (long long)joins.size(); i++) {
            const JoinRef& join = joins[size_t(i)];
            const auto& masks = sources[join.left_source].quotient_masks;
            size_t output = offsets[size_t(i)];
            for (uint64_t mask : masks) {
                mask = transform_pair_mask_local(mask, join.relative_row_map);
                mask = std::min(mask, swap_token_planes_local(mask));
                occurrences[output++] = QueryKey{mask, join.right_source};
            }
        }

        std::cout << std::setprecision(12)
                  << "QUERY_REUSE_INPUT shards=" << split_paths(argv[1]).size()
                  << " records=" << records.size() << " joins=" << joins.size()
                  << " sources=" << sources.size()
                  << " occurrences=" << occurrences.size()
                  << " build_seconds=" << seconds_now() - start << '\n';

        std::vector<QueryGroup> plane_groups = group_queries(std::move(occurrences));
        cache_stats(plane_groups, sources, "plane");

        double automorphism_sum = 0;
        uint32_t maximum_automorphisms = 0;
        uint64_t right_sources = 0;
        for (const Source& source : sources) {
            if (!source.needed_right) continue;
            right_sources++;
            automorphism_sum += source.automorphisms.size();
            maximum_automorphisms = std::max<uint32_t>(
                maximum_automorphisms, source.automorphisms.size());
        }
        std::cout << "QUERY_REUSE_SYMMETRY right_sources=" << right_sources
                  << " mean_half_automorphisms="
                  << (right_sources ? automorphism_sum / right_sources : 0)
                  << " max_half_automorphisms=" << maximum_automorphisms
                  << '\n';

        std::vector<QueryGroup> symmetry_groups(plane_groups.size());
#pragma omp parallel for schedule(dynamic, 64)
        for (long long i = 0; i < (long long)plane_groups.size(); i++) {
            QueryGroup group = plane_groups[size_t(i)];
            group.key.mask = canonical_query_mask(
                group.key.mask, sources[group.key.source]);
            symmetry_groups[size_t(i)] = group;
        }
        const size_t query_checks = std::min<size_t>(16, plane_groups.size());
        for (size_t check = 0; check < query_checks; check++) {
            size_t index = check * plane_groups.size() / query_checks;
            const QueryGroup& group = plane_groups[index];
            const Source& source = sources[group.key.source];
            uint64_t canonical = canonical_query_mask(group.key.mask, source);
            std::vector<FullWeightedEntry> distribution =
                build_full_weighted_distribution(source.key, false);
            if (direct_query(distribution, group.key.mask) !=
                direct_query(distribution, canonical))
                throw std::runtime_error(
                    "right-source automorphism query validation failed");
        }
        symmetry_groups = group_counted_queries(std::move(symmetry_groups));
        cache_stats(symmetry_groups, sources, "plane+half_aut");

        std::cout << "QUERY_REUSE_DONE seconds=" << seconds_now() - start
                  << " exact=OK\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
