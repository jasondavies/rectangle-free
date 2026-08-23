#include <array>
#include <iomanip>
#include <unordered_map>
#include <unordered_set>

#define PAIR_PROJECTION_8X8_CENSUS_NO_MAIN
#include "pair_projection_8x8_census.cpp"

namespace {

// An intentionally optimistic instruction-count model for indexing one exact
// weight class in blocks of 64 suffixes.  Each 42-bit suffix is six 7-bit
// chunks.  A query performs six 64-bit table loads, five ANDs, and one POPC.
// Building all 128 query words for a chunk takes seven two-warp ballots and
// 127 recurrence ANDs.  Reduction, addressing, synchronization, and weighted
// accumulation are omitted, so failure under this model is decisive.
constexpr uint64_t QUERY_CORE_INSTRUCTIONS = 12;
constexpr uint64_t BITSET_CHUNKS = 6;
constexpr uint64_t INDEX_CORE_INSTRUCTIONS =
    BITSET_CHUNKS * (2 * 7 + ((UINT64_C(1) << 7) - 1));
constexpr uint64_t INDEX_BYTES_PER_BLOCK =
    BITSET_CHUNKS * (UINT64_C(1) << 7) * sizeof(uint64_t);

constexpr std::array<uint32_t, 7> CLASS_THRESHOLDS = {
    1, 16, 32, 64, 128, 256, 1024};

struct BitsetStats {
    U128 bmma_tiles = 0;
    U128 fixed_right_query = 0;
    U128 fixed_left_query = 0;
    U128 zero_setup_best = 0;
    U128 zero_setup_hybrid = 0;
    U128 per_pair_setup_best = 0;
    U128 tile_coverage_max[CLASS_THRESHOLDS.size()]{};
    U128 tile_coverage_min[CLASS_THRESHOLDS.size()]{};
    U128 zero_setup_better_tiles = 0;
    U128 zero_setup_25pct_tiles = 0;
    uint64_t class_orientations = 0;
    uint64_t zero_setup_better_pairs = 0;
    uint64_t zero_setup_25pct_pairs = 0;
    std::unordered_set<const ProjectionClass*> indexed_right;
    std::unordered_set<const ProjectionClass*> indexed_left;
    std::unordered_map<const ProjectionClass*, U128> right_savings;
    std::unordered_map<const ProjectionClass*, U128> left_savings;
    std::unordered_map<const ProjectionClass*, U128> oracle_savings;
};

static uint64_t ceil_div_u64(uint64_t value, uint64_t divisor) {
    return (value + divisor - 1) / divisor;
}

static uint64_t index_blocks(uint64_t entries) {
    return ceil_div_u64(entries, 64);
}

static uint64_t query_warps(uint64_t entries) {
    return ceil_div_u64(entries, 32);
}

static uint64_t lookup_query_cost(uint64_t queries, uint64_t indexed) {
    return query_warps(queries) * index_blocks(indexed) *
           QUERY_CORE_INSTRUCTIONS;
}

static uint64_t lookup_setup_cost(uint64_t indexed) {
    return index_blocks(indexed) * INDEX_CORE_INSTRUCTIONS;
}

static void bitset_visit_class_pair(const ProjectionClass& left,
                                    const ProjectionClass& right,
                                    BitsetStats& stats) {
    const uint64_t a = left.suffixes.size();
    const uint64_t b = right.suffixes.size();
    const uint64_t tiles = predicate_bmma_tiles(uint32_t(a), uint32_t(b));
    const uint64_t fixed_right = lookup_query_cost(a, b);
    const uint64_t fixed_left = lookup_query_cost(b, a);
    const uint64_t best = std::min(fixed_right, fixed_left);
    const uint64_t pair_setup = std::min(
        fixed_right + lookup_setup_cost(b),
        fixed_left + lookup_setup_cost(a));

    stats.class_orientations++;
    stats.bmma_tiles += tiles;
    stats.fixed_right_query += fixed_right;
    stats.fixed_left_query += fixed_left;
    stats.zero_setup_best += best;
    stats.zero_setup_hybrid += std::min(tiles, best);
    stats.per_pair_setup_best += pair_setup;
    stats.indexed_right.insert(&right);
    stats.indexed_left.insert(&left);
    if (fixed_right < tiles)
        stats.right_savings[&right] += tiles - fixed_right;
    if (fixed_left < tiles)
        stats.left_savings[&left] += tiles - fixed_left;
    if (best < tiles) {
        const ProjectionClass* indexed =
            fixed_right <= fixed_left ? &right : &left;
        stats.oracle_savings[indexed] += tiles - best;
    }
    if (best < tiles) {
        stats.zero_setup_better_pairs++;
        stats.zero_setup_better_tiles += tiles;
    }
    if (U128(best) * 4 <= U128(tiles) * 3) {
        stats.zero_setup_25pct_pairs++;
        stats.zero_setup_25pct_tiles += tiles;
    }
    for (size_t i = 0; i < CLASS_THRESHOLDS.size(); i++) {
        if (std::max(a, b) >= CLASS_THRESHOLDS[i])
            stats.tile_coverage_max[i] += tiles;
        if (std::min(a, b) >= CLASS_THRESHOLDS[i])
            stats.tile_coverage_min[i] += tiles;
    }
}

static void bitset_distribution_stats(const ProjectionDistribution& left,
                                      const ProjectionDistribution& right,
                                      BitsetStats& stats) {
    auto visit_orientation = [&](const ProjectionBucket& lhs, bool swapped) {
        uint16_t query = swapped ? swap_prefix_planes(lhs.prefix) : lhs.prefix;
        uint16_t allowed = uint16_t(PREFIX_FULL ^ query);
        auto visit_bucket = [&](const ProjectionBucket& rhs) {
            for (uint32_t li = 0; li < lhs.class_count; li++) {
                const ProjectionClass& a =
                    left.classes[lhs.class_offset + li];
                for (uint32_t ri = 0; ri < rhs.class_count; ri++) {
                    const ProjectionClass& b =
                        right.classes[rhs.class_offset + ri];
                    if (swapped && b.orbit_size != 2) continue;
                    bitset_visit_class_pair(a, b, stats);
                }
            }
        };
        uint32_t submask_count = uint32_t(1) << __builtin_popcount(allowed);
        if (submask_count < right.buckets.size()) {
            uint16_t value = allowed;
            for (;;) {
                uint16_t ordinal = right.dense[value];
                if (ordinal != UINT16_MAX) visit_bucket(right.buckets[ordinal]);
                if (!value) break;
                value = uint16_t((value - 1) & allowed);
            }
        } else {
            for (const ProjectionBucket& rhs : right.buckets)
                if (!(query & rhs.prefix)) visit_bucket(rhs);
        }
    };
    for (const ProjectionBucket& bucket : left.buckets) {
        visit_orientation(bucket, false);
        visit_orientation(bucket, true);
    }
}

static U128 persistent_setup_cost(
    const std::unordered_set<const ProjectionClass*>& classes) {
    U128 result = 0;
    for (const ProjectionClass* item : classes)
        result += lookup_setup_cost(item->suffixes.size());
    return result;
}

struct AdaptivePolicy {
    U128 net_savings = 0;
    U128 table_bytes = 0;
    uint64_t indexed_classes = 0;
};

static AdaptivePolicy adaptive_policy(
    const std::unordered_map<const ProjectionClass*, U128>& savings) {
    AdaptivePolicy result;
    for (const auto& item : savings) {
        U128 setup = lookup_setup_cost(item.first->suffixes.size());
        if (item.second <= setup) continue;
        result.net_savings += item.second - setup;
        result.table_bytes +=
            U128(index_blocks(item.first->suffixes.size())) *
            INDEX_BYTES_PER_BLOCK;
        result.indexed_classes++;
    }
    return result;
}

static void print_class_census(const std::vector<ProjectionPair>& pairs) {
    uint64_t class_count = 0;
    uint64_t entries = 0;
    uint64_t maximum = 0;
    std::array<uint64_t, CLASS_THRESHOLDS.size()> counts{};
    std::array<uint64_t, CLASS_THRESHOLDS.size()> covered_entries{};
    for (const ProjectionPair& pair : pairs) {
        for (const ProjectionDistribution* distribution :
             {&pair.selected, &pair.complement}) {
            for (const ProjectionClass& item : distribution->classes) {
                uint64_t size = item.suffixes.size();
                class_count++;
                entries += size;
                maximum = std::max(maximum, size);
                for (size_t i = 0; i < CLASS_THRESHOLDS.size(); i++) {
                    if (size >= CLASS_THRESHOLDS[i]) {
                        counts[i]++;
                        covered_entries[i] += size;
                    }
                }
            }
        }
    }
    std::cout << "WEIGHT_CLASS_BITSET_CLASSES classes=" << class_count
              << " entries=" << entries
              << " mean_size="
              << (class_count ? static_cast<double>(entries) /
                                    static_cast<double>(class_count)
                              : 0.0)
              << " maximum_size=" << maximum << '\n';
    for (size_t i = 0; i < CLASS_THRESHOLDS.size(); i++) {
        std::cout << "WEIGHT_CLASS_BITSET_CLASS_GATE size="
                  << CLASS_THRESHOLDS[i]
                  << " class_ratio="
                  << (class_count ? static_cast<double>(counts[i]) /
                                        static_cast<double>(class_count)
                                  : 0.0)
                  << " entry_ratio="
                  << (entries ? static_cast<double>(covered_entries[i]) /
                                    static_cast<double>(entries)
                              : 0.0)
                  << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) {
            std::cerr << "usage: " << argv[0]
                      << " SHARD[,SHARD...] [SAMPLES_PER_SHARD=16]\n";
            return 2;
        }
        uint64_t samples = argc >= 3 ? std::stoull(argv[2]) : 16;
        initialise_tables();
        initialise_weighted_increments();
        std::vector<SampleRecord> sampled;
        for (const std::string& path : split_paths(argv[1])) {
            std::vector<SampleRecord> part = read_stride_sample(path, samples);
            sampled.insert(sampled.end(), part.begin(), part.end());
        }
        std::vector<uint32_t> prefixes;
        prefixes.reserve(2 * sampled.size());
        for (const SampleRecord& record : sampled) {
            prefixes.push_back(half_prefix(record.key, 0));
            prefixes.push_back(half_prefix(record.key, HALF_COLUMNS));
        }
        std::sort(prefixes.begin(), prefixes.end());
        prefixes.erase(std::unique(prefixes.begin(), prefixes.end()),
                       prefixes.end());
        std::vector<ProjectionPair> pairs(prefixes.size());
        double build_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
        for (long long i = 0; i < (long long)prefixes.size(); i++)
            pairs[size_t(i)] = build_pair_local(prefixes[size_t(i)]);
        double build_seconds = seconds_now() - build_start;
        std::unordered_map<uint32_t, uint32_t> ids;
        for (size_t i = 0; i < prefixes.size(); i++)
            ids.emplace(prefixes[i], uint32_t(i));
        std::vector<ProjectionRecord> records;
        records.reserve(sampled.size());
        for (const SampleRecord& record : sampled) {
            records.push_back(ProjectionRecord{
                ids.at(half_prefix(record.key, 0)),
                ids.at(half_prefix(record.key, HALF_COLUMNS))});
        }

        std::cout << std::setprecision(12)
                  << "WEIGHT_CLASS_BITSET_CACHE records=" << records.size()
                  << " half_keys=" << prefixes.size()
                  << " build_seconds=" << build_seconds
                  << " query_core_instructions=" << QUERY_CORE_INSTRUCTIONS
                  << " index_core_instructions=" << INDEX_CORE_INSTRUCTIONS
                  << " index_bytes_per_64=" << INDEX_BYTES_PER_BLOCK << '\n';
        print_class_census(pairs);

        BitsetStats total;
        double census_start = seconds_now();
        for (const ProjectionRecord& record : records) {
            const ProjectionPair& left = pairs[record.left];
            const ProjectionPair& right = pairs[record.right];
            bitset_distribution_stats(left.selected, right.selected, total);
            bitset_distribution_stats(left.complement, right.complement, total);
        }
        double census_seconds = seconds_now() - census_start;
        U128 right_setup = persistent_setup_cost(total.indexed_right);
        U128 left_setup = persistent_setup_cost(total.indexed_left);
        U128 persistent_right = total.fixed_right_query + right_setup;
        U128 persistent_left = total.fixed_left_query + left_setup;
        AdaptivePolicy adaptive_right = adaptive_policy(total.right_savings);
        AdaptivePolicy adaptive_left = adaptive_policy(total.left_savings);
        AdaptivePolicy adaptive_oracle = adaptive_policy(total.oracle_savings);

        std::cout << "WEIGHT_CLASS_BITSET_TOTAL class_orientations="
                  << total.class_orientations
                  << " bmma_tiles=" << u128_string(total.bmma_tiles)
                  << " fixed_right_zero_setup_ratio="
                  << projection_ratio(total.fixed_right_query,
                                      total.bmma_tiles)
                  << " fixed_left_zero_setup_ratio="
                  << projection_ratio(total.fixed_left_query,
                                      total.bmma_tiles)
                  << " role_oracle_zero_setup_ratio="
                  << projection_ratio(total.zero_setup_best,
                                      total.bmma_tiles)
                  << " role_oracle_hybrid_zero_setup_ratio="
                  << projection_ratio(total.zero_setup_hybrid,
                                      total.bmma_tiles)
                  << " per_pair_setup_ratio="
                  << projection_ratio(total.per_pair_setup_best,
                                      total.bmma_tiles)
                  << " census_seconds=" << census_seconds << '\n';
        std::cout << "WEIGHT_CLASS_BITSET_PERSISTENT policy=right"
                  << " indexed_classes=" << total.indexed_right.size()
                  << " setup=" << u128_string(right_setup)
                  << " query=" << u128_string(total.fixed_right_query)
                  << " total_ratio="
                  << projection_ratio(persistent_right, total.bmma_tiles)
                  << '\n';
        std::cout << "WEIGHT_CLASS_BITSET_PERSISTENT policy=left"
                  << " indexed_classes=" << total.indexed_left.size()
                  << " setup=" << u128_string(left_setup)
                  << " query=" << u128_string(total.fixed_left_query)
                  << " total_ratio="
                  << projection_ratio(persistent_left, total.bmma_tiles)
                  << '\n';
        auto print_adaptive = [&](const char* policy,
                                  const AdaptivePolicy& value) {
            std::cout << "WEIGHT_CLASS_BITSET_ADAPTIVE policy=" << policy
                      << " indexed_classes=" << value.indexed_classes
                      << " table_bytes=" << u128_string(value.table_bytes)
                      << " net_savings=" << u128_string(value.net_savings)
                      << " hybrid_ratio="
                      << projection_ratio(total.bmma_tiles -
                                              value.net_savings,
                                          total.bmma_tiles)
                      << '\n';
        };
        print_adaptive("right", adaptive_right);
        print_adaptive("left", adaptive_left);
        print_adaptive("role_oracle", adaptive_oracle);
        std::cout << "WEIGHT_CLASS_BITSET_ORACLE better_pairs="
                  << total.zero_setup_better_pairs
                  << " better_pair_ratio="
                  << (total.class_orientations
                          ? static_cast<double>(total.zero_setup_better_pairs) /
                                static_cast<double>(total.class_orientations)
                          : 0.0)
                  << " better_tile_coverage="
                  << projection_ratio(total.zero_setup_better_tiles,
                                      total.bmma_tiles)
                  << " pct25_pairs=" << total.zero_setup_25pct_pairs
                  << " pct25_pair_ratio="
                  << (total.class_orientations
                          ? static_cast<double>(total.zero_setup_25pct_pairs) /
                                static_cast<double>(total.class_orientations)
                          : 0.0)
                  << " pct25_tile_coverage="
                  << projection_ratio(total.zero_setup_25pct_tiles,
                                      total.bmma_tiles)
                  << '\n';
        for (size_t i = 0; i < CLASS_THRESHOLDS.size(); i++) {
            std::cout << "WEIGHT_CLASS_BITSET_TILE_GATE size="
                      << CLASS_THRESHOLDS[i]
                      << " max_side_tile_coverage="
                      << projection_ratio(total.tile_coverage_max[i],
                                          total.bmma_tiles)
                      << " min_side_tile_coverage="
                      << projection_ratio(total.tile_coverage_min[i],
                                          total.bmma_tiles)
                      << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
