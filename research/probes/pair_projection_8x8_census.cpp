#include <array>
#include <iomanip>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#define PREFIX_BMMA_COST_CENSUS_NO_MAIN
#include "prefix_bmma_cost_census.cpp"

namespace {

constexpr unsigned PREFIX_COORDINATES = 7;
constexpr unsigned PREFIX_BITS = 2 * PREFIX_COORDINATES;
constexpr unsigned SUFFIX_COORDINATES = PAIRS - PREFIX_COORDINATES;
constexpr unsigned SUFFIX_BITS = 2 * SUFFIX_COORDINATES;
constexpr uint16_t PREFIX_FULL = (uint16_t(1) << PREFIX_BITS) - 1;

struct ProjectionClass {
    uint32_t weight = 0;
    uint8_t orbit_size = 0;
    uint64_t bit_or = 0;
    uint64_t bit_and = 0;
    std::vector<uint64_t> suffixes;
};

struct ProjectionBucket {
    uint16_t prefix = 0;
    uint32_t class_offset = 0;
    uint16_t class_count = 0;
};

struct ProjectionDistribution {
    std::vector<ProjectionBucket> buckets;
    std::vector<ProjectionClass> classes;
    std::vector<uint16_t> dense;
};

struct ProjectionPair {
    ProjectionDistribution selected;
    ProjectionDistribution complement;
};

struct ProjectionRecord {
    uint32_t left = 0;
    uint32_t right = 0;
};

struct Stats {
    U128 comparisons = 0;
    U128 tiles = 0;
    U128 terminal_accept_comparisons = 0;
    U128 terminal_accept_tiles = 0;
    U128 terminal_reject_comparisons = 0;
    U128 terminal_reject_tiles = 0;
    U128 dimension_comparisons[SUFFIX_BITS + 1]{};
    U128 dimension_tiles[SUFFIX_BITS + 1]{};
    uint64_t orientations = 0;
    uint64_t class_pairs = 0;
    uint64_t tested_bucket_pairs = 0;
    uint64_t compatible_bucket_orientations = 0;
    uint64_t dimension_tasks[SUFFIX_BITS + 1]{};
    // Candidate coverage when a dense SOS transform needs at most 1/F of the
    // raw Cartesian operations.  F=128 is approximately the predicate count
    // represented by one BMMA tile, before charging table traffic.
    U128 sos_candidate_tiles[4]{};
    U128 sos_candidate_comparisons[4]{};
    uint64_t sos_candidate_tasks[4]{};
};

constexpr std::array<unsigned, 4> SOS_GATES = {1, 8, 32, 128};

struct HeavyTask {
    const ProjectionClass* left = nullptr;
    const ProjectionClass* right = nullptr;
    uint64_t tiles = 0;
    bool swap_right = false;
};

struct MinTiles {
    bool operator()(const HeavyTask& a, const HeavyTask& b) const {
        return a.tiles > b.tiles;
    }
};

static uint64_t swap_token_planes_local(uint64_t mask) {
    constexpr uint64_t plane_mask = (UINT64_C(1) << PAIRS) - 1;
    return ((mask & plane_mask) << PAIRS) |
           ((mask >> PAIRS) & plane_mask);
}

static uint16_t swap_prefix_planes(uint16_t prefix) {
    constexpr uint16_t plane_mask =
        (uint16_t(1) << PREFIX_COORDINATES) - 1;
    return uint16_t(((prefix & plane_mask) << PREFIX_COORDINATES) |
                    ((prefix >> PREFIX_COORDINATES) & plane_mask));
}

static uint64_t swap_suffix_planes(uint64_t suffix) {
    constexpr uint64_t plane_mask =
        (UINT64_C(1) << SUFFIX_COORDINATES) - 1;
    return ((suffix & plane_mask) << SUFFIX_COORDINATES) |
           ((suffix >> SUFFIX_COORDINATES) & plane_mask);
}

static ProjectionDistribution build_distribution(uint32_t prefix,
                                                 bool complement) {
    struct Item {
        uint16_t prefix;
        uint32_t weight;
        uint8_t orbit_size;
        uint64_t suffix;
    };
    std::vector<FullWeightedEntry> full =
        build_full_weighted_distribution(prefix, complement);
    std::vector<Item> items;
    items.reserve((full.size() + 1) / 2);
    const uint32_t pair_mask = production_mask(PREFIX_COORDINATES);
    uint64_t expanded_entries = 0;
    for (const FullWeightedEntry& entry : full) {
        uint64_t swapped = swap_token_planes_local(entry.mask);
        if (entry.mask > swapped) continue;
        uint16_t bucket_prefix = 0;
        uint64_t suffix = 0;
        split_pair_mask(entry.mask, pair_mask, bucket_prefix, suffix);
        uint8_t orbit_size = uint8_t(entry.mask == swapped ? 1 : 2);
        items.push_back(Item{bucket_prefix, entry.weight, orbit_size, suffix});
        expanded_entries += orbit_size;
    }
    if (expanded_entries != full.size())
        throw std::runtime_error("token-plane quotient invariant failed");
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        if (a.prefix != b.prefix) return a.prefix < b.prefix;
        if (a.weight != b.weight) return a.weight < b.weight;
        if (a.orbit_size != b.orbit_size)
            return a.orbit_size < b.orbit_size;
        return a.suffix < b.suffix;
    });

    ProjectionDistribution result;
    result.dense.assign(size_t(1) << PREFIX_BITS, UINT16_MAX);
    for (size_t begin = 0; begin < items.size();) {
        size_t bucket_end = begin + 1;
        while (bucket_end < items.size() &&
               items[bucket_end].prefix == items[begin].prefix)
            bucket_end++;
        if (result.buckets.size() >= UINT16_MAX)
            throw std::overflow_error("too many prefix buckets");
        ProjectionBucket bucket{items[begin].prefix,
                                uint32_t(result.classes.size()), 0};
        for (size_t class_begin = begin; class_begin < bucket_end;) {
            size_t class_end = class_begin + 1;
            while (class_end < bucket_end &&
                   items[class_end].weight == items[class_begin].weight &&
                   items[class_end].orbit_size ==
                       items[class_begin].orbit_size)
                class_end++;
            ProjectionClass output;
            output.weight = items[class_begin].weight;
            output.orbit_size = items[class_begin].orbit_size;
            output.bit_and = (UINT64_C(1) << SUFFIX_BITS) - 1;
            output.suffixes.reserve(class_end - class_begin);
            for (size_t index = class_begin; index < class_end; index++) {
                uint64_t suffix = items[index].suffix;
                output.suffixes.push_back(suffix);
                output.bit_or |= suffix;
                output.bit_and &= suffix;
            }
            result.classes.push_back(std::move(output));
            bucket.class_count++;
            class_begin = class_end;
        }
        uint16_t ordinal = uint16_t(result.buckets.size());
        result.dense[bucket.prefix] = ordinal;
        result.buckets.push_back(bucket);
        begin = bucket_end;
    }
    return result;
}

static ProjectionPair build_pair_local(uint32_t prefix) {
    return ProjectionPair{build_distribution(prefix, false),
                          build_distribution(prefix, true)};
}

static void add_stats(Stats& destination, const Stats& source) {
    destination.comparisons += source.comparisons;
    destination.tiles += source.tiles;
    destination.terminal_accept_comparisons +=
        source.terminal_accept_comparisons;
    destination.terminal_accept_tiles += source.terminal_accept_tiles;
    destination.terminal_reject_comparisons +=
        source.terminal_reject_comparisons;
    destination.terminal_reject_tiles += source.terminal_reject_tiles;
    destination.orientations += source.orientations;
    destination.class_pairs += source.class_pairs;
    destination.tested_bucket_pairs += source.tested_bucket_pairs;
    destination.compatible_bucket_orientations +=
        source.compatible_bucket_orientations;
    for (unsigned d = 0; d <= SUFFIX_BITS; d++) {
        destination.dimension_comparisons[d] += source.dimension_comparisons[d];
        destination.dimension_tiles[d] += source.dimension_tiles[d];
        destination.dimension_tasks[d] += source.dimension_tasks[d];
    }
    for (unsigned gate = 0; gate < SOS_GATES.size(); gate++) {
        destination.sos_candidate_tiles[gate] +=
            source.sos_candidate_tiles[gate];
        destination.sos_candidate_comparisons[gate] +=
            source.sos_candidate_comparisons[gate];
        destination.sos_candidate_tasks[gate] +=
            source.sos_candidate_tasks[gate];
    }
}

static void retain_heavy(
    std::priority_queue<HeavyTask, std::vector<HeavyTask>, MinTiles>& heap,
    size_t limit, HeavyTask task) {
    if (!limit) return;
    if (heap.size() < limit) {
        heap.push(task);
    } else if (task.tiles > heap.top().tiles) {
        heap.pop();
        heap.push(task);
    }
}

static void visit_class_pair(
    const ProjectionClass& left, const ProjectionClass& right,
    bool swap_right, Stats& stats,
    std::priority_queue<HeavyTask, std::vector<HeavyTask>, MinTiles>& heap,
    size_t heavy_limit) {
    uint64_t right_or = swap_right ? swap_suffix_planes(right.bit_or)
                                   : right.bit_or;
    uint64_t right_and = swap_right ? swap_suffix_planes(right.bit_and)
                                    : right.bit_and;
    uint64_t mask = left.bit_or & right_or;
    uint64_t comparisons =
        uint64_t(left.suffixes.size()) * right.suffixes.size();
    uint64_t tiles = predicate_bmma_tiles(uint32_t(left.suffixes.size()),
                                          uint32_t(right.suffixes.size()));
    unsigned dimension = __builtin_popcountll(mask);
    stats.orientations++;
    stats.class_pairs++;
    stats.comparisons += comparisons;
    stats.tiles += tiles;
    stats.dimension_tasks[dimension]++;
    stats.dimension_comparisons[dimension] += comparisons;
    stats.dimension_tiles[dimension] += tiles;
    U128 sos_operations =
        U128(dimension) * (U128(1) << dimension) +
        left.suffixes.size() + right.suffixes.size();
    for (unsigned gate = 0; gate < SOS_GATES.size(); gate++) {
        if (sos_operations * SOS_GATES[gate] <= comparisons) {
            stats.sos_candidate_tiles[gate] += tiles;
            stats.sos_candidate_comparisons[gate] += comparisons;
            stats.sos_candidate_tasks[gate]++;
        }
    }
    if (!mask) {
        stats.terminal_accept_comparisons += comparisons;
        stats.terminal_accept_tiles += tiles;
    } else if (left.bit_and & right_and) {
        stats.terminal_reject_comparisons += comparisons;
        stats.terminal_reject_tiles += tiles;
    } else {
        retain_heavy(heap, heavy_limit,
                     HeavyTask{&left, &right, tiles, swap_right});
    }
}

static void distribution_stats(
    const ProjectionDistribution& left,
    const ProjectionDistribution& right, Stats& stats,
    std::priority_queue<HeavyTask, std::vector<HeavyTask>, MinTiles>& heap,
    size_t heavy_limit) {
    stats.tested_bucket_pairs +=
        uint64_t(left.buckets.size()) * right.buckets.size();
    auto visit_orientation = [&](const ProjectionBucket& lhs, bool swapped) {
        uint16_t query = swapped ? swap_prefix_planes(lhs.prefix) : lhs.prefix;
        uint16_t allowed = uint16_t(PREFIX_FULL ^ query);
        auto visit_bucket = [&](const ProjectionBucket& rhs) {
            stats.compatible_bucket_orientations++;
            for (uint32_t li = 0; li < lhs.class_count; li++) {
                const ProjectionClass& a =
                    left.classes[lhs.class_offset + li];
                for (uint32_t ri = 0; ri < rhs.class_count; ri++) {
                    const ProjectionClass& b =
                        right.classes[rhs.class_offset + ri];
                    if (swapped && b.orbit_size != 2) continue;
                    visit_class_pair(a, b, swapped, stats, heap, heavy_limit);
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

struct ProjectedSupport {
    std::vector<uint64_t> values;
    std::vector<uint32_t> multiplicities;
    std::vector<uint32_t> weight_class_counts;
};

static ProjectedSupport project_and_aggregate(
    const ProjectionClass& source, uint64_t mask, bool swap) {
    std::vector<uint64_t> values;
    values.reserve(source.suffixes.size());
    for (uint64_t suffix : source.suffixes) {
        if (swap) suffix = swap_suffix_planes(suffix);
        values.push_back(suffix & mask);
    }
    std::sort(values.begin(), values.end());
    std::unordered_map<uint32_t, uint32_t> multiplicity_classes;
    ProjectedSupport result;
    result.values.reserve(values.size());
    for (size_t begin = 0; begin < values.size();) {
        size_t end = begin + 1;
        while (end < values.size() && values[end] == values[begin]) end++;
        result.values.push_back(values[begin]);
        uint32_t multiplicity = uint32_t(end - begin);
        result.multiplicities.push_back(multiplicity);
        multiplicity_classes[multiplicity]++;
        begin = end;
    }
    result.weight_class_counts.reserve(multiplicity_classes.size());
    for (const auto& item : multiplicity_classes)
        result.weight_class_counts.push_back(item.second);
    return result;
}

static uint64_t projected_bmma_tiles(const ProjectedSupport& left,
                                     const ProjectedSupport& right) {
    uint64_t result = 0;
    for (uint32_t a : left.weight_class_counts)
        for (uint32_t b : right.weight_class_counts)
            result += predicate_bmma_tiles(a, b);
    return result;
}

static U128 direct_compatible_pairs(const ProjectionClass& left,
                                    const ProjectionClass& right,
                                    bool swap_right) {
    U128 result = 0;
    for (uint64_t a : left.suffixes) {
        for (uint64_t b : right.suffixes) {
            if (swap_right) b = swap_suffix_planes(b);
            result += !(a & b);
        }
    }
    return result;
}

static U128 projected_compatible_pairs(const ProjectedSupport& left,
                                       const ProjectedSupport& right) {
    U128 result = 0;
    for (size_t a = 0; a < left.values.size(); a++) {
        for (size_t b = 0; b < right.values.size(); b++) {
            if (!(left.values[a] & right.values[b]))
                result += U128(left.multiplicities[a]) *
                          right.multiplicities[b];
        }
    }
    return result;
}

static U128 submask_queries(const std::vector<uint64_t>& queries,
                            unsigned dimension) {
    U128 result = 0;
    for (uint64_t query : queries) {
        unsigned free_bits = dimension - __builtin_popcountll(query);
        result += U128(1) << free_bits;
    }
    return result;
}

static double projection_ratio(U128 numerator, U128 denominator) {
    if (!denominator) return 0;
    return double(static_cast<long double>(numerator) /
                  static_cast<long double>(denominator));
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

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 4) {
            std::cerr << "usage: " << argv[0]
                      << " SHARD[,SHARD...] [SAMPLES_PER_SHARD=16]"
                         " [HEAVY_TASKS=1024]\n";
            return 2;
        }
        uint64_t samples = argc >= 3 ? std::stoull(argv[2]) : 16;
        size_t heavy_limit = argc >= 4 ? std::stoull(argv[3]) : 1024;
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
        uint64_t support_entries = 0, buckets = 0, classes = 0;
        for (const ProjectionPair& pair : pairs) {
            for (const ProjectionDistribution* distribution :
                 {&pair.selected, &pair.complement}) {
                buckets += distribution->buckets.size();
                classes += distribution->classes.size();
                for (const ProjectionClass& item : distribution->classes)
                    support_entries += item.suffixes.size();
            }
        }
        std::cout << std::setprecision(12)
                  << "PAIR_PROJECTION_CACHE records=" << records.size()
                  << " half_keys=" << prefixes.size()
                  << " support_entries=" << support_entries
                  << " buckets=" << buckets << " classes=" << classes
                  << " build_seconds=" << build_seconds << '\n';

        Stats total;
        std::priority_queue<HeavyTask, std::vector<HeavyTask>, MinTiles> heap;
        double census_start = seconds_now();
        for (const ProjectionRecord& record : records) {
            const ProjectionPair& left = pairs[record.left];
            const ProjectionPair& right = pairs[record.right];
            distribution_stats(left.selected, right.selected, total, heap,
                               heavy_limit);
            distribution_stats(left.complement, right.complement, total, heap,
                               heavy_limit);
        }
        double census_seconds = seconds_now() - census_start;
        U128 terminal_tiles =
            total.terminal_accept_tiles + total.terminal_reject_tiles;
        U128 terminal_comparisons = total.terminal_accept_comparisons +
                                    total.terminal_reject_comparisons;
        std::cout << "PAIR_PROJECTION_TOTAL orientations="
                  << total.orientations
                  << " tested_bucket_pairs=" << total.tested_bucket_pairs
                  << " compatible_bucket_orientations="
                  << total.compatible_bucket_orientations
                  << " class_pairs=" << total.class_pairs
                  << " comparisons=" << u128_string(total.comparisons)
                  << " tiles=" << u128_string(total.tiles)
                  << " terminal_accept_tile_ratio="
                  << projection_ratio(total.terminal_accept_tiles, total.tiles)
                  << " terminal_reject_tile_ratio="
                  << projection_ratio(total.terminal_reject_tiles, total.tiles)
                  << " terminal_total_tile_ratio="
                  << projection_ratio(terminal_tiles, total.tiles)
                  << " terminal_total_comparison_ratio="
                  << projection_ratio(terminal_comparisons, total.comparisons)
                  << " census_seconds=" << census_seconds << '\n';
        U128 cumulative_tiles = 0, cumulative_comparisons = 0;
        uint64_t cumulative_tasks = 0;
        for (unsigned d = 0; d <= SUFFIX_BITS; d++) {
            cumulative_tiles += total.dimension_tiles[d];
            cumulative_comparisons += total.dimension_comparisons[d];
            cumulative_tasks += total.dimension_tasks[d];
            if (total.dimension_tasks[d]) {
                std::cout << "PAIR_PROJECTION_DIMENSION d=" << d
                          << " tasks=" << total.dimension_tasks[d]
                          << " tile_ratio="
                          << projection_ratio(total.dimension_tiles[d], total.tiles)
                          << " comparison_ratio="
                          << projection_ratio(total.dimension_comparisons[d],
                                              total.comparisons)
                          << " cumulative_tile_ratio="
                          << projection_ratio(cumulative_tiles, total.tiles)
                          << " cumulative_comparison_ratio="
                          << projection_ratio(cumulative_comparisons,
                                              total.comparisons)
                          << " cumulative_task_ratio="
                          << double(cumulative_tasks) / total.class_pairs
                          << '\n';
            }
        }
        for (unsigned gate = 0; gate < SOS_GATES.size(); gate++) {
            std::cout << "PAIR_PROJECTION_SOS_GATE factor="
                      << SOS_GATES[gate]
                      << " tasks=" << total.sos_candidate_tasks[gate]
                      << " tile_coverage="
                      << projection_ratio(total.sos_candidate_tiles[gate],
                                          total.tiles)
                      << " comparison_coverage="
                      << projection_ratio(
                             total.sos_candidate_comparisons[gate],
                             total.comparisons)
                      << '\n';
        }

        std::vector<HeavyTask> heavy;
        while (!heap.empty()) {
            heavy.push_back(heap.top());
            heap.pop();
        }
        U128 heavy_tiles = 0, projected_tiles = 0, raw_pairs = 0;
        U128 projected_pairs = 0;
        U128 sos_operations = 0, submask_operations = 0;
        uint64_t raw_entries = 0, projected_entries = 0;
        uint64_t validated = 0;
        std::array<uint64_t, SUFFIX_BITS + 1> heavy_dimensions{};
        for (const HeavyTask& task : heavy) {
            uint64_t right_or = task.swap_right
                ? swap_suffix_planes(task.right->bit_or)
                : task.right->bit_or;
            uint64_t mask = task.left->bit_or & right_or;
            unsigned dimension = __builtin_popcountll(mask);
            ProjectedSupport a =
                project_and_aggregate(*task.left, mask, false);
            ProjectedSupport b =
                project_and_aggregate(*task.right, mask, task.swap_right);
            if (validated < 32) {
                U128 direct = direct_compatible_pairs(
                    *task.left, *task.right, task.swap_right);
                U128 projected = projected_compatible_pairs(a, b);
                if (direct != projected)
                    throw std::runtime_error(
                        "pair-specific projection changed an exact join");
                validated++;
            }
            uint64_t original_a = task.left->suffixes.size();
            uint64_t original_b = task.right->suffixes.size();
            heavy_tiles += task.tiles;
            projected_tiles += projected_bmma_tiles(a, b);
            raw_pairs += U128(original_a) * original_b;
            projected_pairs += U128(a.values.size()) * b.values.size();
            raw_entries += original_a + original_b;
            projected_entries += a.values.size() + b.values.size();
            heavy_dimensions[dimension]++;
            if (dimension < 64) {
                sos_operations +=
                    U128(dimension) * (U128(1) << dimension) +
                    a.values.size() + b.values.size();
                U128 query_a = U128(b.values.size()) +
                               submask_queries(a.values, dimension);
                U128 query_b = U128(a.values.size()) +
                               submask_queries(b.values, dimension);
                submask_operations += std::min(query_a, query_b);
            }
        }
        std::cout << "PAIR_PROJECTION_HEAVY tasks=" << heavy.size()
                  << " tile_coverage="
                  << projection_ratio(heavy_tiles, total.tiles)
                  << " projected_tile_ratio="
                  << projection_ratio(projected_tiles, heavy_tiles)
                  << " raw_pairs=" << u128_string(raw_pairs)
                  << " projected_pairs=" << u128_string(projected_pairs)
                  << " projected_pair_ratio="
                  << projection_ratio(projected_pairs, raw_pairs)
                  << " raw_entries=" << raw_entries
                  << " projected_entries=" << projected_entries
                  << " projected_entry_ratio="
                  << (raw_entries ? double(projected_entries) / raw_entries
                                  : 0.0)
                  << " sos_operation_ratio="
                  << projection_ratio(sos_operations, raw_pairs)
                  << " submask_operation_ratio="
                  << projection_ratio(submask_operations, raw_pairs)
                  << " validated=" << validated << " exact=OK\n";
        for (unsigned d = 0; d <= SUFFIX_BITS; d++) {
            if (heavy_dimensions[d])
                std::cout << "PAIR_PROJECTION_HEAVY_DIMENSION d=" << d
                          << " tasks=" << heavy_dimensions[d] << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
