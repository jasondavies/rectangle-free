#include <immintrin.h>

#define PREFIX_BUCKET_TT_RANK_CENSUS_NO_MAIN
#include "prefix_bucket_tt_rank_census.cpp"

namespace {

constexpr uint32_t CENSUS_TASK_CHUNK = 16;

struct CensusEntry {
    uint64_t mask;
    uint8_t weight_ordinal;
};

struct FullDistribution {
    std::vector<CensusEntry> entries;
    std::vector<uint32_t> weights;
};

struct FullPair {
    FullDistribution selected;
    FullDistribution complement;
};

struct CensusRecord {
    uint32_t left;
    uint32_t right;
};

struct Dataset {
    std::string path;
    std::vector<FullPair> pairs;
    std::vector<CensusRecord> records;
    uint64_t entries = 0;
};

struct ClassBucket {
    uint16_t prefix = 0;
    uint32_t count = 0;
    std::vector<uint32_t> classes;
};

struct CandidateDistribution {
    std::vector<ClassBucket> buckets;
};

struct CandidatePair {
    CandidateDistribution selected;
    CandidateDistribution complement;
};

struct Cost {
    U128 tiles = 0;
    U128 suffix_cells = 0;
    U128 class_pairs = 0;
    uint64_t compatible_bucket_pairs = 0;
    uint64_t tested_bucket_pairs = 0;
    uint64_t stored_buckets = 0;
    uint64_t stored_classes = 0;
    U128 adaptive_prefix_probes = 0;
    U128 adaptive_task_chunks = 0;
    U128 cartesian_task_chunks = 0;
    uint64_t enumerated_left_buckets = 0;
    uint64_t visited_left_buckets = 0;
};

static long double u128_long_double(U128 value) {
    return static_cast<long double>(value);
}

static FullDistribution make_full_distribution(
    std::vector<FullWeightedEntry> source) {
    FullDistribution result;
    result.weights.reserve(source.size());
    for (const auto& entry : source) result.weights.push_back(entry.weight);
    std::sort(result.weights.begin(), result.weights.end());
    result.weights.erase(std::unique(result.weights.begin(), result.weights.end()),
                         result.weights.end());
    if (result.weights.size() > 32)
        throw std::runtime_error("full distribution exceeds 32 weights");
    result.entries.reserve(source.size());
    for (const auto& entry : source) {
        auto found = std::lower_bound(result.weights.begin(), result.weights.end(),
                                      entry.weight);
        result.entries.push_back(CensusEntry{
            entry.mask, uint8_t(found - result.weights.begin())});
    }
    return result;
}

static Dataset build_dataset(const std::string& path, uint64_t sample_records) {
    Dataset result;
    result.path = path;
    std::vector<SampleRecord> records =
        read_stride_sample(path, sample_records);
    std::vector<uint32_t> prefixes;
    for (const auto& record : records) {
        prefixes.push_back(half_prefix(record.key, 0));
        prefixes.push_back(half_prefix(record.key, HALF_COLUMNS));
    }
    std::sort(prefixes.begin(), prefixes.end());
    prefixes.erase(std::unique(prefixes.begin(), prefixes.end()), prefixes.end());
    result.pairs.resize(prefixes.size());
#pragma omp parallel for schedule(dynamic, 1)
    for (long long index = 0; index < (long long)prefixes.size(); index++) {
        uint32_t prefix = prefixes[size_t(index)];
        result.pairs[size_t(index)] = FullPair{
            make_full_distribution(
                build_full_weighted_distribution(prefix, false)),
            make_full_distribution(
                build_full_weighted_distribution(prefix, true))};
    }
    std::unordered_map<uint32_t, uint32_t> prefix_index;
    for (size_t index = 0; index < prefixes.size(); index++)
        prefix_index.emplace(prefixes[index], uint32_t(index));
    for (const auto& record : records) {
        result.records.push_back(CensusRecord{
            prefix_index.at(half_prefix(record.key, 0)),
            prefix_index.at(half_prefix(record.key, HALF_COLUMNS))});
    }
    for (const auto& pair : result.pairs)
        result.entries += pair.selected.entries.size() +
                          pair.complement.entries.size();
    return result;
}

static CandidateDistribution bucket_distribution(
    const FullDistribution& source, uint32_t pair_mask) {
    if (__builtin_popcount(pair_mask) > 8)
        throw std::runtime_error("census supports at most eight pair coordinates");
    uint64_t expanded = uint64_t(pair_mask) |
                        (uint64_t(pair_mask) << PAIRS);
    struct TemporaryBucket {
        uint16_t prefix = 0;
        std::array<uint32_t, 32> counts{};
    };
    std::unordered_map<uint16_t, uint32_t> bucket_index;
    bucket_index.reserve(std::min<size_t>(
        source.entries.size(),
        size_t(1) << (2 * __builtin_popcount(pair_mask))));
    std::vector<TemporaryBucket> temporary;
    for (const auto& entry : source.entries) {
        uint16_t prefix = uint16_t(_pext_u64(entry.mask, expanded));
        auto inserted = bucket_index.emplace(prefix, uint32_t(temporary.size()));
        if (inserted.second) temporary.push_back(TemporaryBucket{prefix, {}});
        temporary[inserted.first->second].counts[entry.weight_ordinal]++;
    }
    CandidateDistribution result;
    result.buckets.reserve(temporary.size());
    for (const auto& bucket : temporary) {
        ClassBucket output;
        output.prefix = bucket.prefix;
        for (size_t ordinal = 0; ordinal < source.weights.size(); ordinal++) {
            uint32_t count = bucket.counts[ordinal];
            if (!count) continue;
            output.count += count;
            output.classes.push_back(count);
        }
        result.buckets.push_back(std::move(output));
    }
    std::sort(result.buckets.begin(), result.buckets.end(),
              [](const ClassBucket& a, const ClassBucket& b) {
                  return a.prefix < b.prefix;
              });
    return result;
}

static void add_cost(Cost& destination, const Cost& source) {
    destination.tiles += source.tiles;
    destination.suffix_cells += source.suffix_cells;
    destination.class_pairs += source.class_pairs;
    destination.compatible_bucket_pairs += source.compatible_bucket_pairs;
    destination.tested_bucket_pairs += source.tested_bucket_pairs;
    destination.stored_buckets += source.stored_buckets;
    destination.stored_classes += source.stored_classes;
    destination.adaptive_prefix_probes += source.adaptive_prefix_probes;
    destination.adaptive_task_chunks += source.adaptive_task_chunks;
    destination.cartesian_task_chunks += source.cartesian_task_chunks;
    destination.enumerated_left_buckets += source.enumerated_left_buckets;
    destination.visited_left_buckets += source.visited_left_buckets;
}

class CostEvaluator {
  public:
    explicit CostEvaluator(std::vector<Dataset> datasets)
        : datasets_(std::move(datasets)), start_(seconds_now()) {}

    const Cost& evaluate(uint32_t pair_mask) {
        auto found = cache_.find(pair_mask);
        if (found != cache_.end()) return found->second;
        Cost total;
        for (const auto& dataset : datasets_)
            add_cost(total, evaluate_dataset(dataset, pair_mask));
        auto inserted = cache_.emplace(pair_mask, total);
        if (!(cache_.size() % 100)) {
            std::fprintf(stderr,
                         "CENSUS_PROGRESS candidates=%zu seconds=%.3f\n",
                         cache_.size(), seconds_now() - start_);
        }
        return inserted.first->second;
    }

    size_t candidates() const { return cache_.size(); }

  private:
    static Cost evaluate_dataset(const Dataset& dataset, uint32_t pair_mask) {
        std::vector<CandidatePair> layouts(dataset.pairs.size());
#pragma omp parallel for schedule(dynamic, 1)
        for (long long index = 0; index < (long long)dataset.pairs.size(); index++) {
            layouts[size_t(index)] = CandidatePair{
                bucket_distribution(dataset.pairs[size_t(index)].selected,
                                    pair_mask),
                bucket_distribution(dataset.pairs[size_t(index)].complement,
                                    pair_mask)};
        }
        Cost result;
        for (const auto& pair : layouts) {
            for (const CandidateDistribution* distribution :
                 {&pair.selected, &pair.complement}) {
                result.stored_buckets += distribution->buckets.size();
                for (const auto& bucket : distribution->buckets)
                    result.stored_classes += bucket.classes.size();
            }
        }
        std::vector<Cost> record_costs(dataset.records.size());
#pragma omp parallel for schedule(dynamic, 1)
        for (long long record_index = 0;
             record_index < (long long)dataset.records.size(); record_index++) {
            const auto& record = dataset.records[size_t(record_index)];
            Cost local;
            for (unsigned component_index = 0; component_index < 2;
                 component_index++) {
                const CandidateDistribution& left = component_index
                    ? layouts[record.left].complement
                    : layouts[record.left].selected;
                const CandidateDistribution& right = component_index
                    ? layouts[record.right].complement
                    : layouts[record.right].selected;
                local.tested_bucket_pairs +=
                    uint64_t(left.buckets.size()) * right.buckets.size();
                uint64_t cartesian_tasks =
                    uint64_t(left.buckets.size()) * right.buckets.size();
                local.cartesian_task_chunks +=
                    (cartesian_tasks + CENSUS_TASK_CHUNK - 1) /
                    CENSUS_TASK_CHUNK;
                for (const auto& a : left.buckets) {
                    unsigned prefix_bits =
                        2U * unsigned(__builtin_popcount(pair_mask));
                    uint32_t compatible_values = uint32_t(1) <<
                        (prefix_bits - unsigned(__builtin_popcount(a.prefix)));
                    uint32_t probes = std::min<uint32_t>(
                        uint32_t(right.buckets.size()), compatible_values);
                    local.adaptive_prefix_probes += probes;
                    local.adaptive_task_chunks +=
                        (probes + CENSUS_TASK_CHUNK - 1) /
                        CENSUS_TASK_CHUNK;
                    local.enumerated_left_buckets +=
                        compatible_values < right.buckets.size();
                    local.visited_left_buckets++;
                    for (const auto& b : right.buckets) {
                        if (a.prefix & b.prefix) continue;
                        local.compatible_bucket_pairs++;
                        local.suffix_cells += U128(a.count) * b.count;
                        local.class_pairs +=
                            U128(a.classes.size()) * b.classes.size();
                        for (uint32_t ac : a.classes)
                            for (uint32_t bc : b.classes)
                                local.tiles += predicate_bmma_tiles(ac, bc);
                    }
                }
            }
            record_costs[size_t(record_index)] = local;
        }
        for (const auto& local : record_costs) add_cost(result, local);
        return result;
    }

    std::vector<Dataset> datasets_;
    std::unordered_map<uint32_t, Cost> cache_;
    double start_;
};

static uint32_t production_mask(unsigned pair_count) {
    uint32_t result = 0;
    for (unsigned rank = 0; rank < pair_count; rank++)
        result |= uint32_t(1) << production_pair_order[rank];
    return result;
}

static std::string edge_string(uint32_t mask) {
    std::ostringstream output;
    bool comma = false;
    for (unsigned first = 0; first < ROWS; first++)
        for (unsigned second = first + 1; second < ROWS; second++) {
            unsigned pair = unsigned(g_pair_index[first][second]);
            if (!(mask & (uint32_t(1) << pair))) continue;
            if (comma) output << ',';
            output << first << second;
            comma = true;
        }
    return output.str();
}

static bool better(const Cost& a, const Cost& b) {
    if (a.tiles != b.tiles) return a.tiles < b.tiles;
    if (a.class_pairs != b.class_pairs) return a.class_pairs < b.class_pairs;
    if (a.tested_bucket_pairs != b.tested_bucket_pairs)
        return a.tested_bucket_pairs < b.tested_bucket_pairs;
    return a.stored_buckets < b.stored_buckets;
}

static void print_cost(const char* kind, unsigned width, uint32_t mask,
                       const Cost& cost, const Cost& baseline) {
    long double tile_ratio = baseline.tiles
        ? u128_long_double(cost.tiles) / u128_long_double(baseline.tiles) : 0;
    long double fill = cost.tiles
        ? u128_long_double(cost.suffix_cells) /
              (128.0L * u128_long_double(cost.tiles)) : 0;
    long double probe_ratio = cost.tested_bucket_pairs
        ? u128_long_double(cost.adaptive_prefix_probes) /
              cost.tested_bucket_pairs : 0;
    long double chunk_ratio = cost.cartesian_task_chunks
        ? u128_long_double(cost.adaptive_task_chunks) /
              u128_long_double(cost.cartesian_task_chunks) : 0;
    long double enumeration_fraction = cost.visited_left_buckets
        ? static_cast<long double>(cost.enumerated_left_buckets) /
              cost.visited_left_buckets : 0;
    std::cout << std::setprecision(12)
              << kind << " pair_coordinates=" << width
              << " prefix_bits=" << 2 * width
              << " mask=0x" << std::hex << mask << std::dec
              << " edges=" << edge_string(mask)
              << " tiles=" << u128_string(cost.tiles)
              << " production_tile_ratio=" << double(tile_ratio)
              << " suffix_cells=" << u128_string(cost.suffix_cells)
              << " class_pairs=" << u128_string(cost.class_pairs)
              << " fill=" << double(fill)
              << " compatible_bucket_pairs=" << cost.compatible_bucket_pairs
              << " tested_bucket_pairs=" << cost.tested_bucket_pairs
              << " stored_buckets=" << cost.stored_buckets
              << " stored_classes=" << cost.stored_classes
              << " adaptive_prefix_probes="
              << u128_string(cost.adaptive_prefix_probes)
              << " adaptive_probe_ratio=" << double(probe_ratio)
              << " adaptive_task_chunks="
              << u128_string(cost.adaptive_task_chunks)
              << " cartesian_task_chunks="
              << u128_string(cost.cartesian_task_chunks)
              << " adaptive_chunk_ratio=" << double(chunk_ratio)
              << " enumerated_left_fraction="
              << double(enumeration_fraction) << '\n';
}

static std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> result;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, delimiter))
        if (!item.empty()) result.push_back(item);
    return result;
}

}  // namespace

#ifndef PREFIX_BMMA_COST_CENSUS_NO_MAIN
int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 5) {
            std::cerr << "usage: " << argv[0]
                      << " SHARD[,SHARD...] [SAMPLE_RECORDS_PER_SHARD=4]"
                         " [LOCAL_ROUNDS=2] [MASK[,MASK...]]\n";
            return 2;
        }
        uint64_t sample_records = argc > 2 ? std::stoull(argv[2]) : 4;
        unsigned local_rounds = argc > 3 ? unsigned(std::stoul(argv[3])) : 2;
        if (!sample_records) return 2;
        double start = seconds_now();
        initialise_tables();
        initialise_weighted_increments();
        std::vector<Dataset> datasets;
        for (const auto& path : split(argv[1], ',')) {
            double build_start = seconds_now();
            datasets.push_back(build_dataset(path, sample_records));
            const auto& dataset = datasets.back();
            std::cout << "CENSUS_INPUT path=" << path
                      << " records=" << dataset.records.size()
                      << " distributions=" << dataset.pairs.size() * 2
                      << " entries=" << dataset.entries
                      << " build_seconds=" << seconds_now() - build_start
                      << '\n';
        }
        CostEvaluator evaluator(std::move(datasets));
        if (argc > 4) {
            std::unordered_set<unsigned> printed_baselines;
            for (const auto& item : split(argv[4], ',')) {
                uint32_t mask = uint32_t(std::stoul(item, nullptr, 0));
                unsigned width = unsigned(__builtin_popcount(mask));
                const Cost& baseline = evaluator.evaluate(production_mask(width));
                if (printed_baselines.insert(width).second)
                    print_cost("CENSUS_BASELINE", width,
                               production_mask(width), baseline, baseline);
                print_cost("CENSUS_EVAL", width, mask,
                           evaluator.evaluate(mask), baseline);
            }
            std::cout << "CENSUS_SUMMARY candidates=" << evaluator.candidates()
                      << " seconds=" << seconds_now() - start << '\n';
            return 0;
        }

        std::array<uint32_t, 9> greedy{};
        uint32_t growing = 0;
        for (unsigned width = 1; width <= 8; width++) {
            uint32_t best_mask = 0;
            const Cost* best_cost = nullptr;
            for (unsigned pair = 0; pair < PAIRS; pair++) {
                if (growing & (uint32_t(1) << pair)) continue;
                uint32_t candidate = growing | (uint32_t(1) << pair);
                const Cost& cost = evaluator.evaluate(candidate);
                if (!best_cost || better(cost, *best_cost)) {
                    best_mask = candidate;
                    best_cost = &cost;
                }
            }
            growing = best_mask;
            greedy[width] = growing;
        }

        for (unsigned width = 4; width <= 8; width++) {
            uint32_t baseline_mask = production_mask(width);
            const Cost& baseline = evaluator.evaluate(baseline_mask);
            print_cost("CENSUS_PRODUCTION", width, baseline_mask, baseline,
                       baseline);
            uint32_t current = greedy[width];
            if (better(evaluator.evaluate(baseline_mask),
                       evaluator.evaluate(current)))
                current = baseline_mask;
            for (unsigned round = 0; round < local_rounds; round++) {
                uint32_t best_mask = current;
                const Cost* best_cost = &evaluator.evaluate(current);
                for (unsigned remove = 0; remove < PAIRS; remove++) {
                    if (!(current & (uint32_t(1) << remove))) continue;
                    for (unsigned add = 0; add < PAIRS; add++) {
                        if (current & (uint32_t(1) << add)) continue;
                        uint32_t candidate =
                            (current ^ (uint32_t(1) << remove)) |
                            (uint32_t(1) << add);
                        const Cost& cost = evaluator.evaluate(candidate);
                        if (better(cost, *best_cost)) {
                            best_mask = candidate;
                            best_cost = &cost;
                        }
                    }
                }
                if (best_mask == current) break;
                current = best_mask;
            }
            print_cost("CENSUS_BEST", width, current,
                       evaluator.evaluate(current), baseline);
        }
        std::cout << "CENSUS_SUMMARY candidates=" << evaluator.candidates()
                  << " seconds=" << seconds_now() - start
                  << " weighted_support=EXACT search=greedy-plus-swaps\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
#endif
