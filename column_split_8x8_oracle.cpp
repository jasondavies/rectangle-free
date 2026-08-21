#include <iomanip>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#define PREFIX_BMMA_COST_CENSUS_NO_MAIN
#include "prefix_bmma_cost_census.cpp"

namespace {

constexpr unsigned PREFIX_COORDINATES = 7;
constexpr unsigned PREFIX_BITS = 2 * PREFIX_COORDINATES;
constexpr uint16_t PREFIX_FULL = (uint16_t(1) << PREFIX_BITS) - 1;
// Experiment 345's two-point L40S calibration assigned one tested physical
// bucket pair the cost of 3.247 BMMA tiles.  Treat 3.25 as a ranking aid only:
// the earlier row-gauge gate proved that a direct GPU A/B remains mandatory.
constexpr long double CALIBRATED_BUCKET_TILE_EQUIVALENT = 3.25L;

struct QuotientClass {
    uint32_t count = 0;
    uint8_t orbit_size = 0;
};

struct QuotientBucket {
    uint16_t prefix = 0;
    uint32_t class_offset = 0;
    uint16_t class_count = 0;
};

struct QuotientDistribution {
    std::vector<QuotientBucket> buckets;
    std::vector<QuotientClass> classes;
    std::vector<uint16_t> dense;
    uint64_t entries = 0;
};

struct QuotientPair {
    QuotientDistribution selected;
    QuotientDistribution complement;
};

struct SplitCandidate {
    bool transpose = false;
    uint8_t columns = 0;
};

struct SplitKeys {
    uint32_t first = 0;
    uint32_t second = 0;
};

struct SplitCost {
    U128 tiles = 0;
    U128 suffix_cells = 0;
    U128 class_calls = 0;
    uint64_t tested_bucket_pairs = 0;
    uint64_t compatible_orientations = 0;
};

struct RecordCost {
    SplitCost selected;
    SplitCost complement;
};

struct DatasetCuts {
    std::string path;
    std::vector<SampleRecord> records;
    // Candidate-major arrays, each containing one entry per sampled record.
    std::vector<std::vector<SplitKeys>> keys;
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

static uint64_t transpose_grid(uint64_t key) {
    uint64_t result = 0;
    for (unsigned row = 0; row < ROWS; row++) {
        for (unsigned column = 0; column < COLUMNS; column++) {
            unsigned source = COLUMNS * (ROWS - 1 - row) + column;
            unsigned destination = COLUMNS * (ROWS - 1 - column) + row;
            result |= ((key >> source) & 1U) << destination;
        }
    }
    return result;
}

// Column order is immaterial to a half-distribution.  Sort its four labelled
// column masks before rebuilding the row-major 8x4 key; this cheaply recovers
// all column-permutation reuse without changing the shared row gauge.
static uint32_t extract_sorted_half(uint64_t key, uint8_t columns) {
    if (__builtin_popcount(unsigned(columns)) != HALF_COLUMNS)
        throw std::logic_error("column split is not 4+4");
    std::array<uint8_t, HALF_COLUMNS> column_masks{};
    unsigned output_column = 0;
    for (unsigned column = 0; column < COLUMNS; column++) {
        if (!(columns & (uint8_t(1) << column))) continue;
        uint8_t column_mask = 0;
        for (unsigned row = 0; row < ROWS; row++) {
            unsigned bit = COLUMNS * (ROWS - 1 - row) + column;
            column_mask |= uint8_t((key >> bit) & 1U) << row;
        }
        column_masks[output_column++] = column_mask;
    }
    std::sort(column_masks.begin(), column_masks.end());
    uint32_t result = 0;
    for (unsigned row = 0; row < ROWS; row++) {
        uint32_t pattern = 0;
        for (unsigned column = 0; column < HALF_COLUMNS; column++)
            pattern |= uint32_t((column_masks[column] >> row) & 1U) << column;
        result = (result << HALF_COLUMNS) | pattern;
    }
    return result;
}

static std::vector<SplitCandidate> enumerate_cuts() {
    std::vector<SplitCandidate> result;
    for (unsigned transpose = 0; transpose < 2; transpose++) {
        for (unsigned columns = 0; columns < (1U << COLUMNS); columns++) {
            // P and its complement define the same mathematical cut.  Keep
            // the representative containing labelled column zero.
            if (!(columns & 1U) || __builtin_popcount(columns) != HALF_COLUMNS)
                continue;
            result.push_back(
                SplitCandidate{bool(transpose), uint8_t(columns)});
        }
    }
    if (result.size() != 70)
        throw std::logic_error("8x8 cut enumeration is incomplete");
    return result;
}

static QuotientDistribution build_quotient_distribution(uint32_t prefix,
                                                         bool complement) {
    struct Item {
        uint16_t prefix;
        uint32_t weight;
        uint8_t orbit_size;
    };
    std::vector<FullWeightedEntry> full =
        build_full_weighted_distribution(prefix, complement);
    std::vector<Item> items;
    items.reserve((full.size() + 1) / 2);
    const uint32_t pair_mask = production_mask(PREFIX_COORDINATES);
    uint64_t expanded_entries = 0;
    U128 full_weight = 0, expanded_weight = 0;
    for (const FullWeightedEntry& entry : full) {
        full_weight += entry.weight;
        uint64_t swapped = swap_token_planes_local(entry.mask);
        if (entry.mask > swapped) continue;
        uint16_t bucket_prefix = 0;
        uint64_t suffix = 0;
        split_pair_mask(entry.mask, pair_mask, bucket_prefix, suffix);
        uint8_t orbit_size = uint8_t(entry.mask == swapped ? 1 : 2);
        items.push_back(Item{bucket_prefix, entry.weight, orbit_size});
        expanded_entries += orbit_size;
        expanded_weight += U128(orbit_size) * entry.weight;
    }
    if (expanded_entries != full.size() || expanded_weight != full_weight)
        throw std::runtime_error("token-plane quotient invariant failed");
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        if (a.prefix != b.prefix) return a.prefix < b.prefix;
        if (a.weight != b.weight) return a.weight < b.weight;
        return a.orbit_size < b.orbit_size;
    });

    QuotientDistribution result;
    result.entries = items.size();
    result.dense.assign(size_t(1) << PREFIX_BITS, UINT16_MAX);
    for (size_t begin = 0; begin < items.size();) {
        size_t bucket_end = begin + 1;
        while (bucket_end < items.size() &&
               items[bucket_end].prefix == items[begin].prefix)
            bucket_end++;
        if (result.buckets.size() >= UINT16_MAX)
            throw std::overflow_error("too many quotient prefix buckets");
        QuotientBucket bucket{items[begin].prefix,
                              uint32_t(result.classes.size()), 0};
        for (size_t class_begin = begin; class_begin < bucket_end;) {
            size_t class_end = class_begin + 1;
            while (class_end < bucket_end &&
                   items[class_end].weight == items[class_begin].weight &&
                   items[class_end].orbit_size ==
                       items[class_begin].orbit_size)
                class_end++;
            result.classes.push_back(QuotientClass{
                uint32_t(class_end - class_begin),
                items[class_begin].orbit_size});
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

static QuotientPair build_quotient_pair(uint32_t prefix) {
    return QuotientPair{build_quotient_distribution(prefix, false),
                        build_quotient_distribution(prefix, true)};
}

static void add_cost(SplitCost& target, const SplitCost& source) {
    target.tiles += source.tiles;
    target.suffix_cells += source.suffix_cells;
    target.class_calls += source.class_calls;
    target.tested_bucket_pairs += source.tested_bucket_pairs;
    target.compatible_orientations += source.compatible_orientations;
}

static SplitCost distribution_cost(const QuotientDistribution& left,
                                   const QuotientDistribution& right) {
    SplitCost result;
    result.tested_bucket_pairs =
        uint64_t(left.buckets.size()) * right.buckets.size();
    auto visit = [&](const QuotientBucket& lhs, bool swapped) {
        uint16_t query = swapped ? swap_prefix_planes(lhs.prefix) : lhs.prefix;
        uint16_t allowed = uint16_t(PREFIX_FULL ^ query);
        uint32_t submask_count = uint32_t(1) << __builtin_popcount(allowed);
        auto accumulate = [&](const QuotientBucket& rhs) {
            result.compatible_orientations++;
            for (uint32_t li = 0; li < lhs.class_count; li++) {
                const QuotientClass& a =
                    left.classes[lhs.class_offset + li];
                for (uint32_t ri = 0; ri < rhs.class_count; ri++) {
                    const QuotientClass& b =
                        right.classes[rhs.class_offset + ri];
                    if (swapped && b.orbit_size != 2) continue;
                    result.class_calls++;
                    result.suffix_cells += U128(a.count) * b.count;
                    result.tiles += predicate_bmma_tiles(a.count, b.count);
                }
            }
        };
        if (submask_count < right.buckets.size()) {
            uint16_t value = allowed;
            for (;;) {
                uint16_t ordinal = right.dense[value];
                if (ordinal != UINT16_MAX)
                    accumulate(right.buckets[ordinal]);
                if (!value) break;
                value = uint16_t((value - 1) & allowed);
            }
        } else {
            for (const QuotientBucket& rhs : right.buckets)
                if (!(query & rhs.prefix)) accumulate(rhs);
        }
    };
    for (const QuotientBucket& lhs : left.buckets) {
        visit(lhs, false);
        visit(lhs, true);
    }
    return result;
}

static long double score(const SplitCost& cost,
                         long double bucket_equivalent) {
    return u128_long_double(cost.tiles) +
           bucket_equivalent * cost.tested_bucket_pairs;
}

static SplitCost combined(const RecordCost& cost) {
    SplitCost result = cost.selected;
    add_cost(result, cost.complement);
    return result;
}

static std::string candidate_name(const SplitCandidate& candidate) {
    std::ostringstream output;
    output << (candidate.transpose ? 'H' : 'V') << ":0x"
           << std::hex << unsigned(candidate.columns);
    return output.str();
}

static DatasetCuts prepare_dataset(const std::string& path,
                                   uint64_t sample_records,
                                   const std::vector<SplitCandidate>& cuts,
                                   std::vector<uint32_t>& prefixes) {
    DatasetCuts result;
    result.path = path;
    result.records = read_stride_sample(path, sample_records);
    result.keys.assign(cuts.size(),
                       std::vector<SplitKeys>(result.records.size()));
    for (size_t candidate = 0; candidate < cuts.size(); candidate++) {
        for (size_t record = 0; record < result.records.size(); record++) {
            uint64_t key = result.records[record].key;
            if (cuts[candidate].transpose) key = transpose_grid(key);
            uint8_t first = cuts[candidate].columns;
            uint8_t second = uint8_t(~first);
            SplitKeys split{extract_sorted_half(key, first),
                            extract_sorted_half(key, second)};
            result.keys[candidate][record] = split;
            prefixes.push_back(split.first);
            prefixes.push_back(split.second);
        }
    }
    return result;
}

struct StrategyStats {
    SplitCost cost;
    std::unordered_set<uint32_t> left_keys;
    std::unordered_set<uint32_t> right_keys;
};

static uint64_t layout_entries(
    const std::unordered_set<uint32_t>& keys,
    const std::unordered_map<uint32_t, uint32_t>& prefix_ids,
    const std::vector<QuotientPair>& pairs) {
    uint64_t result = 0;
    for (uint32_t key : keys) {
        const QuotientPair& pair = pairs[prefix_ids.at(key)];
        result += pair.selected.entries + pair.complement.entries;
    }
    return result;
}

static void print_strategy(
    const char* dataset, const char* kind, const StrategyStats& strategy,
    const SplitCost& baseline,
    const std::unordered_map<uint32_t, uint32_t>& prefix_ids,
    const std::vector<QuotientPair>& pairs) {
    long double baseline_score =
        score(baseline, CALIBRATED_BUCKET_TILE_EQUIVALENT);
    long double strategy_score =
        score(strategy.cost, CALIBRATED_BUCKET_TILE_EQUIVALENT);
    std::cout << std::setprecision(12)
              << "COLUMN_SPLIT_STRATEGY dataset=" << dataset
              << " kind=" << kind
              << " score_ratio=" << double(strategy_score / baseline_score)
              << " tile_ratio="
              << double(u128_long_double(strategy.cost.tiles) /
                        u128_long_double(baseline.tiles))
              << " tested_bucket_ratio="
              << double(static_cast<long double>(
                            strategy.cost.tested_bucket_pairs) /
                        baseline.tested_bucket_pairs)
              << " suffix_cell_ratio="
              << double(u128_long_double(strategy.cost.suffix_cells) /
                        u128_long_double(baseline.suffix_cells))
              << " left_keys=" << strategy.left_keys.size()
              << " right_keys=" << strategy.right_keys.size()
              << " left_entries="
              << layout_entries(strategy.left_keys, prefix_ids, pairs)
              << " right_entries="
              << layout_entries(strategy.right_keys, prefix_ids, pairs)
              << '\n';
}

static long double support_product_proxy(const QuotientPair& left,
                                         const QuotientPair& right,
                                         bool complement) {
    const QuotientDistribution& a =
        complement ? left.complement : left.selected;
    const QuotientDistribution& b =
        complement ? right.complement : right.selected;
    return static_cast<long double>(a.entries) * b.entries;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 4) {
            std::cerr << "usage: " << argv[0]
                      << " TRAIN.orbits[,TRAIN.orbits] TEST.orbits "
                         "[SAMPLE_RECORDS_PER_SHARD=32]\n";
            return 2;
        }
        uint64_t sample_records = argc > 3 ? std::stoull(argv[3]) : 32;
        if (!sample_records) return 2;
        initialise_tables();
        initialise_weighted_increments();
        const std::vector<SplitCandidate> cuts = enumerate_cuts();
        std::vector<std::string> paths = split(argv[1], ',');
        size_t training_datasets = paths.size();
        std::vector<std::string> test_paths = split(argv[2], ',');
        paths.insert(paths.end(), test_paths.begin(), test_paths.end());
        if (!training_datasets || test_paths.empty()) return 2;

        double start = seconds_now();
        std::vector<uint32_t> prefixes;
        std::vector<DatasetCuts> datasets;
        for (const std::string& path : paths) {
            datasets.push_back(
                prepare_dataset(path, sample_records, cuts, prefixes));
            std::cout << "COLUMN_SPLIT_INPUT path=" << path
                      << " records=" << datasets.back().records.size()
                      << " cuts=" << cuts.size() << '\n';
        }
        std::sort(prefixes.begin(), prefixes.end());
        prefixes.erase(std::unique(prefixes.begin(), prefixes.end()),
                       prefixes.end());
        std::unordered_map<uint32_t, uint32_t> prefix_ids;
        prefix_ids.reserve(prefixes.size() * 2);
        for (size_t index = 0; index < prefixes.size(); index++)
            prefix_ids.emplace(prefixes[index], uint32_t(index));

        std::vector<QuotientPair> pairs(prefixes.size());
        double distribution_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
        for (long long index = 0; index < (long long)prefixes.size(); index++)
            pairs[size_t(index)] = build_quotient_pair(prefixes[size_t(index)]);
        uint64_t cache_entries = 0, cache_buckets = 0, cache_classes = 0;
        for (const QuotientPair& pair : pairs) {
            for (const QuotientDistribution* distribution :
                 {&pair.selected, &pair.complement}) {
                cache_entries += distribution->entries;
                cache_buckets += distribution->buckets.size();
                cache_classes += distribution->classes.size();
            }
        }
        std::cout << "COLUMN_SPLIT_CACHE prefixes=" << prefixes.size()
                  << " entries=" << cache_entries
                  << " buckets=" << cache_buckets
                  << " classes=" << cache_classes
                  << " seconds=" << seconds_now() - distribution_start
                  << " token_plane_quotient=1\n";

        std::vector<std::vector<RecordCost>> costs(datasets.size());
        double cost_start = seconds_now();
        for (size_t dataset = 0; dataset < datasets.size(); dataset++) {
            size_t count = datasets[dataset].records.size();
            costs[dataset].resize(cuts.size() * count);
#pragma omp parallel for schedule(dynamic, 1)
            for (long long flat = 0;
                 flat < (long long)(cuts.size() * count); flat++) {
                size_t candidate = size_t(flat) / count;
                size_t record = size_t(flat) % count;
                const SplitKeys& keys = datasets[dataset].keys[candidate][record];
                const QuotientPair& left = pairs[prefix_ids.at(keys.first)];
                const QuotientPair& right = pairs[prefix_ids.at(keys.second)];
                costs[dataset][size_t(flat)] = RecordCost{
                    distribution_cost(left.selected, right.selected),
                    distribution_cost(left.complement, right.complement)};
            }
            std::cerr << "COLUMN_SPLIT_PROGRESS dataset=" << dataset
                      << " seconds=" << seconds_now() - cost_start << '\n';
        }

        auto aggregate_candidate = [&](size_t begin_dataset,
                                       size_t end_dataset,
                                       size_t candidate) {
            SplitCost result;
            for (size_t dataset = begin_dataset; dataset < end_dataset;
                 dataset++) {
                size_t count = datasets[dataset].records.size();
                for (size_t record = 0; record < count; record++)
                    add_cost(result,
                             combined(costs[dataset][candidate * count + record]));
            }
            return result;
        };
        size_t baseline_candidate = cuts.size();
        for (size_t candidate = 0; candidate < cuts.size(); candidate++)
            if (!cuts[candidate].transpose && cuts[candidate].columns == 0x0f)
                baseline_candidate = candidate;
        if (baseline_candidate == cuts.size())
            throw std::logic_error("production split is absent");

        std::vector<SplitCost> train_costs(cuts.size());
        std::vector<SplitCost> test_costs(cuts.size());
        for (size_t candidate = 0; candidate < cuts.size(); candidate++) {
            train_costs[candidate] =
                aggregate_candidate(0, training_datasets, candidate);
            test_costs[candidate] = aggregate_candidate(
                training_datasets, datasets.size(), candidate);
        }
        const SplitCost& train_baseline = train_costs[baseline_candidate];
        const SplitCost& test_baseline = test_costs[baseline_candidate];
        std::vector<size_t> order(cuts.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return score(train_costs[a], CALIBRATED_BUCKET_TILE_EQUIVALENT) <
                   score(train_costs[b], CALIBRATED_BUCKET_TILE_EQUIVALENT);
        });
        std::cout << "COLUMN_SPLIT_CALIBRATION bucket_tile_equivalent="
                  << double(CALIBRATED_BUCKET_TILE_EQUIVALENT)
                  << " note=token-quotient-adjusted-from-experiment-345\n";
        for (size_t rank = 0; rank < std::min<size_t>(10, order.size()); rank++) {
            size_t candidate = order[rank];
            const SplitCost& train = train_costs[candidate];
            const SplitCost& test = test_costs[candidate];
            std::cout << std::setprecision(12)
                      << "COLUMN_SPLIT_GLOBAL rank=" << rank + 1
                      << " candidate=" << candidate_name(cuts[candidate])
                      << " train_score_ratio="
                      << double(score(train, CALIBRATED_BUCKET_TILE_EQUIVALENT) /
                                score(train_baseline,
                                      CALIBRATED_BUCKET_TILE_EQUIVALENT))
                      << " test_score_ratio="
                      << double(score(test, CALIBRATED_BUCKET_TILE_EQUIVALENT) /
                                score(test_baseline,
                                      CALIBRATED_BUCKET_TILE_EQUIVALENT))
                      << " test_tile_ratio="
                      << double(u128_long_double(test.tiles) /
                                u128_long_double(test_baseline.tiles))
                      << " test_bucket_ratio="
                      << double(static_cast<long double>(
                                    test.tested_bucket_pairs) /
                                test_baseline.tested_bucket_pairs)
                      << '\n';
        }

        // Report fixed baseline/best strategies and impossible per-record
        // coupled/independent upper bounds on the held-out dataset.  For each
        // fixed cut, choose the execution direction with fewer sampled left
        // layout entries; arithmetic is symmetric under this choice.
        auto fixed_strategy = [&](size_t candidate) {
            StrategyStats forward, reverse;
            for (size_t dataset = training_datasets;
                 dataset < datasets.size(); dataset++) {
                size_t count = datasets[dataset].records.size();
                for (size_t record = 0; record < count; record++) {
                    add_cost(forward.cost,
                             combined(costs[dataset][candidate * count + record]));
                    const SplitKeys& keys =
                        datasets[dataset].keys[candidate][record];
                    forward.left_keys.insert(keys.first);
                    forward.right_keys.insert(keys.second);
                    reverse.left_keys.insert(keys.second);
                    reverse.right_keys.insert(keys.first);
                }
            }
            reverse.cost = forward.cost;
            return layout_entries(forward.left_keys, prefix_ids, pairs) <=
                           layout_entries(reverse.left_keys, prefix_ids, pairs)
                       ? forward : reverse;
        };
        print_strategy("test", "baseline", fixed_strategy(baseline_candidate),
                       test_baseline, prefix_ids, pairs);
        print_strategy("test", "best-global", fixed_strategy(order.front()),
                       test_baseline, prefix_ids, pairs);

        StrategyStats coupled_oracle;
        StrategyStats independent_oracle;
        StrategyStats two_cut_oracle;
        StrategyStats top_four_oracle;
        StrategyStats support_proxy;
        size_t horizontal_baseline = cuts.size();
        for (size_t candidate = 0; candidate < cuts.size(); candidate++)
            if (cuts[candidate].transpose && cuts[candidate].columns == 0x0f)
                horizontal_baseline = candidate;
        if (horizontal_baseline == cuts.size())
            throw std::logic_error("horizontal baseline split is absent");
        std::array<size_t, 2> two_cuts{baseline_candidate,
                                       horizontal_baseline};
        for (size_t test_dataset = training_datasets;
             test_dataset < datasets.size(); test_dataset++) {
          size_t count = datasets[test_dataset].records.size();
          for (size_t record = 0; record < count; record++) {
            size_t coupled = baseline_candidate;
            size_t selected = baseline_candidate;
            size_t complement = baseline_candidate;
            size_t two_cut = baseline_candidate;
            size_t top_four = order.front();
            size_t proxy = baseline_candidate;
            long double proxy_score =
                std::numeric_limits<long double>::infinity();
            for (size_t candidate = 0; candidate < cuts.size(); candidate++) {
                const RecordCost& trial =
                    costs[test_dataset][candidate * count + record];
                const RecordCost& coupled_best =
                    costs[test_dataset][coupled * count + record];
                if (score(combined(trial), CALIBRATED_BUCKET_TILE_EQUIVALENT) <
                    score(combined(coupled_best),
                          CALIBRATED_BUCKET_TILE_EQUIVALENT))
                    coupled = candidate;
                if (score(trial.selected, CALIBRATED_BUCKET_TILE_EQUIVALENT) <
                    score(costs[test_dataset][selected * count + record].selected,
                          CALIBRATED_BUCKET_TILE_EQUIVALENT))
                    selected = candidate;
                if (score(trial.complement,
                          CALIBRATED_BUCKET_TILE_EQUIVALENT) <
                    score(costs[test_dataset][complement * count + record]
                              .complement,
                          CALIBRATED_BUCKET_TILE_EQUIVALENT))
                    complement = candidate;

                const SplitKeys& candidate_keys =
                    datasets[test_dataset].keys[candidate][record];
                const QuotientPair& proxy_left =
                    pairs[prefix_ids.at(candidate_keys.first)];
                const QuotientPair& proxy_right =
                    pairs[prefix_ids.at(candidate_keys.second)];
                long double candidate_proxy =
                    support_product_proxy(proxy_left, proxy_right, false) +
                    support_product_proxy(proxy_left, proxy_right, true);
                if (candidate_proxy < proxy_score) {
                    proxy_score = candidate_proxy;
                    proxy = candidate;
                }
            }
            for (size_t candidate : two_cuts)
                if (score(combined(costs[test_dataset]
                                      [candidate * count + record]),
                          CALIBRATED_BUCKET_TILE_EQUIVALENT) <
                    score(combined(costs[test_dataset]
                                      [two_cut * count + record]),
                          CALIBRATED_BUCKET_TILE_EQUIVALENT))
                    two_cut = candidate;
            for (size_t rank = 0; rank < 4; rank++) {
                size_t candidate = order[rank];
                if (score(combined(costs[test_dataset]
                                      [candidate * count + record]),
                          CALIBRATED_BUCKET_TILE_EQUIVALENT) <
                    score(combined(costs[test_dataset]
                                      [top_four * count + record]),
                          CALIBRATED_BUCKET_TILE_EQUIVALENT))
                    top_four = candidate;
            }
            add_cost(coupled_oracle.cost,
                     combined(costs[test_dataset][coupled * count + record]));
            const SplitKeys& ck = datasets[test_dataset].keys[coupled][record];
            coupled_oracle.left_keys.insert(ck.first);
            coupled_oracle.right_keys.insert(ck.second);

            const RecordCost& selected_cost =
                costs[test_dataset][selected * count + record];
            const RecordCost& complement_cost =
                costs[test_dataset][complement * count + record];
            add_cost(independent_oracle.cost, selected_cost.selected);
            add_cost(independent_oracle.cost, complement_cost.complement);
            const SplitKeys& sk = datasets[test_dataset].keys[selected][record];
            const SplitKeys& nk =
                datasets[test_dataset].keys[complement][record];
            independent_oracle.left_keys.insert(sk.first);
            independent_oracle.right_keys.insert(sk.second);
            independent_oracle.left_keys.insert(nk.first);
            independent_oracle.right_keys.insert(nk.second);

            for (auto item : {std::pair<size_t, StrategyStats*>{
                                  two_cut, &two_cut_oracle},
                              std::pair<size_t, StrategyStats*>{
                                  top_four, &top_four_oracle},
                              std::pair<size_t, StrategyStats*>{
                                  proxy, &support_proxy}}) {
                add_cost(item.second->cost,
                         combined(costs[test_dataset]
                                       [item.first * count + record]));
                const SplitKeys& keys =
                    datasets[test_dataset].keys[item.first][record];
                item.second->left_keys.insert(keys.first);
                item.second->right_keys.insert(keys.second);
            }
          }
        }
        print_strategy("test", "vertical-horizontal-oracle", two_cut_oracle,
                       test_baseline, prefix_ids, pairs);
        print_strategy("test", "learned-top4-oracle", top_four_oracle,
                       test_baseline, prefix_ids, pairs);
        print_strategy("test", "support-product-selector", support_proxy,
                       test_baseline, prefix_ids, pairs);
        print_strategy("test", "coupled-oracle", coupled_oracle,
                       test_baseline, prefix_ids, pairs);
        print_strategy("test", "independent-oracle", independent_oracle,
                       test_baseline, prefix_ids, pairs);
        std::cout << "COLUMN_SPLIT_DONE cuts=" << cuts.size()
                  << " execution_orientations=" << cuts.size() * 2
                  << " seconds=" << seconds_now() - start << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
