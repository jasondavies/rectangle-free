#include <iomanip>
#include <iostream>
#include <tuple>

#define PREFIX_BUCKET_TT_RANK_CENSUS_NO_MAIN
#include "prefix_bucket_tt_rank_census.cpp"

namespace {

constexpr uint64_t TOKEN_MASK = (UINT64_C(1) << (2 * PAIRS)) - 1;

struct CountedIncrement {
    uint64_t mask = 0;
    uint8_t ones = 0;
    uint32_t weight = 0;
};

struct CountedMapEntry {
    uint64_t key = 0;
    uint64_t weight = 0;
    bool used = false;
};

class CountedMap {
  public:
    explicit CountedMap(size_t capacity = 16) {
        size_t size = 16;
        while (size < capacity) size <<= 1;
        entries_.resize(size);
    }

    void add(uint64_t key, uint64_t weight) {
        if ((count_ + 1) * 10 >= entries_.size() * 7) rehash();
        size_t slot = size_t(mix64(key)) & (entries_.size() - 1);
        while (entries_[slot].used) {
            if (entries_[slot].key == key) {
                entries_[slot].weight += weight;
                return;
            }
            slot = (slot + 1) & (entries_.size() - 1);
        }
        entries_[slot] = CountedMapEntry{key, weight, true};
        count_++;
    }

    uint64_t find(uint64_t key) const {
        size_t slot = size_t(mix64(key)) & (entries_.size() - 1);
        while (entries_[slot].used) {
            if (entries_[slot].key == key) return entries_[slot].weight;
            slot = (slot + 1) & (entries_.size() - 1);
        }
        return 0;
    }

    const std::vector<CountedMapEntry>& entries() const { return entries_; }
    size_t count() const { return count_; }

  private:
    void rehash() {
        std::vector<CountedMapEntry> old = std::move(entries_);
        entries_.assign(old.size() * 2, CountedMapEntry{});
        count_ = 0;
        for (const CountedMapEntry& entry : old)
            if (entry.used) add(entry.key, entry.weight);
    }

    std::vector<CountedMapEntry> entries_;
    size_t count_ = 0;
};

static std::array<std::vector<CountedIncrement>, 1U << ROWS>
    counted_increments;

static uint64_t pack_counted(uint64_t mask, unsigned ones) {
    if (mask & ~TOKEN_MASK || ones > ROWS * HALF_COLUMNS)
        throw std::logic_error("counted distribution key overflow");
    return mask | (uint64_t(ones) << 56);
}

static uint64_t counted_mask(uint64_t key) { return key & TOKEN_MASK; }
static unsigned counted_ones(uint64_t key) { return unsigned(key >> 56); }

static uint64_t swap_token_planes_local(uint64_t mask) {
    constexpr uint64_t plane = (UINT64_C(1) << PAIRS) - 1;
    return ((mask & plane) << PAIRS) | ((mask >> PAIRS) & plane);
}

static void initialise_counted_increments() {
    for (unsigned active = 0; active < (1U << ROWS); active++) {
        std::vector<std::pair<uint64_t, uint8_t>> raw;
        unsigned assignment = active;
        for (;;) {
            uint64_t mask = 0;
            for (unsigned first = 0; first < ROWS; first++) {
                for (unsigned second = first + 1; second < ROWS; second++) {
                    if (!(active & (1U << first)) ||
                        !(active & (1U << second)))
                        continue;
                    unsigned a = (assignment >> first) & 1U;
                    unsigned b = (assignment >> second) & 1U;
                    if (a == b)
                        mask |= UINT64_C(1)
                                << (a * PAIRS +
                                    g_pair_index[first][second]);
                }
            }
            raw.emplace_back(mask, uint8_t(__builtin_popcount(assignment)));
            if (!assignment) break;
            assignment = (assignment - 1U) & active;
        }
        std::sort(raw.begin(), raw.end());
        for (size_t begin = 0; begin < raw.size();) {
            size_t end = begin + 1;
            while (end < raw.size() && raw[end] == raw[begin]) end++;
            counted_increments[active].push_back(CountedIncrement{
                raw[begin].first, raw[begin].second,
                uint32_t(end - begin)});
            begin = end;
        }
    }
}

static uint32_t canonical_half(uint32_t key) {
    std::array<uint8_t, ROWS> rows{};
    for (int row = ROWS - 1; row >= 0; row--) {
        rows[size_t(row)] = uint8_t(key & 15U);
        key >>= HALF_COLUMNS;
    }
    std::array<unsigned, HALF_COLUMNS> permutation{0, 1, 2, 3};
    uint32_t best = UINT32_MAX;
    do {
        std::array<uint8_t, ROWS> transformed{};
        for (unsigned row = 0; row < ROWS; row++)
            for (unsigned output = 0; output < HALF_COLUMNS; output++)
                transformed[row] |=
                    ((rows[row] >> permutation[output]) & 1U) << output;
        std::sort(transformed.begin(), transformed.end());
        uint32_t candidate = 0;
        for (uint8_t row : transformed) candidate = (candidate << 4) | row;
        best = std::min(best, candidate);
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    return best;
}

static std::vector<uint32_t> enumerate_canonical_halves() {
    std::vector<uint32_t> keys;
    std::array<uint8_t, ROWS> rows{};
    auto visit = [&](auto&& self, unsigned position, unsigned minimum) -> void {
        if (position == rows.size()) {
            uint32_t key = 0;
            for (uint8_t row : rows) key = (key << HALF_COLUMNS) | row;
            keys.push_back(canonical_half(key));
            return;
        }
        for (unsigned value = minimum; value < (1U << HALF_COLUMNS); value++) {
            rows[position] = uint8_t(value);
            self(self, position + 1, value);
        }
    };
    visit(visit, 0, 0);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    if (keys.size() != 25207)
        throw std::runtime_error("canonical 8x4 census is incomplete");
    return keys;
}

static unsigned active_cells(uint32_t prefix, bool complement) {
    unsigned active = unsigned(__builtin_popcount(prefix));
    return complement ? ROWS * HALF_COLUMNS - active : active;
}

static CountedMap build_counted_distribution(uint32_t prefix,
                                             bool complement) {
    std::array<uint8_t, ROWS> row_patterns{};
    constexpr unsigned pattern_mask = (1U << HALF_COLUMNS) - 1U;
    for (int row = ROWS - 1; row >= 0; row--) {
        row_patterns[size_t(row)] = uint8_t(prefix & pattern_mask);
        prefix >>= HALF_COLUMNS;
    }
    CountedMap current;
    current.add(pack_counted(0, 0), 1);
    for (unsigned column = 0; column < HALF_COLUMNS; column++) {
        unsigned active = 0;
        for (unsigned row = 0; row < ROWS; row++) {
            unsigned pattern = complement
                ? row_patterns[row] ^ pattern_mask
                : row_patterns[row];
            if (pattern & (1U << column)) active |= 1U << row;
        }
        CountedMap next(std::max<size_t>(16, current.count()));
        for (const CountedMapEntry& entry : current.entries()) {
            if (!entry.used) continue;
            uint64_t mask = counted_mask(entry.key);
            unsigned ones = counted_ones(entry.key);
            for (const CountedIncrement& increment : counted_increments[active]) {
                if (mask & increment.mask) continue;
                next.add(pack_counted(mask | increment.mask,
                                      ones + increment.ones),
                         entry.weight * increment.weight);
            }
        }
        current = std::move(next);
    }
    return current;
}

struct DistributionStats {
    uint64_t ordinary_entries = 0;
    uint64_t quotient_entries = 0;
    uint64_t counted_entries = 0;
    uint64_t counted_quotient_entries = 0;
    uint64_t total_weight = 0;
    uint32_t ordinary_weight_classes = 0;
    uint32_t counted_weight_classes = 0;
};

using SupportHistogram = std::array<uint64_t, ROWS * HALF_COLUMNS + 1>;

static SupportHistogram counted_support_histogram(uint32_t prefix,
                                                  bool complement) {
    CountedMap counted = build_counted_distribution(prefix, complement);
    SupportHistogram result{};
    for (const CountedMapEntry& entry : counted.entries())
        if (entry.used) result[counted_ones(entry.key)]++;
    return result;
}

static DistributionStats measure_distribution(uint32_t prefix,
                                              bool complement) {
    std::vector<FullWeightedEntry> ordinary =
        build_full_weighted_distribution(prefix, complement);
    CountedMap counted = build_counted_distribution(prefix, complement);
    unsigned active = active_cells(prefix, complement);
    DistributionStats result;
    result.ordinary_entries = ordinary.size();
    result.counted_entries = counted.count();
    std::unordered_set<uint32_t> ordinary_weights;
    std::unordered_set<uint32_t> counted_weights;
    uint64_t ordinary_total = 0;
    uint64_t ordinary_expanded = 0;
    for (const FullWeightedEntry& entry : ordinary) {
        ordinary_total += entry.weight;
        ordinary_weights.insert(entry.weight);
        uint64_t swapped = swap_token_planes_local(entry.mask);
        if (entry.mask <= swapped) {
            result.quotient_entries++;
            ordinary_expanded += entry.mask == swapped ? 1 : 2;
        }
    }
    if (ordinary_expanded != ordinary.size())
        throw std::logic_error("ordinary token-plane quotient failed");
    uint64_t counted_expanded = 0;
    for (const CountedMapEntry& entry : counted.entries()) {
        if (!entry.used) continue;
        uint64_t mask = counted_mask(entry.key);
        unsigned ones = counted_ones(entry.key);
        uint64_t swapped = swap_token_planes_local(mask);
        unsigned swapped_ones = active - ones;
        uint64_t partner = pack_counted(swapped, swapped_ones);
        if (counted.find(partner) != entry.weight)
            throw std::logic_error("cardinality token-plane equivariance failed");
        result.total_weight += entry.weight;
        if (std::pair(mask, ones) <= std::pair(swapped, swapped_ones)) {
            result.counted_quotient_entries++;
            counted_expanded += partner == entry.key ? 1 : 2;
            if (entry.weight > UINT32_MAX)
                throw std::overflow_error("counted weight exceeds uint32_t");
            counted_weights.insert(uint32_t(entry.weight));
        }
    }
    if (counted_expanded != counted.count())
        throw std::logic_error("cardinality token-plane quotient failed");
    if (ordinary_total != result.total_weight)
        throw std::logic_error("cardinality refinement changed total weight");
    result.ordinary_weight_classes = uint32_t(ordinary_weights.size());
    result.counted_weight_classes = uint32_t(counted_weights.size());
    return result;
}

struct WorkStats {
    U128 baseline = 0;
    U128 retained = 0;
    uint64_t records = 0;
};

static unsigned balanced_side(unsigned cells) {
    constexpr unsigned total = ROWS * 2 * HALF_COLUMNS;
    return std::min(cells, total - cells);
}

static void verify_small_colour_cut(unsigned rows, unsigned columns) {
    unsigned cells = rows * columns;
    uint64_t assignments = UINT64_C(1) << (2 * cells);
    uint64_t valid = 0, selected_sixths = 0;
    for (uint64_t packed = 0; packed < assignments; packed++) {
        bool rectangle_free = true;
        for (unsigned first = 0; first < rows && rectangle_free; first++) {
            for (unsigned second = first + 1;
                 second < rows && rectangle_free; second++) {
                for (unsigned left = 0; left < columns && rectangle_free; left++) {
                    for (unsigned right = left + 1; right < columns; right++) {
                        unsigned a = unsigned((packed >>
                            (2 * (first * columns + left))) & 3U);
                        unsigned b = unsigned((packed >>
                            (2 * (first * columns + right))) & 3U);
                        unsigned c = unsigned((packed >>
                            (2 * (second * columns + left))) & 3U);
                        unsigned d = unsigned((packed >>
                            (2 * (second * columns + right))) & 3U);
                        if (a == b && a == c && a == d) {
                            rectangle_free = false;
                            break;
                        }
                    }
                }
            }
        }
        if (!rectangle_free) continue;
        valid++;
        std::array<unsigned, 3> cut_ones{};
        for (unsigned cell = 0; cell < cells; cell++) {
            unsigned colour = unsigned((packed >> (2 * cell)) & 3U);
            unsigned x = colour >> 1;
            unsigned y = colour & 1U;
            cut_ones[0] += x;
            cut_ones[1] += y;
            cut_ones[2] += x ^ y;
        }
        std::array<unsigned, 3> balance{};
        for (unsigned cut = 0; cut < 3; cut++)
            balance[cut] = std::min(cut_ones[cut], cells - cut_ones[cut]);
        unsigned minimum = *std::min_element(balance.begin(), balance.end());
        unsigned ties = unsigned(std::count(balance.begin(), balance.end(),
                                            minimum));
        if (balance[0] == minimum) selected_sixths += 6 / ties;
    }
    // The order-three colour action cyclically permutes x, y, and x xor y.
    // Weighting every minimizing cut by 1/ties chooses one representative;
    // selected_sixths stores six times that rational weight.
    if (selected_sixths != 2 * valid)
        throw std::logic_error("small colour-cut quotient identity failed");
    std::cout << "COLOUR_CUT_SMALL rows=" << rows
              << " columns=" << columns
              << " valid=" << valid
              << " selected_sixths=" << selected_sixths
              << " exact=OK\n";
}

static WorkStats estimate_record_work(
    const SampleRecord& record,
    const std::unordered_map<uint32_t, size_t>& prefix_index,
    const std::vector<std::array<SupportHistogram, 2>>& histograms) {
    uint32_t left = half_prefix(record.key, 0);
    uint32_t right = half_prefix(record.key, HALF_COLUMNS);
    const auto& left_hist = histograms[prefix_index.at(left)];
    const auto& right_hist = histograms[prefix_index.at(right)];
    constexpr unsigned total_cells = ROWS * 2 * HALF_COLUMNS;
    unsigned selected_cells = unsigned(__builtin_popcountll(record.key));
    unsigned complement_cells = total_cells - selected_cells;
    unsigned selected_balance = balanced_side(selected_cells);
    std::array<bool, total_cells + 1> allowed_selected{};
    std::array<bool, total_cells + 1> allowed_complement{};
    for (unsigned selected_ones = 0;
         selected_ones <= selected_cells; selected_ones++) {
        for (unsigned complement_ones = 0;
             complement_ones <= complement_cells; complement_ones++) {
            unsigned second_cut = selected_ones + complement_ones;
            unsigned xor_cut = selected_ones +
                               (complement_cells - complement_ones);
            if (selected_balance <= balanced_side(second_cut) &&
                selected_balance <= balanced_side(xor_cut)) {
                allowed_selected[selected_ones] = true;
                allowed_complement[complement_ones] = true;
            }
        }
    }
    WorkStats result;
    result.records = 1;
    for (unsigned component = 0; component < 2; component++) {
        const SupportHistogram& a = left_hist[component];
        const SupportHistogram& b = right_hist[component];
        const auto& allowed = component ? allowed_complement : allowed_selected;
        for (unsigned first = 0; first < a.size(); first++) {
            for (unsigned second = 0; second < b.size(); second++) {
                U128 work = U128(a[first]) * b[second];
                result.baseline += work;
                if (allowed[first + second]) result.retained += work;
            }
        }
    }
    return result;
}

static void add_work(WorkStats& destination, const WorkStats& source) {
    destination.baseline += source.baseline;
    destination.retained += source.retained;
    destination.records += source.records;
}

static void run_work_census(const std::string& path, uint64_t sample_records) {
    std::vector<SampleRecord> records = read_stride_sample(path, sample_records);
    std::vector<uint32_t> prefixes;
    prefixes.reserve(records.size() * 2);
    for (const SampleRecord& record : records) {
        prefixes.push_back(half_prefix(record.key, 0));
        prefixes.push_back(half_prefix(record.key, HALF_COLUMNS));
    }
    std::sort(prefixes.begin(), prefixes.end());
    prefixes.erase(std::unique(prefixes.begin(), prefixes.end()), prefixes.end());
    std::vector<std::array<SupportHistogram, 2>> histograms(prefixes.size());
#pragma omp parallel for schedule(dynamic, 1)
    for (long long index = 0; index < (long long)prefixes.size(); index++) {
        histograms[size_t(index)][0] =
            counted_support_histogram(prefixes[size_t(index)], false);
        histograms[size_t(index)][1] =
            counted_support_histogram(prefixes[size_t(index)], true);
    }
    std::unordered_map<uint32_t, size_t> prefix_index;
    prefix_index.reserve(prefixes.size() * 2);
    for (size_t index = 0; index < prefixes.size(); index++)
        prefix_index.emplace(prefixes[index], index);
    std::array<WorkStats, ROWS * 2 * HALF_COLUMNS + 1> by_weight{};
    std::vector<WorkStats> per_record(records.size());
#pragma omp parallel for schedule(dynamic, 16)
    for (long long index = 0; index < (long long)records.size(); index++)
        per_record[size_t(index)] =
            estimate_record_work(records[size_t(index)], prefix_index,
                                 histograms);
    WorkStats total;
    for (size_t index = 0; index < records.size(); index++) {
        add_work(total, per_record[index]);
        unsigned weight = unsigned(__builtin_popcountll(records[index].key));
        add_work(by_weight[weight], per_record[index]);
    }
    std::cout << "COLOUR_CUT_WORK path=" << path
              << " records=" << total.records
              << " unique_halves=" << prefixes.size()
              << " baseline_pairs=" << u128_string(total.baseline)
              << " retained_pairs=" << u128_string(total.retained)
              << " retained_ratio="
              << static_cast<long double>(total.retained) /
                     static_cast<long double>(total.baseline)
              << " exact_count_gate=OK\n";
    for (unsigned weight = 0; weight < by_weight.size(); weight++) {
        const WorkStats& item = by_weight[weight];
        if (!item.records) continue;
        std::cout << "COLOUR_CUT_WORK_WEIGHT weight=" << weight
                  << " records=" << item.records
                  << " retained_ratio="
                  << static_cast<long double>(item.retained) /
                         static_cast<long double>(item.baseline)
                  << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 4) {
            std::cerr << "usage: " << argv[0]
                      << " [CANONICAL_HALF_SAMPLES=256;0=all]"
                         " [8X8_SHARD] [SHARD_RECORD_SAMPLES=4096]\n";
            return 2;
        }
        size_t requested = argc >= 2 ? std::stoull(argv[1]) : 256;
        initialise_tables();
        initialise_weighted_increments();
        initialise_counted_increments();
        verify_small_colour_cut(2, 2);
        verify_small_colour_cut(2, 3);
        verify_small_colour_cut(3, 3);
        std::vector<uint32_t> canonical = enumerate_canonical_halves();
        if (!requested || requested > canonical.size()) requested = canonical.size();
        std::vector<uint32_t> sample;
        sample.reserve(requested);
        for (size_t index = 0; index < requested; index++) {
            size_t source = size_t((U128(index) * canonical.size() +
                                    canonical.size() / 2) / requested);
            if (source >= canonical.size()) source = canonical.size() - 1;
            sample.push_back(canonical[source]);
        }
        std::vector<DistributionStats> stats(2 * sample.size());
        double start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
        for (long long index = 0; index < (long long)sample.size(); index++) {
            stats[2 * size_t(index)] =
                measure_distribution(sample[size_t(index)], false);
            stats[2 * size_t(index) + 1] =
                measure_distribution(sample[size_t(index)], true);
        }
        DistributionStats total;
        double maximum_ratio = 0;
        uint64_t within_three = 0;
        uint32_t maximum_ordinary_classes = 0;
        uint32_t maximum_counted_classes = 0;
        for (const DistributionStats& item : stats) {
            total.ordinary_entries += item.ordinary_entries;
            total.quotient_entries += item.quotient_entries;
            total.counted_entries += item.counted_entries;
            total.counted_quotient_entries += item.counted_quotient_entries;
            total.total_weight += item.total_weight;
            double ratio = item.quotient_entries
                ? double(item.counted_quotient_entries) / item.quotient_entries
                : 1.0;
            maximum_ratio = std::max(maximum_ratio, ratio);
            within_three += ratio <= 3.0;
            maximum_ordinary_classes = std::max(
                maximum_ordinary_classes, item.ordinary_weight_classes);
            maximum_counted_classes = std::max(
                maximum_counted_classes, item.counted_weight_classes);
        }
        std::cout << std::setprecision(12)
                  << "COLOUR_CUT_CARDINALITY samples=" << sample.size()
                  << " distributions=" << stats.size()
                  << " ordinary_entries=" << total.ordinary_entries
                  << " ordinary_quotient_entries=" << total.quotient_entries
                  << " counted_entries=" << total.counted_entries
                  << " counted_quotient_entries="
                  << total.counted_quotient_entries
                  << " unquotiented_ratio="
                  << double(total.counted_entries) / total.ordinary_entries
                  << " quotient_ratio="
                  << double(total.counted_quotient_entries) /
                         total.quotient_entries
                  << " maximum_distribution_ratio=" << maximum_ratio
                  << " distributions_within_three=" << within_three
                  << " maximum_ordinary_weight_classes="
                  << maximum_ordinary_classes
                  << " maximum_counted_weight_classes="
                  << maximum_counted_classes
                  << " seconds=" << seconds_now() - start
                  << " exact=OK\n";
        if (argc >= 3) {
            uint64_t record_samples = argc == 4 ? std::stoull(argv[3]) : 4096;
            run_work_census(argv[2], record_samples);
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
