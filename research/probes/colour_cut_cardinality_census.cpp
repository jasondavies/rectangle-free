#include <iomanip>
#include <iostream>
#include <map>
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

// A stronger colour-cut gate.  The four nibbles store the number of inner
// one-bits in each physical half-column.  Keeping this information alongside
// the support is exact; unlike the cardinality-only census it distinguishes
// column-degree multisets of the three 2+2 colour cuts.
struct DegreeKey {
    uint64_t mask = 0;
    uint16_t degrees = 0;

    bool operator==(const DegreeKey& other) const {
        return mask == other.mask && degrees == other.degrees;
    }
};

struct DegreeMapEntry {
    DegreeKey key;
    uint64_t weight = 0;
    bool used = false;
};

class DegreeMap {
  public:
    explicit DegreeMap(size_t capacity = 16) {
        size_t size = 16;
        while (size < capacity * 2) size <<= 1;
        entries_.resize(size);
    }

    bool add(DegreeKey key, uint64_t weight, size_t state_cap) {
        if ((count_ + 1) * 10 >= entries_.size() * 7) rehash();
        size_t slot = size_t(mix64(key.mask ^
                                   (uint64_t(key.degrees) << 41))) &
                      (entries_.size() - 1);
        while (entries_[slot].used) {
            if (entries_[slot].key == key) {
                entries_[slot].weight += weight;
                return true;
            }
            slot = (slot + 1) & (entries_.size() - 1);
        }
        if (count_ >= state_cap) return false;
        entries_[slot] = DegreeMapEntry{key, weight, true};
        count_++;
        return true;
    }

    const std::vector<DegreeMapEntry>& entries() const { return entries_; }
    size_t count() const { return count_; }

  private:
    void rehash() {
        std::vector<DegreeMapEntry> old = std::move(entries_);
        entries_.assign(old.size() * 2, DegreeMapEntry{});
        count_ = 0;
        for (const DegreeMapEntry& entry : old) {
            if (!entry.used) continue;
            if (!add(entry.key, entry.weight, SIZE_MAX))
                throw std::logic_error("degree-map rehash failed");
        }
    }

    std::vector<DegreeMapEntry> entries_;
    size_t count_ = 0;
};

using DegreeHistogram = std::vector<std::pair<uint16_t, uint64_t>>;

struct DegreeHistogramData {
    DegreeHistogram histogram;
    uint64_t state_count = 0;
    uint64_t transition_count = 0;
    bool capped = false;
};

static DegreeHistogramData degree_support_histogram(uint32_t prefix,
                                                    bool complement,
                                                    size_t state_cap,
                                                    uint64_t transition_cap) {
    uint32_t original_prefix = prefix;
    bool trace = std::getenv("COLOUR_CUT_TRACE") != nullptr;
    std::array<uint8_t, ROWS> row_patterns{};
    constexpr unsigned pattern_mask = (1U << HALF_COLUMNS) - 1U;
    for (int row = ROWS - 1; row >= 0; row--) {
        row_patterns[size_t(row)] = uint8_t(prefix & pattern_mask);
        prefix >>= HALF_COLUMNS;
    }
    DegreeMap current;
    current.add(DegreeKey{}, 1, state_cap);
    uint64_t transitions = 0;
    for (unsigned column = 0; column < HALF_COLUMNS; column++) {
        unsigned active = 0;
        for (unsigned row = 0; row < ROWS; row++) {
            unsigned pattern = complement
                ? row_patterns[row] ^ pattern_mask
                : row_patterns[row];
            if (pattern & (1U << column)) active |= 1U << row;
        }
        DegreeMap next(std::max<size_t>(16, current.count()));
        for (const DegreeMapEntry& entry : current.entries()) {
            if (!entry.used) continue;
            const DegreeKey& key = entry.key;
            for (const CountedIncrement& increment : counted_increments[active]) {
                if (++transitions >= transition_cap)
                    return DegreeHistogramData{
                        {}, current.count(), transitions, true};
                if (key.mask & increment.mask) continue;
                DegreeKey child{
                    key.mask | increment.mask,
                    uint16_t(key.degrees |
                             (uint16_t(increment.ones) << (4 * column)))
                };
                if (!next.add(child, entry.weight * increment.weight,
                              state_cap))
                    return DegreeHistogramData{
                        {}, next.count(), transitions, true};
            }
        }
        current = std::move(next);
        if (trace) {
#pragma omp critical(colour_cut_trace)
            std::cerr << "degree_trace prefix=" << original_prefix
                      << " complement=" << complement
                      << " column=" << column
                      << " states=" << current.count()
                      << " transitions=" << transitions << '\n';
        }
    }
    std::map<uint16_t, uint64_t> histogram;
    for (const DegreeMapEntry& entry : current.entries()) {
        if (!entry.used) continue;
        histogram[entry.key.degrees]++;
    }
    return DegreeHistogramData{
        DegreeHistogram(histogram.begin(), histogram.end()), current.count(),
        transitions, false};
}

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

struct DegreeWorkStats {
    U128 baseline = 0;
    U128 retained = 0;
    U128 refined_entries = 0;
    U128 ordinary_entries = 0;
    uint64_t records = 0;
    uint64_t selected_sectors = 0;
    uint64_t complement_sectors = 0;
    uint64_t sector_pair_capped_records = 0;
};

struct CutSignature {
    uint8_t balance = 0;
    uint64_t degrees = 0;

    auto tuple() const { return std::tie(balance, degrees); }
    bool operator<=(const CutSignature& other) const {
        return tuple() <= other.tuple();
    }
};

static uint8_t degree_nibble(uint32_t code, unsigned column) {
    return uint8_t((code >> (4 * column)) & 15U);
}

static uint64_t pack_sorted_degrees(std::array<uint8_t, 2 * HALF_COLUMNS> values) {
    std::sort(values.begin(), values.end());
    uint64_t packed = 0;
    for (uint8_t value : values) packed = (packed << 4) | value;
    return packed;
}

static CutSignature column_cut_signature(
    const std::array<uint8_t, 2 * HALF_COLUMNS>& input) {
    std::array<uint8_t, 2 * HALF_COLUMNS> complement{};
    unsigned ones = 0;
    for (unsigned column = 0; column < input.size(); column++) {
        if (input[column] > ROWS)
            throw std::logic_error("invalid column degree");
        ones += input[column];
        complement[column] = uint8_t(ROWS - input[column]);
    }
    uint64_t direct = pack_sorted_degrees(input);
    uint64_t inverse = pack_sorted_degrees(complement);
    return CutSignature{
        uint8_t(std::min(ones, ROWS * unsigned(input.size()) - ones)),
        std::min(direct, inverse)
    };
}

using FullDegreeHistogram = std::vector<std::pair<uint32_t, uint64_t>>;

static FullDegreeHistogram combine_degree_histograms(
    const DegreeHistogram& left, const DegreeHistogram& right) {
    FullDegreeHistogram output;
    output.reserve(left.size() * right.size());
    for (const auto& [left_code, left_count] : left) {
        for (const auto& [right_code, right_count] : right) {
            output.emplace_back(uint32_t(left_code) |
                                    (uint32_t(right_code) << 16),
                                left_count * right_count);
        }
    }
    return output;
}

static std::array<uint8_t, 2 * HALF_COLUMNS> outer_column_degrees(
    uint64_t key) {
    std::array<uint8_t, 2 * HALF_COLUMNS> output{};
    for (unsigned row = 0; row < ROWS; row++) {
        for (unsigned column = 0; column < output.size(); column++) {
            output[column] += uint8_t((key >> (row * output.size() + column)) & 1U);
        }
    }
    return output;
}

static DegreeWorkStats estimate_degree_work(
    const SampleRecord& record,
    const std::unordered_map<uint32_t, size_t>& prefix_index,
    const std::vector<std::array<DegreeHistogramData, 2>>& histograms,
    const std::vector<std::array<uint64_t, 2>>& ordinary_counts) {
    uint32_t left = half_prefix(record.key, 0);
    uint32_t right = half_prefix(record.key, HALF_COLUMNS);
    size_t left_index = prefix_index.at(left);
    size_t right_index = prefix_index.at(right);
    FullDegreeHistogram selected = combine_degree_histograms(
        histograms[left_index][0].histogram,
        histograms[right_index][0].histogram);
    FullDegreeHistogram complement = combine_degree_histograms(
        histograms[left_index][1].histogram,
        histograms[right_index][1].histogram);
    if (std::getenv("COLOUR_CUT_TRACE")) {
#pragma omp critical(colour_cut_trace)
        std::cerr << "degree_join_trace selected_sectors=" << selected.size()
                  << " complement_sectors=" << complement.size()
                  << " sector_pairs="
                  << u128_string(U128(selected.size()) * complement.size())
                  << '\n';
    }
    std::array<uint8_t, 2 * HALF_COLUMNS> x_degrees =
        outer_column_degrees(record.key);
    CutSignature x_signature = column_cut_signature(x_degrees);
    std::vector<uint8_t> keep_selected(selected.size());
    std::vector<uint8_t> keep_complement(complement.size());
    constexpr U128 sector_pair_cap = 10'000'000;
    if (U128(selected.size()) * complement.size() > sector_pair_cap) {
        DegreeWorkStats result;
        result.records = 1;
        result.sector_pair_capped_records = 1;
        result.selected_sectors = selected.size();
        result.complement_sectors = complement.size();
        for (const auto& item : selected) result.baseline += item.second;
        for (const auto& item : complement) result.baseline += item.second;
        for (unsigned component = 0; component < 2; component++) {
            for (const auto& [code, count] :
                 histograms[left_index][component].histogram) {
                (void)code;
                result.refined_entries += count;
            }
            for (const auto& [code, count] :
                 histograms[right_index][component].histogram) {
                (void)code;
                result.refined_entries += count;
            }
            result.ordinary_entries += ordinary_counts[left_index][component];
            result.ordinary_entries += ordinary_counts[right_index][component];
        }
        return result;
    }
    for (size_t first = 0; first < selected.size(); first++) {
        uint32_t selected_code = selected[first].first;
        for (size_t second = 0; second < complement.size(); second++) {
            uint32_t complement_code = complement[second].first;
            std::array<uint8_t, 2 * HALF_COLUMNS> y{}, z{};
            for (unsigned column = 0; column < y.size(); column++) {
                unsigned sy = degree_nibble(selected_code, column);
                unsigned cy = degree_nibble(complement_code, column);
                y[column] = uint8_t(sy + cy);
                z[column] = uint8_t(x_degrees[column] - sy + cy);
            }
            if (x_signature <= column_cut_signature(y) &&
                x_signature <= column_cut_signature(z)) {
                keep_selected[first] = 1;
                keep_complement[second] = 1;
            }
        }
    }
    DegreeWorkStats result;
    result.records = 1;
    result.selected_sectors = selected.size();
    result.complement_sectors = complement.size();
    for (size_t index = 0; index < selected.size(); index++) {
        result.baseline += selected[index].second;
        if (keep_selected[index]) result.retained += selected[index].second;
    }
    for (size_t index = 0; index < complement.size(); index++) {
        result.baseline += complement[index].second;
        if (keep_complement[index]) result.retained += complement[index].second;
    }
    for (unsigned component = 0; component < 2; component++) {
        for (const auto& [code, count] :
             histograms[left_index][component].histogram) {
            (void)code;
            result.refined_entries += count;
        }
        for (const auto& [code, count] :
             histograms[right_index][component].histogram) {
            (void)code;
            result.refined_entries += count;
        }
        result.ordinary_entries += ordinary_counts[left_index][component];
        result.ordinary_entries += ordinary_counts[right_index][component];
    }
    return result;
}

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

static void add_degree_work(DegreeWorkStats& destination,
                            const DegreeWorkStats& source) {
    destination.baseline += source.baseline;
    destination.retained += source.retained;
    destination.refined_entries += source.refined_entries;
    destination.ordinary_entries += source.ordinary_entries;
    destination.records += source.records;
    destination.selected_sectors += source.selected_sectors;
    destination.complement_sectors += source.complement_sectors;
    destination.sector_pair_capped_records += source.sector_pair_capped_records;
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

static void run_degree_work_census(const std::string& path,
                                   uint64_t sample_records) {
    std::vector<SampleRecord> records = read_stride_sample(path, sample_records);
    std::vector<uint32_t> prefixes;
    prefixes.reserve(records.size() * 2);
    for (const SampleRecord& record : records) {
        prefixes.push_back(half_prefix(record.key, 0));
        prefixes.push_back(half_prefix(record.key, HALF_COLUMNS));
    }
    std::sort(prefixes.begin(), prefixes.end());
    prefixes.erase(std::unique(prefixes.begin(), prefixes.end()), prefixes.end());
    std::vector<std::array<DegreeHistogramData, 2>> histograms(prefixes.size());
    std::vector<std::array<uint64_t, 2>> ordinary_counts(prefixes.size());
    double build_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
    for (long long index = 0; index < (long long)prefixes.size(); index++) {
        for (unsigned component = 0; component < 2; component++) {
            bool complement = component != 0;
            histograms[size_t(index)][component] = degree_support_histogram(
                prefixes[size_t(index)], complement, 1'000'000, 5'000'000);
            ordinary_counts[size_t(index)][component] =
                build_full_weighted_distribution(prefixes[size_t(index)],
                                                 complement).size();
        }
    }
    double build_seconds = seconds_now() - build_start;
    uint64_t capped_distributions = 0;
    uint64_t complete_distributions = 0;
    U128 complete_refined_states = 0;
    U128 complete_ordinary_states = 0;
    U128 attempted_transitions = 0;
    for (size_t index = 0; index < prefixes.size(); index++) {
        for (unsigned component = 0; component < 2; component++) {
            const DegreeHistogramData& item = histograms[index][component];
            attempted_transitions += item.transition_count;
            if (item.capped) capped_distributions++;
            else {
                complete_distributions++;
                complete_refined_states += item.state_count;
                complete_ordinary_states += ordinary_counts[index][component];
            }
        }
    }
    if (capped_distributions) {
        std::cout << "COLOUR_CUT_DEGREE_WORK path=" << path
                  << " records=" << records.size()
                  << " unique_halves=" << prefixes.size()
                  << " status=capped"
                  << " state_cap=1000000"
                  << " transition_cap=5000000"
                  << " capped_distributions=" << capped_distributions
                  << " complete_distributions=" << complete_distributions
                  << " complete_refined_ratio="
                  << (complete_ordinary_states
                      ? static_cast<long double>(complete_refined_states) /
                            static_cast<long double>(complete_ordinary_states)
                      : 0)
                  << " attempted_transitions="
                  << u128_string(attempted_transitions)
                  << " build_seconds=" << build_seconds
                  << "\n";
        return;
    }
    std::unordered_map<uint32_t, size_t> prefix_index;
    prefix_index.reserve(prefixes.size() * 2);
    for (size_t index = 0; index < prefixes.size(); index++)
        prefix_index.emplace(prefixes[index], index);
    std::vector<DegreeWorkStats> per_record(records.size());
    double census_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
    for (long long index = 0; index < (long long)records.size(); index++) {
        per_record[size_t(index)] = estimate_degree_work(
            records[size_t(index)], prefix_index, histograms, ordinary_counts);
    }
    DegreeWorkStats total;
    for (const DegreeWorkStats& item : per_record) add_degree_work(total, item);
    std::cout << "COLOUR_CUT_DEGREE_WORK path=" << path
              << " records=" << total.records
              << " unique_halves=" << prefixes.size()
              << " baseline_pairs=" << u128_string(total.baseline)
              << " retained_pairs=" << u128_string(total.retained);
    if (total.sector_pair_capped_records) {
        std::cout << " retained_ratio=unmeasured";
    } else {
        std::cout << " retained_ratio="
                  << static_cast<long double>(total.retained) /
                         static_cast<long double>(total.baseline);
    }
    std::cout << " refined_entry_ratio="
              << static_cast<long double>(total.refined_entries) /
                     static_cast<long double>(total.ordinary_entries)
              << " mean_selected_sectors="
              << double(total.selected_sectors) / total.records
              << " mean_complement_sectors="
              << double(total.complement_sectors) / total.records
              << " sector_pair_cap=10000000"
              << " sector_pair_capped_records="
              << total.sector_pair_capped_records
              << " build_seconds=" << build_seconds
              << " census_seconds=" << seconds_now() - census_start
              << " transpose_safe=no"
              << " structural_gate=OK\n";
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
            run_degree_work_census(argv[2], record_samples);
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
