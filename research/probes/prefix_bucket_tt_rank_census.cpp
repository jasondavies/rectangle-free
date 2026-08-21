#include <deque>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <tuple>
#include <unordered_set>

// Reuse the checked R8ORB01 reader, row-pair numbering, and half-prefix
// extraction from the hierarchy census.  The weighted recurrence below is
// separate because tensor ranks depend on coefficients, not only support.
#define main prefix_hierarchy_8x8_census_unused_main
#include "prefix_hierarchy_8x8_census.cpp"
#undef main

namespace {

struct WeightedIncrement {
    uint64_t mask;
    uint32_t weight;
};

struct WeightedMapEntry {
    uint64_t mask = 0;
    uint64_t weight = 0;
    bool used = false;
};

struct WeightedEntry {
    uint64_t suffix;
    uint32_t weight;
};

struct FullWeightedEntry {
    uint64_t mask;
    uint32_t weight;
};

struct WeightedBucket {
    uint16_t prefix = 0;
    std::vector<WeightedEntry> entries;
    std::vector<uint32_t> weight_class_counts;
    size_t candidate = 0;
};

struct BucketedDistribution {
    std::vector<WeightedBucket> buckets;
};

struct WeightedDistributionPair {
    BucketedDistribution selected;
    BucketedDistribution complement;
};

struct Candidate {
    size_t distribution = 0;
    unsigned component = 0;
    size_t bucket = 0;
    long double work_score = 0;
};

struct JoinCandidate {
    size_t left_distribution = 0;
    size_t right_distribution = 0;
    unsigned component = 0;
    size_t left_bucket = 0;
    size_t right_bucket = 0;
    U128 work = 0;
};

struct JoinCandidateGreater {
    bool operator()(const JoinCandidate& a, const JoinCandidate& b) const {
        return a.work > b.work;
    }
};

class WeightedMap {
  public:
    explicit WeightedMap(size_t capacity = 16) {
        size_t size = 16;
        while (size < capacity) size <<= 1;
        entries_.resize(size);
    }

    void add(uint64_t mask, uint64_t weight) {
        if ((count_ + 1) * 10 >= entries_.size() * 7) rehash();
        size_t slot = size_t(mix64(mask)) & (entries_.size() - 1);
        while (entries_[slot].used) {
            if (entries_[slot].mask == mask) {
                entries_[slot].weight += weight;
                return;
            }
            slot = (slot + 1) & (entries_.size() - 1);
        }
        entries_[slot] = WeightedMapEntry{mask, weight, true};
        count_++;
    }

    const std::vector<WeightedMapEntry>& entries() const { return entries_; }
    size_t count() const { return count_; }

  private:
    void rehash() {
        std::vector<WeightedMapEntry> old = std::move(entries_);
        entries_.assign(old.size() * 2, WeightedMapEntry{});
        count_ = 0;
        for (const auto& entry : old)
            if (entry.used) add(entry.mask, entry.weight);
    }

    std::vector<WeightedMapEntry> entries_;
    size_t count_ = 0;
};

static std::array<std::vector<WeightedIncrement>, 1U << ROWS>
    weighted_increments;

static constexpr std::array<uint8_t, PAIRS> production_pair_order = {
    0, 1, 7, 2, 8, 13, 27, 3, 14, 18, 9, 24, 26, 25,
    23, 22, 4, 10, 15, 19, 5, 11, 16, 20, 6, 12, 17, 21};

static void initialise_weighted_increments() {
    for (unsigned active = 0; active < (1U << ROWS); active++) {
        std::vector<uint64_t> masks;
        unsigned assignment = active;
        for (;;) {
            uint64_t mask = 0;
            for (unsigned first = 0; first < ROWS; first++) {
                for (unsigned second = first + 1; second < ROWS; second++) {
                    if (!(active & (1U << first)) ||
                        !(active & (1U << second)))
                        continue;
                    unsigned c1 = (assignment >> first) & 1U;
                    unsigned c2 = (assignment >> second) & 1U;
                    if (c1 == c2)
                        mask |= UINT64_C(1)
                                << (c1 * PAIRS +
                                    g_pair_index[first][second]);
                }
            }
            masks.push_back(mask);
            if (!assignment) break;
            assignment = (assignment - 1U) & active;
        }
        std::sort(masks.begin(), masks.end());
        auto& output = weighted_increments[active];
        for (size_t begin = 0; begin < masks.size();) {
            size_t end = begin + 1;
            while (end < masks.size() && masks[end] == masks[begin]) end++;
            output.push_back(
                WeightedIncrement{masks[begin], uint32_t(end - begin)});
            begin = end;
        }
    }
}

static unsigned production_pair_rank(unsigned pair) {
    for (unsigned rank = 0; rank < PAIRS; rank++)
        if (production_pair_order[rank] == pair) return rank;
    throw std::runtime_error("invalid pair index");
}

static void split_pair_mask(uint64_t mask, uint32_t pair_mask,
                            uint16_t& prefix, uint64_t& suffix) {
    prefix = 0;
    suffix = 0;
    unsigned prefix_bit = 0, suffix_bit = 0;
    for (unsigned colour = 0; colour < 2; colour++) {
        for (unsigned pair = 0; pair < PAIRS; pair++) {
            uint64_t bit = (mask >> (colour * PAIRS + pair)) & 1U;
            if (pair_mask & (uint32_t(1) << pair))
                prefix |= uint16_t(bit << prefix_bit++);
            else
                suffix |= bit << suffix_bit++;
        }
    }
    if (prefix_bit > 16 || prefix_bit + suffix_bit != 56)
        throw std::runtime_error("bad pair-mask prefix split");
}

static void split_production_mask(uint64_t mask, uint16_t& prefix,
                                  uint64_t& suffix) {
    uint32_t pair_mask = 0;
    for (unsigned rank = 0; rank < 7; rank++)
        pair_mask |= uint32_t(1) << production_pair_order[rank];
    split_pair_mask(mask, pair_mask, prefix, suffix);
}

static uint64_t join_production_mask(uint16_t prefix, uint64_t suffix) {
    uint64_t mask = 0;
    unsigned prefix_bit = 0, suffix_bit = 0;
    for (unsigned colour = 0; colour < 2; colour++) {
        for (unsigned pair = 0; pair < PAIRS; pair++) {
            uint64_t bit = production_pair_rank(pair) < 7
                ? (prefix >> prefix_bit++) & 1U
                : (suffix >> suffix_bit++) & 1U;
            mask |= bit << (colour * PAIRS + pair);
        }
    }
    return mask;
}

static std::vector<FullWeightedEntry> build_full_weighted_distribution(
    uint32_t prefix, bool complement) {
    std::array<uint8_t, ROWS> row_patterns{};
    constexpr unsigned pattern_mask = (1U << HALF_COLUMNS) - 1U;
    for (int row = ROWS - 1; row >= 0; row--) {
        row_patterns[size_t(row)] = uint8_t(prefix & pattern_mask);
        prefix >>= HALF_COLUMNS;
    }
    WeightedMap current;
    current.add(0, 1);
    for (unsigned column = 0; column < HALF_COLUMNS; column++) {
        unsigned active = 0;
        for (unsigned row = 0; row < ROWS; row++) {
            unsigned pattern = complement
                ? row_patterns[row] ^ pattern_mask
                : row_patterns[row];
            if (pattern & (1U << column)) active |= 1U << row;
        }
        WeightedMap next(std::max<size_t>(16, current.entries().size()));
        for (const auto& entry : current.entries()) {
            if (!entry.used) continue;
            for (const auto& increment : weighted_increments[active]) {
                if (entry.mask & increment.mask) continue;
                next.add(entry.mask | increment.mask,
                         entry.weight * increment.weight);
            }
        }
        current = std::move(next);
    }

    std::vector<FullWeightedEntry> result;
    result.reserve(current.count());
    for (const auto& entry : current.entries()) {
        if (!entry.used) continue;
        if (entry.weight > UINT32_MAX)
            throw std::overflow_error("distribution weight exceeds uint32_t");
        result.push_back(
            FullWeightedEntry{entry.mask, uint32_t(entry.weight)});
    }
    std::sort(result.begin(), result.end(),
              [](const FullWeightedEntry& a, const FullWeightedEntry& b) {
                  return a.mask < b.mask;
              });
    return result;
}

static BucketedDistribution bucket_weighted_distribution(
    const std::vector<FullWeightedEntry>& entries, uint32_t pair_mask) {
    std::unordered_map<uint16_t, std::vector<WeightedEntry>> buckets;
    for (const auto& entry : entries) {
        uint16_t bucket_prefix = 0;
        uint64_t suffix = 0;
        split_pair_mask(entry.mask, pair_mask, bucket_prefix, suffix);
        buckets[bucket_prefix].push_back(
            WeightedEntry{suffix, entry.weight});
    }
    BucketedDistribution result;
    result.buckets.reserve(buckets.size());
    for (auto& item : buckets) {
        std::sort(item.second.begin(), item.second.end(),
                  [](const WeightedEntry& a, const WeightedEntry& b) {
                      return a.suffix < b.suffix;
                  });
        std::unordered_map<uint32_t, uint32_t> weight_classes;
        for (const auto& entry : item.second) weight_classes[entry.weight]++;
        std::vector<uint32_t> class_counts;
        class_counts.reserve(weight_classes.size());
        for (const auto& weight_class : weight_classes)
            class_counts.push_back(weight_class.second);
        std::sort(class_counts.begin(), class_counts.end(),
                  std::greater<uint32_t>());
        result.buckets.push_back(
            WeightedBucket{item.first, std::move(item.second),
                           std::move(class_counts), 0});
    }
    std::sort(result.buckets.begin(), result.buckets.end(),
              [](const WeightedBucket& a, const WeightedBucket& b) {
                  return a.prefix < b.prefix;
              });
    return result;
}

static BucketedDistribution build_weighted_distribution(uint32_t prefix,
                                                         bool complement) {
    uint32_t pair_mask = 0;
    for (unsigned rank = 0; rank < 7; rank++)
        pair_mask |= uint32_t(1) << production_pair_order[rank];
    return bucket_weighted_distribution(
        build_full_weighted_distribution(prefix, complement), pair_mask);
}

static WeightedDistributionPair build_weighted_pair(uint32_t prefix) {
    return WeightedDistributionPair{
        build_weighted_distribution(prefix, false),
        build_weighted_distribution(prefix, true)};
}

static uint64_t predicate_bmma_tiles(uint64_t left, uint64_t right) {
    uint64_t forward = ((left + 15) / 16) * ((right + 7) / 8);
    uint64_t reverse = ((right + 15) / 16) * ((left + 7) / 8);
    return std::min(forward, reverse);
}

static uint64_t weight_class_bmma_tiles(const WeightedBucket& left,
                                        const WeightedBucket& right) {
    uint64_t tiles = 0;
    for (uint32_t left_count : left.weight_class_counts)
        for (uint32_t right_count : right.weight_class_counts)
            tiles += predicate_bmma_tiles(left_count, right_count);
    return tiles;
}

static void validate_weighted_support(
    uint32_t prefix, bool complement,
    const BucketedDistribution& weighted) {
    std::vector<uint64_t> reconstructed;
    for (const auto& bucket : weighted.buckets)
        for (const auto& entry : bucket.entries) {
            if (!entry.weight)
                throw std::runtime_error("zero weighted entry");
            reconstructed.push_back(
                join_production_mask(bucket.prefix, entry.suffix));
        }
    std::sort(reconstructed.begin(), reconstructed.end());
    if (std::adjacent_find(reconstructed.begin(), reconstructed.end()) !=
        reconstructed.end())
        throw std::runtime_error("duplicate reconstructed weighted mask");
    if (reconstructed != build_distribution(prefix, complement))
        throw std::runtime_error("weighted/support recurrence mismatch");
}

struct RankResult {
    unsigned rank = 0;
    unsigned rows = 0;
    unsigned columns = 0;
    unsigned odd_entries = 0;
    bool capped = false;
};

// Rank over an odd prime is needed because valid distribution weights are
// often even (and can therefore disappear completely over GF(2)).  Rather
// than materialising a potentially enormous unfolding, multiply its columns
// by a fixed dense random matrix with `cap` columns and row-reduce the result
// while streaming rows.  Matrix multiplication cannot increase rank, so the
// reported value is a rigorous lower bound over GF(p); randomness can only
// make the lower bound smaller, never create a false high rank.
static constexpr uint32_t RANK_PRIME = UINT32_C(2147483647);

static uint32_t mod_power(uint32_t value, uint32_t exponent) {
    uint32_t result = 1;
    while (exponent) {
        if (exponent & 1U)
            result = uint32_t(uint64_t(result) * value % RANK_PRIME);
        value = uint32_t(uint64_t(value) * value % RANK_PRIME);
        exponent >>= 1;
    }
    return result;
}

static uint32_t projection_coefficient(uint64_t column, unsigned output) {
    uint64_t key = column ^
        (UINT64_C(0x9e3779b97f4a7c15) * uint64_t(output + 1));
    return 1U + uint32_t(mix64(key) % (RANK_PRIME - 1U));
}

struct ModularEdge {
    uint64_t row;
    uint64_t column;
    uint32_t weight;
};

static unsigned modp_projected_unfolding_rank(
    const std::vector<WeightedEntry>& entries,
    const std::array<uint8_t, 42>& order, unsigned cut, unsigned cap) {
    std::vector<ModularEdge> edges;
    edges.reserve(entries.size());
    for (const auto& entry : entries) {
        uint64_t row = 0, column = 0;
        for (unsigned position = 0; position < 42; position++) {
            uint64_t bit = (entry.suffix >> order[position]) & 1U;
            if (position < cut)
                row |= bit << position;
            else
                column |= bit << (position - cut);
        }
        edges.push_back(ModularEdge{
            row, column, uint32_t(entry.weight % RANK_PRIME)});
    }
    std::sort(edges.begin(), edges.end(),
              [](const ModularEdge& a, const ModularEdge& b) {
                  return std::tie(a.row, a.column) <
                         std::tie(b.row, b.column);
              });

    std::vector<std::vector<uint32_t>> basis(
        cap, std::vector<uint32_t>());
    std::vector<uint32_t> vector(cap);
    unsigned rank = 0;
    for (size_t begin = 0; begin < edges.size();) {
        size_t end = begin + 1;
        while (end < edges.size() && edges[end].row == edges[begin].row) end++;
        std::fill(vector.begin(), vector.end(), 0);
        for (size_t edge = begin; edge < end; edge++) {
            for (unsigned output = 0; output < cap; output++) {
                uint32_t coefficient =
                    projection_coefficient(edges[edge].column, output);
                uint32_t term = uint32_t(
                    uint64_t(edges[edge].weight) * coefficient % RANK_PRIME);
                uint32_t value = vector[output] + term;
                if (value >= RANK_PRIME || value < vector[output])
                    value -= RANK_PRIME;
                vector[output] = value;
            }
        }
        for (unsigned pivot = 0; pivot < cap; pivot++) {
            if (!vector[pivot]) continue;
            if (basis[pivot].empty()) {
                uint32_t inverse =
                    mod_power(vector[pivot], RANK_PRIME - 2U);
                for (unsigned column = pivot; column < cap; column++)
                    vector[column] = uint32_t(
                        uint64_t(vector[column]) * inverse % RANK_PRIME);
                basis[pivot] = vector;
                rank++;
                break;
            }
            uint32_t factor = vector[pivot];
            for (unsigned column = pivot; column < cap; column++) {
                uint32_t term = uint32_t(
                    uint64_t(factor) * basis[pivot][column] % RANK_PRIME);
                vector[column] = vector[column] >= term
                    ? vector[column] - term
                    : vector[column] + (RANK_PRIME - term);
            }
        }
        if (rank == cap) return rank;
        begin = end;
    }
    return rank;
}

struct ModularRankResult {
    unsigned maximum_lower_bound = 0;
    unsigned witness_cut = 0;
};

struct ModularProfile {
    std::array<unsigned, 41> ranks{};
};

static ModularProfile modp_rank_profile(
    const std::vector<WeightedEntry>& entries,
    const std::array<uint8_t, 42>& order, unsigned cap) {
    ModularProfile result;
    for (unsigned cut = 1; cut < 42; cut++)
        result.ranks[cut - 1] = modp_projected_unfolding_rank(
            entries, order, cut, cap);
    return result;
}

static U128 optimistic_dense_tt_operations(const ModularProfile& left,
                                            const ModularProfile& right) {
    U128 operations = 0;
    for (unsigned bit = 0; bit < 42; bit++) {
        uint64_t left_before = bit ? left.ranks[bit - 1] : 1;
        uint64_t left_after = bit < 41 ? left.ranks[bit] : 1;
        uint64_t right_before = bit ? right.ranks[bit - 1] : 1;
        uint64_t right_after = bit < 41 ? right.ranks[bit] : 1;
        // Three compatible local bit pairs.  This is the usual two-step
        // dense contraction A^T E B; using capped lower ranks makes it an
        // intentionally optimistic screening estimate.
        operations += U128(3) *
            (U128(left_before) * left_after * right_before +
             U128(left_after) * right_before * right_after);
    }
    return operations;
}

static ModularRankResult modp_rank_census(
    const std::vector<WeightedEntry>& entries,
    const std::array<uint8_t, 42>& order, unsigned cap) {
    ModularRankResult result;
    // Start at the balanced unfolding, then move outwards.  High-rank orders
    // normally hit the cap immediately and avoid 40 unnecessary reductions.
    for (unsigned distance = 0; distance < 21; distance++) {
        unsigned cuts[2] = {21 - distance, 21 + distance};
        unsigned count = distance ? 2 : 1;
        for (unsigned index = 0; index < count; index++) {
            unsigned cut = cuts[index];
            if (!cut || cut >= 42) continue;
            unsigned rank = modp_projected_unfolding_rank(
                entries, order, cut, cap);
            if (rank > result.maximum_lower_bound) {
                result.maximum_lower_bound = rank;
                result.witness_cut = cut;
            }
            if (rank == cap) return result;
        }
    }
    return result;
}

static RankResult gf2_unfolding_rank(
    const std::vector<WeightedEntry>& entries,
    const std::array<uint8_t, 42>& order, unsigned cut, unsigned cap) {
    std::unordered_map<uint64_t, unsigned> row_ids, column_ids;
    struct Edge { unsigned row, column; };
    std::vector<std::pair<uint64_t, uint64_t>> assignments;
    assignments.reserve(entries.size());
    for (const auto& entry : entries) {
        if (!(entry.weight & 1U)) continue;
        uint64_t row_key = 0, column_key = 0;
        for (unsigned position = 0; position < 42; position++) {
            uint64_t bit = (entry.suffix >> order[position]) & 1U;
            if (position < cut)
                row_key |= bit << position;
            else
                column_key |= bit << (position - cut);
        }
        assignments.emplace_back(row_key, column_key);
        row_ids.emplace(row_key, 0);
        column_ids.emplace(column_key, 0);
    }
    unsigned next = 0;
    for (auto& item : row_ids) item.second = next++;
    next = 0;
    for (auto& item : column_ids) item.second = next++;
    std::vector<Edge> edges;
    edges.reserve(assignments.size());
    for (const auto& item : assignments)
        edges.push_back(Edge{row_ids.at(item.first),
                             column_ids.at(item.second)});

    const unsigned row_count = unsigned(row_ids.size());
    const unsigned column_count = unsigned(column_ids.size());
    std::vector<std::vector<unsigned>> row_edges(row_count),
        column_edges(column_count);
    for (const Edge& edge : edges) {
        row_edges[edge.row].push_back(edge.column);
        column_edges[edge.column].push_back(edge.row);
    }
    std::vector<unsigned> row_degree(row_count), column_degree(column_count);
    std::vector<uint8_t> row_active(row_count, 1),
        column_active(column_count, 1);
    std::deque<std::pair<bool, unsigned>> queue;
    for (unsigned row = 0; row < row_count; row++) {
        row_degree[row] = unsigned(row_edges[row].size());
        if (row_degree[row] == 1) queue.emplace_back(false, row);
    }
    for (unsigned column = 0; column < column_count; column++) {
        column_degree[column] = unsigned(column_edges[column].size());
        if (column_degree[column] == 1) queue.emplace_back(true, column);
    }

    unsigned rank = 0;
    auto remove_pivot = [&](unsigned pivot_row, unsigned pivot_column) {
        row_active[pivot_row] = 0;
        column_active[pivot_column] = 0;
        for (unsigned column : row_edges[pivot_row]) {
            if (!column_active[column]) continue;
            if (--column_degree[column] == 1)
                queue.emplace_back(true, column);
        }
        for (unsigned row : column_edges[pivot_column]) {
            if (!row_active[row]) continue;
            if (--row_degree[row] == 1)
                queue.emplace_back(false, row);
        }
        rank++;
    };
    while (!queue.empty() && rank < cap) {
        auto [is_column, index] = queue.front();
        queue.pop_front();
        if (is_column) {
            if (!column_active[index] || column_degree[index] != 1) continue;
            unsigned row = UINT32_MAX;
            for (unsigned candidate : column_edges[index])
                if (row_active[candidate]) { row = candidate; break; }
            if (row != UINT32_MAX) remove_pivot(row, index);
        } else {
            if (!row_active[index] || row_degree[index] != 1) continue;
            unsigned column = UINT32_MAX;
            for (unsigned candidate : row_edges[index])
                if (column_active[candidate]) { column = candidate; break; }
            if (column != UINT32_MAX) remove_pivot(index, column);
        }
    }
    if (rank >= cap)
        return RankResult{rank, row_count, column_count,
                          unsigned(edges.size()), true};

    std::vector<unsigned> active_rows, active_columns;
    for (unsigned row = 0; row < row_count; row++)
        if (row_active[row] && row_degree[row]) active_rows.push_back(row);
    for (unsigned column = 0; column < column_count; column++)
        if (column_active[column] && column_degree[column])
            active_columns.push_back(column);
    if (active_rows.empty() || active_columns.empty())
        return RankResult{rank, row_count, column_count,
                          unsigned(edges.size()), false};

    // Complete the exact rank on the residual 2-core.  Peeling usually
    // removes the matching-like balanced cuts; cap memory defensively.
    uint64_t words = (active_columns.size() + 63) / 64;
    uint64_t bytes = uint64_t(active_rows.size()) * words * sizeof(uint64_t);
    if (bytes > (UINT64_C(512) << 20))
        return RankResult{rank, row_count, column_count,
                          unsigned(edges.size()), true};
    std::vector<int> new_column(column_count, -1);
    for (size_t index = 0; index < active_columns.size(); index++)
        new_column[active_columns[index]] = int(index);
    std::vector<std::vector<uint64_t>> matrix(
        active_rows.size(), std::vector<uint64_t>(size_t(words), 0));
    for (size_t output_row = 0; output_row < active_rows.size(); output_row++) {
        for (unsigned column : row_edges[active_rows[output_row]]) {
            int mapped = new_column[column];
            if (mapped >= 0)
                matrix[output_row][unsigned(mapped) >> 6] ^=
                    UINT64_C(1) << (unsigned(mapped) & 63);
        }
    }
    size_t pivot_row = 0;
    for (size_t column = 0;
         column < active_columns.size() && pivot_row < matrix.size() &&
         rank < cap;
         column++) {
        size_t selected = pivot_row;
        while (selected < matrix.size() &&
               !(matrix[selected][column >> 6] &
                 (UINT64_C(1) << (column & 63))))
            selected++;
        if (selected == matrix.size()) continue;
        std::swap(matrix[pivot_row], matrix[selected]);
        for (size_t row = pivot_row + 1; row < matrix.size(); row++) {
            if (!(matrix[row][column >> 6] &
                  (UINT64_C(1) << (column & 63))))
                continue;
            for (size_t word = column >> 6; word < size_t(words); word++)
                matrix[row][word] ^= matrix[pivot_row][word];
        }
        pivot_row++;
        rank++;
    }
    bool capped = rank >= cap &&
                  pivot_row < std::min(active_rows.size(),
                                       active_columns.size());
    return RankResult{rank, row_count, column_count,
                      unsigned(edges.size()), capped};
}

static std::array<uint8_t, 42> colour_major_order() {
    std::array<uint8_t, 42> order{};
    std::iota(order.begin(), order.end(), uint8_t(0));
    return order;
}

static std::array<uint8_t, 42> pair_interleaved_order(bool reverse) {
    std::array<uint8_t, 42> order{};
    unsigned output = 0;
    for (unsigned step = 0; step < 21; step++) {
        unsigned pair_position = reverse ? 20 - step : step;
        order[output++] = uint8_t(pair_position);
        order[output++] = uint8_t(21 + pair_position);
    }
    return order;
}

static std::array<uint8_t, 42> lexicographic_pair_order() {
    std::array<unsigned, PAIRS> suffix_position{};
    suffix_position.fill(UINT32_MAX);
    unsigned position = 0;
    for (unsigned pair = 0; pair < PAIRS; pair++)
        if (production_pair_rank(pair) >= 7)
            suffix_position[pair] = position++;
    std::array<uint8_t, 42> order{};
    unsigned output = 0;
    for (unsigned pair = 0; pair < PAIRS; pair++) {
        if (suffix_position[pair] == UINT32_MAX) continue;
        order[output++] = uint8_t(suffix_position[pair]);
        order[output++] = uint8_t(21 + suffix_position[pair]);
    }
    return order;
}

static std::array<uint8_t, 42> frequency_order(
    const std::vector<WeightedEntry>& entries, bool balanced_first) {
    std::array<uint64_t, 42> frequency{};
    uint64_t support = 0;
    for (const auto& entry : entries) {
        support++;
        for (unsigned bit = 0; bit < 42; bit++)
            frequency[bit] += (entry.suffix >> bit) & 1U;
    }
    std::array<uint8_t, 42> order{};
    std::iota(order.begin(), order.end(), uint8_t(0));
    std::sort(order.begin(), order.end(), [&](uint8_t a, uint8_t b) {
        uint64_t score_a =
            std::min(frequency[a], support - frequency[a]);
        uint64_t score_b =
            std::min(frequency[b], support - frequency[b]);
        if (score_a != score_b)
            return balanced_first ? score_a > score_b : score_a < score_b;
        return a < b;
    });
    return order;
}

static std::array<uint8_t, 42> joint_frequency_order(
    const std::vector<WeightedEntry>& left,
    const std::vector<WeightedEntry>& right, bool balanced_first) {
    std::array<uint64_t, 42> frequency{};
    uint64_t support = 0;
    for (const auto* entries : {&left, &right}) {
        support += entries->size();
        for (const auto& entry : *entries)
            for (unsigned bit = 0; bit < 42; bit++)
                frequency[bit] += (entry.suffix >> bit) & 1U;
    }
    std::array<uint8_t, 42> order{};
    std::iota(order.begin(), order.end(), uint8_t(0));
    std::sort(order.begin(), order.end(), [&](uint8_t a, uint8_t b) {
        uint64_t score_a =
            std::min(frequency[a], support - frequency[a]);
        uint64_t score_b =
            std::min(frequency[b], support - frequency[b]);
        if (score_a != score_b)
            return balanced_first ? score_a > score_b : score_a < score_b;
        return a < b;
    });
    return order;
}

static const BucketedDistribution& component(
    const WeightedDistributionPair& pair, unsigned selected) {
    return selected ? pair.complement : pair.selected;
}

}  // namespace

#ifndef PREFIX_BUCKET_TT_RANK_CENSUS_NO_MAIN
int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 5) {
            std::cerr << "usage: " << argv[0]
                      << " SHARD.orbits [SAMPLE_RECORDS=64] [BUCKETS=24]"
                         " [RANK_CAP=1024]\n";
            return 2;
        }
        const std::string path = argv[1];
        uint64_t sample_records = argc > 2 ? std::stoull(argv[2]) : 64;
        size_t wanted_buckets = argc > 3 ? std::stoull(argv[3]) : 24;
        unsigned rank_cap = argc > 4 ? unsigned(std::stoul(argv[4])) : 1024;
        if (!sample_records || !wanted_buckets || !rank_cap) return 2;

        double start = seconds_now();
        initialise_tables();
        initialise_weighted_increments();
        std::vector<SampleRecord> records =
            read_stride_sample(path, sample_records);
        std::vector<uint32_t> prefixes;
        for (const auto& record : records) {
            prefixes.push_back(half_prefix(record.key, 0));
            prefixes.push_back(half_prefix(record.key, HALF_COLUMNS));
        }
        std::sort(prefixes.begin(), prefixes.end());
        prefixes.erase(std::unique(prefixes.begin(), prefixes.end()),
                       prefixes.end());
        std::vector<WeightedDistributionPair> distributions(prefixes.size());
        double build_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
        for (long long index = 0; index < (long long)prefixes.size(); index++)
            distributions[size_t(index)] = build_weighted_pair(prefixes[index]);
        double build_seconds = seconds_now() - build_start;
        size_t validation_prefixes = std::min<size_t>(8, prefixes.size());
        for (size_t index = 0; index < validation_prefixes; index++) {
            validate_weighted_support(prefixes[index], false,
                                      distributions[index].selected);
            validate_weighted_support(prefixes[index], true,
                                      distributions[index].complement);
        }
        std::cout << "TT_VALIDATE prefixes=" << validation_prefixes
                  << " selected_and_complement_support=OK"
                     " positive_weights=OK\n";
        std::unordered_map<uint32_t, size_t> prefix_index;
        for (size_t index = 0; index < prefixes.size(); index++)
            prefix_index.emplace(prefixes[index], index);

        std::vector<Candidate> candidates;
        uint64_t total_entries = 0;
        for (size_t distribution = 0; distribution < distributions.size();
             distribution++) {
            for (unsigned selected = 0; selected < 2; selected++) {
                auto& buckets = selected
                    ? distributions[distribution].complement.buckets
                    : distributions[distribution].selected.buckets;
                for (size_t bucket = 0; bucket < buckets.size(); bucket++) {
                    buckets[bucket].candidate = candidates.size();
                    total_entries += buckets[bucket].entries.size();
                    candidates.push_back(
                        Candidate{distribution, selected, bucket, 0});
                }
            }
        }

        U128 sampled_work = 0;
        U128 predicate_tiles = 0;
        U128 weight_class_tiles = 0;
        U128 weight_class_pairs = 0;
        uint64_t compatible_bucket_pairs = 0;
        constexpr size_t JOIN_CENSUS_SIZE = 12;
        std::priority_queue<JoinCandidate, std::vector<JoinCandidate>,
                            JoinCandidateGreater> heavy_joins;
        for (const auto& record : records) {
            size_t left_index = prefix_index.at(half_prefix(record.key, 0));
            size_t right_index =
                prefix_index.at(half_prefix(record.key, HALF_COLUMNS));
            for (unsigned selected = 0; selected < 2; selected++) {
                auto& left = selected
                    ? distributions[left_index].complement
                    : distributions[left_index].selected;
                auto& right = selected
                    ? distributions[right_index].complement
                    : distributions[right_index].selected;
                for (size_t left_bucket_index = 0;
                     left_bucket_index < left.buckets.size();
                     left_bucket_index++) {
                    auto& left_bucket = left.buckets[left_bucket_index];
                    for (size_t right_bucket_index = 0;
                         right_bucket_index < right.buckets.size();
                         right_bucket_index++) {
                        auto& right_bucket = right.buckets[right_bucket_index];
                        if (left_bucket.prefix & right_bucket.prefix) continue;
                        U128 work = U128(left_bucket.entries.size()) *
                                    right_bucket.entries.size();
                        sampled_work += work;
                        predicate_tiles += predicate_bmma_tiles(
                            left_bucket.entries.size(),
                            right_bucket.entries.size());
                        weight_class_tiles += weight_class_bmma_tiles(
                            left_bucket, right_bucket);
                        weight_class_pairs +=
                            U128(left_bucket.weight_class_counts.size()) *
                            right_bucket.weight_class_counts.size();
                        compatible_bucket_pairs++;
                        long double score = (long double)work;
                        candidates[left_bucket.candidate].work_score += score;
                        candidates[right_bucket.candidate].work_score += score;
                        JoinCandidate join{left_index, right_index, selected,
                                           left_bucket_index,
                                           right_bucket_index, work};
                        if (heavy_joins.size() < JOIN_CENSUS_SIZE)
                            heavy_joins.push(join);
                        else if (work > heavy_joins.top().work) {
                            heavy_joins.pop();
                            heavy_joins.push(join);
                        }
                    }
                }
            }
        }
        std::vector<JoinCandidate> joins;
        while (!heavy_joins.empty()) {
            joins.push_back(heavy_joins.top());
            heavy_joins.pop();
        }
        std::reverse(joins.begin(), joins.end());
        uint64_t total_weight_classes = 0;
        uint64_t multiclass_buckets = 0;
        size_t maximum_weight_classes = 0;
        std::unordered_set<uint32_t> distinct_weights;
        for (const auto& pair : distributions) {
            for (const auto* distribution : {&pair.selected,
                                             &pair.complement}) {
                for (const auto& bucket : distribution->buckets) {
                    total_weight_classes +=
                        bucket.weight_class_counts.size();
                    multiclass_buckets +=
                        bucket.weight_class_counts.size() > 1;
                    maximum_weight_classes = std::max(
                        maximum_weight_classes,
                        bucket.weight_class_counts.size());
                    for (const auto& entry : bucket.entries)
                        distinct_weights.insert(entry.weight);
                }
            }
        }
        long double tile_inflation = predicate_tiles
            ? (long double)weight_class_tiles /
                  (long double)predicate_tiles
            : 0;
        long double baseline_fill = predicate_tiles
            ? (long double)sampled_work /
                  ((long double)predicate_tiles * 128.0L)
            : 0;
        long double class_fill = weight_class_tiles
            ? (long double)sampled_work /
                  ((long double)weight_class_tiles * 128.0L)
            : 0;
        std::cout << std::setprecision(12)
                  << "TT_WEIGHT_CLASS compatible_bucket_pairs="
                  << compatible_bucket_pairs
                  << " class_pairs=" << u128_string(weight_class_pairs)
                  << " predicate_tiles=" << u128_string(predicate_tiles)
                  << " class_tiles=" << u128_string(weight_class_tiles)
                  << " tile_inflation=" << double(tile_inflation)
                  << " baseline_fill=" << double(baseline_fill)
                  << " class_fill=" << double(class_fill)
                  << " stored_buckets=" << candidates.size()
                  << " stored_classes=" << total_weight_classes
                  << " multiclass_buckets=" << multiclass_buckets
                  << " max_classes_per_bucket=" << maximum_weight_classes
                  << " distinct_weights=" << distinct_weights.size()
                  << '\n';
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) {
                      if (a.work_score != b.work_score)
                          return a.work_score > b.work_score;
                      return std::tie(a.distribution, a.component, a.bucket) <
                             std::tie(b.distribution, b.component, b.bucket);
                  });
        wanted_buckets = std::min(wanted_buckets, candidates.size());
        std::cout << std::setprecision(12)
                  << "TT_INPUT records=" << records.size()
                  << " unique_prefixes=" << prefixes.size()
                  << " distributions=" << prefixes.size() * 2
                  << " buckets=" << candidates.size()
                  << " entries=" << total_entries
                  << " sampled_suffix_work=" << u128_string(sampled_work)
                  << " build_seconds=" << build_seconds << '\n';

        unsigned globally_small = 0;
        unsigned globally_large = 0;
        for (size_t selected_index = 0; selected_index < wanted_buckets;
             selected_index++) {
            const Candidate& candidate = candidates[selected_index];
            const auto& distribution = component(
                distributions[candidate.distribution], candidate.component);
            const auto& bucket = distribution.buckets[candidate.bucket];
            std::vector<std::pair<std::string, std::array<uint8_t, 42>>> orders;
            orders.push_back({"colour-major", colour_major_order()});
            orders.push_back({"pair-interleaved", pair_interleaved_order(false)});
            orders.push_back({"pair-interleaved-reverse",
                              pair_interleaved_order(true)});
            orders.push_back({"lex-pair-interleaved",
                              lexicographic_pair_order()});
            orders.push_back({"balanced-first",
                              frequency_order(bucket.entries, true)});
            orders.push_back({"unbalanced-first",
                              frequency_order(bucket.entries, false)});
            unsigned best_max = UINT32_MAX;
            std::string best_name;
            for (const auto& named_order : orders) {
                unsigned maximum = 0;
                unsigned middle = 0;
                unsigned first_cap = 0;
                unsigned odd_entries = 0;
                std::ostringstream ranks;
                for (unsigned cut = 1; cut < 42; cut++) {
                    RankResult result = gf2_unfolding_rank(
                        bucket.entries, named_order.second, cut, rank_cap);
                    maximum = std::max(maximum, result.rank);
                    if (cut == 21) middle = result.rank;
                    odd_entries = result.odd_entries;
                    if (result.capped && !first_cap) first_cap = cut;
                    if (cut > 1) ranks << ',';
                    ranks << result.rank << (result.capped ? "+" : "");
                }
                ModularRankResult modular = modp_rank_census(
                    bucket.entries, named_order.second, rank_cap);
                if (modular.maximum_lower_bound < best_max) {
                    best_max = modular.maximum_lower_bound;
                    best_name = named_order.first;
                }
                std::cout << "TT_RANK bucket_rank=" << selected_index
                          << " outer_prefix=" << prefixes[candidate.distribution]
                          << " component=" << candidate.component
                          << " token_prefix=" << bucket.prefix
                          << " support=" << bucket.entries.size()
                          << " odd_entries=" << odd_entries
                          << " work_score=" << double(candidate.work_score)
                          << " order=" << named_order.first
                          << " max_rank=" << maximum
                          << " middle_rank=" << middle
                          << " first_cap_cut=" << first_cap
                          << " modp_max_lower_bound="
                          << modular.maximum_lower_bound
                          << " modp_witness_cut=" << modular.witness_cut
                          << " modp_capped="
                          << (modular.maximum_lower_bound == rank_cap)
                          << " ranks=" << ranks.str() << '\n';
            }
            globally_small += best_max <= 128;
            globally_large += best_max >= rank_cap;
            std::cout << "TT_BUCKET bucket_rank=" << selected_index
                      << " support=" << bucket.entries.size()
                      << " best_order=" << best_name
                      << " best_max_rank=" << best_max << '\n';
        }
        for (size_t join_index = 0; join_index < joins.size(); join_index++) {
            const JoinCandidate& join = joins[join_index];
            const auto& left_distribution = component(
                distributions[join.left_distribution], join.component);
            const auto& right_distribution = component(
                distributions[join.right_distribution], join.component);
            const auto& left = left_distribution.buckets[join.left_bucket];
            const auto& right = right_distribution.buckets[join.right_bucket];
            std::vector<std::pair<std::string, std::array<uint8_t, 42>>> orders;
            orders.push_back({"pair-interleaved",
                              pair_interleaved_order(false)});
            orders.push_back({"colour-major", colour_major_order()});
            orders.push_back({"joint-balanced-first",
                              joint_frequency_order(left.entries,
                                                    right.entries, true)});
            orders.push_back({"joint-unbalanced-first",
                              joint_frequency_order(left.entries,
                                                    right.entries, false)});
            for (const auto& named_order : orders) {
                ModularRankResult left_rank = modp_rank_census(
                    left.entries, named_order.second, rank_cap);
                ModularRankResult right_rank = modp_rank_census(
                    right.entries, named_order.second, rank_cap);
                std::cout << "TT_JOIN join_rank=" << join_index
                          << " work=" << u128_string(join.work)
                          << " left_support=" << left.entries.size()
                          << " right_support=" << right.entries.size()
                          << " component=" << join.component
                          << " order=" << named_order.first
                          << " left_max_lower_bound="
                          << left_rank.maximum_lower_bound
                          << " left_witness_cut=" << left_rank.witness_cut
                          << " right_max_lower_bound="
                          << right_rank.maximum_lower_bound
                          << " right_witness_cut=" << right_rank.witness_cut
                          << '\n';
                if (join_index < 3) {
                    ModularProfile left_profile = modp_rank_profile(
                        left.entries, named_order.second, rank_cap);
                    ModularProfile right_profile = modp_rank_profile(
                        right.entries, named_order.second, rank_cap);
                    unsigned max_rank_product = 0;
                    for (unsigned cut = 0; cut < 41; cut++)
                        max_rank_product = std::max(
                            max_rank_product,
                            left_profile.ranks[cut] *
                                right_profile.ranks[cut]);
                    std::cout << "TT_JOIN_COST join_rank=" << join_index
                              << " order=" << named_order.first
                              << " direct_comparisons="
                              << u128_string(join.work)
                              << " max_same_cut_rank_product="
                              << max_rank_product
                              << " optimistic_dense_operations="
                              << u128_string(optimistic_dense_tt_operations(
                                     left_profile, right_profile))
                              << " profile_cap=" << rank_cap << '\n';
                }
            }
        }
        std::cout << "TT_SUMMARY tested_buckets=" << wanted_buckets
                  << " best_max_le_128=" << globally_small
                  << " best_max_at_cap=" << globally_large
                  << " rank_cap=" << rank_cap
                  << " total_seconds=" << seconds_now() - start
                  << " exact_gf2=OK modp_lower_bound=OK"
                     " modp_prime=" << RANK_PRIME << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
#endif
