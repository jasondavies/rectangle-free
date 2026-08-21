#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Diagnostic-only CPU oracle for adaptive 7x9 prefix splits.  It deliberately
// shares no state with the production CUDA solver.

namespace {

constexpr int ROWS = 7;
constexpr int COLUMNS = 9;
constexpr int LEFT_COLUMNS = 4;
constexpr int RIGHT_COLUMNS = 5;
constexpr int ROW_BITS = 9;
constexpr int PAIRS = ROWS * (ROWS - 1) / 2;
constexpr uint32_t PAIR_MASK = (uint32_t(1) << PAIRS) - 1;
constexpr uint32_t SUBSET_COUNT = uint32_t(1) << PAIRS;
constexpr size_t RESERVOIR_SIZE = 8;
constexpr char ORBIT_MAGIC[] = "R7ORB09";

using PrefixKey = uint64_t;

struct OrbitRecord {
    uint64_t key;
    uint64_t weight;
};

struct Entry {
    uint64_t mask;
    uint64_t weight;
};

struct Increment {
    uint64_t mask;
    uint16_t weight;
};

struct MapEntry {
    uint64_t mask = 0;
    uint64_t weight = 0;
    bool used = false;
};

struct Distribution {
    std::vector<Entry> entries;
};

struct DistributionPair {
    Distribution selected;
    Distribution complement;
};

struct CanonicalForm {
    PrefixKey key;
    uint32_t row_map;
};

struct Observation {
    PrefixKey left;
    PrefixKey right;
    CanonicalForm selected;
    CanonicalForm complement;
};

struct GroupKey {
    PrefixKey first;
    PrefixKey second;

    bool operator==(const GroupKey& other) const {
        return first == other.first && second == other.second;
    }
};

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

struct GroupKeyHash {
    size_t operator()(const GroupKey& key) const {
        return size_t(mix64(key.first ^ mix64(key.second)));
    }
};

struct Group {
    GroupKey key{};
    uint64_t observations = 0;
    std::vector<Observation> reservoir;
};

struct SelectedGroup {
    size_t index;
    double sampling_weight;
};

class Map {
  public:
    explicit Map(size_t capacity = 16) : entries_(capacity) {}

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
        entries_[slot] = MapEntry{mask, weight, true};
        count_++;
    }

    const std::vector<MapEntry>& entries() const { return entries_; }
    size_t count() const { return count_; }

  private:
    void insert_raw(const MapEntry& entry) {
        size_t slot = size_t(mix64(entry.mask)) & (entries_.size() - 1);
        while (entries_[slot].used) slot = (slot + 1) & (entries_.size() - 1);
        entries_[slot] = entry;
        count_++;
    }

    void rehash() {
        std::vector<MapEntry> old = std::move(entries_);
        entries_.assign(old.size() * 2, MapEntry{});
        count_ = 0;
        for (const MapEntry& entry : old) {
            if (entry.used) insert_raw(entry);
        }
    }

    std::vector<MapEntry> entries_;
    size_t count_ = 0;
};

static std::array<std::vector<Increment>, 1 << ROWS> g_increments;

static double seconds_now() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

static int pair_number(int first, int second) {
    if (first > second) std::swap(first, second);
    return first * (2 * ROWS - first - 1) / 2 + second - first - 1;
}

static void initialise_tables() {
    for (unsigned active = 0; active < (1U << ROWS); active++) {
        auto& increments = g_increments[active];
        unsigned assignment = active;
        for (;;) {
            uint64_t mask = 0;
            for (int u = 0; u < ROWS; u++) {
                for (int v = u + 1; v < ROWS; v++) {
                    if (!(active & (1U << u)) || !(active & (1U << v))) continue;
                    unsigned cu = (assignment >> u) & 1U;
                    unsigned cv = (assignment >> v) & 1U;
                    if (cu == cv) {
                        mask |= UINT64_C(1)
                                << (cu * PAIRS + pair_number(u, v));
                    }
                }
            }
            increments.push_back(Increment{mask, 1});
            if (!assignment) break;
            assignment = (assignment - 1U) & active;
        }
        std::sort(increments.begin(), increments.end(),
                  [](const Increment& a, const Increment& b) {
                      return a.mask < b.mask;
                  });
        size_t unique = 0;
        for (const Increment& increment : increments) {
            if (unique && increments[unique - 1].mask == increment.mask) {
                increments[unique - 1].weight++;
            } else {
                increments[unique++] = increment;
            }
        }
        increments.resize(unique);
    }
}

static Distribution build_distribution(PrefixKey prefix, int columns,
                                       bool complement) {
    std::array<uint8_t, ROWS> rows{};
    const unsigned pattern_mask = (1U << columns) - 1U;
    for (int row = ROWS - 1; row >= 0; row--) {
        rows[size_t(row)] = uint8_t(prefix & pattern_mask);
        prefix >>= columns;
    }
    Map current;
    current.add(0, 1);
    for (int column = 0; column < columns; column++) {
        unsigned active = 0;
        for (int row = 0; row < ROWS; row++) {
            unsigned pattern = complement ? rows[size_t(row)] ^ pattern_mask
                                          : rows[size_t(row)];
            if (pattern & (1U << column)) active |= 1U << row;
        }
        Map next(std::max<size_t>(16, current.entries().size()));
        for (const MapEntry& entry : current.entries()) {
            if (!entry.used) continue;
            for (const Increment& increment : g_increments[active]) {
                if (entry.mask & increment.mask) continue;
                next.add(entry.mask | increment.mask,
                         entry.weight * increment.weight);
            }
        }
        current = std::move(next);
    }
    Distribution result;
    result.entries.reserve(current.count());
    for (const MapEntry& entry : current.entries()) {
        if (entry.used) result.entries.push_back(Entry{entry.mask, entry.weight});
    }
    return result;
}

static DistributionPair build_pair(PrefixKey prefix, int columns) {
    return DistributionPair{build_distribution(prefix, columns, false),
                            build_distribution(prefix, columns, true)};
}

static CanonicalForm canonical_prefix(PrefixKey prefix, int columns) {
    if (columns != RIGHT_COLUMNS) {
        throw std::logic_error("oracle canonicalises only five-column prefixes");
    }
    const uint32_t pattern_mask = (1U << columns) - 1U;
    std::array<uint8_t, ROWS> patterns{};
    for (int row = ROWS - 1; row >= 0; row--) {
        patterns[size_t(row)] = uint8_t(prefix & pattern_mask);
        prefix >>= columns;
    }
    std::array<int, RIGHT_COLUMNS> permutation{};
    for (int column = 0; column < columns; column++) {
        permutation[size_t(column)] = column;
    }
    PrefixKey best_key = UINT64_MAX;
    uint32_t best_row_map = 0;
    do {
        std::array<std::pair<uint8_t, uint8_t>, ROWS> rows{};
        for (int row = 0; row < ROWS; row++) {
            uint8_t transformed = 0;
            for (int column = 0; column < columns; column++) {
                if (patterns[size_t(row)] & (1U << column)) {
                    transformed |= uint8_t(1U << permutation[size_t(column)]);
                }
            }
            rows[size_t(row)] = {transformed, uint8_t(row)};
        }
        std::sort(rows.begin(), rows.end());
        PrefixKey key = 0;
        uint32_t row_map = 0;
        for (int row = 0; row < ROWS; row++) {
            key = (key << columns) | rows[size_t(row)].first;
            row_map |= uint32_t(rows[size_t(row)].second) << (4 * row);
        }
        if (key < best_key) {
            best_key = key;
            best_row_map = row_map;
        }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    return CanonicalForm{best_key, best_row_map};
}

static uint32_t inverse_row_map(uint32_t row_map) {
    uint32_t inverse = 0;
    for (int canonical = 0; canonical < ROWS; canonical++) {
        int raw = int((row_map >> (4 * canonical)) & 15U);
        inverse |= uint32_t(canonical) << (4 * raw);
    }
    return inverse;
}

static uint64_t transform_pair_mask(uint64_t mask, uint32_t row_map) {
    uint64_t transformed = 0;
    for (int colour = 0; colour < 2; colour++) {
        for (int first = 0; first < ROWS; first++) {
            int image_first = int((row_map >> (4 * first)) & 15U);
            for (int second = first + 1; second < ROWS; second++) {
                int source = colour * PAIRS + pair_number(first, second);
                if (!(mask & (UINT64_C(1) << source))) continue;
                int image_second = int((row_map >> (4 * second)) & 15U);
                int destination = colour * PAIRS +
                                  pair_number(image_first, image_second);
                transformed |= UINT64_C(1) << destination;
            }
        }
    }
    return transformed;
}

static PrefixKey left_prefix(uint64_t key) {
    PrefixKey result = 0;
    for (int row = 0; row < ROWS; row++) {
        unsigned shift = ROW_BITS * (ROWS - 1U - row);
        result = (result << LEFT_COLUMNS) |
                 PrefixKey((key >> shift) & ((1U << LEFT_COLUMNS) - 1U));
    }
    return result;
}

static PrefixKey right_prefix(uint64_t key) {
    PrefixKey result = 0;
    for (int row = 0; row < ROWS; row++) {
        unsigned shift = ROW_BITS * (ROWS - 1U - row);
        result = (result << RIGHT_COLUMNS) |
                 PrefixKey((key >> (shift + LEFT_COLUMNS)) &
                           ((1U << RIGHT_COLUMNS) - 1U));
    }
    return result;
}

static int cell_count(uint64_t key) {
    int result = 0;
    const uint64_t row_mask = (UINT64_C(1) << ROW_BITS) - 1U;
    for (int row = 0; row < ROWS; row++) {
        result += __builtin_popcountll(key & row_mask);
        key >>= ROW_BITS;
    }
    return result;
}

static uint64_t random_bounded(uint64_t& state, uint64_t bound) {
    state = mix64(state + UINT64_C(0x9e3779b97f4a7c15));
    return bound ? state % bound : 0;
}

static std::vector<Group> read_groups(const std::string& path,
                                      uint64_t maximum_records) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path);
    char magic[8];
    uint32_t columns = 0;
    uint64_t file_count = 0;
    input.read(magic, sizeof(magic));
    input.read(reinterpret_cast<char*>(&columns), sizeof(columns));
    input.read(reinterpret_cast<char*>(&file_count), sizeof(file_count));
    if (!input || std::memcmp(magic, ORBIT_MAGIC, 7) || columns != COLUMNS) {
        throw std::runtime_error("invalid 7x9 orbit file");
    }
    const uint64_t count = maximum_records && maximum_records < file_count
                               ? maximum_records
                               : file_count;
    const bool strided = count < file_count;

    std::unordered_map<PrefixKey, std::pair<CanonicalForm, CanonicalForm>> forms;
    forms.reserve(size_t(std::min<uint64_t>(count, 1000000)));
    std::unordered_map<GroupKey, size_t, GroupKeyHash> indices;
    indices.reserve(size_t(std::min<uint64_t>(count, 1000000)));
    std::vector<Group> groups;
    const PrefixKey full_mask = (PrefixKey(1) << (RIGHT_COLUMNS * ROWS)) - 1;
    uint64_t retained = 0;
    for (uint64_t index = 0; index < count; index++) {
        if (strided) {
            const uint64_t quotient = file_count / count;
            const uint64_t remainder = file_count % count;
            const uint64_t source_index =
                index * quotient + (index * remainder) / count;
            input.seekg(std::streamoff(20 + source_index * sizeof(OrbitRecord)));
        }
        OrbitRecord record{};
        input.read(reinterpret_cast<char*>(&record), sizeof(record));
        if (!input) throw std::runtime_error("truncated orbit file");
        if (cell_count(record.key) * 2 > ROWS * COLUMNS) continue;
        retained++;
        PrefixKey right = right_prefix(record.key);
        auto found_forms = forms.find(right);
        if (found_forms == forms.end()) {
            found_forms = forms.emplace(
                right, std::make_pair(canonical_prefix(right, RIGHT_COLUMNS),
                                      canonical_prefix(right ^ full_mask,
                                                       RIGHT_COLUMNS)))
                              .first;
        }
        CanonicalForm selected = found_forms->second.first;
        CanonicalForm complement = found_forms->second.second;
        GroupKey key{std::min(selected.key, complement.key),
                     std::max(selected.key, complement.key)};
        auto found = indices.find(key);
        if (found == indices.end()) {
            size_t group_index = groups.size();
            groups.push_back(Group{key, 0, {}});
            found = indices.emplace(key, group_index).first;
        }
        Group& group = groups[found->second];
        group.observations++;
        Observation observation{left_prefix(record.key), right, selected,
                                complement};
        if (group.reservoir.size() < RESERVOIR_SIZE) {
            group.reservoir.push_back(observation);
        } else {
            uint64_t state = mix64(key.first ^ mix64(key.second) ^
                                   group.observations);
            uint64_t slot = state % group.observations;
            if (slot < RESERVOIR_SIZE) group.reservoir[size_t(slot)] = observation;
        }
    }
    std::cerr << "GROUP_READ records=" << count << " retained=" << retained
              << " raw_rights=" << forms.size() << " groups=" << groups.size()
              << '\n';
    return groups;
}

static std::vector<SelectedGroup> select_groups(const std::vector<Group>& groups,
                                                size_t wanted) {
    wanted = std::min(wanted, groups.size());
    size_t top_count = wanted / 2;
    std::vector<size_t> by_count(groups.size());
    std::iota(by_count.begin(), by_count.end(), 0);
    std::partial_sort(by_count.begin(), by_count.begin() + top_count,
                      by_count.end(), [&](size_t a, size_t b) {
                          if (groups[a].observations != groups[b].observations) {
                              return groups[a].observations > groups[b].observations;
                          }
                          return GroupKeyHash{}(groups[a].key) <
                                 GroupKeyHash{}(groups[b].key);
                      });
    std::vector<SelectedGroup> selected;
    selected.reserve(wanted);
    std::unordered_set<size_t> used;
    for (auto iterator = by_count.begin(); iterator != by_count.begin() + top_count;
         ++iterator) {
        selected.push_back(SelectedGroup{*iterator, 1.0});
        used.insert(*iterator);
    }
    std::vector<size_t> by_hash;
    by_hash.reserve(groups.size() - used.size());
    for (size_t index = 0; index < groups.size(); index++) {
        if (!used.count(index)) by_hash.push_back(index);
    }
    std::partial_sort(by_hash.begin(), by_hash.begin() + (wanted - top_count),
                      by_hash.end(), [&](size_t a, size_t b) {
                          return GroupKeyHash{}(groups[a].key) <
                                 GroupKeyHash{}(groups[b].key);
                      });
    const size_t random_count = wanted - top_count;
    const double random_weight =
        random_count ? double(groups.size() - top_count) / random_count : 1.0;
    for (auto iterator = by_hash.begin(); iterator != by_hash.begin() + random_count;
         ++iterator) {
        selected.push_back(SelectedGroup{*iterator, random_weight});
    }
    return selected;
}

static std::vector<uint32_t> enumerate_candidates() {
    std::vector<uint32_t> result;
    for (int a = 0; a < PAIRS; a++)
        for (int b = a + 1; b < PAIRS; b++)
            for (int c = b + 1; c < PAIRS; c++)
                for (int d = c + 1; d < PAIRS; d++)
                    for (int e = d + 1; e < PAIRS; e++) {
                        result.push_back((uint32_t(1) << a) |
                                         (uint32_t(1) << b) |
                                         (uint32_t(1) << c) |
                                         (uint32_t(1) << d) |
                                         (uint32_t(1) << e));
                    }
    return result;
}

static std::string candidate_string(uint32_t candidate) {
    std::string result;
    for (int first = 0; first < ROWS; first++) {
        for (int second = first + 1; second < ROWS; second++) {
            int pair = pair_number(first, second);
            if (!(candidate & (uint32_t(1) << pair))) continue;
            if (!result.empty()) result.push_back(',');
            result += std::to_string(first);
            result += std::to_string(second);
        }
    }
    return result;
}

struct ProfiledGroup {
    GroupKey key{};
    uint64_t observations = 0;
    double direct = 0;
    double current = 0;
    std::vector<double> costs;
};

static void add_component_samples(
    std::vector<double>& histogram, const Distribution& left,
    const Distribution& canonical_right, uint32_t right_map, double scale,
    uint64_t samples, uint32_t current_mask, double& current_cost,
    uint64_t& random_state) {
    if (left.entries.empty() || canonical_right.entries.empty()) return;
    const long double product = static_cast<long double>(left.entries.size()) *
                                canonical_right.entries.size();
    const double sample_weight = double(product * scale / samples);
    uint32_t inverse = inverse_row_map(right_map);
    for (uint64_t sample = 0; sample < samples; sample++) {
        const Entry& lhs = left.entries[size_t(random_bounded(
            random_state, left.entries.size()))];
        const Entry& canonical = canonical_right.entries[size_t(random_bounded(
            random_state, canonical_right.entries.size()))];
        uint64_t rhs_mask = transform_pair_mask(canonical.mask, right_map);
        uint64_t overlap = lhs.mask & rhs_mask;
        uint32_t raw_conflicts =
            (uint32_t(overlap) | uint32_t(overlap >> PAIRS)) & PAIR_MASK;
        if (!(raw_conflicts & current_mask)) current_cost += sample_weight;
        uint64_t canonical_overlap = transform_pair_mask(overlap, inverse);
        uint32_t conflicts = uint32_t(canonical_overlap) |
                             uint32_t(canonical_overlap >> PAIRS);
        histogram[conflicts & PAIR_MASK] += sample_weight;
    }
}

static std::vector<ProfiledGroup> profile_group(
    const Group& group, const std::vector<uint32_t>& candidates,
    uint64_t samples_per_component, uint32_t current_mask,
    double sampling_weight,
    std::unordered_map<PrefixKey, DistributionPair>& left_cache,
    std::unordered_map<PrefixKey, Distribution>& right_cache) {
    const size_t component_count = group.key.first == group.key.second ? 1 : 2;
    std::vector<ProfiledGroup> results(component_count);
    std::vector<std::vector<double>> subsets(
        component_count, std::vector<double>(SUBSET_COUNT, 0.0));
    for (size_t component = 0; component < component_count; component++) {
        PrefixKey key = component ? group.key.second : group.key.first;
        results[component].key = GroupKey{key, key};
        results[component].observations = group.observations;
    }
    const double observation_scale =
        sampling_weight * double(group.observations) / group.reservoir.size();
    uint64_t state = mix64(group.key.first ^ mix64(group.key.second));
    for (const Observation& observation : group.reservoir) {
        auto left_found = left_cache.find(observation.left);
        if (left_found == left_cache.end()) {
            left_found = left_cache
                             .emplace(observation.left,
                                      build_pair(observation.left, LEFT_COLUMNS))
                             .first;
        }
        auto get_right = [&](PrefixKey key) -> const Distribution& {
            auto found = right_cache.find(key);
            if (found == right_cache.end()) {
                found = right_cache
                            .emplace(key, build_distribution(key, RIGHT_COLUMNS,
                                                             false))
                            .first;
            }
            return found->second;
        };
        const DistributionPair& left = left_found->second;
        const Distribution& selected = get_right(observation.selected.key);
        const Distribution& complement = get_right(observation.complement.key);
        const size_t selected_component =
            component_count == 1 || observation.selected.key == group.key.first
                ? 0
                : 1;
        const size_t complement_component =
            component_count == 1 || observation.complement.key == group.key.first
                ? 0
                : 1;
        results[selected_component].direct +=
            observation_scale * double(left.selected.entries.size()) *
            selected.entries.size();
        results[complement_component].direct +=
            observation_scale * double(left.complement.entries.size()) *
            complement.entries.size();
        add_component_samples(subsets[selected_component], left.selected, selected,
                              observation.selected.row_map, observation_scale,
                              samples_per_component, current_mask,
                              results[selected_component].current, state);
        add_component_samples(subsets[complement_component], left.complement,
                              complement,
                              observation.complement.row_map, observation_scale,
                              samples_per_component, current_mask,
                              results[complement_component].current, state);
    }
    for (size_t component = 0; component < component_count; component++) {
        std::vector<double>& subset = subsets[component];
        for (int bit = 0; bit < PAIRS; bit++) {
            uint32_t flag = uint32_t(1) << bit;
            for (uint32_t mask = 0; mask < SUBSET_COUNT; mask++) {
                if (mask & flag) subset[mask] += subset[mask ^ flag];
            }
        }
        const double normalization_error =
            std::abs(subset[PAIR_MASK] - results[component].direct);
        if (normalization_error >
            std::max(1.0, results[component].direct) * 1e-10) {
            throw std::logic_error("sample histogram does not normalize to direct work");
        }
        results[component].costs.resize(candidates.size());
        for (size_t index = 0; index < candidates.size(); index++) {
            results[component].costs[index] =
                subset[PAIR_MASK ^ candidates[index]];
        }
    }
    return results;
}

static double cost_for(const std::vector<ProfiledGroup>& groups,
                       const std::vector<size_t>& group_indices,
                       const std::vector<size_t>& portfolio) {
    double total = 0;
    for (size_t group_index : group_indices) {
        double best = std::numeric_limits<double>::infinity();
        for (size_t candidate : portfolio) {
            best = std::min(best, groups[group_index].costs[candidate]);
        }
        total += best;
    }
    return total;
}

static std::vector<size_t> greedy_portfolio(
    const std::vector<ProfiledGroup>& groups,
    const std::vector<size_t>& train_indices, size_t wanted) {
    if (groups.empty()) return {};
    const size_t candidates = groups.front().costs.size();
    std::vector<double> best(train_indices.size(),
                             std::numeric_limits<double>::infinity());
    std::vector<size_t> portfolio;
    std::vector<bool> used(candidates, false);
    while (portfolio.size() < wanted) {
        size_t best_candidate = 0;
        double best_total = std::numeric_limits<double>::infinity();
        for (size_t candidate = 0; candidate < candidates; candidate++) {
            if (used[candidate]) continue;
            double total = 0;
            for (size_t index = 0; index < train_indices.size(); index++) {
                total += std::min(best[index],
                                  groups[train_indices[index]].costs[candidate]);
            }
            if (total < best_total) {
                best_total = total;
                best_candidate = candidate;
            }
        }
        used[best_candidate] = true;
        portfolio.push_back(best_candidate);
        for (size_t index = 0; index < train_indices.size(); index++) {
            best[index] = std::min(
                best[index], groups[train_indices[index]].costs[best_candidate]);
        }
    }
    return portfolio;
}

static double current_cost(const std::vector<ProfiledGroup>& groups,
                           const std::vector<size_t>& indices) {
    double total = 0;
    for (size_t index : indices) total += groups[index].current;
    return total;
}

static double direct_cost(const std::vector<ProfiledGroup>& groups,
                          const std::vector<size_t>& indices) {
    double total = 0;
    for (size_t index : indices) total += groups[index].direct;
    return total;
}

static void report_split(const char* name,
                         const std::vector<ProfiledGroup>& groups,
                         const std::vector<size_t>& indices,
                         const std::vector<size_t>& portfolio) {
    double direct = direct_cost(groups, indices);
    double current = current_cost(groups, indices);
    double adaptive = cost_for(groups, indices, portfolio);
    double oracle = 0;
    for (size_t group_index : indices) {
        oracle += *std::min_element(groups[group_index].costs.begin(),
                                   groups[group_index].costs.end());
    }
    std::cout << "PORTFOLIO split=" << name << " size=" << portfolio.size()
              << " groups=" << indices.size() << " direct=" << direct
              << " current=" << current << " adaptive=" << adaptive
              << " oracle=" << oracle
              << " current_retained=" << current / direct
              << " adaptive_retained=" << adaptive / direct
              << " reduction_vs_current=" << 1.0 - adaptive / current
              << " oracle_reduction_vs_current=" << 1.0 - oracle / current
              << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 6) {
            std::cerr << "usage: " << argv[0]
                      << " ORBITS [groups=64] [samples_per_component=4096]"
                         " [max_records=0] [maximum_portfolio=16]\n";
            return 2;
        }
        const std::string path = argv[1];
        const size_t wanted_groups =
            argc > 2 ? size_t(std::stoull(argv[2])) : 64;
        const uint64_t samples_per_component =
            argc > 3 ? std::stoull(argv[3]) : 4096;
        const uint64_t maximum_records =
            argc > 4 ? std::stoull(argv[4]) : 0;
        const size_t maximum_portfolio =
            argc > 5 ? size_t(std::stoull(argv[5])) : 16;
        if (!wanted_groups || !samples_per_component || !maximum_portfolio) {
            throw std::runtime_error("numeric arguments must be positive");
        }

        initialise_tables();
        const double start = seconds_now();
        std::vector<Group> all_groups = read_groups(path, maximum_records);
        std::vector<SelectedGroup> selected =
            select_groups(all_groups, wanted_groups);
        std::vector<uint32_t> candidates = enumerate_candidates();
        if (candidates.size() != 20349) {
            throw std::logic_error("five-pair candidate census is incomplete");
        }
        const uint32_t current_mask = (uint32_t(1) << 0) |
                                      (uint32_t(1) << 1) |
                                      (uint32_t(1) << 2) |
                                      (uint32_t(1) << 3) |
                                      (uint32_t(1) << 6);
        std::cerr << "ORACLE_SETUP selected_groups=" << selected.size()
                  << " candidates=" << candidates.size()
                  << " samples_per_component=" << samples_per_component
                  << " current=" << candidate_string(current_mask) << '\n';

        std::unordered_map<PrefixKey, DistributionPair> left_cache;
        std::unordered_map<PrefixKey, Distribution> right_cache;
        left_cache.reserve(selected.size() * RESERVOIR_SIZE);
        right_cache.reserve(selected.size() * 2);
        std::vector<ProfiledGroup> profiled;
        profiled.reserve(selected.size() * 2);
        std::vector<bool> profile_train;
        profile_train.reserve(selected.size() * 2);
        for (size_t ordinal = 0; ordinal < selected.size(); ordinal++) {
            double group_start = seconds_now();
            std::vector<ProfiledGroup> components = profile_group(
                all_groups[selected[ordinal].index], candidates,
                samples_per_component, current_mask,
                selected[ordinal].sampling_weight, left_cache, right_cache);
            double component_direct = 0;
            for (ProfiledGroup& component : components) {
                component_direct += component.direct;
                profiled.push_back(std::move(component));
                profile_train.push_back((ordinal & 1) == 0);
            }
            std::cerr << "GROUP_PROFILE index=" << ordinal + 1 << '/'
                      << selected.size()
                      << " observations="
                      << all_groups[selected[ordinal].index].observations
                      << " sampling_weight="
                      << selected[ordinal].sampling_weight
                      << " components=" << components.size()
                      << " direct=" << component_direct
                      << " seconds=" << seconds_now() - group_start << '\n';
        }

        std::vector<size_t> train;
        std::vector<size_t> test;
        for (size_t index = 0; index < profiled.size(); index++) {
            (profile_train[index] ? train : test).push_back(index);
        }
        std::vector<size_t> all(profiled.size());
        std::iota(all.begin(), all.end(), 0);
        for (size_t size = 1; size <= maximum_portfolio; size *= 2) {
            std::vector<size_t> portfolio =
                greedy_portfolio(profiled, train, size);
            std::cout << "PORTFOLIO_CONFIG size=" << size;
            for (size_t candidate : portfolio) {
                std::cout << " config=" << candidate_string(candidates[candidate]);
            }
            std::cout << '\n';
            report_split("train", profiled, train, portfolio);
            report_split("test", profiled, test, portfolio);
            report_split("all", profiled, all, portfolio);
        }
        std::cout << "ORACLE_DONE groups=" << profiled.size()
                  << " train=" << train.size() << " test=" << test.size()
                  << " left_cache=" << left_cache.size()
                  << " right_cache=" << right_cache.size()
                  << " seconds=" << seconds_now() - start << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
