#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using U128 = unsigned __int128;

constexpr unsigned ROWS = 8;
constexpr unsigned COLUMNS = 8;
constexpr unsigned HALF_COLUMNS = 4;
constexpr unsigned PAIRS = ROWS * (ROWS - 1) / 2;

struct OrbitRecord { uint64_t key, weight; };
struct Increment { uint64_t mask; };
struct MapEntry { uint64_t mask = 0; bool used = false; };
struct DistributionPair {
    std::vector<uint64_t> selected, complement;
};
struct SampleRecord { uint64_t index, key; };
struct JoinView {
    const std::vector<uint64_t>* left;
    const std::vector<uint64_t>* right;
    uint64_t seed;
};
struct Level3 { uint8_t prefix; uint64_t count; };
struct Level2 {
    uint16_t prefix;
    uint64_t count;
    std::vector<Level3> children;
};
struct Level1 {
    uint16_t prefix;
    uint64_t count;
    std::vector<Level2> children;
};
struct Hierarchy {
    std::vector<Level1> roots;
    uint64_t level2_nodes = 0, level3_nodes = 0;
};
struct HierarchyCounts {
    U128 direct = 0, after_first = 0, after_second = 0, after_third = 0;
    U128 compatible_first_tasks = 0, compatible_second_tasks = 0;
    U128 compatible_third_tasks = 0;
};

static int g_pair_index[ROWS][ROWS];
static std::array<std::vector<Increment>, 1U << ROWS> g_increments;

static uint64_t mix64(uint64_t value) {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static std::string u128_string(U128 value) {
    char digits[40];
    unsigned length = 0;
    do {
        digits[length++] = char('0' + value % 10);
        value /= 10;
    } while (value);
    std::string result;
    while (length) result.push_back(digits[--length]);
    return result;
}

static double ratio(U128 numerator, U128 denominator) {
    return denominator ? double((long double)numerator / (long double)denominator)
                       : 0.0;
}

static double seconds_now() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

class SupportMap {
  public:
    explicit SupportMap(size_t capacity = 16) {
        size_t size = 16;
        while (size < capacity) size <<= 1;
        entries_.resize(size);
    }

    void add(uint64_t mask) {
        if ((count_ + 1) * 10 >= entries_.size() * 7) rehash();
        size_t slot = size_t(mix64(mask)) & (entries_.size() - 1);
        while (entries_[slot].used) {
            if (entries_[slot].mask == mask) return;
            slot = (slot + 1) & (entries_.size() - 1);
        }
        entries_[slot] = MapEntry{mask, true};
        count_++;
    }

    const std::vector<MapEntry>& entries() const { return entries_; }
    size_t count() const { return count_; }

  private:
    void rehash() {
        std::vector<MapEntry> old = std::move(entries_);
        entries_.assign(old.size() * 2, MapEntry{});
        count_ = 0;
        for (const MapEntry& entry : old) if (entry.used) add(entry.mask);
    }

    std::vector<MapEntry> entries_;
    size_t count_ = 0;
};

static void initialise_tables() {
    int pair = 0;
    for (unsigned first = 0; first < ROWS; first++) {
        for (unsigned second = first + 1; second < ROWS; second++) {
            g_pair_index[first][second] = pair++;
        }
    }
    for (unsigned active = 0; active < (1U << ROWS); active++) {
        auto& increments = g_increments[active];
        unsigned assignment = active;
        for (;;) {
            uint64_t mask = 0;
            for (unsigned first = 0; first < ROWS; first++) {
                for (unsigned second = first + 1; second < ROWS; second++) {
                    if (!(active & (1U << first)) ||
                        !(active & (1U << second))) continue;
                    unsigned c1 = (assignment >> first) & 1U;
                    unsigned c2 = (assignment >> second) & 1U;
                    if (c1 == c2) {
                        mask |= UINT64_C(1)
                                << (c1 * PAIRS + g_pair_index[first][second]);
                    }
                }
            }
            increments.push_back(Increment{mask});
            if (!assignment) break;
            assignment = (assignment - 1U) & active;
        }
        std::sort(increments.begin(), increments.end(),
                  [](const Increment& a, const Increment& b) {
                      return a.mask < b.mask;
                  });
        increments.erase(std::unique(increments.begin(), increments.end(),
                                     [](const Increment& a, const Increment& b) {
                                         return a.mask == b.mask;
                                     }),
                         increments.end());
    }
}

static std::vector<uint64_t> build_distribution(uint32_t prefix,
                                                bool complement) {
    std::array<uint8_t, ROWS> rows{};
    constexpr unsigned pattern_mask = (1U << HALF_COLUMNS) - 1U;
    for (int row = ROWS - 1; row >= 0; row--) {
        rows[size_t(row)] = uint8_t(prefix & pattern_mask);
        prefix >>= HALF_COLUMNS;
    }
    SupportMap current;
    current.add(0);
    for (unsigned column = 0; column < HALF_COLUMNS; column++) {
        unsigned active = 0;
        for (unsigned row = 0; row < ROWS; row++) {
            unsigned pattern = complement ? rows[row] ^ pattern_mask : rows[row];
            if (pattern & (1U << column)) active |= 1U << row;
        }
        SupportMap next(std::max<size_t>(16, current.entries().size()));
        for (const MapEntry& entry : current.entries()) {
            if (!entry.used) continue;
            for (const Increment& increment : g_increments[active]) {
                if (!(entry.mask & increment.mask))
                    next.add(entry.mask | increment.mask);
            }
        }
        current = std::move(next);
    }
    std::vector<uint64_t> result;
    result.reserve(current.count());
    for (const MapEntry& entry : current.entries())
        if (entry.used) result.push_back(entry.mask);
    std::sort(result.begin(), result.end());
    return result;
}

static DistributionPair build_pair(uint32_t prefix) {
    return DistributionPair{build_distribution(prefix, false),
                            build_distribution(prefix, true)};
}

static uint32_t half_prefix(uint64_t key, unsigned column_offset) {
    uint32_t result = 0;
    for (unsigned row = 0; row < ROWS; row++) {
        unsigned shift = COLUMNS * (ROWS - 1U - row) + column_offset;
        result = (result << HALF_COLUMNS) |
                 uint32_t((key >> shift) & ((1U << HALF_COLUMNS) - 1U));
    }
    return result;
}

static std::vector<SampleRecord> read_stride_sample(const std::string& path,
                                                    uint64_t sample_count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path);
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    input.read(magic, sizeof(magic));
    input.read(reinterpret_cast<char*>(&columns), sizeof(columns));
    input.read(reinterpret_cast<char*>(&count), sizeof(count));
    const bool recognised = !std::memcmp(magic, "R8ORB01", 7) ||
                            !std::memcmp(magic, "R8SQT01", 7);
    if (!input || !recognised || columns != COLUMNS)
        throw std::runtime_error("invalid 8x8 solve file");
    sample_count = std::min(sample_count, count);
    std::vector<SampleRecord> result;
    result.reserve(size_t(sample_count));
    for (uint64_t sample = 0; sample < sample_count; sample++) {
        uint64_t index = uint64_t((U128(sample) * count + count / 2) /
                                  sample_count);
        if (index >= count) index = count - 1;
        input.seekg(std::streamoff(20 + index * sizeof(OrbitRecord)));
        OrbitRecord record{};
        input.read(reinterpret_cast<char*>(&record), sizeof(record));
        if (!input) throw std::runtime_error("truncated 8x8 solve file");
        result.push_back(SampleRecord{index, record.key});
    }
    return result;
}

static uint32_t extract_pair_prefix(uint64_t mask,
                                    const std::array<uint8_t, PAIRS>& order,
                                    unsigned begin, unsigned count) {
    uint32_t result = 0;
    unsigned output = 0;
    for (unsigned colour = 0; colour < 2; colour++) {
        for (unsigned index = 0; index < count; index++) {
            unsigned pair = order[begin + index];
            result |= uint32_t((mask >> (colour * PAIRS + pair)) & 1U)
                      << output++;
        }
    }
    return result;
}

static std::array<uint8_t, PAIRS> learn_pair_order(
    const std::vector<JoinView>& joins, uint64_t samples_per_join) {
    struct Observation { uint32_t conflicts; long double weight; };
    std::vector<Observation> observations;
    observations.reserve(joins.size() * size_t(samples_per_join));
    for (const JoinView& join : joins) {
        const auto& left = *join.left;
        const auto& right = *join.right;
        U128 direct = U128(left.size()) * right.size();
        uint64_t samples = uint64_t(std::min<U128>(direct, samples_per_join));
        if (!samples) continue;
        long double weight = (long double)direct / samples;
        for (uint64_t sample = 0; sample < samples; sample++) {
            uint64_t a = mix64(join.seed + sample * UINT64_C(0x9e3779b97f4a7c15));
            uint64_t b = mix64(a ^ UINT64_C(0xd1b54a32d192ed03));
            uint64_t overlap = left[size_t(a % left.size())] &
                               right[size_t(b % right.size())];
            uint32_t conflicts = 0;
            for (unsigned pair = 0; pair < PAIRS; pair++) {
                if ((overlap & (UINT64_C(1) << pair)) ||
                    (overlap & (UINT64_C(1) << (PAIRS + pair))))
                    conflicts |= 1U << pair;
            }
            observations.push_back(Observation{conflicts, weight});
        }
    }
    std::array<uint8_t, PAIRS> order{};
    uint32_t selected = 0;
    for (unsigned rank = 0; rank < PAIRS; rank++) {
        unsigned best = PAIRS;
        long double best_rejected = -1;
        for (unsigned pair = 0; pair < PAIRS; pair++) {
            if (selected & (1U << pair)) continue;
            long double rejected = 0;
            for (const Observation& observation : observations) {
                if (!(observation.conflicts & selected) &&
                    (observation.conflicts & (1U << pair)))
                    rejected += observation.weight;
            }
            if (rejected > best_rejected) {
                best_rejected = rejected;
                best = pair;
            }
        }
        order[rank] = uint8_t(best);
        selected |= 1U << best;
    }
    return order;
}

static std::array<uint8_t, PAIRS> parse_pair_order(const std::string& text) {
    std::array<uint8_t, PAIRS> order{};
    uint32_t seen = 0;
    std::stringstream stream(text);
    std::string item;
    unsigned count = 0;
    while (std::getline(stream, item, ',')) {
        if (count >= PAIRS) throw std::runtime_error("too many pair IDs");
        unsigned pair = unsigned(std::stoul(item));
        if (pair >= PAIRS || (seen & (1U << pair)))
            throw std::runtime_error("pair order is not a permutation");
        order[count++] = uint8_t(pair);
        seen |= 1U << pair;
    }
    if (count != PAIRS) throw std::runtime_error("pair order is incomplete");
    return order;
}

static U128 flat_retained(const JoinView& join,
                          const std::array<uint8_t, PAIRS>& order,
                          unsigned pair_count) {
    unsigned bits = 2 * pair_count;
    size_t size = size_t(1) << bits;
    std::vector<uint64_t> left_counts(size), right_subset(size);
    for (uint64_t mask : *join.left)
        left_counts[extract_pair_prefix(mask, order, 0, pair_count)]++;
    for (uint64_t mask : *join.right)
        right_subset[extract_pair_prefix(mask, order, 0, pair_count)]++;
    for (unsigned bit = 0; bit < bits; bit++) {
        for (size_t mask = 0; mask < size; mask++) {
            if (mask & (size_t(1) << bit))
                right_subset[mask] += right_subset[mask ^ (size_t(1) << bit)];
        }
    }
    U128 result = 0;
    size_t full = size - 1;
    for (size_t prefix = 0; prefix < size; prefix++)
        result += U128(left_counts[prefix]) * right_subset[full ^ prefix];
    return result;
}

struct Triple { uint16_t first, second; uint8_t third; };

static Hierarchy build_hierarchy(
    const std::vector<uint64_t>& distribution,
    const std::array<uint8_t, PAIRS>& order) {
    std::vector<Triple> triples;
    triples.reserve(distribution.size());
    for (uint64_t mask : distribution) {
        triples.push_back(Triple{
            uint16_t(extract_pair_prefix(mask, order, 0, 5)),
            uint16_t(extract_pair_prefix(mask, order, 5, 5)),
            uint8_t(extract_pair_prefix(mask, order, 10, 2))});
    }
    std::sort(triples.begin(), triples.end(), [](const Triple& a, const Triple& b) {
        if (a.first != b.first) return a.first < b.first;
        if (a.second != b.second) return a.second < b.second;
        return a.third < b.third;
    });
    Hierarchy result;
    for (size_t first_begin = 0; first_begin < triples.size();) {
        size_t first_end = first_begin + 1;
        while (first_end < triples.size() &&
               triples[first_end].first == triples[first_begin].first) first_end++;
        Level1 root{triples[first_begin].first,
                    uint64_t(first_end - first_begin), {}};
        for (size_t second_begin = first_begin; second_begin < first_end;) {
            size_t second_end = second_begin + 1;
            while (second_end < first_end &&
                   triples[second_end].second == triples[second_begin].second)
                second_end++;
            Level2 middle{triples[second_begin].second,
                          uint64_t(second_end - second_begin), {}};
            for (size_t third_begin = second_begin; third_begin < second_end;) {
                size_t third_end = third_begin + 1;
                while (third_end < second_end &&
                       triples[third_end].third == triples[third_begin].third)
                    third_end++;
                middle.children.push_back(Level3{
                    triples[third_begin].third,
                    uint64_t(third_end - third_begin)});
                third_begin = third_end;
            }
            result.level3_nodes += middle.children.size();
            root.children.push_back(std::move(middle));
            second_begin = second_end;
        }
        result.level2_nodes += root.children.size();
        result.roots.push_back(std::move(root));
        first_begin = first_end;
    }
    return result;
}

static HierarchyCounts count_hierarchy(const JoinView& join,
                                       const Hierarchy& left,
                                       const Hierarchy& right) {
    HierarchyCounts result;
    result.direct = U128(join.left->size()) * join.right->size();
    for (const Level1& a1 : left.roots) for (const Level1& b1 : right.roots) {
        if (a1.prefix & b1.prefix) continue;
        result.compatible_first_tasks++;
        result.after_first += U128(a1.count) * b1.count;
        for (const Level2& a2 : a1.children) for (const Level2& b2 : b1.children) {
            if (a2.prefix & b2.prefix) continue;
            result.compatible_second_tasks++;
            result.after_second += U128(a2.count) * b2.count;
            for (const Level3& a3 : a2.children)
                for (const Level3& b3 : b2.children) {
                    if (a3.prefix & b3.prefix) continue;
                    result.compatible_third_tasks++;
                    result.after_third += U128(a3.count) * b3.count;
                }
        }
    }
    return result;
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::fprintf(stderr, "Usage: %s SHARD.orbits [SAMPLE_RECORDS=64] "
                             "[TRAIN_PAIR_SAMPLES=32768] [PAIR_ORDER]\n", argv[0]);
        return 2;
    }
    try {
        std::string path = argv[1];
        uint64_t sample_records = argc > 2 ? std::stoull(argv[2]) : 64;
        uint64_t pair_samples = argc > 3 ? std::stoull(argv[3]) : 32768;
        if (sample_records < 4 || pair_samples < 1) return 2;
        double start = seconds_now();
        initialise_tables();
        std::vector<SampleRecord> records =
            read_stride_sample(path, sample_records);
        std::vector<uint32_t> prefixes;
        for (const SampleRecord& record : records) {
            prefixes.push_back(half_prefix(record.key, 0));
            prefixes.push_back(half_prefix(record.key, HALF_COLUMNS));
        }
        std::sort(prefixes.begin(), prefixes.end());
        prefixes.erase(std::unique(prefixes.begin(), prefixes.end()), prefixes.end());
        std::vector<DistributionPair> pairs(prefixes.size());
        double build_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
        for (long long index = 0; index < (long long)prefixes.size(); index++)
            pairs[size_t(index)] = build_pair(prefixes[size_t(index)]);
        double build_seconds = seconds_now() - build_start;
        std::unordered_map<uint32_t, size_t> prefix_index;
        for (size_t index = 0; index < prefixes.size(); index++)
            prefix_index.emplace(prefixes[index], index);
        std::vector<JoinView> training, holdout;
        uint64_t support_entries = 0;
        for (size_t index = 0; index < records.size(); index++) {
            uint32_t lk = half_prefix(records[index].key, 0);
            uint32_t rk = half_prefix(records[index].key, HALF_COLUMNS);
            const auto& left = pairs.at(prefix_index.at(lk));
            const auto& right = pairs.at(prefix_index.at(rk));
            support_entries += left.selected.size() + left.complement.size() +
                               right.selected.size() + right.complement.size();
            auto& destination = index & 1 ? holdout : training;
            destination.push_back(JoinView{&left.selected, &right.selected,
                                           mix64(records[index].index * 2)});
            destination.push_back(JoinView{&left.complement, &right.complement,
                                           mix64(records[index].index * 2 + 1)});
        }
        auto order = argc > 4 ? parse_pair_order(argv[4])
                              : learn_pair_order(training, pair_samples);
        std::printf("ORDER pairs=");
        for (unsigned rank = 0; rank < PAIRS; rank++) {
            unsigned pair = order[rank], first = 0;
            while (pair >= ROWS - 1 - first) pair -= ROWS - 1 - first++;
            std::printf("%s%u:(%u,%u)", rank ? "," : "", order[rank], first,
                        first + 1 + pair);
        }
        std::printf("\n");
        U128 direct = 0;
        for (const JoinView& join : holdout)
            direct += U128(join.left->size()) * join.right->size();
        for (unsigned pair_count : {4U, 5U, 6U, 7U}) {
            U128 retained = 0;
            double flat_start = seconds_now();
            for (const JoinView& join : holdout)
                retained += flat_retained(join, order, pair_count);
            std::printf("FLAT pair_coordinates=%u prefix_bits=%u direct=%s "
                        "retained=%s retained_fraction=%.9f reduction=%.6fx "
                        "seconds=%.6f\n",
                        pair_count, 2 * pair_count, u128_string(direct).c_str(),
                        u128_string(retained).c_str(), ratio(retained, direct),
                        retained ? 1.0 / ratio(retained, direct) : 0.0,
                        seconds_now() - flat_start);
        }
        HierarchyCounts totals;
        uint64_t roots = 0, level2 = 0, level3 = 0;
        double hierarchy_start = seconds_now();
        for (const JoinView& join : holdout) {
            Hierarchy left = build_hierarchy(*join.left, order);
            Hierarchy right = build_hierarchy(*join.right, order);
            roots += left.roots.size() + right.roots.size();
            level2 += left.level2_nodes + right.level2_nodes;
            level3 += left.level3_nodes + right.level3_nodes;
            HierarchyCounts counts = count_hierarchy(join, left, right);
            totals.direct += counts.direct;
            totals.after_first += counts.after_first;
            totals.after_second += counts.after_second;
            totals.after_third += counts.after_third;
            totals.compatible_first_tasks += counts.compatible_first_tasks;
            totals.compatible_second_tasks += counts.compatible_second_tasks;
            totals.compatible_third_tasks += counts.compatible_third_tasks;
        }
        std::printf("HIERARCHY levels=10+10+4 suffix_bits=32 direct=%s "
                    "after_first=%s first_fraction=%.9f after_second=%s "
                    "second_fraction=%.9f after_third=%s third_fraction=%.9f "
                    "reduction=%.6fx first_tasks=%s second_tasks=%s "
                    "third_tasks=%s mean_leaf_work=%.3f roots=%llu "
                    "level2_nodes=%llu level3_nodes=%llu seconds=%.6f\n",
                    u128_string(totals.direct).c_str(),
                    u128_string(totals.after_first).c_str(),
                    ratio(totals.after_first, totals.direct),
                    u128_string(totals.after_second).c_str(),
                    ratio(totals.after_second, totals.direct),
                    u128_string(totals.after_third).c_str(),
                    ratio(totals.after_third, totals.direct),
                    totals.after_third ? 1.0 / ratio(totals.after_third, totals.direct) : 0,
                    u128_string(totals.compatible_first_tasks).c_str(),
                    u128_string(totals.compatible_second_tasks).c_str(),
                    u128_string(totals.compatible_third_tasks).c_str(),
                    totals.compatible_third_tasks
                        ? double((long double)totals.after_third /
                                 (long double)totals.compatible_third_tasks)
                        : 0.0,
                    (unsigned long long)roots, (unsigned long long)level2,
                    (unsigned long long)level3, seconds_now() - hierarchy_start);
        std::printf("SUMMARY records=%zu train_joins=%zu holdout_joins=%zu "
                    "unique_prefixes=%zu support_entries=%llu build_seconds=%.6f "
                    "total_seconds=%.6f exact_holdout=OK\n",
                    records.size(), training.size(), holdout.size(), prefixes.size(),
                    (unsigned long long)support_entries, build_seconds,
                    seconds_now() - start);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
