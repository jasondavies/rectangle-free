#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using U128 = unsigned __int128;

constexpr unsigned ROWS = 7;
constexpr unsigned COLUMNS = 9;
constexpr unsigned ROW_BITS = 9;
constexpr unsigned LEFT_COLUMNS = 4;
constexpr unsigned RIGHT_COLUMNS = 5;
constexpr unsigned PAIRS = ROWS * (ROWS - 1) / 2;

struct OrbitRecord {
    uint64_t key;
    uint64_t weight;
};

struct RawRight {
    uint64_t key;
    uint64_t weight;
};

struct RightStat {
    uint64_t key;
    uint64_t records;
    uint64_t weight;
};

struct Increment {
    uint64_t mask;
};

struct MapEntry {
    uint64_t mask = 0;
    bool used = false;
};

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

static double seconds_now() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

static uint64_t right_prefix(uint64_t key) {
    uint64_t result = 0;
    for (unsigned row = 0; row < ROWS; row++) {
        unsigned shift = ROW_BITS * (ROWS - 1U - row);
        result = (result << RIGHT_COLUMNS) |
                 ((key >> (shift + LEFT_COLUMNS)) &
                  ((UINT64_C(1) << RIGHT_COLUMNS) - 1));
    }
    return result;
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
        for (const MapEntry& entry : old) {
            if (entry.used) add(entry.mask);
        }
    }

    std::vector<MapEntry> entries_;
    size_t count_ = 0;
};

static std::array<std::vector<Increment>, 1U << ROWS> build_increments() {
    int pair_index[ROWS][ROWS]{};
    int pair = 0;
    for (unsigned first = 0; first < ROWS; first++) {
        for (unsigned second = first + 1; second < ROWS; second++) {
            pair_index[first][second] = pair++;
        }
    }
    std::array<std::vector<Increment>, 1U << ROWS> result;
    for (unsigned active = 0; active < (1U << ROWS); active++) {
        unsigned assignment = active;
        while (true) {
            uint64_t mask = 0;
            for (unsigned first = 0; first < ROWS; first++) {
                for (unsigned second = first + 1; second < ROWS; second++) {
                    if (!(active & (1U << first)) ||
                        !(active & (1U << second))) {
                        continue;
                    }
                    unsigned first_colour = (assignment >> first) & 1U;
                    unsigned second_colour = (assignment >> second) & 1U;
                    if (first_colour == second_colour) {
                        mask |= UINT64_C(1)
                                << (first_colour * PAIRS +
                                    pair_index[first][second]);
                    }
                }
            }
            result[active].push_back(Increment{mask});
            if (!assignment) break;
            assignment = (assignment - 1U) & active;
        }
        auto& increments = result[active];
        std::sort(increments.begin(), increments.end(),
                  [](const Increment& lhs, const Increment& rhs) {
                      return lhs.mask < rhs.mask;
                  });
        increments.erase(
            std::unique(increments.begin(), increments.end(),
                        [](const Increment& lhs, const Increment& rhs) {
                            return lhs.mask == rhs.mask;
                        }),
            increments.end());
    }
    return result;
}

static uint32_t support_count(
    uint64_t prefix, bool complement,
    const std::array<std::vector<Increment>, 1U << ROWS>& increments) {
    std::array<uint8_t, ROWS> patterns{};
    constexpr unsigned pattern_mask = (1U << RIGHT_COLUMNS) - 1U;
    for (int row = ROWS - 1; row >= 0; row--) {
        patterns[size_t(row)] = uint8_t(prefix & pattern_mask);
        prefix >>= RIGHT_COLUMNS;
    }
    SupportMap current;
    current.add(0);
    for (unsigned column = 0; column < RIGHT_COLUMNS; column++) {
        unsigned active = 0;
        for (unsigned row = 0; row < ROWS; row++) {
            unsigned pattern = complement ? patterns[row] ^ pattern_mask
                                          : patterns[row];
            if (pattern & (1U << column)) active |= 1U << row;
        }
        SupportMap next(std::max<size_t>(16, current.entries().size()));
        for (const MapEntry& entry : current.entries()) {
            if (!entry.used) continue;
            for (const Increment& increment : increments[active]) {
                if (!(entry.mask & increment.mask)) {
                    next.add(entry.mask | increment.mask);
                }
            }
        }
        current = std::move(next);
    }
    if (current.count() > UINT32_MAX) {
        throw std::overflow_error("distribution support exceeds uint32_t");
    }
    return uint32_t(current.count());
}

static std::vector<RightStat> read_shard(const std::string& path,
                                         uint64_t& records,
                                         U128& labelled_weight) {
    double start = seconds_now();
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path);
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    input.read(magic, sizeof(magic));
    input.read(reinterpret_cast<char*>(&columns), sizeof(columns));
    input.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!input || std::memcmp(magic, "R7ORB09", 7) || columns != COLUMNS) {
        throw std::runtime_error("invalid 7x9 orbit file " + path);
    }
    std::vector<RawRight> raw;
    raw.reserve(size_t(count));
    constexpr size_t chunk_records = 1U << 20;
    std::vector<OrbitRecord> chunk(chunk_records);
    for (uint64_t offset = 0; offset < count;) {
        size_t wanted = size_t(std::min<uint64_t>(chunk_records, count - offset));
        input.read(reinterpret_cast<char*>(chunk.data()),
                   wanted * sizeof(OrbitRecord));
        if (!input) throw std::runtime_error("truncated orbit file " + path);
        for (size_t index = 0; index < wanted; index++) {
            raw.push_back(RawRight{right_prefix(chunk[index].key),
                                   chunk[index].weight});
            labelled_weight += chunk[index].weight;
        }
        offset += wanted;
    }
    char trailing;
    if (input.read(&trailing, 1)) {
        throw std::runtime_error("trailing orbit data " + path);
    }
    records = count;
    std::sort(raw.begin(), raw.end(), [](const RawRight& lhs,
                                         const RawRight& rhs) {
        return lhs.key < rhs.key;
    });
    std::vector<RightStat> result;
    result.reserve(raw.size() / 3);
    for (size_t begin = 0; begin < raw.size();) {
        size_t end = begin + 1;
        U128 weight = raw[begin].weight;
        while (end < raw.size() && raw[end].key == raw[begin].key) {
            weight += raw[end].weight;
            end++;
        }
        if (weight > std::numeric_limits<uint64_t>::max()) {
            throw std::overflow_error("per-right labelled weight exceeds uint64_t");
        }
        result.push_back(RightStat{raw[begin].key, end - begin,
                                   uint64_t(weight)});
        begin = end;
    }
    std::fprintf(stderr,
                 "SHARD path=%s records=%llu unique_right=%zu seconds=%.6f\n",
                 path.c_str(), (unsigned long long)records, result.size(),
                 seconds_now() - start);
    return result;
}

struct HeapItem {
    uint64_t key;
    uint32_t shard;
    uint64_t index;
};

struct HeapGreater {
    bool operator()(const HeapItem& lhs, const HeapItem& rhs) const {
        if (lhs.key != rhs.key) return lhs.key > rhs.key;
        return lhs.shard > rhs.shard;
    }
};

struct SupportSample {
    uint64_t key;
    uint32_t multiplicity;
    uint32_t support = 0;
};

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "Usage: %s SAMPLE_BITS SHARD.orbits SHARD.orbits [...]\n",
                     argv[0]);
        return 2;
    }
    unsigned sample_bits = unsigned(std::stoul(argv[1]));
    if (sample_bits > 30) return 2;
    const size_t shard_count = size_t(argc - 2);
    std::vector<std::string> paths(shard_count);
    for (size_t shard = 0; shard < shard_count; shard++) {
        paths[shard] = argv[shard + 2];
    }
    std::vector<std::vector<RightStat>> shards(shard_count);
    std::vector<uint64_t> record_counts(shard_count);
    std::vector<U128> labelled_weights(shard_count);
    double read_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
    for (long long shard = 0; shard < (long long)shard_count; shard++) {
        shards[size_t(shard)] =
            read_shard(paths[size_t(shard)], record_counts[size_t(shard)],
                       labelled_weights[size_t(shard)]);
    }
    double read_seconds = seconds_now() - read_start;

    uint64_t sum_distinct = 0;
    uint64_t total_records = 0;
    U128 total_weight = 0;
    std::priority_queue<HeapItem, std::vector<HeapItem>, HeapGreater> heap;
    for (size_t shard = 0; shard < shard_count; shard++) {
        sum_distinct += shards[shard].size();
        total_records += record_counts[shard];
        total_weight += labelled_weights[shard];
        if (!shards[shard].empty()) {
            heap.push(HeapItem{shards[shard][0].key, uint32_t(shard), 0});
        }
    }
    std::vector<uint64_t> multiplicity_histogram(shard_count + 1);
    std::vector<uint64_t> pairwise(shard_count * shard_count);
    uint64_t union_distinct = 0;
    uint64_t repeated_records = 0;
    U128 repeated_weight = 0;
    std::vector<SupportSample> samples;
    const uint64_t sample_mask = sample_bits ? (UINT64_C(1) << sample_bits) - 1 : 0;
    double merge_start = seconds_now();
    while (!heap.empty()) {
        uint64_t key = heap.top().key;
        std::vector<uint32_t> present;
        uint64_t key_records = 0;
        U128 key_weight = 0;
        while (!heap.empty() && heap.top().key == key) {
            HeapItem item = heap.top();
            heap.pop();
            const RightStat& stat = shards[item.shard][item.index];
            present.push_back(item.shard);
            key_records += stat.records;
            key_weight += stat.weight;
            item.index++;
            if (item.index < shards[item.shard].size()) {
                item.key = shards[item.shard][item.index].key;
                heap.push(item);
            }
        }
        union_distinct++;
        multiplicity_histogram[present.size()]++;
        if (present.size() > 1) {
            repeated_records += key_records;
            repeated_weight += key_weight;
        }
        for (size_t first = 0; first < present.size(); first++) {
            for (size_t second = first + 1; second < present.size(); second++) {
                pairwise[size_t(present[first]) * shard_count + present[second]]++;
            }
        }
        if (!(mix64(key) & sample_mask)) {
            samples.push_back(
                SupportSample{key, uint32_t(present.size()), 0});
        }
    }
    double merge_seconds = seconds_now() - merge_start;

    auto increments = build_increments();
    double support_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
    for (long long index = 0; index < (long long)samples.size(); index++) {
        uint64_t key = samples[size_t(index)].key;
        samples[size_t(index)].support =
            support_count(key, false, increments) +
            support_count(key, true, increments);
    }
    double support_seconds = seconds_now() - support_start;
    U128 sampled_build_entries = 0;
    U128 sampled_saved_entries = 0;
    for (const SupportSample& sample : samples) {
        sampled_build_entries += U128(sample.support) * sample.multiplicity;
        sampled_saved_entries += U128(sample.support) * (sample.multiplicity - 1);
    }

    uint64_t duplicate_builds = sum_distinct - union_distinct;
    std::printf(
        "OVERLAP shards=%zu records=%llu labelled_weight=%s "
        "sum_distinct=%llu union_distinct=%llu duplicate_builds=%llu "
        "raw_reuse_fraction=%.9f repeated_record_fraction=%.9f "
        "repeated_weight_fraction=%.9f read_seconds=%.6f merge_seconds=%.6f "
        "exact=OK\n",
        shard_count, (unsigned long long)total_records,
        u128_string(total_weight).c_str(), (unsigned long long)sum_distinct,
        (unsigned long long)union_distinct,
        (unsigned long long)duplicate_builds,
        double(duplicate_builds) / double(sum_distinct),
        double(repeated_records) / double(total_records),
        double(repeated_weight) / double(total_weight), read_seconds,
        merge_seconds);
    for (size_t multiplicity = 1; multiplicity <= shard_count; multiplicity++) {
        if (multiplicity_histogram[multiplicity]) {
            std::printf("MULTIPLICITY shards=%zu keys=%llu fraction=%.9f\n",
                        multiplicity,
                        (unsigned long long)multiplicity_histogram[multiplicity],
                        double(multiplicity_histogram[multiplicity]) /
                            double(union_distinct));
        }
    }
    for (size_t first = 0; first < shard_count; first++) {
        for (size_t second = first + 1; second < shard_count; second++) {
            uint64_t intersection =
                pairwise[first * shard_count + second];
            uint64_t pair_union = shards[first].size() + shards[second].size() -
                                  intersection;
            std::printf(
                "PAIR first=%zu second=%zu intersection=%llu jaccard=%.9f "
                "first_fraction=%.9f second_fraction=%.9f\n",
                first, second, (unsigned long long)intersection,
                double(intersection) / double(pair_union),
                double(intersection) / double(shards[first].size()),
                double(intersection) / double(shards[second].size()));
        }
    }
    std::printf(
        "SUPPORT_SAMPLE bits=%u samples=%zu build_entries=%s saved_entries=%s "
        "expanded_reuse_fraction=%.9f seconds=%.6f exact_support=OK\n",
        sample_bits, samples.size(),
        u128_string(sampled_build_entries).c_str(),
        u128_string(sampled_saved_entries).c_str(),
        sampled_build_entries
            ? double(sampled_saved_entries) / double(sampled_build_entries)
            : 0.0,
        support_seconds);
    return 0;
}
