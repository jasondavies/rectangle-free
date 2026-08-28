#define GRID_ROWS 6
#define GRID_COLUMNS 12
#define LEFT_COLUMNS 6
#define RIGHT_COLUMNS 6
#define ORBIT_ROW_BITS 12
#define ORBIT_MAGIC "R6W1201"
#define TWCOLOUR_WIDE_ORBIT_RECORD 1

#include "../../src/gpu/twocolour_gpu_common.cuh"

#include <iomanip>
#include <queue>

namespace {

constexpr unsigned PREFIX_PAIRS = 5;
constexpr unsigned SUFFIX_PAIRS = PAIRS - PREFIX_PAIRS;
constexpr unsigned SUFFIX_BITS_LOCAL = 2 * SUFFIX_PAIRS;

struct Class {
    uint32_t weight = 0;
    uint8_t orbit = 0;
    uint32_t bit_or = 0;
    uint32_t bit_and = 0;
    std::vector<uint32_t> suffixes;
};

struct Bucket {
    uint16_t prefix = 0;
    std::vector<Class> classes;
};

struct Distribution { std::vector<Bucket> buckets; };
struct Pair { Distribution selected, complement; };
struct Record { uint32_t left = 0, right = 0; };

struct Stats {
    U128 comparisons = 0, tiles = 0;
    U128 terminal_accept = 0, terminal_reject = 0;
    U128 dimension_tiles[SUFFIX_BITS_LOCAL + 1]{};
    U128 sos_tiles[4]{};
    uint64_t class_pairs = 0, bucket_pairs = 0;
};

struct Heavy {
    const Class* left = nullptr;
    const Class* right = nullptr;
    bool swapped = false;
    uint64_t tiles = 0;
};
struct MinHeavy {
    bool operator()(const Heavy& a, const Heavy& b) const {
        return a.tiles > b.tiles;
    }
};

static PrefixKey half_key(U128 key, unsigned begin) {
    PrefixKey result = 0;
    const U128 row_mask = (U128(1) << COLUMNS) - 1U;
    for (unsigned row = 0; row < ROWS; ++row) {
        const unsigned shift = COLUMNS * (ROWS - 1U - row);
        result = (result << 6) |
                 PrefixKey((((key >> shift) & row_mask) >> begin) & 63U);
    }
    return result;
}

static void split_mask(uint64_t mask, uint32_t pair_mask, uint16_t& prefix,
                       uint32_t& suffix) {
    prefix = 0;
    suffix = 0;
    unsigned pi = 0, si = 0;
    for (unsigned plane = 0; plane < 2; ++plane) {
        for (unsigned pair = 0; pair < PAIRS; ++pair) {
            const bool selected = (pair_mask >> pair) & 1U;
            const bool set = (mask >> (plane * PAIRS + pair)) & 1U;
            if (selected) {
                if (set) prefix |= uint16_t(1U << pi);
                ++pi;
            } else {
                if (set) suffix |= uint32_t(1U << si);
                ++si;
            }
        }
    }
}

static uint16_t swap_prefix(uint16_t value) {
    constexpr uint16_t half = (1U << PREFIX_PAIRS) - 1U;
    return uint16_t(((value & half) << PREFIX_PAIRS) |
                    ((value >> PREFIX_PAIRS) & half));
}
static uint32_t swap_suffix(uint32_t value) {
    constexpr uint32_t half = (1U << SUFFIX_PAIRS) - 1U;
    return ((value & half) << SUFFIX_PAIRS) |
           ((value >> SUFFIX_PAIRS) & half);
}
static uint64_t bmma_tiles(uint32_t a, uint32_t b) {
    return std::min(uint64_t((a + 15) / 16) * ((b + 7) / 8),
                    uint64_t((b + 15) / 16) * ((a + 7) / 8));
}

static Distribution make_distribution(PrefixKey key, bool complement,
                                      uint32_t pair_mask) {
    const auto full = quotient_token_planes(
        build_distribution(key, 6, complement));
    struct Item {
        uint16_t prefix;
        uint32_t suffix, weight;
        uint8_t orbit;
    };
    std::vector<Item> items;
    items.reserve(full.entries.size());
    for (const Entry& entry : full.entries) {
        uint16_t prefix;
        uint32_t suffix;
        split_mask(entry.mask, pair_mask, prefix, suffix);
        if (entry.weight > UINT32_MAX)
            throw std::overflow_error("6x6 distribution weight exceeds uint32");
        items.push_back(Item{prefix, suffix, uint32_t(entry.weight),
                             uint8_t(token_plane_orbit_size(entry.mask))});
    }
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        if (a.prefix != b.prefix) return a.prefix < b.prefix;
        if (a.weight != b.weight) return a.weight < b.weight;
        if (a.orbit != b.orbit) return a.orbit < b.orbit;
        return a.suffix < b.suffix;
    });
    Distribution result;
    for (size_t begin = 0; begin < items.size();) {
        size_t bucket_end = begin + 1;
        while (bucket_end < items.size() &&
               items[bucket_end].prefix == items[begin].prefix)
            ++bucket_end;
        Bucket bucket;
        bucket.prefix = items[begin].prefix;
        for (size_t class_begin = begin; class_begin < bucket_end;) {
            size_t class_end = class_begin + 1;
            while (class_end < bucket_end &&
                   items[class_end].weight == items[class_begin].weight &&
                   items[class_end].orbit == items[class_begin].orbit)
                ++class_end;
            Class output;
            output.weight = items[class_begin].weight;
            output.orbit = items[class_begin].orbit;
            output.bit_and = (1U << SUFFIX_BITS_LOCAL) - 1U;
            output.suffixes.reserve(class_end - class_begin);
            for (size_t i = class_begin; i < class_end; ++i) {
                output.suffixes.push_back(items[i].suffix);
                output.bit_or |= items[i].suffix;
                output.bit_and &= items[i].suffix;
            }
            bucket.classes.push_back(std::move(output));
            class_begin = class_end;
        }
        result.buckets.push_back(std::move(bucket));
        begin = bucket_end;
    }
    return result;
}

static void retain(std::priority_queue<Heavy, std::vector<Heavy>, MinHeavy>& q,
                   size_t limit, Heavy value) {
    if (!limit) return;
    if (q.size() < limit) q.push(value);
    else if (value.tiles > q.top().tiles) { q.pop(); q.push(value); }
}

static void visit(const Class& a, const Class& b, bool swapped, Stats& stats,
                  std::priority_queue<Heavy, std::vector<Heavy>, MinHeavy>& q,
                  size_t heavy_limit) {
    const uint32_t bor = swapped ? swap_suffix(b.bit_or) : b.bit_or;
    const uint32_t band = swapped ? swap_suffix(b.bit_and) : b.bit_and;
    const uint32_t active = a.bit_or & bor;
    const uint64_t comparisons = uint64_t(a.suffixes.size()) * b.suffixes.size();
    const uint64_t tiles = bmma_tiles(uint32_t(a.suffixes.size()),
                                      uint32_t(b.suffixes.size()));
    const unsigned d = __builtin_popcount(active);
    stats.comparisons += comparisons;
    stats.tiles += tiles;
    stats.dimension_tiles[d] += tiles;
    ++stats.class_pairs;
    const U128 sos = U128(d) * (U128(1) << d) +
                     a.suffixes.size() + b.suffixes.size();
    constexpr unsigned gates[] = {1, 8, 32, 128};
    for (unsigned i = 0; i < 4; ++i)
        if (sos * gates[i] <= comparisons) stats.sos_tiles[i] += tiles;
    if (!active) stats.terminal_accept += tiles;
    else if (a.bit_and & band) stats.terminal_reject += tiles;
    else retain(q, heavy_limit, Heavy{&a, &b, swapped, tiles});
}

static void join_stats(
    const Distribution& left, const Distribution& right, Stats& stats,
    std::priority_queue<Heavy, std::vector<Heavy>, MinHeavy>& q,
    size_t heavy_limit) {
    stats.bucket_pairs += uint64_t(left.buckets.size()) * right.buckets.size();
    for (const Bucket& a : left.buckets) for (const Bucket& b : right.buckets) {
        const bool ordinary = !(a.prefix & b.prefix);
        const bool swapped = !(a.prefix & swap_prefix(b.prefix));
        if (ordinary)
            for (const Class& ac : a.classes)
                for (const Class& bc : b.classes)
                    visit(ac, bc, false, stats, q, heavy_limit);
        if (swapped)
            for (const Class& ac : a.classes)
                for (const Class& bc : b.classes)
                    if (bc.orbit == 2)
                        visit(ac, bc, true, stats, q, heavy_limit);
    }
}

static std::vector<uint32_t> project(const Class& source, uint32_t mask,
                                     bool swapped) {
    std::vector<uint32_t> result;
    result.reserve(source.suffixes.size());
    for (uint32_t value : source.suffixes)
        result.push_back((swapped ? swap_suffix(value) : value) & mask);
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}
static long double ld(U128 value) { return static_cast<long double>(value); }

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::fprintf(stderr, "Usage: %s SAMPLE [RECORDS=32] [PAIR_MASK=0x67] [HEAVY=512]\n", argv[0]);
        return 2;
    }
    const uint64_t wanted = argc >= 3 ? std::strtoull(argv[2], nullptr, 10) : 32;
    const uint32_t pair_mask = argc >= 4 ? std::strtoul(argv[3], nullptr, 0) : 0x67;
    const size_t heavy_limit = argc >= 5 ? std::strtoull(argv[4], nullptr, 10) : 512;
    if (__builtin_popcount(pair_mask) != PREFIX_PAIRS)
        throw std::runtime_error("pair mask must contain five pairs");
    std::ifstream input(argv[1], std::ios::binary);
    char magic[8]; uint32_t width; uint64_t count;
    input.read(magic, 8); input.read(reinterpret_cast<char*>(&width), 4);
    input.read(reinterpret_cast<char*>(&count), 8);
    if (!input || std::memcmp(magic, ORBIT_MAGIC, 7) || width != COLUMNS || !wanted || wanted > count)
        throw std::runtime_error("invalid sample");
    std::vector<U128> keys;
    for (uint64_t i = 0; i < wanted; ++i) {
        const uint64_t index = uint64_t((U128(i * 2 + 1) * count) / (U128(2) * wanted));
        OrbitRecord record{};
        input.seekg(std::streamoff(20 + index * sizeof(record)));
        input.read(reinterpret_cast<char*>(&record), sizeof(record));
        keys.push_back((U128(record.meta & WIDE_ORBIT_KEY_MASK) << 64) | record.low);
    }
    initialise_tables();
    std::vector<PrefixKey> half_keys;
    for (U128 key : keys) { half_keys.push_back(half_key(key, 6)); half_keys.push_back(half_key(key, 0)); }
    std::sort(half_keys.begin(), half_keys.end());
    half_keys.erase(std::unique(half_keys.begin(), half_keys.end()), half_keys.end());
    std::vector<Pair> pairs(half_keys.size());
    const double build_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
    for (long long i = 0; i < (long long)half_keys.size(); ++i)
        pairs[size_t(i)] = Pair{make_distribution(half_keys[size_t(i)], false, pair_mask), make_distribution(half_keys[size_t(i)], true, pair_mask)};
    const double build_seconds = seconds_now() - build_start;
    std::unordered_map<PrefixKey, uint32_t> ids;
    for (size_t i = 0; i < half_keys.size(); ++i) ids.emplace(half_keys[i], i);
    std::vector<Record> records;
    for (U128 key : keys) records.push_back(Record{ids.at(half_key(key, 6)), ids.at(half_key(key, 0))});
    Stats stats;
    std::priority_queue<Heavy, std::vector<Heavy>, MinHeavy> heavy;
    for (const Record& record : records) {
        const Pair& a = pairs[record.left]; const Pair& b = pairs[record.right];
        join_stats(a.selected, b.selected, stats, heavy, heavy_limit);
        join_stats(a.complement, b.complement, stats, heavy, heavy_limit);
    }
    U128 heavy_tiles = 0, raw_entries = 0, projected_entries = 0;
    while (!heavy.empty()) {
        const Heavy task = heavy.top(); heavy.pop();
        const uint32_t bor = task.swapped ? swap_suffix(task.right->bit_or) : task.right->bit_or;
        const uint32_t mask = task.left->bit_or & bor;
        const auto a = project(*task.left, mask, false);
        const auto b = project(*task.right, mask, task.swapped);
        heavy_tiles += task.tiles;
        raw_entries += task.left->suffixes.size() + task.right->suffixes.size();
        projected_entries += a.size() + b.size();
    }
    const U128 terminal = stats.terminal_accept + stats.terminal_reject;
    std::cout << std::setprecision(12) << "SIX_BY_TWELVE_PROJECTION records=" << records.size()
              << " half_keys=" << half_keys.size() << " pair_mask=0x" << std::hex << pair_mask << std::dec
              << " build_seconds=" << build_seconds << " bucket_pairs=" << stats.bucket_pairs
              << " class_pairs=" << stats.class_pairs << " comparisons=" << u128_string(stats.comparisons)
              << " tiles=" << u128_string(stats.tiles)
              << " terminal_tile_ratio=" << double(ld(terminal) / ld(stats.tiles))
              << " heavy_tile_coverage=" << double(ld(heavy_tiles) / ld(stats.tiles))
              << " projected_entry_ratio=" << double(ld(projected_entries) / ld(raw_entries)) << '\n';
    U128 cumulative = 0;
    for (unsigned d = 0; d <= SUFFIX_BITS_LOCAL; ++d) {
        cumulative += stats.dimension_tiles[d];
        if (stats.dimension_tiles[d])
            std::cout << "SIX_BY_TWELVE_PROJECTION_DIM d=" << d
                      << " tile_ratio=" << double(ld(stats.dimension_tiles[d]) / ld(stats.tiles))
                      << " cumulative=" << double(ld(cumulative) / ld(stats.tiles)) << '\n';
    }
    constexpr unsigned gates[] = {1, 8, 32, 128};
    for (unsigned i = 0; i < 4; ++i)
        std::cout << "SIX_BY_TWELVE_SOS_GATE factor=" << gates[i]
                  << " tile_coverage=" << double(ld(stats.sos_tiles[i]) / ld(stats.tiles)) << '\n';
    std::cout << "SIX_BY_TWELVE_PROJECTION exact=STRUCTURAL_OK\n";
}
