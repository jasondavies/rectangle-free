#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using U128 = unsigned __int128;
using PrefixKey = uint64_t;

#if !defined(GRID_ROWS) || !defined(GRID_COLUMNS) || \
    !defined(LEFT_COLUMNS) || !defined(RIGHT_COLUMNS) || \
    !defined(ORBIT_ROW_BITS) || !defined(ORBIT_MAGIC)
#error "GPU geometry and orbit format must be defined by the entry point"
#endif

enum {
    ROWS = GRID_ROWS,
    COLUMNS = GRID_COLUMNS,
    PAIRS = ROWS * (ROWS - 1) / 2,
    ROW_BITS = ORBIT_ROW_BITS,
    CELLS = ROWS * COLUMNS,
    THREADS = 256
};


struct OrbitRecord {
    uint64_t key;
    uint64_t weight;
};

struct alignas(16) Entry {
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

struct Edge {
    PrefixKey left;
    PrefixKey right;
    uint64_t weight;
    uint8_t factor;
    // Set after loading so exact CPU checks are spread deterministically
    // through the key-sorted workload instead of sampling only its prefix.
    uint8_t validate = 0;
};

static uint64_t mix64(uint64_t x);

static void mark_validation_edges(std::vector<Edge>& edges, uint64_t wanted) {
    const uint64_t count = std::min<uint64_t>(wanted, edges.size());
    for (uint64_t stratum = 0; stratum < count; stratum++) {
        const uint64_t begin = U128(stratum) * edges.size() / count;
        const uint64_t end = U128(stratum + 1) * edges.size() / count;
        const uint64_t width = end - begin;
        // Stable jitter avoids repeatedly selecting the same relative position
        // in every sorted-key stratum while guaranteeing one check per stratum.
        const uint64_t index = begin + mix64(stratum ^ edges.size()) % width;
        edges[size_t(index)].validate = 1;
    }
}

struct CanonicalDescriptor {
    uint64_t offset;
    uint32_t count;
};

struct CanonicalForm {
    PrefixKey key;
    uint32_t row_map;
};

struct CanonicalRef {
    uint32_t distribution;
    uint32_t row_map;
};

struct RawCanonicalPair {
    PrefixKey raw;
    CanonicalRef selected;
    CanonicalRef complement;
};

struct CanonicalFactory {
    int columns;
    std::vector<Entry> entries;
    std::vector<CanonicalDescriptor> descriptors;
    // Sorted in descriptor order. Retaining these keys lets a long-lived
    // factory resolve raw prefixes that were not present in its seed shard.
    std::vector<PrefixKey> canonical_keys;
    std::vector<RawCanonicalPair> raw;
};

static int g_pair_index[ROWS][ROWS];
static std::vector<Increment> g_increments[1 << ROWS];

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error_ = (call);                                             \
        if (error_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                         cudaGetErrorString(error_));                            \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

static double seconds_now() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

static std::string u128_string(U128 value) {
    char digits[40];
    int length = 0;
    do {
        digits[length++] = char('0' + value % 10);
        value /= 10;
    } while (value);
    std::string result;
    while (length) result.push_back(digits[--length]);
    return result;
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

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

static void initialise_tables() {
    int pair = 0;
    for (int u = 0; u < ROWS; u++) {
        for (int v = u + 1; v < ROWS; v++) g_pair_index[u][v] = pair++;
    }
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
                        mask |= UINT64_C(1) << (cu * PAIRS + g_pair_index[u][v]);
                    }
                }
            }
            increments.push_back(Increment{mask, 1});
            if (!assignment) break;
            assignment = (assignment - 1U) & active;
        }
        std::sort(increments.begin(), increments.end(),
                  [](const Increment& a, const Increment& b) { return a.mask < b.mask; });
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

static Distribution build_distribution(PrefixKey prefix, int columns, bool complement) {
    uint8_t rows[ROWS];
    unsigned pattern_mask = (1U << columns) - 1U;
    for (int row = ROWS - 1; row >= 0; row--) {
        rows[row] = uint8_t(prefix & pattern_mask);
        prefix >>= columns;
    }
    Map current;
    current.add(0, 1);
    for (int column = 0; column < columns; column++) {
        unsigned active = 0;
        for (int row = 0; row < ROWS; row++) {
            unsigned pattern = complement ? rows[row] ^ pattern_mask : rows[row];
            if (pattern & (1U << column)) active |= 1U << row;
        }
        Map next(std::max<size_t>(16, current.entries().size()));
        for (const MapEntry& entry : current.entries()) {
            if (!entry.used) continue;
            for (const Increment& increment : g_increments[active]) {
                if (entry.mask & increment.mask) continue;
                next.add(entry.mask | increment.mask, entry.weight * increment.weight);
            }
        }
        current = std::move(next);
    }
    Distribution result;
    result.entries.reserve(current.count());
    for (const MapEntry& entry : current.entries()) {
        if (!entry.used) continue;
        if (entry.weight > UINT32_MAX) {
            throw std::overflow_error("distribution weight exceeds uint32_t");
        }
        result.entries.push_back(Entry{entry.mask, entry.weight});
    }
    return result;
}

// Complementing every inner binary colour swaps the two PAIRS-bit token
// planes.  Each complete half-distribution is invariant under this involution,
// so retain one arbitrary representative per support orbit.  Per-mask weights
// remain unchanged; the join kernel restores the exact orbit multiplicities.
static __host__ __device__ uint64_t swap_token_planes(uint64_t mask) {
    constexpr uint64_t plane_mask =
        (UINT64_C(1) << PAIRS) - UINT64_C(1);
    return ((mask & plane_mask) << PAIRS) |
           ((mask >> PAIRS) & plane_mask);
}

static Distribution quotient_token_planes(Distribution distribution) {
    size_t output = 0;
    for (Entry entry : distribution.entries) {
        if (entry.mask <= swap_token_planes(entry.mask))
            distribution.entries[output++] = entry;
    }
    distribution.entries.resize(output);
    return distribution;
}

static __host__ __device__ uint32_t token_plane_orbit_size(uint64_t mask) {
    return mask == swap_token_planes(mask) ? 1U : 2U;
}

static DistributionPair build_pair(PrefixKey prefix, int columns) {
    return DistributionPair{build_distribution(prefix, columns, false),
                            build_distribution(prefix, columns, true)};
}

static CanonicalForm canonical_prefix(PrefixKey prefix, int columns) {
    const uint32_t pattern_mask = (1U << columns) - 1U;
    std::array<uint8_t, ROWS> patterns{};
    for (int row = ROWS - 1; row >= 0; row--) {
        patterns[size_t(row)] = uint8_t(prefix & pattern_mask);
        prefix >>= columns;
    }

    std::array<uint8_t, 5> permutation{};
    for (int column = 0; column < columns; column++) {
        permutation[size_t(column)] = uint8_t(column);
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
    } while (std::next_permutation(permutation.begin(), permutation.begin() + columns));
    return CanonicalForm{best_key, best_row_map};
}

static __host__ __device__ int pair_number(int first, int second) {
    if (first > second) {
        int temporary = first;
        first = second;
        second = temporary;
    }
    return first * (2 * ROWS - first - 1) / 2 + second - first - 1;
}

static __host__ __device__ uint64_t transform_pair_mask(uint64_t mask, uint64_t row_map) {
    uint64_t transformed = 0;
    for (int colour = 0; colour < 2; colour++) {
        for (int first = 0; first < ROWS; first++) {
            int image_first = int((row_map >> (4 * first)) & 15U);
            for (int second = first + 1; second < ROWS; second++) {
                int source = colour * PAIRS + pair_number(first, second);
                if (!(mask & (UINT64_C(1) << source))) continue;
                int image_second = int((row_map >> (4 * second)) & 15U);
                int destination = colour * PAIRS + pair_number(image_first, image_second);
                transformed |= UINT64_C(1) << destination;
            }
        }
    }
    return transformed;
}

static bool entries_equal(std::vector<Entry> lhs, std::vector<Entry> rhs) {
    auto less = [](const Entry& a, const Entry& b) { return a.mask < b.mask; };
    std::sort(lhs.begin(), lhs.end(), less);
    std::sort(rhs.begin(), rhs.end(), less);
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); i++) {
        if (lhs[i].mask != rhs[i].mask || lhs[i].weight != rhs[i].weight) return false;
    }
    return true;
}

static const RawCanonicalPair& lookup_raw(const CanonicalFactory& factory, PrefixKey raw) {
    auto found = std::lower_bound(
        factory.raw.begin(), factory.raw.end(), raw,
        [](const RawCanonicalPair& item, PrefixKey key) { return item.raw < key; });
    if (found == factory.raw.end() || found->raw != raw) {
        throw std::runtime_error("canonical prefix lookup failed");
    }
    return *found;
}

static bool find_factory_canonical_refs(
    const CanonicalFactory& factory, PrefixKey raw,
    std::array<CanonicalRef, 2>& result) {
    if (factory.canonical_keys.empty()) {
        return false;
    }
    const PrefixKey full_mask = factory.columns * ROWS == 64
                                    ? UINT64_MAX
                                    : (PrefixKey(1) << (factory.columns * ROWS)) - 1U;
    const CanonicalForm forms[2] = {
        canonical_prefix(raw, factory.columns),
        canonical_prefix(raw ^ full_mask, factory.columns)};
    for (int complement = 0; complement < 2; complement++) {
        auto source = std::lower_bound(factory.canonical_keys.begin(),
                                       factory.canonical_keys.end(),
                                       forms[complement].key);
        if (source == factory.canonical_keys.end() ||
            *source != forms[complement].key) {
            return false;
        }
        result[complement] = CanonicalRef{
            uint32_t(source - factory.canonical_keys.begin()),
            forms[complement].row_map};
    }
    return true;
}

static CanonicalFactory build_canonical_factory(std::vector<PrefixKey> raw_keys, int columns) {
    std::sort(raw_keys.begin(), raw_keys.end());
    raw_keys.erase(std::unique(raw_keys.begin(), raw_keys.end()), raw_keys.end());
    struct RawForms {
        PrefixKey raw;
        CanonicalForm selected;
        CanonicalForm complement;
    };
    std::vector<RawForms> forms(raw_keys.size());
    const PrefixKey full_mask = columns * ROWS == 64
                                    ? UINT64_MAX
                                    : (PrefixKey(1) << (columns * ROWS)) - 1U;
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)raw_keys.size(); i++) {
        PrefixKey raw = raw_keys[size_t(i)];
        forms[size_t(i)] = RawForms{raw, canonical_prefix(raw, columns),
                                    canonical_prefix(raw ^ full_mask, columns)};
    }

    std::vector<PrefixKey> canonical_keys;
    canonical_keys.reserve(forms.size() * 2);
    for (const RawForms& item : forms) {
        canonical_keys.push_back(item.selected.key);
        canonical_keys.push_back(item.complement.key);
    }
    std::sort(canonical_keys.begin(), canonical_keys.end());
    canonical_keys.erase(std::unique(canonical_keys.begin(), canonical_keys.end()),
                         canonical_keys.end());

    std::vector<Distribution> distributions(canonical_keys.size());
#pragma omp parallel for schedule(dynamic, 8)
    for (long long i = 0; i < (long long)canonical_keys.size(); i++) {
        distributions[size_t(i)] = build_distribution(canonical_keys[size_t(i)], columns, false);
        distributions[size_t(i)] =
            quotient_token_planes(std::move(distributions[size_t(i)]));
    }

    CanonicalFactory factory{};
    factory.columns = columns;
    factory.canonical_keys = canonical_keys;
    factory.descriptors.resize(distributions.size());
    size_t total_entries = 0;
    for (const Distribution& distribution : distributions) {
        total_entries += distribution.entries.size();
    }
    factory.entries.reserve(total_entries);
    for (size_t i = 0; i < distributions.size(); i++) {
        factory.descriptors[i] =
            CanonicalDescriptor{factory.entries.size(),
                                uint32_t(distributions[i].entries.size())};
        factory.entries.insert(factory.entries.end(),
                               std::make_move_iterator(distributions[i].entries.begin()),
                               std::make_move_iterator(distributions[i].entries.end()));
    }
    factory.raw.resize(forms.size());
    for (size_t i = 0; i < forms.size(); i++) {
        auto distribution_id = [&](PrefixKey key) {
            return uint32_t(std::lower_bound(canonical_keys.begin(), canonical_keys.end(), key) -
                            canonical_keys.begin());
        };
        factory.raw[i] = RawCanonicalPair{
            forms[i].raw,
            CanonicalRef{distribution_id(forms[i].selected.key), forms[i].selected.row_map},
            CanonicalRef{distribution_id(forms[i].complement.key),
                         forms[i].complement.row_map}};
    }

    size_t checks = std::min<size_t>(16, factory.raw.size());
    for (size_t check = 0; check < checks; check++) {
        size_t index = checks == 1 ? 0 : check * (factory.raw.size() - 1) / (checks - 1);
        const RawCanonicalPair& item = factory.raw[index];
        for (int complement = 0; complement < 2; complement++) {
            const CanonicalRef& reference = complement ? item.complement : item.selected;
            const CanonicalDescriptor& descriptor = factory.descriptors[reference.distribution];
            std::vector<Entry> expanded;
            expanded.reserve(descriptor.count);
            for (uint32_t entry_index = 0; entry_index < descriptor.count; entry_index++) {
                Entry entry = factory.entries[descriptor.offset + entry_index];
                entry.mask = transform_pair_mask(entry.mask, reference.row_map);
                entry.mask = std::min(entry.mask, swap_token_planes(entry.mask));
                expanded.push_back(entry);
            }
            Distribution direct = build_distribution(item.raw, columns, complement != 0);
            direct = quotient_token_planes(std::move(direct));
            if (!entries_equal(std::move(expanded), std::move(direct.entries))) {
                throw std::runtime_error("canonical distribution validation failed");
            }
        }
    }
    return factory;
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
        result += __builtin_popcount(unsigned(key & row_mask));
        key >>= ROW_BITS;
    }
    return result;
}

static std::vector<Edge> read_edges(const std::string& path, uint64_t start,
                                    uint64_t end, uint64_t filter_mod,
                                    uint64_t filter_id, U128& labelled_weight,
                                    uint64_t& records) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path);
    char magic[8];
    uint32_t columns;
    uint64_t count;
    input.read(magic, 8);
    input.read(reinterpret_cast<char*>(&columns), sizeof(columns));
    input.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!input || std::memcmp(magic, ORBIT_MAGIC, 7) || columns != COLUMNS) {
        throw std::runtime_error("invalid orbit file");
    }
    if (!end) end = count;
    if (start > end || end > count) throw std::runtime_error("invalid record range");
    // A range and an ownership filter are independent constraints.  Earlier
    // code silently widened a filtered range to the whole file, allowing two
    // distinct manifest items to compute the same records.
    uint64_t read_start = start;
    uint64_t read_end = end;
    input.seekg(std::streamoff(20 + read_start * sizeof(OrbitRecord)));
    std::vector<Edge> edges;
    const uint64_t span = read_end - read_start;
    const uint64_t filtered_span = filter_mod ? span / filter_mod : span;
    edges.reserve(size_t(filtered_span / 2 + 1));
    for (uint64_t index = read_start; index < read_end; index++) {
        OrbitRecord record;
        input.read(reinterpret_cast<char*>(&record), sizeof(record));
        if (!input) throw std::runtime_error("truncated orbit file");
        PrefixKey left = left_prefix(record.key);
        if (filter_mod && mix64(left) % filter_mod != filter_id) continue;
        records++;
        labelled_weight += record.weight;
        int cells = cell_count(record.key);
        if (cells * 2 <= CELLS) {
            uint8_t factor = cells * 2 < CELLS ? 2 : 1;
            edges.push_back(Edge{left, right_prefix(record.key), record.weight, factor});
        }
    }
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        if (a.right != b.right) return a.right < b.right;
        return a.left < b.left;
    });
    return edges;
}

static std::vector<PrefixKey> unique_lefts(const std::vector<Edge>& edges) {
    std::vector<PrefixKey> result;
    result.reserve(edges.size());
    for (const Edge& edge : edges) result.push_back(edge.left);
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

static std::vector<PrefixKey> unique_rights(const std::vector<Edge>& edges) {
    std::vector<PrefixKey> result;
    result.reserve(edges.size());
    // read_edges() already sorts by (right,left).
    for (const Edge& edge : edges) {
        if (result.empty() || result.back() != edge.right) {
            result.push_back(edge.right);
        }
    }
    return result;
}

// Resolve labelled prefixes once when a work item is loaded.  Production
// solvers revisit edges while constructing every device batch; keeping the
// dense ID beside that immutable edge order avoids a hash-table probe in each
// recurring descriptor build without enlarging the on-disk record ABI.
static std::vector<uint32_t> resolve_edge_left_ids(
    const std::vector<Edge>& edges, const std::vector<PrefixKey>& left_keys) {
    if (left_keys.size() > UINT32_MAX) {
        throw std::overflow_error("left prefix index exceeds uint32_t");
    }
    std::unordered_map<PrefixKey, uint32_t> left_index;
    left_index.reserve(left_keys.size() * 2);
    for (size_t index = 0; index < left_keys.size(); index++) {
        left_index.emplace(left_keys[index], uint32_t(index));
    }
    std::vector<uint32_t> result;
    result.reserve(edges.size());
    for (const Edge& edge : edges) result.push_back(left_index.at(edge.left));
    return result;
}

static uint64_t cpu_join(const Entry* lhs, size_t lhs_count, const Entry* rhs,
                         size_t rhs_count) {
    uint64_t result = 0;
    for (size_t i = 0; i < lhs_count; i++) {
        for (size_t j = 0; j < rhs_count; j++) {
            if (!(lhs[i].mask & rhs[j].mask)) result += lhs[i].weight * rhs[j].weight;
        }
    }
    return result;
}
