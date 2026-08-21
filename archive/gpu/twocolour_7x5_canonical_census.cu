#define CANONICAL_PREFIX_CACHE
#define GRID_ROWS 7
#define GRID_COLUMNS 5
#define LEFT_COLUMNS 2
#define RIGHT_COLUMNS 3
#define ORBIT_ROW_BITS 8

#define main twocolour_census_hidden_main
#include "../../twocolour_7x7_gpu.cu"
#undef main

static PrefixKey compact_five_column_key(uint64_t orbit_key) {
    PrefixKey result = 0;
    for (int row = 0; row < ROWS; row++) {
        unsigned shift = ORBIT_ROW_BITS * (ROWS - 1U - row);
        result = (result << 5) | PrefixKey((orbit_key >> shift) & 31U);
    }
    return result;
}

static uint64_t percentile(const std::vector<uint64_t>& sorted, double fraction) {
    size_t index = size_t(fraction * double(sorted.size() - 1));
    return sorted[index];
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s RECT7X5.orbits\n", argv[0]);
        return 2;
    }
    double total_start = seconds_now();
    initialise_tables();

    std::ifstream input(argv[1], std::ios::binary);
    if (!input) throw std::runtime_error("cannot open orbit file");
    char magic[8];
    uint32_t columns = 0;
    uint64_t record_count = 0;
    input.read(magic, sizeof(magic));
    input.read(reinterpret_cast<char*>(&columns), sizeof(columns));
    input.read(reinterpret_cast<char*>(&record_count), sizeof(record_count));
    constexpr uint64_t expected_records = 136758;
    if (!input || std::memcmp(magic, "R7ORB01", 7) || columns != 5 ||
        record_count != expected_records) {
        throw std::runtime_error("invalid complete 7x5 orbit file");
    }

    std::vector<PrefixKey> prefixes;
    prefixes.reserve(record_count);
    U128 labelled_weight = 0;
    for (uint64_t index = 0; index < record_count; index++) {
        OrbitRecord record{};
        input.read(reinterpret_cast<char*>(&record), sizeof(record));
        if (!input) throw std::runtime_error("truncated orbit file");
        prefixes.push_back(
            canonical_prefix(compact_five_column_key(record.key), 5).key);
        labelled_weight += record.weight;
    }
    char trailing;
    if (input.read(&trailing, 1)) throw std::runtime_error("trailing orbit data");
    std::sort(prefixes.begin(), prefixes.end());
    if (std::adjacent_find(prefixes.begin(), prefixes.end()) != prefixes.end() ||
        labelled_weight != (U128(1) << 35)) {
        throw std::runtime_error("orbit uniqueness or labelled-weight check failed");
    }

    const PrefixKey full_mask = (PrefixKey(1) << 35) - 1;
    uint64_t self_complementary = 0;
    for (PrefixKey prefix : prefixes) {
        CanonicalForm canonical = canonical_prefix(prefix, 5);
        if (canonical.key != prefix) {
            throw std::runtime_error("orbit record is not canonical in solver encoding");
        }
        PrefixKey complement = canonical_prefix(prefix ^ full_mask, 5).key;
        if (!std::binary_search(prefixes.begin(), prefixes.end(), complement)) {
            throw std::runtime_error("canonical complement is absent from orbit census");
        }
        if (complement == prefix) self_complementary++;
    }
    std::printf("ORBITS records=%llu labelled_weight=%s self_complementary=%llu "
                "complement_pairs=%llu exact=OK\n",
                (unsigned long long)record_count,
                u128_string(labelled_weight).c_str(),
                (unsigned long long)self_complementary,
                (unsigned long long)((record_count - self_complementary) / 2));

    std::vector<uint64_t> entry_counts(prefixes.size());
    std::vector<uint64_t> maximum_weights(prefixes.size());
    double build_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
    for (long long index = 0; index < (long long)prefixes.size(); index++) {
        Distribution distribution =
            build_distribution(prefixes[size_t(index)], 5, false);
        entry_counts[size_t(index)] = distribution.entries.size();
        uint64_t maximum_weight = 0;
        for (const Entry& entry : distribution.entries) {
            maximum_weight = std::max(maximum_weight, entry.weight);
        }
        maximum_weights[size_t(index)] = maximum_weight;
    }
    double build_seconds = seconds_now() - build_start;

    U128 total_entries = 0;
    uint64_t maximum_entries = 0;
    uint64_t maximum_weight = 0;
    std::array<uint64_t, 65> log2_histogram{};
    for (uint64_t count : entry_counts) {
        total_entries += count;
        maximum_entries = std::max(maximum_entries, count);
        unsigned bucket = count <= 1
                              ? 0
                              : 64U - unsigned(__builtin_clzll(count - 1));
        log2_histogram[bucket]++;
    }
    for (uint64_t weight : maximum_weights) {
        maximum_weight = std::max(maximum_weight, weight);
    }
    unsigned weight_bits = maximum_weight
                               ? 64U - unsigned(__builtin_clzll(maximum_weight))
                               : 0;
    std::vector<uint64_t> sorted_counts = entry_counts;
    std::sort(sorted_counts.begin(), sorted_counts.end());

    std::vector<size_t> heavy(prefixes.size());
    std::iota(heavy.begin(), heavy.end(), 0);
    size_t heavy_count = std::min<size_t>(16, heavy.size());
    std::partial_sort(heavy.begin(), heavy.begin() + heavy_count, heavy.end(),
                      [&](size_t lhs, size_t rhs) {
                          if (entry_counts[lhs] != entry_counts[rhs]) {
                              return entry_counts[lhs] > entry_counts[rhs];
                          }
                          return prefixes[lhs] < prefixes[rhs];
                      });

    double entries_as_double = double(total_entries);
    constexpr double gib = 1024.0 * 1024.0 * 1024.0;
    std::printf("CENSUS distributions=%zu total_entries=%s max_entries=%llu "
                "mean_entries=%.3f p50=%llu p90=%llu p95=%llu p99=%llu "
                "p999=%llu max_weight=%llu weight_bits=%u "
                "packed42_22=%s build_seconds=%.6f threads=%d exact=OK\n",
                prefixes.size(), u128_string(total_entries).c_str(),
                (unsigned long long)maximum_entries,
                entries_as_double / double(prefixes.size()),
                (unsigned long long)percentile(sorted_counts, 0.50),
                (unsigned long long)percentile(sorted_counts, 0.90),
                (unsigned long long)percentile(sorted_counts, 0.95),
                (unsigned long long)percentile(sorted_counts, 0.99),
                (unsigned long long)percentile(sorted_counts, 0.999),
                (unsigned long long)maximum_weight, weight_bits,
                weight_bits <= 22 ? "YES" : "NO",
                build_seconds,
#ifdef _OPENMP
                omp_get_max_threads()
#else
                1
#endif
    );
    std::printf("CAPACITY bytes8_gib=%.6f bytes12_gib=%.6f bytes16_gib=%.6f\n",
                entries_as_double * 8.0 / gib,
                entries_as_double * 12.0 / gib,
                entries_as_double * 16.0 / gib);
    std::printf("HISTOGRAM");
    for (size_t bucket = 0; bucket < log2_histogram.size(); bucket++) {
        if (log2_histogram[bucket]) {
            std::printf(" le2^%zu=%llu", bucket,
                        (unsigned long long)log2_histogram[bucket]);
        }
    }
    std::printf("\n");
    for (size_t rank = 0; rank < heavy_count; rank++) {
        size_t index = heavy[rank];
        std::printf("HEAVY rank=%zu prefix=%llu entries=%llu\n", rank + 1,
                    (unsigned long long)prefixes[index],
                    (unsigned long long)entry_counts[index]);
    }
    std::printf("TOTAL seconds=%.6f\n", seconds_now() - total_start);
    return 0;
}
