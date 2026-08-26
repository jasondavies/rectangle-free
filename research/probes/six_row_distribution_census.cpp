#define GRID_ROWS 6
#define GRID_COLUMNS 6
#define LEFT_COLUMNS 3
#define RIGHT_COLUMNS 3
#define ORBIT_ROW_BITS 6
#define ORBIT_MAGIC "R6ORB01"

#include "../../src/gpu/twocolour_gpu_common.cuh"

#include <atomic>
#include <limits>
#include <mutex>

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::fprintf(stderr, "Usage: %s CANONICAL_6x6.orbits [START END]\n",
                     argv[0]);
        return 2;
    }
    std::ifstream input(argv[1], std::ios::binary);
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    input.read(magic, sizeof(magic));
    input.read(reinterpret_cast<char*>(&columns), sizeof(columns));
    input.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!input || std::memcmp(magic, ORBIT_MAGIC, 7) || columns != 6 ||
        count != UINT64_C(251610)) {
        throw std::runtime_error("invalid canonical 6x6 corpus");
    }
    uint64_t start = argc >= 3 ? std::strtoull(argv[2], nullptr, 10) : 0;
    uint64_t end = argc >= 4 ? std::strtoull(argv[3], nullptr, 10) : count;
    if (start > end || end > count) throw std::runtime_error("invalid range");
    std::vector<PrefixKey> keys(end - start);
    input.seekg(std::streamoff(20 + start * sizeof(OrbitRecord)));
    for (PrefixKey& key : keys) {
        OrbitRecord record;
        input.read(reinterpret_cast<char*>(&record), sizeof(record));
        if (!input) throw std::runtime_error("truncated canonical corpus");
        // R6ORB01 retains the generator's ten-bit row stride at intermediate
        // depths. Repack the six active bits per row for build_distribution.
        PrefixKey stored = record.key;
        PrefixKey compact = 0;
        std::array<uint16_t, ROWS> rows{};
        for (int row = ROWS - 1; row >= 0; row--) {
            rows[size_t(row)] = uint16_t(stored & 1023U);
            stored >>= 10;
        }
        for (int row = 0; row < ROWS; row++)
            compact = (compact << 6) | (rows[size_t(row)] & 63U);
        key = compact;
    }

    initialise_tables();
    std::atomic<bool> failed{false};
    std::atomic<uint64_t> completed{0};
    uint64_t entries = 0;
    uint64_t fixed_entries = 0;
    uint64_t maximum_entries = 0;
    uint64_t maximum_weight = 0;
    uint64_t class_values = 0;
    uint32_t maximum_classes = 0;
    PrefixKey maximum_entry_key = 0;
    double begin = seconds_now();
#pragma omp parallel for schedule(dynamic, 1) reduction(+:entries,fixed_entries,class_values) reduction(max:maximum_weight)
    for (long long index = 0; index < (long long)keys.size(); index++) {
        try {
            Distribution distribution = quotient_token_planes(
                build_distribution(keys[size_t(index)], 6, false));
            entries += distribution.entries.size();
            std::vector<std::pair<uint32_t, uint8_t>> classes;
            for (const Entry& entry : distribution.entries) {
                maximum_weight = std::max(maximum_weight, entry.weight);
                uint8_t orbit = uint8_t(token_plane_orbit_size(entry.mask));
                fixed_entries += orbit == 1;
                std::pair<uint32_t, uint8_t> value{uint32_t(entry.weight), orbit};
                if (std::find(classes.begin(), classes.end(), value) ==
                    classes.end()) {
                    classes.push_back(value);
                }
            }
            class_values += classes.size();
#pragma omp critical(six_row_census_maximum)
            {
                if (distribution.entries.size() > maximum_entries) {
                    maximum_entries = distribution.entries.size();
                    maximum_entry_key = keys[size_t(index)];
                }
                maximum_classes = std::max<uint32_t>(maximum_classes,
                                                     classes.size());
            }
        } catch (const std::exception& error) {
#pragma omp critical(six_row_census_error)
            std::fprintf(stderr, "distribution %lld failed: %s\n", index,
                         error.what());
            failed.store(true, std::memory_order_relaxed);
        }
        uint64_t done = completed.fetch_add(1, std::memory_order_relaxed) + 1;
        if (done % 10000 == 0 || done == keys.size()) {
#pragma omp critical(six_row_census_progress)
            std::fprintf(stderr, "completed=%llu/%zu seconds=%.3f\n",
                         (unsigned long long)done, keys.size(),
                         seconds_now() - begin);
        }
    }
    if (failed.load(std::memory_order_relaxed)) return 1;
    std::printf(
        "SIX_ROW_DISTRIBUTION_CENSUS start=%llu end=%llu distributions=%zu "
        "quotient_entries=%llu fixed_entries=%llu maximum_entries=%llu "
        "maximum_entry_key=%llu class_values=%llu maximum_classes=%u "
        "maximum_weight=%llu seconds=%.6f\n",
        (unsigned long long)start, (unsigned long long)end, keys.size(),
        (unsigned long long)entries, (unsigned long long)fixed_entries,
        (unsigned long long)maximum_entries,
        (unsigned long long)maximum_entry_key,
        (unsigned long long)class_values, maximum_classes,
        (unsigned long long)maximum_weight, seconds_now() - begin);
    return 0;
}
