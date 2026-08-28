#define SIX_BY_TWELVE_PROJECTION_CENSUS_NO_MAIN
#include "six_by_twelve_projection_census.cpp"

namespace {

static std::vector<U128> read_keys(const char* path) {
    std::ifstream input(path, std::ios::binary);
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    input.read(magic, 8);
    input.read(reinterpret_cast<char*>(&columns), 4);
    input.read(reinterpret_cast<char*>(&count), 8);
    if (!input || std::memcmp(magic, ORBIT_MAGIC, 7) || columns != COLUMNS)
        throw std::runtime_error("invalid 6x12 corpus");
    std::vector<U128> keys(count);
    for (uint64_t index = 0; index < count; ++index) {
        OrbitRecord record{};
        input.read(reinterpret_cast<char*>(&record), sizeof(record));
        keys[index] =
            (U128(record.meta & WIDE_ORBIT_KEY_MASK) << 64) | record.low;
    }
    if (!input) throw std::runtime_error("truncated 6x12 corpus");
    return keys;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) {
            std::fprintf(stderr, "Usage: %s CORPUS [PAIR_MASK=0x67]\n",
                         argv[0]);
            return 2;
        }
        const uint32_t pair_mask =
            argc == 3 ? std::strtoul(argv[2], nullptr, 0) : 0x67;
        if (__builtin_popcount(pair_mask) != PREFIX_PAIRS)
            throw std::runtime_error("pair mask must contain five row pairs");
        initialise_tables();
        const std::vector<U128> keys = read_keys(argv[1]);
        std::vector<uint64_t> comparisons(keys.size()), tiles(keys.size());
        const double start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
        for (long long index = 0; index < (long long)keys.size(); ++index) {
            const PrefixKey left = half_key(keys[size_t(index)], 0);
            const PrefixKey right = half_key(keys[size_t(index)], 6);
            const auto left_selected =
                make_distribution(left, false, pair_mask);
            const auto right_selected =
                make_distribution(right, false, pair_mask);
            const auto left_complement =
                make_distribution(left, true, pair_mask);
            const auto right_complement =
                make_distribution(right, true, pair_mask);
            Stats stats;
            std::priority_queue<Heavy, std::vector<Heavy>, MinHeavy> unused;
            join_stats(left_selected, right_selected, stats, unused, 0);
            join_stats(left_complement, right_complement, stats, unused, 0);
            if (stats.comparisons > UINT64_MAX || stats.tiles > UINT64_MAX)
                throw std::overflow_error("per-record cost exceeds uint64");
            comparisons[size_t(index)] = uint64_t(stats.comparisons);
            tiles[size_t(index)] = uint64_t(stats.tiles);
        }
        U128 comparison_total = 0, tile_total = 0;
        for (size_t index = 0; index < keys.size(); ++index) {
            comparison_total += comparisons[index];
            tile_total += tiles[index];
        }
        std::printf("SIX_BY_TWELVE_FIXED_COST records=%zu pair_mask=0x%03x "
                    "comparisons=%s tiles=%s seconds=%.6f exact=STRUCTURAL_OK\n",
                    keys.size(), pair_mask,
                    u128_string(comparison_total).c_str(),
                    u128_string(tile_total).c_str(), seconds_now() - start);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
