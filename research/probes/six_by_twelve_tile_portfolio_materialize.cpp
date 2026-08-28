#define SIX_BY_TWELVE_PROJECTION_CENSUS_NO_MAIN
#include "six_by_twelve_projection_census.cpp"

#include <filesystem>

namespace fs = std::filesystem;

namespace {

constexpr uint16_t ALL_COLUMNS = (1U << COLUMNS) - 1U;
constexpr std::array<uint16_t, 16> TILE_MENU = {
    0x03f, 0x0cf, 0x077, 0x11f, 0x07d, 0x60f, 0x05f, 0x18f,
    0x0e7, 0x51d, 0x61d, 0x273, 0x09f, 0x1c7, 0x07b, 0x25d};
constexpr std::array<unsigned, 3> PORTFOLIO_SIZES = {4, 8, 16};

static PrefixKey extract_columns(U128 key, uint16_t columns) {
    PrefixKey result = 0;
    for (unsigned row = 0; row < ROWS; ++row) {
        const unsigned row_shift = COLUMNS * (ROWS - 1U - row);
        PrefixKey pattern = 0;
        unsigned output = 0;
        for (unsigned column = 0; column < COLUMNS; ++column) {
            if (!(columns & (1U << column))) continue;
            pattern |= PrefixKey((key >> (row_shift + column)) & 1U)
                       << output++;
        }
        result = (result << 6) | pattern;
    }
    return result;
}

static U128 concatenate_halves(PrefixKey left, PrefixKey right) {
    U128 result = 0;
    for (unsigned row = 0; row < ROWS; ++row) {
        const unsigned shift = 6 * (ROWS - 1U - row);
        const PrefixKey a = (left >> shift) & 63U;
        const PrefixKey b = (right >> shift) & 63U;
        result = (result << 12) | (U128(b) << 6) | a;
    }
    return result;
}

static uint64_t layout_entries(const Distribution& distribution) {
    uint64_t result = 0;
    for (const Bucket& bucket : distribution.buckets)
        for (const Class& item : bucket.classes)
            result += item.suffixes.size();
    return result;
}

static uint64_t layout_tiles(const Distribution& left,
                             const Distribution& right) {
    Stats stats;
    std::priority_queue<Heavy, std::vector<Heavy>, MinHeavy> unused;
    join_stats(left, right, stats, unused, 0);
    if (stats.tiles > UINT64_MAX)
        throw std::overflow_error("per-record tensor tiles exceed uint64");
    return uint64_t(stats.tiles);
}

static void write_records(const fs::path& path,
                          const std::vector<OrbitRecord>& source,
                          const std::vector<U128>& keys, unsigned repeat) {
    std::ofstream output(path, std::ios::binary);
    const char magic[8] = ORBIT_MAGIC;
    const uint32_t columns = COLUMNS;
    const uint64_t count = uint64_t(source.size()) * repeat;
    output.write(magic, 8);
    output.write(reinterpret_cast<const char*>(&columns), 4);
    output.write(reinterpret_cast<const char*>(&count), 8);
    for (unsigned copy = 0; copy < repeat; ++copy) {
        for (size_t index = 0; index < source.size(); ++index) {
            OrbitRecord record = source[index];
            const uint64_t weight = record.meta >> WIDE_ORBIT_KEY_BITS;
            record.low = uint64_t(keys[index]);
            record.meta = (weight << WIDE_ORBIT_KEY_BITS) |
                          uint64_t(keys[index] >> 64);
            output.write(reinterpret_cast<const char*>(&record), sizeof(record));
        }
    }
    if (!output) throw std::runtime_error("failed writing " + path.string());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 6) {
            std::fprintf(stderr,
                         "Usage: %s INPUT OUTPUT_DIR [REPEAT=64] "
                         "[PAIR_MASK=0x67] [COST_MATRIX]\n", argv[0]);
            return 2;
        }
        const unsigned repeat =
            argc >= 4 ? std::strtoul(argv[3], nullptr, 10) : 64;
        const uint32_t pair_mask =
            argc >= 5 ? std::strtoul(argv[4], nullptr, 0) : 0x67;
        if (!repeat || __builtin_popcount(pair_mask) != PREFIX_PAIRS)
            throw std::runtime_error("invalid repeat or pair mask");
        std::ifstream input(argv[1], std::ios::binary);
        char magic[8];
        uint32_t columns = 0;
        uint64_t count = 0;
        input.read(magic, 8);
        input.read(reinterpret_cast<char*>(&columns), 4);
        input.read(reinterpret_cast<char*>(&count), 8);
        if (!input || std::memcmp(magic, ORBIT_MAGIC, 7) || columns != COLUMNS)
            throw std::runtime_error("invalid 6x12 input corpus");
        std::vector<OrbitRecord> records(count);
        input.read(reinterpret_cast<char*>(records.data()),
                   std::streamsize(records.size() * sizeof(OrbitRecord)));
        if (!input) throw std::runtime_error("truncated 6x12 input corpus");
        std::vector<U128> keys(count);
        for (size_t index = 0; index < records.size(); ++index)
            keys[index] =
                (U128(records[index].meta & WIDE_ORBIT_KEY_MASK) << 64) |
                records[index].low;
        initialise_tables();
        std::array<std::vector<U128>, PORTFOLIO_SIZES.size()> outputs;
        for (auto& output : outputs) output.resize(records.size());
        std::array<std::vector<uint64_t>, PORTFOLIO_SIZES.size()> record_tiles;
        std::array<std::vector<uint8_t>, PORTFOLIO_SIZES.size()> record_menus;
        for (size_t item = 0; item < PORTFOLIO_SIZES.size(); ++item) {
            record_tiles[item].resize(records.size());
            record_menus[item].resize(records.size());
        }
        std::array<U128, PORTFOLIO_SIZES.size()> tile_totals{};
        std::vector<uint64_t> cut_costs;
        if (argc == 6) cut_costs.resize(records.size() * TILE_MENU.size());
        std::array<std::array<uint64_t, TILE_MENU.size()>,
                   PORTFOLIO_SIZES.size()> selections{};
        const double start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
        for (long long record = 0; record < (long long)records.size(); ++record) {
            uint64_t best_tiles = UINT64_MAX;
            U128 best_key = 0;
            size_t checkpoint = 0;
            size_t best_menu = 0;
            for (size_t menu = 0; menu < TILE_MENU.size(); ++menu) {
                const PrefixKey first =
                    extract_columns(keys[size_t(record)], TILE_MENU[menu]);
                const PrefixKey second = extract_columns(
                    keys[size_t(record)], ALL_COLUMNS ^ TILE_MENU[menu]);
                const auto first_selected =
                    make_distribution(first, false, pair_mask);
                const auto second_selected =
                    make_distribution(second, false, pair_mask);
                const auto first_complement =
                    make_distribution(first, true, pair_mask);
                const auto second_complement =
                    make_distribution(second, true, pair_mask);
                const uint64_t tiles =
                    layout_tiles(first_selected, second_selected) +
                    layout_tiles(first_complement, second_complement);
                if (!cut_costs.empty())
                    cut_costs[size_t(record) * TILE_MENU.size() + menu] =
                        tiles;
                if (tiles < best_tiles) {
                    best_tiles = tiles;
                    best_menu = menu;
                    const uint64_t first_entries =
                        layout_entries(first_selected) +
                        layout_entries(first_complement);
                    const uint64_t second_entries =
                        layout_entries(second_selected) +
                        layout_entries(second_complement);
                    best_key = first_entries <= second_entries
                        ? concatenate_halves(first, second)
                        : concatenate_halves(second, first);
                }
                if (menu + 1 == PORTFOLIO_SIZES[checkpoint]) {
                    outputs[checkpoint][size_t(record)] = best_key;
                    record_tiles[checkpoint][size_t(record)] = best_tiles;
                    record_menus[checkpoint][size_t(record)] =
                        uint8_t(best_menu);
                    ++checkpoint;
                }
            }
        }
        fs::create_directories(argv[2]);
        if (!cut_costs.empty()) {
            std::ofstream costs(argv[5], std::ios::binary);
            const char magic[8] = {'R','6','T','I','L','E','1','\0'};
            const uint32_t menu_size = TILE_MENU.size();
            const uint64_t record_count = records.size();
            costs.write(magic, sizeof(magic));
            costs.write(reinterpret_cast<const char*>(&menu_size),
                        sizeof(menu_size));
            costs.write(reinterpret_cast<const char*>(&record_count),
                        sizeof(record_count));
            costs.write(reinterpret_cast<const char*>(TILE_MENU.data()),
                        sizeof(TILE_MENU));
            costs.write(reinterpret_cast<const char*>(cut_costs.data()),
                        std::streamsize(cut_costs.size() * sizeof(uint64_t)));
            if (!costs)
                throw std::runtime_error("failed writing tile-cost matrix");
        }
        for (size_t item = 0; item < PORTFOLIO_SIZES.size(); ++item) {
            for (size_t record = 0; record < records.size(); ++record) {
                tile_totals[item] += record_tiles[item][record];
                selections[item][record_menus[item][record]]++;
            }
            const unsigned size = PORTFOLIO_SIZES[item];
            const std::string stem = "rect6x12-tile-p" + std::to_string(size);
            write_records(fs::path(argv[2]) / (stem + "-seed.orbits"),
                          records, outputs[item], 1);
            write_records(fs::path(argv[2]) /
                              (stem + "-x" + std::to_string(repeat) +
                               ".orbits"),
                          records, outputs[item], repeat);
            std::printf("TILE_PORTFOLIO_MATERIALIZE size=%u records=%zu "
                        "tiles=%s selections=", size, records.size(),
                        u128_string(tile_totals[item]).c_str());
            for (size_t menu = 0; menu < size; ++menu)
                std::printf("%s0x%03x:%llu", menu ? "," : "",
                            TILE_MENU[menu],
                            (unsigned long long)selections[item][menu]);
            std::printf("\n");
        }
        std::printf("TILE_PORTFOLIO_MATERIALIZE seconds=%.6f exact=OK\n",
                    seconds_now() - start);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
