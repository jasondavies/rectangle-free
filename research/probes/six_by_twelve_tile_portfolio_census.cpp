#define SIX_BY_TWELVE_PROJECTION_CENSUS_NO_MAIN
#include "six_by_twelve_projection_census.cpp"

namespace {

static PrefixKey extract_cut(U128 key, uint16_t columns) {
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

static std::vector<U128> read_tile_sample(const char* path, uint64_t wanted) {
    std::ifstream input(path, std::ios::binary);
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    input.read(magic, 8);
    input.read(reinterpret_cast<char*>(&columns), 4);
    input.read(reinterpret_cast<char*>(&count), 8);
    if (!input || std::memcmp(magic, ORBIT_MAGIC, 7) || columns != COLUMNS ||
        !wanted || wanted > count)
        throw std::runtime_error("invalid 6x12 tile sample");
    std::vector<U128> result(wanted);
    for (uint64_t sample = 0; sample < wanted; ++sample) {
        const uint64_t index = uint64_t(
            (U128(sample * 2 + 1) * count) / (U128(2) * wanted));
        OrbitRecord record{};
        input.seekg(std::streamoff(20 + index * sizeof(record)));
        input.read(reinterpret_cast<char*>(&record), sizeof(record));
        result[sample] =
            (U128(record.meta & WIDE_ORBIT_KEY_MASK) << 64) | record.low;
    }
    if (!input) throw std::runtime_error("truncated 6x12 tile sample");
    return result;
}

static uint64_t tile_cost(const Distribution& left,
                          const Distribution& right) {
    Stats stats;
    std::priority_queue<Heavy, std::vector<Heavy>, MinHeavy> unused;
    join_stats(left, right, stats, unused, 0);
    if (stats.tiles > UINT64_MAX)
        throw std::overflow_error("per-join tile count exceeds uint64");
    return uint64_t(stats.tiles);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 4) {
            std::fprintf(stderr,
                         "Usage: %s SAMPLE [RECORDS=64] [PAIR_MASK=0x67]\n",
                         argv[0]);
            return 2;
        }
        const uint64_t wanted =
            argc >= 3 ? std::strtoull(argv[2], nullptr, 10) : 64;
        const uint32_t pair_mask =
            argc >= 4 ? std::strtoul(argv[3], nullptr, 0) : 0x67;
        if (__builtin_popcount(pair_mask) != PREFIX_PAIRS)
            throw std::runtime_error("pair mask must contain five row pairs");
        initialise_tables();
        const std::vector<U128> records = read_tile_sample(argv[1], wanted);
        std::vector<uint16_t> cuts;
        for (uint16_t mask = 0; mask < (1U << COLUMNS); ++mask)
            if ((mask & 1U) && __builtin_popcount(mask) == 6)
                cuts.push_back(mask);
        if (cuts.size() != 462) throw std::logic_error("bad cut count");
        constexpr uint16_t all_columns = (1U << COLUMNS) - 1U;
        std::vector<std::vector<uint64_t>> costs(
            cuts.size(), std::vector<uint64_t>(records.size()));
        const double start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
        for (long long record = 0; record < (long long)records.size(); ++record) {
            for (size_t cut = 0; cut < cuts.size(); ++cut) {
                const PrefixKey first = extract_cut(records[size_t(record)],
                                                    cuts[cut]);
                const PrefixKey second = extract_cut(
                    records[size_t(record)], all_columns ^ cuts[cut]);
                const auto first_selected =
                    make_distribution(first, false, pair_mask);
                const auto second_selected =
                    make_distribution(second, false, pair_mask);
                const auto first_complement =
                    make_distribution(first, true, pair_mask);
                const auto second_complement =
                    make_distribution(second, true, pair_mask);
                costs[cut][size_t(record)] =
                    tile_cost(first_selected, second_selected) +
                    tile_cost(first_complement, second_complement);
            }
        }
        const auto baseline_it =
            std::find(cuts.begin(), cuts.end(), uint16_t(0x03f));
        const size_t baseline = size_t(baseline_it - cuts.begin());
        const U128 baseline_cost = std::accumulate(
            costs[baseline].begin(), costs[baseline].end(), U128(0));
        std::vector<uint64_t> portfolio(records.size(), UINT64_MAX);
        std::vector<size_t> menu;
        std::vector<bool> in_menu(cuts.size(), false);
        for (unsigned round = 0; round < 16; ++round) {
            size_t choice = 0;
            U128 choice_cost = U128(-1);
            for (size_t cut = 0; cut < cuts.size(); ++cut) {
                if (in_menu[cut]) continue;
                U128 candidate = 0;
                for (size_t record = 0; record < records.size(); ++record)
                    candidate += std::min(portfolio[record],
                                          costs[cut][record]);
                if (candidate < choice_cost) {
                    choice = cut;
                    choice_cost = candidate;
                }
            }
            menu.push_back(choice);
            in_menu[choice] = true;
            for (size_t record = 0; record < records.size(); ++record)
                portfolio[record] =
                    std::min(portfolio[record], costs[choice][record]);
            std::printf(
                "TILE_PORTFOLIO_6X12 size=%u added=0x%03x ratio=%.12f "
                "tiles=%s\n", round + 1, cuts[choice],
                double(choice_cost) / double(baseline_cost),
                u128_string(choice_cost).c_str());
        }
        U128 oracle = 0;
        for (size_t record = 0; record < records.size(); ++record) {
            uint64_t best = UINT64_MAX;
            for (size_t cut = 0; cut < cuts.size(); ++cut)
                best = std::min(best, costs[cut][record]);
            oracle += best;
        }
        std::printf(
            "TILE_PORTFOLIO_6X12_SUMMARY records=%zu cuts=%zu "
            "pair_mask=0x%03x baseline_tiles=%s oracle_ratio=%.12f "
            "seconds=%.6f exact=STRUCTURAL_OK\n",
            records.size(), cuts.size(), pair_mask,
            u128_string(baseline_cost).c_str(),
            double(oracle) / double(baseline_cost), seconds_now() - start);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
