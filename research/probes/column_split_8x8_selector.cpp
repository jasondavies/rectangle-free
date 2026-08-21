#include <atomic>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <unordered_map>

#define main prefix_hierarchy_selector_unused_main
#include "prefix_hierarchy_8x8_census.cpp"
#undef main

namespace {

struct SupportCounts {
    uint32_t selected = 0;
    uint32_t complement = 0;
};

struct Candidate {
    bool transpose = false;
    uint8_t columns = 0;
};

static uint64_t transpose_grid(uint64_t key) {
    uint64_t result = 0;
    for (unsigned row = 0; row < 8; row++)
        for (unsigned column = 0; column < 8; column++) {
            unsigned source = 8 * (7 - row) + column;
            unsigned destination = 8 * (7 - column) + row;
            result |= ((key >> source) & 1U) << destination;
        }
    return result;
}

static uint32_t extract_half(uint64_t key, uint8_t columns) {
    uint32_t result = 0;
    for (unsigned row = 0; row < 8; row++) {
        uint32_t pattern = 0;
        unsigned output = 0;
        for (unsigned column = 0; column < 8; column++) {
            if (!(columns & (uint8_t(1) << column))) continue;
            unsigned source = 8 * (7 - row) + column;
            pattern |= uint32_t((key >> source) & 1U) << output++;
        }
        result = (result << 4) | pattern;
    }
    return result;
}

static uint32_t canonical_half(uint32_t key) {
    std::array<uint8_t, 8> rows{};
    for (int row = 7; row >= 0; row--) {
        rows[size_t(row)] = uint8_t(key & 15U);
        key >>= 4;
    }
    std::array<unsigned, 4> permutation{0, 1, 2, 3};
    uint32_t best = UINT32_MAX;
    do {
        std::array<uint8_t, 8> transformed{};
        for (unsigned row = 0; row < 8; row++)
            for (unsigned output = 0; output < 4; output++)
                transformed[row] |=
                    ((rows[row] >> permutation[output]) & 1U) << output;
        std::sort(transformed.begin(), transformed.end());
        uint32_t candidate = 0;
        for (uint8_t row : transformed) candidate = (candidate << 4) | row;
        best = std::min(best, candidate);
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    return best;
}

static std::vector<Candidate> enumerate_candidates(const std::string& mode) {
    if (mode == "vh")
        return {Candidate{false, 0x0f}, Candidate{true, 0x0f}};
    if (mode != "all") throw std::runtime_error("unknown selector mode");
    std::vector<Candidate> result;
    for (unsigned transpose = 0; transpose < 2; transpose++)
        for (unsigned columns = 0; columns < 256; columns++)
            if ((columns & 1U) && __builtin_popcount(columns) == 4)
                result.push_back(Candidate{bool(transpose), uint8_t(columns)});
    if (result.size() != 70) throw std::logic_error("bad candidate count");
    return result;
}

static uint64_t swap_token_planes(uint64_t mask) {
    constexpr uint64_t plane = (UINT64_C(1) << PAIRS) - 1;
    return ((mask & plane) << PAIRS) | ((mask >> PAIRS) & plane);
}

static uint32_t quotient_support_size(uint32_t prefix, bool complement) {
    std::vector<uint64_t> support = build_distribution(prefix, complement);
    uint32_t result = 0;
    uint64_t expanded = 0;
    for (uint64_t mask : support) {
        uint64_t swapped = swap_token_planes(mask);
        if (mask > swapped) continue;
        result++;
        expanded += mask == swapped ? 1 : 2;
    }
    if (expanded != support.size())
        throw std::runtime_error("token quotient support invariant failed");
    return result;
}

static std::vector<uint32_t> enumerate_canonical_halves() {
    std::vector<uint32_t> keys;
    std::array<uint8_t, 8> rows{};
    auto visit = [&](auto&& self, unsigned position, unsigned minimum) -> void {
        if (position == rows.size()) {
            uint32_t key = 0;
            for (uint8_t row : rows) key = (key << 4) | row;
            keys.push_back(canonical_half(key));
            return;
        }
        for (unsigned value = minimum; value < 16; value++) {
            rows[position] = uint8_t(value);
            self(self, position + 1, value);
        }
    };
    visit(visit, 0, 0);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    if (keys.size() != 25207)
        throw std::runtime_error("canonical 8x4 census is incomplete");
    return keys;
}

static std::array<uint8_t, 8> column_order(uint8_t selected) {
    std::array<uint8_t, 8> result{};
    unsigned output = 0;
    for (unsigned column = 0; column < 8; column++)
        if (selected & (uint8_t(1) << column))
            result[output++] = uint8_t(column);
    for (unsigned column = 0; column < 8; column++)
        if (!(selected & (uint8_t(1) << column)))
            result[output++] = uint8_t(column);
    return result;
}

static uint64_t permute_columns(uint64_t key, uint8_t selected) {
    std::array<uint8_t, 8> order = column_order(selected);
    uint64_t result = 0;
    for (unsigned row = 0; row < 8; row++)
        for (unsigned output = 0; output < 8; output++) {
            unsigned source = 8 * (7 - row) + order[output];
            unsigned destination = 8 * (7 - row) + output;
            result |= ((key >> source) & 1U) << destination;
        }
    return result;
}

static void write_header(FILE* output, uint64_t records) {
    const char magic[8] = {'R', '8', 'O', 'R', 'B', '0', '1', 0};
    const uint32_t columns = 8;
    if (std::fwrite(magic, sizeof(magic), 1, output) != 1 ||
        std::fwrite(&columns, sizeof(columns), 1, output) != 1 ||
        std::fwrite(&records, sizeof(records), 1, output) != 1)
        throw std::runtime_error("cannot write output header");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 6) {
            std::cerr << "usage: " << argv[0]
                      << " INPUT.orbits OUTPUT.orbits [START=0] [END=0] "
                         "[MODE=all|vh]\n";
            return 2;
        }
        uint64_t begin = argc > 3 ? std::stoull(argv[3]) : 0;
        uint64_t end = argc > 4 ? std::stoull(argv[4]) : 0;
        std::string mode = argc > 5 ? argv[5] : "all";
        initialise_tables();
        double start = seconds_now();
        std::vector<uint32_t> canonical = enumerate_canonical_halves();
        std::vector<SupportCounts> support(canonical.size());
#pragma omp parallel for schedule(dynamic, 1)
        for (long long index = 0; index < (long long)canonical.size(); index++)
            support[size_t(index)] = SupportCounts{
                quotient_support_size(canonical[size_t(index)], false),
                quotient_support_size(canonical[size_t(index)], true)};
        std::unordered_map<uint32_t, uint32_t> support_index;
        support_index.reserve(canonical.size() * 2);
        for (size_t index = 0; index < canonical.size(); index++)
            support_index.emplace(canonical[index], uint32_t(index));
        double cache_seconds = seconds_now() - start;

        FILE* input = std::fopen(argv[1], "rb");
        if (!input) throw std::runtime_error("cannot open input");
        char magic[8];
        uint32_t columns = 0;
        uint64_t total_records = 0;
        if (std::fread(magic, sizeof(magic), 1, input) != 1 ||
            std::fread(&columns, sizeof(columns), 1, input) != 1 ||
            std::fread(&total_records, sizeof(total_records), 1, input) != 1 ||
            std::memcmp(magic, "R8ORB01", 7) || columns != 8)
            throw std::runtime_error("invalid input");
        if (!end || end > total_records) end = total_records;
        if (begin >= end) return 2;
        if (std::fseek(input, long(20 + begin * sizeof(OrbitRecord)), SEEK_SET))
            throw std::runtime_error("cannot seek input");
        std::vector<OrbitRecord> records(size_t(end - begin));
        if (std::fread(records.data(), sizeof(OrbitRecord), records.size(),
                       input) != records.size() || std::fclose(input) != 0)
            throw std::runtime_error("cannot read input range");

        const std::vector<Candidate> candidates = enumerate_candidates(mode);
        std::array<std::atomic<uint64_t>, 70> histogram{};
        std::atomic<uint64_t> reversed{0};
        double selector_start = seconds_now();
#pragma omp parallel for schedule(static)
        for (long long record_index = 0;
             record_index < (long long)records.size(); record_index++) {
            uint64_t original = records[size_t(record_index)].key;
            U128 best_cost = U128(-1);
            size_t best = 0;
            uint8_t best_columns = 0;
            for (size_t candidate = 0; candidate < candidates.size();
                 candidate++) {
                uint64_t oriented = candidates[candidate].transpose
                    ? transpose_grid(original) : original;
                uint8_t first_columns = candidates[candidate].columns;
                uint8_t second_columns = uint8_t(~first_columns);
                uint32_t first_key =
                    canonical_half(extract_half(oriented, first_columns));
                uint32_t second_key =
                    canonical_half(extract_half(oriented, second_columns));
                const SupportCounts& first =
                    support[support_index.at(first_key)];
                const SupportCounts& second =
                    support[support_index.at(second_key)];
                U128 cost = U128(first.selected) * second.selected +
                            U128(first.complement) * second.complement;
                uint8_t execution_columns = first_columns;
                if (uint64_t(first.selected) + first.complement >
                    uint64_t(second.selected) + second.complement)
                    execution_columns = second_columns;
                if (cost < best_cost) {
                    best_cost = cost;
                    best = candidate;
                    best_columns = execution_columns;
                }
            }
            uint64_t oriented = candidates[best].transpose
                ? transpose_grid(original) : original;
            records[size_t(record_index)].key =
                permute_columns(oriented, best_columns);
            histogram[best].fetch_add(1, std::memory_order_relaxed);
            reversed.fetch_add(
                best_columns != candidates[best].columns,
                std::memory_order_relaxed);
        }
        double selector_seconds = seconds_now() - selector_start;

        std::string temporary = std::string(argv[2]) + ".tmp";
        FILE* output = std::fopen(temporary.c_str(), "wb");
        if (!output) throw std::runtime_error("cannot open output");
        write_header(output, records.size());
        if (std::fwrite(records.data(), sizeof(OrbitRecord), records.size(),
                        output) != records.size() || std::fclose(output) != 0 ||
            std::rename(temporary.c_str(), argv[2]) != 0)
            throw std::runtime_error("cannot publish output");
        std::cout << std::setprecision(12)
                  << "COLUMN_SPLIT_SELECTOR input=" << argv[1]
                  << " output=" << argv[2]
                  << " records=" << records.size()
                  << " mode=" << mode
                  << " cache_seconds=" << cache_seconds
                  << " selector_seconds=" << selector_seconds
                  << " records_per_second="
                  << records.size() / selector_seconds
                  << " reversed=" << reversed.load() << '\n';
        for (size_t candidate = 0; candidate < candidates.size(); candidate++) {
            uint64_t count = histogram[candidate].load();
            if (!count) continue;
            std::cout << "COLUMN_SPLIT_CHOICE candidate="
                      << (candidates[candidate].transpose ? 'H' : 'V')
                      << ":0x" << std::hex
                      << unsigned(candidates[candidate].columns) << std::dec
                      << " records=" << count << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
