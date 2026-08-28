#define GRID_ROWS 6
#define GRID_COLUMNS 12
#define LEFT_COLUMNS 6
#define RIGHT_COLUMNS 6
#define ORBIT_ROW_BITS 12
#define ORBIT_MAGIC "R6W1201"
#define TWCOLOUR_WIDE_ORBIT_RECORD 1

#include "../../src/gpu/twocolour_gpu_common.cuh"

#include <unordered_map>

namespace {

constexpr uint16_t ALL_COLUMNS = (1U << COLUMNS) - 1U;
constexpr std::array<uint16_t, 4> MENU = {0x03f, 0x09f, 0x0e7, 0x06f};

struct Counts { uint32_t selected = 0, complement = 0; };

static PrefixKey extract(U128 key, uint16_t columns) {
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

static U128 concatenate(PrefixKey left, PrefixKey right) {
    U128 result = 0;
    constexpr PrefixKey row_mask = 63;
    for (unsigned row = 0; row < ROWS; ++row) {
        const unsigned shift = 6 * (ROWS - 1U - row);
        const PrefixKey a = (left >> shift) & row_mask;
        const PrefixKey b = (right >> shift) & row_mask;
        // The production wide-record reader names the low six row bits the
        // resident left half.  Keep the first argument there; the high half
        // is streamed on the right.
        result = (result << 12) | (U128(b) << 6) | a;
    }
    return result;
}

static void write_corpus(const char* path, const std::vector<OrbitRecord>& source,
                         const std::vector<U128>& keys, unsigned repeat) {
    std::ofstream output(path, std::ios::binary);
    const char magic[8] = ORBIT_MAGIC;
    const uint32_t columns = COLUMNS;
    const uint64_t count = uint64_t(source.size()) * repeat;
    output.write(magic, 8);
    output.write(reinterpret_cast<const char*>(&columns), 4);
    output.write(reinterpret_cast<const char*>(&count), 8);
    for (unsigned copy = 0; copy < repeat; ++copy) {
        for (size_t i = 0; i < source.size(); ++i) {
            OrbitRecord record = source[i];
            const uint64_t weight = record.meta >> WIDE_ORBIT_KEY_BITS;
            record.low = uint64_t(keys[i]);
            record.meta = (weight << WIDE_ORBIT_KEY_BITS) |
                          uint64_t(keys[i] >> 64);
            output.write(reinterpret_cast<const char*>(&record), sizeof(record));
        }
    }
    if (!output) throw std::runtime_error("failed writing adaptive corpus");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        std::fprintf(stderr,
                     "Usage: %s INPUT OUTPUT_BASE OUTPUT_REPEAT "
                     "[MENU_SIZE=2] [REPEAT=64]\n", argv[0]);
        return 2;
    }
    const unsigned menu_size = argc >= 5 ? std::strtoul(argv[4], nullptr, 10) : 2;
    const unsigned repeat = argc >= 6 ? std::strtoul(argv[5], nullptr, 10) : 64;
    if (!menu_size || menu_size > MENU.size() || !repeat)
        throw std::runtime_error("invalid menu or repeat");
    std::ifstream input(argv[1], std::ios::binary);
    char magic[8]; uint32_t width; uint64_t count;
    input.read(magic, 8); input.read(reinterpret_cast<char*>(&width), 4);
    input.read(reinterpret_cast<char*>(&count), 8);
    if (!input || std::memcmp(magic, ORBIT_MAGIC, 7) || width != COLUMNS)
        throw std::runtime_error("invalid input corpus");
    std::vector<OrbitRecord> records(count);
    input.read(reinterpret_cast<char*>(records.data()),
               std::streamsize(records.size() * sizeof(OrbitRecord)));
    if (!input) throw std::runtime_error("truncated input corpus");
    std::vector<U128> keys(count);
    for (size_t i = 0; i < records.size(); ++i)
        keys[i] = (U128(records[i].meta & WIDE_ORBIT_KEY_MASK) << 64) |
                  records[i].low;
    std::vector<PrefixKey> raw(records.size() * menu_size * 2);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)records.size(); ++i) {
        for (unsigned menu = 0; menu < menu_size; ++menu) {
            raw[(size_t(i) * menu_size + menu) * 2] = extract(keys[size_t(i)], MENU[menu]);
            raw[(size_t(i) * menu_size + menu) * 2 + 1] = extract(keys[size_t(i)], ALL_COLUMNS ^ MENU[menu]);
        }
    }
    std::vector<PrefixKey> canonical(raw.size());
#pragma omp parallel for schedule(dynamic, 32)
    for (long long i = 0; i < (long long)raw.size(); ++i)
        canonical[size_t(i)] = canonical_prefix(raw[size_t(i)], 6).key;
    std::vector<PrefixKey> unique = canonical;
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    initialise_tables();
    std::vector<Counts> counts(unique.size());
    const double build_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
    for (long long i = 0; i < (long long)unique.size(); ++i) {
        counts[size_t(i)] = Counts{
            uint32_t(quotient_token_planes(build_distribution(unique[size_t(i)], 6, false)).entries.size()),
            uint32_t(quotient_token_planes(build_distribution(unique[size_t(i)], 6, true)).entries.size())};
    }
    std::unordered_map<PrefixKey, uint32_t> ids;
    ids.reserve(unique.size() * 2);
    for (size_t i = 0; i < unique.size(); ++i) ids.emplace(unique[i], i);
    std::vector<U128> output_keys(records.size());
    std::array<uint64_t, MENU.size()> selected{};
    U128 baseline = 0, chosen = 0;
#pragma omp parallel for schedule(static) reduction(+:baseline,chosen)
    for (long long i = 0; i < (long long)records.size(); ++i) {
        uint64_t best = UINT64_MAX;
        unsigned choice = 0;
        bool swap = false;
        for (unsigned menu = 0; menu < menu_size; ++menu) {
            const size_t offset = (size_t(i) * menu_size + menu) * 2;
            const Counts& a = counts[ids.at(canonical[offset])];
            const Counts& b = counts[ids.at(canonical[offset + 1])];
            const uint64_t cost = uint64_t(a.selected) * b.selected +
                                  uint64_t(a.complement) * b.complement;
            if (!menu) baseline += cost;
            if (cost < best) {
                best = cost;
                choice = menu;
                swap = uint64_t(a.selected) + a.complement >
                       uint64_t(b.selected) + b.complement;
            }
        }
        const size_t offset = (size_t(i) * menu_size + choice) * 2;
        output_keys[size_t(i)] = swap ? concatenate(raw[offset + 1], raw[offset])
                                      : concatenate(raw[offset], raw[offset + 1]);
        chosen += best;
#pragma omp atomic
        selected[choice]++;
    }
    write_corpus(argv[2], records, output_keys, 1);
    write_corpus(argv[3], records, output_keys, repeat);
    std::printf("SIX_BY_TWELVE_ADAPTIVE records=%zu menu=%u unique_halves=%zu ratio=%.12f build_seconds=%.6f selections=",
                records.size(), menu_size, unique.size(), double(chosen) / double(baseline), seconds_now() - build_start);
    for (unsigned i = 0; i < menu_size; ++i) std::printf("%s0x%03x:%llu", i ? "," : "", MENU[i], (unsigned long long)selected[i]);
    std::printf(" OK\n");
}
