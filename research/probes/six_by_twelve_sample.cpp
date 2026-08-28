#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using U128 = unsigned __int128;

namespace {

constexpr int rows = 6;
constexpr int parent_columns = 10;
constexpr int columns = 12;
constexpr int midpoint = rows * columns / 2;
constexpr uint64_t high_mask = (UINT64_C(1) << (rows * columns - 64)) - 1U;

struct OrbitRecord {
    uint64_t key;
    uint64_t weight;
};

struct WideRecord {
    uint64_t low;
    uint64_t meta;
};

uint64_t mix64(uint64_t value) {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

template <int width>
std::array<uint16_t, rows> unpack_rows(U128 key) {
    std::array<uint16_t, rows> result{};
    const U128 mask = (U128(1) << width) - 1U;
    for (int row = rows - 1; row >= 0; --row) {
        result[size_t(row)] = uint16_t(key & mask);
        key >>= width;
    }
    return result;
}

template <int width>
U128 pack_rows(const std::array<uint16_t, rows>& patterns) {
    U128 result = 0;
    for (uint16_t pattern : patterns) result = (result << width) | pattern;
    return result;
}

U128 canonical_key(const std::array<uint16_t, rows>& input) {
    std::array<unsigned, rows> permutation{0, 1, 2, 3, 4, 5};
    U128 best = ~U128(0);
    do {
        std::array<uint8_t, columns> vectors{};
        std::array<uint8_t, columns> degrees{};
        std::array<unsigned, columns> column_order{};
        for (int column = 0; column < columns; ++column) {
            uint8_t vector = 0;
            uint8_t degree = 0;
            for (unsigned source : permutation) {
                vector = uint8_t((vector << 1) |
                                 ((input[source] >> column) & 1U));
                degree += uint8_t((input[source] >> column) & 1U);
            }
            vectors[size_t(column)] = vector;
            degrees[size_t(column)] = degree;
            column_order[size_t(column)] = unsigned(column);
        }
        std::sort(column_order.begin(), column_order.end(),
                  [&](unsigned lhs, unsigned rhs) {
                      return degrees[lhs] != degrees[rhs]
                          ? degrees[lhs] < degrees[rhs]
                          : vectors[lhs] < vectors[rhs];
                  });
        std::array<uint16_t, rows> candidate{};
        for (int destination = 0; destination < rows; ++destination) {
            uint16_t pattern = 0;
            const unsigned source = permutation[size_t(destination)];
            for (int position = 0; position < columns; ++position)
                pattern |= uint16_t((input[source] >>
                                     column_order[size_t(position)]) & 1U)
                           << position;
            candidate[size_t(destination)] = pattern;
        }
        best = std::min(best, pack_rows<columns>(candidate));
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    return best;
}

void read_header(std::ifstream& input, const char* magic, uint32_t width,
                 uint64_t& count) {
    char actual_magic[8];
    uint32_t actual_width = 0;
    input.read(actual_magic, sizeof(actual_magic));
    input.read(reinterpret_cast<char*>(&actual_width), sizeof(actual_width));
    input.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!input || std::memcmp(actual_magic, magic, 7) || actual_width != width)
        throw std::runtime_error("invalid orbit input header");
}

void write_wide(const std::string& path, const std::vector<U128>& keys) {
    std::ofstream output(path, std::ios::binary);
    const char magic[8] = "R6W1201";
    const uint32_t width = columns;
    const uint64_t count = keys.size();
    output.write(magic, sizeof(magic));
    output.write(reinterpret_cast<const char*>(&width), sizeof(width));
    output.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (U128 key : keys) {
        const uint64_t low = uint64_t(key);
        const uint64_t high = uint64_t(key >> 64);
        if (high & ~high_mask) throw std::runtime_error("wide key overflow");
        // Unit weights are sufficient for a performance/validation corpus.
        const WideRecord record{low, (UINT64_C(1) << 8) | high};
        output.write(reinterpret_cast<const char*>(&record), sizeof(record));
    }
    if (!output) throw std::runtime_error("cannot write " + path);
}

void sample(const std::string& input_path, const std::string& output_path,
            uint64_t parent_samples, uint64_t extensions) {
    if (!parent_samples || !extensions || extensions > 4096)
        throw std::runtime_error("invalid sample dimensions");
    std::ifstream input(input_path, std::ios::binary);
    uint64_t count = 0;
    read_header(input, "R6ORB01", parent_columns, count);
    std::vector<OrbitRecord> records(static_cast<size_t>(count), OrbitRecord{});
    input.read(reinterpret_cast<char*>(records.data()),
               std::streamsize(records.size() * sizeof(OrbitRecord)));
    if (!input) throw std::runtime_error("truncated parent orbit file");
    parent_samples = std::min(parent_samples, count);
    const size_t thread_count =
#ifdef _OPENMP
        static_cast<size_t>(omp_get_max_threads());
#else
        1;
#endif
    std::vector<std::vector<U128>> thread_keys(thread_count);
#pragma omp parallel for schedule(dynamic, 1)
    for (long long sample_index = 0;
         sample_index < static_cast<long long>(parent_samples);
         ++sample_index) {
        const uint64_t index =
            (U128(uint64_t(sample_index) * 2 + 1) * count) /
            (U128(2) * parent_samples);
        const OrbitRecord parent = records[size_t(index)];
        const auto original = unpack_rows<parent_columns>(parent.key);
        const int original_cells = __builtin_popcountll(parent.key);
        const int orientations = original_cells < rows * parent_columns / 2
            ? 2 : 1;
#ifdef _OPENMP
        auto& local = thread_keys[size_t(omp_get_thread_num())];
#else
        auto& local = thread_keys[0];
#endif
        for (int orientation = 0; orientation < orientations; ++orientation) {
            std::array<uint16_t, rows> base{};
            const uint16_t parent_mask = (1U << parent_columns) - 1U;
            for (int row = 0; row < rows; ++row)
                base[size_t(row)] = orientation
                    ? uint16_t(original[size_t(row)] ^ parent_mask)
                    : original[size_t(row)];
            const int base_cells = orientation
                ? rows * parent_columns - original_cells : original_cells;
            const uint64_t seed = mix64(index ^ (uint64_t(orientation) << 63));
            const uint64_t step = (mix64(seed) | 1U) & 4095U;
            const uint64_t offset = mix64(seed + 1) & 4095U;
            for (uint64_t extension = 0; extension < extensions; ++extension) {
                const unsigned pair = unsigned((offset + extension * step) & 4095U);
                const unsigned first = pair & 63U;
                const unsigned second = pair >> 6;
                if (base_cells + __builtin_popcount(first) +
                                     __builtin_popcount(second) > midpoint)
                    continue;
                auto child = base;
                for (int row = 0; row < rows; ++row) {
                    child[size_t(row)] |= uint16_t((first >> row) & 1U) << 10;
                    child[size_t(row)] |= uint16_t((second >> row) & 1U) << 11;
                }
                local.push_back(canonical_key(child));
            }
        }
    }
    std::vector<U128> keys;
    size_t total = 0;
    for (const auto& local : thread_keys) total += local.size();
    keys.reserve(total);
    for (auto& local : thread_keys)
        keys.insert(keys.end(), local.begin(), local.end());
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    write_wide(output_path, keys);
    std::printf("SIX_BY_TWELVE_SAMPLE parents=%llu extensions=%llu "
                "retained_candidates=%zu unique=%zu output=%s OK\n",
                (unsigned long long)parent_samples,
                (unsigned long long)extensions, total, keys.size(),
                output_path.c_str());
}

void promote_seed(const std::string& input_path, const std::string& output_path) {
    std::ifstream input(input_path, std::ios::binary);
    uint64_t count = 0;
    read_header(input, "R6ORB01", 6, count);
    std::vector<U128> keys;
    keys.reserve(size_t(count));
    for (uint64_t index = 0; index < count; ++index) {
        OrbitRecord record{};
        input.read(reinterpret_cast<char*>(&record), sizeof(record));
        auto source = unpack_rows<6>(record.key);
        std::array<uint16_t, rows> promoted{};
        for (int row = 0; row < rows; ++row)
            promoted[size_t(row)] = source[size_t(row)];
        keys.push_back(pack_rows<columns>(promoted));
    }
    if (!input) throw std::runtime_error("truncated canonical seed");
    write_wide(output_path, keys);
    std::printf("SIX_BY_TWELVE_SEED records=%zu output=%s OK\n",
                keys.size(), output_path.c_str());
}

void repeat_sample(const std::string& input_path, const std::string& output_path,
                   uint64_t repetitions) {
    if (!repetitions) throw std::runtime_error("zero repetitions");
    std::ifstream input(input_path, std::ios::binary);
    uint64_t count = 0;
    read_header(input, "R6W1201", columns, count);
    std::vector<U128> source;
    source.reserve(size_t(count));
    for (uint64_t index = 0; index < count; ++index) {
        WideRecord record{};
        input.read(reinterpret_cast<char*>(&record), sizeof(record));
        source.push_back((U128(record.meta & high_mask) << 64) | record.low);
    }
    if (!input) throw std::runtime_error("truncated wide sample");
    std::vector<U128> repeated;
    repeated.reserve(size_t(count * repetitions));
    for (uint64_t repetition = 0; repetition < repetitions; ++repetition)
        repeated.insert(repeated.end(), source.begin(), source.end());
    write_wide(output_path, repeated);
    std::printf("SIX_BY_TWELVE_REPEAT source=%llu repetitions=%llu records=%zu "
                "output=%s OK\n", (unsigned long long)count,
                (unsigned long long)repetitions, repeated.size(),
                output_path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 6 && !std::strcmp(argv[1], "sample")) {
            sample(argv[2], argv[3], std::strtoull(argv[4], nullptr, 10),
                   std::strtoull(argv[5], nullptr, 10));
            return 0;
        }
        if (argc == 4 && !std::strcmp(argv[1], "promote-seed")) {
            promote_seed(argv[2], argv[3]);
            return 0;
        }
        if (argc == 5 && !std::strcmp(argv[1], "repeat")) {
            repeat_sample(argv[2], argv[3],
                          std::strtoull(argv[4], nullptr, 10));
            return 0;
        }
        std::fprintf(stderr,
                     "Usage:\n  %s sample INPUT_6X10 OUTPUT_6X12 "
                     "PARENTS EXTENSIONS\n  %s promote-seed INPUT_6X6 "
                     "OUTPUT_6X12\n  %s repeat INPUT_6X12 OUTPUT_6X12 N\n",
                     argv[0], argv[0], argv[0]);
        return 2;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
