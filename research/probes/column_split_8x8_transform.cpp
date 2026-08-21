#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

using U128 = unsigned __int128;

struct OrbitRecord {
    uint64_t key;
    uint64_t weight;
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

static std::array<uint8_t, 8> column_order(uint8_t selected) {
    if (__builtin_popcount(unsigned(selected)) != 4)
        throw std::runtime_error("selected column mask must have four bits");
    std::array<uint8_t, 8> order{};
    unsigned output = 0;
    for (unsigned column = 0; column < 8; column++)
        if (selected & (uint8_t(1) << column))
            order[output++] = uint8_t(column);
    for (unsigned column = 0; column < 8; column++)
        if (!(selected & (uint8_t(1) << column)))
            order[output++] = uint8_t(column);
    return order;
}

static uint64_t permute_columns(uint64_t key,
                                const std::array<uint8_t, 8>& order) {
    uint64_t result = 0;
    for (unsigned row = 0; row < 8; row++)
        for (unsigned output = 0; output < 8; output++) {
            unsigned source = 8 * (7 - row) + order[output];
            unsigned destination = 8 * (7 - row) + output;
            result |= ((key >> source) & 1U) << destination;
        }
    return result;
}

static uint64_t inverse_permute_columns(
    uint64_t key, const std::array<uint8_t, 8>& order) {
    uint64_t result = 0;
    for (unsigned row = 0; row < 8; row++)
        for (unsigned output = 0; output < 8; output++) {
            unsigned source = 8 * (7 - row) + output;
            unsigned destination = 8 * (7 - row) + order[output];
            result |= ((key >> source) & 1U) << destination;
        }
    return result;
}

static uint64_t mix64(uint64_t value) {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static uint32_t left_prefix(uint64_t key) {
    uint32_t result = 0;
    for (unsigned row = 0; row < 8; row++) {
        unsigned shift = 8 * (7 - row);
        result = (result << 4) | uint32_t((key >> shift) & 15U);
    }
    return result;
}

static uint64_t transform_key(uint64_t key, bool transpose,
                              const std::array<uint8_t, 8>& order) {
    if (transpose) key = transpose_grid(key);
    return permute_columns(key, order);
}

static uint64_t inverse_transform_key(uint64_t key, bool transpose,
                                      const std::array<uint8_t, 8>& order) {
    key = inverse_permute_columns(key, order);
    return transpose ? transpose_grid(key) : key;
}

static void read_header(FILE* input, uint64_t& records) {
    char magic[8];
    uint32_t columns = 0;
    if (std::fread(magic, sizeof(magic), 1, input) != 1 ||
        std::fread(&columns, sizeof(columns), 1, input) != 1 ||
        std::fread(&records, sizeof(records), 1, input) != 1 ||
        std::memcmp(magic, "R8ORB01", 7) || columns != 8)
        throw std::runtime_error("invalid R8ORB01 input");
}

static void write_header(FILE* output, uint64_t records) {
    const char magic[8] = {'R', '8', 'O', 'R', 'B', '0', '1', 0};
    const uint32_t columns = 8;
    if (std::fwrite(magic, sizeof(magic), 1, output) != 1 ||
        std::fwrite(&columns, sizeof(columns), 1, output) != 1 ||
        std::fwrite(&records, sizeof(records), 1, output) != 1)
        throw std::runtime_error("cannot write output header");
}

static void print_u128(U128 value) {
    char digits[40];
    unsigned count = 0;
    do {
        digits[count++] = char('0' + value % 10);
        value /= 10;
    } while (value);
    while (count) std::putchar(digits[--count]);
}

int main(int argc, char** argv) {
    try {
        if (argc >= 8 && std::strcmp(argv[1], "extract-owner") == 0) {
            const char* output_path = argv[2];
            bool transpose = std::strcmp(argv[3], "H") == 0;
            if (!transpose && std::strcmp(argv[3], "V") != 0) return 2;
            uint8_t selected = uint8_t(std::stoul(argv[4], nullptr, 0));
            std::array<uint8_t, 8> order = column_order(selected);
            uint64_t shards = std::stoull(argv[5]);
            uint64_t owner = std::stoull(argv[6]);
            if (!shards || owner >= shards) return 2;
            std::string temporary = std::string(output_path) + ".tmp";
            FILE* output = std::fopen(temporary.c_str(), "wb+");
            if (!output) throw std::runtime_error("cannot open output");
            write_header(output, 0);
            uint64_t input_records = 0, retained_records = 0;
            U128 input_weight = 0, retained_weight = 0;
            for (int argument = 7; argument < argc; argument++) {
                FILE* input = std::fopen(argv[argument], "rb");
                if (!input) throw std::runtime_error("cannot open input");
                uint64_t records = 0;
                read_header(input, records);
                for (uint64_t index = 0; index < records; index++) {
                    OrbitRecord record{};
                    if (std::fread(&record, sizeof(record), 1, input) != 1 ||
                        !record.weight)
                        throw std::runtime_error("invalid orbit record");
                    uint64_t original = record.key;
                    record.key = transform_key(record.key, transpose, order);
                    if (input_records < 8192 &&
                        inverse_transform_key(record.key, transpose, order) !=
                            original)
                        throw std::runtime_error(
                            "split transform is not invertible");
                    input_records++;
                    input_weight += record.weight;
                    if (mix64(left_prefix(record.key)) % shards != owner)
                        continue;
                    if (std::fwrite(&record, sizeof(record), 1, output) != 1)
                        throw std::runtime_error("cannot write orbit record");
                    retained_records++;
                    retained_weight += record.weight;
                }
                if (std::fgetc(input) != EOF || std::fclose(input) != 0)
                    throw std::runtime_error("orbit input I/O failure");
            }
            if (std::fseek(output, 0, SEEK_SET) != 0)
                throw std::runtime_error("cannot seek output header");
            write_header(output, retained_records);
            if (std::fclose(output) != 0 ||
                std::rename(temporary.c_str(), output_path) != 0)
                throw std::runtime_error("cannot publish output");
            std::printf("COLUMN_SPLIT_EXTRACT output=%s inputs=%d "
                        "input_records=%llu retained_records=%llu "
                        "orientation=%c columns=0x%02x owner=%llu/%llu "
                        "input_weight=",
                        output_path, argc - 7,
                        (unsigned long long)input_records,
                        (unsigned long long)retained_records,
                        transpose ? 'H' : 'V', unsigned(selected),
                        (unsigned long long)owner,
                        (unsigned long long)shards);
            print_u128(input_weight);
            std::printf(" retained_weight=");
            print_u128(retained_weight);
            std::printf(" inverse=OK\n");
            return 0;
        }
        if (argc != 5) {
            std::fprintf(stderr,
                         "usage: %s INPUT.orbits OUTPUT.orbits "
                         "ORIENTATION=V|H SELECTED_COLUMNS\n"
                         "       %s extract-owner OUTPUT.orbits "
                         "ORIENTATION SELECTED_COLUMNS SHARDS OWNER "
                         "INPUT.orbits...\n",
                         argv[0],
                         argv[0]);
            return 2;
        }
        bool transpose = std::strcmp(argv[3], "H") == 0;
        if (!transpose && std::strcmp(argv[3], "V") != 0) return 2;
        uint8_t selected = uint8_t(std::stoul(argv[4], nullptr, 0));
        std::array<uint8_t, 8> order = column_order(selected);
        FILE* input = std::fopen(argv[1], "rb");
        if (!input) throw std::runtime_error("cannot open input");
        uint64_t records = 0;
        read_header(input, records);
        std::string temporary = std::string(argv[2]) + ".tmp";
        FILE* output = std::fopen(temporary.c_str(), "wb");
        if (!output) throw std::runtime_error("cannot open output");
        write_header(output, records);
        U128 weight = 0;
        for (uint64_t index = 0; index < records; index++) {
            OrbitRecord record{};
            if (std::fread(&record, sizeof(record), 1, input) != 1 ||
                !record.weight)
                throw std::runtime_error("invalid orbit record");
            uint64_t original = record.key;
            record.key = transform_key(record.key, transpose, order);
            uint64_t round_trip =
                inverse_transform_key(record.key, transpose, order);
            if (round_trip != original ||
                __builtin_popcountll(record.key) !=
                    __builtin_popcountll(original))
                throw std::runtime_error("split transform is not invertible");
            if (std::fwrite(&record, sizeof(record), 1, output) != 1)
                throw std::runtime_error("cannot write orbit record");
            weight += record.weight;
        }
        if (std::fgetc(input) != EOF || std::fclose(input) != 0 ||
            std::fclose(output) != 0)
            throw std::runtime_error("orbit file I/O failure");
        if (std::rename(temporary.c_str(), argv[2]) != 0)
            throw std::runtime_error("cannot publish transformed file");
        std::printf("COLUMN_SPLIT_TRANSFORM input=%s output=%s records=%llu "
                    "orientation=%c columns=0x%02x weight=",
                    argv[1], argv[2], (unsigned long long)records,
                    transpose ? 'H' : 'V', unsigned(selected));
        print_u128(weight);
        std::printf(" inverse=OK\n");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
