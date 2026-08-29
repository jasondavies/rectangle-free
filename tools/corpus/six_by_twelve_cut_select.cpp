#define GRID_ROWS 6
#define GRID_COLUMNS 12
#define LEFT_COLUMNS 6
#define RIGHT_COLUMNS 6
#define ORBIT_ROW_BITS 12
#define ORBIT_MAGIC "R6W1201"
#define TWCOLOUR_WIDE_ORBIT_RECORD 1

#include "../../src/gpu/twocolour_gpu_common.cuh"
#include "../../src/gpu/six_by_six_cache_artifact.hpp"
#include "../../src/gpu/six_by_six_cache_mapped.hpp"

#include <atomic>
#include <fcntl.h>
#include <filesystem>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using U128 = unsigned __int128;

constexpr uint64_t RAW_MULTISET_COUNT = 119877472;
constexpr uint64_t EXPECTED_CANONICAL_COUNT = 251610;
constexpr uint64_t EXPECTED_SUPPORT_COUNT = 3469067567ULL;
constexpr uint16_t ALL_COLUMNS = (1U << COLUMNS) - 1U;
constexpr std::array<uint16_t, 16> TILE_MENU = {
    0x03f, 0x0cf, 0x077, 0x11f, 0x07d, 0x60f, 0x05f, 0x18f,
    0x0e7, 0x51d, 0x61d, 0x273, 0x09f, 0x1c7, 0x07b, 0x25d};

struct TableHeader {
    char magic[8];
    uint32_t version;
    uint32_t rows;
    uint64_t entries;
};
static_assert(sizeof(TableHeader) == 24);

struct CorpusMapping {
    int descriptor = -1;
    size_t bytes = 0;
    const uint8_t* mapping = nullptr;
    uint64_t records = 0;
};

struct MutableMapping {
    int descriptor = -1;
    size_t bytes = 0;
    uint8_t* mapping = nullptr;
};

struct alignas(64) ThreadSelections {
    std::array<uint64_t, 16> counts{};
};

static uint64_t choose[70][7];

static void initialise_choose() {
    choose[0][0] = 1;
    for (unsigned n = 1; n < 70; ++n) {
        choose[n][0] = 1;
        for (unsigned k = 1; k <= 6; ++k)
            choose[n][k] = choose[n - 1][k - 1] + choose[n - 1][k];
    }
    if (choose[69][6] != RAW_MULTISET_COUNT)
        throw std::logic_error("bad multiset census");
}

static uint32_t multiset_rank(std::array<uint8_t, 6> values) {
    std::sort(values.begin(), values.end());
    uint64_t rank = 0;
    for (unsigned index = 0; index < values.size(); ++index)
        rank += choose[unsigned(values[index]) + index][index + 1];
    if (rank >= RAW_MULTISET_COUNT)
        throw std::logic_error("multiset rank overflow");
    return uint32_t(rank);
}

static std::array<uint8_t, 6> column_vectors(PrefixKey key) {
    std::array<uint8_t, ROWS> row_patterns{};
    for (int row = ROWS - 1; row >= 0; --row) {
        row_patterns[size_t(row)] = uint8_t(key & 63U);
        key >>= 6;
    }
    std::array<uint8_t, 6> result{};
    for (unsigned column = 0; column < 6; ++column)
        for (unsigned row = 0; row < ROWS; ++row)
            result[column] |= uint8_t((row_patterns[row] >> column) & 1U)
                              << row;
    return result;
}

static std::array<uint8_t, COLUMNS> full_column_vectors(U128 key) {
    std::array<uint8_t, COLUMNS> result{};
    for (unsigned row = 0; row < ROWS; ++row) {
        const unsigned shift = COLUMNS * (ROWS - 1U - row);
        const uint16_t pattern = uint16_t(key >> shift) & ALL_COLUMNS;
        for (unsigned column = 0; column < COLUMNS; ++column)
            result[column] |= uint8_t((pattern >> column) & 1U) << row;
    }
    return result;
}

static uint32_t half_multiset_rank(
        const std::array<uint8_t, COLUMNS>& columns, uint16_t selected) {
    std::array<uint8_t, 6> values{};
    unsigned output = 0;
    for (unsigned column = 0; column < COLUMNS; ++column)
        if (selected & (1U << column)) values[output++] = columns[column];
    return multiset_rank(values);
}

static PrefixKey packed_half(U128 key, uint16_t columns) {
    PrefixKey result = 0;
    for (unsigned row = 0; row < ROWS; ++row) {
        const unsigned shift = COLUMNS * (ROWS - 1U - row);
        const uint16_t pattern = uint16_t(key >> shift) & ALL_COLUMNS;
        uint8_t half = 0;
        unsigned output = 0;
        for (unsigned column = 0; column < COLUMNS; ++column) {
            if (!(columns & (1U << column))) continue;
            half |= uint8_t((pattern >> column) & 1U) << output++;
        }
        result = (result << 6) | half;
    }
    return result;
}

static U128 concatenate(PrefixKey resident, PrefixKey streamed) {
    U128 result = 0;
    for (unsigned row = 0; row < ROWS; ++row) {
        const unsigned shift = 6 * (ROWS - 1U - row);
        const PrefixKey left = (resident >> shift) & 63U;
        const PrefixKey right = (streamed >> shift) & 63U;
        result = (result << 12) | (U128(right) << 6) | left;
    }
    return result;
}

static CorpusMapping map_corpus(const char* path) {
    CorpusMapping result;
    result.descriptor = open(path, O_RDONLY);
    if (result.descriptor < 0)
        throw std::runtime_error(std::string("cannot open ") + path);
    struct stat status{};
    if (fstat(result.descriptor, &status) || status.st_size < 20)
        throw std::runtime_error("invalid corpus size");
    result.bytes = size_t(status.st_size);
    result.mapping = static_cast<const uint8_t*>(mmap(
        nullptr, result.bytes, PROT_READ, MAP_PRIVATE, result.descriptor, 0));
    if (result.mapping == MAP_FAILED) throw std::runtime_error("mmap failed");
    uint32_t columns = 0;
    std::memcpy(&columns, result.mapping + 8, 4);
    std::memcpy(&result.records, result.mapping + 12, 8);
    if (std::memcmp(result.mapping, ORBIT_MAGIC, 7) || columns != COLUMNS ||
        result.bytes != 20 + result.records * sizeof(OrbitRecord))
        throw std::runtime_error("invalid 6x12 corpus header");
    return result;
}

static void unmap_corpus(CorpusMapping& value) {
    if (value.mapping && munmap(const_cast<uint8_t*>(value.mapping),
                                value.bytes))
        throw std::runtime_error("corpus munmap failed");
    if (value.descriptor >= 0 && close(value.descriptor))
        throw std::runtime_error("corpus close failed");
    value = CorpusMapping{};
}

static MutableMapping create_output(const char* path, size_t bytes) {
    MutableMapping result;
    result.descriptor = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (result.descriptor < 0 || ftruncate(result.descriptor, off_t(bytes)))
        throw std::runtime_error(std::string("cannot create ") + path);
    result.bytes = bytes;
    result.mapping = static_cast<uint8_t*>(mmap(
        nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
        result.descriptor, 0));
    if (result.mapping == MAP_FAILED)
        throw std::runtime_error("output mmap failed");
    return result;
}

static MutableMapping map_mutable_corpus(const char* path,
                                         uint64_t& records) {
    MutableMapping result;
    result.descriptor = open(path, O_RDWR);
    struct stat status{};
    if (result.descriptor < 0 || fstat(result.descriptor, &status) ||
        status.st_size < 20)
        throw std::runtime_error(std::string("cannot open ") + path);
    result.bytes = size_t(status.st_size);
    result.mapping = static_cast<uint8_t*>(mmap(
        nullptr, result.bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
        result.descriptor, 0));
    if (result.mapping == MAP_FAILED)
        throw std::runtime_error("corpus mmap failed");
    uint32_t columns = 0;
    std::memcpy(&columns, result.mapping + 8, 4);
    std::memcpy(&records, result.mapping + 12, 8);
    if (std::memcmp(result.mapping, ORBIT_MAGIC, 7) || columns != COLUMNS ||
        result.bytes != 20 + records * sizeof(OrbitRecord))
        throw std::runtime_error("invalid 6x12 corpus header");
    return result;
}

static void finish_output(MutableMapping& value) {
    if (msync(value.mapping, value.bytes, MS_SYNC) ||
        munmap(value.mapping, value.bytes) || close(value.descriptor))
        throw std::runtime_error("output flush failed");
    value = MutableMapping{};
}

static std::vector<PrefixKey> read_canonical_keys(const char* path) {
    std::ifstream input(path, std::ios::binary);
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    input.read(magic, 8);
    input.read(reinterpret_cast<char*>(&columns), 4);
    input.read(reinterpret_cast<char*>(&count), 8);
    if (!input || std::memcmp(magic, "R6ORB01", 7) || columns != 6 ||
        count != EXPECTED_CANONICAL_COUNT)
        throw std::runtime_error("invalid canonical 6x6 corpus");
    std::vector<PrefixKey> result(count);
    for (uint64_t index = 0; index < count; ++index) {
        uint64_t key = 0, weight = 0;
        input.read(reinterpret_cast<char*>(&key), 8);
        input.read(reinterpret_cast<char*>(&weight), 8);
        PrefixKey packed = 0;
        for (unsigned row = 0; row < ROWS; ++row) {
            const unsigned shift = 10 * (ROWS - 1U - row);
            packed = (packed << 6) | ((key >> shift) & 63U);
        }
        result[index] = packed;
    }
    if (!input) throw std::runtime_error("truncated canonical corpus");
    // The augmentation corpus retains its ten-bit row stride and its own
    // equivalent canonical convention.  Normalize into the distribution
    // factory's compact six-bit-row convention before lookup.  Each key is
    // independent; doing 720 column permutations serially here needlessly
    // dominated both offline cache builders.
#pragma omp parallel for schedule(static)
    for (long long index = 0; index < (long long)result.size(); ++index)
        result[size_t(index)] = canonical_prefix(result[size_t(index)], 6).key;
    std::sort(result.begin(), result.end());
    if (std::adjacent_find(result.begin(), result.end()) != result.end())
        throw std::runtime_error("duplicate canonical key");
    return result;
}

static void build_table(const char* canonical_path, const char* table_path) {
    initialise_tables();
    const std::vector<PrefixKey> keys = read_canonical_keys(canonical_path);
    std::vector<uint32_t> supports(keys.size());
    const double support_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
    for (long long index = 0; index < (long long)keys.size(); ++index) {
        Distribution distribution = quotient_token_planes(
            build_distribution(keys[size_t(index)], 6, false));
        if (distribution.entries.size() > UINT32_MAX)
            std::terminate();
        supports[size_t(index)] = uint32_t(distribution.entries.size());
    }
    uint64_t support_total = 0;
    for (uint32_t count : supports) support_total += count;
    if (support_total != EXPECTED_SUPPORT_COUNT)
        throw std::runtime_error("canonical support census mismatch");
    std::vector<uint32_t> complement_supports(keys.size());
    const PrefixKey full = (PrefixKey(1) << 36) - 1U;
#pragma omp parallel for schedule(static)
    for (long long index = 0; index < (long long)keys.size(); ++index) {
        const PrefixKey complement = canonical_prefix(
            keys[size_t(index)] ^ full, 6).key;
        const auto found = std::lower_bound(keys.begin(), keys.end(), complement);
        if (found == keys.end() || *found != complement) std::terminate();
        complement_supports[size_t(index)] =
            supports[size_t(found - keys.begin())];
    }

    const size_t bytes = sizeof(TableHeader) +
                         RAW_MULTISET_COUNT * sizeof(uint64_t);
    MutableMapping output = create_output(table_path, bytes);
    const TableHeader header{{'R','6','S','U','P','T','1','\0'}, 1, ROWS,
                             RAW_MULTISET_COUNT};
    std::memcpy(output.mapping, &header, sizeof(header));
    uint64_t* table = reinterpret_cast<uint64_t*>(
        output.mapping + sizeof(TableHeader));
    std::memset(table, 0, RAW_MULTISET_COUNT * sizeof(uint64_t));
    std::array<std::array<uint8_t, 64>, 720> row_images{};
    std::array<uint8_t, 6> permutation{0, 1, 2, 3, 4, 5};
    size_t permutation_count = 0;
    do {
        for (unsigned value = 0; value < 64; ++value)
            for (unsigned destination = 0; destination < 6; ++destination)
                row_images[permutation_count][value] |=
                    uint8_t((value >> permutation[destination]) & 1U)
                    << destination;
        ++permutation_count;
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    if (permutation_count != 720) throw std::logic_error("bad S6 census");
    const double expansion_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 8)
    for (long long index = 0; index < (long long)keys.size(); ++index) {
        const auto columns = column_vectors(keys[size_t(index)]);
        const uint64_t packed = uint64_t(supports[size_t(index)]) |
            (uint64_t(complement_supports[size_t(index)]) << 32);
        for (const auto& images : row_images) {
            std::array<uint8_t, 6> transformed{};
            for (unsigned column = 0; column < 6; ++column)
                transformed[column] = images[columns[column]];
            const uint32_t rank = multiset_rank(transformed);
            uint64_t expected = 0;
            if (!__atomic_compare_exchange_n(
                    table + rank, &expected, packed, false,
                    __ATOMIC_RELAXED, __ATOMIC_RELAXED) && expected != packed)
                std::terminate();
        }
    }
    uint64_t missing = 0;
#pragma omp parallel for schedule(static) reduction(+:missing)
    for (long long index = 0; index < (long long)RAW_MULTISET_COUNT; ++index)
        missing += table[size_t(index)] == 0;
    if (missing) throw std::runtime_error("support table is incomplete");
    finish_output(output);
    std::printf("SIX_BY_TWELVE_SUPPORT_TABLE canonical=%zu entries=%llu "
                "bytes=%zu support_total=%llu support_seconds=%.6f "
                "expansion_seconds=%.6f output=%s exact=OK\n",
                keys.size(), (unsigned long long)RAW_MULTISET_COUNT, bytes,
                (unsigned long long)support_total,
                expansion_start - support_start,
                seconds_now() - expansion_start, table_path);
}

struct ReadOnlyTable {
    int descriptor = -1;
    size_t bytes = 0;
    const uint8_t* mapping = nullptr;
    const uint64_t* entries = nullptr;
};

static ReadOnlyTable map_table(const char* path) {
    ReadOnlyTable result;
    result.descriptor = open(path, O_RDONLY);
    struct stat status{};
    if (result.descriptor < 0 || fstat(result.descriptor, &status))
        throw std::runtime_error("cannot open support table");
    result.bytes = size_t(status.st_size);
    if (result.bytes != sizeof(TableHeader) +
                        RAW_MULTISET_COUNT * sizeof(uint64_t))
        throw std::runtime_error("bad support table size");
    result.mapping = static_cast<const uint8_t*>(mmap(
        nullptr, result.bytes, PROT_READ, MAP_SHARED, result.descriptor, 0));
    if (result.mapping == MAP_FAILED)
        throw std::runtime_error("support table mmap failed");
    const auto* header = reinterpret_cast<const TableHeader*>(result.mapping);
    if (std::memcmp(header->magic, "R6SUPT1", 7) || header->version != 1 ||
        header->rows != ROWS || header->entries != RAW_MULTISET_COUNT)
        throw std::runtime_error("bad support table header");
    result.entries = reinterpret_cast<const uint64_t*>(
        result.mapping + sizeof(TableHeader));
    return result;
}

static void unmap_table(ReadOnlyTable& value) {
    if (munmap(const_cast<uint8_t*>(value.mapping), value.bytes) ||
        close(value.descriptor))
        throw std::runtime_error("support table close failed");
    value = ReadOnlyTable{};
}

static uint16_t permutation_rank(const std::array<uint8_t, 6>& value) {
    static constexpr uint16_t factorial[6] = {1, 1, 2, 6, 24, 120};
    uint16_t rank = 0;
    for (unsigned index = 0; index < 6; ++index) {
        unsigned smaller = 0;
        for (unsigned later = index + 1; later < 6; ++later)
            smaller += value[later] < value[index];
        rank += uint16_t(smaller * factorial[5 - index]);
    }
    return rank;
}

static void atomic_minimum(uint32_t* destination, uint32_t value) {
    uint32_t current = __atomic_load_n(destination, __ATOMIC_RELAXED);
    while (value < current && !__atomic_compare_exchange_n(
               destination, &current, value, true,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
}

static void build_cache_artifact(const char* canonical_path,
                                 const char* support_table_path,
                                 const char* output_path) {
    using namespace six_by_six_cache;
    initialise_tables();
    const std::vector<PrefixKey> keys = read_canonical_keys(canonical_path);
    if (keys.size() != CANONICAL_COUNT)
        throw std::runtime_error("unexpected canonical cache size");
    ReadOnlyTable support = map_table(support_table_path);

    // Fail before allocating the large artifact if the independently built
    // support table does not describe this exact canonical-key convention.
    for (size_t index = 0; index < std::min<size_t>(16, keys.size()); ++index) {
        const uint32_t rank = multiset_rank(column_vectors(keys[index]));
        const uint32_t expected = uint32_t(support.entries[rank]);
        const uint64_t actual = quotient_token_planes(
            build_distribution(keys[index],
                               int(six_by_six_cache::COLUMNS), false))
                                    .entries.size();
        if (actual != expected) {
            throw std::runtime_error(
                "support-table preflight mismatch: distribution=" +
                std::to_string(index) + " expected=" +
                std::to_string(expected) + " actual=" +
                std::to_string(actual) + " key=" +
                std::to_string(keys[index]) + " rank=" +
                std::to_string(rank));
        }
    }

    Header header{};
    std::memcpy(header.magic, "R6C6Q01", 8);
    header.version = FORMAT_VERSION;
    header.rows = six_by_six_cache::ROWS;
    header.columns = six_by_six_cache::COLUMNS;
    header.class_slots = CLASS_SLOTS;
    header.canonical_count = CANONICAL_COUNT;
    header.entry_count = ENTRY_COUNT;
    header.multiset_count = MULTISET_COUNT;
    uint64_t cursor = align_up(sizeof(Header));
    auto section = [&](uint64_t bytes) {
        const uint64_t result = cursor;
        cursor = align_up(cursor + bytes);
        return result;
    };
    header.keys_offset = section(CANONICAL_COUNT * sizeof(uint64_t));
    header.descriptors_offset = section(
        CANONICAL_COUNT * sizeof(Descriptor));
    header.masks_offset = section(ENTRY_COUNT * sizeof(uint32_t));
    header.ordinals_offset = section(ENTRY_COUNT * sizeof(uint8_t));
    header.class_weights_offset = section(
        CANONICAL_COUNT * CLASS_SLOTS * sizeof(uint32_t));
    header.class_orbits_offset = section(
        CANONICAL_COUNT * CLASS_SLOTS * sizeof(uint8_t));
    header.class_counts_offset = section(
        CANONICAL_COUNT * sizeof(uint8_t));
    header.references_offset = section(MULTISET_COUNT * sizeof(uint32_t));
    header.file_bytes = cursor;

    const std::string temporary =
        std::string(output_path) + ".tmp." + std::to_string(getpid());
    MutableMapping output = create_output(temporary.c_str(),
                                          size_t(header.file_bytes));
    auto* output_keys = reinterpret_cast<uint64_t*>(
        output.mapping + header.keys_offset);
    auto* descriptors = reinterpret_cast<Descriptor*>(
        output.mapping + header.descriptors_offset);
    auto* masks = reinterpret_cast<uint32_t*>(
        output.mapping + header.masks_offset);
    auto* ordinals = reinterpret_cast<uint8_t*>(
        output.mapping + header.ordinals_offset);
    auto* class_weights = reinterpret_cast<uint32_t*>(
        output.mapping + header.class_weights_offset);
    auto* class_orbits = reinterpret_cast<uint8_t*>(
        output.mapping + header.class_orbits_offset);
    auto* class_counts = reinterpret_cast<uint8_t*>(
        output.mapping + header.class_counts_offset);
    auto* references = reinterpret_cast<uint32_t*>(
        output.mapping + header.references_offset);
    std::memcpy(output_keys, keys.data(), keys.size() * sizeof(PrefixKey));
    std::memset(class_weights, 0,
                CANONICAL_COUNT * CLASS_SLOTS * sizeof(uint32_t));
    std::memset(class_orbits, 0,
                CANONICAL_COUNT * CLASS_SLOTS * sizeof(uint8_t));
    std::memset(class_counts, 0, CANONICAL_COUNT * sizeof(uint8_t));
    std::memset(references, 0xff, MULTISET_COUNT * sizeof(uint32_t));

    uint64_t entry_cursor = 0;
    for (size_t index = 0; index < keys.size(); ++index) {
        const uint32_t rank = multiset_rank(column_vectors(keys[index]));
        const uint32_t count = uint32_t(support.entries[rank]);
        descriptors[index] = Descriptor{entry_cursor, count, 0};
        entry_cursor += count;
    }
    if (entry_cursor != ENTRY_COUNT)
        throw std::runtime_error("support table/cache census mismatch");

    enum CachePackingFailure : uint32_t {
        CACHE_OK = 0,
        CACHE_SUPPORT_COUNT = 1,
        CACHE_WEIGHT_WIDTH = 2,
        CACHE_MASK_WIDTH = 3,
        CACHE_CLASS_WIDTH = 4,
    };
    std::atomic<uint32_t> failure{CACHE_OK};
    std::atomic<uint32_t> failure_index{UINT32_MAX};
    std::atomic<uint64_t> failure_detail{0};
    auto record_failure = [&](uint32_t reason, uint32_t index,
                              uint64_t detail = 0) {
        uint32_t expected = CACHE_OK;
        if (failure.compare_exchange_strong(expected, reason,
                                            std::memory_order_relaxed)) {
            failure_index.store(index, std::memory_order_relaxed);
            failure_detail.store(detail, std::memory_order_relaxed);
        }
    };
    const double distribution_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
    for (long long wide_index = 0;
         wide_index < (long long)keys.size(); ++wide_index) {
        const size_t index = size_t(wide_index);
        Distribution distribution = quotient_token_planes(
            build_distribution(keys[index],
                               int(six_by_six_cache::COLUMNS), false));
        const Descriptor descriptor = descriptors[index];
        if (distribution.entries.size() != descriptor.count) {
            record_failure(
                CACHE_SUPPORT_COUNT, uint32_t(index),
                (uint64_t(descriptor.count) << 32) |
                    uint32_t(distribution.entries.size()));
            continue;
        }
        uint32_t local_weights[CLASS_SLOTS]{};
        uint8_t local_orbits[CLASS_SLOTS]{};
        uint8_t local_count = 0;
        for (uint32_t item = 0; item < descriptor.count; ++item) {
            const Entry entry = distribution.entries[item];
            if (!entry.weight || entry.weight > UINT32_MAX) {
                record_failure(CACHE_WEIGHT_WIDTH, uint32_t(index));
                break;
            }
            if (entry.mask > UINT32_MAX) {
                record_failure(CACHE_MASK_WIDTH, uint32_t(index));
                break;
            }
            const uint8_t orbit = uint8_t(token_plane_orbit_size(entry.mask));
            uint8_t ordinal = 0;
            while (ordinal < local_count &&
                   (local_weights[ordinal] != uint32_t(entry.weight) ||
                    local_orbits[ordinal] != orbit)) {
                ++ordinal;
            }
            if (ordinal == local_count) {
                if (local_count == CLASS_SLOTS) {
                    record_failure(CACHE_CLASS_WIDTH, uint32_t(index));
                    break;
                }
                local_weights[local_count] = uint32_t(entry.weight);
                local_orbits[local_count] = orbit;
                ++local_count;
            }
            masks[descriptor.offset + item] = uint32_t(entry.mask);
            ordinals[descriptor.offset + item] = ordinal;
        }
        const size_t class_base = index * CLASS_SLOTS;
        std::memcpy(class_weights + class_base, local_weights,
                    sizeof(local_weights));
        std::memcpy(class_orbits + class_base, local_orbits,
                    sizeof(local_orbits));
        class_counts[index] = local_count;
    }
    if (failure.load(std::memory_order_relaxed)) {
        throw std::runtime_error(
            "canonical cache packing failed: reason=" +
            std::to_string(failure.load(std::memory_order_relaxed)) +
            " distribution=" +
            std::to_string(failure_index.load(std::memory_order_relaxed)) +
            " detail=" +
            std::to_string(failure_detail.load(std::memory_order_relaxed)));
    }
    const double distribution_seconds = seconds_now() - distribution_start;

    std::array<std::array<uint8_t, 64>, 720> row_images{};
    std::array<uint16_t, 720> inverse_ranks{};
    std::array<uint8_t, 6> permutation{0, 1, 2, 3, 4, 5};
    size_t permutation_index = 0;
    do {
        std::array<uint8_t, 6> inverse{};
        for (unsigned destination = 0; destination < 6; ++destination)
            inverse[permutation[destination]] = uint8_t(destination);
        inverse_ranks[permutation_index] = permutation_rank(inverse);
        for (unsigned value = 0; value < 64; ++value)
            for (unsigned destination = 0; destination < 6; ++destination)
                row_images[permutation_index][value] |=
                    uint8_t((value >> permutation[destination]) & 1U)
                    << destination;
        ++permutation_index;
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    if (permutation_index != 720)
        throw std::logic_error("bad reference permutation census");

    failure.store(CACHE_OK, std::memory_order_relaxed);
    const double reference_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 8)
    for (long long wide_index = 0;
         wide_index < (long long)keys.size(); ++wide_index) {
        const uint32_t index = uint32_t(wide_index);
        const auto columns = column_vectors(keys[index]);
        for (size_t perm = 0; perm < 720; ++perm) {
            std::array<uint8_t, 6> transformed{};
            for (unsigned column = 0; column < 6; ++column)
                transformed[column] = row_images[perm][columns[column]];
            const uint32_t rank = multiset_rank(transformed);
            const uint32_t packed = (index << 10) | inverse_ranks[perm];
            const uint32_t previous =
                __atomic_load_n(references + rank, __ATOMIC_RELAXED);
            if (previous != UINT32_MAX && (previous >> 10) != index) {
                failure.store(CACHE_SUPPORT_COUNT,
                              std::memory_order_relaxed);
            } else {
                atomic_minimum(references + rank, packed);
            }
        }
    }
    uint64_t missing = 0;
#pragma omp parallel for schedule(static) reduction(+:missing)
    for (long long index = 0; index < (long long)MULTISET_COUNT; ++index)
        missing += references[size_t(index)] == UINT32_MAX;
    if (failure.load(std::memory_order_relaxed) || missing)
        throw std::runtime_error("canonical reference table is incomplete");
    const double reference_seconds = seconds_now() - reference_start;

    std::memcpy(output.mapping, &header, sizeof(header));
    const double flush_start = seconds_now();
    finish_output(output);
    if (rename(temporary.c_str(), output_path))
        throw std::runtime_error("cannot publish canonical cache artifact");
    const double flush_seconds = seconds_now() - flush_start;
    unmap_table(support);
    std::printf(
        "SIX_BY_SIX_CACHE_ARTIFACT canonical=%llu entries=%llu "
        "multisets=%llu bytes=%llu distribution_seconds=%.6f "
        "reference_seconds=%.6f flush_seconds=%.6f output=%s exact=OK\n",
        (unsigned long long)CANONICAL_COUNT,
        (unsigned long long)ENTRY_COUNT,
        (unsigned long long)MULTISET_COUNT,
        (unsigned long long)header.file_bytes, distribution_seconds,
        reference_seconds, flush_seconds, output_path);
}

static void validate_cache_artifact(const char* path) {
    initialise_tables();
    MappedSixBySixCache cache(path);
    const auto& header = cache.header();
    uint64_t cursor = 0;
    for (uint64_t index = 0; index < header.canonical_count; ++index) {
        const auto descriptor = cache.descriptors[index];
        const uint8_t class_count = cache.class_counts[index];
        if (descriptor.offset != cursor ||
            class_count > header.class_slots ||
            (!descriptor.count) != (!class_count))
            throw std::runtime_error("invalid cache descriptor sequence");
        cursor += descriptor.count;
    }
    if (cursor != header.entry_count ||
        !std::is_sorted(cache.keys, cache.keys + header.canonical_count))
        throw std::runtime_error("invalid cache canonical sequence");

    // Verify both selected and complement references against independent
    // construction.  Random-looking fixed masks exercise row maps, column
    // multisets, token-plane fixed points, and ordinary size-two orbits.
    uint64_t state = UINT64_C(0x6a09e667f3bcc909);
    constexpr unsigned CHECKS = 16;
    for (unsigned check = 0; check < CHECKS; ++check) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        const PrefixKey raw = state & ((PrefixKey(1) << 36) - 1U);
        const auto references = cache.resolve(raw);
        for (unsigned complement = 0; complement < 2; ++complement) {
            const CanonicalRef reference = references[complement];
            const auto descriptor = cache.descriptors[reference.distribution];
            std::vector<Entry> expanded;
            expanded.reserve(descriptor.count);
            const size_t class_base =
                size_t(reference.distribution) * header.class_slots;
            for (uint32_t item = 0; item < descriptor.count; ++item) {
                const uint64_t offset = descriptor.offset + item;
                const uint8_t ordinal = cache.ordinals[offset];
                if (ordinal >= cache.class_counts[reference.distribution])
                    throw std::runtime_error("invalid cache weight ordinal");
                uint64_t mask = transform_pair_mask(
                    cache.masks[offset], reference.row_map);
                mask = std::min(mask, swap_token_planes(mask));
                expanded.push_back(Entry{
                    mask, cache.class_weights[class_base + ordinal]});
            }
            Distribution direct = quotient_token_planes(build_distribution(
                raw, 6, complement != 0));
            if (!entries_equal(std::move(expanded),
                               std::move(direct.entries)))
                throw std::runtime_error("cache reference validation failed");
        }
    }
    std::printf("SIX_BY_SIX_CACHE_CHECK bytes=%llu canonical=%llu "
                "entries=%llu reference_checks=%u exact=OK\n",
                (unsigned long long)header.file_bytes,
                (unsigned long long)header.canonical_count,
                (unsigned long long)header.entry_count, CHECKS * 2);
}

static uint32_t inverse_six_row_map(uint32_t row_map) {
    uint32_t inverse = 0;
    for (unsigned source = 0; source < 6; ++source) {
        const unsigned destination = (row_map >> (4 * source)) & 15U;
        inverse |= source << (4 * destination);
    }
    return inverse;
}

static uint32_t compose_six_row_maps(uint32_t first, uint32_t second) {
    uint32_t result = 0;
    for (unsigned source = 0; source < 6; ++source) {
        const unsigned middle = (first >> (4 * source)) & 15U;
        const unsigned destination = (second >> (4 * middle)) & 15U;
        result |= destination << (4 * source);
    }
    return result;
}

template <class T>
static uint64_t unique_count(std::vector<T>& values) {
    std::sort(values.begin(), values.end());
    return uint64_t(std::unique(values.begin(), values.end()) -
                    values.begin());
}

static void census_cache_references(const char* cache_path,
                                    const char* corpus_path) {
    MappedSixBySixCache cache(cache_path);
    CorpusMapping corpus = map_corpus(corpus_path);
    std::vector<PrefixKey> raw_lefts, raw_rights;
    std::vector<std::array<uint32_t, 4>> left_halves, right_halves;
    std::vector<std::array<uint32_t, 6>> complete_joins;
    raw_lefts.reserve(corpus.records);
    raw_rights.reserve(corpus.records);
    left_halves.reserve(corpus.records);
    right_halves.reserve(corpus.records);
    complete_joins.reserve(corpus.records);
    const double start = seconds_now();
    for (uint64_t index = 0; index < corpus.records; ++index) {
        OrbitRecord record{};
        std::memcpy(&record,
                    corpus.mapping + 20 + index * sizeof(OrbitRecord),
                    sizeof(record));
        const PrefixKey left = orbit_left_prefix(record);
        const PrefixKey right = orbit_right_prefix(record);
        const auto left_refs = cache.resolve(left);
        const auto right_refs = cache.resolve(right);
        raw_lefts.push_back(left);
        raw_rights.push_back(right);
        left_halves.push_back({
            left_refs[0].distribution, left_refs[0].row_map,
            left_refs[1].distribution, left_refs[1].row_map});
        right_halves.push_back({
            right_refs[0].distribution, right_refs[0].row_map,
            right_refs[1].distribution, right_refs[1].row_map});
        std::array<uint32_t, 6> signature{};
        for (unsigned complement = 0; complement < 2; ++complement) {
            signature[3 * complement] =
                left_refs[complement].distribution;
            signature[3 * complement + 1] =
                right_refs[complement].distribution;
            signature[3 * complement + 2] = compose_six_row_maps(
                right_refs[complement].row_map,
                inverse_six_row_map(left_refs[complement].row_map));
        }
        complete_joins.push_back(signature);
    }
    const uint64_t raw_left_count = unique_count(raw_lefts);
    const uint64_t raw_right_count = unique_count(raw_rights);
    const uint64_t left_half_count = unique_count(left_halves);
    const uint64_t right_half_count = unique_count(right_halves);
    const uint64_t complete_join_count = unique_count(complete_joins);
    const double elapsed = seconds_now() - start;
    std::printf(
        "SIX_BY_SIX_REFERENCE_CENSUS records=%llu raw_left=%llu "
        "left_half_sig=%llu left_reuse=%.6f raw_right=%llu "
        "right_half_sig=%llu right_reuse=%.6f complete_join_sig=%llu "
        "edge_reuse=%.6f seconds=%.6f\n",
        (unsigned long long)corpus.records,
        (unsigned long long)raw_left_count,
        (unsigned long long)left_half_count,
        left_half_count ? double(raw_left_count) / left_half_count : 0,
        (unsigned long long)raw_right_count,
        (unsigned long long)right_half_count,
        right_half_count ? double(raw_right_count) / right_half_count : 0,
        (unsigned long long)complete_join_count,
        complete_join_count ? double(corpus.records) / complete_join_count : 0,
        elapsed);
    unmap_corpus(corpus);
}

static std::array<uint64_t, TILE_MENU.size()> select_records(
        const uint64_t* table, const uint8_t* input, uint8_t* output,
        uint64_t records, unsigned menu_size) {
    for (unsigned menu = 0; menu < menu_size; ++menu)
        if (__builtin_popcount(TILE_MENU[menu]) != 6)
            throw std::logic_error("cut is not 6+6");
    unsigned thread_count = 1;
#ifdef _OPENMP
    thread_count = unsigned(omp_get_max_threads());
#endif
    std::vector<ThreadSelections> thread_selections(thread_count);
#pragma omp parallel for schedule(static)
    for (long long index = 0; index < (long long)records; ++index) {
        unsigned thread = 0;
#ifdef _OPENMP
        thread = unsigned(omp_get_thread_num());
#endif
        OrbitRecord record{};
        std::memcpy(&record, input + size_t(index) * sizeof(record),
                    sizeof(record));
        const uint64_t weight = record.meta >> WIDE_ORBIT_KEY_BITS;
        const U128 key =
            (U128(record.meta & WIDE_ORBIT_KEY_MASK) << 64) | record.low;
        const auto columns = full_column_vectors(key);
        uint64_t best_cost = UINT64_MAX;
        uint16_t best_cut = 0;
        bool best_swap = false;
        unsigned best_menu = 0;
        for (unsigned menu = 0; menu < menu_size; ++menu) {
            const uint16_t cut = TILE_MENU[menu];
            const uint64_t left_counts =
                table[half_multiset_rank(columns, cut)];
            const uint64_t right_counts =
                table[half_multiset_rank(columns, ALL_COLUMNS ^ cut)];
            const uint64_t left_selected = uint32_t(left_counts);
            const uint64_t right_selected = uint32_t(right_counts);
            const uint64_t left_complement = left_counts >> 32;
            const uint64_t right_complement = right_counts >> 32;
            const uint64_t cost = left_selected * right_selected +
                                  left_complement * right_complement;
            if (cost < best_cost) {
                best_cost = cost;
                best_menu = menu;
                best_cut = cut;
                best_swap = left_selected + left_complement >
                            right_selected + right_complement;
            }
        }
        PrefixKey best_left = packed_half(key, best_cut);
        PrefixKey best_right = packed_half(key, ALL_COLUMNS ^ best_cut);
        if (best_swap) std::swap(best_left, best_right);
        const U128 transformed = concatenate(best_left, best_right);
        OrbitRecord result = record;
        result.low = uint64_t(transformed);
        result.meta = (weight << WIDE_ORBIT_KEY_BITS) |
                      uint64_t(transformed >> 64);
        std::memcpy(output + size_t(index) * sizeof(result), &result,
                    sizeof(result));
        thread_selections[thread].counts[best_menu]++;
    }
    std::array<uint64_t, TILE_MENU.size()> result{};
    for (const ThreadSelections& item : thread_selections)
        for (unsigned menu = 0; menu < menu_size; ++menu)
            result[menu] += item.counts[menu];
    return result;
}

static void print_selection_result(
        uint64_t records, unsigned menu_size, double compute_seconds,
        double flush_seconds,
        const std::array<uint64_t, TILE_MENU.size()>& selection_counts,
        const char* output_path, bool in_place) {
    std::printf("SIX_BY_TWELVE_CUT_SELECT records=%llu menu=%u "
                "compute_seconds=%.6f flush_seconds=%.6f "
                "records_per_second=%.3f selections=",
                (unsigned long long)records, menu_size, compute_seconds,
                flush_seconds, records / compute_seconds);
    for (unsigned menu = 0; menu < menu_size; ++menu)
        std::printf("%s0x%03x:%llu", menu ? "," : "", TILE_MENU[menu],
                    (unsigned long long)selection_counts[menu]);
    std::printf(" output=%s in_place=%u exact=OK\n", output_path,
                unsigned(in_place));
}

static void select_corpus(const char* table_path, const char* input_path,
                          const char* output_path, unsigned menu_size) {
    if (!menu_size || menu_size > TILE_MENU.size())
        throw std::runtime_error("menu size must be in 1..16");
    ReadOnlyTable table = map_table(table_path);
    CorpusMapping input = map_corpus(input_path);
    const std::string temporary_path =
        std::string(output_path) + ".tmp." + std::to_string(getpid());
    MutableMapping output = create_output(temporary_path.c_str(), input.bytes);
    // Publish the valid corpus header only after every record is complete.
    // An interrupted non-destructive rewrite is therefore rejected rather
    // than mistaken for a complete solve corpus.
    std::memset(output.mapping, 0, 20);
    const double start = seconds_now();
    const auto selection_counts = select_records(
        table.entries, input.mapping + 20, output.mapping + 20,
        input.records, menu_size);
    std::memcpy(output.mapping, input.mapping, 20);
    const double compute_seconds = seconds_now() - start;
    const double flush_start = seconds_now();
    finish_output(output);
    if (rename(temporary_path.c_str(), output_path))
        throw std::runtime_error("cannot publish selected corpus");
    const double flush_seconds = seconds_now() - flush_start;
    const uint64_t records = input.records;
    unmap_corpus(input);
    unmap_table(table);
    print_selection_result(records, menu_size, compute_seconds,
                           flush_seconds, selection_counts, output_path,
                           false);
}

static void select_corpus_in_place(const char* table_path,
                                   const char* corpus_path,
                                   unsigned menu_size) {
    if (!menu_size || menu_size > TILE_MENU.size())
        throw std::runtime_error("menu size must be in 1..16");
    ReadOnlyTable table = map_table(table_path);
    uint64_t records = 0;
    MutableMapping corpus = map_mutable_corpus(corpus_path, records);
    const double start = seconds_now();
    const auto selection_counts = select_records(
        table.entries, corpus.mapping + 20, corpus.mapping + 20, records,
        menu_size);
    const double compute_seconds = seconds_now() - start;
    const double flush_start = seconds_now();
    finish_output(corpus);
    const double flush_seconds = seconds_now() - flush_start;
    unmap_table(table);
    print_selection_result(records, menu_size, compute_seconds, flush_seconds,
                           selection_counts, corpus_path, true);
}

static void usage(const char* program) {
    std::fprintf(stderr,
                 "Usage:\n"
                 "  %s build-table CANONICAL_6x6.orbits TABLE.bin\n"
                 "  %s build-cache CANONICAL_6x6.orbits TABLE.bin "
                 "CACHE.bin\n"
                 "  %s check-cache CACHE.bin\n"
                 "  %s census-refs CACHE.bin CORPUS.orbits\n"
                 "  %s select TABLE.bin INPUT.orbits OUTPUT.orbits "
                 "[MENU_SIZE=16]\n"
                 "  %s select-in-place TABLE.bin CORPUS.orbits "
                 "[MENU_SIZE=16]\n",
                 program, program, program, program, program, program);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        initialise_choose();
        if (argc == 4 && !std::strcmp(argv[1], "build-table")) {
            build_table(argv[2], argv[3]);
            return 0;
        }
        if (argc == 5 && !std::strcmp(argv[1], "build-cache")) {
            build_cache_artifact(argv[2], argv[3], argv[4]);
            return 0;
        }
        if (argc == 3 && !std::strcmp(argv[1], "check-cache")) {
            validate_cache_artifact(argv[2]);
            return 0;
        }
        if (argc == 4 && !std::strcmp(argv[1], "census-refs")) {
            census_cache_references(argv[2], argv[3]);
            return 0;
        }
        if ((argc == 5 || argc == 6) && !std::strcmp(argv[1], "select")) {
            select_corpus(argv[2], argv[3], argv[4],
                          argc == 6 ? std::strtoul(argv[5], nullptr, 10) : 16);
            return 0;
        }
        if ((argc == 4 || argc == 5) &&
            !std::strcmp(argv[1], "select-in-place")) {
            select_corpus_in_place(
                argv[2], argv[3],
                argc == 5 ? std::strtoul(argv[4], nullptr, 10) : 16);
            return 0;
        }
        usage(argv[0]);
        return 2;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
