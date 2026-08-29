#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using U128 = unsigned __int128;
using RowPattern = uint16_t;

constexpr unsigned ROWS = 6;
constexpr unsigned PARENT_COLUMNS = 11;
constexpr unsigned COLUMNS = 12;
constexpr unsigned PARENT_CELLS = ROWS * PARENT_COLUMNS;
constexpr unsigned CELLS = ROWS * COLUMNS;
constexpr unsigned MIDPOINT = CELLS / 2;
constexpr unsigned ASSIGNMENTS = 1U << ROWS;
constexpr unsigned OUTPUT_KEY_BITS = CELLS - 64;
constexpr uint64_t OUTPUT_KEY_MASK = (UINT64_C(1) << OUTPUT_KEY_BITS) - 1U;
constexpr uint16_t ALL_COLUMNS = (1U << COLUMNS) - 1U;
constexpr uint64_t EXPECTED_RECORDS = UINT64_C(20230535486);
constexpr uint64_t EXPECTED_MIDPOINT_RECORDS = UINT64_C(3233916267);
constexpr U128 EXPECTED_MIDPOINT_WEIGHT =
    (U128(UINT64_C(23)) << 64) + UINT64_C(18237426581517092036);
constexpr U128 EXPECTED_RETAINED_WEIGHT =
    (U128(UINT64_C(139)) << 64) + UINT64_C(18342085327613321826);
constexpr uint64_t GROUP_ORDER = UINT64_C(344881152000);  // 6! * 12!
constexpr uint64_t RAW_MULTISET_COUNT = UINT64_C(119877472);
constexpr size_t OUTPUT_BUFFER_RECORDS = 256;

constexpr std::array<uint16_t, 16> CUT_MENU = {
    0x03f, 0x0cf, 0x077, 0x11f, 0x07d, 0x60f, 0x05f, 0x18f,
    0x0e7, 0x51d, 0x61d, 0x273, 0x09f, 0x1c7, 0x07b, 0x25d};

constexpr std::array<uint64_t, 13> FACTORIAL = {
    1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800,
    39916800, 479001600};

struct ParentRecord {
    uint64_t low;
    uint64_t meta;
};

struct NarrowRecord {
    uint64_t key;
    uint64_t weight;
};

struct OutputRecord {
    uint64_t low;
    uint64_t meta;
};
static_assert(sizeof(OutputRecord) == 16);

template <typename Record>
static Record load_unaligned_record(const uint8_t* source, uint64_t index) {
    Record result;
    std::memcpy(&result, source + index * sizeof(Record), sizeof(Record));
    return result;
}

struct TableHeader {
    char magic[8];
    uint32_t version;
    uint32_t rows;
    uint64_t entries;
};
static_assert(sizeof(TableHeader) == 24);

struct CanonicalResult {
    U128 key = ~U128(0);
    uint64_t automorphisms = 0;
    bool canonical_extension = false;
};

struct CanonContext {
    std::array<RowPattern, ROWS> rows{};
    std::array<uint8_t, ROWS> row_degree{};
    std::array<uint8_t, ROWS> target_degree{};
    std::array<uint8_t, ROWS> order{};
    std::array<uint8_t, ROWS> used{};
    unsigned columns = 0;
    int distinguished_column = -1;
    U128 best = ~U128(0);
    uint64_t best_count = 0;
    bool canonical_extension = false;
};

struct U128Hash {
    size_t operator()(U128 value) const noexcept;
};

struct MappedFile {
    int descriptor = -1;
    size_t bytes = 0;
    const uint8_t* data = nullptr;
};

struct SupportTable {
    MappedFile file;
    const uint64_t* entries = nullptr;
};

struct RangeStats {
    uint64_t input_records = 0;
    uint64_t ordinary_parents = 0;
    uint64_t candidates = 0;
    uint64_t retained_candidates = 0;
    uint64_t canonical_paths = 0;
    uint64_t emitted_records = 0;
    uint64_t midpoint_records = 0;
    U128 retained_weight = 0;
    U128 midpoint_weight = 0;
};

static uint64_t choose_table[70][7];

static double seconds_now() {
    timespec value{};
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec * 1e-9;
}

static uint64_t mix64(uint64_t value) {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

size_t U128Hash::operator()(U128 value) const noexcept {
    return size_t(mix64(uint64_t(value) ^
                        mix64(uint64_t(value >> 64) +
                              UINT64_C(0x9e3779b97f4a7c15))));
}

static std::string u128_string(U128 value) {
    char digits[64];
    size_t count = 0;
    do {
        digits[count++] = char('0' + value % 10);
        value /= 10;
    } while (value);
    std::string result;
    result.reserve(count);
    while (count) result.push_back(digits[--count]);
    return result;
}

static U128 pack_rows(const std::array<RowPattern, ROWS>& rows,
                      unsigned columns) {
    U128 result = 0;
    for (RowPattern row : rows) result = (result << columns) | row;
    return result;
}

static std::array<RowPattern, ROWS> unpack_rows(U128 key, unsigned columns) {
    std::array<RowPattern, ROWS> result{};
    const U128 mask = (U128(1) << columns) - 1U;
    for (int row = ROWS - 1; row >= 0; --row) {
        result[size_t(row)] = RowPattern(key & mask);
        key >>= columns;
    }
    return result;
}

static void evaluate_row_order(CanonContext& context) {
    std::array<uint16_t, COLUMNS> descriptors{};
    uint16_t distinguished = 0;
    for (unsigned column = 0; column < context.columns; ++column) {
        uint8_t vector = 0;
        for (unsigned position = 0; position < ROWS; ++position) {
            vector = uint8_t((vector << 1) |
                ((context.rows[context.order[position]] >> column) & 1U));
        }
        descriptors[column] =
            uint16_t((__builtin_popcount(vector) << ROWS) | vector);
        if (int(column) == context.distinguished_column)
            distinguished = descriptors[column];
    }
    std::sort(descriptors.begin(), descriptors.begin() + context.columns);
    std::array<RowPattern, ROWS> canonical_rows{};
    for (unsigned position = 0; position < ROWS; ++position) {
        RowPattern pattern = 0;
        for (unsigned column = 0; column < context.columns; ++column) {
            const uint8_t vector = uint8_t(descriptors[column] & 63U);
            pattern |= RowPattern((vector >> (ROWS - 1U - position)) & 1U)
                       << column;
        }
        canonical_rows[position] = pattern;
    }
    const U128 key = pack_rows(canonical_rows, context.columns);
    const bool canonical_extension = context.distinguished_column >= 0 &&
        distinguished == descriptors[0];
    if (key < context.best) {
        context.best = key;
        context.best_count = 1;
        context.canonical_extension = canonical_extension;
    } else if (key == context.best) {
        context.best_count++;
        context.canonical_extension |= canonical_extension;
    }
}

static void canonical_rows_rec(CanonContext& context, unsigned depth) {
    if (depth == ROWS) {
        evaluate_row_order(context);
        return;
    }
    const uint8_t degree = context.target_degree[depth];
    std::array<uint64_t, 64> seen{};
    for (unsigned row = 0; row < ROWS; ++row) {
        if (context.used[row] || context.row_degree[row] != degree) continue;
        const RowPattern pattern = context.rows[row];
        const uint64_t bit = UINT64_C(1) << (pattern & 63U);
        if (seen[pattern >> 6] & bit) continue;
        seen[pattern >> 6] |= bit;
        context.used[row] = 1;
        context.order[depth] = uint8_t(row);
        canonical_rows_rec(context, depth + 1);
        context.used[row] = 0;
    }
}

static CanonicalResult canonicalise(
        const std::array<RowPattern, ROWS>& rows, unsigned columns,
        int distinguished_column = -1) {
    if (!columns || columns > COLUMNS)
        throw std::logic_error("invalid canonical width");
    CanonContext context;
    context.rows = rows;
    context.columns = columns;
    context.distinguished_column = distinguished_column;
    for (unsigned row = 0; row < ROWS; ++row) {
        context.row_degree[row] = uint8_t(__builtin_popcount(rows[row]));
        context.target_degree[row] = context.row_degree[row];
    }
    std::sort(context.target_degree.begin(), context.target_degree.end());
    canonical_rows_rec(context, 0);

    uint64_t row_factor = 1;
    std::array<bool, ROWS> row_seen{};
    for (unsigned row = 0; row < ROWS; ++row) {
        if (row_seen[row]) continue;
        unsigned multiplicity = 1;
        for (unsigned other = row + 1; other < ROWS; ++other) {
            if (rows[other] == rows[row]) {
                row_seen[other] = true;
                multiplicity++;
            }
        }
        row_factor *= FACTORIAL[multiplicity];
    }
    std::array<uint8_t, COLUMNS> vectors{};
    for (unsigned column = 0; column < columns; ++column)
        for (unsigned row = 0; row < ROWS; ++row)
            vectors[column] = uint8_t((vectors[column] << 1) |
                                      ((rows[row] >> column) & 1U));
    uint64_t column_factor = 1;
    std::array<bool, COLUMNS> column_seen{};
    for (unsigned column = 0; column < columns; ++column) {
        if (column_seen[column]) continue;
        unsigned multiplicity = 1;
        for (unsigned other = column + 1; other < columns; ++other) {
            if (vectors[other] == vectors[column]) {
                column_seen[other] = true;
                multiplicity++;
            }
        }
        column_factor *= FACTORIAL[multiplicity];
    }
    return CanonicalResult{context.best,
                           context.best_count * row_factor * column_factor,
                           context.canonical_extension};
}

static void initialise_choose() {
    choose_table[0][0] = 1;
    for (unsigned n = 1; n < 70; ++n) {
        choose_table[n][0] = 1;
        for (unsigned k = 1; k <= 6; ++k)
            choose_table[n][k] =
                choose_table[n - 1][k - 1] + choose_table[n - 1][k];
    }
    if (choose_table[69][6] != RAW_MULTISET_COUNT)
        throw std::logic_error("bad multiset census");
}

static uint32_t multiset_rank(std::array<uint8_t, 6> values) {
    std::sort(values.begin(), values.end());
    uint64_t rank = 0;
    for (unsigned index = 0; index < values.size(); ++index)
        rank += choose_table[unsigned(values[index]) + index][index + 1];
    if (rank >= RAW_MULTISET_COUNT)
        throw std::logic_error("multiset rank overflow");
    return uint32_t(rank);
}

static std::array<uint8_t, COLUMNS> column_vectors(U128 key) {
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

static uint64_t packed_half(U128 key, uint16_t selected) {
    uint64_t result = 0;
    for (unsigned row = 0; row < ROWS; ++row) {
        const unsigned shift = COLUMNS * (ROWS - 1U - row);
        const uint16_t pattern = uint16_t(key >> shift) & ALL_COLUMNS;
        uint8_t half = 0;
        unsigned output = 0;
        for (unsigned column = 0; column < COLUMNS; ++column) {
            if (!(selected & (1U << column))) continue;
            half |= uint8_t((pattern >> column) & 1U) << output++;
        }
        result = (result << 6) | half;
    }
    return result;
}

static U128 concatenate(uint64_t resident, uint64_t streamed) {
    U128 result = 0;
    for (unsigned row = 0; row < ROWS; ++row) {
        const unsigned shift = 6 * (ROWS - 1U - row);
        const uint64_t left = (resident >> shift) & 63U;
        const uint64_t right = (streamed >> shift) & 63U;
        result = (result << 12) | (U128(right) << 6) | left;
    }
    return result;
}

static MappedFile map_read_only(const char* path) {
    MappedFile result;
    result.descriptor = open(path, O_RDONLY);
    struct stat status{};
    if (result.descriptor < 0 || fstat(result.descriptor, &status) ||
        status.st_size <= 0)
        throw std::runtime_error(std::string("cannot map ") + path);
    result.bytes = size_t(status.st_size);
    result.data = static_cast<const uint8_t*>(mmap(
        nullptr, result.bytes, PROT_READ, MAP_SHARED, result.descriptor, 0));
    if (result.data == MAP_FAILED) throw std::runtime_error("mmap failed");
    return result;
}

static void unmap_file(MappedFile& file) {
    if (file.data && munmap(const_cast<uint8_t*>(file.data), file.bytes))
        throw std::runtime_error("munmap failed");
    if (file.descriptor >= 0 && close(file.descriptor))
        throw std::runtime_error("close failed");
    file = MappedFile{};
}

static SupportTable map_support_table(const char* path) {
    SupportTable result;
    result.file = map_read_only(path);
    const size_t expected = sizeof(TableHeader) +
                            RAW_MULTISET_COUNT * sizeof(uint64_t);
    if (result.file.bytes != expected)
        throw std::runtime_error("bad support-table size");
    const auto* header =
        reinterpret_cast<const TableHeader*>(result.file.data);
    if (std::memcmp(header->magic, "R6SUPT1", 8) || header->version != 1 ||
        header->rows != ROWS || header->entries != RAW_MULTISET_COUNT)
        throw std::runtime_error("bad support-table header");
    result.entries = reinterpret_cast<const uint64_t*>(
        result.file.data + sizeof(TableHeader));
    return result;
}

static U128 select_cut(const uint64_t* table, U128 key) {
    const auto columns = column_vectors(key);
    uint64_t best_cost = UINT64_MAX;
    uint16_t best_cut = 0;
    bool best_swap = false;
    for (uint16_t cut : CUT_MENU) {
        const uint64_t left = table[half_multiset_rank(columns, cut)];
        const uint64_t right =
            table[half_multiset_rank(columns, ALL_COLUMNS ^ cut)];
        const uint64_t left_selected = uint32_t(left);
        const uint64_t right_selected = uint32_t(right);
        const uint64_t left_complement = left >> 32;
        const uint64_t right_complement = right >> 32;
        const uint64_t cost = left_selected * right_selected +
                              left_complement * right_complement;
        if (cost < best_cost) {
            best_cost = cost;
            best_cut = cut;
            best_swap = left_selected + left_complement >
                        right_selected + right_complement;
        }
    }
    uint64_t left = packed_half(key, best_cut);
    uint64_t right = packed_half(key, ALL_COLUMNS ^ best_cut);
    if (best_swap) std::swap(left, right);
    return concatenate(left, right);
}

static uint64_t left_prefix(U128 key) {
    uint64_t result = 0;
    for (unsigned row = 0; row < ROWS; ++row) {
        const unsigned shift = COLUMNS * (ROWS - 1U - row);
        result = (result << 6) | (uint64_t(key >> shift) & 63U);
    }
    return result;
}

static uint64_t right_prefix(U128 key) {
    uint64_t result = 0;
    for (unsigned row = 0; row < ROWS; ++row) {
        const unsigned shift = COLUMNS * (ROWS - 1U - row);
        result = (result << 6) | (uint64_t(key >> (shift + 6)) & 63U);
    }
    return result;
}

// Solve owners use a right-major key so ordinary numeric ordering is exactly
// the solver's recurring (right,left) traversal.  Generator fragments retain
// the interleaved grid representation needed by the canonical-parent path.
static U128 right_major_key(U128 interleaved) {
    return (U128(right_prefix(interleaved)) << 36) |
           left_prefix(interleaved);
}

static U128 interleaved_key(U128 right_major) {
    const uint64_t half_mask = (UINT64_C(1) << 36) - 1U;
    return concatenate(uint64_t(right_major) & half_mask,
                       uint64_t(right_major >> 36));
}

static OutputRecord output_record(U128 key, uint64_t weight) {
    if (weight >= (UINT64_C(1) << (64 - OUTPUT_KEY_BITS)))
        throw std::overflow_error("6x12 orbit weight overflow");
    return OutputRecord{uint64_t(key),
                        (weight << OUTPUT_KEY_BITS) |
                        (uint64_t(key >> 64) & OUTPUT_KEY_MASK)};
}

static void raise_file_limit(unsigned required) {
    rlimit limit{};
    if (getrlimit(RLIMIT_NOFILE, &limit))
        throw std::runtime_error("getrlimit failed");
    if (limit.rlim_cur >= required) return;
    if (limit.rlim_max < required)
        throw std::runtime_error("file-descriptor limit is too low");
    limit.rlim_cur = required;
    if (setrlimit(RLIMIT_NOFILE, &limit))
        throw std::runtime_error("setrlimit failed");
}

static void write_header(FILE* file, uint64_t records,
                         const char (&magic)[8]) {
    const uint32_t columns = COLUMNS;
    if (std::fwrite(magic, sizeof(magic), 1, file) != 1 ||
        std::fwrite(&columns, sizeof(columns), 1, file) != 1 ||
        std::fwrite(&records, sizeof(records), 1, file) != 1)
        throw std::runtime_error("header write failed");
}

class ShardWriter {
public:
    ShardWriter(unsigned shards, std::string prefix)
        : shards_(shards), prefix_(std::move(prefix)), files_(shards),
          counts_(shards), locks_(shards), final_paths_(shards),
          temporary_paths_(shards) {
        if (!shards_ || shards_ > 1024)
            throw std::runtime_error("shards must be in 1..1024");
        raise_file_limit(shards_ + 64);
        for (unsigned shard = 0; shard < shards_; ++shard) {
            char suffix[32];
            std::snprintf(suffix, sizeof(suffix), ".s%04u.orbits", shard);
            final_paths_[shard] = prefix_ + suffix;
            temporary_paths_[shard] = final_paths_[shard] + ".tmp";
            files_[shard] = std::fopen(temporary_paths_[shard].c_str(), "wb+");
            if (!files_[shard])
                throw std::runtime_error("cannot create output shard");
            std::setvbuf(files_[shard], nullptr, _IOFBF, 1U << 20);
            write_header(files_[shard], 0, "R6W1201");
#ifdef _OPENMP
            omp_init_lock(&locks_[shard]);
#endif
        }
    }

    ShardWriter(const ShardWriter&) = delete;
    ShardWriter& operator=(const ShardWriter&) = delete;

    void append(unsigned shard, const OutputRecord* records, size_t count) {
        if (!count) return;
#ifdef _OPENMP
        omp_set_lock(&locks_[shard]);
#endif
        const size_t written =
            std::fwrite(records, sizeof(OutputRecord), count, files_[shard]);
        counts_[shard] += written;
#ifdef _OPENMP
        omp_unset_lock(&locks_[shard]);
#endif
        if (written != count) failed_.store(true, std::memory_order_relaxed);
    }

    void finish() {
        if (finished_) return;
        if (failed_.load(std::memory_order_relaxed))
            throw std::runtime_error("output shard write failed");
        for (unsigned shard = 0; shard < shards_; ++shard) {
            if (fseeko(files_[shard], 0, SEEK_SET))
                throw std::runtime_error("output seek failed");
            write_header(files_[shard], counts_[shard], "R6W1201");
            if (std::fclose(files_[shard]))
                throw std::runtime_error("output close failed");
            files_[shard] = nullptr;
            if (rename(temporary_paths_[shard].c_str(),
                       final_paths_[shard].c_str()))
                throw std::runtime_error("output publish failed");
#ifdef _OPENMP
            omp_destroy_lock(&locks_[shard]);
#endif
        }
        finished_ = true;
    }

    const std::vector<uint64_t>& counts() const { return counts_; }

private:
    unsigned shards_;
    std::string prefix_;
    std::vector<FILE*> files_;
    std::vector<uint64_t> counts_;
#ifdef _OPENMP
    std::vector<omp_lock_t> locks_;
#else
    std::vector<unsigned char> locks_;
#endif
    std::vector<std::string> final_paths_;
    std::vector<std::string> temporary_paths_;
    std::atomic<bool> failed_{false};
    bool finished_ = false;
};

struct ThreadOutput {
    explicit ThreadOutput(unsigned shards)
        : buffers(shards), stats{} {}
    std::vector<std::vector<OutputRecord>> buffers;
    RangeStats stats;
};

static void append_buffered(ThreadOutput& thread, ShardWriter& writer,
                            unsigned owner, OutputRecord record) {
    auto& buffer = thread.buffers[owner];
    if (buffer.empty()) buffer.reserve(OUTPUT_BUFFER_RECORDS);
    buffer.push_back(record);
    if (buffer.size() == OUTPUT_BUFFER_RECORDS) {
        writer.append(owner, buffer.data(), buffer.size());
        buffer.clear();
    }
}

static void flush_buffers(ThreadOutput& thread, ShardWriter& writer) {
    for (unsigned shard = 0; shard < thread.buffers.size(); ++shard) {
        auto& buffer = thread.buffers[shard];
        writer.append(shard, buffer.data(), buffer.size());
        buffer.clear();
    }
}

static RangeStats add_stats(const std::vector<ThreadOutput>& threads) {
    RangeStats result;
    for (const ThreadOutput& thread : threads) {
        result.input_records += thread.stats.input_records;
        result.ordinary_parents += thread.stats.ordinary_parents;
        result.candidates += thread.stats.candidates;
        result.retained_candidates += thread.stats.retained_candidates;
        result.canonical_paths += thread.stats.canonical_paths;
        result.emitted_records += thread.stats.emitted_records;
        result.midpoint_records += thread.stats.midpoint_records;
        result.retained_weight += thread.stats.retained_weight;
        result.midpoint_weight += thread.stats.midpoint_weight;
    }
    return result;
}

struct ParentCorpus {
    MappedFile file;
    uint64_t records = 0;
    const uint8_t* entries = nullptr;
};

static ParentCorpus map_parent_corpus(const char* path) {
    ParentCorpus result;
    result.file = map_read_only(path);
    if (result.file.bytes < 20)
        throw std::runtime_error("parent corpus is too short");
    char magic[8];
    uint32_t columns = 0;
    std::memcpy(magic, result.file.data, 8);
    std::memcpy(&columns, result.file.data + 8, 4);
    std::memcpy(&result.records, result.file.data + 12, 8);
    if (std::memcmp(magic, "R6W1101", 8) ||
        columns != PARENT_COLUMNS ||
        result.file.bytes != 20 + result.records * sizeof(ParentRecord))
        throw std::runtime_error("invalid retained 6x11 parent corpus");
    result.entries = result.file.data + 20;
    return result;
}

static void unmap_parent_corpus(ParentCorpus& corpus) {
    unmap_file(corpus.file);
    corpus = ParentCorpus{};
}

static void process_parent_orientation(
        const std::array<RowPattern, ROWS>& parent_rows,
        const uint64_t* support_table, unsigned shards, ThreadOutput& thread,
        ShardWriter& writer, std::atomic<uint64_t>& verification_budget) {
    thread.stats.ordinary_parents++;
    std::array<U128, ASSIGNMENTS> emitted_keys{};
    std::array<uint64_t, ASSIGNMENTS> emitted_weights{};
    unsigned emitted = 0;
    const unsigned parent_cells = unsigned(__builtin_popcountll(
        uint64_t(pack_rows(parent_rows, PARENT_COLUMNS)))) +
        unsigned(__builtin_popcountll(
            uint64_t(pack_rows(parent_rows, PARENT_COLUMNS) >> 64)));
    for (unsigned assignment = 0; assignment < ASSIGNMENTS; ++assignment) {
        thread.stats.candidates++;
        const unsigned cells = parent_cells + unsigned(__builtin_popcount(assignment));
        if (cells > MIDPOINT) continue;
        thread.stats.retained_candidates++;
        std::array<RowPattern, ROWS> child_rows{};
        for (unsigned row = 0; row < ROWS; ++row) {
            child_rows[row] = parent_rows[row] |
                RowPattern(((assignment >> row) & 1U) << PARENT_COLUMNS);
        }
        const CanonicalResult canonical =
            canonicalise(child_rows, COLUMNS, PARENT_COLUMNS);
        if (!canonical.canonical_extension) continue;
        thread.stats.canonical_paths++;
        if (!canonical.automorphisms ||
            GROUP_ORDER % canonical.automorphisms)
            std::terminate();
        const uint64_t weight = GROUP_ORDER / canonical.automorphisms;

        bool duplicate = false;
        for (unsigned previous = 0; previous < emitted; ++previous) {
            if (emitted_keys[previous] != canonical.key) continue;
            if (emitted_weights[previous] != weight) std::terminate();
            duplicate = true;
            break;
        }
        if (duplicate) continue;
        emitted_keys[emitted] = canonical.key;
        emitted_weights[emitted] = weight;
        emitted++;

        uint64_t remaining = verification_budget.load(std::memory_order_relaxed);
        while (remaining && !verification_budget.compare_exchange_weak(
                   remaining, remaining - 1, std::memory_order_relaxed)) {}
        if (remaining) {
            auto canonical_rows = unpack_rows(canonical.key, COLUMNS);
            for (RowPattern& row : canonical_rows) row >>= 1;
            const U128 deleted_parent =
                canonicalise(canonical_rows, PARENT_COLUMNS).key;
            const U128 source_parent =
                canonicalise(parent_rows, PARENT_COLUMNS).key;
            if (deleted_parent != source_parent) std::terminate();
        }

        const U128 transformed = select_cut(support_table, canonical.key);
        const unsigned owner = unsigned(mix64(left_prefix(transformed)) % shards);
        append_buffered(thread, writer, owner,
                        output_record(transformed, weight));
        thread.stats.emitted_records++;
        thread.stats.retained_weight += weight;
        if (cells == MIDPOINT) {
            thread.stats.midpoint_records++;
            thread.stats.midpoint_weight += weight;
        }
    }
}

static void run_generate_range(const char* table_path, const char* input_path,
                               uint64_t start, uint64_t end, unsigned shards,
                               const char* output_prefix) {
    initialise_choose();
    SupportTable support = map_support_table(table_path);
    ParentCorpus parents = map_parent_corpus(input_path);
    if (!end) end = parents.records;
    if (start > end || end > parents.records)
        throw std::runtime_error("invalid parent range");
    ShardWriter writer(shards, output_prefix);
    unsigned thread_count = 1;
#ifdef _OPENMP
    thread_count = unsigned(omp_get_max_threads());
#endif
    std::vector<ThreadOutput> threads;
    threads.reserve(thread_count);
    for (unsigned thread = 0; thread < thread_count; ++thread)
        threads.emplace_back(shards);
    std::atomic<uint64_t> verification_budget{1024};
    const double begin = seconds_now();
#pragma omp parallel for schedule(dynamic, 256)
    for (long long input_index = (long long)start;
         input_index < (long long)end; ++input_index) {
        unsigned thread_index = 0;
#ifdef _OPENMP
        thread_index = unsigned(omp_get_thread_num());
#endif
        ThreadOutput& thread = threads[thread_index];
        const ParentRecord record = load_unaligned_record<ParentRecord>(
            parents.entries, uint64_t(input_index));
        const U128 key = (U128(record.meta & 3U) << 64) | record.low;
        const uint64_t stored_weight = record.meta >> 2;
        const unsigned cells = unsigned(__builtin_popcountll(record.low)) +
                               unsigned(__builtin_popcountll(record.meta & 3U));
        if (!stored_weight || cells > PARENT_CELLS / 2) std::terminate();
        thread.stats.input_records++;
        const auto rows = unpack_rows(key, PARENT_COLUMNS);
        process_parent_orientation(rows, support.entries, shards, thread,
                                   writer, verification_budget);
        if (cells < PARENT_CELLS / 2) {
            auto complement = rows;
            const RowPattern mask = (1U << PARENT_COLUMNS) - 1U;
            for (RowPattern& row : complement) row ^= mask;
            process_parent_orientation(complement, support.entries, shards,
                                       thread, writer, verification_budget);
        }
    }
    for (ThreadOutput& thread : threads) flush_buffers(thread, writer);
    const RangeStats stats = add_stats(threads);
    uint64_t written = 0;
    for (uint64_t count : writer.counts()) written += count;
    if (written != stats.emitted_records)
        throw std::runtime_error("output count mismatch");
    writer.finish();
    const double elapsed = seconds_now() - begin;
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    std::printf("R6X12_CANONICAL_RANGE input=%s range=[%llu,%llu) "
                "input_records=%llu ordinary_parents=%llu candidates=%llu "
                "retained_candidates=%llu canonical_paths=%llu "
                "emitted_records=%llu midpoint_records=%llu "
                "retained_weight=%s midpoint_weight=%s shards=%u "
                "seconds=%.6f candidates_per_second=%.3f peak_rss_mib=%.3f "
                "prefix=%s exact=OK\n",
                input_path, (unsigned long long)start,
                (unsigned long long)end,
                (unsigned long long)stats.input_records,
                (unsigned long long)stats.ordinary_parents,
                (unsigned long long)stats.candidates,
                (unsigned long long)stats.retained_candidates,
                (unsigned long long)stats.canonical_paths,
                (unsigned long long)stats.emitted_records,
                (unsigned long long)stats.midpoint_records,
                u128_string(stats.retained_weight).c_str(),
                u128_string(stats.midpoint_weight).c_str(), shards, elapsed,
                stats.candidates / elapsed, usage.ru_maxrss / 1024.0,
                output_prefix);
    unmap_parent_corpus(parents);
    unmap_file(support.file);
}

static void run_make_parent_sample(const char* input_path, uint64_t start,
                                   uint64_t end, const char* output_path) {
    MappedFile input = map_read_only(input_path);
    if (input.bytes < 20) throw std::runtime_error("short 6x10 corpus");
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    std::memcpy(magic, input.data, 8);
    std::memcpy(&columns, input.data + 8, 4);
    std::memcpy(&count, input.data + 12, 8);
    if (std::memcmp(magic, "R6ORB01", 8) || columns != 10 ||
        input.bytes != 20 + count * sizeof(NarrowRecord))
        throw std::runtime_error("invalid retained 6x10 corpus");
    if (!end) end = count;
    if (start > end || end > count)
        throw std::runtime_error("invalid 6x10 range");
    const uint8_t* source = input.data + 20;
    unsigned thread_count = 1;
#ifdef _OPENMP
    thread_count = unsigned(omp_get_max_threads());
#endif
    std::vector<std::vector<ParentRecord>> outputs(thread_count);
    std::vector<RangeStats> stats(thread_count);
    const uint64_t group_order = FACTORIAL[ROWS] * FACTORIAL[PARENT_COLUMNS];
    const double begin = seconds_now();
#pragma omp parallel for schedule(dynamic, 256)
    for (long long index = (long long)start; index < (long long)end; ++index) {
        unsigned thread = 0;
#ifdef _OPENMP
        thread = unsigned(omp_get_thread_num());
#endif
        const NarrowRecord record = load_unaligned_record<NarrowRecord>(
            source, uint64_t(index));
        const unsigned cells = unsigned(__builtin_popcountll(record.key));
        if (!record.weight || cells > 30) std::terminate();
        stats[thread].input_records++;
        auto original = unpack_rows(record.key, 10);
        const unsigned orientations = cells < 30 ? 2 : 1;
        for (unsigned orientation = 0; orientation < orientations;
             ++orientation) {
            auto rows = original;
            if (orientation)
                for (RowPattern& row : rows) row ^= (1U << 10) - 1U;
            stats[thread].ordinary_parents++;
            const unsigned parent_cells = orientation ? 60 - cells : cells;
            std::array<U128, ASSIGNMENTS> emitted{};
            unsigned emitted_count = 0;
            for (unsigned assignment = 0; assignment < ASSIGNMENTS;
                 ++assignment) {
                stats[thread].candidates++;
                const unsigned child_cells =
                    parent_cells + unsigned(__builtin_popcount(assignment));
                if (child_cells > 33) continue;
                stats[thread].retained_candidates++;
                std::array<RowPattern, ROWS> child{};
                for (unsigned row = 0; row < ROWS; ++row)
                    child[row] = rows[row] |
                        RowPattern(((assignment >> row) & 1U) << 10);
                const CanonicalResult result = canonicalise(child, 11, 10);
                if (!result.canonical_extension) continue;
                stats[thread].canonical_paths++;
                if (!result.automorphisms ||
                    group_order % result.automorphisms)
                    std::terminate();
                bool duplicate = false;
                for (unsigned prior = 0; prior < emitted_count; ++prior)
                    duplicate |= emitted[prior] == result.key;
                if (duplicate) continue;
                emitted[emitted_count++] = result.key;
                const uint64_t weight = group_order / result.automorphisms;
                outputs[thread].push_back(ParentRecord{
                    uint64_t(result.key),
                    (weight << 2) | uint64_t(result.key >> 64)});
                stats[thread].emitted_records++;
                stats[thread].retained_weight += weight;
                if (child_cells == 33) {
                    stats[thread].midpoint_records++;
                    stats[thread].midpoint_weight += weight;
                }
            }
        }
    }
    std::vector<ParentRecord> records;
    size_t output_count = 0;
    for (const auto& output : outputs) output_count += output.size();
    records.reserve(output_count);
    RangeStats total;
    for (unsigned thread = 0; thread < thread_count; ++thread) {
        records.insert(records.end(), outputs[thread].begin(),
                       outputs[thread].end());
        total.input_records += stats[thread].input_records;
        total.ordinary_parents += stats[thread].ordinary_parents;
        total.candidates += stats[thread].candidates;
        total.retained_candidates += stats[thread].retained_candidates;
        total.canonical_paths += stats[thread].canonical_paths;
        total.emitted_records += stats[thread].emitted_records;
        total.midpoint_records += stats[thread].midpoint_records;
        total.retained_weight += stats[thread].retained_weight;
        total.midpoint_weight += stats[thread].midpoint_weight;
    }
    std::sort(records.begin(), records.end(), [](const ParentRecord& a,
                                                  const ParentRecord& b) {
        const U128 ka = (U128(a.meta & 3U) << 64) | a.low;
        const U128 kb = (U128(b.meta & 3U) << 64) | b.low;
        return ka < kb;
    });
    for (size_t index = 1; index < records.size(); ++index) {
        const U128 previous =
            (U128(records[index - 1].meta & 3U) << 64) |
            records[index - 1].low;
        const U128 current = (U128(records[index].meta & 3U) << 64) |
                             records[index].low;
        if (previous == current)
            throw std::runtime_error("duplicate 6x11 canonical child");
    }
    const std::string temporary = std::string(output_path) + ".tmp";
    FILE* output = std::fopen(temporary.c_str(), "wb");
    if (!output) throw std::runtime_error("cannot create parent sample");
    const char output_magic[8] = "R6W1101";
    const uint32_t output_columns = 11;
    const uint64_t output_records = records.size();
    if (std::fwrite(output_magic, 8, 1, output) != 1 ||
        std::fwrite(&output_columns, 4, 1, output) != 1 ||
        std::fwrite(&output_records, 8, 1, output) != 1 ||
        std::fwrite(records.data(), sizeof(ParentRecord), records.size(),
                    output) != records.size() ||
        std::fclose(output) || rename(temporary.c_str(), output_path))
        throw std::runtime_error("cannot publish parent sample");
    std::printf("R6X11_CANONICAL_PARENT_SAMPLE range=[%llu,%llu) "
                "input_records=%llu ordinary_parents=%llu candidates=%llu "
                "retained_candidates=%llu canonical_paths=%llu records=%zu "
                "midpoint_records=%llu retained_weight=%s "
                "midpoint_weight=%s seconds=%.6f output=%s exact=OK\n",
                (unsigned long long)start, (unsigned long long)end,
                (unsigned long long)total.input_records,
                (unsigned long long)total.ordinary_parents,
                (unsigned long long)total.candidates,
                (unsigned long long)total.retained_candidates,
                (unsigned long long)total.canonical_paths, records.size(),
                (unsigned long long)total.midpoint_records,
                u128_string(total.retained_weight).c_str(),
                u128_string(total.midpoint_weight).c_str(),
                seconds_now() - begin, output_path);
    unmap_file(input);
}

using OrbitMap = std::unordered_map<U128, uint64_t, U128Hash>;

static U128 reference_canonical_key(
        const std::array<RowPattern, ROWS>& rows, unsigned columns) {
    std::array<unsigned, ROWS> order{0, 1, 2, 3, 4, 5};
    U128 best = ~U128(0);
    do {
        bool degree_ordered = true;
        for (unsigned position = 1; position < ROWS; ++position) {
            if (__builtin_popcount(rows[order[position - 1]]) >
                __builtin_popcount(rows[order[position]])) {
                degree_ordered = false;
                break;
            }
        }
        if (!degree_ordered) continue;
        std::array<uint16_t, COLUMNS> descriptors{};
        for (unsigned column = 0; column < columns; ++column) {
            uint8_t vector = 0;
            for (unsigned position = 0; position < ROWS; ++position)
                vector = uint8_t((vector << 1) |
                    ((rows[order[position]] >> column) & 1U));
            descriptors[column] =
                uint16_t((__builtin_popcount(vector) << ROWS) | vector);
        }
        std::sort(descriptors.begin(), descriptors.begin() + columns);
        std::array<RowPattern, ROWS> canonical_rows{};
        for (unsigned position = 0; position < ROWS; ++position) {
            for (unsigned column = 0; column < columns; ++column) {
                const uint8_t vector = uint8_t(descriptors[column] & 63U);
                canonical_rows[position] |=
                    RowPattern((vector >> (ROWS - 1U - position)) & 1U)
                    << column;
            }
        }
        best = std::min(best, pack_rows(canonical_rows, columns));
    } while (std::next_permutation(order.begin(), order.end()));
    return best;
}

static void validate_random_canonicalisation() {
    uint64_t state = UINT64_C(0x6429d5f3c17e8a0b);
    for (unsigned sample = 0; sample < 256; ++sample) {
        const unsigned columns = 1 + sample % COLUMNS;
        std::array<RowPattern, ROWS> rows{};
        const RowPattern mask = (1U << columns) - 1U;
        for (RowPattern& row : rows) {
            state = mix64(state + UINT64_C(0x9e3779b97f4a7c15));
            row = RowPattern(state) & mask;
        }
        const CanonicalResult result = canonicalise(rows, columns);
        if (result.key != reference_canonical_key(rows, columns) ||
            !result.automorphisms ||
            (FACTORIAL[ROWS] * FACTORIAL[columns]) % result.automorphisms)
            throw std::runtime_error("canonicalizer reference mismatch");

        std::array<uint8_t, ROWS> row_order{0, 1, 2, 3, 4, 5};
        std::array<uint8_t, COLUMNS> column_order{};
        for (unsigned column = 0; column < columns; ++column)
            column_order[column] = uint8_t(column);
        for (int row = ROWS - 1; row > 0; --row) {
            state = mix64(state + UINT64_C(0x9e3779b97f4a7c15));
            std::swap(row_order[size_t(row)],
                      row_order[size_t(state % unsigned(row + 1))]);
        }
        for (int column = int(columns) - 1; column > 0; --column) {
            state = mix64(state + UINT64_C(0x9e3779b97f4a7c15));
            std::swap(column_order[size_t(column)],
                      column_order[size_t(state % unsigned(column + 1))]);
        }
        std::array<RowPattern, ROWS> permuted{};
        for (unsigned destination_row = 0; destination_row < ROWS;
             ++destination_row) {
            const RowPattern source = rows[row_order[destination_row]];
            for (unsigned destination_column = 0;
                 destination_column < columns; ++destination_column) {
                if (source & (1U << column_order[destination_column]))
                    permuted[destination_row] |=
                        RowPattern(1U << destination_column);
            }
        }
        if (canonicalise(permuted, columns).key != result.key)
            throw std::runtime_error("canonicalizer permutation mismatch");
    }
}

static void run_self_test(unsigned maximum_columns) {
    if (!maximum_columns || maximum_columns > 6)
        throw std::runtime_error("self-test width must be in 1..6");
    U128 midpoint_weight = 1;
    for (unsigned factor = 1; factor <= MIDPOINT; ++factor)
        midpoint_weight = midpoint_weight * (MIDPOINT + factor) / factor;
    const U128 retained_weight = ((U128(1) << CELLS) + midpoint_weight) / 2;
    if (midpoint_weight != EXPECTED_MIDPOINT_WEIGHT ||
        retained_weight != EXPECTED_RETAINED_WEIGHT ||
        retained_weight * 2 - midpoint_weight != (U128(1) << CELLS))
        throw std::runtime_error("6x12 corpus invariant mismatch");
    validate_random_canonicalisation();
    OrbitMap parents;
    parents.emplace(U128(0), 1);
    const double begin = seconds_now();
    for (unsigned columns = 1; columns <= maximum_columns; ++columns) {
        OrbitMap ordinary;
        OrbitMap canonical;
        std::unordered_map<U128, U128, U128Hash> origins;
        ordinary.reserve(parents.size() * 8);
        canonical.reserve(parents.size() * 8);
        origins.reserve(parents.size() * 8);
        uint64_t candidates = 0;
        uint64_t accepted_paths = 0;
        for (const auto& parent : parents) {
            const auto parent_rows = unpack_rows(parent.first, columns - 1);
            for (unsigned assignment = 0; assignment < ASSIGNMENTS;
                 ++assignment) {
                std::array<RowPattern, ROWS> child_rows{};
                for (unsigned row = 0; row < ROWS; ++row) {
                    child_rows[row] = parent_rows[row] |
                        RowPattern(((assignment >> row) & 1U) <<
                                   (columns - 1));
                }
                const CanonicalResult basic =
                    canonicalise(child_rows, columns);
                ordinary[basic.key] += parent.second;
                const CanonicalResult exact = canonicalise(
                    child_rows, columns, int(columns - 1));
                candidates++;
                if (!exact.canonical_extension) continue;
                accepted_paths++;
                const uint64_t group_order =
                    FACTORIAL[ROWS] * FACTORIAL[columns];
                if (!exact.automorphisms ||
                    group_order % exact.automorphisms)
                    throw std::runtime_error("bad automorphism divisor");
                const uint64_t weight = group_order / exact.automorphisms;
                auto [origin, inserted_origin] =
                    origins.emplace(exact.key, parent.first);
                if (!inserted_origin && origin->second != parent.first)
                    throw std::runtime_error("child has two canonical parents");
                auto [item, inserted] = canonical.emplace(exact.key, weight);
                if (!inserted && item->second != weight)
                    throw std::runtime_error("inconsistent canonical weight");

                auto child = unpack_rows(exact.key, columns);
                for (RowPattern& row : child) row >>= 1;
                const U128 deleted_parent = columns == 1 ? U128(0) :
                    canonicalise(child, columns - 1).key;
                if (deleted_parent != parent.first)
                    throw std::runtime_error("distinguished parent mismatch");
            }
        }
        if (ordinary.size() != canonical.size())
            throw std::runtime_error("canonical-parent count mismatch");
        U128 weight_sum = 0;
        for (const auto& item : ordinary) {
            const auto found = canonical.find(item.first);
            if (found == canonical.end() || found->second != item.second)
                throw std::runtime_error("canonical-parent corpus mismatch");
            weight_sum += item.second;
        }
        if (weight_sum != (U128(1) << (ROWS * columns)))
            throw std::runtime_error("labelled-weight mismatch");
        if ((columns == 5 && ordinary.size() != 28576) ||
            (columns == 6 && ordinary.size() != 251610))
            throw std::runtime_error("known orbit census mismatch");
        std::printf("R6_CANONICAL_PARENT_SELF_TEST columns=%u records=%zu "
                    "candidates=%llu accepted_paths=%llu weight=%s OK\n",
                    columns, ordinary.size(),
                    (unsigned long long)candidates,
                    (unsigned long long)accepted_paths,
                    u128_string(weight_sum).c_str());
        parents = std::move(canonical);
    }
    std::printf("R6_CANONICAL_PARENT_SELF_TEST maximum_columns=%u "
                "seconds=%.6f exact=OK\n", maximum_columns,
                seconds_now() - begin);
}

struct CheckedRecord {
    OutputRecord record;
    unsigned owner;
    bool right_major;
};

static void run_check_range(const char* table_path,
                            const std::vector<const char*>& paths) {
    if (paths.empty() || paths.size() > 1024)
        throw std::runtime_error("invalid shard list");
    initialise_choose();
    SupportTable support = map_support_table(table_path);
    std::vector<CheckedRecord> records;
    for (unsigned owner = 0; owner < paths.size(); ++owner) {
        MappedFile file = map_read_only(paths[owner]);
        if (file.bytes < 20) throw std::runtime_error("short output shard");
        char magic[8];
        uint32_t columns = 0;
        uint64_t count = 0;
        std::memcpy(magic, file.data, 8);
        std::memcpy(&columns, file.data + 8, 4);
        std::memcpy(&count, file.data + 12, 8);
        const bool right_major = !std::memcmp(magic, "R6W1202", 8);
        if ((!right_major && std::memcmp(magic, "R6W1201", 8)) ||
            columns != COLUMNS ||
            file.bytes != 20 + count * sizeof(OutputRecord))
            throw std::runtime_error("invalid output shard");
        const uint8_t* input = file.data + 20;
        records.reserve(records.size() + size_t(count));
        for (uint64_t index = 0; index < count; ++index)
            records.push_back(CheckedRecord{
                load_unaligned_record<OutputRecord>(input, index), owner,
                right_major});
        unmap_file(file);
    }
    std::vector<U128> keys(records.size());
    std::vector<uint64_t> weights(records.size());
    std::vector<uint8_t> midpoint(records.size());
    std::atomic<bool> failure{false};
    const double begin = seconds_now();
#pragma omp parallel for schedule(dynamic, 64)
    for (long long index = 0; index < (long long)records.size(); ++index) {
        const CheckedRecord& item = records[size_t(index)];
        const uint64_t weight = item.record.meta >> OUTPUT_KEY_BITS;
        const U128 stored_key =
            (U128(item.record.meta & OUTPUT_KEY_MASK) << 64) |
            item.record.low;
        const U128 key = item.right_major ? interleaved_key(stored_key)
                                          : stored_key;
        const unsigned cells = unsigned(__builtin_popcountll(item.record.low)) +
            unsigned(__builtin_popcountll(item.record.meta & OUTPUT_KEY_MASK));
        const CanonicalResult canonical =
            canonicalise(unpack_rows(key, COLUMNS), COLUMNS);
        if (!weight || cells > MIDPOINT || !canonical.automorphisms ||
            GROUP_ORDER % canonical.automorphisms ||
            weight != GROUP_ORDER / canonical.automorphisms ||
            select_cut(support.entries, canonical.key) != key ||
            mix64(left_prefix(key)) % paths.size() != item.owner)
            failure.store(true, std::memory_order_relaxed);
        keys[size_t(index)] = canonical.key;
        weights[size_t(index)] = weight;
        midpoint[size_t(index)] = cells == MIDPOINT;
    }
    if (failure.load(std::memory_order_relaxed))
        throw std::runtime_error("output record validation failed");
    std::sort(keys.begin(), keys.end());
    if (std::adjacent_find(keys.begin(), keys.end()) != keys.end())
        throw std::runtime_error("duplicate canonical child in range");
    U128 retained_weight = 0;
    U128 midpoint_weight = 0;
    uint64_t midpoint_records = 0;
    for (size_t index = 0; index < records.size(); ++index) {
        retained_weight += weights[index];
        if (midpoint[index]) {
            midpoint_records++;
            midpoint_weight += weights[index];
        }
    }
    std::printf("R6X12_CANONICAL_RANGE_CHECK records=%zu midpoint_records=%llu "
                "retained_weight=%s midpoint_weight=%s shards=%zu "
                "seconds=%.6f exact=OK\n",
                records.size(), (unsigned long long)midpoint_records,
                u128_string(retained_weight).c_str(),
                u128_string(midpoint_weight).c_str(), paths.size(),
                seconds_now() - begin);
    unmap_file(support.file);
}

static U128 record_key(const OutputRecord& record) {
    return (U128(record.meta & OUTPUT_KEY_MASK) << 64) | record.low;
}

static void run_reduce_owner(unsigned shards, unsigned owner,
                             const char* output_path,
                             const std::vector<const char*>& paths,
                             bool delete_inputs) {
    if (!shards || shards > 1024 || owner >= shards || paths.empty())
        throw std::runtime_error("invalid owner reduction");
    std::vector<OutputRecord> records;
    const double begin = seconds_now();
    for (const char* path : paths) {
        MappedFile file = map_read_only(path);
        if (file.bytes < 20) throw std::runtime_error("short fragment");
        char magic[8];
        uint32_t columns = 0;
        uint64_t count = 0;
        std::memcpy(magic, file.data, 8);
        std::memcpy(&columns, file.data + 8, 4);
        std::memcpy(&count, file.data + 12, 8);
        if (std::memcmp(magic, "R6W1201", 8) || columns != COLUMNS ||
            file.bytes != 20 + count * sizeof(OutputRecord))
            throw std::runtime_error("invalid fragment");
        const size_t old_size = records.size();
        records.resize(old_size + size_t(count));
        if (count)
            std::memcpy(records.data() + old_size, file.data + 20,
                        size_t(count) * sizeof(OutputRecord));
        unmap_file(file);
    }
    std::atomic<bool> bad_owner{false};
#pragma omp parallel for schedule(static)
    for (long long index = 0; index < (long long)records.size(); ++index) {
        const OutputRecord& record = records[size_t(index)];
        const U128 key = record_key(record);
        const uint64_t weight = record.meta >> OUTPUT_KEY_BITS;
        const unsigned cells = unsigned(__builtin_popcountll(record.low)) +
            unsigned(__builtin_popcountll(record.meta & OUTPUT_KEY_MASK));
        if (!weight || cells > MIDPOINT ||
            mix64(left_prefix(key)) % shards != owner)
            bad_owner.store(true, std::memory_order_relaxed);
    }
    if (bad_owner.load(std::memory_order_relaxed))
        throw std::runtime_error("bad fragment record");

    // Convert once before the only global sort.  This representation is a
    // bijection, so duplicate detection and all coefficient checks remain
    // exact while every later GPU solve can consume the stored order.
#pragma omp parallel for schedule(static)
    for (long long index = 0; index < (long long)records.size(); ++index) {
        OutputRecord& record = records[size_t(index)];
        const uint64_t weight = record.meta >> OUTPUT_KEY_BITS;
        record = output_record(right_major_key(record_key(record)), weight);
    }
    std::sort(records.begin(), records.end(), [](const OutputRecord& a,
                                                  const OutputRecord& b) {
        return record_key(a) < record_key(b);
    });
    for (size_t index = 1; index < records.size(); ++index)
        if (record_key(records[index - 1]) == record_key(records[index]))
            throw std::runtime_error("duplicate child across generator ranges");
    U128 retained_weight = 0;
    U128 midpoint_weight = 0;
    uint64_t midpoint_records = 0;
    for (const OutputRecord& record : records) {
        const uint64_t weight = record.meta >> OUTPUT_KEY_BITS;
        retained_weight += weight;
        const unsigned cells = unsigned(__builtin_popcountll(record.low)) +
            unsigned(__builtin_popcountll(record.meta & OUTPUT_KEY_MASK));
        if (cells == MIDPOINT) {
            midpoint_records++;
            midpoint_weight += weight;
        }
    }
    const std::string temporary = std::string(output_path) + ".tmp";
    FILE* output = std::fopen(temporary.c_str(), "wb");
    if (!output) throw std::runtime_error("cannot create reduced owner");
    write_header(output, records.size(), "R6W1202");
    if (std::fwrite(records.data(), sizeof(OutputRecord), records.size(),
                    output) != records.size() ||
        std::fclose(output) || rename(temporary.c_str(), output_path))
        throw std::runtime_error("cannot publish reduced owner");
    if (delete_inputs)
        for (const char* path : paths)
            if (unlink(path)) throw std::runtime_error("cannot delete fragment");
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    std::printf("R6X12_REDUCE_OWNER owner=%u shards=%u fragments=%zu "
                "records=%zu midpoint_records=%llu retained_weight=%s "
                "midpoint_weight=%s seconds=%.6f peak_rss_mib=%.3f "
                "delete_inputs=%u output=%s exact=OK\n",
                owner, shards, paths.size(), records.size(),
                (unsigned long long)midpoint_records,
                u128_string(retained_weight).c_str(),
                u128_string(midpoint_weight).c_str(),
                seconds_now() - begin, usage.ru_maxrss / 1024.0,
                unsigned(delete_inputs), output_path);
}

static void run_check_full(const char* table_path,
                           const std::vector<const char*>& paths) {
    if (paths.empty() || paths.size() > 1024)
        throw std::runtime_error("invalid full shard set");
    initialise_choose();
    SupportTable support = map_support_table(table_path);
    uint64_t records = 0;
    uint64_t midpoint_records = 0;
    U128 retained_weight = 0;
    U128 midpoint_weight = 0;
    uint64_t exact_checks = 0;
    const double begin = seconds_now();
    for (unsigned owner = 0; owner < paths.size(); ++owner) {
        MappedFile file = map_read_only(paths[owner]);
        if (file.bytes < 20) throw std::runtime_error("short solve shard");
        char magic[8];
        uint32_t columns = 0;
        uint64_t count = 0;
        std::memcpy(magic, file.data, 8);
        std::memcpy(&columns, file.data + 8, 4);
        std::memcpy(&count, file.data + 12, 8);
        const bool right_major = !std::memcmp(magic, "R6W1202", 8);
        if ((!right_major && std::memcmp(magic, "R6W1201", 8)) ||
            columns != COLUMNS ||
            file.bytes != 20 + count * sizeof(OutputRecord))
            throw std::runtime_error("invalid solve shard");
        const uint8_t* source = file.data + 20;
        U128 previous_stored_key = 0;
        for (uint64_t index = 0; index < count; ++index) {
            const OutputRecord record = load_unaligned_record<OutputRecord>(
                source, index);
            const U128 stored_key = record_key(record);
            if (right_major && index && stored_key <= previous_stored_key)
                throw std::runtime_error(
                    "right-major solve shard is not strictly sorted");
            previous_stored_key = stored_key;
            const U128 key = right_major ? interleaved_key(stored_key)
                                         : stored_key;
            const uint64_t weight = record.meta >> OUTPUT_KEY_BITS;
            const unsigned cells = unsigned(__builtin_popcountll(record.low)) +
                unsigned(__builtin_popcountll(record.meta & OUTPUT_KEY_MASK));
            if (!weight || cells > MIDPOINT ||
                mix64(left_prefix(key)) % paths.size() != owner)
                throw std::runtime_error("solve-shard structural mismatch");
            records++;
            retained_weight += weight;
            if (cells == MIDPOINT) {
                midpoint_records++;
                midpoint_weight += weight;
            }
            if (index < 4 || (count > 4 && index == count - 1)) {
                const CanonicalResult canonical =
                    canonicalise(unpack_rows(key, COLUMNS), COLUMNS);
                if (!canonical.automorphisms ||
                    GROUP_ORDER % canonical.automorphisms ||
                    weight != GROUP_ORDER / canonical.automorphisms ||
                    select_cut(support.entries, canonical.key) != key)
                    throw std::runtime_error("solve-shard exact sample mismatch");
                exact_checks++;
            }
        }
        unmap_file(file);
    }
    const U128 covered = retained_weight * 2U - midpoint_weight;
    if (records != EXPECTED_RECORDS ||
        midpoint_records != EXPECTED_MIDPOINT_RECORDS ||
        retained_weight != EXPECTED_RETAINED_WEIGHT ||
        midpoint_weight != EXPECTED_MIDPOINT_WEIGHT ||
        covered != (U128(1) << CELLS))
        throw std::runtime_error("full 6x12 aggregate mismatch");
    std::printf("R6X12_FULL_CHECK records=%llu midpoint_records=%llu "
                "retained_weight=%s midpoint_weight=%s covered_weight=%s "
                "shards=%zu exact_samples=%llu seconds=%.6f exact=OK\n",
                (unsigned long long)records,
                (unsigned long long)midpoint_records,
                u128_string(retained_weight).c_str(),
                u128_string(midpoint_weight).c_str(),
                u128_string(covered).c_str(), paths.size(),
                (unsigned long long)exact_checks, seconds_now() - begin);
    unmap_file(support.file);
}

static void usage(const char* program) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s self-test [MAX_COLUMNS=6]\n"
        "  %s generate-range SUPPORT_TABLE INPUT_6x11 START END "
        "SHARDS OUTPUT_PREFIX\n"
        "  %s check-range SUPPORT_TABLE SHARD_FILE...\n"
        "  %s make-parent-sample INPUT_6x10 START END OUTPUT_6x11\n"
        "  %s reduce-owner SHARDS OWNER OUTPUT FRAGMENT...\n"
        "  %s reduce-owner-delete SHARDS OWNER OUTPUT FRAGMENT...\n"
        "  %s check-full SUPPORT_TABLE SOLVE_SHARD...\n",
        program, program, program, program, program, program, program);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if ((argc == 2 || argc == 3) && !std::strcmp(argv[1], "self-test")) {
            run_self_test(argc == 3 ? std::strtoul(argv[2], nullptr, 10) : 6);
            return 0;
        }
        if (argc == 8 && !std::strcmp(argv[1], "generate-range")) {
            run_generate_range(argv[2], argv[3], std::strtoull(argv[4], nullptr, 10),
                               std::strtoull(argv[5], nullptr, 10),
                               std::strtoul(argv[6], nullptr, 10), argv[7]);
            return 0;
        }
        if (argc >= 4 && !std::strcmp(argv[1], "check-range")) {
            std::vector<const char*> paths(argv + 3, argv + argc);
            run_check_range(argv[2], paths);
            return 0;
        }
        if (argc == 6 && !std::strcmp(argv[1], "make-parent-sample")) {
            run_make_parent_sample(
                argv[2], std::strtoull(argv[3], nullptr, 10),
                std::strtoull(argv[4], nullptr, 10), argv[5]);
            return 0;
        }
        if (argc >= 6 &&
            (!std::strcmp(argv[1], "reduce-owner") ||
             !std::strcmp(argv[1], "reduce-owner-delete"))) {
            std::vector<const char*> paths(argv + 5, argv + argc);
            run_reduce_owner(std::strtoul(argv[2], nullptr, 10),
                             std::strtoul(argv[3], nullptr, 10), argv[4],
                             paths,
                             !std::strcmp(argv[1], "reduce-owner-delete"));
            return 0;
        }
        if (argc >= 4 && !std::strcmp(argv[1], "check-full")) {
            std::vector<const char*> paths(argv + 3, argv + argc);
            run_check_full(argv[2], paths);
            return 0;
        }
        usage(argv[0]);
        return 2;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
