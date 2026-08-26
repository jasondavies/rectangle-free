#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef unsigned __int128 U128;
typedef uint16_t RowPattern;

enum {
    ROWS = 6,
    PARENT_COLUMNS = 10,
    COLUMNS = 11,
    PARENT_CELLS = 60,
    CELLS = 66,
    MIDPOINT = 33,
    ASSIGNMENTS = 64,
    TABLE_BITS = 32,
    OUTPUT_BUFFER_BYTES = 1 << 20
};

static const uint64_t EXPECTED_PARENTS = UINT64_C(917558397);
static const uint64_t EXPECTED_RETAINED_CANDIDATES = UINT64_C(32058782252);
static const uint64_t EXPECTED_RECORDS = UINT64_C(3294410345);
static const uint64_t EXPECTED_MIDPOINT_RECORDS = UINT64_C(550078210);
static const U128 EXPECTED_RETAINED_WEIGHT =
    ((U128)UINT64_C(2) << 64) + UINT64_C(3609714217008132870);
static const U128 EXPECTED_MIDPOINT_WEIGHT =
    UINT64_C(7219428434016265740);

typedef struct {
    uint64_t key;
    uint64_t weight;
} ParentRecord;

// The 66-bit canonical key occupies low plus the low two bits of meta.  The
// remaining 62 bits store the exact orbit weight.  This is both the compact
// on-disk ABI and, with stored_low shifted by one, the in-memory hash slot.
typedef struct {
    uint64_t low;
    uint64_t meta;
} WideOrbitRecord;

typedef struct {
    uint64_t stored_low;
    uint64_t meta;
} WideSlot;

typedef struct {
    RowPattern rows[ROWS];
    uint8_t row_degree[ROWS];
    uint8_t target_degree[ROWS];
    uint8_t order[ROWS];
    uint8_t used[ROWS];
    U128 best;
} CanonContext;

static double seconds_now(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec * 1e-9;
}

static void print_u128(U128 value) {
    char digits[64];
    int length = 0;
    do {
        digits[length++] = (char)('0' + value % 10);
        value /= 10;
    } while (value);
    while (length) putchar(digits[--length]);
}

static uint64_t mix64(uint64_t value) {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static uint64_t wide_hash(U128 key) {
    return mix64((uint64_t)key ^
                 mix64((uint64_t)(key >> 64) +
                       UINT64_C(0x9e3779b97f4a7c15)));
}

static U128 pack_rows(const RowPattern rows[ROWS], int columns) {
    U128 key = 0;
    for (int row = 0; row < ROWS; row++)
        key = (key << columns) | rows[row];
    return key;
}

static void unpack_rows_u64(uint64_t key, RowPattern rows[ROWS], int columns) {
    uint64_t mask = (UINT64_C(1) << columns) - 1U;
    for (int row = ROWS - 1; row >= 0; row--) {
        rows[row] = (RowPattern)(key & mask);
        key >>= columns;
    }
}

static void unpack_rows_wide(U128 key, RowPattern rows[ROWS], int columns) {
    U128 mask = ((U128)1 << columns) - 1U;
    for (int row = ROWS - 1; row >= 0; row--) {
        rows[row] = (RowPattern)(key & mask);
        key >>= columns;
    }
}

static void evaluate_row_order(CanonContext* context) {
    uint8_t column_vector[COLUMNS];
    uint8_t column_degree[COLUMNS];
    uint8_t column_order[COLUMNS];
    for (int column = 0; column < COLUMNS; column++) {
        uint8_t vector = 0;
        uint8_t degree = 0;
        for (int position = 0; position < ROWS; position++) {
            unsigned bit =
                (context->rows[context->order[position]] >> column) & 1U;
            vector = (uint8_t)((vector << 1) | bit);
            degree += (uint8_t)bit;
        }
        column_vector[column] = vector;
        column_degree[column] = degree;
        column_order[column] = (uint8_t)column;
    }
    for (int i = 1; i < COLUMNS; i++) {
        uint8_t column = column_order[i];
        int j = i;
        while (j > 0) {
            uint8_t previous = column_order[j - 1];
            if (column_degree[previous] < column_degree[column] ||
                (column_degree[previous] == column_degree[column] &&
                 column_vector[previous] <= column_vector[column])) break;
            column_order[j] = previous;
            j--;
        }
        column_order[j] = column;
    }
    RowPattern canonical_rows[ROWS] = {0};
    for (int position = 0; position < ROWS; position++) {
        RowPattern pattern = 0;
        uint8_t original_row = context->order[position];
        for (int column_position = 0; column_position < COLUMNS;
             column_position++) {
            uint8_t original_column = column_order[column_position];
            if ((context->rows[original_row] >> original_column) & 1U)
                pattern |= (RowPattern)(1U << column_position);
        }
        canonical_rows[position] = pattern;
    }
    U128 key = pack_rows(canonical_rows, COLUMNS);
    if (key < context->best) context->best = key;
}

static void canonical_rows_rec(CanonContext* context, int depth) {
    if (depth == ROWS) {
        evaluate_row_order(context);
        return;
    }
    uint8_t degree = context->target_degree[depth];
    uint64_t seen_patterns[32] = {0};
    for (int row = 0; row < ROWS; row++) {
        if (context->used[row] || context->row_degree[row] != degree) continue;
        RowPattern pattern = context->rows[row];
        uint64_t bit = UINT64_C(1) << (pattern & 63U);
        if (seen_patterns[pattern >> 6] & bit) continue;
        seen_patterns[pattern >> 6] |= bit;
        context->used[row] = 1;
        context->order[depth] = (uint8_t)row;
        canonical_rows_rec(context, depth + 1);
        context->used[row] = 0;
    }
}

static U128 canonical_key(const RowPattern rows[ROWS]) {
    CanonContext context = {.best = ~(U128)0};
    memcpy(context.rows, rows, sizeof(context.rows));
    for (int row = 0; row < ROWS; row++) {
        context.row_degree[row] =
            (uint8_t)__builtin_popcount((unsigned)rows[row]);
        context.target_degree[row] = context.row_degree[row];
    }
    for (int i = 1; i < ROWS; i++) {
        uint8_t degree = context.target_degree[i];
        int j = i;
        while (j && context.target_degree[j - 1] > degree) {
            context.target_degree[j] = context.target_degree[j - 1];
            j--;
        }
        context.target_degree[j] = degree;
    }
    canonical_rows_rec(&context, 0);
    return context.best;
}

static void validate_canonicalizer(void) {
    uint64_t state = UINT64_C(0x6429d5f3c17e8a0b);
    for (int sample = 0; sample < 1000; sample++) {
        RowPattern rows[ROWS];
        for (int row = 0; row < ROWS; row++) {
            state = mix64(state + UINT64_C(0x9e3779b97f4a7c15));
            rows[row] = (RowPattern)(state & ((1U << COLUMNS) - 1U));
        }
        uint8_t row_order[ROWS];
        uint8_t column_order[COLUMNS];
        for (int row = 0; row < ROWS; row++) row_order[row] = (uint8_t)row;
        for (int column = 0; column < COLUMNS; column++)
            column_order[column] = (uint8_t)column;
        for (int row = ROWS - 1; row > 0; row--) {
            state = mix64(state + UINT64_C(0x9e3779b97f4a7c15));
            int other = (int)(state % (uint64_t)(row + 1));
            uint8_t temporary = row_order[row];
            row_order[row] = row_order[other];
            row_order[other] = temporary;
        }
        for (int column = COLUMNS - 1; column > 0; column--) {
            state = mix64(state + UINT64_C(0x9e3779b97f4a7c15));
            int other = (int)(state % (uint64_t)(column + 1));
            uint8_t temporary = column_order[column];
            column_order[column] = column_order[other];
            column_order[other] = temporary;
        }
        RowPattern permuted[ROWS] = {0};
        for (int destination_row = 0; destination_row < ROWS;
             destination_row++) {
            RowPattern source = rows[row_order[destination_row]];
            for (int destination_column = 0; destination_column < COLUMNS;
                 destination_column++) {
                if (source & (1U << column_order[destination_column]))
                    permuted[destination_row] |=
                        (RowPattern)(1U << destination_column);
            }
        }
        if (canonical_key(rows) != canonical_key(permuted)) {
            fprintf(stderr, "wide canonicalizer self-test failed\n");
            exit(1);
        }
    }
}

static int wide_table_add(WideSlot* slots, uint64_t mask, U128 key,
                          uint64_t weight) {
    const uint64_t lock = UINT64_MAX;
    uint64_t low = (uint64_t)key;
    uint64_t high = (uint64_t)(key >> 64);
    uint64_t stored = low + 1U;
    if (!stored || stored == lock || high > 3 || weight >= (UINT64_C(1) << 62))
        exit(1);
    uint64_t slot = wide_hash(key) & mask;
    for (;;) {
        WideSlot* entry = &slots[slot];
        uint64_t observed =
            __atomic_load_n(&entry->stored_low, __ATOMIC_ACQUIRE);
        if (observed == stored) {
            uint64_t meta = __atomic_load_n(&entry->meta, __ATOMIC_ACQUIRE);
            if ((meta & 3U) == high) {
                __atomic_fetch_add(&entry->meta, weight << 2,
                                   __ATOMIC_RELAXED);
                return 0;
            }
        } else if (!observed) {
            uint64_t expected = 0;
            if (__atomic_compare_exchange_n(
                    &entry->stored_low, &expected, lock, 0,
                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
                entry->meta = (weight << 2) | high;
                __atomic_store_n(&entry->stored_low, stored, __ATOMIC_RELEASE);
                return 1;
            }
            continue;
        } else if (observed == lock) {
            continue;
        }
        slot = (slot + 1U) & mask;
    }
}

static void validate_table(void) {
    const uint64_t capacity = 1024;
    WideSlot* slots = calloc(capacity, sizeof(*slots));
    if (!slots) exit(1);
    uint64_t inserted = 0;
#pragma omp parallel for schedule(static) reduction(+:inserted)
    for (int index = 0; index < 100000; index++) {
        U128 key = ((U128)(index % 4) << 64) | (uint64_t)(index % 100);
        inserted += wide_table_add(slots, capacity - 1, key, 1);
    }
    uint64_t count = 0;
    uint64_t weight = 0;
    for (uint64_t slot = 0; slot < capacity; slot++) {
        if (!slots[slot].stored_low) continue;
        if (slots[slot].stored_low == UINT64_MAX) exit(1);
        count++;
        weight += slots[slot].meta >> 2;
    }
    free(slots);
    if (inserted != 100 || count != 100 || weight != 100000) {
        fprintf(stderr, "wide table self-test failed\n");
        exit(1);
    }
}

static void raise_file_limit(int required) {
    struct rlimit limit;
    if (getrlimit(RLIMIT_NOFILE, &limit) != 0) exit(1);
    if (limit.rlim_cur >= (rlim_t)required) return;
    if (limit.rlim_max < (rlim_t)required) exit(1);
    limit.rlim_cur = (rlim_t)required;
    if (setrlimit(RLIMIT_NOFILE, &limit) != 0) exit(1);
}

static void write_header(FILE* file, uint64_t count) {
    const char magic[8] = "R6W1101";
    uint32_t columns = COLUMNS;
    if (fwrite(magic, sizeof(magic), 1, file) != 1 ||
        fwrite(&columns, sizeof(columns), 1, file) != 1 ||
        fwrite(&count, sizeof(count), 1, file) != 1) exit(1);
}

static uint64_t packed_half(const RowPattern rows[ROWS], int shift,
                            int columns) {
    uint64_t result = 0;
    uint16_t mask = (uint16_t)((1U << columns) - 1U);
    for (int row = 0; row < ROWS; row++)
        result = (result << columns) | ((rows[row] >> shift) & mask);
    return result;
}

static void write_solve_shards(WideSlot* slots, uint64_t capacity, int shards,
                               const char* prefix) {
    raise_file_limit(shards + 32);
    FILE** outputs = calloc((size_t)shards, sizeof(*outputs));
    uint64_t* counts = calloc((size_t)shards, sizeof(*counts));
    if (!outputs || !counts) exit(1);
    char path[4096];
    for (int shard = 0; shard < shards; shard++) {
        snprintf(path, sizeof(path), "%s.s%04d.orbits", prefix, shard);
        outputs[shard] = fopen(path, "wb+");
        if (!outputs[shard]) exit(1);
        setvbuf(outputs[shard], NULL, _IOFBF, OUTPUT_BUFFER_BYTES);
        write_header(outputs[shard], 0);
    }
    uint64_t records = 0;
    uint64_t midpoint_records = 0;
    U128 retained_weight = 0;
    U128 midpoint_weight = 0;
    double begin = seconds_now();
    for (uint64_t slot = 0; slot < capacity; slot++) {
        uint64_t stored = slots[slot].stored_low;
        if (!stored) continue;
        if (stored == UINT64_MAX) exit(1);
        uint64_t low = stored - 1U;
        uint64_t meta = slots[slot].meta;
        uint64_t high = meta & 3U;
        uint64_t weight = meta >> 2;
        U128 key = ((U128)high << 64) | low;
        RowPattern rows[ROWS];
        unpack_rows_wide(key, rows, COLUMNS);
        int cells = __builtin_popcountll(low) + __builtin_popcountll(high);
        if (!weight || cells > MIDPOINT) exit(1);
        uint64_t left = packed_half(rows, 0, 5);
        int owner = (int)(mix64(left) % (uint64_t)shards);
        WideOrbitRecord record = {low, meta};
        if (fwrite(&record, sizeof(record), 1, outputs[owner]) != 1) exit(1);
        counts[owner]++;
        records++;
        retained_weight += weight;
        if (cells == MIDPOINT) {
            midpoint_records++;
            midpoint_weight += weight;
        }
        if (slot && !(slot & (UINT64_C(0x3fffffff)))) {
            fprintf(stderr, "write_scan=%llu/%llu records=%llu seconds=%.3f\n",
                    (unsigned long long)slot,
                    (unsigned long long)capacity,
                    (unsigned long long)records, seconds_now() - begin);
        }
    }
    U128 covered = retained_weight * 2U - midpoint_weight;
    if (records != EXPECTED_RECORDS ||
        midpoint_records != EXPECTED_MIDPOINT_RECORDS ||
        retained_weight != EXPECTED_RETAINED_WEIGHT ||
        midpoint_weight != EXPECTED_MIDPOINT_WEIGHT ||
        covered != ((U128)1 << CELLS)) {
        fprintf(stderr, "wide corpus aggregate validation failed\n");
        exit(1);
    }
    uint64_t minimum = UINT64_MAX;
    uint64_t maximum = 0;
    for (int shard = 0; shard < shards; shard++) {
        if (fseeko(outputs[shard], 12, SEEK_SET) != 0 ||
            fwrite(&counts[shard], sizeof(counts[shard]), 1,
                   outputs[shard]) != 1 ||
            fclose(outputs[shard]) != 0) exit(1);
        if (counts[shard] < minimum) minimum = counts[shard];
        if (counts[shard] > maximum) maximum = counts[shard];
    }
    printf("WIDE_SOLVE_SHARDS records=%llu midpoint_records=%llu shards=%d "
           "minimum_records=%llu maximum_records=%llu retained_weight=",
           (unsigned long long)records,
           (unsigned long long)midpoint_records, shards,
           (unsigned long long)minimum, (unsigned long long)maximum);
    print_u128(retained_weight);
    printf(" midpoint_weight=");
    print_u128(midpoint_weight);
    printf(" covered_weight=");
    print_u128(covered);
    printf(" seconds=%.3f prefix=%s OK\n", seconds_now() - begin, prefix);
    free(counts);
    free(outputs);
}

static void run_generate(const char* input_path, int shards,
                         const char* output_prefix) {
    if (shards < 1 || shards > 1024) exit(2);
    validate_canonicalizer();
    validate_table();
    int descriptor = open(input_path, O_RDONLY);
    if (descriptor < 0) exit(1);
    struct stat status;
    if (fstat(descriptor, &status) != 0) exit(1);
    void* mapping = mmap(NULL, (size_t)status.st_size, PROT_READ, MAP_PRIVATE,
                         descriptor, 0);
    if (mapping == MAP_FAILED) exit(1);
    const unsigned char* bytes = mapping;
    uint32_t columns;
    uint64_t parent_count;
    memcpy(&columns, bytes + 8, sizeof(columns));
    memcpy(&parent_count, bytes + 12, sizeof(parent_count));
    if (status.st_size != (off_t)(20 + parent_count * sizeof(ParentRecord)) ||
        memcmp(bytes, "R6ORB01", 7) || columns != PARENT_COLUMNS ||
        parent_count != UINT64_C(502732239)) exit(1);
    const uint64_t capacity = UINT64_C(1) << TABLE_BITS;
    const size_t table_bytes = (size_t)capacity * sizeof(WideSlot);
    WideSlot* slots = mmap(NULL, table_bytes, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (slots == MAP_FAILED) {
        fprintf(stderr, "cannot allocate %zu-byte wide table: %s\n",
                table_bytes, strerror(errno));
        exit(1);
    }
#ifdef MADV_HUGEPAGE
    madvise(slots, table_bytes, MADV_HUGEPAGE);
#endif
    uint64_t reconstructed_parents = 0;
    uint64_t raw_candidates = 0;
    uint64_t retained_candidates = 0;
    uint64_t unique = 0;
    double begin = seconds_now();
#pragma omp parallel for schedule(dynamic, 256) reduction(+:reconstructed_parents,raw_candidates,retained_candidates,unique)
    for (long long index = 0; index < (long long)parent_count; index++) {
        ParentRecord parent;
        memcpy(&parent, bytes + 20 + (size_t)index * sizeof(parent),
               sizeof(parent));
        RowPattern original[ROWS];
        unpack_rows_u64(parent.key, original, PARENT_COLUMNS);
        int original_cells = __builtin_popcountll(parent.key);
        int orientations = original_cells < PARENT_CELLS / 2 ? 2 : 1;
        for (int orientation = 0; orientation < orientations; orientation++) {
            RowPattern rows[ROWS];
            int cells = orientation ? PARENT_CELLS - original_cells
                                    : original_cells;
            reconstructed_parents++;
            raw_candidates += ASSIGNMENTS;
            for (int row = 0; row < ROWS; row++)
                rows[row] = orientation
                    ? (RowPattern)(original[row] ^ ((1U << PARENT_COLUMNS) - 1U))
                    : original[row];
            for (unsigned assignment = 0; assignment < ASSIGNMENTS;
                 assignment++) {
                int child_cells = cells + __builtin_popcount(assignment);
                if (child_cells > MIDPOINT) continue;
                retained_candidates++;
                RowPattern child[ROWS];
                for (int row = 0; row < ROWS; row++)
                    child[row] = rows[row] |
                        (RowPattern)(((assignment >> row) & 1U) <<
                                     PARENT_COLUMNS);
                U128 key = canonical_key(child);
                unique += wide_table_add(slots, capacity - 1, key,
                                         parent.weight);
            }
        }
        if (index && !(index & UINT64_C(0x7fffff))) {
#pragma omp critical(wide_generation_progress)
            fprintf(stderr, "parents=%lld/%llu seconds=%.3f\n", index,
                    (unsigned long long)parent_count, seconds_now() - begin);
        }
    }
    double generation_seconds = seconds_now() - begin;
    if (reconstructed_parents != EXPECTED_PARENTS ||
        raw_candidates != EXPECTED_PARENTS * ASSIGNMENTS ||
        retained_candidates != EXPECTED_RETAINED_CANDIDATES ||
        unique != EXPECTED_RECORDS) {
        fprintf(stderr, "wide generation census validation failed: "
                "parents=%llu raw=%llu retained=%llu unique=%llu\n",
                (unsigned long long)reconstructed_parents,
                (unsigned long long)raw_candidates,
                (unsigned long long)retained_candidates,
                (unsigned long long)unique);
        exit(1);
    }
    printf("WIDE_GENERATE parents=%llu raw_candidates=%llu "
           "retained_candidates=%llu unique=%llu generation_seconds=%.3f "
           "table_gib=%.3f OK\n",
           (unsigned long long)reconstructed_parents,
           (unsigned long long)raw_candidates,
           (unsigned long long)retained_candidates,
           (unsigned long long)unique, generation_seconds,
           table_bytes / 1073741824.0);
    write_solve_shards(slots, capacity, shards, output_prefix);
    if (munmap(slots, table_bytes) != 0 ||
        munmap(mapping, (size_t)status.st_size) != 0 ||
        close(descriptor) != 0) exit(1);
}

static void read_half_corpus(const char* path, int expected_columns,
                             uint64_t expected_count, ParentRecord** records) {
    FILE* file = fopen(path, "rb");
    if (!file) exit(1);
    char magic[8];
    uint32_t columns;
    uint64_t count;
    if (fread(magic, sizeof(magic), 1, file) != 1 ||
        fread(&columns, sizeof(columns), 1, file) != 1 ||
        fread(&count, sizeof(count), 1, file) != 1 ||
        memcmp(magic, "R6ORB01", 7) || columns != (uint32_t)expected_columns ||
        count != expected_count) exit(1);
    *records = malloc((size_t)count * sizeof(**records));
    if (!*records || fread(*records, sizeof(**records), count, file) != count ||
        fgetc(file) != EOF || fclose(file) != 0) exit(1);
}

static void run_promote_seed(const char* left_path, const char* right_path,
                             const char* output_path) {
    const uint64_t left_count = UINT64_C(28576);
    const uint64_t right_count = UINT64_C(251610);
    ParentRecord* left = NULL;
    ParentRecord* right = NULL;
    read_half_corpus(left_path, 5, left_count, &left);
    read_half_corpus(right_path, 6, right_count, &right);
    FILE* output = fopen(output_path, "wb");
    if (!output) exit(1);
    write_header(output, right_count);
    for (uint64_t index = 0; index < right_count; index++) {
        RowPattern left_rows[ROWS];
        RowPattern right_rows[ROWS];
        RowPattern rows[ROWS];
        // R6ORB01 canonical half corpora retain the generator's ten-bit row
        // stride even when their header records only five or six columns.
        unpack_rows_u64(left[index % left_count].key, left_rows,
                        PARENT_COLUMNS);
        unpack_rows_u64(right[index].key, right_rows, PARENT_COLUMNS);
        int left_cells = __builtin_popcountll(left[index % left_count].key);
        int right_cells = __builtin_popcountll(right[index].key);
        for (int row = 0; row < ROWS; row++) {
            if (left_cells > 15) left_rows[row] ^= 31U;
            if (right_cells > 18) right_rows[row] ^= 63U;
            rows[row] = left_rows[row] | (RowPattern)(right_rows[row] << 5);
        }
        U128 key = pack_rows(rows, COLUMNS);
        WideOrbitRecord record = {
            (uint64_t)key, (UINT64_C(1) << 2) | (uint64_t)(key >> 64)};
        if (fwrite(&record, sizeof(record), 1, output) != 1) exit(1);
    }
    if (fclose(output) != 0) exit(1);
    free(right);
    free(left);
    printf("WIDE_PROMOTE_SEED records=%llu output=%s OK\n",
           (unsigned long long)right_count, output_path);
}

static void run_check_solve(int shards, int input_count, char** paths) {
    if (shards < 1 || shards > 1024 || input_count != shards) exit(2);
    uint64_t records = 0;
    uint64_t midpoint_records = 0;
    uint64_t minimum = UINT64_MAX;
    uint64_t maximum = 0;
    U128 retained_weight = 0;
    U128 midpoint_weight = 0;
    double begin = seconds_now();
    for (int shard = 0; shard < shards; shard++) {
        int descriptor = open(paths[shard], O_RDONLY);
        if (descriptor < 0) exit(1);
        struct stat status;
        if (fstat(descriptor, &status) != 0 || status.st_size < 20) exit(1);
        void* mapping = mmap(NULL, (size_t)status.st_size, PROT_READ,
                             MAP_PRIVATE, descriptor, 0);
        if (mapping == MAP_FAILED) exit(1);
#ifdef MADV_SEQUENTIAL
        madvise(mapping, (size_t)status.st_size, MADV_SEQUENTIAL);
#endif
        const unsigned char* bytes = mapping;
        uint32_t columns;
        uint64_t count;
        memcpy(&columns, bytes + 8, sizeof(columns));
        memcpy(&count, bytes + 12, sizeof(count));
        if (memcmp(bytes, "R6W1101", 8) || columns != COLUMNS ||
            status.st_size != (off_t)(20 + count * sizeof(WideOrbitRecord))) {
            exit(1);
        }
        for (uint64_t index = 0; index < count; index++) {
            WideOrbitRecord record;
            memcpy(&record, bytes + 20 + index * sizeof(record),
                   sizeof(record));
            uint64_t high = record.meta & 3U;
            uint64_t weight = record.meta >> 2;
            U128 key = ((U128)high << 64) | record.low;
            int cells = __builtin_popcountll(record.low) +
                        __builtin_popcountll(high);
            if (!weight || cells > MIDPOINT) exit(1);
            RowPattern rows[ROWS];
            unpack_rows_wide(key, rows, COLUMNS);
            uint64_t left = packed_half(rows, 0, 5);
            if ((int)(mix64(left) % (uint64_t)shards) != shard) exit(1);
            retained_weight += weight;
            if (cells == MIDPOINT) {
                midpoint_records++;
                midpoint_weight += weight;
            }
        }
        records += count;
        if (count < minimum) minimum = count;
        if (count > maximum) maximum = count;
        if (munmap(mapping, (size_t)status.st_size) != 0 ||
            close(descriptor) != 0) exit(1);
    }
    U128 covered = retained_weight * 2U - midpoint_weight;
    if (records != EXPECTED_RECORDS ||
        midpoint_records != EXPECTED_MIDPOINT_RECORDS ||
        retained_weight != EXPECTED_RETAINED_WEIGHT ||
        midpoint_weight != EXPECTED_MIDPOINT_WEIGHT ||
        covered != ((U128)1 << CELLS)) exit(1);
    printf("WIDE_SOLVE_CHECK records=%llu midpoint_records=%llu shards=%d "
           "minimum_records=%llu maximum_records=%llu retained_weight=",
           (unsigned long long)records,
           (unsigned long long)midpoint_records, shards,
           (unsigned long long)minimum, (unsigned long long)maximum);
    print_u128(retained_weight);
    printf(" midpoint_weight=");
    print_u128(midpoint_weight);
    printf(" covered_weight=");
    print_u128(covered);
    printf(" seconds=%.3f OK\n", seconds_now() - begin);
}

static void usage(const char* program) {
    fprintf(stderr,
            "Usage:\n"
            "  %s self-test\n"
            "  %s generate RETAINED_6x10.orbits SHARDS OUTPUT_PREFIX\n"
            "  %s check-solve SHARDS SHARD_FILE...\n"
            "  %s promote-seed CANONICAL_6x5.orbits CANONICAL_6x6.orbits "
            "OUTPUT_6x11.orbits\n",
            program, program, program, program);
}

int main(int argc, char** argv) {
    if (argc == 2 && !strcmp(argv[1], "self-test")) {
        validate_canonicalizer();
        validate_table();
        puts("WIDE_6X11_SELF_TEST OK");
        return 0;
    }
    if (argc == 5 && !strcmp(argv[1], "generate")) {
        run_generate(argv[2], atoi(argv[3]), argv[4]);
        return 0;
    }
    if (argc == 5 && !strcmp(argv[1], "promote-seed")) {
        run_promote_seed(argv[2], argv[3], argv[4]);
        return 0;
    }
    if (argc >= 4 && !strcmp(argv[1], "check-solve")) {
        run_check_solve(atoi(argv[2]), argc - 3, argv + 3);
        return 0;
    }
    usage(argv[0]);
    return 2;
}
