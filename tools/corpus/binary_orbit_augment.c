/* Weighted canonical augmentation for small binary matrices. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef unsigned __int128 U128;

#ifndef ORBIT_ROWS
#define ORBIT_ROWS 7
#endif
#ifndef ORBIT_MAX_COLUMNS
#define ORBIT_MAX_COLUMNS 7
#endif
#ifndef ORBIT_ROW_BITS
#define ORBIT_ROW_BITS 8
#endif
#ifndef ORBIT_MAGIC
#define ORBIT_MAGIC "R7ORB01"
#endif

#if ORBIT_ROWS == ORBIT_MAX_COLUMNS && ORBIT_ROWS == 7
#define SQUARE_TRANSPOSE_MAGIC "R7SQT01"
#elif ORBIT_ROWS == ORBIT_MAX_COLUMNS && ORBIT_ROWS == 8
#define SQUARE_TRANSPOSE_MAGIC "R8SQT01"
#endif

enum {
    ROWS = ORBIT_ROWS,
    MAX_COLUMNS = ORBIT_MAX_COLUMNS,
    ROW_BITS = ORBIT_ROW_BITS,
    NEW_COLUMNS = 1 << ROWS,
    SEEN_WORDS = ((1 << MAX_COLUMNS) + 63) / 64
};

typedef uint16_t RowPattern;

typedef struct {
    uint64_t key;
    uint64_t weight;
} OrbitRecord;

typedef struct {
    OrbitRecord* entries;
    size_t capacity;
    size_t count;
} OrbitMap;

typedef struct {
    RowPattern rows[ROWS];
    uint8_t row_degree[ROWS];
    uint8_t target_degree[ROWS];
    uint8_t order[ROWS];
    uint8_t used[ROWS];
    int columns;
    uint64_t best;
} CanonContext;

#if ORBIT_ROWS == 7 && ORBIT_MAX_COLUMNS == 7
static const uint64_t g_expected_counts[MAX_COLUMNS + 1] = {
    1, 8, 70, 734, 9343, 136758, 2141733, 33642660
};
#elif ORBIT_ROWS == 7 && ORBIT_MAX_COLUMNS == 8
static const uint64_t g_expected_counts[MAX_COLUMNS + 1] = {
    1, 8, 70, 734, 9343, 136758, 2141733, 33642660, 508147108
};
#elif ORBIT_ROWS == 7 && ORBIT_MAX_COLUMNS == 9
static const uint64_t g_expected_counts[MAX_COLUMNS + 1] = {
    1, 8, 70, 734, 9343, 136758, 2141733, 33642660, 508147108,
    UINT64_C(7216495370)
};
#elif ORBIT_ROWS == 6 && ORBIT_MAX_COLUMNS == 9
static const uint64_t g_expected_counts[MAX_COLUMNS + 1] = {
    1, 7, 50, 386, 3250, 28576, 251610, 2141733, 17256831, 130237768
};
#elif ORBIT_ROWS == 6 && ORBIT_MAX_COLUMNS == 10
static const uint64_t g_expected_counts[MAX_COLUMNS + 1] = {
    1, 7, 50, 386, 3250, 28576, 251610, 2141733, 17256831, 130237768,
    917558397
};
#elif ORBIT_ROWS == 8 && ORBIT_MAX_COLUMNS == 8
static const uint64_t g_expected_counts[MAX_COLUMNS + 1] = {
    1, 9, 95, 1324, 25207, 613894, 17256831, 508147108, 14685630688
};
#else
#error "Add exact expected orbit counts for this augmentation geometry"
#endif

static double seconds_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void print_u128(U128 value) {
    char digits[40];
    int length = 0;
    do {
        digits[length++] = (char)('0' + value % 10);
        value /= 10;
    } while (value);
    while (length) putchar(digits[--length]);
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static void* xcalloc(size_t count, size_t size) {
    void* pointer = calloc(count, size);
    if (!pointer) {
        fprintf(stderr, "allocation failed for %zu bytes\n", count * size);
        exit(1);
    }
    return pointer;
}

static void orbit_map_init(OrbitMap* map, size_t capacity) {
    map->capacity = capacity;
    map->count = 0;
    map->entries = xcalloc(capacity, sizeof(map->entries[0]));
    for (size_t i = 0; i < capacity; i++) map->entries[i].key = UINT64_MAX;
}

static void orbit_map_insert_raw(OrbitMap* map, uint64_t key, uint64_t weight) {
    size_t slot = (size_t)mix64(key) & (map->capacity - 1U);
    while (map->entries[slot].key != UINT64_MAX) {
        slot = (slot + 1U) & (map->capacity - 1U);
    }
    map->entries[slot] = (OrbitRecord){.key = key, .weight = weight};
    map->count++;
}

static void orbit_map_rehash(OrbitMap* map) {
    OrbitRecord* old_entries = map->entries;
    size_t old_capacity = map->capacity;
    orbit_map_init(map, old_capacity << 1);
    for (size_t i = 0; i < old_capacity; i++) {
        if (old_entries[i].key != UINT64_MAX) {
            orbit_map_insert_raw(map, old_entries[i].key, old_entries[i].weight);
        }
    }
    free(old_entries);
}

static void orbit_map_add(OrbitMap* map, uint64_t key, uint64_t weight) {
    if ((map->count + 1U) * 10U >= map->capacity * 7U) orbit_map_rehash(map);
    size_t slot = (size_t)mix64(key) & (map->capacity - 1U);
    while (map->entries[slot].key != UINT64_MAX) {
        if (map->entries[slot].key == key) {
            map->entries[slot].weight += weight;
            return;
        }
        slot = (slot + 1U) & (map->capacity - 1U);
    }
    orbit_map_insert_raw(map, key, weight);
}

#if ORBIT_ROWS == 6 && ORBIT_MAX_COLUMNS == 10 && ORBIT_ROW_BITS == 10
// The retained depth-ten table is allocated at its final capacity, so it can
// be filled concurrently without rehashing.  A transient lock key publishes
// the weight before the real key; readers encountering it retry the same slot.
static int orbit_map_add_concurrent(OrbitMap* map, uint64_t key,
                                    uint64_t weight) {
    const uint64_t lock_key = UINT64_MAX - 1U;
    size_t slot = (size_t)mix64(key) & (map->capacity - 1U);
    for (;;) {
        OrbitRecord* entry = &map->entries[slot];
        uint64_t observed = __atomic_load_n(&entry->key, __ATOMIC_ACQUIRE);
        if (observed == key) {
            __atomic_fetch_add(&entry->weight, weight, __ATOMIC_RELAXED);
            return 0;
        }
        if (observed == UINT64_MAX) {
            uint64_t expected = UINT64_MAX;
            if (__atomic_compare_exchange_n(
                    &entry->key, &expected, lock_key, 0,
                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
                entry->weight = weight;
                __atomic_store_n(&entry->key, key, __ATOMIC_RELEASE);
                return 1;
            }
            continue;
        }
        if (observed == lock_key) continue;
        slot = (slot + 1U) & (map->capacity - 1U);
    }
}

static void validate_concurrent_orbit_map(void) {
    OrbitMap map;
    orbit_map_init(&map, 1024);
    uint64_t inserted = 0;
#pragma omp parallel for schedule(static) reduction(+:inserted)
    for (int index = 0; index < 100000; index++) {
        inserted += orbit_map_add_concurrent(
            &map, (uint64_t)(index % 100), 1
        );
    }
    map.count = (size_t)inserted;
    uint64_t weight = 0;
    for (size_t slot = 0; slot < map.capacity; slot++) {
        if (map.entries[slot].key != UINT64_MAX) weight += map.entries[slot].weight;
    }
    if (map.count != 100 || weight != 100000) {
        fprintf(stderr, "concurrent orbit-map self-test failed\n");
        exit(1);
    }
    free(map.entries);
}
#endif

static uint64_t pack_rows(const RowPattern rows[ROWS]) {
    uint64_t key = 0;
    for (int row = 0; row < ROWS; row++) key = (key << ROW_BITS) | rows[row];
    return key;
}

static void unpack_rows(uint64_t key, RowPattern rows[ROWS]) {
    const uint64_t mask = (UINT64_C(1) << ROW_BITS) - 1U;
    for (int row = ROWS - 1; row >= 0; row--) {
        rows[row] = (RowPattern)(key & mask);
        key >>= ROW_BITS;
    }
}

static void evaluate_row_order(CanonContext* context) {
    uint8_t column_vector[MAX_COLUMNS];
    uint8_t column_degree[MAX_COLUMNS];
    uint8_t column_order[MAX_COLUMNS];
    for (int column = 0; column < context->columns; column++) {
        uint8_t vector = 0;
        uint8_t degree = 0;
        for (int position = 0; position < ROWS; position++) {
            unsigned bit = (context->rows[context->order[position]] >> column) & 1U;
            vector = (uint8_t)((vector << 1) | bit);
            degree += (uint8_t)bit;
        }
        column_vector[column] = vector;
        column_degree[column] = degree;
        column_order[column] = (uint8_t)column;
    }
    for (int i = 1; i < context->columns; i++) {
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
        for (int column_position = 0; column_position < context->columns; column_position++) {
            uint8_t original_column = column_order[column_position];
            if ((context->rows[original_row] >> original_column) & 1U) {
                pattern |= (RowPattern)(1U << column_position);
            }
        }
        canonical_rows[position] = pattern;
    }
    uint64_t key = pack_rows(canonical_rows);
    if (key < context->best) context->best = key;
}

static void canonical_rows_rec(CanonContext* context, int depth) {
    if (depth == ROWS) {
        evaluate_row_order(context);
        return;
    }
    uint8_t degree = context->target_degree[depth];
    uint64_t seen_patterns[SEEN_WORDS];
    memset(seen_patterns, 0, sizeof(seen_patterns));
    for (int row = 0; row < ROWS; row++) {
        if (context->used[row] || context->row_degree[row] != degree) continue;
        RowPattern pattern = context->rows[row];
        uint64_t pattern_bit = UINT64_C(1) << (pattern & 63U);
        if (seen_patterns[pattern >> 6] & pattern_bit) continue;
        seen_patterns[pattern >> 6] |= pattern_bit;
        context->used[row] = 1;
        context->order[depth] = (uint8_t)row;
        canonical_rows_rec(context, depth + 1);
        context->used[row] = 0;
    }
}

static uint64_t canonical_key_reference(const RowPattern rows[ROWS], int columns) {
    CanonContext context = {.columns = columns, .best = UINT64_MAX};
    memcpy(context.rows, rows, sizeof(context.rows));
    for (int row = 0; row < ROWS; row++) {
        context.row_degree[row] = (uint8_t)__builtin_popcount((unsigned)rows[row]);
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

#if ORBIT_ROWS == 8 && ORBIT_MAX_COLUMNS == 8 && ORBIT_ROW_BITS == 8
typedef struct {
    RowPattern rows[8];
    uint8_t row_degree[8];
    uint8_t target_degree[8];
    uint8_t used[8];
    uint8_t column_degree[8];
    int columns;
    int distinguished_column;
    int canonical_extension;
    uint64_t best;
    uint64_t best_count;
} FastCanonContext;

static uint64_t g_spread_row[256];
static uint64_t g_column_contribution[8][256];
static int g_fast_canon_tables_ready = 0;

static void initialise_fast_canon_tables(void) {
    if (g_fast_canon_tables_ready) return;
    for (unsigned pattern = 0; pattern < 256; pattern++) {
        uint64_t spread = 0;
        for (int column = 0; column < 8; column++) {
            spread |= (uint64_t)((pattern >> column) & 1U) << (8 * column);
        }
        g_spread_row[pattern] = spread;
    }
    for (int column_position = 0; column_position < 8; column_position++) {
        for (unsigned vector = 0; vector < 256; vector++) {
            uint64_t contribution = 0;
            for (int row_position = 0; row_position < 8; row_position++) {
                if ((vector >> (7 - row_position)) & 1U) {
                    contribution |= UINT64_C(1) <<
                                    ((7 - row_position) * 8 + column_position);
                }
            }
            g_column_contribution[column_position][vector] = contribution;
        }
    }
    g_fast_canon_tables_ready = 1;
}

static void evaluate_fast_row_order(FastCanonContext* context,
                                    uint64_t column_vectors) {
    uint16_t columns[8];
    for (int column = 0; column < context->columns; column++) {
        uint8_t vector = (uint8_t)(column_vectors >> (8 * column));
        columns[column] = (uint16_t)((context->column_degree[column] << 8) | vector);
    }
    uint16_t distinguished = context->distinguished_column >= 0 ?
        columns[context->distinguished_column] : 0;
    for (int i = 1; i < context->columns; i++) {
        uint16_t value = columns[i];
        int j = i;
        while (j > 0 && columns[j - 1] > value) {
            columns[j] = columns[j - 1];
            j--;
        }
        columns[j] = value;
    }
    uint64_t key = 0;
    for (int column = 0; column < context->columns; column++) {
        key |= g_column_contribution[column][columns[column] & 255U];
    }
    int canonical_extension = context->distinguished_column >= 0 &&
                              distinguished == columns[0];
    if (key < context->best) {
        context->best = key;
        context->best_count = 1;
        context->canonical_extension = canonical_extension;
    } else if (key == context->best) {
        context->best_count++;
        context->canonical_extension |= canonical_extension;
    }
}

static void canonical_rows_fast_rec(FastCanonContext* context, int depth,
                                    uint64_t column_vectors) {
    if (depth == 8) {
        evaluate_fast_row_order(context, column_vectors);
        return;
    }
    uint8_t degree = context->target_degree[depth];
    uint64_t seen_patterns[4] = {0};
    for (int row = 0; row < 8; row++) {
        if (context->used[row] || context->row_degree[row] != degree) continue;
        RowPattern pattern = context->rows[row];
        uint64_t pattern_bit = UINT64_C(1) << (pattern & 63U);
        if (seen_patterns[pattern >> 6] & pattern_bit) continue;
        seen_patterns[pattern >> 6] |= pattern_bit;
        context->used[row] = 1;
        canonical_rows_fast_rec(
            context, depth + 1,
            ((column_vectors & UINT64_C(0x7f7f7f7f7f7f7f7f)) << 1) |
                g_spread_row[pattern]
        );
        context->used[row] = 0;
    }
}

static uint64_t canonical_key_8xn_fast(const RowPattern rows[8], int columns,
                                       uint64_t* automorphisms,
                                       int distinguished_column,
                                       int* canonical_extension) {
    initialise_fast_canon_tables();
    FastCanonContext context = {
        .columns = columns,
        .distinguished_column = distinguished_column,
        .best = UINT64_MAX
    };
    memcpy(context.rows, rows, sizeof(context.rows));
    for (int row = 0; row < 8; row++) {
        context.row_degree[row] = (uint8_t)__builtin_popcount((unsigned)rows[row]);
        context.target_degree[row] = context.row_degree[row];
        for (int column = 0; column < columns; column++) {
            context.column_degree[column] += (uint8_t)((rows[row] >> column) & 1U);
        }
    }
    for (int i = 1; i < 8; i++) {
        uint8_t degree = context.target_degree[i];
        int j = i;
        while (j > 0 && context.target_degree[j - 1] > degree) {
            context.target_degree[j] = context.target_degree[j - 1];
            j--;
        }
        context.target_degree[j] = degree;
    }
    canonical_rows_fast_rec(&context, 0, 0);
    if (automorphisms) {
        static const uint64_t factorial[9] = {
            1, 1, 2, 6, 24, 120, 720, 5040, 40320
        };
        uint64_t row_factor = 1;
        uint8_t row_seen[8] = {0};
        for (int row = 0; row < 8; row++) {
            if (row_seen[row]) continue;
            int multiplicity = 1;
            for (int other = row + 1; other < 8; other++) {
                if (rows[other] == rows[row]) {
                    row_seen[other] = 1;
                    multiplicity++;
                }
            }
            row_factor *= factorial[multiplicity];
        }
        uint8_t column_vectors[8] = {0};
        for (int column = 0; column < columns; column++) {
            for (int row = 0; row < 8; row++) {
                column_vectors[column] = (uint8_t)(
                    (column_vectors[column] << 1) | ((rows[row] >> column) & 1U)
                );
            }
        }
        uint64_t column_factor = 1;
        uint8_t column_seen[8] = {0};
        for (int column = 0; column < columns; column++) {
            if (column_seen[column]) continue;
            int multiplicity = 1;
            for (int other = column + 1; other < columns; other++) {
                if (column_vectors[other] == column_vectors[column]) {
                    column_seen[other] = 1;
                    multiplicity++;
                }
            }
            column_factor *= factorial[multiplicity];
        }
        *automorphisms = context.best_count * row_factor * column_factor;
    }
    if (canonical_extension) *canonical_extension = context.canonical_extension;
    return context.best;
}
#endif

static uint64_t canonical_key(const RowPattern rows[ROWS], int columns) {
#if ORBIT_ROWS == 8 && ORBIT_MAX_COLUMNS == 8 && ORBIT_ROW_BITS == 8 && \
    !defined(FORCE_REFERENCE_CANON)
    {
        uint64_t fast = canonical_key_8xn_fast(rows, columns, NULL, -1, NULL);
#ifdef VERIFY_FAST_CANON
        uint64_t reference = canonical_key_reference(rows, columns);
        if (fast != reference) {
            fprintf(stderr, "fast canonical mismatch fast=%016llx reference=%016llx\n",
                    (unsigned long long)fast, (unsigned long long)reference);
            exit(1);
        }
#endif
        return fast;
    }
#endif
    return canonical_key_reference(rows, columns);
}

#if ORBIT_ROWS == ORBIT_MAX_COLUMNS && defined(SQUARE_TRANSPOSE_MAGIC)
static uint64_t transpose_square_key(uint64_t key) {
    RowPattern source[ROWS];
    RowPattern target[ROWS];
    memset(target, 0, sizeof(target));
    unpack_rows(key, source);
    for (int row = 0; row < ROWS; row++) {
        for (int column = 0; column < ROWS; column++) {
            target[column] |= (RowPattern)(
                ((source[row] >> column) & 1U) << row
            );
        }
    }
    return pack_rows(target);
}

static uint64_t square_transpose_partner(uint64_t key) {
    RowPattern rows[ROWS];
    unpack_rows(transpose_square_key(key), rows);
    return canonical_key(rows, ROWS);
}

// Quotient a complete square row/column-orbit file by transposition.  This is
// the mandatory final corpus step for maintained square geometries;
// complement pairing remains in the solver because transposition preserves
// the number of selected cells.
static void run_square_transpose_filter(const char* input_path,
                                        const char* output_path) {
    FILE* input = fopen(input_path, "rb");
    if (!input) exit(1);
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    if (fread(magic, sizeof(magic), 1, input) != 1 ||
        memcmp(magic, ORBIT_MAGIC, 7) != 0 ||
        fread(&columns, sizeof(columns), 1, input) != 1 ||
        fread(&count, sizeof(count), 1, input) != 1 ||
        columns != ROWS || !count) {
        fprintf(stderr, "invalid square transpose input %s\n", input_path);
        exit(1);
    }
    OrbitRecord* records = xcalloc((size_t)count, sizeof(records[0]));
    if (fread(records, sizeof(records[0]), (size_t)count, input) != count ||
        fgetc(input) != EOF || fclose(input) != 0) exit(1);

    uint64_t fixed = 0;
    uint64_t lower = 0;
    uint64_t higher = 0;
    U128 input_weight = 0;
    double begin = seconds_now();
#pragma omp parallel for schedule(dynamic, 64) reduction(+:fixed,lower,higher,input_weight)
    for (uint64_t index = 0; index < count; index++) {
        OrbitRecord* record = &records[index];
        uint64_t original_weight = record->weight;
        uint64_t partner = square_transpose_partner(record->key);
        if (index < 8192 && square_transpose_partner(partner) != record->key) {
            fprintf(stderr, "square transpose involution mismatch at %llu\n",
                    (unsigned long long)index);
            exit(1);
        }
        input_weight += original_weight;
        if (partner == record->key) {
            fixed++;
        } else if (record->key < partner) {
            if (original_weight > UINT64_MAX / 2U) exit(1);
            record->weight = original_weight * 2U;
            lower++;
        } else {
            record->weight = 0;
            higher++;
        }
    }
    double filter_seconds = seconds_now() - begin;
    uint64_t output_count = fixed + lower;
    char temporary_path[4096];
    int path_length = snprintf(temporary_path, sizeof(temporary_path),
                               "%s.tmp", output_path);
    if (path_length < 0 || (size_t)path_length >= sizeof(temporary_path))
        exit(1);
    FILE* output = fopen(temporary_path, "wb");
    if (!output) exit(1);
    const char output_magic[8] = SQUARE_TRANSPOSE_MAGIC;
    if (fwrite(output_magic, sizeof(output_magic), 1, output) != 1 ||
        fwrite(&columns, sizeof(columns), 1, output) != 1 ||
        fwrite(&output_count, sizeof(output_count), 1, output) != 1) exit(1);
    U128 output_weight = 0;
    for (uint64_t index = 0; index < count; index++) {
        OrbitRecord record = records[index];
        if (!record.weight) continue;
        output_weight += record.weight;
        if (fwrite(&record, sizeof(record), 1, output) != 1) exit(1);
    }
    if (fclose(output) != 0 || rename(temporary_path, output_path) != 0 ||
        fixed + lower + higher != count || output_weight != input_weight) exit(1);
    printf("square_transpose_filter input_records=%llu output_records=%llu "
           "fixed=%llu lower=%llu higher=%llu seconds=%.6f weight=",
           (unsigned long long)count, (unsigned long long)output_count,
           (unsigned long long)fixed, (unsigned long long)lower,
           (unsigned long long)higher, filter_seconds);
    print_u128(output_weight);
    printf(" output=%s OK\n", output_path);
    free(records);
}
#endif

#if ORBIT_ROWS == 8 && ORBIT_MAX_COLUMNS == 8 && ORBIT_ROW_BITS == 8
static uint64_t canonical_key_with_automorphisms(const RowPattern rows[8], int columns,
                                                 uint64_t* automorphisms) {
    uint64_t key = canonical_key_8xn_fast(
        rows, columns, automorphisms, -1, NULL
    );
#ifdef VERIFY_FAST_CANON
    uint64_t reference = canonical_key_reference(rows, columns);
    if (key != reference) exit(1);
#endif
    return key;
}

static uint64_t canonical_extension_analysis(const RowPattern rows[8], int columns,
                                             uint64_t* automorphisms,
                                             int* canonical_extension) {
    return canonical_key_8xn_fast(
        rows, columns, automorphisms, columns - 1, canonical_extension
    );
}
#endif

static OrbitMap extend_map(const OrbitMap* parents, int parent_columns,
                           size_t start, size_t end, uint64_t* candidates) {
    size_t parent_count = end - start;
    size_t initial_capacity = 16;
    while (initial_capacity < parent_count * 16U) initial_capacity <<= 1;
    OrbitMap children;
    orbit_map_init(&children, initial_capacity);
    size_t parent_index = 0;
    for (size_t slot = 0; slot < parents->capacity; slot++) {
        OrbitRecord parent = parents->entries[slot];
        if (parent.key == UINT64_MAX) continue;
        if (parent_index >= start && parent_index < end) {
            RowPattern rows[ROWS];
            unpack_rows(parent.key, rows);
            for (unsigned assignment = 0; assignment < NEW_COLUMNS; assignment++) {
                RowPattern child_rows[ROWS];
                for (int row = 0; row < ROWS; row++) {
                    child_rows[row] = rows[row] |
                                      (RowPattern)(((assignment >> row) & 1U) << parent_columns);
                }
                uint64_t child_key = canonical_key(child_rows, parent_columns + 1);
                orbit_map_add(&children, child_key, parent.weight);
                (*candidates)++;
            }
        }
        parent_index++;
    }
    return children;
}

static void orbit_map_insert_unique(OrbitMap* map, uint64_t key, uint64_t value) {
    if ((map->count + 1U) * 10U >= map->capacity * 7U) orbit_map_rehash(map);
    size_t slot = (size_t)mix64(key) & (map->capacity - 1U);
    while (map->entries[slot].key != UINT64_MAX) {
        if (map->entries[slot].key == key) {
            fprintf(stderr, "duplicate orbit key %016llx\n",
                    (unsigned long long)key);
            exit(1);
        }
        slot = (slot + 1U) & (map->capacity - 1U);
    }
    orbit_map_insert_raw(map, key, value);
}

#if ORBIT_ROWS == 8 && ORBIT_MAX_COLUMNS == 8 && ORBIT_ROW_BITS == 8
static void orbit_map_set_exact(OrbitMap* map, uint64_t key, uint64_t value) {
    size_t slot = (size_t)mix64(key) & (map->capacity - 1U);
    while (map->entries[slot].key != UINT64_MAX) {
        if (map->entries[slot].key == key) {
            if (map->entries[slot].weight != value) {
                fprintf(stderr, "inconsistent exact map value\n");
                exit(1);
            }
            return;
        }
        slot = (slot + 1U) & (map->capacity - 1U);
    }
    if ((map->count + 1U) * 10U >= map->capacity * 7U) {
        orbit_map_rehash(map);
        orbit_map_set_exact(map, key, value);
        return;
    }
    orbit_map_insert_raw(map, key, value);
}

static uint64_t canonical_first_column_parent(uint64_t child, int columns) {
    RowPattern rows[8];
    unpack_rows(child, rows);
    for (int row = 0; row < 8; row++) rows[row] >>= 1;
    return canonical_key(rows, columns - 1);
}

static OrbitMap extend_map_canonical_parent(const OrbitMap* parents,
                                            int parent_columns,
                                            uint64_t* candidates,
                                            uint64_t* accepted_paths) {
    static const uint64_t factorial[9] = {
        1, 1, 2, 6, 24, 120, 720, 5040, 40320
    };
    size_t initial_capacity = 16;
    while (initial_capacity < parents->count * 16U) initial_capacity <<= 1;
    OrbitMap children;
    OrbitMap origins;
    orbit_map_init(&children, initial_capacity);
    orbit_map_init(&origins, initial_capacity);
    int columns = parent_columns + 1;
    uint64_t group_order = factorial[8] * factorial[columns];
    for (size_t slot = 0; slot < parents->capacity; slot++) {
        OrbitRecord parent = parents->entries[slot];
        if (parent.key == UINT64_MAX) continue;
        RowPattern rows[8];
        unpack_rows(parent.key, rows);
        for (unsigned assignment = 0; assignment < NEW_COLUMNS; assignment++) {
            RowPattern child_rows[8];
            for (int row = 0; row < 8; row++) {
                child_rows[row] = rows[row] |
                                  (RowPattern)(((assignment >> row) & 1U) <<
                                               parent_columns);
            }
            uint64_t automorphisms = 0;
            int canonical_extension = 0;
            uint64_t child = canonical_extension_analysis(
                child_rows, columns, &automorphisms, &canonical_extension
            );
            (*candidates)++;
#ifdef VERIFY_CANONICAL_PARENT
            int explicit_extension =
                canonical_first_column_parent(child, columns) == parent.key;
            if (canonical_extension && !explicit_extension) {
                fprintf(stderr, "canonical extension mismatch child=%016llx "
                        "parent=%016llx canonical_extension=%d explicit=%d "
                        "deleted_parent=%016llx\n",
                        (unsigned long long)child,
                        (unsigned long long)parent.key, canonical_extension,
                        explicit_extension,
                        (unsigned long long)canonical_first_column_parent(
                            child, columns));
                exit(1);
            }
#endif
            if (!canonical_extension) continue;
            if (!automorphisms || group_order % automorphisms) exit(1);
            (*accepted_paths)++;
            orbit_map_set_exact(&origins, child, parent.key);
            orbit_map_set_exact(&children, child, group_order / automorphisms);
        }
    }
    if (children.count != origins.count) exit(1);
    free(origins.entries);
    return children;
}
#endif

static void extend_parent_record(OrbitMap* children, OrbitRecord parent,
                                 int parent_columns, uint64_t* candidates) {
    RowPattern rows[ROWS];
    unpack_rows(parent.key, rows);
    for (unsigned assignment = 0; assignment < NEW_COLUMNS; assignment++) {
        RowPattern child_rows[ROWS];
        for (int row = 0; row < ROWS; row++) {
            child_rows[row] = rows[row] |
                              (RowPattern)(((assignment >> row) & 1U) << parent_columns);
        }
        uint64_t child_key = canonical_key(child_rows, parent_columns + 1);
        orbit_map_add(children, child_key, parent.weight);
        (*candidates)++;
    }
}

static U128 map_weight_sum(const OrbitMap* map) {
    U128 total = 0;
    for (size_t i = 0; i < map->capacity; i++) {
        if (map->entries[i].key != UINT64_MAX) total += map->entries[i].weight;
    }
    return total;
}

static int record_compare(const void* lhs_ptr, const void* rhs_ptr) {
    const OrbitRecord* lhs = lhs_ptr;
    const OrbitRecord* rhs = rhs_ptr;
    return lhs->key < rhs->key ? -1 : lhs->key > rhs->key;
}

static OrbitRecord* map_records_sorted(const OrbitMap* map) {
    OrbitRecord* records = xcalloc(map->count, sizeof(records[0]));
    size_t count = 0;
    for (size_t i = 0; i < map->capacity; i++) {
        if (map->entries[i].key != UINT64_MAX) records[count++] = map->entries[i];
    }
    qsort(records, count, sizeof(records[0]), record_compare);
    return records;
}

static void write_orbit_file(const char* path, int columns, const OrbitMap* map) {
    FILE* file = fopen(path, "wb");
    if (!file) {
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    OrbitRecord* records = map_records_sorted(map);
    const char magic[8] = ORBIT_MAGIC;
    uint32_t columns_value = (uint32_t)columns;
    uint64_t count = map->count;
    if (fwrite(magic, sizeof(magic), 1, file) != 1 ||
        fwrite(&columns_value, sizeof(columns_value), 1, file) != 1 ||
        fwrite(&count, sizeof(count), 1, file) != 1 ||
        fwrite(records, sizeof(records[0]), map->count, file) != map->count) {
        fprintf(stderr, "failed writing %s\n", path);
        exit(1);
    }
    if (fclose(file) != 0) {
        fprintf(stderr, "failed closing %s\n", path);
        exit(1);
    }
    free(records);
}

static void write_orbit_file_unsorted(const char* path, int columns,
                                      const OrbitMap* map) {
    FILE* file = fopen(path, "wb");
    if (!file) {
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    const char magic[8] = ORBIT_MAGIC;
    uint32_t columns_value = (uint32_t)columns;
    uint64_t count = map->count;
    if (fwrite(magic, sizeof(magic), 1, file) != 1 ||
        fwrite(&columns_value, sizeof(columns_value), 1, file) != 1 ||
        fwrite(&count, sizeof(count), 1, file) != 1) exit(1);
    for (size_t slot = 0; slot < map->capacity; slot++) {
        OrbitRecord record = map->entries[slot];
        if (record.key != UINT64_MAX &&
            fwrite(&record, sizeof(record), 1, file) != 1) exit(1);
    }
    if (fclose(file) != 0) exit(1);
}

#if ORBIT_ROWS == 7 && ORBIT_MAX_COLUMNS == 9 && ORBIT_ROW_BITS == 9
static void unpack_7x8_rows(uint64_t key, RowPattern rows[7]) {
    for (int row = 6; row >= 0; row--) {
        rows[row] = (RowPattern)(key & UINT64_C(0xff));
        key >>= 8;
    }
}

static void run_sample_extend_7x8(const char* parent_path, size_t wanted,
                                  const char* output_path) {
    if (!wanted || wanted > SIZE_MAX / (NEW_COLUMNS * 2U)) exit(2);
    FILE* input = fopen(parent_path, "rb");
    if (!input) {
        fprintf(stderr, "cannot open %s: %s\n", parent_path, strerror(errno));
        exit(1);
    }
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    if (fread(magic, sizeof(magic), 1, input) != 1 ||
        memcmp(magic, "R7ORB01", 7) != 0 ||
        fread(&columns, sizeof(columns), 1, input) != 1 ||
        fread(&count, sizeof(count), 1, input) != 1 || columns != 8 ||
        count != UINT64_C(508147108)) {
        fprintf(stderr, "invalid complete 8-bit-packed 7x8 corpus %s\n", parent_path);
        exit(1);
    }

    size_t initial_capacity = 16;
    while (initial_capacity < wanted * NEW_COLUMNS * 2U) initial_capacity <<= 1;
    OrbitMap children;
    orbit_map_init(&children, initial_capacity);
    uint64_t* indices = xcalloc(wanted, sizeof(indices[0]));
    size_t selected = 0;
    uint64_t state = UINT64_C(0x7265637437783961);
    uint64_t candidates = 0;
    U128 parent_weight = 0;
    double start = seconds_now();
    while (selected < wanted) {
        state = mix64(state + UINT64_C(0x9e3779b97f4a7c15));
        uint64_t index = state % count;
        int duplicate = 0;
        for (size_t previous = 0; previous < selected; previous++) {
            if (indices[previous] == index) {
                duplicate = 1;
                break;
            }
        }
        if (duplicate) continue;
        indices[selected++] = index;
        if (fseeko(input, (off_t)(20 + index * sizeof(OrbitRecord)), SEEK_SET) != 0) {
            fprintf(stderr, "seek failed in %s\n", parent_path);
            exit(1);
        }
        OrbitRecord parent;
        if (fread(&parent, sizeof(parent), 1, input) != 1) {
            fprintf(stderr, "random parent read failed in %s\n", parent_path);
            exit(1);
        }
        parent_weight += parent.weight;
        RowPattern rows[7];
        unpack_7x8_rows(parent.key, rows);
        for (unsigned assignment = 0; assignment < NEW_COLUMNS; assignment++) {
            RowPattern child_rows[7];
            for (int row = 0; row < 7; row++) {
                child_rows[row] = rows[row] |
                                  (RowPattern)(((assignment >> row) & 1U) << 8);
            }
            orbit_map_add(&children, canonical_key(child_rows, 9), parent.weight);
            candidates++;
        }
    }
    fclose(input);
    free(indices);
    U128 child_weight = map_weight_sum(&children);
    if (candidates != wanted * NEW_COLUMNS ||
        child_weight != parent_weight * NEW_COLUMNS) {
        fprintf(stderr, "sample extension weight/candidate validation failed\n");
        exit(1);
    }
    write_orbit_file(output_path, 9, &children);
    printf("sample_extend_7x8 parents=%zu candidates=%llu local_unique=%zu "
           "parent_weight=", wanted, (unsigned long long)candidates, children.count);
    print_u128(parent_weight);
    printf(" child_weight=");
    print_u128(child_weight);
    printf(" seconds=%.3f output=%s OK\n", seconds_now() - start, output_path);
    free(children.entries);
}
#endif

static OrbitMap read_orbit_file(const char* path, int* columns) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    char magic[8];
    uint32_t columns_value = 0;
    uint64_t count = 0;
    if (fread(magic, sizeof(magic), 1, file) != 1 ||
        memcmp(magic, ORBIT_MAGIC, 7) != 0 ||
        fread(&columns_value, sizeof(columns_value), 1, file) != 1 ||
        fread(&count, sizeof(count), 1, file) != 1 || columns_value > MAX_COLUMNS) {
        fprintf(stderr, "invalid orbit file %s\n", path);
        exit(1);
    }
    size_t capacity = 16;
    while (capacity < (size_t)count * 2U) capacity <<= 1;
    OrbitMap map;
    orbit_map_init(&map, capacity);
    for (uint64_t i = 0; i < count; i++) {
        OrbitRecord record;
        if (fread(&record, sizeof(record), 1, file) != 1) {
            fprintf(stderr, "truncated orbit file %s\n", path);
            exit(1);
        }
        orbit_map_add(&map, record.key, record.weight);
    }
    if (fgetc(file) != EOF) {
        fprintf(stderr, "trailing data in orbit file %s\n", path);
        exit(1);
    }
    fclose(file);
    *columns = (int)columns_value;
    return map;
}

static void validate_map(const OrbitMap* map, int columns) {
    U128 weight = map_weight_sum(map);
    U128 expected_weight = (U128)1 << (ROWS * columns);
    printf("columns=%d orbits=%zu expected_orbits=%llu labelled_weight=", columns,
           map->count, (unsigned long long)g_expected_counts[columns]);
    print_u128(weight);
    printf(" expected_weight=");
    print_u128(expected_weight);
    printf(" %s\n", map->count == g_expected_counts[columns] && weight == expected_weight ?
                       "OK" : "FAIL");
    if (map->count != g_expected_counts[columns] || weight != expected_weight) exit(1);
}

static void run_build(int target_columns, const char* output_path) {
    OrbitMap current;
    orbit_map_init(&current, 16);
    orbit_map_add(&current, 0, 1);
    validate_map(&current, 0);
    for (int columns = 1; columns <= target_columns; columns++) {
        uint64_t candidates = 0;
        double start = seconds_now();
        OrbitMap next = extend_map(&current, columns - 1, 0, current.count, &candidates);
        double seconds = seconds_now() - start;
        free(current.entries);
        current = next;
        validate_map(&current, columns);
        printf("augmentation columns=%d candidates=%llu time=%.3fs rate=%.0f/s\n", columns,
               (unsigned long long)candidates, seconds, (double)candidates / seconds);
    }
    if (output_path) {
        write_orbit_file(output_path, target_columns, &current);
        printf("wrote=%s records=%zu\n", output_path, current.count);
    }
    free(current.entries);
}

static void run_augment(const char* parent_path, const char* output_path) {
    int parent_columns = 0;
    OrbitMap parents = read_orbit_file(parent_path, &parent_columns);
    if (parent_columns >= MAX_COLUMNS) exit(2);
    uint64_t candidates = 0;
    double begin = seconds_now();
    OrbitMap children = extend_map(
        &parents, parent_columns, 0, parents.count, &candidates
    );
    free(parents.entries);
    validate_map(&children, parent_columns + 1);
    double augmentation_seconds = seconds_now() - begin;
    write_orbit_file(output_path, parent_columns + 1, &children);
    printf("augment columns=%d candidates=%llu unique=%zu "
           "augmentation_seconds=%.3f total_seconds=%.3f output=%s OK\n",
           parent_columns + 1, (unsigned long long)candidates, children.count,
           augmentation_seconds, seconds_now() - begin, output_path);
    free(children.entries);
}

#if ORBIT_ROWS == 6 && ORBIT_MAX_COLUMNS == 10 && ORBIT_ROW_BITS == 10
static void run_augment_solve_6x10(const char* parent_path,
                                   const char* output_path) {
    validate_concurrent_orbit_map();
    int parent_columns = 0;
    OrbitMap parents = read_orbit_file(parent_path, &parent_columns);
    if (parent_columns != 9) exit(2);
    validate_map(&parents, parent_columns);
    size_t capacity = 16;
    while ((U128)capacity * 7U <= (U128)UINT64_C(502732239) * 10U)
        capacity <<= 1;
    OrbitMap children;
    orbit_map_init(&children, capacity);
    uint64_t raw_candidates = 0;
    uint64_t retained_candidates = 0;
    uint64_t unique_children = 0;
    double begin = seconds_now();
#pragma omp parallel for schedule(dynamic, 256) \
    reduction(+:raw_candidates,retained_candidates,unique_children)
    for (long long parent_slot = 0;
         parent_slot < (long long)parents.capacity; parent_slot++) {
        OrbitRecord parent = parents.entries[(size_t)parent_slot];
        if (parent.key == UINT64_MAX) continue;
        int parent_cells = __builtin_popcountll(parent.key);
        RowPattern rows[ROWS];
        unpack_rows(parent.key, rows);
        for (unsigned assignment = 0; assignment < NEW_COLUMNS; assignment++) {
            raw_candidates++;
            if (parent_cells + __builtin_popcount(assignment) > 30) continue;
            retained_candidates++;
            RowPattern child_rows[ROWS];
            for (int row = 0; row < ROWS; row++) {
                child_rows[row] = rows[row] |
                    (RowPattern)(((assignment >> row) & 1U) << 9);
            }
            unique_children += orbit_map_add_concurrent(
                &children, canonical_key(child_rows, 10), parent.weight
            );
        }
    }
    children.count = (size_t)unique_children;
    free(parents.entries);
    U128 retained_weight = 0;
    U128 covered_weight = 0;
    uint64_t midpoint = 0;
    for (size_t slot = 0; slot < children.capacity; slot++) {
        OrbitRecord child = children.entries[slot];
        if (child.key == UINT64_MAX) continue;
        int cells = __builtin_popcountll(child.key);
        if (cells > 30) exit(1);
        retained_weight += child.weight;
        covered_weight += (U128)child.weight * (cells < 30 ? 2U : 1U);
        midpoint += cells == 30;
    }
    const int valid = raw_candidates == UINT64_C(8335217152) &&
                      retained_candidates == UINT64_C(4569464882) &&
                      children.count == UINT64_C(502732239) &&
                      retained_weight == (U128)UINT64_C(635593043085854200) &&
                      covered_weight == ((U128)1 << 60);
    if (!valid) {
        fprintf(stderr, "retained 6x10 augmentation validation failed\n");
        exit(1);
    }
    double augmentation_seconds = seconds_now() - begin;
    write_orbit_file(output_path, 10, &children);
    printf("augment_solve_6x10 raw_candidates=%llu retained_candidates=%llu "
           "unique=%zu midpoint=%llu retained_weight=",
           (unsigned long long)raw_candidates,
           (unsigned long long)retained_candidates, children.count,
           (unsigned long long)midpoint);
    print_u128(retained_weight);
    printf(" covered_weight=");
    print_u128(covered_weight);
    printf(" augmentation_seconds=%.3f total_seconds=%.3f output=%s OK\n",
           augmentation_seconds, seconds_now() - begin, output_path);
    free(children.entries);
}
#endif

#if ORBIT_ROWS == 8 && ORBIT_MAX_COLUMNS == 8 && ORBIT_ROW_BITS == 8
static void run_build_canonical_parent(int target_columns, const char* output_path) {
    OrbitMap current;
    orbit_map_init(&current, 16);
    orbit_map_add(&current, 0, 1);
    validate_map(&current, 0);
    for (int columns = 1; columns <= target_columns; columns++) {
        uint64_t candidates = 0;
        uint64_t accepted_paths = 0;
        double start = seconds_now();
        OrbitMap next = extend_map_canonical_parent(
            &current, columns - 1, &candidates, &accepted_paths
        );
        double seconds = seconds_now() - start;
        free(current.entries);
        current = next;
        validate_map(&current, columns);
        printf("canonical_parent columns=%d candidates=%llu accepted_paths=%llu "
               "unique=%zu time=%.3fs rate=%.0f/s\n", columns,
               (unsigned long long)candidates,
               (unsigned long long)accepted_paths, current.count, seconds,
               (double)candidates / seconds);
    }
    if (output_path) {
        write_orbit_file(output_path, target_columns, &current);
        printf("wrote=%s records=%zu\n", output_path, current.count);
    }
    free(current.entries);
}
#endif

static void write_bucket_records(const OrbitMap* map, int bucket_count,
                                 const char* prefix) {
    FILE** files = xcalloc((size_t)bucket_count, sizeof(files[0]));
    char path[4096];
    for (int bucket = 0; bucket < bucket_count; bucket++) {
        snprintf(path, sizeof(path), "%s.b%03d", prefix, bucket);
        files[bucket] = fopen(path, "wb");
        if (!files[bucket]) {
            fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
            exit(1);
        }
    }
    for (size_t i = 0; i < map->capacity; i++) {
        OrbitRecord record = map->entries[i];
        if (record.key == UINT64_MAX) continue;
        int bucket = (int)(mix64(record.key) % (uint64_t)bucket_count);
        if (fwrite(&record, sizeof(record), 1, files[bucket]) != 1) exit(1);
    }
    for (int bucket = 0; bucket < bucket_count; bucket++) {
        if (fclose(files[bucket]) != 0) exit(1);
    }
    free(files);
}

#if ORBIT_ROWS == 7 && ORBIT_MAX_COLUMNS == 9 && ORBIT_ROW_BITS == 9
static uint64_t retained_assignment_count_7x9(int parent_cells) {
    static const uint8_t binomial_prefix[8] = {1, 8, 29, 64, 99, 120, 127, 128};
    if (parent_cells > 31) return 0;
    int remaining = 31 - parent_cells;
    return binomial_prefix[remaining < 7 ? remaining : 7];
}

static void run_solve_plan_7x9(const char* parent_path, int ranges) {
    if (ranges < 1 || ranges > 9999) exit(2);
    FILE* input = fopen(parent_path, "rb");
    if (!input) exit(1);
    char magic[8];
    uint32_t columns = 0;
    uint64_t parent_count = 0;
    if (fread(magic, sizeof(magic), 1, input) != 1 ||
        memcmp(magic, "R7ORB01", 7) != 0 ||
        fread(&columns, sizeof(columns), 1, input) != 1 ||
        fread(&parent_count, sizeof(parent_count), 1, input) != 1 ||
        columns != 8 || parent_count != UINT64_C(508147108)) exit(1);

    uint64_t total_work = 0;
    uint64_t active_parents = 0;
    U128 labelled_weight = 0;
    const uint64_t expected_active_parents = UINT64_C(413704224);
    const uint64_t expected_work = UINT64_C(32521414912);
    double census_begin = seconds_now();
    for (uint64_t index = 0; index < parent_count; index++) {
        OrbitRecord parent;
        if (fread(&parent, sizeof(parent), 1, input) != 1) exit(1);
        uint64_t work =
            retained_assignment_count_7x9(__builtin_popcountll(parent.key));
        total_work += work;
        active_parents += work != 0;
        labelled_weight += (U128)parent.weight * work;
    }
    if (fgetc(input) != EOF || active_parents != expected_active_parents ||
        total_work != expected_work || labelled_weight != ((U128)1 << 62) ||
        fseeko(input, 20, SEEK_SET) != 0) exit(1);
    double census_seconds = seconds_now() - census_begin;

    uint64_t cumulative = 0;
    uint64_t range_start = 0;
    uint64_t range_work = 0;
    int range = 0;
    double plan_begin = seconds_now();
    for (uint64_t index = 0; index < parent_count; index++) {
        OrbitRecord parent;
        if (fread(&parent, sizeof(parent), 1, input) != 1) exit(1);
        uint64_t work =
            retained_assignment_count_7x9(__builtin_popcountll(parent.key));
        cumulative += work;
        range_work += work;
        if (range + 1 < ranges &&
            (U128)cumulative * (uint64_t)ranges >=
                (U128)total_work * (uint64_t)(range + 1)) {
            printf("solve_plan_7x9 range=%d start=%llu end=%llu work=%llu\n",
                   range, (unsigned long long)range_start,
                   (unsigned long long)(index + 1),
                   (unsigned long long)range_work);
            range++;
            range_start = index + 1;
            range_work = 0;
        }
    }
    if (fgetc(input) != EOF || fclose(input) != 0 || cumulative != total_work ||
        range != ranges - 1) exit(1);
    printf("solve_plan_7x9 range=%d start=%llu end=%llu work=%llu\n", range,
           (unsigned long long)range_start, (unsigned long long)parent_count,
           (unsigned long long)range_work);
    printf("solve_plan_7x9 records=%llu active_parents=%llu work=%llu "
           "labelled_weight=",
           (unsigned long long)parent_count,
           (unsigned long long)active_parents,
           (unsigned long long)total_work);
    print_u128(labelled_weight);
    printf(" ranges=%d census_seconds=%.3f plan_seconds=%.3f OK\n", ranges,
           census_seconds, seconds_now() - plan_begin);
}

static void run_extend_7x8(const char* parent_path, uint64_t start, uint64_t end,
                           int bucket_count, const char* output_prefix,
                           int retained_only) {
    if (start > end || bucket_count < 1 || bucket_count > 999 ||
        end - start > SIZE_MAX / (NEW_COLUMNS * 2U)) exit(2);
    FILE* input = fopen(parent_path, "rb");
    if (!input) exit(1);
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    if (fread(magic, sizeof(magic), 1, input) != 1 ||
        memcmp(magic, "R7ORB01", 7) != 0 ||
        fread(&columns, sizeof(columns), 1, input) != 1 ||
        fread(&count, sizeof(count), 1, input) != 1 || columns != 8 ||
        count != UINT64_C(508147108) || end > count ||
        fseeko(input, (off_t)(20 + start * sizeof(OrbitRecord)), SEEK_SET) != 0) {
        fprintf(stderr, "invalid complete 7x8 extension input or range\n");
        exit(1);
    }
    double begin = seconds_now();
    uint64_t planned_work = (end - start) * NEW_COLUMNS;
    if (retained_only) {
        planned_work = 0;
        for (uint64_t index = start; index < end; index++) {
            OrbitRecord parent;
            if (fread(&parent, sizeof(parent), 1, input) != 1) exit(1);
            planned_work += retained_assignment_count_7x9(
                __builtin_popcountll(parent.key));
        }
        if (fseeko(input, (off_t)(20 + start * sizeof(OrbitRecord)), SEEK_SET) !=
            0) exit(1);
    }
    size_t initial_capacity = 16;
    while ((U128)initial_capacity * 7U <= (U128)planned_work * 10U) {
        initial_capacity <<= 1;
    }
    OrbitMap children;
    orbit_map_init(&children, initial_capacity);
    U128 parent_weight = 0;
    uint64_t candidates = 0;
    uint64_t retained_candidates = 0;
    for (uint64_t index = start; index < end; index++) {
        OrbitRecord parent;
        if (fread(&parent, sizeof(parent), 1, input) != 1) exit(1);
        parent_weight += parent.weight;
        RowPattern rows[7];
        unpack_7x8_rows(parent.key, rows);
        int parent_cells = 0;
        for (int row = 0; row < 7; row++) {
            parent_cells += __builtin_popcount((unsigned)rows[row]);
        }
        for (unsigned assignment = 0; assignment < NEW_COLUMNS; assignment++) {
            candidates++;
            if (retained_only &&
                parent_cells + __builtin_popcount(assignment) > 31) continue;
            retained_candidates++;
            RowPattern child_rows[7];
            for (int row = 0; row < 7; row++) {
                child_rows[row] = rows[row] |
                                  (RowPattern)(((assignment >> row) & 1U) << 8);
            }
            orbit_map_add(&children, canonical_key(child_rows, 9), parent.weight);
        }
    }
    if (fclose(input) != 0) exit(1);
    write_bucket_records(&children, bucket_count, output_prefix);
    U128 child_weight = map_weight_sum(&children);
    if (candidates != (end - start) * NEW_COLUMNS ||
        (retained_only && retained_candidates != planned_work) ||
        (!retained_only && child_weight != parent_weight * NEW_COLUMNS)) exit(1);
    printf("extend_7x8 parent_range=[%llu,%llu) parents=%llu candidates=%llu "
           "retained_candidates=%llu planned_work=%llu local_unique=%zu "
           "parent_weight=",
           (unsigned long long)start, (unsigned long long)end,
           (unsigned long long)(end - start),
           (unsigned long long)candidates,
           (unsigned long long)retained_candidates,
           (unsigned long long)planned_work, children.count);
    print_u128(parent_weight);
    printf(" child_weight=");
    print_u128(child_weight);
    printf(" buckets=%d retained_only=%d seconds=%.3f prefix=%s OK\n",
           bucket_count, retained_only, seconds_now() - begin, output_prefix);
    free(children.entries);
}
#endif

static void run_extend(const char* parent_path, size_t start, size_t end,
                       int bucket_count, const char* prefix) {
    int parent_columns = 0;
    OrbitMap parents = read_orbit_file(parent_path, &parent_columns);
    if (parent_columns >= MAX_COLUMNS || start > end || end > parents.count ||
        bucket_count < 1 || bucket_count > 999) {
        fprintf(stderr, "invalid extend range or bucket count\n");
        exit(2);
    }
    uint64_t candidates = 0;
    double begin = seconds_now();
    OrbitMap children = extend_map(&parents, parent_columns, start, end, &candidates);
    double seconds = seconds_now() - begin;
    write_bucket_records(&children, bucket_count, prefix);
    printf("extend columns=%d parent_range=[%zu,%zu) parents=%zu candidates=%llu local_unique=%zu local_weight=",
           parent_columns + 1, start, end, end - start, (unsigned long long)candidates,
           children.count);
    print_u128(map_weight_sum(&children));
    printf(" buckets=%d time=%.3fs\n", bucket_count, seconds);
    free(children.entries);
    free(parents.entries);
}

static void run_sample_extend(const char* parent_path, uint64_t sample_mod,
                              uint64_t sample_id, const char* output_path) {
    if (!sample_mod || sample_id >= sample_mod) exit(2);
    FILE* input = fopen(parent_path, "rb");
    if (!input) exit(1);
    char magic[8];
    uint32_t parent_columns = 0;
    uint64_t parent_count = 0;
    if (fread(magic, sizeof(magic), 1, input) != 1 ||
        memcmp(magic, ORBIT_MAGIC, 7) != 0 ||
        fread(&parent_columns, sizeof(parent_columns), 1, input) != 1 ||
        fread(&parent_count, sizeof(parent_count), 1, input) != 1 ||
        parent_columns >= MAX_COLUMNS) {
        fprintf(stderr, "invalid parent orbit file %s\n", parent_path);
        exit(1);
    }
    size_t expected_parents = (size_t)(parent_count / sample_mod + 1);
    if (expected_parents > SIZE_MAX / (NEW_COLUMNS * 2U)) {
        fprintf(stderr, "sample is too large for an in-memory child map\n");
        exit(2);
    }
    size_t initial_capacity = 16;
    while (initial_capacity < expected_parents * NEW_COLUMNS * 2U) {
        initial_capacity <<= 1;
    }
    OrbitMap children;
    orbit_map_init(&children, initial_capacity);
    uint64_t selected_parents = 0;
    uint64_t candidates = 0;
    U128 selected_parent_weight = 0;
    double start = seconds_now();
    for (uint64_t index = 0; index < parent_count; index++) {
        OrbitRecord parent;
        if (fread(&parent, sizeof(parent), 1, input) != 1) exit(1);
        if (mix64(parent.key) % sample_mod != sample_id) continue;
        selected_parents++;
        selected_parent_weight += parent.weight;
        extend_parent_record(&children, parent, (int)parent_columns, &candidates);
    }
    if (fgetc(input) != EOF || fclose(input) != 0) exit(1);
    write_orbit_file(output_path, (int)parent_columns + 1, &children);
    printf("sample_extend columns=%u sample=%llu/%llu scanned_parents=%llu "
           "selected_parents=%llu candidates=%llu local_unique=%zu "
           "selected_parent_weight=",
           parent_columns + 1, (unsigned long long)sample_id,
           (unsigned long long)sample_mod, (unsigned long long)parent_count,
           (unsigned long long)selected_parents, (unsigned long long)candidates,
           children.count);
    print_u128(selected_parent_weight);
    printf(" child_weight=");
    print_u128(map_weight_sum(&children));
    printf(" seconds=%.3f output=%s\n", seconds_now() - start, output_path);
    free(children.entries);
}

static void run_reduce(int columns, const char* output_path, int input_count,
                       char** input_paths) {
    uint64_t records = 0;
    for (int input = 0; input < input_count; input++) {
        struct stat status;
        if (stat(input_paths[input], &status) != 0 ||
            status.st_size % (off_t)sizeof(OrbitRecord) != 0) {
            fprintf(stderr, "invalid bucket file %s\n", input_paths[input]);
            exit(1);
        }
        records += (uint64_t)(status.st_size / (off_t)sizeof(OrbitRecord));
    }
    size_t capacity = 16;
    while (capacity < (size_t)records * 2U) capacity <<= 1;
    OrbitMap reduced;
    orbit_map_init(&reduced, capacity);
    for (int input = 0; input < input_count; input++) {
        FILE* file = fopen(input_paths[input], "rb");
        if (!file) exit(1);
        OrbitRecord record;
        while (fread(&record, sizeof(record), 1, file) == 1) {
            orbit_map_add(&reduced, record.key, record.weight);
        }
        if (!feof(file)) exit(1);
        fclose(file);
    }
    write_orbit_file(output_path, columns, &reduced);
    printf("reduce columns=%d inputs=%d records=%llu unique=%zu weight=", columns,
           input_count, (unsigned long long)records, reduced.count);
    print_u128(map_weight_sum(&reduced));
    printf(" output=%s\n", output_path);
    free(reduced.entries);
}

static void write_orbit_header(FILE* file, int columns, uint64_t count) {
    const char magic[8] = ORBIT_MAGIC;
    uint32_t columns_value = (uint32_t)columns;
    if (fwrite(magic, sizeof(magic), 1, file) != 1 ||
        fwrite(&columns_value, sizeof(columns_value), 1, file) != 1 ||
        fwrite(&count, sizeof(count), 1, file) != 1) exit(1);
}

#if ORBIT_ROWS == 6 && ORBIT_MAX_COLUMNS == 10 && ORBIT_ROW_BITS == 10
static uint32_t solve_left_prefix_6x10(uint64_t key) {
    RowPattern rows[6];
    unpack_rows(key, rows);
    uint32_t prefix = 0;
    for (int row = 0; row < 6; row++) {
        prefix = (prefix << 5) | (rows[row] & 31U);
    }
    return prefix;
}

static int solve_shard_6x10(uint64_t key, int shards) {
    return (int)(mix64(solve_left_prefix_6x10(key)) % (uint64_t)shards);
}

static void run_solve_census_6x10(const char* parent_path) {
    static const uint8_t binomial[7] = {1, 6, 15, 20, 15, 6, 1};
    static const uint8_t binomial_prefix[7] = {1, 7, 22, 42, 57, 63, 64};
    FILE* input = fopen(parent_path, "rb");
    if (!input) exit(1);
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    if (fread(magic, sizeof(magic), 1, input) != 1 ||
        memcmp(magic, ORBIT_MAGIC, 7) != 0 ||
        fread(&columns, sizeof(columns), 1, input) != 1 ||
        fread(&count, sizeof(count), 1, input) != 1 || columns != 9 ||
        count != UINT64_C(130237768)) exit(1);
    uint64_t retained_candidates = 0;
    U128 retained_weight = 0;
    U128 midpoint_weight = 0;
    double begin = seconds_now();
    for (uint64_t index = 0; index < count; index++) {
        OrbitRecord parent;
        if (fread(&parent, sizeof(parent), 1, input) != 1) exit(1);
        int remaining = 30 - __builtin_popcountll(parent.key);
        if (remaining < 0) continue;
        unsigned maximum = remaining < 6 ? (unsigned)remaining : 6U;
        retained_candidates += binomial_prefix[maximum];
        retained_weight += (U128)parent.weight * binomial_prefix[maximum];
        if (remaining <= 6)
            midpoint_weight += (U128)parent.weight * binomial[remaining];
    }
    if (fgetc(input) != EOF || fclose(input) != 0) exit(1);
    U128 covered_weight = retained_weight * 2U - midpoint_weight;
    const int valid = covered_weight == ((U128)1 << 60);
    printf("solve_census_6x10 parents=%llu raw_candidates=%llu "
           "retained_candidates=%llu retained_weight=",
           (unsigned long long)count,
           (unsigned long long)(count * UINT64_C(64)),
           (unsigned long long)retained_candidates);
    print_u128(retained_weight);
    printf(" midpoint_weight=");
    print_u128(midpoint_weight);
    printf(" covered_weight=");
    print_u128(covered_weight);
    printf(" seconds=%.3f %s\n", seconds_now() - begin,
           valid ? "OK" : "FAIL");
    if (!valid) exit(1);
}

static void raise_file_limit(int required) {
    struct rlimit limit;
    if (getrlimit(RLIMIT_NOFILE, &limit) != 0) exit(1);
    rlim_t wanted = (rlim_t)required;
    if (limit.rlim_cur >= wanted) return;
    if (limit.rlim_max < wanted) {
        fprintf(stderr, "open-file limit is too small for %d solve shards\n",
                required - 1);
        exit(1);
    }
    limit.rlim_cur = wanted;
    if (setrlimit(RLIMIT_NOFILE, &limit) != 0) exit(1);
}

static void run_solve_partition_6x10(const char* input_path, int shards,
                                     const char* output_prefix) {
    if (shards < 1 || shards > 8192) exit(2);
    raise_file_limit(shards + 16);
    FILE* input = fopen(input_path, "rb");
    if (!input) exit(1);
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    if (fread(magic, sizeof(magic), 1, input) != 1 ||
        memcmp(magic, ORBIT_MAGIC, 7) != 0 ||
        fread(&columns, sizeof(columns), 1, input) != 1 ||
        fread(&count, sizeof(count), 1, input) != 1 || columns != 10 ||
        !count) exit(1);
    FILE** outputs = xcalloc((size_t)shards, sizeof(outputs[0]));
    uint64_t* shard_records = xcalloc((size_t)shards, sizeof(shard_records[0]));
    U128 labelled_weight = 0;
    U128 retained_weight = 0;
    U128 covered_weight = 0;
    uint64_t retained = 0;
    uint64_t midpoint = 0;
    char path[4096];
    for (int shard = 0; shard < shards; shard++) {
        snprintf(path, sizeof(path), "%s.s%04d.orbits", output_prefix, shard);
        outputs[shard] = fopen(path, "wb+");
        if (!outputs[shard]) exit(1);
        write_orbit_header(outputs[shard], 10, 0);
    }
    for (uint64_t index = 0; index < count; index++) {
        OrbitRecord record;
        if (fread(&record, sizeof(record), 1, input) != 1) exit(1);
        labelled_weight += record.weight;
        int cells = __builtin_popcountll(record.key);
        if (cells > 30) continue;
        int shard = solve_shard_6x10(record.key, shards);
        if (fwrite(&record, sizeof(record), 1, outputs[shard]) != 1) exit(1);
        shard_records[shard]++;
        retained++;
        retained_weight += record.weight;
        covered_weight += (U128)record.weight * (cells < 30 ? 2U : 1U);
        midpoint += cells == 30;
    }
    if (fgetc(input) != EOF || fclose(input) != 0) exit(1);
    uint64_t minimum_records = UINT64_MAX;
    uint64_t maximum_records = 0;
    for (int shard = 0; shard < shards; shard++) {
        if (fseeko(outputs[shard], 12, SEEK_SET) != 0 ||
            fwrite(&shard_records[shard], sizeof(shard_records[shard]), 1,
                   outputs[shard]) != 1 ||
            fclose(outputs[shard]) != 0) exit(1);
        if (shard_records[shard] < minimum_records)
            minimum_records = shard_records[shard];
        if (shard_records[shard] > maximum_records)
            maximum_records = shard_records[shard];
    }
    const int full = retained == UINT64_C(502732239);
    const int valid = !full || covered_weight == ((U128)1 << 60);
    printf("solve_partition_6x10 input=%s records=%llu retained=%llu "
           "midpoint=%llu shards=%d min_records=%llu max_records=%llu "
           "labelled_weight=", input_path, (unsigned long long)count,
           (unsigned long long)retained, (unsigned long long)midpoint, shards,
           (unsigned long long)minimum_records,
           (unsigned long long)maximum_records);
    print_u128(labelled_weight);
    printf(" retained_weight=");
    print_u128(retained_weight);
    printf(" covered_weight=");
    print_u128(covered_weight);
    printf(" prefix=%s full=%d %s\n", output_prefix, full,
           valid ? "OK" : "FAIL");
    free(shard_records);
    free(outputs);
    if (!valid) exit(1);
}

static void run_promote_half_seed_6x10(const char* input_path,
                                       const char* output_path) {
    FILE* input = fopen(input_path, "rb");
    FILE* output = fopen(output_path, "wb");
    if (!input || !output) exit(1);
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    if (fread(magic, sizeof(magic), 1, input) != 1 ||
        memcmp(magic, ORBIT_MAGIC, 7) != 0 ||
        fread(&columns, sizeof(columns), 1, input) != 1 ||
        fread(&count, sizeof(count), 1, input) != 1 || columns != 5 ||
        count != UINT64_C(28576)) exit(1);
    write_orbit_header(output, 10, count);
    OrbitRecord record;
    U128 weight = 0;
    for (uint64_t index = 0; index < count; index++) {
        if (fread(&record, sizeof(record), 1, input) != 1 ||
            fwrite(&record, sizeof(record), 1, output) != 1) exit(1);
        weight += record.weight;
    }
    if (fgetc(input) != EOF || fclose(input) != 0 || fclose(output) != 0 ||
        weight != ((U128)1 << 30)) exit(1);
    printf("promote_half_seed_6x10 records=%llu input_weight=",
           (unsigned long long)count);
    print_u128(weight);
    printf(" output=%s OK\n", output_path);
}

static void run_solve_check_6x10(int shards, int input_count,
                                 char** input_paths) {
    if (shards < 1 || shards != input_count) exit(2);
    uint64_t records = 0;
    U128 retained_weight = 0;
    U128 covered_weight = 0;
    for (int shard = 0; shard < shards; shard++) {
        FILE* input = fopen(input_paths[shard], "rb");
        if (!input) exit(1);
        char magic[8];
        uint32_t columns = 0;
        uint64_t count = 0;
        if (fread(magic, sizeof(magic), 1, input) != 1 ||
            memcmp(magic, ORBIT_MAGIC, 7) != 0 ||
            fread(&columns, sizeof(columns), 1, input) != 1 ||
            fread(&count, sizeof(count), 1, input) != 1 || columns != 10)
            exit(1);
        records += count;
        for (uint64_t index = 0; index < count; index++) {
            OrbitRecord record;
            if (fread(&record, sizeof(record), 1, input) != 1 ||
                __builtin_popcountll(record.key) > 30 ||
                solve_shard_6x10(record.key, shards) != shard) exit(1);
            int cells = __builtin_popcountll(record.key);
            retained_weight += record.weight;
            covered_weight += (U128)record.weight * (cells < 30 ? 2U : 1U);
        }
        if (fgetc(input) != EOF || fclose(input) != 0) exit(1);
    }
    const int full = records == UINT64_C(502732239);
    const int valid = !full || covered_weight == ((U128)1 << 60);
    printf("solve_check_6x10 shards=%d records=%llu retained_weight=", shards,
           (unsigned long long)records);
    print_u128(retained_weight);
    printf(" covered_weight=");
    print_u128(covered_weight);
    printf(" expected_covered_weight=");
    print_u128((U128)1 << 60);
    printf(" full=%d %s\n", full, valid ? "OK" : "FAIL");
    if (!valid) exit(1);
}
#endif

static void run_partition(const char* input_path, int bucket_count, const char* prefix) {
    if (bucket_count < 1 || bucket_count > 999) exit(2);
    FILE* input = fopen(input_path, "rb");
    if (!input) exit(1);
    char magic[8];
    uint32_t columns = 0;
    uint64_t input_count = 0;
    if (fread(magic, sizeof(magic), 1, input) != 1 ||
        memcmp(magic, ORBIT_MAGIC, 7) != 0 ||
        fread(&columns, sizeof(columns), 1, input) != 1 ||
        fread(&input_count, sizeof(input_count), 1, input) != 1) exit(1);
    FILE** outputs = xcalloc((size_t)bucket_count, sizeof(outputs[0]));
    uint64_t* counts = xcalloc((size_t)bucket_count, sizeof(counts[0]));
    U128* weights = xcalloc((size_t)bucket_count, sizeof(weights[0]));
    char path[4096];
    for (int bucket = 0; bucket < bucket_count; bucket++) {
        snprintf(path, sizeof(path), "%s.b%03d.orbits", prefix, bucket);
        outputs[bucket] = fopen(path, "wb+");
        if (!outputs[bucket]) exit(1);
        write_orbit_header(outputs[bucket], (int)columns, 0);
    }
    for (uint64_t index = 0; index < input_count; index++) {
        OrbitRecord record;
        if (fread(&record, sizeof(record), 1, input) != 1) exit(1);
        int bucket = (int)(mix64(record.key) % (uint64_t)bucket_count);
        if (fwrite(&record, sizeof(record), 1, outputs[bucket]) != 1) exit(1);
        counts[bucket]++;
        weights[bucket] += record.weight;
    }
    if (fgetc(input) != EOF) exit(1);
    fclose(input);
    uint64_t minimum = UINT64_MAX;
    uint64_t maximum = 0;
    U128 total_weight = 0;
    for (int bucket = 0; bucket < bucket_count; bucket++) {
        if (fseeko(outputs[bucket], 12, SEEK_SET) != 0 ||
            fwrite(&counts[bucket], sizeof(counts[bucket]), 1, outputs[bucket]) != 1 ||
            fclose(outputs[bucket]) != 0) exit(1);
        if (counts[bucket] < minimum) minimum = counts[bucket];
        if (counts[bucket] > maximum) maximum = counts[bucket];
        total_weight += weights[bucket];
    }
    printf("partition input=%s columns=%u records=%llu buckets=%d min=%llu max=%llu weight=",
           input_path, columns, (unsigned long long)input_count, bucket_count,
           (unsigned long long)minimum, (unsigned long long)maximum);
    print_u128(total_weight);
    printf(" prefix=%s\n", prefix);
    free(weights);
    free(counts);
    free(outputs);
}

static void run_slice(const char* input_path, uint64_t start, uint64_t end,
                      const char* output_path) {
    FILE* input = fopen(input_path, "rb");
    FILE* output = fopen(output_path, "wb");
    if (!input || !output) exit(1);
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    if (fread(magic, sizeof(magic), 1, input) != 1 ||
        memcmp(magic, ORBIT_MAGIC, 7) != 0 ||
        fread(&columns, sizeof(columns), 1, input) != 1 ||
        fread(&count, sizeof(count), 1, input) != 1 ||
        start > end || end > count) exit(1);
    if (fseeko(input, 20 + (off_t)start * (off_t)sizeof(OrbitRecord), SEEK_SET) != 0) {
        exit(1);
    }
    write_orbit_header(output, (int)columns, end - start);
    U128 weight = 0;
    for (uint64_t index = start; index < end; index++) {
        OrbitRecord record;
        if (fread(&record, sizeof(record), 1, input) != 1 ||
            fwrite(&record, sizeof(record), 1, output) != 1) exit(1);
        weight += record.weight;
    }
    if (fclose(input) != 0 || fclose(output) != 0) exit(1);
    printf("slice input=%s range=[%llu,%llu) records=%llu weight=",
           input_path, (unsigned long long)start, (unsigned long long)end,
           (unsigned long long)(end - start));
    print_u128(weight);
    printf(" output=%s\n", output_path);
}

static void run_combine(const char* output_path, int input_count, char** input_paths) {
    if (input_count < 1) exit(2);
    uint32_t columns = 0;
    uint64_t total_count = 0;
    for (int input = 0; input < input_count; input++) {
        FILE* file = fopen(input_paths[input], "rb");
        if (!file) exit(1);
        char magic[8];
        uint32_t input_columns = 0;
        uint64_t count = 0;
        if (fread(magic, sizeof(magic), 1, file) != 1 ||
            memcmp(magic, ORBIT_MAGIC, 7) != 0 ||
            fread(&input_columns, sizeof(input_columns), 1, file) != 1 ||
            fread(&count, sizeof(count), 1, file) != 1 ||
            (input && input_columns != columns)) exit(1);
        columns = input_columns;
        total_count += count;
        fclose(file);
    }
    FILE* output = fopen(output_path, "wb");
    if (!output) exit(1);
    write_orbit_header(output, (int)columns, total_count);
    unsigned char* buffer = xcalloc(1U << 20, 1);
    for (int input = 0; input < input_count; input++) {
        FILE* file = fopen(input_paths[input], "rb");
        if (!file || fseeko(file, 20, SEEK_SET) != 0) exit(1);
        size_t count;
        while ((count = fread(buffer, 1, 1U << 20, file)) != 0) {
            if (fwrite(buffer, 1, count, output) != count) exit(1);
        }
        if (!feof(file)) exit(1);
        fclose(file);
    }
    free(buffer);
    if (fclose(output) != 0) exit(1);
    printf("combine columns=%u inputs=%d records=%llu output=%s\n", columns,
           input_count, (unsigned long long)total_count, output_path);
}

static void run_check(const char* input_path) {
    FILE* file = fopen(input_path, "rb");
    if (!file) exit(1);
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    if (fread(magic, sizeof(magic), 1, file) != 1 ||
        memcmp(magic, ORBIT_MAGIC, 7) != 0 ||
        fread(&columns, sizeof(columns), 1, file) != 1 ||
        fread(&count, sizeof(count), 1, file) != 1 || columns > MAX_COLUMNS) exit(1);
    U128 weight = 0;
    for (uint64_t index = 0; index < count; index++) {
        OrbitRecord record;
        if (fread(&record, sizeof(record), 1, file) != 1) exit(1);
        weight += record.weight;
    }
    if (fgetc(file) != EOF) exit(1);
    fclose(file);
    U128 expected_weight = (U128)1 << (ROWS * columns);
    int valid = count == g_expected_counts[columns] && weight == expected_weight;
    printf("check input=%s columns=%u records=%llu expected_records=%llu weight=",
           input_path, columns, (unsigned long long)count,
           (unsigned long long)g_expected_counts[columns]);
    print_u128(weight);
    printf(" expected_weight=");
    print_u128(expected_weight);
    printf(" %s\n", valid ? "OK" : "FAIL");
    if (!valid) exit(1);
}

#if ORBIT_ROWS == 7 && ORBIT_MAX_COLUMNS == 9 && ORBIT_ROW_BITS == 9
static uint32_t solve_left_prefix_7x9(uint64_t key) {
    RowPattern rows[ROWS];
    unpack_rows(key, rows);
    uint32_t prefix = 0;
    for (int row = 0; row < ROWS; row++) {
        prefix = (prefix << 4) | (rows[row] & 15U);
    }
    return prefix;
}

static int solve_shard_7x9(uint64_t key, int shards) {
    return (int)(mix64(solve_left_prefix_7x9(key)) % (uint64_t)shards);
}

static void run_reduce_solve_7x9(int columns, int shards,
                                 const char* output_prefix, int input_count,
                                 char** input_paths) {
    if (columns != 9 || shards < 1 || shards > 9999 || input_count < 1) exit(2);
    uint64_t raw_records = 0;
    for (int input = 0; input < input_count; input++) {
        struct stat status;
        if (stat(input_paths[input], &status) != 0 ||
            status.st_size % (off_t)sizeof(OrbitRecord) != 0) exit(1);
        raw_records += (uint64_t)(status.st_size / (off_t)sizeof(OrbitRecord));
    }
    size_t capacity = 16;
    while (capacity < (size_t)raw_records * 2U) capacity <<= 1;
    OrbitMap reduced;
    orbit_map_init(&reduced, capacity);
    for (int input = 0; input < input_count; input++) {
        FILE* file = fopen(input_paths[input], "rb");
        if (!file) exit(1);
        OrbitRecord record;
        while (fread(&record, sizeof(record), 1, file) == 1) {
            orbit_map_add(&reduced, record.key, record.weight);
        }
        if (!feof(file) || fclose(file) != 0) exit(1);
    }

    FILE** outputs = xcalloc((size_t)shards, sizeof(outputs[0]));
    uint64_t* shard_records = xcalloc((size_t)shards, sizeof(shard_records[0]));
    char path[4096];
    for (int shard = 0; shard < shards; shard++) {
        snprintf(path, sizeof(path), "%s.s%04d", output_prefix, shard);
        outputs[shard] = fopen(path, "wb");
        if (!outputs[shard]) exit(1);
    }
    uint64_t retained = 0;
    U128 covered_weight = 0;
    for (size_t slot = 0; slot < reduced.capacity; slot++) {
        OrbitRecord record = reduced.entries[slot];
        if (record.key == UINT64_MAX || __builtin_popcountll(record.key) > 31) continue;
        int shard = solve_shard_7x9(record.key, shards);
        if (fwrite(&record, sizeof(record), 1, outputs[shard]) != 1) exit(1);
        shard_records[shard]++;
        retained++;
        covered_weight += (U128)record.weight * 2U;
    }
    uint64_t minimum_records = UINT64_MAX;
    uint64_t maximum_records = 0;
    for (int shard = 0; shard < shards; shard++) {
        if (fclose(outputs[shard]) != 0) exit(1);
        minimum_records = shard_records[shard] < minimum_records
                              ? shard_records[shard]
                              : minimum_records;
        maximum_records = shard_records[shard] > maximum_records
                              ? shard_records[shard]
                              : maximum_records;
    }
    printf("reduce_solve columns=9 inputs=%d raw_records=%llu reduced=%zu "
           "retained=%llu shards=%d min_records=%llu max_records=%llu "
           "reduced_weight=",
           input_count, (unsigned long long)raw_records, reduced.count,
           (unsigned long long)retained, shards,
           (unsigned long long)minimum_records,
           (unsigned long long)maximum_records);
    print_u128(map_weight_sum(&reduced));
    printf(" covered_weight=");
    print_u128(covered_weight);
    printf(" prefix=%s\n", output_prefix);
    free(shard_records);
    free(outputs);
    free(reduced.entries);
}

static void run_solve_reduce_7x9(int shards, int shard, const char* output_path,
                                 int input_count, char** input_paths,
                                 int require_unique) {
    if (shards < 1 || shard < 0 || shard >= shards || input_count < 1) exit(2);
    uint64_t raw_records = 0;
    for (int input = 0; input < input_count; input++) {
        struct stat status;
        if (stat(input_paths[input], &status) != 0 ||
            status.st_size % (off_t)sizeof(OrbitRecord) != 0) exit(1);
        raw_records += (uint64_t)(status.st_size / (off_t)sizeof(OrbitRecord));
    }
    size_t capacity = 16;
    while (capacity < (size_t)raw_records * 2U) capacity <<= 1;
    OrbitMap reduced;
    orbit_map_init(&reduced, capacity);
    for (int input = 0; input < input_count; input++) {
        FILE* file = fopen(input_paths[input], "rb");
        if (!file) exit(1);
        OrbitRecord record;
        while (fread(&record, sizeof(record), 1, file) == 1) {
            if (__builtin_popcountll(record.key) > 31 ||
                solve_shard_7x9(record.key, shards) != shard) {
                fprintf(stderr, "invalid 7x9 solve record in %s\n",
                        input_paths[input]);
                exit(1);
            }
            if (require_unique) {
                orbit_map_insert_unique(&reduced, record.key, record.weight);
            } else {
                orbit_map_add(&reduced, record.key, record.weight);
            }
        }
        if (!feof(file) || fclose(file) != 0) exit(1);
    }
    U128 labelled_weight = map_weight_sum(&reduced);
    write_orbit_file_unsorted(output_path, 9, &reduced);
    printf("solve_reduce_7x9 shard=%d/%d inputs=%d raw_records=%llu unique=%zu "
           "labelled_weight=",
           shard, shards, input_count, (unsigned long long)raw_records,
           reduced.count);
    print_u128(labelled_weight);
    printf(" covered_weight=");
    print_u128(labelled_weight * 2U);
    printf(" unique_input=%d output=%s\n", require_unique, output_path);
    free(reduced.entries);
}

static void run_solve_partition_7x9(const char* input_path, int shards,
                                    const char* output_prefix) {
    if (shards < 1 || shards > 9999) exit(2);
    FILE* input = fopen(input_path, "rb");
    if (!input) exit(1);
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    if (fread(magic, sizeof(magic), 1, input) != 1 ||
        memcmp(magic, ORBIT_MAGIC, 7) != 0 ||
        fread(&columns, sizeof(columns), 1, input) != 1 ||
        fread(&count, sizeof(count), 1, input) != 1 || columns != 9) exit(1);
    FILE** outputs = xcalloc((size_t)shards, sizeof(outputs[0]));
    uint64_t* shard_records = xcalloc((size_t)shards, sizeof(shard_records[0]));
    U128 labelled_weight = 0;
    U128 retained_weight = 0;
    char path[4096];
    for (int shard = 0; shard < shards; shard++) {
        snprintf(path, sizeof(path), "%s.s%04d.orbits", output_prefix, shard);
        outputs[shard] = fopen(path, "wb+");
        if (!outputs[shard]) exit(1);
        write_orbit_header(outputs[shard], 9, 0);
    }
    uint64_t retained = 0;
    for (uint64_t index = 0; index < count; index++) {
        OrbitRecord record;
        if (fread(&record, sizeof(record), 1, input) != 1) exit(1);
        labelled_weight += record.weight;
        if (__builtin_popcountll(record.key) > 31) continue;
        int shard = solve_shard_7x9(record.key, shards);
        if (fwrite(&record, sizeof(record), 1, outputs[shard]) != 1) exit(1);
        shard_records[shard]++;
        retained_weight += record.weight;
        retained++;
    }
    if (fgetc(input) != EOF || fclose(input) != 0) exit(1);
    uint64_t minimum_records = UINT64_MAX;
    uint64_t maximum_records = 0;
    for (int shard = 0; shard < shards; shard++) {
        if (fseeko(outputs[shard], 12, SEEK_SET) != 0 ||
            fwrite(&shard_records[shard], sizeof(shard_records[shard]), 1,
                   outputs[shard]) != 1 ||
            fclose(outputs[shard]) != 0) exit(1);
        minimum_records = shard_records[shard] < minimum_records
                              ? shard_records[shard]
                              : minimum_records;
        maximum_records = shard_records[shard] > maximum_records
                              ? shard_records[shard]
                              : maximum_records;
    }
    printf("solve_partition_7x9 input=%s records=%llu retained=%llu shards=%d "
           "min_records=%llu max_records=%llu labelled_weight=",
           input_path, (unsigned long long)count, (unsigned long long)retained,
           shards, (unsigned long long)minimum_records,
           (unsigned long long)maximum_records);
    print_u128(labelled_weight);
    printf(" retained_weight=");
    print_u128(retained_weight);
    printf(" covered_weight=");
    print_u128(retained_weight * 2U);
    printf(" prefix=%s\n", output_prefix);
    free(shard_records);
    free(outputs);
}

typedef struct {
    uint64_t records;
    U128 labelled_weight;
    U128 covered_weight;
} SolveCheckTotals7x9;

static SolveCheckTotals7x9 check_solve_file_7x9(int shards, int shard,
                                                const char* path) {
    FILE* input = fopen(path, "rb");
    if (!input) exit(1);
    char magic[8];
    uint32_t columns = 0;
    SolveCheckTotals7x9 totals = {0};
    if (fread(magic, sizeof(magic), 1, input) != 1 ||
        memcmp(magic, ORBIT_MAGIC, 7) != 0 ||
        fread(&columns, sizeof(columns), 1, input) != 1 ||
        fread(&totals.records, sizeof(totals.records), 1, input) != 1 ||
        columns != 9) exit(1);
    for (uint64_t index = 0; index < totals.records; index++) {
        OrbitRecord record;
        if (fread(&record, sizeof(record), 1, input) != 1 ||
            __builtin_popcountll(record.key) > 31 ||
            solve_shard_7x9(record.key, shards) != shard) exit(1);
        totals.labelled_weight += record.weight;
    }
    if (fgetc(input) != EOF || fclose(input) != 0) exit(1);
    totals.covered_weight = totals.labelled_weight * 2U;
    return totals;
}

static void run_solve_check_shard_7x9(int shards, int shard,
                                      const char* input_path) {
    if (shards < 1 || shard < 0 || shard >= shards) exit(2);
    SolveCheckTotals7x9 totals =
        check_solve_file_7x9(shards, shard, input_path);
    printf("solve_check_shard_7x9 shard=%d/%d records=%llu labelled_weight=",
           shard, shards, (unsigned long long)totals.records);
    print_u128(totals.labelled_weight);
    printf(" covered_weight=");
    print_u128(totals.covered_weight);
    printf(" OK\n");
}

static void run_solve_check_7x9(int shards, int input_count,
                                char** input_paths) {
    if (shards != input_count || shards < 1) exit(2);
    SolveCheckTotals7x9 totals = {0};
    for (int shard = 0; shard < shards; shard++) {
        SolveCheckTotals7x9 item =
            check_solve_file_7x9(shards, shard, input_paths[shard]);
        totals.records += item.records;
        totals.labelled_weight += item.labelled_weight;
        totals.covered_weight += item.covered_weight;
    }
    const uint64_t expected_records = UINT64_C(3608247685);
    const U128 expected_labelled_weight = (U128)1 << 62;
    const U128 expected_covered_weight = (U128)1 << 63;
    int valid = totals.records == expected_records &&
                totals.labelled_weight == expected_labelled_weight &&
                totals.covered_weight == expected_covered_weight;
    printf("solve_check_7x9 shards=%d records=%llu expected_records=%llu "
           "labelled_weight=",
           shards, (unsigned long long)totals.records,
           (unsigned long long)expected_records);
    print_u128(totals.labelled_weight);
    printf(" expected_labelled_weight=");
    print_u128(expected_labelled_weight);
    printf(" covered_weight=");
    print_u128(totals.covered_weight);
    printf(" expected_covered_weight=");
    print_u128(expected_covered_weight);
    printf(" %s\n", valid ? "OK" : "FAIL");
    if (!valid) exit(1);
}
#endif

#if ORBIT_ROWS == 8 && ORBIT_MAX_COLUMNS == 8 && ORBIT_ROW_BITS == 8
static uint32_t solve_left_prefix(uint64_t key) {
    RowPattern rows[ROWS];
    unpack_rows(key, rows);
    uint32_t prefix = 0;
    for (int row = 0; row < ROWS; row++) {
        prefix = (prefix << 4) | (rows[row] & 15U);
    }
    return prefix;
}

static uint64_t canonical_complement_key(uint64_t key) {
    RowPattern rows[ROWS];
    unpack_rows(key, rows);
    for (int row = 0; row < ROWS; row++) rows[row] ^= UINT8_MAX;
    return canonical_key(rows, 8);
}

static uint64_t solve_representative(uint64_t key);

static uint64_t transpose_8x8_key(uint64_t key) {
    RowPattern source[8];
    RowPattern target[8] = {0};
    unpack_rows(key, source);
    for (int row = 0; row < 8; row++) {
        for (int column = 0; column < 8; column++) {
            target[column] |=
                (RowPattern)(((source[row] >> column) & 1U) << row);
        }
    }
    return pack_rows(target);
}

static uint64_t solve_transpose_partner(uint64_t key,
                                        int* raw_transpose_canonical) {
    uint64_t raw = transpose_8x8_key(key);
    RowPattern rows[8];
    unpack_rows(raw, rows);
    uint64_t canonical = canonical_key(rows, 8);
    if (raw_transpose_canonical) *raw_transpose_canonical = raw == canonical;
    return solve_representative(canonical);
}

typedef struct {
    uint64_t records;
    uint64_t midpoint;
    uint64_t fixed;
    uint64_t lower;
    uint64_t higher;
    uint64_t same_owner;
    uint64_t raw_transpose_canonical;
    uint64_t involution_checks;
    U128 fixed_weight;
    U128 lower_weight;
    U128 higher_weight;
    U128 covered_weight;
} TransposeCensusTotals;

static void run_solve_transpose_census(int shards, int shard,
                                       const char* input_path,
                                       uint64_t requested_samples) {
    if (shards < 1 || shard < 0 || shard >= shards) exit(2);
    FILE* input = fopen(input_path, "rb");
    if (!input) exit(1);
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    double load_begin = seconds_now();
    if (fread(magic, sizeof(magic), 1, input) != 1 ||
        memcmp(magic, ORBIT_MAGIC, 7) != 0 ||
        fread(&columns, sizeof(columns), 1, input) != 1 ||
        fread(&count, sizeof(count), 1, input) != 1 || columns != 8 || !count) {
        fprintf(stderr, "invalid transpose census input %s\n", input_path);
        exit(1);
    }
    OrbitRecord* records = xcalloc((size_t)count, sizeof(records[0]));
    if (fread(records, sizeof(records[0]), (size_t)count, input) != count ||
        fgetc(input) != EOF || fclose(input) != 0) exit(1);
    double load_seconds = seconds_now() - load_begin;
    uint64_t work = !requested_samples || requested_samples > count
                        ? count : requested_samples;
    initialise_fast_canon_tables();
    int thread_count = 1;
#ifdef _OPENMP
    thread_count = omp_get_max_threads();
#endif
    TransposeCensusTotals* thread_totals =
        xcalloc((size_t)thread_count, sizeof(thread_totals[0]));
    double begin = seconds_now();
#pragma omp parallel
    {
        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif
        TransposeCensusTotals local = {0};
#pragma omp for schedule(static)
        for (uint64_t sample = 0; sample < work; sample++) {
            uint64_t index = work == count
                                 ? sample
                                 : (uint64_t)(((U128)sample * count) / work);
            OrbitRecord record = records[index];
            if (!record.weight || __builtin_popcountll(record.key) > 32 ||
                mix64(solve_left_prefix(record.key)) % (uint64_t)shards !=
                    (uint64_t)shard) {
                fprintf(stderr,
                        "invalid transpose census record at index %llu\n",
                        (unsigned long long)index);
                exit(1);
            }
            int raw_canonical = 0;
            uint64_t partner =
                solve_transpose_partner(record.key, &raw_canonical);
            if (partner == UINT64_MAX ||
                __builtin_popcountll(partner) !=
                    __builtin_popcountll(record.key)) exit(1);
            local.records++;
            int cells = __builtin_popcountll(record.key);
            local.midpoint += cells == 32;
            local.covered_weight +=
                (U128)record.weight * (cells < 32 ? 2U : 1U);
            local.raw_transpose_canonical += (uint64_t)raw_canonical;
            local.same_owner +=
                mix64(solve_left_prefix(partner)) % (uint64_t)shards ==
                (uint64_t)shard;
            if (partner == record.key) {
                local.fixed++;
                local.fixed_weight += record.weight;
            } else if (record.key < partner) {
                local.lower++;
                local.lower_weight += record.weight;
            } else {
                local.higher++;
                local.higher_weight += record.weight;
            }
            if (sample < 8192) {
                uint64_t round_trip =
                    solve_transpose_partner(partner, NULL);
                if (round_trip != record.key) {
                    fprintf(stderr,
                            "transpose involution mismatch key=%016llx "
                            "partner=%016llx round_trip=%016llx\n",
                            (unsigned long long)record.key,
                            (unsigned long long)partner,
                            (unsigned long long)round_trip);
                    exit(1);
                }
                local.involution_checks++;
            }
        }
        thread_totals[thread] = local;
    }
    double census_seconds = seconds_now() - begin;
    TransposeCensusTotals totals = {0};
    for (int thread = 0; thread < thread_count; thread++) {
        totals.records += thread_totals[thread].records;
        totals.midpoint += thread_totals[thread].midpoint;
        totals.fixed += thread_totals[thread].fixed;
        totals.lower += thread_totals[thread].lower;
        totals.higher += thread_totals[thread].higher;
        totals.same_owner += thread_totals[thread].same_owner;
        totals.raw_transpose_canonical +=
            thread_totals[thread].raw_transpose_canonical;
        totals.involution_checks += thread_totals[thread].involution_checks;
        totals.fixed_weight += thread_totals[thread].fixed_weight;
        totals.lower_weight += thread_totals[thread].lower_weight;
        totals.higher_weight += thread_totals[thread].higher_weight;
        totals.covered_weight += thread_totals[thread].covered_weight;
    }
    if (totals.records != work ||
        totals.fixed + totals.lower + totals.higher != work) exit(1);
    printf("solve_transpose_census shard=%d/%d input_records=%llu "
           "samples=%llu midpoint=%llu fixed=%llu lower=%llu higher=%llu "
           "same_owner=%llu "
           "raw_transpose_canonical=%llu involution_checks=%llu threads=%d "
           "load_seconds=%.6f census_seconds=%.6f rate=%.3f fixed_weight=",
           shard, shards, (unsigned long long)count,
           (unsigned long long)totals.records,
           (unsigned long long)totals.midpoint,
           (unsigned long long)totals.fixed,
           (unsigned long long)totals.lower,
           (unsigned long long)totals.higher,
           (unsigned long long)totals.same_owner,
           (unsigned long long)totals.raw_transpose_canonical,
           (unsigned long long)totals.involution_checks, thread_count,
           load_seconds, census_seconds,
           census_seconds ? (double)work / census_seconds : 0.0);
    print_u128(totals.fixed_weight);
    printf(" lower_weight=");
    print_u128(totals.lower_weight);
    printf(" higher_weight=");
    print_u128(totals.higher_weight);
    printf(" covered_weight=");
    print_u128(totals.covered_weight);
    printf(" OK\n");
    free(thread_totals);
    free(records);
}

static void run_automorphism_check(const char* input_path) {
    static const uint64_t factorial[9] = {
        1, 1, 2, 6, 24, 120, 720, 5040, 40320
    };
    FILE* input = fopen(input_path, "rb");
    if (!input) exit(1);
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    if (fread(magic, sizeof(magic), 1, input) != 1 ||
        memcmp(magic, ORBIT_MAGIC, 7) != 0 ||
        fread(&columns, sizeof(columns), 1, input) != 1 ||
        fread(&count, sizeof(count), 1, input) != 1 || columns > 8) exit(1);
    U128 total_weight = 0;
    uint64_t group_order = factorial[8] * factorial[columns];
    double begin = seconds_now();
    for (uint64_t index = 0; index < count; index++) {
        OrbitRecord record;
        RowPattern rows[8];
        uint64_t automorphisms = 0;
        if (fread(&record, sizeof(record), 1, input) != 1) exit(1);
        unpack_rows(record.key, rows);
        uint64_t key = canonical_key_with_automorphisms(
            rows, (int)columns, &automorphisms
        );
        if (!automorphisms || group_order % automorphisms || key != record.key ||
            group_order / automorphisms != record.weight) {
            fprintf(stderr, "automorphism weight mismatch at record %llu\n",
                    (unsigned long long)index);
            exit(1);
        }
        total_weight += record.weight;
    }
    if (fgetc(input) != EOF || fclose(input) != 0) exit(1);
    U128 expected_weight = (U128)1 << (8 * columns);
    printf("automorphism_check columns=%u records=%llu weight=", columns,
           (unsigned long long)count);
    print_u128(total_weight);
    printf(" expected_weight=");
    print_u128(expected_weight);
    printf(" seconds=%.3f %s\n", seconds_now() - begin,
           total_weight == expected_weight ? "OK" : "FAIL");
    if (total_weight != expected_weight) exit(1);
}

static uint64_t solve_representative(uint64_t key) {
    int cells = __builtin_popcountll(key);
    if (cells < 32) return key;
    if (cells > 32) return UINT64_MAX;
    uint64_t complement = canonical_complement_key(key);
    return key < complement ? key : complement;
}

static void run_solve_transpose_filter(int shards, int shard,
                                       const char* input_path,
                                       const char* output_path) {
    if (shards < 1 || shard < 0 || shard >= shards) exit(2);
    FILE* input = fopen(input_path, "rb");
    if (!input) exit(1);
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    double load_begin = seconds_now();
    if (fread(magic, sizeof(magic), 1, input) != 1 ||
        memcmp(magic, ORBIT_MAGIC, 7) != 0 ||
        fread(&columns, sizeof(columns), 1, input) != 1 ||
        fread(&count, sizeof(count), 1, input) != 1 || columns != 8 || !count) {
        fprintf(stderr, "invalid transpose filter input %s\n", input_path);
        exit(1);
    }
    OrbitRecord* records = xcalloc((size_t)count, sizeof(records[0]));
    if (fread(records, sizeof(records[0]), (size_t)count, input) != count ||
        fgetc(input) != EOF || fclose(input) != 0) exit(1);
    double load_seconds = seconds_now() - load_begin;
    initialise_fast_canon_tables();
    int thread_count = 1;
#ifdef _OPENMP
    thread_count = omp_get_max_threads();
#endif
    TransposeCensusTotals* thread_totals =
        xcalloc((size_t)thread_count, sizeof(thread_totals[0]));
    double begin = seconds_now();
#pragma omp parallel
    {
        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif
        TransposeCensusTotals local = {0};
#pragma omp for schedule(static)
        for (uint64_t index = 0; index < count; index++) {
            OrbitRecord* record = &records[index];
            int cells = __builtin_popcountll(record->key);
            if (!record->weight || cells > 32 ||
                mix64(solve_left_prefix(record->key)) % (uint64_t)shards !=
                    (uint64_t)shard) {
                fprintf(stderr,
                        "invalid transpose filter record at index %llu\n",
                        (unsigned long long)index);
                exit(1);
            }
            int raw_canonical = 0;
            uint64_t partner =
                solve_transpose_partner(record->key, &raw_canonical);
            if (partner == UINT64_MAX ||
                __builtin_popcountll(partner) != cells) exit(1);
            local.records++;
            local.midpoint += cells == 32;
            local.covered_weight +=
                (U128)record->weight * (cells < 32 ? 2U : 1U);
            local.raw_transpose_canonical += (uint64_t)raw_canonical;
            local.same_owner +=
                mix64(solve_left_prefix(partner)) % (uint64_t)shards ==
                (uint64_t)shard;
            if (partner == record->key) {
                local.fixed++;
                local.fixed_weight += record->weight;
            } else if (record->key < partner) {
                if (record->weight > UINT64_MAX / 2U) exit(1);
                local.lower++;
                local.lower_weight += record->weight;
                record->weight *= 2U;
            } else {
                local.higher++;
                local.higher_weight += record->weight;
                record->weight = 0;
            }
            if (index < 8192) {
                if (solve_transpose_partner(partner, NULL) != record->key) {
                    fprintf(stderr,
                            "transpose filter involution mismatch at index %llu\n",
                            (unsigned long long)index);
                    exit(1);
                }
                local.involution_checks++;
            }
        }
        thread_totals[thread] = local;
    }
    double filter_seconds = seconds_now() - begin;
    TransposeCensusTotals totals = {0};
    for (int thread = 0; thread < thread_count; thread++) {
        totals.records += thread_totals[thread].records;
        totals.midpoint += thread_totals[thread].midpoint;
        totals.fixed += thread_totals[thread].fixed;
        totals.lower += thread_totals[thread].lower;
        totals.higher += thread_totals[thread].higher;
        totals.same_owner += thread_totals[thread].same_owner;
        totals.raw_transpose_canonical +=
            thread_totals[thread].raw_transpose_canonical;
        totals.involution_checks += thread_totals[thread].involution_checks;
        totals.fixed_weight += thread_totals[thread].fixed_weight;
        totals.lower_weight += thread_totals[thread].lower_weight;
        totals.higher_weight += thread_totals[thread].higher_weight;
        totals.covered_weight += thread_totals[thread].covered_weight;
    }
    uint64_t output_count = totals.fixed + totals.lower;
    char temporary_path[4096];
    int path_length = snprintf(temporary_path, sizeof(temporary_path),
                               "%s.tmp", output_path);
    if (path_length < 0 || (size_t)path_length >= sizeof(temporary_path)) exit(1);
    FILE* output = fopen(temporary_path, "wb");
    if (!output) exit(1);
    const char transpose_magic[8] = SQUARE_TRANSPOSE_MAGIC;
    const uint32_t transpose_columns = 8;
    if (fwrite(transpose_magic, sizeof(transpose_magic), 1, output) != 1 ||
        fwrite(&transpose_columns, sizeof(transpose_columns), 1, output) != 1 ||
        fwrite(&output_count, sizeof(output_count), 1, output) != 1) exit(1);
    U128 output_weight = 0;
    U128 output_covered_weight = 0;
    for (uint64_t index = 0; index < count; index++) {
        OrbitRecord record = records[index];
        if (!record.weight) continue;
        int cells = __builtin_popcountll(record.key);
        output_weight += record.weight;
        output_covered_weight +=
            (U128)record.weight * (cells < 32 ? 2U : 1U);
        if (fwrite(&record, sizeof(record), 1, output) != 1) exit(1);
    }
    if (fclose(output) != 0 || rename(temporary_path, output_path) != 0) {
        fprintf(stderr, "cannot publish transpose filter output %s: %s\n",
                output_path, strerror(errno));
        exit(1);
    }
    U128 expected_output_weight =
        totals.fixed_weight + totals.lower_weight * 2U;
    if (totals.records != count ||
        totals.fixed + totals.lower + totals.higher != count ||
        output_weight != expected_output_weight) exit(1);
    printf("solve_transpose_filter shard=%d/%d input_records=%llu "
           "output_records=%llu midpoint=%llu fixed=%llu lower=%llu "
           "higher=%llu "
           "same_owner=%llu raw_transpose_canonical=%llu "
           "involution_checks=%llu threads=%d load_seconds=%.6f "
           "filter_seconds=%.6f input_weight=",
           shard, shards, (unsigned long long)count,
           (unsigned long long)output_count,
           (unsigned long long)totals.midpoint,
           (unsigned long long)totals.fixed,
           (unsigned long long)totals.lower,
           (unsigned long long)totals.higher,
           (unsigned long long)totals.same_owner,
           (unsigned long long)totals.raw_transpose_canonical,
           (unsigned long long)totals.involution_checks, thread_count,
           load_seconds, filter_seconds);
    print_u128(totals.fixed_weight + totals.lower_weight + totals.higher_weight);
    printf(" output_weight=");
    print_u128(output_weight);
    printf(" output_covered_weight=");
    print_u128(output_covered_weight);
    printf(" output=%s OK\n", output_path);
    free(thread_totals);
    free(records);
}

static void run_solve_transpose_gate(const char* input_path, uint64_t limit,
                                     const char* control_path,
                                     const char* quotient_path) {
    if (!limit) exit(2);
    FILE* input = fopen(input_path, "rb");
    FILE* control = fopen(control_path, "wb+");
    FILE* quotient = fopen(quotient_path, "wb+");
    if (!input || !control || !quotient) exit(1);
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    if (fread(magic, sizeof(magic), 1, input) != 1 ||
        memcmp(magic, ORBIT_MAGIC, 7) != 0 ||
        fread(&columns, sizeof(columns), 1, input) != 1 ||
        fread(&count, sizeof(count), 1, input) != 1 || columns != 8) exit(1);
    write_orbit_header(control, 8, 0);
    const char transpose_magic[8] = SQUARE_TRANSPOSE_MAGIC;
    const uint32_t transpose_columns = 8;
    const uint64_t empty_count = 0;
    if (fwrite(transpose_magic, sizeof(transpose_magic), 1, quotient) != 1 ||
        fwrite(&transpose_columns, sizeof(transpose_columns), 1, quotient) != 1 ||
        fwrite(&empty_count, sizeof(empty_count), 1, quotient) != 1) exit(1);
    uint64_t scanned = 0;
    uint64_t control_count = 0;
    uint64_t quotient_count = 0;
    uint64_t fixed = 0;
    U128 control_weight = 0;
    U128 quotient_weight = 0;
    U128 control_covered_weight = 0;
    U128 quotient_covered_weight = 0;
    while (scanned < count && quotient_count < limit) {
        OrbitRecord record;
        if (fread(&record, sizeof(record), 1, input) != 1) exit(1);
        scanned++;
        uint64_t partner = solve_transpose_partner(record.key, NULL);
        if (record.key > partner) continue;
        int cells = __builtin_popcountll(record.key);
        uint64_t complement_factor = cells < 32 ? 2U : 1U;
        if (fwrite(&record, sizeof(record), 1, control) != 1) exit(1);
        control_count++;
        control_weight += record.weight;
        control_covered_weight += (U128)record.weight * complement_factor;
        OrbitRecord reduced = record;
        if (record.key == partner) {
            fixed++;
        } else {
            if (record.weight > UINT64_MAX / 2U) exit(1);
            OrbitRecord transposed = {.key = partner, .weight = record.weight};
            if (fwrite(&transposed, sizeof(transposed), 1, control) != 1) exit(1);
            control_count++;
            control_weight += record.weight;
            control_covered_weight += (U128)record.weight * complement_factor;
            reduced.weight *= 2U;
        }
        if (fwrite(&reduced, sizeof(reduced), 1, quotient) != 1) exit(1);
        quotient_count++;
        quotient_weight += reduced.weight;
        quotient_covered_weight +=
            (U128)reduced.weight * complement_factor;
    }
    if (fclose(input) != 0 || quotient_count != limit ||
        control_weight != quotient_weight ||
        control_covered_weight != quotient_covered_weight) exit(1);
    if (fseeko(control, 12, SEEK_SET) != 0 ||
        fwrite(&control_count, sizeof(control_count), 1, control) != 1 ||
        fclose(control) != 0 || fseeko(quotient, 12, SEEK_SET) != 0 ||
        fwrite(&quotient_count, sizeof(quotient_count), 1, quotient) != 1 ||
        fclose(quotient) != 0) exit(1);
    printf("solve_transpose_gate input=%s scanned=%llu control_records=%llu "
           "quotient_records=%llu fixed=%llu weight=",
           input_path, (unsigned long long)scanned,
           (unsigned long long)control_count,
           (unsigned long long)quotient_count, (unsigned long long)fixed);
    print_u128(control_weight);
    printf(" covered_weight=");
    print_u128(control_covered_weight);
    printf(" control=%s quotient=%s OK\n", control_path, quotient_path);
}

static uint64_t retained_assignment_count(int parent_cells) {
    static const uint16_t binomial_prefix[9] = {
        1, 9, 37, 93, 163, 219, 247, 255, 256
    };
    if (parent_cells > 32) return 0;
    int remaining = 32 - parent_cells;
    return binomial_prefix[remaining < 8 ? remaining : 8];
}

static void run_solve_plan(const char* parent_path, int ranges) {
    if (ranges < 1 || ranges > 9999) exit(2);
    FILE* input = fopen(parent_path, "rb");
    if (!input) exit(1);
    char magic[8];
    uint32_t columns = 0;
    uint64_t parent_count = 0;
    if (fread(magic, sizeof(magic), 1, input) != 1 ||
        memcmp(magic, ORBIT_MAGIC, 7) != 0 ||
        fread(&columns, sizeof(columns), 1, input) != 1 ||
        fread(&parent_count, sizeof(parent_count), 1, input) != 1 ||
        columns != 7 || parent_count != g_expected_counts[7]) exit(1);
    const uint64_t expected_work = UINT64_C(71256694627);
    uint64_t cumulative = 0;
    uint64_t range_start = 0;
    uint64_t range_work = 0;
    int range = 0;
    double begin = seconds_now();
    for (uint64_t index = 0; index < parent_count; index++) {
        OrbitRecord parent;
        if (fread(&parent, sizeof(parent), 1, input) != 1) exit(1);
        uint64_t work = retained_assignment_count(__builtin_popcountll(parent.key));
        cumulative += work;
        range_work += work;
        if (range + 1 < ranges &&
            (U128)cumulative * (uint64_t)ranges >=
                (U128)expected_work * (uint64_t)(range + 1)) {
            printf("solve_plan range=%d start=%llu end=%llu work=%llu\n", range,
                   (unsigned long long)range_start,
                   (unsigned long long)(index + 1),
                   (unsigned long long)range_work);
            range++;
            range_start = index + 1;
            range_work = 0;
        }
    }
    if (fgetc(input) != EOF || fclose(input) != 0 ||
        cumulative != expected_work || range != ranges - 1) exit(1);
    printf("solve_plan range=%d start=%llu end=%llu work=%llu\n", range,
           (unsigned long long)range_start, (unsigned long long)parent_count,
           (unsigned long long)range_work);
    printf("solve_plan records=%llu work=%llu ranges=%d seconds=%.3f OK\n",
           (unsigned long long)parent_count, (unsigned long long)cumulative,
           ranges, seconds_now() - begin);
}

static void write_solve_map(const OrbitMap* map, FILE** outputs, int shards,
                            uint64_t* records) {
    for (size_t slot = 0; slot < map->capacity; slot++) {
        OrbitRecord record = map->entries[slot];
        if (record.key == UINT64_MAX) continue;
        int shard = (int)(mix64(solve_left_prefix(record.key)) % (uint64_t)shards);
        if (fwrite(&record, sizeof(record), 1, outputs[shard]) != 1) exit(1);
        (*records)++;
    }
}

static void run_solve_extend(const char* parent_path, uint64_t start, uint64_t end,
                             int shards, size_t chunk_parents,
                             const char* output_prefix, int canonical_parent) {
    if (shards < 1 || shards > 9999 || !chunk_parents) exit(2);
    FILE* input = fopen(parent_path, "rb");
    if (!input) exit(1);
    char magic[8];
    uint32_t columns = 0;
    uint64_t parent_count = 0;
    if (fread(magic, sizeof(magic), 1, input) != 1 ||
        memcmp(magic, ORBIT_MAGIC, 7) != 0 ||
        fread(&columns, sizeof(columns), 1, input) != 1 ||
        fread(&parent_count, sizeof(parent_count), 1, input) != 1 ||
        columns != 7 || start > end || end > parent_count) {
        fprintf(stderr, "invalid 8x7 solve-extension input or range\n");
        exit(1);
    }
    if (fseeko(input, 20 + (off_t)start * (off_t)sizeof(OrbitRecord), SEEK_SET) != 0) {
        exit(1);
    }
    FILE** outputs = xcalloc((size_t)shards, sizeof(outputs[0]));
    char path[4096];
    for (int shard = 0; shard < shards; shard++) {
        snprintf(path, sizeof(path), "%s.s%04d", output_prefix, shard);
        outputs[shard] = fopen(path, "wb");
        if (!outputs[shard]) exit(1);
    }
    uint64_t candidates = 0;
    uint64_t retained_candidates = 0;
    uint64_t canonical_paths = 0;
    uint64_t emitted_paths = 0;
    uint64_t local_solve_records = 0;
    uint64_t solve_records = 0;
    U128 parent_weight = 0;
    double begin = seconds_now();
    for (uint64_t chunk_start = start; chunk_start < end;) {
        uint64_t chunk_end = chunk_start + chunk_parents;
        if (chunk_end > end) chunk_end = end;
        size_t capacity = 16;
        size_t chunk_count = (size_t)(chunk_end - chunk_start);
        if (chunk_count > SIZE_MAX / NEW_COLUMNS) exit(1);
        size_t capacity_per_parent = canonical_parent ? 64U : NEW_COLUMNS;
        while (capacity < chunk_count * capacity_per_parent) capacity <<= 1;
        OrbitMap solve;
        orbit_map_init(&solve, capacity);
        for (uint64_t index = chunk_start; index < chunk_end; index++) {
            OrbitRecord parent;
            if (fread(&parent, sizeof(parent), 1, input) != 1) exit(1);
            parent_weight += parent.weight;
            RowPattern rows[ROWS];
            unpack_rows(parent.key, rows);
            int parent_cells = __builtin_popcountll(parent.key);
            uint64_t parent_assignments = retained_assignment_count(parent_cells);
            if (!parent_assignments) {
                candidates += NEW_COLUMNS;
                continue;
            }
            for (unsigned assignment = 0; assignment < NEW_COLUMNS; assignment++) {
                candidates++;
                if (parent_cells + __builtin_popcount(assignment) > 32) continue;
                RowPattern child_rows[ROWS];
                for (int row = 0; row < ROWS; row++) {
                    child_rows[row] = rows[row] |
                                      (RowPattern)(((assignment >> row) & 1U) << 7);
                }
                uint64_t automorphisms = 0;
                int canonical_extension = 0;
                uint64_t child = canonical_parent ?
                    canonical_extension_analysis(
                        child_rows, 8, &automorphisms, &canonical_extension
                    ) : canonical_key(child_rows, 8);
                retained_candidates++;
                if (canonical_parent) {
                    if (!canonical_extension) continue;
                    canonical_paths++;
                    const uint64_t group_order = UINT64_C(40320) * UINT64_C(40320);
                    if (!automorphisms || group_order % automorphisms) exit(1);
                    uint64_t weight = group_order / automorphisms;
                    uint64_t representative = child;
                    if (parent_cells + __builtin_popcount(assignment) == 32) {
                        uint64_t complement = canonical_complement_key(child);
                        if (child > complement) continue;
                        if (child < complement) weight *= 2U;
                    }
                    emitted_paths++;
                    orbit_map_set_exact(&solve, representative, weight);
                } else {
                    uint64_t representative = solve_representative(child);
                    if (representative == UINT64_MAX) exit(1);
                    orbit_map_add(&solve, representative, parent.weight);
                }
            }
        }
        local_solve_records += solve.count;
        write_solve_map(&solve, outputs, shards, &solve_records);
        free(solve.entries);
        chunk_start = chunk_end;
        printf("solve_extend_progress parent=%llu/%llu candidates=%llu "
               "retained=%llu records=%llu elapsed=%.3f\n",
               (unsigned long long)chunk_start, (unsigned long long)end,
               (unsigned long long)candidates,
               (unsigned long long)retained_candidates,
               (unsigned long long)solve_records, seconds_now() - begin);
    }
    if (fclose(input) != 0) exit(1);
    for (int shard = 0; shard < shards; shard++) {
        if (fclose(outputs[shard]) != 0) exit(1);
    }
    free(outputs);
    printf("solve_extend range=[%llu,%llu) parents=%llu parent_weight=",
           (unsigned long long)start, (unsigned long long)end,
           (unsigned long long)(end - start));
    print_u128(parent_weight);
    printf(" candidates=%llu retained_candidates=%llu canonical_paths=%llu "
           "emitted_paths=%llu local_solve_unique=%llu "
           "solve_records=%llu shards=%d chunk_parents=%zu seconds=%.3f "
           "canonical_parent=%d prefix=%s\n",
           (unsigned long long)candidates,
           (unsigned long long)retained_candidates,
           (unsigned long long)canonical_paths,
           (unsigned long long)emitted_paths,
           (unsigned long long)local_solve_records,
           (unsigned long long)solve_records, shards, chunk_parents,
           seconds_now() - begin, canonical_parent, output_prefix);
}

static void run_solve_reduce(int shards, int shard, const char* output_path,
                             int input_count, char** input_paths,
                             int require_unique) {
    if (shards < 1 || shard < 0 || shard >= shards || input_count < 1) exit(2);
    uint64_t raw_records = 0;
    for (int input_index = 0; input_index < input_count; input_index++) {
        struct stat status;
        if (stat(input_paths[input_index], &status) != 0 ||
            status.st_size % (off_t)sizeof(OrbitRecord) != 0) exit(1);
        raw_records += (uint64_t)(status.st_size / (off_t)sizeof(OrbitRecord));
    }
    if (raw_records > SIZE_MAX / 2U) exit(1);
    size_t capacity = 16;
    while (capacity < (size_t)raw_records * 2U) capacity <<= 1;
    OrbitMap reduced;
    orbit_map_init(&reduced, capacity);
    for (int input_index = 0; input_index < input_count; input_index++) {
        FILE* input = fopen(input_paths[input_index], "rb");
        if (!input) exit(1);
        OrbitRecord record;
        while (fread(&record, sizeof(record), 1, input) == 1) {
            if (__builtin_popcountll(record.key) > 32 ||
                mix64(solve_left_prefix(record.key)) % (uint64_t)shards !=
                    (uint64_t)shard) {
                fprintf(stderr, "invalid solve record in %s\n", input_paths[input_index]);
                exit(1);
            }
            if (require_unique) {
                orbit_map_insert_unique(&reduced, record.key, record.weight);
            } else {
                orbit_map_add(&reduced, record.key, record.weight);
            }
        }
        if (!feof(input) || fclose(input) != 0) exit(1);
    }
    U128 covered_weight = 0;
    uint64_t midpoint = 0;
    for (size_t slot = 0; slot < reduced.capacity; slot++) {
        OrbitRecord record = reduced.entries[slot];
        if (record.key == UINT64_MAX) continue;
        int cells = __builtin_popcountll(record.key);
        covered_weight += (U128)record.weight * (cells < 32 ? 2U : 1U);
        if (cells == 32) {
            midpoint++;
        }
    }
    write_orbit_file_unsorted(output_path, 8, &reduced);
    printf("solve_reduce shard=%d/%d inputs=%d raw_records=%llu unique=%zu "
           "midpoint=%llu labelled_weight=",
           shard, shards, input_count, (unsigned long long)raw_records, reduced.count,
           (unsigned long long)midpoint);
    print_u128(map_weight_sum(&reduced));
    printf(" covered_weight=");
    print_u128(covered_weight);
    printf(" unique_input=%d output=%s\n", require_unique, output_path);
    free(reduced.entries);
}

static void run_solve_partition(const char* input_path, int shards,
                                const char* output_prefix) {
    if (shards < 1 || shards > 9999) exit(2);
    FILE* input = fopen(input_path, "rb");
    if (!input) exit(1);
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    if (fread(magic, sizeof(magic), 1, input) != 1 ||
        memcmp(magic, ORBIT_MAGIC, 7) != 0 ||
        fread(&columns, sizeof(columns), 1, input) != 1 ||
        fread(&count, sizeof(count), 1, input) != 1 || columns != 8) exit(1);
    FILE** outputs = xcalloc((size_t)shards, sizeof(outputs[0]));
    char path[4096];
    for (int shard = 0; shard < shards; shard++) {
        snprintf(path, sizeof(path), "%s.s%04d", output_prefix, shard);
        outputs[shard] = fopen(path, "wb");
        if (!outputs[shard]) exit(1);
    }
    uint64_t retained = 0;
    for (uint64_t index = 0; index < count; index++) {
        OrbitRecord record;
        if (fread(&record, sizeof(record), 1, input) != 1) exit(1);
        uint64_t representative = solve_representative(record.key);
        if (representative == UINT64_MAX) continue;
        record.key = representative;
        int shard = (int)(mix64(solve_left_prefix(record.key)) % (uint64_t)shards);
        if (fwrite(&record, sizeof(record), 1, outputs[shard]) != 1) exit(1);
        retained++;
    }
    if (fgetc(input) != EOF || fclose(input) != 0) exit(1);
    for (int shard = 0; shard < shards; shard++) {
        if (fclose(outputs[shard]) != 0) exit(1);
    }
    free(outputs);
    printf("solve_partition input=%s records=%llu retained=%llu shards=%d prefix=%s\n",
           input_path, (unsigned long long)count, (unsigned long long)retained,
           shards, output_prefix);
}

typedef struct {
    uint64_t records;
    uint64_t midpoint;
    uint64_t self_complementary;
    U128 labelled_weight;
    U128 covered_weight;
} SolveCheckTotals;

static SolveCheckTotals check_solve_file(int shards, int shard, const char* path) {
    if (shards < 1 || shard < 0 || shard >= shards) exit(2);
    FILE* input = fopen(path, "rb");
    if (!input) exit(1);
    char magic[8];
    uint32_t columns = 0;
    SolveCheckTotals totals = {0};
    if (fread(magic, sizeof(magic), 1, input) != 1 ||
        memcmp(magic, SQUARE_TRANSPOSE_MAGIC, 7) != 0 ||
        fread(&columns, sizeof(columns), 1, input) != 1 ||
        fread(&totals.records, sizeof(totals.records), 1, input) != 1 ||
        columns != 8) exit(1);
    for (uint64_t index = 0; index < totals.records; index++) {
        OrbitRecord record;
        if (fread(&record, sizeof(record), 1, input) != 1) exit(1);
        int cells = __builtin_popcountll(record.key);
        if (cells > 32 ||
            mix64(solve_left_prefix(record.key)) % (uint64_t)shards !=
                (uint64_t)shard) exit(1);
        totals.labelled_weight += record.weight;
        totals.covered_weight += (U128)record.weight * (cells < 32 ? 2U : 1U);
        if (cells == 32) {
            uint64_t complement = canonical_complement_key(record.key);
            if (record.key > complement) exit(1);
            totals.midpoint++;
            totals.self_complementary += complement == record.key;
        }
    }
    if (fgetc(input) != EOF || fclose(input) != 0) exit(1);
    return totals;
}

static void print_solve_check_shard(int shards, int shard,
                                    const SolveCheckTotals* totals) {
    printf("solve_check_shard shard=%d/%d records=%llu midpoint=%llu "
           "self_complementary=%llu labelled_weight=", shard, shards,
           (unsigned long long)totals->records,
           (unsigned long long)totals->midpoint,
           (unsigned long long)totals->self_complementary);
    print_u128(totals->labelled_weight);
    printf(" covered_weight=");
    print_u128(totals->covered_weight);
    printf(" OK\n");
}

static void run_solve_check_shard(int shards, int shard, const char* input_path) {
    SolveCheckTotals totals = check_solve_file(shards, shard, input_path);
    print_solve_check_shard(shards, shard, &totals);
}

static void run_solve_check(int shards, int input_count, char** input_paths) {
    if (shards != input_count || shards < 1) exit(2);
    initialise_fast_canon_tables();
    // Files are independent.  Keep one exact result per shard so the final
    // 128-bit reduction is deterministic and needs no OpenMP reduction support.
    SolveCheckTotals* shard_totals =
        xcalloc((size_t)shards, sizeof(shard_totals[0]));
    int thread_count = 1;
#ifdef _OPENMP
    thread_count = omp_get_max_threads();
#endif
    double begin = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
    for (int shard = 0; shard < shards; shard++) {
        shard_totals[shard] =
            check_solve_file(shards, shard, input_paths[shard]);
    }
    SolveCheckTotals totals = {0};
    for (int shard = 0; shard < shards; shard++) {
        SolveCheckTotals item = shard_totals[shard];
        totals.records += item.records;
        totals.midpoint += item.midpoint;
        totals.self_complementary += item.self_complementary;
        totals.labelled_weight += item.labelled_weight;
        totals.covered_weight += item.covered_weight;
    }
    free(shard_totals);
    const uint64_t expected_records = UINT64_C(3671999389);
    const uint64_t expected_midpoint = UINT64_C(354110921);
    const uint64_t expected_self_complementary = UINT64_C(217940);
    U128 expected_covered_weight = (U128)1 << 64;
    int valid = totals.records == expected_records &&
                totals.midpoint == expected_midpoint &&
                totals.self_complementary == expected_self_complementary &&
                totals.covered_weight == expected_covered_weight;
    printf("solve_check shards=%d records=%llu expected_records=%llu midpoint=%llu "
           "expected_midpoint=%llu self_complementary=%llu "
           "expected_self_complementary=%llu labelled_weight=",
           shards, (unsigned long long)totals.records,
           (unsigned long long)expected_records, (unsigned long long)totals.midpoint,
           (unsigned long long)expected_midpoint,
           (unsigned long long)totals.self_complementary,
           (unsigned long long)expected_self_complementary);
    print_u128(totals.labelled_weight);
    printf(" covered_weight=");
    print_u128(totals.covered_weight);
    printf(" expected_covered_weight=");
    print_u128(expected_covered_weight);
    printf(" threads=%d seconds=%.3f %s\n", thread_count,
           seconds_now() - begin, valid ? "OK" : "FAIL");
    if (!valid) exit(1);
}

static uint64_t transpose_7x8_key(uint64_t source_key) {
    uint8_t source_rows[7];
    for (int row = 6; row >= 0; row--) {
        source_rows[row] = (uint8_t)(source_key & UINT64_C(0xff));
        source_key >>= 8;
    }
    RowPattern target_rows[8] = {0};
    for (int target_row = 0; target_row < 8; target_row++) {
        for (int target_column = 0; target_column < 7; target_column++) {
            target_rows[target_row] |=
                (RowPattern)(((source_rows[target_column] >> target_row) & 1U)
                             << target_column);
        }
    }
    return pack_rows(target_rows);
}

static uint64_t transpose_8x7_key(uint64_t source_key) {
    RowPattern source_rows[ROWS];
    unpack_rows(source_key, source_rows);
    uint8_t target_rows[7] = {0};
    for (int target_row = 0; target_row < 7; target_row++) {
        for (int target_column = 0; target_column < 8; target_column++) {
            target_rows[target_row] |=
                (uint8_t)(((source_rows[target_column] >> target_row) & 1U)
                          << target_column);
        }
    }
    uint64_t target_key = 0;
    for (int row = 0; row < 7; row++) target_key = (target_key << 8) | target_rows[row];
    return target_key;
}

static void run_transpose_7x8(const char* input_path, const char* output_path) {
    FILE* input = fopen(input_path, "rb");
    FILE* output = fopen(output_path, "wb");
    if (!input || !output) {
        fprintf(stderr, "cannot open transpose input or output: %s\n", strerror(errno));
        exit(1);
    }
    char input_magic[8];
    uint32_t input_columns = 0;
    uint64_t count = 0;
    if (fread(input_magic, sizeof(input_magic), 1, input) != 1 ||
        memcmp(input_magic, "R7ORB01", 7) != 0 ||
        fread(&input_columns, sizeof(input_columns), 1, input) != 1 ||
        fread(&count, sizeof(count), 1, input) != 1 || input_columns != 8 ||
        count != g_expected_counts[7]) {
        fprintf(stderr, "invalid complete 7x8 orbit file %s\n", input_path);
        exit(1);
    }
    write_orbit_header(output, 7, count);
    U128 weight = 0;
    double start = seconds_now();
    for (uint64_t index = 0; index < count; index++) {
        OrbitRecord source;
        if (fread(&source, sizeof(source), 1, input) != 1) {
            fprintf(stderr, "truncated orbit file %s\n", input_path);
            exit(1);
        }
        OrbitRecord target = {.key = transpose_7x8_key(source.key),
                              .weight = source.weight};
        if (transpose_8x7_key(target.key) != source.key ||
            fwrite(&target, sizeof(target), 1, output) != 1) {
            fprintf(stderr, "transpose validation/write failure at record %llu\n",
                    (unsigned long long)index);
            exit(1);
        }
        weight += source.weight;
    }
    if (fgetc(input) != EOF || fclose(input) != 0 || fclose(output) != 0) exit(1);
    U128 expected_weight = (U128)1 << 56;
    printf("transpose source=7x8 target=8x7 records=%llu weight=",
           (unsigned long long)count);
    print_u128(weight);
    printf(" expected_weight=");
    print_u128(expected_weight);
    printf(" seconds=%.3f output=%s %s\n", seconds_now() - start, output_path,
           weight == expected_weight ? "OK" : "FAIL");
    if (weight != expected_weight) exit(1);
}
#endif

static void usage(const char* program) {
    fprintf(stderr,
            "Usage:\n"
            "  %s build COLUMNS [OUTPUT.orbits]\n"
            "  %s augment PARENTS.orbits OUTPUT.orbits\n"
            "  %s extend PARENTS.orbits START END BUCKETS OUTPUT_PREFIX\n"
            "  %s reduce COLUMNS OUTPUT.orbits BUCKET_FILE...\n"
            "  %s combine OUTPUT.orbits INPUT.orbits...\n"
            "  %s check INPUT.orbits\n"
            "  %s partition INPUT.orbits BUCKETS OUTPUT_PREFIX\n"
            "  %s sample-extend PARENTS.orbits MOD ID OUTPUT.orbits\n"
            "  %s slice INPUT.orbits START END OUTPUT.orbits\n",
            program, program, program, program, program, program, program, program,
            program);
#if ORBIT_ROWS == ORBIT_MAX_COLUMNS && defined(SQUARE_TRANSPOSE_MAGIC)
    fprintf(stderr,
            "  %s square-transpose-filter INPUT.orbits OUTPUT.orbits\n",
            program);
#endif
#if ORBIT_ROWS == 6 && ORBIT_MAX_COLUMNS == 10 && ORBIT_ROW_BITS == 10
    fprintf(stderr,
            "  %s augment-solve PARENTS_6x9.orbits OUTPUT_6x10.orbits\n"
            "  %s solve-census PARENTS_6x9.orbits\n"
            "  %s solve-partition INPUT.orbits SHARDS OUTPUT_PREFIX\n"
            "  %s solve-check SHARDS SHARD0.orbits ...\n"
            "  %s promote-half-seed INPUT_6x5.orbits OUTPUT_6x10.orbits\n",
            program, program, program, program, program);
#endif
#if ORBIT_ROWS == 7 && ORBIT_MAX_COLUMNS == 9 && ORBIT_ROW_BITS == 9
    fprintf(stderr,
            "  %s sample-extend7x8 PARENTS.orbits COUNT OUTPUT.orbits\n"
            "  %s solve-plan7x9 PARENTS.orbits RANGES\n"
            "  %s extend7x8 PARENTS.orbits START END BUCKETS OUTPUT_PREFIX\n"
            "  %s solve-extend7x8 PARENTS.orbits START END BUCKETS OUTPUT_PREFIX\n"
            "  %s reduce-solve COLUMNS SHARDS OUTPUT_PREFIX BUCKET_FILE...\n"
            "  %s solve-reduce SHARDS SHARD OUTPUT.orbits RAW_FILE...\n"
            "  %s solve-reduce-unique SHARDS SHARD OUTPUT.orbits RAW_FILE...\n"
            "  %s solve-partition INPUT.orbits SHARDS OUTPUT_PREFIX\n"
            "  %s solve-check-shard SHARDS SHARD INPUT.orbits\n"
            "  %s solve-check SHARDS SHARD0.orbits ...\n",
            program, program, program, program, program, program, program, program,
            program, program);
#endif
#if ORBIT_ROWS == 8 && ORBIT_MAX_COLUMNS == 8 && ORBIT_ROW_BITS == 8
    fprintf(stderr, "  %s transpose7x8 INPUT.orbits OUTPUT.orbits\n", program);
    fprintf(stderr,
            "  %s build-canonical-parent COLUMNS [OUTPUT.orbits]\n"
            "  %s automorphism-check INPUT.orbits\n"
            "  %s solve-plan PARENTS.orbits RANGES\n"
            "  %s solve-extend PARENTS.orbits START END SHARDS CHUNK_PARENTS "
            "OUTPUT_PREFIX\n"
            "  %s solve-extend-reference PARENTS.orbits START END SHARDS "
            "CHUNK_PARENTS OUTPUT_PREFIX\n"
            "  %s solve-reduce SHARDS SHARD OUTPUT.orbits RAW_FILE...\n"
            "  %s solve-reduce-unique SHARDS SHARD OUTPUT.orbits RAW_FILE...\n"
            "  %s solve-partition INPUT.orbits SHARDS OUTPUT_PREFIX\n"
            "  %s solve-check-shard SHARDS SHARD INPUT.orbits\n"
            "  %s solve-check SHARDS SHARD0.orbits ...\n",
            program, program, program, program, program, program, program, program,
            program, program);
    fprintf(stderr,
            "  %s solve-transpose-census SHARDS SHARD INPUT.orbits SAMPLES\n"
            "  %s solve-transpose-filter SHARDS SHARD INPUT.orbits "
            "OUTPUT.orbits\n"
            "  %s solve-transpose-gate INPUT.orbits RECORDS CONTROL.orbits "
            "QUOTIENT.orbits\n",
            program, program, program);
#endif
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
#if ORBIT_ROWS == ORBIT_MAX_COLUMNS && defined(SQUARE_TRANSPOSE_MAGIC)
    if (argc == 4 && strcmp(argv[1], "square-transpose-filter") == 0) {
        run_square_transpose_filter(argv[2], argv[3]);
        return 0;
    }
#endif
    if (argc >= 3 && strcmp(argv[1], "build") == 0) {
        int columns = atoi(argv[2]);
        if (columns < 0 || columns > MAX_COLUMNS || argc > 4) {
            usage(argv[0]);
            return 2;
        }
        run_build(columns, argc == 4 ? argv[3] : NULL);
        return 0;
    }
    if (argc == 4 && strcmp(argv[1], "augment") == 0) {
        run_augment(argv[2], argv[3]);
        return 0;
    }
    if (argc == 7 && strcmp(argv[1], "extend") == 0) {
        size_t start = strtoull(argv[3], NULL, 10);
        size_t end = strtoull(argv[4], NULL, 10);
        int buckets = atoi(argv[5]);
        run_extend(argv[2], start, end, buckets, argv[6]);
        return 0;
    }
    if (argc >= 5 && strcmp(argv[1], "reduce") == 0) {
        int columns = atoi(argv[2]);
        if (columns < 1 || columns > MAX_COLUMNS) return 2;
        run_reduce(columns, argv[3], argc - 4, argv + 4);
        return 0;
    }
    if (argc >= 4 && strcmp(argv[1], "combine") == 0) {
        run_combine(argv[2], argc - 3, argv + 3);
        return 0;
    }
    if (argc == 3 && strcmp(argv[1], "check") == 0) {
        run_check(argv[2]);
        return 0;
    }
    if (argc == 5 && strcmp(argv[1], "partition") == 0) {
        run_partition(argv[2], atoi(argv[3]), argv[4]);
        return 0;
    }
    if (argc == 6 && strcmp(argv[1], "sample-extend") == 0) {
        run_sample_extend(argv[2], strtoull(argv[3], NULL, 10),
                          strtoull(argv[4], NULL, 10), argv[5]);
        return 0;
    }
    if (argc == 6 && strcmp(argv[1], "slice") == 0) {
        run_slice(argv[2], strtoull(argv[3], NULL, 10),
                  strtoull(argv[4], NULL, 10), argv[5]);
        return 0;
    }
#if ORBIT_ROWS == 6 && ORBIT_MAX_COLUMNS == 10 && ORBIT_ROW_BITS == 10
    if (argc == 4 && strcmp(argv[1], "augment-solve") == 0) {
        run_augment_solve_6x10(argv[2], argv[3]);
        return 0;
    }
    if (argc == 3 && strcmp(argv[1], "solve-census") == 0) {
        run_solve_census_6x10(argv[2]);
        return 0;
    }
    if (argc == 5 && strcmp(argv[1], "solve-partition") == 0) {
        run_solve_partition_6x10(argv[2], atoi(argv[3]), argv[4]);
        return 0;
    }
    if (argc >= 4 && strcmp(argv[1], "solve-check") == 0) {
        run_solve_check_6x10(atoi(argv[2]), argc - 3, argv + 3);
        return 0;
    }
    if (argc == 4 && strcmp(argv[1], "promote-half-seed") == 0) {
        run_promote_half_seed_6x10(argv[2], argv[3]);
        return 0;
    }
#endif
#if ORBIT_ROWS == 7 && ORBIT_MAX_COLUMNS == 9 && ORBIT_ROW_BITS == 9
    if (argc == 5 && strcmp(argv[1], "sample-extend7x8") == 0) {
        run_sample_extend_7x8(argv[2], strtoull(argv[3], NULL, 10), argv[4]);
        return 0;
    }
    if (argc == 4 && strcmp(argv[1], "solve-plan7x9") == 0) {
        run_solve_plan_7x9(argv[2], atoi(argv[3]));
        return 0;
    }
    if (argc == 7 && strcmp(argv[1], "extend7x8") == 0) {
        run_extend_7x8(argv[2], strtoull(argv[3], NULL, 10),
                       strtoull(argv[4], NULL, 10), atoi(argv[5]), argv[6], 0);
        return 0;
    }
    if (argc == 7 && strcmp(argv[1], "solve-extend7x8") == 0) {
        run_extend_7x8(argv[2], strtoull(argv[3], NULL, 10),
                       strtoull(argv[4], NULL, 10), atoi(argv[5]), argv[6], 1);
        return 0;
    }
    if (argc >= 6 && strcmp(argv[1], "reduce-solve") == 0) {
        run_reduce_solve_7x9(atoi(argv[2]), atoi(argv[3]), argv[4],
                             argc - 5, argv + 5);
        return 0;
    }
    if (argc >= 6 && strcmp(argv[1], "solve-reduce") == 0) {
        run_solve_reduce_7x9(atoi(argv[2]), atoi(argv[3]), argv[4],
                             argc - 5, argv + 5, 0);
        return 0;
    }
    if (argc >= 6 && strcmp(argv[1], "solve-reduce-unique") == 0) {
        run_solve_reduce_7x9(atoi(argv[2]), atoi(argv[3]), argv[4],
                             argc - 5, argv + 5, 1);
        return 0;
    }
    if (argc == 5 && strcmp(argv[1], "solve-partition") == 0) {
        run_solve_partition_7x9(argv[2], atoi(argv[3]), argv[4]);
        return 0;
    }
    if (argc == 5 && strcmp(argv[1], "solve-check-shard") == 0) {
        run_solve_check_shard_7x9(atoi(argv[2]), atoi(argv[3]), argv[4]);
        return 0;
    }
    if (argc >= 4 && strcmp(argv[1], "solve-check") == 0) {
        run_solve_check_7x9(atoi(argv[2]), argc - 3, argv + 3);
        return 0;
    }
#endif
#if ORBIT_ROWS == 8 && ORBIT_MAX_COLUMNS == 8 && ORBIT_ROW_BITS == 8
    if (argc >= 3 && strcmp(argv[1], "build-canonical-parent") == 0) {
        int columns = atoi(argv[2]);
        if (columns < 0 || columns > MAX_COLUMNS || argc > 4) return 2;
        run_build_canonical_parent(columns, argc == 4 ? argv[3] : NULL);
        return 0;
    }
    if (argc == 4 && strcmp(argv[1], "transpose7x8") == 0) {
        run_transpose_7x8(argv[2], argv[3]);
        return 0;
    }
    if (argc == 3 && strcmp(argv[1], "automorphism-check") == 0) {
        run_automorphism_check(argv[2]);
        return 0;
    }
    if (argc == 4 && strcmp(argv[1], "solve-plan") == 0) {
        run_solve_plan(argv[2], atoi(argv[3]));
        return 0;
    }
    if (argc == 8 && strcmp(argv[1], "solve-extend") == 0) {
        run_solve_extend(argv[2], strtoull(argv[3], NULL, 10),
                         strtoull(argv[4], NULL, 10), atoi(argv[5]),
                         strtoull(argv[6], NULL, 10), argv[7], 1);
        return 0;
    }
    if (argc == 8 && strcmp(argv[1], "solve-extend-reference") == 0) {
        run_solve_extend(argv[2], strtoull(argv[3], NULL, 10),
                         strtoull(argv[4], NULL, 10), atoi(argv[5]),
                         strtoull(argv[6], NULL, 10), argv[7], 0);
        return 0;
    }
    if (argc >= 6 && strcmp(argv[1], "solve-reduce") == 0) {
        run_solve_reduce(
            atoi(argv[2]), atoi(argv[3]), argv[4], argc - 5, argv + 5, 0
        );
        return 0;
    }
    if (argc >= 6 && strcmp(argv[1], "solve-reduce-unique") == 0) {
        run_solve_reduce(
            atoi(argv[2]), atoi(argv[3]), argv[4], argc - 5, argv + 5, 1
        );
        return 0;
    }
    if (argc == 5 && strcmp(argv[1], "solve-partition") == 0) {
        run_solve_partition(argv[2], atoi(argv[3]), argv[4]);
        return 0;
    }
    if (argc == 5 && strcmp(argv[1], "solve-check-shard") == 0) {
        run_solve_check_shard(atoi(argv[2]), atoi(argv[3]), argv[4]);
        return 0;
    }
    if (argc >= 4 && strcmp(argv[1], "solve-check") == 0) {
        run_solve_check(atoi(argv[2]), argc - 3, argv + 3);
        return 0;
    }
    if (argc == 6 && strcmp(argv[1], "solve-transpose-census") == 0) {
        run_solve_transpose_census(
            atoi(argv[2]), atoi(argv[3]), argv[4],
            strtoull(argv[5], NULL, 10));
        return 0;
    }
    if (argc == 6 && strcmp(argv[1], "solve-transpose-filter") == 0) {
        run_solve_transpose_filter(
            atoi(argv[2]), atoi(argv[3]), argv[4], argv[5]);
        return 0;
    }
    if (argc == 6 && strcmp(argv[1], "solve-transpose-gate") == 0) {
        run_solve_transpose_gate(
            argv[2], strtoull(argv[3], NULL, 10), argv[4], argv[5]);
        return 0;
    }
#endif
    usage(argv[0]);
    return 2;
}
