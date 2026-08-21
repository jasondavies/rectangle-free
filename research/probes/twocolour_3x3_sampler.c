/* Exact double-coset sampler for the two-bit 3+3 contraction at 8x6. */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned __int128 U128;

enum {
    ROWS = 8,
    HALF_COLUMNS = 3,
    HALF_PATTERNS = 8,
    FULL_COLUMNS = 6,
    FULL_PATTERNS = 64,
    PAIRS = 28,
    PREFIX_ORBITS = 1324,
    MAX_TABLES = 40320
};

typedef struct {
    uint64_t mask;
    uint64_t weight;
    uint8_t used;
} MapEntry;

typedef struct {
    MapEntry* entries;
    size_t capacity;
    size_t count;
} Map;

typedef struct {
    uint64_t mask;
    uint16_t weight;
} Increment;

typedef struct {
    uint64_t mask;
    uint64_t weight;
} VectorEntry;

typedef struct {
    VectorEntry* entries;
    size_t count;
} Distribution;

typedef struct {
    uint8_t histogram[HALF_PATTERNS];
    size_t support[2];
} Prefix;

typedef struct {
    uint8_t cells[FULL_PATTERNS];
    uint64_t group_multiplicity;
} RelativeTable;

typedef struct {
    uint64_t key[4];
    U128 value;
    uint8_t used;
} KernelEntry;

static int g_pair_index[ROWS][ROWS];
static uint64_t g_factorial[ROWS + 1];
static int g_half_preimages[6][HALF_PATTERNS];
static int g_half_permutation[HALF_COLUMNS];
static int g_half_permutation_count;
static int g_full_preimages[720][FULL_PATTERNS];
static int g_full_permutation[FULL_COLUMNS];
static int g_full_permutation_count;
static uint8_t g_prefix_histogram[HALF_PATTERNS];
static Prefix g_prefixes[PREFIX_ORBITS];
static int g_prefix_count;
static int g_viable_indices[PREFIX_ORBITS];
static int g_viable_count;
static RelativeTable g_tables[MAX_TABLES];
static int g_table_count;
static int g_row_patterns[HALF_PATTERNS];
static int g_column_patterns[HALF_PATTERNS];
static int g_row_type_count;
static int g_column_type_count;
static int g_row_margins[HALF_PATTERNS];
static int g_column_remaining[HALF_PATTERNS];
static uint8_t g_small_table[HALF_PATTERNS][HALF_PATTERNS];
static uint64_t g_table_multiplicity_numerator;
static KernelEntry* g_kernel_cache;
static size_t g_kernel_capacity;
static size_t g_unique_kernels;
static uint64_t g_candidate_tables;
static uint64_t g_duplicate_kernels;
static uint64_t g_duplicate_validations;
static uint64_t g_pair_tests;
static uint64_t g_distribution_transitions;
static uint64_t* g_test_samples;
static size_t g_test_sample_count;
static U128 g_checksum;

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

static void map_init(Map* map, size_t capacity) {
    map->capacity = capacity;
    map->count = 0;
    map->entries = calloc(capacity, sizeof(map->entries[0]));
    if (!map->entries) exit(1);
}

static void map_insert_raw(Map* map, uint64_t mask, uint64_t weight) {
    size_t slot = (size_t)mix64(mask) & (map->capacity - 1U);
    while (map->entries[slot].used) slot = (slot + 1U) & (map->capacity - 1U);
    map->entries[slot] = (MapEntry){.mask = mask, .weight = weight, .used = 1};
    map->count++;
}

static void map_rehash(Map* map) {
    MapEntry* old = map->entries;
    size_t old_capacity = map->capacity;
    map_init(map, old_capacity << 1);
    for (size_t i = 0; i < old_capacity; i++) {
        if (old[i].used) map_insert_raw(map, old[i].mask, old[i].weight);
    }
    free(old);
}

static void map_add(Map* map, uint64_t mask, uint64_t weight) {
    if ((map->count + 1U) * 10U >= map->capacity * 7U) map_rehash(map);
    size_t slot = (size_t)mix64(mask) & (map->capacity - 1U);
    while (map->entries[slot].used) {
        if (map->entries[slot].mask == mask) {
            map->entries[slot].weight += weight;
            return;
        }
        slot = (slot + 1U) & (map->capacity - 1U);
    }
    map_insert_raw(map, mask, weight);
}

static int increment_compare(const void* lhs_ptr, const void* rhs_ptr) {
    const Increment* lhs = lhs_ptr;
    const Increment* rhs = rhs_ptr;
    return lhs->mask < rhs->mask ? -1 : lhs->mask > rhs->mask;
}

static Distribution build_distribution(const uint8_t row_patterns[ROWS], int complement) {
    Map current;
    map_init(&current, 16);
    map_add(&current, 0, 1);
    for (int column = 0; column < HALF_COLUMNS; column++) {
        unsigned active_rows = 0;
        for (int row = 0; row < ROWS; row++) {
            unsigned pattern = complement ? row_patterns[row] ^ (HALF_PATTERNS - 1) :
                                            row_patterns[row];
            if ((pattern >> column) & 1U) active_rows |= 1U << row;
        }
        Increment increments[256];
        size_t increment_count = 0;
        unsigned assignment = active_rows;
        for (;;) {
            uint64_t mask = 0;
            for (int u = 0; u < ROWS; u++) {
                for (int v = u + 1; v < ROWS; v++) {
                    if (((active_rows >> u) & 1U) == 0 ||
                        ((active_rows >> v) & 1U) == 0) continue;
                    unsigned cu = (assignment >> u) & 1U;
                    unsigned cv = (assignment >> v) & 1U;
                    if (cu == cv) mask |= UINT64_C(1) << (cu * PAIRS + g_pair_index[u][v]);
                }
            }
            increments[increment_count++] = (Increment){.mask = mask, .weight = 1};
            if (!assignment) break;
            assignment = (assignment - 1U) & active_rows;
        }
        qsort(increments, increment_count, sizeof(increments[0]), increment_compare);
        size_t unique = 0;
        for (size_t i = 0; i < increment_count; i++) {
            if (unique && increments[i].mask == increments[unique - 1U].mask) {
                increments[unique - 1U].weight++;
            } else {
                increments[unique++] = increments[i];
            }
        }
        Map next;
        map_init(&next, current.capacity);
        for (size_t i = 0; i < current.capacity; i++) {
            if (!current.entries[i].used) continue;
            for (size_t j = 0; j < unique; j++) {
                if (current.entries[i].mask & increments[j].mask) continue;
                map_add(&next, current.entries[i].mask | increments[j].mask,
                        current.entries[i].weight * increments[j].weight);
                g_distribution_transitions++;
            }
        }
        free(current.entries);
        current = next;
    }
    VectorEntry* entries = calloc(current.count, sizeof(entries[0]));
    if (!entries && current.count) exit(1);
    size_t count = 0;
    for (size_t i = 0; i < current.capacity; i++) {
        if (current.entries[i].used) {
            entries[count++] =
                (VectorEntry){.mask = current.entries[i].mask, .weight = current.entries[i].weight};
        }
    }
    free(current.entries);
    return (Distribution){.entries = entries, .count = count};
}

static uint64_t disjoint_join(const Distribution* lhs, const Distribution* rhs,
                              uint64_t* local_tests) {
    uint64_t total = 0;
    *local_tests = 0;
    for (size_t i = 0; i < lhs->count; i++) {
        for (size_t j = 0; j < rhs->count; j++) {
            (*local_tests)++;
            if ((lhs->entries[i].mask & rhs->entries[j].mask) == 0) {
                total += lhs->entries[i].weight * rhs->entries[j].weight;
            }
        }
    }
    return total;
}

static void free_distribution(Distribution* distribution) {
    free(distribution->entries);
    *distribution = (Distribution){0};
}

static void generate_half_permutations(int depth, unsigned used) {
    if (depth == HALF_COLUMNS) {
        for (int pattern = 0; pattern < HALF_PATTERNS; pattern++) {
            int image = 0;
            for (int bit = 0; bit < HALF_COLUMNS; bit++) {
                if ((pattern >> bit) & 1) image |= 1 << g_half_permutation[bit];
            }
            g_half_preimages[g_half_permutation_count][image] = pattern;
        }
        g_half_permutation_count++;
        return;
    }
    for (int bit = 0; bit < HALF_COLUMNS; bit++) {
        if ((used >> bit) & 1U) continue;
        g_half_permutation[depth] = bit;
        generate_half_permutations(depth + 1, used | (1U << bit));
    }
}

static void generate_full_permutations(int depth, unsigned used) {
    if (depth == FULL_COLUMNS) {
        for (int pattern = 0; pattern < FULL_PATTERNS; pattern++) {
            int image = 0;
            for (int bit = 0; bit < FULL_COLUMNS; bit++) {
                if ((pattern >> bit) & 1) image |= 1 << g_full_permutation[bit];
            }
            g_full_preimages[g_full_permutation_count][image] = pattern;
        }
        g_full_permutation_count++;
        return;
    }
    for (int bit = 0; bit < FULL_COLUMNS; bit++) {
        if ((used >> bit) & 1U) continue;
        g_full_permutation[depth] = bit;
        generate_full_permutations(depth + 1, used | (1U << bit));
    }
}

static int prefix_is_canonical(void) {
    for (int permutation = 0; permutation < g_half_permutation_count; permutation++) {
        for (int pattern = 0; pattern < HALF_PATTERNS; pattern++) {
            unsigned transformed = g_prefix_histogram[g_half_preimages[permutation][pattern]];
            if (transformed < g_prefix_histogram[pattern]) return 0;
            if (transformed > g_prefix_histogram[pattern]) break;
        }
    }
    return 1;
}

static void histogram_rows(const uint8_t histogram[HALF_PATTERNS], uint8_t rows[ROWS]) {
    int row = 0;
    for (int pattern = 0; pattern < HALF_PATTERNS; pattern++) {
        for (int copy = 0; copy < histogram[pattern]; copy++) rows[row++] = (uint8_t)pattern;
    }
}

static void record_prefix(void) {
    Prefix* prefix = &g_prefixes[g_prefix_count++];
    memcpy(prefix->histogram, g_prefix_histogram, sizeof(prefix->histogram));
    uint8_t rows[ROWS];
    histogram_rows(prefix->histogram, rows);
    for (int complement = 0; complement < 2; complement++) {
        Distribution distribution = build_distribution(rows, complement);
        prefix->support[complement] = distribution.count;
        free_distribution(&distribution);
    }
    if (prefix->support[0] && prefix->support[1]) {
        g_viable_indices[g_viable_count++] = g_prefix_count - 1;
    }
}

static void enumerate_prefix_histograms(int pattern, int remaining) {
    if (pattern == HALF_PATTERNS - 1) {
        g_prefix_histogram[pattern] = (uint8_t)remaining;
        if (prefix_is_canonical()) record_prefix();
        return;
    }
    for (int count = 0; count <= remaining; count++) {
        g_prefix_histogram[pattern] = (uint8_t)count;
        enumerate_prefix_histograms(pattern + 1, remaining - count);
    }
}

static void store_relative_table(void) {
    if (g_table_count == MAX_TABLES) {
        fprintf(stderr, "relative table capacity exceeded\n");
        exit(1);
    }
    RelativeTable* table = &g_tables[g_table_count++];
    memset(table->cells, 0, sizeof(table->cells));
    uint64_t denominator = 1;
    for (int row = 0; row < g_row_type_count; row++) {
        for (int column = 0; column < g_column_type_count; column++) {
            unsigned count = g_small_table[row][column];
            table->cells[g_row_patterns[row] | (g_column_patterns[column] << HALF_COLUMNS)] =
                (uint8_t)count;
            denominator *= g_factorial[count];
        }
    }
    table->group_multiplicity = g_table_multiplicity_numerator / denominator;
}

static void assign_table_row(int row);

static void assign_table_cells(int row, int column, int remaining) {
    if (column == g_column_type_count - 1) {
        if (remaining > g_column_remaining[column]) return;
        g_small_table[row][column] = (uint8_t)remaining;
        g_column_remaining[column] -= remaining;
        assign_table_row(row + 1);
        g_column_remaining[column] += remaining;
        return;
    }
    int maximum = remaining < g_column_remaining[column] ? remaining : g_column_remaining[column];
    for (int value = 0; value <= maximum; value++) {
        g_small_table[row][column] = (uint8_t)value;
        g_column_remaining[column] -= value;
        assign_table_cells(row, column + 1, remaining - value);
        g_column_remaining[column] += value;
    }
}

static void assign_table_row(int row) {
    if (row == g_row_type_count - 1) {
        int sum = 0;
        for (int column = 0; column < g_column_type_count; column++) {
            g_small_table[row][column] = (uint8_t)g_column_remaining[column];
            sum += g_column_remaining[column];
        }
        if (sum == g_row_margins[row]) store_relative_table();
        return;
    }
    assign_table_cells(row, 0, g_row_margins[row]);
}

static void generate_relative_tables(const Prefix* lhs, const Prefix* rhs) {
    g_row_type_count = 0;
    g_column_type_count = 0;
    g_table_multiplicity_numerator = 1;
    memset(g_small_table, 0, sizeof(g_small_table));
    for (int pattern = 0; pattern < HALF_PATTERNS; pattern++) {
        if (lhs->histogram[pattern]) {
            g_row_patterns[g_row_type_count] = pattern;
            g_row_margins[g_row_type_count++] = lhs->histogram[pattern];
            g_table_multiplicity_numerator *= g_factorial[lhs->histogram[pattern]];
        }
        if (rhs->histogram[pattern]) {
            g_column_patterns[g_column_type_count] = pattern;
            g_column_remaining[g_column_type_count++] = rhs->histogram[pattern];
            g_table_multiplicity_numerator *= g_factorial[rhs->histogram[pattern]];
        }
    }
    g_table_count = 0;
    assign_table_row(0);
    uint64_t multiplicity_sum = 0;
    for (int i = 0; i < g_table_count; i++) multiplicity_sum += g_tables[i].group_multiplicity;
    if (multiplicity_sum != g_factorial[ROWS]) {
        fprintf(stderr, "double-coset checksum failed: %llu != %llu\n",
                (unsigned long long)multiplicity_sum,
                (unsigned long long)g_factorial[ROWS]);
        exit(1);
    }
}

static void canonical_full_key(const uint8_t histogram[FULL_PATTERNS], uint64_t key[4]) {
    uint8_t best[FULL_PATTERNS];
    memcpy(best, histogram, sizeof(best));
    for (int permutation = 0; permutation < g_full_permutation_count; permutation++) {
        for (int pattern = 0; pattern < FULL_PATTERNS; pattern++) {
            uint8_t transformed = histogram[g_full_preimages[permutation][pattern]];
            if (transformed < best[pattern]) {
                for (int tail = pattern; tail < FULL_PATTERNS; tail++) {
                    best[tail] = histogram[g_full_preimages[permutation][tail]];
                }
                break;
            }
            if (transformed > best[pattern]) break;
        }
    }
    memset(key, 0, 4 * sizeof(key[0]));
    for (int pattern = 0; pattern < FULL_PATTERNS; pattern++) {
        key[pattern / 16] |= (uint64_t)best[pattern] << (4 * (pattern % 16));
    }
}

static uint64_t kernel_hash(const uint64_t key[4]) {
    uint64_t hash = UINT64_C(0x9e3779b97f4a7c15);
    for (int i = 0; i < 4; i++) hash = mix64(hash ^ key[i]);
    return hash;
}

static KernelEntry* kernel_lookup(const uint64_t key[4], int* found) {
    size_t slot = (size_t)kernel_hash(key) & (g_kernel_capacity - 1U);
    while (g_kernel_cache[slot].used) {
        if (memcmp(g_kernel_cache[slot].key, key, 4 * sizeof(key[0])) == 0) {
            *found = 1;
            return &g_kernel_cache[slot];
        }
        slot = (slot + 1U) & (g_kernel_capacity - 1U);
    }
    *found = 0;
    return &g_kernel_cache[slot];
}

static void table_rows(const RelativeTable* table, uint8_t lhs_rows[ROWS],
                       uint8_t rhs_rows[ROWS]) {
    int row = 0;
    for (int combined = 0; combined < FULL_PATTERNS; combined++) {
        int lhs = combined & (HALF_PATTERNS - 1);
        int rhs = combined >> HALF_COLUMNS;
        for (int copy = 0; copy < table->cells[combined]; copy++) {
            lhs_rows[row] = (uint8_t)lhs;
            rhs_rows[row] = (uint8_t)rhs;
            row++;
        }
    }
}

static U128 compute_table_value(const RelativeTable* table, uint64_t* tests) {
    uint8_t lhs_rows[ROWS];
    uint8_t rhs_rows[ROWS];
    table_rows(table, lhs_rows, rhs_rows);
    Distribution lhs = build_distribution(lhs_rows, 0);
    Distribution rhs = build_distribution(rhs_rows, 0);
    Distribution lhs_complement = build_distribution(lhs_rows, 1);
    Distribution rhs_complement = build_distribution(rhs_rows, 1);
    uint64_t tests_left = 0;
    uint64_t tests_right = 0;
    uint64_t left = disjoint_join(&lhs, &rhs, &tests_left);
    uint64_t right = disjoint_join(&lhs_complement, &rhs_complement, &tests_right);
    *tests = tests_left + tests_right;
    free_distribution(&lhs);
    free_distribution(&rhs);
    free_distribution(&lhs_complement);
    free_distribution(&rhs_complement);
    return (U128)left * right;
}

static void evaluate_table(const RelativeTable* table) {
    g_candidate_tables++;
    uint64_t key[4];
    canonical_full_key(table->cells, key);
    int found = 0;
    KernelEntry* cache_entry = kernel_lookup(key, &found);
    if (found) {
        g_duplicate_kernels++;
        if (g_duplicate_validations < 32) {
            uint64_t ignored_tests = 0;
            U128 duplicate_value = compute_table_value(table, &ignored_tests);
            if (duplicate_value != cache_entry->value) {
                fprintf(stderr, "column-orbit kernel mismatch\n");
                exit(1);
            }
            g_duplicate_validations++;
        }
        g_checksum ^= cache_entry->value + table->group_multiplicity;
        return;
    }
    uint64_t tests = 0;
    U128 value = compute_table_value(table, &tests);
    g_pair_tests += tests;
    g_test_samples[g_test_sample_count++] = tests;
    memcpy(cache_entry->key, key, sizeof(cache_entry->key));
    cache_entry->value = value;
    cache_entry->used = 1;
    g_unique_kernels++;
    g_checksum ^= value + table->group_multiplicity;
}

static int u64_compare(const void* lhs_ptr, const void* rhs_ptr) {
    uint64_t lhs = *(const uint64_t*)lhs_ptr;
    uint64_t rhs = *(const uint64_t*)rhs_ptr;
    return lhs < rhs ? -1 : lhs > rhs;
}

int main(int argc, char** argv) {
    size_t target_kernels = argc > 1 ? strtoull(argv[1], NULL, 10) : 1000;
    if (!target_kernels) {
        fprintf(stderr, "Usage: %s UNIQUE_KERNELS\n", argv[0]);
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    g_factorial[0] = 1;
    for (int i = 1; i <= ROWS; i++) g_factorial[i] = g_factorial[i - 1] * (uint64_t)i;
    int pair = 0;
    for (int u = 0; u < ROWS; u++) {
        for (int v = u + 1; v < ROWS; v++) g_pair_index[u][v] = pair++;
    }
    generate_half_permutations(0, 0);
    generate_full_permutations(0, 0);
    enumerate_prefix_histograms(0, ROWS);
    g_kernel_capacity = 1;
    while (g_kernel_capacity < target_kernels * 4U) g_kernel_capacity <<= 1;
    g_kernel_cache = calloc(g_kernel_capacity, sizeof(g_kernel_cache[0]));
    g_test_samples = calloc(target_kernels, sizeof(g_test_samples[0]));
    if (!g_kernel_cache || !g_test_samples) return 1;
    uint64_t rng = UINT64_C(0x243f6a8885a308d3);
    uint64_t completed_pairs = 0;
    uint64_t double_cosets = 0;
    size_t next_progress = 1000;
    double start = seconds_now();
    while (g_unique_kernels < target_kernels) {
        rng = mix64(rng);
        int lhs_index = g_viable_indices[rng % (uint64_t)g_viable_count];
        rng = mix64(rng);
        int rhs_index = g_viable_indices[rng % (uint64_t)g_viable_count];
        generate_relative_tables(&g_prefixes[lhs_index], &g_prefixes[rhs_index]);
        completed_pairs++;
        double_cosets += (uint64_t)g_table_count;
        if (g_table_count) {
            size_t start_table = (size_t)(rng % (uint64_t)g_table_count);
            for (int offset = 0; offset < g_table_count && g_unique_kernels < target_kernels;
                 offset++) {
                int table = (int)((start_table + (size_t)offset) % (size_t)g_table_count);
                evaluate_table(&g_tables[table]);
            }
        }
        if (g_unique_kernels >= next_progress) {
            printf("progress unique=%zu candidates=%llu pairs=%llu elapsed=%.2fs\n",
                   g_unique_kernels, (unsigned long long)g_candidate_tables,
                   (unsigned long long)completed_pairs, seconds_now() - start);
            next_progress = (g_unique_kernels / 1000U + 1U) * 1000U;
        }
    }
    double seconds = seconds_now() - start;
    qsort(g_test_samples, g_test_sample_count, sizeof(g_test_samples[0]), u64_compare);
    double projected_hours = seconds / (double)g_unique_kernels * 17256831.0 / 3600.0;
    printf("prefix_orbits=%d viable=%d sampled_pairs=%llu enumerated_double_cosets=%llu\n",
           g_prefix_count, g_viable_count, (unsigned long long)completed_pairs,
           (unsigned long long)double_cosets);
    printf("sampling=uniform_prefix_pairs_then_all_double_cosets (not uniform full orbits)\n");
    printf("unique_kernels=%zu candidates=%llu duplicates=%llu duplicate_rate=%.2f%% duplicate_checks=%llu\n",
           g_unique_kernels, (unsigned long long)g_candidate_tables,
           (unsigned long long)g_duplicate_kernels,
           100.0 * (double)g_duplicate_kernels / (double)g_candidate_tables,
           (unsigned long long)g_duplicate_validations);
    printf("pair_tests total=%llu median=%llu p90=%llu p99=%llu max=%llu\n",
           (unsigned long long)g_pair_tests,
           (unsigned long long)g_test_samples[g_test_sample_count / 2U],
           (unsigned long long)g_test_samples[g_test_sample_count * 9U / 10U],
           (unsigned long long)g_test_samples[g_test_sample_count * 99U / 100U],
           (unsigned long long)g_test_samples[g_test_sample_count - 1U]);
    printf("time=%.3fs seconds_per_unique=%.6f sampled_projected_8x6_core_hours=%.1f transitions=%llu checksum=",
           seconds, seconds / (double)g_unique_kernels, projected_hours,
           (unsigned long long)g_distribution_transitions);
    print_u128(g_checksum);
    printf("\n");
    free(g_kernel_cache);
    free(g_test_samples);
    return g_prefix_count == PREFIX_ORBITS && g_viable_count == PREFIX_ORBITS - 8 ? 0 : 1;
}
