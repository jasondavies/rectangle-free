/* Exact asymmetric 3+4 two-bit kernel feasibility probe for T_4(7,7). */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned __int128 U128;

enum {
    ROWS = 7,
    LEFT_COLUMNS = 3,
    RIGHT_COLUMNS = 4,
    MAX_COLUMNS = 4,
    MAX_PATTERNS = 16,
    PAIRS = 21,
    LEFT_ORBITS = 734,
    RIGHT_ORBITS = 9343,
    ROW_PERMUTATIONS = 5040,
    SHAPES = 15
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
    uint8_t histogram[MAX_PATTERNS];
    size_t support[2];
    uint8_t shape;
} Prefix;

typedef struct {
    int columns;
    int patterns;
    int expected_orbits;
    int permutation_count;
    int permutation[MAX_COLUMNS];
    int preimages[24][MAX_PATTERNS];
    uint8_t histogram[MAX_PATTERNS];
    Prefix* prefixes;
    int prefix_count;
} PrefixSet;

typedef struct {
    uint16_t lhs;
    uint16_t rhs;
    uint64_t tests;
    uint16_t double_cosets;
} PairWork;

typedef struct {
    uint64_t word[4];
} TableKey;

typedef struct {
    TableKey key;
    uint16_t permutation;
} PermutationRecord;

static Prefix g_left_prefixes[LEFT_ORBITS];
static Prefix g_right_prefixes[RIGHT_ORBITS];
static int g_pair_index[ROWS][ROWS];
static uint8_t g_row_permutations[ROW_PERMUTATIONS][ROWS];
static uint8_t g_current_row_permutation[ROWS];
static int g_row_permutation_count;
static uint32_t g_shape_codes[SHAPES];
static uint8_t g_shape_rows[SHAPES][ROWS];
static int g_shape_count;
static uint16_t g_double_coset_counts[SHAPES][SHAPES];
static uint64_t g_distribution_transitions;

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

static Distribution build_distribution(const uint8_t rows[ROWS], int columns,
                                       int complement) {
    int patterns = 1 << columns;
    Map current;
    map_init(&current, 16);
    map_add(&current, 0, 1);
    for (int column = 0; column < columns; column++) {
        unsigned active_rows = 0;
        for (int row = 0; row < ROWS; row++) {
            unsigned pattern = complement ? (unsigned)rows[row] ^ (unsigned)(patterns - 1) :
                                            (unsigned)rows[row];
            if ((pattern >> column) & 1U) active_rows |= 1U << row;
        }
        Increment increments[128];
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
                    if (cu == cv) {
                        mask |= UINT64_C(1) << (cu * PAIRS + (unsigned)g_pair_index[u][v]);
                    }
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

static void free_distribution(Distribution* distribution) {
    free(distribution->entries);
    *distribution = (Distribution){0};
}

static uint64_t disjoint_join(const Distribution* lhs, const Distribution* rhs) {
    uint64_t total = 0;
    for (size_t i = 0; i < lhs->count; i++) {
        for (size_t j = 0; j < rhs->count; j++) {
            if ((lhs->entries[i].mask & rhs->entries[j].mask) == 0) {
                total += lhs->entries[i].weight * rhs->entries[j].weight;
            }
        }
    }
    return total;
}

static void generate_column_permutations(PrefixSet* set, int depth, unsigned used) {
    if (depth == set->columns) {
        for (int pattern = 0; pattern < set->patterns; pattern++) {
            int image = 0;
            for (int bit = 0; bit < set->columns; bit++) {
                if ((pattern >> bit) & 1) image |= 1 << set->permutation[bit];
            }
            set->preimages[set->permutation_count][image] = pattern;
        }
        set->permutation_count++;
        return;
    }
    for (int bit = 0; bit < set->columns; bit++) {
        if ((used >> bit) & 1U) continue;
        set->permutation[depth] = bit;
        generate_column_permutations(set, depth + 1, used | (1U << bit));
    }
}

static int histogram_is_canonical(const PrefixSet* set) {
    for (int permutation = 0; permutation < set->permutation_count; permutation++) {
        for (int pattern = 0; pattern < set->patterns; pattern++) {
            unsigned transformed = set->histogram[set->preimages[permutation][pattern]];
            if (transformed < set->histogram[pattern]) return 0;
            if (transformed > set->histogram[pattern]) break;
        }
    }
    return 1;
}

static void enumerate_histograms(PrefixSet* set, int pattern, int remaining) {
    if (pattern == set->patterns - 1) {
        set->histogram[pattern] = (uint8_t)remaining;
        if (histogram_is_canonical(set)) {
            if (set->prefix_count == set->expected_orbits) exit(1);
            memcpy(set->prefixes[set->prefix_count++].histogram, set->histogram,
                   sizeof(set->histogram));
        }
        return;
    }
    for (int count = 0; count <= remaining; count++) {
        set->histogram[pattern] = (uint8_t)count;
        enumerate_histograms(set, pattern + 1, remaining - count);
    }
}

static void histogram_rows(const uint8_t histogram[MAX_PATTERNS], int patterns,
                           uint8_t rows[ROWS]) {
    int row = 0;
    for (int pattern = 0; pattern < patterns; pattern++) {
        for (int copy = 0; copy < histogram[pattern]; copy++) rows[row++] = (uint8_t)pattern;
    }
}

static uint8_t histogram_shape(const uint8_t histogram[MAX_PATTERNS], int patterns) {
    uint8_t parts[ROWS];
    int count = 0;
    for (int pattern = 0; pattern < patterns; pattern++) {
        if (histogram[pattern]) parts[count++] = histogram[pattern];
    }
    for (int i = 1; i < count; i++) {
        uint8_t value = parts[i];
        int j = i;
        while (j && parts[j - 1] < value) {
            parts[j] = parts[j - 1];
            j--;
        }
        parts[j] = value;
    }
    uint32_t code = 0;
    for (int i = 0; i < count; i++) code = code * 8U + parts[i];
    for (int shape = 0; shape < g_shape_count; shape++) {
        if (g_shape_codes[shape] == code) return (uint8_t)shape;
    }
    if (g_shape_count == SHAPES) exit(1);
    int row = 0;
    for (int group = 0; group < count; group++) {
        for (int copy = 0; copy < parts[group]; copy++) {
            g_shape_rows[g_shape_count][row++] = (uint8_t)group;
        }
    }
    g_shape_codes[g_shape_count] = code;
    return (uint8_t)g_shape_count++;
}

static void generate_row_permutations(unsigned depth, unsigned used) {
    if (depth >= ROWS) {
        if (depth != ROWS) exit(1);
        memcpy(g_row_permutations[g_row_permutation_count++], g_current_row_permutation,
               sizeof(g_current_row_permutation));
        return;
    }
    for (int row = 0; row < ROWS; row++) {
        if ((used >> row) & 1U) continue;
        g_current_row_permutation[depth] = (uint8_t)row;
        generate_row_permutations(depth + 1U, used | (1U << row));
    }
}

static int table_key_compare(const void* lhs_ptr, const void* rhs_ptr) {
    const TableKey* lhs = lhs_ptr;
    const TableKey* rhs = rhs_ptr;
    for (int i = 0; i < 4; i++) {
        if (lhs->word[i] < rhs->word[i]) return -1;
        if (lhs->word[i] > rhs->word[i]) return 1;
    }
    return 0;
}

static int permutation_record_compare(const void* lhs_ptr, const void* rhs_ptr) {
    const PermutationRecord* lhs = lhs_ptr;
    const PermutationRecord* rhs = rhs_ptr;
    return table_key_compare(&lhs->key, &rhs->key);
}

static uint64_t permute_token_mask(uint64_t mask, int permutation) {
    uint8_t old_to_new[ROWS];
    for (int new_row = 0; new_row < ROWS; new_row++) {
        old_to_new[g_row_permutations[permutation][new_row]] = (uint8_t)new_row;
    }
    uint8_t pair_image[PAIRS];
    int old_pair = 0;
    for (int old_u = 0; old_u < ROWS; old_u++) {
        for (int old_v = old_u + 1; old_v < ROWS; old_v++) {
            int new_u = old_to_new[old_u];
            int new_v = old_to_new[old_v];
            if (new_u > new_v) {
                int swap = new_u;
                new_u = new_v;
                new_v = swap;
            }
            pair_image[old_pair++] = (uint8_t)g_pair_index[new_u][new_v];
        }
    }
    uint64_t result = 0;
    while (mask) {
        int bit = __builtin_ctzll(mask);
        int colour = bit / PAIRS;
        int pair = bit % PAIRS;
        result |= UINT64_C(1) << (colour * PAIRS + pair_image[pair]);
        mask &= mask - 1U;
    }
    return result;
}

static uint64_t disjoint_join_permuted(const Distribution* lhs,
                                       const Distribution* rhs,
                                       int permutation) {
    uint64_t* rhs_masks = calloc(rhs->count, sizeof(rhs_masks[0]));
    if (!rhs_masks && rhs->count) exit(1);
    for (size_t j = 0; j < rhs->count; j++) {
        rhs_masks[j] = permute_token_mask(rhs->entries[j].mask, permutation);
    }
    uint64_t total = 0;
    for (size_t i = 0; i < lhs->count; i++) {
        for (size_t j = 0; j < rhs->count; j++) {
            if ((lhs->entries[i].mask & rhs_masks[j]) == 0) {
                total += lhs->entries[i].weight * rhs->entries[j].weight;
            }
        }
    }
    free(rhs_masks);
    return total;
}

static uint16_t double_coset_count(uint8_t lhs_shape, uint8_t rhs_shape) {
    uint16_t cached = g_double_coset_counts[lhs_shape][rhs_shape];
    if (cached) return cached;
    TableKey* keys = calloc(ROW_PERMUTATIONS, sizeof(keys[0]));
    if (!keys) exit(1);
    for (int permutation = 0; permutation < ROW_PERMUTATIONS; permutation++) {
        uint8_t cells[ROWS * ROWS] = {0};
        for (int row = 0; row < ROWS; row++) {
            int lhs = g_shape_rows[lhs_shape][row];
            int rhs = g_shape_rows[rhs_shape][g_row_permutations[permutation][row]];
            cells[lhs * ROWS + rhs]++;
        }
        for (int cell = 0; cell < ROWS * ROWS; cell++) {
            keys[permutation].word[cell / 16] |=
                (uint64_t)cells[cell] << (4 * (cell % 16));
        }
    }
    qsort(keys, ROW_PERMUTATIONS, sizeof(keys[0]), table_key_compare);
    uint16_t count = 0;
    for (int i = 0; i < ROW_PERMUTATIONS; i++) {
        if (!i || table_key_compare(&keys[i - 1], &keys[i]) != 0) count++;
    }
    free(keys);
    g_double_coset_counts[lhs_shape][rhs_shape] = count;
    g_double_coset_counts[rhs_shape][lhs_shape] = count;
    return count;
}

static int size_compare(const void* lhs_ptr, const void* rhs_ptr) {
    size_t lhs = *(const size_t*)lhs_ptr;
    size_t rhs = *(const size_t*)rhs_ptr;
    return lhs < rhs ? -1 : lhs > rhs;
}

static int pair_work_compare(const void* lhs_ptr, const void* rhs_ptr) {
    const PairWork* lhs = lhs_ptr;
    const PairWork* rhs = rhs_ptr;
    return lhs->tests < rhs->tests ? -1 : lhs->tests > rhs->tests;
}

static double build_prefix_supports(Prefix* prefixes, int count, int columns,
                                    size_t support_values[2][RIGHT_ORBITS],
                                    int* viable_indices, int* viable_count) {
    int patterns = 1 << columns;
    double start = seconds_now();
    for (int i = 0; i < count; i++) {
        uint8_t rows[ROWS];
        histogram_rows(prefixes[i].histogram, patterns, rows);
        Distribution selected = build_distribution(rows, columns, 0);
        Distribution complement = build_distribution(rows, columns, 1);
        prefixes[i].support[0] = selected.count;
        prefixes[i].support[1] = complement.count;
        prefixes[i].shape = histogram_shape(prefixes[i].histogram, patterns);
        if (selected.count && complement.count) {
            support_values[0][*viable_count] = selected.count;
            support_values[1][*viable_count] = complement.count;
            viable_indices[(*viable_count)++] = i;
        }
        free_distribution(&selected);
        free_distribution(&complement);
    }
    return seconds_now() - start;
}

static void benchmark_pair(const PairWork* pair, const int* left_viable,
                           const int* right_viable, const char* label,
                           uint64_t max_join_tests, uint64_t* measured_tests,
                           double* measured_seconds) {
    int left_index = left_viable[pair->lhs];
    int right_index = right_viable[pair->rhs];
    uint8_t left_rows[ROWS];
    uint8_t right_base[ROWS];
    uint8_t right_rows[ROWS];
    histogram_rows(g_left_prefixes[left_index].histogram, 1 << LEFT_COLUMNS, left_rows);
    histogram_rows(g_right_prefixes[right_index].histogram, 1 << RIGHT_COLUMNS, right_base);
    int permutation = (int)(mix64((uint64_t)(uint32_t)left_index << 32 |
                                 (uint32_t)right_index) % ROW_PERMUTATIONS);
    for (int row = 0; row < ROWS; row++) {
        right_rows[row] = right_base[g_row_permutations[permutation][row]];
    }
    double build_start = seconds_now();
    Distribution left = build_distribution(left_rows, LEFT_COLUMNS, 0);
    Distribution right = build_distribution(right_rows, RIGHT_COLUMNS, 0);
    Distribution left_complement = build_distribution(left_rows, LEFT_COLUMNS, 1);
    Distribution right_complement = build_distribution(right_rows, RIGHT_COLUMNS, 1);
    double build_seconds = seconds_now() - build_start;
    uint64_t actual_tests = (uint64_t)left.count * right.count +
                            (uint64_t)left_complement.count * right_complement.count;
    if (actual_tests != pair->tests) exit(1);
    printf("join_%s prefixes=%d,%d predicted_tests=%llu double_cosets=%u build_time=%.6fs",
           label, left_index, right_index, (unsigned long long)pair->tests,
           pair->double_cosets, build_seconds);
    if (pair->tests <= max_join_tests) {
        double join_start = seconds_now();
        uint64_t selected = disjoint_join(&left, &right);
        uint64_t complement = disjoint_join(&left_complement, &right_complement);
        double join_seconds = seconds_now() - join_start;
        *measured_tests += pair->tests;
        *measured_seconds += join_seconds;
        printf(" join_time=%.6fs c2=%llu complement_c2=%llu value=", join_seconds,
               (unsigned long long)selected, (unsigned long long)complement);
        print_u128((U128)selected * complement);
    } else {
        printf(" join_time=SKIPPED(limit=%llu)", (unsigned long long)max_join_tests);
    }
    printf("\n");
    free_distribution(&left);
    free_distribution(&right);
    free_distribution(&left_complement);
    free_distribution(&right_complement);
}

static void benchmark_alignment_aggregate(const PairWork* pair,
                                          const int* left_viable,
                                          const int* right_viable,
                                          const char* label, int verify_all) {
    int left_index = left_viable[pair->lhs];
    int right_index = right_viable[pair->rhs];
    uint8_t left_rows[ROWS];
    uint8_t right_rows[ROWS];
    histogram_rows(g_left_prefixes[left_index].histogram, 1 << LEFT_COLUMNS,
                   left_rows);
    histogram_rows(g_right_prefixes[right_index].histogram, 1 << RIGHT_COLUMNS,
                   right_rows);
    Distribution left = build_distribution(left_rows, LEFT_COLUMNS, 0);
    Distribution right = build_distribution(right_rows, RIGHT_COLUMNS, 0);
    Distribution left_complement = build_distribution(left_rows, LEFT_COLUMNS, 1);
    Distribution right_complement = build_distribution(right_rows, RIGHT_COLUMNS, 1);
    PermutationRecord* records = calloc(ROW_PERMUTATIONS, sizeof(records[0]));
    if (!records) exit(1);
    for (int permutation = 0; permutation < ROW_PERMUTATIONS; permutation++) {
        records[permutation].permutation = (uint16_t)permutation;
        uint8_t cells[ROWS * ROWS] = {0};
        for (int row = 0; row < ROWS; row++) {
            int lhs = g_shape_rows[g_left_prefixes[left_index].shape][row];
            int rhs = g_shape_rows[g_right_prefixes[right_index].shape]
                                  [g_row_permutations[permutation][row]];
            cells[lhs * ROWS + rhs]++;
        }
        for (int cell = 0; cell < ROWS * ROWS; cell++) {
            records[permutation].key.word[cell / 16] |=
                (uint64_t)cells[cell] << (4 * (cell % 16));
        }
    }
    qsort(records, ROW_PERMUTATIONS, sizeof(records[0]),
          permutation_record_compare);
    double reduced_start = seconds_now();
    U128 reduced_sum = 0;
    int groups = 0;
    int begin = 0;
    while (begin < ROW_PERMUTATIONS) {
        int end = begin + 1;
        while (end < ROW_PERMUTATIONS &&
               table_key_compare(&records[begin].key, &records[end].key) == 0) end++;
        int permutation = records[begin].permutation;
        uint64_t selected = disjoint_join_permuted(&left, &right, permutation);
        uint64_t complement = disjoint_join_permuted(
            &left_complement, &right_complement, permutation);
        reduced_sum += (U128)(end - begin) * selected * complement;
        groups++;
        begin = end;
    }
    double reduced_seconds = seconds_now() - reduced_start;
    double all_seconds = 0;
    if (verify_all) {
        double all_start = seconds_now();
        U128 all_sum = 0;
        for (int permutation = 0; permutation < ROW_PERMUTATIONS; permutation++) {
            uint64_t selected = disjoint_join_permuted(&left, &right, permutation);
            uint64_t complement = disjoint_join_permuted(
                &left_complement, &right_complement, permutation);
            all_sum += (U128)selected * complement;
        }
        all_seconds = seconds_now() - all_start;
        if (all_sum != reduced_sum) {
            fprintf(stderr, "7x7 alignment aggregation mismatch\n");
            exit(1);
        }
    }
    if (groups != pair->double_cosets) {
        fprintf(stderr, "7x7 alignment aggregation mismatch\n");
        exit(1);
    }
    printf("alignment_%s prefixes=%d,%d permutations=%d double_cosets=%d reduction=%.2fx reduced_time=%.6fs verify_all=%s all_time=%.6fs value=",
           label, left_index, right_index, ROW_PERMUTATIONS, groups,
           (double)ROW_PERMUTATIONS / groups, reduced_seconds,
           verify_all ? "yes" : "no", all_seconds);
    print_u128(reduced_sum);
    printf("\n");
    free(records);
    free_distribution(&left);
    free_distribution(&right);
    free_distribution(&left_complement);
    free_distribution(&right_complement);
}

int main(int argc, char** argv) {
    uint64_t max_join_tests = argc > 1 ? strtoull(argv[1], NULL, 10) : UINT64_C(20000000000);
    int alignment_quantile = argc > 2 ? (int)strtol(argv[2], NULL, 10) : -1;
    int verify_all_alignments = argc > 3 ? atoi(argv[3]) : 1;
    if (!max_join_tests) {
        fprintf(stderr, "Usage: %s MAX_EXACT_JOIN_TESTS [ALIGNMENT_QUANTILE]\n", argv[0]);
        return 2;
    }
    if (alignment_quantile < -1 || alignment_quantile > 4) return 2;
    setvbuf(stdout, NULL, _IOLBF, 0);
    int pair = 0;
    for (int u = 0; u < ROWS; u++) {
        for (int v = u + 1; v < ROWS; v++) g_pair_index[u][v] = pair++;
    }
    generate_row_permutations(0, 0);
    PrefixSet left_set = {.columns = LEFT_COLUMNS,
                          .patterns = 1 << LEFT_COLUMNS,
                          .expected_orbits = LEFT_ORBITS,
                          .prefixes = g_left_prefixes};
    PrefixSet right_set = {.columns = RIGHT_COLUMNS,
                           .patterns = 1 << RIGHT_COLUMNS,
                           .expected_orbits = RIGHT_ORBITS,
                           .prefixes = g_right_prefixes};
    generate_column_permutations(&left_set, 0, 0);
    generate_column_permutations(&right_set, 0, 0);
    enumerate_histograms(&left_set, 0, ROWS);
    enumerate_histograms(&right_set, 0, ROWS);
    if (left_set.prefix_count != LEFT_ORBITS || right_set.prefix_count != RIGHT_ORBITS ||
        g_row_permutation_count != ROW_PERMUTATIONS) return 1;

    static size_t left_support_values[2][RIGHT_ORBITS];
    static size_t right_support_values[2][RIGHT_ORBITS];
    int* left_viable = malloc(LEFT_ORBITS * sizeof(left_viable[0]));
    int* right_viable = malloc(RIGHT_ORBITS * sizeof(right_viable[0]));
    if (!left_viable || !right_viable) return 1;
    int left_viable_count = 0;
    int right_viable_count = 0;
    double left_support_seconds =
        build_prefix_supports(g_left_prefixes, LEFT_ORBITS, LEFT_COLUMNS,
                              left_support_values, left_viable, &left_viable_count);
    double right_support_seconds =
        build_prefix_supports(g_right_prefixes, RIGHT_ORBITS, RIGHT_COLUMNS,
                              right_support_values, right_viable, &right_viable_count);
    for (int complement = 0; complement < 2; complement++) {
        qsort(left_support_values[complement], (size_t)left_viable_count,
              sizeof(left_support_values[complement][0]), size_compare);
        qsort(right_support_values[complement], (size_t)right_viable_count,
              sizeof(right_support_values[complement][0]), size_compare);
    }
    printf("prefix_orbits left=%d right=%d viable_left=%d viable_right=%d shapes=%d row_permutations=%d\n",
           left_set.prefix_count, right_set.prefix_count, left_viable_count,
           right_viable_count, g_shape_count, g_row_permutation_count);
    printf("support_build left=%.3fs right=%.3fs transitions=%llu\n", left_support_seconds,
           right_support_seconds, (unsigned long long)g_distribution_transitions);
    for (int side = 0; side < 2; side++) {
        size_t (*values)[RIGHT_ORBITS] = side ? right_support_values : left_support_values;
        int count = side ? right_viable_count : left_viable_count;
        for (int complement = 0; complement < 2; complement++) {
            /* Quantify the complete canonical cache, not only its support quantiles. */
            U128 support_total = 0;
            for (int index = 0; index < count; index++) {
                support_total += values[complement][index];
            }
            printf("support_%s_%s min=%zu median=%zu p90=%zu p99=%zu max=%zu\n",
                   side ? "right4" : "left3", complement ? "complement" : "selected",
                   values[complement][0], values[complement][count / 2],
                   values[complement][count * 9 / 10], values[complement][count * 99 / 100],
                   values[complement][count - 1]);
            printf("support_%s_%s_total=", side ? "right4" : "left3",
                   complement ? "complement" : "selected");
            print_u128(support_total);
            printf("\n");
        }
    }

    size_t pair_count = (size_t)left_viable_count * (size_t)right_viable_count;
    PairWork* work = malloc(pair_count * sizeof(work[0]));
    if (!work) return 1;
    U128 weighted_test_sum = 0;
    uint64_t total_weight = 0;
    size_t cursor = 0;
    for (int lhs = 0; lhs < left_viable_count; lhs++) {
        const Prefix* left = &g_left_prefixes[left_viable[lhs]];
        for (int rhs = 0; rhs < right_viable_count; rhs++) {
            const Prefix* right = &g_right_prefixes[right_viable[rhs]];
            uint64_t tests = (uint64_t)left->support[0] * right->support[0] +
                             (uint64_t)left->support[1] * right->support[1];
            uint16_t double_cosets = double_coset_count(left->shape, right->shape);
            work[cursor++] = (PairWork){.lhs = (uint16_t)lhs,
                                        .rhs = (uint16_t)rhs,
                                        .tests = tests,
                                        .double_cosets = double_cosets};
            weighted_test_sum += (U128)tests * double_cosets;
            total_weight += double_cosets;
        }
    }
    qsort(work, pair_count, sizeof(work[0]), pair_work_compare);
    size_t quantile_indices[5] = {0};
    const uint64_t quantile_numerators[5] = {0, 50, 90, 99, 100};
    uint64_t cumulative = 0;
    size_t quantile = 0;
    for (size_t i = 0; i < pair_count && quantile < 5; i++) {
        cumulative += work[i].double_cosets;
        while (quantile < 5 &&
               (quantile_numerators[quantile] == 0 ||
                (U128)cumulative * 100U >= (U128)total_weight * quantile_numerators[quantile])) {
            quantile_indices[quantile++] = i;
        }
    }
    quantile_indices[4] = pair_count - 1;
    double mean_tests = (double)weighted_test_sum / (double)total_weight;
    printf("pair_types=%zu double_coset_weight=%llu weighted_tests min=%llu median=%llu p90=%llu p99=%llu max=%llu mean=%.0f\n",
           pair_count, (unsigned long long)total_weight, (unsigned long long)work[0].tests,
           (unsigned long long)work[quantile_indices[1]].tests,
           (unsigned long long)work[quantile_indices[2]].tests,
           (unsigned long long)work[quantile_indices[3]].tests,
           (unsigned long long)work[quantile_indices[4]].tests, mean_tests);
    const char* labels[5] = {"min", "median", "p90", "p99", "max"};
    uint64_t measured_tests = 0;
    double measured_seconds = 0;
    for (int i = 0; i < 5; i++) {
        if (i && quantile_indices[i] == quantile_indices[i - 1]) continue;
        benchmark_pair(&work[quantile_indices[i]], left_viable, right_viable, labels[i],
                       max_join_tests, &measured_tests, &measured_seconds);
    }
    if (alignment_quantile >= 0) {
        benchmark_alignment_aggregate(&work[quantile_indices[alignment_quantile]],
                                      left_viable, right_viable,
                                      labels[alignment_quantile],
                                      verify_all_alignments);
    }
    if (measured_tests && measured_seconds > 0) {
        double throughput = (double)measured_tests / measured_seconds;
        double mean_build = left_support_seconds / left_set.prefix_count +
                            right_support_seconds / right_set.prefix_count;
        double mean_kernel = mean_build + mean_tests / throughput;
        double projected_core_hours = mean_kernel * 16821330.0 / 3600.0;
        printf("measured_join_tests=%llu measured_join_time=%.6fs throughput=%.0f_tests_per_s\n",
               (unsigned long long)measured_tests, measured_seconds, throughput);
        printf("mean_build=%.6fs projected_mean_kernel=%.6fs projected_7x7_core_hours=%.1f complement_paired_orbits=16821330\n",
               mean_build, mean_kernel, projected_core_hours);
    }
    free(work);
    free(left_viable);
    free(right_viable);
    return 0;
}
