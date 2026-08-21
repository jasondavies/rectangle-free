/* Exact 8x4 two-bit contraction through row/column orbits and 2+2 tables. */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned __int128 U128;

enum { ROWS = 8, COLUMNS = 4, PATTERNS = 16, PAIRS = 6, STATES = 1 << 12 };

typedef struct {
    uint16_t increment;
    uint8_t weight;
} RowOption;

typedef struct {
    uint64_t key;
    U128 value;
    uint8_t used;
} CacheEntry;

static int g_permutations[24][COLUMNS];
static int g_current_permutation[COLUMNS];
static int g_permutation_count;
static RowOption g_options[PATTERNS][16];
static int g_option_counts[PATTERNS];
static uint64_t g_factorial[ROWS + 1];
static CacheEntry* g_cache;
static size_t g_cache_capacity;
static size_t g_cache_count;
static uint64_t g_dp_transitions;
static uint64_t g_histograms;
static uint64_t g_canonical_histograms;
static U128 g_row_orbit_total;
static U128 g_full_orbit_total;

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

static void generate_permutations(int depth, unsigned used) {
    if (depth == COLUMNS) {
        for (int bit = 0; bit < COLUMNS; bit++) {
            g_permutations[g_permutation_count][bit] = g_current_permutation[bit];
        }
        g_permutation_count++;
        return;
    }
    for (int bit = 0; bit < COLUMNS; bit++) {
        if ((used >> bit) & 1U) continue;
        g_current_permutation[depth] = bit;
        generate_permutations(depth + 1, used | (1U << bit));
    }
}

static int pattern_image(int pattern, int permutation) {
    int image = 0;
    for (int bit = 0; bit < COLUMNS; bit++) {
        if ((pattern >> bit) & 1) image |= 1 << g_permutations[permutation][bit];
    }
    return image;
}

static uint64_t pack_histogram(const unsigned* histogram) {
    uint64_t key = 0;
    for (int pattern = 0; pattern < PATTERNS; pattern++) {
        key |= (uint64_t)histogram[pattern] << (4 * pattern);
    }
    return key;
}

static void unpack_histogram(uint64_t key, unsigned* histogram) {
    for (int pattern = 0; pattern < PATTERNS; pattern++) {
        histogram[pattern] = (unsigned)((key >> (4 * pattern)) & 15U);
    }
}

static uint64_t transformed_key(const unsigned* histogram, int permutation) {
    unsigned transformed[PATTERNS] = {0};
    for (int pattern = 0; pattern < PATTERNS; pattern++) {
        transformed[pattern_image(pattern, permutation)] = histogram[pattern];
    }
    return pack_histogram(transformed);
}

static uint64_t canonical_key(const unsigned* histogram) {
    uint64_t best = UINT64_MAX;
    for (int permutation = 0; permutation < g_permutation_count; permutation++) {
        uint64_t key = transformed_key(histogram, permutation);
        if (key < best) best = key;
    }
    return best;
}

static void build_row_options(void) {
    int pair_index[COLUMNS][COLUMNS] = {{0}};
    int pair = 0;
    for (int u = 0; u < COLUMNS; u++) {
        for (int v = u + 1; v < COLUMNS; v++) pair_index[u][v] = pair++;
    }
    for (int pattern = 0; pattern < PATTERNS; pattern++) {
        unsigned assignment = (unsigned)pattern;
        for (;;) {
            uint16_t increment = 0;
            for (int u = 0; u < COLUMNS; u++) {
                for (int v = u + 1; v < COLUMNS; v++) {
                    if (((pattern >> u) & 1) == 0 || ((pattern >> v) & 1) == 0) continue;
                    unsigned cu = (assignment >> u) & 1U;
                    unsigned cv = (assignment >> v) & 1U;
                    if (cu == cv) increment |= (uint16_t)(1U << (cu * PAIRS + pair_index[u][v]));
                }
            }
            int option = 0;
            while (option < g_option_counts[pattern] &&
                   g_options[pattern][option].increment != increment) option++;
            if (option == g_option_counts[pattern]) {
                g_options[pattern][option] =
                    (RowOption){.increment = increment, .weight = 1};
                g_option_counts[pattern]++;
            } else {
                g_options[pattern][option].weight++;
            }
            if (assignment == 0) break;
            assignment = (assignment - 1U) & (unsigned)pattern;
        }
    }
}

static uint64_t count_c2(const unsigned* histogram) {
    uint64_t values[2][STATES] = {{0}};
    uint64_t* current = values[0];
    uint64_t* next = values[1];
    uint16_t active[STATES];
    uint16_t next_active[STATES];
    size_t active_count = 1;
    active[0] = 0;
    current[0] = 1;
    for (int pattern = 0; pattern < PATTERNS; pattern++) {
        for (unsigned copy = 0; copy < histogram[pattern]; copy++) {
            size_t next_count = 0;
            for (size_t i = 0; i < active_count; i++) {
                uint16_t state = active[i];
                uint64_t weight = current[state];
                for (int option = 0; option < g_option_counts[pattern]; option++) {
                    RowOption row_option = g_options[pattern][option];
                    if (state & row_option.increment) continue;
                    uint16_t combined = (uint16_t)(state | row_option.increment);
                    if (next[combined] == 0) next_active[next_count++] = combined;
                    next[combined] += weight * row_option.weight;
                    g_dp_transitions++;
                }
            }
            for (size_t i = 0; i < active_count; i++) current[active[i]] = 0;
            uint64_t* swap_values = current;
            current = next;
            next = swap_values;
            memcpy(active, next_active, next_count * sizeof(active[0]));
            active_count = next_count;
        }
    }
    uint64_t total = 0;
    for (size_t i = 0; i < active_count; i++) total += current[active[i]];
    return total;
}

static U128 evaluate_key(uint64_t key) {
    size_t slot = (size_t)mix64(key) & (g_cache_capacity - 1U);
    while (g_cache[slot].used) {
        if (g_cache[slot].key == key) return g_cache[slot].value;
        slot = (slot + 1U) & (g_cache_capacity - 1U);
    }
    unsigned histogram[PATTERNS];
    unsigned complement[PATTERNS];
    unpack_histogram(key, histogram);
    for (int pattern = 0; pattern < PATTERNS; pattern++) {
        complement[pattern] = histogram[pattern ^ (PATTERNS - 1)];
    }
    uint64_t left = count_c2(histogram);
    uint64_t right = count_c2(complement);
    U128 value = (U128)left * right;
    g_cache[slot] = (CacheEntry){.key = key, .value = value, .used = 1};
    g_cache_count++;
    return value;
}

static void process_histogram(const unsigned* histogram) {
    uint64_t own_key = pack_histogram(histogram);
    uint64_t key = canonical_key(histogram);
    U128 value = evaluate_key(key);
    uint64_t row_factor = 1;
    for (int pattern = 0; pattern < PATTERNS; pattern++) {
        row_factor *= g_factorial[histogram[pattern]];
    }
    uint64_t row_orbit_size = g_factorial[ROWS] / row_factor;
    g_row_orbit_total += (U128)row_orbit_size * value;
    g_histograms++;
    if (own_key == key) {
        uint64_t column_stabilizer = 0;
        for (int permutation = 0; permutation < g_permutation_count; permutation++) {
            if (transformed_key(histogram, permutation) == own_key) column_stabilizer++;
        }
        uint64_t full_orbit_size = g_factorial[ROWS] * (uint64_t)g_permutation_count /
                                   (row_factor * column_stabilizer);
        g_full_orbit_total += (U128)full_orbit_size * value;
        g_canonical_histograms++;
    }
}

static unsigned g_histogram[PATTERNS];

static void enumerate_histograms(int pattern, int remaining) {
    if (pattern == PATTERNS - 1) {
        g_histogram[pattern] = (unsigned)remaining;
        process_histogram(g_histogram);
        return;
    }
    for (int count = 0; count <= remaining; count++) {
        g_histogram[pattern] = (unsigned)count;
        enumerate_histograms(pattern + 1, remaining - count);
    }
}

int main(void) {
    const U128 expected = (U128)UINT64_C(1855457222859010944);
    g_factorial[0] = 1;
    for (int i = 1; i <= ROWS; i++) g_factorial[i] = g_factorial[i - 1] * (uint64_t)i;
    generate_permutations(0, 0);
    build_row_options();
    g_cache_capacity = 1U << 16;
    g_cache = calloc(g_cache_capacity, sizeof(g_cache[0]));
    if (!g_cache) return 1;
    double start = seconds_now();
    enumerate_histograms(0, ROWS);
    double seconds = seconds_now() - start;
    printf("labelled-column row orbits=%llu full row/column orbits=%llu cached_kernels=%zu\n",
           (unsigned long long)g_histograms,
           (unsigned long long)g_canonical_histograms, g_cache_count);
    printf("2+2 contingency contraction=");
    print_u128(g_row_orbit_total);
    printf(" %s\n", g_row_orbit_total == expected ? "OK" : "FAIL");
    printf("full-orbit contraction=");
    print_u128(g_full_orbit_total);
    printf(" %s\n", g_full_orbit_total == expected ? "OK" : "FAIL");
    printf("expected=");
    print_u128(expected);
    printf(" dp_transitions=%llu time=%.3fs\n", (unsigned long long)g_dp_transitions,
           seconds);
    free(g_cache);
    return g_row_orbit_total == expected && g_full_orbit_total == expected ? 0 : 1;
}
