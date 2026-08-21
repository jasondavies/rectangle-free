/* Exact two-bit contraction over S8 x Sd binary-mask orbits, d <= 5. */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned __int128 U128;

enum { ROWS = 8, MAX_COLUMNS = 5, MAX_PATTERNS = 32, MAX_PERMUTATIONS = 120 };

typedef struct {
    uint32_t increment;
    uint8_t weight;
} RowOption;

static int g_columns;
static int g_patterns;
static int g_pairs;
static size_t g_state_count;
static int g_permutations[MAX_PERMUTATIONS][MAX_COLUMNS];
static int g_current_permutation[MAX_COLUMNS];
static int g_preimages[MAX_PERMUTATIONS][MAX_PATTERNS];
static int g_permutation_count;
static unsigned g_histogram[MAX_PATTERNS];
static RowOption g_options[MAX_PATTERNS][32];
static int g_option_counts[MAX_PATTERNS];
static uint64_t g_factorial[ROWS + 1];
static uint64_t* g_values[2];
static uint32_t* g_active[2];
static uint64_t g_orbit_count;
static uint64_t g_kernel_count;
static uint64_t g_sample_stride;
static uint64_t g_dp_transitions;
static size_t g_max_active;
static U128 g_total;
static double g_start;

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

static void generate_permutations(int depth, unsigned used) {
    if (depth == g_columns) {
        for (int pattern = 0; pattern < g_patterns; pattern++) {
            int image = 0;
            for (int bit = 0; bit < g_columns; bit++) {
                if ((pattern >> bit) & 1) {
                    image |= 1 << g_current_permutation[bit];
                }
            }
            g_preimages[g_permutation_count][image] = pattern;
        }
        for (int bit = 0; bit < g_columns; bit++) {
            g_permutations[g_permutation_count][bit] = g_current_permutation[bit];
        }
        g_permutation_count++;
        return;
    }
    for (int bit = 0; bit < g_columns; bit++) {
        if ((used >> bit) & 1U) continue;
        g_current_permutation[depth] = bit;
        generate_permutations(depth + 1, used | (1U << bit));
    }
}

static int transformed_compare(int permutation) {
    for (int pattern = 0; pattern < g_patterns; pattern++) {
        unsigned transformed = g_histogram[g_preimages[permutation][pattern]];
        if (transformed < g_histogram[pattern]) return -1;
        if (transformed > g_histogram[pattern]) return 1;
    }
    return 0;
}

static int is_canonical(void) {
    for (int permutation = 0; permutation < g_permutation_count; permutation++) {
        if (transformed_compare(permutation) < 0) return 0;
    }
    return 1;
}

static void build_row_options(void) {
    int pair_index[MAX_COLUMNS][MAX_COLUMNS] = {{0}};
    int pair = 0;
    for (int u = 0; u < g_columns; u++) {
        for (int v = u + 1; v < g_columns; v++) pair_index[u][v] = pair++;
    }
    for (int pattern = 0; pattern < g_patterns; pattern++) {
        unsigned assignment = (unsigned)pattern;
        for (;;) {
            uint32_t increment = 0;
            for (int u = 0; u < g_columns; u++) {
                for (int v = u + 1; v < g_columns; v++) {
                    if (((pattern >> u) & 1) == 0 || ((pattern >> v) & 1) == 0) continue;
                    unsigned cu = (assignment >> u) & 1U;
                    unsigned cv = (assignment >> v) & 1U;
                    if (cu == cv) increment |= 1U << (cu * g_pairs + pair_index[u][v]);
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
    uint64_t* current = g_values[0];
    uint64_t* next = g_values[1];
    uint32_t* active = g_active[0];
    uint32_t* next_active = g_active[1];
    size_t active_count = 1;
    active[0] = 0;
    current[0] = 1;
    for (int pattern = 0; pattern < g_patterns; pattern++) {
        for (unsigned copy = 0; copy < histogram[pattern]; copy++) {
            size_t next_count = 0;
            for (size_t i = 0; i < active_count; i++) {
                uint32_t state = active[i];
                uint64_t state_weight = current[state];
                for (int option = 0; option < g_option_counts[pattern]; option++) {
                    RowOption row_option = g_options[pattern][option];
                    if (state & row_option.increment) continue;
                    uint32_t combined = state | row_option.increment;
                    if (next[combined] == 0) next_active[next_count++] = combined;
                    next[combined] += state_weight * row_option.weight;
                    g_dp_transitions++;
                }
            }
            for (size_t i = 0; i < active_count; i++) current[active[i]] = 0;
            uint64_t* value_swap = current;
            current = next;
            next = value_swap;
            uint32_t* active_swap = active;
            active = next_active;
            next_active = active_swap;
            active_count = next_count;
            if (active_count > g_max_active) g_max_active = active_count;
        }
    }
    uint64_t total = 0;
    for (size_t i = 0; i < active_count; i++) {
        total += current[active[i]];
        current[active[i]] = 0;
    }
    return total;
}

static void record_orbit(void) {
    g_orbit_count++;
    if ((g_orbit_count - 1U) % g_sample_stride != 0) return;
    unsigned complement[MAX_PATTERNS] = {0};
    for (int pattern = 0; pattern < g_patterns; pattern++) {
        complement[pattern] = g_histogram[pattern ^ (g_patterns - 1)];
    }
    uint64_t left = count_c2(g_histogram);
    uint64_t right = count_c2(complement);
    uint64_t row_factor = 1;
    for (int pattern = 0; pattern < g_patterns; pattern++) {
        row_factor *= g_factorial[g_histogram[pattern]];
    }
    uint64_t column_stabilizer = 0;
    for (int permutation = 0; permutation < g_permutation_count; permutation++) {
        if (transformed_compare(permutation) == 0) column_stabilizer++;
    }
    uint64_t orbit_size = g_factorial[ROWS] * (uint64_t)g_permutation_count /
                          (row_factor * column_stabilizer);
    g_total += (U128)orbit_size * left * right;
    g_kernel_count++;
    if (g_kernel_count % 10000U == 0) {
        printf("progress kernels=%llu orbits=%llu elapsed=%.2fs transitions=%llu\n",
               (unsigned long long)g_kernel_count, (unsigned long long)g_orbit_count,
               seconds_now() - g_start, (unsigned long long)g_dp_transitions);
    }
}

static void enumerate_histograms(int pattern, int remaining) {
    if (pattern == g_patterns - 1) {
        g_histogram[pattern] = (unsigned)remaining;
        if (is_canonical()) record_orbit();
        return;
    }
    for (int count = 0; count <= remaining; count++) {
        g_histogram[pattern] = (unsigned)count;
        enumerate_histograms(pattern + 1, remaining - count);
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    g_columns = argc > 1 ? atoi(argv[1]) : 5;
    g_sample_stride = argc > 2 ? strtoull(argv[2], NULL, 10) : 1;
    if (g_columns < 1 || g_columns > MAX_COLUMNS || g_sample_stride == 0) {
        fprintf(stderr, "Usage: %s COLUMNS [SAMPLE_STRIDE], with 1 <= COLUMNS <= 5\n",
                argv[0]);
        return 2;
    }
    g_patterns = 1 << g_columns;
    g_pairs = g_columns * (g_columns - 1) / 2;
    g_state_count = (size_t)1U << (2 * g_pairs);
    g_factorial[0] = 1;
    for (int i = 1; i <= ROWS; i++) g_factorial[i] = g_factorial[i - 1] * (uint64_t)i;
    generate_permutations(0, 0);
    build_row_options();
    for (int i = 0; i < 2; i++) {
        g_values[i] = calloc(g_state_count, sizeof(g_values[i][0]));
        g_active[i] = malloc(g_state_count * sizeof(g_active[i][0]));
        if (!g_values[i] || !g_active[i]) return 1;
    }
    g_start = seconds_now();
    enumerate_histograms(0, ROWS);
    double seconds = seconds_now() - g_start;
    printf("rows=%d columns=%d orbits=%llu sampled_kernels=%llu stride=%llu\n", ROWS,
           g_columns, (unsigned long long)g_orbit_count,
           (unsigned long long)g_kernel_count, (unsigned long long)g_sample_stride);
    printf("sampled_weighted_total=");
    print_u128(g_total);
    printf(" transitions=%llu max_active=%zu time=%.3fs\n",
           (unsigned long long)g_dp_transitions, g_max_active, seconds);
    if (g_sample_stride == 1 && g_columns == 4) {
        U128 expected = (U128)UINT64_C(1855457222859010944);
        printf("expected_8x4=");
        print_u128(expected);
        printf(" %s\n", g_total == expected ? "OK" : "FAIL");
        if (g_total != expected) return 1;
    }
    if (g_sample_stride == 1 && g_columns == 5) {
        U128 expected = (U128)UINT64_C(2575233962780615894) * 10000U + 160U;
        printf("expected_8x5=");
        print_u128(expected);
        printf(" %s\n", g_total == expected ? "OK" : "FAIL");
        if (g_total != expected) return 1;
    }
    for (int i = 0; i < 2; i++) {
        free(g_values[i]);
        free(g_active[i]);
    }
    return 0;
}
