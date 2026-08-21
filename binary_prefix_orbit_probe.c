/* Enumerate binary 8xd matrices modulo row and column permutations. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { ROWS = 8, MAX_COLUMNS = 5, MAX_PATTERNS = 1 << MAX_COLUMNS, MAX_PERMUTATIONS = 120 };

static int g_columns;
static int g_patterns;
static unsigned g_histogram[MAX_PATTERNS];
static int g_permutations[MAX_PERMUTATIONS][MAX_COLUMNS];
static int g_pattern_images[MAX_PERMUTATIONS][MAX_PATTERNS];
static int g_pattern_preimages[MAX_PERMUTATIONS][MAX_PATTERNS];
static int g_current_permutation[MAX_COLUMNS];
static int g_permutation_count;
static uint64_t g_orbit_count;
static uint64_t g_weight_sum;
static uint64_t g_extended_orbit_count;
static uint64_t g_self_complementary;
static uint64_t g_projected_stabilizer_sum;
static uint64_t g_max_projected_stabilizer;
static uint64_t g_max_full_stabilizer;
static uint64_t g_factorial[ROWS + 1];

static void generate_permutations(int depth, unsigned used) {
    if (depth == g_columns) {
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

static int pattern_image(int pattern, int permutation) {
    return g_pattern_images[permutation][pattern];
}

static int transformed_compare(int permutation, int complement) {
    int all_bits = g_patterns - 1;
    for (int pattern = 0; pattern < g_patterns; pattern++) {
        int source = g_pattern_preimages[permutation][pattern];
        if (complement) source ^= all_bits;
        unsigned transformed = g_histogram[source];
        if (transformed < g_histogram[pattern]) return -1;
        if (transformed > g_histogram[pattern]) return 1;
    }
    return 0;
}

static int is_column_canonical(void) {
    for (int permutation = 0; permutation < g_permutation_count; permutation++) {
        if (transformed_compare(permutation, 0) < 0) return 0;
    }
    return 1;
}

static void record_orbit(void) {
    uint64_t row_factor = 1;
    for (int pattern = 0; pattern < g_patterns; pattern++) {
        row_factor *= g_factorial[g_histogram[pattern]];
    }
    uint64_t column_stabilizer = 0;
    uint64_t column_kernel = 0;
    int complement_smaller = 0;
    int complement_equal = 0;
    for (int permutation = 0; permutation < g_permutation_count; permutation++) {
        if (transformed_compare(permutation, 0) == 0) {
            column_stabilizer++;
            int fixes_support = 1;
            for (int pattern = 0; pattern < g_patterns; pattern++) {
                if (g_histogram[pattern] && pattern_image(pattern, permutation) != pattern) {
                    fixes_support = 0;
                    break;
                }
            }
            if (fixes_support) column_kernel++;
        }
        int complement_cmp = transformed_compare(permutation, 1);
        if (complement_cmp < 0) complement_smaller = 1;
        if (complement_cmp == 0) complement_equal = 1;
    }
    uint64_t full_stabilizer = row_factor * column_stabilizer;
    uint64_t projected_row_stabilizer = full_stabilizer / column_kernel;
    uint64_t orbit_size = g_factorial[ROWS] * (uint64_t)g_permutation_count /
                          full_stabilizer;
    g_orbit_count++;
    g_weight_sum += orbit_size;
    g_projected_stabilizer_sum += projected_row_stabilizer;
    if (projected_row_stabilizer > g_max_projected_stabilizer) {
        g_max_projected_stabilizer = projected_row_stabilizer;
    }
    if (full_stabilizer > g_max_full_stabilizer) g_max_full_stabilizer = full_stabilizer;
    if (!complement_smaller) g_extended_orbit_count++;
    if (complement_equal) g_self_complementary++;
}

static void enumerate_histograms(int pattern, int remaining) {
    if (pattern == g_patterns - 1) {
        g_histogram[pattern] = (unsigned)remaining;
        if (is_column_canonical()) record_orbit();
        return;
    }
    for (int count = 0; count <= remaining; count++) {
        g_histogram[pattern] = (unsigned)count;
        enumerate_histograms(pattern + 1, remaining - count);
    }
}

static uint64_t power_u64(uint64_t base, int exponent) {
    uint64_t result = 1;
    while (exponent--) result *= base;
    return result;
}

int main(int argc, char** argv) {
    g_columns = argc > 1 ? atoi(argv[1]) : 4;
    if (g_columns < 1 || g_columns > MAX_COLUMNS) {
        fprintf(stderr, "Usage: %s COLUMNS, with 1 <= COLUMNS <= 5\n", argv[0]);
        return 2;
    }
    g_patterns = 1 << g_columns;
    g_factorial[0] = 1;
    for (int i = 1; i <= ROWS; i++) g_factorial[i] = g_factorial[i - 1] * (uint64_t)i;
    generate_permutations(0, 0);
    for (int permutation = 0; permutation < g_permutation_count; permutation++) {
        for (int pattern = 0; pattern < g_patterns; pattern++) {
            int image = 0;
            for (int bit = 0; bit < g_columns; bit++) {
                if ((pattern >> bit) & 1) {
                    image |= 1 << g_permutations[permutation][bit];
                }
            }
            g_pattern_images[permutation][pattern] = image;
            g_pattern_preimages[permutation][image] = pattern;
        }
    }
    enumerate_histograms(0, ROWS);
    uint64_t expected = power_u64(2, ROWS * g_columns);
    printf("rows=%d columns=%d column_permutations=%d\n", ROWS, g_columns,
           g_permutation_count);
    printf("orbits=%llu labelled_weight=%llu expected=%llu %s\n",
           (unsigned long long)g_orbit_count, (unsigned long long)g_weight_sum,
           (unsigned long long)expected, g_weight_sum == expected ? "OK" : "FAIL");
    printf("complement_orbits=%llu self_complementary=%llu\n",
           (unsigned long long)g_extended_orbit_count,
           (unsigned long long)g_self_complementary);
    printf("projected_row_stabilizer_avg=%.3f max=%llu full_stabilizer_max=%llu\n",
           (double)g_projected_stabilizer_sum / (double)g_orbit_count,
           (unsigned long long)g_max_projected_stabilizer,
           (unsigned long long)g_max_full_stabilizer);
    return g_weight_sum == expected ? 0 : 1;
}
