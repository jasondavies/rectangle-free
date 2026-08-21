/* Measure exact two-colour token distributions for binary 8x3 mask orbits. */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned __int128 U128;

enum { ROWS = 8, COLUMNS = 3, PATTERNS = 8, PAIRS = 28 };
enum { KERNEL_SAMPLE = 24 };

typedef struct {
    uint64_t mask;
    uint64_t weight;
    uint8_t used;
} Entry;

typedef struct {
    Entry* entries;
    size_t capacity;
    size_t count;
} Map;

typedef struct {
    uint64_t mask;
    uint16_t weight;
} Increment;

static int g_permutations[6][COLUMNS];
static int g_current_permutation[COLUMNS];
static int g_preimages[6][PATTERNS];
static int g_permutation_count;
static unsigned g_histogram[PATTERNS];
static size_t g_supports[2 * 1324];
static size_t g_support_count;
static uint64_t g_orbits;
static uint64_t g_transitions;
static size_t g_max_support;
static unsigned g_max_histogram[PATTERNS];
static int g_max_complement;
static int g_pair_index[ROWS][ROWS];
static unsigned g_sample_histograms[KERNEL_SAMPLE][PATTERNS];
static int g_sample_count;
static uint64_t g_dead_orbits;

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
    map->entries[slot] = (Entry){.mask = mask, .weight = weight, .used = 1};
    map->count++;
}

static void map_rehash(Map* map) {
    Entry* old = map->entries;
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

static void generate_permutations(int depth, unsigned used) {
    if (depth == COLUMNS) {
        for (int pattern = 0; pattern < PATTERNS; pattern++) {
            int image = 0;
            for (int bit = 0; bit < COLUMNS; bit++) {
                if ((pattern >> bit) & 1) image |= 1 << g_current_permutation[bit];
            }
            g_preimages[g_permutation_count][image] = pattern;
        }
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

static int is_canonical(void) {
    for (int permutation = 0; permutation < g_permutation_count; permutation++) {
        for (int pattern = 0; pattern < PATTERNS; pattern++) {
            unsigned transformed = g_histogram[g_preimages[permutation][pattern]];
            if (transformed < g_histogram[pattern]) return 0;
            if (transformed > g_histogram[pattern]) break;
        }
    }
    return 1;
}

static int increment_compare(const void* lhs_ptr, const void* rhs_ptr) {
    const Increment* lhs = lhs_ptr;
    const Increment* rhs = rhs_ptr;
    return lhs->mask < rhs->mask ? -1 : lhs->mask > rhs->mask;
}

static Map build_distribution(const unsigned* histogram, int complement) {
    unsigned row_patterns[ROWS];
    int row = 0;
    for (int pattern = 0; pattern < PATTERNS; pattern++) {
        for (unsigned copy = 0; copy < histogram[pattern]; copy++) {
            row_patterns[row++] = complement ? (unsigned)(pattern ^ (PATTERNS - 1)) :
                                              (unsigned)pattern;
        }
    }
    Map current;
    map_init(&current, 16);
    map_add(&current, 0, 1);
    for (int column = 0; column < COLUMNS; column++) {
        unsigned active_rows = 0;
        for (int r = 0; r < ROWS; r++) {
            if ((row_patterns[r] >> column) & 1U) active_rows |= 1U << r;
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
            if (assignment == 0) break;
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
                g_transitions++;
            }
        }
        free(current.entries);
        current = next;
    }
    return current;
}

static void record_orbit(void) {
    g_orbits++;
    size_t orbit_supports[2];
    for (int complement = 0; complement < 2; complement++) {
        Map distribution = build_distribution(g_histogram, complement);
        size_t support = distribution.count;
        orbit_supports[complement] = support;
        g_supports[g_support_count++] = support;
        if (support > g_max_support) {
            g_max_support = support;
            memcpy(g_max_histogram, g_histogram, sizeof(g_max_histogram));
            g_max_complement = complement;
        }
        free(distribution.entries);
    }
    if (!orbit_supports[0] || !orbit_supports[1]) {
        g_dead_orbits++;
    } else if (g_sample_count < KERNEL_SAMPLE &&
               g_orbits - 1U >= (uint64_t)55U * (uint64_t)g_sample_count) {
        memcpy(g_sample_histograms[g_sample_count++], g_histogram, sizeof(g_histogram));
    }
}

typedef struct {
    Entry* entries;
    size_t count;
} CompactDistribution;

static CompactDistribution compact_distribution(const unsigned* histogram, int complement) {
    Map map = build_distribution(histogram, complement);
    Entry* entries = calloc(map.count, sizeof(entries[0]));
    if (!entries) exit(1);
    size_t count = 0;
    for (size_t i = 0; i < map.capacity; i++) {
        if (map.entries[i].used) entries[count++] = map.entries[i];
    }
    free(map.entries);
    return (CompactDistribution){.entries = entries, .count = count};
}

static uint64_t compact_join(const CompactDistribution* lhs,
                             const CompactDistribution* rhs, uint64_t* pair_tests) {
    uint64_t total = 0;
    for (size_t i = 0; i < lhs->count; i++) {
        for (size_t j = 0; j < rhs->count; j++) {
            (*pair_tests)++;
            if ((lhs->entries[i].mask & rhs->entries[j].mask) == 0) {
                total += lhs->entries[i].weight * rhs->entries[j].weight;
            }
        }
    }
    return total;
}

static uint64_t mod_power(uint64_t base, uint64_t exponent, uint64_t modulus) {
    uint64_t result = 1;
    while (exponent) {
        if (exponent & 1U) result = (uint64_t)((U128)result * base % modulus);
        base = (uint64_t)((U128)base * base % modulus);
        exponent >>= 1;
    }
    return result;
}

static int kernel_rank(const U128 matrix[KERNEL_SAMPLE][KERNEL_SAMPLE], int dimension,
                       uint64_t modulus) {
    uint64_t work[KERNEL_SAMPLE][KERNEL_SAMPLE];
    for (int row = 0; row < dimension; row++) {
        for (int column = 0; column < dimension; column++) {
            work[row][column] = (uint64_t)(matrix[row][column] % modulus);
        }
    }
    int rank = 0;
    for (int column = 0; column < dimension && rank < dimension; column++) {
        int pivot = rank;
        while (pivot < dimension && work[pivot][column] == 0) pivot++;
        if (pivot == dimension) continue;
        if (pivot != rank) {
            for (int c = column; c < dimension; c++) {
                uint64_t swap = work[rank][c];
                work[rank][c] = work[pivot][c];
                work[pivot][c] = swap;
            }
        }
        uint64_t inverse = mod_power(work[rank][column], modulus - 2U, modulus);
        for (int c = column; c < dimension; c++) {
            work[rank][c] = (uint64_t)((U128)work[rank][c] * inverse % modulus);
        }
        for (int row = rank + 1; row < dimension; row++) {
            uint64_t factor = work[row][column];
            if (!factor) continue;
            for (int c = column; c < dimension; c++) {
                uint64_t product = (uint64_t)((U128)factor * work[rank][c] % modulus);
                work[row][c] = work[row][c] >= product ? work[row][c] - product :
                                                       work[row][c] + modulus - product;
            }
        }
        rank++;
    }
    return rank;
}

static void probe_kernel_rank(void) {
    CompactDistribution selected[KERNEL_SAMPLE];
    CompactDistribution complements[KERNEL_SAMPLE];
    size_t selected_support = 0;
    size_t complement_support = 0;
    for (int i = 0; i < g_sample_count; i++) {
        selected[i] = compact_distribution(g_sample_histograms[i], 0);
        complements[i] = compact_distribution(g_sample_histograms[i], 1);
        selected_support += selected[i].count;
        complement_support += complements[i].count;
    }
    U128 matrix[KERNEL_SAMPLE][KERNEL_SAMPLE] = {{0}};
    uint64_t pair_tests = 0;
    double start = seconds_now();
    for (int i = 0; i < g_sample_count; i++) {
        for (int j = i; j < g_sample_count; j++) {
            uint64_t left = compact_join(&selected[i], &selected[j], &pair_tests);
            uint64_t right = compact_join(&complements[i], &complements[j], &pair_tests);
            matrix[i][j] = matrix[j][i] = (U128)left * right;
        }
    }
    double seconds = seconds_now() - start;
    int rank1 = kernel_rank(matrix, g_sample_count, UINT64_C(1000000007));
    int rank2 = kernel_rank(matrix, g_sample_count, UINT64_C(1000000009));
    printf("aligned_kernel_sample=%d selected_support=%zu complement_support=%zu pair_tests=%llu time=%.3fs ranks_mod_1000000007=%d mod_1000000009=%d\n",
           g_sample_count, selected_support, complement_support,
           (unsigned long long)pair_tests, seconds, rank1, rank2);
    for (int i = 0; i < g_sample_count; i++) {
        free(selected[i].entries);
        free(complements[i].entries);
    }
}

static U128 disjoint_join(const Map* lhs, const Map* rhs, uint64_t* pair_tests) {
    U128 total = 0;
    *pair_tests = 0;
    for (size_t i = 0; i < lhs->capacity; i++) {
        if (!lhs->entries[i].used) continue;
        for (size_t j = 0; j < rhs->capacity; j++) {
            if (!rhs->entries[j].used) continue;
            (*pair_tests)++;
            if ((lhs->entries[i].mask & rhs->entries[j].mask) == 0) {
                total += (U128)lhs->entries[i].weight * rhs->entries[j].weight;
            }
        }
    }
    return total;
}

static void enumerate_histograms(int pattern, int remaining) {
    if (pattern == PATTERNS - 1) {
        g_histogram[pattern] = (unsigned)remaining;
        if (is_canonical()) record_orbit();
        return;
    }
    for (int count = 0; count <= remaining; count++) {
        g_histogram[pattern] = (unsigned)count;
        enumerate_histograms(pattern + 1, remaining - count);
    }
}

static int size_compare(const void* lhs_ptr, const void* rhs_ptr) {
    size_t lhs = *(const size_t*)lhs_ptr;
    size_t rhs = *(const size_t*)rhs_ptr;
    return lhs < rhs ? -1 : lhs > rhs;
}

int main(void) {
    int pair = 0;
    for (int u = 0; u < ROWS; u++) {
        for (int v = u + 1; v < ROWS; v++) g_pair_index[u][v] = pair++;
    }
    generate_permutations(0, 0);
    enumerate_histograms(0, ROWS);
    qsort(g_supports, g_support_count, sizeof(g_supports[0]), size_compare);
    unsigned long long support_sum = 0;
    for (size_t i = 0; i < g_support_count; i++) support_sum += g_supports[i];
    printf("orbits=%llu dead_orbits=%llu distributions=%zu transitions=%llu\n",
           (unsigned long long)g_orbits, (unsigned long long)g_dead_orbits,
           g_support_count, (unsigned long long)g_transitions);
    printf("support min=%zu median=%zu p90=%zu p99=%zu max=%zu mean=%.1f\n",
           g_supports[0], g_supports[g_support_count / 2U],
           g_supports[g_support_count * 9U / 10U],
           g_supports[g_support_count * 99U / 100U], g_max_support,
           (double)support_sum / (double)g_support_count);
    printf("max_histogram complement=%d:", g_max_complement);
    for (int pattern = 0; pattern < PATTERNS; pattern++) {
        if (g_max_histogram[pattern]) printf(" %d:%u", pattern, g_max_histogram[pattern]);
    }
    printf("\n");
    Map hardest = build_distribution(g_max_histogram, g_max_complement);
    Map opposite = build_distribution(g_max_histogram, !g_max_complement);
    uint64_t pair_tests = 0;
    double join_start = seconds_now();
    U128 joined = disjoint_join(&hardest, &opposite, &pair_tests);
    double join_seconds = seconds_now() - join_start;
    printf("max_cross_join supports=%zu,%zu pair_tests=%llu total=", hardest.count,
           opposite.count, (unsigned long long)pair_tests);
    print_u128(joined);
    printf(" time=%.3fs\n", join_seconds);
    free(hardest.entries);
    free(opposite.entries);
    probe_kernel_rank();
    return g_orbits == 1324 ? 0 : 1;
}
