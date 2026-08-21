/* Exact finite-field rank probe for the one-column 2+2-colour tensor split. */
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    uint64_t left;
    uint64_t right;
    uint32_t weight;
} Entry;

static int bit_permutations[24][4];
static int current_permutation[4];
static int permutation_count;
static unsigned histogram[16];
static uint64_t invariant_orbits;
static uint64_t left_feasible_orbits;
static uint64_t jointly_feasible_orbits;
static uint64_t binary_masks[256][256];
static unsigned binary_mask_counts[256];
static int probe_rows;
static int probe_pairs;
static int probe_pair_index[8][8];

static int packing_feasible_rec(const unsigned* subsets, const int* order, int depth,
                                uint64_t used) {
    if (depth == 4) return 1;
    unsigned subset = subsets[order[depth]];
    for (unsigned i = 0; i < binary_mask_counts[subset]; i++) {
        uint64_t mask = binary_masks[subset][i];
        if ((mask & used) == 0 && packing_feasible_rec(subsets, order, depth + 1, used | mask)) {
            return 1;
        }
    }
    return 0;
}

static int packing_feasible(const unsigned* subsets) {
    int order[4] = {0, 1, 2, 3};
    for (int i = 1; i < 4; i++) {
        int value = order[i];
        int j = i;
        while (j && binary_mask_counts[subsets[order[j - 1]]] >
                        binary_mask_counts[subsets[value]]) {
            order[j] = order[j - 1];
            j--;
        }
        order[j] = value;
    }
    return packing_feasible_rec(subsets, order, 0, 0);
}

static void build_binary_masks(void) {
    unsigned subset_limit = 1U << probe_rows;
    for (unsigned subset = 0; subset < subset_limit; subset++) {
        unsigned colouring = subset;
        for (;;) {
            uint64_t mask = 0;
            for (int u = 0; u < probe_rows; u++) {
                for (int v = u + 1; v < probe_rows; v++) {
                    if (((subset >> u) & 1U) == 0 || ((subset >> v) & 1U) == 0) continue;
                    unsigned cu = (colouring >> u) & 1U;
                    unsigned cv = (colouring >> v) & 1U;
                    if (cu == cv) {
                        mask |= UINT64_C(1) << (cu * probe_pairs + probe_pair_index[u][v]);
                    }
                }
            }
            unsigned count = binary_mask_counts[subset];
            unsigned i = 0;
            while (i < count && binary_masks[subset][i] != mask) i++;
            if (i == count) binary_masks[subset][binary_mask_counts[subset]++] = mask;
            if (colouring == 0) break;
            colouring = (colouring - 1U) & subset;
        }
    }
}

static void generate_bit_permutations(int depth, unsigned used) {
    if (depth == 4) {
        for (int bit = 0; bit < 4; bit++) {
            bit_permutations[permutation_count][bit] = current_permutation[bit];
        }
        permutation_count++;
        return;
    }
    for (int bit = 0; bit < 4; bit++) {
        if ((used >> bit) & 1U) continue;
        current_permutation[depth] = bit;
        generate_bit_permutations(depth + 1, used | (1U << bit));
    }
}

static int histogram_is_canonical(void) {
    unsigned transformed[16];
    for (int permutation = 0; permutation < permutation_count; permutation++) {
        for (int i = 0; i < 16; i++) transformed[i] = 0;
        for (int pattern = 0; pattern < 16; pattern++) {
            int image = 0;
            for (int bit = 0; bit < 4; bit++) {
                if ((pattern >> bit) & 1) image |= 1 << bit_permutations[permutation][bit];
            }
            transformed[image] = histogram[pattern];
        }
        for (int pattern = 0; pattern < 16; pattern++) {
            if (transformed[pattern] < histogram[pattern]) return 0;
            if (transformed[pattern] > histogram[pattern]) break;
        }
    }
    return 1;
}

static void test_histogram_feasibility(void) {
    unsigned subsets[4] = {0, 0, 0, 0};
    int row = 0;
    for (int pattern = 0; pattern < 16; pattern++) {
        for (unsigned copy = 0; copy < histogram[pattern]; copy++, row++) {
            for (int column = 0; column < 4; column++) {
                if ((pattern >> column) & 1) subsets[column] |= 1U << row;
            }
        }
    }
    if (!packing_feasible(subsets)) return;
    left_feasible_orbits++;
    unsigned complement[4];
    unsigned all_rows = (1U << probe_rows) - 1U;
    for (int column = 0; column < 4; column++) complement[column] = all_rows ^ subsets[column];
    if (packing_feasible(complement)) jointly_feasible_orbits++;
}

static void count_histogram_orbits(int pattern, int remaining) {
    if (pattern == 15) {
        histogram[pattern] = (unsigned)remaining;
        if (histogram_is_canonical()) {
            invariant_orbits++;
            test_histogram_feasibility();
        }
        return;
    }
    for (int count = 0; count <= remaining; count++) {
        histogram[pattern] = (unsigned)count;
        count_histogram_orbits(pattern + 1, remaining - count);
    }
}

static int entry_compare(const void* lhs_ptr, const void* rhs_ptr) {
    const Entry* lhs = lhs_ptr;
    const Entry* rhs = rhs_ptr;
    if (lhs->left != rhs->left) return lhs->left < rhs->left ? -1 : 1;
    if (lhs->right != rhs->right) return lhs->right < rhs->right ? -1 : 1;
    return 0;
}

static int u64_compare(const void* lhs_ptr, const void* rhs_ptr) {
    uint64_t lhs = *(const uint64_t*)lhs_ptr;
    uint64_t rhs = *(const uint64_t*)rhs_ptr;
    return lhs < rhs ? -1 : lhs > rhs;
}

static size_t lower_bound_u64(const uint64_t* values, size_t count, uint64_t target) {
    size_t lo = 0;
    size_t hi = count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2U;
        if (values[mid] < target) lo = mid + 1U;
        else hi = mid;
    }
    return lo;
}

static void subtract_multiple(uint8_t* row, const uint8_t* pivot, size_t begin,
                              size_t end, unsigned coefficient, unsigned modulus) {
#if defined(__AVX2__)
    uint8_t negative[32];
    for (int i = 0; i < 32; i++) {
        negative[i] = (uint8_t)((modulus - coefficient * (unsigned)(i & 15) % modulus) %
                                modulus);
    }
    __m256i table = _mm256_loadu_si256((const __m256i*)negative);
    __m256i threshold = _mm256_set1_epi8((char)(modulus - 1U));
    __m256i modulus_vector = _mm256_set1_epi8((char)modulus);
    size_t column = begin;
    for (; column + 32U <= end; column += 32U) {
        __m256i a = _mm256_loadu_si256((const __m256i*)(row + column));
        __m256i b = _mm256_loadu_si256((const __m256i*)(pivot + column));
        __m256i add = _mm256_shuffle_epi8(table, b);
        __m256i sum = _mm256_add_epi8(a, add);
        __m256i reduce =
            _mm256_and_si256(_mm256_cmpgt_epi8(sum, threshold), modulus_vector);
        _mm256_storeu_si256((__m256i*)(row + column), _mm256_sub_epi8(sum, reduce));
    }
#else
    size_t column = begin;
#endif
    for (; column < end; column++) {
        row[column] =
            (uint8_t)((row[column] + modulus - coefficient * pivot[column] % modulus) % modulus);
    }
}

static size_t matrix_rank(const Entry* entries, size_t entry_count,
                          const uint64_t* left_values, size_t left_count,
                          const uint64_t* right_values, size_t right_count,
                          unsigned modulus) {
    uint8_t* matrix = calloc(left_count * right_count, sizeof(matrix[0]));
    uint8_t** rows = malloc(left_count * sizeof(rows[0]));
    if (!matrix || !rows) {
        fprintf(stderr, "matrix allocation failed (%.1f MiB)\n",
                (double)left_count * (double)right_count / 1048576.0);
        exit(1);
    }
    for (size_t row = 0; row < left_count; row++) rows[row] = matrix + row * right_count;
    for (size_t i = 0; i < entry_count; i++) {
        size_t row = lower_bound_u64(left_values, left_count, entries[i].left);
        size_t column = lower_bound_u64(right_values, right_count, entries[i].right);
        rows[row][column] = (uint8_t)(entries[i].weight % modulus);
    }
    uint8_t inverse[16] = {0};
    for (unsigned value = 1; value < modulus; value++) {
        for (unsigned candidate = 1; candidate < modulus; candidate++) {
            if (value * candidate % modulus == 1U) inverse[value] = (uint8_t)candidate;
        }
    }
    size_t rank = 0;
    for (size_t column = 0; column < right_count && rank < left_count; column++) {
        size_t pivot_row = rank;
        while (pivot_row < left_count && rows[pivot_row][column] == 0) pivot_row++;
        if (pivot_row == left_count) continue;
        uint8_t* tmp = rows[rank];
        rows[rank] = rows[pivot_row];
        rows[pivot_row] = tmp;
        unsigned scale = inverse[rows[rank][column]];
        if (scale != 1) {
            for (size_t c = column; c < right_count; c++) {
                rows[rank][c] = (uint8_t)(rows[rank][c] * scale % modulus);
            }
        }
        for (size_t row = rank + 1U; row < left_count; row++) {
            unsigned coefficient = rows[row][column];
            if (coefficient) {
                subtract_multiple(rows[row], rows[rank], column, right_count, coefficient,
                                  modulus);
            }
        }
        rank++;
    }
    free(rows);
    free(matrix);
    return rank;
}

int main(int argc, char** argv) {
    int row_count = argc > 1 ? atoi(argv[1]) : 8;
    if (row_count < 2 || row_count > 8) {
        fprintf(stderr, "Usage: %s ROWS, with 2 <= ROWS <= 8\n", argv[0]);
        return 2;
    }
    int pair_index[8][8] = {{0}};
    int pairs = 0;
    for (int u = 0; u < row_count; u++) {
        for (int v = u + 1; v < row_count; v++) pair_index[u][v] = pairs++;
    }
    probe_rows = row_count;
    probe_pairs = pairs;
    for (int u = 0; u < row_count; u++) {
        for (int v = u + 1; v < row_count; v++) probe_pair_index[u][v] = pair_index[u][v];
    }
    build_binary_masks();
    size_t assignments = (size_t)1U << (2 * row_count);
    Entry* all = calloc(assignments, sizeof(all[0]));
    uint64_t* left_values = malloc(assignments * sizeof(left_values[0]));
    uint64_t* right_values = malloc(assignments * sizeof(right_values[0]));
    if (!all || !left_values || !right_values) return 1;
    for (size_t code = 0; code < assignments; code++) {
        unsigned colours[8];
        size_t value = code;
        for (int row = 0; row < row_count; row++) {
            colours[row] = (unsigned)(value & 3U);
            value >>= 2;
        }
        uint64_t left = 0;
        uint64_t right = 0;
        for (int u = 0; u < row_count; u++) {
            for (int v = u + 1; v < row_count; v++) {
                if (colours[u] != colours[v]) continue;
                unsigned colour = colours[u];
                if (colour < 2) left |= UINT64_C(1) << (colour * pairs + pair_index[u][v]);
                else right |= UINT64_C(1) << ((colour - 2U) * pairs + pair_index[u][v]);
            }
        }
        all[code] = (Entry){.left = left, .right = right, .weight = 1};
        left_values[code] = left;
        right_values[code] = right;
    }
    qsort(all, assignments, sizeof(all[0]), entry_compare);
    size_t entry_count = 0;
    for (size_t i = 0; i < assignments; i++) {
        if (entry_count && all[i].left == all[entry_count - 1U].left &&
            all[i].right == all[entry_count - 1U].right) {
            all[entry_count - 1U].weight++;
        } else {
            all[entry_count++] = all[i];
        }
    }
    qsort(left_values, assignments, sizeof(left_values[0]), u64_compare);
    qsort(right_values, assignments, sizeof(right_values[0]), u64_compare);
    size_t left_count = 0;
    size_t right_count = 0;
    for (size_t i = 0; i < assignments; i++) {
        if (!left_count || left_values[i] != left_values[left_count - 1U]) {
            left_values[left_count++] = left_values[i];
        }
        if (!right_count || right_values[i] != right_values[right_count - 1U]) {
            right_values[right_count++] = right_values[i];
        }
    }
    printf("rows=%d assignments=%zu entries=%zu active_matrix=%zux%zu bytes=%zu\n",
           row_count, assignments, entry_count, left_count, right_count,
           left_count * right_count);
    if (left_count > SIZE_MAX / right_count || left_count * right_count > UINT64_C(1000000000)) {
        fprintf(stderr, "active matrix exceeds 1 GB kill limit\n");
        return 1;
    }
    const unsigned primes[] = {2, 3, 5, 7, 11, 13};
    size_t stable_rank = 0;
    for (size_t i = 0; i < sizeof(primes) / sizeof(primes[0]); i++) {
        size_t rank = matrix_rank(all, entry_count, left_values, left_count, right_values,
                                  right_count, primes[i]);
        printf("mod%u_rank=%zu/%zu\n", primes[i], rank,
               left_count < right_count ? left_count : right_count);
        stable_rank = rank;
    }
    uint64_t sym4_bound = (uint64_t)stable_rank * (stable_rank + 1U) *
                          (stable_rank + 2U) * (stable_rank + 3U) / 24U;
    generate_bit_permutations(0, 0);
    count_histogram_orbits(0, row_count);
    size_t subset_factor_rank_bound = row_count >= 3 ? ((size_t)1U << row_count) - 2U * row_count : 2U;
    printf("subset_factor_rank_bound=%zu symmetric_fourth_rank_bound=%llu subset_channel_orbits=%llu left_feasible=%llu jointly_feasible=%llu\n",
           subset_factor_rank_bound, (unsigned long long)sym4_bound,
           (unsigned long long)invariant_orbits,
           (unsigned long long)left_feasible_orbits,
           (unsigned long long)jointly_feasible_orbits);
    free(all);
    free(left_values);
    free(right_values);
    return 0;
}
