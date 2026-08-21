/* Exact validator for T4(G) = sum_A C2(A) C2(complement A). */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned __int128 U128;

typedef struct {
    uint64_t increment;
    uint8_t selected_rows;
} BinaryOption;

static int g_rows;
static int g_columns;
static int g_pairs;
static int g_pair_index[8][8];
static BinaryOption* g_binary_options;
static size_t g_binary_option_count;
static uint64_t* g_c2;
static uint64_t g_c2_leaves;

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

static void enumerate_masked_binary(int column, uint64_t used, uint32_t selected) {
    if (column == g_columns) {
        g_c2[selected]++;
        g_c2_leaves++;
        return;
    }
    unsigned shift = (unsigned)(column * g_rows);
    for (size_t i = 0; i < g_binary_option_count; i++) {
        BinaryOption option = g_binary_options[i];
        if (option.increment & used) continue;
        enumerate_masked_binary(column + 1, used | option.increment,
                                selected | ((uint32_t)option.selected_rows << shift));
    }
}

static U128 twobit_count(int rows, int columns, double* seconds, uint64_t* leaves) {
    g_rows = rows;
    g_columns = columns;
    g_pairs = 0;
    memset(g_pair_index, 0, sizeof(g_pair_index));
    for (int u = 0; u < rows; u++) {
        for (int v = u + 1; v < rows; v++) g_pair_index[u][v] = g_pairs++;
    }
    size_t option_capacity = 1;
    for (int row = 0; row < rows; row++) option_capacity *= 3U;
    g_binary_options = calloc(option_capacity, sizeof(g_binary_options[0]));
    if (!g_binary_options) exit(1);
    g_binary_option_count = option_capacity;
    for (size_t code = 0; code < option_capacity; code++) {
        unsigned symbols[8];
        size_t value = code;
        uint8_t selected_rows = 0;
        for (int row = 0; row < rows; row++) {
            symbols[row] = (unsigned)(value % 3U);
            value /= 3U;
            if (symbols[row]) selected_rows |= (uint8_t)(1U << row);
        }
        uint64_t increment = 0;
        for (int u = 0; u < rows; u++) {
            for (int v = u + 1; v < rows; v++) {
                if (symbols[u] && symbols[u] == symbols[v]) {
                    unsigned colour = symbols[u] - 1U;
                    increment |= UINT64_C(1) << (colour * g_pairs + g_pair_index[u][v]);
                }
            }
        }
        g_binary_options[code] =
            (BinaryOption){.increment = increment, .selected_rows = selected_rows};
    }

    int cells = rows * columns;
    size_t mask_count = (size_t)1U << cells;
    g_c2 = calloc(mask_count, sizeof(g_c2[0]));
    if (!g_c2) exit(1);
    g_c2_leaves = 0;
    double start = seconds_now();
    enumerate_masked_binary(0, 0, 0);
    uint32_t full = (uint32_t)(mask_count - 1U);
    U128 total = 0;
    for (uint32_t mask = 0; mask <= full; mask++) {
        total += (U128)g_c2[mask] * g_c2[full ^ mask];
    }
    *seconds = seconds_now() - start;
    *leaves = g_c2_leaves;
    free(g_c2);
    free(g_binary_options);
    return total;
}

static uint64_t direct_four_colour_count(int rows, int columns) {
    int pair_index[8][8] = {{0}};
    int pairs = 0;
    for (int u = 0; u < rows; u++) {
        for (int v = u + 1; v < rows; v++) pair_index[u][v] = pairs++;
    }
    int token_bits = 4 * pairs;
    if (token_bits > 20) return 0;
    size_t states = (size_t)1U << token_bits;
    uint64_t* current = calloc(states, sizeof(current[0]));
    uint64_t* next = calloc(states, sizeof(next[0]));
    if (!current || !next) exit(1);
    size_t assignments = (size_t)1U << (2 * rows);
    uint64_t* increments = calloc(assignments, sizeof(increments[0]));
    if (!increments) exit(1);
    for (size_t code = 0; code < assignments; code++) {
        unsigned colours[8];
        size_t value = code;
        for (int row = 0; row < rows; row++) {
            colours[row] = (unsigned)(value & 3U);
            value >>= 2;
        }
        for (int u = 0; u < rows; u++) {
            for (int v = u + 1; v < rows; v++) {
                if (colours[u] == colours[v]) {
                    increments[code] |=
                        UINT64_C(1) << (colours[u] * pairs + pair_index[u][v]);
                }
            }
        }
    }
    current[0] = 1;
    for (int column = 0; column < columns; column++) {
        memset(next, 0, states * sizeof(next[0]));
        for (size_t state = 0; state < states; state++) {
            if (!current[state]) continue;
            for (size_t assignment = 0; assignment < assignments; assignment++) {
                uint64_t increment = increments[assignment];
                if ((state & increment) == 0) next[state | increment] += current[state];
            }
        }
        uint64_t* swap = current;
        current = next;
        next = swap;
    }
    uint64_t total = 0;
    for (size_t state = 0; state < states; state++) total += current[state];
    free(increments);
    free(current);
    free(next);
    return total;
}

int main(void) {
    static const struct {
        int rows;
        int columns;
        uint64_t expected;
    } cases[] = {
        {2, 2, 0},
        {2, 4, 0},
        {3, 3, 0},
        {3, 4, 12870096},
        {3, 5, 0},
        {4, 4, 2545607472ULL},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        double seconds = 0;
        uint64_t leaves = 0;
        U128 twobit = twobit_count(cases[i].rows, cases[i].columns, &seconds, &leaves);
        uint64_t direct = direct_four_colour_count(cases[i].rows, cases[i].columns);
        int direct_available = direct != 0;
        int ok = (!cases[i].expected || twobit == cases[i].expected) &&
                 (!direct_available || twobit == direct);
        printf("%dx%d twobit=", cases[i].rows, cases[i].columns);
        print_u128(twobit);
        if (cases[i].expected) {
            printf(" expected=%llu", (unsigned long long)cases[i].expected);
        }
        if (direct_available) printf(" direct=%llu", (unsigned long long)direct);
        printf(" ternary_leaves=%llu time=%.3fs %s\n", (unsigned long long)leaves,
               seconds, ok ? "OK" : "FAIL");
        if (!ok) return 1;
    }
    return 0;
}
