/* Exact modular rank gate for an S_m-equivariant two-bit  h+h kernel. */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { MAX_ROWS = 5, MAX_GROUP = 120, MAX_PARTITIONS = 7 };
static const uint32_t PRIME = 1000000007U;

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
    uint64_t weight;
} Entry;

typedef struct {
    Entry* entries;
    size_t count;
} Distribution;

typedef struct {
    uint8_t part[MAX_ROWS];
    uint8_t count;
} Partition;

typedef struct {
    uint8_t rows[MAX_ROWS];
    Distribution selected;
    Distribution complement;
    int orbit_begin;
    int orbit_count;
} Seed;

typedef struct {
    uint8_t rows[MAX_ROWS];
    uint8_t representative_permutation;
    uint8_t seed;
} State;

static int g_rows;
static int g_columns;
static int g_pairs;
static int g_pair_index[MAX_ROWS][MAX_ROWS];
static uint8_t g_permutations[MAX_GROUP][MAX_ROWS];
static uint8_t g_inverse[MAX_GROUP][MAX_ROWS];
static uint8_t g_compose[MAX_GROUP][MAX_GROUP];
static uint8_t g_cycle_type[MAX_GROUP];
static int g_group_size;
static Partition g_partitions[MAX_PARTITIONS];
static int g_partition_count;
static uint64_t g_join_tests;

static double seconds_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
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
    if (!pointer) exit(1);
    return pointer;
}

static void map_init(Map* map, size_t capacity) {
    map->capacity = capacity;
    map->count = 0;
    map->entries = xcalloc(capacity, sizeof(map->entries[0]));
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

static Distribution build_distribution(const uint8_t rows[MAX_ROWS], int complement) {
    Map current;
    map_init(&current, 16);
    map_add(&current, 0, 1);
    int patterns = 1 << g_columns;
    for (int column = 0; column < g_columns; column++) {
        unsigned active = 0;
        for (int row = 0; row < g_rows; row++) {
            unsigned pattern = complement ? rows[row] ^ (patterns - 1U) : rows[row];
            if ((pattern >> column) & 1U) active |= 1U << row;
        }
        Map increments;
        map_init(&increments, 16);
        unsigned assignment = active;
        for (;;) {
            uint64_t mask = 0;
            for (int u = 0; u < g_rows; u++) {
                for (int v = u + 1; v < g_rows; v++) {
                    if (((active >> u) & 1U) && ((active >> v) & 1U) &&
                        (((assignment >> u) ^ (assignment >> v)) & 1U) == 0) {
                        int colour = (assignment >> u) & 1U;
                        mask |= UINT64_C(1) << (colour * g_pairs + g_pair_index[u][v]);
                    }
                }
            }
            map_add(&increments, mask, 1);
            if (!assignment) break;
            assignment = (assignment - 1U) & active;
        }
        Map next;
        map_init(&next, current.capacity);
        for (size_t i = 0; i < current.capacity; i++) {
            if (!current.entries[i].used) continue;
            for (size_t j = 0; j < increments.capacity; j++) {
                if (!increments.entries[j].used ||
                    (current.entries[i].mask & increments.entries[j].mask)) continue;
                map_add(&next, current.entries[i].mask | increments.entries[j].mask,
                        current.entries[i].weight * increments.entries[j].weight);
            }
        }
        free(current.entries);
        free(increments.entries);
        current = next;
    }
    Entry* entries = xcalloc(current.count, sizeof(entries[0]));
    size_t count = 0;
    for (size_t i = 0; i < current.capacity; i++) {
        if (current.entries[i].used) {
            entries[count++] = (Entry){current.entries[i].mask, current.entries[i].weight};
        }
    }
    free(current.entries);
    return (Distribution){entries, count};
}

static uint64_t permute_mask(uint64_t mask, int permutation) {
    int old_to_new[MAX_ROWS];
    for (int new_row = 0; new_row < g_rows; new_row++) {
        old_to_new[g_permutations[permutation][new_row]] = new_row;
    }
    int pair_image[10];
    int pair = 0;
    for (int old_u = 0; old_u < g_rows; old_u++) {
        for (int old_v = old_u + 1; old_v < g_rows; old_v++) {
            int u = old_to_new[old_u];
            int v = old_to_new[old_v];
            if (u > v) {
                int swap = u;
                u = v;
                v = swap;
            }
            pair_image[pair++] = g_pair_index[u][v];
        }
    }
    uint64_t result = 0;
    while (mask) {
        int bit = __builtin_ctzll(mask);
        result |= UINT64_C(1) << ((bit / g_pairs) * g_pairs + pair_image[bit % g_pairs]);
        mask &= mask - 1U;
    }
    return result;
}

static uint64_t disjoint_join_permuted(const Distribution* lhs,
                                       const Distribution* rhs, int permutation) {
    uint64_t total = 0;
    g_join_tests += lhs->count * rhs->count;
    for (size_t j = 0; j < rhs->count; j++) {
        uint64_t rhs_mask = permute_mask(rhs->entries[j].mask, permutation);
        for (size_t i = 0; i < lhs->count; i++) {
            if ((lhs->entries[i].mask & rhs_mask) == 0) {
                total += lhs->entries[i].weight * rhs->entries[j].weight;
            }
        }
    }
    return total;
}

static void generate_permutations_rec(int depth, unsigned used, uint8_t current[MAX_ROWS]) {
    if (depth == g_rows) {
        memcpy(g_permutations[g_group_size++], current, (size_t)g_rows);
        return;
    }
    for (int row = 0; row < g_rows; row++) {
        if ((used >> row) & 1U) continue;
        current[depth] = (uint8_t)row;
        generate_permutations_rec(depth + 1, used | (1U << row), current);
    }
}

static int permutation_index(const uint8_t permutation[MAX_ROWS]) {
    for (int i = 0; i < g_group_size; i++) {
        if (!memcmp(g_permutations[i], permutation, (size_t)g_rows)) return i;
    }
    exit(1);
}

static void generate_partitions_rec(int remaining, int maximum, Partition* current) {
    if (!remaining) {
        g_partitions[g_partition_count++] = *current;
        return;
    }
    if (maximum > remaining) maximum = remaining;
    for (int part = maximum; part >= 1; part--) {
        current->part[current->count++] = (uint8_t)part;
        generate_partitions_rec(remaining - part, part, current);
        current->count--;
    }
}

static int partition_size(const Partition* partition) {
    int size = 0;
    for (int i = 0; i < partition->count; i++) size += partition->part[i];
    return size;
}

static int removed_is_border_strip(const Partition* outer, const Partition* inner,
                                   int* height) {
    uint8_t removed[MAX_ROWS][MAX_ROWS] = {{0}};
    int first_row = -1, first_column = -1, removed_count = 0, occupied_rows = 0;
    for (int row = 0; row < outer->count; row++) {
        int inner_length = row < inner->count ? inner->part[row] : 0;
        int row_removed = 0;
        for (int column = inner_length; column < outer->part[row]; column++) {
            removed[row][column] = 1;
            removed_count++;
            row_removed = 1;
            if (first_row < 0) first_row = row, first_column = column;
        }
        occupied_rows += row_removed;
    }
    if (!removed_count) return 0;
    for (int row = 0; row + 1 < MAX_ROWS; row++) {
        for (int column = 0; column + 1 < MAX_ROWS; column++) {
            if (removed[row][column] && removed[row + 1][column] &&
                removed[row][column + 1] && removed[row + 1][column + 1]) return 0;
        }
    }
    int qr[MAX_ROWS * MAX_ROWS], qc[MAX_ROWS * MAX_ROWS], begin = 0, end = 1, seen = 0;
    uint8_t visited[MAX_ROWS][MAX_ROWS] = {{0}};
    qr[0] = first_row;
    qc[0] = first_column;
    visited[first_row][first_column] = 1;
    while (begin < end) {
        int row = qr[begin], column = qc[begin++];
        seen++;
        static const int dr[4] = {-1, 1, 0, 0};
        static const int dc[4] = {0, 0, -1, 1};
        for (int direction = 0; direction < 4; direction++) {
            int nr = row + dr[direction], nc = column + dc[direction];
            if (nr < 0 || nr >= MAX_ROWS || nc < 0 || nc >= MAX_ROWS ||
                !removed[nr][nc] || visited[nr][nc]) continue;
            visited[nr][nc] = 1;
            qr[end] = nr;
            qc[end++] = nc;
        }
    }
    if (seen != removed_count) return 0;
    *height = occupied_rows - 1;
    return 1;
}

static int character_rec(const Partition* shape, const Partition* cycles, int cycle_index);

static void enumerate_inner_rec(const Partition* outer, const Partition* cycles,
                                int cycle_index, int row, int previous, int target,
                                int size, Partition* inner, int* total) {
    if (row == outer->count) {
        if (size != target) return;
        while (inner->count && !inner->part[inner->count - 1]) inner->count--;
        int height = 0;
        if (removed_is_border_strip(outer, inner, &height)) {
            int value = character_rec(inner, cycles, cycle_index + 1);
            *total += (height & 1) ? -value : value;
        }
        inner->count = outer->count;
        return;
    }
    int maximum = outer->part[row] < previous ? outer->part[row] : previous;
    for (int length = maximum; length >= 0; length--) {
        inner->part[row] = (uint8_t)length;
        enumerate_inner_rec(outer, cycles, cycle_index, row + 1, length, target,
                            size + length, inner, total);
    }
}

static int character_rec(const Partition* shape, const Partition* cycles, int cycle_index) {
    if (cycle_index == cycles->count) return partition_size(shape) == 0;
    int target = partition_size(shape) - cycles->part[cycle_index];
    if (target < 0) return 0;
    Partition inner = {.count = shape->count};
    int total = 0;
    enumerate_inner_rec(shape, cycles, cycle_index, 0, MAX_ROWS, target, 0,
                        &inner, &total);
    return total;
}

static int character(const Partition* shape, const Partition* cycles) {
    return character_rec(shape, cycles, 0);
}

static uint32_t mod_pow(uint32_t base, uint32_t exponent) {
    uint64_t result = 1, value = base;
    while (exponent) {
        if (exponent & 1U) result = result * value % PRIME;
        value = value * value % PRIME;
        exponent >>= 1;
    }
    return (uint32_t)result;
}

static int matrix_rank(uint32_t* matrix, int rows, int columns) {
    int rank = 0;
    for (int column = 0; column < columns && rank < rows; column++) {
        int pivot = rank;
        while (pivot < rows && !matrix[(size_t)pivot * columns + column]) pivot++;
        if (pivot == rows) continue;
        if (pivot != rank) {
            for (int j = column; j < columns; j++) {
                uint32_t swap = matrix[(size_t)rank * columns + j];
                matrix[(size_t)rank * columns + j] = matrix[(size_t)pivot * columns + j];
                matrix[(size_t)pivot * columns + j] = swap;
            }
        }
        uint32_t inverse = mod_pow(matrix[(size_t)rank * columns + column], PRIME - 2U);
        for (int j = column; j < columns; j++) {
            matrix[(size_t)rank * columns + j] =
                (uint32_t)((uint64_t)matrix[(size_t)rank * columns + j] * inverse % PRIME);
        }
        for (int i = 0; i < rows; i++) {
            if (i == rank) continue;
            uint32_t factor = matrix[(size_t)i * columns + column];
            if (!factor) continue;
            for (int j = column; j < columns; j++) {
                uint32_t product =
                    (uint32_t)((uint64_t)factor * matrix[(size_t)rank * columns + j] % PRIME);
                uint32_t* cell = &matrix[(size_t)i * columns + j];
                *cell = *cell >= product ? *cell - product : *cell + PRIME - product;
            }
        }
        rank++;
    }
    return rank;
}

static uint64_t rows_code(const uint8_t rows[MAX_ROWS]) {
    uint64_t code = 0;
    for (int row = 0; row < g_rows; row++) code |= (uint64_t)rows[row] << (row * 4);
    return code;
}

static int state_in_seed(const State* states, const Seed* seed,
                         const uint8_t rows[MAX_ROWS]) {
    uint64_t code = rows_code(rows);
    for (int i = 0; i < seed->orbit_count; i++) {
        int state = seed->orbit_begin + i;
        if (rows_code(states[state].rows) == code) return state;
    }
    exit(1);
}

static void print_partition(const Partition* partition) {
    putchar('[');
    for (int i = 0; i < partition->count; i++) {
        if (i) putchar(',');
        printf("%u", partition->part[i]);
    }
    putchar(']');
}

int main(int argc, char** argv) {
    g_rows = argc > 1 ? atoi(argv[1]) : 4;
    g_columns = argc > 2 ? atoi(argv[2]) : 4;
    int requested_seeds = argc > 3 ? atoi(argv[3]) : 8;
    if (g_rows < 3 || g_rows > MAX_ROWS || g_columns < 2 || g_columns > 4 ||
        requested_seeds < 1 || requested_seeds > 32) {
        fprintf(stderr, "Usage: %s ROWS(3..5) HALF_COLUMNS(2..4) SEEDS(1..32)\n", argv[0]);
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    int pair = 0;
    for (int u = 0; u < g_rows; u++) {
        for (int v = u + 1; v < g_rows; v++) g_pair_index[u][v] = pair++;
    }
    g_pairs = pair;
    uint8_t current[MAX_ROWS] = {0};
    generate_permutations_rec(0, 0, current);
    for (int p = 0; p < g_group_size; p++) {
        for (int i = 0; i < g_rows; i++) g_inverse[p][g_permutations[p][i]] = (uint8_t)i;
        for (int q = 0; q < g_group_size; q++) {
            uint8_t composed[MAX_ROWS];
            for (int i = 0; i < g_rows; i++) {
                composed[i] = g_permutations[q][g_inverse[p][i]];
            }
            g_compose[p][q] = (uint8_t)permutation_index(composed);
        }
    }
    Partition empty = {0};
    generate_partitions_rec(g_rows, g_rows, &empty);
    for (int p = 0; p < g_group_size; p++) {
        uint8_t visited[MAX_ROWS] = {0};
        Partition cycles = {0};
        for (int i = 0; i < g_rows; i++) {
            if (visited[i]) continue;
            int length = 0, at = i;
            do {
                visited[at] = 1;
                at = g_permutations[p][at];
                length++;
            } while (!visited[at]);
            cycles.part[cycles.count++] = (uint8_t)length;
        }
        for (int i = 1; i < cycles.count; i++) {
            uint8_t value = cycles.part[i];
            int j = i;
            while (j && cycles.part[j - 1] < value) {
                cycles.part[j] = cycles.part[j - 1];
                j--;
            }
            cycles.part[j] = value;
        }
        for (int type = 0; type < g_partition_count; type++) {
            if (cycles.count == g_partitions[type].count &&
                !memcmp(cycles.part, g_partitions[type].part, cycles.count)) {
                g_cycle_type[p] = (uint8_t)type;
                break;
            }
        }
    }

    Seed* seeds = xcalloc((size_t)requested_seeds, sizeof(seeds[0]));
    int seed_count = 0;
    uint64_t rng = UINT64_C(0x243f6a8885a308d3);
    int seed_attempts = 0;
    while (seed_count < requested_seeds) {
        if (++seed_attempts > 1000000) {
            fprintf(stderr, "could not find %d distinct viable seed histograms\n",
                    requested_seeds);
            return 1;
        }
        rng = mix64(rng);
        uint8_t rows[MAX_ROWS] = {0};
        uint8_t histogram[16] = {0};
        for (int row = 0; row < g_rows; row++) {
            rng = mix64(rng);
            rows[row] = (uint8_t)(rng & ((1U << g_columns) - 1U));
            histogram[rows[row]]++;
        }
        int duplicate = 0;
        uint64_t signature = 0;
        for (int pattern = 0; pattern < (1 << g_columns); pattern++) {
            signature |= (uint64_t)histogram[pattern] << (3 * pattern);
        }
        for (int i = 0; i < seed_count; i++) {
            uint8_t other_histogram[16] = {0};
            uint64_t other = 0;
            for (int row = 0; row < g_rows; row++) other_histogram[seeds[i].rows[row]]++;
            for (int pattern = 0; pattern < (1 << g_columns); pattern++) {
                other |= (uint64_t)other_histogram[pattern] << (3 * pattern);
            }
            if (other == signature) duplicate = 1;
        }
        if (duplicate) continue;
        memcpy(seeds[seed_count].rows, rows, sizeof(rows));
        seeds[seed_count].selected = build_distribution(rows, 0);
        seeds[seed_count].complement = build_distribution(rows, 1);
        if (!seeds[seed_count].selected.count || !seeds[seed_count].complement.count) {
            free(seeds[seed_count].selected.entries);
            free(seeds[seed_count].complement.entries);
            continue;
        }
        seed_count++;
    }
    int state_capacity = seed_count * g_group_size;
    State* states = xcalloc((size_t)state_capacity, sizeof(states[0]));
    int state_count = 0;
    for (int seed = 0; seed < seed_count; seed++) {
        seeds[seed].orbit_begin = state_count;
        for (int p = 0; p < g_group_size; p++) {
            uint8_t rows[MAX_ROWS];
            for (int row = 0; row < g_rows; row++) {
                rows[row] = seeds[seed].rows[g_permutations[p][row]];
            }
            int duplicate = 0;
            for (int i = seeds[seed].orbit_begin; i < state_count; i++) {
                if (!memcmp(states[i].rows, rows, (size_t)g_rows)) duplicate = 1;
            }
            if (duplicate) continue;
            memcpy(states[state_count].rows, rows, sizeof(rows));
            states[state_count].representative_permutation = (uint8_t)p;
            states[state_count].seed = (uint8_t)seed;
            state_count++;
        }
        seeds[seed].orbit_count = state_count - seeds[seed].orbit_begin;
    }
    printf("rows=%d half_columns=%d group=%d seeds=%d states=%d prime=%u\n",
           g_rows, g_columns, g_group_size, seed_count, state_count, PRIME);
    for (int seed = 0; seed < seed_count; seed++) {
        printf("seed=%d orbit=%d support=%zu,%zu\n", seed, seeds[seed].orbit_count,
               seeds[seed].selected.count, seeds[seed].complement.count);
    }

    double kernel_start = seconds_now();
    uint32_t* relative_kernel = xcalloc((size_t)seed_count * seed_count * g_group_size,
                                        sizeof(relative_kernel[0]));
    for (int a = 0; a < seed_count; a++) {
        for (int b = 0; b < seed_count; b++) {
            for (int p = 0; p < g_group_size; p++) {
                uint64_t selected = disjoint_join_permuted(
                    &seeds[a].selected, &seeds[b].selected, p);
                uint64_t complement = disjoint_join_permuted(
                    &seeds[a].complement, &seeds[b].complement, p);
                relative_kernel[((size_t)a * seed_count + b) * g_group_size + p] =
                    (uint32_t)((selected % PRIME) * (complement % PRIME) % PRIME);
            }
        }
    }
    uint32_t* kernel = xcalloc((size_t)state_count * state_count, sizeof(kernel[0]));
    for (int i = 0; i < state_count; i++) {
        for (int j = 0; j < state_count; j++) {
            int relative = g_compose[states[i].representative_permutation]
                                    [states[j].representative_permutation];
            kernel[(size_t)i * state_count + j] =
                relative_kernel[((size_t)states[i].seed * seed_count + states[j].seed) *
                                    g_group_size + relative];
        }
    }
    for (int i = 0; i < state_count; i++) {
        for (int j = 0; j < state_count; j++) {
            if (kernel[(size_t)i * state_count + j] !=
                kernel[(size_t)j * state_count + i]) {
                fprintf(stderr, "kernel symmetry failed\n");
                return 1;
            }
        }
    }
    for (int p = 0; p < g_group_size; p += g_group_size > 8 ? g_group_size / 8 : 1) {
        for (int i = 0; i < state_count; i++) {
            uint8_t rows_i[MAX_ROWS];
            for (int row = 0; row < g_rows; row++) {
                rows_i[row] = states[i].rows[g_permutations[p][row]];
            }
            int image_i = state_in_seed(states, &seeds[states[i].seed], rows_i);
            for (int j = 0; j < state_count; j++) {
                uint8_t rows_j[MAX_ROWS];
                for (int row = 0; row < g_rows; row++) {
                    rows_j[row] = states[j].rows[g_permutations[p][row]];
                }
                int image_j = state_in_seed(states, &seeds[states[j].seed], rows_j);
                if (kernel[(size_t)i * state_count + j] !=
                    kernel[(size_t)image_i * state_count + image_j]) {
                    fprintf(stderr, "kernel equivariance failed\n");
                    return 1;
                }
            }
        }
    }
    printf("kernel_build=%.3fs comparisons=%llu\n", seconds_now() - kernel_start,
           (unsigned long long)g_join_tests);

    uint32_t* projector = xcalloc((size_t)state_count * state_count, sizeof(projector[0]));
    uint32_t* product = xcalloc((size_t)state_count * state_count, sizeof(product[0]));
    uint32_t* scratch = xcalloc((size_t)state_count * state_count, sizeof(scratch[0]));
    int component_total = 0;
    for (int lambda = 0; lambda < g_partition_count; lambda++) {
        memset(projector, 0, (size_t)state_count * state_count * sizeof(projector[0]));
        for (int column = 0; column < state_count; column++) {
            int seed = states[column].seed;
            for (int p = 0; p < g_group_size; p++) {
                uint8_t transformed[MAX_ROWS];
                for (int row = 0; row < g_rows; row++) {
                    transformed[row] = states[column].rows[g_permutations[p][row]];
                }
                int target = state_in_seed(states, &seeds[seed], transformed);
                int chi = character(&g_partitions[lambda], &g_partitions[g_cycle_type[p]]);
                uint32_t add = chi >= 0 ? (uint32_t)chi : PRIME - (uint32_t)(-chi);
                uint32_t* cell = &projector[(size_t)target * state_count + column];
                *cell += add;
                if (*cell >= PRIME) *cell -= PRIME;
            }
        }
        memcpy(scratch, projector,
               (size_t)state_count * state_count * sizeof(scratch[0]));
        int component_rank = matrix_rank(scratch, state_count, state_count);
        component_total += component_rank;
        memset(product, 0, (size_t)state_count * state_count * sizeof(product[0]));
        for (int block = 0; block < seed_count; block++) {
            int begin = seeds[block].orbit_begin;
            int end = begin + seeds[block].orbit_count;
            for (int i = 0; i < state_count; i++) {
                for (int k = begin; k < end; k++) {
                    uint32_t left = kernel[(size_t)i * state_count + k];
                    if (!left) continue;
                    for (int j = begin; j < end; j++) {
                        uint32_t right = projector[(size_t)k * state_count + j];
                        product[(size_t)i * state_count + j] =
                            (uint32_t)((product[(size_t)i * state_count + j] +
                                        (uint64_t)left * right) % PRIME);
                    }
                }
            }
        }
        memcpy(scratch, product,
               (size_t)state_count * state_count * sizeof(scratch[0]));
        int kernel_rank = matrix_rank(scratch, state_count, state_count);
        print_partition(&g_partitions[lambda]);
        printf(" component=%d kernel_rank=%d retained=%.3f\n", component_rank,
               kernel_rank, component_rank ? (double)kernel_rank / component_rank : 0.0);
    }
    if (component_total != state_count) {
        fprintf(stderr, "central-projector dimension checksum failed: %d != %d\n",
                component_total, state_count);
        return 1;
    }
    printf("component_dimension_sum=%d expected=%d OK\n", component_total,
           state_count);
    free(projector);
    free(product);
    free(scratch);
    free(kernel);
    free(relative_kernel);
    free(states);
    for (int seed = 0; seed < seed_count; seed++) {
        free(seeds[seed].selected.entries);
        free(seeds[seed].complement.entries);
    }
    free(seeds);
    return 0;
}
