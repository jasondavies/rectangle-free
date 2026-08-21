/* S8 character decomposition of binary 8x3 masks modulo S3 columns. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N = 8, MAX_PARTITIONS = 22 };

typedef struct {
    uint8_t part[N];
    uint8_t count;
} Partition;

static Partition partitions[MAX_PARTITIONS];
static int partition_count;
static uint64_t factorial[N + 1];
static int g_columns = 3;

static void generate_partitions_rec(int remaining, int maximum, Partition* current) {
    if (!remaining) {
        partitions[partition_count++] = *current;
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
    uint8_t removed[N][N] = {{0}};
    int removed_count = 0;
    int first_row = -1;
    int first_column = -1;
    int occupied_rows = 0;
    for (int row = 0; row < outer->count; row++) {
        int inner_length = row < inner->count ? inner->part[row] : 0;
        int row_removed = 0;
        for (int column = inner_length; column < outer->part[row]; column++) {
            removed[row][column] = 1;
            removed_count++;
            row_removed = 1;
            if (first_row < 0) {
                first_row = row;
                first_column = column;
            }
        }
        occupied_rows += row_removed;
    }
    if (!removed_count) return 0;
    for (int row = 0; row + 1 < N; row++) {
        for (int column = 0; column + 1 < N; column++) {
            if (removed[row][column] && removed[row + 1][column] &&
                removed[row][column + 1] && removed[row + 1][column + 1]) return 0;
        }
    }
    uint8_t visited[N][N] = {{0}};
    int queue_row[N * N];
    int queue_column[N * N];
    int begin = 0;
    int end = 1;
    queue_row[0] = first_row;
    queue_column[0] = first_column;
    visited[first_row][first_column] = 1;
    int connected = 0;
    static const int dr[4] = {-1, 1, 0, 0};
    static const int dc[4] = {0, 0, -1, 1};
    while (begin < end) {
        int row = queue_row[begin];
        int column = queue_column[begin++];
        connected++;
        for (int direction = 0; direction < 4; direction++) {
            int next_row = row + dr[direction];
            int next_column = column + dc[direction];
            if (next_row < 0 || next_row >= N || next_column < 0 || next_column >= N ||
                !removed[next_row][next_column] || visited[next_row][next_column]) continue;
            visited[next_row][next_column] = 1;
            queue_row[end] = next_row;
            queue_column[end++] = next_column;
        }
    }
    if (connected != removed_count) return 0;
    *height = occupied_rows - 1;
    return 1;
}

static int character_rec(const Partition* shape, const Partition* cycles, int cycle_index);

static void enumerate_inner_rec(const Partition* outer, const Partition* cycles,
                                int cycle_index, int row, int previous, int target_size,
                                int current_size, Partition* inner, int* total) {
    if (row == outer->count) {
        if (current_size != target_size) return;
        while (inner->count && inner->part[inner->count - 1] == 0) inner->count--;
        int height = 0;
        if (removed_is_border_strip(outer, inner, &height)) {
            int contribution = character_rec(inner, cycles, cycle_index + 1);
            *total += (height & 1) ? -contribution : contribution;
        }
        inner->count = (uint8_t)outer->count;
        return;
    }
    int maximum = outer->part[row] < previous ? outer->part[row] : previous;
    for (int length = maximum; length >= 0; length--) {
        if (current_size + length > target_size) continue;
        inner->part[row] = (uint8_t)length;
        enumerate_inner_rec(outer, cycles, cycle_index, row + 1, length, target_size,
                            current_size + length, inner, total);
    }
}

static int character_rec(const Partition* shape, const Partition* cycles, int cycle_index) {
    if (cycle_index == cycles->count) return partition_size(shape) == 0;
    int remove = cycles->part[cycle_index];
    int target = partition_size(shape) - remove;
    if (target < 0) return 0;
    Partition inner = {.count = shape->count};
    int total = 0;
    enumerate_inner_rec(shape, cycles, cycle_index, 0, N, target, 0, &inner, &total);
    return total;
}

static int character(const Partition* shape, const Partition* cycles) {
    return character_rec(shape, cycles, 0);
}

static uint64_t conjugacy_class_size(const Partition* cycles) {
    uint64_t denominator = 1;
    for (int length = 1; length <= N; length++) {
        int multiplicity = 0;
        for (int i = 0; i < cycles->count; i++) if (cycles->part[i] == length) multiplicity++;
        for (int i = 0; i < multiplicity; i++) denominator *= (uint64_t)length;
        denominator *= factorial[multiplicity];
    }
    return factorial[N] / denominator;
}

static int gcd_int(int a, int b) {
    while (b) {
        int remainder = a % b;
        a = b;
        b = remainder;
    }
    return a;
}

static uint64_t fixed_column_orbits(const Partition* row_cycles) {
    static const int column_cycles3[3][4] = {{1, 1, 1, 0}, {2, 1, 0, 0}, {3, 0, 0, 0}};
    static const int column_counts3[3] = {3, 2, 1};
    static const int column_class_sizes3[3] = {1, 3, 2};
    static const int column_cycles4[5][4] = {
        {1, 1, 1, 1}, {2, 1, 1, 0}, {2, 2, 0, 0}, {3, 1, 0, 0}, {4, 0, 0, 0}
    };
    static const int column_counts4[5] = {4, 3, 2, 2, 1};
    static const int column_class_sizes4[5] = {1, 6, 3, 8, 6};
    const int (*column_cycles)[4] = g_columns == 3 ? column_cycles3 : column_cycles4;
    const int* column_counts = g_columns == 3 ? column_counts3 : column_counts4;
    const int* column_class_sizes =
        g_columns == 3 ? column_class_sizes3 : column_class_sizes4;
    int types = g_columns == 3 ? 3 : 5;
    int column_factorial = g_columns == 3 ? 6 : 24;
    uint64_t sum = 0;
    for (int type = 0; type < types; type++) {
        int cell_cycles = 0;
        for (int i = 0; i < row_cycles->count; i++) {
            for (int j = 0; j < column_counts[type]; j++) {
                cell_cycles += gcd_int(row_cycles->part[i], column_cycles[type][j]);
            }
        }
        sum += (uint64_t)column_class_sizes[type] * (UINT64_C(1) << cell_cycles);
    }
    return sum / (uint64_t)column_factorial;
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
    g_columns = argc > 1 ? atoi(argv[1]) : 3;
    if (g_columns != 3 && g_columns != 4) {
        fprintf(stderr, "Usage: %s COLUMNS, with COLUMNS equal to 3 or 4\n", argv[0]);
        return 2;
    }
    factorial[0] = 1;
    for (int i = 1; i <= N; i++) factorial[i] = factorial[i - 1] * (uint64_t)i;
    Partition current = {0};
    generate_partitions_rec(N, N, &current);
    uint64_t module_dimension = 0;
    uint64_t commutant_dimension = 0;
    uint64_t trivial_multiplicity = 0;
    printf("columns=%d\n", g_columns);
    printf("lambda dimension multiplicity block_square\n");
    for (int lambda_index = 0; lambda_index < partition_count; lambda_index++) {
        const Partition* lambda = &partitions[lambda_index];
        int64_t numerator = 0;
        int dimension = 0;
        for (int mu_index = 0; mu_index < partition_count; mu_index++) {
            const Partition* mu = &partitions[mu_index];
            int chi = character(lambda, mu);
            if (mu->count == N) dimension = chi;
            numerator += (int64_t)conjugacy_class_size(mu) * chi *
                         (int64_t)fixed_column_orbits(mu);
        }
        int64_t multiplicity = numerator / (int64_t)factorial[N];
        if (lambda_index == 0) trivial_multiplicity = (uint64_t)multiplicity;
        print_partition(lambda);
        printf(" %d %lld %llu\n", dimension, (long long)multiplicity,
               (unsigned long long)(multiplicity * multiplicity));
        module_dimension += (uint64_t)dimension * (uint64_t)multiplicity;
        commutant_dimension += (uint64_t)multiplicity * (uint64_t)multiplicity;
    }
    Partition identity_cycles = {.count = N};
    for (int i = 0; i < N; i++) identity_cycles.part[i] = 1;
    uint64_t expected_dimension = fixed_column_orbits(&identity_cycles);
    uint64_t expected_orbits = g_columns == 3 ? 1324U : 25207U;
    printf("module_dimension=%llu expected=%llu %s\n",
           (unsigned long long)module_dimension, (unsigned long long)expected_dimension,
           module_dimension == expected_dimension ? "OK" : "FAIL");
    printf("trivial_multiplicity=%llu expected_orbits=%llu %s\n",
           (unsigned long long)trivial_multiplicity, (unsigned long long)expected_orbits,
           trivial_multiplicity == expected_orbits ? "OK" : "FAIL");
    printf("commutant_dimension=%llu\n", (unsigned long long)commutant_dimension);
    return module_dimension == expected_dimension && trivial_multiplicity == expected_orbits ? 0 : 1;
}
