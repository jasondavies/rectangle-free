/* Burnside counts for binary matrices modulo row/column permutations and complement. */
#include <stdint.h>
#include <stdio.h>

typedef unsigned __int128 U128;

typedef struct {
    uint8_t length[9];
    uint8_t count;
    uint64_t class_size;
} Partition;

static uint64_t factorial[10];
static Partition partitions[10][32];
static int partition_counts[10];

static uint64_t gcd_u64(uint64_t a, uint64_t b) {
    while (b) {
        uint64_t remainder = a % b;
        a = b;
        b = remainder;
    }
    return a;
}

static void record_partition(int n, const uint8_t* parts, int count) {
    Partition* partition = &partitions[n][partition_counts[n]++];
    partition->count = (uint8_t)count;
    uint64_t denominator = 1;
    for (int i = 0; i < count; i++) partition->length[i] = parts[i];
    for (int length = 1; length <= n; length++) {
        int multiplicity = 0;
        for (int i = 0; i < count; i++) if (parts[i] == length) multiplicity++;
        for (int i = 0; i < multiplicity; i++) denominator *= (uint64_t)length;
        denominator *= factorial[multiplicity];
    }
    partition->class_size = factorial[n] / denominator;
}

static void generate_partitions_rec(int n, int remaining, int minimum,
                                    uint8_t* parts, int count) {
    if (!remaining) {
        record_partition(n, parts, count);
        return;
    }
    for (int part = minimum; part <= remaining; part++) {
        parts[count] = (uint8_t)part;
        generate_partitions_rec(n, remaining - part, part, parts, count + 1);
    }
}

static U128 orbit_count(int rows, int columns) {
    U128 sum = 0;
    for (int i = 0; i < partition_counts[rows]; i++) {
        const Partition* row_partition = &partitions[rows][i];
        for (int j = 0; j < partition_counts[columns]; j++) {
            const Partition* column_partition = &partitions[columns][j];
            int cell_cycles = 0;
            for (int r = 0; r < row_partition->count; r++) {
                for (int c = 0; c < column_partition->count; c++) {
                    cell_cycles += (int)gcd_u64(row_partition->length[r],
                                               column_partition->length[c]);
                }
            }
            sum += (U128)row_partition->class_size * column_partition->class_size *
                   ((U128)1 << cell_cycles);
        }
    }
    return sum / ((U128)factorial[rows] * factorial[columns]);
}

static U128 complement_fixed_orbit_count(int rows, int columns) {
    U128 sum = 0;
    for (int i = 0; i < partition_counts[rows]; i++) {
        const Partition* row_partition = &partitions[rows][i];
        for (int j = 0; j < partition_counts[columns]; j++) {
            const Partition* column_partition = &partitions[columns][j];
            int cell_cycles = 0;
            int all_cycles_even = 1;
            for (int r = 0; r < row_partition->count; r++) {
                for (int c = 0; c < column_partition->count; c++) {
                    int row_length = row_partition->length[r];
                    int column_length = column_partition->length[c];
                    int common = (int)gcd_u64((uint64_t)row_length,
                                              (uint64_t)column_length);
                    cell_cycles += common;
                    if (((row_length / common) * column_length) % 2 != 0) {
                        all_cycles_even = 0;
                    }
                }
            }
            if (all_cycles_even) {
                sum += (U128)row_partition->class_size * column_partition->class_size *
                       ((U128)1 << cell_cycles);
            }
        }
    }
    return sum / ((U128)factorial[rows] * factorial[columns]);
}

static U128 orbit_count_at_weight(int rows, int columns, int target_weight) {
    U128 sum = 0;
    for (int i = 0; i < partition_counts[rows]; i++) {
        const Partition* row_partition = &partitions[rows][i];
        for (int j = 0; j < partition_counts[columns]; j++) {
            const Partition* column_partition = &partitions[columns][j];
            U128 coefficients[73] = {0};
            coefficients[0] = 1;
            for (int r = 0; r < row_partition->count; r++) {
                for (int c = 0; c < column_partition->count; c++) {
                    int row_length = row_partition->length[r];
                    int column_length = column_partition->length[c];
                    int common = (int)gcd_u64((uint64_t)row_length,
                                              (uint64_t)column_length);
                    int cycle_length = row_length / common * column_length;
                    for (int copy = 0; copy < common; copy++) {
                        for (int weight = target_weight; weight >= cycle_length; weight--) {
                            coefficients[weight] += coefficients[weight - cycle_length];
                        }
                    }
                }
            }
            sum += (U128)row_partition->class_size * column_partition->class_size *
                   coefficients[target_weight];
        }
    }
    return sum / ((U128)factorial[rows] * factorial[columns]);
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

int main(void) {
    factorial[0] = 1;
    for (int i = 1; i <= 9; i++) factorial[i] = factorial[i - 1] * (uint64_t)i;
    for (int n = 1; n <= 9; n++) {
        uint8_t parts[9];
        generate_partitions_rec(n, n, 1, parts, 0);
    }
    for (int columns = 1; columns <= 8; columns++) {
        printf("8x%d binary_row_column_orbits=", columns);
        print_u128(orbit_count(8, columns));
        printf("\n");
    }
    U128 seven_orbits = orbit_count(7, 7);
    U128 seven_self_complementary = complement_fixed_orbit_count(7, 7);
    printf("7x7 binary_row_column_orbits=");
    print_u128(seven_orbits);
    printf(" self_complementary_orbits=");
    print_u128(seven_self_complementary);
    printf(" complement_paired_orbits=");
    print_u128((seven_orbits + seven_self_complementary) / 2);
    printf("\n");
    U128 seven_eight_orbits = orbit_count(7, 8);
    U128 seven_eight_self_complementary = complement_fixed_orbit_count(7, 8);
    U128 seven_eight_midpoint = orbit_count_at_weight(7, 8, 28);
    printf("7x8 binary_row_column_orbits=");
    print_u128(seven_eight_orbits);
    printf(" self_complementary_orbits=");
    print_u128(seven_eight_self_complementary);
    printf(" complement_paired_orbits=");
    print_u128((seven_eight_orbits + seven_eight_self_complementary) / 2);
    printf(" midpoint_orbits=");
    print_u128(seven_eight_midpoint);
    printf(" evaluated_without_midpoint_pairing=");
    print_u128((seven_eight_orbits + seven_eight_midpoint) / 2);
    printf("\n");
    U128 eight_orbits = orbit_count(8, 8);
    U128 eight_self_complementary = complement_fixed_orbit_count(8, 8);
    U128 eight_midpoint = orbit_count_at_weight(8, 8, 32);
    printf("8x8 binary_row_column_orbits=");
    print_u128(eight_orbits);
    printf(" self_complementary_orbits=");
    print_u128(eight_self_complementary);
    printf(" complement_paired_orbits=");
    print_u128((eight_orbits + eight_self_complementary) / 2);
    printf(" midpoint_orbits=");
    print_u128(eight_midpoint);
    printf(" evaluated_without_midpoint_pairing=");
    print_u128((eight_orbits + eight_midpoint) / 2);
    printf("\n");
    U128 six_nine_orbits = orbit_count(6, 9);
    U128 six_nine_self_complementary = complement_fixed_orbit_count(6, 9);
    printf("6x9 binary_row_column_orbits=");
    print_u128(six_nine_orbits);
    printf(" self_complementary_orbits=");
    print_u128(six_nine_self_complementary);
    printf(" complement_paired_orbits=");
    print_u128((six_nine_orbits + six_nine_self_complementary) / 2);
    U128 six_nine_midpoint = orbit_count_at_weight(6, 9, 27);
    printf(" midpoint_orbits=");
    print_u128(six_nine_midpoint);
    printf(" evaluated_without_midpoint_pairing=");
    print_u128((six_nine_orbits + six_nine_midpoint) / 2);
    printf("\n");
    return 0;
}
