/* Sharded exact two-bit solver for T_4(7,7) from weighted binary orbit files. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
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
    LEFT_PATTERNS = 8,
    RIGHT_PATTERNS = 16,
    PAIRS = 21
};

typedef struct {
    uint64_t key;
    uint64_t weight;
} OrbitRecord;

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
    uint64_t orbit_records;
    U128 labelled_weight;
    uint64_t kernels;
    U128 selected_weight;
    U128 contribution;
} Result;

static int g_pair_index[ROWS][ROWS];
static uint64_t g_transitions;
static uint64_t g_pair_tests;

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

static void fprint_u128(FILE* file, U128 value) {
    char digits[40];
    int length = 0;
    do {
        digits[length++] = (char)('0' + value % 10);
        value /= 10;
    } while (value);
    while (length) fputc(digits[--length], file);
}

static U128 parse_u128(const char* text_value) {
    U128 value = 0;
    if (!*text_value) exit(1);
    for (const char* cursor = text_value; *cursor; cursor++) {
        if (*cursor < '0' || *cursor > '9') exit(1);
        value = value * 10U + (unsigned)(*cursor - '0');
    }
    return value;
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
                g_transitions++;
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

static uint64_t disjoint_join(const Distribution* lhs, const Distribution* rhs) {
    uint64_t total = 0;
    g_pair_tests += (uint64_t)lhs->count * rhs->count;
    for (size_t i = 0; i < lhs->count; i++) {
        for (size_t j = 0; j < rhs->count; j++) {
            if ((lhs->entries[i].mask & rhs->entries[j].mask) == 0) {
                total += lhs->entries[i].weight * rhs->entries[j].weight;
            }
        }
    }
    return total;
}

static void free_distribution(Distribution* distribution) {
    free(distribution->entries);
    *distribution = (Distribution){0};
}

static void unpack_rows(uint64_t key, uint8_t rows[ROWS]) {
    for (int row = ROWS - 1; row >= 0; row--) {
        rows[row] = (uint8_t)key;
        key >>= 8;
    }
}

static U128 evaluate_kernel(uint64_t key) {
    uint8_t rows[ROWS];
    uint8_t left_rows[ROWS];
    uint8_t right_rows[ROWS];
    unpack_rows(key, rows);
    for (int row = 0; row < ROWS; row++) {
        left_rows[row] = rows[row] & (LEFT_PATTERNS - 1);
        right_rows[row] = rows[row] >> LEFT_COLUMNS;
    }
    Distribution left = build_distribution(left_rows, LEFT_COLUMNS, 0);
    Distribution right = build_distribution(right_rows, RIGHT_COLUMNS, 0);
    Distribution left_complement = build_distribution(left_rows, LEFT_COLUMNS, 1);
    Distribution right_complement = build_distribution(right_rows, RIGHT_COLUMNS, 1);
    uint64_t selected = disjoint_join(&left, &right);
    uint64_t complement = disjoint_join(&left_complement, &right_complement);
    free_distribution(&left);
    free_distribution(&right);
    free_distribution(&left_complement);
    free_distribution(&right_complement);
    return (U128)selected * complement;
}

static void write_result(const char* path, const Result* result) {
    FILE* file = fopen(path, "w");
    if (!file) {
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    fprintf(file, "RECT7T4_V1\n");
    fprintf(file, "orbit_records %llu\n", (unsigned long long)result->orbit_records);
    fprintf(file, "labelled_weight ");
    fprint_u128(file, result->labelled_weight);
    fprintf(file, "\nkernels %llu\n", (unsigned long long)result->kernels);
    fprintf(file, "selected_weight ");
    fprint_u128(file, result->selected_weight);
    fprintf(file, "\ncontribution ");
    fprint_u128(file, result->contribution);
    fprintf(file, "\nend\n");
    if (fclose(file) != 0) exit(1);
}

static Result solve_file(const char* path, uint64_t start_index, uint64_t end_index) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    if (fread(magic, sizeof(magic), 1, file) != 1 || memcmp(magic, "R7ORB01", 7) != 0 ||
        fread(&columns, sizeof(columns), 1, file) != 1 || columns != 7 ||
        fread(&count, sizeof(count), 1, file) != 1) {
        fprintf(stderr, "invalid orbit file %s\n", path);
        exit(1);
    }
    if (end_index == 0) end_index = count;
    if (start_index > end_index || end_index > count) {
        fprintf(stderr, "invalid record range [%llu,%llu) for %llu records\n",
                (unsigned long long)start_index, (unsigned long long)end_index,
                (unsigned long long)count);
        exit(2);
    }
    if (fseeko(file, (off_t)(20U + start_index * sizeof(OrbitRecord)), SEEK_SET) != 0) {
        fprintf(stderr, "failed seeking %s\n", path);
        exit(1);
    }
    uint64_t selected_count = end_index - start_index;
    Result result = {0};
    double start = seconds_now();
    for (uint64_t index = 0; index < selected_count; index++) {
        OrbitRecord record;
        if (fread(&record, sizeof(record), 1, file) != 1) exit(1);
        result.orbit_records++;
        result.labelled_weight += record.weight;
        uint8_t rows[ROWS];
        unpack_rows(record.key, rows);
        int cells = 0;
        for (int row = 0; row < ROWS; row++) cells += __builtin_popcount((unsigned)rows[row]);
        if (cells <= 24) {
            result.kernels++;
            result.selected_weight += record.weight;
            result.contribution += (U128)2 * record.weight * evaluate_kernel(record.key);
        }
        if ((index + 1U) % 10000U == 0) {
            printf("progress file=%s range=[%llu,%llu) records=%llu/%llu kernels=%llu elapsed=%.2fs\n",
                   path, (unsigned long long)start_index, (unsigned long long)end_index,
                   (unsigned long long)(index + 1U), (unsigned long long)selected_count,
                   (unsigned long long)result.kernels, seconds_now() - start);
        }
    }
    fclose(file);
    printf("solved file=%s records=%llu kernels=%llu labelled_weight=", path,
           (unsigned long long)result.orbit_records, (unsigned long long)result.kernels);
    print_u128(result.labelled_weight);
    printf(" selected_weight=");
    print_u128(result.selected_weight);
    printf(" contribution=");
    print_u128(result.contribution);
    printf(" transitions=%llu pair_tests=%llu time=%.3fs\n",
           (unsigned long long)g_transitions, (unsigned long long)g_pair_tests,
           seconds_now() - start);
    return result;
}

static Result read_result(const char* path) {
    FILE* file = fopen(path, "r");
    if (!file) exit(1);
    char line[256];
    Result result = {0};
    if (!fgets(line, sizeof(line), file) || strcmp(line, "RECT7T4_V1\n") != 0) exit(1);
    int fields = 0;
    while (fgets(line, sizeof(line), file)) {
        char* newline = strchr(line, '\n');
        if (newline) *newline = '\0';
        char* space = strchr(line, ' ');
        if (strcmp(line, "end") == 0) break;
        if (!space) exit(1);
        *space++ = '\0';
        if (strcmp(line, "orbit_records") == 0) {
            result.orbit_records = strtoull(space, NULL, 10);
            fields++;
        } else if (strcmp(line, "labelled_weight") == 0) {
            result.labelled_weight = parse_u128(space);
            fields++;
        } else if (strcmp(line, "kernels") == 0) {
            result.kernels = strtoull(space, NULL, 10);
            fields++;
        } else if (strcmp(line, "selected_weight") == 0) {
            result.selected_weight = parse_u128(space);
            fields++;
        } else if (strcmp(line, "contribution") == 0) {
            result.contribution = parse_u128(space);
            fields++;
        } else {
            exit(1);
        }
    }
    fclose(file);
    if (fields != 5) exit(1);
    return result;
}

static void aggregate_results(int count, char** paths) {
    Result total = {0};
    for (int i = 0; i < count; i++) {
        Result part = read_result(paths[i]);
        total.orbit_records += part.orbit_records;
        total.labelled_weight += part.labelled_weight;
        total.kernels += part.kernels;
        total.selected_weight += part.selected_weight;
        total.contribution += part.contribution;
    }
    U128 expected = (U128)UINT64_C(7016720048108792558) * 100000000U +
                    UINT64_C(76925440);
    int valid = total.orbit_records == UINT64_C(33642660) &&
                total.labelled_weight == ((U128)1 << 49) &&
                total.kernels == UINT64_C(16821330) &&
                total.selected_weight == ((U128)1 << 48) &&
                total.contribution == expected;
    printf("aggregate files=%d orbit_records=%llu labelled_weight=", count,
           (unsigned long long)total.orbit_records);
    print_u128(total.labelled_weight);
    printf(" kernels=%llu selected_weight=", (unsigned long long)total.kernels);
    print_u128(total.selected_weight);
    printf("\nT4(7,7)=");
    print_u128(total.contribution);
    printf(" expected=");
    print_u128(expected);
    printf(" %s\n", valid ? "OK" : "INCOMPLETE_OR_FAIL");
    if (!valid) exit(1);
}

static void usage(const char* program) {
    fprintf(stderr,
            "Usage:\n"
            "  %s solve RESULT.txt ORBITS.orbits [START END]\n"
            "  %s aggregate RESULT.txt...\n",
            program, program);
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    int pair = 0;
    for (int u = 0; u < ROWS; u++) {
        for (int v = u + 1; v < ROWS; v++) g_pair_index[u][v] = pair++;
    }
    if ((argc == 4 || argc == 6) && strcmp(argv[1], "solve") == 0) {
        uint64_t start = argc == 6 ? strtoull(argv[4], NULL, 10) : 0;
        uint64_t end = argc == 6 ? strtoull(argv[5], NULL, 10) : 0;
        Result result = solve_file(argv[3], start, end);
        write_result(argv[2], &result);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "aggregate") == 0) {
        aggregate_results(argc - 2, argv + 2);
        return 0;
    }
    usage(argv[0]);
    return 2;
}
