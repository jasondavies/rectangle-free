/* Probe exact weighted-ZDD compression of two-column token histories. */
#define _POSIX_C_SOURCE 200809L
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned __int128 U128;

typedef struct {
    uint64_t mask;
    uint64_t weight;
    uint8_t used;
} SupportEntry;

typedef struct {
    SupportEntry* entries;
    size_t capacity;
    size_t count;
} SupportMap;

typedef struct {
    uint64_t ordered;
    uint64_t weight;
} Item;

typedef struct {
    uint32_t level;
    uint32_t low;
    uint32_t high;
    uint64_t terminal_weight;
    uint8_t terminal;
} ZddNode;

typedef struct {
    ZddNode* nodes;
    size_t count;
    size_t capacity;
    uint32_t* slots;
    size_t slot_capacity;
    uint64_t* memo;
    uint32_t* memo_stamp;
    uint64_t* terminal_weights;
    uint32_t* terminal_ids;
    size_t terminal_count;
    size_t terminal_capacity;
    uint32_t stamp;
    uint64_t visits;
    int bits;
} Zdd;

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static void* xcalloc(size_t count, size_t size) {
    void* ptr = calloc(count, size);
    if (!ptr) {
        fprintf(stderr, "allocation failed for %.1f MiB\n",
                (double)count * (double)size / 1048576.0);
        exit(1);
    }
    return ptr;
}

static void support_init(SupportMap* map, size_t capacity) {
    map->capacity = capacity;
    map->count = 0;
    map->entries = xcalloc(capacity, sizeof(map->entries[0]));
}

static void support_insert_raw(SupportMap* map, uint64_t mask, uint64_t weight) {
    size_t slot = (size_t)mix64(mask) & (map->capacity - 1U);
    while (map->entries[slot].used) slot = (slot + 1U) & (map->capacity - 1U);
    map->entries[slot] = (SupportEntry){.mask = mask, .weight = weight, .used = 1};
    map->count++;
}

static void support_rehash(SupportMap* map) {
    SupportEntry* old = map->entries;
    size_t old_capacity = map->capacity;
    support_init(map, old_capacity << 1);
    for (size_t i = 0; i < old_capacity; i++) {
        if (old[i].used) support_insert_raw(map, old[i].mask, old[i].weight);
    }
    free(old);
}

static void support_add_weight(SupportMap* map, uint64_t mask, uint64_t weight) {
    if ((map->count + 1U) * 10U >= map->capacity * 7U) support_rehash(map);
    size_t slot = (size_t)mix64(mask) & (map->capacity - 1U);
    while (map->entries[slot].used) {
        if (map->entries[slot].mask == mask) {
            map->entries[slot].weight += weight;
            return;
        }
        slot = (slot + 1U) & (map->capacity - 1U);
    }
    support_insert_raw(map, mask, weight);
}

static int item_compare(const void* lhs_ptr, const void* rhs_ptr) {
    const Item* lhs = lhs_ptr;
    const Item* rhs = rhs_ptr;
    return lhs->ordered < rhs->ordered ? -1 : lhs->ordered > rhs->ordered;
}

static uint64_t node_hash(uint32_t level, uint32_t low, uint32_t high) {
    return mix64(((uint64_t)level << 48) ^ ((uint64_t)low << 24) ^ high);
}

static void zdd_slots_rebuild(Zdd* zdd) {
    free(zdd->slots);
    zdd->slot_capacity = 1;
    while (zdd->slot_capacity < zdd->count * 2U + 16U) zdd->slot_capacity <<= 1;
    zdd->slots = xcalloc(zdd->slot_capacity, sizeof(zdd->slots[0]));
    for (uint32_t id = 1; id < zdd->count; id++) {
        if (zdd->nodes[id].terminal) continue;
        ZddNode node = zdd->nodes[id];
        size_t slot = (size_t)node_hash(node.level, node.low, node.high) &
                      (zdd->slot_capacity - 1U);
        while (zdd->slots[slot]) slot = (slot + 1U) & (zdd->slot_capacity - 1U);
        zdd->slots[slot] = id;
    }
}

static uint32_t zdd_append(Zdd* zdd, ZddNode node) {
    if (zdd->count == zdd->capacity) {
        zdd->capacity = zdd->capacity ? zdd->capacity << 1 : 1024;
        zdd->nodes = realloc(zdd->nodes, zdd->capacity * sizeof(zdd->nodes[0]));
        if (!zdd->nodes) exit(1);
    }
    uint32_t id = (uint32_t)zdd->count++;
    zdd->nodes[id] = node;
    return id;
}

static uint32_t zdd_terminal(Zdd* zdd, uint64_t weight) {
    if (weight == 0) return 0;
    for (size_t i = 0; i < zdd->terminal_count; i++) {
        if (zdd->terminal_weights[i] == weight) return zdd->terminal_ids[i];
    }
    if (zdd->terminal_count == zdd->terminal_capacity) {
        zdd->terminal_capacity = zdd->terminal_capacity ? zdd->terminal_capacity << 1 : 16;
        zdd->terminal_weights =
            realloc(zdd->terminal_weights,
                    zdd->terminal_capacity * sizeof(zdd->terminal_weights[0]));
        zdd->terminal_ids =
            realloc(zdd->terminal_ids, zdd->terminal_capacity * sizeof(zdd->terminal_ids[0]));
        if (!zdd->terminal_weights || !zdd->terminal_ids) exit(1);
    }
    uint32_t id = zdd_append(zdd, (ZddNode){.terminal_weight = weight, .terminal = 1});
    zdd->terminal_weights[zdd->terminal_count] = weight;
    zdd->terminal_ids[zdd->terminal_count] = id;
    zdd->terminal_count++;
    return id;
}

static uint32_t zdd_node(Zdd* zdd, uint32_t level, uint32_t low, uint32_t high) {
    if (high == 0) return low;
    if ((zdd->count + 1U) * 10U >= zdd->slot_capacity * 7U) zdd_slots_rebuild(zdd);
    size_t slot = (size_t)node_hash(level, low, high) & (zdd->slot_capacity - 1U);
    while (zdd->slots[slot]) {
        uint32_t id = zdd->slots[slot];
        ZddNode* node = &zdd->nodes[id];
        if (node->level == level && node->low == low && node->high == high) return id;
        slot = (slot + 1U) & (zdd->slot_capacity - 1U);
    }
    uint32_t id = zdd_append(zdd, (ZddNode){.level = level, .low = low, .high = high});
    zdd->slots[slot] = id;
    return id;
}

static uint32_t zdd_build(Zdd* zdd, const Item* items, size_t begin, size_t end, int level) {
    if (begin == end) return 0;
    if (level == zdd->bits) return zdd_terminal(zdd, items[begin].weight);
    uint64_t bit = UINT64_C(1) << (zdd->bits - level - 1);
    size_t split = begin;
    while (split < end && (items[split].ordered & bit) == 0) split++;
    uint32_t low = zdd_build(zdd, items, begin, split, level + 1);
    uint32_t high = zdd_build(zdd, items, split, end, level + 1);
    return zdd_node(zdd, (uint32_t)level, low, high);
}

static uint64_t zdd_query_rec(Zdd* zdd, uint32_t id, uint64_t forbidden_ordered) {
    if (id == 0) return 0;
    if (zdd->nodes[id].terminal) return zdd->nodes[id].terminal_weight;
    if (zdd->memo_stamp[id] == zdd->stamp) return zdd->memo[id];
    zdd->visits++;
    ZddNode node = zdd->nodes[id];
    uint64_t bit = UINT64_C(1) << (zdd->bits - (int)node.level - 1);
    uint64_t value = zdd_query_rec(zdd, node.low, forbidden_ordered);
    if ((forbidden_ordered & bit) == 0) {
        value += zdd_query_rec(zdd, node.high, forbidden_ordered);
    }
    zdd->memo_stamp[id] = zdd->stamp;
    zdd->memo[id] = value;
    return value;
}

static uint64_t permute_mask(uint64_t mask, const int* order, int bits) {
    uint64_t out = 0;
    for (int level = 0; level < bits; level++) {
        if ((mask >> order[level]) & 1U) out |= UINT64_C(1) << (bits - level - 1);
    }
    return out;
}

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

static size_t middle_rank_mod2(const SupportMap* support, int pairs, uint64_t divisor) {
    int split_bits = 2 * pairs;
    if (split_bits >= 31) return 0;
    size_t dimension = (size_t)1U << split_bits;
    size_t words = (dimension + 63U) / 64U;
    uint64_t* matrix = xcalloc(dimension * words, sizeof(matrix[0]));
    uint64_t low_mask = (UINT64_C(1) << split_bits) - 1U;
    for (size_t i = 0; i < support->capacity; i++) {
        if (!support->entries[i].used || ((support->entries[i].weight / divisor) & 1U) == 0) continue;
        size_t row = (size_t)(support->entries[i].mask & low_mask);
        size_t column = (size_t)(support->entries[i].mask >> split_bits);
        matrix[row * words + column / 64U] ^= UINT64_C(1) << (column & 63U);
    }
    size_t rank = 0;
    for (size_t column = 0; column < dimension && rank < dimension; column++) {
        size_t pivot = rank;
        uint64_t bit = UINT64_C(1) << (column & 63U);
        while (pivot < dimension && (matrix[pivot * words + column / 64U] & bit) == 0) {
            pivot++;
        }
        if (pivot == dimension) continue;
        if (pivot != rank) {
            for (size_t word = column / 64U; word < words; word++) {
                uint64_t tmp = matrix[rank * words + word];
                matrix[rank * words + word] = matrix[pivot * words + word];
                matrix[pivot * words + word] = tmp;
            }
        }
        for (size_t row = rank + 1U; row < dimension; row++) {
            if (matrix[row * words + column / 64U] & bit) {
                for (size_t word = column / 64U; word < words; word++) {
                    matrix[row * words + word] ^= matrix[rank * words + word];
                }
            }
        }
        rank++;
    }
    free(matrix);
    return rank;
}

static size_t middle_rank_mod3(const SupportMap* support, int pairs, uint64_t divisor) {
    int split_bits = 2 * pairs;
    if (split_bits >= 31) return 0;
    size_t dimension = (size_t)1U << split_bits;
    size_t words = (dimension + 63U) / 64U;
    uint64_t* ones = xcalloc(dimension * words, sizeof(ones[0]));
    uint64_t* twos = xcalloc(dimension * words, sizeof(twos[0]));
    uint64_t low_mask = (UINT64_C(1) << split_bits) - 1U;
    for (size_t i = 0; i < support->capacity; i++) {
        if (!support->entries[i].used) continue;
        unsigned residue = (unsigned)((support->entries[i].weight / divisor) % 3U);
        if (!residue) continue;
        size_t row = (size_t)(support->entries[i].mask & low_mask);
        size_t column = (size_t)(support->entries[i].mask >> split_bits);
        uint64_t bit = UINT64_C(1) << (column & 63U);
        if (residue == 1) ones[row * words + column / 64U] |= bit;
        else twos[row * words + column / 64U] |= bit;
    }
    size_t rank = 0;
    for (size_t column = 0; column < dimension && rank < dimension; column++) {
        size_t word0 = column / 64U;
        uint64_t bit = UINT64_C(1) << (column & 63U);
        size_t pivot = rank;
        while (pivot < dimension &&
               ((ones[pivot * words + word0] | twos[pivot * words + word0]) & bit) == 0) {
            pivot++;
        }
        if (pivot == dimension) continue;
        if (pivot != rank) {
            for (size_t word = word0; word < words; word++) {
                uint64_t tmp = ones[rank * words + word];
                ones[rank * words + word] = ones[pivot * words + word];
                ones[pivot * words + word] = tmp;
                tmp = twos[rank * words + word];
                twos[rank * words + word] = twos[pivot * words + word];
                twos[pivot * words + word] = tmp;
            }
        }
        if (twos[rank * words + word0] & bit) {
            for (size_t word = word0; word < words; word++) {
                uint64_t tmp = ones[rank * words + word];
                ones[rank * words + word] = twos[rank * words + word];
                twos[rank * words + word] = tmp;
            }
        }
        for (size_t row = rank + 1U; row < dimension; row++) {
            unsigned coefficient = (ones[row * words + word0] & bit) ? 1U :
                                   (twos[row * words + word0] & bit) ? 2U : 0U;
            if (!coefficient) continue;
            for (size_t word = word0; word < words; word++) {
                uint64_t a1 = ones[row * words + word];
                uint64_t a2 = twos[row * words + word];
                uint64_t p1 = ones[rank * words + word];
                uint64_t p2 = twos[rank * words + word];
                uint64_t b1 = coefficient == 1U ? p2 : p1;
                uint64_t b2 = coefficient == 1U ? p1 : p2;
                uint64_t a0 = ~(a1 | a2);
                uint64_t b0 = ~(b1 | b2);
                ones[row * words + word] = (a1 & b0) | (a0 & b1) | (a2 & b2);
                twos[row * words + word] = (a2 & b0) | (a0 & b2) | (a1 & b1);
            }
        }
        rank++;
    }
    free(ones);
    free(twos);
    return rank;
}

static void subtract_multiple_mod_small(uint8_t* row, const uint8_t* pivot, size_t begin,
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

static size_t middle_rank_mod_small(const SupportMap* support, int pairs, uint64_t divisor,
                                    unsigned modulus) {
    int split_bits = 2 * pairs;
    if (split_bits >= 31) return 0;
    size_t dimension = (size_t)1U << split_bits;
    uint8_t* matrix = xcalloc(dimension * dimension, sizeof(matrix[0]));
    uint8_t** rows = xcalloc(dimension, sizeof(rows[0]));
    for (size_t row = 0; row < dimension; row++) rows[row] = matrix + row * dimension;
    uint64_t low_mask = (UINT64_C(1) << split_bits) - 1U;
    for (size_t i = 0; i < support->capacity; i++) {
        if (!support->entries[i].used) continue;
        size_t row = (size_t)(support->entries[i].mask & low_mask);
        size_t column = (size_t)(support->entries[i].mask >> split_bits);
        rows[row][column] = (uint8_t)((support->entries[i].weight / divisor) % modulus);
    }
    uint8_t inverse[16] = {0};
    for (unsigned value = 1; value < modulus; value++) {
        for (unsigned candidate = 1; candidate < modulus; candidate++) {
            if (value * candidate % modulus == 1U) inverse[value] = (uint8_t)candidate;
        }
    }
    size_t rank = 0;
    for (size_t column = 0; column < dimension && rank < dimension; column++) {
        size_t pivot_row = rank;
        while (pivot_row < dimension && rows[pivot_row][column] == 0) pivot_row++;
        if (pivot_row == dimension) continue;
        uint8_t* tmp = rows[rank];
        rows[rank] = rows[pivot_row];
        rows[pivot_row] = tmp;
        unsigned scale = inverse[rows[rank][column]];
        if (scale != 1) {
            for (size_t c = column; c < dimension; c++) {
                rows[rank][c] = (uint8_t)(rows[rank][c] * scale % modulus);
            }
        }
        for (size_t row = rank + 1U; row < dimension; row++) {
            unsigned coefficient = rows[row][column];
            if (coefficient) {
                subtract_multiple_mod_small(rows[row], rows[rank], column, dimension,
                                            coefficient, modulus);
            }
        }
        rank++;
    }
    free(rows);
    free(matrix);
    return rank;
}

int main(int argc, char** argv) {
    int rows = argc > 1 ? atoi(argv[1]) : 5;
    int depth = argc > 2 ? atoi(argv[2]) : 2;
    int run_join = argc > 3 && strcmp(argv[3], "join") == 0;
    int run_rank = argc > 3 && strcmp(argv[3], "rank") == 0;
    if (rows < 2 || rows > 6) {
        fprintf(stderr, "Usage: %s ROWS [DEPTH [join|rank]], with 2 <= ROWS <= 6\n", argv[0]);
        return 2;
    }
    if (depth < 1 || depth > 4) {
        fprintf(stderr, "DEPTH must be between 1 and 4\n");
        return 2;
    }
    int pairs = rows * (rows - 1) / 2;
    int bits = 4 * pairs;
    int pair_index[6][6] = {{0}};
    int next_pair = 0;
    for (int u = 0; u < rows; u++) {
        for (int v = u + 1; v < rows; v++) pair_index[u][v] = next_pair++;
    }

    size_t columns = (size_t)1U << (2 * rows);
    uint64_t* tokens = xcalloc(columns, sizeof(tokens[0]));
    for (size_t code = 0; code < columns; code++) {
        unsigned colours[6];
        size_t value = code;
        for (int row = 0; row < rows; row++) {
            colours[row] = (unsigned)(value & 3U);
            value >>= 2;
        }
        uint64_t mask = 0;
        for (int u = 0; u < rows; u++) {
            for (int v = u + 1; v < rows; v++) {
                if (colours[u] == colours[v]) {
                    mask |= UINT64_C(1) << (colours[u] * pairs + pair_index[u][v]);
                }
            }
        }
        tokens[code] = mask;
    }

    SupportMap increments;
    support_init(&increments, 1U << 12);
    for (size_t column = 0; column < columns; column++) {
        support_add_weight(&increments, tokens[column], 1);
    }

    double build_start = seconds_now();
    SupportMap support;
    support_init(&support, 1U << 12);
    for (size_t i = 0; i < increments.capacity; i++) {
        if (increments.entries[i].used) {
            support_add_weight(&support, increments.entries[i].mask,
                               increments.entries[i].weight);
        }
    }
    for (int d = 2; d <= depth; d++) {
        SupportMap next;
        support_init(&next, support.capacity);
        for (size_t i = 0; i < support.capacity; i++) {
            if (!support.entries[i].used) continue;
            for (size_t j = 0; j < increments.capacity; j++) {
                if (!increments.entries[j].used) continue;
                if ((support.entries[i].mask & increments.entries[j].mask) == 0) {
                    support_add_weight(&next,
                                       support.entries[i].mask | increments.entries[j].mask,
                                       support.entries[i].weight * increments.entries[j].weight);
                }
            }
        }
        free(support.entries);
        support = next;
    }
    uint64_t compatible_pairs = 0;
    for (size_t i = 0; i < support.capacity; i++) {
        if (support.entries[i].used) compatible_pairs += support.entries[i].weight;
    }
    double build_seconds = seconds_now() - build_start;
    printf("rows=%d depth=%d tokens=%d columns=%zu increments=%zu total=%llu support=%zu build=%.3fs\n",
           rows, depth, bits, columns, increments.count, (unsigned long long)compatible_pairs,
           support.count, build_seconds);

    if (run_rank) {
        if (rows > 4) {
            fprintf(stderr, "rank currently requires ROWS<=4 (the dense matricization limit)\n");
            return 2;
        }
        uint64_t content_gcd = 0;
        for (size_t i = 0; i < support.capacity; i++) {
            if (!support.entries[i].used) continue;
            uint64_t a = content_gcd;
            uint64_t b = support.entries[i].weight;
            while (b) {
                uint64_t remainder = a % b;
                a = b;
                b = remainder;
            }
            content_gcd = a;
        }
        double rank_start = seconds_now();
        size_t rank2 = middle_rank_mod2(&support, pairs, content_gcd);
        size_t rank3 = middle_rank_mod3(&support, pairs, content_gcd);
        size_t rank5 = middle_rank_mod_small(&support, pairs, content_gcd, 5);
        size_t rank7 = middle_rank_mod_small(&support, pairs, content_gcd, 7);
        size_t rank11 = middle_rank_mod_small(&support, pairs, content_gcd, 11);
        size_t rank13 = middle_rank_mod_small(&support, pairs, content_gcd, 13);
        size_t rank_dimension = (size_t)1U << (2 * pairs);
        printf("middle_2+2_content_gcd=%llu ranks=mod2:%zu mod3:%zu mod5:%zu mod7:%zu mod11:%zu mod13:%zu /%zu rank=%.3fs\n",
               (unsigned long long)content_gcd, rank2, rank3, rank5, rank7, rank11, rank13,
               rank_dimension, seconds_now() - rank_start);
    }

    if (run_join) {
        if (depth != 2) {
            fprintf(stderr, "join currently requires DEPTH=2\n");
            return 2;
        }
        uint64_t* support_by_size = xcalloc((size_t)bits + 1U, sizeof(uint64_t));
        U128* weight_by_size = xcalloc((size_t)bits + 1U, sizeof(U128));
        for (size_t i = 0; i < support.capacity; i++) {
            if (!support.entries[i].used) continue;
            int size = __builtin_popcountll(support.entries[i].mask);
            support_by_size[size]++;
            weight_by_size[size] += support.entries[i].weight;
        }
        U128 raw_support_pairs = (U128)support.count * support.count;
        U128 kept_support_pairs = 0;
        U128 raw_weight_pairs = (U128)compatible_pairs * compatible_pairs;
        U128 kept_weight_pairs = 0;
        printf("strata:");
        for (int size = 0; size <= bits; size++) {
            if (!support_by_size[size]) continue;
            printf(" %d:%llu/", size, (unsigned long long)support_by_size[size]);
            print_u128(weight_by_size[size]);
            for (int other = 0; other + size <= bits; other++) {
                kept_support_pairs += (U128)support_by_size[size] * support_by_size[other];
                kept_weight_pairs += weight_by_size[size] * weight_by_size[other];
            }
        }
        printf("\ncardinality_kept_support=");
        print_u128(kept_support_pairs);
        printf("/");
        print_u128(raw_support_pairs);
        printf(" (%.3f%%) cardinality_kept_weight=", 100.0 * (double)kept_support_pairs /
                                                        (double)raw_support_pairs);
        print_u128(kept_weight_pairs);
        printf("/");
        print_u128(raw_weight_pairs);
        printf(" (%.3f%%)\n", 100.0 * (double)kept_weight_pairs / (double)raw_weight_pairs);
        free(support_by_size);
        free(weight_by_size);
    }

    Item* items = xcalloc(support.count, sizeof(items[0]));
    size_t item_count = 0;
    for (size_t i = 0; i < support.capacity; i++) {
        if (!support.entries[i].used) continue;
        items[item_count++] = (Item){.ordered = support.entries[i].mask,
                                    .weight = support.entries[i].weight};
    }

    const char* names[2] = {"colour-major", "pair-major"};
    for (int ordering = 0; ordering < 2; ordering++) {
        int order[64];
        int pos = 0;
        if (ordering == 0) {
            for (int c = 0; c < 4; c++) for (int p = 0; p < pairs; p++) order[pos++] = c * pairs + p;
        } else {
            for (int p = 0; p < pairs; p++) for (int c = 0; c < 4; c++) order[pos++] = c * pairs + p;
        }
        item_count = 0;
        for (size_t i = 0; i < support.capacity; i++) {
            if (!support.entries[i].used) continue;
            items[item_count++] =
                (Item){.ordered = permute_mask(support.entries[i].mask, order, bits),
                       .weight = support.entries[i].weight};
        }
        qsort(items, item_count, sizeof(items[0]), item_compare);

        Zdd zdd = {.bits = bits, .slot_capacity = 2048};
        zdd.slots = xcalloc(zdd.slot_capacity, sizeof(zdd.slots[0]));
        zdd_append(&zdd, (ZddNode){.terminal = 1, .terminal_weight = 0});
        double zdd_start = seconds_now();
        uint32_t root = zdd_build(&zdd, items, 0, item_count, 0);
        double zdd_seconds = seconds_now() - zdd_start;
        zdd.memo = xcalloc(zdd.count, sizeof(zdd.memo[0]));
        zdd.memo_stamp = xcalloc(zdd.count, sizeof(zdd.memo_stamp[0]));

        uint64_t check = 0;
        zdd.stamp++;
        uint64_t empty_query = zdd_query_rec(&zdd, root, 0);
        size_t queries = columns < 1000 ? columns : 1000;
        printf("order=%s zdd_nodes=%zu ratio=%.3f build=%.3fs empty=%llu\n",
               names[ordering], zdd.count, (double)zdd.count / (double)support.count,
               zdd_seconds, (unsigned long long)empty_query);
        if (run_join && ordering == 0) {
            U128 joined = 0;
            zdd.visits = 0;
            double join_start = seconds_now();
            for (size_t i = 0; i < support.capacity; i++) {
                if (!support.entries[i].used) continue;
                zdd.stamp++;
                uint64_t forbidden = permute_mask(support.entries[i].mask, order, bits);
                joined += (U128)support.entries[i].weight *
                          zdd_query_rec(&zdd, root, forbidden);
            }
            double join_seconds = seconds_now() - join_start;
            printf("exact_F2_join=");
            print_u128(joined);
            printf(" queries=%zu query=%.3fs avg_visits=%.1f\n", support.count, join_seconds,
                   (double)zdd.visits / (double)support.count);
        }
        for (int depth = 1; depth <= 4; depth++) {
            zdd.visits = 0;
            check = 0;
            double query_start = seconds_now();
            for (size_t i = 0; i < queries; i++) {
                uint64_t used = 0;
                size_t cursor = (i * UINT64_C(11400714819323198485)) & (columns - 1U);
                for (int d = 0; d < depth; d++) {
                    size_t attempts = 0;
                    while ((tokens[cursor] & used) != 0 && attempts < columns) {
                        cursor = (cursor + 1U) & (columns - 1U);
                        attempts++;
                    }
                    if (attempts == columns) break;
                    used |= tokens[cursor];
                    cursor = (cursor + 7919U) & (columns - 1U);
                }
                zdd.stamp++;
                uint64_t forbidden = permute_mask(used, order, bits);
                check ^= zdd_query_rec(&zdd, root, forbidden) + (uint64_t)i;
            }
            double query_seconds = seconds_now() - query_start;
            printf("  depth=%d queries=%zu query=%.3fs avg_visits=%.1f check=%llu\n",
                   depth, queries, query_seconds, (double)zdd.visits / (double)queries,
                   (unsigned long long)check);
        }
        free(zdd.nodes);
        free(zdd.slots);
        free(zdd.memo);
        free(zdd.memo_stamp);
        free(zdd.terminal_weights);
        free(zdd.terminal_ids);
    }

    free(items);
    free(support.entries);
    free(increments.entries);
    free(tokens);
    return 0;
}
