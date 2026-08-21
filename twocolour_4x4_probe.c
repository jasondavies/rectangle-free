/* Stratified exact 4+4 two-bit kernel feasibility probe for T_4(8,8). */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned __int128 U128;

enum {
    ROWS = 8,
    HALF_COLUMNS = 4,
    HALF_PATTERNS = 16,
    PAIRS = 28,
    PREFIX_ORBITS = 25207,
    MAX_TABLES = 40320
};

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
    uint8_t histogram[HALF_PATTERNS];
} Prefix;

typedef struct {
    uint8_t cells[HALF_PATTERNS * HALF_PATTERNS];
    uint64_t multiplicity;
} RelativeTable;

typedef struct {
    int prefix_index;
    size_t support[2];
    double build_seconds;
    uint8_t shape;
} Sample;

typedef struct {
    uint16_t lhs;
    uint16_t rhs;
    uint64_t tests;
    uint32_t double_cosets;
} PairWork;

typedef struct {
    uint64_t ordered;
    uint64_t weight;
} OracleItem;

typedef struct {
    uint64_t union_mask;
    uint64_t common_mask;
    uint64_t sum;
    uint64_t bit;
    uint32_t low;
    uint32_t high;
} OracleNode;

typedef struct {
    OracleNode* nodes;
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
    int ordering;
} Oracle;

typedef struct {
    uint64_t value;
    size_t nodes;
    uint64_t visits;
    double build_seconds;
    double query_seconds;
    const char* method;
} OracleJoinResult;

typedef struct {
    uint64_t key;
    uint64_t value;
} PairMemoEntry;

typedef struct {
    uint64_t* keys;
    uint32_t* frequencies;
    uint8_t* used;
    size_t capacity;
    size_t count;
} KeySet;

typedef struct {
    uint64_t* keys;
    uint32_t* values;
    uint8_t* used;
    size_t capacity;
} PrefixIndex;

typedef struct {
    PairMemoEntry* entries;
    size_t capacity;
    size_t count;
    size_t max_count;
    int aborted;
    uint64_t calls;
    uint64_t disjoint_shortcuts;
    uint64_t common_rejects;
} PairMemo;

typedef struct {
    uint64_t key;
    uint64_t value;
} QueryCacheEntry;

typedef struct {
    QueryCacheEntry* entries;
    size_t capacity;
    size_t count;
    size_t max_count;
    uint64_t hits;
    uint64_t misses;
} QueryCache;

typedef struct {
    uint64_t forbidden;
    uint64_t value;
    uint32_t node;
    uint8_t used;
} CrossMemoEntry;

typedef struct {
    CrossMemoEntry* entries;
    size_t capacity;
    size_t count;
    size_t max_count;
    uint64_t hits;
    uint64_t misses;
} CrossMemo;

static int g_pair_index[ROWS][ROWS];
static uint64_t g_factorial[ROWS + 1];
static int g_preimages[24][HALF_PATTERNS];
static int g_permutation[HALF_COLUMNS];
static int g_permutation_count;
static uint8_t g_histogram[HALF_PATTERNS];
static Prefix g_prefixes[PREFIX_ORBITS];
static int g_prefix_count;
static RelativeTable g_tables[MAX_TABLES];
static int g_table_count;
static int g_row_patterns[HALF_PATTERNS];
static int g_column_patterns[HALF_PATTERNS];
static int g_row_type_count;
static int g_column_type_count;
static int g_row_margins[HALF_PATTERNS];
static int g_column_remaining[HALF_PATTERNS];
static uint8_t g_small_table[HALF_PATTERNS][HALF_PATTERNS];
static uint64_t g_table_numerator;
static uint64_t g_distribution_transitions;
static uint32_t g_shape_codes[22];
static int g_shape_representatives[22];
static int g_shape_count;
static uint32_t g_double_coset_counts[22][22];

static void* xcalloc(size_t count, size_t size) {
    void* pointer = calloc(count, size);
    if (!pointer) exit(1);
    return pointer;
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

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static void key_set_init(KeySet* set, size_t capacity) {
    set->capacity = capacity;
    set->count = 0;
    set->keys = xcalloc(capacity, sizeof(set->keys[0]));
    set->frequencies = xcalloc(capacity, sizeof(set->frequencies[0]));
    set->used = xcalloc(capacity, sizeof(set->used[0]));
}

static void key_set_rehash(KeySet* set) {
    KeySet replacement;
    key_set_init(&replacement, set->capacity << 1);
    for (size_t i = 0; i < set->capacity; i++) {
        if (!set->used[i]) continue;
        size_t slot = (size_t)mix64(set->keys[i]) & (replacement.capacity - 1U);
        while (replacement.used[slot]) slot = (slot + 1U) & (replacement.capacity - 1U);
        replacement.used[slot] = 1;
        replacement.keys[slot] = set->keys[i];
        replacement.frequencies[slot] = set->frequencies[i];
        replacement.count++;
    }
    free(set->used);
    free(set->frequencies);
    free(set->keys);
    *set = replacement;
}

static int key_set_add(KeySet* set, uint64_t key) {
    if ((set->count + 1U) * 10U >= set->capacity * 7U) key_set_rehash(set);
    size_t slot = (size_t)mix64(key) & (set->capacity - 1U);
    while (set->used[slot]) {
        if (set->keys[slot] == key) {
            set->frequencies[slot]++;
            return 0;
        }
        slot = (slot + 1U) & (set->capacity - 1U);
    }
    set->used[slot] = 1;
    set->keys[slot] = key;
    set->frequencies[slot] = 1;
    set->count++;
    return 1;
}

static void key_set_free(KeySet* set) {
    free(set->used);
    free(set->frequencies);
    free(set->keys);
    *set = (KeySet){0};
}

static void prefix_index_init(PrefixIndex* index) {
    index->capacity = 65536;
    index->keys = xcalloc(index->capacity, sizeof(index->keys[0]));
    index->values = xcalloc(index->capacity, sizeof(index->values[0]));
    index->used = xcalloc(index->capacity, sizeof(index->used[0]));
}

static void prefix_index_add(PrefixIndex* index, uint64_t key, uint32_t value) {
    size_t slot = (size_t)mix64(key) & (index->capacity - 1U);
    while (index->used[slot]) {
        if (index->keys[slot] == key) exit(1);
        slot = (slot + 1U) & (index->capacity - 1U);
    }
    index->used[slot] = 1;
    index->keys[slot] = key;
    index->values[slot] = value;
}

static uint32_t prefix_index_get(const PrefixIndex* index, uint64_t key) {
    size_t slot = (size_t)mix64(key) & (index->capacity - 1U);
    while (index->used[slot]) {
        if (index->keys[slot] == key) return index->values[slot];
        slot = (slot + 1U) & (index->capacity - 1U);
    }
    fprintf(stderr, "missing canonical prefix histogram\n");
    exit(1);
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

static Distribution build_distribution(const uint8_t rows[ROWS], int complement) {
    Map current;
    map_init(&current, 16);
    map_add(&current, 0, 1);
    for (int column = 0; column < HALF_COLUMNS; column++) {
        unsigned active_rows = 0;
        for (int row = 0; row < ROWS; row++) {
            unsigned pattern = complement ? rows[row] ^ (HALF_PATTERNS - 1) : rows[row];
            if ((pattern >> column) & 1U) active_rows |= 1U << row;
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
                g_distribution_transitions++;
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

static void free_distribution(Distribution* distribution) {
    free(distribution->entries);
    *distribution = (Distribution){0};
}

static uint64_t oracle_node_hash(uint64_t bit, uint32_t low, uint32_t high) {
    return mix64(bit ^ (uint64_t)low << 24 ^ high);
}

static uint32_t oracle_append(Oracle* oracle, OracleNode node) {
    if (oracle->count == oracle->capacity) {
        oracle->capacity = oracle->capacity ? oracle->capacity << 1 : 1024;
        oracle->nodes = realloc(oracle->nodes, oracle->capacity * sizeof(oracle->nodes[0]));
        if (!oracle->nodes) exit(1);
    }
    uint32_t id = (uint32_t)oracle->count++;
    oracle->nodes[id] = node;
    return id;
}

static void oracle_slots_rebuild(Oracle* oracle) {
    free(oracle->slots);
    oracle->slot_capacity = 1;
    while (oracle->slot_capacity < oracle->count * 2U + 16U) oracle->slot_capacity <<= 1;
    oracle->slots = xcalloc(oracle->slot_capacity, sizeof(oracle->slots[0]));
    for (uint32_t id = 1; id < oracle->count; id++) {
        OracleNode node = oracle->nodes[id];
        if (!node.bit) continue;
        size_t slot = (size_t)oracle_node_hash(node.bit, node.low, node.high) &
                      (oracle->slot_capacity - 1U);
        while (oracle->slots[slot]) slot = (slot + 1U) & (oracle->slot_capacity - 1U);
        oracle->slots[slot] = id;
    }
}

static uint32_t oracle_terminal(Oracle* oracle, uint64_t weight) {
    for (size_t i = 0; i < oracle->terminal_count; i++) {
        if (oracle->terminal_weights[i] == weight) return oracle->terminal_ids[i];
    }
    if (oracle->terminal_count == oracle->terminal_capacity) {
        oracle->terminal_capacity = oracle->terminal_capacity ? oracle->terminal_capacity << 1 : 16;
        oracle->terminal_weights =
            realloc(oracle->terminal_weights,
                    oracle->terminal_capacity * sizeof(oracle->terminal_weights[0]));
        oracle->terminal_ids =
            realloc(oracle->terminal_ids,
                    oracle->terminal_capacity * sizeof(oracle->terminal_ids[0]));
        if (!oracle->terminal_weights || !oracle->terminal_ids) exit(1);
    }
    uint32_t id = oracle_append(oracle, (OracleNode){.sum = weight});
    oracle->terminal_weights[oracle->terminal_count] = weight;
    oracle->terminal_ids[oracle->terminal_count++] = id;
    return id;
}

static uint32_t oracle_node(Oracle* oracle, uint32_t level, uint32_t low, uint32_t high) {
    if (high == 0) return low;
    if ((oracle->count + 1U) * 10U >= oracle->slot_capacity * 7U) {
        oracle_slots_rebuild(oracle);
    }
    uint64_t bit = UINT64_C(1) << (2 * PAIRS - (int)level - 1);
    size_t slot = (size_t)oracle_node_hash(bit, low, high) &
                  (oracle->slot_capacity - 1U);
    while (oracle->slots[slot]) {
        uint32_t id = oracle->slots[slot];
        OracleNode* node = &oracle->nodes[id];
        if (node->bit == bit && node->low == low && node->high == high) return id;
        slot = (slot + 1U) & (oracle->slot_capacity - 1U);
    }
    OracleNode* high_node = &oracle->nodes[high];
    uint64_t high_union = high_node->union_mask | bit;
    uint64_t high_common = high_node->common_mask | bit;
    uint64_t union_mask = high_union;
    uint64_t common_mask = high_common;
    uint64_t sum = high_node->sum;
    if (low) {
        OracleNode* low_node = &oracle->nodes[low];
        union_mask |= low_node->union_mask;
        common_mask &= low_node->common_mask;
        sum += low_node->sum;
    }
    uint32_t id = oracle_append(oracle, (OracleNode){.union_mask = union_mask,
                                                     .common_mask = common_mask,
                                                     .sum = sum,
                                                     .bit = bit,
                                                     .low = low,
                                                     .high = high});
    oracle->slots[slot] = id;
    return id;
}

static uint32_t oracle_build_rec(Oracle* oracle, const OracleItem* items,
                                 size_t begin, size_t end, int level) {
    if (begin == end) return 0;
    if (level == 2 * PAIRS) return oracle_terminal(oracle, items[begin].weight);
    uint64_t bit = UINT64_C(1) << (2 * PAIRS - level - 1);
    size_t split = begin;
    while (split < end && (items[split].ordered & bit) == 0) split++;
    uint32_t low = oracle_build_rec(oracle, items, begin, split, level + 1);
    uint32_t high = oracle_build_rec(oracle, items, split, end, level + 1);
    return oracle_node(oracle, (uint32_t)level, low, high);
}

static int oracle_item_compare(const void* lhs_ptr, const void* rhs_ptr) {
    const OracleItem* lhs = lhs_ptr;
    const OracleItem* rhs = rhs_ptr;
    return lhs->ordered < rhs->ordered ? -1 : lhs->ordered > rhs->ordered;
}

static uint64_t oracle_order_mask(uint64_t mask, int ordering) {
    uint64_t ordered = 0;
    int level = 0;
    if (ordering == 0) {
        for (int colour = 0; colour < 2; colour++) {
            for (int pair = 0; pair < PAIRS; pair++) {
                if ((mask >> (colour * PAIRS + pair)) & 1U) {
                    ordered |= UINT64_C(1) << (2 * PAIRS - level - 1);
                }
                level++;
            }
        }
    } else {
        for (int pair = 0; pair < PAIRS; pair++) {
            for (int colour = 0; colour < 2; colour++) {
                if ((mask >> (colour * PAIRS + pair)) & 1U) {
                    ordered |= UINT64_C(1) << (2 * PAIRS - level - 1);
                }
                level++;
            }
        }
    }
    return ordered;
}

static Oracle oracle_build(const Distribution* distribution, int ordering, uint32_t* root) {
    OracleItem* items = xcalloc(distribution->count, sizeof(items[0]));
    for (size_t i = 0; i < distribution->count; i++) {
        items[i] = (OracleItem){.ordered = oracle_order_mask(distribution->entries[i].mask,
                                                             ordering),
                                .weight = distribution->entries[i].weight};
    }
    qsort(items, distribution->count, sizeof(items[0]), oracle_item_compare);
    Oracle oracle = {.slot_capacity = 2048, .ordering = ordering};
    oracle.slots = xcalloc(oracle.slot_capacity, sizeof(oracle.slots[0]));
    oracle_append(&oracle, (OracleNode){0});
    *root = oracle_build_rec(&oracle, items, 0, distribution->count, 0);
    oracle.memo = xcalloc(oracle.count, sizeof(oracle.memo[0]));
    oracle.memo_stamp = xcalloc(oracle.count, sizeof(oracle.memo_stamp[0]));
    free(items);
    return oracle;
}

static uint64_t oracle_query_rec(Oracle* oracle, uint32_t id, uint64_t forbidden) {
    if (!id) return 0;
    OracleNode* node = &oracle->nodes[id];
    if ((node->union_mask & forbidden) == 0) return node->sum;
    if (node->common_mask & forbidden) return 0;
    if (!node->bit) return node->sum;
    if (oracle->memo_stamp[id] == oracle->stamp) return oracle->memo[id];
    oracle->visits++;
    uint64_t value = oracle_query_rec(oracle, node->low, forbidden);
    if ((forbidden & node->bit) == 0) {
        value += oracle_query_rec(oracle, node->high, forbidden);
    }
    oracle->memo_stamp[id] = oracle->stamp;
    oracle->memo[id] = value;
    return value;
}

static void oracle_free(Oracle* oracle) {
    free(oracle->nodes);
    free(oracle->slots);
    free(oracle->memo);
    free(oracle->memo_stamp);
    free(oracle->terminal_weights);
    free(oracle->terminal_ids);
    *oracle = (Oracle){0};
}

static OracleJoinResult oracle_join(const Distribution* queries,
                                    const Distribution* indexed, int ordering) {
    double build_start = seconds_now();
    uint32_t root = 0;
    Oracle oracle = oracle_build(indexed, ordering, &root);
    double build_seconds = seconds_now() - build_start;
    uint64_t value = 0;
    double query_start = seconds_now();
    for (size_t i = 0; i < queries->count; i++) {
        oracle.stamp++;
        uint64_t forbidden = oracle_order_mask(queries->entries[i].mask, ordering);
        value += queries->entries[i].weight * oracle_query_rec(&oracle, root, forbidden);
    }
    double query_seconds = seconds_now() - query_start;
    OracleJoinResult result = {.value = value,
                               .nodes = oracle.count,
                               .visits = oracle.visits,
                               .build_seconds = build_seconds,
                               .query_seconds = query_seconds,
                               .method = "one-sided"};
    oracle_free(&oracle);
    return result;
}

static void pair_memo_init(PairMemo* memo, size_t capacity, size_t max_count) {
    memo->capacity = capacity;
    memo->count = 0;
    memo->max_count = max_count;
    memo->aborted = 0;
    memo->entries = xcalloc(capacity, sizeof(memo->entries[0]));
}

static void pair_memo_insert_raw(PairMemo* memo, uint64_t key, uint64_t value) {
    size_t slot = (size_t)mix64(key) & (memo->capacity - 1U);
    while (memo->entries[slot].key) slot = (slot + 1U) & (memo->capacity - 1U);
    memo->entries[slot] = (PairMemoEntry){.key = key, .value = value};
    memo->count++;
}

static void pair_memo_rehash(PairMemo* memo) {
    PairMemoEntry* old = memo->entries;
    size_t old_capacity = memo->capacity;
    size_t max_count = memo->max_count;
    pair_memo_init(memo, old_capacity << 1, max_count);
    for (size_t i = 0; i < old_capacity; i++) {
        if (old[i].key) pair_memo_insert_raw(memo, old[i].key, old[i].value);
    }
    free(old);
}

static int pair_memo_get(PairMemo* memo, uint64_t key, uint64_t* value) {
    size_t slot = (size_t)mix64(key) & (memo->capacity - 1U);
    while (memo->entries[slot].key) {
        if (memo->entries[slot].key == key) {
            *value = memo->entries[slot].value;
            return 1;
        }
        slot = (slot + 1U) & (memo->capacity - 1U);
    }
    return 0;
}

static void pair_memo_put(PairMemo* memo, uint64_t key, uint64_t value) {
    if (memo->count >= memo->max_count) {
        memo->aborted = 1;
        return;
    }
    if ((memo->count + 1U) * 10U >= memo->capacity * 7U) pair_memo_rehash(memo);
    pair_memo_insert_raw(memo, key, value);
}

static uint64_t oracle_pair_join_rec(const Oracle* lhs, uint32_t lhs_id,
                                     const Oracle* rhs, uint32_t rhs_id,
                                     PairMemo* memo) {
    if (memo->aborted) return 0;
    if (!lhs_id || !rhs_id) return 0;
    const OracleNode* lhs_node = &lhs->nodes[lhs_id];
    const OracleNode* rhs_node = &rhs->nodes[rhs_id];
    memo->calls++;
    if ((lhs_node->union_mask & rhs_node->union_mask) == 0) {
        memo->disjoint_shortcuts++;
        return lhs_node->sum * rhs_node->sum;
    }
    if (lhs_node->common_mask & rhs_node->common_mask) {
        memo->common_rejects++;
        return 0;
    }
    uint64_t key = (uint64_t)lhs_id << 32 | rhs_id;
    uint64_t cached = 0;
    if (pair_memo_get(memo, key, &cached)) return cached;
    uint64_t bit = lhs_node->bit > rhs_node->bit ? lhs_node->bit : rhs_node->bit;
    if (!bit) return lhs_node->sum * rhs_node->sum;
    uint32_t lhs_low = lhs_id;
    uint32_t lhs_high = 0;
    uint32_t rhs_low = rhs_id;
    uint32_t rhs_high = 0;
    if (lhs_node->bit == bit) {
        lhs_low = lhs_node->low;
        lhs_high = lhs_node->high;
    }
    if (rhs_node->bit == bit) {
        rhs_low = rhs_node->low;
        rhs_high = rhs_node->high;
    }
    uint64_t value = oracle_pair_join_rec(lhs, lhs_low, rhs, rhs_low, memo);
    if (memo->aborted) return 0;
    if (lhs_high) value += oracle_pair_join_rec(lhs, lhs_high, rhs, rhs_low, memo);
    if (memo->aborted) return 0;
    if (rhs_high) value += oracle_pair_join_rec(lhs, lhs_low, rhs, rhs_high, memo);
    if (memo->aborted) return 0;
    pair_memo_put(memo, key, value);
    return value;
}

static OracleJoinResult oracle_pair_join(const Distribution* lhs_distribution,
                                         const Distribution* rhs_distribution,
                                         int ordering) {
    double build_start = seconds_now();
    uint32_t lhs_root = 0;
    uint32_t rhs_root = 0;
    Oracle lhs = oracle_build(lhs_distribution, ordering, &lhs_root);
    Oracle rhs = oracle_build(rhs_distribution, ordering, &rhs_root);
    double build_seconds = seconds_now() - build_start;
    PairMemo memo;
    pair_memo_init(&memo, 2048, SIZE_MAX);
    double query_start = seconds_now();
    uint64_t value = oracle_pair_join_rec(&lhs, lhs_root, &rhs, rhs_root, &memo);
    double query_seconds = seconds_now() - query_start;
    OracleJoinResult result = {.value = value,
                               .nodes = lhs.count + rhs.count,
                               .visits = memo.count,
                               .build_seconds = build_seconds,
                               .query_seconds = query_seconds,
                               .method = "apply"};
    free(memo.entries);
    oracle_free(&lhs);
    oracle_free(&rhs);
    return result;
}

static uint64_t disjoint_join(const Distribution* lhs, const Distribution* rhs) {
    uint64_t total = 0;
    for (size_t i = 0; i < lhs->count; i++) {
        for (size_t j = 0; j < rhs->count; j++) {
            if ((lhs->entries[i].mask & rhs->entries[j].mask) == 0) {
                total += lhs->entries[i].weight * rhs->entries[j].weight;
            }
        }
    }
    return total;
}

static uint64_t oracle_query_built(const Distribution* queries, Oracle* indexed,
                                   uint32_t root) {
    uint64_t value = 0;
    for (size_t i = 0; i < queries->count; i++) {
        indexed->stamp++;
        uint64_t forbidden = oracle_order_mask(queries->entries[i].mask,
                                               indexed->ordering);
        value += queries->entries[i].weight *
                 oracle_query_rec(indexed, root, forbidden);
    }
    return value;
}

static void query_cache_init(QueryCache* cache, size_t max_count) {
    cache->capacity = 2048;
    cache->max_count = max_count;
    cache->entries = xcalloc(cache->capacity, sizeof(cache->entries[0]));
}

static void query_cache_insert_raw(QueryCache* cache, uint64_t key, uint64_t value) {
    size_t slot = (size_t)mix64(key) & (cache->capacity - 1U);
    while (cache->entries[slot].key) slot = (slot + 1U) & (cache->capacity - 1U);
    cache->entries[slot] = (QueryCacheEntry){.key = key, .value = value};
    cache->count++;
}

static void query_cache_rehash(QueryCache* cache) {
    QueryCacheEntry* old = cache->entries;
    size_t old_capacity = cache->capacity;
    cache->capacity <<= 1;
    cache->count = 0;
    cache->entries = xcalloc(cache->capacity, sizeof(cache->entries[0]));
    for (size_t i = 0; i < old_capacity; i++) {
        if (old[i].key) query_cache_insert_raw(cache, old[i].key, old[i].value);
    }
    free(old);
}

static uint64_t oracle_cached_query(Oracle* oracle, uint32_t root, uint64_t mask,
                                    QueryCache* cache, CrossMemo* cross_memo);

static void cross_memo_init(CrossMemo* memo, size_t max_count) {
    memo->capacity = 1;
    while (memo->capacity < max_count * 10U / 7U + 16U) memo->capacity <<= 1;
    memo->max_count = max_count;
    memo->entries = xcalloc(memo->capacity, sizeof(memo->entries[0]));
}

static uint64_t oracle_cross_query_rec(Oracle* oracle, uint32_t id,
                                       uint64_t forbidden, CrossMemo* memo) {
    if (!id) return 0;
    OracleNode* node = &oracle->nodes[id];
    forbidden &= node->union_mask;
    if (!forbidden) return node->sum;
    if (node->common_mask & forbidden) return 0;
    if (!node->bit) return node->sum;
    uint64_t hash = mix64(forbidden ^ mix64(id));
    size_t slot = (size_t)hash & (memo->capacity - 1U);
    while (memo->entries[slot].used) {
        if (memo->entries[slot].node == id &&
            memo->entries[slot].forbidden == forbidden) {
            memo->hits++;
            return memo->entries[slot].value;
        }
        slot = (slot + 1U) & (memo->capacity - 1U);
    }
    memo->misses++;
    oracle->visits++;
    uint64_t value = oracle_cross_query_rec(oracle, node->low, forbidden, memo);
    if ((forbidden & node->bit) == 0) {
        value += oracle_cross_query_rec(oracle, node->high, forbidden, memo);
    }
    if (memo->count < memo->max_count) {
        memo->entries[slot] = (CrossMemoEntry){.forbidden = forbidden,
                                               .value = value,
                                               .node = id,
                                               .used = 1};
        memo->count++;
    }
    return value;
}

static uint64_t oracle_cached_query(Oracle* oracle, uint32_t root, uint64_t mask,
                                    QueryCache* cache, CrossMemo* cross_memo) {
    /* Token masks use only 56 bits, so mask + 1 is an unambiguous nonzero key. */
    uint64_t key = mask + 1U;
    size_t slot = (size_t)mix64(key) & (cache->capacity - 1U);
    while (cache->entries[slot].key) {
        if (cache->entries[slot].key == key) {
            cache->hits++;
            return cache->entries[slot].value;
        }
        slot = (slot + 1U) & (cache->capacity - 1U);
    }
    cache->misses++;
    uint64_t forbidden = oracle_order_mask(mask, oracle->ordering);
    uint64_t value = 0;
    if (cross_memo->count >= cross_memo->max_count) {
        oracle->stamp++;
        value = oracle_query_rec(oracle, root, forbidden);
    } else {
        value = oracle_cross_query_rec(oracle, root, forbidden, cross_memo);
    }
    if (cache->count < cache->max_count) {
        if ((cache->count + 1U) * 10U >= cache->capacity * 7U) {
            query_cache_rehash(cache);
        }
        query_cache_insert_raw(cache, key, value);
    }
    return value;
}

static uint64_t oracle_pair_join_built(const Oracle* lhs, uint32_t lhs_root,
                                       const Oracle* rhs, uint32_t rhs_root,
                                       uint64_t* states, int* completed) {
    PairMemo memo;
    pair_memo_init(&memo, 2048, 2000000);
    uint64_t value = oracle_pair_join_rec(lhs, lhs_root, rhs, rhs_root, &memo);
    *states = memo.count;
    *completed = !memo.aborted;
    free(memo.entries);
    return value;
}

/* A deployable selector derived from the structural probes, rather than an
   oracle that runs every implementation and reports the fastest afterward. */
static OracleJoinResult oracle_hybrid_join(const Distribution* lhs_distribution,
                                           const Distribution* rhs_distribution) {
    uint64_t direct_tests =
        (uint64_t)lhs_distribution->count * rhs_distribution->count;
    if (direct_tests <= UINT64_C(1000000)) {
        double query_start = seconds_now();
        uint64_t value = disjoint_join(lhs_distribution, rhs_distribution);
        return (OracleJoinResult){.value = value,
                                  .query_seconds = seconds_now() - query_start,
                                  .method = "direct"};
    }

    double build_start = seconds_now();
    uint32_t lhs_root = 0;
    uint32_t rhs_root = 0;
    Oracle lhs = oracle_build(lhs_distribution, 1, &lhs_root);
    Oracle rhs = oracle_build(rhs_distribution, 1, &rhs_root);
    double build_seconds = seconds_now() - build_start;
    size_t nodes = lhs.count + rhs.count;
    uint64_t visits = 0;
    uint64_t value = 0;
    const char* method = NULL;
    double query_start = seconds_now();
    if (nodes <= 50000) {
        int apply_completed = 0;
        value = oracle_pair_join_built(&lhs, lhs_root, &rhs, rhs_root, &visits,
                                       &apply_completed);
        if (apply_completed) method = "apply";
    }
    if (!method && lhs.count <= rhs.count) {
        value = oracle_query_built(rhs_distribution, &lhs, lhs_root);
        visits += lhs.visits;
        method = nodes <= 50000 ? "apply->rhs-to-lhs" : "rhs-to-lhs";
    } else if (!method) {
        value = oracle_query_built(lhs_distribution, &rhs, rhs_root);
        visits += rhs.visits;
        method = nodes <= 50000 ? "apply->lhs-to-rhs" : "lhs-to-rhs";
    }
    double query_seconds = seconds_now() - query_start;
    oracle_free(&lhs);
    oracle_free(&rhs);
    return (OracleJoinResult){.value = value,
                              .nodes = nodes,
                              .visits = visits,
                              .build_seconds = build_seconds,
                              .query_seconds = query_seconds,
                              .method = method};
}

static void generate_permutations(int depth, unsigned used) {
    if (depth == HALF_COLUMNS) {
        for (int pattern = 0; pattern < HALF_PATTERNS; pattern++) {
            int image = 0;
            for (int bit = 0; bit < HALF_COLUMNS; bit++) {
                if ((pattern >> bit) & 1) image |= 1 << g_permutation[bit];
            }
            g_preimages[g_permutation_count][image] = pattern;
        }
        g_permutation_count++;
        return;
    }
    for (int bit = 0; bit < HALF_COLUMNS; bit++) {
        if ((used >> bit) & 1U) continue;
        g_permutation[depth] = bit;
        generate_permutations(depth + 1, used | (1U << bit));
    }
}

static int is_canonical(void) {
    for (int permutation = 0; permutation < g_permutation_count; permutation++) {
        for (int pattern = 0; pattern < HALF_PATTERNS; pattern++) {
            unsigned transformed = g_histogram[g_preimages[permutation][pattern]];
            if (transformed < g_histogram[pattern]) return 0;
            if (transformed > g_histogram[pattern]) break;
        }
    }
    return 1;
}

static void enumerate_histograms(int pattern, int remaining) {
    if (pattern == HALF_PATTERNS - 1) {
        g_histogram[pattern] = (uint8_t)remaining;
        if (is_canonical()) {
            if (g_prefix_count == PREFIX_ORBITS) exit(1);
            memcpy(g_prefixes[g_prefix_count++].histogram, g_histogram, sizeof(g_histogram));
        }
        return;
    }
    for (int count = 0; count <= remaining; count++) {
        g_histogram[pattern] = (uint8_t)count;
        enumerate_histograms(pattern + 1, remaining - count);
    }
}

static void histogram_rows(const uint8_t histogram[HALF_PATTERNS], uint8_t rows[ROWS]) {
    int row = 0;
    for (int pattern = 0; pattern < HALF_PATTERNS; pattern++) {
        for (int copy = 0; copy < histogram[pattern]; copy++) rows[row++] = (uint8_t)pattern;
    }
}

static uint8_t histogram_shape(const uint8_t histogram[HALF_PATTERNS], int prefix_index) {
    uint8_t parts[ROWS];
    int count = 0;
    for (int pattern = 0; pattern < HALF_PATTERNS; pattern++) {
        if (histogram[pattern]) parts[count++] = histogram[pattern];
    }
    for (int i = 1; i < count; i++) {
        uint8_t value = parts[i];
        int j = i;
        while (j && parts[j - 1] < value) {
            parts[j] = parts[j - 1];
            j--;
        }
        parts[j] = value;
    }
    uint32_t code = 0;
    for (int i = 0; i < count; i++) code = code * 9U + parts[i];
    for (int shape = 0; shape < g_shape_count; shape++) {
        if (g_shape_codes[shape] == code) return (uint8_t)shape;
    }
    if (g_shape_count == 22) exit(1);
    g_shape_codes[g_shape_count] = code;
    g_shape_representatives[g_shape_count] = prefix_index;
    return (uint8_t)g_shape_count++;
}

static void store_table(void) {
    if (g_table_count == MAX_TABLES) exit(1);
    RelativeTable* table = &g_tables[g_table_count++];
    memset(table->cells, 0, sizeof(table->cells));
    uint64_t denominator = 1;
    for (int row = 0; row < g_row_type_count; row++) {
        for (int column = 0; column < g_column_type_count; column++) {
            unsigned count = g_small_table[row][column];
            table->cells[g_row_patterns[row] |
                         (g_column_patterns[column] << HALF_COLUMNS)] = (uint8_t)count;
            denominator *= g_factorial[count];
        }
    }
    table->multiplicity = g_table_numerator / denominator;
}

static void assign_table_row(int row);

static void assign_table_cells(int row, int column, int remaining) {
    if (column == g_column_type_count - 1) {
        if (remaining > g_column_remaining[column]) return;
        g_small_table[row][column] = (uint8_t)remaining;
        g_column_remaining[column] -= remaining;
        assign_table_row(row + 1);
        g_column_remaining[column] += remaining;
        return;
    }
    int maximum = remaining < g_column_remaining[column] ? remaining : g_column_remaining[column];
    for (int value = 0; value <= maximum; value++) {
        g_small_table[row][column] = (uint8_t)value;
        g_column_remaining[column] -= value;
        assign_table_cells(row, column + 1, remaining - value);
        g_column_remaining[column] += value;
    }
}

static void assign_table_row(int row) {
    if (row == g_row_type_count - 1) {
        int sum = 0;
        for (int column = 0; column < g_column_type_count; column++) {
            g_small_table[row][column] = (uint8_t)g_column_remaining[column];
            sum += g_column_remaining[column];
        }
        if (sum == g_row_margins[row]) store_table();
        return;
    }
    assign_table_cells(row, 0, g_row_margins[row]);
}

static void generate_tables(const Prefix* lhs, const Prefix* rhs) {
    g_row_type_count = 0;
    g_column_type_count = 0;
    g_table_numerator = 1;
    memset(g_small_table, 0, sizeof(g_small_table));
    for (int pattern = 0; pattern < HALF_PATTERNS; pattern++) {
        if (lhs->histogram[pattern]) {
            g_row_patterns[g_row_type_count] = pattern;
            g_row_margins[g_row_type_count++] = lhs->histogram[pattern];
            g_table_numerator *= g_factorial[lhs->histogram[pattern]];
        }
        if (rhs->histogram[pattern]) {
            g_column_patterns[g_column_type_count] = pattern;
            g_column_remaining[g_column_type_count++] = rhs->histogram[pattern];
            g_table_numerator *= g_factorial[rhs->histogram[pattern]];
        }
    }
    g_table_count = 0;
    assign_table_row(0);
    uint64_t sum = 0;
    for (int i = 0; i < g_table_count; i++) sum += g_tables[i].multiplicity;
    if (sum != g_factorial[ROWS]) {
        fprintf(stderr, "double-coset checksum failed: %llu\n", (unsigned long long)sum);
        exit(1);
    }
}

static uint32_t double_coset_count(uint8_t lhs_shape, uint8_t rhs_shape) {
    uint32_t count = g_double_coset_counts[lhs_shape][rhs_shape];
    if (count) return count;
    generate_tables(&g_prefixes[g_shape_representatives[lhs_shape]],
                    &g_prefixes[g_shape_representatives[rhs_shape]]);
    count = (uint32_t)g_table_count;
    g_double_coset_counts[lhs_shape][rhs_shape] = count;
    g_double_coset_counts[rhs_shape][lhs_shape] = count;
    return count;
}

static void table_rows(const RelativeTable* table, uint8_t lhs_rows[ROWS],
                       uint8_t rhs_rows[ROWS]) {
    int row = 0;
    for (int combined = 0; combined < HALF_PATTERNS * HALF_PATTERNS; combined++) {
        for (int copy = 0; copy < table->cells[combined]; copy++) {
            lhs_rows[row] = (uint8_t)(combined & (HALF_PATTERNS - 1));
            rhs_rows[row] = (uint8_t)(combined >> HALF_COLUMNS);
            row++;
        }
    }
}

static void table_rhs_in_canonical_lhs_order(const RelativeTable* table,
                                             uint8_t rhs_rows[ROWS]) {
    int row = 0;
    for (int lhs_pattern = 0; lhs_pattern < HALF_PATTERNS; lhs_pattern++) {
        for (int rhs_pattern = 0; rhs_pattern < HALF_PATTERNS; rhs_pattern++) {
            int combined = lhs_pattern | (rhs_pattern << HALF_COLUMNS);
            for (int copy = 0; copy < table->cells[combined]; copy++) {
                rhs_rows[row++] = (uint8_t)rhs_pattern;
            }
        }
    }
    if (row != ROWS) exit(1);
}

static void rhs_old_to_new_rows(const Prefix* rhs_prefix,
                                const uint8_t rhs_rows[ROWS],
                                uint8_t old_to_new[ROWS]) {
    uint8_t next_old[HALF_PATTERNS];
    int offset = 0;
    for (int pattern = 0; pattern < HALF_PATTERNS; pattern++) {
        next_old[pattern] = (uint8_t)offset;
        offset += rhs_prefix->histogram[pattern];
    }
    for (int new_row = 0; new_row < ROWS; new_row++) {
        int pattern = rhs_rows[new_row];
        int old_row = next_old[pattern]++;
        old_to_new[old_row] = (uint8_t)new_row;
    }
}

static uint64_t permute_token_mask(uint64_t mask,
                                   const uint8_t old_to_new[ROWS]) {
    uint8_t pair_image[PAIRS];
    int old_pair = 0;
    for (int old_u = 0; old_u < ROWS; old_u++) {
        for (int old_v = old_u + 1; old_v < ROWS; old_v++) {
            int new_u = old_to_new[old_u];
            int new_v = old_to_new[old_v];
            if (new_u > new_v) {
                int swap = new_u;
                new_u = new_v;
                new_v = swap;
            }
            pair_image[old_pair++] = (uint8_t)g_pair_index[new_u][new_v];
        }
    }
    uint64_t result = 0;
    while (mask) {
        int bit = __builtin_ctzll(mask);
        int colour = bit / PAIRS;
        int pair = bit % PAIRS;
        result |= UINT64_C(1) << (colour * PAIRS + pair_image[pair]);
        mask &= mask - 1U;
    }
    return result;
}

typedef struct {
    double seconds;
    size_t cache_entries;
    size_t subproblems;
    uint64_t hits;
    uint64_t misses;
    uint64_t subproblem_hits;
    uint64_t visits;
} AlignmentSideStats;

static AlignmentSideStats alignment_side_values(
    const Prefix* lhs_prefix, const Prefix* rhs_prefix,
    const Distribution* lhs, const Distribution* rhs, size_t table_count,
    uint64_t values[MAX_TABLES]) {
    double start = seconds_now();
    uint32_t root = 0;
    Oracle oracle = oracle_build(lhs, 1, &root);
    QueryCache cache = {0};
    query_cache_init(&cache, 2000000);
    CrossMemo cross_memo = {0};
    cross_memo_init(&cross_memo, 200000);
    uint8_t rhs_rows[ROWS];
    uint8_t old_to_new[ROWS];
    for (size_t table_index = 0; table_index < table_count; table_index++) {
        table_rhs_in_canonical_lhs_order(&g_tables[table_index], rhs_rows);
        rhs_old_to_new_rows(rhs_prefix, rhs_rows, old_to_new);
        uint64_t value = 0;
        for (size_t entry = 0; entry < rhs->count; entry++) {
            uint64_t mask = permute_token_mask(rhs->entries[entry].mask, old_to_new);
            value += rhs->entries[entry].weight *
                     oracle_cached_query(&oracle, root, mask, &cache, &cross_memo);
        }
        values[table_index] = value;
    }
    AlignmentSideStats stats = {.seconds = seconds_now() - start,
                                .cache_entries = cache.count,
                                .subproblems = cross_memo.count,
                                .hits = cache.hits,
                                .misses = cache.misses,
                                .subproblem_hits = cross_memo.hits,
                                .visits = oracle.visits};
    free(cache.entries);
    free(cross_memo.entries);
    oracle_free(&oracle);
    (void)lhs_prefix;
    return stats;
}

static void benchmark_alignment_aggregate(const Sample* samples,
                                          const PairWork* pair,
                                          const char* label,
                                          size_t requested_tables) {
    const Prefix* lhs_prefix = &g_prefixes[samples[pair->lhs].prefix_index];
    const Prefix* rhs_prefix = &g_prefixes[samples[pair->rhs].prefix_index];
    generate_tables(lhs_prefix, rhs_prefix);
    size_t table_count = (size_t)g_table_count;
    if (requested_tables && requested_tables < table_count) {
        table_count = requested_tables;
    }
    uint8_t lhs_rows[ROWS];
    uint8_t rhs_rows[ROWS];
    histogram_rows(lhs_prefix->histogram, lhs_rows);
    histogram_rows(rhs_prefix->histogram, rhs_rows);
    Distribution lhs = build_distribution(lhs_rows, 0);
    Distribution rhs = build_distribution(rhs_rows, 0);
    Distribution lhs_complement = build_distribution(lhs_rows, 1);
    Distribution rhs_complement = build_distribution(rhs_rows, 1);
    static uint64_t selected_values[MAX_TABLES];
    static uint64_t complement_values[MAX_TABLES];
    AlignmentSideStats selected =
        alignment_side_values(lhs_prefix, rhs_prefix, &lhs, &rhs, table_count,
                              selected_values);
    AlignmentSideStats complement = alignment_side_values(
        lhs_prefix, rhs_prefix, &lhs_complement, &rhs_complement, table_count,
        complement_values);
    U128 aggregate = 0;
    uint64_t represented_permutations = 0;
    for (size_t table = 0; table < table_count; table++) {
        aggregate += (U128)g_tables[table].multiplicity * selected_values[table] *
                     complement_values[table];
        represented_permutations += g_tables[table].multiplicity;
    }

    /* Independently rebuild and scan the first alignment as a hard gate on
       row-token permutation direction and the cached one-sided query. */
    uint8_t aligned_lhs_rows[ROWS];
    uint8_t aligned_rhs_rows[ROWS];
    table_rows(&g_tables[0], aligned_lhs_rows, aligned_rhs_rows);
    Distribution direct_lhs = build_distribution(aligned_lhs_rows, 0);
    Distribution direct_rhs = build_distribution(aligned_rhs_rows, 0);
    Distribution direct_lhs_complement = build_distribution(aligned_lhs_rows, 1);
    Distribution direct_rhs_complement = build_distribution(aligned_rhs_rows, 1);
    uint64_t direct_selected = disjoint_join(&direct_lhs, &direct_rhs);
    uint64_t direct_complement =
        disjoint_join(&direct_lhs_complement, &direct_rhs_complement);
    if (direct_selected != selected_values[0] ||
        direct_complement != complement_values[0]) {
        fprintf(stderr, "alignment/direct mismatch\n");
        exit(1);
    }
    printf("alignment_%s prefixes=%d,%d tables=%zu/%d permutations=%llu selected_support=%zu,%zu complement_support=%zu,%zu\n",
           label, samples[pair->lhs].prefix_index,
           samples[pair->rhs].prefix_index, table_count, g_table_count,
           (unsigned long long)represented_permutations, lhs.count, rhs.count,
           lhs_complement.count, rhs_complement.count);
    printf("  selected time=%.6fs queries=%zu query_hits=%llu query_misses=%llu subproblems=%zu subproblem_hits=%llu visits=%llu\n",
           selected.seconds, selected.cache_entries,
           (unsigned long long)selected.hits,
           (unsigned long long)selected.misses,
           selected.subproblems,
           (unsigned long long)selected.subproblem_hits,
           (unsigned long long)selected.visits);
    printf("  complement time=%.6fs queries=%zu query_hits=%llu query_misses=%llu subproblems=%zu subproblem_hits=%llu visits=%llu\n",
           complement.seconds, complement.cache_entries,
           (unsigned long long)complement.hits,
           (unsigned long long)complement.misses,
           complement.subproblems,
           (unsigned long long)complement.subproblem_hits,
           (unsigned long long)complement.visits);
    printf("  aggregate_time=%.6fs mean_per_table=%.9fs complete=%s value=",
           selected.seconds + complement.seconds,
           (selected.seconds + complement.seconds) / (double)table_count,
           table_count == (size_t)g_table_count ? "yes" : "no");
    print_u128(aggregate);
    printf("\n");
    free_distribution(&direct_lhs);
    free_distribution(&direct_rhs);
    free_distribution(&direct_lhs_complement);
    free_distribution(&direct_rhs_complement);
    free_distribution(&lhs);
    free_distribution(&rhs);
    free_distribution(&lhs_complement);
    free_distribution(&rhs_complement);
}

static int hybrid_method_index(const char* method) {
    if (!strcmp(method, "direct")) return 0;
    if (!strcmp(method, "apply")) return 1;
    return 2;
}

static void write_distribution(FILE* output, const Distribution* distribution) {
    for (size_t i = 0; i < distribution->count; i++) {
        if (fwrite(&distribution->entries[i].mask, sizeof(uint64_t), 1, output) != 1 ||
            fwrite(&distribution->entries[i].weight, sizeof(uint64_t), 1, output) != 1) {
            perror("write GPU dataset");
            exit(1);
        }
    }
}

static void write_dataset_kernel(FILE* output, const Distribution* lhs,
                                 const Distribution* rhs,
                                 const Distribution* lhs_complement,
                                 const Distribution* rhs_complement,
                                 uint64_t selected, uint64_t complement) {
    uint64_t fields[7] = {(uint64_t)lhs->count,
                          (uint64_t)rhs->count,
                          (uint64_t)lhs_complement->count,
                          (uint64_t)rhs_complement->count,
                          selected,
                          complement,
                          (uint64_t)lhs->count * rhs->count +
                              (uint64_t)lhs_complement->count * rhs_complement->count};
    if (fwrite(fields, sizeof(fields), 1, output) != 1) {
        perror("write GPU dataset record");
        exit(1);
    }
    write_distribution(output, lhs);
    write_distribution(output, rhs);
    write_distribution(output, lhs_complement);
    write_distribution(output, rhs_complement);
}

static double measure_hybrid_pair(const Sample* samples, const PairWork* pair,
                                  uint64_t salt, uint64_t method_counts[3],
                                  double* direct_seconds, FILE* dataset) {
    const Prefix* lhs_prefix = &g_prefixes[samples[pair->lhs].prefix_index];
    const Prefix* rhs_prefix = &g_prefixes[samples[pair->rhs].prefix_index];
    generate_tables(lhs_prefix, rhs_prefix);
    size_t table_index =
        (size_t)mix64(((uint64_t)samples[pair->lhs].prefix_index << 32) ^
                      (uint32_t)samples[pair->rhs].prefix_index ^ salt) %
        (size_t)g_table_count;
    uint8_t lhs_rows[ROWS];
    uint8_t rhs_rows[ROWS];
    table_rows(&g_tables[table_index], lhs_rows, rhs_rows);
    double build_start = seconds_now();
    Distribution lhs = build_distribution(lhs_rows, 0);
    Distribution rhs = build_distribution(rhs_rows, 0);
    Distribution lhs_complement = build_distribution(lhs_rows, 1);
    Distribution rhs_complement = build_distribution(rhs_rows, 1);
    double build_seconds = seconds_now() - build_start;
    double hybrid_start = seconds_now();
    OracleJoinResult selected = oracle_hybrid_join(&lhs, &rhs);
    OracleJoinResult complement =
        oracle_hybrid_join(&lhs_complement, &rhs_complement);
    double elapsed = build_seconds + seconds_now() - hybrid_start;
    double direct_start = seconds_now();
    uint64_t direct_selected = disjoint_join(&lhs, &rhs);
    uint64_t direct_complement = disjoint_join(&lhs_complement, &rhs_complement);
    *direct_seconds += build_seconds + seconds_now() - direct_start;
    if (selected.value != direct_selected ||
        complement.value != direct_complement) {
        fprintf(stderr, "stratified hybrid/direct mismatch\n");
        exit(1);
    }
    if (dataset) {
        write_dataset_kernel(dataset, &lhs, &rhs, &lhs_complement,
                             &rhs_complement, direct_selected,
                             direct_complement);
    }
    method_counts[hybrid_method_index(selected.method)]++;
    method_counts[hybrid_method_index(complement.method)]++;
    free_distribution(&lhs);
    free_distribution(&rhs);
    free_distribution(&lhs_complement);
    free_distribution(&rhs_complement);
    return elapsed;
}

static int size_compare(const void* lhs_ptr, const void* rhs_ptr) {
    size_t lhs = *(const size_t*)lhs_ptr;
    size_t rhs = *(const size_t*)rhs_ptr;
    return lhs < rhs ? -1 : lhs > rhs;
}

static int pair_compare(const void* lhs_ptr, const void* rhs_ptr) {
    const PairWork* lhs = lhs_ptr;
    const PairWork* rhs = rhs_ptr;
    return lhs->tests < rhs->tests ? -1 : lhs->tests > rhs->tests;
}

static void benchmark_pair(const Sample* samples, const PairWork* pair, const char* label,
                           uint64_t max_join_tests, uint64_t* measured_tests,
                           double* measured_seconds) {
    const Sample* lhs_sample = &samples[pair->lhs];
    const Sample* rhs_sample = &samples[pair->rhs];
    const Prefix* lhs_prefix = &g_prefixes[lhs_sample->prefix_index];
    const Prefix* rhs_prefix = &g_prefixes[rhs_sample->prefix_index];
    double table_start = seconds_now();
    generate_tables(lhs_prefix, rhs_prefix);
    double table_seconds = seconds_now() - table_start;
    size_t table_index = (size_t)mix64((uint64_t)lhs_sample->prefix_index << 32 |
                                      (uint32_t)rhs_sample->prefix_index) % (size_t)g_table_count;
    uint8_t lhs_rows[ROWS];
    uint8_t rhs_rows[ROWS];
    table_rows(&g_tables[table_index], lhs_rows, rhs_rows);
    double build_start = seconds_now();
    Distribution lhs = build_distribution(lhs_rows, 0);
    Distribution rhs = build_distribution(rhs_rows, 0);
    Distribution lhs_complement = build_distribution(lhs_rows, 1);
    Distribution rhs_complement = build_distribution(rhs_rows, 1);
    double build_seconds = seconds_now() - build_start;
    uint64_t actual_tests = (uint64_t)lhs.count * rhs.count +
                            (uint64_t)lhs_complement.count * rhs_complement.count;
    if (actual_tests != pair->tests) {
        fprintf(stderr, "support invariance failed\n");
        exit(1);
    }
    printf("join_%s prefixes=%d,%d predicted_tests=%llu tables=%d table_time=%.6fs build_time=%.6fs",
           label, lhs_sample->prefix_index, rhs_sample->prefix_index,
           (unsigned long long)pair->tests, g_table_count, table_seconds, build_seconds);
    printf("\n");
    U128 oracle_value = 0;
    int oracle_value_set = 0;
    double best_oracle_total = 0;
    double best_oracle_query = 0;
    OracleJoinResult selected_hybrid = oracle_hybrid_join(&lhs, &rhs);
    OracleJoinResult complement_hybrid =
        oracle_hybrid_join(&lhs_complement, &rhs_complement);
    U128 hybrid_value = (U128)selected_hybrid.value * complement_hybrid.value;
    oracle_value = hybrid_value;
    oracle_value_set = 1;
    double hybrid_build_seconds =
        selected_hybrid.build_seconds + complement_hybrid.build_seconds;
    double hybrid_query_seconds =
        selected_hybrid.query_seconds + complement_hybrid.query_seconds;
    printf("  hybrid methods=%s+%s nodes=%zu build=%.6fs query=%.6fs states=%llu value=",
           selected_hybrid.method, complement_hybrid.method,
           selected_hybrid.nodes + complement_hybrid.nodes,
           hybrid_build_seconds, hybrid_query_seconds,
           (unsigned long long)(selected_hybrid.visits + complement_hybrid.visits));
    print_u128(hybrid_value);
    printf("\n");
    /* Pair-major exposes the mutually exclusive colour bits for each row pair
       together.  Colour-major was exact, but produced far larger Apply tables
       (85m states and 3 GiB on a small p90 probe), so it is not a viable
       campaign ordering. */
    for (int ordering = 1; ordering < 2; ordering++) {
        OracleJoinResult selected_pair_oracle = oracle_pair_join(&lhs, &rhs, ordering);
        OracleJoinResult complement_pair_oracle =
            oracle_pair_join(&lhs_complement, &rhs_complement, ordering);
        U128 pair_value = (U128)selected_pair_oracle.value * complement_pair_oracle.value;
        if (!oracle_value_set) {
            oracle_value = pair_value;
            oracle_value_set = 1;
        } else if (pair_value != oracle_value) {
            fprintf(stderr, "pair-oracle order mismatch\n");
            exit(1);
        }
        double pair_build_seconds =
            selected_pair_oracle.build_seconds + complement_pair_oracle.build_seconds;
        double pair_query_seconds =
            selected_pair_oracle.query_seconds + complement_pair_oracle.query_seconds;
        double pair_total_seconds = pair_build_seconds + pair_query_seconds;
        if (!best_oracle_total || pair_total_seconds < best_oracle_total) {
            best_oracle_total = pair_total_seconds;
            best_oracle_query = pair_query_seconds;
        }
        printf("  pair_oracle order=%s nodes=%zu build=%.6fs apply=%.6fs memo_states=%llu value=",
               ordering ? "pair-major" : "colour-major",
               selected_pair_oracle.nodes + complement_pair_oracle.nodes,
               pair_build_seconds, pair_query_seconds,
               (unsigned long long)(selected_pair_oracle.visits +
                                    complement_pair_oracle.visits));
        print_u128(pair_value);
        printf("\n");
        for (int orientation = 0; orientation < 2; orientation++) {
            const Distribution* selected_queries = orientation ? &rhs : &lhs;
            const Distribution* selected_indexed = orientation ? &lhs : &rhs;
            const Distribution* complement_queries =
                orientation ? &rhs_complement : &lhs_complement;
            const Distribution* complement_indexed =
                orientation ? &lhs_complement : &rhs_complement;
            OracleJoinResult selected_oracle =
                oracle_join(selected_queries, selected_indexed, ordering);
            OracleJoinResult complement_oracle =
                oracle_join(complement_queries, complement_indexed, ordering);
            U128 value = (U128)selected_oracle.value * complement_oracle.value;
            if (!oracle_value_set) {
                oracle_value = value;
                oracle_value_set = 1;
            } else if (value != oracle_value) {
                fprintf(stderr, "oracle orientation/order mismatch\n");
                exit(1);
            }
            double oracle_build_seconds =
                selected_oracle.build_seconds + complement_oracle.build_seconds;
            double oracle_query_seconds =
                selected_oracle.query_seconds + complement_oracle.query_seconds;
            double oracle_total_seconds = oracle_build_seconds + oracle_query_seconds;
            if (!best_oracle_total || oracle_total_seconds < best_oracle_total) {
                best_oracle_total = oracle_total_seconds;
                best_oracle_query = oracle_query_seconds;
            }
            printf("  oracle order=%s orientation=%s nodes=%zu+%zu build=%.6fs query=%.6fs visits=%llu value=",
                   ordering ? "pair-major" : "colour-major",
                   orientation ? "rhs-to-lhs" : "lhs-to-rhs",
                   selected_oracle.nodes, complement_oracle.nodes,
                   oracle_build_seconds, oracle_query_seconds,
                   (unsigned long long)(selected_oracle.visits + complement_oracle.visits));
            print_u128(value);
            printf("\n");
        }
    }
    printf("  oracle_best total=%.6fs amortized_query=%.6fs\n",
           best_oracle_total, best_oracle_query);
    if (pair->tests <= max_join_tests) {
        double join_start = seconds_now();
        uint64_t left = disjoint_join(&lhs, &rhs);
        uint64_t right = disjoint_join(&lhs_complement, &rhs_complement);
        double join_seconds = seconds_now() - join_start;
        *measured_tests += pair->tests;
        *measured_seconds += join_seconds;
        printf("  direct join_time=%.6fs c2=%llu complement_c2=%llu value=", join_seconds,
               (unsigned long long)left, (unsigned long long)right);
        print_u128((U128)left * right);
        if ((U128)left * right != oracle_value) {
            fprintf(stderr, "direct/oracle mismatch\n");
            exit(1);
        }
    } else {
        printf("  direct join_time=SKIPPED(limit=%llu)",
               (unsigned long long)max_join_tests);
    }
    printf("\n");
    free_distribution(&lhs);
    free_distribution(&rhs);
    free_distribution(&lhs_complement);
    free_distribution(&rhs_complement);
}

typedef struct {
    uint8_t rows[ROWS];
    uint8_t row_degree[ROWS];
    uint8_t target_degree[ROWS];
    uint8_t order[ROWS];
    uint8_t used[ROWS];
    int columns;
    uint64_t best;
} MatrixCanonContext;

static void evaluate_matrix_row_order(MatrixCanonContext* context) {
    uint8_t column_vector[ROWS];
    uint8_t column_degree[ROWS];
    uint8_t column_order[ROWS];
    for (int column = 0; column < context->columns; column++) {
        uint8_t vector = 0;
        uint8_t degree = 0;
        for (int position = 0; position < ROWS; position++) {
            unsigned bit = (context->rows[context->order[position]] >> column) & 1U;
            vector = (uint8_t)((vector << 1) | bit);
            degree += (uint8_t)bit;
        }
        column_vector[column] = vector;
        column_degree[column] = degree;
        column_order[column] = (uint8_t)column;
    }
    for (int i = 1; i < context->columns; i++) {
        uint8_t column = column_order[i];
        int j = i;
        while (j > 0) {
            uint8_t previous = column_order[j - 1];
            if (column_degree[previous] < column_degree[column] ||
                (column_degree[previous] == column_degree[column] &&
                 column_vector[previous] <= column_vector[column])) break;
            column_order[j] = previous;
            j--;
        }
        column_order[j] = column;
    }
    uint64_t key = 0;
    for (int position = 0; position < ROWS; position++) {
        uint8_t pattern = 0;
        uint8_t original_row = context->order[position];
        for (int column_position = 0; column_position < context->columns;
             column_position++) {
            uint8_t original_column = column_order[column_position];
            if ((context->rows[original_row] >> original_column) & 1U) {
                pattern |= (uint8_t)(1U << column_position);
            }
        }
        key = (key << 8) | pattern;
    }
    if (key < context->best) context->best = key;
}

static void canonical_matrix_rows_rec(MatrixCanonContext* context, int depth) {
    if (depth == ROWS) {
        evaluate_matrix_row_order(context);
        return;
    }
    uint8_t degree = context->target_degree[depth];
    uint64_t seen[4] = {0};
    for (int row = 0; row < ROWS; row++) {
        if (context->used[row] || context->row_degree[row] != degree) continue;
        uint8_t pattern = context->rows[row];
        uint64_t bit = UINT64_C(1) << (pattern & 63U);
        if (seen[pattern >> 6] & bit) continue;
        seen[pattern >> 6] |= bit;
        context->used[row] = 1;
        context->order[depth] = (uint8_t)row;
        canonical_matrix_rows_rec(context, depth + 1);
        context->used[row] = 0;
    }
}

static uint64_t canonical_matrix_key_columns(uint64_t key, int columns) {
    MatrixCanonContext context = {.columns = columns, .best = UINT64_MAX};
    for (int row = ROWS - 1; row >= 0; row--) {
        context.rows[row] = (uint8_t)(key & UINT64_C(0xff));
        key >>= 8;
    }
    for (int row = 0; row < ROWS; row++) {
        context.row_degree[row] = (uint8_t)__builtin_popcount(context.rows[row]);
        context.target_degree[row] = context.row_degree[row];
    }
    for (int i = 1; i < ROWS; i++) {
        uint8_t degree = context.target_degree[i];
        int j = i;
        while (j && context.target_degree[j - 1] > degree) {
            context.target_degree[j] = context.target_degree[j - 1];
            j--;
        }
        context.target_degree[j] = degree;
    }
    canonical_matrix_rows_rec(&context, 0);
    return context.best;
}

static uint64_t canonical_matrix_key(uint64_t key) {
    return canonical_matrix_key_columns(key, ROWS);
}

static int distinct_parent_count(uint64_t child_key) {
    uint8_t rows[ROWS];
    for (int row = ROWS - 1; row >= 0; row--) {
        rows[row] = (uint8_t)(child_key & UINT64_C(0xff));
        child_key >>= 8;
    }
    uint64_t parents[ROWS];
    int parent_count = 0;
    for (int deleted = 0; deleted < ROWS; deleted++) {
        uint64_t parent_key = 0;
        uint8_t low_mask = (uint8_t)((1U << deleted) - 1U);
        for (int row = 0; row < ROWS; row++) {
            uint8_t parent_row = (uint8_t)((rows[row] & low_mask) |
                                           ((rows[row] >> 1) & ~low_mask));
            parent_key = (parent_key << 8) | parent_row;
        }
        parent_key = canonical_matrix_key_columns(parent_key, 7);
        int duplicate = 0;
        for (int i = 0; i < parent_count; i++) duplicate |= parents[i] == parent_key;
        if (!duplicate) parents[parent_count++] = parent_key;
    }
    return parent_count;
}

static uint64_t histogram_code(const uint8_t histogram[HALF_PATTERNS]) {
    uint64_t code = 0;
    for (int pattern = 0; pattern < HALF_PATTERNS; pattern++) {
        code = (code << 4) | histogram[pattern];
    }
    return code;
}

static uint64_t canonical_histogram_code(const uint8_t histogram[HALF_PATTERNS]) {
    uint64_t best = UINT64_MAX;
    for (int permutation = 0; permutation < g_permutation_count; permutation++) {
        uint8_t transformed[HALF_PATTERNS];
        for (int pattern = 0; pattern < HALF_PATTERNS; pattern++) {
            transformed[pattern] = histogram[g_preimages[permutation][pattern]];
        }
        uint64_t code = histogram_code(transformed);
        if (code < best) best = code;
    }
    return best;
}

static uint32_t prefix_type(const PrefixIndex* index, uint32_t labelled_prefix) {
    uint8_t histogram[HALF_PATTERNS] = {0};
    for (int row = 0; row < ROWS; row++) {
        histogram[labelled_prefix & 15U]++;
        labelled_prefix >>= 4;
    }
    return prefix_index_get(index, canonical_histogram_code(histogram));
}

static int uint64_compare(const void* lhs_pointer, const void* rhs_pointer) {
    uint64_t lhs = *(const uint64_t*)lhs_pointer;
    uint64_t rhs = *(const uint64_t*)rhs_pointer;
    return lhs < rhs ? -1 : lhs > rhs;
}

typedef struct {
    uint64_t work;
    double inverse_probability;
    uint8_t parent_count;
} DegreeWork;

static int degree_work_compare(const void* lhs_pointer, const void* rhs_pointer) {
    const DegreeWork* lhs = lhs_pointer;
    const DegreeWork* rhs = rhs_pointer;
    return lhs->work < rhs->work ? -1 : lhs->work > rhs->work;
}

static int run_orbit_census(int shards, int input_count, char** input_paths) {
    if (shards < 1 || input_count < 1) return 2;
    uint32_t support[PREFIX_ORBITS][2];
    PrefixIndex prefix_index;
    prefix_index_init(&prefix_index);
    U128 canonical_entries = 0;
    U128 viable_labelled_prefixes = 0;
    U128 viable_labelled_entries = 0;
    uint64_t dead_prefixes = 0;
    double support_start = seconds_now();
    for (int prefix = 0; prefix < g_prefix_count; prefix++) {
        uint8_t rows[ROWS];
        histogram_rows(g_prefixes[prefix].histogram, rows);
        Distribution selected = build_distribution(rows, 0);
        Distribution complement = build_distribution(rows, 1);
        support[prefix][0] = (uint32_t)selected.count;
        support[prefix][1] = (uint32_t)complement.count;
        canonical_entries += selected.count;
        if (!selected.count || !complement.count) dead_prefixes++;
        if (selected.count && complement.count) {
            uint64_t row_stabilizer = 1;
            for (int pattern = 0; pattern < HALF_PATTERNS; pattern++) {
                row_stabilizer *= g_factorial[g_prefixes[prefix].histogram[pattern]];
            }
            uint64_t column_stabilizer = 0;
            for (int permutation = 0; permutation < g_permutation_count; permutation++) {
                int equal = 1;
                for (int pattern = 0; pattern < HALF_PATTERNS; pattern++) {
                    if (g_prefixes[prefix].histogram[g_preimages[permutation][pattern]] !=
                        g_prefixes[prefix].histogram[pattern]) {
                        equal = 0;
                        break;
                    }
                }
                column_stabilizer += equal;
            }
            uint64_t orbit_size = g_factorial[ROWS] * g_factorial[HALF_COLUMNS] /
                                  (row_stabilizer * column_stabilizer);
            viable_labelled_prefixes += orbit_size;
            viable_labelled_entries +=
                (U128)orbit_size * (selected.count + complement.count);
        }
        prefix_index_add(&prefix_index, histogram_code(g_prefixes[prefix].histogram),
                         (uint32_t)prefix);
        free_distribution(&selected);
        free_distribution(&complement);
        if ((prefix + 1) % 1024 == 0 || prefix + 1 == g_prefix_count) {
            printf("census_support_progress prefixes=%d/%d elapsed=%.3f\n",
                   prefix + 1, g_prefix_count, seconds_now() - support_start);
        }
    }
    printf("CENSUS_CANONICAL prefixes=%d dead=%llu selected_entries=",
           g_prefix_count, (unsigned long long)dead_prefixes);
    print_u128(canonical_entries);
    printf(" soa_bytes=");
    print_u128(canonical_entries * 12U);
    printf(" viable_labelled_prefixes=");
    print_u128(viable_labelled_prefixes);
    printf(" viable_labelled_entries=");
    print_u128(viable_labelled_entries);
    printf(" viable_labelled_soa_bytes=");
    print_u128(viable_labelled_entries * 12U);
    printf(" build_seconds=%.3f\n", seconds_now() - support_start);

    KeySet orbit_keys, all_left, shard_left, shard_right;
    key_set_init(&orbit_keys, 1024);
    key_set_init(&all_left, 1024);
    key_set_init(&shard_left, 1024);
    key_set_init(&shard_right, 1024);
    U128 all_left_entries = 0;
    U128 shard_left_entries = 0;
    U128 shard_right_entries = 0;
    U128 comparison_sum = 0;
    uint64_t sample_orbits = 0;
    uint64_t evaluated = 0;
    uint64_t midpoint = 0;
    uint64_t self_complementary = 0;
    uint64_t dead_kernels = 0;
    uint64_t shard_kernels = 0;
    size_t work_count = 0;
    size_t work_capacity = 1024;
    uint64_t* work = xcalloc(work_capacity, sizeof(work[0]));
    size_t degree_count = 0;
    size_t degree_capacity = 1024;
    DegreeWork* degree_work = xcalloc(degree_capacity, sizeof(degree_work[0]));
    uint64_t degree_histogram[9] = {0};
    double census_start = seconds_now();
    for (int input_index = 0; input_index < input_count; input_index++) {
        FILE* input = fopen(input_paths[input_index], "rb");
        if (!input) exit(1);
        char magic[8];
        uint32_t columns = 0;
        uint64_t count = 0;
        if (fread(magic, sizeof(magic), 1, input) != 1 ||
            memcmp(magic, "R8ORB01", 7) != 0 ||
            fread(&columns, sizeof(columns), 1, input) != 1 ||
            fread(&count, sizeof(count), 1, input) != 1 || columns != 8) exit(1);
        for (uint64_t record_index = 0; record_index < count; record_index++) {
            uint64_t key, weight;
            if (fread(&key, sizeof(key), 1, input) != 1 ||
                fread(&weight, sizeof(weight), 1, input) != 1) exit(1);
            (void)weight;
            if (!key_set_add(&orbit_keys, key)) continue;
            sample_orbits++;
            int cells = __builtin_popcountll(key);
            if (cells > 32) continue;
            if (cells == 32) {
                midpoint++;
                uint64_t complement_key = canonical_matrix_key(~key);
                if (key > complement_key) continue;
                if (key == complement_key) self_complementary++;
            }
            uint8_t rows[ROWS];
            uint64_t unpack_key = key;
            for (int row = ROWS - 1; row >= 0; row--) {
                rows[row] = (uint8_t)(unpack_key & UINT64_C(0xff));
                unpack_key >>= 8;
            }
            uint32_t left = 0;
            uint32_t right = 0;
            for (int row = 0; row < ROWS; row++) {
                left = (left << 4) | (rows[row] & 15U);
                right = (right << 4) | (rows[row] >> 4);
            }
            uint32_t left_type = prefix_type(&prefix_index, left);
            uint32_t right_type = prefix_type(&prefix_index, right);
            uint64_t comparisons =
                (uint64_t)support[left_type][0] * support[right_type][0] +
                (uint64_t)support[left_type][1] * support[right_type][1];
            if (!comparisons) dead_kernels++;
            comparison_sum += comparisons;
            if (work_count == work_capacity) {
                work_capacity <<= 1;
                work = realloc(work, work_capacity * sizeof(work[0]));
                if (!work) exit(1);
            }
            work[work_count++] = comparisons;
            if (mix64(key ^ UINT64_C(0x8c3c010cb4754c91)) % 32U == 0) {
                int parents = distinct_parent_count(key);
                if (degree_count == degree_capacity) {
                    degree_capacity <<= 1;
                    degree_work = realloc(degree_work,
                                          degree_capacity * sizeof(degree_work[0]));
                    if (!degree_work) exit(1);
                }
                double selected_parent_fraction = (double)input_count / 262144.0;
                double missed = 1.0;
                for (int i = 0; i < parents; i++) missed *= 1.0 - selected_parent_fraction;
                degree_work[degree_count++] =
                    (DegreeWork){.work = comparisons,
                                 .inverse_probability = 1.0 / (1.0 - missed),
                                 .parent_count = (uint8_t)parents};
                degree_histogram[parents]++;
            }
            evaluated++;
            if (key_set_add(&all_left, left)) {
                all_left_entries += support[left_type][0] + support[left_type][1];
            }
            if (mix64(left) % (uint64_t)shards == 0) {
                shard_kernels++;
                if (key_set_add(&shard_left, left)) {
                    shard_left_entries += support[left_type][0] + support[left_type][1];
                }
                if (key_set_add(&shard_right, right)) {
                    shard_right_entries += support[right_type][0] + support[right_type][1];
                }
            }
        }
        if (fgetc(input) != EOF || fclose(input) != 0) exit(1);
        printf("CENSUS_FILE index=%d records=%llu cumulative_orbits=%llu "
               "evaluated=%llu unique_left=%zu shard0_left=%zu shard0_right=%zu\n",
               input_index, (unsigned long long)count,
               (unsigned long long)sample_orbits, (unsigned long long)evaluated,
               all_left.count, shard_left.count, shard_right.count);
    }
    if (!work_count) exit(1);
    qsort(work, work_count, sizeof(work[0]), uint64_compare);
    double mean = (double)comparison_sum / (double)evaluated;
    double projected_hours = mean * 7343033248.0 / 3.4e12 / 3600.0;
    printf("CENSUS_RESULT sample_orbits=%llu evaluated=%llu midpoint_seen=%llu "
           "self_complementary=%llu dead_kernels=%llu comparisons=",
           (unsigned long long)sample_orbits, (unsigned long long)evaluated,
           (unsigned long long)midpoint, (unsigned long long)self_complementary,
           (unsigned long long)dead_kernels);
    print_u128(comparison_sum);
    printf(" mean=%.3f median=%llu p90=%llu p99=%llu max=%llu "
           "projected_l40s_hours_at_3.4T=%.3f seconds=%.3f\n",
           mean, (unsigned long long)work[work_count / 2U],
           (unsigned long long)work[work_count * 9U / 10U],
           (unsigned long long)work[work_count * 99U / 100U],
           (unsigned long long)work[work_count - 1U], projected_hours,
           seconds_now() - census_start);
    qsort(degree_work, degree_count, sizeof(degree_work[0]), degree_work_compare);
    double degree_weight = 0;
    double degree_weighted_work = 0;
    for (size_t i = 0; i < degree_count; i++) {
        degree_weight += degree_work[i].inverse_probability;
        degree_weighted_work +=
            degree_work[i].inverse_probability * (double)degree_work[i].work;
    }
    uint64_t degree_quantiles[3] = {0};
    const double fractions[3] = {0.5, 0.9, 0.99};
    double cumulative_degree_weight = 0;
    int next_quantile = 0;
    for (size_t i = 0; i < degree_count && next_quantile < 3; i++) {
        cumulative_degree_weight += degree_work[i].inverse_probability;
        while (next_quantile < 3 &&
               cumulative_degree_weight >= degree_weight * fractions[next_quantile]) {
            degree_quantiles[next_quantile++] = degree_work[i].work;
        }
    }
    double corrected_mean = degree_weighted_work / degree_weight;
    printf("CENSUS_DEGREE sampled=%zu histogram_1_to_8=%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu "
           "corrected_mean=%.3f corrected_median=%llu corrected_p90=%llu "
           "corrected_p99=%llu projected_l40s_hours_at_3.4T=%.3f\n",
           degree_count, (unsigned long long)degree_histogram[1],
           (unsigned long long)degree_histogram[2],
           (unsigned long long)degree_histogram[3],
           (unsigned long long)degree_histogram[4],
           (unsigned long long)degree_histogram[5],
           (unsigned long long)degree_histogram[6],
           (unsigned long long)degree_histogram[7],
           (unsigned long long)degree_histogram[8], corrected_mean,
           (unsigned long long)degree_quantiles[0],
           (unsigned long long)degree_quantiles[1],
           (unsigned long long)degree_quantiles[2],
           corrected_mean * 7343033248.0 / 3.4e12 / 3600.0);
    printf("CENSUS_PREFIX shards=%d all_left=%zu all_left_entries=", shards,
           all_left.count);
    print_u128(all_left_entries);
    printf(" all_left_soa_bytes=");
    print_u128(all_left_entries * 12U);
    printf(" shard0_left=%zu shard0_left_entries=", shard_left.count);
    print_u128(shard_left_entries);
    printf(" shard0_left_soa_bytes=");
    print_u128(shard_left_entries * 12U);
    printf(" shard0_right=%zu shard0_right_entries=", shard_right.count);
    print_u128(shard_right_entries);
    printf(" shard0_right_soa_bytes=");
    print_u128(shard_right_entries * 12U);
    uint64_t shard_singletons = 0;
    uint64_t shard_doubletons = 0;
    for (size_t slot = 0; slot < shard_left.capacity; slot++) {
        if (!shard_left.used[slot]) continue;
        shard_singletons += shard_left.frequencies[slot] == 1;
        shard_doubletons += shard_left.frequencies[slot] == 2;
    }
    double chao_lower = shard_left.count;
    if (shard_doubletons) {
        chao_lower += (double)shard_singletons * shard_singletons /
                      (2.0 * shard_doubletons);
    }
    double coverage = shard_kernels
                          ? 1.0 - (double)shard_singletons / shard_kernels
                          : 0;
    printf(" shard_kernels=%llu singletons=%llu doubletons=%llu coverage=%.6f "
           "chao_lower=%.0f\n",
           (unsigned long long)shard_kernels,
           (unsigned long long)shard_singletons,
           (unsigned long long)shard_doubletons, coverage, chao_lower);
    free(degree_work);
    free(work);
    key_set_free(&shard_right);
    key_set_free(&shard_left);
    key_set_free(&all_left);
    key_set_free(&orbit_keys);
    free(prefix_index.used);
    free(prefix_index.values);
    free(prefix_index.keys);
    return 0;
}

int main(int argc, char** argv) {
    int census_mode = argc > 1 && strcmp(argv[1], "--orbit-census") == 0;
    size_t requested_samples = !census_mode && argc > 1 ? strtoull(argv[1], NULL, 10) : 256;
    uint64_t max_join_tests = !census_mode && argc > 2
                                  ? strtoull(argv[2], NULL, 10)
                                  : UINT64_C(1000000000);
    size_t oracle_samples = !census_mode && argc > 3 ? strtoull(argv[3], NULL, 10) : 0;
    int alignment_quantile = !census_mode && argc > 4
                                 ? (int)strtol(argv[4], NULL, 10)
                                 : -1;
    size_t alignment_table_limit =
        !census_mode && argc > 5 ? strtoull(argv[5], NULL, 10) : 0;
    const char* dataset_path = !census_mode && argc > 6 ? argv[6] : NULL;
    if (!requested_samples) {
        fprintf(stderr,
                "Usage: %s PREFIX_SAMPLES MAX_EXACT_JOIN_TESTS [HYBRID_SAMPLES [ALIGNMENT_QUANTILE [ALIGNMENT_TABLE_LIMIT [GPU_DATASET]]]]\n",
                argv[0]);
        return 2;
    }
    if (alignment_quantile < -1 || alignment_quantile > 4) return 2;
    setvbuf(stdout, NULL, _IOLBF, 0);
    g_factorial[0] = 1;
    for (int i = 1; i <= ROWS; i++) g_factorial[i] = g_factorial[i - 1] * (uint64_t)i;
    int pair_index = 0;
    for (int u = 0; u < ROWS; u++) {
        for (int v = u + 1; v < ROWS; v++) g_pair_index[u][v] = pair_index++;
    }
    generate_permutations(0, 0);
    enumerate_histograms(0, ROWS);
    if (g_prefix_count != PREFIX_ORBITS) {
        fprintf(stderr, "prefix orbit count failed: %d\n", g_prefix_count);
        return 1;
    }
    if (census_mode) {
        if (argc < 4) {
            fprintf(stderr, "Usage: %s --orbit-census SHARDS SAMPLE.orbits...\n",
                    argv[0]);
            return 2;
        }
        return run_orbit_census(atoi(argv[2]), argc - 3, argv + 3);
    }
    if (requested_samples > (size_t)g_prefix_count) requested_samples = (size_t)g_prefix_count;
    int* indices = malloc((size_t)g_prefix_count * sizeof(indices[0]));
    Sample* samples = calloc(requested_samples, sizeof(samples[0]));
    size_t* supports[2] = {calloc(requested_samples, sizeof(size_t)),
                           calloc(requested_samples, sizeof(size_t))};
    if (!indices || !samples || !supports[0] || !supports[1]) return 1;
    for (int i = 0; i < g_prefix_count; i++) indices[i] = i;
    uint64_t rng = UINT64_C(0x6a09e667f3bcc909);
    for (int i = g_prefix_count - 1; i > 0; i--) {
        rng = mix64(rng);
        int other = (int)(rng % (uint64_t)(i + 1));
        int swap = indices[i];
        indices[i] = indices[other];
        indices[other] = swap;
    }
    size_t viable = 0;
    double support_start = seconds_now();
    for (size_t i = 0; i < requested_samples; i++) {
        int prefix_index_value = indices[i];
        uint8_t rows[ROWS];
        histogram_rows(g_prefixes[prefix_index_value].histogram, rows);
        double build_start = seconds_now();
        Distribution selected = build_distribution(rows, 0);
        Distribution complement = build_distribution(rows, 1);
        double build_seconds = seconds_now() - build_start;
        if (selected.count && complement.count) {
            samples[viable] = (Sample){.prefix_index = prefix_index_value,
                                       .support = {selected.count, complement.count},
                                       .build_seconds = build_seconds,
                                       .shape = histogram_shape(
                                           g_prefixes[prefix_index_value].histogram,
                                           prefix_index_value)};
            supports[0][viable] = selected.count;
            supports[1][viable] = complement.count;
            viable++;
        }
        free_distribution(&selected);
        free_distribution(&complement);
        if ((i + 1U) % 32U == 0 || i + 1U == requested_samples) {
            printf("support_progress sampled=%zu viable=%zu elapsed=%.2fs\n",
                   i + 1U, viable, seconds_now() - support_start);
        }
    }
    double support_seconds = seconds_now() - support_start;
    if (!viable) return 1;
    qsort(supports[0], viable, sizeof(supports[0][0]), size_compare);
    qsort(supports[1], viable, sizeof(supports[1][0]), size_compare);
    size_t pair_count = viable * (viable + 1U) / 2U;
    PairWork* pair_work = malloc(pair_count * sizeof(pair_work[0]));
    if (!pair_work) return 1;
    size_t cursor = 0;
    U128 ordered_test_sum = 0;
    U128 weighted_test_sum = 0;
    uint64_t total_pair_weight = 0;
    for (size_t i = 0; i < viable; i++) {
        for (size_t j = 0; j < viable; j++) {
            ordered_test_sum += (U128)samples[i].support[0] * samples[j].support[0] +
                                (U128)samples[i].support[1] * samples[j].support[1];
        }
        for (size_t j = i; j < viable; j++) {
            uint32_t double_cosets = double_coset_count(samples[i].shape, samples[j].shape);
            pair_work[cursor++] =
                (PairWork){.lhs = (uint16_t)i,
                           .rhs = (uint16_t)j,
                           .tests = (uint64_t)samples[i].support[0] * samples[j].support[0] +
                                    (uint64_t)samples[i].support[1] * samples[j].support[1],
                           .double_cosets = double_cosets};
            uint64_t weight = (uint64_t)double_cosets * (i == j ? 1U : 2U);
            weighted_test_sum += (U128)pair_work[cursor - 1U].tests * weight;
            total_pair_weight += weight;
        }
    }
    qsort(pair_work, pair_count, sizeof(pair_work[0]), pair_compare);
    double mean_tests = (double)ordered_test_sum / (double)((U128)viable * viable);
    double weighted_mean_tests = (double)weighted_test_sum / (double)total_pair_weight;
    printf("prefix_orbits=%d sampled=%zu viable_sampled=%zu support_time=%.3fs transitions=%llu\n",
           g_prefix_count, requested_samples, viable, support_seconds,
           (unsigned long long)g_distribution_transitions);
    for (int complement = 0; complement < 2; complement++) {
        printf("support_%s min=%zu median=%zu p90=%zu p99=%zu max=%zu\n",
               complement ? "complement" : "selected", supports[complement][0],
               supports[complement][viable / 2U], supports[complement][viable * 9U / 10U],
               supports[complement][viable * 99U / 100U], supports[complement][viable - 1U]);
    }
    printf("pair_tests pairs=%zu min=%llu median=%llu p90=%llu p99=%llu max=%llu mean=%.0f\n",
           pair_count, (unsigned long long)pair_work[0].tests,
           (unsigned long long)pair_work[pair_count / 2U].tests,
           (unsigned long long)pair_work[pair_count * 9U / 10U].tests,
           (unsigned long long)pair_work[pair_count * 99U / 100U].tests,
           (unsigned long long)pair_work[pair_count - 1U].tests, mean_tests);
    size_t quantile_indices[5] = {0};
    const uint64_t quantile_numerators[5] = {0, 50, 90, 99, 100};
    uint64_t cumulative_weight = 0;
    size_t quantile = 0;
    for (size_t i = 0; i < pair_count && quantile < 5; i++) {
        uint64_t weight = (uint64_t)pair_work[i].double_cosets *
                          (pair_work[i].lhs == pair_work[i].rhs ? 1U : 2U);
        cumulative_weight += weight;
        while (quantile < 5 &&
               (quantile_numerators[quantile] == 0 ||
                (U128)cumulative_weight * 100U >=
                    (U128)total_pair_weight * quantile_numerators[quantile])) {
            quantile_indices[quantile++] = i;
        }
    }
    quantile_indices[4] = pair_count - 1U;
    printf("double_coset_weight=%llu shapes=%d weighted_pair_tests median=%llu p90=%llu p99=%llu max=%llu mean=%.0f\n",
           (unsigned long long)total_pair_weight, g_shape_count,
           (unsigned long long)pair_work[quantile_indices[1]].tests,
           (unsigned long long)pair_work[quantile_indices[2]].tests,
           (unsigned long long)pair_work[quantile_indices[3]].tests,
           (unsigned long long)pair_work[quantile_indices[4]].tests, weighted_mean_tests);
    const char* labels[] = {"min", "median", "p90", "p99", "max"};
    uint64_t measured_tests = 0;
    double measured_seconds = 0;
    if (max_join_tests) {
        for (size_t i = 0; i < sizeof(quantile_indices) / sizeof(quantile_indices[0]); i++) {
            if (i && quantile_indices[i] == quantile_indices[i - 1U]) continue;
            benchmark_pair(samples, &pair_work[quantile_indices[i]], labels[i],
                           max_join_tests, &measured_tests, &measured_seconds);
        }
    }
    if (measured_tests && measured_seconds > 0) {
        double throughput = (double)measured_tests / measured_seconds;
        double mean_build = 2.0 * support_seconds / (double)requested_samples;
        double projected_kernel_seconds = mean_build + weighted_mean_tests / throughput;
        double complement_orbits = 14685630688.0 / 2.0;
        double projected_core_hours = projected_kernel_seconds * complement_orbits / 3600.0;
        printf("measured_join_tests=%llu measured_join_time=%.6fs throughput=%.0f_tests_per_s\n",
               (unsigned long long)measured_tests, measured_seconds, throughput);
        printf("sampled_mean_build_per_kernel=%.6fs projected_mean_kernel=%.6fs projected_complement_paired_core_hours=%.0f budget_200k=%s\n",
               mean_build, projected_kernel_seconds, projected_core_hours,
               projected_kernel_seconds <= 0.09813 ? "PASS" : "FAIL");
    }
    if (oracle_samples) {
        double hybrid_sample_seconds = 0;
        double direct_sample_seconds = 0;
        FILE* dataset = NULL;
        if (dataset_path) {
            dataset = fopen(dataset_path, "wb");
            if (!dataset) {
                perror(dataset_path);
                return 1;
            }
            const char magic[8] = {'T', '4', 'G', 'P', 'U', '0', '1', '\0'};
            uint64_t kernel_count = oracle_samples;
            if (fwrite(magic, sizeof(magic), 1, dataset) != 1 ||
                fwrite(&kernel_count, sizeof(kernel_count), 1, dataset) != 1) {
                perror("write GPU dataset header");
                return 1;
            }
        }
        uint64_t hybrid_method_counts[3] = {0};
        size_t pair_cursor = 0;
        uint64_t cumulative =
            (uint64_t)pair_work[0].double_cosets *
            (pair_work[0].lhs == pair_work[0].rhs ? 1U : 2U);
        uint64_t sample_rng = UINT64_C(0xbb67ae8584caa73b);
        for (size_t sample = 0; sample < oracle_samples; sample++) {
            uint64_t bucket_begin =
                (uint64_t)((U128)total_pair_weight * sample / oracle_samples);
            uint64_t bucket_end =
                (uint64_t)((U128)total_pair_weight * (sample + 1U) / oracle_samples);
            sample_rng = mix64(sample_rng);
            uint64_t target = bucket_begin + sample_rng % (bucket_end - bucket_begin);
            while (cumulative <= target && pair_cursor + 1U < pair_count) {
                pair_cursor++;
                cumulative +=
                    (uint64_t)pair_work[pair_cursor].double_cosets *
                    (pair_work[pair_cursor].lhs == pair_work[pair_cursor].rhs ? 1U : 2U);
            }
            hybrid_sample_seconds +=
                measure_hybrid_pair(samples, &pair_work[pair_cursor], sample_rng,
                                    hybrid_method_counts, &direct_sample_seconds,
                                    dataset);
            if ((sample + 1U) % 8U == 0 || sample + 1U == oracle_samples) {
                printf("hybrid_progress sampled=%zu elapsed=%.3fs\n", sample + 1U,
                       hybrid_sample_seconds);
            }
        }
        double hybrid_mean_seconds = hybrid_sample_seconds / (double)oracle_samples;
        double direct_mean_seconds = direct_sample_seconds / (double)oracle_samples;
        double complement_paired_orbits = 7343033248.0;
        double hybrid_core_hours =
            hybrid_mean_seconds * complement_paired_orbits / 3600.0;
        printf("hybrid_stratified_samples=%zu mean_kernel=%.6fs direct_same_sample=%.6fs speedup=%.2fx projected_complement_paired_core_hours=%.0f budget_200k=%s methods_direct_apply_one_sided=%llu,%llu,%llu\n",
               oracle_samples, hybrid_mean_seconds, direct_mean_seconds,
               direct_mean_seconds / hybrid_mean_seconds, hybrid_core_hours,
               hybrid_mean_seconds <= 200000.0 * 3600.0 / complement_paired_orbits
                   ? "PASS"
                   : "FAIL",
               (unsigned long long)hybrid_method_counts[0],
               (unsigned long long)hybrid_method_counts[1],
               (unsigned long long)hybrid_method_counts[2]);
        if (dataset && fclose(dataset) != 0) {
            perror("close GPU dataset");
            return 1;
        }
    }
    if (alignment_quantile >= 0) {
        benchmark_alignment_aggregate(
            samples, &pair_work[quantile_indices[alignment_quantile]],
            labels[alignment_quantile], alignment_table_limit);
    }
    free(pair_work);
    free(supports[0]);
    free(supports[1]);
    free(samples);
    free(indices);
    return 0;
}
