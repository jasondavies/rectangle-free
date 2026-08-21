/* Exact residual-state/ZDD probe for labelled C4-free binary matrices. */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned __int128 U128;

typedef struct {
    uint32_t level;
    uint32_t low;
    uint32_t high;
} Node;

typedef struct {
    uint8_t used;
    uint8_t depth;
    uint16_t rows[8];
    uint32_t root;
    U128 count;
} MemoEntry;

static Node* nodes;
static size_t node_count;
static size_t node_capacity;
static uint32_t* node_slots;
static size_t node_slot_capacity;
static MemoEntry* memo;
static size_t memo_count;
static size_t memo_capacity;
static uint64_t memo_by_depth[9];
static int g_rows;
static int g_cols;

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

static uint64_t node_hash(uint32_t level, uint32_t low, uint32_t high) {
    return mix64(((uint64_t)level << 48) ^ ((uint64_t)low << 24) ^ high);
}

static void rebuild_node_slots(void) {
    free(node_slots);
    node_slot_capacity = 1;
    while (node_slot_capacity < node_count * 2U + 16U) node_slot_capacity <<= 1;
    node_slots = xcalloc(node_slot_capacity, sizeof(node_slots[0]));
    for (uint32_t id = 2; id < node_count; id++) {
        Node node = nodes[id];
        size_t slot = (size_t)node_hash(node.level, node.low, node.high) &
                      (node_slot_capacity - 1U);
        while (node_slots[slot]) slot = (slot + 1U) & (node_slot_capacity - 1U);
        node_slots[slot] = id;
    }
}

static uint32_t make_node(uint32_t level, uint32_t low, uint32_t high) {
    if (high == 0) return low;
    if ((node_count + 1U) * 10U >= node_slot_capacity * 7U) rebuild_node_slots();
    size_t slot = (size_t)node_hash(level, low, high) & (node_slot_capacity - 1U);
    while (node_slots[slot]) {
        uint32_t id = node_slots[slot];
        if (nodes[id].level == level && nodes[id].low == low && nodes[id].high == high) return id;
        slot = (slot + 1U) & (node_slot_capacity - 1U);
    }
    if (node_count == node_capacity) {
        node_capacity <<= 1;
        nodes = realloc(nodes, node_capacity * sizeof(nodes[0]));
        if (!nodes) exit(1);
    }
    uint32_t id = (uint32_t)node_count++;
    nodes[id] = (Node){.level = level, .low = low, .high = high};
    node_slots[slot] = id;
    return id;
}

static uint64_t state_hash(int depth, const uint16_t* rows) {
    uint64_t h = mix64((uint64_t)depth);
    for (int i = 0; i < depth; i++) h = mix64(h ^ rows[i]);
    return h;
}

static int state_equal(const MemoEntry* entry, int depth, const uint16_t* rows) {
    return entry->depth == depth &&
           memcmp(entry->rows, rows, (size_t)depth * sizeof(rows[0])) == 0;
}

static void memo_insert_raw(MemoEntry entry) {
    size_t slot = (size_t)state_hash(entry.depth, entry.rows) & (memo_capacity - 1U);
    while (memo[slot].used) slot = (slot + 1U) & (memo_capacity - 1U);
    memo[slot] = entry;
    memo[slot].used = 1;
    memo_count++;
}

static void memo_rehash(void) {
    MemoEntry* old = memo;
    size_t old_capacity = memo_capacity;
    memo_capacity <<= 1;
    memo = xcalloc(memo_capacity, sizeof(memo[0]));
    memo_count = 0;
    for (size_t i = 0; i < old_capacity; i++) if (old[i].used) memo_insert_raw(old[i]);
    free(old);
}

static uint32_t build_selector(uint32_t* children, int bit, int begin, int end, int row) {
    if (bit == g_cols) return children[begin];
    int middle = (begin + end) / 2;
    uint32_t low = build_selector(children, bit + 1, begin, middle, row);
    uint32_t high = build_selector(children, bit + 1, middle, end, row);
    return make_node((uint32_t)(row * g_cols + bit), low, high);
}

static void insert_sorted(uint16_t* rows, int depth, uint16_t value) {
    int pos = depth;
    while (pos > 0 && rows[pos - 1] > value) {
        rows[pos] = rows[pos - 1];
        pos--;
    }
    rows[pos] = value;
}

static void solve_state(int depth, const uint16_t* previous, uint32_t* root_out, U128* count_out) {
    if (depth == g_rows) {
        *root_out = 1;
        *count_out = 1;
        return;
    }
    uint64_t hash = state_hash(depth, previous);
    size_t slot = (size_t)hash & (memo_capacity - 1U);
    while (memo[slot].used) {
        if (state_equal(&memo[slot], depth, previous)) {
            *root_out = memo[slot].root;
            *count_out = memo[slot].count;
            return;
        }
        slot = (slot + 1U) & (memo_capacity - 1U);
    }

    int choices = 1 << g_cols;
    uint32_t children[256];
    U128 count = 0;
    for (int mask = 0; mask < choices; mask++) {
        int valid = 1;
        for (int i = 0; i < depth; i++) {
            if (__builtin_popcount((unsigned)mask & previous[i]) > 1) {
                valid = 0;
                break;
            }
        }
        if (!valid) {
            children[mask] = 0;
            continue;
        }
        uint16_t next_rows[8];
        memcpy(next_rows, previous, (size_t)depth * sizeof(previous[0]));
        insert_sorted(next_rows, depth, (uint16_t)mask);
        U128 child_count;
        solve_state(depth + 1, next_rows, &children[mask], &child_count);
        count += child_count;
    }
    uint32_t root = build_selector(children, 0, 0, choices, depth);

    if ((memo_count + 1U) * 10U >= memo_capacity * 7U) {
        memo_rehash();
        slot = (size_t)hash & (memo_capacity - 1U);
        while (memo[slot].used) slot = (slot + 1U) & (memo_capacity - 1U);
    }
    MemoEntry entry = {.used = 1, .depth = (uint8_t)depth, .root = root, .count = count};
    memcpy(entry.rows, previous, (size_t)depth * sizeof(previous[0]));
    memo[slot] = entry;
    memo_count++;
    memo_by_depth[depth]++;
    *root_out = root;
    *count_out = count;
}

static void print_u128(U128 value) {
    char digits[64];
    int count = 0;
    do {
        digits[count++] = (char)('0' + value % 10);
        value /= 10;
    } while (value);
    while (count) putchar(digits[--count]);
}

int main(int argc, char** argv) {
    g_rows = argc > 1 ? atoi(argv[1]) : 6;
    g_cols = argc > 2 ? atoi(argv[2]) : g_rows;
    if (g_rows < 1 || g_rows > 8 || g_cols < 1 || g_cols > 8) {
        fprintf(stderr, "Usage: %s ROWS [COLS], both in 1..8\n", argv[0]);
        return 2;
    }
    node_capacity = 1U << 16;
    nodes = xcalloc(node_capacity, sizeof(nodes[0]));
    node_count = 2;
    node_slot_capacity = 1U << 17;
    node_slots = xcalloc(node_slot_capacity, sizeof(node_slots[0]));
    memo_capacity = 1U << 16;
    memo = xcalloc(memo_capacity, sizeof(memo[0]));

    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);
    uint32_t root;
    U128 count;
    uint16_t empty[8] = {0};
    solve_state(0, empty, &root, &count);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    double elapsed = (double)(finish.tv_sec - start.tv_sec) +
                     (double)(finish.tv_nsec - start.tv_nsec) / 1e9;

    printf("C4-free %dx%d count=", g_rows, g_cols);
    print_u128(count);
    printf(" root=%u zdd_nodes=%zu memo=%zu time=%.3fs\n", root, node_count, memo_count, elapsed);
    printf("memo_by_depth:");
    for (int depth = 0; depth < g_rows; depth++) printf(" %d:%llu", depth, (unsigned long long)memo_by_depth[depth]);
    putchar('\n');

    free(nodes);
    free(node_slots);
    free(memo);
    return 0;
}
