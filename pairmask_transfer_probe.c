/* Experimental four-colour transfer over per-colour row-pair histories. */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef ORDERED_CANON_BITS
#define ORDERED_CANON_BITS 20
#endif

enum {
    MAX_ROWS = 8,
    COLORS = 4,
    MAX_PAIRS = 28,
    MAX_VERTICES = 40,
    MAX_ASSIGNMENT_WORDS = (1 << (2 * MAX_ROWS)) / 64
};

typedef struct { uint32_t m[COLORS]; } State;
typedef unsigned __int128 Count;

typedef struct {
    State key;
    Count value;
    uint8_t used;
} Entry;

typedef struct {
    Entry* entries;
    size_t capacity;
    size_t count;
} Map;

typedef struct {
    State state;
    uint32_t weight;
} Increment;

typedef struct {
    uint8_t own;
    uint8_t counts[MAX_VERTICES];
    uint8_t vertex;
} Signature;

static int g_rows;
static int g_pair_index[MAX_ROWS][MAX_ROWS];
static int g_pair_u[MAX_PAIRS];
static int g_pair_v[MAX_PAIRS];
static int g_pairs;
static uint32_t g_clique[1 << MAX_ROWS];
static unsigned long long g_canon_calls;
static unsigned long long g_canon_discrete;
static unsigned long long g_canon_ir;
static unsigned long long g_canon_ir_nodes;

typedef struct {
    State key;
    State value;
    uint8_t used;
} CanonEntry;

static CanonEntry* g_canon_cache;
static size_t g_canon_capacity;
static size_t g_canon_count;
static int g_canon_direct;
static unsigned long long g_canon_cache_hits;

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static uint64_t state_hash(const State* state) {
    uint64_t h = UINT64_C(0x9e3779b97f4a7c15);
    for (int c = 0; c < COLORS; c++) {
        h = mix64(h ^ ((uint64_t)state->m[c] + UINT64_C(0x9e3779b97f4a7c15)));
    }
    return h;
}

static int state_equal(const State* a, const State* b) {
    return memcmp(a, b, sizeof(*a)) == 0;
}

static void map_init(Map* map, size_t capacity) {
    map->capacity = capacity;
    map->count = 0;
    map->entries = calloc(capacity, sizeof(map->entries[0]));
    if (!map->entries) { fprintf(stderr, "OOM allocating map\n"); exit(1); }
}

static void map_free(Map* map) {
    free(map->entries);
    memset(map, 0, sizeof(*map));
}

static void map_insert_raw(Map* map, const State* key, Count value) {
    size_t slot = (size_t)state_hash(key) & (map->capacity - 1);
    while (map->entries[slot].used) slot = (slot + 1) & (map->capacity - 1);
    map->entries[slot].used = 1;
    map->entries[slot].key = *key;
    map->entries[slot].value = value;
    map->count++;
}

static void map_rehash(Map* map) {
    Map next;
    map_init(&next, map->capacity << 1);
    for (size_t i = 0; i < map->capacity; i++) {
        if (map->entries[i].used) {
            map_insert_raw(&next, &map->entries[i].key, map->entries[i].value);
        }
    }
    free(map->entries);
    *map = next;
}

static void map_add(Map* map, const State* key, Count value) {
    if ((map->count + 1) * 10 >= map->capacity * 7) map_rehash(map);
    size_t slot = (size_t)state_hash(key) & (map->capacity - 1);
    while (map->entries[slot].used) {
        if (state_equal(&map->entries[slot].key, key)) {
            map->entries[slot].value += value;
            return;
        }
        slot = (slot + 1) & (map->capacity - 1);
    }
    map->entries[slot].used = 1;
    map->entries[slot].key = *key;
    map->entries[slot].value = value;
    map->count++;
}

static void sort_masks(State* state) {
    for (int i = 1; i < COLORS; i++) {
        uint32_t x = state->m[i];
        int j = i;
        while (j > 0 && state->m[j - 1] > x) {
            state->m[j] = state->m[j - 1];
            j--;
        }
        state->m[j] = x;
    }
}

static int state_less(const State* lhs, const State* rhs) {
    for (int i = 0; i < COLORS; i++) {
        if (lhs->m[i] != rhs->m[i]) return lhs->m[i] < rhs->m[i];
    }
    return 0;
}

static int canon_cache_get(const State* key, State* value) {
    size_t slot = (size_t)state_hash(key) & (g_canon_capacity - 1);
    if (g_canon_direct) {
        if (!g_canon_cache[slot].used ||
            !state_equal(&g_canon_cache[slot].key, key)) return 0;
        *value = g_canon_cache[slot].value;
        g_canon_cache_hits++;
        return 1;
    }
    while (g_canon_cache[slot].used) {
        if (state_equal(&g_canon_cache[slot].key, key)) {
            *value = g_canon_cache[slot].value;
            g_canon_cache_hits++;
            return 1;
        }
        slot = (slot + 1) & (g_canon_capacity - 1);
    }
    return 0;
}

static void canon_cache_rehash(void) {
    CanonEntry* old = g_canon_cache;
    size_t old_capacity = g_canon_capacity;
    g_canon_capacity <<= 1;
    g_canon_cache = calloc(g_canon_capacity, sizeof(g_canon_cache[0]));
    if (!g_canon_cache) { fprintf(stderr, "OOM growing canon cache\n"); exit(1); }
    g_canon_count = 0;
    for (size_t i = 0; i < old_capacity; i++) if (old[i].used) {
        size_t slot = (size_t)state_hash(&old[i].key) & (g_canon_capacity - 1);
        while (g_canon_cache[slot].used) slot = (slot + 1) & (g_canon_capacity - 1);
        g_canon_cache[slot] = old[i];
        g_canon_count++;
    }
    free(old);
}

static void canon_cache_put(const State* key, const State* value) {
    if (g_canon_direct) {
        size_t slot = (size_t)state_hash(key) & (g_canon_capacity - 1);
        if (!g_canon_cache[slot].used) g_canon_count++;
        g_canon_cache[slot].used = 1;
        g_canon_cache[slot].key = *key;
        g_canon_cache[slot].value = *value;
        return;
    }
    if ((g_canon_count + 1) * 10 >= g_canon_capacity * 7) canon_cache_rehash();
    size_t slot = (size_t)state_hash(key) & (g_canon_capacity - 1);
    while (g_canon_cache[slot].used) slot = (slot + 1) & (g_canon_capacity - 1);
    g_canon_cache[slot].used = 1;
    g_canon_cache[slot].key = *key;
    g_canon_cache[slot].value = *value;
    g_canon_count++;
}

static int signature_compare(const void* lhs_ptr, const void* rhs_ptr) {
    const Signature* lhs = lhs_ptr;
    const Signature* rhs = rhs_ptr;
    if (lhs->own != rhs->own) return (int)lhs->own - (int)rhs->own;
    int cmp = memcmp(lhs->counts, rhs->counts, sizeof(lhs->counts));
    if (cmp != 0) return cmp;
    return (int)lhs->vertex - (int)rhs->vertex;
}

static int signature_same(const Signature* lhs, const Signature* rhs) {
    return lhs->own == rhs->own &&
           memcmp(lhs->counts, rhs->counts, sizeof(lhs->counts)) == 0;
}

static uint32_t permute_pair_mask(uint32_t mask, const int row_position[MAX_ROWS]) {
    uint32_t out = 0;
    while (mask) {
        int p = __builtin_ctz(mask);
        int u = row_position[g_pair_u[p]];
        int v = row_position[g_pair_v[p]];
        if (u > v) { int t = u; u = v; v = t; }
        out |= UINT32_C(1) << g_pair_index[u][v];
        mask &= mask - 1;
    }
    return out;
}

static State state_from_row_colours(const State* input, const uint8_t* colours) {
    int row_position[MAX_ROWS];
    int row_order[MAX_ROWS];
    for (int i = 0; i < g_rows; i++) row_order[i] = i;
    for (int i = 1; i < g_rows; i++) {
        int x = row_order[i], j = i;
        while (j > 0 && colours[row_order[j - 1]] > colours[x]) {
            row_order[j] = row_order[j - 1];
            j--;
        }
        row_order[j] = x;
    }
    for (int i = 0; i < g_rows; i++) row_position[row_order[i]] = i;

    State candidate;
    for (int c = 0; c < COLORS; c++) {
        candidate.m[c] = permute_pair_mask(input->m[c], row_position);
    }
    sort_masks(&candidate);
    return candidate;
}

static int refine_colours(int n, const uint64_t* adjacency, uint8_t* colours) {
    Signature signatures[MAX_VERTICES];
    uint8_t next_colours[MAX_VERTICES];
    int colour_count = 0;
    for (int v = 0; v < n; v++) if ((int)colours[v] + 1 > colour_count) {
        colour_count = (int)colours[v] + 1;
    }
    for (;;) {
        memset(signatures, 0, sizeof(signatures));
        for (int v = 0; v < n; v++) {
            signatures[v].own = colours[v];
            signatures[v].vertex = (uint8_t)v;
            uint64_t neighbours = adjacency[v];
            while (neighbours) {
                int u = __builtin_ctzll(neighbours);
                signatures[v].counts[colours[u]]++;
                neighbours &= neighbours - 1;
            }
        }
        qsort(signatures, (size_t)n, sizeof(signatures[0]), signature_compare);
        int next_count = 1;
        next_colours[signatures[0].vertex] = 0;
        for (int i = 1; i < n; i++) {
            if (!signature_same(&signatures[i - 1], &signatures[i])) next_count++;
            next_colours[signatures[i].vertex] = (uint8_t)(next_count - 1);
        }
        int changed = next_count != colour_count || memcmp(colours, next_colours, (size_t)n) != 0;
        memcpy(colours, next_colours, (size_t)n);
        colour_count = next_count;
        if (!changed || colour_count == n) return colour_count;
    }
}

static void ir_canonicalise_rows(const State* state, int n, const uint64_t* adjacency,
                                 uint8_t* colours, State* best) {
    g_canon_ir_nodes++;
    (void)refine_colours(n, adjacency, colours);

    int chosen_colour = -1;
    int chosen_size = MAX_ROWS + 1;
    for (int colour = 0; colour < n; colour++) {
        int size = 0;
        for (int other = 0; other < g_rows; other++) size += colours[other] == colour;
        if (size > 1 && size < chosen_size) {
            chosen_colour = colour;
            chosen_size = size;
        }
    }

    if (chosen_colour < 0) {
        State candidate = state_from_row_colours(state, colours);
        if (state_less(&candidate, best)) *best = candidate;
        return;
    }

    int next_colour = 0;
    for (int v = 0; v < n; v++) if ((int)colours[v] >= next_colour) {
        next_colour = (int)colours[v] + 1;
    }
    for (int row = 0; row < g_rows; row++) if (colours[row] == chosen_colour) {
        uint8_t branch_colours[MAX_VERTICES];
        memcpy(branch_colours, colours, (size_t)n);
        branch_colours[row] = (uint8_t)next_colour;
        ir_canonicalise_rows(state, n, adjacency, branch_colours, best);
    }
}

/* Canonicalise the coloured incidence graph by row individualize/refine. */
static State canonicalise(State state) {
    sort_masks(&state);
    g_canon_calls++;
    State cached;
    if (canon_cache_get(&state, &cached)) return cached;

    int n = g_rows + g_pairs + COLORS;
    uint64_t adjacency[MAX_VERTICES] = {0};
    uint8_t colours[MAX_VERTICES];

    for (int r = 0; r < g_rows; r++) colours[r] = 0;
    for (int p = 0; p < g_pairs; p++) colours[g_rows + p] = 1;
    for (int c = 0; c < COLORS; c++) colours[g_rows + g_pairs + c] = 2;

    for (int p = 0; p < g_pairs; p++) {
        int pv = g_rows + p;
        int u = g_pair_u[p];
        int v = g_pair_v[p];
        adjacency[pv] |= UINT64_C(1) << u;
        adjacency[pv] |= UINT64_C(1) << v;
        adjacency[u] |= UINT64_C(1) << pv;
        adjacency[v] |= UINT64_C(1) << pv;
        for (int c = 0; c < COLORS; c++) {
            if ((state.m[c] >> p) & 1U) {
                int cv = g_rows + g_pairs + c;
                adjacency[pv] |= UINT64_C(1) << cv;
                adjacency[cv] |= UINT64_C(1) << pv;
            }
        }
    }

    int colour_count = refine_colours(n, adjacency, colours);

    if (colour_count != n) {
        g_canon_ir++;
        State best = {{UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX}};
        ir_canonicalise_rows(&state, n, adjacency, colours, &best);
        canon_cache_put(&state, &best);
        return best;
    }
    g_canon_discrete++;

    int row_position[MAX_ROWS];
    int row_order[MAX_ROWS];
    int colour_order[COLORS];
    for (int i = 0; i < g_rows; i++) row_order[i] = i;
    for (int i = 1; i < g_rows; i++) {
        int x = row_order[i], j = i;
        while (j > 0 && colours[row_order[j - 1]] > colours[x]) {
            row_order[j] = row_order[j - 1]; j--;
        }
        row_order[j] = x;
    }
    for (int i = 0; i < g_rows; i++) row_position[row_order[i]] = i;

    for (int i = 0; i < COLORS; i++) colour_order[i] = i;
    for (int i = 1; i < COLORS; i++) {
        int x = colour_order[i], j = i;
        while (j > 0 && colours[g_rows + g_pairs + colour_order[j - 1]] >
                            colours[g_rows + g_pairs + x]) {
            colour_order[j] = colour_order[j - 1]; j--;
        }
        colour_order[j] = x;
    }

    State out;
    for (int c = 0; c < COLORS; c++) {
        out.m[c] = permute_pair_mask(state.m[colour_order[c]], row_position);
    }
    sort_masks(&out);
    canon_cache_put(&state, &out);
    return out;
}

static int increment_compare(const void* lhs_ptr, const void* rhs_ptr) {
    const Increment* lhs = lhs_ptr;
    const Increment* rhs = rhs_ptr;
    return memcmp(&lhs->state, &rhs->state, sizeof(State));
}

static Increment* build_increments(size_t* count_out) {
    uint64_t assignments = UINT64_C(1) << (2 * g_rows);
    Increment* all = malloc((size_t)assignments * sizeof(*all));
    if (!all) { fprintf(stderr, "OOM allocating increments\n"); exit(1); }
    for (uint64_t code = 0; code < assignments; code++) {
        uint8_t row_sets[COLORS] = {0};
        uint64_t x = code;
        for (int r = 0; r < g_rows; r++) {
            row_sets[x & 3U] |= (uint8_t)(1U << r);
            x >>= 2;
        }
        for (int c = 0; c < COLORS; c++) all[code].state.m[c] = g_clique[row_sets[c]];
        all[code].weight = 1;
    }
    qsort(all, (size_t)assignments, sizeof(*all), increment_compare);
    size_t unique = 0;
    for (size_t i = 0; i < assignments; i++) {
        if (unique > 0 && state_equal(&all[unique - 1].state, &all[i].state)) {
            all[unique - 1].weight++;
        } else {
            all[unique++] = all[i];
        }
    }
    *count_out = unique;
    return all;
}

static void print_count(Count value) {
    char digits[64];
    int n = 0;
    if (value == 0) { putchar('0'); return; }
    while (value) {
        digits[n++] = (char)('0' + value % 10);
        value /= 10;
    }
    while (n) putchar(digits[--n]);
}

static Count map_total(const Map* map) {
    Count total = 0;
    for (size_t i = 0; i < map->capacity; i++) {
        if (map->entries[i].used) total += map->entries[i].value;
    }
    return total;
}

typedef struct {
    uint32_t* ids;
    size_t count;
    size_t capacity;
} IncrementIndex;

typedef struct {
    State key;
    Count value;
    uint8_t kmax;
    uint8_t used;
} PackMemoEntry;

typedef struct {
    PackMemoEntry* entries;
    size_t capacity;
    size_t count;
} PackMemo;

typedef struct {
    const Increment* increments;
    IncrementIndex* per_bit;
    PackMemo memo;
    unsigned long long calls;
    unsigned long long hits;
    uint64_t started_ns;
    uint64_t next_report_ns;
} PackSolver;

static void index_push(IncrementIndex* index, uint32_t id) {
    if (index->count == index->capacity) {
        size_t next_capacity = index->capacity ? index->capacity << 1 : 256;
        uint32_t* next = realloc(index->ids, next_capacity * sizeof(next[0]));
        if (!next) { fprintf(stderr, "OOM growing increment index\n"); exit(1); }
        index->ids = next;
        index->capacity = next_capacity;
    }
    index->ids[index->count++] = id;
}

static int state_is_subset(const State* subset, const State* superset) {
    for (int c = 0; c < COLORS; c++) {
        if (subset->m[c] & ~superset->m[c]) return 0;
    }
    return 1;
}

static State state_without(const State* state, const State* removed) {
    State out;
    for (int c = 0; c < COLORS; c++) out.m[c] = state->m[c] & ~removed->m[c];
    return out;
}

static int state_popcount(const State* state) {
    int count = 0;
    for (int c = 0; c < COLORS; c++) count += __builtin_popcount(state->m[c]);
    return count;
}

static uint64_t pack_memo_hash(const State* state, int kmax) {
    return mix64(state_hash(state) ^ ((uint64_t)kmax * UINT64_C(0x9e3779b97f4a7c15)));
}

static void pack_memo_init(PackMemo* memo, size_t capacity) {
    memo->capacity = capacity;
    memo->count = 0;
    memo->entries = calloc(capacity, sizeof(memo->entries[0]));
    if (!memo->entries) { fprintf(stderr, "OOM allocating set-packing memo\n"); exit(1); }
}

static void pack_memo_insert_raw(PackMemo* memo, const State* key, int kmax,
                                 Count value) {
    size_t slot = (size_t)pack_memo_hash(key, kmax) & (memo->capacity - 1);
    while (memo->entries[slot].used) slot = (slot + 1) & (memo->capacity - 1);
    memo->entries[slot].used = 1;
    memo->entries[slot].key = *key;
    memo->entries[slot].kmax = (uint8_t)kmax;
    memo->entries[slot].value = value;
    memo->count++;
}

static void pack_memo_rehash(PackMemo* memo) {
    PackMemo next;
    pack_memo_init(&next, memo->capacity << 1);
    for (size_t i = 0; i < memo->capacity; i++) if (memo->entries[i].used) {
        pack_memo_insert_raw(&next, &memo->entries[i].key, memo->entries[i].kmax,
                             memo->entries[i].value);
    }
    free(memo->entries);
    *memo = next;
}

static int pack_memo_get(const PackMemo* memo, const State* key, int kmax,
                         Count* value) {
    size_t slot = (size_t)pack_memo_hash(key, kmax) & (memo->capacity - 1);
    while (memo->entries[slot].used) {
        if (memo->entries[slot].kmax == kmax && state_equal(&memo->entries[slot].key, key)) {
            *value = memo->entries[slot].value;
            return 1;
        }
        slot = (slot + 1) & (memo->capacity - 1);
    }
    return 0;
}

static void pack_memo_put(PackMemo* memo, const State* key, int kmax,
                          Count value) {
    if ((memo->count + 1) * 10 >= memo->capacity * 7) pack_memo_rehash(memo);
    pack_memo_insert_raw(memo, key, kmax, value);
}

static void pack_report(PackSolver* solver) {
    uint64_t now = now_ns();
    if (now < solver->next_report_ns) return;
    double elapsed = (double)(now - solver->started_ns) / 1e9;
    fprintf(stderr, "setpack elapsed=%.1fs states=%zu calls=%llu hits=%llu canon_cache=%zu\n",
            elapsed, solver->memo.count, solver->calls, solver->hits, g_canon_count);
    fflush(stderr);
    solver->next_report_ns = now + UINT64_C(5000000000);
}

static int minimum_column_pair_tokens(void) {
    int quotient = g_rows / COLORS;
    int remainder = g_rows % COLORS;
    return remainder * (quotient + 1) * quotient / 2 +
           (COLORS - remainder) * quotient * (quotient - 1) / 2;
}

static Count solve_setpack(PackSolver* solver, State available, int k) {
    int available_bits = state_popcount(&available);
    if (k == 0) return 1;
    if (available_bits < k * minimum_column_pair_tokens()) return 0;

    available = canonicalise(available);

    Count cached;
    if (pack_memo_get(&solver->memo, &available, k, &cached)) {
        solver->hits++;
        return cached;
    }
    solver->calls++;
    pack_report(solver);

    Count result = 0;

    int best_bit = -1;
    size_t best_count = SIZE_MAX;
    for (int c = 0; c < COLORS; c++) {
        uint32_t mask = available.m[c];
        while (mask) {
            int pair = __builtin_ctz(mask);
            int bit = c * g_pairs + pair;
            const IncrementIndex* index = &solver->per_bit[bit];
            size_t feasible = 0;
            for (size_t i = 0; i < index->count; i++) {
                if (state_is_subset(&solver->increments[index->ids[i]].state, &available)) {
                    feasible++;
                    if (feasible >= best_count) break;
                }
            }
            if (feasible < best_count) {
                best_count = feasible;
                best_bit = bit;
            }
            mask &= mask - 1;
        }
    }

    if (best_bit < 0) {
        pack_memo_put(&solver->memo, &available, k, 0);
        return result;
    }

    State bit_state = {{0, 0, 0, 0}};
    bit_state.m[best_bit / g_pairs] = UINT32_C(1) << (best_bit % g_pairs);
    State without_bit = state_without(&available, &bit_state);
    result = solve_setpack(solver, without_bit, k);

    const IncrementIndex* choices = &solver->per_bit[best_bit];
    for (size_t i = 0; i < choices->count; i++) {
        const Increment* increment = &solver->increments[choices->ids[i]];
        if (!state_is_subset(&increment->state, &available)) continue;
        State remaining = state_without(&available, &increment->state);
        Count suffix = solve_setpack(solver, remaining, k - 1);
        result += suffix * increment->weight;
    }

    pack_memo_put(&solver->memo, &available, k, result);
    return result;
}

static void run_setpack(const Increment* increments, size_t increment_count, int columns) {
    if (g_rows <= COLORS) {
        fprintf(stderr, "setpack mode currently requires ROWS > 4 (zero-mask columns unsupported)\n");
        exit(2);
    }
    int token_count = COLORS * g_pairs;
    IncrementIndex* per_bit = calloc((size_t)token_count, sizeof(per_bit[0]));
    if (!per_bit) { fprintf(stderr, "OOM allocating token index\n"); exit(1); }
    for (size_t id = 0; id < increment_count; id++) {
        for (int c = 0; c < COLORS; c++) {
            uint32_t mask = increments[id].state.m[c];
            while (mask) {
                int pair = __builtin_ctz(mask);
                index_push(&per_bit[c * g_pairs + pair], (uint32_t)id);
                mask &= mask - 1;
            }
        }
    }

    PackSolver solver = {
        .increments = increments,
        .per_bit = per_bit,
    };
    pack_memo_init(&solver.memo, 1U << 20);
    solver.started_ns = now_ns();
    solver.next_report_ns = solver.started_ns + UINT64_C(5000000000);

    uint32_t all_pairs = g_pairs == 32 ? UINT32_MAX : ((UINT32_C(1) << g_pairs) - 1U);
    State available = {{all_pairs, all_pairs, all_pairs, all_pairs}};
    Count packed = solve_setpack(&solver, available, columns);

    Count factorial = 1;
    printf("setpack states=%zu calls=%llu hits=%llu time=%.3fs\n",
           solver.memo.count, solver.calls, solver.hits,
           (double)(now_ns() - solver.started_ns) / 1e9);
    for (int k = 2; k <= columns; k++) factorial *= (unsigned)k;
    printf("T4(%d,%d)=", g_rows, columns);
    print_count(packed * factorial);
    putchar('\n');

    for (int bit = 0; bit < token_count; bit++) free(per_bit[bit].ids);
    free(per_bit);
    free(solver.memo.entries);
}

typedef struct {
    const Increment* increments;
    size_t increment_count;
    const uint64_t* terminal_uses;
    size_t terminal_words;
    uint64_t terminal_valid[MAX_ASSIGNMENT_WORDS];
    uint8_t* pair_families;
    uint32_t* pair_counts;
    int contract_two;
    unsigned long long contracted_calls;
    unsigned long long contracted_submasks;
    PackMemo memo;
    unsigned long long calls;
    unsigned long long hits;
    size_t completed[65];
    uint64_t started_ns;
    uint64_t next_report_ns;
} OrderedSolver;

enum { PAIR_STATE_COUNT = 1 << (2 * MAX_ROWS) };

static uint64_t* build_terminal_token_bitsets(size_t* word_count_out) {
    size_t assignment_count = (size_t)1U << (2 * g_rows);
    size_t word_count = (assignment_count + 63) / 64;
    size_t token_count = (size_t)COLORS * (size_t)g_pairs;
    uint64_t* uses = calloc(token_count * word_count, sizeof(uses[0]));
    if (!uses) { fprintf(stderr, "OOM allocating terminal bitsets\n"); exit(1); }

    for (size_t code = 0; code < assignment_count; code++) {
        uint8_t row_sets[COLORS] = {0, 0, 0, 0};
        size_t x = code;
        for (int row = 0; row < g_rows; row++) {
            row_sets[x & 3U] |= (uint8_t)(1U << row);
            x >>= 2;
        }
        for (int colour = 0; colour < COLORS; colour++) {
            uint32_t mask = g_clique[row_sets[colour]];
            while (mask) {
                int pair = __builtin_ctz(mask);
                size_t token = (size_t)colour * (size_t)g_pairs + (size_t)pair;
                uses[token * word_count + code / 64] |= UINT64_C(1) << (code % 64);
                mask &= mask - 1;
            }
        }
    }
    *word_count_out = word_count;
    return uses;
}

static Count count_terminal_columns(OrderedSolver* solver, const State* available) {
    memset(solver->terminal_valid, 0xff,
           solver->terminal_words * sizeof(solver->terminal_valid[0]));
    uint32_t all_pairs = (UINT32_C(1) << g_pairs) - 1U;
    for (int colour = 0; colour < COLORS; colour++) {
        uint32_t forbidden = all_pairs & ~available->m[colour];
        while (forbidden) {
            int pair = __builtin_ctz(forbidden);
            size_t token = (size_t)colour * (size_t)g_pairs + (size_t)pair;
            const uint64_t* uses = solver->terminal_uses + token * solver->terminal_words;
            for (size_t word = 0; word < solver->terminal_words; word++) {
                solver->terminal_valid[word] &= ~uses[word];
            }
            forbidden &= forbidden - 1;
        }
    }

    Count result = 0;
    for (size_t word = 0; word < solver->terminal_words; word++) {
        result += (unsigned)__builtin_popcountll(solver->terminal_valid[word]);
    }
    return result;
}

static uint64_t build_colour_pair_family(const State* available, int colour,
                                         uint8_t* family) {
    size_t state_count = (size_t)1U << (2 * g_rows);
    memset(family, 0, state_count * sizeof(family[0]));
    uint8_t legal[1 << MAX_ROWS];
    for (int rows = 0; rows < (1 << g_rows); rows++) {
        legal[rows] = !(g_clique[rows] & ~available->m[colour]);
    }
    uint64_t work = 0;
    for (int first = 0; first < (1 << g_rows); first++) if (legal[first]) {
        for (int second = 0; second < (1 << g_rows); second++) {
            if (legal[second] && __builtin_popcount((unsigned)(first & second)) <= 1) {
                unsigned state = (unsigned)first | ((unsigned)second << g_rows);
                family[state] = 1;
                work += UINT64_C(1) << (2 * g_rows - __builtin_popcount(state));
            }
        }
    }
    return work;
}

static void convolve_colour_pair(const uint8_t* first, const uint8_t* second,
                                 uint32_t* counts) {
    unsigned state_count = 1U << (2 * g_rows);
    unsigned full = state_count - 1U;
    memset(counts, 0, (size_t)state_count * sizeof(counts[0]));
    for (unsigned left = 0; left < state_count; left++) if (first[left]) {
        unsigned available = full ^ left;
        unsigned right = available;
        for (;;) {
            if (second[right]) counts[left | right]++;
            if (right == 0) break;
            right = (right - 1U) & available;
        }
    }
}

static Count count_two_columns_contracted(OrderedSolver* solver,
                                          const State* available) {
    size_t state_count = (size_t)1U << (2 * g_rows);
    unsigned full = (unsigned)state_count - 1U;
    uint8_t* families[COLORS];
    uint64_t work[COLORS];
    int order[COLORS] = {0, 1, 2, 3};
    for (int colour = 0; colour < COLORS; colour++) {
        families[colour] = solver->pair_families + (size_t)colour * PAIR_STATE_COUNT;
        work[colour] = build_colour_pair_family(available, colour, families[colour]);
    }
    for (int i = 1; i < COLORS; i++) {
        int x = order[i], j = i;
        while (j > 0 && work[order[j - 1]] > work[x]) {
            order[j] = order[j - 1];
            j--;
        }
        order[j] = x;
    }
    uint32_t* low = solver->pair_counts;
    uint32_t* high = low + PAIR_STATE_COUNT;

    convolve_colour_pair(families[order[0]], families[order[3]], low);
    convolve_colour_pair(families[order[1]], families[order[2]], high);

    Count result = 0;
    for (unsigned used = 0; used < state_count; used++) {
        result += (Count)low[used] * high[full ^ used];
    }
    solver->contracted_calls++;
    solver->contracted_submasks += work[order[0]] + work[order[1]];
    return result;
}

static void ordered_report(OrderedSolver* solver) {
    uint64_t now = now_ns();
    if (now < solver->next_report_ns) return;
    fprintf(stderr, "ordered elapsed=%.1fs states=%zu calls=%llu hits=%llu done1=%zu done2=%zu done3=%zu canon_cache=%zu/%zu canon_hits=%llu\n",
            (double)(now - solver->started_ns) / 1e9, solver->memo.count,
            solver->calls, solver->hits, solver->completed[1], solver->completed[2],
            solver->completed[3], g_canon_count, g_canon_capacity, g_canon_cache_hits);
    fflush(stderr);
    solver->next_report_ns = now + UINT64_C(5000000000);
}

static Count solve_ordered(OrderedSolver* solver, State available, int columns) {
    if (columns == 0) return 1;
    if (state_popcount(&available) < columns * minimum_column_pair_tokens()) return 0;
    available = canonicalise(available);

    Count cached;
    if (pack_memo_get(&solver->memo, &available, columns, &cached)) {
        solver->hits++;
        return cached;
    }
    solver->calls++;
    ordered_report(solver);

    Count result = 0;
    if (solver->contract_two && columns == 2) {
        result = count_two_columns_contracted(solver, &available);
    } else if (columns == 1) {
        result = count_terminal_columns(solver, &available);
    } else {
        for (size_t i = 0; i < solver->increment_count; i++) {
            const Increment* increment = &solver->increments[i];
            if (!state_is_subset(&increment->state, &available)) continue;
            State remaining = state_without(&available, &increment->state);
            result += increment->weight * solve_ordered(solver, remaining, columns - 1);
        }
    }

    pack_memo_put(&solver->memo, &available, columns, result);
    solver->completed[columns]++;
    return result;
}

typedef struct {
    State available;
    uint32_t multiplicity;
    uint8_t sizes[COLORS];
    uint8_t parts;
} RootOrbit;

typedef struct {
    State state;
    Count weight;
} WeightedState;

static int weighted_state_compare(const void* lhs_ptr, const void* rhs_ptr) {
    const WeightedState* lhs = lhs_ptr;
    const WeightedState* rhs = rhs_ptr;
    return memcmp(&lhs->state, &rhs->state, sizeof(State));
}

static WeightedState* build_child_shards(const Increment* increments,
                                         size_t increment_count, State available,
                                         int remaining_columns,
                                         size_t* count_out) {
    available = canonicalise(available);
    Map children;
    map_init(&children, 1U << 13);
    for (size_t i = 0; i < increment_count; i++) {
        const Increment* increment = &increments[i];
        if (!state_is_subset(&increment->state, &available)) continue;
        State remaining = state_without(&available, &increment->state);
        if (state_popcount(&remaining) <
            remaining_columns * minimum_column_pair_tokens()) continue;
        remaining = canonicalise(remaining);
        map_add(&children, &remaining, increment->weight);
    }

    WeightedState* shards = malloc(children.count * sizeof(shards[0]));
    if (!shards) { fprintf(stderr, "OOM allocating child shards\n"); exit(1); }
    size_t count = 0;
    for (size_t i = 0; i < children.capacity; i++) if (children.entries[i].used) {
        shards[count].state = children.entries[i].key;
        shards[count].weight = children.entries[i].value;
        count++;
    }
    map_free(&children);
    qsort(shards, count, sizeof(shards[0]), weighted_state_compare);
    *count_out = count;
    return shards;
}

static uint32_t small_factorial(int n) {
    uint32_t result = 1;
    for (int i = 2; i <= n; i++) result *= (uint32_t)i;
    return result;
}

static void build_root_orbits_rec(RootOrbit orbits[32], size_t* count,
                                  uint8_t sizes[COLORS], int parts,
                                  int remaining, int maximum) {
    if (remaining == 0) {
        uint32_t all_pairs = (UINT32_C(1) << g_pairs) - 1U;
        RootOrbit* orbit = &orbits[(*count)++];
        memset(orbit, 0, sizeof(*orbit));
        orbit->parts = (uint8_t)parts;
        uint32_t denominator = 1;
        for (int i = 0; i < parts; i++) {
            orbit->sizes[i] = sizes[i];
            denominator *= small_factorial(sizes[i]);
            uint8_t row_set = 0;
            int first = 0;
            for (int j = 0; j < i; j++) first += sizes[j];
            for (int row = first; row < first + sizes[i]; row++) {
                row_set |= (uint8_t)(1U << row);
            }
            orbit->available.m[i] = all_pairs & ~g_clique[row_set];
        }
        for (int i = parts; i < COLORS; i++) orbit->available.m[i] = all_pairs;
        for (int size = 1; size <= g_rows; size++) {
            int multiplicity = 0;
            for (int i = 0; i < parts; i++) multiplicity += sizes[i] == size;
            denominator *= small_factorial(multiplicity);
        }
        uint32_t inject_colours = small_factorial(COLORS) /
                                  small_factorial(COLORS - parts);
        orbit->multiplicity = small_factorial(g_rows) / denominator * inject_colours;
        return;
    }
    if (parts == COLORS) return;
    int upper = remaining < maximum ? remaining : maximum;
    for (int size = upper; size >= 1; size--) {
        sizes[parts] = (uint8_t)size;
        build_root_orbits_rec(orbits, count, sizes, parts + 1,
                              remaining - size, size);
    }
}

static size_t build_root_orbits(RootOrbit orbits[32]) {
    size_t count = 0;
    uint8_t sizes[COLORS] = {0, 0, 0, 0};
    build_root_orbits_rec(orbits, &count, sizes, 0, g_rows, g_rows);
    return count;
}

static void print_root_orbit(const RootOrbit* orbit, size_t index,
                             Count contribution) {
    printf("orbit=%zu sizes=", index);
    for (int i = 0; i < orbit->parts; i++) {
        if (i) putchar('+');
        printf("%u", orbit->sizes[i]);
    }
    printf(" multiplicity=%u contribution=", orbit->multiplicity);
    print_count(contribution);
    putchar('\n');
    fflush(stdout);
}

static void run_ordered(const Increment* increments, size_t increment_count, int columns,
                        int selected_orbit, int selected_child, int contract_two) {
    if (g_rows <= COLORS) {
        fprintf(stderr, "ordered mode currently requires ROWS > 4 (zero-mask columns unsupported)\n");
        exit(2);
    }
    size_t terminal_words = 0;
    uint64_t* terminal_uses = build_terminal_token_bitsets(&terminal_words);
    uint8_t* pair_families = NULL;
    uint32_t* pair_counts = NULL;
    if (contract_two) {
        pair_families = calloc(COLORS * PAIR_STATE_COUNT, sizeof(pair_families[0]));
        pair_counts = calloc(2U * PAIR_STATE_COUNT, sizeof(pair_counts[0]));
        if (!pair_families || !pair_counts) {
            fprintf(stderr, "OOM allocating two-column contraction workspace\n");
            exit(1);
        }
    }
    OrderedSolver solver = {
        .increments = increments,
        .increment_count = increment_count,
        .terminal_uses = terminal_uses,
        .terminal_words = terminal_words,
        .pair_families = pair_families,
        .pair_counts = pair_counts,
        .contract_two = contract_two,
    };
    pack_memo_init(&solver.memo, 1U << 20);
    solver.started_ns = now_ns();
    solver.next_report_ns = solver.started_ns + UINT64_C(5000000000);

    Count result = 0;
    if (columns == 0) {
        result = 1;
    } else {
        RootOrbit orbits[32];
        size_t orbit_count = build_root_orbits(orbits);
        uint32_t total_multiplicity = 0;
        for (size_t i = 0; i < orbit_count; i++) total_multiplicity += orbits[i].multiplicity;
        if (total_multiplicity != (UINT32_C(1) << (2 * g_rows))) {
            fprintf(stderr, "Internal error: root orbit multiplicities sum to %u\n",
                    total_multiplicity);
            exit(1);
        }
        if (selected_orbit >= (int)orbit_count) {
            fprintf(stderr, "ORBIT must be 0..%zu for %d rows\n", orbit_count - 1, g_rows);
            exit(2);
        }
        size_t first = selected_orbit < 0 ? 0 : (size_t)selected_orbit;
        size_t end = selected_orbit < 0 ? orbit_count : first + 1;
        for (size_t i = first; i < end; i++) {
            Count contribution;
            if (selected_child >= 0) {
                if (columns < 2) {
                    fprintf(stderr, "CHILD requires at least two columns\n");
                    exit(2);
                }
                size_t child_count = 0;
                WeightedState* children = build_child_shards(increments, increment_count,
                                                              orbits[i].available,
                                                              columns - 2,
                                                              &child_count);
                printf("root_orbit=%zu child_shards=%zu\n", i, child_count);
                if (selected_child >= (int)child_count) {
                    fprintf(stderr, "CHILD must be 0..%zu for root orbit %zu\n",
                            child_count - 1, i);
                    exit(2);
                }
                const WeightedState* child = &children[selected_child];
                Count suffix = solve_ordered(&solver, child->state, columns - 2);
                contribution = (Count)orbits[i].multiplicity * child->weight * suffix;
                printf("child=%d transition_weight=", selected_child);
                print_count(child->weight);
                printf(" contribution=");
                print_count(contribution);
                putchar('\n');
                free(children);
            } else {
                contribution = (Count)orbits[i].multiplicity *
                               solve_ordered(&solver, orbits[i].available, columns - 1);
                print_root_orbit(&orbits[i], i, contribution);
            }
            result += contribution;
        }
    }
    printf("ordered states=%zu calls=%llu hits=%llu contracted=%llu submasks=%llu time=%.3fs\n",
           solver.memo.count, solver.calls, solver.hits,
           solver.contracted_calls, solver.contracted_submasks,
           (double)(now_ns() - solver.started_ns) / 1e9);
    printf(selected_child >= 0 ? "T4-shard(%d,%d)=" :
           (selected_orbit < 0 ? "T4(%d,%d)=" : "T4-orbit(%d,%d)="),
           g_rows, columns);
    print_count(result);
    putchar('\n');
    free(pair_counts);
    free(pair_families);
    free(terminal_uses);
    free(solver.memo.entries);
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 6 ||
        (argc == 5 && strcmp(argv[3], "--ordered") != 0 &&
         strcmp(argv[3], "--contracted") != 0) ||
        (argc == 6 && strcmp(argv[3], "--contracted") != 0)) {
        fprintf(stderr, "Usage: %s ROWS COLUMNS [--setpack|--ordered|--contracted [ORBIT [CHILD]]]\n",
                argv[0]);
        return 2;
    }
    g_rows = atoi(argv[1]);
    int columns = atoi(argv[2]);
    if (g_rows < 2 || g_rows > MAX_ROWS || columns < 0 || g_rows * columns > 64) {
        fprintf(stderr, "ROWS must be 2..8 and ROWS*COLUMNS at most 64\n");
        return 2;
    }

    memset(g_pair_index, -1, sizeof(g_pair_index));
    for (int u = 0; u < g_rows; u++) {
        for (int v = u + 1; v < g_rows; v++) {
            g_pair_index[u][v] = g_pair_index[v][u] = g_pairs;
            g_pair_u[g_pairs] = u;
            g_pair_v[g_pairs] = v;
            g_pairs++;
        }
    }
    for (int set = 0; set < (1 << g_rows); set++) {
        uint32_t mask = 0;
        for (int u = 0; u < g_rows; u++) if ((set >> u) & 1U) {
            for (int v = u + 1; v < g_rows; v++) if ((set >> v) & 1U) {
                mask |= UINT32_C(1) << g_pair_index[u][v];
            }
        }
        g_clique[set] = mask;
    }
    g_canon_direct = argc >= 4 &&
                     (strcmp(argv[3], "--ordered") == 0 ||
                      strcmp(argv[3], "--contracted") == 0);
    g_canon_capacity = g_canon_direct ? (1U << ORDERED_CANON_BITS) : (1U << 20);
    g_canon_cache = calloc(g_canon_capacity, sizeof(g_canon_cache[0]));
    if (!g_canon_cache) { fprintf(stderr, "OOM allocating canon cache\n"); exit(1); }

    size_t increment_count = 0;
    Increment* increments = build_increments(&increment_count);
    printf("rows=%d colours=4 pair_bits=%d increments=%zu\n",
           g_rows, g_pairs * COLORS, increment_count);

    if (argc >= 4) {
        if (strcmp(argv[3], "--setpack") == 0) {
            run_setpack(increments, increment_count, columns);
        } else if (strcmp(argv[3], "--ordered") == 0 ||
                   strcmp(argv[3], "--contracted") == 0) {
            int selected_orbit = argc == 5 ? atoi(argv[4]) : -1;
            if (argc == 6) selected_orbit = atoi(argv[4]);
            int selected_child = argc == 6 ? atoi(argv[5]) : -1;
            if (selected_orbit < -1) {
                fprintf(stderr, "ORBIT must be nonnegative\n");
                return 2;
            }
            if (selected_child < -1) {
                fprintf(stderr, "CHILD must be nonnegative\n");
                return 2;
            }
            run_ordered(increments, increment_count, columns, selected_orbit,
                        selected_child,
                        strcmp(argv[3], "--contracted") == 0);
        } else {
            fprintf(stderr, "Unknown mode: %s\n", argv[3]);
            return 2;
        }
        free(g_canon_cache);
        free(increments);
        return 0;
    }

    Map current;
    map_init(&current, 16);
    State zero = {{0, 0, 0, 0}};
    map_add(&current, &zero, 1);
    printf("column=0 states=1 total=1\n");

    for (int column = 1; column <= columns; column++) {
        uint64_t started = now_ns();
        unsigned long long calls_before = g_canon_calls;
        unsigned long long discrete_before = g_canon_discrete;
        unsigned long long ir_before = g_canon_ir;
        unsigned long long ir_nodes_before = g_canon_ir_nodes;
        Map next;
        size_t initial_capacity = 16;
        while (initial_capacity < current.count * 4) initial_capacity <<= 1;
        map_init(&next, initial_capacity);

        unsigned long long transitions = 0;
        for (size_t slot = 0; slot < current.capacity; slot++) {
            if (!current.entries[slot].used) continue;
            const State* state = &current.entries[slot].key;
            Count ways = current.entries[slot].value;
            for (size_t i = 0; i < increment_count; i++) {
                State out;
                int legal = 1;
                for (int c = 0; c < COLORS; c++) {
                    if (state->m[c] & increments[i].state.m[c]) { legal = 0; break; }
                    out.m[c] = state->m[c] | increments[i].state.m[c];
                }
                if (!legal) continue;
                transitions++;
                out = canonicalise(out);
                map_add(&next, &out, ways * increments[i].weight);
            }
        }
        map_free(&current);
        current = next;
        double elapsed = (double)(now_ns() - started) / 1e9;
        unsigned long long canon_delta = g_canon_calls - calls_before;
        unsigned long long discrete_delta = g_canon_discrete - discrete_before;
        unsigned long long ir_delta = g_canon_ir - ir_before;
        unsigned long long ir_nodes_delta = g_canon_ir_nodes - ir_nodes_before;
        printf("column=%d states=%zu transitions=%llu wl_discrete=%llu ir=%llu/%llu calls=%llu time=%.3fs total=",
               column, current.count, transitions, discrete_delta, ir_delta, ir_nodes_delta,
               canon_delta, elapsed);
        print_count(map_total(&current));
        putchar('\n');
        fflush(stdout);
    }

    map_free(&current);
    free(g_canon_cache);
    free(increments);
    return 0;
}
