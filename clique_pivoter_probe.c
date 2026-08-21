/* Weighted succinct-clique-tree probe for the column compatibility graph. */
#define _POSIX_C_SOURCE 200809L
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned __int128 U128;

typedef struct {
    uint64_t lo;
    uint64_t hi;
    uint32_t weight;
} Increment;

static Increment* increments;
static uint64_t* adjacency;
static uint64_t* work_sets;
static uint64_t* allowed_sets;
static size_t vertex_count;
static size_t words;
static int target_k;
static uint64_t node_limit;
static uint64_t nodes_visited;
static uint64_t leaves;
static int stopped;
static U128 total;

static int increment_compare(const void* lhs_ptr, const void* rhs_ptr) {
    const Increment* lhs = lhs_ptr;
    const Increment* rhs = rhs_ptr;
    if (lhs->hi != rhs->hi) return lhs->hi < rhs->hi ? -1 : 1;
    if (lhs->lo != rhs->lo) return lhs->lo < rhs->lo ? -1 : 1;
    return 0;
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
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

static int set_empty(const uint64_t* set) {
    for (size_t w = 0; w < words; w++) if (set[w]) return 0;
    return 1;
}

static int choose_pivot(const uint64_t* set) {
    int best = -1;
    unsigned best_degree = 0;
    for (size_t word = 0; word < words; word++) {
        uint64_t candidates = set[word];
        while (candidates) {
            int bit = __builtin_ctzll(candidates);
            int vertex = (int)(word * 64U + (unsigned)bit);
            unsigned degree = 0;
            const uint64_t* neighbours = adjacency + (size_t)vertex * words;
            for (size_t w = 0; w < words; w++) {
                degree += (unsigned)__builtin_popcountll(set[w] & neighbours[w]);
            }
            if (best < 0 || degree > best_degree) {
                best = vertex;
                best_degree = degree;
            }
            candidates &= candidates - 1U;
        }
    }
    return best;
}

static void pivoter(const uint64_t* set, int depth, int hold_count,
                    U128 hold_weight, const U128* pivot_poly) {
    if (stopped || hold_count > target_k) return;
    if (++nodes_visited >= node_limit) {
        stopped = 1;
        return;
    }
    if (set_empty(set)) {
        int need = target_k - hold_count;
        if (need >= 0 && need <= target_k) total += hold_weight * pivot_poly[need];
        leaves++;
        return;
    }
    if (depth >= 28) {
        fprintf(stderr, "unexpected clique-tree depth >= 28\n");
        stopped = 1;
        return;
    }

    int pivot = choose_pivot(set);
    const uint64_t* pivot_neighbours = adjacency + (size_t)pivot * words;
    uint64_t* child = work_sets + (size_t)(depth + 1) * words;
    for (size_t w = 0; w < words; w++) child[w] = set[w] & pivot_neighbours[w];
    U128 next_poly[9];
    memcpy(next_poly, pivot_poly, (size_t)(target_k + 1) * sizeof(next_poly[0]));
    U128 pivot_weight = increments[pivot].weight;
    for (int degree = target_k; degree >= 1; degree--) {
        next_poly[degree] += next_poly[degree - 1] * pivot_weight;
    }
    pivoter(child, depth + 1, hold_count, hold_weight, next_poly);

    uint64_t* allowed = allowed_sets + (size_t)depth * words;
    memcpy(allowed, set, words * sizeof(allowed[0]));
    for (size_t word = 0; word < words && !stopped; word++) {
        uint64_t nonneighbors = set[word] & ~pivot_neighbours[word];
        nonneighbors &= ~(UINT64_C(1) << (pivot & 63)) | (word != (size_t)pivot / 64U ? UINT64_MAX : 0);
        while (nonneighbors && !stopped) {
            int bit = __builtin_ctzll(nonneighbors);
            int vertex = (int)(word * 64U + (unsigned)bit);
            allowed[(size_t)vertex / 64U] &= ~(UINT64_C(1) << (vertex & 63));
            const uint64_t* neighbours = adjacency + (size_t)vertex * words;
            for (size_t w = 0; w < words; w++) child[w] = set[w] & neighbours[w] & allowed[w];
            pivoter(child, depth + 1, hold_count + 1,
                    hold_weight * increments[vertex].weight, pivot_poly);
            nonneighbors &= nonneighbors - 1U;
        }
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    target_k = argc > 1 ? atoi(argv[1]) : 4;
    long long root_end_arg = argc > 2 ? atoll(argv[2]) : -1;
    node_limit = argc > 3 ? strtoull(argv[3], NULL, 10) : UINT64_C(1000000);
    if (target_k < 1 || target_k > 8) {
        fprintf(stderr, "Usage: %s K [ROOT_END [NODE_LIMIT]]\n", argv[0]);
        return 2;
    }

    int pair_index[8][8] = {{0}};
    int pairs = 0;
    for (int u = 0; u < 8; u++) for (int v = u + 1; v < 8; v++) pair_index[u][v] = pairs++;
    size_t assignments = 1U << 16;
    Increment* all = calloc(assignments, sizeof(all[0]));
    if (!all) return 1;
    for (size_t code = 0; code < assignments; code++) {
        unsigned colours[8];
        size_t value = code;
        for (int row = 0; row < 8; row++) {
            colours[row] = (unsigned)(value & 3U);
            value >>= 2;
        }
        unsigned __int128 mask = 0;
        for (int u = 0; u < 8; u++) for (int v = u + 1; v < 8; v++) {
            if (colours[u] == colours[v]) {
                mask |= (unsigned __int128)1 << (colours[u] * pairs + pair_index[u][v]);
            }
        }
        all[code] = (Increment){.lo = (uint64_t)mask, .hi = (uint64_t)(mask >> 64), .weight = 1};
    }
    qsort(all, assignments, sizeof(all[0]), increment_compare);
    vertex_count = 0;
    for (size_t i = 0; i < assignments; i++) {
        if (vertex_count && all[i].lo == all[vertex_count - 1].lo &&
            all[i].hi == all[vertex_count - 1].hi) {
            all[vertex_count - 1].weight++;
        } else {
            all[vertex_count++] = all[i];
        }
    }
    increments = all;
    words = (vertex_count + 63U) / 64U;
    adjacency = calloc(vertex_count * words, sizeof(adjacency[0]));
    if (!adjacency) {
        fprintf(stderr, "failed to allocate %.1f MiB adjacency\n",
                (double)vertex_count * (double)words * 8.0 / 1048576.0);
        return 1;
    }
    uint64_t* token_uses = calloc((size_t)112 * words, sizeof(token_uses[0]));
    if (!token_uses) return 1;
    for (size_t vertex = 0; vertex < vertex_count; vertex++) {
        uint64_t lo = increments[vertex].lo;
        while (lo) {
            int token = __builtin_ctzll(lo);
            token_uses[(size_t)token * words + vertex / 64U] |=
                UINT64_C(1) << (vertex & 63U);
            lo &= lo - 1U;
        }
        uint64_t hi = increments[vertex].hi;
        while (hi) {
            int token = 64 + __builtin_ctzll(hi);
            token_uses[(size_t)token * words + vertex / 64U] |=
                UINT64_C(1) << (vertex & 63U);
            hi &= hi - 1U;
        }
    }
    double build_start = now_seconds();
    #pragma omp parallel for schedule(dynamic, 8)
    for (size_t i = 0; i < vertex_count; i++) {
        uint64_t* row = adjacency + i * words;
        memset(row, 0xff, words * sizeof(row[0]));
        uint64_t lo = increments[i].lo;
        while (lo) {
            int token = __builtin_ctzll(lo);
            const uint64_t* blocked = token_uses + (size_t)token * words;
            for (size_t w = 0; w < words; w++) row[w] &= ~blocked[w];
            lo &= lo - 1U;
        }
        uint64_t hi = increments[i].hi;
        while (hi) {
            int token = 64 + __builtin_ctzll(hi);
            const uint64_t* blocked = token_uses + (size_t)token * words;
            for (size_t w = 0; w < words; w++) row[w] &= ~blocked[w];
            hi &= hi - 1U;
        }
        row[i / 64U] &= ~(UINT64_C(1) << (i & 63U));
        if (vertex_count & 63U) {
            row[words - 1] &= (UINT64_C(1) << (vertex_count & 63U)) - 1U;
        }
    }
    double build_seconds = now_seconds() - build_start;
    printf("vertices=%zu words=%zu adjacency=%.1fMiB build=%.3fs threads=%d\n",
           vertex_count, words, (double)vertex_count * (double)words * 8.0 / 1048576.0,
           build_seconds, omp_get_max_threads());

    work_sets = calloc(29U * words, sizeof(work_sets[0]));
    allowed_sets = calloc(29U * words, sizeof(allowed_sets[0]));
    if (!work_sets || !allowed_sets) return 1;
    size_t root_end = root_end_arg < 0 || (size_t)root_end_arg > vertex_count
        ? vertex_count : (size_t)root_end_arg;
    double count_start = now_seconds();
    for (size_t vertex = 0; vertex < root_end && !stopped; vertex++) {
        uint64_t* root = work_sets;
        const uint64_t* neighbours = adjacency + vertex * words;
        memcpy(root, neighbours, words * sizeof(root[0]));
        for (size_t w = 0; w < vertex / 64U; w++) root[w] = 0;
        unsigned root_bit = (unsigned)(vertex & 63U);
        root[vertex / 64U] &= root_bit == 63U
            ? 0 : ~((UINT64_C(1) << (root_bit + 1U)) - 1U);
        U128 pivot_poly[9] = {1};
        pivoter(root, 0, 1, increments[vertex].weight, pivot_poly);
    }
    double count_seconds = now_seconds() - count_start;
    printf("k=%d roots=%zu nodes=%llu leaves=%llu stopped=%d count=",
           target_k, root_end, (unsigned long long)nodes_visited,
           (unsigned long long)leaves, stopped);
    print_u128(total);
    printf(" time=%.3fs\n", count_seconds);
    free(adjacency);
    free(token_uses);
    free(work_sets);
    free(allowed_sets);
    free(all);
    return stopped ? 3 : 0;
}
