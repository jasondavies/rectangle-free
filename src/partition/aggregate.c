#include "partition_poly_internal.h"
#ifdef RECT_RESIDUAL_CENSUS
#include "../../research/probes/partition_residual_census.h"
#endif

struct TerminalAggregator {
#if RECT_COUNT_K4
    int unused;
#else
    struct TerminalAggregateEntry* entries;
    size_t capacity;
    size_t mask;
    size_t count;
    size_t flush_at;
    ResultAccum* total;
#endif
};

#if !RECT_COUNT_K4
typedef struct TerminalAggregateEntry {
    uint64_t hash;
    Graph graph;
    Poly weight;
    uint8_t used;
} TerminalAggregateEntry;

static uint64_t terminal_graph_hash(const Graph* graph) {
    uint64_t hash = UINT64_C(0x9e3779b97f4a7c15) ^ graph->n;
    for (int i = 0; i < graph->n; i++) {
        uint64_t x = (uint64_t)graph->adj[i] + UINT64_C(0x9e3779b97f4a7c15);
        x ^= x >> 30;
        x *= UINT64_C(0xbf58476d1ce4e5b9);
        x ^= x >> 27;
        x *= UINT64_C(0x94d049bb133111eb);
        x ^= x >> 31;
        hash ^= x + (hash << 6) + (hash >> 2);
    }
    return hash;
}

static int terminal_graph_equal(const Graph* a, const Graph* b) {
    return a->n == b->n && a->vertex_mask == b->vertex_mask &&
           memcmp(a->adj, b->adj, (size_t)a->n * sizeof(a->adj[0])) == 0;
}
#endif

TerminalAggregator* terminal_aggregator_create(int bits, ResultAccum* total) {
    if (bits <= 0) return NULL;
    TerminalAggregator* aggregator = (TerminalAggregator*)calloc(1, sizeof(*aggregator));
    if (!aggregator) {
        fprintf(stderr, "Failed to allocate terminal aggregator\n");
        exit(1);
    }
#if RECT_COUNT_K4
    (void)total;
#else
    aggregator->capacity = (size_t)UINT64_C(1) << bits;
    aggregator->mask = aggregator->capacity - 1;
    aggregator->flush_at = aggregator->capacity * 3 / 4;
    aggregator->total = total;
    aggregator->entries =
        (TerminalAggregateEntry*)calloc(aggregator->capacity, sizeof(aggregator->entries[0]));
    if (!aggregator->entries) {
        fprintf(stderr, "Failed to allocate terminal aggregator entries\n");
        exit(1);
    }
#endif
    return aggregator;
}

void terminal_aggregator_flush(TerminalAggregator* aggregator,
                               RowGraphCache* cache, RowGraphCache* raw_cache,
                               GraphCanonWorkspace* ws, long long* local_canon_calls,
                               long long* local_cache_hits, long long* local_raw_cache_hits,
                               ProfileStats* profile) {
    if (!aggregator) return;
#if RECT_COUNT_K4
    (void)cache;
    (void)raw_cache;
    (void)ws;
    (void)local_canon_calls;
    (void)local_cache_hits;
    (void)local_raw_cache_hits;
    (void)profile;
#else
    if (aggregator->count == 0) return;
#ifdef RECT_RESIDUAL_CENSUS
    residual_census_begin();
#endif
    double t0 = (PROFILE_BUILD && profile) ? omp_get_wtime() : 0.0;
    for (size_t i = 0; i < aggregator->capacity; i++) {
        TerminalAggregateEntry* entry = &aggregator->entries[i];
        if (!entry->used) continue;
        GraphResult graph_result;
        ResultAccum contribution;
        solve_graph_poly(&entry->graph, cache, raw_cache, ws,
                         local_canon_calls, local_cache_hits, local_raw_cache_hits,
                         profile, &graph_result);
        poly_mul_graph_ref(&entry->weight, &graph_result, &contribution);
#ifdef RECT_RESIDUAL_CENSUS
        residual_census_visit(&entry->graph, &entry->weight, &contribution, ws);
#endif
        poly_accumulate_checked(aggregator->total, &contribution);
        entry->used = 0;
    }
    aggregator->count = 0;
#ifdef RECT_RESIDUAL_CENSUS
    residual_census_end(cache, raw_cache, ws);
#endif
    if (PROFILE_BUILD && profile) {
        profile->terminal_aggregate_flushes++;
        profile->terminal_aggregate_time += omp_get_wtime() - t0;
    }
#endif
}

int terminal_aggregator_defer(TerminalAggregator* aggregator, const Graph* graph,
                              const Poly* weight, RowGraphCache* cache,
                              RowGraphCache* raw_cache, GraphCanonWorkspace* ws,
                              long long* local_canon_calls, long long* local_cache_hits,
                              long long* local_raw_cache_hits, ProfileStats* profile) {
    if (!aggregator) return 0;
#if RECT_COUNT_K4
    (void)graph;
    (void)weight;
    (void)cache;
    (void)raw_cache;
    (void)ws;
    (void)local_canon_calls;
    (void)local_cache_hits;
    (void)local_raw_cache_hits;
    (void)profile;
    return 0;
#else
    if (PROFILE_BUILD && profile) profile->terminal_aggregate_inputs++;
    uint64_t hash = terminal_graph_hash(graph);
    size_t slot = (size_t)hash & aggregator->mask;
    for (;;) {
        TerminalAggregateEntry* entry = &aggregator->entries[slot];
        if (!entry->used) break;
        if (entry->hash == hash && terminal_graph_equal(&entry->graph, graph)) {
            poly_accumulate_checked(&entry->weight, weight);
            if (PROFILE_BUILD && profile) profile->terminal_aggregate_hits++;
            return 1;
        }
        slot = (slot + 1) & aggregator->mask;
    }

    if (aggregator->count >= aggregator->flush_at) {
        terminal_aggregator_flush(aggregator, cache, raw_cache, ws,
                                  local_canon_calls, local_cache_hits,
                                  local_raw_cache_hits, profile);
        slot = (size_t)hash & aggregator->mask;
        while (aggregator->entries[slot].used) slot = (slot + 1) & aggregator->mask;
    }

    TerminalAggregateEntry* entry = &aggregator->entries[slot];
    entry->used = 1;
    entry->hash = hash;
    entry->graph.n = graph->n;
    entry->graph.vertex_mask = graph->vertex_mask;
    memcpy(entry->graph.adj, graph->adj, (size_t)graph->n * sizeof(graph->adj[0]));
    entry->weight.deg = weight->deg;
    memcpy(entry->weight.coeffs, weight->coeffs,
           ((size_t)weight->deg + 1U) * sizeof(weight->coeffs[0]));
    aggregator->count++;
    if (PROFILE_BUILD && profile) profile->terminal_aggregate_unique++;
    return 1;
#endif
}

void terminal_aggregator_destroy(TerminalAggregator* aggregator) {
    if (!aggregator) return;
#if !RECT_COUNT_K4
    free(aggregator->entries);
#endif
    free(aggregator);
}
