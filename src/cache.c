#include "partition_poly.h"

// --- GRAPH CACHE HELPERS ---

static inline uint64_t* graph_cache_sig_slot(const GraphCache* cache, int slot) {
    return cache->sigs + (size_t)slot * GRAPH_SIG_WORDS;
}

static inline AdjWord* row_graph_cache_row_slot(const RowGraphCache* cache, int slot) {
    return cache->rows + (size_t)slot * MAXN_NAUTY;
}

static inline void graph_pack_signature(const Graph* g, uint32_t key_n, uint64_t* out) {
    memset(out, 0, (size_t)GRAPH_SIG_WORDS * sizeof(*out));
    int bit = 0;
    for (uint32_t j = 1; j < key_n; j++) {
        for (uint32_t i = 0; i < j; i++, bit++) {
            if ((g->adj[i] >> j) & 1ULL) {
                out[(unsigned)bit >> 6] |= 1ULL << (bit & 63);
            }
        }
    }
}

static inline PolyCoeff* graph_cache_coeff_slot(const GraphCache* cache, int slot) {
    return cache->coeffs + (size_t)slot * (size_t)cache->poly_len;
}

static inline PolyCoeff* row_graph_cache_coeff_slot(const RowGraphCache* cache, int slot) {
    return cache->coeffs + (size_t)slot * (size_t)cache->poly_len;
}

static inline uint32_t graph_cache_next_stamp(GraphCache* cache) {
    cache->next_stamp++;
    if (cache->next_stamp == 0) cache->next_stamp = 1;
    return cache->next_stamp;
}

static inline void graph_cache_touch_slot(GraphCache* cache, int slot) {
    cache->stamps[slot] = graph_cache_next_stamp(cache);
}

static inline uint32_t row_graph_cache_next_stamp(RowGraphCache* cache) {
    cache->next_stamp++;
    if (cache->next_stamp == 0) cache->next_stamp = 1;
    return cache->next_stamp;
}

static inline void row_graph_cache_touch_slot(RowGraphCache* cache, int slot) {
    cache->stamps[slot] = row_graph_cache_next_stamp(cache);
}

static inline int graph_cache_slot_matches_sig(const GraphCache* cache, int slot, uint64_t key_hash,
                                               uint32_t key_n, const uint64_t* sig) {
    if (!cache->keys[slot].used || cache->keys[slot].key_hash != key_hash ||
        cache->keys[slot].key_n != key_n) {
        return 0;
    }
    const uint64_t* slot_sig = graph_cache_sig_slot(cache, slot);
    return memcmp(slot_sig, sig, sizeof(uint64_t) * GRAPH_SIG_WORDS) == 0;
}

static inline int row_graph_cache_slot_matches_graph(const RowGraphCache* cache, int slot,
                                                     uint64_t key_hash, uint32_t key_n,
                                                     const Graph* g, AdjWord row_mask) {
    if (!cache->keys[slot].used || cache->keys[slot].key_hash != key_hash ||
        cache->keys[slot].key_n != key_n) {
        return 0;
    }
    const AdjWord* slot_rows = row_graph_cache_row_slot(cache, slot);
    if (row_mask == (AdjWord)ADJWORD_MASK) {
        return memcmp(slot_rows, g->adj, (size_t)key_n * sizeof(AdjWord)) == 0;
    }
    for (uint32_t i = 0; i < key_n; i++) {
        if (slot_rows[i] != (g->adj[i] & row_mask)) return 0;
    }
    return 1;
}

static inline int row_graph_cache_slot_matches_rows(const RowGraphCache* cache, int slot,
                                                    uint64_t key_hash, uint32_t key_n,
                                                    const AdjWord* rows) {
    if (!cache->keys[slot].used || cache->keys[slot].key_hash != key_hash ||
        cache->keys[slot].key_n != key_n) {
        return 0;
    }
    const AdjWord* slot_rows = row_graph_cache_row_slot(cache, slot);
    return memcmp(slot_rows, rows, (size_t)key_n * sizeof(AdjWord)) == 0;
}

static inline void graph_cache_load_poly(const GraphCache* cache, int slot, GraphPoly* value) {
#if RECT_COUNT_K4
    value->x_pow = cache->x_pows[slot];
    int deg = cache->degs[slot];
    value->deg = (uint8_t)deg;
    memcpy(value->coeffs, graph_cache_coeff_slot(cache, slot),
           (size_t)(deg + 1) * sizeof(value->coeffs[0]));
#else
    uint32_t key_n = cache->keys[slot].key_n;
    value->x_pow = key_n == 0 ? 0 : 1;
    value->deg = key_n == 0 ? 0 : (uint8_t)(key_n - 1);
    memcpy(value->coeffs, graph_cache_coeff_slot(cache, slot),
           (size_t)(key_n == 0 ? 1 : key_n) * sizeof(value->coeffs[0]));
#endif
}

static inline void row_graph_cache_load_poly(const RowGraphCache* cache, int slot, GraphPoly* value) {
#if RECT_COUNT_K4
    value->x_pow = cache->x_pows[slot];
    int deg = cache->degs[slot];
    value->deg = (uint8_t)deg;
    memcpy(value->coeffs, row_graph_cache_coeff_slot(cache, slot),
           (size_t)(deg + 1) * sizeof(value->coeffs[0]));
#else
    uint32_t key_n = cache->keys[slot].key_n;
    value->x_pow = key_n == 0 ? 0 : 1;
    value->deg = key_n == 0 ? 0 : (uint8_t)(key_n - 1);
    memcpy(value->coeffs, row_graph_cache_coeff_slot(cache, slot),
           (size_t)(key_n == 0 ? 1 : key_n) * sizeof(value->coeffs[0]));
#endif
}

static int graph_cache_lookup_poly(GraphCache* cache, uint64_t key_hash, uint32_t key_n,
                                   const Graph* g, uint64_t row_mask, GraphPoly* value, int touch) {
    (void)row_mask;
    uint64_t sig[GRAPH_SIG_WORDS];
    graph_pack_signature(g, key_n, sig);
    int cache_idx = (int)(key_hash & (uint64_t)cache->mask);
    for (int k = 0; k < cache->probe; k++) {
        int p = (cache_idx + k) & cache->mask;
        if (graph_cache_slot_matches_sig(cache, p, key_hash, key_n, sig)) {
            graph_cache_load_poly(cache, p, value);
            if (touch) graph_cache_touch_slot(cache, p);
            return 1;
        }
    }
    return 0;
}

int row_graph_cache_lookup_poly(RowGraphCache* cache, uint64_t key_hash, uint32_t key_n,
                                const Graph* g, AdjWord row_mask, GraphPoly* value,
                                int touch) {
    int cache_idx = (int)(key_hash & (uint64_t)cache->mask);
    for (int k = 0; k < cache->probe; k++) {
        int p = (cache_idx + k) & cache->mask;
        if (row_graph_cache_slot_matches_graph(cache, p, key_hash, key_n, g, row_mask)) {
            row_graph_cache_load_poly(cache, p, value);
            if (touch) row_graph_cache_touch_slot(cache, p);
            return 1;
        }
    }
    return 0;
}

int row_graph_cache_lookup_rows(RowGraphCache* cache, uint64_t key_hash, uint32_t key_n,
                                const AdjWord* rows, GraphPoly* value, int touch) {
    int cache_idx = (int)(key_hash & (uint64_t)cache->mask);
    for (int k = 0; k < cache->probe; k++) {
        int p = (cache_idx + k) & cache->mask;
        if (row_graph_cache_slot_matches_rows(cache, p, key_hash, key_n, rows)) {
            row_graph_cache_load_poly(cache, p, value);
            if (touch) row_graph_cache_touch_slot(cache, p);
            return 1;
        }
    }
    return 0;
}

void shared_graph_cache_init(SharedGraphCache* shared, int bits, int poly_len) {
    size_t size = (size_t)1 << bits;
    memset(shared, 0, sizeof(*shared));
    pthread_rwlock_init(&shared->lock, NULL);
    shared->cache.mask = (int)size - 1;
    shared->cache.probe = CACHE_PROBE;
    shared->cache.poly_len = poly_len;
    shared->cache.keys = checked_aligned_alloc(64, sizeof(CacheKey) * size, "shared_cache_keys");
    shared->cache.stamps = checked_aligned_alloc(64, sizeof(uint32_t) * size, "shared_cache_stamps");
    shared->cache.sigs = checked_aligned_alloc(64, sizeof(uint64_t) * size * GRAPH_SIG_WORDS, "shared_cache_sigs");
#if RECT_COUNT_K4
    shared->cache.x_pows = checked_aligned_alloc(64, sizeof(uint8_t) * size, "shared_cache_x_pows");
    shared->cache.degs = checked_aligned_alloc(64, sizeof(uint8_t) * size, "shared_cache_degs");
#endif
    shared->cache.coeffs =
        checked_aligned_alloc(64, sizeof(PolyCoeff) * size * (size_t)poly_len, "shared_cache_coeffs");
    memset(shared->cache.keys, 0, sizeof(CacheKey) * size);
    memset(shared->cache.stamps, 0, sizeof(uint32_t) * size);
    shared->cache.next_stamp = 0;
    shared->enabled = 1;
}

void shared_graph_cache_free(SharedGraphCache* shared) {
    if (!shared) return;
    free(shared->cache.keys);
    free(shared->cache.stamps);
    free(shared->cache.sigs);
#if RECT_COUNT_K4
    free(shared->cache.x_pows);
    free(shared->cache.degs);
#endif
    free(shared->cache.coeffs);
    pthread_rwlock_destroy(&shared->lock);
    memset(shared, 0, sizeof(*shared));
}

int shared_graph_cache_lookup_poly(SharedGraphCache* shared, uint64_t key_hash, uint32_t key_n,
                                   const Graph* g, uint64_t row_mask, GraphPoly* value) {
    if (!shared || !shared->enabled) return 0;
    int found = 0;
    pthread_rwlock_rdlock(&shared->lock);
    found = graph_cache_lookup_poly(&shared->cache, key_hash, key_n, g, row_mask, value, 0);
    pthread_rwlock_unlock(&shared->lock);
    return found;
}

static void store_graph_cache_entry(GraphCache* cache, uint64_t key_hash, uint32_t key_n, const Graph* g,
                                    uint64_t row_mask, const GraphPoly* value) {
    (void)row_mask;
    uint64_t sig[GRAPH_SIG_WORDS];
    graph_pack_signature(g, key_n, sig);
    int cache_idx = (int)(key_hash & (uint64_t)cache->mask);
    int empty_slot = -1;
    int oldest_same_n_slot = -1;
    int oldest_other_n_slot = -1;
    uint32_t oldest_same_n_stamp = UINT32_MAX;
    uint32_t oldest_other_n_stamp = UINT32_MAX;
    for (int k = 0; k < cache->probe; k++) {
        int p = (cache_idx + k) & cache->mask;
        if (graph_cache_slot_matches_sig(cache, p, key_hash, key_n, sig)) {
            empty_slot = p;
            break;
        }
        if (!cache->keys[p].used) {
            if (empty_slot < 0) empty_slot = p;
            continue;
        }

        uint32_t stamp = cache->stamps[p];
        if (cache->keys[p].key_n != key_n) {
            if (stamp < oldest_other_n_stamp) {
                oldest_other_n_stamp = stamp;
                oldest_other_n_slot = p;
            }
        } else if (stamp < oldest_same_n_stamp) {
            oldest_same_n_stamp = stamp;
            oldest_same_n_slot = p;
        }
    }
    int best_slot = empty_slot;
    if (best_slot < 0) {
        best_slot = (oldest_other_n_slot >= 0) ? oldest_other_n_slot : oldest_same_n_slot;
    }
    if (best_slot < 0) best_slot = cache_idx;
    cache->keys[best_slot].key_hash = key_hash;
    cache->keys[best_slot].key_n = key_n;
    memcpy(graph_cache_sig_slot(cache, best_slot), sig, sizeof(sig));
#if RECT_COUNT_K4
    cache->x_pows[best_slot] = value->x_pow;
    cache->degs[best_slot] = (uint8_t)value->deg;
#endif
    memcpy(graph_cache_coeff_slot(cache, best_slot), value->coeffs,
           (size_t)(
#if RECT_COUNT_K4
               value->deg + 1
#else
               key_n == 0 ? 1 : key_n
#endif
               ) * sizeof(value->coeffs[0]));
    cache->keys[best_slot].used = 1;
    graph_cache_touch_slot(cache, best_slot);
}

void shared_graph_cache_flush_exports(void) {
    SharedGraphCacheExporter* exporter = tls_shared_cache_exporter;
    SharedGraphCache* shared = g_shared_graph_cache;
    if (!shared || !shared->enabled || !exporter || exporter->count <= 0) return;
    pthread_rwlock_wrlock(&shared->lock);
    for (int i = 0; i < exporter->count; i++) {
        SharedGraphCacheExportEntry* e = &exporter->entries[i];
        store_graph_cache_entry(&shared->cache, e->key_hash, e->key_n, &e->g, e->row_mask, &e->value);
    }
    pthread_rwlock_unlock(&shared->lock);
    exporter->count = 0;
}

void shared_graph_cache_export(uint64_t key_hash, uint32_t key_n, const Graph* g,
                               uint64_t row_mask, const GraphPoly* value) {
    SharedGraphCacheExporter* exporter = tls_shared_cache_exporter;
    SharedGraphCache* shared = g_shared_graph_cache;
    if (!shared || !shared->enabled || !exporter) return;
    if (exporter->count >= SHARED_CACHE_EXPORT_CAP) {
        shared_graph_cache_flush_exports();
    }
    SharedGraphCacheExportEntry* e = &exporter->entries[exporter->count++];
    e->key_hash = key_hash;
    e->key_n = key_n;
    e->g = *g;
    e->row_mask = row_mask;
    e->value = *value;
}

void store_row_graph_cache_entry(RowGraphCache* cache, uint64_t key_hash, uint32_t key_n,
                                 const Graph* g, AdjWord row_mask,
                                 const GraphPoly* value) {
    int cache_idx = (int)(key_hash & (uint64_t)cache->mask);
    int empty_slot = -1;
    int oldest_same_n_slot = -1;
    int oldest_other_n_slot = -1;
    uint32_t oldest_same_n_stamp = UINT32_MAX;
    uint32_t oldest_other_n_stamp = UINT32_MAX;
    for (int k = 0; k < cache->probe; k++) {
        int p = (cache_idx + k) & cache->mask;
        if (row_graph_cache_slot_matches_graph(cache, p, key_hash, key_n, g, row_mask)) {
            empty_slot = p;
            break;
        }
        if (!cache->keys[p].used) {
            if (empty_slot < 0) empty_slot = p;
            continue;
        }

        uint32_t stamp = cache->stamps[p];
        if (cache->keys[p].key_n != key_n) {
            if (stamp < oldest_other_n_stamp) {
                oldest_other_n_stamp = stamp;
                oldest_other_n_slot = p;
            }
        } else if (stamp < oldest_same_n_stamp) {
            oldest_same_n_stamp = stamp;
            oldest_same_n_slot = p;
        }
    }
    int best_slot = empty_slot;
    if (best_slot < 0) {
        best_slot = (oldest_other_n_slot >= 0) ? oldest_other_n_slot : oldest_same_n_slot;
    }
    if (best_slot < 0) best_slot = cache_idx;
    cache->keys[best_slot].key_hash = key_hash;
    cache->keys[best_slot].key_n = key_n;
    AdjWord* slot_rows = row_graph_cache_row_slot(cache, best_slot);
    if (row_mask == (AdjWord)ADJWORD_MASK) {
        memcpy(slot_rows, g->adj, (size_t)key_n * sizeof(AdjWord));
    } else {
        for (uint32_t i = 0; i < key_n; i++) {
            slot_rows[i] = g->adj[i] & row_mask;
        }
    }
#if RECT_COUNT_K4
    cache->x_pows[best_slot] = value->x_pow;
    cache->degs[best_slot] = (uint8_t)value->deg;
#endif
    memcpy(row_graph_cache_coeff_slot(cache, best_slot), value->coeffs,
           (size_t)(
#if RECT_COUNT_K4
               value->deg + 1
#else
               key_n == 0 ? 1 : key_n
#endif
               ) * sizeof(value->coeffs[0]));
    cache->keys[best_slot].used = 1;
    row_graph_cache_touch_slot(cache, best_slot);
}

void store_row_graph_cache_entry_rows(RowGraphCache* cache, uint64_t key_hash, uint32_t key_n,
                                      const AdjWord* rows, const GraphPoly* value) {
    int cache_idx = (int)(key_hash & (uint64_t)cache->mask);
    int empty_slot = -1;
    int oldest_same_n_slot = -1;
    int oldest_other_n_slot = -1;
    uint32_t oldest_same_n_stamp = UINT32_MAX;
    uint32_t oldest_other_n_stamp = UINT32_MAX;
    for (int k = 0; k < cache->probe; k++) {
        int p = (cache_idx + k) & cache->mask;
        if (row_graph_cache_slot_matches_rows(cache, p, key_hash, key_n, rows)) {
            empty_slot = p;
            break;
        }
        if (!cache->keys[p].used) {
            if (empty_slot < 0) empty_slot = p;
            continue;
        }

        uint32_t stamp = cache->stamps[p];
        if (cache->keys[p].key_n != key_n) {
            if (stamp < oldest_other_n_stamp) {
                oldest_other_n_stamp = stamp;
                oldest_other_n_slot = p;
            }
        } else if (stamp < oldest_same_n_stamp) {
            oldest_same_n_stamp = stamp;
            oldest_same_n_slot = p;
        }
    }
    int best_slot = empty_slot;
    if (best_slot < 0) {
        best_slot = (oldest_other_n_slot >= 0) ? oldest_other_n_slot : oldest_same_n_slot;
    }
    if (best_slot < 0) best_slot = cache_idx;
    cache->keys[best_slot].key_hash = key_hash;
    cache->keys[best_slot].key_n = key_n;
    AdjWord* slot_rows = row_graph_cache_row_slot(cache, best_slot);
#if RECT_COUNT_K4
    memcpy(slot_rows, rows, (size_t)key_n * sizeof(AdjWord));
    cache->x_pows[best_slot] = value->x_pow;
    cache->degs[best_slot] = (uint8_t)value->deg;
#else
    memcpy(slot_rows, rows, (size_t)key_n * sizeof(AdjWord));
#endif
    memcpy(row_graph_cache_coeff_slot(cache, best_slot), value->coeffs,
           (size_t)(
#if RECT_COUNT_K4
               value->deg + 1
#else
               key_n == 0 ? 1 : key_n
#endif
               ) * sizeof(value->coeffs[0]));
    cache->keys[best_slot].used = 1;
    row_graph_cache_touch_slot(cache, best_slot);
}
