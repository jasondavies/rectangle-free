#include "partition_poly.h"

// --- GRAPH CACHE HELPERS ---

static inline AdjWord* row_graph_cache_row_slot(const RowGraphCache* cache, int slot) {
    return cache->rows + (size_t)slot * MAXN_NAUTY;
}

static inline GraphCacheValue* row_graph_cache_coeff_slot(const RowGraphCache* cache, int slot) {
#if RECT_COUNT_K4
    return cache->coeffs + (size_t)slot;
#else
    return cache->coeffs + (size_t)slot * (size_t)cache->poly_len;
#endif
}

static inline uint32_t row_graph_cache_next_stamp(RowGraphCache* cache) {
    cache->next_stamp++;
    if (cache->next_stamp == 0) cache->next_stamp = 1;
    return cache->next_stamp;
}

static inline void row_graph_cache_touch_slot(RowGraphCache* cache, int slot) {
    cache->stamps[slot] = row_graph_cache_next_stamp(cache);
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

static inline void row_graph_cache_load_poly(const RowGraphCache* cache, int slot, GraphResult* value) {
#if RECT_COUNT_K4
    *value = *row_graph_cache_coeff_slot(cache, slot);
#else
    uint32_t key_n = cache->keys[slot].key_n;
    value->x_pow = key_n == 0 ? 0 : 1;
    value->deg = key_n == 0 ? 0 : (uint8_t)(key_n - 1);
    memcpy(value->coeffs, row_graph_cache_coeff_slot(cache, slot),
           (size_t)(key_n == 0 ? 1 : key_n) * sizeof(value->coeffs[0]));
#endif
}

int row_graph_cache_lookup_poly(RowGraphCache* cache, uint64_t key_hash, uint32_t key_n,
                                const Graph* g, AdjWord row_mask, GraphResult* value,
                                int touch) {
    int cache_idx = (int)(key_hash & (uint64_t)cache->mask);
    for (int k = 0; k < cache->probe; k++) {
        int p = (cache_idx + k) & cache->mask;
        if (!cache->keys[p].used) {
            return 0;
        }
        if (row_graph_cache_slot_matches_graph(cache, p, key_hash, key_n, g, row_mask)) {
            row_graph_cache_load_poly(cache, p, value);
            if (touch) row_graph_cache_touch_slot(cache, p);
            return 1;
        }
    }
    return 0;
}

int row_graph_cache_lookup_rows(RowGraphCache* cache, uint64_t key_hash, uint32_t key_n,
                                const AdjWord* rows, GraphResult* value, int touch) {
    int cache_idx = (int)(key_hash & (uint64_t)cache->mask);
    for (int k = 0; k < cache->probe; k++) {
        int p = (cache_idx + k) & cache->mask;
        if (!cache->keys[p].used) {
            return 0;
        }
        if (row_graph_cache_slot_matches_rows(cache, p, key_hash, key_n, rows)) {
            row_graph_cache_load_poly(cache, p, value);
            if (touch) row_graph_cache_touch_slot(cache, p);
            return 1;
        }
    }
    return 0;
}

void store_row_graph_cache_entry(RowGraphCache* cache, uint64_t key_hash, uint32_t key_n,
                                 const Graph* g, AdjWord row_mask,
                                 const GraphResult* value) {
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
    *row_graph_cache_coeff_slot(cache, best_slot) = *value;
#else
    memcpy(row_graph_cache_coeff_slot(cache, best_slot), value->coeffs,
           (size_t)(key_n == 0 ? 1 : key_n) * sizeof(value->coeffs[0]));
#endif
    cache->keys[best_slot].used = 1;
    row_graph_cache_touch_slot(cache, best_slot);
}

void store_row_graph_cache_entry_rows(RowGraphCache* cache, uint64_t key_hash, uint32_t key_n,
                                      const AdjWord* rows, const GraphResult* value) {
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
    memcpy(slot_rows, rows, (size_t)key_n * sizeof(AdjWord));
#if RECT_COUNT_K4
    *row_graph_cache_coeff_slot(cache, best_slot) = *value;
#else
    memcpy(row_graph_cache_coeff_slot(cache, best_slot), value->coeffs,
           (size_t)(key_n == 0 ? 1 : key_n) * sizeof(value->coeffs[0]));
#endif
    cache->keys[best_slot].used = 1;
    row_graph_cache_touch_slot(cache, best_slot);
}
