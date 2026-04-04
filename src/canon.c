#include "partition_poly_internal.h"

// --- SYMMETRY LOGIC ---

#define REP_ORBIT_MARK_WORDS ((CANON_PARTITION_ID_LIMIT + 63u) / 64u)

static inline uint16_t* canon_state_changed_first_greater_idx_row(CanonState* st, int depth) {
    return st->changed_first_greater_idx + (size_t)depth * (size_t)st->limit;
}

static inline uint8_t* canon_state_changed_first_greater_old_idx_row(CanonState* st, int depth) {
    return st->changed_first_greater_old_idx + (size_t)depth * (size_t)st->limit;
}

static inline uint16_t* canon_state_changed_first_greater_old_val_row(CanonState* st, int depth) {
    return st->changed_first_greater_old_val + (size_t)depth * (size_t)st->limit;
}

static inline uint16_t* canon_state_equal_perm_row(CanonState* st, int depth) {
    return st->equal_perm + (size_t)depth * (size_t)st->limit;
}

static inline const uint16_t* canon_state_equal_perm_row_const(const CanonState* st, int depth) {
    return st->equal_perm + (size_t)depth * (size_t)st->limit;
}

static void graph_apply_permutation_dense_rows(uint32_t n, const AdjWord* dense_rows,
                                               const uint8_t* new_index_of_old, Graph* dst);
static inline void graph_cache_load_poly(const GraphCache* cache, int slot, GraphPoly* value);
static inline void row_graph_cache_load_poly(const RowGraphCache* cache, int slot, GraphPoly* value);
void store_graph_cache_entry(GraphCache* cache, uint64_t key_hash, uint32_t key_n, const Graph* g,
                             uint64_t row_mask, const GraphPoly* value);

static void solve_structure_with_row_orbit(const Graph* partial_graph, long long row_orbit,
                                           RowGraphCache* cache, RowGraphCache* raw_cache,
                                           NautyWorkspace* ws, long long* local_canon_calls,
                                           long long* local_cache_hits,
                                           long long* local_raw_cache_hits,
                                           const WeightAccum* weight_prod, long long mult_coeff,
                                           ProfileStats* profile, Poly* out_result);
static inline int row_insert_sorted(uint16_t* row, int len, uint16_t val) {
    switch (len) {
        case 0:
            row[0] = val;
            return 0;
        case 1:
            if (row[0] > val) {
                row[1] = row[0];
                row[0] = val;
                return 0;
            }
            row[1] = val;
            return 1;
        case 2:
            if (row[1] > val) {
                row[2] = row[1];
                if (row[0] > val) {
                    row[1] = row[0];
                    row[0] = val;
                    return 0;
                }
                row[1] = val;
                return 1;
            }
            row[2] = val;
            return 2;
        case 3:
            if (row[2] > val) {
                row[3] = row[2];
                if (row[1] > val) {
                    row[2] = row[1];
                    if (row[0] > val) {
                        row[1] = row[0];
                        row[0] = val;
                        return 0;
                    }
                    row[1] = val;
                    return 1;
                }
                row[2] = val;
                return 2;
            }
            row[3] = val;
            return 3;
        case 4:
            if (row[3] > val) {
                row[4] = row[3];
                if (row[2] > val) {
                    row[3] = row[2];
                    if (row[1] > val) {
                        row[2] = row[1];
                        if (row[0] > val) {
                            row[1] = row[0];
                            row[0] = val;
                            return 0;
                        }
                        row[1] = val;
                        return 1;
                    }
                    row[2] = val;
                    return 2;
                }
                row[3] = val;
                return 3;
            }
            row[4] = val;
            return 4;
        case 5:
            if (row[4] > val) {
                row[5] = row[4];
                if (row[3] > val) {
                    row[4] = row[3];
                    if (row[2] > val) {
                        row[3] = row[2];
                        if (row[1] > val) {
                            row[2] = row[1];
                            if (row[0] > val) {
                                row[1] = row[0];
                                row[0] = val;
                                return 0;
                            }
                            row[1] = val;
                            return 1;
                        }
                        row[2] = val;
                        return 2;
                    }
                    row[3] = val;
                    return 3;
                }
                row[4] = val;
                return 4;
            }
            row[5] = val;
            return 5;
        default: {
            int j = len;
            while (j > 0 && row[j - 1] > val) j--;
            for (int k = len; k > j; k--) row[k] = row[k - 1];
            row[j] = val;
            return j;
        }
    }
}

static inline void sort5_u16(uint16_t* a, uint16_t* b, uint16_t* c, uint16_t* d, uint16_t* e) {
#define SORT_SWAP_U16(x, y) \
    do { \
        if (*(x) > *(y)) { \
            uint16_t tmp = *(x); \
            *(x) = *(y); \
            *(y) = tmp; \
        } \
    } while (0)
    SORT_SWAP_U16(a, b);
    SORT_SWAP_U16(d, e);
    SORT_SWAP_U16(a, c);
    SORT_SWAP_U16(b, c);
    SORT_SWAP_U16(a, d);
    SORT_SWAP_U16(c, d);
    SORT_SWAP_U16(b, e);
    SORT_SWAP_U16(b, c);
    SORT_SWAP_U16(d, e);
    SORT_SWAP_U16(c, d);
#undef SORT_SWAP_U16
}

static inline void sort4_u16(uint16_t* a, uint16_t* b, uint16_t* c, uint16_t* d) {
#define SORT_SWAP_U16(x, y) \
    do { \
        if (*(x) > *(y)) { \
            uint16_t tmp = *(x); \
            *(x) = *(y); \
            *(y) = tmp; \
        } \
    } while (0)
    SORT_SWAP_U16(a, b);
    SORT_SWAP_U16(c, d);
    SORT_SWAP_U16(a, c);
    SORT_SWAP_U16(b, d);
    SORT_SWAP_U16(b, c);
#undef SORT_SWAP_U16
}

static inline int canon_rebuild_equal_case(const CanonState* st, int p, int g, uint16_t pid,
                                           uint8_t* next_first_greater, uint16_t* next_first_greater_val) {
    int depth = st->depth;
    int new_depth = depth + 1;

    if (depth == 3 && g == 2) {
        uint16_t a = st->stack_perm_rows[0][p];
        uint16_t b = st->stack_perm_rows[1][p];
        uint16_t c = st->stack_perm_rows[2][p];
        uint16_t last = a > b ? a : b;
        if (c > last) last = c;
        if (last < pid) return 0;
        if (last > pid) {
            *next_first_greater = 3;
            *next_first_greater_val = last;
        } else {
            *next_first_greater = (uint8_t)new_depth;
            *next_first_greater_val = 0;
        }
        return 1;
    }

    if (depth == 4 && g == 3) {
        uint16_t a = st->stack_perm_rows[0][p];
        uint16_t b = st->stack_perm_rows[1][p];
        uint16_t c = st->stack_perm_rows[2][p];
        uint16_t d = st->stack_perm_rows[3][p];
        uint16_t last = a > b ? a : b;
        if (c > last) last = c;
        if (d > last) last = d;
        if (last < pid) return 0;
        if (last > pid) {
            *next_first_greater = 4;
            *next_first_greater_val = last;
        } else {
            *next_first_greater = (uint8_t)new_depth;
            *next_first_greater_val = 0;
        }
        return 1;
    }

    if (depth == 4 && g == 2) {
        uint16_t a = st->stack_perm_rows[0][p];
        uint16_t b = st->stack_perm_rows[1][p];
        uint16_t c = st->stack_perm_rows[2][p];
        uint16_t d = st->stack_perm_rows[3][p];
        sort4_u16(&a, &b, &c, &d);

        if (c < st->stack_vals[3]) return 0;
        if (c > st->stack_vals[3]) {
            *next_first_greater = 3;
            *next_first_greater_val = c;
            return 1;
        }
        if (d < pid) return 0;
        if (d > pid) {
            *next_first_greater = 4;
            *next_first_greater_val = d;
        } else {
            *next_first_greater = (uint8_t)new_depth;
            *next_first_greater_val = 0;
        }
        return 1;
    }

    if (depth == 5 && g == 2) {
        uint16_t a = st->stack_perm_rows[0][p];
        uint16_t b = st->stack_perm_rows[1][p];
        uint16_t c = st->stack_perm_rows[2][p];
        uint16_t d = st->stack_perm_rows[3][p];
        uint16_t e = st->stack_perm_rows[4][p];
        sort5_u16(&a, &b, &c, &d, &e);

        if (c < st->stack_vals[3]) return 0;
        if (c > st->stack_vals[3]) {
            *next_first_greater = 3;
            *next_first_greater_val = c;
            return 1;
        }
        if (d < st->stack_vals[4]) return 0;
        if (d > st->stack_vals[4]) {
            *next_first_greater = 4;
            *next_first_greater_val = d;
            return 1;
        }
        if (e < pid) return 0;
        if (e > pid) {
            *next_first_greater = 5;
            *next_first_greater_val = e;
        } else {
            *next_first_greater = (uint8_t)new_depth;
            *next_first_greater_val = 0;
        }
        return 1;
    }

    if (depth == 5 && g == 4) {
        uint16_t a = st->stack_perm_rows[0][p];
        uint16_t b = st->stack_perm_rows[1][p];
        uint16_t c = st->stack_perm_rows[2][p];
        uint16_t d = st->stack_perm_rows[3][p];
        uint16_t e = st->stack_perm_rows[4][p];
        uint16_t last = a > b ? a : b;
        if (c > last) last = c;
        if (d > last) last = d;
        if (e > last) last = e;
        if (last < pid) return 0;
        if (last > pid) {
            *next_first_greater = 5;
            *next_first_greater_val = last;
        } else {
            *next_first_greater = (uint8_t)new_depth;
            *next_first_greater_val = 0;
        }
        return 1;
    }

    if (depth == 5 && g == 3) {
        uint16_t a = st->stack_perm_rows[0][p];
        uint16_t b = st->stack_perm_rows[1][p];
        uint16_t c = st->stack_perm_rows[2][p];
        uint16_t d = st->stack_perm_rows[3][p];
        uint16_t e = st->stack_perm_rows[4][p];
        sort5_u16(&a, &b, &c, &d, &e);

        if (d < st->stack_vals[4]) return 0;
        if (d > st->stack_vals[4]) {
            *next_first_greater = 4;
            *next_first_greater_val = d;
            return 1;
        }
        if (e < pid) return 0;
        if (e > pid) {
            *next_first_greater = 5;
            *next_first_greater_val = e;
        } else {
            *next_first_greater = (uint8_t)new_depth;
            *next_first_greater_val = 0;
        }
        return 1;
    }

    uint16_t row[MAX_COLS];
    int len = 0;

    for (int t = 0; t < depth; t++) {
        row_insert_sorted(row, len, st->stack_perm_rows[t][p]);
        len++;
    }

    for (int k = g + 1; k < depth; k++) {
        uint16_t rv = row[k - 1];
        uint16_t cv = st->stack_vals[k];
        if (rv < cv) return 0;
        if (rv > cv) {
            *next_first_greater = (uint8_t)k;
            *next_first_greater_val = rv;
            return 1;
        }
    }

    uint16_t last = row[depth - 1];
    if (last < pid) return 0;
    if (last > pid) {
        *next_first_greater = (uint8_t)depth;
        *next_first_greater_val = last;
    } else {
        *next_first_greater = (uint8_t)new_depth;
        *next_first_greater_val = 0;
    }
    return 1;
}

static inline int canon_state_prepare_terminal_fast(const CanonState* st, int partition_id,
                                                    int* next_stabilizer) {
    int depth = st->depth;
    int new_depth = depth + 1;
    int stabilizer = 0;
    uint16_t pid = (uint16_t)partition_id;
    uint16_t max_threshold = pid;
    const uint16_t* partition_perm_row =
        perm_table + (size_t)partition_id * (size_t)perm_count;
    const uint16_t* perm_order_row =
        perm_order_by_value + (size_t)partition_id * (size_t)perm_count;
    const uint16_t* prefix_end_row =
        perm_value_prefix_end + (size_t)partition_id * (size_t)num_partitions;
    const uint8_t* first_greater = st->first_greater;
    const uint16_t* stack_vals = st->stack_vals;

    for (int g = 0; g < depth; g++) {
        if (stack_vals[g] > max_threshold) max_threshold = stack_vals[g];
    }

    uint16_t scan_count = prefix_end_row[max_threshold];

    for (uint16_t i = 0; i < scan_count; i++) {
        uint16_t p = perm_order_row[i];
        uint16_t x = partition_perm_row[p];
        uint8_t g = first_greater[p];

        if (__builtin_expect(g != (uint8_t)depth, 1)) {
            uint16_t c = stack_vals[g];
            if (__builtin_expect(x > c, 1)) {
                continue;
            } else if (x < c) {
                return 0;
            } else {
                uint8_t next_fg;
                uint16_t next_fg_val;
                if (!canon_rebuild_equal_case(st, p, g, pid, &next_fg, &next_fg_val)) {
                    return 0;
                }
                if (next_fg == new_depth) {
                    stabilizer++;
                }
            }
        } else {
            if (x < pid) {
                return 0;
            }
            if (x == pid) {
                stabilizer++;
            }
        }
    }

    *next_stabilizer = stabilizer;
    return 1;
}

static int canon_state_prepare_terminal_profiled(const CanonState* st, int partition_id,
                                                 int* next_stabilizer) {
    int depth = st->depth;
    int new_depth = depth + 1;
    int stabilizer = 0;
    uint16_t pid = (uint16_t)partition_id;
    uint16_t max_threshold = pid;
    const uint16_t* partition_perm_row =
        perm_table + (size_t)partition_id * (size_t)perm_count;
    const uint16_t* perm_order_row =
        perm_order_by_value + (size_t)partition_id * (size_t)perm_count;
    const uint16_t* prefix_end_row =
        perm_value_prefix_end + (size_t)partition_id * (size_t)num_partitions;
    const uint8_t* first_greater = st->first_greater;
    const uint16_t* stack_vals = st->stack_vals;
    ProfileStats* prof = tls_profile;

    prof->canon_prepare_terminal_calls_by_depth[depth]++;

    for (int g = 0; g < depth; g++) {
        if (stack_vals[g] > max_threshold) max_threshold = stack_vals[g];
    }

    uint16_t scan_count = prefix_end_row[max_threshold];
    prof->canon_prepare_terminal_continue_by_depth[depth] += (long long)st->limit - scan_count;

    for (uint16_t i = 0; i < scan_count; i++) {
        uint16_t p = perm_order_row[i];
        uint16_t x = partition_perm_row[p];
        uint8_t g = first_greater[p];

        if (__builtin_expect(g != (uint8_t)depth, 1)) {
            uint16_t c = stack_vals[g];
            if (__builtin_expect(x > c, 1)) {
                prof->canon_prepare_terminal_continue_by_depth[depth]++;
                continue;
            } else if (x < c) {
                prof->canon_prepare_order_rejects_by_depth[depth]++;
                return 0;
            } else {
                uint8_t next_fg;
                uint16_t next_fg_val;
                prof->canon_prepare_equal_case_calls_by_depth[depth]++;
                if (!canon_rebuild_equal_case(st, p, g, pid, &next_fg, &next_fg_val)) {
                    prof->canon_prepare_equal_case_rejects_by_depth[depth]++;
                    return 0;
                }
                if (next_fg == new_depth) {
                    stabilizer++;
                }
            }
        } else {
            if (x < pid) {
                prof->canon_prepare_order_rejects_by_depth[depth]++;
                return 0;
            }
            if (x == pid) {
                stabilizer++;
            }
        }
    }

    *next_stabilizer = stabilizer;
    return 1;
}

static int canon_state_prepare_terminal(const CanonState* st, int partition_id,
                                        int* next_stabilizer) {
#if RECT_PROFILE
    if (tls_profile == NULL) {
        return canon_state_prepare_terminal_fast(st, partition_id, next_stabilizer);
    }
    return canon_state_prepare_terminal_profiled(st, partition_id, next_stabilizer);
#else
    return canon_state_prepare_terminal_fast(st, partition_id, next_stabilizer);
#endif
}

static inline int canon_state_partition_is_rep(const CanonState* st, int min_idx,
                                               int partition_id) {
    uint16_t pid = (uint16_t)partition_id;
    uint16_t min_pid = (uint16_t)min_idx;
    uint16_t eq_count = st->equal_count[st->depth];
    if (eq_count <= 1) return 1;

    const uint16_t* eq = canon_state_equal_perm_row_const(st, st->depth);
    const uint16_t* partition_perm_row =
        perm_table + (size_t)partition_id * (size_t)perm_count;
    for (uint16_t i = 0; i < eq_count; i++) {
        uint16_t image = partition_perm_row[eq[i]];
        if (image >= min_pid && image < pid) {
            return 0;
        }
    }
    return 1;
}

static inline int canon_state_use_orbit_marking(const CanonState* st) {
    return st->equal_count[st->depth] >= RECT_REP_ORBIT_MARK_THRESHOLD;
}

static inline int orbit_mark_bit_test(const uint64_t* bits, int pid) {
    return (bits[(unsigned)pid >> 6] >> ((unsigned)pid & 63u)) & 1u;
}

static inline void orbit_mark_bit_set(uint64_t* bits, int pid) {
    bits[(unsigned)pid >> 6] |= 1ULL << ((unsigned)pid & 63u);
}

static inline void canon_state_mark_orbit_nonreps(const CanonState* st, int min_idx,
                                                  int partition_id, uint64_t* orbit_mark_bits) {
    uint16_t pid = (uint16_t)partition_id;
    uint16_t min_pid = (uint16_t)min_idx;
    uint16_t eq_count = st->equal_count[st->depth];
    const uint16_t* eq = canon_state_equal_perm_row_const(st, st->depth);
    const uint16_t* partition_perm_row =
        perm_table + (size_t)partition_id * (size_t)perm_count;

    for (uint16_t i = 0; i < eq_count; i++) {
        uint16_t image = partition_perm_row[eq[i]];
        if (image >= min_pid && image > pid) {
            orbit_mark_bit_set(orbit_mark_bits, image);
        }
    }
}

static inline void canon_state_seed_orbit_marks(const CanonState* st, int min_idx, int start_pid,
                                                uint64_t* orbit_mark_bits) {
    for (int pid = min_idx; pid < start_pid; pid++) {
        if (orbit_mark_bit_test(orbit_mark_bits, pid)) continue;
        canon_state_mark_orbit_nonreps(st, min_idx, pid, orbit_mark_bits);
    }
}

void canon_state_init(CanonState* st, int limit) {
    memset(st, 0, sizeof(*st));
    st->capacity = limit;
    st->limit = limit;
    st->first_greater =
        checked_calloc((size_t)limit, sizeof(*st->first_greater), "canon_state_first_greater");
    st->first_greater_val =
        checked_calloc((size_t)limit, sizeof(*st->first_greater_val), "canon_state_first_greater_val");
    st->equal_perm =
        checked_calloc((size_t)(g_cols + 1) * (size_t)limit, sizeof(*st->equal_perm),
                       "canon_state_equal_perm");
    st->changed_first_greater_idx =
        checked_calloc((size_t)g_cols * (size_t)limit, sizeof(*st->changed_first_greater_idx),
                       "canon_state_changed_first_greater_idx");
    st->changed_first_greater_old_idx =
        checked_calloc((size_t)g_cols * (size_t)limit, sizeof(*st->changed_first_greater_old_idx),
                       "canon_state_changed_first_greater_old_idx");
    st->changed_first_greater_old_val =
        checked_calloc((size_t)g_cols * (size_t)limit, sizeof(*st->changed_first_greater_old_val),
                       "canon_state_changed_first_greater_old_val");
}

void canon_state_free(CanonState* st) {
    free(st->first_greater);
    free(st->first_greater_val);
    free(st->equal_perm);
    free(st->changed_first_greater_idx);
    free(st->changed_first_greater_old_idx);
    free(st->changed_first_greater_old_val);
    memset(st, 0, sizeof(*st));
}

void canon_scratch_init(CanonScratch* scratch, int limit) {
    memset(scratch, 0, sizeof(*scratch));
    scratch->limit = limit;
    scratch->changed_first_greater_new_idx =
        checked_calloc((size_t)limit, sizeof(*scratch->changed_first_greater_new_idx),
                       "canon_scratch_changed_first_greater_new_idx");
    scratch->changed_first_greater_new_val =
        checked_calloc((size_t)limit, sizeof(*scratch->changed_first_greater_new_val),
                       "canon_scratch_changed_first_greater_new_val");
    scratch->next_equal_perm =
        checked_calloc((size_t)limit, sizeof(*scratch->next_equal_perm),
                       "canon_scratch_next_equal_perm");
    scratch->changed_first_greater_idx = checked_calloc((size_t)limit,
                                                        sizeof(*scratch->changed_first_greater_idx),
                                                        "canon_scratch_changed_first_greater_idx");
}

void canon_scratch_free(CanonScratch* scratch) {
    free(scratch->changed_first_greater_new_idx);
    free(scratch->changed_first_greater_new_val);
    free(scratch->next_equal_perm);
    free(scratch->changed_first_greater_idx);
    memset(scratch, 0, sizeof(*scratch));
}

void canon_state_reset(CanonState* st, int limit) {
    if (limit < 0 || limit > st->capacity) {
        fprintf(stderr, "canon_state_reset limit %d out of range [0, %d]\n", limit, st->capacity);
        abort();
    }
    size_t count = (size_t)limit;
    st->limit = limit;
    st->depth = 0;
    st->stabilizer[0] = limit;
    st->equal_count[0] = (uint16_t)limit;
    uint16_t* equal_perm0 = canon_state_equal_perm_row(st, 0);
    for (int p = 0; p < limit; p++) {
        equal_perm0[p] = (uint16_t)p;
    }
    memset(st->first_greater, 0, count * sizeof(*st->first_greater));
    memset(st->first_greater_val, 0, count * sizeof(*st->first_greater_val));
}

static inline int canon_state_prepare_push_fast(const CanonState* st, int partition_id,
                                                CanonScratch* scratch,
                                                int* next_stabilizer) {
    int depth = st->depth;
    int new_depth = depth + 1;
    int stabilizer = 0;
    uint16_t pid = (uint16_t)partition_id;
    const uint16_t* partition_perm_row =
        perm_table + (size_t)partition_id * (size_t)perm_count;
    const uint8_t* first_greater = st->first_greater;
    const uint16_t* stack_vals = st->stack_vals;
    const uint16_t* first_greater_val = st->first_greater_val;
    uint8_t* changed_first_greater_new_idx = scratch->changed_first_greater_new_idx;
    uint16_t* changed_first_greater_new_val = scratch->changed_first_greater_new_val;
    uint16_t* next_equal_perm = scratch->next_equal_perm;
    uint16_t* changed_first_greater_idx = scratch->changed_first_greater_idx;
    uint16_t next_equal_count = 0;
    uint16_t changed_first_greater_count = 0;

    for (int p = 0; p < st->limit; p++) {
        uint8_t old_fg = first_greater[p];
        uint16_t old_fg_val = first_greater_val[p];
        uint16_t x = partition_perm_row[p];
        uint8_t g = old_fg;
        uint8_t next_fg;
        uint16_t next_fg_val;

        if (__builtin_expect(g != (uint8_t)depth, 1)) {
            uint16_t r = old_fg_val;
            if (__builtin_expect(x >= r, 1)) {
                continue;
            }

            uint16_t c = stack_vals[g];
            if (__builtin_expect(x > c, 1)) {
                next_fg = g;
                next_fg_val = x;
            } else if (x < c) {
                return 0;
            } else {
                if (!canon_rebuild_equal_case(st, p, g, pid, &next_fg, &next_fg_val)) {
                    return 0;
                }
                if (next_fg == new_depth) {
                    stabilizer++;
                }
            }
        } else {
            if (x < pid) {
                return 0;
            }
            if (x == pid) {
                next_fg = (uint8_t)new_depth;
                next_fg_val = 0;
                stabilizer++;
            } else {
                next_fg = (uint8_t)depth;
                next_fg_val = x;
            }
        }

        uint16_t new_fg_val = (next_fg < new_depth) ? next_fg_val : 0;
        if (next_fg == new_depth) {
            next_equal_perm[next_equal_count++] = (uint16_t)p;
        }
        if (old_fg != next_fg || old_fg_val != new_fg_val) {
            changed_first_greater_idx[changed_first_greater_count] = (uint16_t)p;
            changed_first_greater_new_idx[changed_first_greater_count] = next_fg;
            changed_first_greater_new_val[changed_first_greater_count] = new_fg_val;
            changed_first_greater_count++;
        }
    }
    scratch->next_equal_count = next_equal_count;
    scratch->changed_first_greater_count = changed_first_greater_count;
    *next_stabilizer = stabilizer;
    return 1;
}

static int canon_state_prepare_push_profiled(const CanonState* st, int partition_id,
                                             CanonScratch* scratch,
                                             int* next_stabilizer) {
    int depth = st->depth;
    int new_depth = depth + 1;
    int stabilizer = 0;
    uint16_t pid = (uint16_t)partition_id;
    const uint16_t* partition_perm_row =
        perm_table + (size_t)partition_id * (size_t)perm_count;
    const uint8_t* first_greater = st->first_greater;
    const uint16_t* stack_vals = st->stack_vals;
    const uint16_t* first_greater_val = st->first_greater_val;
    uint8_t* changed_first_greater_new_idx = scratch->changed_first_greater_new_idx;
    uint16_t* changed_first_greater_new_val = scratch->changed_first_greater_new_val;
    uint16_t* next_equal_perm = scratch->next_equal_perm;
    uint16_t* changed_first_greater_idx = scratch->changed_first_greater_idx;
    uint16_t active_count = 0;
    uint16_t next_equal_count = 0;
    uint16_t changed_first_greater_count = 0;
    ProfileStats* prof = tls_profile;

    for (int p = 0; p < st->limit; p++) {
        uint8_t old_fg = first_greater[p];
        uint16_t old_fg_val = first_greater_val[p];
        uint16_t x = partition_perm_row[p];
        uint8_t g = old_fg;
        uint8_t next_fg;
        uint16_t next_fg_val;

        if (__builtin_expect(g != (uint8_t)depth, 1)) {
            uint16_t r = old_fg_val;
            if (__builtin_expect(x >= r, 1)) {
                prof->canon_prepare_fast_continue_by_depth[depth]++;
                continue;
            }

            uint16_t c = stack_vals[g];
            if (__builtin_expect(x > c, 1)) {
                next_fg = g;
                next_fg_val = x;
            } else if (x < c) {
                prof->canon_prepare_order_rejects_by_depth[depth]++;
                return 0;
            } else {
                prof->canon_prepare_equal_case_calls_by_depth[depth]++;
                if (!canon_rebuild_equal_case(st, p, g, pid, &next_fg, &next_fg_val)) {
                    prof->canon_prepare_equal_case_rejects_by_depth[depth]++;
                    return 0;
                }
                if (next_fg == new_depth) {
                    stabilizer++;
                }
            }
        } else {
            if (x < pid) {
                prof->canon_prepare_order_rejects_by_depth[depth]++;
                return 0;
            }
            if (x == pid) {
                next_fg = (uint8_t)new_depth;
                next_fg_val = 0;
                stabilizer++;
            } else {
                next_fg = (uint8_t)depth;
                next_fg_val = x;
            }
        }

        uint16_t new_fg_val = (next_fg < new_depth) ? next_fg_val : 0;
        active_count++;
        if (next_fg == new_depth) {
            next_equal_perm[next_equal_count++] = (uint16_t)p;
        }
        if (old_fg != next_fg || old_fg_val != new_fg_val) {
            changed_first_greater_idx[changed_first_greater_count] = (uint16_t)p;
            changed_first_greater_new_idx[changed_first_greater_count] = next_fg;
            changed_first_greater_new_val[changed_first_greater_count] = new_fg_val;
            changed_first_greater_count++;
        }
    }
    scratch->next_equal_count = next_equal_count;
    scratch->changed_first_greater_count = changed_first_greater_count;
    prof->canon_prepare_scanned_by_depth[depth] += st->limit;
    prof->canon_prepare_active_by_depth[depth] += active_count;
    *next_stabilizer = stabilizer;
    return 1;
}

int canon_state_prepare_push(const CanonState* st, int partition_id, CanonScratch* scratch,
                             int* next_stabilizer) {
#if RECT_PROFILE
    if (tls_profile == NULL) {
        return canon_state_prepare_push_fast(st, partition_id, scratch, next_stabilizer);
    }
    return canon_state_prepare_push_profiled(st, partition_id, scratch, next_stabilizer);
#else
    return canon_state_prepare_push_fast(st, partition_id, scratch, next_stabilizer);
#endif
}

void canon_state_commit_push(CanonState* st, int partition_id, const CanonScratch* scratch,
                             int next_stabilizer) {
    int depth = st->depth;
    int new_depth = depth + 1;
    uint16_t* equal_perm = canon_state_equal_perm_row(st, new_depth);
    uint16_t* changed_first_greater_idx = canon_state_changed_first_greater_idx_row(st, depth);
    uint8_t* changed_first_greater_old_idx = canon_state_changed_first_greater_old_idx_row(st, depth);
    uint16_t* changed_first_greater_old_val = canon_state_changed_first_greater_old_val_row(st, depth);
    uint16_t changed_fg_count = scratch->changed_first_greater_count;
    st->stack_vals[depth] = (uint16_t)partition_id;
    st->stack_perm_rows[depth] = perm_table + (size_t)partition_id * (size_t)perm_count;

    for (uint16_t i = 0; i < scratch->next_equal_count; i++) {
        equal_perm[i] = scratch->next_equal_perm[i];
    }
    for (uint16_t i = 0; i < changed_fg_count; i++) {
        uint16_t p = scratch->changed_first_greater_idx[i];
        changed_first_greater_idx[i] = p;
        changed_first_greater_old_idx[i] = st->first_greater[p];
        changed_first_greater_old_val[i] = st->first_greater_val[p];
        st->first_greater[p] = scratch->changed_first_greater_new_idx[i];
        st->first_greater_val[p] = scratch->changed_first_greater_new_val[i];
    }

    st->equal_count[new_depth] = scratch->next_equal_count;
    st->changed_first_greater_count[depth] = changed_fg_count;
    st->stabilizer[new_depth] = next_stabilizer;
    st->depth = new_depth;
}

void canon_state_pop(CanonState* st) {
    int depth = st->depth - 1;
    uint16_t* changed_first_greater_idx = canon_state_changed_first_greater_idx_row(st, depth);
    uint8_t* changed_first_greater_old_idx = canon_state_changed_first_greater_old_idx_row(st, depth);
    uint16_t* changed_first_greater_old_val = canon_state_changed_first_greater_old_val_row(st, depth);
    for (uint16_t i = 0; i < st->changed_first_greater_count[depth]; i++) {
        uint16_t p = changed_first_greater_idx[i];
        st->first_greater[p] = changed_first_greater_old_idx[i];
        st->first_greater_val[p] = changed_first_greater_old_val[i];
    }
    st->depth = depth;
}

long long get_orbit_multiplier_state(const CanonState* st) {
    int stabilizer = st->stabilizer[st->depth];
    return factorial[g_rows] / stabilizer;
}

// --- NAUTY CANONICALISATION ---

static inline setword pack_row_to_nauty1(uint64_t row_bits, int n) {
    if (n < 64) row_bits &= (1ULL << n) - 1ULL;
    setword row = 0;
    while (row_bits) {
        int j = __builtin_ctzll(row_bits);
        row |= bit[j];
        row_bits &= row_bits - 1;
    }
    return row;
}

// Convert our graph to nauty format and compute canonical form
void nauty_workspace_init(NautyWorkspace* ws, int n) {
    int m = SETWORDSNEEDED(n);
    if (n <= ws->nmax && m <= ws->mmax) return;
    free(ws->ng);
    free(ws->cg);
    free(ws->lab);
    free(ws->ptn);
    free(ws->orbits);
    ws->ng = checked_calloc((size_t)n * (size_t)m, sizeof(graph), "nauty_ng");
    ws->cg = checked_calloc((size_t)n * (size_t)m, sizeof(graph), "nauty_cg");
    ws->lab = checked_calloc((size_t)n, sizeof(int), "nauty_lab");
    ws->ptn = checked_calloc((size_t)n, sizeof(int), "nauty_ptn");
    ws->orbits = checked_calloc((size_t)n, sizeof(int), "nauty_orbits");
    ws->nmax = n;
    ws->mmax = m;
}

void nauty_workspace_free(NautyWorkspace* ws) {
    free(ws->ng);
    free(ws->cg);
    free(ws->lab);
    free(ws->ptn);
    free(ws->orbits);
    memset(ws, 0, sizeof(*ws));
}

void get_canonical_graph_from_dense_rows(int n, const AdjWord* rows, Graph* canon,
                                         NautyWorkspace* ws, ProfileStats* profile) {
    double total_t0 = 0.0;
    double phase_t0 = 0.0;
    
    if (n == 0) {
        canon->n = 0;
        canon->vertex_mask = 0;
        memset(canon->adj, 0, sizeof(canon->adj));
        return;
    }
    
    if (n == 1) {
        canon->n = 1;
        canon->vertex_mask = 1;
        canon->adj[0] = 0;
        return;
    }
    
    int m = SETWORDSNEEDED(n);
    nauty_workspace_init(ws, n);
    
    graph* ng = ws->ng;
    graph* cg = ws->cg;
    int* lab = ws->lab;
    int* ptn = ws->ptn;
    int* orbits = ws->orbits;
    uint8_t degrees[MAXN_NAUTY];
    if (PROFILE_BUILD && profile) {
        total_t0 = omp_get_wtime();
        phase_t0 = total_t0;
    }
    
    if (m == 1) {
        for (int i = 0; i < n; i++) {
            GRAPHROW(ng, i, 1)[0] = pack_row_to_nauty1((uint64_t)rows[i], n);
        }
    } else {
        EMPTYGRAPH(ng, m, n);
        for (int i = 0; i < n; i++) {
            uint64_t upper = (uint64_t)rows[i] & ~((UINT64_C(1) << (i + 1)) - 1U);
            while (upper) {
                int j = __builtin_ctzll(upper);
                ADDONEEDGE(ng, i, j, m);
                upper &= upper - 1;
            }
        }
    }
    if (PROFILE_BUILD && profile) {
        profile->get_canonical_graph_build_input_time += omp_get_wtime() - phase_t0;
    }

    int degree_counts[MAXN_NAUTY + 1] = {0};
    for (int i = 0; i < n; i++) {
        degrees[i] = (uint8_t)__builtin_popcountll((uint64_t)rows[i]);
        degree_counts[degrees[i]]++;
    }
    int degree_offsets[MAXN_NAUTY + 1];
    int pos = 0;
    for (int deg = 0; deg <= MAXN_NAUTY; deg++) {
        degree_offsets[deg] = pos;
        pos += degree_counts[deg];
    }
    for (int v = 0; v < n; v++) {
        lab[degree_offsets[degrees[v]]++] = v;
    }
    for (int i = 0; i < n; i++) {
        ptn[i] = (i + 1 < n && degrees[lab[i]] == degrees[lab[i + 1]]) ? 1 : 0;
    }
    
    // Set up options for canonical labelling
    DEFAULTOPTIONS_GRAPH(options);
    options.getcanon = TRUE;
    options.defaultptn = FALSE;
    
    statsblk stats;
    
    // Compute canonical form
    if (PROFILE_BUILD && profile) phase_t0 = omp_get_wtime();
    densenauty(ng, lab, ptn, orbits, &options, &stats, m, n, cg);
    if (PROFILE_BUILD && profile) {
        profile->nauty_calls++;
        profile->nauty_time += omp_get_wtime() - phase_t0;
        phase_t0 = omp_get_wtime();
    }

    // nauty returns lab[i] = original vertex now placed at canonical position i.
    uint8_t new_index_of_old[MAXN_NAUTY];
    for (int i = 0; i < n; i++) new_index_of_old[lab[i]] = (uint8_t)i;
    graph_apply_permutation_dense_rows((uint32_t)n, rows, new_index_of_old, canon);
    if (PROFILE_BUILD && profile) {
        profile->get_canonical_graph_rebuild_time += omp_get_wtime() - phase_t0;
        profile->get_canonical_graph_time += omp_get_wtime() - total_t0;
    }
}

void get_canonical_graph(Graph* g, Graph* canon, NautyWorkspace* ws, ProfileStats* profile) {
    AdjWord rows[MAXN_NAUTY];
    double phase_t0 = 0.0;
    if (PROFILE_BUILD && profile) phase_t0 = omp_get_wtime();
    int n = (int)graph_build_dense_rows(g, rows);
    if (PROFILE_BUILD && profile) {
        double dt = omp_get_wtime() - phase_t0;
        profile->get_canonical_graph_dense_rows_time += dt;
        profile->get_canonical_graph_time += dt;
    }
    get_canonical_graph_from_dense_rows(n, rows, canon, ws, profile);
}

// --- GRAPH SOLVER ---

static inline uint32_t small_graph_stride(int n) {
    return (uint32_t)(n + 1);
}

static inline uint32_t small_graph_edge_total(int n) {
    return (uint32_t)(n * (n - 1) / 2);
}

int32_t* small_graph_poly_slot(int n, uint32_t mask) {
    return g_small_graph_lookup_coeffs[n] + (size_t)mask * (size_t)small_graph_stride(n);
}

static void small_graph_lookup_init_layout(void) {
    g_small_graph_graph_count[0] = 1;
    g_small_graph_edge_count[0] = 0;
    for (int n = 1; n <= SMALL_GRAPH_LOOKUP_MAX_N; n++) {
        uint32_t edge_total = small_graph_edge_total(n);
        g_small_graph_graph_count[n] = 1U << edge_total;
        g_small_graph_edge_count[n] = (uint8_t)edge_total;

        int bit = 0;
        for (int j = 1; j < n; j++) {
            for (int i = 0; i < j; i++, bit++) {
                g_small_graph_edge_u[n][bit] = (uint8_t)i;
                g_small_graph_edge_v[n][bit] = (uint8_t)j;
            }
        }
    }
}

static int small_graph_lookup_allocate_storage(void) {
    if (g_small_graph_lookup_coeffs[0]) return 1;
    for (int n = 0; n <= SMALL_GRAPH_LOOKUP_MAX_N; n++) {
        uint32_t stride = small_graph_stride(n);
        uint32_t graph_count = g_small_graph_graph_count[n];
        g_small_graph_lookup_coeffs[n] =
            checked_calloc((size_t)graph_count * (size_t)stride, sizeof(int32_t), "small_graph_lookup");
    }
    return 1;
}

static uint32_t small_graph_pack_mask_from_rows(const uint8_t* rows, int n) {
    uint32_t mask = 0;
    int bit = 0;
    for (int j = 1; j < n; j++) {
        for (int i = 0; i < j; i++, bit++) {
            if ((rows[i] >> j) & 1U) mask |= 1U << bit;
        }
    }
    return mask;
}

uint32_t small_graph_pack_mask(const Graph* g) {
    AdjWord rows[MAXN_NAUTY];
    uint32_t n = graph_build_dense_rows(g, rows);
    uint32_t mask = 0;
    int bit = 0;
    for (uint32_t j = 1; j < n; j++) {
        for (uint32_t i = 0; i < j; i++, bit++) {
            if (((uint64_t)rows[i] >> j) & 1ULL) mask |= 1U << bit;
        }
    }
    return mask;
}

uint64_t graph_pack_upper_mask64(const Graph* g) {
    int edge_total = g->n * (g->n - 1) / 2;
    if (edge_total > 64) {
        fprintf(stderr, "graph_pack_upper_mask64 only supports graphs with at most 64 edge bits\n");
        exit(1);
    }
    AdjWord rows[MAXN_NAUTY];
    uint32_t n = graph_build_dense_rows(g, rows);
    uint64_t mask = 0;
    int bit = 0;
    for (uint32_t j = 1; j < n; j++) {
        for (uint32_t i = 0; i < j; i++, bit++) {
            if (((uint64_t)rows[i] >> j) & 1ULL) mask |= 1ULL << bit;
        }
    }
    return mask;
}

static uint32_t small_graph_contract_mask(uint32_t mask, int n, int u, int v) {
    uint8_t rows[SMALL_GRAPH_LOOKUP_MAX_N] = {0};
    int bit = 0;
    for (int j = 1; j < n; j++) {
        for (int i = 0; i < j; i++, bit++) {
            if ((mask >> bit) & 1U) {
                rows[i] |= (uint8_t)(1U << j);
                rows[j] |= (uint8_t)(1U << i);
            }
        }
    }

    rows[u] |= rows[v];
    rows[u] &= (uint8_t)~((1U << u) | (1U << v));
    for (int k = 0; k < n; k++) {
        if (k == u || k == v) continue;
        if ((rows[k] >> v) & 1U) {
            rows[k] &= (uint8_t)~(1U << v);
            rows[k] |= (uint8_t)(1U << u);
            rows[u] |= (uint8_t)(1U << k);
        }
    }

    uint8_t compact[SMALL_GRAPH_LOOKUP_MAX_N] = {0};
    int remap[SMALL_GRAPH_LOOKUP_MAX_N];
    int next = 0;
    for (int i = 0; i < n; i++) remap[i] = -1;
    for (int i = 0; i < n; i++) {
        if (i == v) continue;
        remap[i] = next++;
    }
    for (int i = 0; i < n; i++) {
        if (i == v) continue;
        uint8_t new_row = 0;
        for (int j = 0; j < n; j++) {
            if (j == v || !((rows[i] >> j) & 1U)) continue;
            new_row |= (uint8_t)(1U << remap[j]);
        }
        compact[remap[i]] = new_row;
    }
    return small_graph_pack_mask_from_rows(compact, n - 1);
}

static void small_graph_lookup_generate_tables(void) {
    int32_t* empty0 = small_graph_poly_slot(0, 0);
    empty0[0] = 1;

    for (int n = 1; n <= SMALL_GRAPH_LOOKUP_MAX_N; n++) {
        uint32_t graph_count = g_small_graph_graph_count[n];
        int32_t* empty_poly = small_graph_poly_slot(n, 0);
        empty_poly[n] = 1;

        for (uint32_t mask = 1; mask < graph_count; mask++) {
            uint32_t del_mask = mask & (mask - 1U);
            int edge_bit = __builtin_ctz(mask);
            int u = g_small_graph_edge_u[n][edge_bit];
            int v = g_small_graph_edge_v[n][edge_bit];
            uint32_t cont_mask = small_graph_contract_mask(mask, n, u, v);
            int32_t* out = small_graph_poly_slot(n, mask);
            const int32_t* del_poly = small_graph_poly_slot(n, del_mask);
            const int32_t* cont_poly = small_graph_poly_slot(n - 1, cont_mask);
            for (int k = 0; k <= n; k++) {
                int32_t cont_coeff = (k < n) ? cont_poly[k] : 0;
                out[k] = del_poly[k] - cont_coeff;
            }
        }
    }
}

static const char* small_graph_lookup_default_path(void) {
    const char* env_path = getenv("RECT_SMALL_GRAPH_TABLE");
    if (env_path && *env_path) return env_path;
    return "small_graph_lookup_n7.bin";
}

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t max_n;
} SmallGraphTableHeader;

#define SMALL_GRAPH_TABLE_MAGIC UINT64_C(0x534750375441424c)
#define SMALL_GRAPH_TABLE_VERSION 1U

static int small_graph_lookup_try_load_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;

    SmallGraphTableHeader header;
    int ok = fread(&header, sizeof(header), 1, f) == 1 &&
             header.magic == SMALL_GRAPH_TABLE_MAGIC &&
             header.version == SMALL_GRAPH_TABLE_VERSION &&
             header.max_n == SMALL_GRAPH_LOOKUP_MAX_N;
    if (!ok) {
        fclose(f);
        return 0;
    }

    small_graph_lookup_init_layout();
    small_graph_lookup_allocate_storage();
    for (int n = 0; n <= SMALL_GRAPH_LOOKUP_MAX_N; n++) {
        uint32_t count = g_small_graph_graph_count[n] * small_graph_stride(n);
        if (fread(g_small_graph_lookup_coeffs[n], sizeof(int32_t), count, f) != count) {
            fclose(f);
            small_graph_lookup_free();
            return 0;
        }
    }
    fclose(f);
    g_small_graph_lookup_loaded_from_file = 1;
    g_small_graph_lookup_ready = 1;
    return 1;
}

void small_graph_lookup_init(void) {
    if (g_small_graph_lookup_ready) return;

    double t0 = omp_get_wtime();
    g_small_graph_lookup_loaded_from_file = 0;
    if (!small_graph_lookup_try_load_file(small_graph_lookup_default_path())) {
        small_graph_lookup_init_layout();
        small_graph_lookup_allocate_storage();
        small_graph_lookup_generate_tables();
        g_small_graph_lookup_loaded_from_file = 0;
        g_small_graph_lookup_ready = 1;
    }
    g_small_graph_lookup_init_time = omp_get_wtime() - t0;
}

void small_graph_lookup_free(void) {
    for (int n = 0; n <= SMALL_GRAPH_LOOKUP_MAX_N; n++) {
        free(g_small_graph_lookup_coeffs[n]);
        g_small_graph_lookup_coeffs[n] = NULL;
    }
    g_small_graph_lookup_ready = 0;
    g_small_graph_lookup_init_time = 0.0;
    g_small_graph_lookup_loaded_from_file = 0;
}

static void small_graph_lookup_load_poly(int n, uint32_t mask, Poly* out) {
    const int32_t* coeffs = small_graph_poly_slot(n, mask);
    out->deg = n;
    for (int i = 0; i <= n; i++) out->coeffs[i] = coeffs[i];
}

void small_graph_lookup_load_graph_poly(int n, uint32_t mask, GraphPoly* out) {
    const int32_t* coeffs = small_graph_poly_slot(n, mask);
    out->deg = (uint8_t)n;
    for (int i = 0; i <= n; i++) out->coeffs[i] = coeffs[i];
}

static const char* connected_canon_lookup_default_path(void) {
    const char* env_path = getenv("RECT_CONNECTED_CANON_LOOKUP");
    if (env_path && *env_path) return env_path;
    return NULL;
}

int connected_canon_lookup_entry_cmp(const void* lhs, const void* rhs) {
    const ConnectedCanonLookupEntry* a = lhs;
    const ConnectedCanonLookupEntry* b = rhs;
    if (a->mask < b->mask) return -1;
    if (a->mask > b->mask) return 1;
    return 0;
}

static size_t connected_canon_lookup_entry_file_size(int n) {
    return sizeof(uint64_t) + (size_t)(n + 1) * sizeof(int32_t);
}

static int connected_canon_lookup_try_load_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;

    ConnectedCanonLookupHeader header;
    int ok = fread(&header, sizeof(header), 1, f) == 1 &&
             header.magic == CONNECTED_CANON_LOOKUP_MAGIC &&
             header.version == CONNECTED_CANON_LOOKUP_VERSION &&
             header.n > 0 &&
             header.n <= CONNECTED_CANON_LOOKUP_MAX_N;
    if (!ok) {
        fclose(f);
        return 0;
    }

    ConnectedCanonLookupEntry* entries =
        checked_calloc((size_t)header.count, sizeof(*entries), "connected_canon_lookup");
    for (uint32_t i = 0; i < header.count; i++) {
        if (fread(&entries[i].mask, sizeof(entries[i].mask), 1, f) != 1 ||
            fread(entries[i].coeffs, sizeof(entries[i].coeffs[0]), (size_t)header.n + 1, f) !=
                (size_t)header.n + 1) {
            fclose(f);
            free(entries);
            return 0;
        }
    }
    long expected_size = (long)(sizeof(header) +
                                (size_t)header.count * connected_canon_lookup_entry_file_size((int)header.n));
    if (fseek(f, 0, SEEK_END) != 0 || ftell(f) != expected_size) {
        fclose(f);
        free(entries);
        return 0;
    }
    fclose(f);

    g_connected_canon_lookup = entries;
    g_connected_canon_lookup_count = header.count;
    g_connected_canon_lookup_n = (int)header.n;
    g_connected_canon_lookup_ready = 1;
    g_connected_canon_lookup_loaded = 1;
    return 1;
}

void connected_canon_lookup_init(void) {
    if (g_connected_canon_lookup_ready) return;
    double t0 = omp_get_wtime();
    const char* env_path = connected_canon_lookup_default_path();
    g_connected_canon_lookup_loaded = 0;
    if (env_path) {
        g_connected_canon_lookup_loaded =
            connected_canon_lookup_try_load_file(env_path);
    } else {
        g_connected_canon_lookup_loaded =
            connected_canon_lookup_try_load_file("connected_canon_lookup_n10.bin") ||
            connected_canon_lookup_try_load_file("connected_canon_lookup_n9.bin");
    }
    g_connected_canon_lookup_load_time = omp_get_wtime() - t0;
}

void connected_canon_lookup_free(void) {
    free(g_connected_canon_lookup);
    g_connected_canon_lookup = NULL;
    g_connected_canon_lookup_count = 0;
    g_connected_canon_lookup_ready = 0;
    g_connected_canon_lookup_loaded = 0;
    g_connected_canon_lookup_n = 0;
    g_connected_canon_lookup_load_time = 0.0;
}

int connected_canon_lookup_load_graph_poly(const Graph* g, GraphPoly* out) {
    if (!g_connected_canon_lookup_ready || g->n != g_connected_canon_lookup_n) return 0;

    uint64_t mask = graph_pack_upper_mask64(g);
    ConnectedCanonLookupEntry key = {.mask = mask};
    ConnectedCanonLookupEntry* entry = bsearch(&key, g_connected_canon_lookup,
                                               g_connected_canon_lookup_count,
                                               sizeof(*g_connected_canon_lookup),
                                               connected_canon_lookup_entry_cmp);
    if (!entry) return 0;

    out->deg = (uint8_t)g_connected_canon_lookup_n;
    for (int i = 0; i <= g_connected_canon_lookup_n; i++) out->coeffs[i] = entry->coeffs[i];
    return 1;
}

uint64_t graph_row_mask(int n) {
    if (n >= 64) return ~0ULL;
    if (n <= 0) return 0ULL;
    return (1ULL << n) - 1ULL;
}

uint32_t graph_build_dense_rows(const Graph* g, AdjWord* rows) {
    uint64_t active = g->vertex_mask & ADJWORD_MASK;
    uint32_t n = (uint32_t)__builtin_popcountll(active);

    if (n != g->n) {
        fprintf(stderr, "Graph vertex mask/count mismatch: n=%u mask_popcount=%u\n",
                (unsigned)g->n, (unsigned)n);
        exit(1);
    }

    uint64_t rem = active;
    uint32_t dense_v = 0;
    while (rem) {
        int v = __builtin_ctzll(rem);
#if defined(__BMI2__) && (defined(__x86_64__) || defined(__i386__))
        rows[dense_v] = (AdjWord)_pext_u64((uint64_t)g->adj[v] & active, active);
#else
        uint64_t row_bits = (uint64_t)g->adj[v] & active;
        AdjWord row = 0;
        uint32_t dense_u = 0;
        uint64_t bit = active;
        while (bit) {
            int u = __builtin_ctzll(bit);
            if ((row_bits >> u) & 1U) row |= (AdjWord)(UINT64_C(1) << dense_u);
            dense_u++;
            bit &= bit - 1;
        }
        rows[dense_v] = row;
#endif
        dense_v++;
        rem &= rem - 1;
    }
    return n;
}

static void graph_transpose_dense_rows(uint32_t n, const AdjWord* src_rows, AdjWord* dst_rows) {
    uint64_t mask = graph_row_mask((int)n);
    for (uint32_t i = 0; i < n; i++) dst_rows[i] = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint64_t row = (uint64_t)src_rows[i] & mask;
        while (row) {
            uint32_t j = (uint32_t)__builtin_ctzll(row);
            dst_rows[j] |= (AdjWord)(UINT64_C(1) << i);
            row &= row - 1;
        }
    }
}

static void graph_apply_permutation_dense_rows(uint32_t n, const AdjWord* dense_rows,
                                               const uint8_t* new_index_of_old, Graph* dst) {
    AdjWord row_permuted[MAXN_NAUTY];
    AdjWord transposed[MAXN_NAUTY];

    for (uint32_t i = 0; i < n; i++) {
        row_permuted[new_index_of_old[i]] = dense_rows[i];
    }
    graph_transpose_dense_rows(n, row_permuted, transposed);
    for (uint32_t i = 0; i < n; i++) {
        dst->adj[new_index_of_old[i]] = transposed[i];
    }
    dst->n = (uint8_t)n;
    dst->vertex_mask = graph_row_mask((int)n);
}

static void graph_apply_permutation(const Graph* src, const uint8_t* new_index_of_old,
                                    Graph* dst) {
    AdjWord dense_rows[MAXN_NAUTY];
    uint32_t n = graph_build_dense_rows(src, dense_rows);
    graph_apply_permutation_dense_rows(n, dense_rows, new_index_of_old, dst);
}

static uint32_t graph_build_dense_rows_from_mask(const Graph* src, uint64_t mask, AdjWord* rows) {
    uint64_t active = mask & src->vertex_mask & ADJWORD_MASK;
    uint32_t n = (uint32_t)__builtin_popcountll(active);

    uint64_t rem = active;
    uint32_t dense_v = 0;
    while (rem) {
        int v = __builtin_ctzll(rem);
#if defined(__BMI2__) && (defined(__x86_64__) || defined(__i386__))
        rows[dense_v] = (AdjWord)_pext_u64((uint64_t)src->adj[v] & active, active);
#else
        uint64_t row_bits = (uint64_t)src->adj[v] & active;
        AdjWord row = 0;
        uint32_t dense_u = 0;
        uint64_t bit = active;
        while (bit) {
            int u = __builtin_ctzll(bit);
            if ((row_bits >> u) & 1U) row |= (AdjWord)(UINT64_C(1) << dense_u);
            dense_u++;
            bit &= bit - 1;
        }
        rows[dense_v] = row;
#endif
        dense_v++;
        rem &= rem - 1;
    }
    return n;
}

void induced_subgraph_from_mask(const Graph* src, uint64_t mask, Graph* dst) {
    uint32_t n = graph_build_dense_rows_from_mask(src, mask, dst->adj);
    dst->n = (uint8_t)n;
    dst->vertex_mask = graph_row_mask((int)n);
}

int graph_collect_components(const Graph* g, uint64_t* component_masks) {
    uint64_t remaining = g->vertex_mask;
    int count = 0;
    while (remaining) {
        int start = __builtin_ctzll(remaining);
        uint64_t component = 0;
        uint64_t frontier = 1ULL << start;
        while (frontier) {
            component |= frontier;
            uint64_t next = 0;
            uint64_t current = frontier;
            while (current) {
                int v = __builtin_ctzll(current);
                next |= (uint64_t)g->adj[v] & g->vertex_mask;
                current &= current - 1;
            }
            frontier = next & remaining & ~component;
        }
        component_masks[count++] = component;
        remaining &= ~component;
    }
    return count;
}

int graph_has_articulation_point(const Graph* g) {
    if (g->n <= 2) return 0;
    uint64_t full = g->vertex_mask;
    uint64_t rem_vertices = full;
    while (rem_vertices) {
        int v = __builtin_ctzll(rem_vertices);
        uint64_t remaining = full & ~(1ULL << v);
        if (remaining == 0) return 0;
        int start = __builtin_ctzll(remaining);
        uint64_t visited = 0;
        uint64_t frontier = 1ULL << start;
        while (frontier) {
            visited |= frontier;
            uint64_t next = 0;
            uint64_t current = frontier;
            while (current) {
                int u = __builtin_ctzll(current);
                next |= ((uint64_t)g->adj[u] & g->vertex_mask) & remaining;
                current &= current - 1;
            }
            frontier = next & remaining & ~visited;
        }
        if (visited != remaining) return 1;
        rem_vertices &= rem_vertices - 1;
    }
    return 0;
}

int graph_has_k2_separator(const Graph* g) {
    if (g->n <= 3) return 0;
    uint64_t full = g->vertex_mask;
    uint64_t rem_u = full;
    while (rem_u) {
        int u = __builtin_ctzll(rem_u);
        uint64_t nbrs = ((uint64_t)g->adj[u] & full) & ~((1ULL << (u + 1)) - 1ULL);
        while (nbrs) {
            int v = __builtin_ctzll(nbrs);
            nbrs &= nbrs - 1;
            uint64_t remaining = full & ~(1ULL << u) & ~(1ULL << v);
            if (remaining == 0) continue;
            int start = __builtin_ctzll(remaining);
            uint64_t visited = 0;
            uint64_t frontier = 1ULL << start;
            while (frontier) {
                visited |= frontier;
                uint64_t next = 0;
                uint64_t current = frontier;
                while (current) {
                    int w = __builtin_ctzll(current);
                    next |= ((uint64_t)g->adj[w] & g->vertex_mask) & remaining;
                    current &= current - 1;
                }
                frontier = next & remaining & ~visited;
            }
            if (visited != remaining) return 1;
        }
        rem_u &= rem_u - 1;
    }
    return 0;
}

uint64_t hash_graph(const Graph* g) {
    uint64_t h = 14695981039346656037ULL;
    h ^= g->vertex_mask;
    h *= 1099511628211ULL;
    uint64_t rem = g->vertex_mask;
    while (rem) {
        int i = __builtin_ctzll(rem);
        h ^= ((uint64_t)g->adj[i] & g->vertex_mask);
        h *= 1099511628211ULL;
        rem &= rem - 1;
    }
    h ^= (uint64_t)g->n;
    h *= 1099511628211ULL;
    return h;
}

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

static inline uint64_t hash_dense_rows(uint32_t n, const AdjWord* rows) {
    uint64_t h = 14695981039346656037ULL;
    for (uint32_t i = 0; i < n; i++) {
        h ^= (uint64_t)rows[i];
        h *= 1099511628211ULL;
    }
    h ^= (uint64_t)n;
    h *= 1099511628211ULL;
    return h;
}

uint64_t graph_fill_dense_key_rows(const Graph* g, AdjWord row_mask, AdjWord* rows) {
    if (g->vertex_mask == graph_row_mask(g->n)) {
        uint64_t h = 14695981039346656037ULL;
        for (uint32_t i = 0; i < g->n; i++) {
            AdjWord row = g->adj[i] & row_mask;
            rows[i] = row;
            h ^= (uint64_t)row;
            h *= 1099511628211ULL;
        }
        h ^= (uint64_t)g->n;
        h *= 1099511628211ULL;
        return h;
    }

    int dense_index[MAXN_NAUTY];
    int dense_vertices[MAXN_NAUTY];
    uint64_t rem = g->vertex_mask & ADJWORD_MASK;
    uint32_t n = 0;
    while (rem) {
        int v = __builtin_ctzll(rem);
        dense_index[v] = (int)n;
        dense_vertices[n++] = v;
        rem &= rem - 1;
    }

    if (n != g->n) {
        fprintf(stderr, "Graph vertex mask/count mismatch: n=%u mask_popcount=%u\n",
                (unsigned)g->n, (unsigned)n);
        exit(1);
    }

    uint64_t h = 14695981039346656037ULL;
    for (uint32_t dense_v = 0; dense_v < n; dense_v++) {
        int v = dense_vertices[dense_v];
        uint64_t row_bits = (uint64_t)g->adj[v] & g->vertex_mask;
        AdjWord row = 0;
        while (row_bits) {
            int u = __builtin_ctzll(row_bits);
            row |= (AdjWord)(UINT64_C(1) << dense_index[u]);
            row_bits &= row_bits - 1;
        }
        rows[dense_v] = row;
        h ^= (uint64_t)row;
        h *= 1099511628211ULL;
    }
    h ^= (uint64_t)n;
    h *= 1099511628211ULL;
    return h;
}

static inline int graph_cache_slot_matches_sig(const GraphCache* cache, int slot, uint64_t key_hash,
                                               uint32_t key_n, const uint64_t* sig) {
    if (!cache->keys[slot].used || cache->keys[slot].key_hash != key_hash ||
        cache->keys[slot].key_n != key_n) {
        return 0;
    }
    const uint64_t* slot_sig = graph_cache_sig_slot(cache, slot);
    for (int i = 0; i < GRAPH_SIG_WORDS; i++) {
        if (slot_sig[i] != sig[i]) return 0;
    }
    return 1;
}

static inline int row_graph_cache_slot_matches_graph(const RowGraphCache* cache, int slot,
                                                     uint64_t key_hash, uint32_t key_n,
                                                     const Graph* g, AdjWord row_mask) {
    if (!cache->keys[slot].used || cache->keys[slot].key_hash != key_hash ||
        cache->keys[slot].key_n != key_n) {
        return 0;
    }
    const AdjWord* slot_rows = row_graph_cache_row_slot(cache, slot);
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
    for (uint32_t i = 0; i < key_n; i++) {
        if (slot_rows[i] != rows[i]) return 0;
    }
    return 1;
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

static inline void graph_cache_load_poly(const GraphCache* cache, int slot, GraphPoly* value) {
    int deg = cache->degs[slot];
    value->deg = (uint8_t)deg;
    memcpy(value->coeffs, graph_cache_coeff_slot(cache, slot),
           (size_t)(deg + 1) * sizeof(value->coeffs[0]));
}

static inline void row_graph_cache_load_poly(const RowGraphCache* cache, int slot, GraphPoly* value) {
    int deg = cache->degs[slot];
    value->deg = (uint8_t)deg;
    memcpy(value->coeffs, row_graph_cache_coeff_slot(cache, slot),
           (size_t)(deg + 1) * sizeof(value->coeffs[0]));
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
    shared->cache.degs = checked_aligned_alloc(64, sizeof(uint8_t) * size, "shared_cache_degs");
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
    free(shared->cache.degs);
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

void store_graph_cache_entry(GraphCache* cache, uint64_t key_hash, uint32_t key_n, const Graph* g,
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
    cache->degs[best_slot] = (uint8_t)value->deg;
    memcpy(graph_cache_coeff_slot(cache, best_slot), value->coeffs,
           (size_t)(value->deg + 1) * sizeof(value->coeffs[0]));
    cache->keys[best_slot].used = 1;
    graph_cache_touch_slot(cache, best_slot);
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
    for (uint32_t i = 0; i < key_n; i++) {
        slot_rows[i] = g->adj[i] & row_mask;
    }
    cache->degs[best_slot] = (uint8_t)value->deg;
    memcpy(row_graph_cache_coeff_slot(cache, best_slot), value->coeffs,
           (size_t)(value->deg + 1) * sizeof(value->coeffs[0]));
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
    for (uint32_t i = 0; i < key_n; i++) {
        slot_rows[i] = rows[i];
    }
    cache->degs[best_slot] = (uint8_t)value->deg;
    memcpy(row_graph_cache_coeff_slot(cache, best_slot), value->coeffs,
           (size_t)(value->deg + 1) * sizeof(value->coeffs[0]));
    cache->keys[best_slot].used = 1;
    row_graph_cache_touch_slot(cache, best_slot);
}

void partial_graph_reset(PartialGraphState* st) {
    st->g.n = 0;
    st->g.vertex_mask = 0;
    memset(st->g.adj, 0, sizeof(st->g.adj));
    memset(st->base, 0, sizeof(st->base));
#if RECT_COUNT_K4_FEASIBILITY
    memset(st->pair_count, 0, sizeof(st->pair_count));
    st->remaining_capacity = (uint8_t)(4 * num_row_pairs);
    st->full_pair_mask = 0;
    st->last_base = 0;
    st->last_num_new = 0;
#endif
}

static int partial_graph_append(PartialGraphState* st, int depth, int pid, const int* stack) {
    int base_new = st->g.n;
    int num_complex = partitions[pid].num_complex;
    st->base[depth] = base_new;
#if RECT_COUNT_K4_FEASIBILITY
    st->last_base = (uint8_t)base_new;
    st->last_num_new = (uint8_t)num_complex;
#endif
    st->g.n = (uint8_t)(st->g.n + num_complex);
    st->g.vertex_mask |= ((UINT64_C(1) << num_complex) - 1U) << base_new;
    for (int i = 0; i < num_complex; i++) st->g.adj[base_new + i] = 0;

    for (int i1 = 0; i1 < num_complex; i1++) {
        int u = base_new + i1;
        st->g.adj[u] |= ((uint64_t)intra_mask_get(pid, i1)) << base_new;
    }

    for (int prev = 0; prev < depth; prev++) {
        int prev_pid = stack[prev];
        int prev_base = st->base[prev];
        for (int i1 = 0; i1 < num_complex; i1++) {
            int u = base_new + i1;
            uint32_t mask = overlap_mask_get(pid, prev_pid, i1);
            if (mask == 0) continue;
            st->g.adj[u] |= ((uint64_t)mask) << prev_base;
            while (mask) {
                int i2 = __builtin_ctz(mask);
                int v = prev_base + i2;
                st->g.adj[v] |= (1ULL << u);
                mask &= mask - 1;
            }
        }
    }

#if RECT_COUNT_K4_FEASIBILITY
    uint32_t shadow = pair_shadow_mask[pid];
    while (shadow) {
        int pair = __builtin_ctz(shadow);
        st->pair_count[pair]++;
        if (st->pair_count[pair] == 4) {
            st->full_pair_mask |= (uint32_t)(1U << pair);
        }
        shadow &= shadow - 1;
    }
    st->remaining_capacity = (uint8_t)(st->remaining_capacity - pair_shadow_pairs[pid]);
#endif
    return 1;
}

static inline int partial_graph_candidate_can_fit(const PartialGraphState* st, int pid,
                                                  int cols_left) {
#if RECT_COUNT_K4_FEASIBILITY
    uint32_t shadow = pair_shadow_mask[pid];
    if (shadow & st->full_pair_mask) return 0;
    if (st->remaining_capacity <
        (int)pair_shadow_pairs[pid] + (int)suffix_min_pairs[pid] * cols_left) {
        return 0;
    }
#else
    (void)st;
    (void)pid;
    (void)cols_left;
#endif
    return 1;
}

int partial_graph_append_checked(PartialGraphState* st, int depth, int pid,
                                 const int* stack, int cols_left) {
    if (!partial_graph_candidate_can_fit(st, pid, cols_left)) return 0;
    if (!partial_graph_append(st, depth, pid, stack)) return 0;
#if RECT_COUNT_K4_FEASIBILITY
    if (!partial_graph_is_feasible(st, cols_left)) return 0;
#endif
    return 1;
}

void build_live_prefix2_tasks(PrefixId** live_i_out, PrefixId** live_j_out,
                              long long* live_count_out) {
    PrefixTaskBuffer live = {0};
    CanonState canon_state;
    CanonScratch canon_scratch;
    PartialGraphState partial_graph;
    int stack[MAX_COLS];

    prefix_task_buffer_init(&live, num_partitions);
    canon_state_init(&canon_state, perm_count);
    canon_scratch_init(&canon_scratch, perm_count);

    for (int i = 0; i < num_partitions; i++) {
        int next_stabilizer = 0;
        canon_state_reset(&canon_state, perm_count);
        partial_graph_reset(&partial_graph);
        stack[0] = i;
        if (!canon_state_prepare_push(&canon_state, i, &canon_scratch, &next_stabilizer)) {
            continue;
        }
        canon_state_commit_push(&canon_state, i, &canon_scratch, next_stabilizer);
        if (!partial_graph_append_checked(&partial_graph, 0, i, stack, g_cols - 1)) {
            canon_state_pop(&canon_state);
            continue;
        }

        for (int j = i; j < num_partitions; j++) {
            stack[1] = j;
            if (!canon_state_prepare_push(&canon_state, j, &canon_scratch, &next_stabilizer)) {
                continue;
            }
            canon_state_commit_push(&canon_state, j, &canon_scratch, next_stabilizer);
            PartialGraphState prefix_graph = partial_graph;
            if (partial_graph_append_checked(&prefix_graph, 1, j, stack, g_cols - 2)) {
                prefix_task_buffer_push2(&live, i, j);
            }
            canon_state_pop(&canon_state);
        }

        canon_state_pop(&canon_state);
    }

    canon_state_free(&canon_state);
    canon_scratch_free(&canon_scratch);
    *live_i_out = live.i;
    *live_j_out = live.j;
    *live_count_out = live.count;
}

static void solve_structure_with_row_orbit(const Graph* partial_graph, long long row_orbit,
                                           RowGraphCache* cache, RowGraphCache* raw_cache,
                                           NautyWorkspace* ws, long long* local_canon_calls,
                                           long long* local_cache_hits,
                                           long long* local_raw_cache_hits,
                                           const WeightAccum* weight_prod, long long mult_coeff,
                                           ProfileStats* profile, Poly* out_result) {
    double t0 = 0.0;
    if (PROFILE_BUILD && profile) {
        profile->solve_structure_calls++;
        t0 = omp_get_wtime();
    }
#if RECT_COUNT_K4
    unsigned __int128 structure_weight =
        (*weight_prod) * (WeightAccum)mult_coeff * (WeightAccum)row_orbit;
    if (PROFILE_BUILD && profile) profile->build_weight_time += omp_get_wtime() - t0;
    if (structure_weight == 0) {
        poly_zero(out_result);
        return;
    }
#else
    Poly weight;
    poly_scale_ref(weight_prod, mult_coeff * row_orbit, &weight);
    if (PROFILE_BUILD && profile) profile->build_weight_time += omp_get_wtime() - t0;
#endif
    GraphPoly graph_poly_small;
    solve_graph_poly(partial_graph, cache, raw_cache, ws,
                     local_canon_calls, local_cache_hits, local_raw_cache_hits,
                     profile, &graph_poly_small);
#if RECT_COUNT_K4
    weight_accum_scale_to_poly(weight_prod, mult_coeff, row_orbit,
                               graph_poly_get_count4(&graph_poly_small), out_result);
#else
    poly_mul_graph_ref(&weight, &graph_poly_small, out_result);
#endif
}

static void solve_structure(const Graph* partial_graph, CanonState* canon_state,
                            RowGraphCache* cache, RowGraphCache* raw_cache, NautyWorkspace* ws,
                            long long* local_canon_calls, long long* local_cache_hits,
                            long long* local_raw_cache_hits, const WeightAccum* weight_prod,
                            long long mult_coeff, ProfileStats* profile, Poly* out_result) {
    long long row_orbit = get_orbit_multiplier_state(canon_state);
    solve_structure_with_row_orbit(partial_graph, row_orbit, cache, raw_cache, ws,
                                   local_canon_calls, local_cache_hits, local_raw_cache_hits,
                                   weight_prod, mult_coeff, profile, out_result);
}

void dfs(int depth, int min_idx, int* stack, CanonState* canon_state,
         const PartialGraphState* partial_graph, RowGraphCache* cache,
         RowGraphCache* raw_cache, NautyWorkspace* ws, Poly* local_total,
         long long* local_canon_calls, long long* local_cache_hits,
         long long* local_raw_cache_hits, const WeightAccum* weight_prod,
         long long mult_coeff, int run_len, ProfileStats* profile,
         CanonScratch* canon_scratch);

static void dfs_fast_orbit(int depth, int min_idx, int* stack, CanonState* canon_state,
                           const PartialGraphState* partial_graph, RowGraphCache* cache,
                           RowGraphCache* raw_cache, NautyWorkspace* ws, Poly* local_total,
                           long long* local_canon_calls, long long* local_cache_hits,
                           long long* local_raw_cache_hits, const WeightAccum* weight_prod,
                           long long mult_coeff, int run_len, ProfileStats* profile,
                           CanonScratch* canon_scratch) {
    int next_stabilizer = 0;
    int is_terminal = (depth + 1 == g_cols);
    int cols_left = g_cols - depth - 1;
    uint64_t orbit_mark_bits[REP_ORBIT_MARK_WORDS];
    memset(orbit_mark_bits, 0, sizeof(orbit_mark_bits));

    for (int i = min_idx; i < num_partitions; i++) {
        if (orbit_mark_bit_test(orbit_mark_bits, i)) {
            continue;
        }
        canon_state_mark_orbit_nonreps(canon_state, min_idx, i, orbit_mark_bits);
        if (!partial_graph_candidate_can_fit(partial_graph, i, cols_left)) {
            continue;
        }
        int ok_prepare = is_terminal
            ? canon_state_prepare_terminal(canon_state, i, &next_stabilizer)
            : canon_state_prepare_push(canon_state, i, canon_scratch, &next_stabilizer);
        if (!ok_prepare) {
            continue;
        }

        stack[depth] = i;
        WeightAccum next_weight_prod;
        weight_accum_mul_partition(weight_prod, i, &next_weight_prod);
        long long next_mult_coeff = mult_coeff * (depth + 1);
        int next_run_len = 1;
        if (depth > 0 && i == stack[depth - 1]) {
            next_run_len = run_len + 1;
            next_mult_coeff /= next_run_len;
        }
        PartialGraphState next_graph = *partial_graph;
        if (!is_terminal) {
            canon_state_commit_push(canon_state, i, canon_scratch, next_stabilizer);
        }
        int ok = partial_graph_append_checked(&next_graph, depth, i, stack, cols_left);
        if (ok) {
            if (is_terminal) {
                long long row_orbit = factorial[g_rows] / next_stabilizer;
                Poly res;
                solve_structure_with_row_orbit(&next_graph.g, row_orbit, cache, raw_cache, ws,
                                               local_canon_calls, local_cache_hits,
                                               local_raw_cache_hits, &next_weight_prod,
                                               next_mult_coeff, profile, &res);
                poly_accumulate_checked(local_total, &res);
            } else {
                dfs(depth + 1, i, stack, canon_state, &next_graph, cache, raw_cache, ws, local_total,
                    local_canon_calls, local_cache_hits, local_raw_cache_hits, &next_weight_prod,
                    next_mult_coeff, next_run_len, profile, canon_scratch);
            }
        }
        if (!is_terminal) {
            canon_state_pop(canon_state);
        }
    }
}

static void dfs_fast_rep(int depth, int min_idx, int* stack, CanonState* canon_state,
                         const PartialGraphState* partial_graph, RowGraphCache* cache,
                         RowGraphCache* raw_cache, NautyWorkspace* ws, Poly* local_total,
                         long long* local_canon_calls, long long* local_cache_hits,
                         long long* local_raw_cache_hits, const WeightAccum* weight_prod,
                         long long mult_coeff, int run_len, ProfileStats* profile,
                         CanonScratch* canon_scratch) {
    int next_stabilizer = 0;
    int is_terminal = (depth + 1 == g_cols);
    int cols_left = g_cols - depth - 1;

    for (int i = min_idx; i < num_partitions; i++) {
        if (!partial_graph_candidate_can_fit(partial_graph, i, cols_left)) {
            continue;
        }
        if (!canon_state_partition_is_rep(canon_state, min_idx, i)) {
            continue;
        }
        int ok_prepare = is_terminal
            ? canon_state_prepare_terminal(canon_state, i, &next_stabilizer)
            : canon_state_prepare_push(canon_state, i, canon_scratch, &next_stabilizer);
        if (!ok_prepare) {
            continue;
        }

        stack[depth] = i;
        WeightAccum next_weight_prod;
        weight_accum_mul_partition(weight_prod, i, &next_weight_prod);
        long long next_mult_coeff = mult_coeff * (depth + 1);
        int next_run_len = 1;
        if (depth > 0 && i == stack[depth - 1]) {
            next_run_len = run_len + 1;
            next_mult_coeff /= next_run_len;
        }
        PartialGraphState next_graph = *partial_graph;
        if (!is_terminal) {
            canon_state_commit_push(canon_state, i, canon_scratch, next_stabilizer);
        }
        int ok = partial_graph_append_checked(&next_graph, depth, i, stack, cols_left);
        if (ok) {
            if (is_terminal) {
                long long row_orbit = factorial[g_rows] / next_stabilizer;
                Poly res;
                solve_structure_with_row_orbit(&next_graph.g, row_orbit, cache, raw_cache, ws,
                                               local_canon_calls, local_cache_hits,
                                               local_raw_cache_hits, &next_weight_prod,
                                               next_mult_coeff, profile, &res);
                poly_accumulate_checked(local_total, &res);
            } else {
                dfs(depth + 1, i, stack, canon_state, &next_graph, cache, raw_cache, ws, local_total,
                    local_canon_calls, local_cache_hits, local_raw_cache_hits, &next_weight_prod,
                    next_mult_coeff, next_run_len, profile, canon_scratch);
            }
        }
        if (!is_terminal) {
            canon_state_pop(canon_state);
        }
    }
}

void dfs(int depth, int min_idx, int* stack, CanonState* canon_state, const PartialGraphState* partial_graph,
         RowGraphCache* cache, RowGraphCache* raw_cache, NautyWorkspace* ws, Poly* local_total,
         long long* local_canon_calls, long long* local_cache_hits,
         long long* local_raw_cache_hits, const WeightAccum* weight_prod, long long mult_coeff,
         int run_len, ProfileStats* profile, CanonScratch* canon_scratch) {
    if (depth == g_cols) {
        Poly res;
        solve_structure(&partial_graph->g, canon_state, cache, raw_cache, ws,
                        local_canon_calls, local_cache_hits, local_raw_cache_hits,
                        weight_prod, mult_coeff, profile, &res);
        poly_accumulate_checked(local_total, &res);
        return;
    }

#if !RECT_PROFILE
    if (canon_state_use_orbit_marking(canon_state)) {
        dfs_fast_orbit(depth, min_idx, stack, canon_state, partial_graph, cache, raw_cache, ws,
                       local_total, local_canon_calls, local_cache_hits, local_raw_cache_hits,
                       weight_prod, mult_coeff, run_len, profile, canon_scratch);
    } else {
        dfs_fast_rep(depth, min_idx, stack, canon_state, partial_graph, cache, raw_cache, ws,
                     local_total, local_canon_calls, local_cache_hits, local_raw_cache_hits,
                     weight_prod, mult_coeff, run_len, profile, canon_scratch);
    }
    return;
#endif

    int next_stabilizer = 0;
    uint64_t orbit_mark_bits[REP_ORBIT_MARK_WORDS];
    int use_orbit_marking = canon_state_use_orbit_marking(canon_state);
    if (use_orbit_marking) memset(orbit_mark_bits, 0, sizeof(orbit_mark_bits));
    for (int i = min_idx; i < num_partitions; i++) {
        double t0 = 0.0;
        int is_terminal = (depth + 1 == g_cols);
        int cols_left = g_cols - depth - 1;
        if (use_orbit_marking) {
            if (orbit_mark_bit_test(orbit_mark_bits, i)) {
                continue;
            }
            canon_state_mark_orbit_nonreps(canon_state, min_idx, i, orbit_mark_bits);
        }
        if (!partial_graph_candidate_can_fit(partial_graph, i, cols_left)) {
            continue;
        }
        if (!use_orbit_marking && !canon_state_partition_is_rep(canon_state, min_idx, i)) {
            continue;
        }
        if (PROFILE_BUILD && profile) {
            profile->canon_prepare_calls++;
            profile->canon_prepare_calls_by_depth[depth]++;
            t0 = omp_get_wtime();
        }
        int ok_prepare = is_terminal
            ? canon_state_prepare_terminal(canon_state, i, &next_stabilizer)
            : canon_state_prepare_push(canon_state, i, canon_scratch, &next_stabilizer);
        if (!ok_prepare) {
            if (PROFILE_BUILD && profile) profile->canon_prepare_time += omp_get_wtime() - t0;
            continue;
        }
        if (PROFILE_BUILD && profile) {
            profile->canon_prepare_time += omp_get_wtime() - t0;
            profile->canon_prepare_accepts++;
            profile->canon_prepare_accepts_by_depth[depth]++;
            profile->stabilizer_sum_by_depth[depth] += next_stabilizer;
            if (!is_terminal) {
                profile->canon_commit_calls++;
                t0 = omp_get_wtime();
            }
        }
        stack[depth] = i;
        WeightAccum next_weight_prod;
        weight_accum_mul_partition(weight_prod, i, &next_weight_prod);
        long long next_mult_coeff = mult_coeff * (depth + 1);
        int next_run_len = 1;
        if (depth > 0 && i == stack[depth - 1]) {
            next_run_len = run_len + 1;
            next_mult_coeff /= next_run_len;
        }
        PartialGraphState next_graph = *partial_graph;
        if (!is_terminal) {
            canon_state_commit_push(canon_state, i, canon_scratch, next_stabilizer);
            if (PROFILE_BUILD && profile) profile->canon_commit_time += omp_get_wtime() - t0;
        }
        if (PROFILE_BUILD && profile) {
            profile->partial_append_calls++;
            t0 = omp_get_wtime();
        }
        int ok = partial_graph_append_checked(&next_graph, depth, i, stack, cols_left);
        if (PROFILE_BUILD && profile) profile->partial_append_time += omp_get_wtime() - t0;
        if (ok) {
            if (is_terminal) {
                long long row_orbit = factorial[g_rows] / next_stabilizer;
                Poly res;
                solve_structure_with_row_orbit(&next_graph.g, row_orbit, cache, raw_cache, ws,
                                               local_canon_calls, local_cache_hits,
                                               local_raw_cache_hits, &next_weight_prod,
                                               next_mult_coeff, profile, &res);
                poly_accumulate_checked(local_total, &res);
            } else {
                dfs(depth + 1, i, stack, canon_state, &next_graph, cache, raw_cache, ws, local_total,
                    local_canon_calls, local_cache_hits, local_raw_cache_hits, &next_weight_prod,
                    next_mult_coeff, next_run_len, profile, canon_scratch);
            }
        }
        if (!is_terminal) {
            canon_state_pop(canon_state);
        }
    }
}

PolyCoeff poly_eval(Poly p, long long x) {
    PolyCoeff res = 0;
    PolyCoeff xp = 1;
    for(int i=0; i<=p.deg; i++) {
        res += p.coeffs[i] * xp;
        xp *= x;
    }
    return res;
}

void execute_prefix2_fixed_batch(PrefixId i, const PrefixId* js, const long long* ps, int count,
                                 RowGraphCache* cache, RowGraphCache* raw_cache, NautyWorkspace* ws,
                                 CanonState* canon_state, CanonScratch* canon_scratch,
                                 PartialGraphState* partial_graph, int* stack, Poly* local_total,
                                 long long* local_canon_calls, long long* local_cache_hits,
                                 long long* local_raw_cache_hits, ProfileStats* profile,
                                 long long total_tasks, long long progress_report_step,
                                 double start_time, long long* pending_completed,
                                 TaskTimingStats* task_timing) {
    double t0 = 0.0;
    int next_stabilizer = 0;

    canon_state_reset(canon_state, perm_count);
    partial_graph_reset(partial_graph);

    stack[0] = (int)i;
    if (PROFILE_BUILD) {
        profile->canon_prepare_calls++;
        profile->canon_prepare_calls_by_depth[0]++;
        t0 = omp_get_wtime();
    }
    if (!canon_state_prepare_push(canon_state, (int)i, canon_scratch, &next_stabilizer)) {
        if (PROFILE_BUILD) profile->canon_prepare_time += omp_get_wtime() - t0;
        return;
    }
    if (PROFILE_BUILD) {
        profile->canon_prepare_time += omp_get_wtime() - t0;
        profile->canon_prepare_accepts++;
        profile->canon_prepare_accepts_by_depth[0]++;
        profile->stabilizer_sum_by_depth[0] += next_stabilizer;
        profile->canon_commit_calls++;
        t0 = omp_get_wtime();
    }
    canon_state_commit_push(canon_state, (int)i, canon_scratch, next_stabilizer);
    if (PROFILE_BUILD) profile->canon_commit_time += omp_get_wtime() - t0;
    if (PROFILE_BUILD) {
        profile->partial_append_calls++;
        t0 = omp_get_wtime();
    }
    if (!partial_graph_append_checked(partial_graph, 0, (int)i, stack, g_cols - 1)) {
        if (PROFILE_BUILD) profile->partial_append_time += omp_get_wtime() - t0;
        canon_state_pop(canon_state);
        return;
    }
    if (PROFILE_BUILD) profile->partial_append_time += omp_get_wtime() - t0;

    for (int idx = 0; idx < count; idx++) {
        long long p = ps[idx];
        double task_t0 = PROFILE_BUILD ? omp_get_wtime() : 0.0;
        PrefixId j = js[idx];
        Poly task_total;
        poly_zero(&task_total);

        stack[1] = (int)j;
        if (PROFILE_BUILD) {
            profile->canon_prepare_calls++;
            profile->canon_prepare_calls_by_depth[1]++;
            t0 = omp_get_wtime();
        }
        if (!canon_state_prepare_push(canon_state, (int)j, canon_scratch, &next_stabilizer)) {
            if (PROFILE_BUILD) profile->canon_prepare_time += omp_get_wtime() - t0;
            complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                          pending_completed, task_timing, p, task_t0);
            continue;
        }
        if (PROFILE_BUILD) {
            profile->canon_prepare_time += omp_get_wtime() - t0;
            profile->canon_prepare_accepts++;
            profile->canon_prepare_accepts_by_depth[1]++;
            profile->stabilizer_sum_by_depth[1] += next_stabilizer;
            profile->canon_commit_calls++;
            t0 = omp_get_wtime();
        }
        canon_state_commit_push(canon_state, (int)j, canon_scratch, next_stabilizer);
        if (PROFILE_BUILD) profile->canon_commit_time += omp_get_wtime() - t0;
        PartialGraphState prefix_graph = *partial_graph;
        if (PROFILE_BUILD) {
            profile->partial_append_calls++;
            t0 = omp_get_wtime();
        }
        int ok = partial_graph_append_checked(&prefix_graph, 1, (int)j, stack, g_cols - 2);
        if (PROFILE_BUILD) profile->partial_append_time += omp_get_wtime() - t0;
        if (ok) {
            WeightAccum prefix_weight;
            weight_accum_from_partition((int)i, &prefix_weight);
            weight_accum_mul_partition(&prefix_weight, (int)j, &prefix_weight);
            long long prefix_mult = (i == j) ? 1 : 2;
            int prefix_run = (i == j) ? 2 : 1;
            dfs(2, (int)j, stack, canon_state, &prefix_graph, cache, raw_cache, ws, &task_total,
                local_canon_calls, local_cache_hits, local_raw_cache_hits,
                &prefix_weight, prefix_mult, prefix_run, profile, canon_scratch);
            poly_accumulate_checked(local_total, &task_total);
        }
        canon_state_pop(canon_state);
        complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                      pending_completed, task_timing, p, task_t0);
    }

    canon_state_pop(canon_state);
}

static int replay_local_task_prefix(const LocalTask* task, WorkerCtx* ctx,
                                    WeightAccum* weight_prod, long long* mult_coeff,
                                    int* run_len, int* min_idx) {
    int next_stabilizer = 0;
    int prev_pid = -1;

    canon_state_reset(&ctx->canon_state, perm_count);
    partial_graph_reset(&ctx->partial_graph);
    weight_accum_one(weight_prod);
    *mult_coeff = 1;
    *run_len = 0;
    *min_idx = 0;

    for (int depth = 0; depth < task->depth; depth++) {
        int pid = (int)task->prefix[depth];
        double t0 = 0.0;
        ctx->stack[depth] = pid;

        if (PROFILE_BUILD) {
            tls_profile->canon_prepare_calls++;
            tls_profile->canon_prepare_calls_by_depth[depth]++;
            t0 = omp_get_wtime();
        }
        if (!canon_state_prepare_push(&ctx->canon_state, pid,
                                      &ctx->canon_scratch, &next_stabilizer)) {
            if (PROFILE_BUILD) tls_profile->canon_prepare_time += omp_get_wtime() - t0;
            return 0;
        }
        if (PROFILE_BUILD) {
            tls_profile->canon_prepare_time += omp_get_wtime() - t0;
            tls_profile->canon_prepare_accepts++;
            tls_profile->canon_prepare_accepts_by_depth[depth]++;
            tls_profile->stabilizer_sum_by_depth[depth] += next_stabilizer;
            tls_profile->canon_commit_calls++;
            t0 = omp_get_wtime();
        }
        canon_state_commit_push(&ctx->canon_state, pid,
                                &ctx->canon_scratch, next_stabilizer);
        if (PROFILE_BUILD) tls_profile->canon_commit_time += omp_get_wtime() - t0;

        if (PROFILE_BUILD) {
            tls_profile->partial_append_calls++;
            t0 = omp_get_wtime();
        }
        if (!partial_graph_append_checked(&ctx->partial_graph, depth, pid, ctx->stack,
                                          g_cols - depth - 1)) {
            if (PROFILE_BUILD) tls_profile->partial_append_time += omp_get_wtime() - t0;
            canon_state_pop(&ctx->canon_state);
            return 0;
        }
        if (PROFILE_BUILD) tls_profile->partial_append_time += omp_get_wtime() - t0;

        weight_accum_mul_partition(weight_prod, pid, weight_prod);

        long long next_mult = (*mult_coeff) * (depth + 1);
        int next_run = 1;
        if (depth > 0 && pid == prev_pid) {
            next_run = *run_len + 1;
            next_mult /= next_run;
        }
        *mult_coeff = next_mult;
        *run_len = next_run;
        *min_idx = pid;
        prev_pid = pid;
    }

    return 1;
}

static void dfs_runtime_split_local(int depth, int start_pid, int end_pid, long long root_id, WorkerCtx* ctx,
                                    Poly* local_total, const WeightAccum* weight_prod,
                                    long long mult_coeff, int run_len, ProfileStats* profile,
                                    LocalTaskQueue* queue) {
    if (depth == g_cols) {
        Poly res;
        solve_structure(&ctx->partial_graph.g, &ctx->canon_state,
                        &ctx->cache, &ctx->raw_cache, &ctx->ws,
                        &ctx->local_canon_calls, &ctx->local_cache_hits,
                        &ctx->local_raw_cache_hits, weight_prod, mult_coeff, profile, &res);
        poly_accumulate_checked(local_total, &res);
        return;
    }

    int next_stabilizer = 0;
    int local_end = end_pid;
    int rep_min_idx = (depth > 0) ? ctx->stack[depth - 1] : 0;
    uint64_t orbit_mark_bits[REP_ORBIT_MARK_WORDS];
    int use_orbit_marking = canon_state_use_orbit_marking(&ctx->canon_state);
    if (use_orbit_marking) {
        memset(orbit_mark_bits, 0, sizeof(orbit_mark_bits));
        if (start_pid > rep_min_idx) {
            canon_state_seed_orbit_marks(&ctx->canon_state, rep_min_idx, start_pid, orbit_mark_bits);
        }
    }
    for (int pid = start_pid; pid < local_end; pid++) {
        if (g_adaptive_work_budget > 0 &&
            tls_adaptive_work_counter &&
            *tls_adaptive_work_counter >= g_adaptive_work_budget &&
            depth < g_adaptive_max_depth &&
            (local_end - pid) >= 2 &&
            atomic_load_explicit(&queue->idle_threads, memory_order_relaxed) > 0) {
            LocalTask continuation = {0};
            local_task_from_stack(&continuation, root_id, depth, ctx->stack);
            continuation.lo = (PrefixId)pid;
            continuation.hi = (PrefixId)local_end;
            if (local_queue_try_push(queue, &continuation)) {
                atomic_fetch_add_explicit(&queue->work_budget_continuations, 1, memory_order_relaxed);
                return;
            }
        }

        if ((depth + 1) <= g_adaptive_max_depth && depth >= 3 && depth <= 5 &&
            (local_end - pid) >= 16 &&
            atomic_load_explicit(&queue->idle_threads, memory_order_relaxed) > 0) {
            int mid = pid + (local_end - pid) / 2;
            LocalTask range_task = {0};
            local_task_from_stack(&range_task, root_id, depth, ctx->stack);
            range_task.lo = (PrefixId)mid;
            range_task.hi = (PrefixId)local_end;
            if (local_queue_try_push(queue, &range_task)) {
                atomic_fetch_add_explicit(&queue->donated_tasks, 1, memory_order_relaxed);
                local_end = mid;
            }
        }

        double t0 = 0.0;
        int is_terminal = (depth + 1 == g_cols);
        if (use_orbit_marking) {
            if (orbit_mark_bit_test(orbit_mark_bits, pid)) {
                continue;
            }
            canon_state_mark_orbit_nonreps(&ctx->canon_state, rep_min_idx, pid, orbit_mark_bits);
        } else if (!canon_state_partition_is_rep(&ctx->canon_state, rep_min_idx, pid)) {
            continue;
        }
        ctx->stack[depth] = pid;
        if (PROFILE_BUILD) {
            tls_profile->canon_prepare_calls++;
            tls_profile->canon_prepare_calls_by_depth[depth]++;
            t0 = omp_get_wtime();
        }
        int ok_prepare = is_terminal
            ? canon_state_prepare_terminal(&ctx->canon_state, pid, &next_stabilizer)
            : canon_state_prepare_push(&ctx->canon_state, pid, &ctx->canon_scratch, &next_stabilizer);
        if (!ok_prepare) {
            if (PROFILE_BUILD) tls_profile->canon_prepare_time += omp_get_wtime() - t0;
            continue;
        }
        if (PROFILE_BUILD) {
            tls_profile->canon_prepare_time += omp_get_wtime() - t0;
            tls_profile->canon_prepare_accepts++;
            tls_profile->canon_prepare_accepts_by_depth[depth]++;
            tls_profile->stabilizer_sum_by_depth[depth] += next_stabilizer;
            if (!is_terminal) {
                tls_profile->canon_commit_calls++;
                t0 = omp_get_wtime();
            }
        }

        PartialGraphState saved_graph = ctx->partial_graph;
        if (!is_terminal) {
            canon_state_commit_push(&ctx->canon_state, pid, &ctx->canon_scratch, next_stabilizer);
            if (PROFILE_BUILD) tls_profile->canon_commit_time += omp_get_wtime() - t0;
        }

        if (PROFILE_BUILD) {
            tls_profile->partial_append_calls++;
            t0 = omp_get_wtime();
        }
        if (partial_graph_append_checked(&ctx->partial_graph, depth, pid, ctx->stack,
                                         g_cols - depth - 1)) {
            if (PROFILE_BUILD) tls_profile->partial_append_time += omp_get_wtime() - t0;
            WeightAccum next_weight_prod;
            weight_accum_mul_partition(weight_prod, pid, &next_weight_prod);
            long long next_mult_coeff = mult_coeff * (depth + 1);
            int next_run_len = 1;
            if (depth > 0 && pid == ctx->stack[depth - 1]) {
                next_run_len = run_len + 1;
                next_mult_coeff /= next_run_len;
            }

            if (is_terminal) {
                long long row_orbit = factorial[g_rows] / next_stabilizer;
                Poly res;
                solve_structure_with_row_orbit(&ctx->partial_graph.g, row_orbit, &ctx->cache,
                                               &ctx->raw_cache, &ctx->ws,
                                               &ctx->local_canon_calls, &ctx->local_cache_hits,
                                               &ctx->local_raw_cache_hits, &next_weight_prod,
                                               next_mult_coeff, profile, &res);
                poly_accumulate_checked(local_total, &res);
            } else {
                dfs_runtime_split_local(depth + 1, pid, num_partitions, root_id, ctx, local_total,
                                        &next_weight_prod, next_mult_coeff, next_run_len,
                                        profile, queue);
            }
        } else if (PROFILE_BUILD) {
            tls_profile->partial_append_time += omp_get_wtime() - t0;
        }

        ctx->partial_graph = saved_graph;
        if (!is_terminal) {
            canon_state_pop(&ctx->canon_state);
        }
    }
}

void execute_local_runtime_task(const LocalTask* task, WorkerCtx* ctx, Poly* thread_total,
                                LocalTaskQueue* queue, ProfileStats* profile,
                                long long total_tasks, long long report_step,
                                double start_time, long long* pending_completed,
                                TaskTimingStats* task_timing,
                                QueueSubtaskTimingStats* queue_subtask_stats) {
    WeightAccum weight_prod;
    long long mult_coeff = 1;
    int run_len = 0;
    int min_idx = 0;
    double subtask_t0 = PROFILE_BUILD ? omp_get_wtime() : 0.0;
    long long solve_graph_before = PROFILE_BUILD ? profile->solve_graph_calls : 0;
    long long nauty_before = PROFILE_BUILD ? profile->nauty_calls : 0;
    long long adaptive_work_counter = 0;
    GraphHardStats subtask_hard = {0};
    GraphHardStats* prev_hard_stats = tls_hard_graph_stats;
    if (PROFILE_BUILD) tls_hard_graph_stats = &subtask_hard;
    long long* prev_work_counter = tls_adaptive_work_counter;
    if (g_adaptive_work_budget > 0) tls_adaptive_work_counter = &adaptive_work_counter;

    if (replay_local_task_prefix(task, ctx, &weight_prod, &mult_coeff, &run_len, &min_idx)) {
        if (task->depth == g_cols) {
            Poly res;
            solve_structure(&ctx->partial_graph.g, &ctx->canon_state,
                            &ctx->cache, &ctx->raw_cache, &ctx->ws,
                            &ctx->local_canon_calls, &ctx->local_cache_hits,
                            &ctx->local_raw_cache_hits, &weight_prod, mult_coeff, profile, &res);
            poly_accumulate_checked(thread_total, &res);
        } else {
            int start_pid = (int)task->lo;
            int end_pid = (int)task->hi;
            if (start_pid < min_idx) start_pid = min_idx;
            if (end_pid > num_partitions) end_pid = num_partitions;
            dfs_runtime_split_local(task->depth, start_pid, end_pid, task->root_id, ctx, thread_total,
                                    &weight_prod, mult_coeff, run_len, profile, queue);
        }

        for (int depth = (int)task->depth - 1; depth >= 0; depth--) {
            canon_state_pop(&ctx->canon_state);
        }
    }
    tls_hard_graph_stats = prev_hard_stats;
    tls_adaptive_work_counter = prev_work_counter;

    if (PROFILE_BUILD && queue_subtask_stats && task->depth <= MAX_COLS) {
        double elapsed = omp_get_wtime() - subtask_t0;
        long long solve_graph_delta = profile->solve_graph_calls - solve_graph_before;
        long long nauty_delta = profile->nauty_calls - nauty_before;
        queue_subtask_record(&queue_subtask_stats[task->depth], task, elapsed,
                             solve_graph_delta, nauty_delta,
                             subtask_hard.hard_graph_nodes, subtask_hard.max_n,
                             subtask_hard.max_degree);
        if (queue) {
            local_queue_record_profile(queue, task, elapsed, solve_graph_delta, nauty_delta,
                                       subtask_hard.hard_graph_nodes, subtask_hard.max_n,
                                       subtask_hard.max_degree);
        }
    }

    local_queue_finish_item(queue, task->root_id, total_tasks, report_step, start_time,
                            pending_completed, task_timing);
}
