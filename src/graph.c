#include "partition_poly.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

// --- GRAPH CANONICALISATION AND LOOKUPS ---

int g_small_graph_lookup_ready = 0;
double g_small_graph_lookup_init_time = 0.0;
int g_small_graph_lookup_loaded_from_file = 0;
int32_t* g_small_graph_lookup_coeffs[SMALL_GRAPH_LOOKUP_MAX_N + 1] = {0};
uint8_t* g_small_graph_lookup_x_pows[SMALL_GRAPH_LOOKUP_MAX_N + 1] = {0};
#if RECT_COUNT_K4
uint16_t* g_small_graph_lookup_count4[SMALL_GRAPH_LOOKUP_MAX_N + 1] = {0};
#endif
uint8_t g_small_graph_edge_u[SMALL_GRAPH_LOOKUP_MAX_N + 1][21];
uint8_t g_small_graph_edge_v[SMALL_GRAPH_LOOKUP_MAX_N + 1][21];
uint32_t g_small_graph_graph_count[SMALL_GRAPH_LOOKUP_MAX_N + 1] = {0};
uint8_t g_small_graph_edge_count[SMALL_GRAPH_LOOKUP_MAX_N + 1] = {0};
uint32_t g_connected_canon_lookup_count = 0;
int g_connected_canon_lookup_ready = 0;
int g_connected_canon_lookup_loaded = 0;
int g_connected_canon_lookup_n = 0;
double g_connected_canon_lookup_load_time = 0.0;
static const uint8_t* g_connected_canon_lookup_map = NULL;
static size_t g_connected_canon_lookup_map_len = 0;
static const uint8_t* g_connected_canon_lookup_entries = NULL;
static size_t g_connected_canon_lookup_entry_size = 0;

static inline uint64_t bit_reverse64(uint64_t x) {
    x = ((x & 0x5555555555555555ULL) << 1) | ((x >> 1) & 0x5555555555555555ULL);
    x = ((x & 0x3333333333333333ULL) << 2) | ((x >> 2) & 0x3333333333333333ULL);
    x = ((x & 0x0f0f0f0f0f0f0f0fULL) << 4) | ((x >> 4) & 0x0f0f0f0f0f0f0f0fULL);
    x = ((x & 0x00ff00ff00ff00ffULL) << 8) | ((x >> 8) & 0x00ff00ff00ff00ffULL);
    x = ((x & 0x0000ffff0000ffffULL) << 16) | ((x >> 16) & 0x0000ffff0000ffffULL);
    return (x << 32) | (x >> 32);
}

static inline setword pack_row_to_nauty1(uint64_t row_bits, int n) {
    if (n < 64) row_bits &= (1ULL << n) - 1ULL;
    return (setword)bit_reverse64(row_bits);
}

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

static inline uint64_t graph_pack_upper_mask_from_dense_rows(uint32_t n, const AdjWord* rows) {
    uint64_t mask = 0;
    int bit = 0;
    for (uint32_t j = 1; j < n; j++) {
        for (uint32_t i = 0; i < j; i++, bit++) {
            if (((uint64_t)rows[i] >> j) & 1ULL) mask |= 1ULL << bit;
        }
    }
    return mask;
}

static uint64_t graph_extract_dense_rows_from_nauty(graph* cg, int m, uint32_t n, Graph* dst,
                                                    uint64_t* upper_mask_out) {
    uint64_t mask = graph_row_mask((int)n);
    uint64_t h = 14695981039346656037ULL;
    uint64_t upper_mask = 0;
    int pack_upper = upper_mask_out && ((uint64_t)n * (uint64_t)(n - 1) / 2ULL) <= 64ULL;
    dst->n = (uint8_t)n;
    dst->vertex_mask = mask;
    h ^= mask;
    h *= 1099511628211ULL;

    if (m == 1) {
        for (uint32_t i = 0; i < n; i++) {
            AdjWord row = (AdjWord)(bit_reverse64((uint64_t)GRAPHROW(cg, (int)i, 1)[0]) & mask);
            dst->adj[i] = row;
            h ^= (uint64_t)row;
            h *= 1099511628211ULL;
        }
        if (pack_upper) {
            upper_mask = graph_pack_upper_mask_from_dense_rows(n, dst->adj);
        }
        h ^= (uint64_t)n;
        h *= 1099511628211ULL;
        if (upper_mask_out) *upper_mask_out = upper_mask;
        return h;
    }

    for (uint32_t i = 0; i < n; i++) {
        AdjWord row = 0;
        graph* row_words = GRAPHROW(cg, (int)i, m);
        for (uint32_t j = 0; j < n; j++) {
            if (ISELEMENT(row_words, (int)j)) {
                row |= (AdjWord)(UINT64_C(1) << j);
                if (pack_upper && j > i) {
                    upper_mask |= UINT64_C(1) << (((uint64_t)j * (uint64_t)(j - 1) / 2ULL) + i);
                }
            }
        }
        dst->adj[i] = row & (AdjWord)mask;
        h ^= (uint64_t)dst->adj[i];
        h *= 1099511628211ULL;
    }
    h ^= (uint64_t)n;
    h *= 1099511628211ULL;
    if (upper_mask_out) *upper_mask_out = upper_mask;
    return h;
}

uint64_t get_canonical_graph_from_dense_rows_hashed(int n, const AdjWord* rows, Graph* canon,
                                                    NautyWorkspace* ws,
                                                    ProfileStats* profile,
                                                    uint64_t* upper_mask_out) {
    double total_t0 = 0.0;
    double phase_t0 = 0.0;

    if (n == 0) {
        canon->n = 0;
        canon->vertex_mask = 0;
        memset(canon->adj, 0, sizeof(canon->adj));
        if (upper_mask_out) *upper_mask_out = 0;
        return hash_graph(canon);
    }

    if (n == 1) {
        canon->n = 1;
        canon->vertex_mask = 1;
        canon->adj[0] = 0;
        if (upper_mask_out) *upper_mask_out = 0;
        return hash_graph(canon);
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
            uint64_t upper = (uint64_t)rows[i] & ~graph_row_mask(i + 1);
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

    DEFAULTOPTIONS_GRAPH(options);
    options.getcanon = TRUE;
    options.defaultptn = FALSE;

    statsblk stats;

    if (PROFILE_BUILD && profile) phase_t0 = omp_get_wtime();
    densenauty(ng, lab, ptn, orbits, &options, &stats, m, n, cg);
    if (PROFILE_BUILD && profile) {
        profile->nauty_calls++;
        profile->nauty_time += omp_get_wtime() - phase_t0;
        phase_t0 = omp_get_wtime();
    }

    uint64_t canon_hash =
        graph_extract_dense_rows_from_nauty(cg, m, (uint32_t)n, canon, upper_mask_out);
    if (PROFILE_BUILD && profile) {
        profile->get_canonical_graph_rebuild_time += omp_get_wtime() - phase_t0;
        profile->get_canonical_graph_time += omp_get_wtime() - total_t0;
    }
    return canon_hash;
}

void get_canonical_graph_from_dense_rows(int n, const AdjWord* rows, Graph* canon,
                                         NautyWorkspace* ws, ProfileStats* profile) {
    (void)get_canonical_graph_from_dense_rows_hashed(n, rows, canon, ws, profile, NULL);
}

uint64_t get_canonical_graph_hashed(Graph* g, Graph* canon, NautyWorkspace* ws,
                                    ProfileStats* profile, uint64_t* upper_mask_out) {
    AdjWord rows[MAXN_NAUTY];
    double phase_t0 = 0.0;
    if (PROFILE_BUILD && profile) phase_t0 = omp_get_wtime();
    int n = (int)graph_build_dense_rows(g, rows);
    if (PROFILE_BUILD && profile) {
        double dt = omp_get_wtime() - phase_t0;
        profile->get_canonical_graph_dense_rows_time += dt;
        profile->get_canonical_graph_time += dt;
    }
    return get_canonical_graph_from_dense_rows_hashed(n, rows, canon, ws, profile,
                                                      upper_mask_out);
}

void get_canonical_graph(Graph* g, Graph* canon, NautyWorkspace* ws, ProfileStats* profile) {
    (void)get_canonical_graph_hashed(g, canon, ws, profile, NULL);
}

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
        g_small_graph_lookup_x_pows[n] =
            checked_calloc((size_t)graph_count, sizeof(uint8_t), "small_graph_lookup_x_pows");
#if RECT_COUNT_K4
        g_small_graph_lookup_count4[n] =
            checked_calloc((size_t)graph_count, sizeof(uint16_t), "small_graph_lookup_count4");
#endif
    }
    return 1;
}

#if RECT_COUNT_K4
static uint16_t small_graph_eval_factorised_at_4(int n, uint32_t mask) {
    int x_pow = g_small_graph_lookup_x_pows[n][mask];
    const int32_t* coeffs = small_graph_poly_slot(n, mask);
    __int128 acc = 0;
    for (int i = n - x_pow; i >= 0; i--) acc = (acc * 4) + coeffs[i];
    acc <<= 2 * x_pow;
    if (acc < 0 || acc > UINT16_MAX) {
        fprintf(stderr, "small graph 4-colouring count out of uint16 range\n");
        exit(1);
    }
    return (uint16_t)acc;
}

static void small_graph_lookup_build_count4_tables(void) {
    for (int n = 0; n <= SMALL_GRAPH_LOOKUP_MAX_N; n++) {
        uint32_t graph_count = g_small_graph_graph_count[n];
        for (uint32_t mask = 0; mask < graph_count; mask++) {
            g_small_graph_lookup_count4[n][mask] = small_graph_eval_factorised_at_4(n, mask);
        }
    }
}
#endif

static void small_graph_lookup_factorise_tables(void) {
    for (int n = 0; n <= SMALL_GRAPH_LOOKUP_MAX_N; n++) {
        uint32_t graph_count = g_small_graph_graph_count[n];
        for (uint32_t mask = 0; mask < graph_count; mask++) {
            int32_t* coeffs = small_graph_poly_slot(n, mask);
            int shift = 0;
            while (shift < n && coeffs[shift] == 0) shift++;
            g_small_graph_lookup_x_pows[n][mask] = (uint8_t)shift;
            if (shift > 0) {
                for (int i = 0; i <= n - shift; i++) coeffs[i] = coeffs[i + shift];
                for (int i = n - shift + 1; i <= n; i++) coeffs[i] = 0;
            }
        }
    }
#if RECT_COUNT_K4
    small_graph_lookup_build_count4_tables();
#endif
}

static inline uint32_t small_graph_pack_mask_from_row_values(const AdjWord* rows, int n) {
    uint32_t mask = 0;
    int bit = 0;
    for (int j = 1; j < n; j++) {
        mask |= (uint32_t)(((uint64_t)rows[j] & graph_row_mask(j)) << bit);
        bit += j;
    }
    return mask;
}

static uint32_t small_graph_pack_mask_from_rows(const uint8_t* rows, int n) {
    AdjWord dense_rows[SMALL_GRAPH_LOOKUP_MAX_N];
    for (int i = 0; i < n; i++) dense_rows[i] = rows[i];
    return small_graph_pack_mask_from_row_values(dense_rows, n);
}

uint32_t small_graph_pack_mask(const Graph* g) {
    if (g->vertex_mask == graph_row_mask(g->n)) {
        return small_graph_pack_mask_from_row_values(g->adj, g->n);
    }
    AdjWord rows[MAXN_NAUTY];
    uint32_t n = graph_build_dense_rows(g, rows);
    return small_graph_pack_mask_from_row_values(rows, (int)n);
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
    small_graph_lookup_factorise_tables();
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
        small_graph_lookup_factorise_tables();
        g_small_graph_lookup_loaded_from_file = 0;
        g_small_graph_lookup_ready = 1;
    }
    g_small_graph_lookup_init_time = omp_get_wtime() - t0;
}

void small_graph_lookup_free(void) {
    for (int n = 0; n <= SMALL_GRAPH_LOOKUP_MAX_N; n++) {
        free(g_small_graph_lookup_coeffs[n]);
        g_small_graph_lookup_coeffs[n] = NULL;
        free(g_small_graph_lookup_x_pows[n]);
        g_small_graph_lookup_x_pows[n] = NULL;
#if RECT_COUNT_K4
        free(g_small_graph_lookup_count4[n]);
        g_small_graph_lookup_count4[n] = NULL;
#endif
    }
    g_small_graph_lookup_ready = 0;
    g_small_graph_lookup_init_time = 0.0;
    g_small_graph_lookup_loaded_from_file = 0;
}

void small_graph_lookup_load_graph_poly(int n, uint32_t mask, GraphPoly* out) {
    const int32_t* coeffs = small_graph_poly_slot(n, mask);
    out->x_pow = g_small_graph_lookup_x_pows[n][mask];
    out->deg = (uint8_t)(n - out->x_pow);
    for (int i = 0; i <= out->deg; i++) out->coeffs[i] = coeffs[i];
}

static const char* connected_canon_lookup_default_path(void) {
    const char* env_path = getenv("RECT_CONNECTED_CANON_LOOKUP");
    if (env_path && *env_path) return env_path;
    return NULL;
}

static size_t connected_canon_lookup_entry_file_size(int n) {
    return sizeof(uint64_t) + (size_t)n * sizeof(int32_t);
}

static const uint8_t* connected_canon_lookup_find_entry(uint64_t mask) {
    uint32_t lo = 0;
    uint32_t hi = g_connected_canon_lookup_count;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        const uint8_t* entry =
            g_connected_canon_lookup_entries + (size_t)mid * g_connected_canon_lookup_entry_size;
        uint64_t entry_mask;
        memcpy(&entry_mask, entry, sizeof(entry_mask));
        if (entry_mask < mask) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo >= g_connected_canon_lookup_count) return NULL;

    {
        const uint8_t* entry =
            g_connected_canon_lookup_entries + (size_t)lo * g_connected_canon_lookup_entry_size;
        uint64_t entry_mask;
        memcpy(&entry_mask, entry, sizeof(entry_mask));
        return (entry_mask == mask) ? entry : NULL;
    }
}

static int connected_canon_lookup_try_load_file(const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < (off_t)sizeof(ConnectedCanonLookupHeader)) {
        close(fd);
        return 0;
    }

    void* map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return 0;

    {
        const ConnectedCanonLookupHeader* header = (const ConnectedCanonLookupHeader*)map;
        if (header->magic != CONNECTED_CANON_LOOKUP_MAGIC ||
            header->version != CONNECTED_CANON_LOOKUP_VERSION ||
            header->n == 0 || header->n > CONNECTED_CANON_LOOKUP_MAX_N) {
            munmap(map, (size_t)st.st_size);
            return 0;
        }

        {
            size_t entry_size = connected_canon_lookup_entry_file_size((int)header->n);
            size_t expected_size = sizeof(*header) + (size_t)header->count * entry_size;
            if ((size_t)st.st_size != expected_size) {
                munmap(map, (size_t)st.st_size);
                return 0;
            }

            g_connected_canon_lookup_map = map;
            g_connected_canon_lookup_map_len = (size_t)st.st_size;
            g_connected_canon_lookup_entries = (const uint8_t*)map + sizeof(*header);
            g_connected_canon_lookup_entry_size = entry_size;
            g_connected_canon_lookup_count = header->count;
            g_connected_canon_lookup_n = (int)header->n;
            g_connected_canon_lookup_ready = 1;
            g_connected_canon_lookup_loaded = 1;
#ifdef MADV_RANDOM
            madvise((void*)g_connected_canon_lookup_map, g_connected_canon_lookup_map_len,
                    MADV_RANDOM);
#endif
            return 1;
        }
    }

    return 0;
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
    if (g_connected_canon_lookup_map) {
        munmap((void*)g_connected_canon_lookup_map, g_connected_canon_lookup_map_len);
    }
    g_connected_canon_lookup_map = NULL;
    g_connected_canon_lookup_map_len = 0;
    g_connected_canon_lookup_entries = NULL;
    g_connected_canon_lookup_count = 0;
    g_connected_canon_lookup_ready = 0;
    g_connected_canon_lookup_loaded = 0;
    g_connected_canon_lookup_n = 0;
    g_connected_canon_lookup_entry_size = 0;
    g_connected_canon_lookup_load_time = 0.0;
}

const int32_t* connected_canon_lookup_find_coeffs(uint64_t mask) {
    const uint8_t* entry;
    const int32_t* coeffs;

    if (!g_connected_canon_lookup_ready) return NULL;

    entry = connected_canon_lookup_find_entry(mask);
    if (!entry) return NULL;

    coeffs = (const int32_t*)(entry + sizeof(uint64_t));
    return coeffs;
}

int connected_canon_lookup_load_graph_poly(const Graph* g, GraphPoly* out) {
    if (!g_connected_canon_lookup_ready || g->n != g_connected_canon_lookup_n) return 0;

    {
        const int32_t* coeffs = connected_canon_lookup_find_coeffs(graph_pack_upper_mask64(g));
        if (!coeffs) return 0;
        out->x_pow = 1;
        out->deg = (uint8_t)(g_connected_canon_lookup_n - 1);
        for (int i = 0; i <= out->deg; i++) out->coeffs[i] = coeffs[i];
    }
    return 1;
}

int connected_canon_lookup_load_graph_poly_mask(uint64_t mask, int n, GraphPoly* out) {
    if (!g_connected_canon_lookup_ready || n != g_connected_canon_lookup_n) return 0;

    {
        const int32_t* coeffs = connected_canon_lookup_find_coeffs(mask);
        if (!coeffs) return 0;
        out->x_pow = 1;
        out->deg = (uint8_t)(g_connected_canon_lookup_n - 1);
        for (int i = 0; i <= out->deg; i++) out->coeffs[i] = coeffs[i];
    }
    return 1;
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

typedef struct {
    int time;
    int disc[MAXN_NAUTY];
    int low[MAXN_NAUTY];
    int parent[MAXN_NAUTY];
    uint64_t open_mask[MAXN_NAUTY];
    uint64_t* block_masks;
    int block_count;
    uint64_t articulation_mask;
} BiconnectedSearch;

typedef struct {
    uint8_t u;
    uint8_t child_count;
    uint64_t remaining_neighbors;
} BiconnectedFrame;

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

int graph_collect_biconnected_components(const Graph* g, uint64_t* block_masks,
                                         uint64_t* articulation_mask) {
    BiconnectedSearch st;
    memset(&st, 0, sizeof(st));
    st.block_masks = block_masks;
    for (int i = 0; i < MAXN_NAUTY; i++) st.parent[i] = -1;

    uint64_t active = g->vertex_mask;
    uint64_t discovered = 0;
    BiconnectedFrame stack[MAXN_NAUTY];
    int depth = 0;

    while ((active & ~discovered) != 0) {
        int root = __builtin_ctzll(active & ~discovered);
        st.disc[root] = ++st.time;
        st.low[root] = st.disc[root];
        st.parent[root] = -1;
        st.open_mask[root] = UINT64_C(1) << root;
        discovered |= UINT64_C(1) << root;
        stack[depth++] =
            (BiconnectedFrame){.u = (uint8_t)root,
                               .child_count = 0,
                               .remaining_neighbors = (uint64_t)g->adj[root] & active};

        while (depth > 0) {
            BiconnectedFrame* frame = &stack[depth - 1];
            int u = frame->u;

            if (frame->remaining_neighbors != 0) {
                int v = __builtin_ctzll(frame->remaining_neighbors);
                frame->remaining_neighbors &= frame->remaining_neighbors - 1;

                if (((discovered >> v) & 1U) == 0) {
                    st.parent[v] = u;
                    frame->child_count++;
                    st.disc[v] = ++st.time;
                    st.low[v] = st.disc[v];
                    st.open_mask[v] = UINT64_C(1) << v;
                    discovered |= UINT64_C(1) << v;
                    stack[depth++] =
                        (BiconnectedFrame){.u = (uint8_t)v,
                                           .child_count = 0,
                                           .remaining_neighbors = (uint64_t)g->adj[v] & active};
                } else if (v != st.parent[u] && st.disc[v] < st.disc[u]) {
                    if (st.disc[v] < st.low[u]) st.low[u] = st.disc[v];
                }
                continue;
            }

            int parent = st.parent[u];
            int child_count = frame->child_count;
            depth--;

            if (parent == -1) {
                if (child_count > 1) st.articulation_mask |= UINT64_C(1) << u;
                if (child_count == 0) st.block_masks[st.block_count++] = UINT64_C(1) << u;
                continue;
            }

            if (st.low[u] < st.low[parent]) st.low[parent] = st.low[u];
            if (st.low[u] >= st.disc[parent]) {
                if (st.parent[parent] != -1) st.articulation_mask |= UINT64_C(1) << parent;
                uint64_t block = st.open_mask[u] | (UINT64_C(1) << parent);
                if (block != 0) st.block_masks[st.block_count++] = block;
            } else {
                st.open_mask[parent] |= st.open_mask[u];
            }
        }
    }

    if (articulation_mask) *articulation_mask = st.articulation_mask & g->vertex_mask;
    return st.block_count;
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
        uint64_t nbrs = ((uint64_t)g->adj[u] & full) & ~graph_row_mask(u + 1);
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

    uint64_t active = g->vertex_mask & ADJWORD_MASK;
    uint32_t n = (uint32_t)__builtin_popcountll(active);

    if (n != g->n) {
        fprintf(stderr, "Graph vertex mask/count mismatch: n=%u mask_popcount=%u\n",
                (unsigned)g->n, (unsigned)n);
        exit(1);
    }

    uint64_t h = 14695981039346656037ULL;
#if defined(__BMI2__) && (defined(__x86_64__) || defined(__i386__))
    uint64_t rem = active;
    uint32_t dense_v = 0;
    while (rem) {
        int v = __builtin_ctzll(rem);
        AdjWord row = (AdjWord)_pext_u64((uint64_t)g->adj[v] & active, active);
        rows[dense_v] = row;
        h ^= (uint64_t)row;
        h *= 1099511628211ULL;
        dense_v++;
        rem &= rem - 1;
    }
#else
    int dense_index[MAXN_NAUTY];
    int dense_vertices[MAXN_NAUTY];
    uint64_t rem = active;
    uint32_t dense_v = 0;
    while (rem) {
        int v = __builtin_ctzll(rem);
        dense_index[v] = (int)dense_v;
        dense_vertices[dense_v++] = v;
        rem &= rem - 1;
    }

    for (uint32_t i = 0; i < n; i++) {
        int v = dense_vertices[i];
        uint64_t row_bits = (uint64_t)g->adj[v] & active;
        AdjWord row = 0;
        while (row_bits) {
            int u = __builtin_ctzll(row_bits);
            row |= (AdjWord)(UINT64_C(1) << dense_index[u]);
            row_bits &= row_bits - 1;
        }
        rows[i] = row;
        h ^= (uint64_t)row;
        h *= 1099511628211ULL;
    }
#endif
    h ^= (uint64_t)n;
    h *= 1099511628211ULL;
    return h;
}
