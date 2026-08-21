#include "partition_poly.h"

#if RECT_COUNT_K4

int solve_graph_poly_treewidth(const Graph* g, int width_limit,
                               GraphPoly* out, int* width_out) {
    (void)g;
    (void)width_limit;
    (void)out;
    if (width_out) *width_out = -1;
    return 0;
}

#else

#define TW_MAX_WIDTH 6
#define TW_MAX_BAG (TW_MAX_WIDTH + 1)
#define TW_MAX_STATES 877
#define TW_CATALOG_HASH 2048

typedef struct {
    int count;
    uint32_t codes[TW_MAX_STATES];
    uint8_t colours[TW_MAX_STATES][TW_MAX_BAG];
    uint8_t blocks[TW_MAX_STATES];
    uint16_t hash_slots[TW_CATALOG_HASH];
} TwPartitionCatalog;

typedef struct {
    int n;
    int width;
    uint8_t order[MAX_GRAPH_VERTICES];
    uint8_t position[MAX_GRAPH_VERTICES];
    int8_t parent[MAX_GRAPH_VERTICES];
    int8_t first_child[MAX_GRAPH_VERTICES];
    int8_t next_sibling[MAX_GRAPH_VERTICES];
    uint8_t bag_n[MAX_GRAPH_VERTICES];
    uint8_t bags[MAX_GRAPH_VERTICES][TW_MAX_BAG];
} TwDecomposition;

typedef struct {
    uint8_t bag_n;
    uint8_t bag[TW_MAX_BAG];
    int state_count;
    int stride;
    int8_t* degree;
    PolyCoeff* coeffs;
} TwTable;

typedef struct {
    const Graph* graph;
    const TwDecomposition* decomposition;
    int stride;
    int overflow;
} TwSolveContext;

static TwPartitionCatalog tw_catalogs[TW_MAX_BAG + 1];
static pthread_once_t tw_catalog_once = PTHREAD_ONCE_INIT;

static void tw_catalog_generate_rec(TwPartitionCatalog* catalog, int bag_n,
                                    int pos, int max_colour, uint32_t code,
                                    uint8_t colours[TW_MAX_BAG]) {
    if (pos == bag_n) {
        int idx = catalog->count++;
        if (idx >= TW_MAX_STATES) abort();
        catalog->codes[idx] = code;
        memcpy(catalog->colours[idx], colours, (size_t)bag_n);
        catalog->blocks[idx] = (uint8_t)(bag_n == 0 ? 0 : max_colour + 1);
        return;
    }

    int upper = (pos == 0) ? 0 : max_colour + 1;
    for (int colour = 0; colour <= upper; colour++) {
        colours[pos] = (uint8_t)colour;
        int next_max = colour > max_colour ? colour : max_colour;
        tw_catalog_generate_rec(catalog, bag_n, pos + 1, next_max,
                                code | ((uint32_t)colour << (3 * pos)), colours);
    }
}

static inline uint32_t tw_catalog_hash(uint32_t code) {
    return (code * UINT32_C(2654435761)) & (TW_CATALOG_HASH - 1);
}

static void tw_catalogs_init_once(void) {
    for (int bag_n = 0; bag_n <= TW_MAX_BAG; bag_n++) {
        TwPartitionCatalog* catalog = &tw_catalogs[bag_n];
        memset(catalog, 0, sizeof(*catalog));
        uint8_t colours[TW_MAX_BAG] = {0};
        tw_catalog_generate_rec(catalog, bag_n, 0, -1, 0, colours);
        for (int i = 0; i < catalog->count; i++) {
            uint32_t slot = tw_catalog_hash(catalog->codes[i]);
            while (catalog->hash_slots[slot] != 0) {
                slot = (slot + 1) & (TW_CATALOG_HASH - 1);
            }
            catalog->hash_slots[slot] = (uint16_t)(i + 1);
        }
    }
}

static int tw_catalog_find(const TwPartitionCatalog* catalog, uint32_t code) {
    uint32_t slot = tw_catalog_hash(code);
    for (;;) {
        uint16_t stored = catalog->hash_slots[slot];
        if (stored == 0) return -1;
        int idx = (int)stored - 1;
        if (catalog->codes[idx] == code) return idx;
        slot = (slot + 1) & (TW_CATALOG_HASH - 1);
    }
}

static int tw_choose_min_fill_vertex(const uint64_t* adj, uint64_t active,
                                     uint64_t* neighbors_out, int* fill_out) {
    int best_vertex = -1;
    int best_fill = INT_MAX;
    int best_degree = INT_MAX;
    uint64_t rem_vertices = active;
    while (rem_vertices) {
        int v = __builtin_ctzll(rem_vertices);
        uint64_t neighbors = adj[v] & active;
        int degree = __builtin_popcountll(neighbors);
        int fill = 0;
        uint64_t rem = neighbors;
        while (rem) {
            int u = __builtin_ctzll(rem);
            fill += __builtin_popcountll(neighbors & ~adj[u] & ~graph_row_mask(u + 1));
            if (fill > best_fill) break;
            rem &= rem - 1;
        }
        if (fill < best_fill || (fill == best_fill && degree < best_degree)) {
            best_vertex = v;
            best_fill = fill;
            best_degree = degree;
            *neighbors_out = neighbors;
        }
        rem_vertices &= rem_vertices - 1;
    }
    *fill_out = best_fill;
    return best_vertex;
}

static int tw_build_decomposition(const Graph* g, int width_limit,
                                  TwDecomposition* decomposition) {
    if (width_limit < 0 || width_limit > TW_MAX_WIDTH || g->n > MAX_GRAPH_VERTICES) return 0;
    memset(decomposition, 0, sizeof(*decomposition));
    decomposition->n = g->n;
    decomposition->width = 0;
    for (int i = 0; i < MAX_GRAPH_VERTICES; i++) {
        decomposition->parent[i] = -1;
        decomposition->first_child[i] = -1;
        decomposition->next_sibling[i] = -1;
    }

    uint64_t filled_adj[MAX_GRAPH_VERTICES] = {0};
    for (int v = 0; v < g->n; v++) filled_adj[v] = (uint64_t)g->adj[v] & g->vertex_mask;
    uint64_t active = g->vertex_mask;

    for (int step = 0; step < g->n; step++) {
        uint64_t neighbors = 0;
        int fill = 0;
        int v = tw_choose_min_fill_vertex(filled_adj, active, &neighbors, &fill);
        (void)fill;
        if (v < 0) return 0;
        int degree = __builtin_popcountll(neighbors);
        if (degree > width_limit) return 0;
        if (degree > decomposition->width) decomposition->width = degree;

        decomposition->order[step] = (uint8_t)v;
        decomposition->position[v] = (uint8_t)step;
        decomposition->bag_n[v] = (uint8_t)(degree + 1);
        int bag_pos = 0;
        for (int u = 0; u < g->n; u++) {
            if (u == v || (neighbors & (UINT64_C(1) << u))) {
                decomposition->bags[v][bag_pos++] = (uint8_t)u;
            }
        }

        uint64_t rem = neighbors;
        while (rem) {
            int u = __builtin_ctzll(rem);
            filled_adj[u] |= neighbors & ~(UINT64_C(1) << u);
            rem &= rem - 1;
        }
        active &= ~(UINT64_C(1) << v);
    }

    for (int step = 0; step < g->n; step++) {
        int v = decomposition->order[step];
        int parent = -1;
        int parent_pos = INT_MAX;
        for (int i = 0; i < decomposition->bag_n[v]; i++) {
            int u = decomposition->bags[v][i];
            if (u == v) continue;
            int pos = decomposition->position[u];
            if (pos > step && pos < parent_pos) {
                parent = u;
                parent_pos = pos;
            }
        }
        decomposition->parent[v] = (int8_t)parent;
        if (parent >= 0) {
            decomposition->next_sibling[v] = decomposition->first_child[parent];
            decomposition->first_child[parent] = (int8_t)v;
        }
    }
    return 1;
}

static TwTable* tw_table_alloc(const uint8_t* bag, int bag_n, int stride) {
    const TwPartitionCatalog* catalog = &tw_catalogs[bag_n];
    TwTable* table = (TwTable*)calloc(1, sizeof(*table));
    if (!table) return NULL;
    table->bag_n = (uint8_t)bag_n;
    memcpy(table->bag, bag, (size_t)bag_n);
    table->state_count = catalog->count;
    table->stride = stride;
    table->degree = (int8_t*)malloc((size_t)table->state_count * sizeof(table->degree[0]));
    table->coeffs = (PolyCoeff*)calloc((size_t)table->state_count * (size_t)stride,
                                      sizeof(table->coeffs[0]));
    if (!table->degree || !table->coeffs) {
        free(table->degree);
        free(table->coeffs);
        free(table);
        return NULL;
    }
    memset(table->degree, -1, (size_t)table->state_count * sizeof(table->degree[0]));
    return table;
}

static void tw_table_free(TwTable* table) {
    if (!table) return;
    free(table->degree);
    free(table->coeffs);
    free(table);
}

static inline PolyCoeff* tw_table_poly(TwTable* table, int state) {
    return table->coeffs + (size_t)state * (size_t)table->stride;
}

static inline const PolyCoeff* tw_table_poly_const(const TwTable* table, int state) {
    return table->coeffs + (size_t)state * (size_t)table->stride;
}

static uint32_t tw_restrict_code(const TwTable* source, int source_state,
                                 const uint8_t* target_bag, int target_n) {
    const uint8_t* source_colours = tw_catalogs[source->bag_n].colours[source_state];
    int8_t relabel[TW_MAX_BAG];
    memset(relabel, -1, sizeof(relabel));
    int next_colour = 0;
    uint32_t code = 0;
    for (int i = 0; i < target_n; i++) {
        int source_pos = -1;
        for (int j = 0; j < source->bag_n; j++) {
            if (source->bag[j] == target_bag[i]) {
                source_pos = j;
                break;
            }
        }
        if (source_pos < 0) abort();
        int old_colour = source_colours[source_pos];
        if (relabel[old_colour] < 0) relabel[old_colour] = (int8_t)next_colour++;
        code |= (uint32_t)relabel[old_colour] << (3 * i);
    }
    return code;
}

static int tw_add_coeff(TwSolveContext* context, PolyCoeff* dst, PolyCoeff value) {
    PolyCoeff sum;
    if (__builtin_add_overflow(*dst, value, &sum)) {
        context->overflow = 1;
        return 0;
    }
    *dst = sum;
    return 1;
}

static int tw_add_poly(TwSolveContext* context, TwTable* dst, int dst_state,
                       const PolyCoeff* src, int src_degree, int linear_constant) {
    PolyCoeff* out = tw_table_poly(dst, dst_state);
    if (linear_constant < 0) {
        for (int i = 0; i <= src_degree; i++) {
            if (!tw_add_coeff(context, &out[i], src[i])) return 0;
        }
        if (dst->degree[dst_state] < src_degree) dst->degree[dst_state] = (int8_t)src_degree;
        return 1;
    }

    for (int i = 0; i <= src_degree; i++) {
        PolyCoeff scaled;
        if (__builtin_mul_overflow(src[i], (PolyCoeff)(-linear_constant), &scaled) ||
            !tw_add_coeff(context, &out[i], scaled) ||
            !tw_add_coeff(context, &out[i + 1], src[i])) {
            context->overflow = 1;
            return 0;
        }
    }
    int degree = src_degree + 1;
    if (dst->degree[dst_state] < degree) dst->degree[dst_state] = (int8_t)degree;
    return 1;
}

static TwTable* tw_forget_child_root(TwSolveContext* context, const TwTable* child,
                                     int forgotten_vertex) {
    uint8_t separator[TW_MAX_BAG];
    int separator_n = 0;
    int forgotten_pos = -1;
    for (int i = 0; i < child->bag_n; i++) {
        if (child->bag[i] == forgotten_vertex) {
            forgotten_pos = i;
        } else {
            separator[separator_n++] = child->bag[i];
        }
    }
    if (forgotten_pos < 0) return NULL;
    TwTable* result = tw_table_alloc(separator, separator_n, context->stride);
    if (!result) return NULL;
    const TwPartitionCatalog* child_catalog = &tw_catalogs[child->bag_n];
    const TwPartitionCatalog* separator_catalog = &tw_catalogs[separator_n];

    for (int state = 0; state < child->state_count; state++) {
        int degree = child->degree[state];
        if (degree < 0) continue;
        uint32_t restricted = tw_restrict_code(child, state, separator, separator_n);
        int target_state = tw_catalog_find(separator_catalog, restricted);
        if (target_state < 0) abort();
        int forgotten_colour = child_catalog->colours[state][forgotten_pos];
        int singleton = 1;
        for (int i = 0; i < child->bag_n; i++) {
            if (i != forgotten_pos && child_catalog->colours[state][i] == forgotten_colour) {
                singleton = 0;
                break;
            }
        }
        int linear_constant = singleton ? separator_catalog->blocks[target_state] : -1;
        if (!tw_add_poly(context, result, target_state,
                         tw_table_poly_const(child, state), degree, linear_constant)) {
            tw_table_free(result);
            return NULL;
        }
    }
    for (int state = 0; state < result->state_count; state++) {
        PolyCoeff* poly = tw_table_poly(result, state);
        while (result->degree[state] >= 0 && poly[(int)result->degree[state]] == 0) {
            result->degree[state]--;
        }
    }
    return result;
}

static int tw_multiply_state_poly(TwSolveContext* context, TwTable* dst, int dst_state,
                                  const PolyCoeff* rhs, int rhs_degree) {
    int lhs_degree = dst->degree[dst_state];
    if (lhs_degree < 0 || rhs_degree < 0) {
        dst->degree[dst_state] = -1;
        return 1;
    }
    PolyCoeff tmp[MAX_GRAPH_VERTICES + 1] = {0};
    const PolyCoeff* lhs = tw_table_poly_const(dst, dst_state);
    for (int i = 0; i <= lhs_degree; i++) {
        for (int j = 0; j <= rhs_degree; j++) {
            PolyCoeff product;
            if (__builtin_mul_overflow(lhs[i], rhs[j], &product) ||
                !tw_add_coeff(context, &tmp[i + j], product)) {
                context->overflow = 1;
                return 0;
            }
        }
    }
    int degree = lhs_degree + rhs_degree;
    while (degree >= 0 && tmp[degree] == 0) degree--;
    if (degree < 0) {
        dst->degree[dst_state] = -1;
        return 1;
    }
    PolyCoeff* out = tw_table_poly(dst, dst_state);
    memcpy(out, tmp, ((size_t)degree + 1U) * sizeof(out[0]));
    dst->degree[dst_state] = (int8_t)degree;
    return 1;
}

static TwTable* tw_build_node_table(TwSolveContext* context, int node) {
    const TwDecomposition* decomposition = context->decomposition;
    int bag_n = decomposition->bag_n[node];
    const uint8_t* bag = decomposition->bags[node];
    TwTable* table = tw_table_alloc(bag, bag_n, context->stride);
    if (!table) return NULL;
    const TwPartitionCatalog* catalog = &tw_catalogs[bag_n];
    int node_pos = -1;
    for (int i = 0; i < bag_n; i++) {
        if (bag[i] == node) node_pos = i;
    }
    if (node_pos < 0) abort();

    for (int state = 0; state < table->state_count; state++) {
        int valid = 1;
        for (int i = 0; i < bag_n; i++) {
            int other = bag[i];
            if (other != node &&
                ((uint64_t)context->graph->adj[node] & (UINT64_C(1) << other)) &&
                catalog->colours[state][node_pos] == catalog->colours[state][i]) {
                valid = 0;
                break;
            }
        }
        if (valid) {
            table->degree[state] = 0;
            tw_table_poly(table, state)[0] = 1;
        }
    }

    for (int child = decomposition->first_child[node]; child >= 0;
         child = decomposition->next_sibling[child]) {
        TwTable* child_table = tw_build_node_table(context, child);
        if (!child_table) {
            tw_table_free(table);
            return NULL;
        }
        TwTable* separator_table = tw_forget_child_root(context, child_table, child);
        tw_table_free(child_table);
        if (!separator_table) {
            tw_table_free(table);
            return NULL;
        }
        const TwPartitionCatalog* separator_catalog = &tw_catalogs[separator_table->bag_n];
        for (int state = 0; state < table->state_count; state++) {
            if (table->degree[state] < 0) continue;
            uint32_t restricted = tw_restrict_code(table, state,
                                                   separator_table->bag,
                                                   separator_table->bag_n);
            int separator_state = tw_catalog_find(separator_catalog, restricted);
            if (separator_state < 0) abort();
            if (!tw_multiply_state_poly(context, table, state,
                                        tw_table_poly_const(separator_table, separator_state),
                                        separator_table->degree[separator_state])) {
                tw_table_free(separator_table);
                tw_table_free(table);
                return NULL;
            }
        }
        tw_table_free(separator_table);
    }
    return table;
}

static int tw_multiply_linear_dense(TwSolveContext* context, PolyCoeff* poly,
                                    int* degree, int constant) {
    PolyCoeff tmp[MAX_GRAPH_VERTICES + 1] = {0};
    for (int i = 0; i <= *degree; i++) {
        PolyCoeff scaled;
        if (__builtin_mul_overflow(poly[i], (PolyCoeff)(-constant), &scaled) ||
            !tw_add_coeff(context, &tmp[i], scaled) ||
            !tw_add_coeff(context, &tmp[i + 1], poly[i])) {
            context->overflow = 1;
            return 0;
        }
    }
    (*degree)++;
    memcpy(poly, tmp, ((size_t)*degree + 1U) * sizeof(poly[0]));
    return 1;
}

static int tw_table_finish(TwSolveContext* context, const TwTable* root, GraphPoly* out) {
    PolyCoeff total[MAX_GRAPH_VERTICES + 1] = {0};
    int total_degree = 0;
    const TwPartitionCatalog* catalog = &tw_catalogs[root->bag_n];
    for (int state = 0; state < root->state_count; state++) {
        int degree = root->degree[state];
        if (degree < 0) continue;
        PolyCoeff term[MAX_GRAPH_VERTICES + 1] = {0};
        memcpy(term, tw_table_poly_const(root, state),
               ((size_t)degree + 1U) * sizeof(term[0]));
        int blocks = catalog->blocks[state];
        for (int i = 0; i < blocks; i++) {
            if (!tw_multiply_linear_dense(context, term, &degree, i)) return 0;
        }
        if (degree > total_degree) total_degree = degree;
        for (int i = 0; i <= degree; i++) {
            if (!tw_add_coeff(context, &total[i], term[i])) return 0;
        }
    }

    int low = 0;
    while (low <= total_degree && total[low] == 0) low++;
    if (low > total_degree) {
        out->x_pow = 0;
        out->deg = 0;
        out->coeffs[0] = 0;
        return 1;
    }
    out->x_pow = (uint8_t)low;
    out->deg = (uint8_t)(total_degree - low);
    for (int i = low; i <= total_degree; i++) out->coeffs[i - low] = total[i];
    return 1;
}

int solve_graph_poly_treewidth(const Graph* g, int width_limit,
                               GraphPoly* out, int* width_out) {
    if (width_limit < 0 || width_limit > TW_MAX_WIDTH || g->n <= 0) return 0;
    pthread_once(&tw_catalog_once, tw_catalogs_init_once);

    TwDecomposition decomposition;
    if (!tw_build_decomposition(g, width_limit, &decomposition)) {
        if (width_out) *width_out = width_limit + 1;
        return 0;
    }
    if (width_out) *width_out = decomposition.width;

    int root = -1;
    int roots = 0;
    for (int v = 0; v < g->n; v++) {
        if (decomposition.parent[v] < 0) {
            root = v;
            roots++;
        }
    }
    if (roots != 1) return 0;

    TwSolveContext context = {
        .graph = g,
        .decomposition = &decomposition,
        .stride = g->n + 1,
        .overflow = 0,
    };
    TwTable* root_table = tw_build_node_table(&context, root);
    if (!root_table) return 0;
    int ok = !context.overflow && tw_table_finish(&context, root_table, out);
    tw_table_free(root_table);
    return ok && !context.overflow;
}

#endif
