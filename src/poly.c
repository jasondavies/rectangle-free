static void degree_overflow(int deg) {
    fprintf(stderr, "Polynomial degree %d exceeds MAX_DEGREE=%d\n", deg, MAX_DEGREE - 1);
    exit(1);
}

// --- POLYNOMIAL ARITHMETIC ---

void poly_zero(Poly* p) {
    p->deg = 0;
    memset(p->coeffs, 0, sizeof(p->coeffs));
}

static inline void poly_one_ref(Poly* p) {
    poly_zero(p);
    p->coeffs[0] = 1;
}

Poly poly_one() {
    Poly p;
    poly_one_ref(&p);
    return p;
}

PolyCoeff poly_eval(Poly p, long long x);

static long long parse_ll_or_die(const char* text, const char* label) {
    char* end = NULL;
    errno = 0;
    long long value = strtoll(text, &end, 10);
    if (!text || *text == '\0' || !end || *end != '\0' || errno != 0) {
        fprintf(stderr, "Invalid integer for %s: %s\n", label, text ? text : "(null)");
        exit(1);
    }
    return value;
}

static inline void poly_add_ref(const Poly* a, const Poly* b, Poly* out) {
    Poly tmp;
    Poly* r = out;
    if (out == a || out == b) r = &tmp;
    r->deg = (a->deg > b->deg) ? a->deg : b->deg;
    for (int i = 0; i <= r->deg; i++) {
        PolyCoeff av = (i <= a->deg) ? a->coeffs[i] : 0;
        PolyCoeff bv = (i <= b->deg) ? b->coeffs[i] : 0;
        r->coeffs[i] = av + bv;
    }
    while (r->deg > 0 && r->coeffs[r->deg] == 0) r->deg--;
    if (r != out) *out = *r;
}

static inline void poly_add_ref_checked(const Poly* a, const Poly* b, Poly* out) {
    Poly tmp;
    Poly* r = out;
    if (out == a || out == b) r = &tmp;
    r->deg = (a->deg > b->deg) ? a->deg : b->deg;
    for (int i = 0; i <= r->deg; i++) {
        PolyCoeff av = (i <= a->deg) ? a->coeffs[i] : 0;
        PolyCoeff bv = (i <= b->deg) ? b->coeffs[i] : 0;
        if (__builtin_add_overflow(av, bv, &r->coeffs[i])) {
            fprintf(stderr, "128-bit overflow in polynomial accumulation\n");
            abort();
        }
    }
    while (r->deg > 0 && r->coeffs[r->deg] == 0) r->deg--;
    if (r != out) *out = *r;
}

static inline void poly_accumulate_checked(Poly* acc, const Poly* add) {
    int old_deg = acc->deg;
    if (add->deg > old_deg) {
        memset(acc->coeffs + old_deg + 1, 0,
               (size_t)(add->deg - old_deg) * sizeof(acc->coeffs[0]));
        acc->deg = add->deg;
    }
    for (int i = 0; i <= add->deg; i++) {
        if (__builtin_add_overflow(acc->coeffs[i], add->coeffs[i], &acc->coeffs[i])) {
            fprintf(stderr, "128-bit overflow in polynomial accumulation\n");
            abort();
        }
    }
    while (acc->deg > 0 && acc->coeffs[acc->deg] == 0) acc->deg--;
}

Poly poly_add(Poly a, Poly b) {
    Poly r;
    poly_add_ref(&a, &b, &r);
    return r;
}

static inline void poly_sub_ref(const Poly* a, const Poly* b, Poly* out) {
    Poly tmp;
    Poly* r = out;
    if (out == a || out == b) r = &tmp;
    r->deg = (a->deg > b->deg) ? a->deg : b->deg;
    for (int i = 0; i <= r->deg; i++) {
        PolyCoeff av = (i <= a->deg) ? a->coeffs[i] : 0;
        PolyCoeff bv = (i <= b->deg) ? b->coeffs[i] : 0;
        r->coeffs[i] = av - bv;
    }
    while (r->deg > 0 && r->coeffs[r->deg] == 0) r->deg--;
    if (r != out) *out = *r;
}

Poly poly_sub(Poly a, Poly b) {
    Poly r;
    poly_sub_ref(&a, &b, &r);
    return r;
}

static inline void poly_mul_ref(const Poly* a, const Poly* b, Poly* out) {
    if ((a->deg == 0 && a->coeffs[0] == 0) ||
        (b->deg == 0 && b->coeffs[0] == 0)) {
        poly_zero(out);
        return;
    }

    Poly tmp;
    Poly* r = out;
    if (out == a || out == b) r = &tmp;
    r->deg = a->deg + b->deg;
    if (r->deg >= MAX_DEGREE) {
        degree_overflow(r->deg);
    }
    memset(r->coeffs, 0, (size_t)(r->deg + 1) * sizeof(r->coeffs[0]));
    for (int i = 0; i <= a->deg; i++) {
        if (a->coeffs[i] == 0) continue;
        for (int j = 0; j <= b->deg; j++) {
            r->coeffs[i + j] += a->coeffs[i] * b->coeffs[j];
        }
    }
    while (r->deg > 0 && r->coeffs[r->deg] == 0) r->deg--;
    if (r != out) *out = *r;
}

Poly poly_mul(Poly a, Poly b) {
    Poly r;
    poly_mul_ref(&a, &b, &r);
    return r;
}

static inline void poly_scale_ref(const Poly* a, long long s, Poly* out) {
    if (s == 0) {
        poly_zero(out);
        return;
    }
    out->deg = a->deg;
    for (int i = 0; i <= a->deg; i++) out->coeffs[i] = a->coeffs[i] * (PolyCoeff)s;
}

Poly poly_scale(Poly a, long long s) {
    Poly r;
    poly_scale_ref(&a, s, &r);
    return r;
}

static inline void poly_mul_linear_ref(const Poly* a, int c, Poly* out) {
    if (a->deg == 0 && a->coeffs[0] == 0) {
        poly_zero(out);
        return;
    }

    Poly tmp;
    Poly* r = out;
    if (out == a) r = &tmp;
    r->deg = a->deg + 1;
    if (r->deg >= MAX_DEGREE) degree_overflow(r->deg);
    memset(r->coeffs, 0, (size_t)(r->deg + 1) * sizeof(r->coeffs[0]));

    for (int i = 0; i <= a->deg; i++) r->coeffs[i + 1] += a->coeffs[i];
    if (c != 0) {
        for (int i = 0; i <= a->deg; i++) r->coeffs[i] -= a->coeffs[i] * (PolyCoeff)c;
    }
    while (r->deg > 0 && r->coeffs[r->deg] == 0) r->deg--;
    if (r != out) *out = *r;
}

Poly poly_mul_linear(Poly a, int c) {
    Poly r;
    poly_mul_linear_ref(&a, c, &r);
    return r;
}

static inline void poly_mul_falling_ref(const Poly* p, int start, int count, Poly* out) {
    if (out != p) *out = *p;
    for (int i = 0; i < count; i++) {
        poly_mul_linear_ref(out, start + i, out);
    }
}

Poly poly_mul_falling(Poly p, int start, int count) {
    Poly r;
    poly_mul_falling_ref(&p, start, count, &r);
    return r;
}

static inline void graph_poly_degree_overflow(int deg) {
    fprintf(stderr, "Graph polynomial degree %d exceeds MAXN_NAUTY=%d\n", deg, MAXN_NAUTY);
    exit(1);
}

static inline void graph_poly_zero(GraphPoly* p) {
    p->deg = 0;
    memset(p->coeffs, 0, sizeof(p->coeffs));
}

static inline void graph_poly_one_ref(GraphPoly* p) {
    graph_poly_zero(p);
    p->coeffs[0] = 1;
}

static inline void graph_poly_from_poly(const Poly* src, GraphPoly* dst) {
    if (src->deg > MAXN_NAUTY) graph_poly_degree_overflow(src->deg);
    dst->deg = (uint8_t)src->deg;
    memcpy(dst->coeffs, src->coeffs, (size_t)(src->deg + 1) * sizeof(src->coeffs[0]));
}

static inline void graph_poly_to_poly(const GraphPoly* src, Poly* dst) {
    dst->deg = src->deg;
    memcpy(dst->coeffs, src->coeffs, (size_t)(src->deg + 1) * sizeof(src->coeffs[0]));
}

static inline void poly_mul_graph_ref(const Poly* a, const GraphPoly* b, Poly* out) {
    if ((a->deg == 0 && a->coeffs[0] == 0) || (b->deg == 0 && b->coeffs[0] == 0)) {
        poly_zero(out);
        return;
    }

    Poly tmp;
    Poly* r = out;
    if (out == a) r = &tmp;
    r->deg = a->deg + b->deg;
    if (r->deg >= MAX_DEGREE) degree_overflow(r->deg);
    memset(r->coeffs, 0, (size_t)(r->deg + 1) * sizeof(r->coeffs[0]));
    for (int i = 0; i <= a->deg; i++) {
        if (a->coeffs[i] == 0) continue;
        for (int j = 0; j <= b->deg; j++) {
            r->coeffs[i + j] += a->coeffs[i] * b->coeffs[j];
        }
    }
    while (r->deg > 0 && r->coeffs[r->deg] == 0) r->deg--;
    if (r != out) *out = *r;
}

static inline void graph_poly_mul_ref(const GraphPoly* a, const GraphPoly* b, GraphPoly* out) {
    if ((a->deg == 0 && a->coeffs[0] == 0) || (b->deg == 0 && b->coeffs[0] == 0)) {
        graph_poly_zero(out);
        return;
    }

    GraphPoly tmp;
    GraphPoly* r = out;
    if (out == a || out == b) r = &tmp;
    r->deg = (uint8_t)(a->deg + b->deg);
    if (r->deg > MAXN_NAUTY) graph_poly_degree_overflow(r->deg);
    memset(r->coeffs, 0, (size_t)(r->deg + 1) * sizeof(r->coeffs[0]));
    for (int i = 0; i <= a->deg; i++) {
        if (a->coeffs[i] == 0) continue;
        for (int j = 0; j <= b->deg; j++) {
            r->coeffs[i + j] += a->coeffs[i] * b->coeffs[j];
        }
    }
    while (r->deg > 0 && r->coeffs[r->deg] == 0) r->deg--;
    if (r != out) *out = *r;
}

static inline void graph_poly_sub_ref(const GraphPoly* a, const GraphPoly* b, GraphPoly* out) {
    GraphPoly tmp;
    GraphPoly* r = out;
    if (out == a || out == b) r = &tmp;
    r->deg = (a->deg > b->deg) ? a->deg : b->deg;
    for (int i = 0; i <= r->deg; i++) {
        PolyCoeff av = (i <= a->deg) ? a->coeffs[i] : 0;
        PolyCoeff bv = (i <= b->deg) ? b->coeffs[i] : 0;
        r->coeffs[i] = av - bv;
    }
    while (r->deg > 0 && r->coeffs[r->deg] == 0) r->deg--;
    if (r != out) *out = *r;
}

static inline void graph_poly_mul_linear_ref(const GraphPoly* a, int c, GraphPoly* out) {
    if (a->deg == 0 && a->coeffs[0] == 0) {
        graph_poly_zero(out);
        return;
    }

    GraphPoly tmp;
    GraphPoly* r = out;
    if (out == a) r = &tmp;
    r->deg = (uint8_t)(a->deg + 1);
    if (r->deg > MAXN_NAUTY) graph_poly_degree_overflow(r->deg);
    memset(r->coeffs, 0, (size_t)(r->deg + 1) * sizeof(r->coeffs[0]));

    for (int i = 0; i <= a->deg; i++) r->coeffs[i + 1] += a->coeffs[i];
    if (c != 0) {
        for (int i = 0; i <= a->deg; i++) r->coeffs[i] -= a->coeffs[i] * (PolyCoeff)c;
    }
    while (r->deg > 0 && r->coeffs[r->deg] == 0) r->deg--;
    if (r != out) *out = *r;
}

#if RECT_COUNT_K4
static inline void graph_poly_set_count4(uint64_t count, GraphPoly* out) {
    out->deg = 0;
    out->coeffs[0] = (PolyCoeff)count;
}

static inline uint64_t graph_poly_get_count4(const GraphPoly* p) {
    return (uint64_t)p->coeffs[0];
}

static inline void poly_set_count4(unsigned __int128 count, Poly* out) {
    poly_zero(out);
    out->coeffs[0] = (PolyCoeff)count;
}

static uint64_t eval_int32_poly_at_4(const int32_t* coeffs, int deg) {
    __int128 acc = 0;
    for (int i = deg; i >= 0; i--) acc = (acc * 4) + coeffs[i];
    if (acc < 0 || acc > UINT64_MAX) {
        fprintf(stderr, "4-colouring count out of uint64 range\n");
        exit(1);
    }
    return (uint64_t)acc;
}

static uint64_t small_graph_lookup_load_count4(int n, uint32_t mask) {
    return eval_int32_poly_at_4(small_graph_poly_slot(n, mask), n);
}

static uint64_t connected_canon_lookup_load_count4(const Graph* g) {
    if (!g_connected_canon_lookup_ready || g->n != g_connected_canon_lookup_n) return UINT64_MAX;

    uint64_t mask = graph_pack_upper_mask64(g);
    ConnectedCanonLookupEntry key = {.mask = mask};
    ConnectedCanonLookupEntry* entry = bsearch(&key, g_connected_canon_lookup,
                                               g_connected_canon_lookup_count,
                                               sizeof(*g_connected_canon_lookup),
                                               connected_canon_lookup_entry_cmp);
    if (!entry) return UINT64_MAX;
    return eval_int32_poly_at_4(entry->coeffs, g_connected_canon_lookup_n);
}

static const uint64_t g_fall4[5] = {1, 4, 12, 24, 24};

static uint64_t count_graph_4_rec(const Graph* g, AdjWord uncoloured,
                                  uint8_t forbid[MAXN_NAUTY], int used) {
    if (!uncoloured) return g_fall4[used];

    int best = -1;
    int best_sat = -1;
    int best_deg = -1;
    AdjWord rem = uncoloured;
    while (rem) {
        int v = __builtin_ctzll((uint64_t)rem);
        rem &= rem - 1;
        int sat = __builtin_popcount((unsigned)forbid[v]);
        int deg = __builtin_popcountll((uint64_t)(g->adj[v] & uncoloured));
        if (sat > best_sat || (sat == best_sat && deg > best_deg)) {
            best = v;
            best_sat = sat;
            best_deg = deg;
        }
    }

    AdjWord rest = uncoloured & ~((AdjWord)1 << best);
    AdjWord neigh = g->adj[best] & rest;
    uint8_t fb = forbid[best];
    uint64_t total = 0;

    for (int c = 0; c < used; c++) {
        uint8_t bit = (uint8_t)(1u << c);
        if (fb & bit) continue;

        int changed[MAXN_NAUTY];
        int changed_count = 0;
        AdjWord nbrs = neigh;
        while (nbrs) {
            int u = __builtin_ctzll((uint64_t)nbrs);
            nbrs &= nbrs - 1;
            if (!(forbid[u] & bit)) {
                forbid[u] |= bit;
                changed[changed_count++] = u;
            }
        }

        total += count_graph_4_rec(g, rest, forbid, used);

        while (changed_count) {
            int u = changed[--changed_count];
            forbid[u] &= (uint8_t)~bit;
        }
    }

    if (used < 4) {
        uint8_t bit = (uint8_t)(1u << used);
        if (!(fb & bit)) {
            int changed[MAXN_NAUTY];
            int changed_count = 0;
            AdjWord nbrs = neigh;
            while (nbrs) {
                int u = __builtin_ctzll((uint64_t)nbrs);
                nbrs &= nbrs - 1;
                if (!(forbid[u] & bit)) {
                    forbid[u] |= bit;
                    changed[changed_count++] = u;
                }
            }

            total += count_graph_4_rec(g, rest, forbid, used + 1);

            while (changed_count) {
                int u = changed[--changed_count];
                forbid[u] &= (uint8_t)~bit;
            }
        }
    }

    return total;
}

static uint64_t count_graph_4_dsat(const Graph* g) {
    uint8_t forbid[MAXN_NAUTY] = {0};
    return count_graph_4_rec(g, (AdjWord)g->vertex_mask, forbid, 0);
}

#if RECT_COUNT_K4_FEASIBILITY
static int contains_edge_mask(const Graph* g, uint64_t mask) {
    while (mask) {
        int a = __builtin_ctzll(mask);
        uint64_t na = (uint64_t)g->adj[a] & mask & ~((UINT64_C(1) << (a + 1)) - 1U);
        if (na) return 1;
        mask &= mask - 1;
    }
    return 0;
}

static int contains_triangle_mask(const Graph* g, uint64_t mask) {
    while (mask) {
        int a = __builtin_ctzll(mask);
        uint64_t na = (uint64_t)g->adj[a] & mask & ~((UINT64_C(1) << (a + 1)) - 1U);
        while (na) {
            int b = __builtin_ctzll(na);
            if ((na & (uint64_t)g->adj[b]) & ~((UINT64_C(1) << (b + 1)) - 1U)) return 1;
            na &= na - 1;
        }
        mask &= mask - 1;
    }
    return 0;
}

static int contains_k4_mask(const Graph* g, uint64_t mask) {
    while (mask) {
        int a = __builtin_ctzll(mask);
        uint64_t na = (uint64_t)g->adj[a] & mask & ~((UINT64_C(1) << (a + 1)) - 1U);
        while (na) {
            int b = __builtin_ctzll(na);
            uint64_t nb = (na & (uint64_t)g->adj[b]) & ~((UINT64_C(1) << (b + 1)) - 1U);
            while (nb) {
                int c = __builtin_ctzll(nb);
                if ((nb & (uint64_t)g->adj[c]) & ~((UINT64_C(1) << (c + 1)) - 1U)) return 1;
                nb &= nb - 1;
            }
            na &= na - 1;
        }
        mask &= mask - 1;
    }
    return 0;
}

static int partial_graph_new_has_k5(const PartialGraphState* st) {
    if (st->g.n < 5 || st->last_num_new == 0) return 0;

    uint64_t old_mask = st->last_base > 0 ? ((UINT64_C(1) << st->last_base) - 1U) : 0;
    int base_new = st->last_base;
    int num_new = st->last_num_new;

    for (int i = 0; i < num_new; i++) {
        int u = base_new + i;
        if (contains_k4_mask(&st->g, (uint64_t)st->g.adj[u] & old_mask)) return 1;
    }

    for (int i = 0; i < num_new; i++) {
        int u = base_new + i;
        for (int j = i + 1; j < num_new; j++) {
            int v = base_new + j;
            if (contains_triangle_mask(&st->g,
                                       ((uint64_t)st->g.adj[u] & (uint64_t)st->g.adj[v] & old_mask))) {
                return 1;
            }
        }
    }

    if (num_new >= 3) {
        uint64_t common = old_mask;
        for (int i = 0; i < 3; i++) common &= (uint64_t)st->g.adj[base_new + i];
        if (contains_edge_mask(&st->g, common)) return 1;
    }

    return 0;
}

static int choose_dsat_vertex_colourable(const Graph* g, const int8_t* colour,
                                         const uint8_t* saturation) {
    int best = -1;
    int best_sat = -1;
    int best_deg = -1;
    uint64_t rem = g->vertex_mask;
    while (rem) {
        int v = __builtin_ctzll(rem);
        rem &= rem - 1;
        if (colour[v] >= 0) continue;
        int sat = __builtin_popcount((unsigned)saturation[v]);
        int deg = __builtin_popcountll((uint64_t)g->adj[v] & g->vertex_mask);
        if (sat > best_sat || (sat == best_sat && deg > best_deg)) {
            best = v;
            best_sat = sat;
            best_deg = deg;
        }
    }
    return best;
}

static int dsatur_is_4_colourable(const Graph* g, int coloured, const int8_t* colour,
                                  const uint8_t* saturation, int8_t* solution) {
    if (coloured == g->n) {
        if (solution) memcpy(solution, colour, sizeof(int8_t) * MAXN_NAUTY);
        return 1;
    }

    int v = choose_dsat_vertex_colourable(g, colour, saturation);
    if (v < 0) {
        if (solution) memcpy(solution, colour, sizeof(int8_t) * MAXN_NAUTY);
        return 1;
    }

    uint8_t available = (uint8_t)(0x0fU & (uint8_t)~saturation[v]);
    while (available) {
        unsigned bit_u = (unsigned)available & (unsigned)(-(int)available);
        uint8_t bit = (uint8_t)bit_u;
        int c = __builtin_ctz((unsigned)bit);
        available &= (uint8_t)(available - 1);

        int8_t next_colour[MAXN_NAUTY];
        uint8_t next_saturation[MAXN_NAUTY];
        memcpy(next_colour, colour, sizeof(next_colour));
        memcpy(next_saturation, saturation, sizeof(next_saturation));
        next_colour[v] = (int8_t)c;

        uint64_t neighbours = (uint64_t)g->adj[v] & g->vertex_mask;
        int stuck = 0;
        while (neighbours) {
            int u = __builtin_ctzll(neighbours);
            if (next_colour[u] < 0) {
                next_saturation[u] |= bit;
                if (next_saturation[u] == 0x0fU) stuck = 1;
            }
            neighbours &= neighbours - 1;
        }

        if (!stuck && dsatur_is_4_colourable(g, coloured + 1, next_colour, next_saturation,
                                             solution)) {
            return 1;
        }
    }
    return 0;
}

static int induced_subgraph_with_vertices(const Graph* src, uint64_t mask, Graph* dst, int* verts) {
    int n = 0;
    uint64_t rem = mask & src->vertex_mask;
    while (rem) {
        verts[n++] = __builtin_ctzll(rem);
        rem &= rem - 1;
    }

    dst->n = (uint8_t)n;
    dst->vertex_mask = graph_row_mask(n);
    memset(dst->adj, 0, (size_t)n * sizeof(dst->adj[0]));
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (((uint64_t)src->adj[verts[i]] >> verts[j]) & 1U) {
                dst->adj[i] |= (AdjWord)(UINT64_C(1) << j);
                dst->adj[j] |= (AdjWord)(UINT64_C(1) << i);
            }
        }
    }
    return n;
}

static int peel_to_4_core(const Graph* src, Graph* core, int* peel_order, int* peel_len,
                          int* core_vertices) {
    int degree[MAXN_NAUTY];
    uint64_t active = src->vertex_mask;
    uint64_t queue = 0;

    uint64_t rem = active;
    while (rem) {
        int v = __builtin_ctzll(rem);
        degree[v] = __builtin_popcountll((uint64_t)src->adj[v] & active);
        if (degree[v] <= 3) queue |= UINT64_C(1) << v;
        rem &= rem - 1;
    }

    *peel_len = 0;
    while (queue) {
        int v = __builtin_ctzll(queue);
        queue &= queue - 1;
        if (((active >> v) & 1U) == 0) continue;

        active &= ~(UINT64_C(1) << v);
        peel_order[(*peel_len)++] = v;

        uint64_t neighbours = (uint64_t)src->adj[v] & active;
        while (neighbours) {
            int u = __builtin_ctzll(neighbours);
            degree[u]--;
            if (degree[u] == 3) queue |= UINT64_C(1) << u;
            neighbours &= neighbours - 1;
        }
    }

    return induced_subgraph_with_vertices(src, active, core, core_vertices);
}

static int extend_colouring_over_peel(const Graph* g, const int* peel_order, int peel_len,
                                      int8_t* colour) {
    for (int i = peel_len - 1; i >= 0; i--) {
        int v = peel_order[i];
        uint8_t used = 0;
        uint64_t neighbours = (uint64_t)g->adj[v] & g->vertex_mask;
        while (neighbours) {
            int u = __builtin_ctzll(neighbours);
            if (colour[u] >= 0) used |= (uint8_t)(1U << colour[u]);
            neighbours &= neighbours - 1;
        }
        uint8_t available = (uint8_t)(0x0fU & (uint8_t)~used);
        if (available == 0) return 0;
        colour[v] = (int8_t)__builtin_ctz((unsigned)available);
    }
    return 1;
}

static int graph_component_colourable(const Graph* g, int8_t* out_colour) {
    if (g->n == 0) return 1;
    if (g->n == 1) {
        if (out_colour) {
            memset(out_colour, -1, sizeof(int8_t) * MAXN_NAUTY);
            out_colour[__builtin_ctzll(g->vertex_mask)] = 0;
        }
        return 1;
    }

    int start = __builtin_ctzll(g->vertex_mask);
    int best_deg = -1;
    uint64_t rem = g->vertex_mask;
    while (rem) {
        int v = __builtin_ctzll(rem);
        int deg = __builtin_popcountll((uint64_t)g->adj[v] & g->vertex_mask);
        if (deg > best_deg) {
            best_deg = deg;
            start = v;
        }
        rem &= rem - 1;
    }

    int8_t colour[MAXN_NAUTY];
    uint8_t saturation[MAXN_NAUTY];
    for (int i = 0; i < MAXN_NAUTY; i++) {
        colour[i] = -1;
        saturation[i] = 0;
    }

    colour[start] = 0;
    uint64_t neighbours = (uint64_t)g->adj[start] & g->vertex_mask;
    while (neighbours) {
        int u = __builtin_ctzll(neighbours);
        saturation[u] |= 1U;
        neighbours &= neighbours - 1;
    }

    return dsatur_is_4_colourable(g, 1, colour, saturation, out_colour);
}

static int partial_graph_is_feasible(const PartialGraphState* st, int cols_left) {
    if (cols_left <= 0) return 1;
    if (st->remaining_capacity < min_partition_pairs * cols_left) return 0;

    Graph core;
    int peel_order[MAXN_NAUTY];
    int peel_len = 0;
    int core_vertices[MAXN_NAUTY];
    int core_n = peel_to_4_core(&st->g, &core, peel_order, &peel_len, core_vertices);
    if (core_n == 0) {
        int8_t colour[MAXN_NAUTY];
        memset(colour, -1, sizeof(colour));
        return extend_colouring_over_peel(&st->g, peel_order, peel_len, colour);
    }

    if (partial_graph_new_has_k5(st)) return 0;

    int8_t core_colour[MAXN_NAUTY];
    if (!graph_component_colourable(&core, core_colour)) return 0;

    int8_t colour[MAXN_NAUTY];
    memset(colour, -1, sizeof(colour));
    for (int i = 0; i < core_n; i++) colour[core_vertices[i]] = core_colour[i];
    return extend_colouring_over_peel(&st->g, peel_order, peel_len, colour);
}
#endif
#endif

void print_u128(PolyCoeff n) {
    if (n == 0) { printf("0"); return; }
    if (n < 0) { printf("-"); n = -n; }
    char str[50];
    int idx = 0;
    while (n > 0) {
        str[idx++] = (int)(n % 10) + '0';
        n /= 10;
    }
    for (int i = idx - 1; i >= 0; i--) putchar(str[i]);
}

void print_poly(Poly p) {
    printf("P(x) = ");
    int first = 1;
    for (int i = p.deg; i >= 0; i--) {
        if (p.coeffs[i] == 0) continue;
        if (!first) {
            if (p.coeffs[i] > 0) printf(" + ");
            else printf(" - ");
        } else {
            if (p.coeffs[i] < 0) printf("-");
        }

        PolyCoeff abs_val = (p.coeffs[i] < 0) ? -p.coeffs[i] : p.coeffs[i];
        if (abs_val != 1 || i == 0) {
            print_u128(abs_val);
            if (i > 0) printf("*");
        }

        if (i > 0) {
            printf("x");
            if (i > 1) printf("^%d", i);
        }
        first = 0;
    }
    if (first) {
        printf("0\n");
        return;
    }
    printf("\n");
}

static void write_poly_file_stream(FILE* f, const Poly* poly, const PolyFileMeta* meta) {
    fprintf(f, "RECT_POLY_V1\n");
    fprintf(f, "rows %d\n", meta->rows);
    fprintf(f, "cols %d\n", meta->cols);
    fprintf(f, "task_start %lld\n", meta->task_start);
    fprintf(f, "task_end %lld\n", meta->task_end);
    fprintf(f, "full_tasks %lld\n", meta->full_tasks);
    fprintf(f, "deg %d\n", poly->deg);
    for (int i = 0; i <= poly->deg; i++) {
        fprintf(f, "coeff %d ", i);
        PolyCoeff value = poly->coeffs[i];
        if (value < 0) {
            fputc('-', f);
            value = -value;
        }
        if (value == 0) {
            fputc('0', f);
        } else {
            char digits[64];
            int idx = 0;
            while (value > 0) {
                digits[idx++] = (char)('0' + (int)(value % 10));
                value /= 10;
            }
            while (idx-- > 0) fputc(digits[idx], f);
        }
        fputc('\n', f);
    }
    fprintf(f, "end\n");
}

static void write_poly_file(const char* path, const Poly* poly, const PolyFileMeta* meta) {
    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Failed to open %s for writing\n", path);
        exit(1);
    }

    write_poly_file_stream(f, poly, meta);

    if (fclose(f) != 0) {
        fprintf(stderr, "Failed to close %s\n", path);
        exit(1);
    }
}
