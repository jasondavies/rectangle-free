#include "partition_poly.h"

static void degree_overflow(int deg) {
    fprintf(stderr, "Polynomial degree %d exceeds MAX_DEGREE=%d\n", deg, MAX_DEGREE - 1);
    exit(1);
}

// --- POLYNOMIAL ARITHMETIC ---

void poly_zero(Poly* p) {
    p->deg = 0;
    memset(p->coeffs, 0, sizeof(p->coeffs));
}

void poly_one_ref(Poly* p) {
    poly_zero(p);
    p->coeffs[0] = 1;
}

long long parse_ll_or_die(const char* text, const char* label) {
    char* end = NULL;
    errno = 0;
    long long value = strtoll(text, &end, 10);
    if (!text || *text == '\0' || !end || *end != '\0' || errno != 0) {
        fprintf(stderr, "Invalid integer for %s: %s\n", label, text ? text : "(null)");
        exit(1);
    }
    return value;
}

void poly_accumulate_checked(Poly* acc, const Poly* add) {
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

void poly_mul_ref(const Poly* a, const Poly* b, Poly* out) {
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

void poly_scale_ref(const Poly* a, long long s, Poly* out) {
    if (s == 0) {
        poly_zero(out);
        return;
    }
    out->deg = a->deg;
    for (int i = 0; i <= a->deg; i++) out->coeffs[i] = a->coeffs[i] * (PolyCoeff)s;
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

void poly_mul_falling_ref(const Poly* p, int start, int count, Poly* out) {
    if (out != p) *out = *p;
    for (int i = 0; i < count; i++) {
        poly_mul_linear_ref(out, start + i, out);
    }
}

static inline void graph_poly_degree_overflow(int deg) {
    fprintf(stderr, "Graph polynomial degree %d exceeds MAXN_NAUTY=%d\n", deg, MAXN_NAUTY);
    exit(1);
}

static inline int graph_poly_is_zero(const GraphPoly* p) {
    return p->deg == 0 && p->coeffs[0] == 0;
}

static inline void graph_poly_zero(GraphPoly* p) {
    p->x_pow = 0;
    p->deg = 0;
    p->coeffs[0] = 0;
}

static inline void graph_poly_mul_monomial_ref(const GraphPoly* poly, const GraphPoly* mono,
                                               GraphPoly* out) {
    PolyCoeff scale = mono->coeffs[0];
    if (scale == 0) {
        graph_poly_zero(out);
        return;
    }

    GraphPoly tmp;
    GraphPoly* r = out;
    if (out == poly || out == mono) r = &tmp;
    r->x_pow = (uint8_t)(poly->x_pow + mono->x_pow);
    r->deg = poly->deg;
    if ((int)r->x_pow + (int)r->deg > MAXN_NAUTY) {
        graph_poly_degree_overflow((int)r->x_pow + (int)r->deg);
    }

    if (scale == 1) {
        memcpy(r->coeffs, poly->coeffs, (size_t)(poly->deg + 1) * sizeof(r->coeffs[0]));
    } else {
        for (int i = 0; i <= poly->deg; i++) {
            r->coeffs[i] = poly->coeffs[i] * scale;
        }
    }

    while (r->deg > 0 && r->coeffs[r->deg] == 0) r->deg--;
    if (r != out) *out = *r;
}

void graph_poly_normalize_ref(GraphPoly* p) {
    if (graph_poly_is_zero(p)) {
        graph_poly_zero(p);
        return;
    }

    int shift = 0;
    while (shift <= p->deg && p->coeffs[shift] == 0) shift++;
    if (shift > p->deg) {
        graph_poly_zero(p);
        return;
    }

    if (shift > 0) {
        if ((int)p->x_pow + shift > MAXN_NAUTY) graph_poly_degree_overflow((int)p->x_pow + shift);
        for (int i = 0; i <= p->deg - shift; i++) p->coeffs[i] = p->coeffs[i + shift];
        for (int i = p->deg - shift + 1; i <= p->deg; i++) p->coeffs[i] = 0;
        p->x_pow = (uint8_t)(p->x_pow + shift);
        p->deg = (uint8_t)(p->deg - shift);
    }

    while (p->deg > 0 && p->coeffs[p->deg] == 0) p->deg--;
}

void graph_poly_one_ref(GraphPoly* p) {
    p->x_pow = 0;
    p->deg = 0;
    p->coeffs[0] = 1;
}

void poly_mul_graph_ref(const Poly* a, const GraphPoly* b, Poly* out) {
    if ((a->deg == 0 && a->coeffs[0] == 0) || graph_poly_is_zero(b)) {
        poly_zero(out);
        return;
    }

    Poly tmp;
    Poly* r = out;
    if (out == a) r = &tmp;
    r->deg = a->deg + b->x_pow + b->deg;
    if (r->deg >= MAX_DEGREE) degree_overflow(r->deg);
    memset(r->coeffs, 0, (size_t)(r->deg + 1) * sizeof(r->coeffs[0]));
    for (int i = 0; i <= a->deg; i++) {
        if (a->coeffs[i] == 0) continue;
        for (int j = 0; j <= b->deg; j++) {
            r->coeffs[i + j + b->x_pow] += a->coeffs[i] * b->coeffs[j];
        }
    }
    while (r->deg > 0 && r->coeffs[r->deg] == 0) r->deg--;
    if (r != out) *out = *r;
}

void graph_poly_mul_ref(const GraphPoly* a, const GraphPoly* b, GraphPoly* out) {
    if (graph_poly_is_zero(a) || graph_poly_is_zero(b)) {
        graph_poly_zero(out);
        return;
    }
    if (a->deg == 0) {
        graph_poly_mul_monomial_ref(b, a, out);
        return;
    }
    if (b->deg == 0) {
        graph_poly_mul_monomial_ref(a, b, out);
        return;
    }

    GraphPoly tmp;
    GraphPoly* r = out;
    if (out == a || out == b) r = &tmp;
    r->x_pow = (uint8_t)(a->x_pow + b->x_pow);
    r->deg = (uint8_t)(a->deg + b->deg);
    if ((int)r->x_pow + (int)r->deg > MAXN_NAUTY) {
        graph_poly_degree_overflow((int)r->x_pow + (int)r->deg);
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

void graph_poly_sub_ref(const GraphPoly* a, const GraphPoly* b, GraphPoly* out) {
    GraphPoly tmp;
    GraphPoly* r = out;
    if (out == a || out == b) r = &tmp;
    if (graph_poly_is_zero(a)) {
        *r = *b;
        for (int i = 0; i <= r->deg; i++) r->coeffs[i] = -r->coeffs[i];
        if (r != out) *out = *r;
        return;
    }
    if (graph_poly_is_zero(b)) {
        *r = *a;
        if (r != out) *out = *r;
        return;
    }

    int base_x_pow = (a->x_pow < b->x_pow) ? a->x_pow : b->x_pow;
    int shift_a = (int)a->x_pow - base_x_pow;
    int shift_b = (int)b->x_pow - base_x_pow;
    r->x_pow = (uint8_t)base_x_pow;
    r->deg = (uint8_t)((a->deg + shift_a > b->deg + shift_b) ? (a->deg + shift_a)
                                                              : (b->deg + shift_b));
    for (int i = 0; i <= r->deg; i++) {
        PolyCoeff av = (i >= shift_a && i - shift_a <= a->deg) ? a->coeffs[i - shift_a] : 0;
        PolyCoeff bv = (i >= shift_b && i - shift_b <= b->deg) ? b->coeffs[i - shift_b] : 0;
        r->coeffs[i] = av - bv;
    }
    while (r->deg > 0 && r->coeffs[r->deg] == 0) r->deg--;
    if (r->coeffs[0] == 0) graph_poly_normalize_ref(r);
    if (r != out) *out = *r;
}

void graph_poly_mul_linear_ref(const GraphPoly* a, int c, GraphPoly* out) {
    if (graph_poly_is_zero(a)) {
        graph_poly_zero(out);
        return;
    }

    GraphPoly tmp;
    GraphPoly* r = out;
    if (out == a) r = &tmp;
    if (c == 0) {
        *r = *a;
        if ((int)r->x_pow + 1 > MAXN_NAUTY) graph_poly_degree_overflow((int)r->x_pow + 1);
        r->x_pow++;
    } else {
        r->x_pow = a->x_pow;
        r->deg = (uint8_t)(a->deg + 1);
        if ((int)r->x_pow + (int)r->deg > MAXN_NAUTY) {
            graph_poly_degree_overflow((int)r->x_pow + (int)r->deg);
        }
        r->coeffs[0] = -a->coeffs[0] * (PolyCoeff)c;
        for (int i = 1; i <= a->deg; i++) {
            r->coeffs[i] = a->coeffs[i - 1] - (a->coeffs[i] * (PolyCoeff)c);
        }
        r->coeffs[a->deg + 1] = a->coeffs[a->deg];
    }
    if (r != out) *out = *r;
}

void graph_poly_div_x_ref(const GraphPoly* a, GraphPoly* out) {
    if (graph_poly_is_zero(a)) {
        graph_poly_zero(out);
        return;
    }
    if (a->x_pow == 0) {
        fprintf(stderr, "graph_poly_div_x_ref requires divisibility by x\n");
        exit(1);
    }

    GraphPoly tmp;
    GraphPoly* r = out;
    if (out == a) r = &tmp;
    *r = *a;
    r->x_pow--;
    if (r != out) *out = *r;
}

#if RECT_COUNT_K4
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

uint64_t small_graph_lookup_load_count4(int n, uint32_t mask) {
    int x_pow = g_small_graph_lookup_x_pows[n][mask];
    uint64_t value = eval_int32_poly_at_4(small_graph_poly_slot(n, mask), n - x_pow);
    return value << (2 * x_pow);
}

uint64_t connected_canon_lookup_load_count4(const Graph* g) {
    if (!g_connected_canon_lookup_ready || g->n != g_connected_canon_lookup_n) return UINT64_MAX;

    {
        const int32_t* coeffs = connected_canon_lookup_find_coeffs(graph_pack_upper_mask64(g));
        if (!coeffs) return UINT64_MAX;
        return eval_int32_poly_at_4(coeffs, g_connected_canon_lookup_n - 1) << 2;
    }
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

uint64_t count_graph_4_dsat(const Graph* g) {
    uint8_t forbid[MAXN_NAUTY] = {0};
    return count_graph_4_rec(g, (AdjWord)g->vertex_mask, forbid, 0);
}


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

void write_poly_file(const char* path, const Poly* poly, const PolyFileMeta* meta) {
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
