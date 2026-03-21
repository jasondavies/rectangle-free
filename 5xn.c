// 5xn.c
// Rectangle-free 4-colourings of a 5×n grid for n=0..40.
// Self-contained (no GMP), includes:
//  - BigInt (base 1e9)
//  - progress indicator
//  - per-k incremental stats
//  - symmetry reduction: 5! row permutations + colour permutations (sort 4 blocks)
//
// Build: cc -O3 -march=native -std=c11 5xn.c -o 5xn
// Run:   ./5xn

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum { ROWS=5, COLORS=4, BITS_PER_COLOR=10, TOTAL_BITS=40 };
enum { BASE = 1000000000u }; // 1e9

// ------------------------- timing -------------------------
static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

// ------------------------- BigInt -------------------------
typedef struct {
    uint32_t *d;  // little-endian limbs in base 1e9
    int len;
    int cap;
} Big;

static void big_init(Big *a) { a->d=NULL; a->len=0; a->cap=0; }

static void big_free(Big *a) { free(a->d); a->d=NULL; a->len=0; a->cap=0; }

static void big_reserve(Big *a, int cap) {
    if (cap <= a->cap) return;
    int ncap = a->cap ? a->cap : 1;
    while (ncap < cap) ncap <<= 1;
    uint32_t *nd = (uint32_t*)realloc(a->d, (size_t)ncap * sizeof(uint32_t));
    if (!nd) { fprintf(stderr, "OOM\n"); exit(1); }
    a->d = nd; a->cap = ncap;
}

static void big_set_zero(Big *a) { a->len = 0; }

static int big_is_zero(const Big *a) { return a->len == 0; }

static void big_normalize(Big *a) {
    while (a->len > 0 && a->d[a->len-1] == 0) a->len--;
}

static void big_set_u64(Big *a, uint64_t x) {
    big_set_zero(a);
    if (x == 0) return;
    big_reserve(a, 3);
    while (x) {
        a->d[a->len++] = (uint32_t)(x % BASE);
        x /= BASE;
    }
}

static void big_copy(Big *dst, const Big *src) {
    if (src->len <= 0) {
        dst->len = 0;
        return;
    }
    big_reserve(dst, src->len);
    dst->len = src->len;
    memcpy(dst->d, src->d, (size_t)src->len * sizeof(uint32_t));
}

static void big_add_inplace(Big *a, const Big *b) {
    int n = (a->len > b->len) ? a->len : b->len;
    big_reserve(a, n + 1);
    uint64_t carry = 0;
    for (int i=0;i<n;i++) {
        uint64_t av = (i < a->len) ? a->d[i] : 0;
        uint64_t bv = (i < b->len) ? b->d[i] : 0;
        uint64_t s  = av + bv + carry;
        a->d[i] = (uint32_t)(s % BASE);
        carry = s / BASE;
    }
    a->len = n;
    if (carry) a->d[a->len++] = (uint32_t)carry;
    big_normalize(a);
}

// a += (b * mul) where mul fits in uint32
static void big_addmul_u32(Big *a, const Big *b, uint32_t mul) {
    if (mul == 0 || b->len == 0) return;
    int need = (a->len > b->len ? a->len : b->len) + 2;
    big_reserve(a, need);

    uint64_t carry = 0;
    int i = 0;
    // ensure a->len at least b->len for the loop
    if (a->len < b->len) {
        // initialise missing digits to 0
        for (int k=a->len; k<b->len; k++) a->d[k] = 0;
        a->len = b->len;
    }
    for (; i < b->len; i++) {
        uint64_t cur = (uint64_t)a->d[i] + (uint64_t)b->d[i] * mul + carry;
        a->d[i] = (uint32_t)(cur % BASE);
        carry = cur / BASE;
    }
    while (carry && i < a->len) {
        uint64_t cur = (uint64_t)a->d[i] + carry;
        a->d[i] = (uint32_t)(cur % BASE);
        carry = cur / BASE;
        i++;
    }
    if (carry) {
        a->d[a->len++] = (uint32_t)(carry % BASE);
        carry /= BASE;
        if (carry) a->d[a->len++] = (uint32_t)carry;
    }
    big_normalize(a);
}

// a *= mul (mul <= 40 typically here)
static void big_mul_u32_inplace(Big *a, uint32_t mul) {
    if (mul == 0 || a->len == 0) { a->len = 0; return; }
    uint64_t carry = 0;
    for (int i=0;i<a->len;i++) {
        uint64_t cur = (uint64_t)a->d[i] * mul + carry;
        a->d[i] = (uint32_t)(cur % BASE);
        carry = cur / BASE;
    }
    while (carry) {
        big_reserve(a, a->len + 1);
        a->d[a->len++] = (uint32_t)(carry % BASE);
        carry /= BASE;
    }
    big_normalize(a);
}

static void big_print(const Big *a) {
    if (a->len == 0) { putchar('0'); return; }
    int i = a->len - 1;
    printf("%u", a->d[i]);
    for (i = i-1; i >= 0; i--) printf("%09u", a->d[i]);
}

// ------------------------- Poly -------------------------
typedef struct {
    int deg;   // max index that may be nonzero
    Big *c;    // coefficients 0..deg
} Poly;

static Poly* poly_new(int deg) {
    Poly *p = (Poly*)malloc(sizeof(Poly));
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    p->deg = deg;
    p->c = (Big*)malloc((size_t)(deg+1) * sizeof(Big));
    if (!p->c) { fprintf(stderr, "OOM\n"); exit(1); }
    for (int i=0;i<=deg;i++) big_init(&p->c[i]);
    return p;
}

// deep copy into new poly with degree deg_new (>= src->deg ok; extra coeffs = 0)
static Poly* poly_clone_with_deg(const Poly *src, int deg_new) {
    Poly *p = poly_new(deg_new);
    int m = (src->deg < deg_new) ? src->deg : deg_new;
    for (int i=0;i<=m;i++) big_copy(&p->c[i], &src->c[i]);
    return p;
}

static void poly_trim(Poly *p) {
    int d = p->deg;
    while (d > 0 && big_is_zero(&p->c[d])) d--;
    p->deg = d;
}

static void poly_free(Poly *p) {
    if (!p) return;
    for (int i=0;i<=p->deg;i++) big_free(&p->c[i]);
    free(p->c);
    free(p);
}

// p += wt * x * q  (truncate to kmax)
static void poly_add_shift_mul_u32(Poly *p, const Poly *q, uint32_t wt, int kmax) {
    if (wt == 0) return;
    int qdeg = q->deg;
    if (qdeg > kmax-1) qdeg = kmax-1;
    for (int i=0;i<=qdeg;i++) {
        int j = i+1;
        if (j > kmax) break;
        big_addmul_u32(&p->c[j], &q->c[i], wt);
    }
}

// ------------------------- small utilities -------------------------
static inline int popcount64(uint64_t x) { return __builtin_popcountll(x); }

// ------------------------- masks/vectors -------------------------
typedef struct { uint64_t mask; uint16_t wt; } MaskWt;

typedef struct {
    MaskWt *a;
    int n, cap;
} Vec;

static void vec_init(Vec *v) { v->a=NULL; v->n=0; v->cap=0; }

static void vec_push(Vec *v, uint64_t mask, uint16_t wt) {
    if (v->n == v->cap) {
        int ncap = v->cap ? (v->cap * 2) : 8;
        MaskWt *na = (MaskWt*)realloc(v->a, (size_t)ncap * sizeof(MaskWt));
        if (!na) { fprintf(stderr, "OOM\n"); exit(1); }
        v->a = na; v->cap = ncap;
    }
    v->a[v->n++] = (MaskWt){mask, wt};
}

static int cmp_maskwt(const void *A, const void *B) {
    const MaskWt *a = (const MaskWt*)A;
    const MaskWt *b = (const MaskWt*)B;
    int pa = popcount64(a->mask);
    int pb = popcount64(b->mask);
    if (pa != pb) return (pb - pa); // descending popcount
    if (a->mask < b->mask) return -1;
    if (a->mask > b->mask) return 1;
    return (int)a->wt - (int)b->wt;
}

// ------------------------- hashing -------------------------
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// memo: key -> Poly*
typedef struct {
    uint64_t key;
    Poly *val;
    uint8_t used;
} MemoEnt;

typedef struct {
    MemoEnt *t;
    size_t cap;
    size_t sz;
} MemoMap;

static void memomap_init(MemoMap *m) {
    m->cap = 1<<20; // start ~1M slots; grows
    m->sz = 0;
    m->t = (MemoEnt*)calloc(m->cap, sizeof(MemoEnt));
    if (!m->t) { fprintf(stderr, "OOM\n"); exit(1); }
}

static void memomap_free(MemoMap *m) {
    free(m->t);
    m->t=NULL; m->cap=0; m->sz=0;
}

static void memomap_rehash(MemoMap *m, size_t newcap) {
    MemoEnt *old = m->t;
    size_t oldcap = m->cap;
    m->t = (MemoEnt*)calloc(newcap, sizeof(MemoEnt));
    if (!m->t) { fprintf(stderr, "OOM\n"); exit(1); }
    m->cap = newcap;
    m->sz = 0;

    for (size_t i=0;i<oldcap;i++) if (old[i].used) {
        uint64_t key = old[i].key;
        Poly *val = old[i].val;
        size_t mask = newcap - 1;
        size_t h = (size_t)splitmix64(key) & mask;
        while (m->t[h].used) h = (h + 1) & mask;
        m->t[h].used = 1;
        m->t[h].key = key;
        m->t[h].val = val;
        m->sz++;
    }
    free(old);
}

static Poly* memomap_get(MemoMap *m, uint64_t key) {
    size_t mask = m->cap - 1;
    size_t h = (size_t)splitmix64(key) & mask;
    while (m->t[h].used) {
        if (m->t[h].key == key) return m->t[h].val;
        h = (h + 1) & mask;
    }
    return NULL;
}

static void memomap_put(MemoMap *m, uint64_t key, Poly *val) {
    if ((m->sz + 1) * 10 >= m->cap * 7) { // load > 0.7
        memomap_rehash(m, m->cap << 1);
    }
    size_t mask = m->cap - 1;
    size_t h = (size_t)splitmix64(key) & mask;
    while (m->t[h].used) {
        if (m->t[h].key == key) { m->t[h].val = val; return; }
        h = (h + 1) & mask;
    }
    m->t[h].used = 1;
    m->t[h].key = key;
    m->t[h].val = val;
    m->sz++;
}

// canon cache: U -> canon(U)
typedef struct {
    uint64_t key;
    uint64_t val;
    uint8_t used;
} CEnt;

typedef struct {
    CEnt *t;
    size_t cap;
    size_t sz;
} CanonMap;

static void canonmap_init(CanonMap *m) {
    m->cap = 1<<20;
    m->sz = 0;
    m->t = (CEnt*)calloc(m->cap, sizeof(CEnt));
    if (!m->t) { fprintf(stderr, "OOM\n"); exit(1); }
}

static void canonmap_rehash(CanonMap *m, size_t newcap) {
    CEnt *old = m->t;
    size_t oldcap = m->cap;
    m->t = (CEnt*)calloc(newcap, sizeof(CEnt));
    if (!m->t) { fprintf(stderr, "OOM\n"); exit(1); }
    m->cap = newcap;
    m->sz = 0;

    for (size_t i=0;i<oldcap;i++) if (old[i].used) {
        uint64_t key = old[i].key;
        uint64_t val = old[i].val;
        size_t mask = newcap - 1;
        size_t h = (size_t)splitmix64(key) & mask;
        while (m->t[h].used) h = (h + 1) & mask;
        m->t[h].used = 1;
        m->t[h].key = key;
        m->t[h].val = val;
        m->sz++;
    }
    free(old);
}

static uint8_t canonmap_get(CanonMap *m, uint64_t key, uint64_t *out) {
    size_t mask = m->cap - 1;
    size_t h = (size_t)splitmix64(key) & mask;
    while (m->t[h].used) {
        if (m->t[h].key == key) { *out = m->t[h].val; return 1; }
        h = (h + 1) & mask;
    }
    return 0;
}

static void canonmap_put(CanonMap *m, uint64_t key, uint64_t val) {
    if ((m->sz + 1) * 10 >= m->cap * 7) canonmap_rehash(m, m->cap << 1);
    size_t mask = m->cap - 1;
    size_t h = (size_t)splitmix64(key) & mask;
    while (m->t[h].used) {
        if (m->t[h].key == key) { m->t[h].val = val; return; }
        h = (h + 1) & mask;
    }
    m->t[h].used = 1;
    m->t[h].key = key;
    m->t[h].val = val;
    m->sz++;
}

// ------------------------- symmetry tables -------------------------
static int pair_idx[ROWS][ROWS]; // -1 except i<j
static uint16_t rowperm_table[120][1<<BITS_PER_COLOR]; // 120 * 1024

static int next_permutation_int(int *a, int n) {
    int i = n-2;
    while (i >= 0 && a[i] >= a[i+1]) i--;
    if (i < 0) return 0;
    int j = n-1;
    while (a[j] <= a[i]) j--;
    int tmp=a[i]; a[i]=a[j]; a[j]=tmp;
    for (int l=i+1, r=n-1; l<r; l++,r--) {
        tmp=a[l]; a[l]=a[r]; a[r]=tmp;
    }
    return 1;
}

static void precompute_pair_idx(void) {
    for (int i=0;i<ROWS;i++) for (int j=0;j<ROWS;j++) pair_idx[i][j] = -1;
    int k=0;
    for (int i=0;i<ROWS;i++) for (int j=i+1;j<ROWS;j++) pair_idx[i][j] = k++;
}

static void precompute_rowperm_table(void) {
    int perm[ROWS];
    for (int i=0;i<ROWS;i++) perm[i]=i;

    int pid=0;
    do {
        int map[BITS_PER_COLOR]; // old pair idx -> new pair idx
        int old=0;
        for (int i=0;i<ROWS;i++) for (int j=i+1;j<ROWS;j++) {
            int ni = perm[i], nj = perm[j];
            if (ni > nj) { int t=ni; ni=nj; nj=t; }
            map[old++] = pair_idx[ni][nj];
        }
        for (int v=0; v < (1<<BITS_PER_COLOR); v++) {
            uint16_t out=0;
            int x=v;
            while (x) {
                int lsb = x & -x;
                int b = __builtin_ctz((unsigned)x);
                out |= (uint16_t)(1u << map[b]);
                x ^= lsb;
            }
            rowperm_table[pid][v] = out;
        }

        pid++;
    } while (next_permutation_int(perm, ROWS));
    if (pid != 120) { fprintf(stderr, "perm count mismatch: %d\n", pid); exit(1); }
}

// canonicalise under row perms + colour perms (sort 4 blocks)
static inline uint64_t canon_rows_and_colors(uint64_t U) {
    const uint64_t MASK10 = (1ULL<<BITS_PER_COLOR) - 1ULL;
    uint16_t b0 = (uint16_t)((U >> (0*BITS_PER_COLOR)) & MASK10);
    uint16_t b1 = (uint16_t)((U >> (1*BITS_PER_COLOR)) & MASK10);
    uint16_t b2 = (uint16_t)((U >> (2*BITS_PER_COLOR)) & MASK10);
    uint16_t b3 = (uint16_t)((U >> (3*BITS_PER_COLOR)) & MASK10);

    uint64_t best = UINT64_MAX;
    for (int pid=0; pid<120; pid++) {
        uint16_t c0 = rowperm_table[pid][b0];
        uint16_t c1 = rowperm_table[pid][b1];
        uint16_t c2 = rowperm_table[pid][b2];
        uint16_t c3 = rowperm_table[pid][b3];

        // sorting network for 4 items
        uint16_t x0=c0, x1=c1, x2=c2, x3=c3;
        if (x0 > x1) { uint16_t t=x0; x0=x1; x1=t; }
        if (x2 > x3) { uint16_t t=x2; x2=x3; x3=t; }
        if (x0 > x2) { uint16_t t=x0; x0=x2; x2=t; }
        if (x1 > x3) { uint16_t t=x1; x1=x3; x3=t; }
        if (x1 > x2) { uint16_t t=x1; x1=x2; x2=t; }

        uint64_t V = (uint64_t)x0
            | ((uint64_t)x1 << (1*BITS_PER_COLOR))
            | ((uint64_t)x2 << (2*BITS_PER_COLOR))
            | ((uint64_t)x3 << (3*BITS_PER_COLOR));

        if (V < best) best = V;
    }
    return (best == UINT64_MAX) ? 0 : best;
}

// canon cache wrapper
static inline uint64_t canon_cached(CanonMap *cm, uint64_t U) {
    uint64_t v;
    if (canonmap_get(cm, U, &v)) return v;
    v = canon_rows_and_colors(U);
    canonmap_put(cm, U, v);
    return v;
}

// ------------------------- build perbit lists -------------------------
static uint16_t clique10[1<<ROWS]; // for subset of rows (5 bits) -> 10-bit clique mask

static void precompute_clique10(void) {
    for (int s=0; s < (1<<ROWS); s++) {
        uint16_t out=0;
        for (int i=0;i<ROWS;i++) if (s & (1<<i))
            for (int j=i+1;j<ROWS;j++) if (s & (1<<j)) {
                int idx = pair_idx[i][j];
                out |= (uint16_t)(1u<<idx);
            }
        clique10[s] = out;
    }
}

static void build_perbit(Vec perbit[TOTAL_BITS]) {
    for (int b=0;b<TOTAL_BITS;b++) vec_init(&perbit[b]);

    // compress mask -> multiplicity by enumerating all 4^5 columns
    // simple open addressing table for up to 1024 entries
    typedef struct { uint64_t key; uint16_t val; uint8_t used; } WEnt;
    size_t cap = 2048;
    WEnt *tab = (WEnt*)calloc(cap, sizeof(WEnt));
    if (!tab) { fprintf(stderr, "OOM\n"); exit(1); }
    size_t sz=0;

    for (int a=0;a<4;a++) for (int b=0;b<4;b++) for (int c=0;c<4;c++) for (int d=0;d<4;d++) for (int e=0;e<4;e++) {
        int col[5] = {a,b,c,d,e};
        uint8_t rows_of_color[4] = {0,0,0,0};
        for (int r=0;r<ROWS;r++) rows_of_color[col[r]] |= (uint8_t)(1u<<r);

        uint64_t mask = 0;
        for (int colr=0; colr<4; colr++) {
            uint16_t blk = clique10[rows_of_color[colr]];
            mask |= (uint64_t)blk << (colr*BITS_PER_COLOR);
        }
        // mask cannot be 0 for 5 rows/4 colours
        if (mask == 0) { fprintf(stderr, "Unexpected mask 0\n"); exit(1); }

        // insert/increment
        size_t m = cap - 1;
        size_t h = (size_t)splitmix64(mask) & m;
        while (tab[h].used) {
            if (tab[h].key == mask) { tab[h].val++; goto done_inc; }
            h = (h + 1) & m;
        }
        tab[h].used = 1;
        tab[h].key = mask;
        tab[h].val = 1;
        sz++;
done_inc: ;
    }

    // build perbit from weights
    for (size_t i=0;i<cap;i++) if (tab[i].used) {
        uint64_t mask = tab[i].key;
        uint16_t wt = tab[i].val;
        uint64_t x = mask;
        while (x) {
            uint64_t lsb = x & (~x + 1ULL);
            int bit = __builtin_ctzll(x);
            vec_push(&perbit[bit], mask, wt);
            x ^= lsb;
        }
    }

    for (int bit=0; bit<TOTAL_BITS; bit++) {
        qsort(perbit[bit].a, (size_t)perbit[bit].n, sizeof(MaskWt), cmp_maskwt);
    }

    free(tab);
}

// ------------------------- solver -------------------------
typedef struct {
    uint64_t miss, hit;
    double start, last;
    uint64_t last_miss;
    double report_every;
    int stderr_is_tty;
} Stats;

static void report_progress(const Stats *st, size_t cache_sz, int force) {
    double t = now_seconds();
    if (!force && (t - st->last) < st->report_every) return;
    // can't mutate st here; caller should
    (void)cache_sz;
}

static void stats_print_line(Stats *st, size_t cache_sz, int force) {
    double t = now_seconds();
    if (!force && (t - st->last) < st->report_every) return;
    double elapsed = t - st->start;
    double dt = t - st->last;
    uint64_t miss = st->miss;
    uint64_t hit = st->hit;
    uint64_t dmiss = miss - st->last_miss;
    double rate = (dt > 0) ? ((double)dmiss / dt) : 0.0;

    if (st->stderr_is_tty) {
        fprintf(stderr,
            "\rElapsed %8.1fs | states %12llu | hits %12llu | cache %12zu | miss/s %10.1f",
            elapsed,
            (unsigned long long)miss,
            (unsigned long long)hit,
            cache_sz,
            rate
        );
    } else {
        fprintf(stderr,
            "Elapsed %8.1fs | states %12llu | hits %12llu | cache %12zu | miss/s %10.1f\n",
            elapsed,
            (unsigned long long)miss,
            (unsigned long long)hit,
            cache_sz,
            rate
        );
    }
    fflush(stderr);
    st->last = t;
    st->last_miss = miss;
}

// key pack: (Uc << 6) | kmax  (kmax <= 40 fits)
static inline uint64_t pack_key(uint64_t Uc, int kmax) {
    return (Uc << 6) | (uint64_t)kmax;
}

static Poly* solve(uint64_t U, int kmax,
                   Vec perbit[TOTAL_BITS],
                   MemoMap *memo,
                   CanonMap *canon_cache,
                   Stats *st);

static Poly* solve(uint64_t U, int kmax,
                   Vec perbit[TOTAL_BITS],
                   MemoMap *memo,
                   CanonMap *canon_cache,
                   Stats *st)
{
    uint64_t Uc = canon_cached(canon_cache, U);

    int pc = popcount64(Uc);
    if (kmax > pc) kmax = pc;

    uint64_t key = pack_key(Uc, kmax);
    Poly *cached = memomap_get(memo, key);
    if (cached) { st->hit++; return cached; }

    st->miss++;
    stats_print_line(st, memo->sz, 0);

    if (kmax == 0 || Uc == 0) {
        Poly *p = poly_new(0);
        big_set_u64(&p->c[0], 1);
        memomap_put(memo, key, p);
        return p;
    }

    // choose pivot bit with fewest feasible masks
    int best_b = -1;
    int best_cnt = 1<<30;
    uint64_t x = Uc;
    while (x) {
        uint64_t lsb = x & (~x + 1ULL);
        int b = __builtin_ctzll(x);
        int cnt = 0;
        Vec *vb = &perbit[b];
        for (int i=0;i<vb->n;i++) {
            uint64_t m = vb->a[i].mask;
            if ((m & ~Uc) == 0) {
                cnt++;
                if (cnt >= best_cnt) break;
            }
        }
        if (cnt < best_cnt) { best_cnt = cnt; best_b = b; if (best_cnt == 0) break; }
        x ^= lsb;
    }
    if (best_b < 0) {
        // no bits? shouldn't happen given checks, but return 1
        Poly *p = poly_new(0);
        big_set_u64(&p->c[0], 1);
        memomap_put(memo, key, p);
        return p;
    }

    uint64_t bit = 1ULL << best_b;

    // Case 1: don't use bit
    Poly *base = solve(Uc ^ bit, kmax, perbit, memo, canon_cache, st);
    Poly *p = poly_clone_with_deg(base, kmax);

    // Case 2: choose exactly one mask containing bit
    Vec *vb = &perbit[best_b];
    for (int i=0;i<vb->n;i++) {
        uint64_t m = vb->a[i].mask;
        uint16_t wt = vb->a[i].wt;
        if ((m & ~Uc) != 0) continue;
        Poly *q = solve(Uc & ~m, kmax - 1, perbit, memo, canon_cache, st);
        poly_add_shift_mul_u32(p, q, (uint32_t)wt, kmax);
    }

    poly_trim(p);
    memomap_put(memo, key, p);
    return p;
}

// ------------------------- main -------------------------
int main(void) {
    precompute_pair_idx();
    precompute_clique10();
    precompute_rowperm_table();

    Vec perbit[TOTAL_BITS];
    build_perbit(perbit);

    MemoMap memo;
    memomap_init(&memo);

    CanonMap canon_cache;
    canonmap_init(&canon_cache);

    Stats st = {0};
    st.start = now_seconds();
    st.last = st.start;
    st.last_miss = 0;
    st.report_every = 2.0;
    st.stderr_is_tty = isatty(fileno(stderr));

    uint64_t U0 = (1ULL << TOTAL_BITS) - 1ULL;

    // compute F(k) = k! * c_k for k=0..40 incrementally
    double last_time = now_seconds();
    uint64_t last_miss = st.miss;

    // store final F(k) as Big
    Big F[41];
    for (int i=0;i<=40;i++) big_init(&F[i]);

    for (int k=0;k<=40;k++) {
        Poly *c = solve(U0, k, perbit, &memo, &canon_cache, &st);

        double tnow = now_seconds();
        uint64_t dm = st.miss - last_miss;
        double dt = tnow - last_time;

        Big ck; big_init(&ck);
        if (k <= c->deg) big_copy(&ck, &c->c[k]); else big_set_zero(&ck);

        // Fk = ck * k!
        Big fk; big_init(&fk);
        big_copy(&fk, &ck);
        for (int i=2;i<=k;i++) big_mul_u32_inplace(&fk, (uint32_t)i);

        big_copy(&F[k], &fk);

        if (st.stderr_is_tty) {
            fputs("\r\033[2K", stderr);
            fflush(stderr);
        }

        printf("k=%2d  Δstates=%10llu  Δt=%8.2fs  F(k)=",
               k, (unsigned long long)dm, dt);
        big_print(&fk);
        putchar('\n');

        big_free(&ck);
        big_free(&fk);

        last_time = tnow;
        last_miss = st.miss;
    }

    stats_print_line(&st, memo.sz, 1);
    if (st.stderr_is_tty) fprintf(stderr, "\n");

    printf("\nF(n) table (5×n, 4 colours):\n");
    for (int n=0;n<=40;n++) {
        printf("%d ", n);
        big_print(&F[n]);
        putchar('\n');
    }
    printf("\nFor n > 40, F(n) = 0 (only 40 tokens total).\n");

    for (int i=0;i<=40;i++) big_free(&F[i]);

    // free perbit vectors
    for (int b=0;b<TOTAL_BITS;b++) free(perbit[b].a);

    // Note: memo stores Polys we never free before process exit. If you want,
    // we can add a pass to free all memo values.
    memomap_free(&memo);
    free(canon_cache.t);

    return 0;
}
