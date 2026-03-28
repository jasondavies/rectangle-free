#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SMALL_GRAPH_LOOKUP_MAX_N 7
#define SMALL_GRAPH_TABLE_MAGIC UINT64_C(0x534750375441424c)
#define SMALL_GRAPH_TABLE_VERSION 1U

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t max_n;
} SmallGraphTableHeader;

static int32_t* g_coeffs[SMALL_GRAPH_LOOKUP_MAX_N + 1] = {0};
static uint8_t g_edge_u[SMALL_GRAPH_LOOKUP_MAX_N + 1][21];
static uint8_t g_edge_v[SMALL_GRAPH_LOOKUP_MAX_N + 1][21];
static uint32_t g_graph_count[SMALL_GRAPH_LOOKUP_MAX_N + 1] = {0};

static void* checked_calloc(size_t count, size_t size, const char* label) {
    void* p = calloc(count, size);
    if (!p) {
        fprintf(stderr, "Allocation failed for %s\n", label);
        exit(1);
    }
    return p;
}

static double wall_time_seconds(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static inline uint32_t small_graph_stride(int n) {
    return (uint32_t)(n + 1);
}

static inline uint32_t small_graph_edge_total(int n) {
    return (uint32_t)(n * (n - 1) / 2);
}

static inline int32_t* small_graph_poly_slot(int n, uint32_t mask) {
    return g_coeffs[n] + (size_t)mask * (size_t)small_graph_stride(n);
}

static void init_layout(void) {
    g_graph_count[0] = 1;
    for (int n = 1; n <= SMALL_GRAPH_LOOKUP_MAX_N; n++) {
        g_graph_count[n] = 1U << small_graph_edge_total(n);
        int bit = 0;
        for (int j = 1; j < n; j++) {
            for (int i = 0; i < j; i++, bit++) {
                g_edge_u[n][bit] = (uint8_t)i;
                g_edge_v[n][bit] = (uint8_t)j;
            }
        }
    }
}

static uint32_t pack_mask_from_rows(const uint8_t* rows, int n) {
    uint32_t mask = 0;
    int bit = 0;
    for (int j = 1; j < n; j++) {
        for (int i = 0; i < j; i++, bit++) {
            if ((rows[i] >> j) & 1U) mask |= 1U << bit;
        }
    }
    return mask;
}

static uint32_t contract_mask(uint32_t mask, int n, int u, int v) {
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
    return pack_mask_from_rows(compact, n - 1);
}

static void allocate_tables(void) {
    for (int n = 0; n <= SMALL_GRAPH_LOOKUP_MAX_N; n++) {
        uint32_t graph_count = g_graph_count[n];
        uint32_t stride = small_graph_stride(n);
        g_coeffs[n] = checked_calloc((size_t)graph_count * (size_t)stride, sizeof(int32_t),
                                     "small_graph_lookup_table");
    }
}

static void generate_tables(void) {
    small_graph_poly_slot(0, 0)[0] = 1;
    for (int n = 1; n <= SMALL_GRAPH_LOOKUP_MAX_N; n++) {
        uint32_t graph_count = g_graph_count[n];
        small_graph_poly_slot(n, 0)[n] = 1;
        for (uint32_t mask = 1; mask < graph_count; mask++) {
            uint32_t del_mask = mask & (mask - 1U);
            int edge_bit = __builtin_ctz(mask);
            int u = g_edge_u[n][edge_bit];
            int v = g_edge_v[n][edge_bit];
            uint32_t cont_mask = contract_mask(mask, n, u, v);
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

static void write_tables(const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        perror(path);
        exit(1);
    }

    SmallGraphTableHeader header = {
        .magic = SMALL_GRAPH_TABLE_MAGIC,
        .version = SMALL_GRAPH_TABLE_VERSION,
        .max_n = SMALL_GRAPH_LOOKUP_MAX_N,
    };
    if (fwrite(&header, sizeof(header), 1, f) != 1) {
        perror("fwrite header");
        exit(1);
    }
    for (int n = 0; n <= SMALL_GRAPH_LOOKUP_MAX_N; n++) {
        uint32_t count = g_graph_count[n] * small_graph_stride(n);
        if (fwrite(g_coeffs[n], sizeof(int32_t), count, f) != count) {
            perror("fwrite coeffs");
            exit(1);
        }
    }
    fclose(f);
}

static void free_tables(void) {
    for (int n = 0; n <= SMALL_GRAPH_LOOKUP_MAX_N; n++) free(g_coeffs[n]);
}

int main(int argc, char** argv) {
    const char* out_path = (argc >= 2) ? argv[1] : "small_graph_lookup_n7.bin";
    double t0 = wall_time_seconds();
    init_layout();
    allocate_tables();
    generate_tables();
    write_tables(out_path);
    double elapsed = wall_time_seconds() - t0;

    long long total_graphs = 0;
    for (int n = 0; n <= SMALL_GRAPH_LOOKUP_MAX_N; n++) total_graphs += g_graph_count[n];
    printf("Wrote %s for %lld labelled graphs up to n=%d in %.2f seconds\n",
           out_path, total_graphs, SMALL_GRAPH_LOOKUP_MAX_N, elapsed);

    free_tables();
    return 0;
}
