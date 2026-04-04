#define main partition_poly_solver_main
#include "partition_poly.c"
#undef main

#include <ctype.h>

typedef struct {
    uint64_t mask;
    int32_t coeffs[CONNECTED_CANON_LOOKUP_MAX_N + 1];
} GeneratorEntry;

static void* checked_realloc(void* ptr, size_t size, const char* label) {
    void* out = realloc(ptr, size);
    if (!out) {
        fprintf(stderr, "Reallocation failed for %s\n", label);
        exit(1);
    }
    return out;
}

static int parse_graph6_line(const char* line, Graph* out) {
    while (*line && isspace((unsigned char)*line)) line++;
    if (!*line) return 0;
    if (line[0] == '>' || line[0] == ':') return 0;

    int n = line[0] - 63;
    if (n <= 0 || n > MAXN_NAUTY) return 0;

    out->n = (uint8_t)n;
    out->vertex_mask = graph_row_mask(n);
    memset(out->adj, 0, sizeof(out->adj));

    int payload_idx = 1;
    int bits_left = 0;
    int chunk = 0;
    for (int j = 1; j < n; j++) {
        for (int i = 0; i < j; i++) {
            if (bits_left == 0) {
                if (line[payload_idx] == '\0' || line[payload_idx] == '\n') {
                    fprintf(stderr, "Truncated graph6 line\n");
                    exit(1);
                }
                chunk = line[payload_idx++] - 63;
                bits_left = 6;
            }
            bits_left--;
            if ((chunk >> bits_left) & 1) {
                out->adj[i] |= (AdjWord)(1ULL << j);
                out->adj[j] |= (AdjWord)(1ULL << i);
            }
        }
    }
    return 1;
}

static int generator_entry_cmp(const void* lhs, const void* rhs) {
    const GeneratorEntry* a = lhs;
    const GeneratorEntry* b = rhs;
    if (a->mask < b->mask) return -1;
    if (a->mask > b->mask) return 1;
    return 0;
}

int main(int argc, char** argv) {
    int target_n = (argc >= 3) ? atoi(argv[2]) : 9;
    const char* out_path = (argc >= 2) ? argv[1] :
        (target_n == 10 ? "connected_canon_lookup_n10.bin" : "connected_canon_lookup_n9.bin");

    if (target_n <= 0 || target_n > CONNECTED_CANON_LOOKUP_MAX_N) {
        fprintf(stderr, "target n must be in [1, %d]\n", CONNECTED_CANON_LOOKUP_MAX_N);
        return 1;
    }

    nauty_check(WORDSIZE, SETWORDSNEEDED(MAXN_NAUTY), MAXN_NAUTY, NAUTYVERSIONID);
    factorial[0] = 1;
    for (int i = 1; i <= 19; i++) factorial[i] = factorial[i - 1] * i;
    small_graph_lookup_init();

    RowGraphCache cache = {0};
    RowGraphCache raw_cache = {0};
    cache.mask = CACHE_MASK;
    cache.probe = CACHE_PROBE;
    cache.poly_len = target_n + 1;
    cache.keys = checked_aligned_alloc(64, sizeof(CacheKey) * CACHE_SIZE, "cache_keys");
    cache.stamps = checked_aligned_alloc(64, sizeof(uint32_t) * CACHE_SIZE, "cache_stamps");
    cache.rows = checked_aligned_alloc(64, sizeof(AdjWord) * CACHE_SIZE * MAXN_NAUTY, "cache_rows");
    cache.degs = checked_aligned_alloc(64, sizeof(uint8_t) * CACHE_SIZE, "cache_degs");
    cache.coeffs =
        checked_aligned_alloc(64, sizeof(PolyCoeff) * CACHE_SIZE * (size_t)cache.poly_len, "cache_coeffs");
    memset(cache.keys, 0, sizeof(CacheKey) * CACHE_SIZE);
    memset(cache.stamps, 0, sizeof(uint32_t) * CACHE_SIZE);

    raw_cache.mask = RAW_CACHE_MASK;
    raw_cache.probe = RAW_CACHE_PROBE;
    raw_cache.poly_len = target_n + 1;
    raw_cache.keys = checked_aligned_alloc(64, sizeof(CacheKey) * RAW_CACHE_SIZE, "raw_cache_keys");
    raw_cache.stamps = checked_aligned_alloc(64, sizeof(uint32_t) * RAW_CACHE_SIZE, "raw_cache_stamps");
    raw_cache.rows =
        checked_aligned_alloc(64, sizeof(AdjWord) * RAW_CACHE_SIZE * MAXN_NAUTY, "raw_cache_rows");
    raw_cache.degs = checked_aligned_alloc(64, sizeof(uint8_t) * RAW_CACHE_SIZE, "raw_cache_degs");
    raw_cache.coeffs = checked_aligned_alloc(64, sizeof(PolyCoeff) * RAW_CACHE_SIZE * (size_t)raw_cache.poly_len,
                                             "raw_cache_coeffs");
    memset(raw_cache.keys, 0, sizeof(CacheKey) * RAW_CACHE_SIZE);
    memset(raw_cache.stamps, 0, sizeof(uint32_t) * RAW_CACHE_SIZE);

    NautyWorkspace ws = {0};
    nauty_workspace_init(&ws, target_n);

    size_t capacity = 300000;
    size_t count = 0;
    GeneratorEntry* entries = checked_calloc(capacity, sizeof(*entries), "generator_entries");
    long long local_canon_calls = 0;
    long long local_cache_hits = 0;
    long long local_raw_cache_hits = 0;
    double start = omp_get_wtime();
    long long max_abs_coeff = 0;

    char line[256];
    while (fgets(line, sizeof(line), stdin)) {
        Graph g;
        if (!parse_graph6_line(line, &g)) continue;
        if (g.n != target_n) continue;

        GraphPoly poly;
        solve_graph_poly(&g, &cache, &raw_cache, &ws,
                         &local_canon_calls, &local_cache_hits, &local_raw_cache_hits,
                         NULL, &poly);
        Graph canon;
        get_canonical_graph(&g, &canon, &ws, NULL);

        if (count == capacity) {
            capacity *= 2;
            entries = checked_realloc(entries, capacity * sizeof(*entries), "generator_entries");
        }

        entries[count].mask = graph_pack_upper_mask64(&canon);
        for (int i = 0; i <= target_n; i++) {
            PolyCoeff coeff = (i <= poly.deg) ? poly.coeffs[i] : 0;
            if (coeff < INT32_MIN || coeff > INT32_MAX) {
                fprintf(stderr, "Coefficient out of int32 range for mask %llu\n",
                        (unsigned long long)entries[count].mask);
                return 1;
            }
            entries[count].coeffs[i] = (int32_t)coeff;
            long long abs_coeff = (coeff < 0) ? (long long)(-coeff) : (long long)coeff;
            if (abs_coeff > max_abs_coeff) max_abs_coeff = abs_coeff;
        }
        count++;
    }

    qsort(entries, count, sizeof(*entries), generator_entry_cmp);
    for (size_t i = 1; i < count; i++) {
        if (entries[i - 1].mask == entries[i].mask) {
            fprintf(stderr, "Duplicate canonical mask in generator output: %llu\n",
                    (unsigned long long)entries[i].mask);
            return 1;
        }
    }

    FILE* out = fopen(out_path, "wb");
    if (!out) {
        perror(out_path);
        return 1;
    }
    ConnectedCanonLookupHeader header = {
        .magic = CONNECTED_CANON_LOOKUP_MAGIC,
        .version = CONNECTED_CANON_LOOKUP_VERSION,
        .n = (uint32_t)target_n,
        .count = (uint32_t)count,
    };
    if (fwrite(&header, sizeof(header), 1, out) != 1) {
        perror("fwrite connected canonical lookup");
        fclose(out);
        return 1;
    }
    for (size_t i = 0; i < count; i++) {
        if (fwrite(&entries[i].mask, sizeof(entries[i].mask), 1, out) != 1 ||
            fwrite(entries[i].coeffs, sizeof(entries[i].coeffs[0]), (size_t)target_n + 1, out) !=
                (size_t)target_n + 1) {
            perror("fwrite connected canonical lookup");
            fclose(out);
            return 1;
        }
    }
    fclose(out);

    double elapsed = omp_get_wtime() - start;
    printf("Wrote %s with %zu connected canonical graphs on n=%d in %.2f seconds\n",
           out_path, count, target_n, elapsed);
    printf("Entry payload bytes: %zu\n", connected_canon_lookup_entry_file_size(target_n));
    printf("Generator stats: canon_calls=%lld cache_hits=%lld raw_hits=%lld max_abs_coeff=%lld\n",
           local_canon_calls, local_cache_hits, local_raw_cache_hits, max_abs_coeff);

    free(entries);
    nauty_workspace_free(&ws);
    free(cache.keys);
    free(cache.stamps);
    free(cache.rows);
    free(cache.degs);
    free(cache.coeffs);
    free(raw_cache.keys);
    free(raw_cache.stamps);
    free(raw_cache.rows);
    free(raw_cache.degs);
    free(raw_cache.coeffs);
    small_graph_lookup_free();
    return 0;
}
