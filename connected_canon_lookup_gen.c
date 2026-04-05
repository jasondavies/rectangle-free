#include "src/partition_poly_internal.h"

#include <ctype.h>

typedef struct {
    uint64_t mask;
    int32_t coeffs[CONNECTED_CANON_LOOKUP_MAX_N];
} GeneratorEntry;

static void* checked_realloc(void* ptr, size_t size, const char* label) {
    void* out = realloc(ptr, size);
    if (!out) {
        fprintf(stderr, "Reallocation failed for %s\n", label);
        exit(1);
    }
    return out;
}

static size_t connected_canon_lookup_entry_file_size_local(int n) {
    return sizeof(uint64_t) + (size_t)n * sizeof(int32_t);
}

static void row_graph_cache_init_poly(RowGraphCache* cache, int size, int mask, int probe,
                                      int poly_len, const char* prefix) {
    memset(cache, 0, sizeof(*cache));
    cache->mask = mask;
    cache->probe = probe;
    cache->poly_len = poly_len;

    {
        char label[64];
        snprintf(label, sizeof(label), "%s_keys", prefix);
        cache->keys = checked_aligned_alloc(64, sizeof(CacheKey) * (size_t)size, label);
        snprintf(label, sizeof(label), "%s_stamps", prefix);
        cache->stamps = checked_aligned_alloc(64, sizeof(uint32_t) * (size_t)size, label);
        snprintf(label, sizeof(label), "%s_rows", prefix);
        cache->rows =
            checked_aligned_alloc(64, sizeof(AdjWord) * (size_t)size * MAXN_NAUTY, label);
        snprintf(label, sizeof(label), "%s_coeffs", prefix);
        cache->coeffs = checked_aligned_alloc(
            64, sizeof(PolyCoeff) * (size_t)size * (size_t)poly_len, label);
    }

    memset(cache->keys, 0, sizeof(CacheKey) * (size_t)size);
    memset(cache->stamps, 0, sizeof(uint32_t) * (size_t)size);
}

static void row_graph_cache_free_all(RowGraphCache* cache) {
    free(cache->keys);
    free(cache->stamps);
    free(cache->rows);
    free(cache->x_pows);
    free(cache->degs);
    free(cache->coeffs);
    memset(cache, 0, sizeof(*cache));
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

    {
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
                    out->adj[i] |= (AdjWord)(UINT64_C(1) << j);
                    out->adj[j] |= (AdjWord)(UINT64_C(1) << i);
                }
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
    const char* out_path = (argc >= 2) ? argv[1]
                                       : (target_n == 10 ? "connected_canon_lookup_n10.bin"
                                                         : "connected_canon_lookup_n9.bin");

    if (target_n <= 0 || target_n > CONNECTED_CANON_LOOKUP_MAX_N) {
        fprintf(stderr, "target n must be in [1, %d]\n", CONNECTED_CANON_LOOKUP_MAX_N);
        return 1;
    }

    nauty_check(WORDSIZE, SETWORDSNEEDED(MAXN_NAUTY), MAXN_NAUTY, NAUTYVERSIONID);
    factorial[0] = 1;
    for (int i = 1; i <= 19; i++) factorial[i] = factorial[i - 1] * i;
    small_graph_lookup_init();

    RowGraphCache cache;
    RowGraphCache raw_cache;
    row_graph_cache_init_poly(&cache, CACHE_SIZE, CACHE_MASK, CACHE_PROBE, target_n + 1, "cache");
    row_graph_cache_init_poly(&raw_cache, RAW_CACHE_SIZE, RAW_CACHE_MASK, RAW_CACHE_PROBE,
                              target_n + 1, "raw_cache");

    NautyWorkspace ws = {0};
    nauty_workspace_init(&ws, target_n);

    size_t capacity = 300000;
    size_t count = 0;
    GeneratorEntry* entries = checked_calloc(capacity, sizeof(*entries), "generator_entries");
    long long local_canon_calls = 0;
    long long local_cache_hits = 0;
    long long local_raw_cache_hits = 0;
    long long max_abs_coeff = 0;
    double start = omp_get_wtime();

    {
        char line[256];
        while (fgets(line, sizeof(line), stdin)) {
            Graph g;
            if (!parse_graph6_line(line, &g) || g.n != target_n) continue;

            GraphPoly poly;
            solve_graph_poly(&g, &cache, &raw_cache, &ws, &local_canon_calls, &local_cache_hits,
                             &local_raw_cache_hits, NULL, &poly);

            Graph canon;
            get_canonical_graph(&g, &canon, &ws, NULL);

            if (count == capacity) {
                capacity *= 2;
                entries = checked_realloc(entries, capacity * sizeof(*entries),
                                          "generator_entries");
            }

            if (poly.x_pow != 1 || poly.deg != target_n - 1) {
                fprintf(stderr, "Expected connected residual form x*q(x) for mask %llu\n",
                        (unsigned long long)graph_pack_upper_mask64(&canon));
                row_graph_cache_free_all(&cache);
                row_graph_cache_free_all(&raw_cache);
                nauty_workspace_free(&ws);
                small_graph_lookup_free();
                free(entries);
                return 1;
            }

            entries[count].mask = graph_pack_upper_mask64(&canon);
            for (int i = 0; i < target_n; i++) {
                PolyCoeff coeff = poly.coeffs[i];
                if (coeff < INT32_MIN || coeff > INT32_MAX) {
                    fprintf(stderr, "Coefficient out of int32 range for mask %llu\n",
                            (unsigned long long)entries[count].mask);
                    row_graph_cache_free_all(&cache);
                    row_graph_cache_free_all(&raw_cache);
                    nauty_workspace_free(&ws);
                    small_graph_lookup_free();
                    free(entries);
                    return 1;
                }
                entries[count].coeffs[i] = (int32_t)coeff;
                {
                    long long abs_coeff =
                        (coeff < 0) ? (long long)(-coeff) : (long long)coeff;
                    if (abs_coeff > max_abs_coeff) max_abs_coeff = abs_coeff;
                }
            }
            count++;
        }
    }

    qsort(entries, count, sizeof(*entries), generator_entry_cmp);
    for (size_t i = 1; i < count; i++) {
        if (entries[i - 1].mask == entries[i].mask) {
            fprintf(stderr, "Duplicate canonical mask in generator output: %llu\n",
                    (unsigned long long)entries[i].mask);
            row_graph_cache_free_all(&cache);
            row_graph_cache_free_all(&raw_cache);
            nauty_workspace_free(&ws);
            small_graph_lookup_free();
            free(entries);
            return 1;
        }
    }

    {
        FILE* out = fopen(out_path, "wb");
        if (!out) {
            perror(out_path);
            row_graph_cache_free_all(&cache);
            row_graph_cache_free_all(&raw_cache);
            nauty_workspace_free(&ws);
            small_graph_lookup_free();
            free(entries);
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
            row_graph_cache_free_all(&cache);
            row_graph_cache_free_all(&raw_cache);
            nauty_workspace_free(&ws);
            small_graph_lookup_free();
            free(entries);
            return 1;
        }

        for (size_t i = 0; i < count; i++) {
            if (fwrite(&entries[i].mask, sizeof(entries[i].mask), 1, out) != 1 ||
                fwrite(entries[i].coeffs, sizeof(entries[i].coeffs[0]), (size_t)target_n,
                       out) != (size_t)target_n) {
                perror("fwrite connected canonical lookup");
                fclose(out);
                row_graph_cache_free_all(&cache);
                row_graph_cache_free_all(&raw_cache);
                nauty_workspace_free(&ws);
                small_graph_lookup_free();
                free(entries);
                return 1;
            }
        }
        fclose(out);
    }

    printf("Wrote %s with %zu connected canonical graphs on n=%d in %.2f seconds\n", out_path,
           count, target_n, omp_get_wtime() - start);
    printf("Entry payload bytes: %zu\n",
           connected_canon_lookup_entry_file_size_local(target_n));
    printf("Generator stats: canon_calls=%lld cache_hits=%lld raw_hits=%lld max_abs_coeff=%lld\n",
           local_canon_calls, local_cache_hits, local_raw_cache_hits, max_abs_coeff);

    free(entries);
    nauty_workspace_free(&ws);
    row_graph_cache_free_all(&cache);
    row_graph_cache_free_all(&raw_cache);
    small_graph_lookup_free();
    return 0;
}
