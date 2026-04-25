#include "partition_poly.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

PrefixId* g_live_prefix2_i = NULL;
PrefixId* g_live_prefix2_j = NULL;
long long g_live_prefix2_count = 0;

long long completed_tasks = 0;
Poly global_poly = {0};

int g_rows = DEFAULT_ROWS;
int g_cols = DEFAULT_COLS;
ProgressReporter progress_reporter;
int g_use_raw_cache = 1;
int g_use_wl_canon = 1;
int g_disable_nauty = 1;
int g_wl_canon_min_n = 8;
long long g_wl_canon_enum_limit = 16;
long long progress_last_reported = 0;
int g_adaptive_subdivide = DEFAULT_ADAPTIVE_SUBDIVIDE;
int g_adaptive_max_depth = DEFAULT_ADAPTIVE_MAX_DEPTH;
long long g_adaptive_work_budget = DEFAULT_ADAPTIVE_WORK_BUDGET;
__thread ProfileStats* tls_profile = NULL;
__thread GraphHardStats* tls_hard_graph_stats = NULL;
__thread long long* tls_adaptive_work_counter = NULL;
const char* g_task_times_out_path = NULL;
long long g_task_times_first_task = 0;
long long g_task_times_count = 0;
double* g_task_times_values = NULL;
int g_effective_prefix_depth = 0;
double g_queue_profile_report_step = 0.0;
int g_profile_separators = 0;
static FILE* g_hard_miss_log_file = NULL;
static pthread_mutex_t g_hard_miss_log_mutex = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_hard_miss_log_records = 0;

#define HARD_GRAPH_CACHE_LOCKS 4096
#define HARD_GRAPH_CACHE_SLAB_ENTRIES 32768
#define HARD_GRAPH_CACHE_FILE_VERSION 1U

static const unsigned char HARD_GRAPH_CACHE_FILE_MAGIC[8] =
    {'R', 'H', 'C', 'A', 'C', 'H', '1', 0};

typedef struct HardGraphCacheEntry {
    struct HardGraphCacheEntry* next;
    uint64_t hash;
    uint64_t sig[GRAPH_SIG_WORDS];
    GraphResult value;
    uint8_t n;
} HardGraphCacheEntry;

typedef struct HardGraphCacheSlab {
    struct HardGraphCacheSlab* next;
    size_t used;
    HardGraphCacheEntry entries[HARD_GRAPH_CACHE_SLAB_ENTRIES];
} HardGraphCacheSlab;

typedef struct {
    HardGraphCacheEntry** buckets;
    size_t bucket_count;
    uint64_t mask;
    pthread_mutex_t locks[HARD_GRAPH_CACHE_LOCKS];
    pthread_mutex_t alloc_lock;
    pthread_mutex_t save_lock;
    HardGraphCacheSlab* slabs;
    HardGraphCacheSlab* active_slab;
    size_t entry_count;
    size_t max_entries;
    FILE* save_file;
    int enabled;
    int use_locks;
    atomic_uint_fast64_t lookups;
    atomic_uint_fast64_t hits;
    atomic_uint_fast64_t stores;
    atomic_uint_fast64_t skipped_stores;
    atomic_uint_fast64_t duplicate_stores;
    atomic_uint_fast64_t loaded_entries;
    atomic_uint_fast64_t saved_entries;
} HardGraphCache;

static HardGraphCache g_hard_graph_cache = {0};

static int checked_fwrite(const void* ptr, size_t size, size_t count, FILE* f,
                          const char* label) {
    if (fwrite(ptr, size, count, f) == count) return 1;
    fprintf(stderr, "Failed to write %s\n", label);
    return 0;
}

typedef struct {
    const uint8_t* cur;
    const uint8_t* end;
    const char* path;
} HardGraphCacheReader;

static int hard_graph_cache_read_bytes(HardGraphCacheReader* reader, void* dst,
                                       size_t len, const char* label) {
    if ((size_t)(reader->end - reader->cur) < len) {
        fprintf(stderr, "Unexpected EOF while reading %s from %s\n", label, reader->path);
        return 0;
    }
    memcpy(dst, reader->cur, len);
    reader->cur += len;
    return 1;
}

int hard_miss_log_init_from_env(void) {
    const char* path = getenv("RECT_HARD_MISS_LOG");
    if (!path || !*path) return 1;

    g_hard_miss_log_file = fopen(path, "wb");
    if (!g_hard_miss_log_file) {
        fprintf(stderr, "Failed to open RECT_HARD_MISS_LOG=%s: %s\n", path, strerror(errno));
        return 0;
    }

    const unsigned char magic[8] = {'R', 'H', 'M', 'I', 'S', 'S', '1', 0};
    uint32_t version = 1;
    uint32_t maxn = MAXN_NAUTY;
    uint32_t adjword_bits = (uint32_t)(8U * sizeof(AdjWord));
    uint32_t rows = (uint32_t)g_rows;
    uint32_t cols = (uint32_t)g_cols;

    if (!checked_fwrite(magic, sizeof(magic), 1, g_hard_miss_log_file, "magic") ||
        !checked_fwrite(&version, sizeof(version), 1, g_hard_miss_log_file, "version") ||
        !checked_fwrite(&maxn, sizeof(maxn), 1, g_hard_miss_log_file, "maxn") ||
        !checked_fwrite(&adjword_bits, sizeof(adjword_bits), 1, g_hard_miss_log_file, "adjword_bits") ||
        !checked_fwrite(&rows, sizeof(rows), 1, g_hard_miss_log_file, "rows") ||
        !checked_fwrite(&cols, sizeof(cols), 1, g_hard_miss_log_file, "cols")) {
        fclose(g_hard_miss_log_file);
        g_hard_miss_log_file = NULL;
        return 0;
    }

    printf("Hard-miss log: %s\n", path);
    return 1;
}

void hard_miss_log_record(const Graph* g, uint64_t hash, int max_degree) {
    if (!g_hard_miss_log_file) return;

    uint8_t n = g->n;
    uint8_t degree = (max_degree < 0) ? 0 : (uint8_t)max_degree;
    uint16_t reserved = 0;

    pthread_mutex_lock(&g_hard_miss_log_mutex);
    checked_fwrite(&n, sizeof(n), 1, g_hard_miss_log_file, "record n");
    checked_fwrite(&degree, sizeof(degree), 1, g_hard_miss_log_file, "record degree");
    checked_fwrite(&reserved, sizeof(reserved), 1, g_hard_miss_log_file, "record reserved");
    checked_fwrite(&hash, sizeof(hash), 1, g_hard_miss_log_file, "record hash");
    for (int i = 0; i < MAXN_NAUTY; i++) {
        uint64_t row = (i < g->n) ? (uint64_t)g->adj[i] : 0;
        checked_fwrite(&row, sizeof(row), 1, g_hard_miss_log_file, "record row");
    }
    g_hard_miss_log_records++;
    pthread_mutex_unlock(&g_hard_miss_log_mutex);
}

void hard_miss_log_close(void) {
    if (!g_hard_miss_log_file) return;
    fclose(g_hard_miss_log_file);
    g_hard_miss_log_file = NULL;
    printf("Hard-miss log records: %llu\n", (unsigned long long)g_hard_miss_log_records);
}

static int hard_graph_cache_write_file_header(FILE* f) {
    uint32_t version = HARD_GRAPH_CACHE_FILE_VERSION;
    uint32_t count_k4 = RECT_COUNT_K4;
    uint32_t max_rows = MAX_ROWS;
    uint32_t max_cols = MAX_COLS;
    uint32_t maxn = MAXN_NAUTY;
    uint32_t sig_words = GRAPH_SIG_WORDS;
    uint32_t result_size = sizeof(GraphResult);
    uint32_t coeff_size = sizeof(PolyCoeff);

    return checked_fwrite(HARD_GRAPH_CACHE_FILE_MAGIC, sizeof(HARD_GRAPH_CACHE_FILE_MAGIC), 1, f,
                          "hard-cache magic") &&
           checked_fwrite(&version, sizeof(version), 1, f, "hard-cache version") &&
           checked_fwrite(&count_k4, sizeof(count_k4), 1, f, "hard-cache mode") &&
           checked_fwrite(&max_rows, sizeof(max_rows), 1, f, "hard-cache max rows") &&
           checked_fwrite(&max_cols, sizeof(max_cols), 1, f, "hard-cache max cols") &&
           checked_fwrite(&maxn, sizeof(maxn), 1, f, "hard-cache maxn") &&
           checked_fwrite(&sig_words, sizeof(sig_words), 1, f, "hard-cache sig words") &&
           checked_fwrite(&result_size, sizeof(result_size), 1, f, "hard-cache result size") &&
           checked_fwrite(&coeff_size, sizeof(coeff_size), 1, f, "hard-cache coeff size");
}

static int hard_graph_cache_read_file_header_mapped(HardGraphCacheReader* reader) {
    unsigned char magic[8];
    uint32_t version = 0;
    uint32_t count_k4 = 0;
    uint32_t max_rows = 0;
    uint32_t max_cols = 0;
    uint32_t maxn = 0;
    uint32_t sig_words = 0;
    uint32_t result_size = 0;
    uint32_t coeff_size = 0;

    if (!hard_graph_cache_read_bytes(reader, magic, sizeof(magic), "hard-cache magic") ||
        !hard_graph_cache_read_bytes(reader, &version, sizeof(version), "hard-cache version") ||
        !hard_graph_cache_read_bytes(reader, &count_k4, sizeof(count_k4), "hard-cache mode") ||
        !hard_graph_cache_read_bytes(reader, &max_rows, sizeof(max_rows), "hard-cache max rows") ||
        !hard_graph_cache_read_bytes(reader, &max_cols, sizeof(max_cols), "hard-cache max cols") ||
        !hard_graph_cache_read_bytes(reader, &maxn, sizeof(maxn), "hard-cache maxn") ||
        !hard_graph_cache_read_bytes(reader, &sig_words, sizeof(sig_words), "hard-cache sig words") ||
        !hard_graph_cache_read_bytes(reader, &result_size, sizeof(result_size),
                                     "hard-cache result size") ||
        !hard_graph_cache_read_bytes(reader, &coeff_size, sizeof(coeff_size),
                                     "hard-cache coeff size")) {
        return 0;
    }

    if (memcmp(magic, HARD_GRAPH_CACHE_FILE_MAGIC, sizeof(magic)) != 0 ||
        version != HARD_GRAPH_CACHE_FILE_VERSION ||
        count_k4 != RECT_COUNT_K4 ||
        max_rows != MAX_ROWS ||
        max_cols != MAX_COLS ||
        maxn != MAXN_NAUTY ||
        sig_words != GRAPH_SIG_WORDS ||
        result_size != sizeof(GraphResult) ||
        coeff_size != sizeof(PolyCoeff)) {
        fprintf(stderr, "Hard graph cache file %s is not compatible with this build\n",
                reader->path);
        return 0;
    }
    return 1;
}

static int hard_graph_cache_write_value(FILE* f, const GraphResult* value) {
#if RECT_COUNT_K4
    return checked_fwrite(value, sizeof(*value), 1, f, "hard-cache count4 value");
#else
    uint8_t x_pow = value->x_pow;
    uint8_t deg = value->deg;
    uint16_t reserved = 0;
    uint32_t coeff_count = (uint32_t)value->deg + 1U;
    if (value->deg > MAXN_NAUTY) {
        fprintf(stderr, "Refusing to write invalid hard-cache GraphPoly degree %u\n",
                (unsigned)value->deg);
        return 0;
    }
    return checked_fwrite(&x_pow, sizeof(x_pow), 1, f, "hard-cache x_pow") &&
           checked_fwrite(&deg, sizeof(deg), 1, f, "hard-cache degree") &&
           checked_fwrite(&reserved, sizeof(reserved), 1, f, "hard-cache reserved") &&
           checked_fwrite(&coeff_count, sizeof(coeff_count), 1, f, "hard-cache coeff count") &&
           checked_fwrite(value->coeffs, sizeof(value->coeffs[0]), coeff_count, f,
                          "hard-cache coeffs");
#endif
}

static int hard_graph_cache_read_value_mapped(HardGraphCacheReader* reader,
                                              GraphResult* value) {
#if RECT_COUNT_K4
    return hard_graph_cache_read_bytes(reader, value, sizeof(*value),
                                       "hard-cache count4 value");
#else
    uint8_t x_pow = 0;
    uint8_t deg = 0;
    uint16_t reserved = 0;
    uint32_t coeff_count = 0;
    memset(value, 0, sizeof(*value));
    if (!hard_graph_cache_read_bytes(reader, &x_pow, sizeof(x_pow), "hard-cache x_pow") ||
        !hard_graph_cache_read_bytes(reader, &deg, sizeof(deg), "hard-cache degree") ||
        !hard_graph_cache_read_bytes(reader, &reserved, sizeof(reserved),
                                     "hard-cache reserved") ||
        !hard_graph_cache_read_bytes(reader, &coeff_count, sizeof(coeff_count),
                                     "hard-cache coeff count")) {
        return 0;
    }
    if (reserved != 0 || deg > MAXN_NAUTY || coeff_count != (uint32_t)deg + 1U) {
        fprintf(stderr, "Invalid hard-cache GraphPoly metadata\n");
        return 0;
    }
    value->x_pow = x_pow;
    value->deg = deg;
    return hard_graph_cache_read_bytes(reader, value->coeffs,
                                       sizeof(value->coeffs[0]) * coeff_count,
                                       "hard-cache coeffs");
#endif
}

static inline uint64_t hard_graph_cache_index_mix(uint64_t x) {
    x ^= x >> 33;
    x *= UINT64_C(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x *= UINT64_C(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    return x;
}

static void hard_graph_cache_signature(const Graph* g, uint64_t sig[GRAPH_SIG_WORDS]) {
    memset(sig, 0, sizeof(uint64_t) * GRAPH_SIG_WORDS);
    uint64_t bit_pos = 0;
    for (int j = 1; j < g->n; j++) {
        uint64_t row = (uint64_t)g->adj[j];
        for (int i = 0; i < j; i++, bit_pos++) {
            if (row & (UINT64_C(1) << i)) {
                sig[bit_pos >> 6] |= UINT64_C(1) << (bit_pos & 63);
            }
        }
    }
}

static int hard_graph_cache_sig_words(int n) {
    int bits = n * (n - 1) / 2;
    return (bits + 63) / 64;
}

static int hard_graph_cache_entry_matches(const HardGraphCacheEntry* entry, uint64_t hash,
                                          int n, const uint64_t sig[GRAPH_SIG_WORDS]) {
    if (entry->hash != hash || entry->n != (uint8_t)n) return 0;
    int words = hard_graph_cache_sig_words(n);
    return memcmp(entry->sig, sig, (size_t)words * sizeof(uint64_t)) == 0;
}

static HardGraphCacheEntry* hard_graph_cache_alloc_entry_unlocked(int* capacity_full) {
    HardGraphCache* cache = &g_hard_graph_cache;
    *capacity_full = 0;
    if (cache->max_entries && cache->entry_count >= cache->max_entries) {
        *capacity_full = 1;
        return NULL;
    }
    if (!cache->active_slab || cache->active_slab->used == HARD_GRAPH_CACHE_SLAB_ENTRIES) {
        HardGraphCacheSlab* slab = (HardGraphCacheSlab*)malloc(sizeof(*slab));
        if (!slab) {
            return NULL;
        }
        slab->next = cache->slabs;
        slab->used = 0;
        cache->slabs = slab;
        cache->active_slab = slab;
    }
    HardGraphCacheEntry* entry = &cache->active_slab->entries[cache->active_slab->used++];
    cache->entry_count++;
    return entry;
}

static HardGraphCacheEntry* hard_graph_cache_alloc_entry(int* capacity_full) {
    HardGraphCache* cache = &g_hard_graph_cache;
    if (cache->use_locks) pthread_mutex_lock(&cache->alloc_lock);
    HardGraphCacheEntry* entry = hard_graph_cache_alloc_entry_unlocked(capacity_full);
    if (cache->use_locks) pthread_mutex_unlock(&cache->alloc_lock);
    return entry;
}

static int hard_graph_cache_insert_signature_fresh(uint64_t hash, int n,
                                                   const uint64_t sig[GRAPH_SIG_WORDS],
                                                   const GraphResult* value,
                                                   int* capacity_full) {
    HardGraphCache* cache = &g_hard_graph_cache;
    *capacity_full = 0;

    HardGraphCacheEntry* entry = hard_graph_cache_alloc_entry_unlocked(capacity_full);
    if (!entry) return -1;

    uint64_t bucket = hard_graph_cache_index_mix(hash) & cache->mask;
    entry->next = cache->buckets[bucket];
    entry->hash = hash;
    memcpy(entry->sig, sig, sizeof(uint64_t) * GRAPH_SIG_WORDS);
    entry->value = *value;
    entry->n = (uint8_t)n;
    cache->buckets[bucket] = entry;
    return 1;
}

static int hard_graph_cache_insert_signature(uint64_t hash, int n,
                                             const uint64_t sig[GRAPH_SIG_WORDS],
                                             const GraphResult* value,
                                             int count_duplicate_store,
                                             int* capacity_full) {
    HardGraphCache* cache = &g_hard_graph_cache;
    uint64_t bucket = hard_graph_cache_index_mix(hash) & cache->mask;
    pthread_mutex_t* lock = &cache->locks[bucket & (HARD_GRAPH_CACHE_LOCKS - 1)];
    *capacity_full = 0;

    if (cache->use_locks) pthread_mutex_lock(lock);
    for (HardGraphCacheEntry* entry = cache->buckets[bucket]; entry; entry = entry->next) {
        if (hard_graph_cache_entry_matches(entry, hash, n, sig)) {
            if (cache->use_locks) pthread_mutex_unlock(lock);
            if (count_duplicate_store) {
                atomic_fetch_add_explicit(&cache->duplicate_stores, 1, memory_order_relaxed);
            }
            return 0;
        }
    }

    HardGraphCacheEntry* entry = hard_graph_cache_alloc_entry(capacity_full);
    if (!entry) {
        if (cache->use_locks) pthread_mutex_unlock(lock);
        return -1;
    }
    entry->next = cache->buckets[bucket];
    entry->hash = hash;
    memcpy(entry->sig, sig, sizeof(uint64_t) * GRAPH_SIG_WORDS);
    entry->value = *value;
    entry->n = (uint8_t)n;
    cache->buckets[bucket] = entry;
    if (cache->use_locks) pthread_mutex_unlock(lock);
    return 1;
}

static int hard_graph_cache_write_record(FILE* f, uint64_t hash, int n,
                                         const uint64_t sig[GRAPH_SIG_WORDS],
                                         const GraphResult* value) {
    if (n < 0 || n > MAXN_NAUTY) {
        fprintf(stderr, "Refusing to write invalid hard-cache record n=%d\n", n);
        return 0;
    }
    uint8_t record_n = (uint8_t)n;
    uint8_t sig_words = (uint8_t)hard_graph_cache_sig_words(n);
    uint16_t reserved = 0;
    return checked_fwrite(&record_n, sizeof(record_n), 1, f, "hard-cache record n") &&
           checked_fwrite(&sig_words, sizeof(sig_words), 1, f, "hard-cache record sig words") &&
           checked_fwrite(&reserved, sizeof(reserved), 1, f, "hard-cache record reserved") &&
           checked_fwrite(&hash, sizeof(hash), 1, f, "hard-cache record hash") &&
           checked_fwrite(sig, sizeof(sig[0]), sig_words, f, "hard-cache record sig") &&
           hard_graph_cache_write_value(f, value);
}

static int hard_graph_cache_read_record_mapped(HardGraphCacheReader* reader,
                                               uint64_t* hash_out, int* n_out,
                                               uint64_t sig_out[GRAPH_SIG_WORDS],
                                               GraphResult* value_out, int* eof_out) {
    uint8_t record_n = 0;
    uint8_t sig_words = 0;
    uint16_t reserved = 0;
    *eof_out = 0;
    memset(sig_out, 0, sizeof(uint64_t) * GRAPH_SIG_WORDS);

    if (reader->cur == reader->end) {
        *eof_out = 1;
        return 1;
    }
    if (!hard_graph_cache_read_bytes(reader, &record_n, sizeof(record_n),
                                     "hard-cache record n") ||
        !hard_graph_cache_read_bytes(reader, &sig_words, sizeof(sig_words),
                                     "hard-cache record sig words") ||
        !hard_graph_cache_read_bytes(reader, &reserved, sizeof(reserved),
                                     "hard-cache record reserved") ||
        !hard_graph_cache_read_bytes(reader, hash_out, sizeof(*hash_out),
                                     "hard-cache record hash")) {
        return 0;
    }
    if (reserved != 0 || record_n > MAXN_NAUTY || sig_words > GRAPH_SIG_WORDS ||
        sig_words != hard_graph_cache_sig_words(record_n)) {
        fprintf(stderr, "Invalid hard-cache record metadata\n");
        return 0;
    }
    if (!hard_graph_cache_read_bytes(reader, sig_out, sizeof(sig_out[0]) * sig_words,
                                     "hard-cache record sig") ||
        !hard_graph_cache_read_value_mapped(reader, value_out)) {
        return 0;
    }
    *n_out = record_n;
    return 1;
}

static int hard_graph_cache_load_file(const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Failed to open RECT_HARD_CACHE_LOAD=%s: %s\n", path, strerror(errno));
        return 0;
    }

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "Failed to stat RECT_HARD_CACHE_LOAD=%s: %s\n", path, strerror(errno));
        close(fd);
        return 0;
    }
    size_t map_len = (size_t)st.st_size;
    void* map = mmap(NULL, map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) {
        fprintf(stderr, "Failed to mmap RECT_HARD_CACHE_LOAD=%s: %s\n", path, strerror(errno));
        return 0;
    }
#ifdef MADV_SEQUENTIAL
    madvise(map, map_len, MADV_SEQUENTIAL);
#endif

    HardGraphCacheReader reader = {
        .cur = (const uint8_t*)map,
        .end = (const uint8_t*)map + map_len,
        .path = path,
    };
    if (!hard_graph_cache_read_file_header_mapped(&reader)) {
        munmap(map, map_len);
        return 0;
    }

    uint64_t records = 0;
    uint64_t loaded = 0;
    uint64_t duplicates = 0;
    uint64_t capacity_skips = 0;
    int fast_load = (g_hard_graph_cache.entry_count == 0);
    for (;;) {
        uint64_t hash = 0;
        int n = 0;
        uint64_t sig[GRAPH_SIG_WORDS];
        GraphResult value;
        int eof = 0;
        if (!hard_graph_cache_read_record_mapped(&reader, &hash, &n, sig, &value, &eof)) {
            munmap(map, map_len);
            return 0;
        }
        if (eof) break;
        records++;
        int capacity_full = 0;
        int insert_status = fast_load
            ? hard_graph_cache_insert_signature_fresh(hash, n, sig, &value, &capacity_full)
            : hard_graph_cache_insert_signature(hash, n, sig, &value, 0, &capacity_full);
        if (insert_status > 0) {
            loaded++;
            continue;
        }
        if (capacity_full) {
            capacity_skips++;
            break;
        }
        if (insert_status < 0) {
            munmap(map, map_len);
            fprintf(stderr, "Failed to allocate hard graph cache entry while loading %s\n", path);
            return 0;
        }
        duplicates++;
    }
    munmap(map, map_len);
    atomic_fetch_add_explicit(&g_hard_graph_cache.loaded_entries, loaded, memory_order_relaxed);
    if (capacity_skips) {
        printf("Hard graph cache load: %s records=%llu, loaded=%llu, duplicates=%llu, capacity reached\n",
               path, (unsigned long long)records, (unsigned long long)loaded,
               (unsigned long long)duplicates);
    } else {
        printf("Hard graph cache load: %s records=%llu, loaded=%llu, duplicates=%llu\n",
               path, (unsigned long long)records, (unsigned long long)loaded,
               (unsigned long long)duplicates);
    }
    return 1;
}

static int hard_graph_cache_open_save_file(const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open RECT_HARD_CACHE_SAVE=%s: %s\n", path, strerror(errno));
        return 0;
    }
    if (!hard_graph_cache_write_file_header(f)) {
        fclose(f);
        return 0;
    }
    g_hard_graph_cache.save_file = f;
    printf("Hard graph cache save: %s\n", path);
    return 1;
}

int hard_graph_cache_init_from_env(void) {
    const char* bits_env = getenv("RECT_HARD_CACHE_BITS");
    const char* load_path = getenv("RECT_HARD_CACHE_LOAD");
    const char* save_path = getenv("RECT_HARD_CACHE_SAVE");
    int explicit_bits = bits_env && *bits_env;
    int has_load_path = load_path && *load_path;
    int has_save_path = save_path && *save_path;
    char* end = NULL;
    errno = 0;
    unsigned long bits = DEFAULT_HARD_CACHE_BITS;
    if (explicit_bits) {
        bits = strtoul(bits_env, &end, 10);
        if (errno || end == bits_env || *end != '\0') {
            fprintf(stderr,
                    "Invalid RECT_HARD_CACHE_BITS=%s; expected 0 or an integer in [10, 31]\n",
                    bits_env);
            return 0;
        }
    }
    if (explicit_bits && bits == 0) return 1;
    if (!explicit_bits && bits == 0 && (has_load_path || has_save_path)) bits = 22;
    if (!explicit_bits && bits != 0 && g_rows < 7 && !has_load_path && !has_save_path) return 1;
    if (bits == 0) return 1;
    if (bits < 10 || bits > 31) {
        fprintf(stderr,
                "Invalid hard cache bit count %lu; expected 0 or an integer in [10, 31]\n",
                bits);
        return 0;
    }

    const char* max_entries_env = getenv("RECT_HARD_CACHE_MAX_ENTRIES");
    unsigned long long max_entries = DEFAULT_HARD_CACHE_MAX_ENTRIES;
    if (max_entries_env && *max_entries_env) {
        char* max_end = NULL;
        errno = 0;
        max_entries = strtoull(max_entries_env, &max_end, 10);
        if (errno || max_end == max_entries_env || *max_end != '\0') {
            fprintf(stderr,
                    "Invalid RECT_HARD_CACHE_MAX_ENTRIES=%s; expected a non-negative integer\n",
                    max_entries_env);
            return 0;
        }
    }

    HardGraphCache* cache = &g_hard_graph_cache;
    cache->bucket_count = (size_t)1ULL << bits;
    cache->mask = (uint64_t)cache->bucket_count - 1ULL;
    cache->max_entries = (size_t)max_entries;
    cache->buckets = (HardGraphCacheEntry**)calloc(cache->bucket_count, sizeof(cache->buckets[0]));
    if (!cache->buckets) {
        fprintf(stderr, "Failed to allocate hard graph cache buckets (%zu entries)\n",
                cache->bucket_count);
        return 0;
    }
    for (int i = 0; i < HARD_GRAPH_CACHE_LOCKS; i++) {
        pthread_mutex_init(&cache->locks[i], NULL);
    }
    pthread_mutex_init(&cache->alloc_lock, NULL);
    pthread_mutex_init(&cache->save_lock, NULL);
    atomic_store_explicit(&cache->lookups, 0, memory_order_relaxed);
    atomic_store_explicit(&cache->hits, 0, memory_order_relaxed);
    atomic_store_explicit(&cache->stores, 0, memory_order_relaxed);
    atomic_store_explicit(&cache->skipped_stores, 0, memory_order_relaxed);
    atomic_store_explicit(&cache->duplicate_stores, 0, memory_order_relaxed);
    atomic_store_explicit(&cache->loaded_entries, 0, memory_order_relaxed);
    atomic_store_explicit(&cache->saved_entries, 0, memory_order_relaxed);
    cache->enabled = 1;
    cache->use_locks = omp_get_max_threads() > 1;

    printf("Hard graph cache: enabled, buckets=%zu, max_entries=%zu, locks=%s\n",
           cache->bucket_count, cache->max_entries, cache->use_locks ? "on" : "off");
    if (has_load_path && !hard_graph_cache_load_file(load_path)) {
        hard_graph_cache_close();
        return 0;
    }
    if (has_save_path && !hard_graph_cache_open_save_file(save_path)) {
        hard_graph_cache_close();
        return 0;
    }
    return 1;
}

int hard_graph_cache_lookup(uint64_t hash, const Graph* g, GraphResult* out) {
    HardGraphCache* cache = &g_hard_graph_cache;
    if (!cache->enabled) return 0;

    uint64_t sig[GRAPH_SIG_WORDS];
    hard_graph_cache_signature(g, sig);
    uint64_t bucket = hard_graph_cache_index_mix(hash) & cache->mask;
    pthread_mutex_t* lock = &cache->locks[bucket & (HARD_GRAPH_CACHE_LOCKS - 1)];

    atomic_fetch_add_explicit(&cache->lookups, 1, memory_order_relaxed);
    if (cache->use_locks) pthread_mutex_lock(lock);
    for (HardGraphCacheEntry* entry = cache->buckets[bucket]; entry; entry = entry->next) {
        if (hard_graph_cache_entry_matches(entry, hash, g->n, sig)) {
            *out = entry->value;
            if (cache->use_locks) pthread_mutex_unlock(lock);
            atomic_fetch_add_explicit(&cache->hits, 1, memory_order_relaxed);
            return 1;
        }
    }
    if (cache->use_locks) pthread_mutex_unlock(lock);
    return 0;
}

void hard_graph_cache_store(uint64_t hash, const Graph* g, const GraphResult* value) {
    HardGraphCache* cache = &g_hard_graph_cache;
    if (!cache->enabled) return;

    uint64_t sig[GRAPH_SIG_WORDS];
    hard_graph_cache_signature(g, sig);
    int capacity_full = 0;
    int insert_status = hard_graph_cache_insert_signature(hash, g->n, sig, value, 1, &capacity_full);
    if (insert_status <= 0) {
        if (capacity_full) {
            atomic_fetch_add_explicit(&cache->skipped_stores, 1, memory_order_relaxed);
        } else if (insert_status < 0) {
            fprintf(stderr, "Failed to allocate hard graph cache entry\n");
        }
        return;
    }
    atomic_fetch_add_explicit(&cache->stores, 1, memory_order_relaxed);

    if (cache->save_file) {
        pthread_mutex_lock(&cache->save_lock);
        if (hard_graph_cache_write_record(cache->save_file, hash, g->n, sig, value)) {
            atomic_fetch_add_explicit(&cache->saved_entries, 1, memory_order_relaxed);
        }
        pthread_mutex_unlock(&cache->save_lock);
    }
}

void hard_graph_cache_close(void) {
    HardGraphCache* cache = &g_hard_graph_cache;
    if (!cache->enabled) return;
    if (cache->save_file) {
        fclose(cache->save_file);
        cache->save_file = NULL;
    }

    uint64_t lookups = atomic_load_explicit(&cache->lookups, memory_order_relaxed);
    uint64_t hits = atomic_load_explicit(&cache->hits, memory_order_relaxed);
    uint64_t stores = atomic_load_explicit(&cache->stores, memory_order_relaxed);
    uint64_t skipped_stores =
        atomic_load_explicit(&cache->skipped_stores, memory_order_relaxed);
    uint64_t duplicate_stores =
        atomic_load_explicit(&cache->duplicate_stores, memory_order_relaxed);
    uint64_t loaded_entries =
        atomic_load_explicit(&cache->loaded_entries, memory_order_relaxed);
    uint64_t saved_entries =
        atomic_load_explicit(&cache->saved_entries, memory_order_relaxed);
    double hit_rate = lookups ? (100.0 * (double)hits / (double)lookups) : 0.0;
    printf("Hard graph cache hits: %llu/%llu (%.1f%%), stores=%llu, skipped stores=%llu, "
           "duplicate stores=%llu, loaded=%llu, saved=%llu\n",
           (unsigned long long)hits, (unsigned long long)lookups, hit_rate,
           (unsigned long long)stores, (unsigned long long)skipped_stores,
           (unsigned long long)duplicate_stores, (unsigned long long)loaded_entries,
           (unsigned long long)saved_entries);

    HardGraphCacheSlab* slab = cache->slabs;
    while (slab) {
        HardGraphCacheSlab* next = slab->next;
        free(slab);
        slab = next;
    }
    for (int i = 0; i < HARD_GRAPH_CACHE_LOCKS; i++) {
        pthread_mutex_destroy(&cache->locks[i]);
    }
    pthread_mutex_destroy(&cache->alloc_lock);
    pthread_mutex_destroy(&cache->save_lock);
    free(cache->buckets);
    memset(cache, 0, sizeof(*cache));
}

void task_timing_insert_topk(TaskTimingStats* stats, long long task_index, double elapsed) {
    for (int i = 0; i < TASK_PROFILE_TOPK; i++) {
        if (elapsed > stats->top_times[i]) {
            for (int j = TASK_PROFILE_TOPK - 1; j > i; j--) {
                stats->top_times[j] = stats->top_times[j - 1];
                stats->top_indices[j] = stats->top_indices[j - 1];
            }
            stats->top_times[i] = elapsed;
            stats->top_indices[i] = task_index;
            break;
        }
    }
}

static void task_timing_record(TaskTimingStats* stats, long long task_index, double elapsed) {
    stats->task_count++;
    stats->task_time_sum += elapsed;
    if (elapsed > stats->task_time_max) {
        stats->task_time_max = elapsed;
        stats->task_max_index = task_index;
    }
    task_timing_insert_topk(stats, task_index, elapsed);
}

static void queue_subtask_insert_topk(QueueSubtaskTimingStats* stats, const LocalTask* task,
                                      double elapsed, long long solve_graph_calls,
                                      long long nauty_calls, long long hard_graph_nodes,
                                      int max_hard_graph_n, int max_hard_graph_degree) {
    for (int i = 0; i < TASK_PROFILE_TOPK; i++) {
        if (elapsed > stats->top[i].elapsed) {
            for (int j = TASK_PROFILE_TOPK - 1; j > i; j--) {
                stats->top[j] = stats->top[j - 1];
            }
            stats->top[i].depth = task->depth;
            for (int d = 0; d < task->depth; d++) stats->top[i].prefix[d] = task->prefix[d];
            stats->top[i].elapsed = elapsed;
            stats->top[i].solve_graph_calls = solve_graph_calls;
            stats->top[i].nauty_calls = nauty_calls;
            stats->top[i].hard_graph_nodes = hard_graph_nodes;
            stats->top[i].max_hard_graph_n = (uint8_t)max_hard_graph_n;
            stats->top[i].max_hard_graph_degree = (uint8_t)max_hard_graph_degree;
            break;
        }
    }
}

void queue_subtask_record(QueueSubtaskTimingStats* stats, const LocalTask* task,
                          double elapsed, long long solve_graph_calls,
                          long long nauty_calls, long long hard_graph_nodes,
                          int max_hard_graph_n, int max_hard_graph_degree) {
    stats->task_count++;
    stats->task_time_sum += elapsed;
    if (elapsed > stats->task_time_max) stats->task_time_max = elapsed;
    stats->solve_graph_call_sum += solve_graph_calls;
    stats->nauty_call_sum += nauty_calls;
    stats->hard_graph_node_sum += hard_graph_nodes;
    if (max_hard_graph_n > stats->max_hard_graph_n) stats->max_hard_graph_n = max_hard_graph_n;
    if (max_hard_graph_degree > stats->max_hard_graph_degree) stats->max_hard_graph_degree = max_hard_graph_degree;
    queue_subtask_insert_topk(stats, task, elapsed, solve_graph_calls, nauty_calls,
                              hard_graph_nodes, max_hard_graph_n, max_hard_graph_degree);
}

void queue_subtask_merge(QueueSubtaskTimingStats* dst, const QueueSubtaskTimingStats* src) {
    dst->task_count += src->task_count;
    dst->task_time_sum += src->task_time_sum;
    if (src->task_time_max > dst->task_time_max) dst->task_time_max = src->task_time_max;
    dst->solve_graph_call_sum += src->solve_graph_call_sum;
    dst->nauty_call_sum += src->nauty_call_sum;
    for (int i = 0; i < TASK_PROFILE_TOPK; i++) {
        if (src->top[i].elapsed <= 0.0) break;
        LocalTask task = {0};
        task.depth = src->top[i].depth;
        for (int d = 0; d < task.depth; d++) task.prefix[d] = src->top[i].prefix[d];
        queue_subtask_insert_topk(dst, &task, src->top[i].elapsed,
                                  src->top[i].solve_graph_calls, src->top[i].nauty_calls,
                                  src->top[i].hard_graph_nodes,
                                  src->top[i].max_hard_graph_n,
                                  src->top[i].max_hard_graph_degree);
    }
    dst->hard_graph_node_sum += src->hard_graph_node_sum;
    if (src->max_hard_graph_n > dst->max_hard_graph_n) dst->max_hard_graph_n = src->max_hard_graph_n;
    if (src->max_hard_graph_degree > dst->max_hard_graph_degree) dst->max_hard_graph_degree = src->max_hard_graph_degree;
}

static void print_queue_subtask_prefix(const QueueSubtaskTopEntry* e) {
    printf("(");
    for (int p = 0; p < e->depth; p++) {
        if (p > 0) printf(",");
        printf("%u", (unsigned)e->prefix[p]);
    }
    printf(")");
}

static void record_task_time_value(long long task_index, double elapsed) {
    if (!g_task_times_values || g_task_times_count <= 0) return;
    long long delta = task_index - g_task_times_first_task;
    if (delta < 0) return;
    long long slot = delta;
    if (slot < 0 || slot >= g_task_times_count) return;
    g_task_times_values[slot] = elapsed;
}

int decode_task_prefix(long long task_index, int* i, int* j, int* k, int* l) {
    *i = -1;
    *j = -1;
    *k = -1;
    *l = -1;
    if (g_effective_prefix_depth == 2) {
        if (g_live_prefix2_i && task_index >= 0 && task_index < g_live_prefix2_count) {
            *i = (int)g_live_prefix2_i[task_index];
            *j = (int)g_live_prefix2_j[task_index];
            return 1;
        }
        long long rank = task_index;
        for (int a = 0; a < num_partitions; a++) {
            long long count = num_partitions - a;
            if (rank < count) {
                *i = a;
                *j = a + (int)rank;
                return 1;
            }
            rank -= count;
        }
        return 0;
    }
    if (g_effective_prefix_depth == 3) {
        long long rank = task_index;
        for (int a = 0; a < num_partitions; a++) {
            long long count_a = repeated_combo_count(num_partitions - a, 2);
            if (rank >= count_a) {
                rank -= count_a;
                continue;
            }
            for (int b = a; b < num_partitions; b++) {
                long long count_b = num_partitions - b;
                if (rank < count_b) {
                    *i = a;
                    *j = b;
                    *k = b + (int)rank;
                    return 1;
                }
                rank -= count_b;
            }
            return 0;
        }
        return 0;
    }
    if (g_effective_prefix_depth == 4) {
        long long rank = task_index;
        for (int a = 0; a < num_partitions; a++) {
            long long count_a = repeated_combo_count(num_partitions - a, 3);
            if (rank >= count_a) {
                rank -= count_a;
                continue;
            }
            for (int b = a; b < num_partitions; b++) {
                long long count_b = repeated_combo_count(num_partitions - b, 2);
                if (rank >= count_b) {
                    rank -= count_b;
                    continue;
                }
                for (int c = b; c < num_partitions; c++) {
                    long long count_c = num_partitions - c;
                    if (rank < count_c) {
                        *i = a;
                        *j = b;
                        *k = c;
                        *l = c + (int)rank;
                        return 1;
                    }
                    rank -= count_c;
                }
                return 0;
            }
        }
        return 0;
    }
    return 0;
}

void write_task_times_file(const char* path) {
    if (!path || !g_task_times_values || g_task_times_count <= 0) return;
    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Failed to open task timing output %s: %s\n", path, strerror(errno));
        exit(1);
    }
    fprintf(f, "task_index,elapsed_seconds,i,j,k,l\n");
    for (long long t = 0; t < g_task_times_count; t++) {
        double elapsed = g_task_times_values[t];
        if (elapsed < 0.0) continue;
        long long task_index = g_task_times_first_task + t;
        int i, j, k, l;
        int have_prefix = decode_task_prefix(task_index, &i, &j, &k, &l);
        fprintf(f, "%lld,%.9f,", task_index, elapsed);
        if (have_prefix && i >= 0) fprintf(f, "%d", i);
        fprintf(f, ",");
        if (have_prefix && j >= 0) fprintf(f, "%d", j);
        fprintf(f, ",");
        if (have_prefix && k >= 0) fprintf(f, "%d", k);
        fprintf(f, ",");
        if (have_prefix && l >= 0) fprintf(f, "%d", l);
        fprintf(f, "\n");
    }
    fclose(f);
}

#define PROGRESS_FLUSH_BATCH 64

static inline void maybe_report_progress(long long done, long long total_tasks, long long report_step,
                                         double start_time) {
    #pragma omp critical(progress_report)
    {
        progress_reporter.last_reported = progress_last_reported;
        progress_reporter_maybe_report(&progress_reporter, done, total_tasks, report_step,
                                       start_time, omp_get_wtime());
        progress_last_reported = progress_reporter.last_reported;
    }
}

void flush_completed_tasks(long long total_tasks, long long report_step,
                           double start_time, long long* pending_completed) {
    if (*pending_completed == 0) return;
    long long done = 0;
    #pragma omp atomic capture
    {
        completed_tasks += *pending_completed;
        done = completed_tasks;
    }
    *pending_completed = 0;
    maybe_report_progress(done, total_tasks, report_step, start_time);
}

static inline void complete_task_and_report(long long total_tasks, long long report_step,
                                            double start_time, long long* pending_completed) {
    (*pending_completed)++;
    if (*pending_completed >= PROGRESS_FLUSH_BATCH) {
        flush_completed_tasks(total_tasks, report_step, start_time, pending_completed);
    }
}

void complete_task_report_and_time(long long total_tasks, long long report_step,
                                   double start_time, long long* pending_completed,
                                   TaskTimingStats* task_timing, long long task_index,
                                   double task_t0) {
    complete_task_and_report(total_tasks, report_step, start_time, pending_completed);
    if (PROFILE_BUILD && task_timing) {
        double elapsed = omp_get_wtime() - task_t0;
        task_timing_record(task_timing, task_index, elapsed);
        record_task_time_value(task_index, elapsed);
    }
}

static void local_queue_note_outstanding(LocalTaskQueue* queue, int outstanding) {
    long current = atomic_load_explicit(&queue->max_outstanding_tasks, memory_order_relaxed);
    while ((long)outstanding > current &&
           !atomic_compare_exchange_weak_explicit(&queue->max_outstanding_tasks, &current, outstanding,
                                                  memory_order_relaxed, memory_order_relaxed)) {
    }
}

static inline void local_queue_note_idle_locked(LocalTaskQueue* queue, int new_idle_threads) {
    double now = omp_get_wtime();
    if (queue->occupancy_last_at > 0.0) {
        queue->idle_thread_seconds +=
            (now - queue->occupancy_last_at) * (double)queue->occupancy_idle_threads;
    }
    queue->occupancy_last_at = now;
    queue->occupancy_idle_threads = new_idle_threads;
}

void local_queue_init(LocalTaskQueue* queue, int capacity,
                      long long root_count, int total_threads) {
    memset(queue, 0, sizeof(*queue));
    pthread_mutex_init(&queue->mutex, NULL);
    pthread_cond_init(&queue->cond, NULL);
    queue->tasks = checked_calloc((size_t)capacity, sizeof(*queue->tasks), "local_task_queue");
    queue->roots = checked_calloc((size_t)root_count, sizeof(*queue->roots), "local_root_state");
    queue->capacity = capacity;
    queue->root_count = root_count;
    queue->total_threads = total_threads;
    atomic_init(&queue->outstanding_tasks, 0);
    atomic_init(&queue->queue_count, 0);
    atomic_init(&queue->idle_threads, 0);
    atomic_init(&queue->donated_tasks, 0);
    atomic_init(&queue->work_budget_continuations, 0);
    atomic_init(&queue->max_outstanding_tasks, 0);
    queue->profile_started_at = omp_get_wtime();
    queue->occupancy_last_at = queue->profile_started_at;
    queue->occupancy_idle_threads = 0;
    queue->idle_thread_seconds = 0.0;
    queue->next_profile_report_at =
        (g_queue_profile_report_step > 0.0) ? (queue->profile_started_at + g_queue_profile_report_step) : 0.0;
    for (long long i = 0; i < root_count; i++) {
        queue->roots[i].launched_at = -1.0;
    }
}

void local_queue_free(LocalTaskQueue* queue) {
    free(queue->tasks);
    free(queue->roots);
    pthread_cond_destroy(&queue->cond);
    pthread_mutex_destroy(&queue->mutex);
    memset(queue, 0, sizeof(*queue));
}

void local_task_from_stack(LocalTask* task, long long root_id, int depth, const int* stack) {
    task->depth = (uint8_t)depth;
    task->root_id = root_id;
    for (int i = 0; i < depth; i++) task->prefix[i] = (PrefixId)stack[i];
}

int local_queue_try_push(LocalTaskQueue* queue, const LocalTask* task) {
    int pushed = 0;
    int outstanding = 0;
    pthread_mutex_lock(&queue->mutex);
    if (!queue->stop && queue->count < queue->capacity) {
        __atomic_add_fetch(&queue->roots[task->root_id].pending, 1, __ATOMIC_RELAXED);
        queue->tasks[queue->tail] = *task;
        queue->tail = (queue->tail + 1) % queue->capacity;
        queue->count++;
        atomic_store_explicit(&queue->queue_count, queue->count, memory_order_relaxed);
        outstanding = atomic_fetch_add_explicit(&queue->outstanding_tasks, 1, memory_order_relaxed) + 1;
        local_queue_note_outstanding(queue, outstanding);
        pushed = 1;
        pthread_cond_signal(&queue->cond);
    }
    pthread_mutex_unlock(&queue->mutex);
    return pushed;
}

void local_queue_seed_push(LocalTaskQueue* queue, const LocalTask* task) {
    if (!local_queue_try_push(queue, task)) {
        fprintf(stderr, "Failed to seed local task queue\n");
        exit(1);
    }
}

int local_queue_pop(LocalTaskQueue* queue, LocalTask* task) {
    pthread_mutex_lock(&queue->mutex);
    int marked_idle = 0;
    for (;;) {
        if (queue->count > 0) {
            if (marked_idle) {
                int idle = atomic_fetch_sub_explicit(&queue->idle_threads, 1, memory_order_relaxed) - 1;
                local_queue_note_idle_locked(queue, idle);
            }
            *task = queue->tasks[queue->head];
            queue->head = (queue->head + 1) % queue->capacity;
            queue->count--;
            atomic_store_explicit(&queue->queue_count, queue->count, memory_order_relaxed);
            queue->inflight++;
            if (queue->roots[task->root_id].launched_at < 0.0) {
                queue->roots[task->root_id].launched_at = omp_get_wtime();
            }
            pthread_mutex_unlock(&queue->mutex);
            return 1;
        }
        if (queue->inflight == 0) {
            if (marked_idle) {
                int idle = atomic_fetch_sub_explicit(&queue->idle_threads, 1, memory_order_relaxed) - 1;
                local_queue_note_idle_locked(queue, idle);
            }
            queue->stop = 1;
            pthread_cond_broadcast(&queue->cond);
            pthread_mutex_unlock(&queue->mutex);
            return 0;
        }
        if (!marked_idle) {
            int idle = atomic_fetch_add_explicit(&queue->idle_threads, 1, memory_order_relaxed) + 1;
            local_queue_note_idle_locked(queue, idle);
            marked_idle = 1;
        }
        pthread_cond_wait(&queue->cond, &queue->mutex);
    }
}

void local_queue_finish_item(LocalTaskQueue* queue, long long root_id,
                             long long total_tasks, long long report_step,
                             double start_time, long long* pending_completed,
                             TaskTimingStats* task_timing) {
    long long remaining =
        __atomic_sub_fetch(&queue->roots[root_id].pending, 1, __ATOMIC_ACQ_REL);
    atomic_fetch_sub_explicit(&queue->outstanding_tasks, 1, memory_order_relaxed);

    if (remaining == 0) {
        complete_task_and_report(total_tasks, report_step, start_time, pending_completed);
        if (PROFILE_BUILD && task_timing && queue->roots[root_id].launched_at >= 0.0) {
            double elapsed = omp_get_wtime() - queue->roots[root_id].launched_at;
            task_timing_record(task_timing, queue->roots[root_id].task_index, elapsed);
            record_task_time_value(queue->roots[root_id].task_index, elapsed);
        }
    }

    pthread_mutex_lock(&queue->mutex);
    queue->inflight--;
    if (queue->count == 0 && queue->inflight == 0) {
        pthread_cond_broadcast(&queue->cond);
    } else if (queue->count > 0) {
        pthread_cond_signal(&queue->cond);
    }
    pthread_mutex_unlock(&queue->mutex);
}

void local_queue_record_profile(LocalTaskQueue* queue, const LocalTask* task,
                                double elapsed, long long solve_graph_calls,
                                long long nauty_calls, long long hard_graph_nodes,
                                int max_hard_graph_n, int max_hard_graph_degree) {
    if (g_queue_profile_report_step <= 0.0 || task->depth > MAX_COLS) return;

    pthread_mutex_lock(&queue->mutex);
    queue_subtask_record(&queue->profile_stats[task->depth], task, elapsed, solve_graph_calls, nauty_calls,
                         hard_graph_nodes, max_hard_graph_n, max_hard_graph_degree);
    double now = omp_get_wtime();
    if (queue->next_profile_report_at > 0.0 && now >= queue->next_profile_report_at) {
        double idle_thread_seconds =
            queue->idle_thread_seconds + (now - queue->occupancy_last_at) * (double)queue->occupancy_idle_threads;
        double occupancy_elapsed = now - queue->profile_started_at;
        double avg_active = (occupancy_elapsed > 0.0)
                                ? ((double)queue->total_threads - idle_thread_seconds / occupancy_elapsed)
                                : (double)queue->total_threads;
        int current_active = queue->total_threads - queue->occupancy_idle_threads;
        double util_pct = (queue->total_threads > 0)
                              ? (100.0 * avg_active / (double)queue->total_threads)
                              : 0.0;
        printf("Queue profile after %.2fs (active now %d/%d, avg %.2f/%d = %.1f%%):\n",
               occupancy_elapsed, current_active, queue->total_threads,
               avg_active, queue->total_threads, util_pct);
        for (int d = 0; d <= g_cols && d <= MAX_COLS; d++) {
            QueueSubtaskTimingStats* qs = &queue->profile_stats[d];
            if (qs->task_count == 0) continue;
            printf("  depth %d: %lld subtasks, avg %.6fs, max %.6fs, avg solve_graph %.1f, avg nauty %.1f, avg hard nodes %.1f, max hard n %d, max hard deg %d",
                   d, qs->task_count, qs->task_time_sum / (double)qs->task_count, qs->task_time_max,
                   (double)qs->solve_graph_call_sum / (double)qs->task_count,
                   (double)qs->nauty_call_sum / (double)qs->task_count,
                   (double)qs->hard_graph_node_sum / (double)qs->task_count,
                   qs->max_hard_graph_n, qs->max_hard_graph_degree);
            if (qs->top[0].elapsed > 0.0) {
                printf(", top ");
                print_queue_subtask_prefix(&qs->top[0]);
                printf(" %.6fs", qs->top[0].elapsed);
            }
            printf("\n");
        }
        fflush(stdout);
        queue->next_profile_report_at = now + g_queue_profile_report_step;
    }
    pthread_mutex_unlock(&queue->mutex);
}

void local_queue_print_occupancy_summary(LocalTaskQueue* queue) {
    pthread_mutex_lock(&queue->mutex);
    double now = omp_get_wtime();
    double idle_thread_seconds =
        queue->idle_thread_seconds + (now - queue->occupancy_last_at) * (double)queue->occupancy_idle_threads;
    double occupancy_elapsed = now - queue->profile_started_at;
    double avg_active = (occupancy_elapsed > 0.0)
                            ? ((double)queue->total_threads - idle_thread_seconds / occupancy_elapsed)
                            : (double)queue->total_threads;
    double util_pct = (queue->total_threads > 0)
                          ? (100.0 * avg_active / (double)queue->total_threads)
                          : 0.0;
    pthread_mutex_unlock(&queue->mutex);

    printf("Runtime queue occupancy: avg active %.2f/%d (%.1f%%)\n",
           avg_active, queue->total_threads, util_pct);
    if (g_adaptive_work_budget > 0) {
        printf("Runtime queue work-budget continuations: %lld\n",
               (long long)atomic_load_explicit(&queue->work_budget_continuations, memory_order_relaxed));
    }
}

void runtime_task_system_init(RuntimeTaskSystem* system, int capacity,
                              long long root_count, int total_threads) {
    local_queue_init(&system->shared_queue, capacity, root_count, total_threads);
}

void runtime_task_system_free(RuntimeTaskSystem* system) {
    local_queue_free(&system->shared_queue);
}

void runtime_task_system_seed_task(RuntimeTaskSystem* system, const LocalTask* task) {
    local_queue_seed_push(&system->shared_queue, task);
}

int runtime_task_system_pop_task(RuntimeTaskSystem* system, LocalTask* task) {
    return local_queue_pop(&system->shared_queue, task);
}

int runtime_task_system_push_local(RuntimeTaskSystem* system, const LocalTask* task) {
    return local_queue_try_push(&system->shared_queue, task);
}

int runtime_task_system_push_balance(RuntimeTaskSystem* system, const LocalTask* task) {
    return local_queue_try_push(&system->shared_queue, task);
}

int runtime_task_system_has_idle_workers(const RuntimeTaskSystem* system) {
    return atomic_load_explicit(&system->shared_queue.idle_threads, memory_order_relaxed) > 0;
}

int runtime_task_system_needs_balance(const RuntimeTaskSystem* system) {
    int idle_workers = atomic_load_explicit(&system->shared_queue.idle_threads, memory_order_relaxed);
    if (idle_workers <= 0) return 0;
    int min_global = system->shared_queue.total_threads;
    if (min_global < 4) min_global = 4;
    int count = atomic_load_explicit(&system->shared_queue.queue_count, memory_order_relaxed);
    return count < min_global;
}

void runtime_task_system_note_balance_push(RuntimeTaskSystem* system) {
    atomic_fetch_add_explicit(&system->shared_queue.donated_tasks, 1, memory_order_relaxed);
}

void runtime_task_system_note_work_budget_split(RuntimeTaskSystem* system) {
    atomic_fetch_add_explicit(&system->shared_queue.work_budget_continuations, 1,
                              memory_order_relaxed);
}

void runtime_task_system_finish_task(RuntimeTaskSystem* system, long long root_id,
                                     long long total_tasks, long long report_step,
                                     double start_time, long long* pending_completed,
                                     TaskTimingStats* task_timing) {
    local_queue_finish_item(&system->shared_queue, root_id, total_tasks, report_step,
                            start_time, pending_completed, task_timing);
}

void runtime_task_system_record_profile(RuntimeTaskSystem* system, const LocalTask* task,
                                        double elapsed, long long solve_graph_calls,
                                        long long nauty_calls, long long hard_graph_nodes,
                                        int max_hard_graph_n, int max_hard_graph_degree) {
    local_queue_record_profile(&system->shared_queue, task, elapsed, solve_graph_calls,
                               nauty_calls, hard_graph_nodes,
                               max_hard_graph_n, max_hard_graph_degree);
}

void runtime_task_system_print_summary(RuntimeTaskSystem* system) {
    local_queue_print_occupancy_summary(&system->shared_queue);
}

void* checked_aligned_alloc(size_t alignment, size_t size, const char* label) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        fprintf(stderr, "Failed to allocate %s (%zu bytes)\n", label, size);
        exit(1);
    }
    return ptr;
}

void* checked_calloc(size_t count, size_t size, const char* label) {
    if (count == 0 || size == 0) {
        count = 1;
        size = 1;
    }
    void* ptr = calloc(count, size);
    if (!ptr) {
        fprintf(stderr, "Failed to allocate %s (%zu bytes)\n", label, count * size);
        exit(1);
    }
    return ptr;
}

void prefix_task_buffer_init(PrefixTaskBuffer* buf, long long initial_capacity) {
    memset(buf, 0, sizeof(*buf));
    if (initial_capacity < 16) initial_capacity = 16;
    buf->capacity = initial_capacity;
    buf->i = checked_calloc((size_t)buf->capacity, sizeof(*buf->i), "prefix_buffer_i");
    buf->j = checked_calloc((size_t)buf->capacity, sizeof(*buf->j), "prefix_buffer_j");
}

static void prefix_task_buffer_reserve(PrefixTaskBuffer* buf, long long needed) {
    if (needed <= buf->capacity) return;
    long long new_capacity = buf->capacity;
    while (new_capacity < needed) {
        if (new_capacity > LLONG_MAX / 2) {
            fprintf(stderr, "Prefix task buffer capacity overflow\n");
            exit(1);
        }
        new_capacity *= 2;
    }
    PrefixId* new_i = realloc(buf->i, (size_t)new_capacity * sizeof(*buf->i));
    PrefixId* new_j = realloc(buf->j, (size_t)new_capacity * sizeof(*buf->j));
    if (!new_i || !new_j) {
        fprintf(stderr, "Failed to grow adaptive prefix buffers to %lld entries\n", new_capacity);
        exit(1);
    }
    buf->i = new_i;
    buf->j = new_j;
    buf->capacity = new_capacity;
}

void prefix_task_buffer_push2(PrefixTaskBuffer* buf, int i, int j) {
    prefix_task_buffer_reserve(buf, buf->count + 1);
    buf->i[buf->count] = (PrefixId)i;
    buf->j[buf->count] = (PrefixId)j;
    buf->count++;
}

long long repeated_combo_count(int values, int slots) {
    switch (slots) {
        case 0:
            return 1;
        case 1:
            return values;
        case 2:
            return (long long)values * (values + 1) / 2;
        case 3:
            return (long long)values * (values + 1) * (values + 2) / 6;
        default:
            fprintf(stderr, "Unsupported repeated combination slot count: %d\n", slots);
            exit(1);
    }
}

static void unrank_prefix2(long long rank, int* i, int* j) {
    for (int a = 0; a < num_partitions; a++) {
        long long count = num_partitions - a;
        if (rank < count) {
            *i = a;
            *j = a + (int)rank;
            return;
        }
        rank -= count;
    }
    fprintf(stderr, "Depth-2 prefix rank out of range\n");
    exit(1);
}

void get_prefix2_task(long long task_index, int* i, int* j) {
    if (g_live_prefix2_i && task_index >= 0 && task_index < g_live_prefix2_count) {
        *i = (int)g_live_prefix2_i[task_index];
        *j = (int)g_live_prefix2_j[task_index];
        return;
    }
    unrank_prefix2(task_index, i, j);
}

void build_fixed_prefix2_batches(const PrefixId* live_i, const PrefixId* live_j,
                                 long long task_start,
                                 long long total_tasks, Prefix2Batch** batches_out,
                                 long long* batch_count_out, PrefixId** js_out,
                                 long long** ps_out) {
    int* counts = checked_calloc((size_t)num_partitions, sizeof(*counts), "prefix2_batch_counts");
    int* offsets = checked_calloc((size_t)num_partitions, sizeof(*offsets), "prefix2_batch_offsets");
    int* cursor = checked_calloc((size_t)num_partitions, sizeof(*cursor), "prefix2_batch_cursor");

    for (long long t = 0; t < total_tasks; t++) {
        long long p = task_start + t;
        int i = (int)live_i[p];
        counts[i]++;
    }

    long long batch_count = 0;
    int running = 0;
    for (int i = 0; i < num_partitions; i++) {
        offsets[i] = running;
        cursor[i] = running;
        running += counts[i];
        if (counts[i] > 0) {
            batch_count += (counts[i] + FIXED_PREFIX2_BATCH_SIZE - 1) / FIXED_PREFIX2_BATCH_SIZE;
        }
    }

    PrefixId* js = checked_calloc((size_t)total_tasks, sizeof(*js), "prefix2_batch_js");
    long long* ps = checked_calloc((size_t)total_tasks, sizeof(*ps), "prefix2_batch_ps");
    Prefix2Batch* batches = checked_calloc((size_t)batch_count, sizeof(*batches), "prefix2_batches");

    for (long long t = 0; t < total_tasks; t++) {
        long long p = task_start + t;
        int i = (int)live_i[p];
        int j = (int)live_j[p];
        int pos = cursor[i]++;
        js[pos] = (PrefixId)j;
        ps[pos] = p;
    }

    long long batch_index = 0;
    for (int i = 0; i < num_partitions; i++) {
        for (int pos = offsets[i]; pos < offsets[i] + counts[i]; pos += FIXED_PREFIX2_BATCH_SIZE) {
            int remaining = offsets[i] + counts[i] - pos;
            int batch_size = remaining < FIXED_PREFIX2_BATCH_SIZE ? remaining : FIXED_PREFIX2_BATCH_SIZE;
            batches[batch_index].i = (PrefixId)i;
            batches[batch_index].start = (uint32_t)pos;
            batches[batch_index].count = (uint16_t)batch_size;
            batch_index++;
        }
    }

    free(counts);
    free(offsets);
    free(cursor);

    *batches_out = batches;
    *batch_count_out = batch_count;
    *js_out = js;
    *ps_out = ps;
}

void unrank_prefix3(long long rank, int* i, int* j, int* k) {
    for (int a = 0; a < num_partitions; a++) {
        long long count_a = repeated_combo_count(num_partitions - a, 2);
        if (rank < count_a) {
            *i = a;
            for (int b = a; b < num_partitions; b++) {
                long long count_b = num_partitions - b;
                if (rank < count_b) {
                    *j = b;
                    *k = b + (int)rank;
                    return;
                }
                rank -= count_b;
            }
        }
        rank -= count_a;
    }
    fprintf(stderr, "Depth-3 prefix rank out of range\n");
    exit(1);
}

void unrank_prefix4(long long rank, int* i, int* j, int* k, int* l) {
    for (int a = 0; a < num_partitions; a++) {
        long long count_a = repeated_combo_count(num_partitions - a, 3);
        if (rank < count_a) {
            *i = a;
            for (int b = a; b < num_partitions; b++) {
                long long count_b = repeated_combo_count(num_partitions - b, 2);
                if (rank < count_b) {
                    *j = b;
                    for (int c = b; c < num_partitions; c++) {
                        long long count_c = num_partitions - c;
                        if (rank < count_c) {
                            *k = c;
                            *l = c + (int)rank;
                            return;
                        }
                        rank -= count_c;
                    }
                }
                rank -= count_b;
            }
        }
        rank -= count_a;
    }
    fprintf(stderr, "Depth-4 prefix rank out of range\n");
    exit(1);
}
