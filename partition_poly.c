#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <stdatomic.h>
#include <omp.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include "progress_util.h"

#ifndef RECT_PROFILE
#define RECT_PROFILE 0
#endif

#define PROFILE_BUILD RECT_PROFILE

// --- CONFIGURATION ---
#ifndef DEFAULT_ROWS
#define DEFAULT_ROWS 6
#endif
#ifndef DEFAULT_COLS
#define DEFAULT_COLS 6
#endif
#ifndef MAX_ROWS
#define MAX_ROWS 7
#endif
#ifndef MAX_COLS
#define MAX_COLS 16
#endif

// Define MAXN before including nauty
#define MAX_COMPLEX_PER_COL (MAX_ROWS / 2)
#define MAXN_NAUTY (MAX_COLS * (MAX_COMPLEX_PER_COL > 0 ? MAX_COMPLEX_PER_COL : 1))

#if MAXN_NAUTY > 64
#error "This 7x7-specialised build expects MAXN_NAUTY <= 64"
#endif

#include "nauty.h"

#if MAXN_NAUTY <= 16
typedef uint16_t AdjWord;
#define ADJWORD_MASK ((uint64_t)((1ULL << MAXN_NAUTY) - 1ULL))
#elif MAXN_NAUTY <= 32
typedef uint32_t AdjWord;
#define ADJWORD_MASK ((uint64_t)((1ULL << MAXN_NAUTY) - 1ULL))
#else
typedef uint64_t AdjWord;
#if MAXN_NAUTY >= 64
#define ADJWORD_MASK (~0ULL)
#else
#define ADJWORD_MASK ((uint64_t)((1ULL << MAXN_NAUTY) - 1ULL))
#endif
#endif

#define MAX_PERMUTATIONS 5040 // 7!
#define MAX_DEGREE ((MAX_ROWS * MAX_COLS) + 1)

// Cache settings - tuned for better locality
#ifndef CACHE_BITS
#define CACHE_BITS 18
#endif
#define CACHE_SIZE (1 << CACHE_BITS)
#define CACHE_MASK (CACHE_SIZE - 1)
#ifndef CACHE_PROBE
#define CACHE_PROBE 16
#endif

// Lookaside cache for exact labelled graphs before nauty canonicalisation.
#ifndef RAW_CACHE_BITS
#define RAW_CACHE_BITS 13
#endif
#define RAW_CACHE_SIZE (1 << RAW_CACHE_BITS)
#define RAW_CACHE_MASK (RAW_CACHE_SIZE - 1)
#ifndef RAW_CACHE_PROBE
#define RAW_CACHE_PROBE 8
#endif

#ifndef FIXED_PREFIX2_BATCH_SIZE
#define FIXED_PREFIX2_BATCH_SIZE 16
#endif

#define MAX_ROW_PAIRS ((MAX_ROWS * (MAX_ROWS - 1)) / 2)

#ifndef RECT_COUNT_K4
#define RECT_COUNT_K4 0
#endif

#ifndef RECT_COUNT_K4_FEASIBILITY
#define RECT_COUNT_K4_FEASIBILITY 0
#endif

#ifndef RECT_REP_ORBIT_MARK_THRESHOLD
#define RECT_REP_ORBIT_MARK_THRESHOLD 8
#endif

// --- DATA TYPES ---

typedef __int128_t PolyCoeff;
typedef uint16_t PrefixId;

typedef struct {
    int rows;
    int cols;
    long long task_start;
    long long task_end;
    long long full_tasks;
} PolyFileMeta;

typedef struct {
    int deg;
    PolyCoeff coeffs[MAX_DEGREE];
} Poly;

typedef struct {
    uint8_t deg;
    PolyCoeff coeffs[MAXN_NAUTY + 1];
} GraphPoly;

typedef struct {
    uint8_t mapping[MAX_ROWS];
    int num_blocks;
    uint32_t block_masks[MAX_ROWS]; 
    int is_complex[MAX_ROWS];       
    uint8_t complex_blocks[MAX_ROWS];
    int num_complex;
    int num_singletons;
} Partition;

typedef struct {
    uint8_t n;
    uint64_t vertex_mask;
    AdjWord adj[MAXN_NAUTY];
} Graph;

#define GRAPH_SIG_BITS ((MAXN_NAUTY * (MAXN_NAUTY - 1)) / 2)
#define GRAPH_SIG_WORDS ((GRAPH_SIG_BITS + 63) / 64)

typedef struct {
    Graph g;
    int base[MAX_COLS];
#if RECT_COUNT_K4_FEASIBILITY
    uint8_t pair_count[MAX_ROW_PAIRS];
    uint8_t remaining_capacity;
    uint32_t full_pair_mask;
    uint8_t last_base;
    uint8_t last_num_new;
#endif
} PartialGraphState;

typedef struct {
    int nmax;
    int mmax;
    graph* ng;
    graph* cg;
    int* lab;
    int* ptn;
    int* orbits;
} NautyWorkspace;

// Modified cache entry to store canonical adjacency
typedef struct {
    uint64_t key_hash;
    uint32_t key_n;
    uint8_t used;
} CacheKey;

typedef struct {
    CacheKey* keys;
    uint32_t* stamps;
    uint64_t* sigs;
    uint8_t* degs;
    PolyCoeff* coeffs;
    int mask;
    int probe;
    int poly_len;
    uint32_t next_stamp;
} GraphCache;

typedef struct {
    CacheKey* keys;
    uint32_t* stamps;
    AdjWord* rows;
    uint8_t* degs;
    PolyCoeff* coeffs;
    int mask;
    int probe;
    int poly_len;
    uint32_t next_stamp;
} RowGraphCache;

typedef struct {
    long long canon_prepare_calls;
    long long canon_prepare_accepts;
    long long canon_commit_calls;
    long long partial_append_calls;
    long long solve_structure_calls;
    long long solve_graph_calls;
    long long nauty_calls;
    long long hard_graph_nodes;
    long long canon_prepare_calls_by_depth[MAX_COLS + 1];
    long long canon_prepare_accepts_by_depth[MAX_COLS + 1];
    long long stabilizer_sum_by_depth[MAX_COLS + 1];
    long long canon_prepare_scanned_by_depth[MAX_COLS + 1];
    long long canon_prepare_active_by_depth[MAX_COLS + 1];
    long long canon_prepare_terminal_calls_by_depth[MAX_COLS + 1];
    long long canon_prepare_fast_continue_by_depth[MAX_COLS + 1];
    long long canon_prepare_terminal_continue_by_depth[MAX_COLS + 1];
    long long canon_prepare_equal_case_calls_by_depth[MAX_COLS + 1];
    long long canon_prepare_equal_case_rejects_by_depth[MAX_COLS + 1];
    long long canon_prepare_order_rejects_by_depth[MAX_COLS + 1];
    long long solve_graph_calls_by_n[MAXN_NAUTY + 1];
    long long solve_graph_raw_hits_by_n[MAXN_NAUTY + 1];
    long long solve_graph_canon_hits_by_n[MAXN_NAUTY + 1];
    long long hard_graph_nodes_by_n[MAXN_NAUTY + 1];
    long long solve_graph_lookup_calls_by_n[MAXN_NAUTY + 1];
    long long solve_graph_connected_lookup_calls_by_n[MAXN_NAUTY + 1];
    long long solve_graph_component_calls_by_n[MAXN_NAUTY + 1];
    long long solve_graph_hard_misses_by_n[MAXN_NAUTY + 1];
    long long hard_graph_articulation_by_n[MAXN_NAUTY + 1];
    long long hard_graph_k2_separator_by_n[MAXN_NAUTY + 1];
    long long hard_graph_nodes_by_n_degree[MAXN_NAUTY + 1][MAXN_NAUTY + 1];
    int hard_graph_max_n;
    int hard_graph_max_degree;
    double canon_prepare_time;
    double canon_commit_time;
    double partial_append_time;
    double build_weight_time;
    double solve_graph_time;
    double get_canonical_graph_time;
    double get_canonical_graph_dense_rows_time;
    double get_canonical_graph_build_input_time;
    double nauty_time;
    double get_canonical_graph_rebuild_time;
    double solve_graph_time_by_n[MAXN_NAUTY + 1];
    double solve_graph_lookup_time_by_n[MAXN_NAUTY + 1];
    double solve_graph_connected_lookup_time_by_n[MAXN_NAUTY + 1];
    double solve_graph_raw_hit_time_by_n[MAXN_NAUTY + 1];
    double solve_graph_canon_hit_time_by_n[MAXN_NAUTY + 1];
    double solve_graph_component_time_by_n[MAXN_NAUTY + 1];
    double solve_graph_hard_miss_time_by_n[MAXN_NAUTY + 1];
    double solve_graph_hard_miss_separator_time_by_n[MAXN_NAUTY + 1];
    double solve_graph_hard_miss_pick_time_by_n[MAXN_NAUTY + 1];
    double solve_graph_hard_miss_delete_time_by_n[MAXN_NAUTY + 1];
    double solve_graph_hard_miss_contract_build_time_by_n[MAXN_NAUTY + 1];
    double solve_graph_hard_miss_contract_solve_time_by_n[MAXN_NAUTY + 1];
    double solve_graph_hard_miss_store_time_by_n[MAXN_NAUTY + 1];
} ProfileStats;

#define TASK_PROFILE_TOPK 8

typedef struct {
    long long task_count;
    double task_time_sum;
    double task_time_max;
    long long task_max_index;
    double top_times[TASK_PROFILE_TOPK];
    long long top_indices[TASK_PROFILE_TOPK];
} TaskTimingStats;

typedef struct {
    uint8_t depth;
    PrefixId prefix[MAX_COLS];
    double elapsed;
    long long solve_graph_calls;
    long long nauty_calls;
    long long hard_graph_nodes;
    uint8_t max_hard_graph_n;
    uint8_t max_hard_graph_degree;
} QueueSubtaskTopEntry;

typedef struct {
    long long task_count;
    double task_time_sum;
    double task_time_max;
    long long solve_graph_call_sum;
    long long nauty_call_sum;
    long long hard_graph_node_sum;
    int max_hard_graph_n;
    int max_hard_graph_degree;
    QueueSubtaskTopEntry top[TASK_PROFILE_TOPK];
} QueueSubtaskTimingStats;

typedef struct {
    long long hard_graph_nodes;
    int max_n;
    int max_degree;
} GraphHardStats;

#define SHARED_CACHE_EXPORT_CAP 64

typedef struct {
    uint64_t key_hash;
    uint32_t key_n;
    Graph g;
    uint64_t row_mask;
    GraphPoly value;
} SharedGraphCacheExportEntry;

typedef struct {
    SharedGraphCacheExportEntry entries[SHARED_CACHE_EXPORT_CAP];
    int count;
} SharedGraphCacheExporter;

typedef struct {
    GraphCache cache;
    pthread_rwlock_t lock;
    int enabled;
} SharedGraphCache;

typedef struct {
    PrefixId* i;
    PrefixId* j;
    long long count;
    long long capacity;
} PrefixTaskBuffer;

typedef struct {
    PrefixId i;
    uint32_t start;
    uint16_t count;
} Prefix2Batch;

typedef struct {
    long long pending;
    long long task_index;
    double launched_at;
} RootTaskState;

typedef struct {
    uint8_t depth;
    PrefixId prefix[MAX_COLS];
    PrefixId lo;
    PrefixId hi;
    long long root_id;
} LocalTask;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    LocalTask* tasks;
    int capacity;
    int head;
    int tail;
    int count;
    int inflight;
    int stop;
    atomic_int outstanding_tasks;
    atomic_int idle_threads;
    atomic_long donated_tasks;
    atomic_long work_budget_continuations;
    atomic_long max_outstanding_tasks;
    RootTaskState* roots;
    long long root_count;
    int total_threads;
    double occupancy_last_at;
    int occupancy_idle_threads;
    double idle_thread_seconds;
    QueueSubtaskTimingStats profile_stats[MAX_COLS + 1];
    double profile_started_at;
    double next_profile_report_at;
} LocalTaskQueue;

#define CONNECTED_CANON_LOOKUP_MAX_N 10
#define CONNECTED_CANON_LOOKUP_MAGIC UINT64_C(0x43434c394741424c)
#define CONNECTED_CANON_LOOKUP_VERSION 1U

typedef struct {
    uint64_t mask;
    int32_t coeffs[CONNECTED_CANON_LOOKUP_MAX_N + 1];
} ConnectedCanonLookupEntry;

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t n;
    uint32_t count;
} ConnectedCanonLookupHeader;

// --- GLOBALS ---
static int num_partitions = 0;
static int perm_count = 0;
static int max_partition_capacity = 0;
static int max_complex_per_partition = 0;
static Partition* partitions = NULL;
static int (*perms)[MAX_ROWS] = NULL;
static uint16_t* perm_table = NULL;
static uint16_t* perm_order_by_value = NULL;
static uint16_t* perm_value_prefix_end = NULL;
static uint16_t* partition_id_lookup = NULL;
static uint32_t partition_id_lookup_size = 0;
static uint64_t factorial[20];
#if MAX_ROWS <= 7
typedef uint8_t ComplexMask;
#else
typedef uint32_t ComplexMask;
#endif
static ComplexMask* overlap_mask = NULL;
static ComplexMask* intra_mask = NULL;
static Poly* partition_weight_poly = NULL;
static uint8_t* partition_weight4 = NULL;
#if RECT_COUNT_K4_FEASIBILITY
static uint32_t* pair_shadow_mask = NULL;
static uint8_t* pair_shadow_pairs = NULL;
static uint8_t* suffix_min_pairs = NULL;
static int pair_index[MAX_ROWS][MAX_ROWS];
static int num_row_pairs = 0;
static int min_partition_pairs = 0;
#endif
static PrefixId* g_live_prefix2_i = NULL;
static PrefixId* g_live_prefix2_j = NULL;
static long long g_live_prefix2_count = 0;

static long long completed_tasks = 0;
static Poly global_poly = {0}; 

static int g_rows = DEFAULT_ROWS;
static int g_cols = DEFAULT_COLS;
static ProgressReporter progress_reporter;
static int g_use_raw_cache = 1;
static long long progress_last_reported = 0;
static int g_adaptive_subdivide = 0;
static int g_adaptive_max_depth = 3;
static long long g_adaptive_work_budget = 0;
static __thread ProfileStats* tls_profile = NULL;
static __thread GraphHardStats* tls_hard_graph_stats = NULL;
static __thread long long* tls_adaptive_work_counter = NULL;
static __thread SharedGraphCacheExporter* tls_shared_cache_exporter = NULL;
static const char* g_task_times_out_path = NULL;
static long long g_task_times_first_task = 0;
static long long g_task_times_count = 0;
static double* g_task_times_values = NULL;
static int g_effective_prefix_depth = 0;
static double g_queue_profile_report_step = 0.0;
static int g_shared_cache_merge = 0;
static int g_shared_cache_bits = 16;
static int g_profile_separators = 0;
static SharedGraphCache* g_shared_graph_cache = NULL;
#define SMALL_GRAPH_LOOKUP_MAX_N 7
static int g_small_graph_lookup_ready = 0;
static double g_small_graph_lookup_init_time = 0.0;
static int g_small_graph_lookup_loaded_from_file = 0;
static int32_t* g_small_graph_lookup_coeffs[SMALL_GRAPH_LOOKUP_MAX_N + 1] = {0};
static uint8_t g_small_graph_edge_u[SMALL_GRAPH_LOOKUP_MAX_N + 1][21];
static uint8_t g_small_graph_edge_v[SMALL_GRAPH_LOOKUP_MAX_N + 1][21];
static uint32_t g_small_graph_graph_count[SMALL_GRAPH_LOOKUP_MAX_N + 1] = {0};
static uint8_t g_small_graph_edge_count[SMALL_GRAPH_LOOKUP_MAX_N + 1] = {0};
static ConnectedCanonLookupEntry* g_connected_canon_lookup = NULL;
static uint32_t g_connected_canon_lookup_count = 0;
static int g_connected_canon_lookup_ready = 0;
static int g_connected_canon_lookup_loaded = 0;
static int g_connected_canon_lookup_n = 0;
static double g_connected_canon_lookup_load_time = 0.0;

static void* checked_calloc(size_t count, size_t size, const char* label);
static void* checked_aligned_alloc(size_t alignment, size_t size, const char* label);
static void shared_graph_cache_flush_exports(void);
static inline void graph_cache_load_poly(const GraphCache* cache, int slot, GraphPoly* value);
static inline uint32_t graph_cache_next_stamp(GraphCache* cache);
static inline void graph_cache_touch_slot(GraphCache* cache, int slot);
static inline int graph_cache_slot_matches_sig(const GraphCache* cache, int slot, uint64_t key_hash,
                                               uint32_t key_n, const uint64_t* sig);
void store_graph_cache_entry(GraphCache* cache, uint64_t key_hash, uint32_t key_n, const Graph* g,
                             uint64_t row_mask, const GraphPoly* value);
static inline void row_graph_cache_load_poly(const RowGraphCache* cache, int slot, GraphPoly* value);
static inline uint32_t row_graph_cache_next_stamp(RowGraphCache* cache);
static inline void row_graph_cache_touch_slot(RowGraphCache* cache, int slot);
static inline int row_graph_cache_slot_matches_graph(const RowGraphCache* cache, int slot,
                                                     uint64_t key_hash, uint32_t key_n,
                                                     const Graph* g, AdjWord row_mask);
static int row_graph_cache_lookup_poly(RowGraphCache* cache, uint64_t key_hash, uint32_t key_n,
                                       const Graph* g, AdjWord row_mask, GraphPoly* value,
                                       int touch);
static void store_row_graph_cache_entry(RowGraphCache* cache, uint64_t key_hash, uint32_t key_n,
                                        const Graph* g, AdjWord row_mask,
                                        const GraphPoly* value);
static void small_graph_lookup_init(void);
static void small_graph_lookup_free(void);
static void connected_canon_lookup_init(void);
static void connected_canon_lookup_free(void);
static inline int32_t* small_graph_poly_slot(int n, uint32_t mask);
static uint32_t small_graph_pack_mask(const Graph* g);
static uint64_t graph_pack_upper_mask64(const Graph* g);
static int connected_canon_lookup_entry_cmp(const void* lhs, const void* rhs);
static inline uint64_t graph_row_mask(int n);
static uint32_t graph_build_dense_rows(const Graph* g, AdjWord* rows);
static void graph_apply_permutation_dense_rows(uint32_t n, const AdjWord* dense_rows,
                                               const uint8_t* new_index_of_old, Graph* dst);
static void get_canonical_graph_from_dense_rows(int n, const AdjWord* rows, Graph* canon,
                                                NautyWorkspace* ws, ProfileStats* profile);

static inline ComplexMask* intra_mask_row(int partition_id) {
    return intra_mask + (size_t)partition_id * (size_t)max_complex_per_partition;
}

static inline ComplexMask intra_mask_get(int partition_id, int complex_idx) {
    return intra_mask_row(partition_id)[complex_idx];
}

static inline ComplexMask* overlap_mask_row(int lhs_partition_id, int rhs_partition_id) {
    return overlap_mask +
           (((size_t)lhs_partition_id * (size_t)num_partitions + (size_t)rhs_partition_id) *
            (size_t)max_complex_per_partition);
}

static inline ComplexMask overlap_mask_get(int lhs_partition_id, int rhs_partition_id, int complex_idx) {
    return overlap_mask_row(lhs_partition_id, rhs_partition_id)[complex_idx];
}

static void unrank_prefix2(long long rank, int* i, int* j);
static void unrank_prefix3(long long rank, int* i, int* j, int* k);
static void unrank_prefix4(long long rank, int* i, int* j, int* k, int* l);
static inline long long repeated_combo_count(int values, int slots);

#if RECT_COUNT_K4
typedef unsigned __int128 WeightAccum;
#else
typedef Poly WeightAccum;
#endif

static void task_timing_insert_topk(TaskTimingStats* stats, long long task_index, double elapsed) {
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

static void queue_subtask_record(QueueSubtaskTimingStats* stats, const LocalTask* task,
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

static void queue_subtask_merge(QueueSubtaskTimingStats* dst, const QueueSubtaskTimingStats* src) {
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

static int decode_task_prefix(long long task_index, int* i, int* j, int* k, int* l) {
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

static void write_task_times_file(const char* path) {
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

#define DEFAULT_PROGRESS_UPDATES 2000
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

static inline void flush_completed_tasks(long long total_tasks, long long report_step,
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
    shared_graph_cache_flush_exports();
    (*pending_completed)++;
    if (*pending_completed >= PROGRESS_FLUSH_BATCH) {
        flush_completed_tasks(total_tasks, report_step, start_time, pending_completed);
    }
}

static inline void complete_task_report_and_time(long long total_tasks, long long report_step,
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

static void local_queue_init(LocalTaskQueue* queue, int capacity,
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

static void local_queue_free(LocalTaskQueue* queue) {
    free(queue->tasks);
    free(queue->roots);
    pthread_cond_destroy(&queue->cond);
    pthread_mutex_destroy(&queue->mutex);
    memset(queue, 0, sizeof(*queue));
}

static inline void local_task_from_stack(LocalTask* task, long long root_id, int depth, const int* stack) {
    task->depth = (uint8_t)depth;
    task->root_id = root_id;
    for (int i = 0; i < depth; i++) task->prefix[i] = (PrefixId)stack[i];
}

static int local_queue_try_push(LocalTaskQueue* queue, const LocalTask* task) {
    int pushed = 0;
    int outstanding = 0;
    pthread_mutex_lock(&queue->mutex);
    if (!queue->stop && queue->count < queue->capacity) {
        __atomic_add_fetch(&queue->roots[task->root_id].pending, 1, __ATOMIC_RELAXED);
        queue->tasks[queue->tail] = *task;
        queue->tail = (queue->tail + 1) % queue->capacity;
        queue->count++;
        outstanding = atomic_fetch_add_explicit(&queue->outstanding_tasks, 1, memory_order_relaxed) + 1;
        local_queue_note_outstanding(queue, outstanding);
        pushed = 1;
        pthread_cond_signal(&queue->cond);
    }
    pthread_mutex_unlock(&queue->mutex);
    return pushed;
}

static void local_queue_seed_push(LocalTaskQueue* queue, const LocalTask* task) {
    if (!local_queue_try_push(queue, task)) {
        fprintf(stderr, "Failed to seed local task queue\n");
        exit(1);
    }
}

static int local_queue_pop(LocalTaskQueue* queue, LocalTask* task) {
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

static void local_queue_finish_item(LocalTaskQueue* queue, long long root_id,
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

static void local_queue_record_profile(LocalTaskQueue* queue, const LocalTask* task,
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

static void local_queue_print_occupancy_summary(LocalTaskQueue* queue) {
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

static void* checked_aligned_alloc(size_t alignment, size_t size, const char* label) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        fprintf(stderr, "Failed to allocate %s (%zu bytes)\n", label, size);
        exit(1);
    }
    return ptr;
}

static void* checked_calloc(size_t count, size_t size, const char* label) {
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

static void prefix_task_buffer_init(PrefixTaskBuffer* buf, long long initial_capacity) {
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

static void prefix_task_buffer_push2(PrefixTaskBuffer* buf, int i, int j) {
    prefix_task_buffer_reserve(buf, buf->count + 1);
    buf->i[buf->count] = (PrefixId)i;
    buf->j[buf->count] = (PrefixId)j;
    buf->count++;
}

static inline long long repeated_combo_count(int values, int slots) {
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

static void get_prefix2_task(long long task_index, int* i, int* j) {
    if (g_live_prefix2_i && task_index >= 0 && task_index < g_live_prefix2_count) {
        *i = (int)g_live_prefix2_i[task_index];
        *j = (int)g_live_prefix2_j[task_index];
        return;
    }
    unrank_prefix2(task_index, i, j);
}

static void build_fixed_prefix2_batches(const PrefixId* live_i, const PrefixId* live_j,
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

static void unrank_prefix3(long long rank, int* i, int* j, int* k) {
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

static void unrank_prefix4(long long rank, int* i, int* j, int* k, int* l) {
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

static int bell_number_upper_bound(int rows) {
    static const int bell_numbers[] = {0, 1, 2, 5, 15, 52, 203, 877};
    if (rows < 0 || rows >= (int)(sizeof(bell_numbers) / sizeof(bell_numbers[0]))) {
        fprintf(stderr, "Unsupported row count for Bell number lookup: %d\n", rows);
        exit(1);
    }
    return bell_numbers[rows];
}

static void init_row_dependent_tables(void) {
    max_partition_capacity = bell_number_upper_bound(g_rows);
    perm_count = (int)factorial[g_rows];
    max_complex_per_partition = g_rows / 2;

    partitions = checked_calloc((size_t)max_partition_capacity, sizeof(*partitions), "partitions");
    perms = checked_calloc((size_t)perm_count, sizeof(*perms), "perms");
}

static void init_partition_lookup_tables(void) {
    size_t partition_count = (size_t)num_partitions;
    partition_id_lookup_size = 1u << (3 * g_rows);

    perm_table = checked_calloc(partition_count * (size_t)perm_count, sizeof(*perm_table), "perm_table");
    perm_order_by_value =
        checked_calloc(partition_count * (size_t)perm_count, sizeof(*perm_order_by_value),
                       "perm_order_by_value");
    perm_value_prefix_end =
        checked_calloc(partition_count * partition_count, sizeof(*perm_value_prefix_end),
                       "perm_value_prefix_end");
    partition_id_lookup =
        checked_aligned_alloc(64, (size_t)partition_id_lookup_size * sizeof(*partition_id_lookup),
                              "partition_id_lookup");
    memset(partition_id_lookup, 0xff, (size_t)partition_id_lookup_size * sizeof(*partition_id_lookup));
    overlap_mask = checked_calloc(partition_count * partition_count * (size_t)max_complex_per_partition,
                                  sizeof(*overlap_mask), "overlap_mask");
    intra_mask = checked_calloc(partition_count * (size_t)max_complex_per_partition,
                                sizeof(*intra_mask), "intra_mask");
    partition_weight_poly =
        checked_calloc(partition_count, sizeof(*partition_weight_poly), "partition_weight_poly");
    partition_weight4 = checked_calloc(partition_count, sizeof(*partition_weight4), "partition_weight4");
#if RECT_COUNT_K4_FEASIBILITY
    pair_shadow_mask = checked_calloc(partition_count, sizeof(*pair_shadow_mask), "pair_shadow_mask");
    pair_shadow_pairs = checked_calloc(partition_count, sizeof(*pair_shadow_pairs), "pair_shadow_pairs");
    suffix_min_pairs = checked_calloc(partition_count, sizeof(*suffix_min_pairs), "suffix_min_pairs");
#endif
}

static void free_row_dependent_tables(void) {
    free(partitions);
    free(perms);
    free(perm_table);
    free(perm_order_by_value);
    free(perm_value_prefix_end);
    free(partition_id_lookup);
    free(overlap_mask);
    free(intra_mask);
    free(partition_weight_poly);
    free(partition_weight4);
#if RECT_COUNT_K4_FEASIBILITY
    free(pair_shadow_mask);
    free(pair_shadow_pairs);
    free(suffix_min_pairs);
#endif

    partitions = NULL;
    perms = NULL;
    perm_table = NULL;
    perm_order_by_value = NULL;
    perm_value_prefix_end = NULL;
    partition_id_lookup = NULL;
    partition_id_lookup_size = 0;
    overlap_mask = NULL;
    intra_mask = NULL;
    partition_weight_poly = NULL;
    partition_weight4 = NULL;
#if RECT_COUNT_K4_FEASIBILITY
    pair_shadow_mask = NULL;
    pair_shadow_pairs = NULL;
    suffix_min_pairs = NULL;
    num_row_pairs = 0;
    min_partition_pairs = 0;
#endif
    num_partitions = 0;
    perm_count = 0;
    max_partition_capacity = 0;
    max_complex_per_partition = 0;
}

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

static void usage(const char* prog) {
    fprintf(stderr,
            "Usage:\n"
            "  %s [rows cols] [--task-start N] [--task-end N] [--prefix-depth N] [--reorder] [--adaptive-subdivide] [--adaptive-max-depth N] [--adaptive-work-budget N] [--poly-out FILE]"
#if RECT_PROFILE
            " [--task-times-out FILE]"
#endif
            "\n"
            "\n"
            "Notes:\n"
            "  --task-start/--task-end define a half-open task range [start, end).\n"
            "  --prefix-depth may be 2, 3, or 4.\n"
            "  --reorder changes partition IDs and task numbering.\n"
            "  Adaptive subdivision currently supports only --prefix-depth 2.\n"
            "  In full polynomial mode it uses a local runtime queue of donated subtrees.\n"
            "  Profiling is selected at compile time.\n",
            prog);
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

// --- INITIALISATION ---

void generate_permutations() {
    int p[MAX_ROWS];
    for (int i = 0; i < g_rows; i++) p[i] = i;
    int count = 0;
    while (1) {
        memcpy(perms[count], p, sizeof(int) * g_rows);
        count++;
        int i = g_rows - 2;
        while (i >= 0 && p[i] >= p[i + 1]) i--;
        if (i < 0) break;
        int j = g_rows - 1;
        while (p[j] <= p[i]) j--;
        int temp = p[i]; p[i] = p[j]; p[j] = temp;
        int l = i + 1, r = g_rows - 1;
        while (l < r) { temp = p[l]; p[l] = p[r]; p[r] = temp; l++; r--; }
    }
    perm_count = count;
}

void normalize_partition(uint8_t* p) {
    uint8_t map[MAX_ROWS];
    memset(map, 255, sizeof(map));
    uint8_t next = 0;
    for (int i = 0; i < g_rows; i++) {
        if (map[p[i]] == 255) map[p[i]] = next++;
        p[i] = map[p[i]];
    }
}

static int partition_pair_shadow_size(const Partition* part) {
    int pairs = 0;
    for (int ci = 0; ci < part->num_complex; ci++) {
        int block = part->complex_blocks[ci];
        int sz = __builtin_popcount(part->block_masks[block]);
        pairs += (sz * (sz - 1)) / 2;
    }
    return pairs;
}

static int partition_largest_complex_block(const Partition* part) {
    int best = 0;
    for (int ci = 0; ci < part->num_complex; ci++) {
        int block = part->complex_blocks[ci];
        int sz = __builtin_popcount(part->block_masks[block]);
        if (sz > best) best = sz;
    }
    return best;
}

static int compare_partition_hardness(const void* lhs, const void* rhs) {
    const Partition* a = (const Partition*)lhs;
    const Partition* b = (const Partition*)rhs;

    if (a->num_complex != b->num_complex) return b->num_complex - a->num_complex;

    int ap = partition_pair_shadow_size(a);
    int bp = partition_pair_shadow_size(b);
    if (ap != bp) return bp - ap;

    int al = partition_largest_complex_block(a);
    int bl = partition_largest_complex_block(b);
    if (al != bl) return bl - al;

    if (a->num_blocks != b->num_blocks) return b->num_blocks - a->num_blocks;

    return memcmp(a->mapping, b->mapping, (size_t)g_rows);
}

static void reorder_partitions_by_hardness(void) {
    qsort(partitions, (size_t)num_partitions, sizeof(partitions[0]), compare_partition_hardness);
}

void generate_partitions_recursive(int idx, uint8_t* current, int max_val) {
    if (idx == g_rows) {
        Partition part;
        memset(&part, 0, sizeof(part));
        memcpy(part.mapping, current, g_rows);
        part.num_blocks = max_val + 1;
#if RECT_COUNT_K4
        if (part.num_blocks > 4) return;
#endif
        
        int counts[MAX_ROWS];
        for(int k=0; k<g_rows; k++) counts[k]=0;
        
        for (int r = 0; r < g_rows; r++) {
            part.block_masks[current[r]] |= (1 << r);
            counts[current[r]]++;
        }
        for (int b = 0; b < part.num_blocks; b++) {
            part.is_complex[b] = (counts[b] >= 2);
            if (part.is_complex[b]) part.num_complex++;
            else part.num_singletons++;
        }
        int ci = 0;
        for (int b = 0; b < part.num_blocks; b++) {
            if (part.is_complex[b]) part.complex_blocks[ci++] = (uint8_t)b;
        }
        if (num_partitions >= max_partition_capacity) {
            fprintf(stderr, "Partition table capacity exceeded for %d rows\n", g_rows);
            exit(1);
        }
        partitions[num_partitions++] = part;
        return;
    }
    for (int i = 0; i <= max_val; i++) {
        current[idx] = i;
        generate_partitions_recursive(idx + 1, current, max_val);
    }
    if (max_val < g_rows - 1
#if RECT_COUNT_K4
        && max_val + 2 <= 4
#endif
    ) {
        current[idx] = max_val + 1;
        generate_partitions_recursive(idx + 1, current, max_val + 1);
    }
}

int get_partition_id(uint8_t* map) {
    uint32_t key = 0;
    for (int i = 0; i < g_rows; i++) {
        key |= (uint32_t)map[i] << (3 * i);
    }
    if (key >= partition_id_lookup_size) return -1;
    uint16_t val = partition_id_lookup[key];
    return (val == UINT16_MAX) ? -1 : (int)val;
}

static void build_partition_id_lookup(void) {
    for (int id = 0; id < num_partitions; id++) {
        uint32_t key = 0;
        for (int i = 0; i < g_rows; i++) {
            key |= (uint32_t)partitions[id].mapping[i] << (3 * i);
        }
        partition_id_lookup[key] = (uint16_t)id;
    }
}

void build_perm_table() {
    uint8_t temp[MAX_ROWS];
    
    for (int id = 0; id < num_partitions; id++) {
        for (int pi = 0; pi < perm_count; pi++) {
            for (int r = 0; r < g_rows; r++) {
                temp[r] = partitions[id].mapping[perms[pi][r]];
            }
            normalize_partition(temp);
            int pid = get_partition_id(temp);
            if (pid < 0 || pid > UINT16_MAX) {
                fprintf(stderr, "partition id out of range in build_perm_table: %d\n", pid);
                exit(1);
            }
            perm_table[(size_t)id * (size_t)perm_count + (size_t)pi] = (uint16_t)pid;
        }
    }
}

static void build_terminal_perm_order_tables(void) {
    uint16_t* counts =
        checked_calloc((size_t)num_partitions, sizeof(*counts), "terminal_perm_order_counts");
    uint16_t* offsets =
        checked_calloc((size_t)num_partitions, sizeof(*offsets), "terminal_perm_order_offsets");

    for (int id = 0; id < num_partitions; id++) {
        uint16_t* sorted_row = perm_order_by_value + (size_t)id * (size_t)perm_count;
        uint16_t* prefix_row = perm_value_prefix_end + (size_t)id * (size_t)num_partitions;
        const uint16_t* perm_row = perm_table + (size_t)id * (size_t)perm_count;

        memset(counts, 0, (size_t)num_partitions * sizeof(*counts));
        for (int p = 0; p < perm_count; p++) {
            counts[perm_row[p]]++;
        }

        uint16_t next = 0;
        for (int value = 0; value < num_partitions; value++) {
            offsets[value] = next;
            next = (uint16_t)(next + counts[value]);
            prefix_row[value] = next;
        }

        for (int p = 0; p < perm_count; p++) {
            uint16_t value = perm_row[p];
            sorted_row[offsets[value]++] = (uint16_t)p;
        }
    }

    free(counts);
    free(offsets);
}

void build_overlap_table() {
    memset(overlap_mask, 0,
           (size_t)num_partitions * (size_t)num_partitions * (size_t)max_complex_per_partition *
               sizeof(*overlap_mask));
    memset(intra_mask, 0,
           (size_t)num_partitions * (size_t)max_complex_per_partition * sizeof(*intra_mask));
    for (int pid1 = 0; pid1 < num_partitions; pid1++) {
        for (int i1 = 0; i1 < partitions[pid1].num_complex; i1++) {
            ComplexMask mask = 0;
            for (int i2 = 0; i2 < partitions[pid1].num_complex; i2++) {
                if (i1 != i2) mask |= (ComplexMask)(1u << i2);
            }
            intra_mask_row(pid1)[i1] = mask;
        }
        for (int i1 = 0; i1 < partitions[pid1].num_complex; i1++) {
            int b1 = partitions[pid1].complex_blocks[i1];
            uint32_t m1 = partitions[pid1].block_masks[b1];
            for (int pid2 = 0; pid2 < num_partitions; pid2++) {
                ComplexMask mask = 0;
                for (int i2 = 0; i2 < partitions[pid2].num_complex; i2++) {
                    int b2 = partitions[pid2].complex_blocks[i2];
                    uint32_t m2 = partitions[pid2].block_masks[b2];
                    if (__builtin_popcount(m1 & m2) >= 2) {
                        mask |= (ComplexMask)(1u << i2);
                    }
                }
                overlap_mask_row(pid1, pid2)[i1] = mask;
            }
        }
    }
}

#if RECT_COUNT_K4_FEASIBILITY
static void init_pair_index(void) {
    memset(pair_index, -1, sizeof(pair_index));
    num_row_pairs = 0;
    for (int i = 0; i < g_rows; i++) {
        for (int j = i + 1; j < g_rows; j++) {
            pair_index[i][j] = num_row_pairs;
            pair_index[j][i] = num_row_pairs;
            num_row_pairs++;
        }
    }
}

static void build_partition_shadow_table(void) {
    min_partition_pairs = MAX_ROW_PAIRS;
    memset(pair_shadow_mask, 0, (size_t)num_partitions * sizeof(*pair_shadow_mask));
    memset(pair_shadow_pairs, 0, (size_t)num_partitions * sizeof(*pair_shadow_pairs));
    memset(suffix_min_pairs, 0, (size_t)num_partitions * sizeof(*suffix_min_pairs));

    for (int pid = 0; pid < num_partitions; pid++) {
        uint32_t shadow = 0;
        const Partition* part = &partitions[pid];
        for (int ci = 0; ci < part->num_complex; ci++) {
            int block = part->complex_blocks[ci];
            uint32_t mask = part->block_masks[block];
            for (int i = 0; i < g_rows; i++) {
                if (((mask >> i) & 1U) == 0) continue;
                for (int j = i + 1; j < g_rows; j++) {
                    if ((mask >> j) & 1U) {
                        shadow |= (uint32_t)(1U << pair_index[i][j]);
                    }
                }
            }
        }
        pair_shadow_mask[pid] = shadow;
        pair_shadow_pairs[pid] = (uint8_t)__builtin_popcount(shadow);
        if (pair_shadow_pairs[pid] < min_partition_pairs) {
            min_partition_pairs = pair_shadow_pairs[pid];
        }
    }

    if (num_partitions == 0) min_partition_pairs = 0;
    for (int pid = num_partitions - 1; pid >= 0; pid--) {
        if (pid == num_partitions - 1) suffix_min_pairs[pid] = pair_shadow_pairs[pid];
        else suffix_min_pairs[pid] = pair_shadow_pairs[pid] < suffix_min_pairs[pid + 1]
                                       ? pair_shadow_pairs[pid]
                                       : suffix_min_pairs[pid + 1];
    }
}
#endif

static void build_partition_weight_table(void) {
    for (int pid = 0; pid < num_partitions; pid++) {
        Poly weight;
        poly_one_ref(&weight);
        int c = partitions[pid].num_complex;
        int s = partitions[pid].num_singletons;
        if (s > 0) {
            poly_mul_falling_ref(&weight, c, s, &weight);
        }
        partition_weight_poly[pid] = weight;
    }
}

static inline uint8_t falling4_weight(int c, int s) {
    if (c + s > 4) return 0;
    uint8_t w = 1;
    for (int i = 0; i < s; i++) w = (uint8_t)(w * (uint8_t)(4 - c - i));
    return w;
}

static void build_partition_weight4_table(void) {
    for (int pid = 0; pid < num_partitions; pid++) {
        partition_weight4[pid] =
            falling4_weight(partitions[pid].num_complex, partitions[pid].num_singletons);
    }
}

#if RECT_COUNT_K4
static inline void weight_accum_one(WeightAccum* out) {
    *out = 1;
}

static inline void weight_accum_from_partition(int pid, WeightAccum* out) {
    *out = (WeightAccum)partition_weight4[pid];
}

static inline void weight_accum_mul_partition(const WeightAccum* src, int pid, WeightAccum* out) {
    *out = (*src) * (WeightAccum)partition_weight4[pid];
}

static inline void weight_accum_scale_to_poly(const WeightAccum* weight_prod, long long mult_coeff,
                                              long long row_orbit, uint64_t graph_count4, Poly* out) {
    WeightAccum total = *weight_prod;
    total *= (WeightAccum)mult_coeff;
    total *= (WeightAccum)row_orbit;
    total *= (WeightAccum)graph_count4;
    poly_zero(out);
    out->coeffs[0] = (PolyCoeff)total;
}
#else
static inline void weight_accum_one(WeightAccum* out) {
    poly_one_ref(out);
}

static inline void weight_accum_from_partition(int pid, WeightAccum* out) {
    *out = partition_weight_poly[pid];
}

static inline void weight_accum_mul_partition(const WeightAccum* src, int pid, WeightAccum* out) {
    poly_mul_ref(src, &partition_weight_poly[pid], out);
}
#endif

// --- SYMMETRY LOGIC ---

#define CANON_PARTITION_ID_LIMIT (1u << 11)
#define REP_ORBIT_MARK_WORDS ((CANON_PARTITION_ID_LIMIT + 63u) / 64u)

typedef struct {
    int limit;
    int depth;
    uint8_t* first_greater;
    uint16_t* first_greater_val;
    uint16_t* equal_perm;
    uint16_t* changed_first_greater_idx;
    uint8_t* changed_first_greater_old_idx;
    uint16_t* changed_first_greater_old_val;
    uint16_t equal_count[MAX_COLS + 1];
    uint16_t changed_first_greater_count[MAX_COLS];
    uint16_t stack_vals[MAX_COLS];
    const uint16_t* stack_perm_rows[MAX_COLS];
    int stabilizer[MAX_COLS + 1];
} CanonState;

typedef struct {
    int limit;
    uint8_t* changed_first_greater_new_idx;
    uint16_t* changed_first_greater_new_val;
    uint16_t* next_equal_perm;
    uint16_t* changed_first_greater_idx;
    uint16_t next_equal_count;
    uint16_t changed_first_greater_count;
} CanonScratch;

typedef struct {
    RowGraphCache cache;
    RowGraphCache raw_cache;
    NautyWorkspace ws;
    CanonState canon_state;
    CanonScratch canon_scratch;
    PartialGraphState partial_graph;
    int stack[MAX_COLS];
    long long local_canon_calls;
    long long local_cache_hits;
    long long local_raw_cache_hits;
} WorkerCtx;

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

static void canon_state_init(CanonState* st, int limit) {
    memset(st, 0, sizeof(*st));
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

static void canon_state_free(CanonState* st) {
    free(st->first_greater);
    free(st->first_greater_val);
    free(st->equal_perm);
    free(st->changed_first_greater_idx);
    free(st->changed_first_greater_old_idx);
    free(st->changed_first_greater_old_val);
    memset(st, 0, sizeof(*st));
}

static void canon_scratch_init(CanonScratch* scratch, int limit) {
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

static void canon_scratch_free(CanonScratch* scratch) {
    free(scratch->changed_first_greater_new_idx);
    free(scratch->changed_first_greater_new_val);
    free(scratch->next_equal_perm);
    free(scratch->changed_first_greater_idx);
    memset(scratch, 0, sizeof(*scratch));
}

void canon_state_reset(CanonState* st, int limit) {
    st->limit = limit;
    st->depth = 0;
    st->stabilizer[0] = limit;
    st->equal_count[0] = (uint16_t)limit;
    uint16_t* equal_perm0 = canon_state_equal_perm_row(st, 0);
    for (int p = 0; p < limit; p++) {
        equal_perm0[p] = (uint16_t)p;
    }
    memset(st->first_greater, 0, (size_t)limit * sizeof(*st->first_greater));
    memset(st->first_greater_val, 0, (size_t)limit * sizeof(*st->first_greater_val));
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

static void get_canonical_graph_from_dense_rows(int n, const AdjWord* rows, Graph* canon,
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

static inline int32_t* small_graph_poly_slot(int n, uint32_t mask) {
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

static uint32_t small_graph_pack_mask(const Graph* g) {
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

static uint64_t graph_pack_upper_mask64(const Graph* g) {
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

static void small_graph_lookup_init(void) {
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

static void small_graph_lookup_free(void) {
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

static void small_graph_lookup_load_graph_poly(int n, uint32_t mask, GraphPoly* out) {
    const int32_t* coeffs = small_graph_poly_slot(n, mask);
    out->deg = (uint8_t)n;
    for (int i = 0; i <= n; i++) out->coeffs[i] = coeffs[i];
}

static const char* connected_canon_lookup_default_path(void) {
    const char* env_path = getenv("RECT_CONNECTED_CANON_LOOKUP");
    if (env_path && *env_path) return env_path;
    return NULL;
}

static int connected_canon_lookup_entry_cmp(const void* lhs, const void* rhs) {
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

static void connected_canon_lookup_init(void) {
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

static void connected_canon_lookup_free(void) {
    free(g_connected_canon_lookup);
    g_connected_canon_lookup = NULL;
    g_connected_canon_lookup_count = 0;
    g_connected_canon_lookup_ready = 0;
    g_connected_canon_lookup_loaded = 0;
    g_connected_canon_lookup_n = 0;
    g_connected_canon_lookup_load_time = 0.0;
}

static int connected_canon_lookup_load_graph_poly(const Graph* g, GraphPoly* out) {
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

static inline uint64_t graph_row_mask(int n) {
    if (n >= 64) return ~0ULL;
    if (n <= 0) return 0ULL;
    return (1ULL << n) - 1ULL;
}

static uint32_t graph_build_dense_rows(const Graph* g, AdjWord* rows) {
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

static void induced_subgraph_from_mask(const Graph* src, uint64_t mask, Graph* dst) {
    uint32_t n = graph_build_dense_rows_from_mask(src, mask, dst->adj);
    dst->n = (uint8_t)n;
    dst->vertex_mask = graph_row_mask((int)n);
}

static int graph_collect_components(const Graph* g, uint64_t* component_masks) {
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

static int graph_has_articulation_point(const Graph* g) {
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

static int graph_has_k2_separator(const Graph* g) {
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

static inline uint64_t graph_fill_dense_key_rows(const Graph* g, AdjWord row_mask, AdjWord* rows) {
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

static int row_graph_cache_lookup_poly(RowGraphCache* cache, uint64_t key_hash, uint32_t key_n,
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

static int row_graph_cache_lookup_rows(RowGraphCache* cache, uint64_t key_hash, uint32_t key_n,
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

static void shared_graph_cache_init(SharedGraphCache* shared, int bits, int poly_len) {
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

static void shared_graph_cache_free(SharedGraphCache* shared) {
    if (!shared) return;
    free(shared->cache.keys);
    free(shared->cache.stamps);
    free(shared->cache.sigs);
    free(shared->cache.degs);
    free(shared->cache.coeffs);
    pthread_rwlock_destroy(&shared->lock);
    memset(shared, 0, sizeof(*shared));
}

static int shared_graph_cache_lookup_poly(SharedGraphCache* shared, uint64_t key_hash, uint32_t key_n,
                                          const Graph* g, uint64_t row_mask, GraphPoly* value) {
    if (!shared || !shared->enabled) return 0;
    int found = 0;
    pthread_rwlock_rdlock(&shared->lock);
    found = graph_cache_lookup_poly(&shared->cache, key_hash, key_n, g, row_mask, value, 0);
    pthread_rwlock_unlock(&shared->lock);
    return found;
}

static void shared_graph_cache_flush_exports(void) {
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

static void shared_graph_cache_export(uint64_t key_hash, uint32_t key_n, const Graph* g,
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

static void store_row_graph_cache_entry(RowGraphCache* cache, uint64_t key_hash, uint32_t key_n,
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

static void store_row_graph_cache_entry_rows(RowGraphCache* cache, uint64_t key_hash, uint32_t key_n,
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

void remove_vertex(Graph* g, int i) {
    uint64_t bit = UINT64_C(1) << i;
    if ((g->vertex_mask & bit) == 0) return;
    g->vertex_mask &= ~bit;
    g->n--;
}

static inline void record_hard_graph_node(ProfileStats* profile, int n, int max_degree) {
    if (PROFILE_BUILD && profile) {
        profile->hard_graph_nodes++;
        if (n >= 0 && n <= MAXN_NAUTY) profile->hard_graph_nodes_by_n[n]++;
        if (n >= 0 && n <= MAXN_NAUTY && max_degree >= 0 && max_degree <= MAXN_NAUTY) {
            profile->hard_graph_nodes_by_n_degree[n][max_degree]++;
        }
        if (n > profile->hard_graph_max_n) profile->hard_graph_max_n = n;
        if (max_degree > profile->hard_graph_max_degree) profile->hard_graph_max_degree = max_degree;
    }
    if (tls_hard_graph_stats) {
        tls_hard_graph_stats->hard_graph_nodes++;
        if (n > tls_hard_graph_stats->max_n) tls_hard_graph_stats->max_n = n;
        if (max_degree > tls_hard_graph_stats->max_degree) tls_hard_graph_stats->max_degree = max_degree;
    }
    if (tls_adaptive_work_counter) {
        (*tls_adaptive_work_counter)++;
    }
}

static inline int graph_neighbors_form_clique(const Graph* g, uint64_t neighbors) {
    uint64_t rem = neighbors;
    while (rem) {
        int u = __builtin_ctzll(rem);
        if ((neighbors & ~((uint64_t)g->adj[u] & g->vertex_mask)) != (UINT64_C(1) << u)) {
            return 0;
        }
        rem &= rem - 1;
    }
    return 1;
}

static void graph_choose_branch_edge(const Graph* g, int* u_out, int* v_out, int* max_deg_out) {
    uint64_t active = g->vertex_mask;
    int max_deg = -1;
    int fallback_u = -1;
    int fallback_v = -1;
    int best_u = -1;
    int best_v = -1;
    int best_score = INT_MIN;

    while (active) {
        int u = __builtin_ctzll(active);
        uint64_t u_neighbors = (uint64_t)g->adj[u] & g->vertex_mask;
        int u_deg = __builtin_popcountll(u_neighbors);
        if (u_deg > max_deg) max_deg = u_deg;
        uint64_t rem = u_neighbors & ~((UINT64_C(1) << (u + 1)) - 1);
        while (rem) {
            int v = __builtin_ctzll(rem);
            uint64_t v_neighbors = (uint64_t)g->adj[v] & g->vertex_mask;
            int v_deg = __builtin_popcountll(v_neighbors);
            if (fallback_u < 0 || u_deg > __builtin_popcountll((uint64_t)g->adj[fallback_u] & g->vertex_mask)) {
                fallback_u = u;
                fallback_v = v;
            }

            uint64_t u_after = u_neighbors & ~(UINT64_C(1) << v);
            uint64_t v_after = v_neighbors & ~(UINT64_C(1) << u);
            int u_clique = graph_neighbors_form_clique(g, u_after);
            int v_clique = graph_neighbors_form_clique(g, v_after);
            uint64_t merged_neighbors =
                (u_neighbors | v_neighbors) & ~((UINT64_C(1) << u) | (UINT64_C(1) << v));
            int merged_clique = graph_neighbors_form_clique(g, merged_neighbors);
            int common = __builtin_popcountll(u_neighbors & v_neighbors);
            int score = 1000 * (u_clique + v_clique + merged_clique) + 16 * common + u_deg + v_deg;
            if (score > best_score) {
                best_score = score;
                best_u = u;
                best_v = v;
            }
            rem &= rem - 1;
        }
        active &= active - 1;
    }

    if (best_u >= 0) {
        *u_out = best_u;
        *v_out = best_v;
    } else {
        *u_out = fallback_u;
        *v_out = fallback_v;
    }
    *max_deg_out = max_deg;
}

static void solve_graph_poly(const Graph* input_g, RowGraphCache* cache, RowGraphCache* raw_cache,
                             NautyWorkspace* ws, long long* local_canon_calls,
                             long long* local_cache_hits, long long* local_raw_cache_hits,
                             ProfileStats* profile, GraphPoly* out_result) {
#if RECT_COUNT_K4
    Graph g = *input_g;
    double solve_t0 = 0.0;
    int profile_n = 0;
    enum {
        SG_OUTCOME_NONE = 0,
        SG_OUTCOME_LOOKUP,
        SG_OUTCOME_CONNECTED_LOOKUP,
        SG_OUTCOME_RAW_HIT,
        SG_OUTCOME_CANON_HIT,
        SG_OUTCOME_COMPONENTS,
        SG_OUTCOME_HARD_MISS,
    } outcome = SG_OUTCOME_NONE;
    if (PROFILE_BUILD && profile) {
        profile->solve_graph_calls++;
        solve_t0 = omp_get_wtime();
    }

    uint64_t multiplier = 1;

    int changed = 1;
    while (changed && g.n > SMALL_GRAPH_LOOKUP_MAX_N) {
        changed = 0;
        uint64_t active = g.vertex_mask;
        while (active) {
            int i = __builtin_ctzll(active);
            active &= active - 1;
            uint64_t neighbors = (uint64_t)g.adj[i] & g.vertex_mask;
            int degree = __builtin_popcountll(neighbors);

            if (degree == 0) {
                multiplier *= 4;
                remove_vertex(&g, i);
                changed = 1;
                continue;
            }

            int is_clique = 1;
            uint64_t rem = neighbors;
            while (rem) {
                int u = __builtin_ctzll(rem);
                if ((neighbors & ~((uint64_t)g.adj[u] & g.vertex_mask)) != (1ULL << u)) {
                    is_clique = 0;
                    break;
                }
                rem &= rem - 1;
            }

            if (is_clique) {
                if (degree >= 4) {
                    graph_poly_set_count4(0, out_result);
                    outcome = SG_OUTCOME_HARD_MISS;
                    goto done;
                }
                multiplier *= (uint64_t)(4 - degree);
                remove_vertex(&g, i);
                changed = 1;
            }
        }
    }

    profile_n = g.n;
    if (PROFILE_BUILD && profile && profile_n >= 0 && profile_n <= MAXN_NAUTY) {
        profile->solve_graph_calls_by_n[profile_n]++;
    }

    if (g.n == 0) {
        graph_poly_set_count4(multiplier, out_result);
        goto done;
    }

    if (g.n <= SMALL_GRAPH_LOOKUP_MAX_N) {
        uint64_t count4 = small_graph_lookup_load_count4(g.n, small_graph_pack_mask(&g));
        graph_poly_set_count4(multiplier * count4, out_result);
        outcome = SG_OUTCOME_LOOKUP;
        goto done;
    }

    AdjWord row_mask = (AdjWord)graph_row_mask(g.n);
    AdjWord raw_rows[MAXN_NAUTY];
    uint64_t raw_hash = 0;
    GraphPoly raw_cached;
    if (g_use_raw_cache) {
        raw_hash = graph_fill_dense_key_rows(&g, row_mask, raw_rows);
        if (row_graph_cache_lookup_rows(raw_cache, raw_hash, (uint32_t)g.n, raw_rows,
                                        &raw_cached, 1)) {
            (*local_raw_cache_hits)++;
            if (PROFILE_BUILD && profile && g.n <= MAXN_NAUTY) {
                profile->solve_graph_raw_hits_by_n[g.n]++;
            }
            graph_poly_set_count4(multiplier * graph_poly_get_count4(&raw_cached), out_result);
            outcome = SG_OUTCOME_RAW_HIT;
            goto done;
        }
    }

    GraphPoly res;
    uint64_t component_masks[MAXN_NAUTY];
    int component_count = graph_collect_components(&g, component_masks);
    if (component_count > 1) {
        uint64_t total = 1;
        outcome = SG_OUTCOME_COMPONENTS;
        for (int i = 0; i < component_count; i++) {
            Graph subgraph;
            induced_subgraph_from_mask(&g, component_masks[i], &subgraph);
            GraphPoly part;
            solve_graph_poly(&subgraph, cache, raw_cache, ws,
                             local_canon_calls, local_cache_hits, local_raw_cache_hits,
                             profile, &part);
            total *= graph_poly_get_count4(&part);
        }
        graph_poly_set_count4(total, &res);
    } else {
        Graph canon;
        if (g_use_raw_cache) {
            get_canonical_graph_from_dense_rows((int)g.n, raw_rows, &canon, ws, profile);
        } else {
            get_canonical_graph(&g, &canon, ws, profile);
        }
        (*local_canon_calls)++;
        uint64_t hash = hash_graph(&canon);

        if (row_graph_cache_lookup_poly(cache, hash, (uint32_t)canon.n, &canon,
                                        (AdjWord)ADJWORD_MASK, &res, 1)) {
            (*local_cache_hits)++;
            if (g_use_raw_cache) {
                store_row_graph_cache_entry_rows(raw_cache, raw_hash, (uint32_t)g.n, raw_rows, &res);
            }
            if (PROFILE_BUILD && profile && canon.n <= MAXN_NAUTY) {
                profile->solve_graph_canon_hits_by_n[canon.n]++;
            }
            graph_poly_set_count4(multiplier * graph_poly_get_count4(&res), out_result);
            outcome = SG_OUTCOME_CANON_HIT;
            goto done;
        }

        if (shared_graph_cache_lookup_poly(g_shared_graph_cache, hash, (uint32_t)canon.n,
                                           &canon, ADJWORD_MASK, &res)) {
            store_row_graph_cache_entry(cache, hash, (uint32_t)canon.n, &canon,
                                        (AdjWord)ADJWORD_MASK, &res);
            (*local_cache_hits)++;
            if (PROFILE_BUILD && profile && canon.n <= MAXN_NAUTY) {
                profile->solve_graph_canon_hits_by_n[canon.n]++;
            }
            if (g_use_raw_cache) {
                store_row_graph_cache_entry_rows(raw_cache, raw_hash, (uint32_t)g.n, raw_rows, &res);
            }
            graph_poly_set_count4(multiplier * graph_poly_get_count4(&res), out_result);
            outcome = SG_OUTCOME_CANON_HIT;
            goto done;
        }

        uint64_t connected_lookup = connected_canon_lookup_load_count4(&canon);
        if (connected_lookup != UINT64_MAX) {
            graph_poly_set_count4(connected_lookup, &res);
            store_row_graph_cache_entry(cache, hash, (uint32_t)canon.n, &canon,
                                        (AdjWord)ADJWORD_MASK, &res);
            if (g_use_raw_cache) {
                store_row_graph_cache_entry_rows(raw_cache, raw_hash, (uint32_t)g.n, raw_rows, &res);
            }
            graph_poly_set_count4(multiplier * connected_lookup, out_result);
            outcome = SG_OUTCOME_CONNECTED_LOOKUP;
            goto done;
        }

        const Graph* branch_g = &canon;
        int max_deg = -1;
        for (int i = 0; i < branch_g->n; i++) {
            int d = __builtin_popcountll(branch_g->adj[i]);
            if (d > max_deg) max_deg = d;
        }
        if (max_deg > 0) record_hard_graph_node(profile, branch_g->n, max_deg);
        outcome = SG_OUTCOME_HARD_MISS;
        uint64_t count4 = count_graph_4_dsat(branch_g);
        graph_poly_set_count4(count4, &res);
        store_row_graph_cache_entry(cache, hash, (uint32_t)canon.n, &canon,
                                    (AdjWord)ADJWORD_MASK, &res);
        shared_graph_cache_export(hash, (uint32_t)canon.n, &canon, ADJWORD_MASK, &res);
    }

    if (g_use_raw_cache) {
        store_row_graph_cache_entry_rows(raw_cache, raw_hash, (uint32_t)g.n, raw_rows, &res);
    }
    graph_poly_set_count4(multiplier * graph_poly_get_count4(&res), out_result);
done:
    if (PROFILE_BUILD && profile) {
        double dt = omp_get_wtime() - solve_t0;
        profile->solve_graph_time += dt;
        if (profile_n >= 0 && profile_n <= MAXN_NAUTY) {
            profile->solve_graph_time_by_n[profile_n] += dt;
            switch (outcome) {
                case SG_OUTCOME_LOOKUP:
                    profile->solve_graph_lookup_calls_by_n[profile_n]++;
                    profile->solve_graph_lookup_time_by_n[profile_n] += dt;
                    break;
                case SG_OUTCOME_CONNECTED_LOOKUP:
                    profile->solve_graph_connected_lookup_calls_by_n[profile_n]++;
                    profile->solve_graph_connected_lookup_time_by_n[profile_n] += dt;
                    break;
                case SG_OUTCOME_RAW_HIT:
                    profile->solve_graph_raw_hit_time_by_n[profile_n] += dt;
                    break;
                case SG_OUTCOME_CANON_HIT:
                    profile->solve_graph_canon_hit_time_by_n[profile_n] += dt;
                    break;
                case SG_OUTCOME_COMPONENTS:
                    profile->solve_graph_component_calls_by_n[profile_n]++;
                    profile->solve_graph_component_time_by_n[profile_n] += dt;
                    break;
                case SG_OUTCOME_HARD_MISS:
                    profile->solve_graph_hard_misses_by_n[profile_n]++;
                    profile->solve_graph_hard_miss_time_by_n[profile_n] += dt;
                    break;
                default:
                    break;
            }
        }
    }
    return;
#else
    Graph g = *input_g;
    double solve_t0 = 0.0;
    int profile_n = 0;
    enum {
        SG_OUTCOME_NONE = 0,
        SG_OUTCOME_LOOKUP,
        SG_OUTCOME_CONNECTED_LOOKUP,
        SG_OUTCOME_RAW_HIT,
        SG_OUTCOME_CANON_HIT,
        SG_OUTCOME_COMPONENTS,
        SG_OUTCOME_HARD_MISS,
    } outcome = SG_OUTCOME_NONE;
    if (PROFILE_BUILD && profile) {
        profile->solve_graph_calls++;
        solve_t0 = omp_get_wtime();
    }
    GraphPoly multiplier;
    graph_poly_one_ref(&multiplier);
    
    int changed = 1;
    while (changed && g.n > SMALL_GRAPH_LOOKUP_MAX_N) {
        changed = 0;
        uint64_t active = g.vertex_mask;
        while (active) {
            int i = __builtin_ctzll(active);
            active &= active - 1;
            uint64_t neighbors = (uint64_t)g.adj[i] & g.vertex_mask;
            int degree = __builtin_popcountll(neighbors);
            
            if (degree == 0) {
                graph_poly_mul_linear_ref(&multiplier, 0, &multiplier);
                remove_vertex(&g, i);
                changed = 1;
                continue;
            }
            
            int is_clique = 1;
            uint64_t rem = neighbors;
            while (rem) {
                int u = __builtin_ctzll(rem);
                if ((neighbors & ~((uint64_t)g.adj[u] & g.vertex_mask)) != (1ULL << u)) {
                    is_clique = 0;
                    break;
                }
                rem &= rem - 1;
            }
            
            if (is_clique) {
                graph_poly_mul_linear_ref(&multiplier, degree, &multiplier);
                remove_vertex(&g, i);
                changed = 1;
            }
        }
    }

    profile_n = g.n;
    if (PROFILE_BUILD && profile && profile_n >= 0 && profile_n <= MAXN_NAUTY) {
        profile->solve_graph_calls_by_n[profile_n]++;
    }
    
    if (g.n == 0) {
        *out_result = multiplier;
        goto done;
    }

    if (g.n <= SMALL_GRAPH_LOOKUP_MAX_N) {
        GraphPoly small_poly;
        small_graph_lookup_load_graph_poly(g.n, small_graph_pack_mask(&g), &small_poly);
        graph_poly_mul_ref(&multiplier, &small_poly, out_result);
        outcome = SG_OUTCOME_LOOKUP;
        goto done;
    }

    AdjWord row_mask = (AdjWord)graph_row_mask(g.n);
    AdjWord raw_rows[MAXN_NAUTY];
    uint64_t raw_hash = 0;
    GraphPoly raw_cached;

    // Fast exact lookup on labelled graph before canonicalisation.
    if (g_use_raw_cache) {
        raw_hash = graph_fill_dense_key_rows(&g, row_mask, raw_rows);
        if (row_graph_cache_lookup_rows(raw_cache, raw_hash, (uint32_t)g.n, raw_rows,
                                        &raw_cached, 1)) {
            (*local_raw_cache_hits)++;
            if (PROFILE_BUILD && profile && g.n <= MAXN_NAUTY) {
                profile->solve_graph_raw_hits_by_n[g.n]++;
            }
            graph_poly_mul_ref(&multiplier, &raw_cached, out_result);
            outcome = SG_OUTCOME_RAW_HIT;
            goto done;
        }
    }

    GraphPoly res;
    uint64_t component_masks[MAXN_NAUTY];
    int component_count = graph_collect_components(&g, component_masks);
    if (component_count > 1) {
        outcome = SG_OUTCOME_COMPONENTS;
        graph_poly_one_ref(&res);
        for (int i = 0; i < component_count; i++) {
            Graph subgraph;
            induced_subgraph_from_mask(&g, component_masks[i], &subgraph);
            GraphPoly part;
            solve_graph_poly(&subgraph, cache, raw_cache, ws,
                             local_canon_calls, local_cache_hits, local_raw_cache_hits,
                             profile, &part);
            graph_poly_mul_ref(&res, &part, &res);
        }
    } else {
        // Canonicalise only if exact lookup missed and the graph is still connected.
        Graph canon;
        if (g_use_raw_cache) {
            get_canonical_graph_from_dense_rows((int)g.n, raw_rows, &canon, ws, profile);
        } else {
            get_canonical_graph(&g, &canon, ws, profile);
        }
        (*local_canon_calls)++;
        
        uint64_t hash = hash_graph(&canon);

        // Cache lookup using canonical form
        if (row_graph_cache_lookup_poly(cache, hash, (uint32_t)canon.n, &canon,
                                        (AdjWord)ADJWORD_MASK, &res, 1)) {
            (*local_cache_hits)++;
            if (g_use_raw_cache) {
                store_row_graph_cache_entry_rows(raw_cache, raw_hash, (uint32_t)g.n, raw_rows, &res);
            }
            if (PROFILE_BUILD && profile && canon.n <= MAXN_NAUTY) {
                profile->solve_graph_canon_hits_by_n[canon.n]++;
            }
            graph_poly_mul_ref(&multiplier, &res, out_result);
            outcome = SG_OUTCOME_CANON_HIT;
            goto done;
        }

        if (shared_graph_cache_lookup_poly(g_shared_graph_cache, hash, (uint32_t)canon.n,
                                           &canon, ADJWORD_MASK, &res)) {
            store_row_graph_cache_entry(cache, hash, (uint32_t)canon.n, &canon,
                                        (AdjWord)ADJWORD_MASK, &res);
            (*local_cache_hits)++;
            if (PROFILE_BUILD && profile && canon.n <= MAXN_NAUTY) {
                profile->solve_graph_canon_hits_by_n[canon.n]++;
            }
            if (g_use_raw_cache) {
                store_row_graph_cache_entry_rows(raw_cache, raw_hash, (uint32_t)g.n, raw_rows, &res);
            }
            graph_poly_mul_ref(&multiplier, &res, out_result);
            outcome = SG_OUTCOME_CANON_HIT;
            goto done;
        }

        if (connected_canon_lookup_load_graph_poly(&canon, &res)) {
            store_row_graph_cache_entry(cache, hash, (uint32_t)canon.n, &canon,
                                        (AdjWord)ADJWORD_MASK, &res);
            if (g_use_raw_cache) {
                store_row_graph_cache_entry_rows(raw_cache, raw_hash, (uint32_t)g.n, raw_rows, &res);
            }
            graph_poly_mul_ref(&multiplier, &res, out_result);
            outcome = SG_OUTCOME_CONNECTED_LOOKUP;
            goto done;
        }

        // Deletion-contraction on canonical graph
        double hard_sep_t = 0.0;
        double hard_pick_t = 0.0;
        double hard_del_t = 0.0;
        double hard_cont_build_t = 0.0;
        double hard_cont_solve_t = 0.0;
        double hard_store_t = 0.0;
        const Graph* branch_g = &canon;
        double phase_t0 = 0.0;
        if (PROFILE_BUILD && profile && branch_g->n >= 0 && branch_g->n <= MAXN_NAUTY) {
            phase_t0 = omp_get_wtime();
        }
        int max_deg = -1, u = -1, v = -1;
        graph_choose_branch_edge(branch_g, &u, &v, &max_deg);
        if (PROFILE_BUILD && profile && branch_g->n >= 0 && branch_g->n <= MAXN_NAUTY) {
            hard_pick_t += omp_get_wtime() - phase_t0;
        }
        if (u != -1 && max_deg > 0) record_hard_graph_node(profile, branch_g->n, max_deg);
        outcome = SG_OUTCOME_HARD_MISS;
        if (PROFILE_BUILD && g_profile_separators && profile &&
            branch_g->n >= 10 && branch_g->n <= MAXN_NAUTY) {
            phase_t0 = omp_get_wtime();
            if (graph_has_articulation_point(branch_g)) {
                profile->hard_graph_articulation_by_n[branch_g->n]++;
            }
            if (graph_has_k2_separator(branch_g)) {
                profile->hard_graph_k2_separator_by_n[branch_g->n]++;
            }
            hard_sep_t += omp_get_wtime() - phase_t0;
        }

        if (u != -1 && v != -1) {
            // Deletion: remove edge (u,v)
            Graph g_del = *branch_g;
            g_del.adj[u] &= ~(1ULL << v);
            g_del.adj[v] &= ~(1ULL << u);
            GraphPoly p_del;
            if (PROFILE_BUILD && profile && branch_g->n >= 0 && branch_g->n <= MAXN_NAUTY) {
                phase_t0 = omp_get_wtime();
            }
            solve_graph_poly(&g_del, cache, raw_cache, ws,
                             local_canon_calls, local_cache_hits, local_raw_cache_hits,
                             profile, &p_del);
            if (PROFILE_BUILD && profile && branch_g->n >= 0 && branch_g->n <= MAXN_NAUTY) {
                hard_del_t += omp_get_wtime() - phase_t0;
            }

            // Contraction: merge v into u
            Graph g_cont = *branch_g;
            if (PROFILE_BUILD && profile && branch_g->n >= 0 && branch_g->n <= MAXN_NAUTY) {
                phase_t0 = omp_get_wtime();
            }
            uint64_t merged_nbrs =
                ((uint64_t)g_cont.adj[u] | (uint64_t)g_cont.adj[v]) &
                g_cont.vertex_mask & ~((UINT64_C(1) << u) | (UINT64_C(1) << v));
            g_cont.adj[u] = (AdjWord)merged_nbrs;
            uint64_t nbrs = merged_nbrs;
            while (nbrs) {
                int k = __builtin_ctzll(nbrs);
                g_cont.adj[k] |= (AdjWord)(UINT64_C(1) << u);
                nbrs &= nbrs - 1;
            }
            remove_vertex(&g_cont, v);
            if (PROFILE_BUILD && profile && branch_g->n >= 0 && branch_g->n <= MAXN_NAUTY) {
                hard_cont_build_t += omp_get_wtime() - phase_t0;
                phase_t0 = omp_get_wtime();
            }
            GraphPoly p_cont;
            solve_graph_poly(&g_cont, cache, raw_cache, ws,
                             local_canon_calls, local_cache_hits, local_raw_cache_hits,
                             profile, &p_cont);
            if (PROFILE_BUILD && profile && branch_g->n >= 0 && branch_g->n <= MAXN_NAUTY) {
                hard_cont_solve_t += omp_get_wtime() - phase_t0;
            }

            graph_poly_sub_ref(&p_del, &p_cont, &res);
        } else {
            graph_poly_one_ref(&res);
            for (int k = 0; k < branch_g->n; k++) graph_poly_mul_linear_ref(&res, 0, &res);
        }

        if (PROFILE_BUILD && profile && branch_g->n >= 0 && branch_g->n <= MAXN_NAUTY) {
            phase_t0 = omp_get_wtime();
        }
        store_row_graph_cache_entry(cache, hash, (uint32_t)canon.n, &canon,
                                    (AdjWord)ADJWORD_MASK, &res);
        shared_graph_cache_export(hash, (uint32_t)canon.n, &canon, ADJWORD_MASK, &res);
        if (PROFILE_BUILD && profile && branch_g->n >= 0 && branch_g->n <= MAXN_NAUTY) {
            hard_store_t += omp_get_wtime() - phase_t0;
            profile->solve_graph_hard_miss_separator_time_by_n[branch_g->n] += hard_sep_t;
            profile->solve_graph_hard_miss_pick_time_by_n[branch_g->n] += hard_pick_t;
            profile->solve_graph_hard_miss_delete_time_by_n[branch_g->n] += hard_del_t;
            profile->solve_graph_hard_miss_contract_build_time_by_n[branch_g->n] += hard_cont_build_t;
            profile->solve_graph_hard_miss_contract_solve_time_by_n[branch_g->n] += hard_cont_solve_t;
            profile->solve_graph_hard_miss_store_time_by_n[branch_g->n] += hard_store_t;
        }
    }

    if (g_use_raw_cache) {
        store_row_graph_cache_entry_rows(raw_cache, raw_hash, (uint32_t)g.n, raw_rows, &res);
    }
    graph_poly_mul_ref(&multiplier, &res, out_result);
done:
    if (PROFILE_BUILD && profile) {
        double dt = omp_get_wtime() - solve_t0;
        profile->solve_graph_time += dt;
        if (profile_n >= 0 && profile_n <= MAXN_NAUTY) {
            profile->solve_graph_time_by_n[profile_n] += dt;
            switch (outcome) {
                case SG_OUTCOME_LOOKUP:
                    profile->solve_graph_lookup_calls_by_n[profile_n]++;
                    profile->solve_graph_lookup_time_by_n[profile_n] += dt;
                    break;
                case SG_OUTCOME_RAW_HIT:
                    profile->solve_graph_raw_hit_time_by_n[profile_n] += dt;
                    break;
                case SG_OUTCOME_CONNECTED_LOOKUP:
                    profile->solve_graph_connected_lookup_calls_by_n[profile_n]++;
                    profile->solve_graph_connected_lookup_time_by_n[profile_n] += dt;
                    break;
                case SG_OUTCOME_CANON_HIT:
                    profile->solve_graph_canon_hit_time_by_n[profile_n] += dt;
                    break;
                case SG_OUTCOME_COMPONENTS:
                    profile->solve_graph_component_calls_by_n[profile_n]++;
                    profile->solve_graph_component_time_by_n[profile_n] += dt;
                    break;
                case SG_OUTCOME_HARD_MISS:
                    profile->solve_graph_hard_misses_by_n[profile_n]++;
                    profile->solve_graph_hard_miss_time_by_n[profile_n] += dt;
                    break;
                default:
                    break;
            }
        }
    }
#endif
}

static void partial_graph_reset(PartialGraphState* st) {
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

static inline int partial_graph_append_checked(PartialGraphState* st, int depth, int pid,
                                               const int* stack, int cols_left) {
    if (!partial_graph_candidate_can_fit(st, pid, cols_left)) return 0;
    if (!partial_graph_append(st, depth, pid, stack)) return 0;
#if RECT_COUNT_K4_FEASIBILITY
    if (!partial_graph_is_feasible(st, cols_left)) return 0;
#endif
    return 1;
}

static void build_live_prefix2_tasks(PrefixId** live_i_out, PrefixId** live_j_out,
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

static void dfs(int depth, int min_idx, int* stack, CanonState* canon_state,
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

static void execute_prefix2_fixed_batch(PrefixId i, const PrefixId* js, const long long* ps, int count,
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

static void execute_local_runtime_task(const LocalTask* task, WorkerCtx* ctx, Poly* thread_total,
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

int main(int argc, char** argv) {
    long long task_start = 0;
    long long task_end = -1;
    const char* poly_out_path = NULL;
    int prefix_depth_override = -1;
    int reorder_partitions_flag = 0;
    int positional_count = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--poly-out") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            poly_out_path = argv[++i];
        } else if (strcmp(argv[i], "--task-start") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            task_start = parse_ll_or_die(argv[++i], "--task-start");
        } else if (strcmp(argv[i], "--task-end") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            task_end = parse_ll_or_die(argv[++i], "--task-end");
        } else if (strcmp(argv[i], "--prefix-depth") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            prefix_depth_override = (int)parse_ll_or_die(argv[++i], "--prefix-depth");
        } else if (strcmp(argv[i], "--reorder") == 0) {
            reorder_partitions_flag = 1;
        } else if (strcmp(argv[i], "--adaptive-subdivide") == 0) {
            g_adaptive_subdivide = 1;
        } else if (strcmp(argv[i], "--adaptive-max-depth") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            g_adaptive_max_depth = (int)parse_ll_or_die(argv[++i], "--adaptive-max-depth");
        } else if (strcmp(argv[i], "--adaptive-work-budget") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            g_adaptive_work_budget = parse_ll_or_die(argv[++i], "--adaptive-work-budget");
        } else if (strcmp(argv[i], "--task-times-out") == 0) {
#if !RECT_PROFILE
            fprintf(stderr, "--task-times-out requires a profiling build\n");
            return 1;
#else
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            g_task_times_out_path = argv[++i];
#endif
        } else if (argv[i][0] == '-') {
            usage(argv[0]);
            return 1;
        } else if (positional_count == 0) {
            g_rows = (int)parse_ll_or_die(argv[i], "rows");
            positional_count++;
        } else if (positional_count == 1) {
            g_cols = (int)parse_ll_or_die(argv[i], "cols");
            positional_count++;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (g_rows < 1 || g_cols < 1 || g_rows > MAX_ROWS || g_cols > MAX_COLS) {
        fprintf(stderr, "Rows/cols must be in range 1..%d and 1..%d\n", MAX_ROWS, MAX_COLS);
        return 1;
    }

    // Verify nauty build/runtime compatibility.
    int max_n = MAXN_NAUTY;
    int max_m = SETWORDSNEEDED(max_n);
    nauty_check(WORDSIZE, max_m, max_n, NAUTYVERSIONID);
    
    // 1. Initialise maths tables
    factorial[0] = 1;
    for(int i=1; i<=19; i++) factorial[i] = factorial[i-1]*i;

    // 2. Data structures
    init_row_dependent_tables();
    generate_permutations();
    uint8_t buffer[MAX_ROWS] = {0};
    generate_partitions_recursive(0, buffer, -1);
    if (reorder_partitions_flag) {
        reorder_partitions_by_hardness();
    }
#if RECT_COUNT_K4_FEASIBILITY
    init_pair_index();
#endif
    if (num_partitions >= CANON_PARTITION_ID_LIMIT) {
        fprintf(stderr, "Partition ID limit too small for %d partitions\n", num_partitions);
        return 1;
    }
    
    // 3. Build lookup tables
    init_partition_lookup_tables();
    build_partition_id_lookup();
    build_perm_table();
    build_terminal_perm_order_tables();
    build_overlap_table();
#if RECT_COUNT_K4
    build_partition_weight4_table();
#else
    build_partition_weight_table();
#endif
#if RECT_COUNT_K4_FEASIBILITY
    build_partition_shadow_table();
#endif
    
    printf("Grid: %dx%d\n", g_rows, g_cols);
    printf("Partitions: %d\n", num_partitions);
    printf("Threads: %d\n", omp_get_max_threads());
    if (reorder_partitions_flag) {
        printf("Partition hardness reorder: enabled\n");
    }
#if RECT_COUNT_K4
    printf("Mode: fixed 4-colour count\n");
#else
    printf("Mode: chromatic polynomial\n");
#endif
#if RECT_PROFILE
    printf("Profiling build: enabled\n");
#else
    printf("Profiling build: disabled\n");
#endif
    printf("Using nauty for canonical graph caching\n");

    // Build prefix list for work distribution.
    int prefix_depth = 0;
    if (prefix_depth_override != -1) {
        prefix_depth = prefix_depth_override;
    } else if (g_rows == 7 && g_cols >= 6) {
        prefix_depth = 2;
    } else if (g_cols >= 6) {
        prefix_depth = 3;
    } else if (g_cols >= 2) {
        prefix_depth = 2;
    }
    if (prefix_depth != 0 && prefix_depth != 2 && prefix_depth != 3 && prefix_depth != 4) {
        fprintf(stderr, "--prefix-depth must be 2, 3, or 4\n");
        return 1;
    }
    if (prefix_depth > g_cols) {
        fprintf(stderr, "--prefix-depth must not exceed cols\n");
        return 1;
    }
    if (g_cols >= 2 && prefix_depth == 0) {
        fprintf(stderr, "Internal error: invalid zero prefix depth for cols >= 2\n");
        return 1;
    }
    if (g_adaptive_subdivide && prefix_depth != 2) {
        fprintf(stderr, "Adaptive subdivision currently supports only --prefix-depth 2\n");
        return 1;
    }
    if (g_adaptive_subdivide && g_cols < 3) {
        fprintf(stderr, "--adaptive-subdivide requires cols >= 3\n");
        return 1;
    }
    if (g_adaptive_work_budget < 0) {
        fprintf(stderr, "--adaptive-work-budget must be non-negative\n");
        return 1;
    }
    if (g_adaptive_subdivide && g_adaptive_max_depth < 3) {
        fprintf(stderr, "--adaptive-max-depth must be at least 3 with --adaptive-subdivide\n");
        return 1;
    }
    if (g_adaptive_work_budget > 0 && !g_adaptive_subdivide) {
        fprintf(stderr, "--adaptive-work-budget requires --adaptive-subdivide\n");
        return 1;
    }
    {
        const char* queue_profile_step_env = getenv("RECT_QUEUE_PROFILE_STEP");
        if (queue_profile_step_env && *queue_profile_step_env) {
            g_queue_profile_report_step = strtod(queue_profile_step_env, NULL);
            if (g_queue_profile_report_step < 0.0) g_queue_profile_report_step = 0.0;
        }
        const char* shared_cache_env = getenv("RECT_SHARED_CACHE_MERGE");
        if (shared_cache_env && *shared_cache_env && strcmp(shared_cache_env, "0") != 0) {
            g_shared_cache_merge = 1;
        }
        const char* shared_cache_bits_env = getenv("RECT_SHARED_CACHE_BITS");
        if (shared_cache_bits_env && *shared_cache_bits_env) {
            g_shared_cache_bits = (int)parse_ll_or_die(shared_cache_bits_env, "RECT_SHARED_CACHE_BITS");
            if (g_shared_cache_bits < 10 || g_shared_cache_bits > 24) {
                fprintf(stderr, "RECT_SHARED_CACHE_BITS must be between 10 and 24\n");
                return 1;
            }
        }
#if RECT_PROFILE
        const char* profile_separators_env = getenv("RECT_PROFILE_SEPARATORS");
        if (profile_separators_env && *profile_separators_env &&
            strcmp(profile_separators_env, "0") != 0) {
            g_profile_separators = 1;
        }
#endif
        const char* raw_cache_env = getenv("RECT_USE_RAW_CACHE");
        if (raw_cache_env && *raw_cache_env) {
            g_use_raw_cache = (strcmp(raw_cache_env, "0") != 0);
        }
    }
    if (task_start < 0) {
        fprintf(stderr, "--task-start must be non-negative\n");
        return 1;
    }
    if (task_end >= 0 && task_end < task_start) {
        fprintf(stderr, "Task range must satisfy 0 <= start <= end\n");
        return 1;
    }
    printf("Raw cache: %s\n", g_use_raw_cache ? "enabled" : "disabled");
    g_effective_prefix_depth = prefix_depth;

    int graph_poly_len = RECT_COUNT_K4 ? 1 : (g_cols * (g_rows / 2) + 1);
    SharedGraphCache shared_graph_cache;
    int shared_graph_cache_active = 0;
    if (g_shared_cache_merge) {
        shared_graph_cache_init(&shared_graph_cache, g_shared_cache_bits, graph_poly_len);
        g_shared_graph_cache = &shared_graph_cache;
        shared_graph_cache_active = 1;
        printf("Shared canonical cache merge enabled: 2^%d slots\n", g_shared_cache_bits);
    }

    long long total_prefixes = 0;
    long long nominal_prefixes = 0;
    Prefix2Batch* prefix2_batches = NULL;
    PrefixId* prefix2_batch_js = NULL;
    long long* prefix2_batch_ps = NULL;
    long long prefix2_batch_count = 0;
    double prefix_generation_time = 0.0;
    int use_runtime_split_queue = (prefix_depth == 2 && g_adaptive_subdivide);

    if (prefix_depth > 0) {
        double prefix_start_time = omp_get_wtime();
        if (prefix_depth == 2) {
            long long base_prefixes = (long long)num_partitions * (num_partitions + 1) / 2;
            nominal_prefixes = base_prefixes;
            build_live_prefix2_tasks(&g_live_prefix2_i, &g_live_prefix2_j, &g_live_prefix2_count);
            total_prefixes = g_live_prefix2_count;
        } else if (prefix_depth == 3) {
            total_prefixes = (long long)num_partitions * (num_partitions + 1) * (num_partitions + 2) / 6;
        } else if (prefix_depth == 4) {
            total_prefixes = (long long)num_partitions * (num_partitions + 1) *
                             (num_partitions + 2) * (num_partitions + 3) / 24;
        }
        prefix_generation_time = omp_get_wtime() - prefix_start_time;
    }

    completed_tasks = 0;
    long long full_tasks = (g_cols == 1) ? (long long)num_partitions : total_prefixes;
    if (task_end < 0) task_end = full_tasks;
    if (task_end < task_start || task_end > full_tasks) {
        fprintf(stderr, "Task range must satisfy 0 <= start <= end <= %lld\n", full_tasks);
        return 1;
    }
    long long active_task_start = task_start;
    long long active_task_end = task_end;
    long long total_tasks = active_task_end - active_task_start;
    long long first_task = active_task_start;
    g_task_times_first_task = first_task;
    g_task_times_count = total_tasks;
    if (g_task_times_out_path && total_tasks > 0) {
        g_task_times_values = checked_calloc((size_t)total_tasks, sizeof(*g_task_times_values),
                                             "task_times_values");
        for (long long t = 0; t < total_tasks; t++) {
            g_task_times_values[t] = -1.0;
        }
    }
    if (prefix_depth == 2 && !g_adaptive_subdivide && total_tasks > 0) {
        double batch_start_time = omp_get_wtime();
        build_fixed_prefix2_batches(g_live_prefix2_i, g_live_prefix2_j,
                                    active_task_start,
                                    total_tasks, &prefix2_batches, &prefix2_batch_count,
                                    &prefix2_batch_js, &prefix2_batch_ps);
        prefix_generation_time += omp_get_wtime() - batch_start_time;
    }
    printf("Prefix depth: %d (%lld tasks)\n", prefix_depth, total_prefixes);
    if (prefix_depth == 2 && nominal_prefixes > 0) {
        printf("Live depth-2 prefixes: %lld of %lld nominal\n", total_prefixes, nominal_prefixes);
    }
    if (g_adaptive_subdivide) {
        if (use_runtime_split_queue) {
            printf("Runtime subdivision enabled: max depth %d",
                   g_adaptive_max_depth);
            if (g_adaptive_work_budget > 0) {
                printf(", work budget %lld", g_adaptive_work_budget);
            }
            printf("\n");
        } else {
            printf("Adaptive subdivision: max depth %d\n",
                   g_adaptive_max_depth);
        }
    }
    printf("Prefix generation: %.2f seconds\n", prefix_generation_time);
    if (prefix_depth > 0) {
        if (prefix_depth == 2 && prefix2_batch_count > 0) {
            size_t bytes_per_task = sizeof(PrefixId) + sizeof(long long);
            size_t bytes_per_batch = sizeof(*prefix2_batches);
            double total_mib =
                (((double)bytes_per_task * (double)total_tasks) +
                 ((double)bytes_per_batch * (double)prefix2_batch_count)) /
                (1024.0 * 1024.0);
            printf("Fixed depth-2 batching: %lld batches, %.2f MiB total\n",
                   prefix2_batch_count, total_mib);
        } else {
            printf("Prefix task storage: unranked on demand for selected tasks\n");
        }
    }
    long long progress_report_step = 0;
    long long progress_updates = DEFAULT_PROGRESS_UPDATES;
    const char* progress_step_env = getenv("RECT_PROGRESS_STEP");
    const char* progress_updates_env = getenv("RECT_PROGRESS_UPDATES");
    if (progress_step_env && *progress_step_env) {
        char* end = NULL;
        long long parsed = strtoll(progress_step_env, &end, 10);
        if (end && *end == '\0' && parsed > 0) {
            progress_report_step = parsed;
        }
    }
    if (progress_report_step == 0 && progress_updates_env && *progress_updates_env) {
        char* end = NULL;
        long long parsed = strtoll(progress_updates_env, &end, 10);
        if (end && *end == '\0' && parsed > 0) {
            progress_updates = parsed;
        }
    }
    if (progress_report_step == 0) {
        progress_report_step = total_tasks / progress_updates;
        if (progress_report_step < 1) progress_report_step = 1;
    }
    if (total_tasks > 0 && progress_report_step > total_tasks) progress_report_step = total_tasks;
    printf("Task range: [%lld, %lld) of %lld\n", active_task_start, active_task_end, full_tasks);
    const char* omp_static_env = getenv("RECT_OMP_STATIC");
    int use_static_schedule =
        (omp_static_env && *omp_static_env && strcmp(omp_static_env, "0") != 0);
    const char* omp_schedule_env = getenv("OMP_SCHEDULE");
    int omp_chunk = 1;
    if (!use_static_schedule) {
        omp_chunk = (prefix_depth == 2 && g_rows < 7 && !g_adaptive_subdivide) ? 8 : 1;
    }
    if (use_static_schedule) {
        printf("OpenMP scheduling: static,1 (RECT_OMP_STATIC override)\n");
        omp_set_schedule(omp_sched_static, 1);
    } else if (omp_schedule_env && *omp_schedule_env) {
        printf("OpenMP scheduling: runtime from OMP_SCHEDULE=%s\n", omp_schedule_env);
    } else {
        printf("OpenMP scheduling: dynamic,%d\n", omp_chunk);
        omp_set_schedule(omp_sched_dynamic, omp_chunk);
    }
    if (total_tasks == 0) {
        printf("No tasks selected; producing the zero polynomial for this shard.\n");
    }
    printf("Progress updates every %lld tasks", progress_report_step);
    if (progress_step_env && *progress_step_env) {
        printf(" (RECT_PROGRESS_STEP override)");
    } else {
        printf(" (target ~%lld updates)", progress_updates);
    }
    printf("\n");
    progress_last_reported = 0;
    progress_reporter_init(&progress_reporter, stdout);
    progress_reporter_print_initial(&progress_reporter, total_tasks);

    double start_time = omp_get_wtime();
    if (total_tasks > 0) {
        small_graph_lookup_init();
        connected_canon_lookup_init();
        if (PROFILE_BUILD) {
            printf("Small-graph lookup %s: %.2f seconds\n",
                   g_small_graph_lookup_loaded_from_file ? "load" : "initialisation",
                   g_small_graph_lookup_init_time);
            if (g_connected_canon_lookup_loaded) {
                printf("Connected canonical lookup n=%d load: %.2f seconds\n",
                       g_connected_canon_lookup_n, g_connected_canon_lookup_load_time);
            }
        }
    }
    
    int num_threads = omp_get_max_threads();
    LocalTaskQueue local_queue;
    int local_queue_active = 0;
    if (use_runtime_split_queue) {
        int queue_capacity_slack = 4 * num_threads;

        long long queue_cap_ll = total_tasks + queue_capacity_slack + 64;
        if (queue_cap_ll > INT_MAX) {
            fprintf(stderr, "Local task queue too large\n");
            return 1;
        }

        local_queue_init(&local_queue, (int)queue_cap_ll, total_tasks, num_threads);
        for (long long t = 0; t < total_tasks; t++) {
            long long p = first_task + t;
            int i = 0;
            int j = 0;
            LocalTask task;
            memset(&task, 0, sizeof(task));
            get_prefix2_task(p, &i, &j);
            task.depth = 2;
            task.root_id = t;
            task.prefix[0] = (PrefixId)i;
            task.prefix[1] = (PrefixId)j;
            task.lo = (PrefixId)j;
            task.hi = (PrefixId)num_partitions;
            local_queue.roots[t].pending = 0;
            local_queue.roots[t].task_index = p;
            local_queue_seed_push(&local_queue, &task);
        }
        local_queue_active = 1;
        printf("Runtime queue: capacity=%d, split-max-depth=%d",
               (int)queue_cap_ll, g_adaptive_max_depth);
        if (g_adaptive_work_budget > 0) {
            printf(", work budget %lld", g_adaptive_work_budget);
        }
        printf("\n");
    }

    Poly* thread_polys = checked_aligned_alloc(64, (size_t)num_threads * sizeof(Poly), "thread_polys");
    ProfileStats* thread_profiles =
        checked_aligned_alloc(64, (size_t)num_threads * sizeof(ProfileStats), "thread_profiles");
    TaskTimingStats* thread_task_timing =
        checked_aligned_alloc(64, (size_t)num_threads * sizeof(TaskTimingStats), "thread_task_timing");
    QueueSubtaskTimingStats* thread_queue_subtask_timing = NULL;
    for(int i=0; i<num_threads; i++) poly_zero(&thread_polys[i]);
    memset(thread_profiles, 0, (size_t)num_threads * sizeof(ProfileStats));
    memset(thread_task_timing, 0, (size_t)num_threads * sizeof(TaskTimingStats));
    if (use_runtime_split_queue && PROFILE_BUILD) {
        thread_queue_subtask_timing = checked_aligned_alloc(
            64, (size_t)num_threads * (size_t)(MAX_COLS + 1) * sizeof(QueueSubtaskTimingStats),
            "thread_queue_subtask_timing");
        memset(thread_queue_subtask_timing, 0,
               (size_t)num_threads * (size_t)(MAX_COLS + 1) * sizeof(QueueSubtaskTimingStats));
    }

    long long total_canon_calls = 0;
    long long total_cache_hits = 0;
    long long total_raw_cache_hits = 0;

    #pragma omp parallel reduction(+:total_canon_calls, total_cache_hits, total_raw_cache_hits)
    {
        int tid = omp_get_thread_num();
        RowGraphCache cache = {0};
        RowGraphCache raw_cache = {0};
        NautyWorkspace ws;
        memset(&ws, 0, sizeof(ws));
        cache.mask = CACHE_MASK;
        cache.probe = CACHE_PROBE;
        cache.poly_len = graph_poly_len;
        cache.keys = checked_aligned_alloc(64, sizeof(CacheKey) * CACHE_SIZE, "cache_keys");
        cache.stamps = checked_aligned_alloc(64, sizeof(uint32_t) * CACHE_SIZE, "cache_stamps");
        cache.rows = checked_aligned_alloc(64, sizeof(AdjWord) * CACHE_SIZE * MAXN_NAUTY, "cache_rows");
        cache.degs = checked_aligned_alloc(64, sizeof(uint8_t) * CACHE_SIZE, "cache_degs");
        cache.coeffs =
            checked_aligned_alloc(64, sizeof(PolyCoeff) * CACHE_SIZE * (size_t)graph_poly_len, "cache_coeffs");
        cache.next_stamp = 0;

        raw_cache.mask = RAW_CACHE_MASK;
        raw_cache.probe = RAW_CACHE_PROBE;
        raw_cache.poly_len = graph_poly_len;
        raw_cache.keys = checked_aligned_alloc(64, sizeof(CacheKey) * RAW_CACHE_SIZE, "raw_cache_keys");
        raw_cache.stamps = checked_aligned_alloc(64, sizeof(uint32_t) * RAW_CACHE_SIZE, "raw_cache_stamps");
        raw_cache.rows =
            checked_aligned_alloc(64, sizeof(AdjWord) * RAW_CACHE_SIZE * MAXN_NAUTY, "raw_cache_rows");
        raw_cache.degs = checked_aligned_alloc(64, sizeof(uint8_t) * RAW_CACHE_SIZE, "raw_cache_degs");
        raw_cache.coeffs = checked_aligned_alloc(64, sizeof(PolyCoeff) * RAW_CACHE_SIZE * (size_t)graph_poly_len,
                                                 "raw_cache_coeffs");
        raw_cache.next_stamp = 0;

        memset(cache.keys, 0, sizeof(CacheKey) * CACHE_SIZE);
        memset(cache.stamps, 0, sizeof(uint32_t) * CACHE_SIZE);
        memset(raw_cache.keys, 0, sizeof(CacheKey) * RAW_CACHE_SIZE);
        memset(raw_cache.stamps, 0, sizeof(uint32_t) * RAW_CACHE_SIZE);

        int stack[MAX_COLS];
        CanonState canon_state;
        CanonScratch canon_scratch;
        PartialGraphState partial_graph;
        canon_state_init(&canon_state, perm_count);
        canon_scratch_init(&canon_scratch, perm_count);
        canon_state_reset(&canon_state, perm_count);
        partial_graph_reset(&partial_graph);
        long long local_canon_calls = 0;
        long long local_cache_hits = 0;
        long long local_raw_cache_hits = 0;
        ProfileStats* profile = &thread_profiles[tid];
        TaskTimingStats* task_timing = &thread_task_timing[tid];
        QueueSubtaskTimingStats* queue_subtask_timing =
            thread_queue_subtask_timing ? thread_queue_subtask_timing + (size_t)tid * (size_t)(MAX_COLS + 1) : NULL;
        SharedGraphCacheExporter shared_cache_exporter = {0};
        long long pending_completed = 0;
        tls_profile = profile;
        tls_shared_cache_exporter = g_shared_cache_merge ? &shared_cache_exporter : NULL;
        
        if (g_cols == 1) {
            // Original 1-column parallelism (nothing to prefix)
            #pragma omp for schedule(runtime)
            for (long long i = first_task; i < active_task_end; i++) {
                double task_t0 = PROFILE_BUILD ? omp_get_wtime() : 0.0;
                double t0 = 0.0;
                stack[0] = i;
                canon_state_reset(&canon_state, perm_count);
                partial_graph_reset(&partial_graph);
                int next_stabilizer = 0;
                if (PROFILE_BUILD) {
                    profile->canon_prepare_calls++;
                    profile->canon_prepare_calls_by_depth[0]++;
                    t0 = omp_get_wtime();
                }
                if (!canon_state_prepare_push(&canon_state, (int)i, &canon_scratch, &next_stabilizer)) {
                    if (PROFILE_BUILD) profile->canon_prepare_time += omp_get_wtime() - t0;
                    complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                  &pending_completed, task_timing, i, task_t0);
                    continue;
                }
                if (PROFILE_BUILD) {
                    profile->canon_prepare_time += omp_get_wtime() - t0;
                    profile->canon_prepare_accepts++;
                    profile->canon_prepare_accepts_by_depth[0]++;
                    profile->stabilizer_sum_by_depth[0] += next_stabilizer;
                    profile->canon_commit_calls++;
                    t0 = omp_get_wtime();
                }
                canon_state_commit_push(&canon_state, (int)i, &canon_scratch, next_stabilizer);
                if (PROFILE_BUILD) profile->canon_commit_time += omp_get_wtime() - t0;
                if (PROFILE_BUILD) {
                    profile->partial_append_calls++;
                    t0 = omp_get_wtime();
                }
                int ok = partial_graph_append_checked(&partial_graph, 0, (int)i, stack, g_cols - 1);
                if (PROFILE_BUILD) profile->partial_append_time += omp_get_wtime() - t0;
                if (ok) {
                    WeightAccum initial_weight;
                    weight_accum_from_partition((int)i, &initial_weight);
                    dfs(1, (int)i, stack, &canon_state, &partial_graph, &cache, &raw_cache, &ws,
                        &thread_polys[tid], &local_canon_calls, &local_cache_hits, &local_raw_cache_hits,
                        &initial_weight, 1, 1, profile, &canon_scratch);
                }
                complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                              &pending_completed, task_timing, i, task_t0);
            }
        } else if (prefix_depth == 2) {
            if (!g_adaptive_subdivide && prefix2_batch_count > 0) {
                #pragma omp for schedule(runtime)
                for (long long b = 0; b < prefix2_batch_count; b++) {
                    Prefix2Batch batch = prefix2_batches[b];
                    execute_prefix2_fixed_batch(batch.i,
                                                prefix2_batch_js + batch.start,
                                                prefix2_batch_ps + batch.start,
                                                batch.count,
                                                &cache, &raw_cache, &ws, &canon_state,
                                                &canon_scratch, &partial_graph, stack, &thread_polys[tid],
                                                &local_canon_calls, &local_cache_hits,
                                                &local_raw_cache_hits, profile,
                                                total_tasks, progress_report_step, start_time,
                                                &pending_completed, task_timing);
                }
            } else if (use_runtime_split_queue) {
                WorkerCtx ctx = {0};
                ctx.cache = cache;
                ctx.raw_cache = raw_cache;
                ctx.ws = ws;
                ctx.canon_state = canon_state;
                ctx.canon_scratch = canon_scratch;
                ctx.partial_graph = partial_graph;
                ctx.local_canon_calls = 0;
                ctx.local_cache_hits = 0;
                ctx.local_raw_cache_hits = 0;

                for (;;) {
                    LocalTask task;
                    if (!local_queue_pop(&local_queue, &task)) break;
                    execute_local_runtime_task(&task, &ctx, &thread_polys[tid], &local_queue,
                                               profile, total_tasks, progress_report_step,
                                               start_time, &pending_completed, task_timing,
                                               queue_subtask_timing);
                }

                ws = ctx.ws;
                memset(&ctx.ws, 0, sizeof(ctx.ws));
                local_canon_calls += ctx.local_canon_calls;
                local_cache_hits += ctx.local_cache_hits;
                local_raw_cache_hits += ctx.local_raw_cache_hits;
            } else {
                #pragma omp for schedule(runtime)
                for (long long t = 0; t < total_tasks; t++) {
                    double task_t0 = PROFILE_BUILD ? omp_get_wtime() : 0.0;
                    double t0 = 0.0;
                    long long p = first_task + t;
                    int i = 0;
                    int j = 0;
                    get_prefix2_task(p, &i, &j);
                    int next_stabilizer = 0;

                    canon_state_reset(&canon_state, perm_count);
                    partial_graph_reset(&partial_graph);

                    stack[0] = i;
                    if (PROFILE_BUILD) {
                        profile->canon_prepare_calls++;
                        profile->canon_prepare_calls_by_depth[0]++;
                        t0 = omp_get_wtime();
                    }
                    if (!canon_state_prepare_push(&canon_state, i, &canon_scratch, &next_stabilizer)) {
                        if (PROFILE_BUILD) profile->canon_prepare_time += omp_get_wtime() - t0;
                        complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                      &pending_completed, task_timing, p, task_t0);
                        continue;
                    }
                    if (PROFILE_BUILD) {
                        profile->canon_prepare_time += omp_get_wtime() - t0;
                        profile->canon_prepare_accepts++;
                        profile->canon_prepare_accepts_by_depth[0]++;
                        profile->stabilizer_sum_by_depth[0] += next_stabilizer;
                        profile->canon_commit_calls++;
                        t0 = omp_get_wtime();
                    }
                    canon_state_commit_push(&canon_state, i, &canon_scratch, next_stabilizer);
                    if (PROFILE_BUILD) profile->canon_commit_time += omp_get_wtime() - t0;
                    if (PROFILE_BUILD) {
                        profile->partial_append_calls++;
                        t0 = omp_get_wtime();
                    }
                    int ok = partial_graph_append_checked(&partial_graph, 0, i, stack, g_cols - 1);
                    if (PROFILE_BUILD) profile->partial_append_time += omp_get_wtime() - t0;
                    if (!ok) {
                        canon_state_pop(&canon_state);
                        complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                      &pending_completed, task_timing, p, task_t0);
                        continue;
                    }

                    stack[1] = j;
                    if (PROFILE_BUILD) {
                        profile->canon_prepare_calls++;
                        profile->canon_prepare_calls_by_depth[1]++;
                        t0 = omp_get_wtime();
                    }
                    if (!canon_state_prepare_push(&canon_state, j, &canon_scratch, &next_stabilizer)) {
                        if (PROFILE_BUILD) profile->canon_prepare_time += omp_get_wtime() - t0;
                        canon_state_pop(&canon_state);
                        complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                      &pending_completed, task_timing, p, task_t0);
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
                    canon_state_commit_push(&canon_state, j, &canon_scratch, next_stabilizer);
                    if (PROFILE_BUILD) profile->canon_commit_time += omp_get_wtime() - t0;
                    PartialGraphState prefix_graph = partial_graph;
                    if (PROFILE_BUILD) {
                        profile->partial_append_calls++;
                        t0 = omp_get_wtime();
                    }
                    ok = partial_graph_append_checked(&prefix_graph, 1, j, stack, g_cols - 2);
                    if (PROFILE_BUILD) profile->partial_append_time += omp_get_wtime() - t0;
                    if (ok) {
                        WeightAccum prefix_weight;
                        weight_accum_from_partition(i, &prefix_weight);
                        weight_accum_mul_partition(&prefix_weight, j, &prefix_weight);
                        long long prefix_mult = (i == j) ? 1 : 2;
                        int prefix_run = (i == j) ? 2 : 1;
                        dfs(2, j, stack, &canon_state, &prefix_graph, &cache, &raw_cache, &ws,
                            &thread_polys[tid], &local_canon_calls, &local_cache_hits, &local_raw_cache_hits,
                            &prefix_weight, prefix_mult, prefix_run, profile, &canon_scratch);
                    }

                    canon_state_pop(&canon_state);
                    canon_state_pop(&canon_state);

                    complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                  &pending_completed, task_timing, p, task_t0);
                }
            }
        } else if (prefix_depth == 3) {
            #pragma omp for schedule(runtime)
            for (long long t = 0; t < total_tasks; t++) {
                double task_t0 = PROFILE_BUILD ? omp_get_wtime() : 0.0;
                long long p = first_task + t;
                int i = 0;
                int j = 0;
                int k = 0;
                unrank_prefix3(p, &i, &j, &k);
                int next_stabilizer = 0;

                canon_state_reset(&canon_state, perm_count);
                partial_graph_reset(&partial_graph);

                stack[0] = i;
                if (!canon_state_prepare_push(&canon_state, i, &canon_scratch, &next_stabilizer)) {
                    complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                  &pending_completed, task_timing, p, task_t0);
                    continue;
                }
                canon_state_commit_push(&canon_state, i, &canon_scratch, next_stabilizer);
                if (!partial_graph_append_checked(&partial_graph, 0, i, stack, g_cols - 1)) {
                    canon_state_pop(&canon_state);
                    complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                  &pending_completed, task_timing, p, task_t0);
                    continue;
                }

                stack[1] = j;
                if (!canon_state_prepare_push(&canon_state, j, &canon_scratch, &next_stabilizer)) {
                    canon_state_pop(&canon_state);
                    complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                  &pending_completed, task_timing, p, task_t0);
                    continue;
                }
                canon_state_commit_push(&canon_state, j, &canon_scratch, next_stabilizer);
                PartialGraphState prefix_graph = partial_graph;
                if (!partial_graph_append_checked(&prefix_graph, 1, j, stack, g_cols - 2)) {
                    canon_state_pop(&canon_state);
                    canon_state_pop(&canon_state);
                    complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                  &pending_completed, task_timing, p, task_t0);
                    continue;
                }

                stack[2] = k;
                if (!canon_state_prepare_push(&canon_state, k, &canon_scratch, &next_stabilizer)) {
                    canon_state_pop(&canon_state);
                    canon_state_pop(&canon_state);
                    complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                  &pending_completed, task_timing, p, task_t0);
                    continue;
                }
                canon_state_commit_push(&canon_state, k, &canon_scratch, next_stabilizer);
                PartialGraphState prefix_graph2 = prefix_graph;
                int ok = partial_graph_append_checked(&prefix_graph2, 2, k, stack, g_cols - 3);
                if (ok) {
                    WeightAccum prefix_weight;
                    weight_accum_from_partition(i, &prefix_weight);
                    weight_accum_mul_partition(&prefix_weight, j, &prefix_weight);
                    weight_accum_mul_partition(&prefix_weight, k, &prefix_weight);
                    long long prefix_mult = (i == j) ? 1 : 2;
                    int prefix_run = (i == j) ? 2 : 1;
                    if (k == j) {
                        prefix_mult = prefix_mult * 3 / (prefix_run + 1);
                        prefix_run += 1;
                    } else {
                        prefix_mult *= 3;
                        prefix_run = 1;
                    }
                    dfs(3, k, stack, &canon_state, &prefix_graph2, &cache, &raw_cache, &ws,
                        &thread_polys[tid], &local_canon_calls, &local_cache_hits, &local_raw_cache_hits,
                        &prefix_weight, prefix_mult, prefix_run, profile, &canon_scratch);
                }

                canon_state_pop(&canon_state);
                canon_state_pop(&canon_state);
                canon_state_pop(&canon_state);

                complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                              &pending_completed, task_timing, p, task_t0);
            }
        } else {
            #pragma omp for schedule(runtime)
            for (long long t = 0; t < total_tasks; t++) {
                double task_t0 = PROFILE_BUILD ? omp_get_wtime() : 0.0;
                long long p = first_task + t;
                int i = 0;
                int j = 0;
                int k = 0;
                int l = 0;
                unrank_prefix4(p, &i, &j, &k, &l);
                int next_stabilizer = 0;

                canon_state_reset(&canon_state, perm_count);
                partial_graph_reset(&partial_graph);

                stack[0] = i;
                if (!canon_state_prepare_push(&canon_state, i, &canon_scratch, &next_stabilizer)) {
                    complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                  &pending_completed, task_timing, p, task_t0);
                    continue;
                }
                canon_state_commit_push(&canon_state, i, &canon_scratch, next_stabilizer);
                if (!partial_graph_append_checked(&partial_graph, 0, i, stack, g_cols - 1)) {
                    canon_state_pop(&canon_state);
                    complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                  &pending_completed, task_timing, p, task_t0);
                    continue;
                }

                stack[1] = j;
                if (!canon_state_prepare_push(&canon_state, j, &canon_scratch, &next_stabilizer)) {
                    canon_state_pop(&canon_state);
                    complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                  &pending_completed, task_timing, p, task_t0);
                    continue;
                }
                canon_state_commit_push(&canon_state, j, &canon_scratch, next_stabilizer);
                PartialGraphState prefix_graph = partial_graph;
                if (!partial_graph_append_checked(&prefix_graph, 1, j, stack, g_cols - 2)) {
                    canon_state_pop(&canon_state);
                    canon_state_pop(&canon_state);
                    complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                  &pending_completed, task_timing, p, task_t0);
                    continue;
                }

                stack[2] = k;
                if (!canon_state_prepare_push(&canon_state, k, &canon_scratch, &next_stabilizer)) {
                    canon_state_pop(&canon_state);
                    canon_state_pop(&canon_state);
                    complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                  &pending_completed, task_timing, p, task_t0);
                    continue;
                }
                canon_state_commit_push(&canon_state, k, &canon_scratch, next_stabilizer);
                PartialGraphState prefix_graph2 = prefix_graph;
                if (!partial_graph_append_checked(&prefix_graph2, 2, k, stack, g_cols - 3)) {
                    canon_state_pop(&canon_state);
                    canon_state_pop(&canon_state);
                    canon_state_pop(&canon_state);
                    complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                  &pending_completed, task_timing, p, task_t0);
                    continue;
                }

                stack[3] = l;
                if (!canon_state_prepare_push(&canon_state, l, &canon_scratch, &next_stabilizer)) {
                    canon_state_pop(&canon_state);
                    canon_state_pop(&canon_state);
                    canon_state_pop(&canon_state);
                    complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                  &pending_completed, task_timing, p, task_t0);
                    continue;
                }
                canon_state_commit_push(&canon_state, l, &canon_scratch, next_stabilizer);
                PartialGraphState prefix_graph3 = prefix_graph2;
                int ok = partial_graph_append_checked(&prefix_graph3, 3, l, stack, g_cols - 4);
                if (ok) {
                    WeightAccum prefix_weight;
                    weight_accum_from_partition(i, &prefix_weight);
                    weight_accum_mul_partition(&prefix_weight, j, &prefix_weight);
                    weight_accum_mul_partition(&prefix_weight, k, &prefix_weight);
                    weight_accum_mul_partition(&prefix_weight, l, &prefix_weight);
                    long long prefix_mult = (i == j) ? 1 : 2;
                    int prefix_run = (i == j) ? 2 : 1;
                    if (k == j) {
                        prefix_mult = prefix_mult * 3 / (prefix_run + 1);
                        prefix_run += 1;
                    } else {
                        prefix_mult *= 3;
                        prefix_run = 1;
                    }
                    if (l == k) {
                        prefix_mult = prefix_mult * 4 / (prefix_run + 1);
                        prefix_run += 1;
                    } else {
                        prefix_mult *= 4;
                        prefix_run = 1;
                    }
                    dfs(4, l, stack, &canon_state, &prefix_graph3, &cache, &raw_cache, &ws,
                        &thread_polys[tid], &local_canon_calls, &local_cache_hits, &local_raw_cache_hits,
                        &prefix_weight, prefix_mult, prefix_run, profile, &canon_scratch);
                }

                canon_state_pop(&canon_state);
                canon_state_pop(&canon_state);
                canon_state_pop(&canon_state);
                canon_state_pop(&canon_state);

                complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                              &pending_completed, task_timing, p, task_t0);
            }
        }

        flush_completed_tasks(total_tasks, progress_report_step, start_time, &pending_completed);
        shared_graph_cache_flush_exports();
        tls_profile = NULL;
        tls_shared_cache_exporter = NULL;

        total_canon_calls += local_canon_calls;
        total_cache_hits += local_cache_hits;
        total_raw_cache_hits += local_raw_cache_hits;
        
        canon_state_free(&canon_state);
        canon_scratch_free(&canon_scratch);
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
    }

    if (local_queue_active) {
        local_queue_print_occupancy_summary(&local_queue);
        local_queue_free(&local_queue);
    }
    
    for(int i=0; i<num_threads; i++) {
        poly_accumulate_checked(&global_poly, &thread_polys[i]);
    }

    ProfileStats total_profile = {0};
    TaskTimingStats total_task_timing = {0};
    QueueSubtaskTimingStats total_queue_subtask_timing[MAX_COLS + 1];
    memset(total_queue_subtask_timing, 0, sizeof(total_queue_subtask_timing));
    for (int i = 0; i < num_threads; i++) {
        ProfileStats* src = &thread_profiles[i];
        total_profile.canon_prepare_calls += src->canon_prepare_calls;
        total_profile.canon_prepare_accepts += src->canon_prepare_accepts;
        total_profile.canon_commit_calls += src->canon_commit_calls;
        total_profile.partial_append_calls += src->partial_append_calls;
        total_profile.solve_structure_calls += src->solve_structure_calls;
        total_profile.solve_graph_calls += src->solve_graph_calls;
        total_profile.nauty_calls += src->nauty_calls;
        total_profile.hard_graph_nodes += src->hard_graph_nodes;
        total_profile.canon_prepare_time += src->canon_prepare_time;
        total_profile.canon_commit_time += src->canon_commit_time;
        total_profile.partial_append_time += src->partial_append_time;
        total_profile.build_weight_time += src->build_weight_time;
        total_profile.solve_graph_time += src->solve_graph_time;
        total_profile.get_canonical_graph_time += src->get_canonical_graph_time;
        total_profile.get_canonical_graph_dense_rows_time += src->get_canonical_graph_dense_rows_time;
        total_profile.get_canonical_graph_build_input_time += src->get_canonical_graph_build_input_time;
        total_profile.nauty_time += src->nauty_time;
        total_profile.get_canonical_graph_rebuild_time += src->get_canonical_graph_rebuild_time;
        if (src->hard_graph_max_n > total_profile.hard_graph_max_n) {
            total_profile.hard_graph_max_n = src->hard_graph_max_n;
        }
        if (src->hard_graph_max_degree > total_profile.hard_graph_max_degree) {
            total_profile.hard_graph_max_degree = src->hard_graph_max_degree;
        }
        for (int d = 0; d <= MAX_COLS; d++) {
            total_profile.canon_prepare_calls_by_depth[d] += src->canon_prepare_calls_by_depth[d];
            total_profile.canon_prepare_accepts_by_depth[d] += src->canon_prepare_accepts_by_depth[d];
            total_profile.stabilizer_sum_by_depth[d] += src->stabilizer_sum_by_depth[d];
            total_profile.canon_prepare_scanned_by_depth[d] += src->canon_prepare_scanned_by_depth[d];
            total_profile.canon_prepare_active_by_depth[d] += src->canon_prepare_active_by_depth[d];
            total_profile.canon_prepare_terminal_calls_by_depth[d] +=
                src->canon_prepare_terminal_calls_by_depth[d];
            total_profile.canon_prepare_fast_continue_by_depth[d] +=
                src->canon_prepare_fast_continue_by_depth[d];
            total_profile.canon_prepare_terminal_continue_by_depth[d] +=
                src->canon_prepare_terminal_continue_by_depth[d];
            total_profile.canon_prepare_equal_case_calls_by_depth[d] +=
                src->canon_prepare_equal_case_calls_by_depth[d];
            total_profile.canon_prepare_equal_case_rejects_by_depth[d] +=
                src->canon_prepare_equal_case_rejects_by_depth[d];
            total_profile.canon_prepare_order_rejects_by_depth[d] +=
                src->canon_prepare_order_rejects_by_depth[d];
        }
        for (int n = 0; n <= MAXN_NAUTY; n++) {
            total_profile.solve_graph_calls_by_n[n] += src->solve_graph_calls_by_n[n];
            total_profile.solve_graph_raw_hits_by_n[n] += src->solve_graph_raw_hits_by_n[n];
            total_profile.solve_graph_canon_hits_by_n[n] += src->solve_graph_canon_hits_by_n[n];
            total_profile.hard_graph_nodes_by_n[n] += src->hard_graph_nodes_by_n[n];
            total_profile.solve_graph_lookup_calls_by_n[n] += src->solve_graph_lookup_calls_by_n[n];
            total_profile.solve_graph_connected_lookup_calls_by_n[n] +=
                src->solve_graph_connected_lookup_calls_by_n[n];
            total_profile.solve_graph_component_calls_by_n[n] += src->solve_graph_component_calls_by_n[n];
            total_profile.solve_graph_hard_misses_by_n[n] += src->solve_graph_hard_misses_by_n[n];
            total_profile.hard_graph_articulation_by_n[n] += src->hard_graph_articulation_by_n[n];
            total_profile.hard_graph_k2_separator_by_n[n] += src->hard_graph_k2_separator_by_n[n];
            total_profile.solve_graph_time_by_n[n] += src->solve_graph_time_by_n[n];
            total_profile.solve_graph_lookup_time_by_n[n] += src->solve_graph_lookup_time_by_n[n];
            total_profile.solve_graph_connected_lookup_time_by_n[n] +=
                src->solve_graph_connected_lookup_time_by_n[n];
            total_profile.solve_graph_raw_hit_time_by_n[n] += src->solve_graph_raw_hit_time_by_n[n];
            total_profile.solve_graph_canon_hit_time_by_n[n] += src->solve_graph_canon_hit_time_by_n[n];
            total_profile.solve_graph_component_time_by_n[n] += src->solve_graph_component_time_by_n[n];
            total_profile.solve_graph_hard_miss_time_by_n[n] += src->solve_graph_hard_miss_time_by_n[n];
            total_profile.solve_graph_hard_miss_separator_time_by_n[n] +=
                src->solve_graph_hard_miss_separator_time_by_n[n];
            total_profile.solve_graph_hard_miss_pick_time_by_n[n] +=
                src->solve_graph_hard_miss_pick_time_by_n[n];
            total_profile.solve_graph_hard_miss_delete_time_by_n[n] +=
                src->solve_graph_hard_miss_delete_time_by_n[n];
            total_profile.solve_graph_hard_miss_contract_build_time_by_n[n] +=
                src->solve_graph_hard_miss_contract_build_time_by_n[n];
            total_profile.solve_graph_hard_miss_contract_solve_time_by_n[n] +=
                src->solve_graph_hard_miss_contract_solve_time_by_n[n];
            total_profile.solve_graph_hard_miss_store_time_by_n[n] +=
                src->solve_graph_hard_miss_store_time_by_n[n];
            for (int d = 0; d <= MAXN_NAUTY; d++) {
                total_profile.hard_graph_nodes_by_n_degree[n][d] +=
                    src->hard_graph_nodes_by_n_degree[n][d];
            }
        }

        total_task_timing.task_count += thread_task_timing[i].task_count;
        total_task_timing.task_time_sum += thread_task_timing[i].task_time_sum;
        if (thread_task_timing[i].task_time_max > total_task_timing.task_time_max) {
            total_task_timing.task_time_max = thread_task_timing[i].task_time_max;
            total_task_timing.task_max_index = thread_task_timing[i].task_max_index;
        }
        for (int k = 0; k < TASK_PROFILE_TOPK; k++) {
            task_timing_insert_topk(&total_task_timing,
                                    thread_task_timing[i].top_indices[k],
                                    thread_task_timing[i].top_times[k]);
        }
        if (thread_queue_subtask_timing) {
            QueueSubtaskTimingStats* src_sub =
                thread_queue_subtask_timing + (size_t)i * (size_t)(MAX_COLS + 1);
            for (int d = 0; d <= MAX_COLS; d++) {
                queue_subtask_merge(&total_queue_subtask_timing[d], &src_sub[d]);
            }
        }
    }

    free(thread_polys);
    free(thread_profiles);
    free(thread_task_timing);
    free(thread_queue_subtask_timing);
    progress_reporter_finish(&progress_reporter);

    double end_time = omp_get_wtime();
    double worker_time = end_time - start_time;
    double total_elapsed = worker_time + prefix_generation_time;
    
    printf("\nWorker Complete in %.2f seconds.\n", worker_time);
    if (prefix_depth > 0) {
        printf("Total elapsed including prefix generation: %.2f seconds.\n", total_elapsed);
    }
    printf("Canonicalisation calls: %lld\n", total_canon_calls);
    printf("Canonical cache hits: %lld (%.1f%%)\n", total_cache_hits,
           total_canon_calls > 0 ? 100.0 * total_cache_hits / total_canon_calls : 0.0);
    printf("Raw cache hits: %lld\n", total_raw_cache_hits);
    if (PROFILE_BUILD) {
        printf("Profile:\n");
        printf("  canon_state_prepare_push: %lld calls, %.3fs\n",
               total_profile.canon_prepare_calls, total_profile.canon_prepare_time);
        printf("  canon_state_commit_push: %lld calls, %.3fs\n",
               total_profile.canon_commit_calls, total_profile.canon_commit_time);
        printf("  partial_graph_append: %lld calls, %.3fs\n",
               total_profile.partial_append_calls, total_profile.partial_append_time);
        printf("  build_structure_weight: %lld calls, %.3fs\n",
               total_profile.solve_structure_calls, total_profile.build_weight_time);
        printf("  solve_graph_poly: %lld calls, %.3fs\n",
               total_profile.solve_graph_calls, total_profile.solve_graph_time);
        printf("  get_canonical_graph: %lld calls, %.3fs\n",
               total_profile.nauty_calls, total_profile.get_canonical_graph_time);
        printf("    dense rows: %.3fs\n",
               total_profile.get_canonical_graph_dense_rows_time);
        printf("    build nauty input: %.3fs\n",
               total_profile.get_canonical_graph_build_input_time);
        printf("    densenauty: %.3fs\n",
               total_profile.nauty_time);
        printf("    rebuild canon graph: %.3fs\n",
               total_profile.get_canonical_graph_rebuild_time);
        printf("  hard graph nodes: %lld, max n %d, max degree %d\n",
               total_profile.hard_graph_nodes,
               total_profile.hard_graph_max_n,
               total_profile.hard_graph_max_degree);
        printf("  Graph solver by simplified n (inclusive time):\n");
        for (int n = 0; n <= MAXN_NAUTY; n++) {
            long long calls = total_profile.solve_graph_calls_by_n[n];
            long long raw_hits = total_profile.solve_graph_raw_hits_by_n[n];
            long long canon_hits = total_profile.solve_graph_canon_hits_by_n[n];
            long long hard = total_profile.hard_graph_nodes_by_n[n];
            double time_s = total_profile.solve_graph_time_by_n[n];
            if (calls == 0 && raw_hits == 0 && canon_hits == 0 && hard == 0) continue;
            printf("    n=%d: calls %lld, time %.3fs, raw hits %lld, canon hits %lld, hard nodes %lld\n",
                   n, calls, time_s, raw_hits, canon_hits, hard);
        }
        printf("  Graph solver outcomes by simplified n:\n");
        for (int n = 0; n <= MAXN_NAUTY; n++) {
            long long lookup_calls = total_profile.solve_graph_lookup_calls_by_n[n];
            long long connected_lookup_calls = total_profile.solve_graph_connected_lookup_calls_by_n[n];
            long long raw_hits = total_profile.solve_graph_raw_hits_by_n[n];
            long long canon_hits = total_profile.solve_graph_canon_hits_by_n[n];
            long long component_calls = total_profile.solve_graph_component_calls_by_n[n];
            long long hard_misses = total_profile.solve_graph_hard_misses_by_n[n];
            if (lookup_calls == 0 && connected_lookup_calls == 0 &&
                raw_hits == 0 && canon_hits == 0 &&
                component_calls == 0 && hard_misses == 0) {
                continue;
            }
            printf("    n=%d: lookup %lld/%.3fs, connected-lookup %lld/%.3fs, raw-hit %lld/%.3fs, canon-hit %lld/%.3fs, components %lld/%.3fs, hard-miss %lld/%.3fs\n",
                   n,
                   lookup_calls, total_profile.solve_graph_lookup_time_by_n[n],
                   connected_lookup_calls, total_profile.solve_graph_connected_lookup_time_by_n[n],
                   raw_hits, total_profile.solve_graph_raw_hit_time_by_n[n],
                   canon_hits, total_profile.solve_graph_canon_hit_time_by_n[n],
                   component_calls, total_profile.solve_graph_component_time_by_n[n],
                   hard_misses, total_profile.solve_graph_hard_miss_time_by_n[n]);
        }
        printf("  Hard-miss subphases by simplified n:\n");
        for (int n = 0; n <= MAXN_NAUTY; n++) {
            long long hard_misses = total_profile.solve_graph_hard_misses_by_n[n];
            if (hard_misses == 0) continue;
            printf("    n=%d: separator %.3fs, pick %.3fs, delete %.3fs, contract-build %.3fs, contract-solve %.3fs, store %.3fs\n",
                   n,
                   total_profile.solve_graph_hard_miss_separator_time_by_n[n],
                   total_profile.solve_graph_hard_miss_pick_time_by_n[n],
                   total_profile.solve_graph_hard_miss_delete_time_by_n[n],
                   total_profile.solve_graph_hard_miss_contract_build_time_by_n[n],
                   total_profile.solve_graph_hard_miss_contract_solve_time_by_n[n],
                   total_profile.solve_graph_hard_miss_store_time_by_n[n]);
        }
        if (g_profile_separators) {
            printf("  Hard-miss separator detection by simplified n:\n");
            for (int n = 0; n <= MAXN_NAUTY; n++) {
                long long hard_misses = total_profile.solve_graph_hard_misses_by_n[n];
                long long articulation = total_profile.hard_graph_articulation_by_n[n];
                long long k2 = total_profile.hard_graph_k2_separator_by_n[n];
                if (hard_misses == 0 && articulation == 0 && k2 == 0) continue;
                printf("    n=%d: hard-miss %lld, articulation %lld, k2-separator %lld\n",
                       n, hard_misses, articulation, k2);
            }
        } else {
            printf("  Hard-miss separator detection: disabled"
                   " (set RECT_PROFILE_SEPARATORS=1 to enable)\n");
        }
        printf("  Hard graph nodes by simplified n and max degree:\n");
        for (int n = 10; n <= MAXN_NAUTY; n++) {
            long long total_n = total_profile.hard_graph_nodes_by_n[n];
            if (total_n == 0) continue;
            printf("    n=%d:", n);
            for (int d = 0; d <= MAXN_NAUTY; d++) {
                long long count = total_profile.hard_graph_nodes_by_n_degree[n][d];
                if (count == 0) continue;
                printf(" deg%d=%lld", d, count);
            }
            printf("\n");
        }
        printf("  CanonState by depth:\n");
        for (int d = 0; d <= g_cols && d <= MAX_COLS; d++) {
            long long calls = total_profile.canon_prepare_calls_by_depth[d];
            long long accepts = total_profile.canon_prepare_accepts_by_depth[d];
            if (calls == 0) continue;
            double accept_rate = 100.0 * (double)accepts / (double)calls;
            double avg_stabilizer =
                accepts > 0 ? (double)total_profile.stabilizer_sum_by_depth[d] / (double)accepts : 0.0;
            double avg_scanned =
                calls > 0 ? (double)total_profile.canon_prepare_scanned_by_depth[d] / (double)calls : 0.0;
            double avg_active =
                calls > 0 ? (double)total_profile.canon_prepare_active_by_depth[d] / (double)calls : 0.0;
            double active_rate =
                total_profile.canon_prepare_scanned_by_depth[d] > 0
                    ? 100.0 * (double)total_profile.canon_prepare_active_by_depth[d] /
                          (double)total_profile.canon_prepare_scanned_by_depth[d]
                    : 0.0;
            printf("    depth %d: prepare %lld, accept %lld (%.1f%%), avg stabiliser %.1f, avg active %.1f/%.0f (%.1f%%)\n",
                   d, calls, accepts, accept_rate, avg_stabilizer, avg_active, avg_scanned, active_rate);
        }
        printf("  CanonState prepare branch mix by depth:\n");
        for (int d = 0; d <= g_cols && d <= MAX_COLS; d++) {
            long long calls = total_profile.canon_prepare_calls_by_depth[d];
            long long terminal_calls = total_profile.canon_prepare_terminal_calls_by_depth[d];
            long long fast_continue = total_profile.canon_prepare_fast_continue_by_depth[d];
            long long terminal_continue = total_profile.canon_prepare_terminal_continue_by_depth[d];
            long long equal_case = total_profile.canon_prepare_equal_case_calls_by_depth[d];
            long long equal_reject = total_profile.canon_prepare_equal_case_rejects_by_depth[d];
            long long order_reject = total_profile.canon_prepare_order_rejects_by_depth[d];
            if (calls == 0 && equal_case == 0 && order_reject == 0) continue;
            printf("    depth %d: terminal %lld, fast-continue %lld, terminal-continue %lld, equal-case %lld, equal-reject %lld, order-reject %lld\n",
                   d, terminal_calls, fast_continue, terminal_continue, equal_case,
                   equal_reject, order_reject);
        }
        if (total_task_timing.task_count > 0) {
            printf("  Task timings: %lld tasks, avg %.6fs, max %.6fs (task %lld)\n",
                   total_task_timing.task_count,
                   total_task_timing.task_time_sum / (double)total_task_timing.task_count,
                   total_task_timing.task_time_max,
                   total_task_timing.task_max_index);
            printf("  Slowest tasks:\n");
            for (int i = 0; i < TASK_PROFILE_TOPK; i++) {
                if (total_task_timing.top_times[i] <= 0.0) break;
                int pi, pj, pk, pl;
                if (decode_task_prefix(total_task_timing.top_indices[i], &pi, &pj, &pk, &pl)) {
                    if (pk >= 0 && pl >= 0) {
                        printf("    task %lld (%d,%d,%d,%d): %.6fs\n",
                               total_task_timing.top_indices[i], pi, pj, pk, pl,
                               total_task_timing.top_times[i]);
                    } else if (pk >= 0) {
                        printf("    task %lld (%d,%d,%d): %.6fs\n",
                               total_task_timing.top_indices[i], pi, pj, pk,
                               total_task_timing.top_times[i]);
                    } else if (pj >= 0) {
                        printf("    task %lld (%d,%d): %.6fs\n",
                               total_task_timing.top_indices[i], pi, pj,
                               total_task_timing.top_times[i]);
                    } else {
                        printf("    task %lld (%d): %.6fs\n",
                               total_task_timing.top_indices[i], pi,
                               total_task_timing.top_times[i]);
                    }
                } else {
                    printf("    task %lld: %.6fs\n",
                           total_task_timing.top_indices[i],
                           total_task_timing.top_times[i]);
                }
            }
        }
        if (use_runtime_split_queue) {
            printf("  Queue subtasks by depth:\n");
            for (int d = 0; d <= g_cols && d <= MAX_COLS; d++) {
                QueueSubtaskTimingStats* qs = &total_queue_subtask_timing[d];
                if (qs->task_count == 0) continue;
                printf("    depth %d: %lld subtasks, avg %.6fs, max %.6fs, avg solve_graph %.1f, avg nauty %.1f, avg hard nodes %.1f, max hard n %d, max hard deg %d\n",
                       d, qs->task_count, qs->task_time_sum / (double)qs->task_count, qs->task_time_max,
                       (double)qs->solve_graph_call_sum / (double)qs->task_count,
                       (double)qs->nauty_call_sum / (double)qs->task_count,
                       (double)qs->hard_graph_node_sum / (double)qs->task_count,
                       qs->max_hard_graph_n, qs->max_hard_graph_degree);
                for (int i = 0; i < TASK_PROFILE_TOPK; i++) {
                    QueueSubtaskTopEntry* e = &qs->top[i];
                    if (e->elapsed <= 0.0) break;
                    printf("      (");
                    for (int p = 0; p < e->depth; p++) {
                        if (p > 0) printf(",");
                        printf("%u", (unsigned)e->prefix[p]);
                    }
                    printf("): %.6fs, solve_graph %lld, nauty %lld, hard_nodes %lld, max_hard_n %u, max_hard_deg %u\n",
                           e->elapsed, e->solve_graph_calls, e->nauty_calls,
                           e->hard_graph_nodes,
                           (unsigned)e->max_hard_graph_n,
                           (unsigned)e->max_hard_graph_degree);
                }
            }
        }
    }

    if (g_task_times_out_path) {
        write_task_times_file(g_task_times_out_path);
        printf("Task timing CSV: %s\n", g_task_times_out_path);
    }
    
#if RECT_COUNT_K4
    printf("\nRectangle-free 4-colourings:\n");
    print_u128(global_poly.coeffs[0]);
    printf("\n");
#else
    printf("\nChromatic Polynomial P(x):\n");
    print_poly(global_poly);

    printf("\nValues:\n");
    long long k_test = 4;
    printf("P(%lld) = ", k_test);
    print_u128(poly_eval(global_poly, k_test));
    printf("\n");

    k_test = 5;
    printf("P(%lld) = ", k_test);
    print_u128(poly_eval(global_poly, k_test));
    printf("\n");
#endif

    if (poly_out_path) {
        PolyFileMeta meta = {
            .rows = g_rows,
            .cols = g_cols,
            .task_start = active_task_start,
            .task_end = active_task_end,
            .full_tasks = full_tasks,
        };
        write_poly_file(poly_out_path, &global_poly, &meta);
#if RECT_COUNT_K4
        printf("\nWrote fixed-4 shard to %s\n", poly_out_path);
#else
        printf("\nWrote polynomial shard to %s\n", poly_out_path);
#endif
    }

    free(g_live_prefix2_i);
    free(g_live_prefix2_j);
    g_live_prefix2_i = NULL;
    g_live_prefix2_j = NULL;
    g_live_prefix2_count = 0;
    free(prefix2_batches);
    free(prefix2_batch_js);
    free(prefix2_batch_ps);
    free(g_task_times_values);
    g_task_times_values = NULL;
    if (shared_graph_cache_active) {
        shared_graph_cache_free(&shared_graph_cache);
        g_shared_graph_cache = NULL;
    }
    small_graph_lookup_free();
    connected_canon_lookup_free();
    free_row_dependent_tables();

    return 0;
}
