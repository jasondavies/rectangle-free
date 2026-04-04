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
#define CANON_PARTITION_ID_LIMIT (1u << 11)

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

typedef struct {
    long long task_start;
    long long task_end;
    const char* poly_out_path;
    int prefix_depth_override;
    int reorder_partitions_flag;
} MainOptions;

typedef struct {
    int prefix_depth;
    int graph_poly_len;
    long long total_prefixes;
    long long nominal_prefixes;
    Prefix2Batch* prefix2_batches;
    PrefixId* prefix2_batch_js;
    long long* prefix2_batch_ps;
    long long prefix2_batch_count;
    double prefix_generation_time;
    int use_runtime_split_queue;
    SharedGraphCache shared_graph_cache;
    int shared_graph_cache_active;
    long long full_tasks;
    long long active_task_start;
    long long active_task_end;
    long long total_tasks;
    long long first_task;
    long long progress_report_step;
} RunConfig;

typedef struct {
    int num_threads;
    LocalTaskQueue local_queue;
    int local_queue_active;
    Poly* thread_polys;
    ProfileStats* thread_profiles;
    TaskTimingStats* thread_task_timing;
    QueueSubtaskTimingStats* thread_queue_subtask_timing;
    long long total_canon_calls;
    long long total_cache_hits;
    long long total_raw_cache_hits;
} ExecutionState;

typedef struct {
    ProfileStats profile;
    TaskTimingStats task_timing;
    QueueSubtaskTimingStats queue_subtask_timing[MAX_COLS + 1];
} ExecutionSummary;

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

static void generate_permutations(void);
static void generate_partitions_recursive(int idx, uint8_t* current, int max_val);
static void reorder_partitions_by_hardness(void);
static void build_partition_id_lookup(void);
static void build_perm_table(void);
static void build_terminal_perm_order_tables(void);
static void build_overlap_table(void);
static void build_partition_weight_table(void);
static void shared_graph_cache_init(SharedGraphCache* shared, int bits, int poly_len);
static void shared_graph_cache_free(SharedGraphCache* shared);
static void build_live_prefix2_tasks(PrefixId** live_i_out, PrefixId** live_j_out,
                                     long long* live_count_out);

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


// Partition generation, table building, and weight initialisation live separately.
#include "src/partitions.c"
// Canonical-state, DFS, and runtime-prefix replay logic live separately.
#include "src/canon.c"
// Main/orchestration code lives separately to keep the core solver readable.
#include "src/main.c"
