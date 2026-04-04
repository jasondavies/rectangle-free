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

// Polynomial arithmetic and polynomial-format I/O live separately.
#include "src/poly.c"
// Runtime support, queueing, and prefix-index helpers live separately.
#include "src/runtime.c"
// Partition generation, table building, and weight initialisation live separately.
#include "src/partitions.c"
// Canonical-state, DFS, and runtime-prefix replay logic live separately.
#include "src/canon.c"
// Main/orchestration code lives separately to keep the core solver readable.
#include "src/main.c"
