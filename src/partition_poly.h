#ifndef PARTITION_POLY_H
#define PARTITION_POLY_H

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

#include "../progress_util.h"

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

#define MAX_PERMUTATIONS 5040
#define PERM_BITSET_WORDS ((MAX_PERMUTATIONS + 63) / 64)
#define MAX_DEGREE ((MAX_ROWS * MAX_COLS) + 1)
#define CANON_PARTITION_ID_LIMIT (1u << 11)

#ifndef CACHE_BITS
#define CACHE_BITS 18
#endif
#define CACHE_SIZE (1 << CACHE_BITS)
#define CACHE_MASK (CACHE_SIZE - 1)
#ifndef CACHE_PROBE
#define CACHE_PROBE 16
#endif

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
    uint8_t x_pow;
    uint8_t deg;
    PolyCoeff coeffs[MAXN_NAUTY + 1];
} GraphPoly;

#if RECT_COUNT_K4
typedef uint64_t GraphResult;
typedef uint64_t GraphCacheValue;
#else
typedef GraphPoly GraphResult;
typedef PolyCoeff GraphCacheValue;
#endif

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

typedef struct {
    uint64_t key_hash;
    uint32_t key_n;
    uint8_t used;
} CacheKey;

typedef struct {
    CacheKey* keys;
    uint32_t* stamps;
    uint64_t* sigs;
    GraphCacheValue* coeffs;
    int mask;
    int probe;
    int poly_len;
    uint32_t next_stamp;
} GraphCache;

typedef struct {
    CacheKey* keys;
    uint32_t* stamps;
    AdjWord* rows;
    GraphCacheValue* coeffs;
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
    GraphResult value;
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

typedef struct {
    LocalTaskQueue shared_queue;
} RuntimeTaskSystem;

#define CONNECTED_CANON_LOOKUP_MAX_N 10
#define CONNECTED_CANON_LOOKUP_MAGIC UINT64_C(0x43434c394741424c)
#define CONNECTED_CANON_LOOKUP_VERSION 2U

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
    RuntimeTaskSystem runtime_tasks;
    int runtime_tasks_active;
#if RECT_COUNT_K4
    unsigned __int128* thread_totals;
#else
    Poly* thread_totals;
#endif
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

#if MAX_ROWS <= 7
typedef uint8_t ComplexMask;
#else
typedef uint32_t ComplexMask;
#endif

#if RECT_COUNT_K4
typedef unsigned __int128 WeightAccum;
typedef unsigned __int128 ResultAccum;
#else
typedef Poly WeightAccum;
typedef Poly ResultAccum;
#endif

extern int num_partitions;
extern int perm_count;
extern int max_partition_capacity;
extern int max_complex_per_partition;
extern Partition* partitions;
extern int (*perms)[MAX_ROWS];
extern uint16_t* perm_table;
extern uint16_t* perm_order_by_value;
extern uint16_t* perm_value_prefix_end;
extern uint64_t* perm_value_prefix_bits;
extern uint16_t* partition_id_lookup;
extern uint32_t partition_id_lookup_size;
extern uint64_t factorial[20];
extern ComplexMask* overlap_mask;
extern ComplexMask* intra_mask;
extern Poly* partition_weight_poly;
extern uint8_t* partition_weight4;
#if RECT_COUNT_K4_FEASIBILITY
extern uint32_t* pair_shadow_mask;
extern uint8_t* pair_shadow_pairs;
extern uint8_t* suffix_min_pairs;
extern int pair_index[MAX_ROWS][MAX_ROWS];
extern int num_row_pairs;
extern int min_partition_pairs;
#endif
extern PrefixId* g_live_prefix2_i;
extern PrefixId* g_live_prefix2_j;
extern long long g_live_prefix2_count;
extern long long completed_tasks;
extern Poly global_poly;
extern int g_rows;
extern int g_cols;
extern ProgressReporter progress_reporter;
extern int g_use_raw_cache;
extern long long progress_last_reported;
extern int g_adaptive_subdivide;
extern int g_adaptive_max_depth;
extern long long g_adaptive_work_budget;
extern __thread ProfileStats* tls_profile;
extern __thread GraphHardStats* tls_hard_graph_stats;
extern __thread long long* tls_adaptive_work_counter;
extern __thread SharedGraphCacheExporter* tls_shared_cache_exporter;
extern const char* g_task_times_out_path;
extern long long g_task_times_first_task;
extern long long g_task_times_count;
extern double* g_task_times_values;
extern int g_effective_prefix_depth;
extern double g_queue_profile_report_step;
extern int g_shared_cache_merge;
extern int g_shared_cache_bits;
extern int g_profile_separators;
extern SharedGraphCache* g_shared_graph_cache;

#define SMALL_GRAPH_LOOKUP_MAX_N 7
extern int g_small_graph_lookup_ready;
extern double g_small_graph_lookup_init_time;
extern int g_small_graph_lookup_loaded_from_file;
extern int32_t* g_small_graph_lookup_coeffs[SMALL_GRAPH_LOOKUP_MAX_N + 1];
extern uint8_t* g_small_graph_lookup_x_pows[SMALL_GRAPH_LOOKUP_MAX_N + 1];
extern uint8_t g_small_graph_edge_u[SMALL_GRAPH_LOOKUP_MAX_N + 1][21];
extern uint8_t g_small_graph_edge_v[SMALL_GRAPH_LOOKUP_MAX_N + 1][21];
extern uint32_t g_small_graph_graph_count[SMALL_GRAPH_LOOKUP_MAX_N + 1];
extern uint8_t g_small_graph_edge_count[SMALL_GRAPH_LOOKUP_MAX_N + 1];
extern uint32_t g_connected_canon_lookup_count;
extern int g_connected_canon_lookup_ready;
extern int g_connected_canon_lookup_loaded;
extern int g_connected_canon_lookup_n;
extern double g_connected_canon_lookup_load_time;

#define DEFAULT_PROGRESS_UPDATES 2000

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

void* checked_calloc(size_t count, size_t size, const char* label);
void* checked_aligned_alloc(size_t alignment, size_t size, const char* label);
void shared_graph_cache_flush_exports(void);
void task_timing_insert_topk(TaskTimingStats* stats, long long task_index, double elapsed);
void queue_subtask_record(QueueSubtaskTimingStats* stats, const LocalTask* task,
                          double elapsed, long long solve_graph_calls,
                          long long nauty_calls, long long hard_graph_nodes,
                          int max_hard_graph_n, int max_hard_graph_degree);
void flush_completed_tasks(long long total_tasks, long long report_step,
                           double start_time, long long* pending_completed);
void complete_task_report_and_time(long long total_tasks, long long report_step,
                                   double start_time, long long* pending_completed,
                                   TaskTimingStats* task_timing, long long task_index,
                                   double task_t0);
void prefix_task_buffer_init(PrefixTaskBuffer* buf, long long initial_capacity);
void prefix_task_buffer_push2(PrefixTaskBuffer* buf, int i, int j);
void local_task_from_stack(LocalTask* task, long long root_id, int depth, const int* stack);
void runtime_task_system_init(RuntimeTaskSystem* system, int capacity,
                              long long root_count, int total_threads);
void runtime_task_system_free(RuntimeTaskSystem* system);
void runtime_task_system_seed_task(RuntimeTaskSystem* system, const LocalTask* task);
int runtime_task_system_pop_task(RuntimeTaskSystem* system, LocalTask* task);
int runtime_task_system_push_local(RuntimeTaskSystem* system, const LocalTask* task);
int runtime_task_system_push_balance(RuntimeTaskSystem* system, const LocalTask* task);
int runtime_task_system_has_idle_workers(const RuntimeTaskSystem* system);
int runtime_task_system_needs_balance(const RuntimeTaskSystem* system);
void runtime_task_system_note_balance_push(RuntimeTaskSystem* system);
void runtime_task_system_note_work_budget_split(RuntimeTaskSystem* system);
void runtime_task_system_finish_task(RuntimeTaskSystem* system, long long root_id,
                                     long long total_tasks, long long report_step,
                                     double start_time, long long* pending_completed,
                                     TaskTimingStats* task_timing);
void runtime_task_system_record_profile(RuntimeTaskSystem* system, const LocalTask* task,
                                        double elapsed, long long solve_graph_calls,
                                        long long nauty_calls, long long hard_graph_nodes,
                                        int max_hard_graph_n, int max_hard_graph_degree);
void runtime_task_system_print_summary(RuntimeTaskSystem* system);
long long repeated_combo_count(int values, int slots);
void get_prefix2_task(long long task_index, int* i, int* j);
void build_fixed_prefix2_batches(const PrefixId* live_i, const PrefixId* live_j,
                                 long long task_start,
                                 long long total_tasks, Prefix2Batch** batches_out,
                                 long long* batch_count_out, PrefixId** js_out,
                                 long long** ps_out);
void unrank_prefix3(long long rank, int* i, int* j, int* k);
void unrank_prefix4(long long rank, int* i, int* j, int* k, int* l);
void queue_subtask_merge(QueueSubtaskTimingStats* dst, const QueueSubtaskTimingStats* src);
int decode_task_prefix(long long task_index, int* i, int* j, int* k, int* l);
void write_task_times_file(const char* path);

void init_row_dependent_tables(void);
void init_partition_lookup_tables(void);
void free_row_dependent_tables(void);
void generate_permutations(void);
void generate_partitions_recursive(int idx, uint8_t* current, int max_val);
void reorder_partitions_by_hardness(void);
void build_partition_id_lookup(void);
void build_perm_table(void);
void build_terminal_perm_order_tables(void);
void build_overlap_table(void);
#if RECT_COUNT_K4_FEASIBILITY
void init_pair_index(void);
void build_partition_shadow_table(void);
#endif
#if RECT_COUNT_K4
void build_partition_weight4_table(void);
#else
void build_partition_weight_table(void);
#endif
void weight_accum_one(WeightAccum* out);
void weight_accum_from_partition(int pid, WeightAccum* out);
void weight_accum_mul_partition(const WeightAccum* src, int pid, WeightAccum* out);
#if RECT_COUNT_K4
void weight_accum_scale_to_poly(const WeightAccum* weight_prod, long long mult_coeff,
                                long long row_orbit, uint64_t graph_count4, Poly* out);
#endif

long long parse_ll_or_die(const char* text, const char* label);
void poly_zero(Poly* p);
void poly_one_ref(Poly* p);
void poly_accumulate_checked(Poly* acc, const Poly* add);
void poly_mul_ref(const Poly* a, const Poly* b, Poly* out);
void poly_scale_ref(const Poly* a, long long s, Poly* out);
void poly_mul_falling_ref(const Poly* p, int start, int count, Poly* out);
void poly_mul_graph_ref(const Poly* a, const GraphPoly* b, Poly* out);
void graph_poly_normalize_ref(GraphPoly* p);
void graph_poly_one_ref(GraphPoly* p);
void graph_poly_mul_ref(const GraphPoly* a, const GraphPoly* b, GraphPoly* out);
void graph_poly_sub_ref(const GraphPoly* a, const GraphPoly* b, GraphPoly* out);
void graph_poly_mul_linear_ref(const GraphPoly* a, int c, GraphPoly* out);
void graph_poly_div_x_ref(const GraphPoly* a, GraphPoly* out);
int32_t* small_graph_poly_slot(int n, uint32_t mask);
uint64_t graph_pack_upper_mask64(const Graph* g);
uint64_t graph_row_mask(int n);
void get_canonical_graph(Graph* g, Graph* canon, NautyWorkspace* ws, ProfileStats* profile);
void get_canonical_graph_from_dense_rows(int n, const AdjWord* rows, Graph* canon,
                                         NautyWorkspace* ws, ProfileStats* profile);
uint32_t small_graph_pack_mask(const Graph* g);
void small_graph_lookup_load_graph_poly(int n, uint32_t mask, GraphPoly* out);
uint32_t graph_build_dense_rows(const Graph* g, AdjWord* rows);
uint64_t graph_fill_dense_key_rows(const Graph* g, AdjWord row_mask, AdjWord* rows);
int row_graph_cache_lookup_poly(RowGraphCache* cache, uint64_t key_hash, uint32_t key_n,
                                const Graph* g, AdjWord row_mask, GraphResult* value, int touch);
int row_graph_cache_lookup_rows(RowGraphCache* cache, uint64_t key_hash, uint32_t key_n,
                                const AdjWord* rows, GraphResult* value, int touch);
int shared_graph_cache_lookup_poly(SharedGraphCache* shared, uint64_t key_hash, uint32_t key_n,
                                   const Graph* g, uint64_t row_mask, GraphResult* value);
void shared_graph_cache_export(uint64_t key_hash, uint32_t key_n, const Graph* g,
                               uint64_t row_mask, const GraphResult* value);
void store_row_graph_cache_entry(RowGraphCache* cache, uint64_t key_hash, uint32_t key_n,
                                 const Graph* g, AdjWord row_mask, const GraphResult* value);
void store_row_graph_cache_entry_rows(RowGraphCache* cache, uint64_t key_hash, uint32_t key_n,
                                      const AdjWord* rows, const GraphResult* value);
const int32_t* connected_canon_lookup_find_coeffs(uint64_t mask);
int connected_canon_lookup_load_graph_poly(const Graph* g, GraphPoly* out);
void induced_subgraph_from_mask(const Graph* src, uint64_t mask, Graph* dst);
int graph_collect_components(const Graph* g, uint64_t* component_masks);
int graph_collect_biconnected_components(const Graph* g, uint64_t* block_masks,
                                         uint64_t* articulation_mask);
int graph_has_articulation_point(const Graph* g);
int graph_has_k2_separator(const Graph* g);
uint64_t hash_graph(const Graph* g);
void nauty_workspace_init(NautyWorkspace* ws, int n);
void nauty_workspace_free(NautyWorkspace* ws);
void small_graph_lookup_init(void);
void small_graph_lookup_free(void);
void connected_canon_lookup_init(void);
void connected_canon_lookup_free(void);
void shared_graph_cache_init(SharedGraphCache* shared, int bits, int poly_len);
void shared_graph_cache_free(SharedGraphCache* shared);
#if RECT_COUNT_K4
static inline void graph_result_set_count4(uint64_t count, GraphResult* out) {
    *out = count;
}

static inline uint64_t graph_result_get_count4(const GraphResult* p) {
    return *p;
}

uint64_t small_graph_lookup_load_count4(int n, uint32_t mask);
uint64_t connected_canon_lookup_load_count4(const Graph* g);
uint64_t count_graph_4_dsat(const Graph* g);
#endif
PolyCoeff poly_eval(Poly p, long long x);
void print_u128(PolyCoeff n);
void print_poly(Poly p);
void write_poly_file(const char* path, const Poly* poly, const PolyFileMeta* meta);

#endif
