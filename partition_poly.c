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
#include "progress_util.h"

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
    uint8_t mapping[MAX_ROWS];
    int num_blocks;
    uint32_t block_masks[MAX_ROWS]; 
    int is_complex[MAX_ROWS];       
    uint8_t complex_blocks[MAX_ROWS];
    int num_complex;
    int num_singletons;
} Partition;

typedef struct {
    int n;
    uint64_t adj[MAXN_NAUTY]; 
} Graph;

#define GRAPH_SIG_BITS ((MAXN_NAUTY * (MAXN_NAUTY - 1)) / 2)
#define GRAPH_SIG_WORDS ((GRAPH_SIG_BITS + 63) / 64)

typedef struct {
    Graph g;
    int base[MAX_COLS];
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
    int hard_graph_max_n;
    int hard_graph_max_degree;
    double canon_prepare_time;
    double canon_commit_time;
    double partial_append_time;
    double build_weight_time;
    double solve_graph_time;
    double nauty_time;
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
    Poly value;
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

// --- GLOBALS ---
static int num_partitions = 0;
static int perm_count = 0;
static int max_partition_capacity = 0;
static int max_complex_per_partition = 0;
static Partition* partitions = NULL;
static int (*perms)[MAX_ROWS] = NULL;
static uint16_t* perm_table = NULL;
static uint16_t* partition_id_lookup = NULL;
static uint32_t partition_id_lookup_size = 0;
static uint64_t factorial[20];
static uint32_t* overlap_mask = NULL;
static uint32_t* intra_mask = NULL;
static Poly* partition_weight_poly = NULL;
static PrefixId* g_live_prefix2_i = NULL;
static PrefixId* g_live_prefix2_j = NULL;
static long long g_live_prefix2_count = 0;

static long long completed_tasks = 0;
static Poly global_poly = {0}; 

static int g_rows = DEFAULT_ROWS;
static int g_cols = DEFAULT_COLS;
static ProgressReporter progress_reporter;
static long long progress_last_reported = 0;
static int g_adaptive_subdivide = 0;
static int g_adaptive_max_depth = 3;
static long long g_adaptive_work_budget = 0;
static int g_profile = 0;
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

static void* checked_calloc(size_t count, size_t size, const char* label);
static void* checked_aligned_alloc(size_t alignment, size_t size, const char* label);
static void shared_graph_cache_flush_exports(void);
static inline void graph_cache_load_poly(const GraphCache* cache, int slot, Poly* value);
static inline uint32_t graph_cache_next_stamp(GraphCache* cache);
static inline void graph_cache_touch_slot(GraphCache* cache, int slot);
static inline int graph_cache_slot_matches_sig(const GraphCache* cache, int slot, uint64_t key_hash,
                                               uint32_t key_n, const uint64_t* sig);
void store_graph_cache_entry(GraphCache* cache, uint64_t key_hash, uint32_t key_n, const Graph* g,
                             uint64_t row_mask, const Poly* value);
static void small_graph_lookup_init(void);
static void small_graph_lookup_free(void);

static inline uint32_t* intra_mask_row(int partition_id) {
    return intra_mask + (size_t)partition_id * (size_t)max_complex_per_partition;
}

static inline uint32_t intra_mask_get(int partition_id, int complex_idx) {
    return intra_mask_row(partition_id)[complex_idx];
}

static inline uint32_t* overlap_mask_row(int lhs_partition_id, int rhs_partition_id) {
    return overlap_mask +
           (((size_t)lhs_partition_id * (size_t)num_partitions + (size_t)rhs_partition_id) *
            (size_t)max_complex_per_partition);
}

static inline uint32_t overlap_mask_get(int lhs_partition_id, int rhs_partition_id, int complex_idx) {
    return overlap_mask_row(lhs_partition_id, rhs_partition_id)[complex_idx];
}

static void unrank_prefix2(long long rank, int* i, int* j);
static void unrank_prefix3(long long rank, int* i, int* j, int* k);
static void unrank_prefix4(long long rank, int* i, int* j, int* k, int* l);
static inline long long repeated_combo_count(int values, int slots);

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
    if (g_profile && task_timing) {
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
        if (g_profile && task_timing && queue->roots[root_id].launched_at >= 0.0) {
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
}

static void free_row_dependent_tables(void) {
    free(partitions);
    free(perms);
    free(perm_table);
    free(partition_id_lookup);
    free(overlap_mask);
    free(intra_mask);
    free(partition_weight_poly);

    partitions = NULL;
    perms = NULL;
    perm_table = NULL;
    partition_id_lookup = NULL;
    partition_id_lookup_size = 0;
    overlap_mask = NULL;
    intra_mask = NULL;
    partition_weight_poly = NULL;
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
            "  %s [rows cols] [--task-start N] [--task-end N] [--prefix-depth N] [--adaptive-subdivide] [--adaptive-max-depth N] [--adaptive-work-budget N] [--poly-out FILE] [--profile] [--task-times-out FILE]\n"
            "\n"
            "Notes:\n"
            "  --task-start/--task-end define a half-open task range [start, end).\n"
            "  --prefix-depth may be 2, 3, or 4.\n"
            "  Adaptive subdivision currently supports only --prefix-depth 2.\n"
            "  In full polynomial mode it uses a local runtime queue of donated subtrees.\n"
            "  --profile prints coarse timing counters for the main phases.\n",
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

void generate_partitions_recursive(int idx, uint8_t* current, int max_val) {
    if (idx == g_rows) {
        Partition part;
        memset(&part, 0, sizeof(part));
        memcpy(part.mapping, current, g_rows);
        part.num_blocks = max_val + 1;
        
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
    if (max_val < g_rows - 1) {
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

void build_overlap_table() {
    memset(overlap_mask, 0,
           (size_t)num_partitions * (size_t)num_partitions * (size_t)max_complex_per_partition *
               sizeof(*overlap_mask));
    memset(intra_mask, 0,
           (size_t)num_partitions * (size_t)max_complex_per_partition * sizeof(*intra_mask));
    for (int pid1 = 0; pid1 < num_partitions; pid1++) {
        for (int i1 = 0; i1 < partitions[pid1].num_complex; i1++) {
            uint32_t mask = 0;
            for (int i2 = 0; i2 < partitions[pid1].num_complex; i2++) {
                if (i1 != i2) mask |= (1u << i2);
            }
            intra_mask_row(pid1)[i1] = mask;
        }
        for (int i1 = 0; i1 < partitions[pid1].num_complex; i1++) {
            int b1 = partitions[pid1].complex_blocks[i1];
            uint32_t m1 = partitions[pid1].block_masks[b1];
            for (int pid2 = 0; pid2 < num_partitions; pid2++) {
                uint32_t mask = 0;
                for (int i2 = 0; i2 < partitions[pid2].num_complex; i2++) {
                    int b2 = partitions[pid2].complex_blocks[i2];
                    uint32_t m2 = partitions[pid2].block_masks[b2];
                    if (__builtin_popcount(m1 & m2) >= 2) {
                        mask |= (1u << i2);
                    }
                }
                overlap_mask_row(pid1, pid2)[i1] = mask;
            }
        }
    }
}

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

// --- SYMMETRY LOGIC ---

typedef struct {
    int limit;
    int depth;
    uint32_t* state;
    uint16_t* changed_first_greater_idx;
    uint32_t* changed_first_greater_old_state;
    uint16_t changed_first_greater_count[MAX_COLS];
    uint16_t stack_vals[MAX_COLS];
    int stabilizer[MAX_COLS + 1];
} CanonState;

typedef struct {
    int limit;
    uint32_t* next_state;
    uint16_t* active_idx;
    uint16_t* changed_first_greater_idx;
    uint16_t active_count;
    uint16_t changed_first_greater_count;
} CanonScratch;

typedef struct {
    GraphCache cache;
    GraphCache raw_cache;
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

static inline uint32_t* canon_state_changed_first_greater_old_state_row(CanonState* st, int depth) {
    return st->changed_first_greater_old_state + (size_t)depth * (size_t)st->limit;
}

static inline uint32_t canon_state_pack(uint8_t first_greater, uint16_t first_greater_val) {
    return (uint32_t)first_greater | ((uint32_t)first_greater_val << 8);
}

static inline uint8_t canon_state_unpack_first_greater(uint32_t state) {
    return (uint8_t)(state & 0xffU);
}

static inline uint16_t canon_state_unpack_first_greater_val(uint32_t state) {
    return (uint16_t)(state >> 8);
}

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

static inline int canon_rebuild_equal_case(const CanonState* st, int p, int g, uint16_t pid,
                                           uint8_t* next_first_greater, uint16_t* next_first_greater_val) {
    int depth = st->depth;
    int new_depth = depth + 1;
    uint16_t row[MAX_COLS];
    int len = 0;

    for (int t = 0; t < depth; t++) {
        const uint16_t* perm_row = perm_table + (size_t)st->stack_vals[t] * (size_t)perm_count;
        row_insert_sorted(row, len, perm_row[p]);
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

static void canon_state_init(CanonState* st, int limit) {
    memset(st, 0, sizeof(*st));
    st->limit = limit;
    st->state =
        checked_calloc((size_t)limit, sizeof(*st->state), "canon_state_state");
    st->changed_first_greater_idx =
        checked_calloc((size_t)g_cols * (size_t)limit, sizeof(*st->changed_first_greater_idx),
                       "canon_state_changed_first_greater_idx");
    st->changed_first_greater_old_state =
        checked_calloc((size_t)g_cols * (size_t)limit, sizeof(*st->changed_first_greater_old_state),
                       "canon_state_changed_first_greater_old_state");
}

static void canon_state_free(CanonState* st) {
    free(st->state);
    free(st->changed_first_greater_idx);
    free(st->changed_first_greater_old_state);
    memset(st, 0, sizeof(*st));
}

static void canon_scratch_init(CanonScratch* scratch, int limit) {
    memset(scratch, 0, sizeof(*scratch));
    scratch->limit = limit;
    scratch->next_state =
        checked_calloc((size_t)limit, sizeof(*scratch->next_state), "canon_scratch_next_state");
    scratch->active_idx =
        checked_calloc((size_t)limit, sizeof(*scratch->active_idx), "canon_scratch_active_idx");
    scratch->changed_first_greater_idx = checked_calloc((size_t)limit,
                                                        sizeof(*scratch->changed_first_greater_idx),
                                                        "canon_scratch_changed_first_greater_idx");
}

static void canon_scratch_free(CanonScratch* scratch) {
    free(scratch->next_state);
    free(scratch->active_idx);
    free(scratch->changed_first_greater_idx);
    memset(scratch, 0, sizeof(*scratch));
}

void canon_state_reset(CanonState* st, int limit) {
    st->limit = limit;
    st->depth = 0;
    st->stabilizer[0] = limit;
    memset(st->state, 0, (size_t)limit * sizeof(*st->state));
}

int canon_state_prepare_push(const CanonState* st, int partition_id, CanonScratch* scratch,
                             int* next_stabilizer) {
    int depth = st->depth;
    int new_depth = depth + 1;
    int stabilizer = 0;
    uint16_t pid = (uint16_t)partition_id;
    const uint16_t* partition_perm_row =
        perm_table + (size_t)partition_id * (size_t)perm_count;
    scratch->active_count = 0;
    scratch->changed_first_greater_count = 0;

    for (int p = 0; p < st->limit; p++) {
        uint32_t old_state = st->state[p];
        uint16_t x = partition_perm_row[p];
        int g = (int)canon_state_unpack_first_greater(old_state);
        uint8_t next_fg;
        uint16_t next_fg_val;

        if (g == depth) {
            if (x < pid) return 0;
            if (x == pid) {
                next_fg = (uint8_t)new_depth;
                next_fg_val = 0;
                stabilizer++;
            } else {
                next_fg = (uint8_t)depth;
                next_fg_val = x;
            }
        } else {
            uint16_t r = canon_state_unpack_first_greater_val(old_state);
            if (x >= r) continue;

            uint16_t c = st->stack_vals[g];
            if (x < c) return 0;
            if (x > c) {
                next_fg = (uint8_t)g;
                next_fg_val = x;
            } else {
                if (!canon_rebuild_equal_case(st, p, g, pid, &next_fg, &next_fg_val)) {
                    return 0;
                }
                if (next_fg == new_depth) {
                    stabilizer++;
                }
            }
        }

        uint16_t new_fg_val = (next_fg < new_depth) ? next_fg_val : 0;
        uint32_t new_state = canon_state_pack(next_fg, new_fg_val);
        scratch->next_state[p] = new_state;
        scratch->active_idx[scratch->active_count++] = (uint16_t)p;
        if (old_state != new_state) {
            scratch->changed_first_greater_idx[scratch->changed_first_greater_count++] = (uint16_t)p;
        }

    }
    if (g_profile && tls_profile) {
        tls_profile->canon_prepare_scanned_by_depth[depth] += st->limit;
        tls_profile->canon_prepare_active_by_depth[depth] += scratch->active_count;
    }
    *next_stabilizer = stabilizer;
    return 1;
}

void canon_state_commit_push(CanonState* st, int partition_id, const CanonScratch* scratch,
                             int next_stabilizer) {
    int depth = st->depth;
    int new_depth = depth + 1;
    uint16_t* changed_first_greater_idx = canon_state_changed_first_greater_idx_row(st, depth);
    uint32_t* changed_first_greater_old_state = canon_state_changed_first_greater_old_state_row(st, depth);
    uint16_t changed_fg_count = scratch->changed_first_greater_count;
    st->stack_vals[depth] = (uint16_t)partition_id;

    for (uint16_t i = 0; i < changed_fg_count; i++) {
        uint16_t p = scratch->changed_first_greater_idx[i];
        changed_first_greater_idx[i] = p;
        changed_first_greater_old_state[i] = st->state[p];
        st->state[p] = scratch->next_state[p];
    }

    st->changed_first_greater_count[depth] = changed_fg_count;
    st->stabilizer[new_depth] = next_stabilizer;
    st->depth = new_depth;
}

void canon_state_pop(CanonState* st) {
    int depth = st->depth - 1;
    uint16_t* changed_first_greater_idx = canon_state_changed_first_greater_idx_row(st, depth);
    uint32_t* changed_first_greater_old_state = canon_state_changed_first_greater_old_state_row(st, depth);
    for (uint16_t i = 0; i < st->changed_first_greater_count[depth]; i++) {
        uint16_t p = changed_first_greater_idx[i];
        st->state[p] = changed_first_greater_old_state[i];
    }
    st->depth = depth;
}

long long get_orbit_multiplier_state(const CanonState* st) {
    int stabilizer = st->stabilizer[st->depth];
    return factorial[g_rows] / stabilizer;
}

// --- NAUTY CANONICALISATION ---

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

void get_canonical_graph(Graph* g, Graph* canon, NautyWorkspace* ws, ProfileStats* profile) {
    int n = g->n;
    double t0 = 0.0;
    
    if (n == 0) {
        canon->n = 0;
        memset(canon->adj, 0, sizeof(canon->adj));
        return;
    }
    
    if (n == 1) {
        canon->n = 1;
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
    
    EMPTYGRAPH(ng, m, n);

    // Convert our adjacency representation to nauty format
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if ((g->adj[i] >> j) & 1ULL) {
                ADDONEEDGE(ng, i, j, m);
            }
        }
    }
    
    // Initialise labelling
    for (int i = 0; i < n; i++) {
        lab[i] = i;
        ptn[i] = 1;
    }
    ptn[n-1] = 0;
    
    // Set up options for canonical labelling
    DEFAULTOPTIONS_GRAPH(options);
    options.getcanon = TRUE;
    options.defaultptn = TRUE;
    
    statsblk stats;
    
    // Compute canonical form
    if (g_profile && profile) t0 = omp_get_wtime();
    densenauty(ng, lab, ptn, orbits, &options, &stats, m, n, cg);
    if (g_profile && profile) {
        profile->nauty_calls++;
        profile->nauty_time += omp_get_wtime() - t0;
    }
    
    // Convert canonical graph back to our format
    canon->n = n;
    memset(canon->adj, 0, (size_t)n * sizeof(canon->adj[0]));
    
    for (int i = 0; i < n; i++) {
        set *row = GRAPHROW(cg, i, m);
        for (int j = i + 1; j < n; j++) {
            if (ISELEMENT(row, j)) {
                canon->adj[i] |= (1ULL << j);
                canon->adj[j] |= (1ULL << i);
            }
        }
    }
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
    uint32_t mask = 0;
    int bit = 0;
    for (int j = 1; j < g->n; j++) {
        for (int i = 0; i < j; i++, bit++) {
            if ((g->adj[i] >> j) & 1ULL) mask |= 1U << bit;
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

static inline uint64_t graph_row_mask(int n) {
    if (n >= 64) return ~0ULL;
    if (n <= 0) return 0ULL;
    return (1ULL << n) - 1ULL;
}

static void induced_subgraph_from_mask(const Graph* src, uint64_t mask, Graph* dst) {
    int verts[MAXN_NAUTY];
    int n = 0;
    uint64_t rem = mask;
    while (rem) {
        verts[n++] = __builtin_ctzll(rem);
        rem &= rem - 1;
    }

    dst->n = n;
    memset(dst->adj, 0, sizeof(dst->adj));
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if ((src->adj[verts[i]] >> verts[j]) & 1ULL) {
                dst->adj[i] |= 1ULL << j;
                dst->adj[j] |= 1ULL << i;
            }
        }
    }
}

static int graph_collect_components(const Graph* g, uint64_t* component_masks) {
    uint64_t remaining = graph_row_mask(g->n);
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
                next |= g->adj[v];
                current &= current - 1;
            }
            frontier = next & remaining & ~component;
        }
        component_masks[count++] = component;
        remaining &= ~component;
    }
    return count;
}

uint64_t hash_graph(const Graph* g) {
    uint64_t row_mask = graph_row_mask(g->n);
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < g->n; i++) {
        h ^= (g->adj[i] & row_mask);
        h *= 1099511628211ULL;
    }
    h ^= (uint64_t)g->n;
    h *= 1099511628211ULL;
    return h;
}

static inline uint64_t* graph_cache_sig_slot(const GraphCache* cache, int slot) {
    return cache->sigs + (size_t)slot * GRAPH_SIG_WORDS;
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

static inline uint32_t graph_cache_next_stamp(GraphCache* cache) {
    cache->next_stamp++;
    if (cache->next_stamp == 0) cache->next_stamp = 1;
    return cache->next_stamp;
}

static inline void graph_cache_touch_slot(GraphCache* cache, int slot) {
    cache->stamps[slot] = graph_cache_next_stamp(cache);
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

static int graph_cache_lookup_poly(GraphCache* cache, uint64_t key_hash, uint32_t key_n,
                                   const Graph* g, uint64_t row_mask, Poly* value, int touch) {
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

static inline void graph_cache_load_poly(const GraphCache* cache, int slot, Poly* value) {
    int deg = cache->degs[slot];
    value->deg = deg;
    memcpy(value->coeffs, graph_cache_coeff_slot(cache, slot),
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
                                          const Graph* g, uint64_t row_mask, Poly* value) {
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
                                      uint64_t row_mask, const Poly* value) {
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
                             uint64_t row_mask, const Poly* value) {
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

void remove_vertex(Graph* g, int i) {
    int last = g->n - 1;
    for(int k=0; k<g->n; k++) g->adj[k] &= ~(1ULL << i);
    
    if (i != last) {
        g->adj[i] = g->adj[last];
        for (int k = 0; k < last; k++) {
            if (k == i) continue;
            if ((g->adj[k] >> last) & 1ULL) {
                g->adj[k] &= ~(1ULL << last);
                g->adj[k] |= (1ULL << i);
            } else {
                g->adj[k] &= ~(1ULL << last);
            }
        }
        g->adj[i] &= ~(1ULL << i);
    } else {
        for (int k = 0; k < last; k++) g->adj[k] &= ~(1ULL << last);
    }
    g->n--;
}

static inline void record_hard_graph_node(ProfileStats* profile, int n, int max_degree) {
    if (g_profile && profile) {
        profile->hard_graph_nodes++;
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

static void solve_graph_poly(const Graph* input_g, GraphCache* cache, GraphCache* raw_cache,
                             NautyWorkspace* ws, long long* local_canon_calls,
                             long long* local_cache_hits, long long* local_raw_cache_hits,
                             ProfileStats* profile, Poly* out_result) {
    Graph g = *input_g;
    double solve_t0 = 0.0;
    if (g_profile && profile) {
        profile->solve_graph_calls++;
        solve_t0 = omp_get_wtime();
    }
    Poly multiplier;
    poly_one_ref(&multiplier);
    
    // Simplification loop - same as before
    int changed = 1;
    while (changed && g.n > 0) {
        changed = 0;
        for (int i = 0; i < g.n; i++) {
            uint64_t neighbors = g.adj[i];
            int degree = __builtin_popcountll(neighbors);
            
            if (degree == 0) {
                poly_mul_linear_ref(&multiplier, 0, &multiplier);
                remove_vertex(&g, i);
                changed = 1; i--; continue;
            }
            
            int is_clique = 1;
            uint64_t rem = neighbors;
            while (rem) {
                int u = __builtin_ctzll(rem);
                if ((neighbors & ~g.adj[u]) != (1ULL << u)) {
                    is_clique = 0;
                    break;
                }
                rem &= rem - 1;
            }
            
            if (is_clique) {
                poly_mul_linear_ref(&multiplier, degree, &multiplier);
                remove_vertex(&g, i);
                changed = 1; i--;
            }
        }
    }
    
    if (g.n == 0) {
        *out_result = multiplier;
        return;
    }

    if (g.n <= SMALL_GRAPH_LOOKUP_MAX_N) {
        Poly small_poly;
        small_graph_lookup_load_poly(g.n, small_graph_pack_mask(&g), &small_poly);
        poly_mul_ref(&multiplier, &small_poly, out_result);
        if (g_profile && profile) profile->solve_graph_time += omp_get_wtime() - solve_t0;
        return;
    }

    uint64_t row_mask = graph_row_mask(g.n);
    uint64_t raw_hash = hash_graph(&g);
    uint64_t raw_sig[GRAPH_SIG_WORDS];
    graph_pack_signature(&g, (uint32_t)g.n, raw_sig);
    int raw_cache_idx = (int)(raw_hash & (uint64_t)raw_cache->mask);

    // Fast exact lookup on labelled graph before canonicalisation.
    for (int k = 0; k < raw_cache->probe; k++) {
        int p = (raw_cache_idx + k) & raw_cache->mask;
        if (graph_cache_slot_matches_sig(raw_cache, p, raw_hash, (uint32_t)g.n, raw_sig)) {
            Poly cached;
            (*local_raw_cache_hits)++;
            graph_cache_load_poly(raw_cache, p, &cached);
            graph_cache_touch_slot(raw_cache, p);
            poly_mul_ref(&multiplier, &cached, out_result);
            if (g_profile && profile) profile->solve_graph_time += omp_get_wtime() - solve_t0;
            return;
        }
    }

    Poly res;
    uint64_t component_masks[MAXN_NAUTY];
    int component_count = graph_collect_components(&g, component_masks);
    if (component_count > 1) {
        poly_one_ref(&res);
        for (int i = 0; i < component_count; i++) {
            Graph subgraph;
            induced_subgraph_from_mask(&g, component_masks[i], &subgraph);
            Poly part;
            solve_graph_poly(&subgraph, cache, raw_cache, ws,
                             local_canon_calls, local_cache_hits, local_raw_cache_hits,
                             profile, &part);
            poly_mul_ref(&res, &part, &res);
        }
    } else {
        // Canonicalise only if exact lookup missed and the graph is still connected.
        Graph canon;
        get_canonical_graph(&g, &canon, ws, profile);
        (*local_canon_calls)++;
        
        uint64_t hash = hash_graph(&canon);
        uint64_t canon_sig[GRAPH_SIG_WORDS];
        graph_pack_signature(&canon, (uint32_t)canon.n, canon_sig);
        int cache_idx = (int)(hash & (uint64_t)cache->mask);
        
        // Cache lookup using canonical form
        for (int k = 0; k < cache->probe; k++) {
            int p = (cache_idx + k) & cache->mask;
            if (graph_cache_slot_matches_sig(cache, p, hash, (uint32_t)canon.n, canon_sig)) {
                Poly cached;
                (*local_cache_hits)++;
                graph_cache_load_poly(cache, p, &cached);
                graph_cache_touch_slot(cache, p);
                store_graph_cache_entry(raw_cache, raw_hash, (uint32_t)g.n, &g, row_mask, &cached);
                poly_mul_ref(&multiplier, &cached, out_result);
                if (g_profile && profile) profile->solve_graph_time += omp_get_wtime() - solve_t0;
                return;
            }
        }

        if (shared_graph_cache_lookup_poly(g_shared_graph_cache, hash, (uint32_t)canon.n,
                                           &canon, ADJWORD_MASK, &res)) {
            store_graph_cache_entry(cache, hash, (uint32_t)canon.n, &canon, ADJWORD_MASK, &res);
            (*local_cache_hits)++;
            store_graph_cache_entry(raw_cache, raw_hash, (uint32_t)g.n, &g, row_mask, &res);
            poly_mul_ref(&multiplier, &res, out_result);
            if (g_profile && profile) profile->solve_graph_time += omp_get_wtime() - solve_t0;
            return;
        }

        // Deletion-contraction on canonical graph
        const Graph* branch_g = &canon;
        int max_deg = -1, u = -1;
        for (int i = 0; i < branch_g->n; i++) {
            int d = __builtin_popcountll(branch_g->adj[i]);
            if (d > max_deg) { max_deg = d; u = i; }
        }
        if (u != -1 && max_deg > 0) record_hard_graph_node(profile, branch_g->n, max_deg);

        int v = -1;
        if (u != -1) {
            for (int k = 0; k < branch_g->n; k++) {
                if ((branch_g->adj[u] >> k) & 1ULL) { v = k; break; }
            }
        }

        if (u != -1 && v != -1) {
            // Deletion: remove edge (u,v)
            Graph g_del = *branch_g;
            g_del.adj[u] &= ~(1ULL << v);
            g_del.adj[v] &= ~(1ULL << u);
            Poly p_del;
            solve_graph_poly(&g_del, cache, raw_cache, ws,
                             local_canon_calls, local_cache_hits, local_raw_cache_hits,
                             profile, &p_del);

            // Contraction: merge v into u
            Graph g_cont = *branch_g;
            g_cont.adj[u] |= g_cont.adj[v];
            g_cont.adj[u] &= ~(1ULL << u);
            g_cont.adj[u] &= ~(1ULL << v);
            for (int k = 0; k < g_cont.n; k++) {
                if (k == u || k == v) continue;
                if ((g_cont.adj[k] >> v) & 1ULL) {
                    g_cont.adj[k] &= ~(1ULL << v);
                    g_cont.adj[k] |= (1ULL << u);
                    g_cont.adj[u] |= (1ULL << k);
                }
            }
            remove_vertex(&g_cont, v);
            Poly p_cont;
            solve_graph_poly(&g_cont, cache, raw_cache, ws,
                             local_canon_calls, local_cache_hits, local_raw_cache_hits,
                             profile, &p_cont);

            poly_sub_ref(&p_del, &p_cont, &res);
        } else {
            poly_one_ref(&res);
            for (int k = 0; k < branch_g->n; k++) poly_mul_linear_ref(&res, 0, &res);
        }

        store_graph_cache_entry(cache, hash, (uint32_t)canon.n, &canon, ADJWORD_MASK, &res);
        shared_graph_cache_export(hash, (uint32_t)canon.n, &canon, ADJWORD_MASK, &res);
    }

    store_graph_cache_entry(raw_cache, raw_hash, (uint32_t)g.n, &g, row_mask, &res);
    poly_mul_ref(&multiplier, &res, out_result);
    if (g_profile && profile) profile->solve_graph_time += omp_get_wtime() - solve_t0;
}

static void partial_graph_reset(PartialGraphState* st) {
    st->g.n = 0;
    memset(st->g.adj, 0, sizeof(st->g.adj));
    memset(st->base, 0, sizeof(st->base));
}

static int partial_graph_append(PartialGraphState* st, int depth, int pid, const int* stack) {
    int base_new = st->g.n;
    int num_complex = partitions[pid].num_complex;
    st->base[depth] = base_new;
    st->g.n += num_complex;
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
        if (!partial_graph_append(&partial_graph, 0, i, stack)) {
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
            if (partial_graph_append(&prefix_graph, 1, j, stack)) {
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

static void solve_structure(const Graph* partial_graph, CanonState* canon_state,
                            GraphCache* cache, GraphCache* raw_cache, NautyWorkspace* ws,
                            long long* local_canon_calls, long long* local_cache_hits,
                            long long* local_raw_cache_hits, const Poly* weight_prod,
                            long long mult_coeff, ProfileStats* profile, Poly* out_result) {
    double t0 = 0.0;
    if (g_profile && profile) {
        profile->solve_structure_calls++;
        t0 = omp_get_wtime();
    }
    long long row_orbit = get_orbit_multiplier_state(canon_state);
    Poly weight;
    poly_scale_ref(weight_prod, mult_coeff * row_orbit, &weight);
    if (g_profile && profile) profile->build_weight_time += omp_get_wtime() - t0;
    Poly graph_poly;
    solve_graph_poly(partial_graph, cache, raw_cache, ws,
                     local_canon_calls, local_cache_hits, local_raw_cache_hits,
                     profile, &graph_poly);
    poly_mul_ref(&weight, &graph_poly, out_result);
}

void dfs(int depth, int min_idx, int* stack, CanonState* canon_state, const PartialGraphState* partial_graph,
         GraphCache* cache, GraphCache* raw_cache, NautyWorkspace* ws, Poly* local_total,
         long long* local_canon_calls, long long* local_cache_hits,
         long long* local_raw_cache_hits, const Poly* weight_prod, long long mult_coeff,
         int run_len, ProfileStats* profile, CanonScratch* canon_scratch) {
    if (depth == g_cols) {
        Poly res;
        solve_structure(&partial_graph->g, canon_state, cache, raw_cache, ws,
                        local_canon_calls, local_cache_hits, local_raw_cache_hits,
                        weight_prod, mult_coeff, profile, &res);
        poly_add_ref_checked(local_total, &res, local_total);
        return;
    }

    int next_stabilizer = 0;
    for (int i = min_idx; i < num_partitions; i++) {
        double t0 = 0.0;
        if (g_profile && profile) {
            profile->canon_prepare_calls++;
            profile->canon_prepare_calls_by_depth[depth]++;
            t0 = omp_get_wtime();
        }
        if (!canon_state_prepare_push(canon_state, i, canon_scratch, &next_stabilizer)) {
            if (g_profile && profile) profile->canon_prepare_time += omp_get_wtime() - t0;
            continue;
        }
        if (g_profile && profile) {
            profile->canon_prepare_time += omp_get_wtime() - t0;
            profile->canon_prepare_accepts++;
            profile->canon_prepare_accepts_by_depth[depth]++;
            profile->stabilizer_sum_by_depth[depth] += next_stabilizer;
            profile->canon_commit_calls++;
            t0 = omp_get_wtime();
        }
        stack[depth] = i;
        Poly next_weight_prod;
        poly_mul_ref(weight_prod, &partition_weight_poly[i], &next_weight_prod);
        long long next_mult_coeff = mult_coeff * (depth + 1);
        int next_run_len = 1;
        if (depth > 0 && i == stack[depth - 1]) {
            next_run_len = run_len + 1;
            next_mult_coeff /= next_run_len;
        }
        canon_state_commit_push(canon_state, i, canon_scratch, next_stabilizer);
        if (g_profile && profile) profile->canon_commit_time += omp_get_wtime() - t0;
        PartialGraphState next_graph = *partial_graph;
        if (g_profile && profile) {
            profile->partial_append_calls++;
            t0 = omp_get_wtime();
        }
        int ok = partial_graph_append(&next_graph, depth, i, stack);
        if (g_profile && profile) profile->partial_append_time += omp_get_wtime() - t0;
        if (ok) {
            dfs(depth + 1, i, stack, canon_state, &next_graph, cache, raw_cache, ws, local_total,
                local_canon_calls, local_cache_hits, local_raw_cache_hits, &next_weight_prod,
                next_mult_coeff, next_run_len, profile, canon_scratch);
        }
        canon_state_pop(canon_state);
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
                                        GraphCache* cache, GraphCache* raw_cache, NautyWorkspace* ws,
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
    if (g_profile) {
        profile->canon_prepare_calls++;
        profile->canon_prepare_calls_by_depth[0]++;
        t0 = omp_get_wtime();
    }
    if (!canon_state_prepare_push(canon_state, (int)i, canon_scratch, &next_stabilizer)) {
        if (g_profile) profile->canon_prepare_time += omp_get_wtime() - t0;
        return;
    }
    if (g_profile) {
        profile->canon_prepare_time += omp_get_wtime() - t0;
        profile->canon_prepare_accepts++;
        profile->canon_prepare_accepts_by_depth[0]++;
        profile->stabilizer_sum_by_depth[0] += next_stabilizer;
        profile->canon_commit_calls++;
        t0 = omp_get_wtime();
    }
    canon_state_commit_push(canon_state, (int)i, canon_scratch, next_stabilizer);
    if (g_profile) profile->canon_commit_time += omp_get_wtime() - t0;
    if (g_profile) {
        profile->partial_append_calls++;
        t0 = omp_get_wtime();
    }
    if (!partial_graph_append(partial_graph, 0, (int)i, stack)) {
        if (g_profile) profile->partial_append_time += omp_get_wtime() - t0;
        canon_state_pop(canon_state);
        return;
    }
    if (g_profile) profile->partial_append_time += omp_get_wtime() - t0;

    for (int idx = 0; idx < count; idx++) {
        long long p = ps[idx];
        double task_t0 = g_profile ? omp_get_wtime() : 0.0;
        PrefixId j = js[idx];
        Poly task_total;
        poly_zero(&task_total);

        stack[1] = (int)j;
        if (g_profile) {
            profile->canon_prepare_calls++;
            profile->canon_prepare_calls_by_depth[1]++;
            t0 = omp_get_wtime();
        }
        if (!canon_state_prepare_push(canon_state, (int)j, canon_scratch, &next_stabilizer)) {
            if (g_profile) profile->canon_prepare_time += omp_get_wtime() - t0;
            complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                          pending_completed, task_timing, p, task_t0);
            continue;
        }
        if (g_profile) {
            profile->canon_prepare_time += omp_get_wtime() - t0;
            profile->canon_prepare_accepts++;
            profile->canon_prepare_accepts_by_depth[1]++;
            profile->stabilizer_sum_by_depth[1] += next_stabilizer;
            profile->canon_commit_calls++;
            t0 = omp_get_wtime();
        }
        canon_state_commit_push(canon_state, (int)j, canon_scratch, next_stabilizer);
        if (g_profile) profile->canon_commit_time += omp_get_wtime() - t0;
        PartialGraphState prefix_graph = *partial_graph;
        if (g_profile) {
            profile->partial_append_calls++;
            t0 = omp_get_wtime();
        }
        int ok = partial_graph_append(&prefix_graph, 1, (int)j, stack);
        if (g_profile) profile->partial_append_time += omp_get_wtime() - t0;
        if (ok) {
            Poly prefix_weight;
            poly_mul_ref(&partition_weight_poly[i], &partition_weight_poly[j], &prefix_weight);
            long long prefix_mult = (i == j) ? 1 : 2;
            int prefix_run = (i == j) ? 2 : 1;
            dfs(2, (int)j, stack, canon_state, &prefix_graph, cache, raw_cache, ws, &task_total,
                local_canon_calls, local_cache_hits, local_raw_cache_hits,
                &prefix_weight, prefix_mult, prefix_run, profile, canon_scratch);
            poly_add_ref_checked(local_total, &task_total, local_total);
        }
        canon_state_pop(canon_state);
        complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                      pending_completed, task_timing, p, task_t0);
    }

    canon_state_pop(canon_state);
}

static int replay_local_task_prefix(const LocalTask* task, WorkerCtx* ctx,
                                    Poly* weight_prod, long long* mult_coeff,
                                    int* run_len, int* min_idx) {
    int next_stabilizer = 0;
    int prev_pid = -1;

    canon_state_reset(&ctx->canon_state, perm_count);
    partial_graph_reset(&ctx->partial_graph);
    poly_one_ref(weight_prod);
    *mult_coeff = 1;
    *run_len = 0;
    *min_idx = 0;

    for (int depth = 0; depth < task->depth; depth++) {
        int pid = (int)task->prefix[depth];
        double t0 = 0.0;
        ctx->stack[depth] = pid;

        if (g_profile) {
            tls_profile->canon_prepare_calls++;
            tls_profile->canon_prepare_calls_by_depth[depth]++;
            t0 = omp_get_wtime();
        }
        if (!canon_state_prepare_push(&ctx->canon_state, pid,
                                      &ctx->canon_scratch, &next_stabilizer)) {
            if (g_profile) tls_profile->canon_prepare_time += omp_get_wtime() - t0;
            return 0;
        }
        if (g_profile) {
            tls_profile->canon_prepare_time += omp_get_wtime() - t0;
            tls_profile->canon_prepare_accepts++;
            tls_profile->canon_prepare_accepts_by_depth[depth]++;
            tls_profile->stabilizer_sum_by_depth[depth] += next_stabilizer;
            tls_profile->canon_commit_calls++;
            t0 = omp_get_wtime();
        }
        canon_state_commit_push(&ctx->canon_state, pid,
                                &ctx->canon_scratch, next_stabilizer);
        if (g_profile) tls_profile->canon_commit_time += omp_get_wtime() - t0;

        if (g_profile) {
            tls_profile->partial_append_calls++;
            t0 = omp_get_wtime();
        }
        if (!partial_graph_append(&ctx->partial_graph, depth, pid, ctx->stack)) {
            if (g_profile) tls_profile->partial_append_time += omp_get_wtime() - t0;
            canon_state_pop(&ctx->canon_state);
            return 0;
        }
        if (g_profile) tls_profile->partial_append_time += omp_get_wtime() - t0;

        poly_mul_ref(weight_prod, &partition_weight_poly[pid], weight_prod);

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
                                    Poly* local_total, const Poly* weight_prod,
                                    long long mult_coeff, int run_len, ProfileStats* profile,
                                    LocalTaskQueue* queue) {
    if (depth == g_cols) {
        Poly res;
        solve_structure(&ctx->partial_graph.g, &ctx->canon_state,
                        &ctx->cache, &ctx->raw_cache, &ctx->ws,
                        &ctx->local_canon_calls, &ctx->local_cache_hits,
                        &ctx->local_raw_cache_hits, weight_prod, mult_coeff, profile, &res);
        poly_add_ref_checked(local_total, &res, local_total);
        return;
    }

    int next_stabilizer = 0;
    int local_end = end_pid;
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
        ctx->stack[depth] = pid;
        if (g_profile) {
            tls_profile->canon_prepare_calls++;
            tls_profile->canon_prepare_calls_by_depth[depth]++;
            t0 = omp_get_wtime();
        }
        if (!canon_state_prepare_push(&ctx->canon_state, pid, &ctx->canon_scratch, &next_stabilizer)) {
            if (g_profile) tls_profile->canon_prepare_time += omp_get_wtime() - t0;
            continue;
        }
        if (g_profile) {
            tls_profile->canon_prepare_time += omp_get_wtime() - t0;
            tls_profile->canon_prepare_accepts++;
            tls_profile->canon_prepare_accepts_by_depth[depth]++;
            tls_profile->stabilizer_sum_by_depth[depth] += next_stabilizer;
            tls_profile->canon_commit_calls++;
            t0 = omp_get_wtime();
        }

        canon_state_commit_push(&ctx->canon_state, pid, &ctx->canon_scratch, next_stabilizer);
        if (g_profile) tls_profile->canon_commit_time += omp_get_wtime() - t0;
        PartialGraphState saved_graph = ctx->partial_graph;

        if (g_profile) {
            tls_profile->partial_append_calls++;
            t0 = omp_get_wtime();
        }
        if (partial_graph_append(&ctx->partial_graph, depth, pid, ctx->stack)) {
            if (g_profile) tls_profile->partial_append_time += omp_get_wtime() - t0;
            Poly next_weight_prod;
            poly_mul_ref(weight_prod, &partition_weight_poly[pid], &next_weight_prod);
            long long next_mult_coeff = mult_coeff * (depth + 1);
            int next_run_len = 1;
            if (depth > 0 && pid == ctx->stack[depth - 1]) {
                next_run_len = run_len + 1;
                next_mult_coeff /= next_run_len;
            }

            dfs_runtime_split_local(depth + 1, pid, num_partitions, root_id, ctx, local_total,
                                    &next_weight_prod, next_mult_coeff, next_run_len,
                                    profile, queue);
        } else if (g_profile) {
            tls_profile->partial_append_time += omp_get_wtime() - t0;
        }

        ctx->partial_graph = saved_graph;
        canon_state_pop(&ctx->canon_state);
    }
}

static void execute_local_runtime_task(const LocalTask* task, WorkerCtx* ctx, Poly* thread_total,
                                       LocalTaskQueue* queue, ProfileStats* profile,
                                       long long total_tasks, long long report_step,
                                       double start_time, long long* pending_completed,
                                       TaskTimingStats* task_timing,
                                       QueueSubtaskTimingStats* queue_subtask_stats) {
    Poly weight_prod;
    long long mult_coeff = 1;
    int run_len = 0;
    int min_idx = 0;
    double subtask_t0 = g_profile ? omp_get_wtime() : 0.0;
    long long solve_graph_before = g_profile ? profile->solve_graph_calls : 0;
    long long nauty_before = g_profile ? profile->nauty_calls : 0;
    long long adaptive_work_counter = 0;
    GraphHardStats subtask_hard = {0};
    GraphHardStats* prev_hard_stats = tls_hard_graph_stats;
    if (g_profile) tls_hard_graph_stats = &subtask_hard;
    long long* prev_work_counter = tls_adaptive_work_counter;
    if (g_adaptive_work_budget > 0) tls_adaptive_work_counter = &adaptive_work_counter;

    if (replay_local_task_prefix(task, ctx, &weight_prod, &mult_coeff, &run_len, &min_idx)) {
        if (task->depth == g_cols) {
            Poly res;
            solve_structure(&ctx->partial_graph.g, &ctx->canon_state,
                            &ctx->cache, &ctx->raw_cache, &ctx->ws,
                            &ctx->local_canon_calls, &ctx->local_cache_hits,
                            &ctx->local_raw_cache_hits, &weight_prod, mult_coeff, profile, &res);
            poly_add_ref_checked(thread_total, &res, thread_total);
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

    if (g_profile && queue_subtask_stats && task->depth <= MAX_COLS) {
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
        } else if (strcmp(argv[i], "--profile") == 0) {
            g_profile = 1;
        } else if (strcmp(argv[i], "--task-times-out") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            g_task_times_out_path = argv[++i];
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
    
    // 3. Build lookup tables
    init_partition_lookup_tables();
    build_partition_id_lookup();
    build_perm_table();
    build_overlap_table();
    build_partition_weight_table();
    
    printf("Grid: %dx%d\n", g_rows, g_cols);
    printf("Partitions: %d\n", num_partitions);
    printf("Threads: %d\n", omp_get_max_threads());
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
    if (g_task_times_out_path && !g_profile) {
        fprintf(stderr, "--task-times-out requires --profile\n");
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
    }
    if (task_start < 0) {
        fprintf(stderr, "--task-start must be non-negative\n");
        return 1;
    }
    if (task_end >= 0 && task_end < task_start) {
        fprintf(stderr, "Task range must satisfy 0 <= start <= end\n");
        return 1;
    }
    g_effective_prefix_depth = prefix_depth;

    int graph_poly_len = g_cols * (g_rows / 2) + 1;
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
        if (g_profile) {
            printf("Small-graph lookup %s: %.2f seconds\n",
                   g_small_graph_lookup_loaded_from_file ? "load" : "initialisation",
                   g_small_graph_lookup_init_time);
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
    if (use_runtime_split_queue && g_profile) {
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
        GraphCache cache = {0};
        GraphCache raw_cache = {0};
        NautyWorkspace ws;
        memset(&ws, 0, sizeof(ws));
        cache.mask = CACHE_MASK;
        cache.probe = CACHE_PROBE;
        cache.poly_len = graph_poly_len;
        cache.keys = checked_aligned_alloc(64, sizeof(CacheKey) * CACHE_SIZE, "cache_keys");
        cache.stamps = checked_aligned_alloc(64, sizeof(uint32_t) * CACHE_SIZE, "cache_stamps");
        cache.sigs = checked_aligned_alloc(64, sizeof(uint64_t) * CACHE_SIZE * GRAPH_SIG_WORDS, "cache_sigs");
        cache.degs = checked_aligned_alloc(64, sizeof(uint8_t) * CACHE_SIZE, "cache_degs");
        cache.coeffs =
            checked_aligned_alloc(64, sizeof(PolyCoeff) * CACHE_SIZE * (size_t)graph_poly_len, "cache_coeffs");
        cache.next_stamp = 0;

        raw_cache.mask = RAW_CACHE_MASK;
        raw_cache.probe = RAW_CACHE_PROBE;
        raw_cache.poly_len = graph_poly_len;
        raw_cache.keys = checked_aligned_alloc(64, sizeof(CacheKey) * RAW_CACHE_SIZE, "raw_cache_keys");
        raw_cache.stamps = checked_aligned_alloc(64, sizeof(uint32_t) * RAW_CACHE_SIZE, "raw_cache_stamps");
        raw_cache.sigs =
            checked_aligned_alloc(64, sizeof(uint64_t) * RAW_CACHE_SIZE * GRAPH_SIG_WORDS, "raw_cache_sigs");
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
                double task_t0 = g_profile ? omp_get_wtime() : 0.0;
                double t0 = 0.0;
                stack[0] = i;
                canon_state_reset(&canon_state, perm_count);
                partial_graph_reset(&partial_graph);
                int next_stabilizer = 0;
                if (g_profile) {
                    profile->canon_prepare_calls++;
                    profile->canon_prepare_calls_by_depth[0]++;
                    t0 = omp_get_wtime();
                }
                if (!canon_state_prepare_push(&canon_state, (int)i, &canon_scratch, &next_stabilizer)) {
                    if (g_profile) profile->canon_prepare_time += omp_get_wtime() - t0;
                    complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                  &pending_completed, task_timing, i, task_t0);
                    continue;
                }
                if (g_profile) {
                    profile->canon_prepare_time += omp_get_wtime() - t0;
                    profile->canon_prepare_accepts++;
                    profile->canon_prepare_accepts_by_depth[0]++;
                    profile->stabilizer_sum_by_depth[0] += next_stabilizer;
                    profile->canon_commit_calls++;
                    t0 = omp_get_wtime();
                }
                canon_state_commit_push(&canon_state, (int)i, &canon_scratch, next_stabilizer);
                if (g_profile) profile->canon_commit_time += omp_get_wtime() - t0;
                if (g_profile) {
                    profile->partial_append_calls++;
                    t0 = omp_get_wtime();
                }
                int ok = partial_graph_append(&partial_graph, 0, (int)i, stack);
                if (g_profile) profile->partial_append_time += omp_get_wtime() - t0;
                if (ok) {
                    dfs(1, (int)i, stack, &canon_state, &partial_graph, &cache, &raw_cache, &ws,
                        &thread_polys[tid], &local_canon_calls, &local_cache_hits, &local_raw_cache_hits,
                        &partition_weight_poly[i], 1, 1, profile, &canon_scratch);
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
                    double task_t0 = g_profile ? omp_get_wtime() : 0.0;
                    double t0 = 0.0;
                    long long p = first_task + t;
                    int i = 0;
                    int j = 0;
                    get_prefix2_task(p, &i, &j);
                    int next_stabilizer = 0;

                    canon_state_reset(&canon_state, perm_count);
                    partial_graph_reset(&partial_graph);

                    stack[0] = i;
                    if (g_profile) {
                        profile->canon_prepare_calls++;
                        profile->canon_prepare_calls_by_depth[0]++;
                        t0 = omp_get_wtime();
                    }
                    if (!canon_state_prepare_push(&canon_state, i, &canon_scratch, &next_stabilizer)) {
                        if (g_profile) profile->canon_prepare_time += omp_get_wtime() - t0;
                        complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                      &pending_completed, task_timing, p, task_t0);
                        continue;
                    }
                    if (g_profile) {
                        profile->canon_prepare_time += omp_get_wtime() - t0;
                        profile->canon_prepare_accepts++;
                        profile->canon_prepare_accepts_by_depth[0]++;
                        profile->stabilizer_sum_by_depth[0] += next_stabilizer;
                        profile->canon_commit_calls++;
                        t0 = omp_get_wtime();
                    }
                    canon_state_commit_push(&canon_state, i, &canon_scratch, next_stabilizer);
                    if (g_profile) profile->canon_commit_time += omp_get_wtime() - t0;
                    if (g_profile) {
                        profile->partial_append_calls++;
                        t0 = omp_get_wtime();
                    }
                    int ok = partial_graph_append(&partial_graph, 0, i, stack);
                    if (g_profile) profile->partial_append_time += omp_get_wtime() - t0;
                    if (!ok) {
                        canon_state_pop(&canon_state);
                        complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                      &pending_completed, task_timing, p, task_t0);
                        continue;
                    }

                    stack[1] = j;
                    if (g_profile) {
                        profile->canon_prepare_calls++;
                        profile->canon_prepare_calls_by_depth[1]++;
                        t0 = omp_get_wtime();
                    }
                    if (!canon_state_prepare_push(&canon_state, j, &canon_scratch, &next_stabilizer)) {
                        if (g_profile) profile->canon_prepare_time += omp_get_wtime() - t0;
                        canon_state_pop(&canon_state);
                        complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                      &pending_completed, task_timing, p, task_t0);
                        continue;
                    }
                    if (g_profile) {
                        profile->canon_prepare_time += omp_get_wtime() - t0;
                        profile->canon_prepare_accepts++;
                        profile->canon_prepare_accepts_by_depth[1]++;
                        profile->stabilizer_sum_by_depth[1] += next_stabilizer;
                        profile->canon_commit_calls++;
                        t0 = omp_get_wtime();
                    }
                    canon_state_commit_push(&canon_state, j, &canon_scratch, next_stabilizer);
                    if (g_profile) profile->canon_commit_time += omp_get_wtime() - t0;
                    PartialGraphState prefix_graph = partial_graph;
                    if (g_profile) {
                        profile->partial_append_calls++;
                        t0 = omp_get_wtime();
                    }
                    ok = partial_graph_append(&prefix_graph, 1, j, stack);
                    if (g_profile) profile->partial_append_time += omp_get_wtime() - t0;
                    if (ok) {
                        Poly prefix_weight;
                        poly_mul_ref(&partition_weight_poly[i], &partition_weight_poly[j],
                                     &prefix_weight);
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
                double task_t0 = g_profile ? omp_get_wtime() : 0.0;
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
                if (!partial_graph_append(&partial_graph, 0, i, stack)) {
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
                if (!partial_graph_append(&prefix_graph, 1, j, stack)) {
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
                int ok = partial_graph_append(&prefix_graph2, 2, k, stack);
                if (ok) {
                    Poly prefix_weight;
                    poly_mul_ref(&partition_weight_poly[i], &partition_weight_poly[j], &prefix_weight);
                    poly_mul_ref(&prefix_weight, &partition_weight_poly[k], &prefix_weight);
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
                double task_t0 = g_profile ? omp_get_wtime() : 0.0;
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
                if (!partial_graph_append(&partial_graph, 0, i, stack)) {
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
                if (!partial_graph_append(&prefix_graph, 1, j, stack)) {
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
                if (!partial_graph_append(&prefix_graph2, 2, k, stack)) {
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
                int ok = partial_graph_append(&prefix_graph3, 3, l, stack);
                if (ok) {
                    Poly prefix_weight;
                    poly_mul_ref(&partition_weight_poly[i], &partition_weight_poly[j], &prefix_weight);
                    poly_mul_ref(&prefix_weight, &partition_weight_poly[k], &prefix_weight);
                    poly_mul_ref(&prefix_weight, &partition_weight_poly[l], &prefix_weight);
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
        free(cache.sigs);
        free(cache.degs);
        free(cache.coeffs);
        free(raw_cache.keys);
        free(raw_cache.stamps);
        free(raw_cache.sigs);
        free(raw_cache.degs);
        free(raw_cache.coeffs);
    }

    if (local_queue_active) {
        local_queue_print_occupancy_summary(&local_queue);
        local_queue_free(&local_queue);
    }
    
    for(int i=0; i<num_threads; i++) {
        poly_add_ref_checked(&global_poly, &thread_polys[i], &global_poly);
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
        total_profile.nauty_time += src->nauty_time;
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
    if (g_profile) {
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
        printf("  get_canonical_graph/densenauty: %lld calls, %.3fs\n",
               total_profile.nauty_calls, total_profile.nauty_time);
        printf("  hard graph nodes: %lld, max n %d, max degree %d\n",
               total_profile.hard_graph_nodes,
               total_profile.hard_graph_max_n,
               total_profile.hard_graph_max_degree);
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

    if (poly_out_path) {
        PolyFileMeta meta = {
            .rows = g_rows,
            .cols = g_cols,
            .task_start = active_task_start,
            .task_end = active_task_end,
            .full_tasks = full_tasks,
        };
        write_poly_file(poly_out_path, &global_poly, &meta);
        printf("\nWrote polynomial shard to %s\n", poly_out_path);
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
    free_row_dependent_tables();

    return 0;
}
