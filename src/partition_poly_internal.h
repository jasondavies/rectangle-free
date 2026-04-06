#ifndef PARTITION_POLY_INTERNAL_H
#define PARTITION_POLY_INTERNAL_H

#include "partition_poly.h"

typedef struct {
    int capacity;
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
    uint16_t first_greater_bucket_count[MAX_COLS + 1];
    uint16_t stack_vals[MAX_COLS];
    const uint16_t* stack_perm_rows[MAX_COLS];
    uint64_t first_greater_bucket_bits[MAX_COLS + 1][PERM_BITSET_WORDS];
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

typedef struct {
    uint8_t old_n;
    uint64_t old_vertex_mask;
    uint8_t touched_prev_count;
    uint8_t touched_prev_idx[MAXN_NAUTY];
    AdjWord touched_prev_old_adj[MAXN_NAUTY];
#if RECT_COUNT_K4_FEASIBILITY
    uint8_t old_remaining_capacity;
    uint32_t old_full_pair_mask;
    uint32_t pair_shadow;
#endif
} PartialGraphAppendFrame;

void canon_state_init(CanonState* st, int limit);
void canon_state_free(CanonState* st);
void canon_scratch_init(CanonScratch* scratch, int limit);
void canon_scratch_free(CanonScratch* scratch);
void canon_state_reset(CanonState* st, int limit);
int canon_state_prepare_push(const CanonState* st, int partition_id, CanonScratch* scratch,
                             int* next_stabilizer);
void canon_state_commit_push(CanonState* st, int partition_id, const CanonScratch* scratch,
                             int next_stabilizer);
void canon_state_pop(CanonState* st);
void partial_graph_reset(PartialGraphState* st);
int partial_graph_append_checked(PartialGraphState* st, int depth, int pid,
                                 const int* stack, int cols_left);
void build_live_prefix2_tasks(PrefixId** live_i_out, PrefixId** live_j_out,
                              long long* live_count_out);
void dfs(int depth, int min_idx, int* stack, CanonState* canon_state,
         PartialGraphState* partial_graph, RowGraphCache* cache,
         RowGraphCache* raw_cache, NautyWorkspace* ws, Poly* local_total,
         long long* local_canon_calls, long long* local_cache_hits,
         long long* local_raw_cache_hits, const WeightAccum* weight_prod,
         long long mult_coeff, int run_len, ProfileStats* profile,
         CanonScratch* canon_scratch);
void execute_prefix2_fixed_batch(PrefixId i, const PrefixId* js, const long long* ps, int count,
                                 RowGraphCache* cache, RowGraphCache* raw_cache,
                                 NautyWorkspace* ws, CanonState* canon_state,
                                 CanonScratch* canon_scratch,
                                 PartialGraphState* partial_graph, int* stack,
                                 Poly* local_total, long long* local_canon_calls,
                                 long long* local_cache_hits,
                                 long long* local_raw_cache_hits, ProfileStats* profile,
                                 long long total_tasks, long long progress_report_step,
                                 double start_time, long long* pending_completed,
                                 TaskTimingStats* task_timing);
void execute_local_runtime_task(const LocalTask* task, WorkerCtx* ctx, Poly* thread_total,
                                RuntimeTaskSystem* runtime_tasks, ProfileStats* profile,
                                long long total_tasks, long long progress_report_step,
                                double start_time, long long* pending_completed,
                                TaskTimingStats* task_timing,
                                QueueSubtaskTimingStats* queue_subtask_stats);
void solve_graph_poly(const Graph* input_g, RowGraphCache* cache, RowGraphCache* raw_cache,
                      NautyWorkspace* ws, long long* local_canon_calls,
                      long long* local_cache_hits, long long* local_raw_cache_hits,
                      ProfileStats* profile, GraphResult* out_result);

#endif
