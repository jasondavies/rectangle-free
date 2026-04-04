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

int32_t* small_graph_poly_slot(int n, uint32_t mask);
uint64_t graph_pack_upper_mask64(const Graph* g);
int connected_canon_lookup_entry_cmp(const void* lhs, const void* rhs);
uint64_t graph_row_mask(int n);
void get_canonical_graph(Graph* g, Graph* canon, NautyWorkspace* ws, ProfileStats* profile);
void get_canonical_graph_from_dense_rows(int n, const AdjWord* rows, Graph* canon,
                                         NautyWorkspace* ws, ProfileStats* profile);
uint32_t small_graph_pack_mask(const Graph* g);
void small_graph_lookup_load_graph_poly(int n, uint32_t mask, GraphPoly* out);
uint32_t graph_build_dense_rows(const Graph* g, AdjWord* rows);
uint64_t graph_fill_dense_key_rows(const Graph* g, AdjWord row_mask, AdjWord* rows);
int row_graph_cache_lookup_poly(RowGraphCache* cache, uint64_t key_hash, uint32_t key_n,
                                const Graph* g, AdjWord row_mask, GraphPoly* value, int touch);
int row_graph_cache_lookup_rows(RowGraphCache* cache, uint64_t key_hash, uint32_t key_n,
                                const AdjWord* rows, GraphPoly* value, int touch);
int shared_graph_cache_lookup_poly(SharedGraphCache* shared, uint64_t key_hash, uint32_t key_n,
                                   const Graph* g, uint64_t row_mask, GraphPoly* value);
void shared_graph_cache_export(uint64_t key_hash, uint32_t key_n, const Graph* g,
                               uint64_t row_mask, const GraphPoly* value);
void store_row_graph_cache_entry(RowGraphCache* cache, uint64_t key_hash, uint32_t key_n,
                                 const Graph* g, AdjWord row_mask, const GraphPoly* value);
void store_row_graph_cache_entry_rows(RowGraphCache* cache, uint64_t key_hash, uint32_t key_n,
                                      const AdjWord* rows, const GraphPoly* value);
int connected_canon_lookup_load_graph_poly(const Graph* g, GraphPoly* out);
void induced_subgraph_from_mask(const Graph* src, uint64_t mask, Graph* dst);
int graph_collect_components(const Graph* g, uint64_t* component_masks);
int graph_has_articulation_point(const Graph* g);
int graph_has_k2_separator(const Graph* g);
uint64_t hash_graph(const Graph* g);
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
void nauty_workspace_init(NautyWorkspace* ws, int n);
void nauty_workspace_free(NautyWorkspace* ws);
void small_graph_lookup_init(void);
void small_graph_lookup_free(void);
void connected_canon_lookup_init(void);
void connected_canon_lookup_free(void);
void shared_graph_cache_init(SharedGraphCache* shared, int bits, int poly_len);
void shared_graph_cache_free(SharedGraphCache* shared);
void partial_graph_reset(PartialGraphState* st);
int partial_graph_append_checked(PartialGraphState* st, int depth, int pid,
                                 const int* stack, int cols_left);
void build_live_prefix2_tasks(PrefixId** live_i_out, PrefixId** live_j_out,
                              long long* live_count_out);
void dfs(int depth, int min_idx, int* stack, CanonState* canon_state,
         const PartialGraphState* partial_graph, RowGraphCache* cache,
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
                                LocalTaskQueue* queue, ProfileStats* profile,
                                long long total_tasks, long long progress_report_step,
                                double start_time, long long* pending_completed,
                                TaskTimingStats* task_timing,
                                QueueSubtaskTimingStats* queue_subtask_stats);
void solve_graph_poly(const Graph* input_g, RowGraphCache* cache, RowGraphCache* raw_cache,
                      NautyWorkspace* ws, long long* local_canon_calls,
                      long long* local_cache_hits, long long* local_raw_cache_hits,
                      ProfileStats* profile, GraphPoly* out_result);

#endif
