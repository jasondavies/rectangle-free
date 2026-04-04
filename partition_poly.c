#include "src/partition_poly.h"

// --- GLOBALS ---
int num_partitions = 0;
int perm_count = 0;
int max_partition_capacity = 0;
int max_complex_per_partition = 0;
Partition* partitions = NULL;
int (*perms)[MAX_ROWS] = NULL;
uint16_t* perm_table = NULL;
uint16_t* perm_order_by_value = NULL;
uint16_t* perm_value_prefix_end = NULL;
uint16_t* partition_id_lookup = NULL;
uint32_t partition_id_lookup_size = 0;
uint64_t factorial[20];
ComplexMask* overlap_mask = NULL;
ComplexMask* intra_mask = NULL;
Poly* partition_weight_poly = NULL;
uint8_t* partition_weight4 = NULL;
#if RECT_COUNT_K4_FEASIBILITY
uint32_t* pair_shadow_mask = NULL;
uint8_t* pair_shadow_pairs = NULL;
uint8_t* suffix_min_pairs = NULL;
int pair_index[MAX_ROWS][MAX_ROWS];
int num_row_pairs = 0;
int min_partition_pairs = 0;
#endif
PrefixId* g_live_prefix2_i = NULL;
PrefixId* g_live_prefix2_j = NULL;
long long g_live_prefix2_count = 0;

long long completed_tasks = 0;
Poly global_poly = {0};

int g_rows = DEFAULT_ROWS;
int g_cols = DEFAULT_COLS;
ProgressReporter progress_reporter;
int g_use_raw_cache = 1;
long long progress_last_reported = 0;
int g_adaptive_subdivide = 0;
int g_adaptive_max_depth = 3;
long long g_adaptive_work_budget = 0;
__thread ProfileStats* tls_profile = NULL;
__thread GraphHardStats* tls_hard_graph_stats = NULL;
__thread long long* tls_adaptive_work_counter = NULL;
__thread SharedGraphCacheExporter* tls_shared_cache_exporter = NULL;
const char* g_task_times_out_path = NULL;
long long g_task_times_first_task = 0;
long long g_task_times_count = 0;
double* g_task_times_values = NULL;
int g_effective_prefix_depth = 0;
double g_queue_profile_report_step = 0.0;
int g_shared_cache_merge = 0;
int g_shared_cache_bits = 16;
int g_profile_separators = 0;
SharedGraphCache* g_shared_graph_cache = NULL;

int g_small_graph_lookup_ready = 0;
double g_small_graph_lookup_init_time = 0.0;
int g_small_graph_lookup_loaded_from_file = 0;
int32_t* g_small_graph_lookup_coeffs[SMALL_GRAPH_LOOKUP_MAX_N + 1] = {0};
uint8_t g_small_graph_edge_u[SMALL_GRAPH_LOOKUP_MAX_N + 1][21];
uint8_t g_small_graph_edge_v[SMALL_GRAPH_LOOKUP_MAX_N + 1][21];
uint32_t g_small_graph_graph_count[SMALL_GRAPH_LOOKUP_MAX_N + 1] = {0};
uint8_t g_small_graph_edge_count[SMALL_GRAPH_LOOKUP_MAX_N + 1] = {0};
ConnectedCanonLookupEntry* g_connected_canon_lookup = NULL;
uint32_t g_connected_canon_lookup_count = 0;
int g_connected_canon_lookup_ready = 0;
int g_connected_canon_lookup_loaded = 0;
int g_connected_canon_lookup_n = 0;
double g_connected_canon_lookup_load_time = 0.0;

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

static void unrank_prefix2(long long rank, int* i, int* j);
static void unrank_prefix3(long long rank, int* i, int* j, int* k);
static void unrank_prefix4(long long rank, int* i, int* j, int* k, int* l);
static inline long long repeated_combo_count(int values, int slots);

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
// Graph solving and canonical cache logic live separately.
#include "src/solver.c"
// Main/orchestration code lives separately to keep the core solver readable.
#include "src/main.c"
