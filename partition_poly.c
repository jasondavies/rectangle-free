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
