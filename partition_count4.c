#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <omp.h>
#include "progress_util.h"

// --- CONFIGURATION ---
#define DEFAULT_ROWS 6
#define DEFAULT_COLS 6
#define K_COLOURS 4
#define MAX_ROWS 6
#define MAX_COLS 8
#define MAX_ROW_PAIRS ((MAX_ROWS * (MAX_ROWS - 1)) / 2)

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

#define PERM_SIZE 720      // 6!
#define MAX_PARTITIONS 300 // Sufficient for B(6)=203
// Cache settings - tuned for better locality
#define CACHE_BITS 18
#define CACHE_SIZE (1 << CACHE_BITS)
#define CACHE_MASK (CACHE_SIZE - 1)
#define CACHE_PROBE 16

// Lookaside cache for exact labelled graphs before nauty canonicalisation.
#define RAW_CACHE_BITS 13
#define RAW_CACHE_SIZE (1 << RAW_CACHE_BITS)
#define RAW_CACHE_MASK (RAW_CACHE_SIZE - 1)
#define RAW_CACHE_PROBE 8

// --- DATA TYPES ---

typedef unsigned __int128 Count;

typedef struct {
    int rows;
    int cols;
    long long task_start;
    long long task_end;
    long long full_tasks;
    long long task_stride;
    long long task_offset;
} CountFileMeta;

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

typedef struct {
    Graph g;
    int base[MAX_COLS];
    uint8_t pair_count[MAX_ROW_PAIRS];
    uint8_t pair_sum;
    uint8_t remaining_capacity;
    uint16_t full_pair_mask;
    uint8_t last_base;
    uint8_t last_num_new;
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
    AdjWord* adj;
    Count* values;
    int mask;
    int probe;
} CountCache;

typedef struct {
    unsigned long long prepare_push_calls;
    unsigned long long prepare_push_rejects;
    unsigned long long append_calls;
    unsigned long long append_pair_overflows;
    unsigned long long feasible_calls;
    unsigned long long feasible_core_empty_accepts;
    unsigned long long feasible_capacity_prunes;
    unsigned long long feasible_k5_prunes;
    unsigned long long feasible_exact_calls;
    unsigned long long feasible_colour_prunes;
    unsigned long long feasible_exact_failures;
    unsigned long long count_calls;
    double prepare_push_time;
    double feasible_time;
    double count_time;
} ProfileStats;

// --- GLOBALS ---
static int num_partitions = 0;
static Partition partitions[MAX_PARTITIONS];
static int perms[PERM_SIZE][MAX_ROWS];
static uint8_t perm_table[MAX_PARTITIONS][PERM_SIZE];
static uint64_t factorial[20];
static uint32_t overlap_mask[MAX_PARTITIONS][MAX_PARTITIONS][MAX_ROWS];
static uint32_t intra_mask[MAX_PARTITIONS][MAX_ROWS];
static uint16_t pair_shadow_mask[MAX_PARTITIONS];
static uint8_t pair_shadow_pairs[MAX_PARTITIONS];
static uint8_t suffix_min_pairs[MAX_PARTITIONS];
static uint8_t partition_weight4[MAX_PARTITIONS];
static int pair_index[MAX_ROWS][MAX_ROWS];
static int num_row_pairs = 0;
static int min_partition_pairs = 0;

static volatile long long completed_tasks = 0;
static Count global_count = 0;

static int g_rows = DEFAULT_ROWS;
static int g_cols = DEFAULT_COLS;
static int g_profile_enabled = 0;
static ProgressReporter progress_reporter;

#define DEFAULT_PROGRESS_UPDATES 2000
#define FULL_COLOUR_MASK ((uint8_t)((1U << K_COLOURS) - 1U))

static inline void maybe_report_progress(long long done, long long total_tasks, long long report_step,
                                         long long* last_reported, double start_time) {
    progress_reporter.last_reported = *last_reported;
    progress_reporter_maybe_report(&progress_reporter, done, total_tasks, report_step,
                                   start_time, omp_get_wtime());
    *last_reported = progress_reporter.last_reported;
}

static inline void complete_task_and_report(long long total_tasks, long long report_step,
                                            long long* last_reported, double start_time) {
    long long done = 0;
    #pragma omp atomic capture
    done = ++completed_tasks;
    maybe_report_progress(done, total_tasks, report_step, last_reported, start_time);
}

static void* checked_aligned_alloc(size_t alignment, size_t size, const char* label) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        fprintf(stderr, "Failed to allocate %s (%zu bytes)\n", label, size);
        exit(1);
    }
    return ptr;
}

static Count parse_u128_or_die(const char* text, const char* label) {
    if (!text || !*text) {
        fprintf(stderr, "Missing integer for %s\n", label);
        exit(1);
    }

    const unsigned char* p = (const unsigned char*)text;
    if (*p == '+') {
        p++;
    }
    if (*p == '-') {
        fprintf(stderr, "Negative integer for %s: %s\n", label, text);
        exit(1);
    }
    if (*p == '\0') {
        fprintf(stderr, "Invalid integer for %s: %s\n", label, text);
        exit(1);
    }

    Count value = 0;
    while (*p) {
        if (*p < '0' || *p > '9') {
            fprintf(stderr, "Invalid integer for %s: %s\n", label, text);
            exit(1);
        }
        value = value * 10 + (*p - '0');
        p++;
    }
    return value;
}

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

static void trim_newline(char* s) {
    if (!s) return;
    size_t len = strlen(s);
    while (len > 0 && (s[len - 1] == '\n' || s[len - 1] == '\r')) {
        s[--len] = '\0';
    }
}

static void print_count(Count n) {
    if (n == 0) { printf("0"); return; }
    char str[50];
    int idx = 0;
    while (n > 0) {
        str[idx++] = (int)(n % 10) + '0';
        n /= 10;
    }
    for (int i = idx - 1; i >= 0; i--) putchar(str[i]);
}

static void usage(const char* prog) {
    fprintf(stderr,
            "Usage:\n"
            "  %s [rows cols] [--profile] [--prefix-depth N] [--task-start N] [--task-end N] [--task-stride N] [--task-offset N] [--count-out FILE]\n"
            "  %s --merge [--count-out FILE] INPUT...\n"
            "\n"
            "Notes:\n"
            "  --profile prints prune counters and time split for key hot paths.\n"
            "  --prefix-depth may be 2, 3, or 4; otherwise the solver picks a default.\n"
            "  --task-start/--task-end define a half-open task range [start, end).\n"
            "  --task-stride/--task-offset select interleaved tasks within that range.\n"
            "  For prefix depth d, tasks correspond to sorted feasible d-column prefixes.\n",
            prog, prog);
}

static long long normalise_task_offset(long long task_stride, long long task_offset) {
    long long normalised = task_offset % task_stride;
    if (normalised < 0) normalised += task_stride;
    return normalised;
}

static long long first_selected_task(long long task_start, long long task_end,
                                     long long task_stride, long long task_offset) {
    if (task_start >= task_end) return task_end;
    long long normalised_offset = normalise_task_offset(task_stride, task_offset);
    long long remainder = task_start % task_stride;
    if (remainder < 0) remainder += task_stride;
    long long delta = (normalised_offset - remainder + task_stride) % task_stride;
    long long first = task_start + delta;
    return first < task_end ? first : task_end;
}

static long long count_selected_tasks(long long task_start, long long task_end,
                                      long long task_stride, long long task_offset) {
    long long first = first_selected_task(task_start, task_end, task_stride, task_offset);
    if (first >= task_end) return 0;
    return 1 + ((task_end - 1 - first) / task_stride);
}

static void write_count_file(const char* path, Count count, const CountFileMeta* meta) {
    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Failed to open %s for writing\n", path);
        exit(1);
    }

    fprintf(f, "RECT_COUNT4_V1\n");
    fprintf(f, "rows %d\n", meta->rows);
    fprintf(f, "cols %d\n", meta->cols);
    fprintf(f, "task_start %lld\n", meta->task_start);
    fprintf(f, "task_end %lld\n", meta->task_end);
    fprintf(f, "full_tasks %lld\n", meta->full_tasks);
    fprintf(f, "task_stride %lld\n", meta->task_stride);
    fprintf(f, "task_offset %lld\n", meta->task_offset);
    fprintf(f, "count ");
    if (count == 0) {
        fputc('0', f);
    } else {
        char digits[64];
        int idx = 0;
        while (count > 0) {
            digits[idx++] = (char)('0' + (int)(count % 10));
            count /= 10;
        }
        while (idx-- > 0) fputc(digits[idx], f);
    }
    fputc('\n', f);
    fprintf(f, "end\n");

    if (fclose(f) != 0) {
        fprintf(stderr, "Failed to close %s\n", path);
        exit(1);
    }
}

static void read_count_file(const char* path, Count* count, CountFileMeta* meta) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open %s for reading\n", path);
        exit(1);
    }

    char line[512];
    if (!fgets(line, sizeof(line), f)) {
        fprintf(stderr, "Failed to read header from %s\n", path);
        exit(1);
    }
    trim_newline(line);
    if (strcmp(line, "RECT_COUNT4_V1") != 0) {
        fprintf(stderr, "Invalid count file header in %s\n", path);
        exit(1);
    }

    *count = 0;
    meta->rows = -1;
    meta->cols = -1;
    meta->task_start = 0;
    meta->task_end = 0;
    meta->full_tasks = -1;
    meta->task_stride = 1;
    meta->task_offset = 0;

    while (fgets(line, sizeof(line), f)) {
        trim_newline(line);
        if (strcmp(line, "end") == 0) break;
        if (sscanf(line, "rows %d", &meta->rows) == 1) continue;
        if (sscanf(line, "cols %d", &meta->cols) == 1) continue;
        if (sscanf(line, "task_start %lld", &meta->task_start) == 1) continue;
        if (sscanf(line, "task_end %lld", &meta->task_end) == 1) continue;
        if (sscanf(line, "full_tasks %lld", &meta->full_tasks) == 1) continue;
        if (sscanf(line, "task_stride %lld", &meta->task_stride) == 1) continue;
        if (sscanf(line, "task_offset %lld", &meta->task_offset) == 1) continue;
        if (strncmp(line, "count ", 6) == 0) {
            *count = parse_u128_or_die(line + 6, path);
            continue;
        }
        fprintf(stderr, "Unrecognised line in %s: %s\n", path, line);
        exit(1);
    }

    if (meta->rows < 0 || meta->cols < 0 || meta->full_tasks < 0 ||
        meta->task_stride <= 0) {
        fprintf(stderr, "Incomplete metadata in %s\n", path);
        exit(1);
    }
    meta->task_offset = normalise_task_offset(meta->task_stride, meta->task_offset);

    if (fclose(f) != 0) {
        fprintf(stderr, "Failed to close %s\n", path);
        exit(1);
    }
}

static int run_merge_mode(const char* prog, const char* count_out_path, int input_count, char** inputs) {
    if (input_count <= 0) {
        usage(prog);
        return 1;
    }

    Count merged = 0;
    CountFileMeta merged_meta = {0};
    long long covered_tasks = 0;
    unsigned char* task_seen = NULL;

    for (int i = 0; i < input_count; i++) {
        Count current = 0;
        CountFileMeta current_meta;
        read_count_file(inputs[i], &current, &current_meta);

        if (current_meta.task_start < 0 ||
            current_meta.task_end < current_meta.task_start ||
            current_meta.task_end > current_meta.full_tasks ||
            current_meta.task_offset < 0 ||
            current_meta.task_offset >= current_meta.task_stride) {
            fprintf(stderr, "Invalid task selection in shard: %s\n", inputs[i]);
            free(task_seen);
            return 1;
        }

        if (i == 0) {
            merged_meta = current_meta;
            task_seen = (unsigned char*)calloc((size_t)merged_meta.full_tasks, sizeof(unsigned char));
            if (!task_seen) {
                fprintf(stderr, "Failed to allocate merge task bitmap\n");
                return 1;
            }
        } else if (current_meta.rows != merged_meta.rows ||
                   current_meta.cols != merged_meta.cols ||
                   current_meta.full_tasks != merged_meta.full_tasks) {
            fprintf(stderr, "Incompatible count shard: %s\n", inputs[i]);
            free(task_seen);
            return 1;
        }

        for (long long task = first_selected_task(current_meta.task_start, current_meta.task_end,
                                                  current_meta.task_stride, current_meta.task_offset);
             task < current_meta.task_end;
             task += current_meta.task_stride) {
            if (task_seen[task]) {
                fprintf(stderr, "Overlapping shard task %lld in %s\n", task, inputs[i]);
                free(task_seen);
                return 1;
            }
            task_seen[task] = 1;
            covered_tasks++;
        }

        merged += current;
    }

    if (!task_seen) {
        fprintf(stderr, "Failed to allocate merge task tracking\n");
        return 1;
    }

    long long min_task = -1;
    long long max_task = -1;
    for (long long task = 0; task < merged_meta.full_tasks; task++) {
        if (task_seen[task]) {
            if (min_task < 0) min_task = task;
            max_task = task + 1;
        }
    }
    if (min_task < 0) {
        min_task = 0;
        max_task = 0;
    }

    merged_meta.task_start = min_task;
    merged_meta.task_end = max_task;
    merged_meta.task_stride = 1;
    merged_meta.task_offset = 0;

    int contiguous_cover = 1;
    for (long long task = min_task; task < max_task; task++) {
        if (!task_seen[task]) {
            contiguous_cover = 0;
            break;
        }
    }

    if (covered_tasks == merged_meta.full_tasks) {
        merged_meta.task_start = 0;
        merged_meta.task_end = merged_meta.full_tasks;
        contiguous_cover = 1;
    }

    if (!contiguous_cover && input_count == 1) {
        CountFileMeta single_meta;
        read_count_file(inputs[0], &merged, &single_meta);
        merged_meta = single_meta;
    } else if (!contiguous_cover && count_out_path) {
        fprintf(stderr,
                "Cannot write merged shard %s: input tasks are non-contiguous and incomplete\n",
                count_out_path);
        free(task_seen);
        return 1;
    }

    printf("Merged %d shard(s) for %dx%d\n", input_count, merged_meta.rows, merged_meta.cols);
    printf("Covered tasks: %lld / %lld\n", covered_tasks, merged_meta.full_tasks);
    printf("\nT_%d(%d,%d) = ", K_COLOURS, merged_meta.rows, merged_meta.cols);
    print_count(merged);
    printf("\n");

    if (count_out_path) {
        write_count_file(count_out_path, merged, &merged_meta);
        printf("\nWrote merged count to %s\n", count_out_path);
    }

    free(task_seen);
    return 0;
}

// --- INITIALIZATION ---

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
}

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

static uint8_t eval_partition_weight4(const Partition* part) {
    int value = 1;
    for (int i = 0; i < part->num_singletons; i++) {
        int factor = K_COLOURS - part->num_complex - i;
        if (factor <= 0) return 0;
        value *= factor;
    }
    return (uint8_t)value;
}

static void build_partition_shadow_table(void) {
    min_partition_pairs = MAX_ROW_PAIRS;
    memset(pair_shadow_mask, 0, sizeof(pair_shadow_mask));
    memset(pair_shadow_pairs, 0, sizeof(pair_shadow_pairs));
    memset(suffix_min_pairs, 0, sizeof(suffix_min_pairs));
    memset(partition_weight4, 0, sizeof(partition_weight4));

    for (int pid = 0; pid < num_partitions; pid++) {
        uint16_t shadow = 0;
        const Partition* part = &partitions[pid];
        for (int ci = 0; ci < part->num_complex; ci++) {
            int block = part->complex_blocks[ci];
            uint32_t mask = part->block_masks[block];
            for (int i = 0; i < g_rows; i++) {
                if (((mask >> i) & 1U) == 0) continue;
                for (int j = i + 1; j < g_rows; j++) {
                    if ((mask >> j) & 1U) {
                        shadow |= (uint16_t)(1U << pair_index[i][j]);
                    }
                }
            }
        }
        pair_shadow_mask[pid] = shadow;
        pair_shadow_pairs[pid] = (uint8_t)__builtin_popcount((unsigned)shadow);
        partition_weight4[pid] = eval_partition_weight4(part);
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
        if (part.num_blocks > K_COLOURS) return;
        
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
    for (int i = 0; i < num_partitions; i++) {
        if (memcmp(partitions[i].mapping, map, g_rows) == 0) return i;
    }
    return -1;
}

void build_perm_table() {
    int limit = (int)factorial[g_rows];
    uint8_t temp[MAX_ROWS];
    
    for (int id = 0; id < num_partitions; id++) {
        for (int pi = 0; pi < PERM_SIZE; pi++) {
            if (pi >= limit) break;
            
            for (int r = 0; r < g_rows; r++) {
                temp[r] = partitions[id].mapping[perms[pi][r]];
            }
            normalize_partition(temp);
            int pid = get_partition_id(temp);
            if (pid < 0 || pid > 255) {
                fprintf(stderr, "partition id out of range in build_perm_table: %d\n", pid);
                exit(1);
            }
            perm_table[id][pi] = (uint8_t)pid;
        }
    }
}

void build_overlap_table() {
    memset(overlap_mask, 0, sizeof(overlap_mask));
    memset(intra_mask, 0, sizeof(intra_mask));
    for (int pid1 = 0; pid1 < num_partitions; pid1++) {
        for (int i1 = 0; i1 < partitions[pid1].num_complex; i1++) {
            uint32_t mask = 0;
            for (int i2 = 0; i2 < partitions[pid1].num_complex; i2++) {
                if (i1 != i2) mask |= (1u << i2);
            }
            intra_mask[pid1][i1] = mask;
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
                overlap_mask[pid1][pid2][i1] = mask;
            }
        }
    }
}

// --- SYMMETRY LOGIC ---

typedef struct {
    int limit;
    int depth;
    uint8_t transformed[PERM_SIZE][MAX_COLS];
    uint8_t insert_pos[MAX_COLS][PERM_SIZE];
    uint8_t stack_vals[MAX_COLS];
    uint16_t active[MAX_COLS + 1][PERM_SIZE];
    int active_count[MAX_COLS + 1];
    int stabilizer[MAX_COLS + 1];
} CanonState;

typedef struct {
    uint8_t depth;
    uint8_t min_idx;
    int stack[MAX_COLS];
    PartialGraphState partial_graph;
} PrefixTask;

void canon_state_reset(CanonState* st, int limit) {
    st->limit = limit;
    st->depth = 0;
    st->active_count[0] = limit;
    for (int p = 0; p < limit; p++) st->active[0][p] = (uint16_t)p;
    st->stabilizer[0] = limit;
}

int canon_state_prepare_push(const CanonState* st, int partition_id, uint8_t* next_insert_pos,
                             int* next_stabilizer, uint16_t* next_active, ProfileStats* profile) {
    double t0 = 0.0;
    if (profile) {
        profile->prepare_push_calls++;
        if (g_profile_enabled) t0 = omp_get_wtime();
    }
    int depth = st->depth;
    int new_depth = depth + 1;
    int stabilizer = 0;
    uint8_t pid = (uint8_t)partition_id;
    for (int p = 0; p < st->limit; p++) {
        const uint8_t* row = st->transformed[p];
        uint8_t val = perm_table[partition_id][p];
        int j = depth;
        while (j > 0 && row[j - 1] > val) j--;
        next_insert_pos[p] = (uint8_t)j;

        int cmp = 0;
        for (int k = 0; k < new_depth; k++) {
            int tv;
            if (k < j) tv = row[k];
            else if (k == j) tv = val;
            else tv = row[k - 1];

            int sv = (k < depth) ? st->stack_vals[k] : pid;
            if (tv < sv) {
                cmp = -1;
                break;
            }
            if (tv > sv) {
                cmp = 1;
                break;
            }
        }
        if (cmp < 0) {
            if (profile) {
                profile->prepare_push_rejects++;
                if (g_profile_enabled) profile->prepare_push_time += omp_get_wtime() - t0;
            }
            return 0;
        }
        if (cmp == 0) {
            next_active[stabilizer++] = (uint16_t)p;
        }
    }
    *next_stabilizer = stabilizer;
    if (profile && g_profile_enabled) profile->prepare_push_time += omp_get_wtime() - t0;
    return 1;
}

void canon_state_commit_push(CanonState* st, int partition_id, const uint8_t* next_insert_pos,
                             int next_stabilizer, const uint16_t* next_active) {
    int depth = st->depth;
    int new_depth = depth + 1;
    st->stack_vals[depth] = (uint8_t)partition_id;
    for (int p = 0; p < st->limit; p++) {
        int j = next_insert_pos[p];
        uint8_t* row = st->transformed[p];
        uint8_t val = perm_table[partition_id][p];
        if (j < depth) {
            memmove(&row[j + 1], &row[j], (size_t)(depth - j) * sizeof(row[0]));
        }
        row[j] = val;
        st->insert_pos[depth][p] = (uint8_t)j;
    }

    memcpy(st->active[new_depth], next_active, (size_t)next_stabilizer * sizeof(next_active[0]));
    st->active_count[new_depth] = next_stabilizer;
    st->stabilizer[new_depth] = next_stabilizer;
    st->depth = new_depth;
}

void canon_state_pop(CanonState* st) {
    int depth = st->depth - 1;
    for (int p = 0; p < st->limit; p++) {
        int j = st->insert_pos[depth][p];
        for (int k = j; k < depth; k++) {
            st->transformed[p][k] = st->transformed[p][k + 1];
        }
    }
    st->depth = depth;
}

static void canon_state_rebuild_from_stack(CanonState* st, const int* stack, int depth) {
    canon_state_reset(st, (int)factorial[g_rows]);
    for (int i = 0; i < depth; i++) {
        uint8_t next_insert_pos[PERM_SIZE];
        uint16_t next_active[PERM_SIZE];
        int next_stabilizer = 0;
        if (!canon_state_prepare_push(st, stack[i], next_insert_pos, &next_stabilizer, next_active, NULL)) {
            fprintf(stderr, "Internal error: failed to rebuild canonical prefix state\n");
            exit(1);
        }
        canon_state_commit_push(st, stack[i], next_insert_pos, next_stabilizer, next_active);
    }
}

long long get_orbit_multiplier_state(const CanonState* st) {
    int stabilizer = st->stabilizer[st->depth];
    return factorial[g_rows] / stabilizer;
}

static int candidate_is_stabiliser_orbit_rep(const CanonState* st, int min_idx, int partition_id) {
    int depth = st->depth;
    int active_count = st->active_count[depth];
    for (int i = 0; i < active_count; i++) {
        int perm_idx = st->active[depth][i];
        int image = perm_table[partition_id][perm_idx];
        if (image >= min_idx && image < partition_id) return 0;
    }
    return 1;
}

// --- NAUTY CANONICALIZATION ---

// Convert our graph to nauty format and compute canonical form
void nauty_workspace_init(NautyWorkspace* ws, int n) {
    int m = SETWORDSNEEDED(n);
    if (n <= ws->nmax && m <= ws->mmax) return;
    free(ws->ng);
    free(ws->cg);
    free(ws->lab);
    free(ws->ptn);
    free(ws->orbits);
    ws->ng = (graph*)malloc((size_t)n * (size_t)m * sizeof(graph));
    ws->cg = (graph*)malloc((size_t)n * (size_t)m * sizeof(graph));
    ws->lab = (int*)malloc((size_t)n * sizeof(int));
    ws->ptn = (int*)malloc((size_t)n * sizeof(int));
    ws->orbits = (int*)malloc((size_t)n * sizeof(int));
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

void get_canonical_graph(Graph* g, Graph* canon, NautyWorkspace* ws) {
    int n = g->n;
    
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
    
    // Initialize labeling
    for (int i = 0; i < n; i++) {
        lab[i] = i;
        ptn[i] = 1;
    }
    ptn[n-1] = 0;
    
    // Set up options for canonical labeling
    DEFAULTOPTIONS_GRAPH(options);
    options.getcanon = TRUE;
    options.defaultptn = TRUE;
    
    statsblk stats;
    
    // Compute canonical form
    densenauty(ng, lab, ptn, orbits, &options, &stats, m, n, cg);
    
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

static inline uint64_t graph_row_mask(int n) {
    if (n >= 64) return ~0ULL;
    if (n <= 0) return 0ULL;
    return (1ULL << n) - 1ULL;
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

static void store_count_cache_entry(CountCache* cache, uint64_t key_hash, uint32_t key_n,
                                    const Graph* g, uint64_t row_mask, Count value) {
    int cache_idx = (int)(key_hash & (uint64_t)cache->mask);
    int best_slot = cache_idx;
    for (int k = 0; k < cache->probe; k++) {
        int p = (cache_idx + k) & cache->mask;
        if (!cache->keys[p].used) {
            best_slot = p;
            break;
        }
    }
    cache->keys[best_slot].key_hash = key_hash;
    cache->keys[best_slot].key_n = key_n;
    for (int i = 0; i < (int)key_n; i++) {
        cache->adj[(size_t)best_slot * MAXN_NAUTY + i] = (AdjWord)(g->adj[i] & row_mask);
    }
    cache->values[best_slot] = value;
    cache->keys[best_slot].used = 1;
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

static void partial_graph_reset(PartialGraphState* st) {
    st->g.n = 0;
    memset(st->g.adj, 0, sizeof(st->g.adj));
    memset(st->base, 0, sizeof(st->base));
    memset(st->pair_count, 0, sizeof(st->pair_count));
    st->pair_sum = 0;
    st->remaining_capacity = (uint8_t)(K_COLOURS * num_row_pairs);
    st->full_pair_mask = 0;
    st->last_base = 0;
    st->last_num_new = 0;
}

static int partial_graph_append(PartialGraphState* st, int depth, int pid, const int* stack,
                                ProfileStats* profile) {
    if (profile) profile->append_calls++;
    uint16_t shadow = pair_shadow_mask[pid];
    uint16_t rem = shadow;
    while (rem) {
        int pair = __builtin_ctz((unsigned)rem);
        if (st->pair_count[pair] >= K_COLOURS) {
            if (profile) profile->append_pair_overflows++;
            return 0;
        }
        rem &= (uint16_t)(rem - 1);
    }

    int base_new = st->g.n;
    int num_complex = partitions[pid].num_complex;
    st->base[depth] = base_new;
    st->last_base = (uint8_t)base_new;
    st->last_num_new = (uint8_t)num_complex;
    st->g.n += num_complex;
    for (int i = 0; i < num_complex; i++) {
        st->g.adj[base_new + i] = 0;
    }

    for (int i1 = 0; i1 < num_complex; i1++) {
        int u = base_new + i1;
        st->g.adj[u] |= ((uint64_t)intra_mask[pid][i1]) << base_new;
    }

    for (int prev = 0; prev < depth; prev++) {
        int prev_pid = stack[prev];
        int prev_base = st->base[prev];
        for (int i1 = 0; i1 < num_complex; i1++) {
            int u = base_new + i1;
            uint32_t mask = overlap_mask[pid][prev_pid][i1];
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

    while (shadow) {
        int pair = __builtin_ctz((unsigned)shadow);
        st->pair_count[pair]++;
        if (st->pair_count[pair] == K_COLOURS) {
            st->full_pair_mask |= (uint16_t)(1U << pair);
        }
        shadow &= (uint16_t)(shadow - 1);
    }
    st->pair_sum = (uint8_t)(st->pair_sum + pair_shadow_pairs[pid]);
    st->remaining_capacity = (uint8_t)(st->remaining_capacity - pair_shadow_pairs[pid]);
    return 1;
}

static int contains_edge_mask(const Graph* g, uint64_t mask) {
    while (mask) {
        int a = __builtin_ctzll(mask);
        uint64_t na = g->adj[a] & mask & ~((1ULL << (a + 1)) - 1ULL);
        if (na) return 1;
        mask &= mask - 1;
    }
    return 0;
}

static int contains_triangle_mask(const Graph* g, uint64_t mask) {
    while (mask) {
        int a = __builtin_ctzll(mask);
        uint64_t na = g->adj[a] & mask & ~((1ULL << (a + 1)) - 1ULL);
        while (na) {
            int b = __builtin_ctzll(na);
            if ((na & g->adj[b]) & ~((1ULL << (b + 1)) - 1ULL)) return 1;
            na &= na - 1;
        }
        mask &= mask - 1;
    }
    return 0;
}

static int contains_k4_mask(const Graph* g, uint64_t mask) {
    while (mask) {
        int a = __builtin_ctzll(mask);
        uint64_t na = g->adj[a] & mask & ~((1ULL << (a + 1)) - 1ULL);
        while (na) {
            int b = __builtin_ctzll(na);
            uint64_t nb = (na & g->adj[b]) & ~((1ULL << (b + 1)) - 1ULL);
            while (nb) {
                int c = __builtin_ctzll(nb);
                if ((nb & g->adj[c]) & ~((1ULL << (c + 1)) - 1ULL)) return 1;
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

    uint64_t old_mask = st->last_base > 0 ? ((1ULL << st->last_base) - 1ULL) : 0ULL;
    int base_new = st->last_base;
    int num_new = st->last_num_new;

    for (int i = 0; i < num_new; i++) {
        int u = base_new + i;
        if (contains_k4_mask(&st->g, st->g.adj[u] & old_mask)) return 1;
    }

    for (int i = 0; i < num_new; i++) {
        int u = base_new + i;
        for (int j = i + 1; j < num_new; j++) {
            int v = base_new + j;
            if (contains_triangle_mask(&st->g, st->g.adj[u] & st->g.adj[v] & old_mask)) return 1;
        }
    }

    if (num_new >= 3) {
        uint64_t common = old_mask;
        for (int i = 0; i < 3; i++) common &= st->g.adj[base_new + i];
        if (contains_edge_mask(&st->g, common)) return 1;
    }

    return 0;
}

static int choose_dsat_vertex(const Graph* g, const int8_t* colour, const uint8_t* saturation) {
    int best = -1;
    int best_sat = -1;
    int best_deg = -1;
    for (int v = 0; v < g->n; v++) {
        if (colour[v] >= 0) continue;
        int sat = __builtin_popcount((unsigned)saturation[v]);
        int deg = __builtin_popcountll(g->adj[v]);
        if (sat > best_sat || (sat == best_sat && deg > best_deg)) {
            best = v;
            best_sat = sat;
            best_deg = deg;
        }
    }
    return best;
}

static int dsatur_is_4_colourable(const Graph* g, int coloured,
                                  const int8_t* colour, const uint8_t* saturation,
                                  int8_t* solution) {
    if (coloured == g->n) {
        if (solution) memcpy(solution, colour, (size_t)g->n * sizeof(solution[0]));
        return 1;
    }

    int v = choose_dsat_vertex(g, colour, saturation);
    if (v < 0) {
        if (solution) memcpy(solution, colour, (size_t)g->n * sizeof(solution[0]));
        return 1;
    }

    uint8_t available = (uint8_t)(FULL_COLOUR_MASK & ~saturation[v]);
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

        uint64_t neighbours = g->adj[v];
        int stuck = 0;
        while (neighbours) {
            int u = __builtin_ctzll(neighbours);
            if (next_colour[u] < 0) {
                next_saturation[u] |= bit;
                if (next_saturation[u] == FULL_COLOUR_MASK) stuck = 1;
            }
            neighbours &= neighbours - 1;
        }

        if (!stuck && dsatur_is_4_colourable(g, coloured + 1, next_colour, next_saturation, solution)) {
            return 1;
        }
    }
    return 0;
}

static void induced_subgraph(const Graph* src, uint64_t mask, Graph* dst) {
    int verts[MAXN_NAUTY];
    int n = 0;
    while (mask) {
        verts[n++] = __builtin_ctzll(mask);
        mask &= mask - 1;
    }

    dst->n = n;
    memset(dst->adj, 0, (size_t)n * sizeof(dst->adj[0]));
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if ((src->adj[verts[i]] >> verts[j]) & 1ULL) {
                dst->adj[i] |= (1ULL << j);
                dst->adj[j] |= (1ULL << i);
            }
        }
    }
}

static int induced_subgraph_with_vertices(const Graph* src, uint64_t mask, Graph* dst, int* verts) {
    int n = 0;
    uint64_t rem = mask;
    while (rem) {
        verts[n++] = __builtin_ctzll(rem);
        rem &= rem - 1;
    }

    dst->n = n;
    memset(dst->adj, 0, (size_t)n * sizeof(dst->adj[0]));
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if ((src->adj[verts[i]] >> verts[j]) & 1ULL) {
                dst->adj[i] |= 1ULL << j;
                dst->adj[j] |= 1ULL << i;
            }
        }
    }
    return n;
}

static int peel_to_4_core(const Graph* src, Graph* core, int* peel_order, int* peel_len, int* core_vertices) {
    int degree[MAXN_NAUTY];
    uint64_t active = graph_row_mask(src->n);
    uint64_t queue = 0;

    for (int v = 0; v < src->n; v++) {
        degree[v] = __builtin_popcountll(src->adj[v] & active);
        if (degree[v] <= 3) queue |= 1ULL << v;
    }

    *peel_len = 0;
    while (queue) {
        int v = __builtin_ctzll(queue);
        queue &= queue - 1;
        if (((active >> v) & 1ULL) == 0) continue;

        active &= ~(1ULL << v);
        peel_order[(*peel_len)++] = v;

        uint64_t neighbours = src->adj[v] & active;
        while (neighbours) {
            int u = __builtin_ctzll(neighbours);
            degree[u]--;
            if (degree[u] == 3) queue |= 1ULL << u;
            neighbours &= neighbours - 1;
        }
    }

    return induced_subgraph_with_vertices(src, active, core, core_vertices);
}

static int extend_colouring_over_peel(const Graph* g, const int* peel_order, int peel_len, int8_t* colour) {
    for (int i = peel_len - 1; i >= 0; i--) {
        int v = peel_order[i];
        uint8_t used = 0;
        uint64_t neighbours = g->adj[v];
        while (neighbours) {
            int u = __builtin_ctzll(neighbours);
            if (colour[u] >= 0) used |= (uint8_t)(1U << colour[u]);
            neighbours &= neighbours - 1;
        }
        uint8_t available = (uint8_t)(FULL_COLOUR_MASK & ~used);
        if (available == 0) return 0;
        colour[v] = (int8_t)__builtin_ctz((unsigned)available);
    }
    return 1;
}

static int graph_component_colourable(const Graph* g, int8_t* out_colour) {
    if (g->n == 0) return 1;
    if (g->n == 1) {
        if (out_colour) out_colour[0] = 0;
        return 1;
    }

    int start = 0;
    int best_deg = -1;
    for (int v = 0; v < g->n; v++) {
        int deg = __builtin_popcountll(g->adj[v]);
        if (deg > best_deg) {
            best_deg = deg;
            start = v;
        }
    }

    int8_t colour[MAXN_NAUTY];
    uint8_t saturation[MAXN_NAUTY];
    for (int i = 0; i < MAXN_NAUTY; i++) {
        colour[i] = -1;
        saturation[i] = 0;
    }

    colour[start] = 0;
    uint64_t neighbours = g->adj[start];
    while (neighbours) {
        int u = __builtin_ctzll(neighbours);
        saturation[u] |= 1U;
        neighbours &= neighbours - 1;
    }

    return dsatur_is_4_colourable(g, 1, colour, saturation, out_colour);
}

static int is_4_colourable(const Graph* g, int8_t* out_colour) {
    if (out_colour) {
        for (int i = 0; i < g->n; i++) out_colour[i] = -1;
    }

    uint64_t remaining = graph_row_mask(g->n);
    while (remaining) {
        int start = __builtin_ctzll(remaining);
        uint64_t component = 0;
        uint64_t frontier = 1ULL << start;
        while (frontier) {
            component |= frontier;
            uint64_t next = 0;
            uint64_t f = frontier;
            while (f) {
                int v = __builtin_ctzll(f);
                next |= g->adj[v];
                f &= f - 1;
            }
            frontier = next & remaining & ~component;
        }

        remaining &= ~component;
        if (__builtin_popcountll(component) <= 1) {
            if (out_colour) out_colour[start] = 0;
            continue;
        }

        Graph subgraph;
        int verts[MAXN_NAUTY];
        int8_t component_colour[MAXN_NAUTY];
        int sub_n = induced_subgraph_with_vertices(g, component, &subgraph, verts);
        if (!graph_component_colourable(&subgraph, out_colour ? component_colour : NULL)) return 0;
        if (out_colour) {
            for (int i = 0; i < sub_n; i++) out_colour[verts[i]] = component_colour[i];
        }
    }
    return 1;
}

static Count dsatur_count_4_colourings(const Graph* g, int coloured,
                                       const int8_t* colour, const uint8_t* saturation) {
    if (coloured == g->n) return 1;

    int v = choose_dsat_vertex(g, colour, saturation);
    if (v < 0) return 1;

    Count total = 0;
    uint8_t available = (uint8_t)(FULL_COLOUR_MASK & ~saturation[v]);
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

        uint64_t neighbours = g->adj[v];
        int stuck = 0;
        while (neighbours) {
            int u = __builtin_ctzll(neighbours);
            if (next_colour[u] < 0) {
                next_saturation[u] |= bit;
                if (next_saturation[u] == FULL_COLOUR_MASK) stuck = 1;
            }
            neighbours &= neighbours - 1;
        }

        if (!stuck) total += dsatur_count_4_colourings(g, coloured + 1, next_colour, next_saturation);
    }
    return total;
}

static Count graph_component_count_4_colourings(const Graph* g) {
    if (g->n == 0) return 1;

    int start = 0;
    int best_deg = -1;
    for (int v = 0; v < g->n; v++) {
        int deg = __builtin_popcountll(g->adj[v]);
        if (deg > best_deg) {
            best_deg = deg;
            start = v;
        }
    }

    int8_t colour[MAXN_NAUTY];
    uint8_t saturation[MAXN_NAUTY];
    for (int i = 0; i < MAXN_NAUTY; i++) {
        colour[i] = -1;
        saturation[i] = 0;
    }

    colour[start] = 0;
    uint64_t neighbours = g->adj[start];
    while (neighbours) {
        int u = __builtin_ctzll(neighbours);
        saturation[u] |= 1U;
        neighbours &= neighbours - 1;
    }

    return (Count)K_COLOURS * dsatur_count_4_colourings(g, 1, colour, saturation);
}

static Count count_4_colourings(Graph g, CountCache* cache, CountCache* raw_cache, NautyWorkspace* ws,
                                long long* local_canon_calls, long long* local_cache_hits,
                                long long* local_raw_cache_hits, ProfileStats* profile) {
    double t0 = 0.0;
    if (profile) {
        profile->count_calls++;
        if (g_profile_enabled) t0 = omp_get_wtime();
    }
    Count multiplier = 1;

    int changed = 1;
    while (changed && g.n > 0) {
        changed = 0;
        for (int i = 0; i < g.n; i++) {
            uint64_t neighbours = g.adj[i];
            int degree = __builtin_popcountll(neighbours);

            if (degree == 0) {
                multiplier *= (Count)K_COLOURS;
                remove_vertex(&g, i);
                changed = 1;
                i--;
                continue;
            }

            int is_clique = 1;
            uint64_t rem = neighbours;
            while (rem) {
                int u = __builtin_ctzll(rem);
                if ((neighbours & ~g.adj[u]) != (1ULL << u)) {
                    is_clique = 0;
                    break;
                }
                rem &= rem - 1;
            }

            if (is_clique) {
                int factor = K_COLOURS - degree;
                if (factor <= 0) return 0;
                multiplier *= (Count)factor;
                remove_vertex(&g, i);
                changed = 1;
                i--;
            }
        }
    }

    if (g.n == 0) {
        if (profile && g_profile_enabled) profile->count_time += omp_get_wtime() - t0;
        return multiplier;
    }

    uint64_t row_mask = graph_row_mask(g.n);
    uint64_t raw_hash = hash_graph(&g);
    int raw_cache_idx = (int)(raw_hash & (uint64_t)raw_cache->mask);
    for (int k = 0; k < raw_cache->probe; k++) {
        int p = (raw_cache_idx + k) & raw_cache->mask;
        if (raw_cache->keys[p].used && raw_cache->keys[p].key_hash == raw_hash &&
            raw_cache->keys[p].key_n == (uint32_t)g.n) {
            int match = 1;
            for (int i = 0; i < g.n && match; i++) {
                uint64_t row = g.adj[i] & row_mask;
                if (raw_cache->adj[(size_t)p * MAXN_NAUTY + i] != (AdjWord)row) match = 0;
            }
            if (match) {
                (*local_raw_cache_hits)++;
                Count result = multiplier * raw_cache->values[p];
                if (profile && g_profile_enabled) profile->count_time += omp_get_wtime() - t0;
                return result;
            }
        }
    }

    Graph canon;
    get_canonical_graph(&g, &canon, ws);
    (*local_canon_calls)++;

    uint64_t hash = hash_graph(&canon);
    int cache_idx = (int)(hash & (uint64_t)cache->mask);
    for (int k = 0; k < cache->probe; k++) {
        int p = (cache_idx + k) & cache->mask;
        if (cache->keys[p].used && cache->keys[p].key_hash == hash &&
            cache->keys[p].key_n == (uint32_t)canon.n) {
            int match = 1;
            for (int i = 0; i < canon.n && match; i++) {
                uint64_t row = canon.adj[i] & ADJWORD_MASK;
                if (cache->adj[(size_t)p * MAXN_NAUTY + i] != (AdjWord)row) match = 0;
            }
            if (match) {
                (*local_cache_hits)++;
                store_count_cache_entry(raw_cache, raw_hash, (uint32_t)g.n, &g, row_mask, cache->values[p]);
                Count result = multiplier * cache->values[p];
                if (profile && g_profile_enabled) profile->count_time += omp_get_wtime() - t0;
                return result;
            }
        }
    }

    Count result = 1;
    uint64_t remaining = graph_row_mask(g.n);
    int component_count = 0;
    while (remaining) {
        int start = __builtin_ctzll(remaining);
        uint64_t component = 0;
        uint64_t frontier = 1ULL << start;
        while (frontier) {
            component |= frontier;
            uint64_t next = 0;
            uint64_t f = frontier;
            while (f) {
                int v = __builtin_ctzll(f);
                next |= g.adj[v];
                f &= f - 1;
            }
            frontier = next & remaining & ~component;
        }

        remaining &= ~component;
        component_count++;

        Graph subgraph;
        induced_subgraph(&g, component, &subgraph);
        result *= graph_component_count_4_colourings(&subgraph);
    }

    if (component_count == 0) result = 1;

    store_count_cache_entry(cache, hash, (uint32_t)canon.n, &canon, ADJWORD_MASK, result);
    store_count_cache_entry(raw_cache, raw_hash, (uint32_t)g.n, &g, row_mask, result);
    result = multiplier * result;
    if (profile && g_profile_enabled) profile->count_time += omp_get_wtime() - t0;
    return result;
}

static int partial_graph_is_feasible(PartialGraphState* st, int cols_left, ProfileStats* profile) {
    double t0 = 0.0;
    if (profile) {
        profile->feasible_calls++;
        if (g_profile_enabled) t0 = omp_get_wtime();
    }
    if (st->remaining_capacity < min_partition_pairs * cols_left) {
        if (profile) {
            profile->feasible_capacity_prunes++;
            if (g_profile_enabled) profile->feasible_time += omp_get_wtime() - t0;
        }
        return 0;
    }

    Graph core;
    int peel_order[MAXN_NAUTY];
    int peel_len = 0;
    int core_vertices[MAXN_NAUTY];
    int core_n = peel_to_4_core(&st->g, &core, peel_order, &peel_len, core_vertices);
    if (core_n == 0) {
        int8_t colour[MAXN_NAUTY];
        memset(colour, -1, sizeof(colour));
        if (!extend_colouring_over_peel(&st->g, peel_order, peel_len, colour)) {
            if (profile) {
                profile->feasible_exact_calls++;
                profile->feasible_colour_prunes++;
                profile->feasible_exact_failures++;
                if (g_profile_enabled) profile->feasible_time += omp_get_wtime() - t0;
            }
            return 0;
        }
        if (profile) {
            profile->feasible_core_empty_accepts++;
            if (g_profile_enabled) profile->feasible_time += omp_get_wtime() - t0;
        }
        return 1;
    }

    if (partial_graph_new_has_k5(st)) {
        if (profile) {
            profile->feasible_k5_prunes++;
            if (g_profile_enabled) profile->feasible_time += omp_get_wtime() - t0;
        }
        return 0;
    }

    if (profile) profile->feasible_exact_calls++;
    int8_t core_colour[MAXN_NAUTY];
    if (!is_4_colourable(&core, core_colour)) {
        if (profile) {
            profile->feasible_colour_prunes++;
            profile->feasible_exact_failures++;
            if (g_profile_enabled) profile->feasible_time += omp_get_wtime() - t0;
        }
        return 0;
    }

    int8_t colour[MAXN_NAUTY];
    memset(colour, -1, sizeof(colour));
    for (int i = 0; i < core_n; i++) colour[core_vertices[i]] = core_colour[i];
    if (!extend_colouring_over_peel(&st->g, peel_order, peel_len, colour)) {
        if (profile) {
            profile->feasible_colour_prunes++;
            profile->feasible_exact_failures++;
            if (g_profile_enabled) profile->feasible_time += omp_get_wtime() - t0;
        }
        return 0;
    }

    if (profile && g_profile_enabled) profile->feasible_time += omp_get_wtime() - t0;
    return 1;
}

static Count build_structure_weight(int* stack, const CanonState* canon_state) {
    long long mult_coeff = factorial[g_cols];
    int run = 1;
    for (int i = 1; i < g_cols; i++) {
        if (stack[i] == stack[i-1]) run++;
        else {
            mult_coeff /= factorial[run];
            run = 1;
        }
    }
    mult_coeff /= factorial[run];

    long long row_orbit = get_orbit_multiplier_state(canon_state);
    Count weight = (Count)(mult_coeff * row_orbit);

    for (int i = 0; i < g_cols; i++) {
        int pid = stack[i];
        weight *= partition_weight4[pid];
    }
    return weight;
}

static Count solve_structure(int* stack, const Graph* partial_graph, const CanonState* canon_state,
                             CountCache* cache, CountCache* raw_cache, NautyWorkspace* ws,
                             long long* local_canon_calls, long long* local_cache_hits,
                             long long* local_raw_cache_hits, ProfileStats* profile) {
    Count weight = build_structure_weight(stack, canon_state);
    Count graph_count = count_4_colourings(*partial_graph, cache, raw_cache, ws,
                                           local_canon_calls, local_cache_hits, local_raw_cache_hits, profile);
    return weight * graph_count;
}

void dfs(int depth, int min_idx, int* stack, CanonState* canon_state, const PartialGraphState* partial_graph,
         CountCache* cache, CountCache* raw_cache, NautyWorkspace* ws, Count* local_total,
         long long* local_canon_calls, long long* local_cache_hits,
         long long* local_raw_cache_hits, ProfileStats* profile) {
    if (depth == g_cols) {
        Count res = solve_structure(stack, &partial_graph->g, canon_state, cache, raw_cache, ws,
                                    local_canon_calls, local_cache_hits, local_raw_cache_hits, profile);
        *local_total += res;
        return;
    }

    uint8_t next_insert_pos[PERM_SIZE];
    uint16_t next_active[PERM_SIZE];
    int next_stabilizer = 0;
    int cols_left = g_cols - depth - 1;
    for (int i = min_idx; i < num_partitions; i++) {
        uint16_t shadow = pair_shadow_mask[i];
        if (shadow & partial_graph->full_pair_mask) continue;
        if (partial_graph->remaining_capacity < (int)pair_shadow_pairs[i] + (int)suffix_min_pairs[i] * cols_left) continue;
        if (!candidate_is_stabiliser_orbit_rep(canon_state, min_idx, i)) continue;
        if (!canon_state_prepare_push(canon_state, i, next_insert_pos, &next_stabilizer, next_active, profile)) continue;
        stack[depth] = i;
        canon_state_commit_push(canon_state, i, next_insert_pos, next_stabilizer, next_active);
        PartialGraphState next_graph = *partial_graph;
        if (partial_graph_append(&next_graph, depth, i, stack, profile) &&
            partial_graph_is_feasible(&next_graph, cols_left, profile)) {
            dfs(depth + 1, i, stack, canon_state, &next_graph, cache, raw_cache, ws, local_total,
                local_canon_calls, local_cache_hits, local_raw_cache_hits, profile);
        }
        canon_state_pop(canon_state);
    }
}

static void append_prefix_task(PrefixTask** tasks, long long* count, long long* capacity,
                               int depth, int min_idx, const int* stack,
                               const PartialGraphState* partial_graph) {
    if (*count >= *capacity) {
        long long new_capacity = (*capacity == 0) ? 1024 : (*capacity * 2);
        PrefixTask* grown = (PrefixTask*)realloc(*tasks, (size_t)new_capacity * sizeof(PrefixTask));
        if (!grown) {
            fprintf(stderr, "Failed to grow prefix task buffer\n");
            exit(1);
        }
        *tasks = grown;
        *capacity = new_capacity;
    }

    PrefixTask* task = &(*tasks)[*count];
    task->depth = (uint8_t)depth;
    task->min_idx = (uint8_t)min_idx;
    memcpy(task->stack, stack, sizeof(task->stack));
    task->partial_graph = *partial_graph;
    (*count)++;
}

static void collect_prefix_tasks_recursive(int depth, int target_depth, int min_idx, int* stack,
                                           CanonState* canon_state, const PartialGraphState* partial_graph,
                                           PrefixTask** tasks, long long* count, long long* capacity,
                                           ProfileStats* profile) {
    if (depth == target_depth) {
        append_prefix_task(tasks, count, capacity, target_depth, min_idx, stack, partial_graph);
        return;
    }

    uint8_t next_insert_pos[PERM_SIZE];
    uint16_t next_active[PERM_SIZE];
    int next_stabilizer = 0;
    int cols_left = g_cols - depth - 1;

    for (int pid = min_idx; pid < num_partitions; pid++) {
        uint16_t shadow = pair_shadow_mask[pid];
        if (shadow & partial_graph->full_pair_mask) continue;
        if (partial_graph->remaining_capacity < (int)pair_shadow_pairs[pid] + (int)suffix_min_pairs[pid] * cols_left) continue;
        if (!candidate_is_stabiliser_orbit_rep(canon_state, min_idx, pid)) continue;
        if (!canon_state_prepare_push(canon_state, pid, next_insert_pos, &next_stabilizer, next_active, profile)) continue;
        stack[depth] = pid;
        canon_state_commit_push(canon_state, pid, next_insert_pos, next_stabilizer, next_active);
        PartialGraphState next_graph = *partial_graph;
        if (partial_graph_append(&next_graph, depth, pid, stack, profile) &&
            partial_graph_is_feasible(&next_graph, cols_left, profile)) {
            collect_prefix_tasks_recursive(depth + 1, target_depth, pid, stack, canon_state, &next_graph,
                                           tasks, count, capacity, profile);
        }
        canon_state_pop(canon_state);
    }
}

int main(int argc, char** argv) {
    long long task_start = 0;
    long long task_end = -1;
    long long task_stride = 1;
    long long task_offset = 0;
    const char* count_out_path = NULL;
    int prefix_depth_override = -1;
    int merge_mode = 0;
    char** merge_inputs = (char**)malloc((size_t)argc * sizeof(char*));
    if (!merge_inputs) {
        fprintf(stderr, "Failed to allocate merge input list\n");
        return 1;
    }
    int merge_input_count = 0;
    int positional_count = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--merge") == 0) {
            merge_mode = 1;
        } else if (strcmp(argv[i], "--profile") == 0) {
            g_profile_enabled = 1;
        } else if (strcmp(argv[i], "--count-out") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                free(merge_inputs);
                return 1;
            }
            count_out_path = argv[++i];
        } else if (strcmp(argv[i], "--task-start") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                free(merge_inputs);
                return 1;
            }
            task_start = parse_ll_or_die(argv[++i], "--task-start");
        } else if (strcmp(argv[i], "--task-end") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                free(merge_inputs);
                return 1;
            }
            task_end = parse_ll_or_die(argv[++i], "--task-end");
        } else if (strcmp(argv[i], "--task-stride") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                free(merge_inputs);
                return 1;
            }
            task_stride = parse_ll_or_die(argv[++i], "--task-stride");
        } else if (strcmp(argv[i], "--task-offset") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                free(merge_inputs);
                return 1;
            }
            task_offset = parse_ll_or_die(argv[++i], "--task-offset");
        } else if (strcmp(argv[i], "--prefix-depth") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                free(merge_inputs);
                return 1;
            }
            prefix_depth_override = (int)parse_ll_or_die(argv[++i], "--prefix-depth");
        } else if (merge_mode) {
            merge_inputs[merge_input_count++] = argv[i];
        } else if (positional_count == 0) {
            g_rows = (int)parse_ll_or_die(argv[i], "rows");
            positional_count++;
        } else if (positional_count == 1) {
            g_cols = (int)parse_ll_or_die(argv[i], "cols");
            positional_count++;
        } else {
            usage(argv[0]);
            free(merge_inputs);
            return 1;
        }
    }

    if (merge_mode) {
        if (positional_count != 0 || task_start != 0 || task_end != -1 ||
            task_stride != 1 || task_offset != 0 || prefix_depth_override != -1 ||
            g_profile_enabled) {
            usage(argv[0]);
            free(merge_inputs);
            return 1;
        }
        int rc = run_merge_mode(argv[0], count_out_path, merge_input_count, merge_inputs);
        free(merge_inputs);
        return rc;
    }
    free(merge_inputs);

    if (g_rows < 1 || g_cols < 1 || g_rows > MAX_ROWS || g_cols > MAX_COLS) {
        fprintf(stderr, "Rows/cols must be in range 1..%d and 1..%d\n", MAX_ROWS, MAX_COLS);
        return 1;
    }

    // Initialize nauty's thread-local storage
    nauty_check(WORDSIZE, MAXN_NAUTY, MAXN_NAUTY, NAUTYVERSIONID);
    
    // 1. Initialize math tables
    factorial[0] = 1;
    for(int i=1; i<=19; i++) factorial[i] = factorial[i-1]*i;

    // 2. Data structures
    generate_permutations();
    init_pair_index();
    uint8_t buffer[MAX_ROWS] = {0};
    generate_partitions_recursive(0, buffer, -1);
    reorder_partitions_by_hardness();
    
    // 3. Build lookup tables
    build_perm_table();
    build_overlap_table();
    build_partition_shadow_table();
    
    printf("Grid: %dx%d\n", g_rows, g_cols);
    printf("Partitions: %d\n", num_partitions);
    printf("Threads: %d\n", omp_get_max_threads());
    printf("Using nauty for canonical graph caching\n");

    // Build prefix list for work distribution.
    // For wider grids, use 3-column prefixes to reduce long-tail imbalance.
    int prefix_depth = 0;
    if (prefix_depth_override != -1) {
        prefix_depth = prefix_depth_override;
    } else if (g_rows == 6 && g_cols >= 4) {
        prefix_depth = 3;
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

    long long total_prefixes = 0;
    long long prefix_capacity = 0;
    PrefixTask* prefix_tasks = NULL;
    double prefix_generation_time = 0.0;

    if (prefix_depth > 0) {
        CanonState prefix_canon;
        PartialGraphState prefix_graph;
        int prefix_stack[MAX_COLS];
        canon_state_reset(&prefix_canon, (int)factorial[g_rows]);
        partial_graph_reset(&prefix_graph);
        double prefix_start_time = omp_get_wtime();
        collect_prefix_tasks_recursive(0, prefix_depth, 0, prefix_stack, &prefix_canon, &prefix_graph,
                                       &prefix_tasks, &total_prefixes, &prefix_capacity, NULL);
        prefix_generation_time = omp_get_wtime() - prefix_start_time;
    }

    if (prefix_depth > 0) {
        printf("Prefix depth: %d (%lld tasks)\n", prefix_depth, total_prefixes);
        printf("Prefix generation: %.2f seconds\n", prefix_generation_time);
        printf("Prefix task bytes: %zu each, %.2f MiB total\n",
               sizeof(PrefixTask),
               ((double)sizeof(PrefixTask) * (double)total_prefixes) / (1024.0 * 1024.0));
    }

    completed_tasks = 0;
    long long full_tasks = (g_cols == 1) ? (long long)num_partitions : total_prefixes;
    if (task_start < 0) {
        fprintf(stderr, "--task-start must be non-negative\n");
        return 1;
    }
    if (task_stride <= 0) {
        fprintf(stderr, "--task-stride must be positive\n");
        return 1;
    }
    if (task_end < 0) task_end = full_tasks;
    if (task_end < task_start || task_end > full_tasks) {
        fprintf(stderr, "Task range must satisfy 0 <= start <= end <= %lld\n", full_tasks);
        return 1;
    }
    task_offset = normalise_task_offset(task_stride, task_offset);
    long long active_task_start = task_start;
    long long active_task_end = task_end;
    long long total_tasks = count_selected_tasks(active_task_start, active_task_end,
                                                task_stride, task_offset);
    long long first_task = first_selected_task(active_task_start, active_task_end,
                                               task_stride, task_offset);
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
    long long progress_last_reported = 0;
    printf("Task range: [%lld, %lld) of %lld\n", active_task_start, active_task_end, full_tasks);
    printf("Task selection: stride %lld, offset %lld\n", task_stride, task_offset);
    if (total_tasks == 0) {
        printf("No tasks selected; producing the zero count for this shard.\n");
    }
    printf("Progress updates every %lld tasks", progress_report_step);
    if (progress_step_env && *progress_step_env) {
        printf(" (RECT_PROGRESS_STEP override)");
    } else {
        printf(" (target ~%lld updates)", progress_updates);
    }
    printf("\n");
    progress_reporter_init(&progress_reporter, stdout);
    progress_reporter_print_initial(&progress_reporter, total_tasks);

    double start_time = omp_get_wtime();
    
    int num_threads = omp_get_max_threads();
    Count* thread_counts = checked_aligned_alloc(64, (size_t)num_threads * sizeof(Count), "thread_counts");
    for (int i = 0; i < num_threads; i++) thread_counts[i] = 0;
    ProfileStats* thread_profiles = checked_aligned_alloc(64, (size_t)num_threads * sizeof(ProfileStats), "thread_profiles");
    memset(thread_profiles, 0, (size_t)num_threads * sizeof(ProfileStats));

    long long total_canon_calls = 0;
    long long total_cache_hits = 0;
    long long total_raw_cache_hits = 0;

    #pragma omp parallel reduction(+:total_canon_calls, total_cache_hits, total_raw_cache_hits)
    {
        int tid = omp_get_thread_num();
        CountCache cache = {0};
        CountCache raw_cache = {0};
        NautyWorkspace ws;
        memset(&ws, 0, sizeof(ws));
        cache.mask = CACHE_MASK;
        cache.probe = CACHE_PROBE;
        cache.keys = checked_aligned_alloc(64, sizeof(CacheKey) * CACHE_SIZE, "cache_keys");
        cache.adj = checked_aligned_alloc(64, sizeof(AdjWord) * CACHE_SIZE * MAXN_NAUTY, "cache_adj");
        cache.values = checked_aligned_alloc(64, sizeof(Count) * CACHE_SIZE, "cache_values");

        raw_cache.mask = RAW_CACHE_MASK;
        raw_cache.probe = RAW_CACHE_PROBE;
        raw_cache.keys = checked_aligned_alloc(64, sizeof(CacheKey) * RAW_CACHE_SIZE, "raw_cache_keys");
        raw_cache.adj = checked_aligned_alloc(64, sizeof(AdjWord) * RAW_CACHE_SIZE * MAXN_NAUTY, "raw_cache_adj");
        raw_cache.values = checked_aligned_alloc(64, sizeof(Count) * RAW_CACHE_SIZE, "raw_cache_values");

        memset(cache.keys, 0, sizeof(CacheKey) * CACHE_SIZE);
        memset(raw_cache.keys, 0, sizeof(CacheKey) * RAW_CACHE_SIZE);

        int stack[MAX_COLS];
        CanonState canon_state;
        PartialGraphState partial_graph;
        canon_state_reset(&canon_state, (int)factorial[g_rows]);
        partial_graph_reset(&partial_graph);
        long long local_canon_calls = 0;
        long long local_cache_hits = 0;
        long long local_raw_cache_hits = 0;
        ProfileStats local_profile;
        memset(&local_profile, 0, sizeof(local_profile));
        
        if (g_cols == 1) {
            // Original 1-column parallelism (nothing to prefix)
            #pragma omp for schedule(dynamic, 1)
            for (long long i = first_task; i < active_task_end; i += task_stride) {
                stack[0] = i;
                canon_state_reset(&canon_state, (int)factorial[g_rows]);
                partial_graph_reset(&partial_graph);
                uint8_t next_insert_pos[PERM_SIZE];
                uint16_t next_active[PERM_SIZE];
                int next_stabilizer = 0;
                if (!canon_state_prepare_push(&canon_state, (int)i, next_insert_pos, &next_stabilizer, next_active, &local_profile)) {
                    complete_task_and_report(total_tasks, progress_report_step,
                                             &progress_last_reported, start_time);
                    continue;
                }
                canon_state_commit_push(&canon_state, (int)i, next_insert_pos, next_stabilizer, next_active);
                if (partial_graph_append(&partial_graph, 0, (int)i, stack, &local_profile) &&
                    partial_graph_is_feasible(&partial_graph, g_cols - 1, &local_profile)) {
                    dfs(1, (int)i, stack, &canon_state, &partial_graph, &cache, &raw_cache, &ws,
                        &thread_counts[tid], &local_canon_calls, &local_cache_hits, &local_raw_cache_hits,
                        &local_profile);
                }
                canon_state_pop(&canon_state);

                complete_task_and_report(total_tasks, progress_report_step,
                                         &progress_last_reported, start_time);
            }
        } else {
            // Prefix-task parallelism: each task already stores the feasible prefix state.
            if (prefix_depth == 2) {
                #pragma omp for schedule(dynamic, 8)
                for (long long p = first_task; p < active_task_end; p += task_stride) {
                    const PrefixTask* task = &prefix_tasks[p];
                    canon_state_rebuild_from_stack(&canon_state, task->stack, task->depth);
                    partial_graph = task->partial_graph;
                    memcpy(stack, task->stack, sizeof(stack));
                    dfs(task->depth, task->min_idx, stack, &canon_state, &partial_graph, &cache, &raw_cache, &ws,
                        &thread_counts[tid], &local_canon_calls, &local_cache_hits, &local_raw_cache_hits,
                        &local_profile);

                    complete_task_and_report(total_tasks, progress_report_step,
                                             &progress_last_reported, start_time);
                }
            } else {
                #pragma omp for schedule(dynamic, 1)
                for (long long p = first_task; p < active_task_end; p += task_stride) {
                    const PrefixTask* task = &prefix_tasks[p];
                    canon_state_rebuild_from_stack(&canon_state, task->stack, task->depth);
                    partial_graph = task->partial_graph;
                    memcpy(stack, task->stack, sizeof(stack));
                    dfs(task->depth, task->min_idx, stack, &canon_state, &partial_graph, &cache, &raw_cache, &ws,
                        &thread_counts[tid], &local_canon_calls, &local_cache_hits, &local_raw_cache_hits,
                        &local_profile);

                    complete_task_and_report(total_tasks, progress_report_step,
                                             &progress_last_reported, start_time);
                }
            }
        }
        
        total_canon_calls += local_canon_calls;
        total_cache_hits += local_cache_hits;
        total_raw_cache_hits += local_raw_cache_hits;
        thread_profiles[tid] = local_profile;
        
        nauty_workspace_free(&ws);
        free(cache.keys);
        free(cache.adj);
        free(cache.values);
        free(raw_cache.keys);
        free(raw_cache.adj);
        free(raw_cache.values);
    }
    
    ProfileStats profile_total;
    memset(&profile_total, 0, sizeof(profile_total));
    for (int i = 0; i < num_threads; i++) {
        global_count += thread_counts[i];
        profile_total.prepare_push_calls += thread_profiles[i].prepare_push_calls;
        profile_total.prepare_push_rejects += thread_profiles[i].prepare_push_rejects;
        profile_total.append_calls += thread_profiles[i].append_calls;
        profile_total.append_pair_overflows += thread_profiles[i].append_pair_overflows;
        profile_total.feasible_calls += thread_profiles[i].feasible_calls;
        profile_total.feasible_core_empty_accepts += thread_profiles[i].feasible_core_empty_accepts;
        profile_total.feasible_capacity_prunes += thread_profiles[i].feasible_capacity_prunes;
        profile_total.feasible_k5_prunes += thread_profiles[i].feasible_k5_prunes;
        profile_total.feasible_exact_calls += thread_profiles[i].feasible_exact_calls;
        profile_total.feasible_colour_prunes += thread_profiles[i].feasible_colour_prunes;
        profile_total.feasible_exact_failures += thread_profiles[i].feasible_exact_failures;
        profile_total.count_calls += thread_profiles[i].count_calls;
        profile_total.prepare_push_time += thread_profiles[i].prepare_push_time;
        profile_total.feasible_time += thread_profiles[i].feasible_time;
        profile_total.count_time += thread_profiles[i].count_time;
    }
    free(thread_counts);
    free(thread_profiles);
    free(prefix_tasks);

    progress_reporter_finish(&progress_reporter);
    
    double end_time = omp_get_wtime();
    double worker_time = end_time - start_time;
    double total_elapsed = worker_time + prefix_generation_time;

    printf("\nWorker Complete in %.2f seconds.\n", worker_time);
    if (prefix_depth > 0) {
        printf("Total elapsed including prefix generation: %.2f seconds.\n", total_elapsed);
    }
    printf("Canonicalization calls: %lld\n", total_canon_calls);
    printf("Canonical cache hits: %lld (%.1f%%)\n", total_cache_hits,
           total_canon_calls > 0 ? 100.0 * total_cache_hits / total_canon_calls : 0.0);
    printf("Raw cache hits: %lld\n", total_raw_cache_hits);
    if (g_profile_enabled) {
        unsigned long long feasible_prunes = profile_total.feasible_capacity_prunes +
                                             profile_total.feasible_k5_prunes +
                                             profile_total.feasible_colour_prunes;
        printf("Profile:\n");
        printf("  canon_state_prepare_push: %llu calls, %llu rejects, %.3fs\n",
               profile_total.prepare_push_calls, profile_total.prepare_push_rejects,
               profile_total.prepare_push_time);
        printf("  partial_graph_append: %llu calls, %llu pair-overflow prunes\n",
               profile_total.append_calls, profile_total.append_pair_overflows);
        printf("  partial_graph_is_feasible: %llu calls, %llu prunes [capacity %llu, K5 %llu, 4-colour %llu], %.3fs\n",
               profile_total.feasible_calls, feasible_prunes,
               profile_total.feasible_capacity_prunes, profile_total.feasible_k5_prunes,
               profile_total.feasible_colour_prunes, profile_total.feasible_time);
        printf("    4-core empty accepts: %llu\n", profile_total.feasible_core_empty_accepts);
        printf("    exact feasibility calls: %llu\n", profile_total.feasible_exact_calls);
        printf("    exact feasibility failures: %llu\n", profile_total.feasible_exact_failures);
        printf("  count_4_colourings: %llu calls, %.3fs\n",
               profile_total.count_calls, profile_total.count_time);
    }
    
    printf("\nT_%d(%d,%d) = ", K_COLOURS, g_rows, g_cols);
    print_count(global_count);
    printf("\n");

    if (count_out_path) {
        CountFileMeta meta = {
            .rows = g_rows,
            .cols = g_cols,
            .task_start = active_task_start,
            .task_end = active_task_end,
            .full_tasks = full_tasks,
            .task_stride = task_stride,
            .task_offset = task_offset,
        };
        write_count_file(count_out_path, global_count, &meta);
        printf("\nWrote count shard to %s\n", count_out_path);
    }

    return 0;
}
