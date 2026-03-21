#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <omp.h>
#include "progress_util.h"

// --- CONFIGURATION ---
#define FIXED_ROWS 7
#define DEFAULT_COLS 7
#define MAX_ROWS 7
#define MAX_COLS 7

// For 7 rows every column has at most 3 complex blocks.
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

#define PERM_SIZE 5040      // 7!
#define MAX_PARTITIONS 900  // B(7)=877
#define MAX_EVALS ((MAX_ROWS * MAX_COLS) + 1)

#define CACHE_BITS 18
#define CACHE_SIZE (1 << CACHE_BITS)
#define CACHE_MASK (CACHE_SIZE - 1)
#define CACHE_PROBE 16

#define RAW_CACHE_BITS 13
#define RAW_CACHE_SIZE (1 << RAW_CACHE_BITS)
#define RAW_CACHE_MASK (RAW_CACHE_SIZE - 1)
#define RAW_CACHE_PROBE 8

#define DEFAULT_PROGRESS_UPDATES 2000
#define TERM_MAP_INITIAL_BITS 12

typedef __int128_t Count;

typedef struct {
    int rows;
    int cols;
    long long task_start;
    long long task_end;
    long long full_tasks;
    long long task_stride;
    long long task_offset;
    int eval_len;
} EvalFileMeta;

typedef struct {
    Count values[MAX_EVALS];
} EvalVec;

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
    AdjWord* adj;
    Count* values;
    int mask;
    int probe;
    int eval_len;
} GraphEvalCache;

typedef struct {
    CacheKey* keys;
    AdjWord* adj;
    Count* values;
    size_t cap;
    size_t sz;
    int eval_len;
} GraphEvalMap;

typedef struct {
    int limit;
    int depth;
    uint16_t transformed[PERM_SIZE][MAX_COLS];
    uint8_t insert_pos[MAX_COLS][PERM_SIZE];
    uint16_t stack_vals[MAX_COLS];
    int stabilizer[MAX_COLS + 1];
} CanonState;

typedef struct {
    int vertex;
    uint32_t old_forbidden;
} DSaturUpdate;

typedef struct {
    int n;
    uint32_t adj[MAXN_NAUTY];
    int degree[MAXN_NAUTY];
    uint32_t uncolored;
    uint32_t forbidden[MAXN_NAUTY];
    Count counts[MAXN_NAUTY + 1];
} DSaturState;

// --- GLOBALS ---
static int num_partitions = 0;
static Partition partitions[MAX_PARTITIONS];
static int perms[PERM_SIZE][MAX_ROWS];
static uint16_t perm_table[MAX_PARTITIONS][PERM_SIZE];
static uint64_t factorial[20];
static uint32_t overlap_mask[MAX_PARTITIONS][MAX_PARTITIONS][MAX_ROWS];
static uint32_t intra_mask[MAX_PARTITIONS][MAX_ROWS];
static EvalVec singleton_eval_factor[MAX_PARTITIONS];
static Count falling_table[MAX_EVALS][MAXN_NAUTY + 1];

static volatile long long completed_tasks = 0;
static EvalVec global_eval = {0};

static int g_rows = FIXED_ROWS;
static int g_cols = DEFAULT_COLS;
static int g_eval_len = 0;
static ProgressReporter progress_reporter;

static inline void maybe_report_progress(long long done, long long total_tasks, long long report_step,
                                         long long* last_reported, double start_time) {
    (void)last_reported;
    progress_reporter_maybe_report(&progress_reporter, done, total_tasks, report_step,
                                   start_time, omp_get_wtime());
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

static Count parse_i128_or_die(const char* text, const char* label) {
    if (!text || !*text) {
        fprintf(stderr, "Missing integer for %s\n", label);
        exit(1);
    }

    int negative = 0;
    const unsigned char* p = (const unsigned char*)text;
    if (*p == '+' || *p == '-') {
        negative = (*p == '-');
        p++;
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
    return negative ? -value : value;
}

static void trim_newline(char* s) {
    if (!s) return;
    size_t len = strlen(s);
    while (len > 0 && (s[len - 1] == '\n' || s[len - 1] == '\r')) {
        s[--len] = '\0';
    }
}

static void print_i128(Count n) {
    if (n == 0) {
        printf("0");
        return;
    }
    if (n < 0) {
        printf("-");
        n = -n;
    }
    char str[64];
    int idx = 0;
    while (n > 0) {
        str[idx++] = (char)('0' + (int)(n % 10));
        n /= 10;
    }
    for (int i = idx - 1; i >= 0; i--) putchar(str[i]);
}

static void eval_zero(EvalVec* v) {
    memset(v->values, 0, sizeof(v->values));
}

static void eval_one(EvalVec* v) {
    for (int i = 0; i < g_eval_len; i++) v->values[i] = 1;
    for (int i = g_eval_len; i < MAX_EVALS; i++) v->values[i] = 0;
}

static void eval_add_inplace(EvalVec* dst, const EvalVec* src) {
    for (int i = 0; i < g_eval_len; i++) dst->values[i] += src->values[i];
}

static EvalVec eval_add(EvalVec a, const EvalVec* b) {
    eval_add_inplace(&a, b);
    return a;
}

static void eval_scale_inplace(EvalVec* v, Count scale) {
    for (int i = 0; i < g_eval_len; i++) v->values[i] *= scale;
}

static void eval_pointwise_mul_inplace(EvalVec* dst, const EvalVec* src) {
    for (int i = 0; i < g_eval_len; i++) dst->values[i] *= src->values[i];
}

static EvalVec eval_pointwise_mul(EvalVec a, const EvalVec* b) {
    eval_pointwise_mul_inplace(&a, b);
    return a;
}

static void eval_mul_linear_inplace(EvalVec* dst, int c) {
    for (int q = 0; q < g_eval_len; q++) {
        dst->values[q] *= (Count)(q - c);
    }
}

static void print_newton_poly(const EvalVec* values) {
    Count diffs[MAX_EVALS];
    for (int i = 0; i < g_eval_len; i++) diffs[i] = values->values[i];

    printf("P(x) = ");
    int first = 1;
    for (int k = 0; k < g_eval_len; k++) {
        Count coeff = diffs[0];
        if (coeff != 0) {
            if (!first) {
                if (coeff > 0) printf(" + ");
                else printf(" - ");
            } else if (coeff < 0) {
                printf("-");
            }

            Count abs_coeff = coeff < 0 ? -coeff : coeff;
            if (k == 0 || abs_coeff != 1) {
                print_i128(abs_coeff);
                if (k > 0) printf("*");
            }
            if (k > 0) {
                printf("binom(x,%d)", k);
            }
            first = 0;
        }

        for (int i = 0; i < g_eval_len - k - 1; i++) {
            diffs[i] = diffs[i + 1] - diffs[i];
        }
    }
    if (first) printf("0");
    printf("\n");
}

static void usage(const char* prog) {
    fprintf(stderr,
            "Usage:\n"
            "  %s [cols] [--task-start N] [--task-end N] [--task-stride N] [--task-offset N] [--prefix-depth 2] [--poly-out FILE]\n"
            "  %s 7 cols [--task-start N] [--task-end N] [--task-stride N] [--task-offset N] [--prefix-depth 2] [--poly-out FILE]\n"
            "  %s --merge [--poly-out FILE] INPUT...\n"
            "\n"
            "Notes:\n"
            "  - this solver is specialised to 7 rows and currently supports up to 7 columns.\n"
            "  - output is stored and printed in evaluation/Newton form, not dense monomial form.\n"
            "  - --task-start/--task-end define a half-open task range [start, end).\n"
            "  - --task-stride/--task-offset select interleaved tasks within that range.\n"
            "  - only --prefix-depth 2 is supported for 7-row runs.\n",
            prog, prog, prog);
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

static void write_eval_file(const char* path, const EvalVec* values, const EvalFileMeta* meta) {
    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Failed to open %s for writing\n", path);
        exit(1);
    }

    fprintf(f, "RECT_EVAL_V1\n");
    fprintf(f, "rows %d\n", meta->rows);
    fprintf(f, "cols %d\n", meta->cols);
    fprintf(f, "task_start %lld\n", meta->task_start);
    fprintf(f, "task_end %lld\n", meta->task_end);
    fprintf(f, "full_tasks %lld\n", meta->full_tasks);
    fprintf(f, "task_stride %lld\n", meta->task_stride);
    fprintf(f, "task_offset %lld\n", meta->task_offset);
    fprintf(f, "eval_len %d\n", meta->eval_len);
    for (int i = 0; i < meta->eval_len; i++) {
        fprintf(f, "value %d ", i);
        Count value = values->values[i];
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

    if (fclose(f) != 0) {
        fprintf(stderr, "Failed to close %s\n", path);
        exit(1);
    }
}

static void read_eval_file(const char* path, EvalVec* values, EvalFileMeta* meta) {
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
    if (strcmp(line, "RECT_EVAL_V1") != 0) {
        fprintf(stderr, "Invalid evaluation file header in %s\n", path);
        exit(1);
    }

    eval_zero(values);
    meta->rows = -1;
    meta->cols = -1;
    meta->task_start = 0;
    meta->task_end = 0;
    meta->full_tasks = -1;
    meta->task_stride = 1;
    meta->task_offset = 0;
    meta->eval_len = -1;

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
        if (sscanf(line, "eval_len %d", &meta->eval_len) == 1) continue;
        if (strncmp(line, "value ", 6) == 0) {
            char* p = line + 6;
            char* end = NULL;
            long idx = strtol(p, &end, 10);
            if (!end || *end != ' ' || idx < 0 || idx >= MAX_EVALS) {
                fprintf(stderr, "Invalid value line in %s: %s\n", path, line);
                exit(1);
            }
            while (*end == ' ') end++;
            values->values[idx] = parse_i128_or_die(end, path);
            continue;
        }
        fprintf(stderr, "Unrecognised line in %s: %s\n", path, line);
        exit(1);
    }

    if (meta->rows != FIXED_ROWS || meta->cols < 1 || meta->cols > MAX_COLS ||
        meta->full_tasks < 0 || meta->task_stride <= 0 || meta->eval_len < 1 ||
        meta->eval_len > MAX_EVALS) {
        fprintf(stderr, "Incomplete metadata in %s\n", path);
        exit(1);
    }
    meta->task_offset = normalise_task_offset(meta->task_stride, meta->task_offset);

    if (fclose(f) != 0) {
        fprintf(stderr, "Failed to close %s\n", path);
        exit(1);
    }
}

static int run_merge_mode(const char* prog, const char* poly_out_path, int input_count, char** inputs) {
    if (input_count <= 0) {
        usage(prog);
        return 1;
    }

    EvalVec merged;
    eval_zero(&merged);
    EvalFileMeta merged_meta = {0};
    long long covered_tasks = 0;
    unsigned char* task_seen = NULL;

    for (int i = 0; i < input_count; i++) {
        EvalVec current;
        EvalFileMeta current_meta;
        read_eval_file(inputs[i], &current, &current_meta);

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
                   current_meta.full_tasks != merged_meta.full_tasks ||
                   current_meta.eval_len != merged_meta.eval_len) {
            fprintf(stderr, "Incompatible evaluation shard: %s\n", inputs[i]);
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

        eval_add_inplace(&merged, &current);
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
        EvalFileMeta single_meta;
        read_eval_file(inputs[0], &merged, &single_meta);
        merged_meta = single_meta;
    } else if (!contiguous_cover && poly_out_path) {
        fprintf(stderr,
                "Cannot write merged shard %s: input tasks are non-contiguous and incomplete\n",
                poly_out_path);
        free(task_seen);
        return 1;
    }

    printf("Merged %d shard(s) for %dx%d\n", input_count, merged_meta.rows, merged_meta.cols);
    printf("Covered tasks: %lld / %lld\n", covered_tasks, merged_meta.full_tasks);
    printf("\nNewton-Form Polynomial:\n");
    g_eval_len = merged_meta.eval_len;
    print_newton_poly(&merged);

    printf("\nValues:\n");
    printf("P(4) = ");
    print_i128(merged.values[4]);
    printf("\n");
    printf("P(5) = ");
    print_i128(merged.values[5]);
    printf("\n");

    if (poly_out_path) {
        write_eval_file(poly_out_path, &merged, &merged_meta);
        printf("\nWrote merged evaluation shard to %s\n", poly_out_path);
    }

    free(task_seen);
    return 0;
}

// --- INITIALISATION ---

static void generate_permutations(void) {
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
        int temp = p[i];
        p[i] = p[j];
        p[j] = temp;
        int l = i + 1;
        int r = g_rows - 1;
        while (l < r) {
            temp = p[l];
            p[l] = p[r];
            p[r] = temp;
            l++;
            r--;
        }
    }
}

static void normalize_partition(uint8_t* p) {
    uint8_t map[MAX_ROWS];
    memset(map, 255, sizeof(map));
    uint8_t next = 0;
    for (int i = 0; i < g_rows; i++) {
        if (map[p[i]] == 255) map[p[i]] = next++;
        p[i] = map[p[i]];
    }
}

static void generate_partitions_recursive(int idx, uint8_t* current, int max_val) {
    if (idx == g_rows) {
        if (num_partitions >= MAX_PARTITIONS) {
            fprintf(stderr, "MAX_PARTITIONS=%d too small\n", MAX_PARTITIONS);
            exit(1);
        }
        Partition part;
        memset(&part, 0, sizeof(part));
        memcpy(part.mapping, current, g_rows);
        part.num_blocks = max_val + 1;

        int counts[MAX_ROWS];
        memset(counts, 0, sizeof(counts));
        for (int r = 0; r < g_rows; r++) {
            part.block_masks[current[r]] |= (1u << r);
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
        current[idx] = (uint8_t)i;
        generate_partitions_recursive(idx + 1, current, max_val);
    }
    if (max_val < g_rows - 1) {
        current[idx] = (uint8_t)(max_val + 1);
        generate_partitions_recursive(idx + 1, current, max_val + 1);
    }
}

static int get_partition_id(uint8_t* map) {
    for (int i = 0; i < num_partitions; i++) {
        if (memcmp(partitions[i].mapping, map, g_rows) == 0) return i;
    }
    return -1;
}

static void build_perm_table(void) {
    int limit = (int)factorial[g_rows];
    uint8_t temp[MAX_ROWS];

    for (int id = 0; id < num_partitions; id++) {
        for (int pi = 0; pi < limit; pi++) {
            for (int r = 0; r < g_rows; r++) {
                temp[r] = partitions[id].mapping[perms[pi][r]];
            }
            normalize_partition(temp);
            int pid = get_partition_id(temp);
            if (pid < 0 || pid >= MAX_PARTITIONS) {
                fprintf(stderr, "partition id out of range in build_perm_table: %d\n", pid);
                exit(1);
            }
            perm_table[id][pi] = (uint16_t)pid;
        }
    }
}

static void build_overlap_table(void) {
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
                    if (__builtin_popcount(m1 & m2) >= 2) mask |= (1u << i2);
                }
                overlap_mask[pid1][pid2][i1] = mask;
            }
        }
    }
}

static Count falling_value(int x, int start, int count) {
    Count value = 1;
    for (int i = 0; i < count; i++) value *= (Count)(x - (start + i));
    return value;
}

static void build_eval_tables(void) {
    for (int q = 0; q < g_eval_len; q++) {
        falling_table[q][0] = 1;
        for (int r = 1; r <= MAXN_NAUTY; r++) {
            falling_table[q][r] = falling_table[q][r - 1] * (Count)(q - (r - 1));
        }
    }
    for (int pid = 0; pid < num_partitions; pid++) {
        eval_zero(&singleton_eval_factor[pid]);
        int c = partitions[pid].num_complex;
        int s = partitions[pid].num_singletons;
        for (int q = 0; q < g_eval_len; q++) {
            singleton_eval_factor[pid].values[q] = falling_value(q, c, s);
        }
    }
}

// --- SYMMETRY LOGIC ---

static void canon_state_reset(CanonState* st, int limit) {
    st->limit = limit;
    st->depth = 0;
    st->stabilizer[0] = limit;
}

static int canon_state_prepare_push(const CanonState* st, int partition_id, uint8_t* next_insert_pos,
                                    int* next_stabilizer) {
    int depth = st->depth;
    int new_depth = depth + 1;
    int stabilizer = 0;
    uint16_t pid = (uint16_t)partition_id;

    for (int p = 0; p < st->limit; p++) {
        const uint16_t* row = st->transformed[p];
        uint16_t val = perm_table[partition_id][p];
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
        if (cmp < 0) return 0;
        stabilizer += (cmp == 0);
    }
    *next_stabilizer = stabilizer;
    return 1;
}

static void canon_state_commit_push(CanonState* st, int partition_id, const uint8_t* next_insert_pos,
                                    int next_stabilizer) {
    int depth = st->depth;
    int new_depth = depth + 1;
    st->stack_vals[depth] = (uint16_t)partition_id;

    for (int p = 0; p < st->limit; p++) {
        int j = next_insert_pos[p];
        uint16_t* row = st->transformed[p];
        uint16_t val = perm_table[partition_id][p];
        if (j < depth) {
            memmove(&row[j + 1], &row[j], (size_t)(depth - j) * sizeof(row[0]));
        }
        row[j] = val;
        st->insert_pos[depth][p] = (uint8_t)j;
    }

    st->stabilizer[new_depth] = next_stabilizer;
    st->depth = new_depth;
}

static void canon_state_pop(CanonState* st) {
    int depth = st->depth - 1;
    for (int p = 0; p < st->limit; p++) {
        int j = st->insert_pos[depth][p];
        for (int k = j; k < depth; k++) st->transformed[p][k] = st->transformed[p][k + 1];
    }
    st->depth = depth;
}

static long long get_orbit_multiplier_state(const CanonState* st) {
    int stabilizer = st->stabilizer[st->depth];
    return (long long)(factorial[g_rows] / (uint64_t)stabilizer);
}

// --- NAUTY CANONICALISATION ---

static void nauty_workspace_init(NautyWorkspace* ws, int n) {
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

static void nauty_workspace_free(NautyWorkspace* ws) {
    free(ws->ng);
    free(ws->cg);
    free(ws->lab);
    free(ws->ptn);
    free(ws->orbits);
    memset(ws, 0, sizeof(*ws));
}

static void get_canonical_graph(Graph* g, Graph* canon, NautyWorkspace* ws) {
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
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if ((g->adj[i] >> j) & 1ULL) ADDONEEDGE(ng, i, j, m);
        }
    }

    for (int i = 0; i < n; i++) {
        lab[i] = i;
        ptn[i] = 1;
    }
    ptn[n - 1] = 0;

    DEFAULTOPTIONS_GRAPH(options);
    options.getcanon = TRUE;
    options.defaultptn = TRUE;
    statsblk stats;
    densenauty(ng, lab, ptn, orbits, &options, &stats, m, n, cg);

    canon->n = n;
    memset(canon->adj, 0, sizeof(canon->adj));
    for (int i = 0; i < n; i++) {
        set* row = GRAPHROW(cg, i, m);
        for (int j = i + 1; j < n; j++) {
            if (ISELEMENT(row, j)) {
                canon->adj[i] |= 1ULL << j;
                canon->adj[j] |= 1ULL << i;
            }
        }
    }
}

// --- GRAPH UTILITIES ---

static inline uint64_t graph_row_mask(int n) {
    if (n >= 64) return ~0ULL;
    if (n <= 0) return 0ULL;
    return (1ULL << n) - 1ULL;
}

static uint64_t hash_graph(const Graph* g) {
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

static inline Count* graph_cache_value_slot(const GraphEvalCache* cache, int slot) {
    return cache->values + (size_t)slot * (size_t)cache->eval_len;
}

static inline void graph_cache_load_eval(const GraphEvalCache* cache, int slot, EvalVec* value) {
    memcpy(value->values, graph_cache_value_slot(cache, slot),
           (size_t)cache->eval_len * sizeof(value->values[0]));
    for (int i = cache->eval_len; i < MAX_EVALS; i++) value->values[i] = 0;
}

static inline Count* graph_map_value_slot(const GraphEvalMap* map, size_t slot) {
    return map->values + slot * (size_t)map->eval_len;
}

static void graph_map_init(GraphEvalMap* map, int eval_len) {
    map->cap = (size_t)1 << TERM_MAP_INITIAL_BITS;
    map->sz = 0;
    map->eval_len = eval_len;
    map->keys = (CacheKey*)calloc(map->cap, sizeof(CacheKey));
    map->adj = (AdjWord*)malloc(sizeof(AdjWord) * map->cap * MAXN_NAUTY);
    map->values = (Count*)malloc(sizeof(Count) * map->cap * (size_t)eval_len);
    if (!map->keys || !map->adj || !map->values) {
        fprintf(stderr, "Failed to allocate graph term map\n");
        exit(1);
    }
}

static void graph_map_free(GraphEvalMap* map) {
    free(map->keys);
    free(map->adj);
    free(map->values);
    memset(map, 0, sizeof(*map));
}

static void graph_map_insert_raw(GraphEvalMap* map, uint64_t key_hash, uint32_t key_n,
                                 const AdjWord* rows, const Count* values) {
    size_t mask = map->cap - 1;
    size_t slot = (size_t)(key_hash & mask);
    while (map->keys[slot].used) slot = (slot + 1) & mask;
    map->keys[slot].used = 1;
    map->keys[slot].key_hash = key_hash;
    map->keys[slot].key_n = key_n;
    memcpy(map->adj + slot * MAXN_NAUTY, rows, (size_t)key_n * sizeof(rows[0]));
    memcpy(graph_map_value_slot(map, slot), values, (size_t)map->eval_len * sizeof(values[0]));
    map->sz++;
}

static void graph_map_rehash(GraphEvalMap* map, size_t new_cap) {
    GraphEvalMap fresh = {0};
    fresh.cap = new_cap;
    fresh.sz = 0;
    fresh.eval_len = map->eval_len;
    fresh.keys = (CacheKey*)calloc(fresh.cap, sizeof(CacheKey));
    fresh.adj = (AdjWord*)malloc(sizeof(AdjWord) * fresh.cap * MAXN_NAUTY);
    fresh.values = (Count*)malloc(sizeof(Count) * fresh.cap * (size_t)fresh.eval_len);
    if (!fresh.keys || !fresh.adj || !fresh.values) {
        fprintf(stderr, "Failed to grow graph term map\n");
        exit(1);
    }

    for (size_t i = 0; i < map->cap; i++) {
        if (!map->keys[i].used) continue;
        graph_map_insert_raw(&fresh, map->keys[i].key_hash, map->keys[i].key_n,
                             map->adj + i * MAXN_NAUTY, graph_map_value_slot(map, i));
    }

    graph_map_free(map);
    *map = fresh;
}

static void graph_map_add(GraphEvalMap* map, const Graph* g, const EvalVec* value) {
    if ((map->sz + 1) * 10 >= map->cap * 7) graph_map_rehash(map, map->cap << 1);

    uint64_t key_hash = hash_graph(g);
    uint64_t row_mask = graph_row_mask(g->n);
    size_t mask = map->cap - 1;
    size_t slot = (size_t)(key_hash & mask);
    while (map->keys[slot].used) {
        if (map->keys[slot].key_hash == key_hash &&
            map->keys[slot].key_n == (uint32_t)g->n) {
            int match = 1;
            for (int i = 0; i < g->n && match; i++) {
                if (map->adj[slot * MAXN_NAUTY + (size_t)i] != (AdjWord)(g->adj[i] & row_mask)) {
                    match = 0;
                }
            }
            if (match) {
                Count* dst = graph_map_value_slot(map, slot);
                for (int i = 0; i < map->eval_len; i++) dst[i] += value->values[i];
                return;
            }
        }
        slot = (slot + 1) & mask;
    }

    map->keys[slot].used = 1;
    map->keys[slot].key_hash = key_hash;
    map->keys[slot].key_n = (uint32_t)g->n;
    for (int i = 0; i < g->n; i++) map->adj[slot * MAXN_NAUTY + (size_t)i] = (AdjWord)(g->adj[i] & row_mask);
    memcpy(graph_map_value_slot(map, slot), value->values, (size_t)map->eval_len * sizeof(value->values[0]));
    map->sz++;
}

static void graph_map_load_eval(const GraphEvalMap* map, size_t slot, EvalVec* value) {
    memcpy(value->values, graph_map_value_slot(map, slot),
           (size_t)map->eval_len * sizeof(value->values[0]));
    for (int i = map->eval_len; i < MAX_EVALS; i++) value->values[i] = 0;
}

static void graph_map_load_graph(const GraphEvalMap* map, size_t slot, Graph* g) {
    int n = (int)map->keys[slot].key_n;
    g->n = n;
    memset(g->adj, 0, sizeof(g->adj));
    for (int i = 0; i < n; i++) g->adj[i] = (uint64_t)map->adj[slot * MAXN_NAUTY + (size_t)i];
}

static void store_graph_cache_entry(GraphEvalCache* cache, uint64_t key_hash, uint32_t key_n,
                                    const Graph* g, uint64_t row_mask, const EvalVec* value) {
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
        cache->adj[(size_t)best_slot * MAXN_NAUTY + (size_t)i] = (AdjWord)(g->adj[i] & row_mask);
    }
    memcpy(graph_cache_value_slot(cache, best_slot), value->values,
           (size_t)cache->eval_len * sizeof(value->values[0]));
    cache->keys[best_slot].used = 1;
}

static void remove_vertex(Graph* g, int i) {
    int last = g->n - 1;
    for (int k = 0; k < g->n; k++) g->adj[k] &= ~(1ULL << i);

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
            uint64_t f = frontier;
            while (f) {
                int v = __builtin_ctzll(f);
                next |= g->adj[v];
                f &= f - 1;
            }
            frontier = next & remaining & ~component;
        }
        component_masks[count++] = component;
        remaining &= ~component;
    }
    return count;
}

// --- DSATUR LEAF SOLVER ---

static int dsatur_choose_vertex(const DSaturState* st) {
    int best = -1;
    int best_sat = -1;
    int best_deg = -1;
    uint32_t rem = st->uncolored;
    while (rem) {
        int v = __builtin_ctz(rem);
        int sat = __builtin_popcount(st->forbidden[v]);
        int deg = st->degree[v];
        if (sat > best_sat || (sat == best_sat && deg > best_deg)) {
            best = v;
            best_sat = sat;
            best_deg = deg;
        }
        rem &= rem - 1;
    }
    return best;
}

static void dsatur_search(DSaturState* st, int used_colors) {
    if (st->uncolored == 0) {
        st->counts[used_colors]++;
        return;
    }

    int v = dsatur_choose_vertex(st);
    uint32_t saved_uncolored = st->uncolored;
    st->uncolored &= ~(1u << v);
    uint32_t uncolored_neighbors = st->adj[v] & st->uncolored;

    uint32_t available = ((1u << used_colors) - 1u) & ~st->forbidden[v];
    while (available) {
        int color = __builtin_ctz(available);
        uint32_t color_bit = 1u << color;
        DSaturUpdate updates[MAXN_NAUTY];
        int update_count = 0;
        uint32_t rem = uncolored_neighbors;
        while (rem) {
            int u = __builtin_ctz(rem);
            if ((st->forbidden[u] & color_bit) == 0) {
                updates[update_count].vertex = u;
                updates[update_count].old_forbidden = st->forbidden[u];
                st->forbidden[u] |= color_bit;
                update_count++;
            }
            rem &= rem - 1;
        }
        dsatur_search(st, used_colors);
        for (int i = update_count - 1; i >= 0; i--) {
            st->forbidden[updates[i].vertex] = updates[i].old_forbidden;
        }
        available &= available - 1;
    }

    if (used_colors < st->n) {
        int color = used_colors;
        uint32_t color_bit = 1u << color;
        DSaturUpdate updates[MAXN_NAUTY];
        int update_count = 0;
        uint32_t rem = uncolored_neighbors;
        while (rem) {
            int u = __builtin_ctz(rem);
            if ((st->forbidden[u] & color_bit) == 0) {
                updates[update_count].vertex = u;
                updates[update_count].old_forbidden = st->forbidden[u];
                st->forbidden[u] |= color_bit;
                update_count++;
            }
            rem &= rem - 1;
        }
        dsatur_search(st, used_colors + 1);
        for (int i = update_count - 1; i >= 0; i--) {
            st->forbidden[updates[i].vertex] = updates[i].old_forbidden;
        }
    }

    st->uncolored = saved_uncolored;
}

static EvalVec solve_graph_dsatur(const Graph* g) {
    DSaturState st;
    memset(&st, 0, sizeof(st));
    st.n = g->n;
    st.uncolored = (g->n == 32) ? UINT32_MAX : ((1u << g->n) - 1u);

    for (int i = 0; i < g->n; i++) {
        st.adj[i] = (uint32_t)g->adj[i];
        st.degree[i] = __builtin_popcount(st.adj[i]);
    }

    dsatur_search(&st, 0);

    EvalVec out;
    eval_zero(&out);
    for (int q = 0; q < g_eval_len; q++) {
        Count sum = 0;
        for (int r = 0; r <= g->n; r++) {
            if (st.counts[r] == 0) continue;
            sum += st.counts[r] * falling_table[q][r];
        }
        out.values[q] = sum;
    }
    return out;
}

static EvalVec solve_graph_eval(Graph g, GraphEvalCache* cache, GraphEvalCache* raw_cache,
                                NautyWorkspace* ws, long long* local_canon_calls,
                                long long* local_cache_hits, long long* local_raw_cache_hits) {
    EvalVec multiplier;
    eval_one(&multiplier);

    int changed = 1;
    while (changed && g.n > 0) {
        changed = 0;
        for (int i = 0; i < g.n; i++) {
            uint64_t neighbors = g.adj[i];
            int degree = __builtin_popcountll(neighbors);
            if (degree == 0) {
                eval_mul_linear_inplace(&multiplier, 0);
                remove_vertex(&g, i);
                changed = 1;
                i--;
                continue;
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
                eval_mul_linear_inplace(&multiplier, degree);
                remove_vertex(&g, i);
                changed = 1;
                i--;
            }
        }
    }

    if (g.n == 0) return multiplier;

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
                if (raw_cache->adj[(size_t)p * MAXN_NAUTY + (size_t)i] != (AdjWord)row) match = 0;
            }
            if (match) {
                EvalVec cached;
                (*local_raw_cache_hits)++;
                graph_cache_load_eval(raw_cache, p, &cached);
                return eval_pointwise_mul(multiplier, &cached);
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
                if (cache->adj[(size_t)p * MAXN_NAUTY + (size_t)i] != (AdjWord)row) match = 0;
            }
            if (match) {
                EvalVec cached;
                (*local_cache_hits)++;
                graph_cache_load_eval(cache, p, &cached);
                store_graph_cache_entry(raw_cache, raw_hash, (uint32_t)g.n, &g, row_mask, &cached);
                return eval_pointwise_mul(multiplier, &cached);
            }
        }
    }

    EvalVec res;
    uint64_t component_masks[MAXN_NAUTY];
    int component_count = graph_collect_components(&g, component_masks);
    if (component_count > 1) {
        eval_one(&res);
        for (int i = 0; i < component_count; i++) {
            Graph subgraph;
            induced_subgraph_from_mask(&g, component_masks[i], &subgraph);
            EvalVec part = solve_graph_eval(subgraph, cache, raw_cache, ws,
                                            local_canon_calls, local_cache_hits, local_raw_cache_hits);
            eval_pointwise_mul_inplace(&res, &part);
        }
    } else {
        res = solve_graph_dsatur(&g);
    }

    store_graph_cache_entry(cache, hash, (uint32_t)canon.n, &canon, ADJWORD_MASK, &res);
    store_graph_cache_entry(raw_cache, raw_hash, (uint32_t)g.n, &g, row_mask, &res);
    return eval_pointwise_mul(multiplier, &res);
}

// --- STRUCTURE SEARCH ---

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
    return 1;
}

static EvalVec build_structure_weight(int* stack, const CanonState* canon_state) {
    long long mult_coeff = (long long)factorial[g_cols];
    int run = 1;
    for (int i = 1; i < g_cols; i++) {
        if (stack[i] == stack[i - 1]) run++;
        else {
            mult_coeff /= (long long)factorial[run];
            run = 1;
        }
    }
    mult_coeff /= (long long)factorial[run];

    long long row_orbit = get_orbit_multiplier_state(canon_state);
    EvalVec weight;
    eval_one(&weight);
    for (int i = 0; i < g_cols; i++) {
        eval_pointwise_mul_inplace(&weight, &singleton_eval_factor[stack[i]]);
    }
    eval_scale_inplace(&weight, (Count)(mult_coeff * row_orbit));
    return weight;
}

static void accumulate_structure(int* stack, const Graph* partial_graph, CanonState* canon_state,
                                 GraphEvalMap* term_map, NautyWorkspace* ws,
                                 long long* local_leaf_canon_calls) {
    EvalVec weight = build_structure_weight(stack, canon_state);
    Graph canon;
    get_canonical_graph((Graph*)partial_graph, &canon, ws);
    (*local_leaf_canon_calls)++;
    graph_map_add(term_map, &canon, &weight);
}

static void dfs(int depth, int min_idx, int* stack, CanonState* canon_state,
                const PartialGraphState* partial_graph, GraphEvalMap* term_map,
                NautyWorkspace* ws, long long* local_leaf_canon_calls) {
    if (depth == g_cols) {
        accumulate_structure(stack, &partial_graph->g, canon_state, term_map, ws,
                             local_leaf_canon_calls);
        return;
    }

    uint8_t next_insert_pos[PERM_SIZE];
    int next_stabilizer = 0;
    for (int i = min_idx; i < num_partitions; i++) {
        if (!canon_state_prepare_push(canon_state, i, next_insert_pos, &next_stabilizer)) continue;
        stack[depth] = i;
        canon_state_commit_push(canon_state, i, next_insert_pos, next_stabilizer);
        PartialGraphState next_graph = *partial_graph;
        if (partial_graph_append(&next_graph, depth, i, stack)) {
            dfs(depth + 1, i, stack, canon_state, &next_graph, term_map, ws, local_leaf_canon_calls);
        }
        canon_state_pop(canon_state);
    }
}

int main(int argc, char** argv) {
    long long task_start = 0;
    long long task_end = -1;
    long long task_stride = 1;
    long long task_offset = 0;
    const char* poly_out_path = NULL;
    int prefix_depth_override = -1;
    int merge_mode = 0;
    int positional_values[2];
    int positional_count = 0;
    char** merge_inputs = (char**)malloc((size_t)argc * sizeof(char*));
    if (!merge_inputs) {
        fprintf(stderr, "Failed to allocate merge input list\n");
        return 1;
    }
    int merge_input_count = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--merge") == 0) {
            merge_mode = 1;
        } else if (strcmp(argv[i], "--poly-out") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                free(merge_inputs);
                return 1;
            }
            poly_out_path = argv[++i];
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
        } else if (positional_count < 2) {
            positional_values[positional_count++] = (int)parse_ll_or_die(argv[i], "positional");
        } else {
            usage(argv[0]);
            free(merge_inputs);
            return 1;
        }
    }

    if (merge_mode) {
        if (positional_count != 0 || task_start != 0 || task_end != -1 ||
            task_stride != 1 || task_offset != 0 || prefix_depth_override != -1) {
            usage(argv[0]);
            free(merge_inputs);
            return 1;
        }
        int rc = run_merge_mode(argv[0], poly_out_path, merge_input_count, merge_inputs);
        free(merge_inputs);
        return rc;
    }
    free(merge_inputs);

    g_rows = FIXED_ROWS;
    if (positional_count == 1) {
        g_cols = positional_values[0];
    } else if (positional_count == 2) {
        if (positional_values[0] != FIXED_ROWS) {
            fprintf(stderr, "This program is specialised to %d rows\n", FIXED_ROWS);
            return 1;
        }
        g_cols = positional_values[1];
    } else if (positional_count > 2) {
        usage(argv[0]);
        return 1;
    }

    if (g_cols < 1 || g_cols > MAX_COLS) {
        fprintf(stderr, "Columns must be in range 1..%d\n", MAX_COLS);
        return 1;
    }

    g_eval_len = g_rows * g_cols + 1;

    nauty_check(WORDSIZE, MAXN_NAUTY, MAXN_NAUTY, NAUTYVERSIONID);

    factorial[0] = 1;
    for (int i = 1; i <= 19; i++) factorial[i] = factorial[i - 1] * (uint64_t)i;

    generate_permutations();
    uint8_t buffer[MAX_ROWS] = {0};
    generate_partitions_recursive(0, buffer, -1);
    build_perm_table();
    build_overlap_table();
    build_eval_tables();

    printf("Grid: %dx%d\n", g_rows, g_cols);
    printf("Partitions: %d\n", num_partitions);
    printf("Threads: %d\n", omp_get_max_threads());
    printf("Leaf backend: DSATUR counting in evaluation space\n");
    printf("Using nauty for canonical graph caching\n");

    int prefix_depth = 0;
    if (prefix_depth_override != -1) prefix_depth = prefix_depth_override;
    else if (g_cols >= 2) prefix_depth = 2;
    if (prefix_depth != 0 && prefix_depth != 2) {
        fprintf(stderr, "--prefix-depth must be 2 for this solver\n");
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
    int* prefix_i = NULL;
    int* prefix_j = NULL;
    double prefix_generation_time = 0.0;
    if (prefix_depth == 2) {
        double prefix_start_time = omp_get_wtime();
        total_prefixes = (long long)num_partitions * (num_partitions + 1) / 2;
        prefix_i = (int*)malloc((size_t)total_prefixes * sizeof(int));
        prefix_j = (int*)malloc((size_t)total_prefixes * sizeof(int));
        if (!prefix_i || !prefix_j) {
            fprintf(stderr, "Failed to allocate prefix arrays\n");
            return 1;
        }
        long long idx = 0;
        for (int i = 0; i < num_partitions; i++) {
            for (int j = i; j < num_partitions; j++) {
                prefix_i[idx] = i;
                prefix_j[idx] = j;
                idx++;
            }
        }
        prefix_generation_time = omp_get_wtime() - prefix_start_time;
        printf("Prefix depth: %d (%lld tasks)\n", prefix_depth, total_prefixes);
        printf("Prefix generation: %.2f seconds\n", prefix_generation_time);
        printf("Prefix task bytes: %zu each, %.2f MiB total\n",
               sizeof(int) * 2,
               ((double)(sizeof(int) * 2) * (double)total_prefixes) / (1024.0 * 1024.0));
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
        if (end && *end == '\0' && parsed > 0) progress_report_step = parsed;
    }
    if (progress_report_step == 0 && progress_updates_env && *progress_updates_env) {
        char* end = NULL;
        long long parsed = strtoll(progress_updates_env, &end, 10);
        if (end && *end == '\0' && parsed > 0) progress_updates = parsed;
    }
    if (progress_report_step == 0) {
        progress_report_step = total_tasks / progress_updates;
        if (progress_report_step < 1) progress_report_step = 1;
    }
    if (total_tasks > 0 && progress_report_step > total_tasks) progress_report_step = total_tasks;

    long long progress_last_reported = 0;
    printf("Task range: [%lld, %lld) of %lld\n", active_task_start, active_task_end, full_tasks);
    printf("Task selection: stride %lld, offset %lld\n", task_stride, task_offset);
    if (total_tasks == 0) printf("No tasks selected; producing the zero evaluation shard.\n");
    printf("Progress updates every %lld tasks", progress_report_step);
    if (progress_step_env && *progress_step_env) printf(" (RECT_PROGRESS_STEP override)");
    else printf(" (target ~%lld updates)", progress_updates);
    printf("\n");

    progress_reporter_init(&progress_reporter, stdout);
    progress_reporter_print_initial(&progress_reporter, total_tasks);

    double start_time = omp_get_wtime();
    int num_threads = omp_get_max_threads();
    GraphEvalMap* thread_terms = (GraphEvalMap*)calloc((size_t)num_threads, sizeof(GraphEvalMap));
    if (!thread_terms) {
        fprintf(stderr, "Failed to allocate graph term maps\n");
        return 1;
    }

    long long total_leaf_canon_calls = 0;
    #pragma omp parallel reduction(+:total_leaf_canon_calls)
    {
        int tid = omp_get_thread_num();
        GraphEvalMap* term_map = &thread_terms[tid];
        NautyWorkspace ws;
        memset(&ws, 0, sizeof(ws));
        graph_map_init(term_map, g_eval_len);

        int stack[MAX_COLS];
        CanonState canon_state;
        PartialGraphState partial_graph;
        canon_state_reset(&canon_state, (int)factorial[g_rows]);
        partial_graph_reset(&partial_graph);
        long long local_leaf_canon_calls = 0;

        if (g_cols == 1) {
            #pragma omp for schedule(dynamic, 1)
            for (long long i = first_task; i < active_task_end; i += task_stride) {
                stack[0] = (int)i;
                canon_state_reset(&canon_state, (int)factorial[g_rows]);
                partial_graph_reset(&partial_graph);
                uint8_t next_insert_pos[PERM_SIZE];
                int next_stabilizer = 0;
                if (!canon_state_prepare_push(&canon_state, (int)i, next_insert_pos, &next_stabilizer)) {
                    complete_task_and_report(total_tasks, progress_report_step,
                                             &progress_last_reported, start_time);
                    continue;
                }
                canon_state_commit_push(&canon_state, (int)i, next_insert_pos, next_stabilizer);
                if (partial_graph_append(&partial_graph, 0, (int)i, stack)) {
                    dfs(1, (int)i, stack, &canon_state, &partial_graph, term_map, &ws,
                        &local_leaf_canon_calls);
                }
                complete_task_and_report(total_tasks, progress_report_step,
                                         &progress_last_reported, start_time);
            }
        } else {
            #pragma omp for schedule(dynamic, 8)
            for (long long p = first_task; p < active_task_end; p += task_stride) {
                int i = prefix_i[p];
                int j = prefix_j[p];
                uint8_t next_insert_pos[PERM_SIZE];
                int next_stabilizer = 0;

                canon_state_reset(&canon_state, (int)factorial[g_rows]);
                partial_graph_reset(&partial_graph);

                stack[0] = i;
                if (!canon_state_prepare_push(&canon_state, i, next_insert_pos, &next_stabilizer)) {
                    complete_task_and_report(total_tasks, progress_report_step,
                                             &progress_last_reported, start_time);
                    continue;
                }
                canon_state_commit_push(&canon_state, i, next_insert_pos, next_stabilizer);
                if (!partial_graph_append(&partial_graph, 0, i, stack)) {
                    canon_state_pop(&canon_state);
                    complete_task_and_report(total_tasks, progress_report_step,
                                             &progress_last_reported, start_time);
                    continue;
                }

                stack[1] = j;
                if (!canon_state_prepare_push(&canon_state, j, next_insert_pos, &next_stabilizer)) {
                    canon_state_pop(&canon_state);
                    complete_task_and_report(total_tasks, progress_report_step,
                                             &progress_last_reported, start_time);
                    continue;
                }
                canon_state_commit_push(&canon_state, j, next_insert_pos, next_stabilizer);
                PartialGraphState prefix_graph = partial_graph;
                if (partial_graph_append(&prefix_graph, 1, j, stack)) {
                    dfs(2, j, stack, &canon_state, &prefix_graph, term_map, &ws,
                        &local_leaf_canon_calls);
                }

                canon_state_pop(&canon_state);
                canon_state_pop(&canon_state);
                complete_task_and_report(total_tasks, progress_report_step,
                                         &progress_last_reported, start_time);
            }
        }

        total_leaf_canon_calls += local_leaf_canon_calls;
        nauty_workspace_free(&ws);
        nauty_freedyn();
        nautil_freedyn();
        naugraph_freedyn();
    }

    GraphEvalMap global_terms;
    graph_map_init(&global_terms, g_eval_len);
    for (int tid = 0; tid < num_threads; tid++) {
        GraphEvalMap* term_map = &thread_terms[tid];
        for (size_t slot = 0; slot < term_map->cap; slot++) {
            if (!term_map->keys[slot].used) continue;
            Graph g;
            EvalVec weight;
            graph_map_load_graph(term_map, slot, &g);
            graph_map_load_eval(term_map, slot, &weight);
            graph_map_add(&global_terms, &g, &weight);
        }
        graph_map_free(term_map);
    }
    free(thread_terms);

    long long total_canon_calls = 0;
    long long total_cache_hits = 0;
    long long total_raw_cache_hits = 0;
    size_t distinct_graph_terms = global_terms.sz;
    double deferred_start = omp_get_wtime();
    EvalVec* deferred_thread_evals = checked_aligned_alloc(64, (size_t)num_threads * sizeof(EvalVec),
                                                           "deferred_thread_evals");
    for (int i = 0; i < num_threads; i++) eval_zero(&deferred_thread_evals[i]);

    #pragma omp parallel reduction(+:total_canon_calls, total_cache_hits, total_raw_cache_hits)
    {
        int tid = omp_get_thread_num();
        GraphEvalCache cache = {0};
        GraphEvalCache raw_cache = {0};
        NautyWorkspace ws;
        memset(&ws, 0, sizeof(ws));

        cache.mask = CACHE_MASK;
        cache.probe = CACHE_PROBE;
        cache.eval_len = g_eval_len;
        cache.keys = checked_aligned_alloc(64, sizeof(CacheKey) * CACHE_SIZE, "cache_keys");
        cache.adj = checked_aligned_alloc(64, sizeof(AdjWord) * CACHE_SIZE * MAXN_NAUTY, "cache_adj");
        cache.values = checked_aligned_alloc(64, sizeof(Count) * CACHE_SIZE * (size_t)g_eval_len,
                                             "cache_values");

        raw_cache.mask = RAW_CACHE_MASK;
        raw_cache.probe = RAW_CACHE_PROBE;
        raw_cache.eval_len = g_eval_len;
        raw_cache.keys = checked_aligned_alloc(64, sizeof(CacheKey) * RAW_CACHE_SIZE, "raw_cache_keys");
        raw_cache.adj = checked_aligned_alloc(64, sizeof(AdjWord) * RAW_CACHE_SIZE * MAXN_NAUTY,
                                              "raw_cache_adj");
        raw_cache.values = checked_aligned_alloc(64, sizeof(Count) * RAW_CACHE_SIZE * (size_t)g_eval_len,
                                                 "raw_cache_values");

        memset(cache.keys, 0, sizeof(CacheKey) * CACHE_SIZE);
        memset(raw_cache.keys, 0, sizeof(CacheKey) * RAW_CACHE_SIZE);

        long long local_canon_calls = 0;
        long long local_cache_hits = 0;
        long long local_raw_cache_hits = 0;

        #pragma omp for schedule(dynamic, 8)
        for (size_t slot = 0; slot < global_terms.cap; slot++) {
            if (!global_terms.keys[slot].used) continue;
            Graph g;
            EvalVec weight;
            graph_map_load_graph(&global_terms, slot, &g);
            graph_map_load_eval(&global_terms, slot, &weight);
            EvalVec graph_eval = solve_graph_eval(g, &cache, &raw_cache, &ws,
                                                  &local_canon_calls, &local_cache_hits,
                                                  &local_raw_cache_hits);
            EvalVec term = eval_pointwise_mul(weight, &graph_eval);
            eval_add_inplace(&deferred_thread_evals[tid], &term);
        }

        total_canon_calls += local_canon_calls;
        total_cache_hits += local_cache_hits;
        total_raw_cache_hits += local_raw_cache_hits;

        nauty_workspace_free(&ws);
        nauty_freedyn();
        nautil_freedyn();
        naugraph_freedyn();
        free(cache.keys);
        free(cache.adj);
        free(cache.values);
        free(raw_cache.keys);
        free(raw_cache.adj);
        free(raw_cache.values);
    }

    eval_zero(&global_eval);
    for (int i = 0; i < num_threads; i++) eval_add_inplace(&global_eval, &deferred_thread_evals[i]);
    free(deferred_thread_evals);
    double deferred_time = omp_get_wtime() - deferred_start;
    graph_map_free(&global_terms);

    free(prefix_i);
    free(prefix_j);

    progress_reporter_finish(&progress_reporter);

    double end_time = omp_get_wtime();
    double worker_time = end_time - start_time;
    double total_elapsed = worker_time + prefix_generation_time;

    printf("\nWorker Complete in %.2f seconds.\n", worker_time);
    if (prefix_depth > 0) printf("Total elapsed including prefix generation: %.2f seconds.\n", total_elapsed);
    printf("Deferred graph phase: %.2f seconds.\n", deferred_time);
    printf("Leaf graph canonicalisations: %lld\n", total_leaf_canon_calls);
    printf("Distinct final graphs: %zu\n", distinct_graph_terms);
    printf("Canonicalisation calls: %lld\n", total_canon_calls);
    printf("Canonical cache hits: %lld (%.1f%%)\n", total_cache_hits,
           total_canon_calls > 0 ? 100.0 * total_cache_hits / total_canon_calls : 0.0);
    printf("Raw cache hits: %lld\n", total_raw_cache_hits);

    printf("\nNewton-Form Polynomial:\n");
    print_newton_poly(&global_eval);

    printf("\nValues:\n");
    printf("P(4) = ");
    print_i128(global_eval.values[4]);
    printf("\n");
    printf("P(5) = ");
    print_i128(global_eval.values[5]);
    printf("\n");

    if (poly_out_path) {
        EvalFileMeta meta = {
            .rows = g_rows,
            .cols = g_cols,
            .task_start = active_task_start,
            .task_end = active_task_end,
            .full_tasks = full_tasks,
            .task_stride = task_stride,
            .task_offset = task_offset,
            .eval_len = g_eval_len,
        };
        write_eval_file(poly_out_path, &global_eval, &meta);
        printf("\nWrote evaluation shard to %s\n", poly_out_path);
    }

    return 0;
}
