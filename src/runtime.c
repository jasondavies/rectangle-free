#include "partition_poly.h"

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
int g_adaptive_subdivide = DEFAULT_ADAPTIVE_SUBDIVIDE;
int g_adaptive_max_depth = DEFAULT_ADAPTIVE_MAX_DEPTH;
long long g_adaptive_work_budget = DEFAULT_ADAPTIVE_WORK_BUDGET;
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

void task_timing_insert_topk(TaskTimingStats* stats, long long task_index, double elapsed) {
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

void queue_subtask_record(QueueSubtaskTimingStats* stats, const LocalTask* task,
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

void queue_subtask_merge(QueueSubtaskTimingStats* dst, const QueueSubtaskTimingStats* src) {
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

int decode_task_prefix(long long task_index, int* i, int* j, int* k, int* l) {
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

void write_task_times_file(const char* path) {
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

void flush_completed_tasks(long long total_tasks, long long report_step,
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
    (*pending_completed)++;
    if (*pending_completed >= PROGRESS_FLUSH_BATCH) {
        flush_completed_tasks(total_tasks, report_step, start_time, pending_completed);
    }
}

void complete_task_report_and_time(long long total_tasks, long long report_step,
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

void local_queue_init(LocalTaskQueue* queue, int capacity,
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

void local_queue_free(LocalTaskQueue* queue) {
    free(queue->tasks);
    free(queue->roots);
    pthread_cond_destroy(&queue->cond);
    pthread_mutex_destroy(&queue->mutex);
    memset(queue, 0, sizeof(*queue));
}

void local_task_from_stack(LocalTask* task, long long root_id, int depth, const int* stack) {
    task->depth = (uint8_t)depth;
    task->root_id = root_id;
    for (int i = 0; i < depth; i++) task->prefix[i] = (PrefixId)stack[i];
}

int local_queue_try_push(LocalTaskQueue* queue, const LocalTask* task) {
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

void local_queue_seed_push(LocalTaskQueue* queue, const LocalTask* task) {
    if (!local_queue_try_push(queue, task)) {
        fprintf(stderr, "Failed to seed local task queue\n");
        exit(1);
    }
}

int local_queue_pop(LocalTaskQueue* queue, LocalTask* task) {
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

void local_queue_finish_item(LocalTaskQueue* queue, long long root_id,
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

void local_queue_record_profile(LocalTaskQueue* queue, const LocalTask* task,
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

void local_queue_print_occupancy_summary(LocalTaskQueue* queue) {
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

void runtime_task_system_init(RuntimeTaskSystem* system, int capacity,
                              long long root_count, int total_threads) {
    local_queue_init(&system->shared_queue, capacity, root_count, total_threads);
}

void runtime_task_system_free(RuntimeTaskSystem* system) {
    local_queue_free(&system->shared_queue);
}

void runtime_task_system_seed_task(RuntimeTaskSystem* system, const LocalTask* task) {
    local_queue_seed_push(&system->shared_queue, task);
}

int runtime_task_system_pop_task(RuntimeTaskSystem* system, LocalTask* task) {
    return local_queue_pop(&system->shared_queue, task);
}

int runtime_task_system_push_local(RuntimeTaskSystem* system, const LocalTask* task) {
    return local_queue_try_push(&system->shared_queue, task);
}

int runtime_task_system_push_balance(RuntimeTaskSystem* system, const LocalTask* task) {
    return local_queue_try_push(&system->shared_queue, task);
}

int runtime_task_system_has_idle_workers(const RuntimeTaskSystem* system) {
    return atomic_load_explicit(&system->shared_queue.idle_threads, memory_order_relaxed) > 0;
}

int runtime_task_system_needs_balance(const RuntimeTaskSystem* system) {
    int idle_workers = atomic_load_explicit(&system->shared_queue.idle_threads, memory_order_relaxed);
    if (idle_workers <= 0) return 0;
    int min_global = system->shared_queue.total_threads;
    if (min_global < 4) min_global = 4;
    return system->shared_queue.count < min_global;
}

void runtime_task_system_note_balance_push(RuntimeTaskSystem* system) {
    atomic_fetch_add_explicit(&system->shared_queue.donated_tasks, 1, memory_order_relaxed);
}

void runtime_task_system_note_work_budget_split(RuntimeTaskSystem* system) {
    atomic_fetch_add_explicit(&system->shared_queue.work_budget_continuations, 1,
                              memory_order_relaxed);
}

void runtime_task_system_finish_task(RuntimeTaskSystem* system, long long root_id,
                                     long long total_tasks, long long report_step,
                                     double start_time, long long* pending_completed,
                                     TaskTimingStats* task_timing) {
    local_queue_finish_item(&system->shared_queue, root_id, total_tasks, report_step,
                            start_time, pending_completed, task_timing);
}

void runtime_task_system_record_profile(RuntimeTaskSystem* system, const LocalTask* task,
                                        double elapsed, long long solve_graph_calls,
                                        long long nauty_calls, long long hard_graph_nodes,
                                        int max_hard_graph_n, int max_hard_graph_degree) {
    local_queue_record_profile(&system->shared_queue, task, elapsed, solve_graph_calls,
                               nauty_calls, hard_graph_nodes,
                               max_hard_graph_n, max_hard_graph_degree);
}

void runtime_task_system_print_summary(RuntimeTaskSystem* system) {
    local_queue_print_occupancy_summary(&system->shared_queue);
}

void* checked_aligned_alloc(size_t alignment, size_t size, const char* label) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        fprintf(stderr, "Failed to allocate %s (%zu bytes)\n", label, size);
        exit(1);
    }
    return ptr;
}

void* checked_calloc(size_t count, size_t size, const char* label) {
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

void prefix_task_buffer_init(PrefixTaskBuffer* buf, long long initial_capacity) {
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

void prefix_task_buffer_push2(PrefixTaskBuffer* buf, int i, int j) {
    prefix_task_buffer_reserve(buf, buf->count + 1);
    buf->i[buf->count] = (PrefixId)i;
    buf->j[buf->count] = (PrefixId)j;
    buf->count++;
}

long long repeated_combo_count(int values, int slots) {
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

void get_prefix2_task(long long task_index, int* i, int* j) {
    if (g_live_prefix2_i && task_index >= 0 && task_index < g_live_prefix2_count) {
        *i = (int)g_live_prefix2_i[task_index];
        *j = (int)g_live_prefix2_j[task_index];
        return;
    }
    unrank_prefix2(task_index, i, j);
}

void build_fixed_prefix2_batches(const PrefixId* live_i, const PrefixId* live_j,
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

void unrank_prefix3(long long rank, int* i, int* j, int* k) {
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

void unrank_prefix4(long long rank, int* i, int* j, int* k, int* l) {
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
