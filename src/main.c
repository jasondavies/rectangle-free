#include "partition_poly_internal.h"

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

static int parse_main_options(int argc, char** argv, MainOptions* opts) {
    int positional_count = 0;

    memset(opts, 0, sizeof(*opts));
    opts->task_end = -1;
    opts->prefix_depth_override = -1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--poly-out") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 0;
            }
            opts->poly_out_path = argv[++i];
        } else if (strcmp(argv[i], "--task-start") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 0;
            }
            opts->task_start = parse_ll_or_die(argv[++i], "--task-start");
        } else if (strcmp(argv[i], "--task-end") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 0;
            }
            opts->task_end = parse_ll_or_die(argv[++i], "--task-end");
        } else if (strcmp(argv[i], "--prefix-depth") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 0;
            }
            opts->prefix_depth_override = (int)parse_ll_or_die(argv[++i], "--prefix-depth");
        } else if (strcmp(argv[i], "--reorder") == 0) {
            opts->reorder_partitions_flag = 1;
        } else if (strcmp(argv[i], "--adaptive-subdivide") == 0) {
            g_adaptive_subdivide = 1;
        } else if (strcmp(argv[i], "--adaptive-max-depth") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 0;
            }
            g_adaptive_max_depth = (int)parse_ll_or_die(argv[++i], "--adaptive-max-depth");
        } else if (strcmp(argv[i], "--adaptive-work-budget") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 0;
            }
            g_adaptive_work_budget = parse_ll_or_die(argv[++i], "--adaptive-work-budget");
        } else if (strcmp(argv[i], "--task-times-out") == 0) {
#if !RECT_PROFILE
            fprintf(stderr, "--task-times-out requires a profiling build\n");
            return 0;
#else
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 0;
            }
            g_task_times_out_path = argv[++i];
#endif
        } else if (argv[i][0] == '-') {
            usage(argv[0]);
            return 0;
        } else if (positional_count == 0) {
            g_rows = (int)parse_ll_or_die(argv[i], "rows");
            positional_count++;
        } else if (positional_count == 1) {
            g_cols = (int)parse_ll_or_die(argv[i], "cols");
            positional_count++;
        } else {
            usage(argv[0]);
            return 0;
        }
    }

    return 1;
}

static int choose_prefix_depth(int prefix_depth_override) {
    if (prefix_depth_override != -1) return prefix_depth_override;
    if (g_rows == 7 && g_cols >= 6) return 2;
    if (g_cols >= 6) return 3;
    if (g_cols >= 2) return 2;
    return 0;
}

static int init_problem_and_run_config(const MainOptions* opts, RunConfig* cfg) {
    memset(cfg, 0, sizeof(*cfg));

    if (g_rows < 1 || g_cols < 1 || g_rows > MAX_ROWS || g_cols > MAX_COLS) {
        fprintf(stderr, "Rows/cols must be in range 1..%d and 1..%d\n", MAX_ROWS, MAX_COLS);
        return 0;
    }

    {
        int max_n = MAXN_NAUTY;
        int max_m = SETWORDSNEEDED(max_n);
        nauty_check(WORDSIZE, max_m, max_n, NAUTYVERSIONID);
    }

    factorial[0] = 1;
    for (int i = 1; i <= 19; i++) factorial[i] = factorial[i - 1] * i;

    init_row_dependent_tables();
    generate_permutations();
    {
        uint8_t buffer[MAX_ROWS] = {0};
        generate_partitions_recursive(0, buffer, -1);
    }
    if (opts->reorder_partitions_flag) {
        reorder_partitions_by_hardness();
    }
#if RECT_COUNT_K4_FEASIBILITY
    init_pair_index();
#endif
    if (num_partitions >= CANON_PARTITION_ID_LIMIT) {
        fprintf(stderr, "Partition ID limit too small for %d partitions\n", num_partitions);
        return 0;
    }

    init_partition_lookup_tables();
    build_partition_id_lookup();
    build_perm_table();
    build_terminal_perm_order_tables();
    build_overlap_table();
#if RECT_COUNT_K4
    build_partition_weight4_table();
#else
    build_partition_weight_table();
#endif
#if RECT_COUNT_K4_FEASIBILITY
    build_partition_shadow_table();
#endif

    printf("Grid: %dx%d\n", g_rows, g_cols);
    printf("Partitions: %d\n", num_partitions);
    printf("Threads: %d\n", omp_get_max_threads());
    if (opts->reorder_partitions_flag) {
        printf("Partition hardness reorder: enabled\n");
    }
#if RECT_COUNT_K4
    printf("Mode: fixed 4-colour count\n");
#else
    printf("Mode: chromatic polynomial\n");
#endif
#if RECT_PROFILE
    printf("Profiling build: enabled\n");
#else
    printf("Profiling build: disabled\n");
#endif
    printf("Using nauty for canonical graph caching\n");

    cfg->prefix_depth = choose_prefix_depth(opts->prefix_depth_override);
    if (cfg->prefix_depth != 0 && cfg->prefix_depth != 2 && cfg->prefix_depth != 3 &&
        cfg->prefix_depth != 4) {
        fprintf(stderr, "--prefix-depth must be 2, 3, or 4\n");
        return 0;
    }
    if (cfg->prefix_depth > g_cols) {
        fprintf(stderr, "--prefix-depth must not exceed cols\n");
        return 0;
    }
    if (g_cols >= 2 && cfg->prefix_depth == 0) {
        fprintf(stderr, "Internal error: invalid zero prefix depth for cols >= 2\n");
        return 0;
    }
    if (g_adaptive_subdivide && cfg->prefix_depth != 2) {
        fprintf(stderr, "Adaptive subdivision currently supports only --prefix-depth 2\n");
        return 0;
    }
    if (g_adaptive_subdivide && g_cols < 3) {
        fprintf(stderr, "--adaptive-subdivide requires cols >= 3\n");
        return 0;
    }
    if (g_adaptive_work_budget < 0) {
        fprintf(stderr, "--adaptive-work-budget must be non-negative\n");
        return 0;
    }
    if (g_adaptive_subdivide && g_adaptive_max_depth < 3) {
        fprintf(stderr, "--adaptive-max-depth must be at least 3 with --adaptive-subdivide\n");
        return 0;
    }
    if (g_adaptive_work_budget > 0 && !g_adaptive_subdivide) {
        fprintf(stderr, "--adaptive-work-budget requires --adaptive-subdivide\n");
        return 0;
    }

    {
        const char* queue_profile_step_env = getenv("RECT_QUEUE_PROFILE_STEP");
        if (queue_profile_step_env && *queue_profile_step_env) {
            g_queue_profile_report_step = strtod(queue_profile_step_env, NULL);
            if (g_queue_profile_report_step < 0.0) g_queue_profile_report_step = 0.0;
        }
        {
            const char* shared_cache_env = getenv("RECT_SHARED_CACHE_MERGE");
            if (shared_cache_env && *shared_cache_env && strcmp(shared_cache_env, "0") != 0) {
                g_shared_cache_merge = 1;
            }
        }
        {
            const char* shared_cache_bits_env = getenv("RECT_SHARED_CACHE_BITS");
            if (shared_cache_bits_env && *shared_cache_bits_env) {
                g_shared_cache_bits =
                    (int)parse_ll_or_die(shared_cache_bits_env, "RECT_SHARED_CACHE_BITS");
                if (g_shared_cache_bits < 10 || g_shared_cache_bits > 24) {
                    fprintf(stderr, "RECT_SHARED_CACHE_BITS must be between 10 and 24\n");
                    return 0;
                }
            }
        }
#if RECT_PROFILE
        {
            const char* profile_separators_env = getenv("RECT_PROFILE_SEPARATORS");
            if (profile_separators_env && *profile_separators_env &&
                strcmp(profile_separators_env, "0") != 0) {
                g_profile_separators = 1;
            }
        }
#endif
        {
            const char* raw_cache_env = getenv("RECT_USE_RAW_CACHE");
            if (raw_cache_env && *raw_cache_env) {
                g_use_raw_cache = (strcmp(raw_cache_env, "0") != 0);
            }
        }
    }

    if (opts->task_start < 0) {
        fprintf(stderr, "--task-start must be non-negative\n");
        return 0;
    }
    if (opts->task_end >= 0 && opts->task_end < opts->task_start) {
        fprintf(stderr, "Task range must satisfy 0 <= start <= end\n");
        return 0;
    }

    printf("Raw cache: %s\n", g_use_raw_cache ? "enabled" : "disabled");
    g_effective_prefix_depth = cfg->prefix_depth;
    cfg->graph_poly_len = RECT_COUNT_K4 ? 1 : (g_cols * (g_rows / 2) + 1);

    return 1;
}

static int prepare_run_config(const MainOptions* opts, RunConfig* cfg) {
    if (g_shared_cache_merge) {
        shared_graph_cache_init(&cfg->shared_graph_cache, g_shared_cache_bits, cfg->graph_poly_len);
        g_shared_graph_cache = &cfg->shared_graph_cache;
        cfg->shared_graph_cache_active = 1;
        printf("Shared canonical cache merge enabled: 2^%d slots\n", g_shared_cache_bits);
    }

    if (cfg->prefix_depth > 0) {
        double prefix_start_time = omp_get_wtime();
        if (cfg->prefix_depth == 2) {
            cfg->nominal_prefixes = (long long)num_partitions * (num_partitions + 1) / 2;
            build_live_prefix2_tasks(&g_live_prefix2_i, &g_live_prefix2_j, &g_live_prefix2_count);
            cfg->total_prefixes = g_live_prefix2_count;
        } else if (cfg->prefix_depth == 3) {
            cfg->total_prefixes =
                (long long)num_partitions * (num_partitions + 1) * (num_partitions + 2) / 6;
        } else if (cfg->prefix_depth == 4) {
            cfg->total_prefixes = (long long)num_partitions * (num_partitions + 1) *
                                  (num_partitions + 2) * (num_partitions + 3) / 24;
        }
        cfg->prefix_generation_time = omp_get_wtime() - prefix_start_time;
    }

    completed_tasks = 0;
    cfg->use_runtime_split_queue = (cfg->prefix_depth == 2 && g_adaptive_subdivide);
    cfg->full_tasks = (g_cols == 1) ? (long long)num_partitions : cfg->total_prefixes;
    cfg->active_task_start = opts->task_start;
    cfg->active_task_end = (opts->task_end < 0) ? cfg->full_tasks : opts->task_end;
    if (cfg->active_task_end < cfg->active_task_start || cfg->active_task_end > cfg->full_tasks) {
        fprintf(stderr, "Task range must satisfy 0 <= start <= end <= %lld\n", cfg->full_tasks);
        return 0;
    }
    cfg->total_tasks = cfg->active_task_end - cfg->active_task_start;
    cfg->first_task = cfg->active_task_start;
    g_task_times_first_task = cfg->first_task;
    g_task_times_count = cfg->total_tasks;

    if (g_task_times_out_path && cfg->total_tasks > 0) {
        g_task_times_values = checked_calloc((size_t)cfg->total_tasks, sizeof(*g_task_times_values),
                                             "task_times_values");
        for (long long t = 0; t < cfg->total_tasks; t++) {
            g_task_times_values[t] = -1.0;
        }
    }

    if (cfg->prefix_depth == 2 && !g_adaptive_subdivide && cfg->total_tasks > 0) {
        double batch_start_time = omp_get_wtime();
        build_fixed_prefix2_batches(g_live_prefix2_i, g_live_prefix2_j, cfg->active_task_start,
                                    cfg->total_tasks, &cfg->prefix2_batches,
                                    &cfg->prefix2_batch_count, &cfg->prefix2_batch_js,
                                    &cfg->prefix2_batch_ps);
        cfg->prefix_generation_time += omp_get_wtime() - batch_start_time;
    }

    printf("Prefix depth: %d (%lld tasks)\n", cfg->prefix_depth, cfg->total_prefixes);
    if (cfg->prefix_depth == 2 && cfg->nominal_prefixes > 0) {
        printf("Live depth-2 prefixes: %lld of %lld nominal\n",
               cfg->total_prefixes, cfg->nominal_prefixes);
    }
    if (g_adaptive_subdivide) {
        if (cfg->use_runtime_split_queue) {
            printf("Runtime subdivision enabled: max depth %d", g_adaptive_max_depth);
            if (g_adaptive_work_budget > 0) {
                printf(", work budget %lld", g_adaptive_work_budget);
            }
            printf("\n");
        } else {
            printf("Adaptive subdivision: max depth %d\n", g_adaptive_max_depth);
        }
    }
    printf("Prefix generation: %.2f seconds\n", cfg->prefix_generation_time);
    if (cfg->prefix_depth > 0) {
        if (cfg->prefix_depth == 2 && cfg->prefix2_batch_count > 0) {
            size_t bytes_per_task = sizeof(PrefixId) + sizeof(long long);
            size_t bytes_per_batch = sizeof(*cfg->prefix2_batches);
            double total_mib =
                (((double)bytes_per_task * (double)cfg->total_tasks) +
                 ((double)bytes_per_batch * (double)cfg->prefix2_batch_count)) /
                (1024.0 * 1024.0);
            printf("Fixed depth-2 batching: %lld batches, %.2f MiB total\n",
                   cfg->prefix2_batch_count, total_mib);
        } else {
            printf("Prefix task storage: unranked on demand for selected tasks\n");
        }
    }

    {
        long long progress_updates = DEFAULT_PROGRESS_UPDATES;
        const char* progress_step_env = getenv("RECT_PROGRESS_STEP");
        const char* progress_updates_env = getenv("RECT_PROGRESS_UPDATES");
        if (progress_step_env && *progress_step_env) {
            char* end = NULL;
            long long parsed = strtoll(progress_step_env, &end, 10);
            if (end && *end == '\0' && parsed > 0) {
                cfg->progress_report_step = parsed;
            }
        }
        if (cfg->progress_report_step == 0 && progress_updates_env && *progress_updates_env) {
            char* end = NULL;
            long long parsed = strtoll(progress_updates_env, &end, 10);
            if (end && *end == '\0' && parsed > 0) {
                progress_updates = parsed;
            }
        }
        if (cfg->progress_report_step == 0) {
            cfg->progress_report_step = cfg->total_tasks / progress_updates;
            if (cfg->progress_report_step < 1) cfg->progress_report_step = 1;
        }
        if (cfg->total_tasks > 0 && cfg->progress_report_step > cfg->total_tasks) {
            cfg->progress_report_step = cfg->total_tasks;
        }
        printf("Task range: [%lld, %lld) of %lld\n",
               cfg->active_task_start, cfg->active_task_end, cfg->full_tasks);

        {
            const char* omp_static_env = getenv("RECT_OMP_STATIC");
            const char* omp_schedule_env = getenv("OMP_SCHEDULE");
            int use_static_schedule =
                (omp_static_env && *omp_static_env && strcmp(omp_static_env, "0") != 0);
            int omp_chunk = 1;
            if (!use_static_schedule) {
                omp_chunk = (cfg->prefix_depth == 2 && g_rows < 7 && !g_adaptive_subdivide) ? 8 : 1;
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
        }
        if (cfg->total_tasks == 0) {
            printf("No tasks selected; producing the zero polynomial for this shard.\n");
        }
        printf("Progress updates every %lld tasks", cfg->progress_report_step);
        if (progress_step_env && *progress_step_env) {
            printf(" (RECT_PROGRESS_STEP override)");
        } else {
            printf(" (target ~%lld updates)", progress_updates);
        }
        printf("\n");
    }

    progress_last_reported = 0;
    progress_reporter_init(&progress_reporter, stdout);
    progress_reporter_print_initial(&progress_reporter, cfg->total_tasks);

    return 1;
}

static void cleanup_run_config(RunConfig* cfg) {
    free(g_live_prefix2_i);
    free(g_live_prefix2_j);
    g_live_prefix2_i = NULL;
    g_live_prefix2_j = NULL;
    g_live_prefix2_count = 0;

    free(cfg->prefix2_batches);
    free(cfg->prefix2_batch_js);
    free(cfg->prefix2_batch_ps);
    cfg->prefix2_batches = NULL;
    cfg->prefix2_batch_js = NULL;
    cfg->prefix2_batch_ps = NULL;
    cfg->prefix2_batch_count = 0;

    free(g_task_times_values);
    g_task_times_values = NULL;

    if (cfg->shared_graph_cache_active) {
        shared_graph_cache_free(&cfg->shared_graph_cache);
        g_shared_graph_cache = NULL;
        cfg->shared_graph_cache_active = 0;
    }

    small_graph_lookup_free();
    connected_canon_lookup_free();
    free_row_dependent_tables();
}

static int init_execution_state(const RunConfig* run, ExecutionState* exec) {
    memset(exec, 0, sizeof(*exec));
    exec->num_threads = omp_get_max_threads();

    if (run->use_runtime_split_queue) {
        int queue_capacity_slack = 4 * exec->num_threads;
        long long queue_cap_ll = run->total_tasks + queue_capacity_slack + 64;
        if (queue_cap_ll > INT_MAX) {
            fprintf(stderr, "Local task queue too large\n");
            return 0;
        }

        runtime_task_system_init(&exec->runtime_tasks, (int)queue_cap_ll,
                                 run->total_tasks, exec->num_threads);
        for (long long t = 0; t < run->total_tasks; t++) {
            long long p = run->first_task + t;
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
            exec->runtime_tasks.shared_queue.roots[t].pending = 0;
            exec->runtime_tasks.shared_queue.roots[t].task_index = p;
            runtime_task_system_seed_task(&exec->runtime_tasks, &task);
        }
        exec->runtime_tasks_active = 1;
        printf("Runtime queue: capacity=%d, split-max-depth=%d",
               (int)queue_cap_ll, g_adaptive_max_depth);
        if (g_adaptive_work_budget > 0) {
            printf(", work budget %lld", g_adaptive_work_budget);
        }
        printf("\n");
    }

    exec->thread_totals =
        checked_aligned_alloc(64, (size_t)exec->num_threads * sizeof(*exec->thread_totals), "thread_totals");
    exec->thread_profiles = checked_aligned_alloc(
        64, (size_t)exec->num_threads * sizeof(ProfileStats), "thread_profiles");
    exec->thread_task_timing = checked_aligned_alloc(
        64, (size_t)exec->num_threads * sizeof(TaskTimingStats), "thread_task_timing");
    for (int i = 0; i < exec->num_threads; i++) {
#if RECT_COUNT_K4
        exec->thread_totals[i] = 0;
#else
        poly_zero(&exec->thread_totals[i]);
#endif
    }
    memset(exec->thread_profiles, 0, (size_t)exec->num_threads * sizeof(ProfileStats));
    memset(exec->thread_task_timing, 0, (size_t)exec->num_threads * sizeof(TaskTimingStats));
    if (run->use_runtime_split_queue && PROFILE_BUILD) {
        exec->thread_queue_subtask_timing = checked_aligned_alloc(
            64, (size_t)exec->num_threads * (size_t)(MAX_COLS + 1) * sizeof(QueueSubtaskTimingStats),
            "thread_queue_subtask_timing");
        memset(exec->thread_queue_subtask_timing, 0,
               (size_t)exec->num_threads * (size_t)(MAX_COLS + 1) *
                   sizeof(QueueSubtaskTimingStats));
    }

    return 1;
}

static void cleanup_execution_state(ExecutionState* exec) {
    if (exec->runtime_tasks_active) {
        runtime_task_system_free(&exec->runtime_tasks);
        exec->runtime_tasks_active = 0;
    }
    free(exec->thread_totals);
    free(exec->thread_profiles);
    free(exec->thread_task_timing);
    free(exec->thread_queue_subtask_timing);
    exec->thread_totals = NULL;
    exec->thread_profiles = NULL;
    exec->thread_task_timing = NULL;
    exec->thread_queue_subtask_timing = NULL;
}

static int replay_task_prefix(const int* prefix, int prefix_depth, CanonState* canon_state,
                              CanonScratch* canon_scratch, PartialGraphState* partial_graph,
                              int* stack, ProfileStats* profile, WeightAccum* prefix_weight,
                              long long* prefix_mult, int* prefix_run) {
    int next_stabilizer = 0;
    int prev_pid = -1;
    double t0 = 0.0;
    int pushed = 0;

    weight_accum_one(prefix_weight);
    *prefix_mult = 1;
    *prefix_run = 0;

    for (int depth = 0; depth < prefix_depth; depth++) {
        int pid = prefix[depth];
        stack[depth] = pid;

        if (PROFILE_BUILD) {
            profile->canon_prepare_calls++;
            profile->canon_prepare_calls_by_depth[depth]++;
            t0 = omp_get_wtime();
        }
        if (!canon_state_prepare_push(canon_state, pid, canon_scratch, &next_stabilizer)) {
            if (PROFILE_BUILD) profile->canon_prepare_time += omp_get_wtime() - t0;
            goto fail;
        }
        if (PROFILE_BUILD) {
            profile->canon_prepare_time += omp_get_wtime() - t0;
            profile->canon_prepare_accepts++;
            profile->canon_prepare_accepts_by_depth[depth]++;
            profile->stabilizer_sum_by_depth[depth] += next_stabilizer;
            profile->partial_append_calls++;
            t0 = omp_get_wtime();
        }
        if (!partial_graph_append_checked(partial_graph, depth, pid, stack, g_cols - depth - 1)) {
            if (PROFILE_BUILD) profile->partial_append_time += omp_get_wtime() - t0;
            goto fail;
        }
        if (PROFILE_BUILD) profile->partial_append_time += omp_get_wtime() - t0;

        if (PROFILE_BUILD) {
            profile->canon_commit_calls++;
            t0 = omp_get_wtime();
        }
        canon_state_commit_push(canon_state, pid, canon_scratch, next_stabilizer);
        pushed++;
        if (PROFILE_BUILD) profile->canon_commit_time += omp_get_wtime() - t0;

        weight_accum_mul_partition(prefix_weight, pid, prefix_weight);
        {
            long long next_mult = (*prefix_mult) * (depth + 1);
            int next_run = 1;
            if (depth > 0 && pid == prev_pid) {
                next_run = *prefix_run + 1;
                next_mult /= next_run;
            }
            *prefix_mult = next_mult;
            *prefix_run = next_run;
        }
        prev_pid = pid;
    }

    return 1;

fail:
    while (pushed-- > 0) canon_state_pop(canon_state);
    return 0;
}

static void execute_run_tasks(const RunConfig* run, double start_time, ExecutionState* exec) {
    const int prefix_depth = run->prefix_depth;
    const int graph_poly_len = run->graph_poly_len;
    const Prefix2Batch* prefix2_batches = run->prefix2_batches;
    const PrefixId* prefix2_batch_js = run->prefix2_batch_js;
    const long long* prefix2_batch_ps = run->prefix2_batch_ps;
    const long long prefix2_batch_count = run->prefix2_batch_count;
    const int use_runtime_split_queue = run->use_runtime_split_queue;
    const long long active_task_end = run->active_task_end;
    const long long total_tasks = run->total_tasks;
    const long long first_task = run->first_task;
    const long long progress_report_step = run->progress_report_step;
    long long total_canon_calls = 0;
    long long total_cache_hits = 0;
    long long total_raw_cache_hits = 0;

    #pragma omp parallel reduction(+:total_canon_calls, total_cache_hits, total_raw_cache_hits)
    {
        int tid = omp_get_thread_num();
        RowGraphCache cache = {0};
        RowGraphCache raw_cache = {0};
        NautyWorkspace ws;
        memset(&ws, 0, sizeof(ws));
        cache.mask = CACHE_MASK;
        cache.probe = CACHE_PROBE;
        cache.poly_len = graph_poly_len;
        cache.keys = checked_aligned_alloc(64, sizeof(CacheKey) * CACHE_SIZE, "cache_keys");
        cache.stamps = checked_aligned_alloc(64, sizeof(uint32_t) * CACHE_SIZE, "cache_stamps");
        cache.rows = checked_aligned_alloc(64, sizeof(AdjWord) * CACHE_SIZE * MAXN_NAUTY, "cache_rows");
        cache.coeffs =
            checked_aligned_alloc(64,
                                  sizeof(GraphCacheValue) * CACHE_SIZE *
                                      (size_t)(
#if RECT_COUNT_K4
                                          1
#else
                                          graph_poly_len
#endif
                                          ),
                                  "cache_coeffs");
        cache.next_stamp = 0;

        raw_cache.mask = RAW_CACHE_MASK;
        raw_cache.probe = RAW_CACHE_PROBE;
        raw_cache.poly_len = graph_poly_len;
        raw_cache.keys = checked_aligned_alloc(64, sizeof(CacheKey) * RAW_CACHE_SIZE, "raw_cache_keys");
        raw_cache.stamps = checked_aligned_alloc(64, sizeof(uint32_t) * RAW_CACHE_SIZE, "raw_cache_stamps");
        raw_cache.rows =
            checked_aligned_alloc(64, sizeof(AdjWord) * RAW_CACHE_SIZE * MAXN_NAUTY, "raw_cache_rows");
        raw_cache.coeffs =
            checked_aligned_alloc(64,
                                  sizeof(GraphCacheValue) * RAW_CACHE_SIZE *
                                      (size_t)(
#if RECT_COUNT_K4
                                          1
#else
                                          graph_poly_len
#endif
                                          ),
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
        ProfileStats* profile = &exec->thread_profiles[tid];
        TaskTimingStats* task_timing = &exec->thread_task_timing[tid];
        QueueSubtaskTimingStats* queue_subtask_timing = exec->thread_queue_subtask_timing
            ? exec->thread_queue_subtask_timing + (size_t)tid * (size_t)(MAX_COLS + 1)
            : NULL;
        SharedGraphCacheExporter shared_cache_exporter = {0};
        long long pending_completed = 0;
        tls_profile = profile;
        tls_shared_cache_exporter = g_shared_cache_merge ? &shared_cache_exporter : NULL;

        if (g_cols == 1) {
            #pragma omp for schedule(runtime)
            for (long long i = first_task; i < active_task_end; i++) {
                double task_t0 = PROFILE_BUILD ? omp_get_wtime() : 0.0;
                double t0 = 0.0;
                stack[0] = i;
                canon_state_reset(&canon_state, perm_count);
                partial_graph_reset(&partial_graph);
                int next_stabilizer = 0;
                if (PROFILE_BUILD) {
                    profile->canon_prepare_calls++;
                    profile->canon_prepare_calls_by_depth[0]++;
                    t0 = omp_get_wtime();
                }
                if (!canon_state_prepare_push(&canon_state, (int)i, &canon_scratch, &next_stabilizer)) {
                    if (PROFILE_BUILD) profile->canon_prepare_time += omp_get_wtime() - t0;
                    complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                  &pending_completed, task_timing, i, task_t0);
                    continue;
                }
                if (PROFILE_BUILD) {
                    profile->canon_prepare_time += omp_get_wtime() - t0;
                    profile->canon_prepare_accepts++;
                    profile->canon_prepare_accepts_by_depth[0]++;
                    profile->stabilizer_sum_by_depth[0] += next_stabilizer;
                    profile->partial_append_calls++;
                    t0 = omp_get_wtime();
                }
                int ok = partial_graph_append_checked(&partial_graph, 0, (int)i, stack, g_cols - 1);
                if (PROFILE_BUILD) profile->partial_append_time += omp_get_wtime() - t0;
                if (ok) {
                    WeightAccum initial_weight;
                    weight_accum_from_partition((int)i, &initial_weight);
                    if (PROFILE_BUILD) {
                        profile->canon_commit_calls++;
                        t0 = omp_get_wtime();
                    }
                    canon_state_commit_push(&canon_state, (int)i, &canon_scratch, next_stabilizer);
                    if (PROFILE_BUILD) profile->canon_commit_time += omp_get_wtime() - t0;
                    dfs(1, (int)i, stack, &canon_state, &partial_graph, &cache, &raw_cache, &ws,
                        &exec->thread_totals[tid], &local_canon_calls, &local_cache_hits,
                        &local_raw_cache_hits, &initial_weight, 1, 1, profile, &canon_scratch);
                    canon_state_pop(&canon_state);
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
                                                &canon_scratch, &partial_graph, stack,
                                                &exec->thread_totals[tid], &local_canon_calls,
                                                &local_cache_hits, &local_raw_cache_hits, profile,
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
                    if (!runtime_task_system_pop_task(&exec->runtime_tasks, &task)) break;
                    execute_local_runtime_task(&task, &ctx, &exec->thread_totals[tid], &exec->runtime_tasks,
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
                    double task_t0 = PROFILE_BUILD ? omp_get_wtime() : 0.0;
                    long long p = first_task + t;
                    int prefix[2] = {0};
                    WeightAccum prefix_weight;
                    long long prefix_mult = 0;
                    int prefix_run = 0;

                    get_prefix2_task(p, &prefix[0], &prefix[1]);

                    canon_state_reset(&canon_state, perm_count);
                    partial_graph_reset(&partial_graph);

                    if (replay_task_prefix(prefix, 2, &canon_state, &canon_scratch,
                                           &partial_graph, stack, profile, &prefix_weight,
                                           &prefix_mult, &prefix_run)) {
                        dfs(2, prefix[1], stack, &canon_state, &partial_graph, &cache, &raw_cache, &ws,
                            &exec->thread_totals[tid], &local_canon_calls, &local_cache_hits,
                            &local_raw_cache_hits, &prefix_weight, prefix_mult, prefix_run,
                            profile, &canon_scratch);
                        canon_state_pop(&canon_state);
                        canon_state_pop(&canon_state);
                    }

                    complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                                  &pending_completed, task_timing, p, task_t0);
                }
            }
        } else if (prefix_depth == 3) {
            #pragma omp for schedule(runtime)
            for (long long t = 0; t < total_tasks; t++) {
                double task_t0 = PROFILE_BUILD ? omp_get_wtime() : 0.0;
                long long p = first_task + t;
                int prefix[3] = {0};
                WeightAccum prefix_weight;
                long long prefix_mult = 0;
                int prefix_run = 0;

                unrank_prefix3(p, &prefix[0], &prefix[1], &prefix[2]);

                canon_state_reset(&canon_state, perm_count);
                partial_graph_reset(&partial_graph);

                if (replay_task_prefix(prefix, 3, &canon_state, &canon_scratch,
                                       &partial_graph, stack, profile, &prefix_weight,
                                       &prefix_mult, &prefix_run)) {
                    dfs(3, prefix[2], stack, &canon_state, &partial_graph, &cache, &raw_cache, &ws,
                        &exec->thread_totals[tid], &local_canon_calls, &local_cache_hits,
                        &local_raw_cache_hits, &prefix_weight, prefix_mult, prefix_run,
                        profile, &canon_scratch);
                    canon_state_pop(&canon_state);
                    canon_state_pop(&canon_state);
                    canon_state_pop(&canon_state);
                }

                complete_task_report_and_time(total_tasks, progress_report_step, start_time,
                                              &pending_completed, task_timing, p, task_t0);
            }
        } else {
            #pragma omp for schedule(runtime)
            for (long long t = 0; t < total_tasks; t++) {
                double task_t0 = PROFILE_BUILD ? omp_get_wtime() : 0.0;
                long long p = first_task + t;
                int prefix[4] = {0};
                WeightAccum prefix_weight;
                long long prefix_mult = 0;
                int prefix_run = 0;

                unrank_prefix4(p, &prefix[0], &prefix[1], &prefix[2], &prefix[3]);

                canon_state_reset(&canon_state, perm_count);
                partial_graph_reset(&partial_graph);

                if (replay_task_prefix(prefix, 4, &canon_state, &canon_scratch,
                                       &partial_graph, stack, profile, &prefix_weight,
                                       &prefix_mult, &prefix_run)) {
                    dfs(4, prefix[3], stack, &canon_state, &partial_graph, &cache, &raw_cache, &ws,
                        &exec->thread_totals[tid], &local_canon_calls, &local_cache_hits,
                        &local_raw_cache_hits, &prefix_weight, prefix_mult, prefix_run,
                        profile, &canon_scratch);
                    canon_state_pop(&canon_state);
                    canon_state_pop(&canon_state);
                    canon_state_pop(&canon_state);
                    canon_state_pop(&canon_state);
                }

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
        free(cache.rows);
        free(cache.coeffs);
        free(raw_cache.keys);
        free(raw_cache.stamps);
        free(raw_cache.rows);
        free(raw_cache.coeffs);
    }

    exec->total_canon_calls = total_canon_calls;
    exec->total_cache_hits = total_cache_hits;
    exec->total_raw_cache_hits = total_raw_cache_hits;
}

static void accumulate_execution_poly(const ExecutionState* exec, Poly* total_poly) {
#if RECT_COUNT_K4
    unsigned __int128 total = 0;
    for (int i = 0; i < exec->num_threads; i++) {
        total += exec->thread_totals[i];
    }
    poly_zero(total_poly);
    total_poly->coeffs[0] = (PolyCoeff)total;
#else
    for (int i = 0; i < exec->num_threads; i++) {
        poly_accumulate_checked(total_poly, &exec->thread_totals[i]);
    }
#endif
}

static void aggregate_execution_summary(const ExecutionState* exec, ExecutionSummary* summary) {
    memset(summary, 0, sizeof(*summary));
    for (int i = 0; i < exec->num_threads; i++) {
        ProfileStats* src = &exec->thread_profiles[i];
        summary->profile.canon_prepare_calls += src->canon_prepare_calls;
        summary->profile.canon_prepare_accepts += src->canon_prepare_accepts;
        summary->profile.canon_commit_calls += src->canon_commit_calls;
        summary->profile.partial_append_calls += src->partial_append_calls;
        summary->profile.solve_structure_calls += src->solve_structure_calls;
        summary->profile.solve_graph_calls += src->solve_graph_calls;
        summary->profile.nauty_calls += src->nauty_calls;
        summary->profile.hard_graph_nodes += src->hard_graph_nodes;
        summary->profile.canon_prepare_time += src->canon_prepare_time;
        summary->profile.canon_commit_time += src->canon_commit_time;
        summary->profile.partial_append_time += src->partial_append_time;
        summary->profile.build_weight_time += src->build_weight_time;
        summary->profile.solve_graph_time += src->solve_graph_time;
        summary->profile.get_canonical_graph_time += src->get_canonical_graph_time;
        summary->profile.get_canonical_graph_dense_rows_time += src->get_canonical_graph_dense_rows_time;
        summary->profile.get_canonical_graph_build_input_time += src->get_canonical_graph_build_input_time;
        summary->profile.nauty_time += src->nauty_time;
        summary->profile.get_canonical_graph_rebuild_time += src->get_canonical_graph_rebuild_time;
        if (src->hard_graph_max_n > summary->profile.hard_graph_max_n) {
            summary->profile.hard_graph_max_n = src->hard_graph_max_n;
        }
        if (src->hard_graph_max_degree > summary->profile.hard_graph_max_degree) {
            summary->profile.hard_graph_max_degree = src->hard_graph_max_degree;
        }
        for (int d = 0; d <= MAX_COLS; d++) {
            summary->profile.canon_prepare_calls_by_depth[d] += src->canon_prepare_calls_by_depth[d];
            summary->profile.canon_prepare_accepts_by_depth[d] += src->canon_prepare_accepts_by_depth[d];
            summary->profile.stabilizer_sum_by_depth[d] += src->stabilizer_sum_by_depth[d];
            summary->profile.canon_prepare_scanned_by_depth[d] += src->canon_prepare_scanned_by_depth[d];
            summary->profile.canon_prepare_active_by_depth[d] += src->canon_prepare_active_by_depth[d];
            summary->profile.canon_prepare_terminal_calls_by_depth[d] +=
                src->canon_prepare_terminal_calls_by_depth[d];
            summary->profile.canon_prepare_fast_continue_by_depth[d] +=
                src->canon_prepare_fast_continue_by_depth[d];
            summary->profile.canon_prepare_terminal_continue_by_depth[d] +=
                src->canon_prepare_terminal_continue_by_depth[d];
            summary->profile.canon_prepare_equal_case_calls_by_depth[d] +=
                src->canon_prepare_equal_case_calls_by_depth[d];
            summary->profile.canon_prepare_equal_case_rejects_by_depth[d] +=
                src->canon_prepare_equal_case_rejects_by_depth[d];
            summary->profile.canon_prepare_order_rejects_by_depth[d] +=
                src->canon_prepare_order_rejects_by_depth[d];
        }
        for (int n = 0; n <= MAXN_NAUTY; n++) {
            summary->profile.solve_graph_calls_by_n[n] += src->solve_graph_calls_by_n[n];
            summary->profile.solve_graph_raw_hits_by_n[n] += src->solve_graph_raw_hits_by_n[n];
            summary->profile.solve_graph_canon_hits_by_n[n] += src->solve_graph_canon_hits_by_n[n];
            summary->profile.hard_graph_nodes_by_n[n] += src->hard_graph_nodes_by_n[n];
            summary->profile.solve_graph_lookup_calls_by_n[n] += src->solve_graph_lookup_calls_by_n[n];
            summary->profile.solve_graph_connected_lookup_calls_by_n[n] +=
                src->solve_graph_connected_lookup_calls_by_n[n];
            summary->profile.solve_graph_component_calls_by_n[n] += src->solve_graph_component_calls_by_n[n];
            summary->profile.solve_graph_hard_misses_by_n[n] += src->solve_graph_hard_misses_by_n[n];
            summary->profile.hard_graph_articulation_by_n[n] += src->hard_graph_articulation_by_n[n];
            summary->profile.hard_graph_k2_separator_by_n[n] += src->hard_graph_k2_separator_by_n[n];
            summary->profile.solve_graph_time_by_n[n] += src->solve_graph_time_by_n[n];
            summary->profile.solve_graph_lookup_time_by_n[n] += src->solve_graph_lookup_time_by_n[n];
            summary->profile.solve_graph_connected_lookup_time_by_n[n] +=
                src->solve_graph_connected_lookup_time_by_n[n];
            summary->profile.solve_graph_raw_hit_time_by_n[n] += src->solve_graph_raw_hit_time_by_n[n];
            summary->profile.solve_graph_canon_hit_time_by_n[n] += src->solve_graph_canon_hit_time_by_n[n];
            summary->profile.solve_graph_component_time_by_n[n] += src->solve_graph_component_time_by_n[n];
            summary->profile.solve_graph_hard_miss_time_by_n[n] += src->solve_graph_hard_miss_time_by_n[n];
            summary->profile.solve_graph_hard_miss_separator_time_by_n[n] +=
                src->solve_graph_hard_miss_separator_time_by_n[n];
            summary->profile.solve_graph_hard_miss_pick_time_by_n[n] +=
                src->solve_graph_hard_miss_pick_time_by_n[n];
            summary->profile.solve_graph_hard_miss_delete_time_by_n[n] +=
                src->solve_graph_hard_miss_delete_time_by_n[n];
            summary->profile.solve_graph_hard_miss_contract_build_time_by_n[n] +=
                src->solve_graph_hard_miss_contract_build_time_by_n[n];
            summary->profile.solve_graph_hard_miss_contract_solve_time_by_n[n] +=
                src->solve_graph_hard_miss_contract_solve_time_by_n[n];
            summary->profile.solve_graph_hard_miss_store_time_by_n[n] +=
                src->solve_graph_hard_miss_store_time_by_n[n];
            for (int d = 0; d <= MAXN_NAUTY; d++) {
                summary->profile.hard_graph_nodes_by_n_degree[n][d] +=
                    src->hard_graph_nodes_by_n_degree[n][d];
            }
        }

        summary->task_timing.task_count += exec->thread_task_timing[i].task_count;
        summary->task_timing.task_time_sum += exec->thread_task_timing[i].task_time_sum;
        if (exec->thread_task_timing[i].task_time_max > summary->task_timing.task_time_max) {
            summary->task_timing.task_time_max = exec->thread_task_timing[i].task_time_max;
            summary->task_timing.task_max_index = exec->thread_task_timing[i].task_max_index;
        }
        for (int k = 0; k < TASK_PROFILE_TOPK; k++) {
            task_timing_insert_topk(&summary->task_timing,
                                    exec->thread_task_timing[i].top_indices[k],
                                    exec->thread_task_timing[i].top_times[k]);
        }
        if (exec->thread_queue_subtask_timing) {
            QueueSubtaskTimingStats* src_sub =
                exec->thread_queue_subtask_timing + (size_t)i * (size_t)(MAX_COLS + 1);
            for (int d = 0; d <= MAX_COLS; d++) {
                queue_subtask_merge(&summary->queue_subtask_timing[d], &src_sub[d]);
            }
        }
    }
}

static void print_execution_report(const RunConfig* run, const ExecutionState* exec,
                                   const ExecutionSummary* summary, double worker_time) {
    double total_elapsed = worker_time + run->prefix_generation_time;
    const ProfileStats* total_profile = &summary->profile;
    const TaskTimingStats* total_task_timing = &summary->task_timing;

    printf("\nWorker Complete in %.2f seconds.\n", worker_time);
    if (run->prefix_depth > 0) {
        printf("Total elapsed including prefix generation: %.2f seconds.\n", total_elapsed);
    }
    printf("Canonicalisation calls: %lld\n", exec->total_canon_calls);
    printf("Canonical cache hits: %lld (%.1f%%)\n", exec->total_cache_hits,
           exec->total_canon_calls > 0 ? 100.0 * exec->total_cache_hits / exec->total_canon_calls : 0.0);
    printf("Raw cache hits: %lld\n", exec->total_raw_cache_hits);
    if (PROFILE_BUILD) {
        printf("Profile:\n");
        printf("  canon_state_prepare_push: %lld calls, %.3fs\n",
               total_profile->canon_prepare_calls, total_profile->canon_prepare_time);
        printf("  canon_state_commit_push: %lld calls, %.3fs\n",
               total_profile->canon_commit_calls, total_profile->canon_commit_time);
        printf("  partial_graph_append: %lld calls, %.3fs\n",
               total_profile->partial_append_calls, total_profile->partial_append_time);
        printf("  build_structure_weight: %lld calls, %.3fs\n",
               total_profile->solve_structure_calls, total_profile->build_weight_time);
        printf("  solve_graph_poly: %lld calls, %.3fs\n",
               total_profile->solve_graph_calls, total_profile->solve_graph_time);
        printf("  get_canonical_graph: %lld calls, %.3fs\n",
               total_profile->nauty_calls, total_profile->get_canonical_graph_time);
        printf("    dense rows: %.3fs\n",
               total_profile->get_canonical_graph_dense_rows_time);
        printf("    build nauty input: %.3fs\n",
               total_profile->get_canonical_graph_build_input_time);
        printf("    densenauty: %.3fs\n",
               total_profile->nauty_time);
        printf("    rebuild canon graph: %.3fs\n",
               total_profile->get_canonical_graph_rebuild_time);
        printf("  hard graph nodes: %lld, max n %d, max degree %d\n",
               total_profile->hard_graph_nodes,
               total_profile->hard_graph_max_n,
               total_profile->hard_graph_max_degree);
        printf("  Graph solver by simplified n (inclusive time):\n");
        for (int n = 0; n <= MAXN_NAUTY; n++) {
            long long calls = total_profile->solve_graph_calls_by_n[n];
            long long raw_hits = total_profile->solve_graph_raw_hits_by_n[n];
            long long canon_hits = total_profile->solve_graph_canon_hits_by_n[n];
            long long hard = total_profile->hard_graph_nodes_by_n[n];
            double time_s = total_profile->solve_graph_time_by_n[n];
            if (calls == 0 && raw_hits == 0 && canon_hits == 0 && hard == 0) continue;
            printf("    n=%d: calls %lld, time %.3fs, raw hits %lld, canon hits %lld, hard nodes %lld\n",
                   n, calls, time_s, raw_hits, canon_hits, hard);
        }
        printf("  Graph solver outcomes by simplified n:\n");
        for (int n = 0; n <= MAXN_NAUTY; n++) {
            long long lookup_calls = total_profile->solve_graph_lookup_calls_by_n[n];
            long long connected_lookup_calls = total_profile->solve_graph_connected_lookup_calls_by_n[n];
            long long raw_hits = total_profile->solve_graph_raw_hits_by_n[n];
            long long canon_hits = total_profile->solve_graph_canon_hits_by_n[n];
            long long component_calls = total_profile->solve_graph_component_calls_by_n[n];
            long long hard_misses = total_profile->solve_graph_hard_misses_by_n[n];
            if (lookup_calls == 0 && connected_lookup_calls == 0 &&
                raw_hits == 0 && canon_hits == 0 &&
                component_calls == 0 && hard_misses == 0) {
                continue;
            }
            printf("    n=%d: lookup %lld/%.3fs, connected-lookup %lld/%.3fs, raw-hit %lld/%.3fs, canon-hit %lld/%.3fs, components %lld/%.3fs, hard-miss %lld/%.3fs\n",
                   n,
                   lookup_calls, total_profile->solve_graph_lookup_time_by_n[n],
                   connected_lookup_calls, total_profile->solve_graph_connected_lookup_time_by_n[n],
                   raw_hits, total_profile->solve_graph_raw_hit_time_by_n[n],
                   canon_hits, total_profile->solve_graph_canon_hit_time_by_n[n],
                   component_calls, total_profile->solve_graph_component_time_by_n[n],
                   hard_misses, total_profile->solve_graph_hard_miss_time_by_n[n]);
        }
        printf("  Hard-miss subphases by simplified n:\n");
        for (int n = 0; n <= MAXN_NAUTY; n++) {
            long long hard_misses = total_profile->solve_graph_hard_misses_by_n[n];
            if (hard_misses == 0) continue;
            printf("    n=%d: separator %.3fs, pick %.3fs, delete %.3fs, contract-build %.3fs, contract-solve %.3fs, store %.3fs\n",
                   n,
                   total_profile->solve_graph_hard_miss_separator_time_by_n[n],
                   total_profile->solve_graph_hard_miss_pick_time_by_n[n],
                   total_profile->solve_graph_hard_miss_delete_time_by_n[n],
                   total_profile->solve_graph_hard_miss_contract_build_time_by_n[n],
                   total_profile->solve_graph_hard_miss_contract_solve_time_by_n[n],
                   total_profile->solve_graph_hard_miss_store_time_by_n[n]);
        }
        if (g_profile_separators) {
            printf("  Hard-miss separator detection by simplified n:\n");
            for (int n = 0; n <= MAXN_NAUTY; n++) {
                long long hard_misses = total_profile->solve_graph_hard_misses_by_n[n];
                long long articulation = total_profile->hard_graph_articulation_by_n[n];
                long long k2 = total_profile->hard_graph_k2_separator_by_n[n];
                if (hard_misses == 0 && articulation == 0 && k2 == 0) continue;
                printf("    n=%d: hard-miss %lld, articulation %lld, k2-separator %lld\n",
                       n, hard_misses, articulation, k2);
            }
        } else {
            printf("  Hard-miss separator detection: disabled"
                   " (set RECT_PROFILE_SEPARATORS=1 to enable)\n");
        }
        printf("  Hard graph nodes by simplified n and max degree:\n");
        for (int n = 10; n <= MAXN_NAUTY; n++) {
            long long total_n = total_profile->hard_graph_nodes_by_n[n];
            if (total_n == 0) continue;
            printf("    n=%d:", n);
            for (int d = 0; d <= MAXN_NAUTY; d++) {
                long long count = total_profile->hard_graph_nodes_by_n_degree[n][d];
                if (count == 0) continue;
                printf(" deg%d=%lld", d, count);
            }
            printf("\n");
        }
        printf("  CanonState by depth:\n");
        for (int d = 0; d <= g_cols && d <= MAX_COLS; d++) {
            long long calls = total_profile->canon_prepare_calls_by_depth[d];
            long long accepts = total_profile->canon_prepare_accepts_by_depth[d];
            if (calls == 0) continue;
            double accept_rate = 100.0 * (double)accepts / (double)calls;
            double avg_stabilizer =
                accepts > 0 ? (double)total_profile->stabilizer_sum_by_depth[d] / (double)accepts : 0.0;
            double avg_scanned =
                calls > 0 ? (double)total_profile->canon_prepare_scanned_by_depth[d] / (double)calls : 0.0;
            double avg_active =
                calls > 0 ? (double)total_profile->canon_prepare_active_by_depth[d] / (double)calls : 0.0;
            double active_rate =
                total_profile->canon_prepare_scanned_by_depth[d] > 0
                    ? 100.0 * (double)total_profile->canon_prepare_active_by_depth[d] /
                          (double)total_profile->canon_prepare_scanned_by_depth[d]
                    : 0.0;
            printf("    depth %d: prepare %lld, accept %lld (%.1f%%), avg stabiliser %.1f, avg active %.1f/%.0f (%.1f%%)\n",
                   d, calls, accepts, accept_rate, avg_stabilizer, avg_active, avg_scanned, active_rate);
        }
        printf("  CanonState prepare branch mix by depth:\n");
        for (int d = 0; d <= g_cols && d <= MAX_COLS; d++) {
            long long calls = total_profile->canon_prepare_calls_by_depth[d];
            long long terminal_calls = total_profile->canon_prepare_terminal_calls_by_depth[d];
            long long fast_continue = total_profile->canon_prepare_fast_continue_by_depth[d];
            long long terminal_continue = total_profile->canon_prepare_terminal_continue_by_depth[d];
            long long equal_case = total_profile->canon_prepare_equal_case_calls_by_depth[d];
            long long equal_reject = total_profile->canon_prepare_equal_case_rejects_by_depth[d];
            long long order_reject = total_profile->canon_prepare_order_rejects_by_depth[d];
            if (calls == 0 && equal_case == 0 && order_reject == 0) continue;
            printf("    depth %d: terminal %lld, fast-continue %lld, terminal-continue %lld, equal-case %lld, equal-reject %lld, order-reject %lld\n",
                   d, terminal_calls, fast_continue, terminal_continue, equal_case,
                   equal_reject, order_reject);
        }
        if (total_task_timing->task_count > 0) {
            printf("  Task timings: %lld tasks, avg %.6fs, max %.6fs (task %lld)\n",
                   total_task_timing->task_count,
                   total_task_timing->task_time_sum / (double)total_task_timing->task_count,
                   total_task_timing->task_time_max,
                   total_task_timing->task_max_index);
            printf("  Slowest tasks:\n");
            for (int i = 0; i < TASK_PROFILE_TOPK; i++) {
                if (total_task_timing->top_times[i] <= 0.0) break;
                int pi, pj, pk, pl;
                if (decode_task_prefix(total_task_timing->top_indices[i], &pi, &pj, &pk, &pl)) {
                    if (pk >= 0 && pl >= 0) {
                        printf("    task %lld (%d,%d,%d,%d): %.6fs\n",
                               total_task_timing->top_indices[i], pi, pj, pk, pl,
                               total_task_timing->top_times[i]);
                    } else if (pk >= 0) {
                        printf("    task %lld (%d,%d,%d): %.6fs\n",
                               total_task_timing->top_indices[i], pi, pj, pk,
                               total_task_timing->top_times[i]);
                    } else if (pj >= 0) {
                        printf("    task %lld (%d,%d): %.6fs\n",
                               total_task_timing->top_indices[i], pi, pj,
                               total_task_timing->top_times[i]);
                    } else {
                        printf("    task %lld (%d): %.6fs\n",
                               total_task_timing->top_indices[i], pi,
                               total_task_timing->top_times[i]);
                    }
                } else {
                    printf("    task %lld: %.6fs\n",
                           total_task_timing->top_indices[i],
                           total_task_timing->top_times[i]);
                }
            }
        }
        if (run->use_runtime_split_queue) {
            printf("  Queue subtasks by depth:\n");
            for (int d = 0; d <= g_cols && d <= MAX_COLS; d++) {
                const QueueSubtaskTimingStats* qs = &summary->queue_subtask_timing[d];
                if (qs->task_count == 0) continue;
                printf("    depth %d: %lld subtasks, avg %.6fs, max %.6fs, avg solve_graph %.1f, avg nauty %.1f, avg hard nodes %.1f, max hard n %d, max hard deg %d\n",
                       d, qs->task_count, qs->task_time_sum / (double)qs->task_count, qs->task_time_max,
                       (double)qs->solve_graph_call_sum / (double)qs->task_count,
                       (double)qs->nauty_call_sum / (double)qs->task_count,
                       (double)qs->hard_graph_node_sum / (double)qs->task_count,
                       qs->max_hard_graph_n, qs->max_hard_graph_degree);
                for (int i = 0; i < TASK_PROFILE_TOPK; i++) {
                    const QueueSubtaskTopEntry* e = &qs->top[i];
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
}

static void write_task_times_report(void) {
    if (g_task_times_out_path) {
        write_task_times_file(g_task_times_out_path);
        printf("Task timing CSV: %s\n", g_task_times_out_path);
    }
}

static void print_final_output(const MainOptions* opts, const RunConfig* run, const Poly* poly) {
#if RECT_COUNT_K4
    printf("\nRectangle-free 4-colourings:\n");
    print_u128(poly->coeffs[0]);
    printf("\n");
#else
    printf("\nChromatic Polynomial P(x):\n");
    print_poly(*poly);

    printf("\nValues:\n");
    long long k_test = 4;
    printf("P(%lld) = ", k_test);
    print_u128(poly_eval(*poly, k_test));
    printf("\n");

    k_test = 5;
    printf("P(%lld) = ", k_test);
    print_u128(poly_eval(*poly, k_test));
    printf("\n");
#endif

    if (opts->poly_out_path) {
        PolyFileMeta meta = {
            .rows = g_rows,
            .cols = g_cols,
            .task_start = run->active_task_start,
            .task_end = run->active_task_end,
            .full_tasks = run->full_tasks,
        };
        write_poly_file(opts->poly_out_path, poly, &meta);
#if RECT_COUNT_K4
        printf("\nWrote fixed-4 shard to %s\n", opts->poly_out_path);
#else
        printf("\nWrote polynomial shard to %s\n", opts->poly_out_path);
#endif
    }
}

int main(int argc, char** argv) {
    MainOptions opts;
    RunConfig run;
    ExecutionState exec;
    ExecutionSummary summary;

    if (!parse_main_options(argc, argv, &opts)) return 1;
    if (!init_problem_and_run_config(&opts, &run)) {
        cleanup_run_config(&run);
        return 1;
    }
    if (!prepare_run_config(&opts, &run)) {
        cleanup_run_config(&run);
        return 1;
    }

    double start_time = omp_get_wtime();
    if (run.total_tasks > 0) {
        small_graph_lookup_init();
        connected_canon_lookup_init();
        if (PROFILE_BUILD) {
            printf("Small-graph lookup %s: %.2f seconds\n",
                   g_small_graph_lookup_loaded_from_file ? "load" : "initialisation",
                   g_small_graph_lookup_init_time);
            if (g_connected_canon_lookup_loaded) {
                printf("Connected canonical lookup n=%d load: %.2f seconds\n",
                       g_connected_canon_lookup_n, g_connected_canon_lookup_load_time);
            }
        }
    }
    if (!init_execution_state(&run, &exec)) {
        cleanup_run_config(&run);
        return 1;
    }

    execute_run_tasks(&run, start_time, &exec);

    if (exec.runtime_tasks_active) {
        runtime_task_system_print_summary(&exec.runtime_tasks);
    }
    accumulate_execution_poly(&exec, &global_poly);
    aggregate_execution_summary(&exec, &summary);

    cleanup_execution_state(&exec);
    progress_reporter_finish(&progress_reporter);

    double end_time = omp_get_wtime();
    double worker_time = end_time - start_time;
    print_execution_report(&run, &exec, &summary, worker_time);
    write_task_times_report();
    print_final_output(&opts, &run, &global_poly);

    cleanup_run_config(&run);

    return 0;
}
