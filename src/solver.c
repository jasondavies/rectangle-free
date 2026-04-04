#include "partition_poly.h"

void remove_vertex(Graph* g, int i) {
    uint64_t bit = UINT64_C(1) << i;
    if ((g->vertex_mask & bit) == 0) return;
    g->vertex_mask &= ~bit;
    g->n--;
}

static inline void record_hard_graph_node(ProfileStats* profile, int n, int max_degree) {
    if (PROFILE_BUILD && profile) {
        profile->hard_graph_nodes++;
        if (n >= 0 && n <= MAXN_NAUTY) profile->hard_graph_nodes_by_n[n]++;
        if (n >= 0 && n <= MAXN_NAUTY && max_degree >= 0 && max_degree <= MAXN_NAUTY) {
            profile->hard_graph_nodes_by_n_degree[n][max_degree]++;
        }
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

typedef enum {
    SG_OUTCOME_NONE = 0,
    SG_OUTCOME_LOOKUP,
    SG_OUTCOME_CONNECTED_LOOKUP,
    SG_OUTCOME_RAW_HIT,
    SG_OUTCOME_CANON_HIT,
    SG_OUTCOME_COMPONENTS,
    SG_OUTCOME_HARD_MISS,
} SolveGraphOutcome;

typedef struct {
    AdjWord row_mask;
    AdjWord raw_rows[MAXN_NAUTY];
    uint64_t raw_hash;
    int has_raw_rows;
} SolveGraphKeyRows;

static inline double begin_solve_graph_profile(ProfileStats* profile) {
    if (PROFILE_BUILD && profile) {
        profile->solve_graph_calls++;
        return omp_get_wtime();
    }
    return 0.0;
}

static inline void finish_solve_graph_profile(ProfileStats* profile, double solve_t0,
                                              int profile_n, SolveGraphOutcome outcome) {
    if (!(PROFILE_BUILD && profile)) return;

    double dt = omp_get_wtime() - solve_t0;
    profile->solve_graph_time += dt;
    if (profile_n < 0 || profile_n > MAXN_NAUTY) return;

    profile->solve_graph_time_by_n[profile_n] += dt;
    switch (outcome) {
        case SG_OUTCOME_LOOKUP:
            profile->solve_graph_lookup_calls_by_n[profile_n]++;
            profile->solve_graph_lookup_time_by_n[profile_n] += dt;
            break;
        case SG_OUTCOME_CONNECTED_LOOKUP:
            profile->solve_graph_connected_lookup_calls_by_n[profile_n]++;
            profile->solve_graph_connected_lookup_time_by_n[profile_n] += dt;
            break;
        case SG_OUTCOME_RAW_HIT:
            profile->solve_graph_raw_hit_time_by_n[profile_n] += dt;
            break;
        case SG_OUTCOME_CANON_HIT:
            profile->solve_graph_canon_hit_time_by_n[profile_n] += dt;
            break;
        case SG_OUTCOME_COMPONENTS:
            profile->solve_graph_component_calls_by_n[profile_n]++;
            profile->solve_graph_component_time_by_n[profile_n] += dt;
            break;
        case SG_OUTCOME_HARD_MISS:
            profile->solve_graph_hard_misses_by_n[profile_n]++;
            profile->solve_graph_hard_miss_time_by_n[profile_n] += dt;
            break;
        case SG_OUTCOME_NONE:
        default:
            break;
    }
}

static int solve_graph_prepare_raw_cache(const Graph* g, RowGraphCache* raw_cache,
                                         ProfileStats* profile, long long* local_raw_cache_hits,
                                         SolveGraphKeyRows* prep, GraphPoly* raw_cached) {
    memset(prep, 0, sizeof(*prep));
    prep->row_mask = (AdjWord)graph_row_mask(g->n);
    if (!g_use_raw_cache) return 0;

    prep->raw_hash = graph_fill_dense_key_rows(g, prep->row_mask, prep->raw_rows);
    prep->has_raw_rows = 1;
    if (!row_graph_cache_lookup_rows(raw_cache, prep->raw_hash, (uint32_t)g->n,
                                     prep->raw_rows, raw_cached, 1)) {
        return 0;
    }

    (*local_raw_cache_hits)++;
    if (PROFILE_BUILD && profile && g->n <= MAXN_NAUTY) {
        profile->solve_graph_raw_hits_by_n[g->n]++;
    }
    return 1;
}

static void solve_graph_build_canon(Graph* g, const SolveGraphKeyRows* prep,
                                    NautyWorkspace* ws, ProfileStats* profile,
                                    long long* local_canon_calls, Graph* canon) {
    if (prep->has_raw_rows) {
        get_canonical_graph_from_dense_rows((int)g->n, prep->raw_rows, canon, ws, profile);
    } else {
        get_canonical_graph(g, canon, ws, profile);
    }
    (*local_canon_calls)++;
}

static void solve_graph_store_raw_cache(const Graph* g, RowGraphCache* raw_cache,
                                        const SolveGraphKeyRows* prep, const GraphPoly* value) {
    if (!prep->has_raw_rows) return;
    store_row_graph_cache_entry_rows(raw_cache, prep->raw_hash, (uint32_t)g->n,
                                     prep->raw_rows, value);
}

static inline int graph_neighbors_form_clique(const Graph* g, uint64_t neighbors) {
    uint64_t rem = neighbors;
    while (rem) {
        int u = __builtin_ctzll(rem);
        if ((neighbors & ~((uint64_t)g->adj[u] & g->vertex_mask)) != (UINT64_C(1) << u)) {
            return 0;
        }
        rem &= rem - 1;
    }
    return 1;
}

static void graph_choose_branch_edge(const Graph* g, int* u_out, int* v_out, int* max_deg_out) {
    uint64_t active = g->vertex_mask;
    int max_deg = -1;
    int fallback_u = -1;
    int fallback_v = -1;
    int best_u = -1;
    int best_v = -1;
    int best_score = INT_MIN;

    while (active) {
        int u = __builtin_ctzll(active);
        uint64_t u_neighbors = (uint64_t)g->adj[u] & g->vertex_mask;
        int u_deg = __builtin_popcountll(u_neighbors);
        if (u_deg > max_deg) max_deg = u_deg;
        uint64_t rem = u_neighbors & ~((UINT64_C(1) << (u + 1)) - 1);
        while (rem) {
            int v = __builtin_ctzll(rem);
            uint64_t v_neighbors = (uint64_t)g->adj[v] & g->vertex_mask;
            int v_deg = __builtin_popcountll(v_neighbors);
            if (fallback_u < 0 || u_deg > __builtin_popcountll((uint64_t)g->adj[fallback_u] & g->vertex_mask)) {
                fallback_u = u;
                fallback_v = v;
            }

            uint64_t u_after = u_neighbors & ~(UINT64_C(1) << v);
            uint64_t v_after = v_neighbors & ~(UINT64_C(1) << u);
            int u_clique = graph_neighbors_form_clique(g, u_after);
            int v_clique = graph_neighbors_form_clique(g, v_after);
            uint64_t merged_neighbors =
                (u_neighbors | v_neighbors) & ~((UINT64_C(1) << u) | (UINT64_C(1) << v));
            int merged_clique = graph_neighbors_form_clique(g, merged_neighbors);
            int common = __builtin_popcountll(u_neighbors & v_neighbors);
            int score = 1000 * (u_clique + v_clique + merged_clique) + 16 * common + u_deg + v_deg;
            if (score > best_score) {
                best_score = score;
                best_u = u;
                best_v = v;
            }
            rem &= rem - 1;
        }
        active &= active - 1;
    }

    if (best_u >= 0) {
        *u_out = best_u;
        *v_out = best_v;
    } else {
        *u_out = fallback_u;
        *v_out = fallback_v;
    }
    *max_deg_out = max_deg;
}

#if RECT_COUNT_K4
static int simplify_graph_count4(Graph* g, uint64_t* multiplier,
                                 SolveGraphOutcome* outcome, GraphPoly* out_result) {
    int changed = 1;
    while (changed && g->n > SMALL_GRAPH_LOOKUP_MAX_N) {
        changed = 0;
        uint64_t active = g->vertex_mask;
        while (active) {
            int i = __builtin_ctzll(active);
            active &= active - 1;
            uint64_t neighbors = (uint64_t)g->adj[i] & g->vertex_mask;
            int degree = __builtin_popcountll(neighbors);

            if (degree == 0) {
                *multiplier *= 4;
                remove_vertex(g, i);
                changed = 1;
                continue;
            }

            if (!graph_neighbors_form_clique(g, neighbors)) {
                continue;
            }
            if (degree >= 4) {
                graph_poly_set_count4(0, out_result);
                *outcome = SG_OUTCOME_HARD_MISS;
                return 0;
            }
            *multiplier *= (uint64_t)(4 - degree);
            remove_vertex(g, i);
            changed = 1;
        }
    }
    return 1;
}
#endif

static void simplify_graph_poly_multiplier(Graph* g, GraphPoly* multiplier) {
    int changed = 1;
    while (changed && g->n > SMALL_GRAPH_LOOKUP_MAX_N) {
        changed = 0;
        uint64_t active = g->vertex_mask;
        while (active) {
            int i = __builtin_ctzll(active);
            active &= active - 1;
            uint64_t neighbors = (uint64_t)g->adj[i] & g->vertex_mask;
            int degree = __builtin_popcountll(neighbors);

            if (degree == 0) {
                graph_poly_mul_linear_ref(multiplier, 0, multiplier);
                remove_vertex(g, i);
                changed = 1;
                continue;
            }

            if (!graph_neighbors_form_clique(g, neighbors)) {
                continue;
            }
            graph_poly_mul_linear_ref(multiplier, degree, multiplier);
            remove_vertex(g, i);
            changed = 1;
        }
    }
}

void solve_graph_poly(const Graph* input_g, RowGraphCache* cache, RowGraphCache* raw_cache,
                      NautyWorkspace* ws, long long* local_canon_calls,
                      long long* local_cache_hits, long long* local_raw_cache_hits,
                      ProfileStats* profile, GraphPoly* out_result) {
#if RECT_COUNT_K4
    Graph g = *input_g;
    double solve_t0 = begin_solve_graph_profile(profile);
    int profile_n = 0;
    SolveGraphOutcome outcome = SG_OUTCOME_NONE;

    uint64_t multiplier = 1;
    if (!simplify_graph_count4(&g, &multiplier, &outcome, out_result)) goto done;

    profile_n = g.n;
    if (PROFILE_BUILD && profile && profile_n >= 0 && profile_n <= MAXN_NAUTY) {
        profile->solve_graph_calls_by_n[profile_n]++;
    }

    if (g.n == 0) {
        graph_poly_set_count4(multiplier, out_result);
        goto done;
    }

    if (g.n <= SMALL_GRAPH_LOOKUP_MAX_N) {
        uint64_t count4 = small_graph_lookup_load_count4(g.n, small_graph_pack_mask(&g));
        graph_poly_set_count4(multiplier * count4, out_result);
        outcome = SG_OUTCOME_LOOKUP;
        goto done;
    }

    SolveGraphKeyRows key_rows;
    GraphPoly raw_cached;
    if (solve_graph_prepare_raw_cache(&g, raw_cache, profile, local_raw_cache_hits,
                                      &key_rows, &raw_cached)) {
        graph_poly_set_count4(multiplier * graph_poly_get_count4(&raw_cached), out_result);
        outcome = SG_OUTCOME_RAW_HIT;
        goto done;
    }

    GraphPoly res;
    uint64_t component_masks[MAXN_NAUTY];
    int component_count = graph_collect_components(&g, component_masks);
    if (component_count > 1) {
        uint64_t total = 1;
        outcome = SG_OUTCOME_COMPONENTS;
        for (int i = 0; i < component_count; i++) {
            Graph subgraph;
            induced_subgraph_from_mask(&g, component_masks[i], &subgraph);
            GraphPoly part;
            solve_graph_poly(&subgraph, cache, raw_cache, ws,
                             local_canon_calls, local_cache_hits, local_raw_cache_hits,
                             profile, &part);
            total *= graph_poly_get_count4(&part);
        }
        graph_poly_set_count4(total, &res);
    } else {
        Graph canon;
        solve_graph_build_canon(&g, &key_rows, ws, profile, local_canon_calls, &canon);
        uint64_t hash = hash_graph(&canon);

        if (row_graph_cache_lookup_poly(cache, hash, (uint32_t)canon.n, &canon,
                                        (AdjWord)ADJWORD_MASK, &res, 1)) {
            (*local_cache_hits)++;
            solve_graph_store_raw_cache(&g, raw_cache, &key_rows, &res);
            if (PROFILE_BUILD && profile && canon.n <= MAXN_NAUTY) {
                profile->solve_graph_canon_hits_by_n[canon.n]++;
            }
            graph_poly_set_count4(multiplier * graph_poly_get_count4(&res), out_result);
            outcome = SG_OUTCOME_CANON_HIT;
            goto done;
        }

        if (shared_graph_cache_lookup_poly(g_shared_graph_cache, hash, (uint32_t)canon.n,
                                           &canon, ADJWORD_MASK, &res)) {
            store_row_graph_cache_entry(cache, hash, (uint32_t)canon.n, &canon,
                                        (AdjWord)ADJWORD_MASK, &res);
            (*local_cache_hits)++;
            if (PROFILE_BUILD && profile && canon.n <= MAXN_NAUTY) {
                profile->solve_graph_canon_hits_by_n[canon.n]++;
            }
            solve_graph_store_raw_cache(&g, raw_cache, &key_rows, &res);
            graph_poly_set_count4(multiplier * graph_poly_get_count4(&res), out_result);
            outcome = SG_OUTCOME_CANON_HIT;
            goto done;
        }

        uint64_t connected_lookup = connected_canon_lookup_load_count4(&canon);
        if (connected_lookup != UINT64_MAX) {
            graph_poly_set_count4(connected_lookup, &res);
            store_row_graph_cache_entry(cache, hash, (uint32_t)canon.n, &canon,
                                        (AdjWord)ADJWORD_MASK, &res);
            solve_graph_store_raw_cache(&g, raw_cache, &key_rows, &res);
            graph_poly_set_count4(multiplier * connected_lookup, out_result);
            outcome = SG_OUTCOME_CONNECTED_LOOKUP;
            goto done;
        }

        const Graph* branch_g = &canon;
        int max_deg = -1;
        for (int i = 0; i < branch_g->n; i++) {
            int d = __builtin_popcountll(branch_g->adj[i]);
            if (d > max_deg) max_deg = d;
        }
        if (max_deg > 0) record_hard_graph_node(profile, branch_g->n, max_deg);
        outcome = SG_OUTCOME_HARD_MISS;
        uint64_t count4 = count_graph_4_dsat(branch_g);
        graph_poly_set_count4(count4, &res);
        store_row_graph_cache_entry(cache, hash, (uint32_t)canon.n, &canon,
                                    (AdjWord)ADJWORD_MASK, &res);
        shared_graph_cache_export(hash, (uint32_t)canon.n, &canon, ADJWORD_MASK, &res);
    }

    solve_graph_store_raw_cache(&g, raw_cache, &key_rows, &res);
    graph_poly_set_count4(multiplier * graph_poly_get_count4(&res), out_result);
done:
    finish_solve_graph_profile(profile, solve_t0, profile_n, outcome);
    return;
#else
    Graph g = *input_g;
    double solve_t0 = begin_solve_graph_profile(profile);
    int profile_n = 0;
    SolveGraphOutcome outcome = SG_OUTCOME_NONE;
    GraphPoly multiplier;
    graph_poly_one_ref(&multiplier);

    simplify_graph_poly_multiplier(&g, &multiplier);

    profile_n = g.n;
    if (PROFILE_BUILD && profile && profile_n >= 0 && profile_n <= MAXN_NAUTY) {
        profile->solve_graph_calls_by_n[profile_n]++;
    }
    
    if (g.n == 0) {
        *out_result = multiplier;
        goto done;
    }

    if (g.n <= SMALL_GRAPH_LOOKUP_MAX_N) {
        GraphPoly small_poly;
        small_graph_lookup_load_graph_poly(g.n, small_graph_pack_mask(&g), &small_poly);
        graph_poly_mul_ref(&multiplier, &small_poly, out_result);
        outcome = SG_OUTCOME_LOOKUP;
        goto done;
    }

    SolveGraphKeyRows key_rows;
    GraphPoly raw_cached;

    if (solve_graph_prepare_raw_cache(&g, raw_cache, profile, local_raw_cache_hits,
                                      &key_rows, &raw_cached)) {
        graph_poly_mul_ref(&multiplier, &raw_cached, out_result);
        outcome = SG_OUTCOME_RAW_HIT;
        goto done;
    }

    GraphPoly res;
    uint64_t component_masks[MAXN_NAUTY];
    int component_count = graph_collect_components(&g, component_masks);
    if (component_count > 1) {
        outcome = SG_OUTCOME_COMPONENTS;
        graph_poly_one_ref(&res);
        for (int i = 0; i < component_count; i++) {
            Graph subgraph;
            induced_subgraph_from_mask(&g, component_masks[i], &subgraph);
            GraphPoly part;
            solve_graph_poly(&subgraph, cache, raw_cache, ws,
                             local_canon_calls, local_cache_hits, local_raw_cache_hits,
                             profile, &part);
            graph_poly_mul_ref(&res, &part, &res);
        }
    } else {
        Graph canon;
        solve_graph_build_canon(&g, &key_rows, ws, profile, local_canon_calls, &canon);
        
        uint64_t hash = hash_graph(&canon);

        if (row_graph_cache_lookup_poly(cache, hash, (uint32_t)canon.n, &canon,
                                        (AdjWord)ADJWORD_MASK, &res, 1)) {
            (*local_cache_hits)++;
            solve_graph_store_raw_cache(&g, raw_cache, &key_rows, &res);
            if (PROFILE_BUILD && profile && canon.n <= MAXN_NAUTY) {
                profile->solve_graph_canon_hits_by_n[canon.n]++;
            }
            graph_poly_mul_ref(&multiplier, &res, out_result);
            outcome = SG_OUTCOME_CANON_HIT;
            goto done;
        }

        if (shared_graph_cache_lookup_poly(g_shared_graph_cache, hash, (uint32_t)canon.n,
                                           &canon, ADJWORD_MASK, &res)) {
            store_row_graph_cache_entry(cache, hash, (uint32_t)canon.n, &canon,
                                        (AdjWord)ADJWORD_MASK, &res);
            (*local_cache_hits)++;
            if (PROFILE_BUILD && profile && canon.n <= MAXN_NAUTY) {
                profile->solve_graph_canon_hits_by_n[canon.n]++;
            }
            solve_graph_store_raw_cache(&g, raw_cache, &key_rows, &res);
            graph_poly_mul_ref(&multiplier, &res, out_result);
            outcome = SG_OUTCOME_CANON_HIT;
            goto done;
        }

        if (connected_canon_lookup_load_graph_poly(&canon, &res)) {
            store_row_graph_cache_entry(cache, hash, (uint32_t)canon.n, &canon,
                                        (AdjWord)ADJWORD_MASK, &res);
            solve_graph_store_raw_cache(&g, raw_cache, &key_rows, &res);
            graph_poly_mul_ref(&multiplier, &res, out_result);
            outcome = SG_OUTCOME_CONNECTED_LOOKUP;
            goto done;
        }

        // Deletion-contraction on canonical graph
        double hard_sep_t = 0.0;
        double hard_pick_t = 0.0;
        double hard_del_t = 0.0;
        double hard_cont_build_t = 0.0;
        double hard_cont_solve_t = 0.0;
        double hard_store_t = 0.0;
        const Graph* branch_g = &canon;
        double phase_t0 = 0.0;
        if (PROFILE_BUILD && profile && branch_g->n >= 0 && branch_g->n <= MAXN_NAUTY) {
            phase_t0 = omp_get_wtime();
        }
        int max_deg = -1, u = -1, v = -1;
        graph_choose_branch_edge(branch_g, &u, &v, &max_deg);
        if (PROFILE_BUILD && profile && branch_g->n >= 0 && branch_g->n <= MAXN_NAUTY) {
            hard_pick_t += omp_get_wtime() - phase_t0;
        }
        if (u != -1 && max_deg > 0) record_hard_graph_node(profile, branch_g->n, max_deg);
        outcome = SG_OUTCOME_HARD_MISS;
        if (PROFILE_BUILD && g_profile_separators && profile &&
            branch_g->n >= 10 && branch_g->n <= MAXN_NAUTY) {
            phase_t0 = omp_get_wtime();
            if (graph_has_articulation_point(branch_g)) {
                profile->hard_graph_articulation_by_n[branch_g->n]++;
            }
            if (graph_has_k2_separator(branch_g)) {
                profile->hard_graph_k2_separator_by_n[branch_g->n]++;
            }
            hard_sep_t += omp_get_wtime() - phase_t0;
        }

        if (u != -1 && v != -1) {
            // Deletion: remove edge (u,v)
            Graph g_del = *branch_g;
            g_del.adj[u] &= ~(1ULL << v);
            g_del.adj[v] &= ~(1ULL << u);
            GraphPoly p_del;
            if (PROFILE_BUILD && profile && branch_g->n >= 0 && branch_g->n <= MAXN_NAUTY) {
                phase_t0 = omp_get_wtime();
            }
            solve_graph_poly(&g_del, cache, raw_cache, ws,
                             local_canon_calls, local_cache_hits, local_raw_cache_hits,
                             profile, &p_del);
            if (PROFILE_BUILD && profile && branch_g->n >= 0 && branch_g->n <= MAXN_NAUTY) {
                hard_del_t += omp_get_wtime() - phase_t0;
            }

            // Contraction: merge v into u
            Graph g_cont = *branch_g;
            if (PROFILE_BUILD && profile && branch_g->n >= 0 && branch_g->n <= MAXN_NAUTY) {
                phase_t0 = omp_get_wtime();
            }
            uint64_t merged_nbrs =
                ((uint64_t)g_cont.adj[u] | (uint64_t)g_cont.adj[v]) &
                g_cont.vertex_mask & ~((UINT64_C(1) << u) | (UINT64_C(1) << v));
            g_cont.adj[u] = (AdjWord)merged_nbrs;
            uint64_t nbrs = merged_nbrs;
            while (nbrs) {
                int k = __builtin_ctzll(nbrs);
                g_cont.adj[k] |= (AdjWord)(UINT64_C(1) << u);
                nbrs &= nbrs - 1;
            }
            remove_vertex(&g_cont, v);
            if (PROFILE_BUILD && profile && branch_g->n >= 0 && branch_g->n <= MAXN_NAUTY) {
                hard_cont_build_t += omp_get_wtime() - phase_t0;
                phase_t0 = omp_get_wtime();
            }
            GraphPoly p_cont;
            solve_graph_poly(&g_cont, cache, raw_cache, ws,
                             local_canon_calls, local_cache_hits, local_raw_cache_hits,
                             profile, &p_cont);
            if (PROFILE_BUILD && profile && branch_g->n >= 0 && branch_g->n <= MAXN_NAUTY) {
                hard_cont_solve_t += omp_get_wtime() - phase_t0;
            }

            graph_poly_sub_ref(&p_del, &p_cont, &res);
        } else {
            graph_poly_one_ref(&res);
            for (int k = 0; k < branch_g->n; k++) graph_poly_mul_linear_ref(&res, 0, &res);
        }

        if (PROFILE_BUILD && profile && branch_g->n >= 0 && branch_g->n <= MAXN_NAUTY) {
            phase_t0 = omp_get_wtime();
        }
        store_row_graph_cache_entry(cache, hash, (uint32_t)canon.n, &canon,
                                    (AdjWord)ADJWORD_MASK, &res);
        shared_graph_cache_export(hash, (uint32_t)canon.n, &canon, ADJWORD_MASK, &res);
        if (PROFILE_BUILD && profile && branch_g->n >= 0 && branch_g->n <= MAXN_NAUTY) {
            hard_store_t += omp_get_wtime() - phase_t0;
            profile->solve_graph_hard_miss_separator_time_by_n[branch_g->n] += hard_sep_t;
            profile->solve_graph_hard_miss_pick_time_by_n[branch_g->n] += hard_pick_t;
            profile->solve_graph_hard_miss_delete_time_by_n[branch_g->n] += hard_del_t;
            profile->solve_graph_hard_miss_contract_build_time_by_n[branch_g->n] += hard_cont_build_t;
            profile->solve_graph_hard_miss_contract_solve_time_by_n[branch_g->n] += hard_cont_solve_t;
            profile->solve_graph_hard_miss_store_time_by_n[branch_g->n] += hard_store_t;
        }
    }

    solve_graph_store_raw_cache(&g, raw_cache, &key_rows, &res);
    graph_poly_mul_ref(&multiplier, &res, out_result);
done:
    finish_solve_graph_profile(profile, solve_t0, profile_n, outcome);
#endif
}
