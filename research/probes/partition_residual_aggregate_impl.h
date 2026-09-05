// Experimental second-stage aggregation. Included only by aggregate.c's A/B
// build, after its private entry definition and exact hash/equality helpers.
#ifdef RECT_RESIDUAL_CENSUS
#error "Run the census and substitution experiment in separate builds"
#endif

static TerminalAggregateEntry* prepare_residual_entries(
    TerminalAggregator* aggregator, ProfileStats* profile) {
    double start = (PROFILE_BUILD && profile) ? omp_get_wtime() : 0.0;
    for (size_t i = 0; i < aggregator->capacity; i++) {
        TerminalAggregateEntry* source = &aggregator->entries[i];
        if (!source->used) continue;
        Graph reduced = source->graph, dense;
        GraphPoly multiplier;
        graph_poly_one_ref(&multiplier);
        simplify_graph_poly_multiplier(&reduced, &multiplier);
        dense.n = (uint8_t)graph_build_dense_rows(&reduced, dense.adj);
        dense.vertex_mask = graph_row_mask(dense.n);

        Poly weight;
        poly_mul_graph_ref(&source->weight, &multiplier, &weight);
        uint64_t hash = terminal_graph_hash(&dense);
        size_t slot = (size_t)hash & aggregator->mask;
        // There are at most as many residuals as raw entries, so the existing
        // raw table's <=75% load bound also bounds this table's probing.
        for (;;) {
            TerminalAggregateEntry* target = &aggregator->residual_entries[slot];
            if (!target->used) {
                target->used = 1;
                target->hash = hash;
                target->graph.n = dense.n;
                target->graph.vertex_mask = dense.vertex_mask;
                memcpy(target->graph.adj, dense.adj, dense.n * sizeof(AdjWord));
                target->weight.deg = weight.deg;
                memcpy(target->weight.coeffs, weight.coeffs,
                       (weight.deg + 1) * sizeof(PolyCoeff));
                aggregator->residual_unique++;
                break;
            }
            if (target->hash == hash && terminal_graph_equal(&target->graph, &dense)) {
                poly_accumulate_checked(&target->weight, &weight);
                break;
            }
            slot = (slot + 1) & aggregator->mask;
        }
        aggregator->residual_inputs++;
        source->used = 0;
    }
    if (PROFILE_BUILD && profile)
        aggregator->residual_prepare_seconds += omp_get_wtime() - start;
    return aggregator->residual_entries;
}
