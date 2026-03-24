# Partition Poly 7xn Experiments

## 2026-03-24

### Experiment 0: Prefix geometry baseline for `7x5`
- Goal: understand how much parallel slack the current prefixing strategy exposes before changing code.
- Command: `OMP_NUM_THREADS=1 ./partition_poly 7 5 --task-end 0 --prefix-depth 2`
- Result: `385003` prefixes, `2.94 MiB` of prefix arrays, negligible generation time (`0.00s`).
- Command: `OMP_NUM_THREADS=1 ./partition_poly 7 5 --task-end 0 --prefix-depth 3`
- Result: `112805879` prefixes, `1290.96 MiB` of prefix arrays, `0.78s` prefix generation.
- Conclusion: full depth-3 prefixing is too memory-hungry to use as the default route to better 32-core scaling on `7x5`. Future experiments should prefer adaptive subdivision or cheaper runtime scheduling changes over unconditional depth-3 task materialisation.

### Experiment 1: Sampled `7x5` baseline and scaling failure
- Goal: find the real hot path on `7x5` and see whether the current depth-2 tasking scales on one machine.
- Command: `OMP_NUM_THREADS=1 ./partition_poly 7 5 --prefix-depth 2 --task-stride 3073 --profile`
- Sample size: `126` depth-2 prefixes spread across the full `385003`-task space.
- Result: `36.47s` wall time.
- Hotspots:
  - `canon_state_prepare_push`: `21.900s`
  - `canon_state_commit_push`: `9.662s`
  - `solve_graph_poly`: `4.587s`
  - `get_canonical_graph/densenauty`: `0.336s`
- Depth profile: depth `4` dominates (`1608145` prepares, `664476` accepts).
- Task balance: the slowest sampled task (`12292`) took `33.67s` by itself.
- Matching command: `OMP_NUM_THREADS=32 ./partition_poly 7 5 --prefix-depth 2 --task-stride 3073`
- Result: `35.98s` wall time, essentially no speedup.
- Conclusion: the first-order problem for `7x5` is not nauty or graph caching. It is the symmetry front-end plus catastrophic task imbalance at depth-2 prefix granularity. A single pathological prefix can dominate the entire run, so better coarse task splitting is a prerequisite for anything close to linear scaling.

### Experiment 2: Existing adaptive subdivision geometry for `7x5`
- Goal: check whether the current adaptive prefix generation already offers a practical way to split heavy depth-2 prefixes without the cost of full depth-3 materialisation.
- Command: `OMP_NUM_THREADS=1 ./partition_poly 7 5 --task-end 0 --prefix-depth 2 --adaptive-subdivide`
- Result: `407604` prefixes, `4.66 MiB` of prefix arrays, `4.89s` prefix generation.
- Conclusion: adaptive subdivision looks geometrically cheap enough to be worth pushing further. It avoids the `1.29 GiB` depth-3 blow-up while creating only `22601` extra tasks. The next code experiments should be about making that adaptive path cheap enough, broad enough, or automatic enough to recover single-machine scaling on `7x5+`.

### Experiment 3: Remove `CanonState` row snapshots and undo by deletion
- Goal: cut `canon_state_commit_push` memory traffic by avoiding per-row snapshot copies in the hot path.
- Change: replaced saved-row copies with an undo path that removed the lazily materialised values from the sorted row during `canon_state_pop`.
- Benchmark command: `OMP_NUM_THREADS=1 ./partition_poly 7 5 --prefix-depth 2 --task-stride 3073 --profile`
- Result: regression.
  - Baseline: `36.47s`
  - Experiment: `40.23s`
  - `canon_state_commit_push` improved from `9.662s` to `7.899s`, but total runtime still got worse.
- Interpretation: the extra delete/repack work in `pop`, plus the worse locality from touching each row repeatedly during undo, costs more than the snapshot copies saved in `commit`.
- Outcome: rejected and reverted.

### Experiment 4: Compare in `prepare_push`, materialise only in `commit_push`
- Goal: reduce wasted work on rejected candidates by avoiding full-row scratch writes during `canon_state_prepare_push`.
- Change: replaced row materialisation in `prepare` with a small on-stack merge/compare, then rebuilt the accepted rows during `commit`.
- Benchmark command: `OMP_NUM_THREADS=1 ./partition_poly 7 5 --prefix-depth 2 --task-stride 3073 --profile`
- Result: regression.
  - Baseline: `36.47s`
  - Experiment: `38.13s`
  - `canon_state_prepare_push` improved from `21.900s` to `17.440s`
- `canon_state_commit_push` regressed from `9.662s` to `15.814s`
- Interpretation: the saved scratch writes in `prepare` were real, but rebuilding accepted rows in `commit` costs even more on this workload because depth-4 acceptance is still high (`41.3%`).
- Outcome: rejected and reverted.

### Experiment 5: Replace adaptive worst-case prefix allocation with a growable buffer
- Goal: remove the hidden worst-case memory reservation in adaptive mode and make deeper adaptive splitting feasible.
- Problem: the old adaptive `7x5` path allocated arrays for the full depth-3 worst case (`112805879` prefixes) even though only `407604` adaptive tasks were emitted.
- Change: replaced the preallocation with a growable `PrefixTaskBuffer` that reserves only what is actually appended.
- Verification:
  - `OMP_NUM_THREADS=1 ./partition_poly 7 5 --task-end 0 --prefix-depth 2 --adaptive-subdivide`
  - Result: still `407604` adaptive tasks, `4.66 MiB` logical task payload, `4.90s` prefix generation.
  - `OMP_NUM_THREADS=1 ./partition_poly 7 2 --task-end 200 --prefix-depth 2`
  - Result: unchanged polynomial on the smoke test.
- Interpretation: this is a real memory optimisation and a prerequisite for more aggressive adaptive prefixing. It does not change the current task list or output, but it removes a very large avoidable allocation from the `7x5+` adaptive path.
- Outcome: accepted.

### Experiment 6: Adaptive depth-4 splitting from `prefix-depth 2`
- Goal: break the remaining heavy adaptive tasks down one level further without materialising the full depth-4 space.
- Change: extended adaptive subdivision so a depth-2 prefix could expand to depth-3 children and then selectively to depth-4 grandchildren.
- Probe command: `OMP_NUM_THREADS=1 ./partition_poly 7 5 --task-end 0 --prefix-depth 2 --adaptive-subdivide --adaptive-max-depth 4`
- Result: rejected before full benchmarking. Prefix generation alone stayed busy for well over `50s` without finishing, which is already too expensive for the intended use.
- Interpretation: the extra branching factor at depth 4 overwhelms the cheap-task advantage. If we revisit deeper adaptive splitting later, it will need a much sharper heavy-prefix predictor or a cheaper counting pass than the naïve recursive expansion attempted here.
- Outcome: rejected and reverted.

### Experiment 7: Use `dynamic,1` for adaptive `prefix-depth 2`
- Goal: improve 32-core utilisation once adaptive subdivision turns a few huge depth-2 tasks into a modest number of uneven depth-3 tasks.
- Change: switched adaptive `prefix-depth 2` scheduling from `dynamic,8` to `dynamic,1`, while leaving the non-adaptive depth-2 path at `dynamic,8`.
- Benchmark command: `OMP_NUM_THREADS=32 ./partition_poly 7 5 --prefix-depth 2 --adaptive-subdivide --task-stride 3235`
- Result: strong win on the same sampled adaptive workload.
  - Before: `Worker Complete 17.41s`, `Total 22.29s`
  - After: `Worker Complete 6.55s`, `Total 11.45s`
- Interpretation: chunk size `8` was starving the scheduler on this kind of irregular adaptive workload. Once each adaptive task became its own schedulable unit, 32-thread execution improved by about `2.66x` in the worker phase and about `1.95x` overall, with identical polynomial output.
- Outcome: accepted.

### Experiment 8: Use `dynamic,1` for 7-row non-adaptive `prefix-depth 2`
- Goal: check whether the same scheduler-granularity change helps the non-adaptive `7x5` path as well.
- Temporary change for the experiment: widened the `dynamic,1` rule to all depth-2 runs.
- Benchmark command: `OMP_NUM_THREADS=32 ./partition_poly 7 5 --prefix-depth 2 --task-stride 3073`
- Result: modest but real improvement on the exact same sampled workload.
  - Before: `Worker Complete 35.97s`, `Total 35.98s`
  - Experiment: `Worker Complete 34.41s`, `Total 34.41s`
- Interpretation: non-adaptive depth-2 still suffers mostly from one giant coarse task, so chunk size alone cannot fix scaling there. But `dynamic,1` does not hurt and still buys about `4%` on the sampled `7x5` run.
- Final decision: keep `dynamic,1` for `g_rows == 7` depth-2 work, but preserve the older `dynamic,8` default for smaller-row non-adaptive cases.
- Outcome: accepted.

### Experiment 9: Make `7x7` structural defaults cheaper before chasing more parallel speed
- Goal: remove obvious `7x7` startup and memory hazards before the next wave of scaling experiments.
- Change:
  - force the default `prefix_depth` to `2` for `g_rows == 7 && g_cols >= 6`
  - shrink graph-cache coefficient slabs to the actual conflict-graph bound `g_cols * floor(g_rows / 2) + 1`
  - move connected-component splitting ahead of nauty canonicalisation in `solve_graph_poly()`
- Validation command: `RECT_PROGRESS_STEP=1000000 OMP_NUM_THREADS=1 ./partition_poly 7 7 --task-end 0`
- Result:
  - default `7x7` prefixing now stays at depth `2`
  - task count remains `385003`
  - prefix storage stays at `2.94 MiB`
  - scheduling remains `dynamic,1`
- Validation command: `RECT_PROGRESS_STEP=1000000 OMP_NUM_THREADS=1 ./partition_poly 7 2 --task-end 200 --prefix-depth 2`
- Result: unchanged smoke-test polynomial
  - `P(x) = 280*x^6 - 2590*x^5 + 9562*x^4 - 17262*x^3 + 15037*x^2 - 5027*x`
  - `P(4) = 58308`
  - `P(5) = 450540`
- Memory impact: for `7x7`, each graph-cache slot now stores `22` coefficients instead of `50`, a `56%` reduction in coefficient-slab footprint for both the canonical and raw per-thread caches.
- Interpretation: this is the right structural baseline for `7x7`. It removes the bad default depth-3 route, cuts a large chunk of per-thread cache memory, and avoids wasting nauty work on disconnected conflict graphs. I could not take a clean 32-thread timing on this batch because the host was already saturated by another long-running 32-thread solver job, so the throughput comparison is deferred to a later experiment.
- Outcome: accepted.
