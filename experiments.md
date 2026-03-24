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
