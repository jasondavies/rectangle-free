# Reuse-budgeted 8×8 column cuts

CPU research gate following Experiment 369's lost-left-reuse result. This
does not implement shared-column responses or change the production solver.
See Experiments 505–512 in [the log](../../docs/experiments.md) for measurements.

## What is optimized

Each original outer record has exact alternative column permutations, and
optionally a transpose, producing 4+4 selected/complement joins. Their outer
coefficients are unchanged. Both counts use the same cut; either half may be
the resident left side. Candidate zero is the **literal production gauge**,
not a separately normalized or more expensive reference.

The C++ exporter imports the production canonical distribution factory and
token-plane representative convention through the existing family-census
model. It counts ordinary/swapped padded 16×8 weight-class tiles. The metric
does not predict instruction latency, screening, occupancy, or wall time.

Every labelled half-pair is a layout identity. Its modeled output size is
eight bytes per suffix, twelve per bucket, sixteen per weight class, and
48 bytes for the selected/complement pair descriptor. Builder scratch,
allocation high-water effects, and repeated construction in different GPU
batches are **not** included. Identity count is capped as well as bytes so
that many tiny layouts cannot hide extra planning/build work. Keeping raw
half keys, rather than row-canonical IDs, preserves labelled-layout costs.

The selector starts with all baseline left/right layouts reserved. For each
candidate left layout it builds a bundle of improving record moves, charging
that left once and each additional right once. It greedily opens the bundle
with the best saving relative to normalized added bytes/identities. It never
exceeds either side's budget or increases total modeled tiles. Unused
baseline layouts stay reserved: this deliberately conservative heuristic is
not an optimal facility-location solver.

Two **optimistic savings bounds** distinguish heuristic failure from a weak
candidate menu:

- Ignore all right-layout costs. Give existing left layouts for free. Sum
  individual new-left savings, allowing overlap to be counted repeatedly.
  Bound additional lefts by the best identity-count bound and fractional
  byte-knapsack bound. This bounds the reserved-baseline problem.
- Also permit up to the whole final left budget in *additional* left layouts,
  while continuing to give baseline layouts for free. This relaxed bound
  applies even to selectors that retire unused baseline layouts.

Both bounds are clipped by the unconstrained per-record oracle. They bound
only the exported candidate menu on the sampled population, not all possible
8×8 algorithms or campaign-wide performance. Small exhaustive synthetic
instances test that neither bound understates a feasible optimum.

The optional `solve --memory-only` diagnostic removes effective identity
caps while retaining the same byte caps. Its additional small layouts are
reported explicitly; it does not claim that their planning/build cost is
free. The identity-constrained bounds are not attached to this relaxed run.

## Population and reproduction

Requires a C++17 compiler and the nauty development library. The group sampler
also uses NumPy; the selector itself uses only the Python standard library.

Sample **complete left groups**, not isolated records, from four input solve
shards. Choose one group uniformly within each fanout band 8–31, 64–255 and
256–1023 per shard. Record all selected group sizes and source indices.
This stratified panel is intentionally bounded and is not an unbiased work
sample: singleton groups and groups above 1023 records are not represented.
Uniqueness and memory totals pool the selected groups, optimistically allowing
right layouts to be reused once across the panel. A future GPU A/B must charge
real reownership, batch rebuilding, and all offline preparation.

The four-cut menu is fixed before this experiment from historical candidates:
vertical `0x0f`, horizontal `0x0f`, horizontal `0x17`, vertical `0x17`.
It has eight execution directions. The all-cut mode includes all 35 unordered
vertical and 35 unordered horizontal cuts, each in both execution directions.
All-cut enumeration on smaller complete groups is a menu-width check, not
evidence about the omitted high-fanout groups.

```sh
make reuse-budget-cut-test
python3 research/probes/reuse_budget_cut.py sample build/reuse-cut-panel \
  ../rectangle-free-data-v2/8x8-transpose/solve/s0000.orbits \
  ../rectangle-free-data-v2/8x8-transpose/solve/s0256.orbits \
  ../rectangle-free-data-v2/8x8-transpose/solve/s0512.orbits \
  ../rectangle-free-data-v2/8x8-transpose/solve/s0768.orbits

ulimit -v 8388608
timeout 600 build/reuse_budget_cut_census build/reuse-cut-panel/records.tsv four \
  > build/reuse-cut-panel/candidates.jsonl
python3 research/probes/reuse_budget_cut.py solve \
  build/reuse-cut-panel/candidates.jsonl build/reuse-cut-panel/result.json
```

Use `all` instead of `four` for the full candidate menu. The exporter caps
inputs at 16,384 records and refuses invalid/truncated rows. The selector
rejects missing completion markers, duplicate source records, duplicate
layout IDs and unresolved layout references. Output selection indices are
analysis artifacts, **not a solve corpus**. No cloud resources are used.

## Interpretation

Experiments with four cuts and with all seventy cuts show that independent
per-record selection still buys tile savings by creating many left layouts.
Budgeted selection recovers a smaller part of the saving. In the initial
all-cut panel, doubling both identity and byte caps saves 11.3% of tiles;
doubling bytes alone saves 20.7%, but needs 33 left identities instead of eight.
The byte-only result is therefore a different operating point, not a result
inside the stricter 2×-identity budget.

Separate-shard all-cut validation on 572 records in eight complete groups
saves 17.0% with both caps doubled, or 22.4% with bytes doubled and identity
caps relaxed (27 left identities rather than eight). This supports keeping
the byte-budgeted approach as a modest research candidate, not claiming an
order-of-magnitude gain.

No measured GPU speedup follows. In particular, one must charge real batches
and planning, and replace this intentionally expensive exact scoring census
with a scalable selector before discussing a campaign. The experiment log
records separate-shard validation and the qualified decision.

## Support-only shortlisting (Experiment 506)

`cut_support_counts.cpp` enumerates all seventy unordered cuts but builds
only canonical **support counts**, not labelled prefix/weight-class layouts.
An in-memory row-multiset lookup avoids repeated canonicalization. The small
`R8SUPPORT1` research cache stores canonical key / quotient support count
pairs; it is lazily populated, not a complete universal cache or a production
artifact. Use only caches generated by this probe with matching sources.
The format has structural validation but no content checksum/provenance.
Existing output cache files are refused. Main count-pass time and total
process time (including cache loading) are reported separately.

`cut_shortlist.py` ranks cuts using only selected/complement support products,
baseline layout membership, and candidate reuse frequencies. A reuse score
divides positive product saving by amortized new support entries, normalized
by each side's baseline entries. Frequencies count distinct records, not
duplicate candidate slots. Two or four **nonbaseline cuts** retain both
directions; baseline slot zero is always retained, giving five or nine scored
alternatives rather than 140. Product-only and mixed rankings are controls.

The exact exporter accepts the resulting shortlist TSV and skips discarded
candidates **before** layout construction and tile counting. The existing
byte-budgeted selector then uses exact output sizes and tile counts. No exact
tile, bucket, class or byte data enters shortlisting. The optional `--exact`
mode compares against a previously completed all-cut export; it does not
change the shortlist. Neither method changes outer coefficients or is a
production corpus writer.

```sh
make cut-shortlist-test
build/cut_support_counts build/reuse-cut-panel/records.tsv \
  build/reuse-cut-panel/count-cache.tsv > build/reuse-cut-panel/counts.jsonl
python3 research/probes/cut_shortlist.py build/reuse-cut-panel/counts.jsonl \
  build/reuse-cut-panel/shortlist --mode reuse --cuts 4
build/reuse_budget_cut_census build/reuse-cut-panel/records.tsv all \
  build/reuse-cut-panel/shortlist/shortlist.tsv \
  > build/reuse-cut-panel/shortlisted.jsonl
python3 research/probes/reuse_budget_cut.py solve \
  build/reuse-cut-panel/shortlisted.jsonl build/reuse-cut-panel/shortlisted-result.json \
  --memory-only
```

Pass an existing count cache as the optional third argument of
`cut_support_counts` to measure warm reuse, with a different output cache
path. The sampler's repeatable `--band LOW HIGH` option overrides its default
fanout bands, allowing complete groups above 1023 members without truncation.
All panels remain capped at 16,384 records. The new tests check warm/cold
parity, support counts against labelled layouts, candidate ordering,
shortlist determinism, exact filtered-export parity, malformed filters and
caches, and complete large-group sampling.

The frozen reuse-four rule retains 96% / 90% of the all-menu heuristic's
saved tiles on the 687/572-record panels. On a separate 8,251-record panel
of four complete groups above 1023 members, it saves 12.17% versus 3.55%
for the historical fixed menu, at doubled output-byte budgets. No all-menu
control is available for that larger panel. Exact metadata construction
still dominates preprocessing (95% in the instrumented small-panel repeat),
so this passes a shortlist-quality gate, **not** a production-cost or GPU
performance gate. The next task is improving the cached histogram producer.

## Cached projected histogram producer (Experiment 507)

`cut_histogram_census` reuses the same candidate exporter with an alternate
research-only metadata model. The production solvers and default reference
exporter are unchanged. The `cached` control retains compact canonical
supports but uses the old full-mask transform and ordered-map histogram.
The `projected` version additionally:

- Packs a canonical support into 56 mask bits plus a weight/orbit-class ordinal.
- Composes the exact production row map with the 14-bit prefix projection.
  Seven 256-entry byte lookup tables then compute only those bits; suffix
  coordinates never need transforming for this cost model.
- Counts into reusable flat storage, sorting and clearing only touched bins.
  Sorted class ordinals preserve the reference's `(weight, orbit)` order.
  Orbit size is invariant under row permutation, and representatives are
  **not** re-minimized after permutation.

The source cache never evicts/rebuilds entries silently. It fails at a
300-million-entry research cap (about 2.4 GB of packed payload; not a cap on
total process memory). The driver produces separate cold and warm JSONL
exports. Both passes reconstruct every labelled histogram and count every
requested tile; only canonical source supports survive between passes.

```sh
make cut-histogram-test
build/cut_histogram_census build/reuse-cut-panel/records.tsv \
  build/reuse-cut-panel/shortlist/shortlist.tsv \
  build/reuse-cut-panel/projected projected
```

Output files are `projected.cold.jsonl` and `projected.warm.jsonl`; existing
files are refused. Pass metrics on stdout separate DP/packing, histograms,
and canonicalization. Each export includes total/layout/tile timing and is
accepted by the same selector. Change the final argument to `cached` for the
cache-only control. `--self-test` checks both backends against reference
prefix/class counts on 130 deterministic halves, including empty/full halves.

On the 8,251-record panel, old scoring takes 222.64 seconds, projected cold
32.45 seconds and projected warm 19.62 seconds. All exports match exactly.
Peak process RSS is 2.66 GiB. Tile enumeration now occupies 85% of the warm
pass: this experiment addresses metadata production, not that next bottleneck.
No GPU/runtime speedup or affordable whole-corpus preprocessing is established.

## Exact tile-cost indexing and subset sums (Experiment 508)

The driver additionally accepts `indexed` and `zeta`; both use projected
histograms and preserve the reference cost formula exactly. These are
**CPU estimator backends, not faster production GPU joins**.

`indexed` builds a two-level occupancy index over the 14-bit prefix domain.
A prefix's low six bits select a bit within a word; its high eight bits
select that word's index. Small precomputed disjointness tables intersect only occupied words
and bits compatible in the ordinary or plane-swapped orientation. A rank
lookup finds the physical bucket. Union the two orientation masks before
enumerating; count a doubly compatible class twice only when the right
orbit has size two. Cache class counts rounded to eight and sixteen once.

For a right class let `b8=ceil(count/8)` and `b16=ceil(b8/2)`. For each left
prefix `p`, form

```text
F_b8[p] = sum over left classes at p:
          min(left16 * b8, b16 * left8)
Z_b8[S] = sum over p subset of S: F_b8[p]
```

A 14-bit subset-sum transform computes `Z`. A right class at prefix `q`
then contributes `Z[full XOR q]`, plus `Z[full XOR swap(q)]` for a size-two
right orbit. This aggregates exactly the *tile cost*, including padding and
the cheaper 16×8 orientation. It does not aggregate actual suffix predicates
or evaluate the colouring count.

`zeta` lazily builds a table for a left layout and rounded right size only
after four uses of that left, eight class queries of that size, and at least
32,768 cumulative `left_class_count × size_query_count`. These are empirical
amortization gates, not a proof that every individual table pays for itself.
The payload cap is 2,048 tables / 256 MiB. Beyond the cap, or below the gates,
use exact indexed enumeration. Remove table-covered classes/buckets from
the fallback scan. Ordinary right-class weights do not enter this estimator;
equal rounded class sizes may therefore share a table despite differing
weights. Source masks and actual solver weights are never merged.

Tables and reuse counters are cleared before **each** cold/warm pass; table
construction, index construction and lookups are all included in timings.
Only canonical source supports persist between passes. The driver reports
table counts/build time, class lookups and candidate word/pair visits.
`--tile-self-test` checks indexed/table/fallback paths, empty distributions,
fixed/swapped orientations, prefix-word boundaries, tile-rounding boundaries,
maximum supported counts and exhausted table budgets against the original
Cartesian scorer.

The large-panel same-build repeat takes 23.31 seconds warm with Cartesian
scoring versus 15.51 seconds with indexed/zeta scoring; all outputs match.
Against Experiment 507's older 19.62-second run, the improvement is about
21%, not 33.5%. Peak process memory rises to 3.91 GiB. Indexing supplies most
of the gain, with the subset tables adding a smaller improvement. See the
experiment log for controls, cold costs and the remaining scalability limits.

## Left-grouped cost queries (Experiment 509)

The research driver also accepts `grouped` and `planned`. Both build the
same labelled metadata in the original record order, defer candidate costs,
and group those queries by raw left-half identity. Selected and complement
distributions are processed separately. The query tables are released after
each group, so later layouts no longer compete with earlier ones for the
process-wide table budget. Candidate results are written back by their
original record/choice indices; no records, coefficients or queries are
deduplicated or reweighted. Output records retain their original order.

`grouped` is a scheduling-only control with the same lazy table eligibility
as `zeta`. `planned` additionally counts the complete group's right-class
queries before its first score. Eligible tables can then benefit the first
query too. Planning does not use exact tile results and retains the same
four-use/eight-query/32,768-work gates. Each pass rebuilds the group plan and
all query tables; canonical support arrays alone remain warm. Scheduling,
planning, construction, querying and table release are charged in the pass.

The driver reports cumulative table builds, peak simultaneously live tables
and number of selected/complement groups. The cap remains 2,048 live tables,
now applied within each group. A large-panel pilot builds 14,824 tables over
the pass but needs only 60 simultaneously (7.5 MiB payload). This is not a
7.5 MiB total-memory claim: canonical sources and all labelled metadata still
remain resident. The experiment log records matched timing/memory results.

```sh
build/cut_histogram_census build/reuse-cut-panel/records.tsv \
  build/reuse-cut-panel/shortlist/shortlist.tsv \
  build/reuse-cut-panel/planned planned
```

The default research reference and all production GPU solvers are unchanged.
As before, an exporter input has at most 16,384 records; grouped score plans
have not been integrated into a full campaign corpus generator.

## Demand-only eligibility (Experiment 510)

`demand` uses the same complete-group plan as `planned`, but removes the
four-join minimum. A table still requires at least eight right-class queries
of its rounded size and `queries × left_class_count >= 32768`. Thus even one
logical join may qualify if it contains sufficient class work. No threshold
or memory-budget change is made, and streaming/unplanned sources retain the
old four-join gate. Query tables are still released after each group.

This changes only a heuristic choice between two exact **CPU cost-estimator**
paths. It does not change candidate costs, the selected cuts or any colouring
count. The driver reports `short_group_tables`, the number of tables built
for sources used fewer than four times. Their construction is included in
the timed pass. `--demand-self-test` checks just-below/at-work-threshold cases,
the eight-query requirement, complete-plan requirement, unchanged indexed
fallback and exhausted table budgets against Cartesian cost enumeration.

Matched warm scoring falls from 10.47 to 7.76 seconds on the 8,251-record
panel (25.9%), with unchanged exact outputs and peak live-table storage.
More table construction is charged: 1.05 seconds instead of 0.33 seconds,
offset by substantially less indexed enumeration. These are preprocessing
measurements only; neither the selected tile saving nor GPU code changes.

## Independent owners and bounded output batches (Experiment 511)

`cut_batch_gate.py` runs the accepted reuse-four shortlist, demand scorer and
2× byte-only selector on separate owner panels, never pooling budgets between
owners. Each panel samples one complete group in each of the 64–255,
256–1023, 1024–4095 and 4096–8191 fanout bands. Seeds and populations are
recorded before costs are seen; a panel has at most 13,564 records. Process
one panel at a time. Support-count caches are passed between panels; exact
canonical support arrays are built cold and reused for one warm pass within
each panel, not retained across all owners.

The batch simulator sorts chosen records by right layout, retaining all left
outputs for that owner. It never splits a right group. Compare 64, 256 and
1024 MiB **right-output** caps and a 4,096-record batch cap under the same
policy for baseline and candidate. A right group that cannot fit is rejected,
not silently oversized. Each right layout is built once per owner; reuse in
different owners is charged repeatedly. This matches the production solver's
right grouping, but not its full structural memory planner: scratch, canonical
cache, allocation high-water marks, join/result buffers and CUDA reserves are
excluded. These are not total-VRAM limits or measured GPU batches.

```sh
make cut-batch-gate-test cut-histogram-test cut-shortlist-test
python3 -u research/probes/cut_batch_gate.py build/reuse-owner-gate \
  --cache build/review-506/large/cache.tsv \
  ../rectangle-free-data-v2/8x8-transpose/solve/s0000.orbits \
  ../rectangle-free-data-v2/8x8-transpose/solve/s0064.orbits
```

Use an existing trusted `R8SUPPORT1` cache from the support-count probe, or a
probe-generated cache with only its header for a fully cold count run. The
driver refuses an existing output directory and saves selections, counts,
shortlists, cold/warm exports, choices, per-batch statistics and process timing
logs. `check_reference(panel)` separately validates the largest baseline-tile
record of each group against the original Cartesian exporter; run this after
timing. `population_census(paths, bands)` records full input SHA-256 digests
and fanout coverage. Neither helper changes a corpus.

Across eight owner files, 73,228 records / 32 fresh groups retain **15.51% fewer
modeled tiles**. Right build-output bytes fall **8.63%**, while left output
bytes nearly double and left identities rise from 32 to 119. The 256 MiB
output model needs 65 batches rather than 72. Warm exact scoring takes 54.64
seconds; the sampled preparation-stage subtotal is about 92.3 seconds,
excluding cache initialization, validation and some process plumbing. Peak
scorer RSS is 4.23 GiB. All cold/warm metadata and independent reference costs
agree; no GPU wall-time or colouring count is newly measured.

This supports a bounded GPU A/B, not production integration. The sampled
fanout bands contain 67.84% of records in these files, but sampling is uniform
over groups within bands, not over records or tile work. Groups above 8,191
records account for another 28.84% and remain outside this gate. Full-corpus
preparation scaling and actual GPU memory/runtime remain unresolved.

## Matched production GPU A/B (Experiment 512)

`cut_gpu_gate.py prepare` writes small baseline/adaptive probe inputs from the
saved choices, plus one union canonical seed. The inputs preserve each
record's coefficient and complement factor, but are not certified canonical
orbit corpora. `run` uses the unchanged production solver, separate CPU
verification, two warm-ups and four alternating baseline/adaptive rounds.
The same cache is reused across both alternatives. Output checks bind the
binary/seed/input hashes and v3 result checksums and require identical exact
partial contributions across every variant/repeat.

```sh
python3 research/probes/cut_gpu_gate.py prepare build/review-511 \
  build/cut-gpu-payload build/twocolour_8x8_solve_gpu
# On a compatible GPU host, with the payload copied into its own directory:
python3 build/cut-gpu-payload/gate.py run build/cut-gpu-payload
```

Compile the production solver for the target GPU; Blackwell's native NVFP4
path requires `sm_120a`. The completed RTX PRO 6000 gate reduces the sum of
eight panel median recurring times from **2.777 to 2.568 seconds (7.51%)**.
External checkpoint intervals corroborate 7.47%, and all 32 scalar join
checks plus the exact partial-result comparisons pass. Both actual layout
construction and GPU joins are charged; cold cache setup is separate.

However, s0256 becomes **14.48% slower**, despite fewer modeled tiles. GPU
join time, rather than just extra construction, regresses on that panel.
The current tile metric is not a reliable per-panel runtime guarantee.
Moreover the preceding 92.29-second CPU preparation subtotal dwarfs the
0.208-second recurring saving for one traversal of this sample. Keep the
method research-only pending a cheaper, runtime-aware selector; neither
one-run end-to-end benefit nor Ada/L40S gains have been established.
