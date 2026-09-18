# Canonical deletion families and workload coverage

This is a CPU research census, not a production generator or GPU timing.
It follows the [shared response probe](shared_column_response_probe.md).
Measured results and qualifications are in Experiments 503–504 of
[the experiment log](../../docs/experiments.md).

## Two different grouping questions

Deleting the **stored last column** and canonicalizing the remaining 8×7 core
under row and column permutations finds only the families visible at that
boundary in the loaded files. Even scanning several complete shards does not
give the possible reuse after choosing a different deletion or regrouping the
whole campaign.

Instead define an invariant parent rule:

1. Normalize an 8×8 mask to at most 32 selected cells by complementation.
2. Consider the mask and its transpose. At exactly 32 cells also consider
   their complements, matching the production `solve_representative()` rule.
3. For each image, delete each of its eight columns.
4. Canonicalize each remaining 8×7 incidence graph under `S8 × S7`, keeping
   rows and columns in separate nauty colour classes.
5. Choose the numerically smallest canonical core key.

No new outer symmetry is claimed. Nauty's IDs differ from the production
lexicographic IDs, but classify the same isomorphisms. The removed column is
transformed with the canonicalizer's actual row permutation.

## Exact global family size without scanning the whole corpus

For a canonical core `K`, append all 256 active-row masks. Reject children
with more than 32 selected cells. Keep a child only if its invariant parent
is `K`, and deduplicate its full row/column/transpose/complement orbit ID.
Choose one extension query per surviving orbit.

This gives the **complete assigned family**, not merely the children seen in
the sampled shards. The justification is constructive:

- Any orbit assigned to `K` has a representative obtained by appending some
  column to the canonical matrix `K`, so enumeration cannot miss it.
- The parent rule is invariant on the full outer orbit, and its minimum key
  is unique; therefore no child belongs to two different parents.
- Equal-key deletion ties and stabilizers are resolved by full child-orbit
  deduplication, not by a presumed generic multiplicity.

For example, the empty core has nine assigned child orbits: a column with
zero through eight selected cells. A two-record sample can therefore contain
two members of a nine-member family; sample occupancy is not its global size.

This counts required response queries, not labelled-grid coefficients. A
future solver must carry the original exact orbit coefficients into the new
partition, including balanced-complement and transpose stabilizers. If an
individual selected/complement result is exposed, its orientation must be
tracked; their product is invariant. No such campaign rewrite is implemented.

## Measuring workload rather than record count

The census imports `src/gpu/twocolour_gpu_common.cuh`. For each sampled original
record it uses production half canonicalization, distribution construction,
token-plane representatives and the canonical-to-labelled row map. It does
**not** re-minimize plane representatives in labelled coordinates or move the
baseline into a new family gauge.

It models padded 16×8 weight-class tiles for both ordinary and distinct swapped
orientations, using the production eight-row prefix coordinates. This remains
a tile-count proxy: it is not a GPU trace, wall time, load-balancing model or
builder-cost estimate. Outer orbit weights are deliberately not used as work
weights: each retained record invokes two joins regardless of its coefficient.

Sample one deterministic pseudorandom record per equal-size stratum, retaining
the stratum's population as its sampling weight. Global assigned family sizes
are exact for those sampled records; their share of total tile work is an
estimate. Four chosen shards do not constitute a random campaign-wide sample.
When `parents` uses fewer rows than its input TSV, it takes a further stratified
subsample and adjusts the weights. The final Experiment 503 census uses all
4,096 input rows, with no second-stage thinning.

## Reproduce

Requires the system nauty development library; Experiment 503 used nauty 2.8.8
with 64-bit setwords and GCC 13.3.

```sh
make shared-column-family-test
mkdir -p build/shared-family

# Use a new output prefix; this bounded example examines one complete shard.
timeout 300 build/shared_column_family_census \
  ../rectangle-free-data-v2/8x8-transpose/solve/s0000.orbits \
  10000000 1024 503 > build/shared-family/census.jsonl

# Require the census to finish before consuming its sample.
jq -e 'select(.type=="complete")' build/shared-family/census.jsonl
jq -r 'select(.type=="sample") | [.file,.index,.key,.stratum_records,.tiles] | @tsv' \
  build/shared-family/census.jsonl > build/shared-family/sample.tsv
timeout 180 build/shared_column_family_census parents \
  build/shared-family/sample.tsv 4096 505 > build/shared-family/parents.jsonl
```

The measured four-shard run supplies a comma-separated list of
`s0000,s0256,s0512,s0768` paths. Limits are ten million records per file,
four files, and 2,048 samples per file. External timeout/address-space caps
should be used for full censuses. Require the final `complete` or
`parent_complete` marker; intermediate JSON lines alone do not certify a
completed census. Keep the input list, commands and source hashes with logs.

Tests cover row/column maps, deletion invariance (including moving the removed
column), transposition, balanced complementation, exact family membership,
the nine-child empty-core fixture, production half-distribution expansion,
malformed TSV/records and duplicate input paths.

## Complete-family timing gate (Experiment 504)

The `family` mode enumerates a parent's complete assigned child orbits. It
computes **all requested counts on both sides**, using independent 4+4 joins,
cached shared 4+3 responses, streamed shared responses, and a repeat of the
independent control. Every completed shared count is compared exactly with
the corresponding independent count. Capped methods discard their answers;
the final marker includes the number of capped method-sides. Assignment time
is reported separately and charged once to each shared-family total, not once
per child. Distribution construction is included; tile-accounting time is
excluded from computation times and reported separately.

The companion runner chooses the 50th and 90th sampled-record tile-cost
quantiles in each fanout band 1–7, 8–19, 20–79, and 80–256. This is an
eight-family **purposive panel**, not an unbiased campaign performance sample.
The selection is made before timing and is saved with input/binary hashes.
Each method-side has an operation/time cap; the process also has an external
deadline and an 8 GiB address-space limit. No cloud resources are used.

```sh
make shared-column-family-test
build/shared_column_family_census check-production-keys build/shared-family/sample.tsv
python3 research/probes/shared_column_family_benchmark.py \
  build/shared-family/parents.jsonl build/shared-family/timing-panel

# Individual controls / supplemental production-layout model:
build/shared_column_family_census family 21695310254508064 100000 100000000000 30
build/shared_column_family_census model-parent 21695310254508064
```

**Do not confuse three baselines:**

- CPU timings: independent and shared calculations both use the new parent
  row/column gauge and label-minimized token-plane representatives.
- `production_builder_parent_gauge_tiles`: production canonical half
  representatives, but still the parent row/column gauge.
- `model-parent`'s `production_tiles`: actual production corpus representatives
  **and** production canonical half representatives. A small C bridge imports
  the existing generator's canonicalization, balanced-complement and transpose
  rules. It never runs the generator. All 4,096 source census keys reproduce
  exactly, as do the eight original source tile costs.

For 396 complete child orbits / 792 selected-complement counts, independent
CPU joins take 36.44 s, cached responses 36.54 s, and streaming 30.99 s.
Assignment adds 0.0257 s to either response method. All answers match, no
method is capped, and the repeated control takes 36.50 s. Streaming therefore
gives only **1.175× aggregate CPU speedup on this panel**. It wins in four of
eight families. Bounded union caching removes just 0.0121% of response
evaluations here and is slower than streaming.

The best apparent CPU win has 6.55× more parent-gauge modeled tiles than its
production representatives; the new grouping has partly made its own control
harder. Another 110-child family gives a 2.58× CPU improvement with only 3.3%
modeled gauge inflation, so the family route is not ruled out. None of these
ratios measures GPU performance. The next useful gate is a selective GPU
response-evaluation probe against original warmed production joins, not
production integration or a blanket GPU speedup extrapolation.

## Reuse outcome and remaining gate

The fixed-boundary row/column grouping covers only about 0.17% of modeled tile
work in families of at least twenty observed extensions. The invariant
deletion partition puts **87.63%** of sampled modeled work in complete assigned
families of at least twenty children (**95.23%** for at least eight).

That clears a reuse gate, not a performance gate. Experiment 504 now times
complete assigned families, including construction and response evaluation,
but finds a mixed CPU result. Comparison against a warmed production GPU
control is still outstanding. Experiment 502's 2.07× CPU result used different
families and cannot be applied as a multiplier here. No 9×9 time, GPU gain or
final grid result is implied.
