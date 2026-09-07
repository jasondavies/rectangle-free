# Checkpointed shared-core campaign runner

The measured Experiment-484 shared-core kernel and the independent tail now
use one checkpoint/reduction workflow. Experiment 486 passed a bounded CUDA
stop/resume pilot on one RTX PRO 6000: ten shared children plus one independent
query match all 33 archived production residues. CUDA memcheck and synccheck
pass. Experiment 487 then measured the full workload's strata and tail.
Experiment 488 enables bounded durable transaction batching and repeats the
eleven-count/33-residue GPU pilot. Experiment 489 extends sharing to the
order-52/54 tail and passes six further complete-count checks (18 residues).
The current forecast is **about 119 RTX PRO 6000 GPU-hours**. Experiment 490
audits the accepted plan and produces 64 whole-group work items, balanced over
eight workers at **14.83–14.88 hours each** before startup/audit, transfer,
reduction and interruption allowances. This combines sampled computation and
measured storage replay; it is not a full campaign or eight-GPU scaling result.
Experiment 491 adds indexed plan access, one audit/worker per queue process,
and ordered journal reduction. These reduce CPU overhead, not GPU arithmetic;
the forecast remains unchanged. Use the freshly bound Experiment-491 manifest.

The maintained components are:

- `src/hafnian/hafnian_common_core.cuh`: the same arithmetic body used by the
  research A/B harness, with no dependency on its benchmark/CPU probe files.
- `src/hafnian/hafnian_common_core_runner.cuh`: persistent device allocations,
  exact per-child reduction and bounded sign chunks.
- `src/hafnian/hafnian_common_worker.cu`: persistent computation service;
  locks in the measured pool-11, order-16, inverse/live-moment/sync-cleanup,
  warp-polynomial/sparse-moment configuration, with 128 CUDA threads.
- `tools/common_core_campaign.py`: streaming plan reader, durable journals,
  resume validation, original-query reduction and certified CRT bounds.

One service owns one GPU. It prepares topology once per group/active-child
set, reuses it across checkpoint ranges and primes with the same active set,
and retains high-water allocations across groups. Only child residue totals
cross back to the host. Singleton IDs remain owned by the original audited
plan and use the existing independent Gray/resolvent engine where supported.
The exact Hessenberg backend covers the remaining orders, including 66.
Its graph matcher uses 128-bit masks above 64 vertices; matrix and inverse
buffers include the extra entries. The reducer derives the sign domain
independently: common-core order for shared groups, full augmented order
for singletons. No explicit tail import or separate reduction is required.

## Build and audit

```sh
make build/hafnian_common_worker NVCC=/usr/local/cuda-12.8/bin/nvcc \
  NVCCFLAGS='-O3 -arch=sm_120 -std=c++17'
# Use sm_89 for Ada instead. Compile verification alone is not a GPU pilot.

python3 tools/common_core_campaign.py \
  --catalog build/common-core-6x27.catalog \
  --plan build/common-core-489/tail-candidate.plan audit
```

The accepted extended plan has 45,007,139 queries, 7,282,729 groups and 13,100
independent singletons. Its content checksum is
`18c8e7e1a1df645b203875f76d38a19d1c06f8287b82b23b56005d9c1e74cb57`.
The audit checks content checksums, catalog binding, structural dimensions,
exact-once query ownership and coefficient coverage. It scans the plan and
uses a byte per query for coverage; it does not load all groups into RAM.
The catalog is read-only memory-mapped. The worker additionally verifies
each actual canonical member embedding before evaluating that group.
These checks do not independently rederive the original defect census.

## Offline cost-balanced manifest

The `build`, `verify` and `commands` subcommands of
`tools/common_core_manifest.py` do not launch solves or rent hardware; its
explicit `run` subcommand executes a local queue. It audits the plan using the production
controller, scans every group's measured stratum, and requires exact agreement
with the combined baseline/replacement census. Missing or mismatched strata
are errors, never zero-cost work. Small group costs are retained in a compact
double array (about 56 MiB for this plan), without materializing all query data.

```sh
python3 tools/common_core_manifest.py build \
  --catalog build/common-core-6x27.catalog \
  --plan build/common-core-489/tail-candidate.plan \
  --baseline-sample build/common-core-487/sample.json \
  --baseline-projection build/common-core-487/projection-random.json \
  --replacement-sample build/common-core-489/candidate-sample.json \
  --replacement-projection build/common-core-489/candidate-projection.json \
  --storage-replay build/common-core-488/build/common-core-488/replay.log \
  --worker build/hafnian_common_worker --workers 8 --target-shards 64 \
  --output build/common-core-491/ready-manifest.json
```

Use a fresh output path: existing manifests are not overwritten. The model
replaces the target compute strata, retains old overhead per replaced child,
and applies the paired transaction-replay ratio to unchanged shared-journal
costs. It yields 118.862 projected GPU-hours. Whole groups are partitioned into
contiguous, near-target-cost work items; the requested shard count is a target,
not a guarantee. A group larger than the target is not split. Longest-first
greedy assignment gives eight queues of eight items here, with the heaviest
worker only 0.16% above ideal average load. The 4.20-hour order-66 item starts
early. Actual runtime variance and interruptions still require supervision.

The checksummed manifest binds the catalog/plan digests, tested worker binary,
controller/configuration, timing sources, task intervals and worker queues.
Journal names include the manifest digest, preventing reuse of old task IDs.
Generate a queue script, without executing it:

```sh
python3 tools/common_core_manifest.py commands \
  --manifest build/common-core-491/ready-manifest.json --worker-id 0 \
  --worker build/hafnian_common_worker --journal-dir build/common-core-491/journals \
  > build/common-core-491/worker-0.sh
```

The script starts one persistent queue process. It verifies artifact and binary
bindings, audits once, then calls the existing controller sequentially for its
assigned intervals with normal durable checkpoint/resume behavior. Run from the repository root; preserve the relative
artifact paths when deploying. **Explicitly select one GPU per queue**, e.g.
`CUDA_VISIBLE_DEVICES=0 bash build/common-core-491/worker-0.sh`. Queue IDs are
logical scheduling slots, not physical device IDs. Re-running a queue resumes
its journals; never run the same task concurrently or reuse another manifest's
journals. Cross-queue/task interval union is independently checked for gaps and
overlaps. One persistent worker keeps its GPU allocations across tasks; each
task retains its own journal and exclusive claim. The audited plan's offsets
are retained in memory (55.6 MiB), allowing direct seeks to assigned intervals.
No persisted sidecar can assert that an audit passed. Every process restart
rehashes and audits the inputs; within a process, file identity/size/mtime/ctime
are checked before reusing the audit. Inputs must remain immutable while used.
The standalone single-interval `run` command still performs its own audit.
Initial startup/audit remains outside the forecast. These scripts do not implement
automatic result pulling, spot recovery or provider cleanup. A short supervised
multi-worker rehearsal remains the next gate before the full campaign.

For a bounded queue rehearsal, invoke the emitted `common_core_manifest.py run`
command with `--max-checkpoints 1`; when a task reaches the limit the **whole
queue stops**, rather than starting the next task. Omit that flag to resume.
Chunk/checkpoint and transaction-batch options are also available on this queue
command. Controller provenance remains strict: do not reuse older-controller
journals or replace their headers just because the arithmetic is unchanged.

## Streaming final reduction

The reducer holds a read-only SQLite snapshot for each input journal, reads
group metadata and sign ranges using two ordered cursors, and merges group
IDs through a heap. It no longer tests all journal intervals for every group
or performs per-group SQL queries. Only the next group from each journal is
buffered; memory scales with journals and their current groups, not the whole
campaign. Overlapping ownership intervals may contain disjoint complementary
sign ranges, but overlapping actual ranges still fail verification.

Ordered and point reads share the same checksum/range decoder. Missing fields,
primes, groups, gaps, duplicate coverage and bad CRT bounds are still rejected
or reported incomplete as appropriate. Normalization inverses use a bounded
cache keyed by `(domain, unmatched, prime index)`; CRT prefix products and
their inverses are precomputed. These change neither residues nor the v2
journal format. An old journal can still be reduced with matching peers, but
cannot be resumed by a changed controller.

## Bounded pilot and resume

Choose a half-open **group-ID** interval, not original query IDs. For example:

```sh
CUDA_VISIBLE_DEVICES=0 python3 tools/common_core_campaign.py \
  --catalog build/common-core-6x28.catalog \
  --plan build/common-core-6x28-order50-481.plan run \
  --worker build/hafnian_common_worker --journal build/core-pilot.sqlite \
  --group-start 0 --group-end 1 --chunk-terms 32768 \
  --checkpoint-terms 65536 --max-checkpoints 1
```

Repeat without `--max-checkpoints` to finish the group. The default checkpoint
range is 1,048,576 signs; device chunks default to 32,768 signs. Their sizes
can change on resume because exact completed ranges, rather than checkpoint
ordinal numbers, determine coverage. A killed process loses only the current
uncommitted transaction batch and any in-flight request. The startup structural
audit repeats on resume.

Checkpoint data is committed in SQLite FULL-synchronous transactions. The
journal binds catalog and plan content digests, solver binary, controller
source, fixed configuration, backend, prime schedule and owned group interval.
Each group and result payload is also checksummed. A second writer to the same
journal is rejected by an advisory file lock. Separate journals can still be
assigned overlapping work accidentally; reduction **rejects** overlapping
contributions rather than double-counting them.

The solver batches up to **32 completed ranges or approximately one second**
per durable transaction (`--commit-ranges`, `--commit-seconds`). It checks age
before worker requests and after completed ranges; there is no background
timer interrupting a GPU request. The maximum time exposed to a crash is thus
the interval plus one in-flight request/preparation, not a hard one-second
deadline. Group metadata and ranges can span multiple groups in one atomic
transaction. Dirty-page spilling is disabled only during this bounded batch
to avoid holding an early exclusive database lock while computing.

Normal completion and `--max-checkpoints` stops flush before reporting success.
Exceptions, Ctrl-C and abrupt termination discard only the current pending
batch; earlier commits remain resumable. A `checkpoint` progress line reports
`durable: false` while buffered; **only** `checkpoints_committed` or a successful
final status acknowledges durable coverage. Commit failure never acknowledges
the batch. The same reducer and row checksums remain in use; no pending results
are visible to other database connections. Set `--commit-ranges 1` for one
transaction per completed range (including new group metadata).

The batching policy may change on resume, like range/chunk sizes. Controller
source provenance remains strict: old-controller journals require their
original controller for resume, rather than silently accepting changed code.

For multiple GPUs, launch one process/journal per GPU with disjoint group
intervals and `CUDA_VISIBLE_DEVICES`. Equal group counts are not necessarily
equal work; campaign scheduling should use the cost model before a full run.
There is no cloud provisioning, result polling or auto-destruction in this
tool.

Per-checkpoint `compute_seconds` covers the device calculation/reduction
(CPU arithmetic for the explicitly marked reference backend). The journal's
`wall` field covers the computation-service request. Neither includes the
startup audit, group preparation, nor the subsequent durable commit; use
process wall time for end-to-end campaign measurements.

Do not copy a live SQLite database as an ordinary file while it is being
modified. Stop the writer, or use Python `sqlite3.Connection.backup()` to
produce a consistent snapshot for pulling. Merely copying the main database
does not guarantee a transaction-consistent backup.
Keep live reader snapshots short: the DELETE-journal writer must wait for
readers at commit. Run full reductions on a consistent backup or pause writers.

## Reduction and independent tail

```sh
python3 tools/common_core_campaign.py \
  --catalog build/common-core-6x28.catalog \
  --plan build/common-core-6x28-order50-481.plan reduce \
  build/core-pilot.sqlite --query-results

python3 tools/common_core_campaign.py \
  --catalog build/common-core-6x27.catalog \
  --plan build/common-core-484/combined.plan tail > build/core-independent.jsonl
```

Reduction maps child columns back to the exact original query IDs, checks
range overlap and full coverage separately in every required field, divides
by the group's correct sign domain and the unmatched-vertex factorial, then
CRT-reconstructs the matching count. It independently recomputes each query's
matching bound and prime count. A reconstructed count exceeding that bound
is an error. Defect coefficients, powers of two and the labelled-column
factorial are applied only after exact matching reconstruction.

`--require-complete` fails if any shared prime/range or independent query is
missing. A complete reduction includes both backends. For `status=partial`,
the reported `partial_labelled_count` is a subtotal, **not** a new grid result.
Do not add partial reports to `results.txt`. The `tail` command is an optional
inventory export, not a prerequisite for solving singletons.

The integrated worker uses protocol/journal version 2 and a new configuration
identity. Version-1 shared-only journals are deliberately not silently mixed
with this verification campaign.

Different binary/controller digests cannot be mixed or resumed silently.
When updating the software, retain the original executable/controller for
resuming old journals or explicitly restart the affected work with new ones.
There is no flag bypassing this provenance check.

## Local regression gates

```sh
make hafnian-common-core-campaign-test
python3 tests/hafnian/common_core_campaign_test.py --real-plan
python3 tests/hafnian/common_core_variants_test.py --preintegration
```

The host service is built with `make build/hafnian_common_worker_host` and
requires `--cpu-reference`. Its provenance cannot be accepted as GPU output.
Shared groups exercise the same cooperative body on one CPU thread. Singletons
use an independent full-characteristic/Newton-trace reference. Neither is a
GPU performance model. The optional real-plan smoke test requires the ignored
6×28 catalog/plan artifacts. Tests cover durable restart, short chunks,
exclusive claims, interrupted transactions, payload tampering, changed
provenance, overlaps, missing fields and out-of-bound CRT reconstructions.

A complete local reference run of 6×28 group zero covers ten children and
524,288 shared signs in each of three primes. It was stopped after its first
65,536-sign checkpoint and resumed. All ten exact matching counts and all
thirty normalized residues match the archived independent production GPU
results. To repeat that comparison once the complete journal exists:

```sh
python3 tests/hafnian/common_core_campaign_test.py --archived-parity \
  build/common-core-485-complete.sqlite \
  ../rectangle-free-data-v2/verda-6x28-hafnian-l40s
```

## Completed bounded GPU pilot (Experiment 486)

```sh
python3 tests/hafnian/common_core_gpu_pilot.py --output build/core-pilot
```

On one RTX PRO 6000, the complete shared group (ten children, three fields)
took 1.35 seconds, and one complete order-48 independent query (three fields)
took 1.89 seconds. These are separate small-pilot wall times including their
intentional stop/resume, not steady-state campaign rates. They exclude the
initial plan audit, VM startup and transfer. All eleven counts match archived
GPU results (33 field residues), with idempotent completed-work reruns.

252 additional range checks cover zero, disjoint-matching and random graphs
at orders 42/48/50/58/62/64/66, four fields, unaligned ranges and domain tails.
They match the independent CPU reference; memcheck and synccheck report zero
errors. The reference itself passes 288 small-graph brute-force checks.
Ada and Blackwell builds compile; this new runtime pilot was Blackwell only.
Logs and journals are retained under `build/common-core-486/`; the temporary
spot VM and OS volume were deleted after pulling the results.

Experiment 487 subsequently measured the integrated workload: **about 166
RTX PRO 6000 GPU-hours**, including 101 shared-compute hours, 39 independent
hours and 26 preparation/checkpoint hours. Two repeated stratified sweeps give
165.74 and 165.94 hours. Approximately 21 hours on eight work-balanced GPUs is
an ideal scaling estimate, before setup, audit/final reduction and interruption
allowances; multi-GPU storage contention is unmeasured. The earlier 102.35-hour
figure excluded independent work and production overhead. Roughly 25 hours
now come from durable journal operations, making bounded transaction batching
the next practical optimization. No full campaign has been launched.

The Experiment-489 order-52/54 extension is accepted for **fresh campaign
plans** after CPU coverage/embedding checks, GPU A/B, and complete-query parity.
It preserves all old shared groups and non-target singletons; 3,222 formerly
independent queries become 447 shared groups, while 51 target queries retain
independent evaluation. Reproducible preparation is:

```sh
build/hafnian_common_core_plan --catalog build/common-core-6x27.catalog \
  --output build/tail-candidate.plan --extend-tail build/common-core-484/combined.plan \
  --threads 16
python3 tests/hafnian/common_core_tail_extension_test.py \
  --catalog build/common-core-6x27.catalog --source build/common-core-484/combined.plan \
  --candidate build/tail-candidate.plan --worker build/hafnian_common_worker_host
```

On an idle GPU, use `common_core_steady_bench.py census --orders 52 54` for
both plans, then its `bench` and `project` commands. These projections cover
**only the selected residual orders**, not the whole campaign. Run the
complete-count gate with `tests/hafnian/common_core_tail_gpu_test.py --sample
CANDIDATE_SAMPLE --worker build/hafnian_common_worker`. New plan hashes/IDs
must never be substituted into existing campaign journals.

The measured selected-order projection is 1.12 compute GPU-hours versus 25.09
for the independent control on the same RTX PRO 6000. All six complete counts
(18 prime residues) agree. Seventeen sampled narrow-pool groups also beat
their own 54 independent children, with a minimum 1.76x projected compute
gain. The new full-campaign estimate is **about 119 GPU-hours**, conservatively
retaining old overhead and all untested-order costs. This is not a full solve;
fresh plan hashes and normal campaign audit/checkpoint checks remain required.

## Reproducible steady-state timing

This original sweep retains immediate Journal commits as the Experiment-487
control; it does not enable the production controller's new batch policy.
Use `research/probes/common_core_commit_bench.py` for the matched storage-only
A/B, supplying this sweep's sample, log and first-pass journal. That replay
advances a virtual clock by measured compute times, implements the real
32-range/one-second policy, and verifies all reopened payloads. Its replica
residues are deliberately non-production, not newly computed sign ranges.

The bounded benchmark uses the production service and SQLite journal methods,
but tags its journals `common-core-benchmark-only-v1`. They are deliberately
not accepted by the production reducer. Uniform reservoir samples within each
exact `(core order, boundary pool, active children per prime)` stratum retain
the full plan's counts; omitted tail strata cannot silently acquire zero cost.

```sh
python3 research/probes/common_core_steady_bench.py census \
  --catalog build/common-core-6x27.catalog \
  --plan build/common-core-484/combined.plan \
  --output build/steady-sample.json
python3 research/probes/common_core_steady_bench.py bench \
  --sample build/steady-sample.json --worker build/hafnian_common_worker \
  --output build/steady-journals --range-seed 487 > build/steady-timing.jsonl
python3 research/probes/common_core_steady_bench.py project \
  --sample build/steady-sample.json --log build/steady-timing.jsonl
```

Run with an unused output directory on one idle GPU. Defaults use three sampled
groups per stratum, two repetitions, 32,768-sign kernel chunks, and timing
ranges of 32,768 and up to 262,144 signs. Seed zero measures ranges starting
at zero; a nonzero seed measures aligned windows throughout the Gray domain.
Check repeated residues and read back each durable checkpoint. The CPU and
worker independently validate preparation metadata. This repeats existing
arithmetic, not an independent proof of every sampled GPU range.

Projection scales the larger range's computation rate by each stratum's full
sign domain. It adds preparation once per group, measured first-call excess
once per field, and request/transaction overhead per production checkpoint
(default 1,048,576 signs). Shared compute time is CUDA-event time; independent
compute time is the existing engine's chunk-loop elapsed time, including its
downloads and host block reduction. The sample minimum/maximum envelope is
**not a statistical confidence interval**. Audit, final cross-query reduction,
provisioning, transfers and interruption losses remain separate allowances.
