# Shared-core residual hafnians: research record

## Current status

The shared-core method has passed full production validation. The complete
6×27 run evaluated 45,007,139 queries in **115.74 timed GPU-hours** on eight
RTX PRO 6000 GPUs, with **15 h 37 min** elapsed including setup and validation.
The later 6×28 verification completed all 36,398 queries and reproduced the exact
grid result in **4.13 timed GPU-hours**, **65 min solving / 72 min total**, on
four RTX PRO 6000 GPUs (Experiment 500).

For current commands, use [the whole-group workflow](common_core_campaign.md)
or [the mixed whole-group/sign-range workflow](mixed_campaign.md). Completed
values are in [results.txt](../../results.txt), and measured campaign summaries
are in the [README](../../README.md#selected-completed-computations).

## Historical gates and projections

The sections below preserve the research sequence. Statements about pending
integration, next experiments, or unlaunched campaigns describe the stage of
the named experiment, not today's deployment status. In particular, the
119/142/166-hour figures are successive sampled forecasts, not additional
completed runs. Research-only flags and artifact formats below are not the
current production interface.

Experiment 491 completes the local CPU follow-ons. Ordered-journal reduction
passes exact A/B replay parity with 15–17% lower elapsed time on bounded local
synthetic workload. The full audited plan gains a 55.6-MiB in-memory offset
index; queue processes reuse the audit and worker across tasks, rebuilding
validation after every restart. Fresh commands/manifests are under
`build/common-core-491/`. The ~119-GPU-hour forecast is unchanged; these costs
were largely outside that projection. No GPU or cloud instance was used.

Experiment 490 completes the local production audit and cost-balanced manifest:
45,007,139 queries covered exactly once, coefficient sum 47,983,269,684,673,
7,282,729 groups, 13,100 independent queries. The 64 work items are assigned
to eight workers with projected loads **14.83–14.88 hours** (118.862 total).
The manifest and generated commands bind the audited inputs and tested binary;
all interval/queue coverage checks pass. No worker was rented or solve launched
in Experiment 490. Its remaining rehearsal and full-campaign gates subsequently
passed; see the current status above and [the campaign instructions](common_core_campaign.md).

Experiment 489's order-52/54 extension now **passes the GPU gate**. It groups
3,222 of 3,273 old independent queries into 447 shared groups, preserving all
other assignments. The targeted compute projects to **1.12 GPU-hours versus
25.09** for independent evaluation on the same RTX PRO 6000. Six complete
counts (18 prime residues) match independent full-matrix evaluation; all 17
matched narrow-pool samples beat their own independent children (1.76–32.57x).
Accept the extension for a fresh campaign plan, retaining the 51 unsupported
target queries as independent. Existing plans/journals are not overwritten.

The revised full solve forecast is **about 119 RTX PRO 6000 GPU-hours**, or
roughly 15 hours on eight work-balanced GPUs before setup, audit/reduction and
interruptions. This replaces only the previously measured order-52/54 compute
component and conservatively retains old overhead. It is a stratified sampled
projection, not a completed campaign or measured whole-campaign speedup.
Logs and four checked journals are under `build/common-core-489/`.

Experiment 488 enables bounded durable transaction batching (32 ranges or
approximately one second, plus an in-flight request). The VM storage replay
reduces journal time **96.4%**, with exact A/B payload parity, crash-recovery
tests and all eleven GPU-pilot counts/33 archived residues passing again.
That experiment's full solve forecast was **about 142 RTX PRO 6000 GPU-hours**,
roughly 18 hours on eight balanced GPUs before setup and interruptions.
This is a computation-plus-storage projection, not measured end-to-end GPU
speedup. The temporary worker/disk have been deleted. No campaign was launched.

Experiment 487 completes the integrated timing gate on one RTX PRO 6000.
The full plan now projects to **about 166 GPU-hours**, including approximately
101 hours of shared computation, 39 hours of independent work and 26 hours of
preparation/checkpoint overhead. Two repeated stratified sweeps, including
off-zero Gray windows, give 165.74 and 165.94 hours. Eight work-balanced GPUs
would need about 21 hours before audit/reduction, setup and interruption
allowances; multi-GPU storage contention has not been measured. This is a
sampled forecast, not a completed campaign. Durable transactions alone account
for about 25 hours: bounded group-commit batching is the next practical target.
The benchmark VM/disk were deleted after journal and log verification.

Experiment 486 connects the independent tail (orders 42 through 66) to the
same version-2 checkpoint/reduction workflow. A bounded RTX PRO 6000 pilot
passes stop/resume, all eleven complete query comparisons (33 archived
residues), 252 independent-reference ranges, memcheck and synccheck. The VM
and disk have been deleted. The remaining gate is representative campaign
timing, not an unimplemented tail or checkpoint path.

Experiment 485 extracts the shared arithmetic into maintained headers and
adds a persistent service, durable sign-range journals and exact per-query
CRT reduction. See [the campaign runner](common_core_campaign.md).
The notes below preserve the earlier research status and measured projections.

Experiment 484's full repeated GPU gate accepts shared fixed inverse chains,
live-only moment columns, and redundant barrier/memo-clear removal:
**112.22 -> 102.35 grouped GPU-hours (8.79% less)** on the same RTX PRO 6000.
These are Experiment-480 grouped kernels, not the complete 6x27 campaign.
The settings remain independently opt-in
(`CORE_OPT_INVERSE_CHAIN`, `CORE_OPT_LIVE_MOMENTS`, `CORE_OPT_SYNC_CLEAR`).
Local UBSan/cooperative gates and CUDA memory/race/synchronization gates pass.
The reduced-output runner is exact and performance-neutral in its pilot;
it is a pre-production component, not a checkpointed campaign solver.
HCCOST02 binds the new configuration; measured pool-eleven curves have been
exported. All-width/order-50 retuning is pending because the benchmark VM
repeatedly lost SSH connectivity. All completed logs were pulled and the VM
and disk deleted. No resources remain from this experiment.

The repaired plan and order-50 extension are now explicitly combined and
fully audited: **45,007,139 queries, 7,285,504 groups, 16,322 singletons**,
all row maps and coefficient coverage verified. Primary shared groups are
unchanged; 5,546 donor groups cover 36,857 previously independent queries.
This is an audited assignment, not a newly measured full-campaign estimate.

Status through Experiment 483: **the new measured kernel configuration
projects the Experiment-480 grouped workload to 112.17 RTX PRO 6000
GPU-hours**, versus 140.38 for the matched control (20.1% less time).
An independently audited extension groups nearly all order-50 leftovers;
that new grouped work projects to another 2.82 hours. These are not yet one
integrated production plan. Uniform thirteen-token padding was rejected.
The earlier cost-aware plan saved 19.45% (174.24 -> 140.35 hours).
Experiment 479 improved the kernel from 305 to 175 hours; Experiment 478's
older control was 313 hours.
That projection excludes independent leftovers and production overhead;
it is not yet a complete campaign estimate. Earlier gates are retained below
as the experimental record.
Nothing here changes the production solver or campaign checkpoint format.

The prime-independent recurrence plan and representation-independent inverse
chains are now shared headers under `src/hafnian/`. Production inverse calls
use the same existing chains; no production arithmetic algorithm changed.
`prepare()` no longer allocates a CPU polynomial workspace, and `set_field()`
reuses the small integer-inverse table. Callers can retain prepared metadata
across chunks; the research one-shot wrapper still prepares per invocation.

Local pre-integration gates:

```sh
make hafnian-common-core-preintegration-test
python3 tests/hafnian/common_core_variants_test.py \
  --preintegration --groups build/common-core-repair480.log
```

Bounded GPU reduction A/B (the reference still downloads all per-sign values
so every reduced total can be checked):

```sh
build/hafnian_common_core_gpu --sweep --groups SAMPLE_LOG --threads 128 \
  --count 131072 --reduced-ab 32768
build/hafnian_common_core_plan --catalog build/common-core-6x27.catalog \
  --output NEW_PLAN --merge build/common-core-6x27-repair480.plan \
  --extension build/common-core-6x27-order50-481.plan --threads 16
build/hafnian_common_core_plan --catalog build/common-core-6x27.catalog \
  --verify NEW_PLAN --all-maps --threads 16
```

Compile the GPU binary with the three Experiment-481 settings and the three
Experiment-484 settings all enabled (ordering is 16, all other switches 1).
`--reduced-ab` reports prepared-group time including device reduction and
download, excluding reference checks and CPU problem construction. It is
not a production campaign command. Remaining work is all-width retuning,
production provenance/checkpoint wiring, and a representative campaign pilot.

## Experiment 481 configuration and integration status (historical)

Experiment 481 adds three opt-in research settings:

- `CORE_OPT_WARP_POLY=1`: warp-local characteristic-polynomial phase and
  cooperative determinant-series sums.
- `CORE_OPT_SPARSE_MOMENTS=1`: precomputed neighbours and one reduction per
  bounded sum, rather than repeated adjacency tests/modular additions.
- `CORE_BOUNDARY_ORDER=16`: deterministic offline boundary-coordinate search
  to reduce the reachable subset recurrence; preserve every query and sign.

Use **128 threads** with this combination. The control is all three settings
zero at 256 threads, with the accepted H/B/S switches still all one. Full
513-case/three-field repeated sweeps give 140.389725/140.368259 control,
117.242664/117.250037 warp+sparse, and 112.165440/112.174780 with ordering.
The search and packing time is excluded from kernel timing. No complete
campaign time is claimed. The switches remain opt-in to preserve the current
HCCOST01 model contract, which rejects new-kernel timings instead of silently
using an incompatible model; production integration needs a versioned model.

The explicit fresh-plan gate `--group-max-order 50` covers 36,857 additional
queries in 5,546 shared groups, leaving 62 order-50 singletons. All 201 new
group/field samples pass their CPU reference checks and project to 2.817296
grouped hours. Six complete residues for three order-50 6x28 queries match
historical results. The full assignment/row-map audit passes, but this fresh
plan must be combined explicitly with the Experiment-480 repaired groups;
do not replace the old plan by its header or add two overlapping outputs.
The extension reduces the independent tail to **552,100,429,824 signs**.
Their runtime, production overhead and larger orders still require timing.

Experiment 483's `CORE_MAX_POOL=13` is a bounded probe only. Twelve complete
padded residues match the historical 6x28 results, and CUDA memory/race/sync
checks pass. Nevertheless, two full sweeps project uniform padding to
129.313893/129.307225 hours, worse than pool eleven. Even a per-field/stratum
oracle projects only 110.59 hours and is not an audited selection policy.
Do not integrate mixed widths from the encouraging unweighted pilot.

Reproduce local gates:

```sh
make hafnian-common-core-candidates-test hafnian-common-core-plan-test \
     hafnian-common-core-projection-test
python3 tests/hafnian/common_core_variants_test.py \
  --candidates --groups build/common-core-repair480.log
build/hafnian_common_core_plan --catalog build/common-core-6x27.catalog \
  --output build/common-core-new-order50.plan --group-max-order 50 --threads 16
build/hafnian_common_core_plan --catalog build/common-core-6x27.catalog \
  --verify build/common-core-new-order50.plan --all-maps --threads 16
```

GPU binaries require explicit H/B/S and candidate defines for reproducible
A/B. `hafnian_common_core_ab.py --threads 128` selects the candidate launch.
Raw timing and verification artifacts are under `build/common-core-gpu-481/`;
Experiments 481--483 record the decisions and scope separately.

## What is shared

The existing Gray/resolvent kernel reuses work among signs for one adjacency
matrix. This experiment groups different residual adjacency matrices.

Start with a reachable defect parent `D`. Each child removes one additional
size-three support. Choose a small boundary pool `B` containing several such
supports. The vertices outside `D union B`, plus the fixed monomer dummies,
form a common core `C`. Every child is exactly `C union (B minus its support)`.
Canonical child IDs are deduplicated: multiple labelled placements of one
query must not be advertised as multiple saved solves. Original catalog
coefficients remain attached to their original queries.

Choose `|B|` odd. The core and each child boundary are then even and can be
paired separately. Pair core vertices first in a fixed order; all children
have the identical signed core matrix, not just isomorphic token sets. Dummy
vertices stay in the common core and retain their original adjacency.

The sample is deterministic bottom-k hashing of **canonical residual queries**
within each `(excess, defect count)` sector. It is not weighted by colouring
coefficients and does not sample labelled defect sequences. For each sampled
root, test up to four reachable parent orbits and a greedy boundary menu of
sizes at most 5, 7, 9 and 11. Parent reachability is checked in the exact orbit
DP, rather than inferred merely from an occupied-token subset relation.
Greedy groups are feasible constructions, not optimal groups.

## Census

The complete 6x27 orbit/CRT census is reproduced before sampling: 45,007,139
queries and 134,616,715,362,304 adaptive sign terms. The run takes 5m52s wall
with 16 local threads and peaks at 29,807,268 KiB RSS (28.43 GiB). Lightweight
tests also ran during this diagnostic; its timing is not an isolated benchmark.
The seed is 475, with 128 queries in each of six sectors.

| Sector `(excess, defects)` | Matrix order | Population | Mean group, six-vertex tail | Mean group, eight-vertex tail |
|---|---:|---:|---:|---:|
| (6,6) | 42 | 34,604,824 | 10.554688 | 15.398438 |
| (6,5) | 44 | 8,512,818 | 10.718750 | 16.460938 |
| (5,5) | 46 | 1,299,727 | 10.828125 | 16.695312 |
| (6,4) | 46 | 368,485 | 10.382812 | 15.617188 |
| (5,4) | 48 | 178,433 | 10.718750 | 16.710938 |
| (6,3) | 48 | 2,593 | 3.468750 | 4.562500 |

All sampled roots in the dominant four sectors have valid groups. In the
small `(6,3)` sector, 91 of 128 roots have no reachable size-three-deletion
parent; these are counted as singletons, not excluded from the mean.
Orders 42/44/46 account for 94.89% of the original adaptive sign-term work.

For order 42, a boundary pool of nine leaves a 36-vertex core and six live
boundary vertices per child. A pool of eleven leaves a 34-vertex core and
eight live boundary vertices. **The pool size and child tail size differ by
three.** Logs contain each chosen parent, boundary pool, canonical child ID
and labelled removed support, so groups are reconstructible.

The diagnostic `rebuild_ratio8` compares numbers of common-core sign
assignments with an assumed eight-term baseline rebuild cadence:

```
ratio = 8 / (group_size * 2^(live_boundary_size/2)).
```

This only illustrates potential reuse of expensive setup. It excludes moments,
boundary arithmetic, rank failures, global group overlap, scheduling and CRT.
It is neither a wall-time estimate nor a proved speedup bound. The current
production eight-term kernel is not yet measured at orders 42/44/46.

## Eliminate boundary signs exactly

There is a stronger alternative to running a separate small determinant for
every child's boundary-sign assignment. For a common-core sign matrix `R`
(signed swaps of fixed core pairs), partition the adjacency into core and
boundary-pool blocks and work with formal power series modulo an odd prime:

\[
 f_R(z)=\det(I-z A_{CC}R)^{-1/2},
 \qquad
 K_R(z)=A_{BB}+z A_{BC}R(I-z A_{CC}R)^{-1}A_{CB}.
\]

If `S` is a child's live boundary subset and `m=|C|/2`, then

\[
 \operatorname{haf}(A_{C\cup S})=
 2^{-(m-1)}\sum_{s_1=1,\ s_2,\ldots,s_m=\pm1}
 \left(\prod_i s_i\right)
 [z^m]\,f_R(z)\operatorname{haf}(K_R(z)[S,S]).
\]

For an empty core use the ordinary boundary hafnian, with no sign sum.
Truncate every polynomial at degree `m`; choose primes large enough for the
coefficient recurrence (the tests use 1,000,003 and 1,000,033).

Derivation: introduce formal centered Gaussian core variables with covariance
`zR` and boundary sources `y`. Integrating the core quadratic exponential
gives `f_R(z) exp(y^T K_R(z)y/2)`. Extracting each boundary variable once is
the hafnian of its `K_R` submatrix. The weighted core-sign sum kills every
even pair multiplicity; at total degree `m`, the only surviving term uses each
of the `m` core pairs once. Fixing the first sign halves the sum because a
global sign flip changes both the degree-m coefficient and the sign product
by `(-1)^m`. This is a formal coefficient identity, not numerical integration.

The core series and the full small `K_R` pool are built once per core sign.
All child answers use principal minors of this one polynomial matrix. Their
small boundary hafnian recurrences can share a memo over boundary subsets,
reset for each core sign. This memo has a 9- or 11-vertex universe, unlike
the rejected 60-token global matching memo from Experiment 405.

`research/probes/hafnian_common_core_identity.py` verifies 1,326 small minors
against independent direct perfect-matching enumeration over two primes,
including complete graphs, singular/zero cores, signed weights and varying
boundary minor sizes. This is an exactness prototype only: its dense Python
matrix powers are not an intended production implementation.

The partial-core formula has **different summands** from the original full
Glynn formula. Do not compare arbitrary sign ranges term-by-term or reuse old
range checkpoints. Complete small counts agree; a future backend needs its
own range identity, catalog/provenance and full-query regression gates.

## CPU arithmetic and ownership gate (Experiment 476)

The complete partial-core formula now has a C++ finite-field implementation:
`research/probes/hafnian_common_core_bench.cpp`. It builds the core determinant
series, all boundary moments and all requested polynomial hafnian minors.
The boundary-subset dependency graph and storage are reused between signs;
its numeric values are recomputed at every sign. The moment implementation
exploits the actual 0/1 adjacency, without assuming the core is invertible.

Local single-thread timings on the Ryzen 7 9700X use four actual 6x27 groups
(one in each of the main order-42/44/46 sectors), each at three boundary caps.
Each test runs 128 core signs twice, reversing the shared/control execution
order on the second repetition. Input roots and full-domain sign-range starts
are deterministic hashes, not manually chosen fast cases. All boundary work
is included; one-time CPU control factorization/workspace setup is reported
separately. These are short arithmetic probes, **not complete campaign timings**.

| Pool vertices | Live boundary | Queries in tested group | Sharing gain versus individual partial-core evaluations | Normalized CPU four-term resolvent work ratio |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 4 | 6 | 5.20–5.25x | 10.52–11.46x |
| 9 | 6 | 10–11 | 7.45–8.30x | 31.71–38.56x |
| 11 | 8 | 14–18 | 6.79–8.75x | 59.77–78.88x |

The last column multiplies the measured full-Glynn CPU cost by the ratio of
sign-domain sizes, `2^(live_boundary/2)`, before comparing it to the measured
partial-core cost. **It is not a GPU speedup.** This CPU control uses the old
four-term resolvent implementation, not the maintained tuned CUDA eight-term
hybrid. The larger pools spend about 48–50% of arithmetic time in boundary
polynomials: removing boundary signs is valuable but the replacement is not
free. These locally large groups also overstate global reuse.

As a sensitivity check, trimming those same groups to six requested children
retains 4.75–4.86x within-method sharing at pool 9 and 3.80–4.11x at pool 11.
The corresponding normalized CPU resolvent ratios fall to 20.79–23.45x and
34.46–38.12x. These tests include unused pool vertices and recompute the
required minor dependency graph; they are not a new optimized assignment.
Repeating the original real-order arithmetic checks with prime 2,147,483,629
gives similar timings and all exact per-sign comparisons pass.

An actual non-overlapping assignment was built for the entire 6x28 `(e=4,d=4)`
sector: 33,077 queries, using all 664 reachable `(3,3)` parents. Two hashed
parent orders give:

| Maximum pool size | Mean queries per assigned group | Queries assigned to multi-query groups |
| ---: | ---: | ---: |
| 7 | 3.527–3.534 | 99.915–99.924% |
| 9 | 4.951–4.975 | 99.918–99.952% |
| 11 | 6.527–6.538 | 99.924–99.937% |

Groups are grown using only still-unassigned child IDs. Every remaining query
is explicitly a singleton, and the coefficient total stays **8,126,516,160**.
This is a feasible greedy assignment, not an optimal cover. Family construction
took about 1.02 s after catalog construction; each assignment took about
0.06–0.07 s. A separate 6x27 check trims overlap in the logged sibling union,
but that sparse union is not evidence of full 45-million-query coverage.

Validation includes 1,350 complete small minors against both direct matching
enumeration and independent full-Glynn/Hessenberg evaluation, over two primes.
For a real 18-query 6x28 group, the complete 524,288-sign partial-core sum
(40 core vertices, 11-vertex pool) is compared to saved production residues.
All 18 saved residues agree over **both** primes 2,147,483,647 and
2,147,483,629 (36 comparisons). The first-prime arithmetic takes 32.54 s on
eight local CPU threads, excluding catalog construction. Result payload hashes,
catalog/query identities and complete independent sign-range coverage are
checked before comparison. The C++ complete-minor tests also pass UBSan.
Do not mistake a per-prime comparison for a new
CRT reconstruction of the entire 6x28 result.

Reproduce the arithmetic and assignment probes:

```sh
make hafnian-common-core-test
build/hafnian_common_core_bench --groups build/common-core-6x27.log \
  --steps 128 --repeats 2
build/hafnian_common_core_bench --coverage6x28
build/hafnian_common_core_bench --complete6x28 \
  --groups build/common-core-6x28-smoke.log --threads 8
python3 research/probes/hafnian_common_core_audit.py \
  --complete build/common-core-complete6x28.log --results PATH_TO_SAVED_RESULTS
```

Use `--prime 2147483629` for the second complete field image and
`--query-limit 6` for sensitivity to smaller, ownership-trimmed groups.
The latter does not constitute a complete 6x27 assignment.

## CUDA prototype (Experiment 477)

`research/gpu/hafnian_common_core_gpu.cu` implements the same partial-core
identity in one cooperative CTA per core sign. Hessenberg elimination uses
parallel commuting row operations followed by their joint inverse-column
update. The characteristic recurrence, determinant series, dressed boundary
moments and polynomial minor memo all remain in shared memory; only child
residues are written out. The memo uses compact reachable-subset slots, with
even-cardinality levels providing a parallel dependency order.

Two compile-time fields, 2,147,483,647 and 2,147,483,629, use exact
pseudo-Mersenne products. No tensor-core arithmetic or numerical approximation
is used. Local Ada and Blackwell compilation reports 39–40 registers and no
spills. The full real group uses 44,192 shared bytes; two 256-thread CTAs fit
per RTX PRO 6000 SM. Smaller groups reach three or four CTAs. These are
resource measurements, not a substitute for instruction-level profiling.

On one Verda spot RTX PRO 6000 Blackwell, compare the same complete 18-query
6x28 group with the freshly compiled maintained solver. Both controls use
its normal Gray path without breakdown/fallback. Timings are:

| Timed solve section, first prime | Maintained solver | Shared-core prototype | Ratio |
| --- | ---: | ---: | ---: |
| All 18 queries | 6.037 s | 0.790 s | 7.64x |
| First six of those queries | 2.013 s | 0.623 s | 3.23x |

The prototype column includes host descriptor packing, allocations, upload,
kernel execution and result download. The control column sums its per-query
reported solve times (including reduction/checkpoint publication); control
rank-factor setup is outside that timer. Both exclude canonical catalog
construction and independent reference validation. **These are measured
complete-group solve sections, not full campaign end-to-end timings.**
The comparison is conservative about prototype setup but not a uniform
campaign pipeline benchmark.

The 18-query kernel alone takes 0.703–0.707 s with 256 threads, versus
0.933 s with 128 and 1.394 s with 64. Before/after control totals are 6.0406 s
and 6.0342 s. The six-query kernel takes about 0.554 s; ownership losses
therefore matter, even though the exact formula still wins in this test.
The complete group sums 524,288 core signs, replacing 18 independent domains
of 8,388,608 full-Glynn signs each.

Actual order-42/44/46 6x27 groups were also tested with six children, both
fields, pools 9/11 and 32,768 core signs. First-prime pool-11 kernels take
20.05/21.55/24.49 ms respectively. Pool-9 kernels take 14.31/15.99/17.32 ms,
but their complete core-sign domain is twice as large: per-sign speed alone
is not the selection metric. These are sampled costs, not complete 6x27
query counts or comparisons against a retuned order-42/44/46 production path.

Validation:

- 3,960 cooperative child/sign comparisons across both fields; all complete
  small sums also match direct perfect-matching enumeration.
- Real-order samples compare 128 evenly spread sign positions per launch
  with the independent CPU formula.
- All 18 complete real-group residues agree with the saved 6x28 campaign
  over both fields. The first field also agrees with the fresh same-worker
  control after validating query identities, checksums and range coverage.
- CUDA memcheck, racecheck and synccheck pass, including singular-core cases and the
  real group. Host OpenMP execution of the same cooperative body is an
  additional concurrency check, not a source of GPU timings.

Commands (set `NVCC` if CUDA is not on `PATH`):

```sh
make hafnian-common-core-cooperative-test
make build/hafnian_common_core_gpu NVCC=/usr/local/cuda/bin/nvcc \
  NVCCFLAGS='-O3 -arch=sm_120 -std=c++17'
build/hafnian_common_core_gpu --self-test
build/hafnian_common_core_gpu --groups build/common-core-6x27.log \
  --order 42 --cap 11 --query-limit 6 --count 32768 --threads 256
build/hafnian_common_core_gpu --groups build/common-core-6x28-smoke.log \
  --complete6x28 --threads 256
```

Local logs, control residues and binary/input hashes are retained under
`build/common-core-gpu-477/`. Use the audit script's `--control-batch` option
to generate an exact control query list from a complete prototype log.
The temporary worker and its OS disk were deleted after downloading the
artifacts; approximately ten minutes at $0.945/GPU-hour cost about $0.16,
including the small disk charge, before any provider billing adjustments.

## Historical decision after the initial CUDA gate

The GPU gate is positive, including a realistic smaller group. Next build a
scalable once-only 6x27 assignment and measure a stratified, ownership-trimmed
workload. Add and validate the remaining CRT fields before production, then
project the complete adaptive-prime catalog using measured group costs.
Production defaults, checkpoint identity and historical results remain
unchanged. There is not yet a justified replacement for the 6x27 campaign
estimate.

## Full assignment and weighted GPU census (Experiment 478)

The planner now covers the complete **45,007,139-query** 6x27 catalog:

- 7,266,103 multi-query groups containing 44,953,960 queries;
- 53,179 singleton fallbacks;
- mean multi-query group size **6.1868**;
- 99.8818% of queries, and **98.2118% of original adaptive sign work**,
  assigned to the shared-core backend.

This is an actual deterministic partition, not independently selected best
families. Only order-42--48 queries are grouped in this first plan; larger
orders are explicit fallbacks. Parent families are constructed in parallel
chunks of 1,024 and consumed in a fixed hashed parent order (seed 478).
Already-owned canonical children are removed before each group is grown.
Parent batches are discarded rather than materializing every overlap.

The exact census is exported once as a 765,121,451-byte, checksummed catalog
of packed query keys, coefficients and per-query prime counts. The
895,777,176-byte plan references that catalog digest and stores each query
ID exactly once with its parent/boundary embedding. Both files have explicit
little-endian encodings, versioned magic, exclusive creation and SHA-256
footers. They are local research artifacts, not production checkpoints.

Measured locally on 16 logical CPUs:

| Stage | Wall time | Peak RSS |
| --- | ---: | ---: |
| Exact census and catalog export | 5m53s | 28.40 GiB |
| Complete assignment from saved catalog | 4m07s | 2.57 GiB |
| Independent audit, including every row-map embedding | 42s | 1.05 GiB |

The auditor checks all IDs once, matching catalog/plan digests, complete
coverage, preserved coefficient total **47,983,269,684,673**, even core
orders and every canonicalized residual embedding. Tests reject truncated
artifacts, checksum-valid duplicate ownership and incomplete catalogs.
The complete 6x28 plan is byte-identical with one and four producer threads.

### Weighted timing, not division by group size

There are 256 grouped timing strata, keyed by residual order, core size,
boundary pool, number of active children and prime index. Later field images
drop children whose certified bounds already have sufficient CRT coverage.
Up to three deterministic bottom-hash group samples are retained per stratum:
**757 real owned-group/field cases**, each timed for 32,768 core signs and
checked at 128 spread positions against the CPU formula.

The CUDA prototype now supports all four existing CRT primes. Four-field
cooperative tests pass (7,920 child/sign comparisons and complete small
brute-force sums), as does GPU memcheck. The third-prime complete 18-query
6x28 group also matches the saved production residues. The grouped 6x27
strata only require the first three fields; all fourth-field queries in this
plan remain independent fallbacks.

One spot RTX PRO 6000 Blackwell gives this weighted **kernel-only** projection:

| Residual order | Grouped GPU-hours |
| ---: | ---: |
| 42 | 156.75 |
| 44 | 82.99 |
| 46 | 58.24 |
| 48 | 15.38 |
| **Total** | **313.36** |

Weighting each stratum's fastest/slowest sampled group gives 307.01--320.56
hours. This is observed within-stratum sample variation, **not a confidence
interval** or an allowance for long-run thermal/power/scheduling effects.
Every grouped stratum is measured; the projection tool rejects missing bins
and checks that the histogram preserves the original adaptive workload.

The omitted independent tail still contains **2,407,173,980,160** original
sign terms, including orders 50--66 and ungrouped low-order residuals. Zero
grouped time for those orders does not mean their solve cost is zero.
Neither tail time, production reductions, artifact I/O, job dispatch nor
restart overhead is included in 313 hours. Thus the old 1,700--1,900-hour
estimate must not simply be divided by the favorable 7.64x pilot ratio.

### Reproduce and continue

```sh
make six_by_twenty_eight_defect_census build/hafnian_common_core_plan
build/six_by_twenty_eight_defect_census --slack 3 --threads 16 \
  --export-catalog build/common-core-6x27.catalog
build/hafnian_common_core_plan --catalog build/common-core-6x27.catalog \
  --output build/common-core-6x27.plan --threads 16 >build/common-core-plan6x27.log
build/hafnian_common_core_plan --catalog build/common-core-6x27.catalog \
  --verify build/common-core-6x27.plan --all-maps --threads 16
make hafnian-common-core-plan-test
# Repeat sweep with each of the first three primes; these use real plan samples.
build/hafnian_common_core_gpu --sweep --groups build/common-core-plan6x27.log \
  --prime 2147483647 --count 32768 --threads 256
python3 research/probes/hafnian_common_core_projection.py \
  --plan build/common-core-plan6x27.log build/common-core-gpu-478/sweep-p*.log
```

Fresh-output commands intentionally refuse to overwrite existing artifacts.
Logs/catalog/plan remain under `build/`; GPU timings and checks are under
`build/common-core-gpu-478/`. The temporary worker and disk were removed
after downloading the artifacts.

Next: implement the persistent grouped runner and its distinct result/reducer
format, and implement/benchmark the independent tail (including currently
unsupported matrix orders). Only then quote a complete campaign estimate or
launch a 6x27 production run. Historical result values remain unchanged.

## Accepted kernel optimisation (Experiment 479)

Three independently gated changes are now the research CUDA default:

1. **Hessenberg:** skip identity row/column swaps, visit only the active
   matrix region, pad the shared matrix stride and reuse characteristic
   recurrence factors across coefficients.
2. **Boundary:** precompute subset transitions and only the consumed upper
   triangle of the dressed symmetric matrix; copy size-two hafnians directly;
   accumulate four 31-bit products before modular reduction. Four products
   fit in `uint64_t`; both pseudo-Mersenne folds remain valid on that range.
3. **Scratch:** share an arena among the core work, moment powers and boundary
   memo, whose lifetimes are disjoint. Preserve the determinant series and
   boundary series separately; barrier before reusing core storage and clear
   the boundary memo before reading its constant/zero coefficients.

The 39-case diagnostic pilot gives 1.024x/1.172x/1.140x for the individual
changes, 1.200x for the first two and 1.466x combined. A separate comparison
also favours retaining the Hessenberg changes alongside boundary+scratch.
These ratios are not multiplied and are not the campaign weighting.

The complete 757-case, three-field sweep is repeated twice, reversing kernel
order. Matched projections are 304.879/305.172 GPU-hours for control and
175.037/175.064 for the candidate: **42.61% less grouped kernel time**.
The order-42/44/46/48 candidate contributions are approximately
92.36/46.30/29.29/7.09 hours. The independent tail and production overhead
remain outside this projection. The assignment/corpus has not changed.

On one 34-core/11-pool/17-child example, shared bytes fall 36,632 -> 21,600
and calculated occupancy rises from two to four CTAs/SM. The phase logger
records summed CTA elapsed cycles, including stalls; these are diagnostic,
not exclusive SM utilisation or an additive attribution of speedup. Timing
binaries compile out the logger. The projection tool rejects instrumented
logs and different kernel configurations mixed in one projection.

All eight compile-time combinations pass cooperative CPU/UBSan tests over
four fields, including real large-core cases. The accepted candidate passes
CUDA memory, race and synchronization checks and matches 36 complete saved
6x28 campaign residues (18 queries over two fields). It compiles without
spills at 40 registers for both sm_89 and sm_120; no Ada timing is claimed.

Reproduce local exactness checks:

```sh
make hafnian-common-core-variants-test hafnian-common-core-projection-test
python3 tests/hafnian/common_core_variants_test.py \
  --groups build/common-core-plan6x27.log
```

For CUDA A/B, compile separate binaries with explicit values for **all three**
`CORE_OPT_HESS`, `CORE_OPT_BOUNDARY`, `CORE_OPT_SCRATCH` definitions. The
control is `0,0,0`; the accepted default is `1,1,1`. Phase binaries additionally
set `CORE_PROFILE=1` and must never supply performance projections. The
bounded driver does not provision workers:

```sh
python3 research/probes/hafnian_common_core_ab.py \
  --binaries build/common-core-gpu-479 --plan build/common-core-plan6x27.log \
  --output build/common-core-new-ab --variants control all --full --repeats 2
```

Logs and download checksums are in `build/common-core-gpu-479/`. Both temporary
workers and their disks were deleted; the first, confidential-computing
worker could not initialise CUDA and supplied no benchmark results.
No maintained production solver, historical result or campaign checkpoint
format changed. Next improve group selection using actual measured costs,
then implement persistent execution and measure the independent tail before
quoting a complete 6x27 campaign cost.

## Accepted cost-aware plan (Experiment 480)

Boundary size is a computational choice, not the query identity. Adding two
unused-by-defect tokens to a boundary removes one signed pair from the core,
halving its sign domain even if the group has no new children. The boundary
polynomial gets more expensive, so select the trade-off using actual timings.

The new planner repair keeps the old parent's owned queries and coefficients
fixed. It first tests odd boundaries up to eleven tokens and beneficial
merges; then tests bounded two-group rearrangements using two deterministic
anchors and a 7/9/11 menu. Every accepted change improves an integer cost
model including core-sign count and actual per-query CRT requirements.
Interpolation/clamping in that model is a search heuristic, not evidence of
speedup. Unknown shapes and group sizes above the measured range are refused.

Models predicted 175.05 -> 142.41 -> 140.96 grouped GPU-hours. Fresh full
sweeps on one GPU, with the CUDA binary unchanged, measured:

| Plan | Forward run | Reverse-order run |
| --- | ---: | ---: |
| Original | 174.241 h | 174.242 h |
| Boundary choices / merges | 141.822 h | 141.837 h |
| Plus bounded regrouping | 140.362 h | 140.341 h |

The last plan is accepted: **140.35 grouped GPU-hours**, 1.241x throughput.
All 513 samples cover its 171 grouped strata, each at 32,768 signs with 128
CPU-reference comparisons. Both plans preserve all 45,007,139 query IDs and
47,983,269,684,673 coefficient weight, with every row-map embedding audited.
The final plan has 7,316,815 groups, including the unchanged 53,179 singletons.
Boundary choices, not greater query reuse, account for most of the gain.

An explicit six-query 6x28 fixture moves four additional tokens to its
boundary (core 44 -> 40) without adding queries. All twelve complete residues
over two primes match saved independent campaign results. The fixture is in
`tests/hafnian/common_core_padded6x28.groups`. Small complete-catalog repairs
also validate ownership, row maps and deterministic output across producer
thread counts. The kernel and historical result formats are unchanged.

To reproduce the **two-stage** accepted plan (outputs must not already exist):

```sh
python3 research/probes/hafnian_common_core_cost.py \
  --output build/common-core-480.costs \
  build/common-core-gpu-479/full/all-r{0,1}-p{2147483647,2147483629,2147483587}.log
make build/hafnian_common_core_plan
build/hafnian_common_core_plan --catalog build/common-core-6x27.catalog \
  --replan build/common-core-6x27.plan --costs build/common-core-480.costs \
  --output build/common-core-6x27-cost480.plan --threads 16
build/hafnian_common_core_plan --catalog build/common-core-6x27.catalog \
  --replan build/common-core-6x27-cost480.plan --costs build/common-core-480.costs \
  --repair-anchors 2 --output build/common-core-6x27-repair480.plan --threads 16
build/hafnian_common_core_plan --catalog build/common-core-6x27.catalog \
  --verify build/common-core-6x27-repair480.plan --all-maps --threads 16
```

The two planning steps took 38s and 2m17s at about 2.56 GiB RSS. Final full
row-map verification took 40s at about 1.05 GiB. The accepted final plan digest
is `387b52f312cd2837de62671e39b53a3bf1de8513a9aaa785813bf2e70cf9b9b4`.
The model's source timing hashes are recorded in its table; GPU logs and
download checksum verification are under `build/common-core-gpu-480/`.
The worker and disk have been deleted.

The tail, including larger orders, still has 2,407,173,980,160 independent sign
terms, and persistent execution/reduction/dispatch/restart overhead remains.
Consequently 140 hours is **not a complete campaign estimate**. Cross-parent
reassignment and boundary sizes above eleven have not been tested here.

## Historical CPU-gate decision

The CPU gate is positive. The next meaningful test is a **bounded CUDA
partial-core prototype**, comparing complete real query groups against the
maintained GPU kernel, including boundary polynomial memory, synchronization,
occupancy and smaller groups after ownership. Do not convert the CPU ratios
into campaign hours or change production defaults. A scalable once-only
6x27 assignment remains unimplemented.

The original gate checklist, now with CPU measurements above:

Proceed to an optimized CPU finite-field gate on the actual logged groups:

1. Build common-core factors/moments once and all child boundary polynomials.
2. Compute six- and eight-vertex polynomial hafnian minors with shared small
   subset memoization; measure all boundary work, not just core reuse.
3. Compare complete small counts and instrument real-order arithmetic costs
   against independent Gray/resolvent evaluations. Preserve exact fallbacks
   for deficient cores; no assumption of a nonsingular adjacency matrix.
4. Measure how feasible local groups become a non-overlapping campaign
   assignment. The census does not establish global coverage or division of
   total runtime by the mean group size.
5. Only if this is promising, prototype CUDA and measure weighted complete
   throughput. Do not revise the historical 6x27 estimate yet.

Reproduce the bounded tests and census:

```sh
make hafnian-common-core-test six-by-twenty-eight-census-test
make six_by_twenty_eight_defect_census
build/six_by_twenty_eight_defect_census --slack 3 --threads 16 \
  --common-core-samples 128 --common-core-parents 4 --common-core-seed 475
```

Local artifacts: `build/common-core-6x27.{log,time}`, the 6x28 smoke log,
and `build/common-core-{tests,regression}.log`.
