# Rectangle-Free Grid Colourings

This repository contains exact solvers for counting colourings of an
`r x n` grid with no monochromatic axis-aligned rectangle. For `k` colours,
let

```text
T_k(r,n) = number of labelled rectangle-free k-colourings of an r x n grid.
```

Solvers may quotient row, column, colour, graph, or token symmetries
internally, but every reported answer counts labelled grid colourings.

## Algorithms at a glance

The repository uses four complementary exact methods. No single method is
best across all geometries.

| Method | Main use | Core reduction |
| --- | --- | --- |
| Token-state dynamic programming | `T_4(r,n)` for two through five rows | Process columns while recording which colour/row-pair tokens have already appeared. |
| Partition and structure-graph search | Exact counts and full chromatic polynomials at moderate sizes | Canonicalise column partitions, build a conflict graph, then count its proper colourings or chromatic polynomial. |
| GPU distribution contraction | Large single evaluations such as `T_4(7,9)` and `T_4(8,8)` | Split four colours into two bits, reduce the outer binary masks by symmetry, and contract cached half-grid token distributions with an exact weighted-disjointness join. |
| Endpoint matching and hafnians | `T_4(6,30)` and the near-endpoint `T_4(6,29)` | At six rows, token saturation turns minimum columns into edges of a 60-vertex graph; count perfect matchings exactly over finite fields. |

### 1. Token-state dynamic programming

A token `(colour, row pair)` records that two rows have the same colour in a
column. Reusing the token in another column creates exactly one monochromatic
rectangle, so a colouring is a sequence of columns with disjoint token sets.

- `src/small/count4.py`: closed forms for two and three rows.
- `src/small/4xn_count4.py`: weighted set-packing DP for four rows.
- `src/small/5xn_count4.c`: canonical state-space DP for five rows.

### 2. Partition and structure-graph search

The shared C implementation under `src/` enumerates canonical multisets of
column partitions. Each surviving structure induces a conflict graph whose
graph contribution is combined with exact multinomial and symmetry weights.

- `partition_poly`: computes the full chromatic polynomial.
- `partition_count4`: evaluates the same decomposition directly at four
  colours, with specialised pruning.
- `partition_poly_7` and `partition_poly_8`: bounded builds for seven- and
  eight-row work.

This family produced the full polynomial computations; it is distinct from
the GPU solvers, which compute only the single value `T_4(r,n)`.

### 3. GPU binary-mask distribution contraction

Write a four-colour as two binary coordinates. For a fixed first-bit mask
`G`, the second-bit choices on `G` and its complement are independent:

```text
T_4(r,n) = sum_G C(G) C(complement(G)).
```

The production pipeline:

1. quotients `G` by row, column, complement, and—on square grids—transpose
   symmetries;
2. splits each binary half-grid into two column blocks;
3. builds or reuses sparse distributions of row-pair tokens;
4. joins the two distributions by exact weighted set disjointness; and
5. performs the dominant predicate work with binary tensor-core MMA.

Every maintained distribution uses the global inner-bit token-plane quotient.
The 7x9 and 8x8 solvers share the grouped-layout builder, BMMA join,
checkpoint format, and provider-neutral reducer.

See [gpu_algorithm.md](docs/gpu_algorithm.md) for the mathematics and
[GPU_CODE.md](docs/GPU_CODE.md) for the maintained implementation surface and
campaign interface.

### 4. Six-row endpoint hafnians

For six rows there are 60 `(colour, row pair)` tokens. At 30 columns, every
column must consume exactly two tokens and all tokens must be used. Columns
therefore form a perfect matching in a fixed 60-vertex graph:

```text
T_4(6,30) = 30! * 2^30 * hafnian(A).
```

The implementation evaluates the hafnian modulo several primes on GPU and
uses CRT for the exact integer. For 6x29, a defect expansion reduces the
calculation to 29 related residual hafnians.

See [six_by_thirty_hafnian.md](docs/hafnian/six_by_thirty_hafnian.md) and
[six_by_twenty_nine_hafnian.md](docs/hafnian/six_by_twenty_nine_hafnian.md).

## Building

The CPU solvers require a C/C++ compiler and OpenMP:

```bash
make -j all
```

Executables and generated review artifacts are written to `build/`. Override
that location with `BUILD_DIR=/path/to/output` when needed.

On macOS, the Makefile expects Homebrew `libomp` under
`/opt/homebrew/opt/libomp` by default.

Build the maintained CUDA production surface with:

```bash
make gpu-production
```

The default CUDA target is `sm_89`; override `NVCCFLAGS` for another GPU
architecture.

## Quick examples

Small-row solvers:

```bash
python3 src/small/count4.py 2
python3 src/small/4xn_count4.py
make 5xn_count4
./build/5xn_count4
```

Direct four-colour and polynomial calculations:

```bash
./build/partition_count4 6 8
./build/partition_poly 6 8
./build/partition_poly_8 8 5 --prefix-depth 2 --task-end 1
```

The partition solvers support deterministic sharding:

```bash
./build/partition_poly 6 8 --task-start 0 --task-end 100 --poly-out a.poly
./build/partition_poly 6 8 --task-start 100 --task-end 200 --poly-out b.poly
./merge_poly.py --poly-out merged.poly a.poly b.poly
```

Useful shared options include `--prefix-depth`, `--task-start`, `--task-end`,
`--reorder`, `--adaptive-subdivide`, and their corresponding `--no-*`
controls. Profiling is selected by building the `_profile` targets.

Exercise the exact hafnian implementations with:

```bash
make six-by-thirty-hafnian-test
make six-by-twenty-nine-hafnian-test
```

The full GPU campaign commands, manifests, cache formats, checkpointing, and
validation procedure are documented in [GPU_CODE.md](docs/GPU_CODE.md).

## Validation

Run the non-CUDA regression suites with:

```bash
make gpu_result_checkpoint_test
./build/gpu_result_checkpoint_test
make gpu-campaign-test
python3 -m unittest -v \
  tests.research.test_reachable_distribution_rank_probe \
  tests.hafnian.test_six_by_thirty_matching_probe \
  tests.research.test_universal_state_dd_probe \
  tests.research.test_universal_state_tensor_rank_probe \
  tests.research.test_universal_state_symmetry_probe
make universal_state_symmetry_probe
./build/universal_state_symmetry_probe --self-test
```

Production GPU results use self-identifying, checksummed checkpoints bound to
the solver, algorithm configuration, cache, input corpus, and exact work
range. `tools/aggregate_gpu_v3.py` verifies and reduces complete 7x9 and 8x8
campaigns.

## Repository map

- `src/`: production solvers and their shared implementation.
- `tools/`: provider-neutral corpus, reduction, and campaign utilities.
- `tests/`: non-CUDA regression suites and exact solver fixtures.
- `research/probes/`: falsifiable algorithm and performance experiments.
- `research/gpu/`: rejected or profiling-only GPU prototypes.
- `docs/`: algorithm notes, implementation maps, and experiment history.
- `archive/gpu/`: isolated historical experiments.
- `legacy/gpu/`: exact regression implementations outside production.
- `results.txt`: recorded exact values.
- `docs/research/`: feasibility gates for proposed larger-grid algorithms.
- `lean/`: formalisation notes.

Research probes are intentionally not part of `make gpu-production`.

## Current scope

- The general partition source supports up to eight rows and sixteen columns.
- `partition_poly_7` is bounded to seven rows and seven columns.
- `partition_poly_8` is bounded to eight rows and eight columns.
- The maintained GPU production targets are specialised for 7x7, 7x9, and
  8x8 distribution-join campaigns.
- The hafnian solvers are geometry-specific implementations for 6x30 and
  6x29.

Exact recorded values are in [results.txt](results.txt); detailed performance
logs and experiment provenance are in
[experiments.md](docs/experiments.md).

## Selected completed computations

These are measured campaign totals, not projections. Aggregate GPU time sums
the time used by every GPU and is therefore distinct from elapsed wall time.

| Computation | Hardware and parallelism | Aggregate compute | Elapsed time |
| --- | --- | ---: | ---: |
| Full 7x7 chromatic polynomial | Distributed CPU cluster | About 200,000 core-hours | Historical distributed run |
| `T_4(7,7)` independent check | 1 L40S | 30.1 GPU-seconds | 35.6 seconds |
| `T_4(8,8)` | 1,024 one-GPU shards; heterogeneous GPUs and variable concurrency | 335.3 GPU-hours | 19.6 GPU-minutes per shard on average |
| `T_4(7,9)` independent check | 128 shards; up to 8 RTX PRO 6000 GPUs | 28.64 GPU-hours | About 5 hours |
| `T_4(6,30)` | 8 RTX PRO 6000 GPUs | 0.79 GPU-hours | About 10 minutes |
| `T_4(6,29)` | 8 RTX PRO 6000 GPUs; interrupted and resumed | 4.14 GPU-hours | About 39 minutes including setup |

The exact integers are collected in [results.txt](results.txt), while the
algorithms, validation checks, and detailed timing breakdowns are recorded in
[experiments.md](docs/experiments.md).

## Acknowledgements and OEIS

The partition-based solver line owes a great deal to
[Adam P. Goucher](https://cp4space.hatsya.com/), particularly the
polynomial/structure-graph formulation and canonical graph keys.

Related OEIS entries:

- [A200045](https://oeis.org/A200045): four-colour rectangle-free grid
  colourings.
- [A391612](https://oeis.org/A391612): colourings of the 6x6 grid as a
  function of the number of colours.
