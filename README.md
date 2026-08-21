# Rectangle-Free Grid Colourings

This repository contains exact solvers for counting colourings of an `r x n`
grid with no monochromatic axis-aligned rectangle: no four corners of a
grid-aligned rectangle may all share the same colour.

The reported values are counts of labelled grid colourings. Some solvers
quotient out row, column, colour, or graph symmetries internally, but the
final answers are still exact counts of concrete colourings.

## Problem statement

For fixed numbers of rows `r`, columns `n`, and colours `k`, let `T_k(r, n)`
be the number of `k`-colourings of the `r x n` grid with no monochromatic
axis-aligned rectangle.

The repository currently contains:

- small-width direct scripts for `k = 4`,
- a 5-row state-space solver in C,
- partition / structure-graph solvers in C for exact counting and chromatic
  polynomial computation,
- helper tools for merging polynomial shards,
- Lean notes under `lean/`,
- recorded results in `results.txt`,
- and longer experimental notes in `experiments.md`.

## Repository guide

### Small-width exact solvers

- `count4.py`
  Closed forms for `T_4(2, n)` and `T_4(3, n)`.

- `4xn_count4.py`
  Weighted set-packing dynamic program for `T_4(4, n)`.

- `5xn_count4.c`
  State-space dynamic program for `T_4(5, n)` with row / colour
  canonicalisation.

- `pairmask_transfer_probe.c`
  Experimental fixed-4 transfer over row-pair histories. It is a research
  probe for wider grids rather than the current production solver.

### Partition-based C solvers

- `partition_poly`
  General partition / structure-graph solver built from the shared sources
  under `src/`. It enumerates canonical column multisets, builds the induced
  conflict graph, and computes the chromatic polynomial.

- `partition_count4`
  Fixed-`k = 4` build target for the shared partition solver. It uses the same
  search, but runs the direct exact 4-colouring path with the special-4
  pruning enabled.

- `partition_poly_7`
  Build target that compiles the same shared solver with `DEFAULT_ROWS=7`,
  `DEFAULT_COLS=7`, and `MAX_COLS=7`. This is the current 7-row polynomial
  executable in the tree.

- `partition_poly_8`
  Build target specialised for grids up to `8 x 8`. Keeping `MAX_COLS=8`
  bounds conflict graphs at 32 vertices and avoids the larger graph/cache
  representation required by the general 16-column executable.

### Helper scripts and data

- `merge_poly.py`
  Merge `RECT_POLY_V1` shard files produced by `partition_poly` or
  `partition_poly_7`.

- `results.txt`
  Table of recorded exact counts.

- `experiments.md`
  Working notes, measurements, and algorithmic experiments.

## Algorithm split

There are two main solver families.

### 1. State-space / token-mask dynamic programming

Used by:

- `4xn_count4.py`
- `5xn_count4.c`

These programs track which row-pair / colour combinations remain legal after a
sequence of columns and recurse with memoisation. The 5-row solver also
canonicalises under row and colour permutations.

### 2. Partition / structure-graph search

Used by:

- `partition_poly`
- `partition_count4`
- `partition_poly_7`

These programs enumerate canonical multisets of column partitions, build a
conflict graph for the complex colour classes, and weight each structure by:

1. the multinomial factor for repeated columns,
2. the row-orbit factor from the surviving row stabiliser,
3. the singleton-colour contribution from the partition type,
4. and the graph contribution.

The graph contribution differs by solver:

- `partition_poly`
  computes the chromatic polynomial symbolically, with WL-based canonical graph
  keys and a labelled fallback.

- `partition_count4`
  counts proper 4-colourings directly inside the shared solver, with
  special-4 pruning such as pair-shadow bounds, cheap obstruction checks,
  and exact 4-colourability tests.

- `partition_poly_7`
  is the `7 x 7` build of `partition_poly`, used for the current 7-row
  experiments.

## Running the small solvers

The Python scripts are standalone:

```bash
python3 count4.py 2
python3 count4.py 3
python3 4xn_count4.py
```

The 5-row C solver is also standalone:

```bash
make 5xn_count4
./5xn_count4
```

The optional transfer probe supports two through eight rows:

```bash
make pairmask_transfer_probe
./pairmask_transfer_probe 8 2
./pairmask_transfer_probe 8 3 --ordered
./pairmask_transfer_probe 8 4 --ordered 0
./pairmask_transfer_probe 8 4 --contracted
./pairmask_transfer_probe 8 5 --contracted 13 350
```

It uses an exact colored-incidence individualize/refine key for row/colour
symmetry. It is intended for state-growth experiments rather than full `8x8`
runs. The
`--ordered` mode is a fixed-four recurrence: it memoises the exact
number of ordered remaining columns from each available row-pair-token state.
It decomposes the first column into exact row/colour orbits; an optional orbit
index runs one independently checkpointable contribution. Contracted mode also
accepts a second index selecting a deterministic canonical second-column shard,
which is the intended unit for cluster jobs. The `--setpack`
mode retains an unordered exact-depth comparator. The `--contracted` mode
counts the final two columns together by subset convolution instead of
materialising one-column terminal states. It is the low-memory experimental
alternative and provides deterministic second-column shards. For throughput,
compare it against `partition_count4`: on the current one-core `8x4` benchmark,
`partition_count4` is substantially faster, while contracted transfer uses
substantially less memory. See Experiment 213 in `experiments.md`.

## Building the partition-based solvers

`partition_poly` and `partition_count4` require OpenMP.

Normal build:

```bash
make
```

This builds the tracked top-level executables:

- `5xn_count4`
- `partition_count4`
- `partition_poly`
- `partition_poly_7`
- `partition_poly_8`
- `partition_poly_profile`
- `partition_poly_7_profile`
- `partition_poly_8_profile`

On macOS with Apple clang, OpenMP usually also needs Homebrew `libomp`. The
current `Makefile` uses `/opt/homebrew/opt/libomp` automatically on Darwin.

## Using `partition_poly`

Default run:

```bash
./partition_poly
```

Explicit size:

```bash
./partition_poly 6 8
```

Useful options:

- `--prefix-depth N`
- `--task-start N --task-end M`
- `--reorder`
- `--no-reorder`
- `--adaptive-subdivide`
- `--no-adaptive-subdivide`
- `--adaptive-max-depth N`
- `--adaptive-work-budget N`
- `--poly-out FILE`

`partition_poly` defaults to adaptive subdivision enabled with max depth `5`
and work budget `1000`. Use `--no-adaptive-subdivide` to force the legacy
non-adaptive path or to run with `--prefix-depth 3/4`.

Partition hardness reorder is enabled by default. Use `--no-reorder` to
restore the legacy partition IDs and task numbering.

Profiling is selected at build time, not by a runtime `--profile` flag. Build
`partition_poly_profile`, `partition_poly_7_profile`, or
`partition_poly_8_profile` for a profiling binary;
only profiling builds accept `--task-times-out FILE`.

For sharded runs:

```bash
./partition_poly 6 8 --task-start 0 --task-end 100 --poly-out shard_a.poly
./partition_poly 6 8 --task-start 100 --task-end 200 --poly-out shard_b.poly
./merge_poly.py --poly-out merged.poly shard_a.poly shard_b.poly
```

Polynomial shard files use the header `RECT_POLY_V1`.

## Using `partition_count4`

Default run:

```bash
./partition_count4
```

Explicit size:

```bash
./partition_count4 6 8
```

Useful options:

- `--prefix-depth N`
- `--task-start N --task-end M`
- `--reorder`
- `--no-reorder`
- `--adaptive-subdivide`
- `--no-adaptive-subdivide`
- `--adaptive-max-depth N`
- `--adaptive-work-budget N`

Examples:

```bash
./partition_count4 6 8 --prefix-depth 4
./partition_count4 6 8 --no-reorder
./partition_count4 6 8 --task-start 0 --task-end 100
```

## Using `partition_poly_7`

Build the specialised 7-row target:

```bash
make partition_poly_7
```

Run it with the default `7 x 7` parameters:

```bash
./partition_poly_7
```

Or pass the dimensions explicitly:

```bash
./partition_poly_7 7 7
```

## Using `partition_poly_8`

Build and run the specialised 8-row target with:

```bash
make partition_poly_8
./partition_poly_8 8 5 --prefix-depth 2 --task-end 1
```

This target enables a bounded shared hard-graph cache by default. Set
`RECT_HARD_CACHE_BITS=0` to disable it for low-reuse shards.
It also dispatches connected residual graphs with at least 18 vertices and
greedy min-fill width at most 5 to the tree-decomposition polynomial solver.
Override this with `RECT_TREEWIDTH_LIMIT` and `RECT_TREEWIDTH_MIN_N`; a limit
of `0` disables the dispatch.

Repeated terminal graphs are combined before graph-polynomial evaluation. The
default aggregation table has 4096 slots for a single-thread run and 1024 slots
per worker for a multi-thread run. Set `RECT_TERMINAL_AGGREGATE_BITS` to choose
a table size of `2^bits`, or to `0` to disable aggregation.

For a faster host-specific production binary, build with profile-guided
optimization:

```bash
make partition_poly_8_pgo
```

This runs an instrumented single-thread `8x5` task-0 training shard, then builds
`partition_poly_8_pgo` from the resulting profile. It is intentionally not part
of `make all`, because training takes longer and uses the normal 8-row cache
memory. Re-run the target after changing compiler versions or relevant flags.

## Compile-time limits

Current limits in the checked-in C sources:

- `partition_poly`: up to 8 rows and 16 columns.
- `partition_count4`: same solver limits as `partition_poly`.
- `partition_poly_7`: 7 rows and up to 7 columns.
- `partition_poly_8`: 8 rows and up to 8 columns.

These limits come from the current fixed-size structures and the size of the
induced conflict graphs.

## Optional local helpers

If your working tree also contains the local `gcloud/` helper scripts, they can
build an Arm64 `partition_poly_7` binary and launch sharded GCP workers. They
are operational helpers rather than part of the core solver codepath described
above.

## Acknowledgements

The partition-based solver line owes a lot to [Adam P. Goucher][adam]. In
particular, the polynomial / graph approach and the focus on canonical graph
keys came directly out of those discussions.

## OEIS

This work has contributed to the [On-Line Encyclopedia of Integer Sequences](https://oeis.org/):

- [A200045](https://oeis.org/A200045): Number of 4-colourings of an nxm grid with no monochromatic axis-aligned rectangle (best known: 6x8).
- [A391612](https://oeis.org/A391612): Number of n-colourings of a 6x6 grid with no monochromatic axis-aligned rectangle (all n via chromatic polynomial).

[adam]: https://cp4space.hatsya.com/
