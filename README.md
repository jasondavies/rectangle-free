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

### Partition-based C solvers

- `partition_poly.c`
  General partition / structure-graph solver. It enumerates canonical column
  multisets, builds the induced conflict graph, and computes the chromatic
  polynomial.

- `partition_count4.c`
  `k = 4` specialisation of the same search, with direct exact 4-colouring
  counts, stronger pruning, and shard merge support.

- `partition_poly_7`
  Build target that compiles `partition_poly.c` with `DEFAULT_ROWS=7`,
  `DEFAULT_COLS=7`, and `MAX_COLS=7`. This is the current 7-row polynomial
  executable in the tree.

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

- `partition_poly.c`
- `partition_count4.c`
- `partition_poly_7`

These programs enumerate canonical multisets of column partitions, build a
conflict graph for the complex colour classes, and weight each structure by:

1. the multinomial factor for repeated columns,
2. the row-orbit factor from the surviving row stabiliser,
3. the singleton-colour contribution from the partition type,
4. and the graph contribution.

The graph contribution differs by solver:

- `partition_poly.c`
  computes the chromatic polynomial symbolically, with [nauty][nauty]-backed
  canonical graph caching.

- `partition_count4.c`
  counts proper 4-colourings directly and adds pruning such as pair-shadow
  bounds, cheap obstruction checks, and exact 4-colourability tests.

- `partition_poly_7`
  is the `7 x 7` build of `partition_poly.c`, used for the current 7-row
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

## Building the partition-based solvers

`partition_poly.c` and `partition_count4.c` require:

- OpenMP,
- the vendored nauty source under `third_party/nauty`.

The top-level `Makefile` builds [nauty][nauty] in a private configured copy
under `third_party/nauty-build`, then links against `nautyT.a`.

Normal build:

```bash
make
```

This builds the tracked top-level executables:

- `5xn_count4`
- `partition_count4`
- `partition_poly`
- `partition_poly_7`

If you want to override the nauty source path:

```bash
make NAUTY_DIR=/path/to/nauty
```

The default layout is:

```text
third_party/nauty
third_party/nauty-build
```

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
- `--adaptive-subdivide`
- `--adaptive-max-depth N`
- `--adaptive-work-budget N`
- `--poly-out FILE`
- `--profile`
- `--task-times-out FILE`

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

- `--profile`
- `--prefix-depth N`
- `--count-out FILE`
- `--task-start N --task-end M`
- `--task-stride N --task-offset N`
- `--merge`

Examples:

```bash
./partition_count4 6 8 --profile
./partition_count4 6 8 --prefix-depth 4 --count-out shard.count
./partition_count4 6 8 --task-start 0 --task-end 100000 --count-out shard0.count
./partition_count4 --merge --count-out merged.count shard0.count shard1.count
```

Count shard files use the header `RECT_COUNT4_V1`.

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

## Compile-time limits

Current limits in the checked-in C sources:

- `partition_poly.c`: up to 7 rows and 16 columns.
- `partition_count4.c`: up to 6 rows and 8 columns.
- `partition_poly_7`: 7 rows and up to 7 columns.

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
labelling with [nauty][nauty] came directly out of those discussions.

[adam]: https://cp4space.hatsya.com/
[nauty]: https://pallini.di.uniroma1.it/
