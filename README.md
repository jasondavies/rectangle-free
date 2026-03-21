# Rectangle-Free Grid Colourings

This repository contains exact solvers for counting colourings of an `r x n`
grid with no monochromatic axis-aligned rectangle.

Equivalently, if four cells form the corners of a rectangle with sides parallel
to the grid axes, those four corner cells are not allowed to all have the same
colour.

The code here does not count colourings "up to symmetry". Some solvers use row,
colour, or graph symmetries internally to reduce the search space, but the final
counts are counts of actual labelled grid colourings.

## Problem statement

For fixed numbers of rows `r`, columns `n`, and colours `k`, let `T_k(r, n)` be
the number of `k`-colourings of the `r x n` grid with no monochromatic
axis-aligned rectangle.

This repo contains a mix of:

- small-width closed forms / direct Python programs,
- state-space dynamic programs with symmetry reduction,
- partition-based structure-graph solvers with nauty-backed canonical caching.

## File guide

### Small-width Python scripts

- `2xn.py`
  Closed form for `T_4(2, n)`.

- `3xn.py`
  Small symbolic / polynomial computation for `T_k(3, n)`, printed here for
  `k = 4`.

- `4xn.py`
  Weighted set-packing DP on row-pair masks for `T_4(4, n)`.

- `5xn.py`
  Python implementation of the `5 x n`, `k = 4` solver using memoized DP with
  canonicalization under row permutations and colour permutations.

- `6xn.py`
  Python implementation for `6 x n`, `k = 4` using a weighted set-packing DP on
  60 pair-colour tokens, with optional row+colour canonicalization.

### 5-row C solver

- `5xn.c`
  Fast C version of the `5 x n`, `k = 4` state-space DP. This is the
  row/colour-canonicalized token-mask solver.

### Partition-based C solvers

- `partition_poly.c`
  Partition/structure-graph solver that computes the chromatic polynomial
  `P(x)` for the reduced graph induced by a column multiset. It uses nauty to
  canonicalize graphs for caching and prints `P(4)` and `P(5)` at the end.

- `partition_count4.c`
  Partition/structure-graph solver specialized to `k = 4`. Instead of computing
  the full polynomial, it counts 4-colourings directly. It adds stronger
  pruning, 4-colourability checks, and prefix-task generation for parallel
  sharding.

### Data

- `counts.txt`
  Table of currently recorded counts. Rows correspond to the number of grid
  rows, columns correspond to `n`.

## Algorithm split

There are two main algorithm families in this repo.

### 1. State-space / token-mask DP

Used by:

- `4xn.py`
- `5xn.py`
- `5xn.c`
- `6xn.py`

These programs encode which row-pair / colour combinations are still available,
then recurse with memoization. For `5 x n`, the important optimization is
canonicalization under row permutations and colour permutations.

### 2. Partition / structure-graph search

Used by:

- `partition_poly.c`
- `partition_count4.c`

These programs enumerate column partitions, build a structure graph describing
interactions between non-singleton colour classes, and use nauty to cache
isomorphic graphs via canonical labeling.

This is the reason the files were renamed from the old `solver.c` /
`solver4.c`: they are meant to be distinguished from the `5xn` state-space DP
solvers by method, not just by output format.

## Running the Python scripts

Each Python file is a standalone script. Examples:

```bash
python3 2xn.py
python3 3xn.py
python3 4xn.py
python3 5xn.py
python3 6xn.py
```

The scripts print tables of exact counts.

## Building the C programs

### `5xn.c`

This file is self-contained.

Example:

```bash
cc -O3 -march=native -std=c11 5xn.c -o 5xn
```

Run with:

```bash
./5xn
```

### `partition_poly.c` and `partition_count4.c`

These require:

- OpenMP,
- a vendored official nauty release under `third_party/nauty`.

Recommended setup:

- download an official nauty release tarball from the nauty distribution site,
- unpack it into `third_party/nauty`,
- run `./configure --enable-tls` inside that directory,
- build `nautyT.a` inside that directory,
- then build this repo with `make`.

The default [Makefile](/Users/jason/src/rectangle-free/Makefile) assumes:

```text
third_party/nauty
```

and uses:

```make
NAUTY_DIR ?= ./third_party/nauty
```

You can override that path at build time if needed:

```bash
make NAUTY_DIR=/path/to/nauty
```

If you want to compile manually, the command shape is:

```bash
cc -O3 -march=native -fopenmp -I./third_party/nauty \
  -DUSE_TLS -o partition_poly partition_poly.c ./third_party/nauty/nautyT.a -lm
cc -O3 -march=native -fopenmp -I./third_party/nauty \
  -DUSE_TLS -o partition_count4 partition_count4.c ./third_party/nauty/nautyT.a -lm
```

On macOS with Apple clang, OpenMP usually also needs Homebrew `libomp`. The
current [Makefile](/Users/jason/src/rectangle-free/Makefile) uses
`/opt/homebrew/opt/libomp` automatically on Darwin.

For the self-contained `5xn` solver, simply run:

```bash
make 5xn
```

To build everything once nauty is present:

```bash
make
```

## Using `partition_poly`

Default run:

```bash
./partition_poly
```

Explicit size:

```bash
./partition_poly 6 8
```

This prints:

- progress information,
- the final chromatic polynomial `P(x)`,
- `P(4)` and `P(5)`,
- cache statistics.

It also supports sharding:

```bash
./partition_poly 6 8 --task-start 0 --task-end 100 --poly-out shard_a.poly
./partition_poly 6 8 --task-start 100 --task-end 200 --poly-out shard_b.poly
./partition_poly --merge --poly-out merged.poly shard_a.poly shard_b.poly
```

The shard file format starts with the header `RECT_POLY_V1`.

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

- `--profile` prints counters and timing for the main pruning paths.
- `--prefix-depth N` chooses the prefix-task depth explicitly.
- `--count-out FILE` writes a shard file.
- `--task-start`, `--task-end`, `--task-stride`, `--task-offset` support
  distributed runs.

Example:

```bash
./partition_count4 6 8 --profile --prefix-depth 4 --count-out shard.count
./partition_count4 --merge --count-out merged.count shard1.count shard2.count
```

The shard file format starts with the header `RECT_COUNT4_V1`.

The final line printed by the solver is:

```text
T_4(r,n) = ...
```

## Notes on limits

Current compile-time limits in the partition-based C solvers:

- `partition_poly.c`: up to 6 rows and 16 columns.
- `partition_count4.c`: up to 6 rows and 8 columns.

These limits come from the current fixed-size data structures and the size of
the induced structure graphs.

## Current status

- Small widths `2, 3, 4` have compact direct programs.
- Width `5` has a dedicated state-space DP solver in Python and C.
- Width `6` has both a Python token-mask solver and partition-based C solvers.
- `counts.txt` is the current in-repo table of known values.
