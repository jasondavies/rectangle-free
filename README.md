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
- partition-based structure-graph solvers with [nauty][nauty]-backed canonical caching.

The two partition-based C solvers in particular came out of a sequence of
algorithmic experiments. The important distinction is:

- `partition_poly.c` computes the full chromatic polynomial associated with a
  column structure.
- `partition_count4.c` keeps the same structure search, but specialises hard to
  `k = 4` and turns the graph side into an exact branch-and-bound search.

## File guide

### Small-width Python scripts

- `count4.py`
  Small dispatcher for exact `k = 4` counts in the easy cases. It currently
  handles `T_4(2, n)` and `T_4(3, n)`.

- `4xn_count4.py`
  Weighted set-packing DP on row-pair masks for `T_4(4, n)`.

### 5-row C solver

- `5xn_count4.c`
  Fast C version of the `5 x n`, `k = 4` state-space DP. This is the
  row/colour-canonicalised token-mask solver.

### 6-row C solvers

- `partition_poly.c`
  Partition/structure-graph solver that computes the chromatic polynomial
  `P(x)` for the reduced graph induced by a column multiset. It uses nauty to
  canonicalise graphs for caching and prints `P(4)` and `P(5)` at the end.

- `partition_count4.c`
  Partition/structure-graph solver specialised to `k = 4`. Instead of computing
  the full polynomial, it counts 4-colourings directly. It adds stronger
  pruning, 4-colourability checks, and prefix-task generation for parallel
  sharding.

### 7-row C solver

- `7xn_poly.c`
  Partition/structure-graph solver specialised to 7 rows and currently up to
  7 columns. It keeps the same canonical structure search, but switches the
  leaf backend to DSATUR counting in evaluation space and prints the result in
  Newton / binomial-basis form together with sampled values such as `P(4)` and
  `P(5)`.

### Data

- `counts.txt`
  Table of currently recorded counts. Rows correspond to the number of grid
  rows, columns correspond to `n`.

## Algorithm split

There are two main algorithm families in this repo.

### 1. State-space / token-mask DP

Used by:

- `4xn.py`
- `5xn_count4.c`

These programs encode which row-pair / colour combinations are still available,
then recurse with memoisation. For `5 x n`, the important optimisation is
canonicalisation under row permutations and colour permutations.

### 2. Partition / structure-graph search

Used by:

- `partition_poly.c`
- `partition_count4.c`
- `7xn_poly.c`

These programs enumerate column partitions, build a structure graph describing
interactions between non-singleton colour classes, and use nauty to cache
isomorphic graphs via canonical labelling.

At a high level, the partition solvers work like this:

1. Enumerate multisets of column partitions in canonical order, quotienting out
   column permutations and row symmetries.
2. For each structure, build the conflict graph on the complex blocks
   (non-singleton colour classes) induced by the chosen columns.
3. Weight each structure by:
   - the multinomial factor for repeated columns,
   - the row-orbit factor from the surviving row stabiliser,
   - the singleton-colour contribution coming from the partition type,
   - and finally either the chromatic polynomial of the conflict graph or the
     number of proper 4-colourings of that graph.

This gives two related but different solvers:

- `partition_poly.c`
  treats the graph contribution symbolically and computes a chromatic
  polynomial `P(x)` by deletion-contraction, with canonical graph caching via
  nauty.

- `partition_count4.c`
  keeps the same canonical partition search, but replaces the symbolic graph
  stage with a direct 4-colouring counter and adds monotone pruning:
  incremental conflict graphs, pair-shadow capacity bounds, cheap `K5`
  obstruction checks, and exact 4-colourability tests on the current prefix
  graph.

- `7xn_poly.c`
  keeps the partition search and graph canonicalisation, but accumulates
  structure weights at evaluation points and computes each distinct leaf graph
  with a DSATUR-based falling-factorial counter instead of deletion-contraction.

In practice, that division is useful:

- the polynomial solver is the more general method and is the right tool when
  you want `P(x)` itself or values for several numbers of colours,
- the `k = 4` solver is much more aggressive and is the practical route for the
  largest current `6 x n` runs.

## Running the Python scripts

Each Python file is a standalone script. Examples:

```bash
python3 count4.py 2
python3 count4.py 3
python3 4xn_count4.py
```

The scripts print tables of exact counts.

## Building the C programs

### `5xn_count4.c`

This file is self-contained.

Example:

```bash
cc -O3 -march=native -std=c11 5xn_count4.c -o 5xn_count4
```

Run with:

```bash
./5xn_count4
```

### `partition_poly.c`, `partition_count4.c`, and `7xn_poly.c`

These require:

- OpenMP,
- a vendored official nauty release under `third_party/nauty`.

The repository already vendors the nauty source in `third_party/nauty`. The
top-level `make` does not build in place there; instead it creates a private
configured copy in `third_party/nauty-build`, enables TLS there, and builds
`nautyT.a`.

In the default layout:

- `third_party/nauty` is the checked-in nauty source tree,
- `third_party/nauty-build` is a generated build directory.

So the normal build is simply:

```bash
make
```

You only need to override the nauty path if you deliberately want to use a
different source tree.

The default `Makefile` assumes:

```text
third_party/nauty
third_party/nauty-build
```

and uses:

```make
NAUTY_DIR ?= ./third_party/nauty
NAUTY_BUILD_DIR ?= ./third_party/nauty-build
```

You can override that path at build time if needed:

```bash
make NAUTY_DIR=/path/to/nauty
```

If you want to compile manually, the command shape is:

```bash
rm -rf third_party/nauty-build
cp -R third_party/nauty third_party/nauty-build
(cd third_party/nauty-build && ./configure --enable-tls && make nautyT.a)

cc -O3 -march=native -fopenmp -I./third_party/nauty-build -I./third_party/nauty \
  -DUSE_TLS -o partition_poly partition_poly.c ./third_party/nauty-build/nautyT.a -lm
cc -O3 -march=native -fopenmp -I./third_party/nauty-build -I./third_party/nauty \
  -DUSE_TLS -o partition_count4 partition_count4.c ./third_party/nauty-build/nautyT.a -lm
cc -O3 -march=native -fopenmp -I./third_party/nauty-build -I./third_party/nauty \
  -DUSE_TLS -o 7xn_poly 7xn_poly.c ./third_party/nauty-build/nautyT.a -lm
```

On macOS with Apple clang, OpenMP usually also needs Homebrew `libomp`. The
current `Makefile` uses
`/opt/homebrew/opt/libomp` automatically on Darwin.

For the self-contained 5-row solver, simply run:

```bash
make 5xn_count4
```

To build everything:

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
./6xn_poly 6 8
```

This prints:

- progress information,
- the final chromatic polynomial `P(x)`,
- `P(4)` and `P(5)`,
- cache statistics.

It also supports sharding:

```bash
./6xn_poly 6 8 --task-start 0 --task-end 100 --poly-out shard_a.poly
./6xn_poly 6 8 --task-start 100 --task-end 200 --poly-out shard_b.poly
./6xn_poly --merge --poly-out merged.poly shard_a.poly shard_b.poly
```

The shard file format starts with the header `RECT_POLY_V1`.

## Using `6xn`

Default run:

```bash
./6xn
```

Explicit size:

```bash
./6xn 6 8
```

Useful options:

- `--profile` prints counters and timing for the main pruning paths.
- `--prefix-depth N` chooses the prefix-task depth explicitly.
- `--count-out FILE` writes a shard file.
- `--task-start` and `--task-end` support distributed runs over contiguous task ranges.

Example:

```bash
./6xn 6 8 --profile --prefix-depth 4 --count-out shard.count
./6xn --merge --count-out merged.count shard1.count shard2.count
```

The shard file format starts with the header `RECT_COUNT4_V1`.

The final line printed by the solver is:

```text
T_4(r,n) = ...
```

## Using `7xn_poly`

Default run:

```bash
./7xn_poly
```

Explicit size:

```bash
./7xn_poly 7
```

This solver is specialised to 7 rows. It currently supports up to 7 columns,
prints the final result in Newton / binomial-basis form, and writes shard files
with the header `RECT_EVAL_V1`.

## Notes on limits

Current compile-time limits in the partition-based C solvers:

- `6xn_poly.c`: up to 6 rows and 16 columns.
- `6xn.c`: up to 6 rows and 8 columns.
- `7xn_poly.c`: fixed at 7 rows and currently up to 7 columns.

These limits come from the current fixed-size data structures and the size of
the induced structure graphs.

## Current status

- Small widths `2, 3, 4` have compact direct scripts.
- Width `5` has a dedicated state-space DP solver in C.
- Width `6` currently has the partition-based C solvers in-tree.
- Width `7` now has an evaluation-space polynomial prototype for `n <= 7`.
- `counts.txt` is the current in-repo table of known values.

## Acknowledgements

The partition-based solver line owes a lot to [Adam P. Goucher][adam]. In
particular, the main ideas behind the polynomial/graph approach came out of
discussions with him:

- reframing the structure contribution in terms of chromatic polynomials of
  conflict graphs,
- using canonical graph labelling with nauty/traces so memoisation works up to
  graph isomorphism rather than on raw labelled graphs,
- reducing graphs by peeling vertices whose neighbourhoods form cliques, which
  only contributes a multiplicative factor to the chromatic polynomial,
- and using bounded per-thread graph-to-polynomial caches rather than trying to
  maintain one giant global symbolic state.

Other useful suggestions from Adam that influenced the implementations and the
way they were tested include:

- using deletion-contraction with memoisation as the basic graph-polynomial
  engine,
- treating the transpose identity `T_k(m, n) = T_k(n, m)` as an end-to-end
  correctness check,
- improving canonical-prefix logic by borrowing ideas from canonical Boolean
  chain search rather than relying purely on a naive permutation loop,
- and exploring lazy per-thread sums over canonical graphs as an alternative
  way to organise the symbolic computation.

More broadly, there are two Adam-inspired lines in this repository:

- the obstruction-pattern / bucket-mask line, which led to the `5 x n` and
  `6 x n` state-space solvers,
- and the structures + graph-colouring + nauty line, which led to the
  partition-based C solvers.

[adam]: https://cp4space.hatsya.com/
[nauty]: https://pallini.di.uniroma1.it/
