# Dense colour-class gate for nine-by-nine

## Exact decomposition

A direct four-colouring partitions the 81 cells into four `C4`-free binary
matrices.  Order those matrices by `(number of cells, bit mask)`, and call the
two largest `A` and `B`.  Then

\[
|A|\geq\lceil81/4\rceil=21,
\qquad
|B|\geq\left\lceil\frac{81-|A|}{3}\right\rceil.
\]

For fixed disjoint `A` and `B`, put

\[
R=K_{9,9}\setminus(A\cup B).
\]

The remaining two colour classes are an ordered binary partition `C,D` of
`R`.  Both must be `C4`-free, and their ordering keys must not exceed the key
of `B`.  If `N(A,B)` is that filtered binary-completion count, then

\[
T_4(9,9)=12\sum_{A>B}N(A,B),
\]

where the sum uses the unique two densest classes.  Ordinarily `N` includes
the two orientations of `C,D`, while the unordered four-block partition has
24 direct-colour labellings.  The same factor 12 also handles the degenerate
`C=D=empty` case exactly.  Exhaustive tests reproduce `T_4(2,2)=252`,
`T_4(2,3)=3912`, and `T_4(3,3)=228984`.

This is materially different from the two-bit outer-mask sum.  It first
enumerates only unusually dense `C4`-free colour classes, then reuses an exact
binary-completion primitive on a much smaller residual mask.

## Density gates

For a `C4`-free bipartite graph with degrees `d_j` on one side,

\[
\sum_j\binom{d_j}{2}\leq\binom92=36,
\]

because any pair of vertices on the other side has at most one common
neighbour.  Applying this on both sides leaves very few degree sequences near
the extremum.  On one side the counts for 21 through 30 edges are

```text
edges:       21  22  23  24  25  26  27  28  29  30
sequences:  198 189 169 143 112  78  48  23   8   1
```

The final 30-edge sequence is only `(4,4,4,3,3,3,3,3,3)`, with all 36
two-path slots saturated.  It is necessary but not realizable.

Using nauty 2.8.8's exact bicoloured isomorph-free generator,

```text
genbg -q -u -v -Z1 9 9 21:30
```

gives the following `S9 x S9` orbit census:

| cells | canonical `C4`-free classes |
|---:|---:|
| 21 | 908,041 |
| 22 | 640,970 |
| 23 | 334,631 |
| 24 | 120,986 |
| 25 | 28,052 |
| 26 | 3,794 |
| 27 | 302 |
| 28 | 14 |
| 29 | 1 |
| 30 | 0 |
| **total** | **2,036,791** |

Thus the exact first-class interval is 21--29 and the exact second-class
minimum is 18, attained when `|A|` is 27--29.  The often quoted 17 gate uses
the elementary 30-edge upper bound, but the nonexistent 30-edge class removes
that case.

The first-class census is small enough to store and shard.  This is a real
reduction from trillions of outer binary-mask representatives.

## Second-class gate

For a fixed `A`, represent a partial `B` by the set of column pairs already
used together in a row.  There are 36 possible column pairs, and `B` is
`C4`-free exactly when no pair is used by two rows.  The compiled probe counts
labelled second classes inside the complement of `A` with this exact state.

The unique 29-cell first class is already difficult:

```text
rows processed       1       2        3         4          5
DP states            28     717   26,486   780,533  18,909,760
```

The sixth row exceeds a 100-million-state cap.  The full uncoloured
automorphism group of this extremal graph has size 24, so quotienting by its
colour-preserving stabilizer cannot provide more than a factor 24 on complete
second classes.

This does **not** reject the dense-first algorithm.  It rejects a conventional
single-table row DP for the second class.  A balanced row split produces two
sparse families of 36-bit pair-resource masks; counting compatible halves is
again a weighted set-disjointness join, which is directly suitable for the
existing GPU/BMMA machinery.  The remaining feasibility questions are:

1. How many complete `B` orbits survive for representative first classes?
2. How much reuse exists among residual masks `R` and their binary completion
   signatures?
3. Can second-class construction and residual completion be fused so that
   individual `B` masks are not written as a huge intermediate corpus?

## Verdict and next gate

Accept the dense-first formulation as the strongest surviving `9x9`
candidate.  The first stage is decisively tractable: only 2.04 million
canonical dense classes.  The straightforward second stage is decisively not
tractable on CPU, even for the most symmetric/densest first class.

The next experiment should implement a GPU `4+5`-row meet-in-the-middle join
for fixed `A`, beginning with the unique 29-edge class and the 14 28-edge
classes.  It must report complete `B` counts by cardinality and materialized
residual-signature uniqueness.  Continue toward all 2.04 million first
classes only if the measured pair/residual reuse projects below the existing
outer-corpus scale by at least two orders of magnitude.

## Reproduction

```text
python3 -m unittest -v tests.research.test_dense_colour_class_probe
python3 research/probes/dense_colour_class_probe.py degrees
make dense_c4free_pair_probe
./build/dense_c4free_pair_probe \
  --first 181,086,118,142,124,0c8,0b0,02b,055 \
  --minimum 18 --maximum 29 --state-cap 100000000
```
