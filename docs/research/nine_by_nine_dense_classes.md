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

The balanced row split produces two sparse families of 36-bit pair-resource
masks.  Their exact weighted-disjointness join is directly suitable for the
existing GPU machinery.

## GPU meet-in-the-middle result

On one RTX PRO 6000, the exact `4+5`-row join for the unique 29-cell `A`
returns

\[
22{,}708{,}949{,}741{,}198
\]

labelled `B` classes of sizes 18--29.  It screens 26.638 trillion half-state
pairs in 8.393 GPU seconds, or 3.174 trillion pairs/s.  Dividing by the full
automorphism-group order 24 still leaves at least about 946 billion complete
second-class orbits.  The actual colour-preserving stabilizer can only be
smaller.

The fourteen 28-cell first-class orbits are no better:

```text
labelled B per A:  29,948,575,180,670 .. 34,898,523,900,958
mean:              32,634,520,100,208.5
sum over 14 A:    456,883,281,402,919
GPU comparisons: 582,054,016,767,646
GPU join time:                    178.842 s
aggregate rate:                    3.255 T/s
```

An independent monolithic CPU DP and the GPU meet-in-the-middle join agree in
every cardinality bin from zero through eight for the 29-cell class, including
the exact total `876,001,660`.

Residual infeasibility also fails as a possible filter.  A stratified,
nonuniform sample of 100 exact `B` classes at every size 18 through 29 gives
1,200/1,200 residuals with an exact binary completion.  The residual
NAE-rectangle solver needs only about 22 search nodes on average.  This is not
a statistical estimate for a uniform orbit sample, but it decisively rules
out the extraordinary near-zero completion rate required to offset trillions
of second classes.

## Verdict and next gate

Reject explicit enumeration of the two densest classes.  The first stage is
decisively tractable, but one first-class representative already expands to
tens of trillions of admissible labelled second classes, and the residual is
usually easy rather than impossible.

The dense-first identity remains useful only if all three remaining colours
are contracted together for each canonical `A`, without materializing `B`.
That is a direct three-colour completion operator on the complement of `A`,
not the proposed `A,B` corpus followed by binary completions.  It needs a new
compressed transfer or tensor representation; another faster disjointness
kernel cannot remove the measured combinatorial expansion.

## Reproduction

```text
python3 -m unittest -v tests.research.test_dense_colour_class_probe
python3 research/probes/dense_colour_class_probe.py degrees
make dense_c4free_pair_probe
make dense_c4free_mitm_probe
./build/dense_c4free_pair_probe \
  --first 181,086,118,142,124,0c8,0b0,02b,055 \
  --minimum 18 --maximum 29 --state-cap 100000000
python3 research/probes/dense_residual_completion_sample.py \
  --first 181,086,118,142,124,0c8,0b0,02b,055 \
  --samples-per-size 100 --minimum 18 --maximum 29
```
