# Exact GPU algorithm for \(T_4(7,9)\) and \(T_4(8,8)\)

This note describes the exact counting algorithm implemented by the current
7x9 and 8x8 GPU solvers.  It deliberately separates the mathematical
reduction from batching, caching, and CUDA optimisations.  The program computes
the single evaluation \(T_4(r,n)\), the number of labelled four-colourings of
an \(r\times n\) grid having no monochromatic axis-aligned rectangle; it does
not compute the full chromatic polynomial.

## 1. Split four colours into two binary coordinates

Write a colour as \((g,h)\in\{0,1\}^2\).  For a set of cells \(A\), let

\[
  C(A)=\#\{h:A\to\{0,1\}: h\text{ has no monochromatic rectangle wholly in }A\}.
\]

Fixing the first-bit mask \(G=\{x:g(x)=1\}\), any monochromatic four-colour
rectangle must lie wholly in either \(G\) or \(G^c\).  The choices of the
second bit on those two sets are independent.  Hence

\[
  \boxed{T_4(r,n)=\sum_{G\subseteq[r]\times[n]} C(G)C(G^c).}
\]

This identity is the main high-level reduction: a four-colour problem becomes
an outer sum over binary masks and two exact binary subproblems.

## 2. Reduce the outer sum by symmetry

Binary masks are canonicalised under row permutations, column permutations,
and global complementation.  For the square 8x8 geometry, matrix transposition
is included as an additional exact outer symmetry.  The corpus stores one
representative \(G\) and an
exact coefficient \(\alpha_G\), equal to the number of labelled masks covered
by that record.  Thus

\[
  T_4(r,n)=\sum_{[G]}\alpha_G C(G)C(G^c).
\]

The coefficient handles strict complement pairs, self-complementary midpoint
orbits, and ordinary row/column stabilisers; no division is performed by the
GPU solver.  As an independent completeness invariant, the coefficients cover
exactly \(2^{rn}\) labelled first-bit masks.

For 8x8, let \(t(G)\) be the row/column/complement-canonical representative of
\(G^{\mathsf T}\).  This is an involution on the complement-paired corpus and

\[
  C(G)C(G^c)=C(G^{\mathsf T})C((G^{\mathsf T})^c).
\]

The production corpus keeps \(G\) unchanged when \(G=t(G)\), keeps the smaller
of each nonfixed pair with twice its old coefficient, and discards the larger.
Because the kept key remains in its original left-owned shard, this exact
quotient needs no cross-shard shuffle even though almost all transpose partners
have a different owner.

All 35 vertical and 35 horizontal 4+4 column partitions are mathematically
valid, and the selected and complement contractions may choose their partitions
independently.  A measured performance-aware selector reduced representative
Cartesian work by about threefold and CUDA time by 35.8% on a matched one-million
record sample.  It also increased the number of resident left identities by
17.96x, however, so layout construction reduced the recurring end-to-end gain
to 14.2%.  A bounded vertical/horizontal portfolio was 12.6% slower overall.
Production therefore retains the fixed vertical split: stable left ownership
and reuse are more valuable than the available per-record join reduction.

Each representative is split vertically into \(G=[L\mid R]\): 4+5 columns for
7x9 and 4+4 columns for 8x8.  It becomes a weighted edge
\((L,R,\alpha_G)\).  Solve shards are owned by the left half, both to bound
memory and to reuse each constructed left layout across many edges.

## 3. Compute a half-grid distribution

Let

\[
  \Omega_r=\{0,1\}\times { [r]\choose 2},
  \qquad |\Omega_r|=2{r\choose2}.
\]

A token \((b,\{i,j\})\) records that, in one column, active rows \(i,j\)
both received inner bit \(b\).  The same token appearing in two columns is
exactly a monochromatic rectangle.

For a half-mask \(A\), define the sparse distribution

\[
  D_A(U)=\#\{\text{inner-bit assignments on }A
                 \text{ whose set of used tokens is exactly }U\},
  \quad U\subseteq\Omega_r,
\]

where only assignments with no repeated token are admitted.  It is built by a
column DP.  Starting with \(D(\varnothing)=1\), enumerate the inner assignments
of the active cells in the next column.  If such an assignment produces token
increment \(I\), apply

\[
  D'(U\cup I)\mathrel{+}=D(U)m(I)
  \quad\text{when }U\cap I=\varnothing,
\]

where \(m(I)\) combines assignments producing the same increment.  All
coefficients are exact integers.

For sparse distributions define the weighted disjointness contraction

\[
  J(D,E)=\sum_{U\cap V=\varnothing}D(U)E(V).
\]

No rectangle can cross the vertical split precisely when the token sets used
by its two halves are disjoint.  Therefore

\[
  C(G)=J(D_{G_L},D_{G_R}),\qquad
  C(G^c)=J(D_{G_L^c},D_{G_R^c}),
\]

and one outer edge contributes

\[
  \alpha_G\,J(D_{G_L},D_{G_R})
              J(D_{G_L^c},D_{G_R^c}).
\]

The final products and sum are accumulated as unsigned 128-bit integers.

## 4. Canonical half-distribution cache

Half-masks are canonicalised under row permutations and permutations of the
columns within that half.  Column order does not affect a token union, while a
row permutation merely relabels every row-pair token:

\[
  D_{\pi A}(\pi U)=D_A(U).
\]

Consequently a labelled half is represented by a canonical distribution ID
and a row permutation.  Its distribution is recovered by permuting support
masks; its weights are unchanged.  Complements are canonicalised into this
same cache rather than constructed independently.  This exact equivariance is
what makes reuse across billions of outer-mask records possible.

### Inner-colour token-plane quotient

Globally complementing every inner bit swaps the two token planes.  Write this
involution as (S).  Complementation is a weight-preserving bijection of
assignments, so every half-distribution obeys

\[
  D_A(U)=D_A(SU).
\]

The maintained production solvers therefore store one arbitrary representative of
each support orbit ([U]=\{U,SU\}), its per-mask weight, and its orbit size
(o_U\in\{1,2\}).  For independently chosen representatives the exact join is

\[
 J(D,E)=\sum_{[U],[V]} o_U D(U)E(V)
 \left([U\cap V=\varnothing]
       +[o_V=2][U\cap SV=\varnothing]\right).
\]

This asymmetric formula returns (J) directly, without first accumulating
(2J), so the established unsigned 64-bit per-join bound is unchanged.  Row
permutations commute with (S), and the prefix uses the same row-pair
coordinates in both planes, allowing prefix and suffix halves to be swapped
independently.  Almost every support orbit has size two: the complete 8x4
cache falls from 565,306,220 entries to 282,659,250, while the complete packed
7x5 cache falls from 4,740,574,641 entries to 2,370,316,739 (17.66 GiB).  The
complete known 7x7 regression uses the same quotient identity in its scalar
CUDA join.

## 5. Prefix-factorised weighted-disjointness join

The remaining contraction is still a Cartesian comparison of sparse support
sets.  To reject most pairs cheaply, choose some row-pair coordinates and split
each token mask as \(U=(p_U,s_U)\), using the same coordinates in both colour
planes.  Then

\[
  U\cap V=\varnothing
  \iff (p_U\mathbin{\&}p_V)=0
       \ \text{and}\ (s_U\mathbin{\&}s_V)=0.
\]

Entries are placed in sparse physical buckets by prefix \(p\).  Within a
bucket they are grouped by their equal distribution weight.  A weight class
stores one coefficient and a contiguous array of suffixes, so the join is

\[
  J(D,E)=
  \sum_{p\&q=0}\ \sum_{a,b}
  w_a w_b\,
  \#\{(s,t)\in S_{p,a}\times S_{q,b}:s\&t=0\}.
\]

The current prefix coordinates (with rows numbered from zero) are empirical
cost optima, not assumptions needed for correctness:

| Geometry | Token bits | Prefix row pairs | Prefix bits | Suffix bits |
|---|---:|---|---:|---:|
| 7x9 | 42 | \(01,02,03,12,13\) | 10 | 32 |
| 8x8 | 56 | \(01,02,03,12,13,23,67\) | 14 | 42 |

The first set is \(K_4\) minus edge 23; the second is \(K_4\sqcup K_2\).

### Boolean tensor-core primitive

For each compatible prefix-bucket pair and pair of weight classes, the CUDA
kernel tiles the suffix Cartesian product into 16x8 groups.  The Boolean MMA
instruction

```text
mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc
```

computes intersection popcounts for all 128 suffix pairs in a tile (unused
suffix coordinates are zero).  A zero result is exactly the disjointness
predicate.  The number of zero outputs is multiplied by the two exact class
weights.  Thus BMMA changes the constant factor of the Cartesian join but not
its mathematics or asymptotic form.

One logical selected/complement join is assigned to a CUDA block.  Warps take
prefix-bucket pairs dynamically, discard incompatible prefixes, process their
weight-class tiles, and reduce to one exact join value.  The implementation
checks all narrowed representations and promotes weights before multiplication.
For quotient layouts, a weight class additionally records the common support
orbit size.  The kernel evaluates the stored relative orientation and, for a
non-fixed right orbit, the swapped-right orientation, then multiplies by the
left orbit size exactly as in the formula above.  When both prefix
orientations are compatible, the two contractions share one tile traversal
and reuse the operand unchanged by token-plane exchange.

## 6. Geometry-specific execution plans

### 7x9: asymmetric 4+5 split

- The 7x4 distributions needed by one solve shard are built once and retained
  on the GPU as its labelled left layout.
- The universal 7x5 canonical cache contains 136,758 distributions and
  2,370,316,739 support-orbit representatives.  Its packed read-only
  representation is about 17.66 GiB.  Production may reconstruct it from the
  canonical orbit census or memory-map a versioned, checksummed artifact.  The
  artifact validates metadata eagerly and payload blocks on first use, so it
  is a warm-page-cache/local-NVMe optimization rather than a requirement.
- Every labelled right half is planned as canonical selected/complement IDs
  plus row maps.  Rights are grouped into source-aware batches so canonical
  support is gathered once where possible.
- CPU gathering and host-to-device transfer of the next right batch are
  overlapped with the current GPU join.  A persistent pinned staging area and
  persistent device buffers avoid repacking it per batch.  GPU kernels read
  only the resulting device-resident batch; they never issue lookups into the
  host cache.
- Full cache residency now fits on a 46-GiB L40S, but a matched production
  shard is 2.56% slower because it leaves too little recurring-layout
  headroom.  Production therefore chooses placement automatically: residency
  requires another 32 GiB plus the safety reserve after the cache allocation.
  This retains streaming on the L40S while using the otherwise idle capacity
  of 96-GB devices.

The asymmetry is important: the large five-column side is an immutable
canonical source, either resident or streamed according to device capacity;
the reused four-column side is always resident.

### 8x8: symmetric 4+4 split

- There are 25,207 canonical 8x4 half-masks.  Token-plane quotienting reduces
  their universal cache from 565,306,220 ordinary support entries to
  282,659,250 representatives; the cache is retained across solve shards in a
  long-lived worker.
- A shard-specific labelled left layout remains resident.  Right layouts are
  constructed in batches directly into prefix buckets and weight classes from
  canonical sources and row maps.
- Dense edge IDs, persistent high-water workspaces, heavy-first scheduling,
  and GPU histogram/scan/scatter construction reduce host work and allocation
  overhead.  They do not alter the contraction above.

The exact outer corpora currently contain 3,608,247,685 complement-paired
records for 7x9 (covering \(2^{63}\) first-bit masks after complement factors)
and 3,671,999,389 complement-and-transpose-paired records for 8x8 (covering
\(2^{64}\)).  The 8x8 transpose quotient contains 965,530 fixed records and
3,671,033,859 nonfixed representatives, down from 7,343,033,248 records.  The
intended solve layouts use 128 and 1,024 left-owned shards respectively.

## 7. End-to-end algorithm

```text
build once, or load, the canonical half-distribution cache

for each left-owned solve shard:
    read and validate weighted outer-orbit edges (L, R, alpha)
    canonicalise the required L and complement(L)
    build the resident labelled left layouts

    plan/group R values into right-source batches
    for each batch, with production overlapped with the previous GPU join:
        gather or construct labelled R and complement(R) layouts
        bucket by prefix and group suffixes by exact weight
        run two exact BMMA disjointness joins per edge,
            restoring token-plane orbit orientations
        accumulate alpha * selected_join * complement_join in uint128

reduce shard uint128 totals
verify record counts and total covered labelled weight
```

Each shard also reconstructs sample selected and complement joins with the
simple sparse DP and scalar Cartesian disjointness test.  Complete 7x7 and 7x8
runs have matched independent exact results; these checks guard the same cache,
permutation, packing, and GPU-join machinery used by 7x9 and 8x8.

## 8. Remaining mathematical bottleneck

After symmetry reduction and cache reuse, the dominant operation is still the
weighted suffix Cartesian product surviving prefix rejection.  BMMA evaluates
many predicates at once, but does not eliminate them.  The most valuable new
idea would therefore need to replace

\[
  \sum_{s\in A}\sum_{t\in B}[s\&t=0]w(s)w(t)
\]

by a substantially cheaper exact query, or contract many relative row
alignments/outer edges together without materialising a prohibitively large
representation.  Generic full subset transforms are too large at 32 or 42
suffix bits, and the trie/ZDD, blocked subset-sum, generic symmetry-block, and
bitset-BMMA variants tested so far did not beat the current prefix/BMMA join.
Pair-specific coordinate projection was also tested after the 8x8 token-plane
quotient.  Although it can nearly halve the projected BMMA tiles of selected
heavy class pairs, the useful work is spread across hundreds of thousands of
different projections; exact terminal OR/AND cases cover under 0.5% of tiles,
and dense subset transforms with a BMMA-competitive operation count cover only
about 0.05%.  Constructing a different projected weighted layout for each pair
therefore has no production path with a material end-to-end ceiling.
Exact behavioral minimization of the complete 8x4 cache is similarly small.
Normalizing sparse vectors by their weight GCD saves under 1% of production
cache entries.  Removing empty/singleton columns before row/column
canonicalization raises this to only 3.37%, and collapses under 1% of linked
selected/complement source pairs.  A complete stabilizer census found no
nonzero distribution symmetry beyond that structural core.  These semantic
quotients therefore do not justify another cache/layout representation.
This weighted-disjointness contraction is therefore the clean mathematical
target for further analysis.

The production entry points are
[`twocolour_7x9_packed_solve.cu`](../src/gpu/twocolour_7x9_packed_solve.cu) and
[`twocolour_8x8_prefix_solve.cu`](../src/gpu/twocolour_8x8_prefix_solve.cu); their shared
mathematical primitives live in
[`twocolour_gpu_common.cuh`](../src/gpu/twocolour_gpu_common.cuh),
[`twocolour_prefix_core.cuh`](../src/gpu/twocolour_prefix_core.cuh), and
[`twocolour_weight_class_join.cuh`](../src/gpu/twocolour_weight_class_join.cuh).
