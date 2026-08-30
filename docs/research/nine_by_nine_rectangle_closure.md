# Rectangle-closure lattice gate for nine-by-nine

## Exact formulation

Let the 81 grid cells be vertices and associate one equality partition to
each of the 1,296 rectangles by joining its four corners.  The join-closure of
these generators is a finite lattice `L`.  Rectangle inclusion-exclusion can
be grouped by the generated cell partition:

\[
T_4(r,c)=\sum_{\pi\in L}\mu_L(\hat 0,\pi)4^{|\pi|}.
\]

This is materially stronger than ordinary labelled deletion-contraction:
every set of rectangles generating the same equality partition is represented
once.  The lattice is invariant under row permutations, column permutations,
and transpose for square grids.

## Implementation and validation

The compiled probe stores a cell partition as a restricted-growth byte string.
Adding a rectangle merges its four block labels and normalises the result.
For the orbit census, an incidence graph with row, column, cell, and unlabeled
block vertices is canonicalised by nauty.  Giving row and column vertices one
colour class on square grids includes global transposition exactly.

Independent enumeration validates both components:

- canonicalising every labelled state produces the same orbit set as direct
  orbit-closure generation through `4x4`;
- labelled Möbius evaluation reproduces
  `T_4(2,3)=3912`, `T_4(3,3)=228984`, and
  `T_4(4,4)=2545607472`;
- all 5, 44, and 9,939 labelled lattice states respectively have nonzero
  Möbius coefficients.

## Census

| geometry | labelled closure states | closure orbits |
|---:|---:|---:|
| `2x3` | 5 | 3 |
| `3x3` | 44 | 6 |
| `4x4` | 9,939 | 58 |
| `5x5` | more than 1,710,496 after depth 4 | 3,350 complete |
| `6x6` | not attempted | more than 587,213, incomplete |

The complete orbit count grows by `57.8x` from four to five rows.  At six
rows, the partial count already gives a further factor above `175x`.
Generation through depth six reaches 367,613 orbits; a time-capped portion of
depth seven raises the lower bound to 587,213.  That run performs 17.33
million rectangle additions in 120.1s, with 9.95 million nauty calls and a
2.09 GiB peak resident set.  A cache of labelled children avoids 7.38 million
additional canonicalisations but cannot change the orbit census.

The observed growth is already much faster than the roughly constant factor
that would be required to reach nine rows.  Möbius cancellation does not
remove any state through `4x4`, and symmetry-equivalent states have identical
Möbius values rather than cancelling within an orbit.

## Verdict

Reject explicit enumeration of the rectangle-closure lattice as a route to
`9x9`.  It is a genuinely different and highly effective quotient—`4x4`
collapses from 9,939 labelled states to 58 orbits—but the quotient itself
fills super-exponentially with the row count.  A GPU sort/reduce engine could
complete the `6x6` census faster; it would not bridge the observed factors of
58 and at least 175 on the way to nine rows.

## Cycle-core follow-up

Peeling incidence-degree-zero/one cells from every rectangle hypergraph inside
an equality block removes all hypertree attachments and retains only a cyclic
2-core.  This further quotient is real but still grows too quickly: complete
`4x4` and `5x5` censuses contain 26 and 638 cumulative core orbits, while the
complete depth-seven `6x6` frontier already raises the cumulative lower bound
to 12,560.  The successive factors are `24.5x` and at least `19.7x`, before
retaining any of the attachment multiplicities required by an exact solver.
See Experiment 462 for the detailed depth census.

## Reproduction

The research target requires Ubuntu's `libnauty-dev` package.

```text
make rectangle_closure_lattice_probe
./build/rectangle_closure_lattice_probe --self-test
./build/rectangle_closure_lattice_probe \
  --rows 4 --columns 4 --labelled --mobius
./build/rectangle_closure_lattice_probe \
  --rows 5 --columns 5 --orbit
./build/rectangle_closure_lattice_probe \
  --rows 6 --columns 6 --orbit --time-cap 120
./build/rectangle_closure_lattice_probe \
  --rows 6 --columns 6 --orbit --cycle-core --time-cap 180
```
