# Full-symmetry universal-state gate for nine-by-nine

## Why this is distinct

The linear-space, ordered-DD, and tensor-rank gates all retained labelled token
coordinates.  The universal state

\[
F_k=P^{\star k}
\]

is invariant under simultaneous row permutations and permutations of the four
direct colours.  Quotienting its support under `S_r x S_4` can therefore merge
states that no site ordering in an MDD or tensor train can identify.

Represent a token state as four graph masks on the same `r` row vertices.
Rows act simultaneously on all four graphs and `S_4` permutes the graph
planes.  Store one canonical representative and the total coefficient mass of
its orbit.

The one-column transfer remains exact without expanding an orbit.  For a
representative `s`, count transitions into each target orbit.  Since the
one-column element is invariant, every member of the source orbit has exactly
the same target-orbit transition counts.  Multiplying by the source orbit's
total mass therefore produces exact target orbit totals.

This is an invariant-subalgebra computation, not post-hoc canonicalization of
an ordered decision diagram.  Relative row and colour alignments reappear as
orbit-algebra transition multiplicities, so the quotient does not make the
eventual contraction free.

## Initial exact transfer census

The reference probe gives:

| rows | state | exact `S_r x S_4` support orbits |
|---:|:---:|---:|
| 3 | `P^3` | 18 |
| 3 | `P^6` | 74 |
| 3 | `P^9` | 87 |
| 4 | `P^3` | 190 |
| 4 | `P^6` | 15,945 |
| 4 | `P^9` | 34,744 |
| 5 | `P^3` | 2,679 |
| 5 | `P^4` | 84,349 |

This is materially better than the ordered representation.  In particular,
the four-row quotient constructs `P^6` with 15,945 states, whereas the prior
MDD construction of `P^3 star P^3` exceeded a million allocated nodes.

The aggregate coefficient sums reproduce the independent direct counts

\[
T_4(3,3)=228984,
\qquad
T_4(4,4)=2545607472.
\]

## Nine-row three-column upper bound

A second exact census avoids token-state construction.  A row of a
three-column block is one of 64 colour words.  Rectangle freedom says that two
selected words may agree in the same colour in at most one coordinate.  After
row quotienting, a grid is therefore a compatible subset of the 64 words.

Burnside quotienting these subsets by global `S_4` colour permutations and
the internal `S_3` permutation of the three block columns gives:

| rows | compatible row sets | colour/column orbits |
|---:|---:|---:|
| 4 | 513,360 | 3,826 |
| 5 | 5,365,548 | 37,987 |
| 6 | 44,516,718 | 311,650 |
| 7 | 301,709,268 | 2,101,024 |
| 8 | 1,705,559,544 | 11,860,157 |
| 9 | 8,169,452,832 | 56,763,377 |

Different word-set orbits can yield the same union token support, so the last
column is an upper bound on `F3` support orbits.  It shows that the nine-row
three-column quotient may be large but is not automatically impossible: tens
of millions of compact records are a realistic storage scale.

## Current verdict and next gate

Keep this route open.  It is the first tested representation to rescue a
contraction that the ordered MDD could not construct.  The existing Python
implementation is only a correctness reference; its factorial canonicalizer
is not intended for rows nine.

The decisive next experiment is a compiled canonical-orbit transfer:

1. complete `P^5` and `P^6` at five rows;
2. measure `P^3` and, if feasible, `P^4` at six rows;
3. introduce a coloured-graph canonicalizer and stabilizer data rather than
   enumerating all row permutations;
4. extrapolate both state count and transition count before attempting the
   nine-row `F3` cache;
5. proceed only if the quotient of `F6`, not merely `F3`, remains manageable.

The final condition matters because `T_4(9,9)=tau(F_6 star F_3)`.  A compact
`F3` orbit table alone is insufficient if multiplication fills an enormous
invariant algebra.

## Reproduction

```text
python3 -m unittest -v tests.research.test_universal_state_symmetry_probe
python3 research/probes/universal_state_symmetry_probe.py burnside \
  --maximum-rows 9
python3 research/probes/universal_state_symmetry_probe.py transfer 4 9 \
  --max-states 100000
python3 research/probes/universal_state_symmetry_probe.py transfer 5 4 \
  --max-states 200000
```
