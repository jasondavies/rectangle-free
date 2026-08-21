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
| 5 | `P^5` | 1,807,714 |
| 5 | `P^6` | 21,037,687 |
| 6 | `P^3` | 50,497 |
| 6 | `P^4` | 8,863,353 |

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

## Compiled contraction gate

The compiled exact transfer completes the previously unknown gates on one
Ryzen 7 9700X core:

| target | source/support tests in last step | last-step seconds | peak RSS |
|:---:|---:|---:|---:|
| `P^5`, five rows | 34,076,996 | 17.8 | 0.62 GiB |
| `P^6`, five rows | 730,316,456 | 307.8 | 2.49 GiB |
| `P^3`, six rows | 764,208 | 2.5 | 0.10 GiB |
| `P^4`, six rows | 105,437,736 | 309.3 | 1.55 GiB |

The coefficient sums agree with the recorded exact values.  In particular,
the six-row `P^4` sum is `79102304162784`, equal by transposition to the
four-row `P^6` result, and the five-row `P^6` sum is
`140221383170146560`.

The state growth, rather than the brute-force canonicalizer, closes the direct
route:

- `P^3` grows by `10.6x`, `14.1x`, and `18.8x` as rows increase from three
  through six;
- `P^4` grows from 1,182 to 84,349 to 8,863,353 at rows four through six;
- `P^6` grows from 15,945 to 21,037,687 when moving only from four to five
  rows, a factor of about 1,319.

At six rows, advancing the 8.86-million-state `P^4` table by one column would
already require about 18.5 billion source/support tests before constructing
`P^6`.  There is no credible extrapolation from these figures to a nine-row
explicit `F6` orbit table.

## Verdict

Reject explicit closure of the full `S_r x S_4` invariant support algebra as
a production route to `9x9`.  The quotient is real and much stronger than an
ordered site representation: it rescues the four-row contraction and may make
a nine-row `F3` table itself storable.  It nevertheless fills far too quickly
when multiplied toward `F6`.

This does not erase the symmetry result.  A future algorithm could still use
canonical `F3` states and stabilizers inside an implicit contraction, but it
must avoid materializing `F4`, `F5`, or `F6` orbit states.  Merely replacing
the row-permutation loop with a faster graph canonicalizer changes the
five-minute timing, not the state-count obstruction.

## Reproduction

```text
python3 -m unittest -v tests.research.test_universal_state_symmetry_probe
python3 research/probes/universal_state_symmetry_probe.py burnside \
  --maximum-rows 9
python3 research/probes/universal_state_symmetry_probe.py transfer 4 9 \
  --max-states 100000
python3 research/probes/universal_state_symmetry_probe.py transfer 5 4 \
  --max-states 200000
make universal_state_symmetry_probe
./build/universal_state_symmetry_probe --self-test
./build/universal_state_symmetry_probe 5 6 --max-states 50000000 \
  --max-cache 20000000 --max-transitions 1000000000
./build/universal_state_symmetry_probe 6 4 --max-states 30000000 \
  --max-cache 20000000 --max-transitions 200000000
```
