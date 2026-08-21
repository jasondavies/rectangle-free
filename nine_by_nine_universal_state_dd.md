# Nine-by-nine single-state decision-diagram gate

## Question

Experiment 381 rejected explicit bases for the complete reachable spaces, but
left open an implicit nonlinear representation of the one universal state.
For the one-column element

\[
P=\sum_{x\in[4]^r}e_{I(x)}
\]

in the squarefree algebra on colour/row-pair tokens, the proposed balanced
identity is

\[
T_4(9,9)=\tau(F_3\star F_3\star F_3),\qquad F_3=P^{\star3}.
\]

The gate asks whether `P`, `P^2`, and especially the single state `F_3` have a
compact exact decision-diagram representation even though their ambient
linear span is almost full.

## Exact representation

`universal_state_dd_probe.py` implements a reduced edge-weighted MDD modulo a
prime.  A reference is a normalized node plus a modular scalar, so scalar
multiples share structure.  Squarefree convolution is an exact memoized Apply
operation.  Two site representations were tested:

- `bundled`: one 16-valued site for the four colour tokens of each row pair;
- `colour-major`: 144 binary sites at nine rows, used here as an ordering and
  granularity baseline.

The pair-site orders were lexicographic, reverse lexicographic, and a balanced
recursive vertex split.  Hard node and Apply-operation caps turn explosive
growth into a controlled rejection rather than an OOM.

The coefficient sum after `k` convolutions is exactly `T_4(r,k)` modulo the
chosen prime.  Exhaustive tests for `(r,k)=(2,3),(3,2),(3,3)` agree in every
site mode and ordering.  Both primes 1,000,003 and 1,000,033 give identical
decision-diagram structure counts for the completed five-row `F_3` run.

## Scaling results

The best or representative completed states are:

| rows | state | representation | reachable nodes | maximum width | allocated nodes | seconds |
|---:|---:|:---|---:|---:|---:|---:|
| 4 | `P^3` | bundled lex | 4,323 | 2,078 | 8,005 | 0.177 |
| 4 | `P^3` | colour-major | 4,623 | 549 | 7,378 | 0.036 |
| 5 | `P^3` | bundled lex | 230,790 | 75,282 | 523,482 | 10.80 |
| 5 | `P^3` | bundled balanced | 304,833 | 112,658 | 863,915 | 16.54 |
| 5 | `P^3` | colour-major | 258,125 | 22,177 | 514,779 | 4.76 |
| 6 | `P^2` | bundled lex | 201,587 | 49,330 | 267,391 | 3.18 |
| 6 | `P^2` | colour-major | 259,415 | 12,288 | 298,115 | 1.47 |

Further gates fail a one-million allocated-node cap:

- every tested six-row `P^3` ordering, in 8.3--18.0 seconds;
- both seven-row `P^2` site representations, in 5.7--11.4 seconds;
- the actual balanced `P^3 \star P^3` construction at only four rows, in
  8.2--17.3 seconds.

The allocated count includes memoized intermediate nodes and is therefore not
the same as the live size of the final state.  Garbage collection and a
compiled implementation could raise the absolute cap.  The completed-state
numbers are the decisive trend: from four to five rows, the bundled `F_3`
grows by 53.4x in reachable nodes and 36.2x in maximum width.  Reordering
trades width against node count and runtime but does not create the many orders
of magnitude required to reach nine rows.

## Verdict

Reject a conventional ordered weighted ZDD/MDD for `F_3` as a credible 9x9
production representation.  It does exploit nonlinear sharing, but that
sharing collapses far too quickly: degree two is already beyond the cap at
seven rows, degree three beyond it at six rows, and the first balanced
three-block contraction beyond it at four rows.

This is deliberately narrower than rejecting all tensor networks.  MDD width
is an upper bound, not an exact tensor-train bond rank; residual MDD functions
could still have linear dependencies.  A second gate would need to measure
exact or certified-lower-bound separator ranks without materializing the full
coefficient tensor.  The DD result sets a high bar: such a method must compress
tens of thousands of distinct five-row residual functions and then scale
through four additional rows.

## Reproduction

```text
python3 -m unittest -v test_universal_state_dd_probe.py
python3 universal_state_dd_probe.py 5 3 --order lex --mode bundled
python3 universal_state_dd_probe.py 5 3 --order lex --mode colour-major
python3 universal_state_dd_probe.py 6 3 --order lex --mode bundled
python3 universal_state_dd_probe.py 7 2 --order lex --mode bundled
python3 universal_state_dd_probe.py 4 9 --strategy three-block
```
