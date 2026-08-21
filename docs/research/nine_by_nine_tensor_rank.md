# Nine-by-nine universal-state tensor-rank gate

## Question

The balanced universal identity

\[
T_4(9,9)=\tau(F_3\star F_3\star F_3),\qquad F_3=P^{\star3},
\]

removes the outer-mask corpus completely.  Experiment 382 rejected a
conventional weighted decision diagram for `F3`, but its diagram width is only
an upper bound on tensor bond dimension.  Distinct residual functions could
still be linearly dependent.  This gate measures that dependence directly.

## Certified rank method

Build the exact reduced weighted MDD for `F3` modulo a prime.  At a cut after
site `k`, each reachable MDD node represents a residual coefficient function
on the remaining sites.  Prefix assignments reaching different normalized
nodes have disjoint support, so the flattening rank is exactly the rank of
these residual functions.

The probe selects deterministic residual nodes and accepted suffix assignments,
evaluates their literal coefficient submatrix, and performs modular Gaussian
elimination.  A rank `r` modulo a prime certifies rank at least `r` over the
rationals.  The result is exact when every residual and suffix assignment is
included.  Two primes, 1,000,003 and 1,000,033, produced identical bounds in
the principal bundled experiments.

This measures the tensor of the single aggregate state.  It does not construct
the ambient reachable vector space rejected by Experiment 381.

## Results

For 16-valued row-pair sites in the balanced recursive vertex ordering, the
natural separator places the internal edges of the two vertex parts on one
side and their cross edges on the other:

| rows | vertex split | sites at cut | residuals | certified rank |
|---:|:---:|:---:|---:|---:|
| 3 | `1+2` | `1 / 2` | 15 | 15 exact |
| 4 | `2+2` | `2 / 4` | 219 | at least 219 |
| 5 | `2+3` | `4 / 6` | 7,869 | at least 2,759 |

The four-row lower bound uses all 219 residual rows and 1,024 suffix columns;
its full row rank is already sufficient for the bound.  The five-row bound is
the rank of a deterministic `4096 x 4096` submatrix.  It is not an estimate of
the complete rank, which can be as large as 7,869.

Other cuts and orders do not expose a small tensor-train route:

- the balanced bundled ordering has rank at least 911 at the `3 / 3`
  four-row cut and at least 1,022 at a `7 / 3` five-row cut;
- the lexicographic five-row ordering reaches rank at least 1,008 by its
  `3 / 7` cut and at least 1,021 at `6 / 4`;
- splitting binary sites by colour gives exact midpoint ranks 4 and 114 at
  three and four rows, then rank at least 885 at the five-row `20 / 20` cut.

The completed `F3` representation itself grows from 4,258 reachable nodes at
four rows to 304,833 at five rows in the balanced bundled ordering.  It already
exceeds the one-million construction cap at six rows, so a six- through
nine-row rank sequence cannot be obtained by this representation.

## Verdict

Reject an ordinary exact tensor train or hierarchical tensor network over the
proposed row-pair or colour-major site trees as a credible route to `9x9`.
The single invariant state does have linear dependence, but its natural bond
ranks are already in the thousands at only five rows and are growing alongside
the decision-diagram explosion.  Reaching nine rows needs many orders of
magnitude of additional structure, not a better implementation of these
factorizations.

This verdict is deliberately narrow.  It does not prove that every possible
symmetry-adapted circuit for the universal identity is large.  A future method
would need a new quotient or analytic contraction that avoids these certified
separator ranks, rather than a different site ordering alone.

## Reproduction

```text
python3 -m unittest -v tests.research.test_universal_state_tensor_rank_probe
python3 research/probes/universal_state_tensor_rank_probe.py 4 \
  --order balanced --rank-cap 1024 --levels 2,3
python3 research/probes/universal_state_tensor_rank_probe.py 5 \
  --order balanced --rank-cap 4096 --levels 4
python3 research/probes/universal_state_tensor_rank_probe.py 5 \
  --order lex --rank-cap 1024
python3 research/probes/universal_state_tensor_rank_probe.py 5 \
  --order balanced --mode colour-major --rank-cap 2048 --levels 20
```
