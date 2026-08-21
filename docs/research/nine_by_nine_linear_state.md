# Nine-by-nine universal linear-state contraction

## Summary

Commuting the outer-mask sum through the two disjointness contractions is
mathematically correct.  For a binary half distribution `D_A`, define

\[
H_k(U,U')=\sum_A D_A(U)D_{A^c}(U').
\]

Then a split `n=a+b` gives

\[
T_4(r,n)=\sum_{U\cap V=\varnothing\atop U'\cap V'=\varnothing}
H_a(U,U')H_b(V,V').
\]

Equivalently, this is a four-colour transfer in the squarefree algebra on
colour/row-pair tokens.  The complement covariance recurrence

\[
M_{k+1}=\sum_{S\subseteq[r]}T_{k,S}M_kT_{k,S^c}^{\mathsf T}
\]

and final trace

\[
T_4(9,9)=\operatorname{tr}(M_4^{\mathsf T}K M_5K^{\mathsf T})
\]

also follow exactly, with `T` expressed in bases for the reachable spaces
`V_k` and `K` the restricted disjointness pairing.

The proposed *explicit linear-state compression*, however, fails its first
scaling gate.  The measurements reject explicit bases for the complete
reachable spaces and generic dense contractions on their symmetry
multiplicity blocks.  They do not rule out an implicit nonlinear
representation of the one universal aggregate state.

## Exact modular ranks

`reachable_distribution_rank_probe.py` represents each one-column `p_S` in
the exact squarefree token algebra and streams commutative products through
sparse Gaussian elimination.  A modular rank is a lower bound on rational
rank.  When it equals the symmetric-power dimension, it proves injectivity
over the rationals.

Let `R_1=2^r-r`.  Degree two is completely injective for every tested row
count through nine.  In particular, two independent primes give

\[
R_2(9)=\binom{503+1}{2}=126{,}756.
\]

Here `R_1(9)=503`.  The 131,328 raw unordered products are
`C(512+1,2)` before quotienting the nine degree-one relations, whereas
`C(503+1,2)=C(504,2)` is the symmetric-square ceiling after quotienting.

The measured degree-three ranks are:

| rows | symmetric-cube bound | measured modular rank | surviving |
|---:|---:|---:|---:|
| 4 | 364 | 304 | 83.516% |
| 5 | 3,654 | 3,424 | 93.706% |
| 6 | 34,220 | 33,535 | 97.998% |
| 7 | 302,621 | 300,849 | at least 99.414% |

Rows four through six agree modulo both 1,000,003 and 1,000,033.  The
seven-row run used 68,847,392 KiB peak RSS and 602.165 seconds.  Its modular
result alone proves that rational `R_3(7)` is at least 300,849; a bad prime
could only make the actual compression still weaker.

The shrinking deficiency is the opposite of what the proposed 9x9 method
needs.  It strongly indicates that `R_3(9)` is close to the full

\[
\dim \operatorname{Sym}^3(V_1)=21{,}337{,}260.
\]

## Row-symmetry forecast

As an `S_r` module,

\[
V_1\cong \mathbf 1\oplus\bigoplus_{s=2}^r \mathbf F[\{s\text{-subsets}\}],
\]

so its character is `2^cycles(g)-fixed_points(g)`.  The probe computes exact
symmetric-power characters and decomposes them with Murnaghan--Nakayama.
For `S_9`, before higher-degree relations, the relevant sizes are:

| degree | dimension | largest multiplicity | commutant entries |
|---:|---:|---:|---:|
| 1 | 503 | 9 | 186 |
| 2 | 126,756 | 261 | 295,493 |
| 3 | 21,337,260 | 20,235 | 2,135,876,320 |
| 4 | 2,699,163,390 | 1,864,762 | 23,066,444,763,935 |
| 5 | 273,695,167,746 | 165,455,238 | 213,959,296,751,779,386 |

The degree-four/degree-five cross space contains
2,184,725,045,552,167 multiplicity entries.  Thus block diagonalisation does
not by itself make `M_4`, `M_5`, or `K` tractable.  Degree-four and degree-five
relations would have to create reductions of several orders of magnitude,
despite degree-three compression already falling below 0.6% at only seven
rows.

## Verdict

The universal outer-sum identity is valid and may still inspire a nonlinear
tensor-network or decision-diagram method.  The particular proposal to build
explicit exact bases for `V_4` and `V_5`, generic dense multiplicity blocks,
and a basis-level `K` is not a credible 9x9 route.  It replaces the
ten-trillion-edge outer corpus with linear algebra whose state spaces are
already nearly maximal at degree three and whose projected degree-four/five
blocks are much too large.

Revisit this path only if a new theorem supplies a dramatic degree-four/five
quotient or proves additional structure in `M_k` and `K` that avoids explicit
multiplicity matrices.  Ordinary exact rank reduction and `S_9`
block-diagonalisation are insufficient.

## Reproduction

```text
python3 research/probes/reachable_distribution_rank_probe.py rank 9 2 --prime 1000003
python3 research/probes/reachable_distribution_rank_probe.py rank 9 2 --prime 1000033
python3 research/probes/reachable_distribution_rank_probe.py rank 7 3 --prime 1000003 --allow-large
python3 research/probes/reachable_distribution_rank_probe.py forecast 9 5
python3 -m unittest -v tests.research.test_reachable_distribution_rank_probe
```
