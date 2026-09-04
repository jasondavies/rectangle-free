# Exact residual-hafnian solver for `T_4(6,29)`

Use the same 60 `(colour,row-pair)` tokens as the `6x30` endpoint. Every
six-entry column consumes at least two tokens. At width 29 the total excess
above that minimum, plus the number of unused tokens, is exactly two.

Let `D` be the collection of columns whose support has more than two tokens,
let `d=|D|`, and let

```text
e(D) = sum_{U in D} (|U|-2).
```

Only four defect sectors are possible:

| excess | defect columns | unmatched tokens | canonical queries |
|---:|---:|---:|---:|
| 0 | 0 | 2 | 1 |
| 1 | 1 | 1 | 2 |
| 2 | 1 | 0 | 1 |
| 2 | 2 | 0 | 25 |

The 83,071 exact occupied-token unions reduce under `S_6 x S_4` to 29
queries. Their defect coefficients already include column-support
multiplicities and orbit aggregation.

For one query, delete its occupied tokens from
`H=K_4 x KG(6,2)`. If `r=2-e(D)` tokens must remain unmatched, append `r`
labelled dummy vertices adjacent to every remaining original vertex and to no
other dummy. If `G_D^+` is this even augmented graph, then

```text
m_{29-d}(H-D) = pm(G_D^+) / r!.
```

The augmented orders are 62, 58, 56, and 54. Therefore

```text
T_4(6,29) = 29! * sum_D coefficient(D) * 2^(29-d) * pm(G_D^+) / r!.
```

The original campaign evaluated these 29 hafnians with nine 31-bit primes,
using the bound `T_4(6,29) < 2^272` and reconstructing the final colouring
count directly. Its v1 files remain readable by the historical reducer.

## Optimised production formulation

The zero-defect query need not use two dummy vertices. Instead, choose the
two unused tokens. Their unordered pairs have five `S_6 x S_4` orbits, with
multiplicities 90, 720, 240, 540 and 180 (sum 1,770). Thus

```text
m_29(H) = sum_{five orbits O} |O| * pm(H - representative(O)).
```

This replaces the single order-62 query with five order-58 queries. There
are now 33 queries: seven of order 58, one of order 56, and 25 of order 54.
The five new coefficients account for the choice of unused tokens, so no
dummy factorial remains for those queries. Their removed-token masks denote
monomers rather than defect supports.

Reconstruct each matching count independently before multiplying by its
coefficient, `2^(29-d)`, and `29!`. The shared exact degree bound gives powers
85 for the five new minors, 89 for the original one-monomer queries, and at
most 81 for the remaining queries. Three 31-bit primes suffice for every
query. Total work is 11,072,962,560 sign terms versus 30,802,968,576 originally.

`hafnian_residual_engine.cuh` shares the persistent workspace, Gray-chain
dispatch, fixed-field arithmetic, checkpoint writer and exact fallback with
6x28. Checkpoints use global Gray indices and a distinct v2 catalog; they
must never be combined with the historical binary-order v1 pieces. A fresh
verification campaign also requires one solver binary digest throughout.

## Build and validate

```bash
make six-by-twenty-nine-hafnian-test
make six-by-twenty-nine-optimized-test

make NVCCFLAGS='-O3 -std=c++17 -arch=sm_120 -lineinfo' \
  six_by_twenty_nine_hafnian_gpu
```

The optimised CPU evaluator is `build/six_by_twenty_nine_optimized_cpu`;
the original CPU target retains the historical catalog and binary ordering.
`tools/check_six_by_twenty_nine_optimized.py` checks all 33 queries under all
three primes at three ranges, including unaligned Gray boundaries and the
end of the term domain. Its `--historical` mode checks the complete result
and all 87 original query/prime combinations, including the weighted sum of
the five minors against the old dummy-augmented query.

## Run

```bash
python3 tools/run_six_by_twenty_nine_hafnian_gpu.py \
  --binary ./build/six_by_twenty_nine_hafnian_gpu \
  --gpus 0,1,2,3,4,5,6,7 \
  --output hafnian-6x29-results
```

The shared scheduler assigns largest jobs first and runs one persistent
process per GPU. Results are published after each 2^24-term chunk; restarting
resumes exact retained prefixes. `tools/reduce_six_by_twenty_nine_optimized.py`
authenticates each payload, validates metadata against the certified catalog,
rejects gaps/overlaps and mixed binaries, and requires complete per-query CRT
coverage before combining exact matching counts.
