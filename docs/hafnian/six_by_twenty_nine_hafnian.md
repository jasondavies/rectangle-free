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

The shared CUDA core evaluates each perfect-matching count with the exact
Glynn power-trace hafnian formula. Its compile-time graph-order
specializations share one implementation and use finite-field Montgomery
arithmetic. Bounding every residual matching count by the corresponding
complete-graph matching count proves `T_4(6,29) < 2^272`. Nine 31-bit primes
give a 279-bit CRT modulus and therefore determine the integer uniquely.

## Build and validate

```bash
make six-by-twenty-nine-hafnian-test

nvcc -O3 -std=c++17 -arch=sm_120 -lineinfo \
  -o six_by_twenty_nine_hafnian_gpu six_by_twenty_nine_hafnian_gpu.cu
```

The CPU and GPU range evaluators must agree for representatives of all four
graph orders. The original `6x30` CUDA fixture also tests the shared kernel at
order 60.

## Run

```bash
python3 tools/run_six_by_twenty_nine_hafnian_gpu.py \
  --binary ./six_by_twenty_nine_hafnian_gpu \
  --gpus 0,1,2,3,4,5,6,7 \
  --output hafnian-6x29-results
```

Jobs are scheduled largest-first and checkpoint exact cumulative sign ranges
after every bounded chunk. Re-running the driver resumes retained prefixes.
The reducer authenticates every payload, requires exact nonoverlapping
coverage of all 29 queries for each prime, checks the defect-sector census,
and performs CRT only after complete modular reduction.
