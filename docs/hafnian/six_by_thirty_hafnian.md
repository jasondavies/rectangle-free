# Exact hafnian solver for `T_4(6,30)`

For six rows, associate one token with every `(colour, row-pair)`. There are
`4 * C(6,2) = 60` tokens. At width 30 every column must consume exactly two
tokens, and all tokens must be consumed exactly once. The two tokens in a
column have different colours and disjoint row-pairs. Hence they are the edges
of

```text
H = K_4 x KG(6,2),  |V(H)| = 60, degree(H) = 18, |E(H)| = 540.
```

Every edge represents two physical columns, according to the order of the two
remaining singleton colours. Therefore

```text
T_4(6,30) = 30! * 2^30 * pm(H) = 30! * 2^30 * haf(A_H).
```

## Optimised production algorithm

Expand a perfect matching at a fixed vertex. Its 18 neighbours are equivalent
under row and colour permutations, so for any edge uv,

```text
pm(H) = 18 * pm(H - {u,v})
T_4(6,30) = 30! * 2^30 * 18 * haf(A_minor).
```

The minor has 58 vertices and needs `2^28` sign terms per prime. It uses the
same persistent Gray-chain, fixed-field and exact fallback engine as 6x29 and
6x28; no new order-60 Gray kernel is required.

The certified degree bound gives `pm(A_minor) <= 2^85`. Reconstruct that
small integer first using primes 2147483647, 2147483629 and 2147483587, then
multiply by 18, `2^30` and `30!` in arbitrary precision. Total work is
805,306,368 sign terms, versus 5,368,709,120 in the original ten-prime solver.

## Build and validation

```bash
make six-by-thirty-hafnian-test six-by-thirty-optimized-test
make NVCCFLAGS='-O3 -std=c++17 -arch=sm_120 -lineinfo' six_by_thirty_hafnian_gpu
./build/six_by_thirty_hafnian_gpu --list
```

Use `-arch=sm_89` for Ada. The host tests cover the Laplace identity on a
smaller token graph, certified CRT, checkpoint provenance, and exact ranges
against the independently implemented 6x29 minor evaluator.

Before a GPU campaign, compare short, unaligned and final term ranges against:

```bash
./build/six_by_thirty_optimized_cpu --query 0 --prime 2147483647 \
  --begin 12345 --end 12473 --threads 1
./build/six_by_thirty_hafnian_gpu --run --query 0 --prime 2147483647 \
  --begin 12345 --end 12473 --output gate.result
```

Both use global Gray indices. Repeat for the other two primes and compare
the CPU residue with `partial_glynn_sum`. The optional
`six_by_thirty_hafnian_gpu_control` Make target uses the independent
runtime-Montgomery kernel on the same minor for matched A/B timings.

## Resumable execution

```bash
python3 tools/run_six_by_thirty_hafnian_gpu.py \
  --binary ./build/six_by_thirty_hafnian_gpu \
  --gpus 0 --output hafnian-6x30-v2-results
python3 tools/reduce_six_by_thirty_optimized.py \
  --directory hafnian-6x30-v2-results
```

There are three prime jobs, so the current runner can use at most three GPUs
(`--gpus 0,1,2`). Each prime must cover `[0,2^28)` exactly. Results bind the
catalog, query, arithmetic backend, binary, term range and payload checksum.
The shared runner resumes from completed prefixes; fresh validation of a new
binary requires a separate result directory.

For manual sharding, use `--query 0 --begin B --end E` and distinct output
paths. Do not use overlapping ranges or mix these results with v1 files.

## Existing independent result and reuse

The edge minor is exactly query 3 of the optimised 6x29 catalog. Its count is

```text
pm(A_minor) = 1133887175503385561722350.
```

Saved, validated 6x29 residues reproduce the historical endpoint answer:

```bash
python3 tools/check_six_by_thirty_optimized.py --from-6x29 PATH_TO_6X29_RESULTS
```

This derives a separate identity check without rewriting source provenance or
pretending a new GPU campaign ran. Those three prime images took 56.055761
GPU-seconds in the 6x29 campaign. That suggests roughly one GPU-minute of
solve work on comparable hardware, excluding setup, but a standalone v2
campaign has not yet been timed.

The original independent 60-vertex CPU solver remains
`six_by_thirty_hafnian`. Its binary-order ranges cover `[0,2^29)`, and its
v1 result files use `tools/reduce_six_by_thirty_hafnian.py`. They are not
range-comparable with the new minor. The historical full GPU campaign used
0.79 GPU-hours; it predates the optimisations above.
