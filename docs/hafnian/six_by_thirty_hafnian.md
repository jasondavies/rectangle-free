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

## Algorithm

The production solver uses the exact Glynn power-trace formula. A supported
perfect matching partitions the 60 vertices into 30 reference pairs. Fixing
one sign leaves `2^29 = 536,870,912` independent terms. For each sign vector it
computes the leading characteristic-polynomial coefficients of `A X D` by:

1. finite-field similarity reduction to upper-Hessenberg form;
2. the La Budde characteristic-polynomial recurrence, truncated after degree
   30;
3. Newton identities for `tr((A X D)^k)`;
4. the coefficient of
   `exp(sum_k tr((A X D)^k) z^k / (2k))`.

All arithmetic is modulo an odd 31-bit prime. Ten primes near `2^31` provide a
310-bit CRT modulus, exceeding the exact bound `T_4(6,30) <= 60! < 2^273`.

The CPU implementation uses Barrett reduction and OpenMP. The CUDA version
uses 32-bit Montgomery arithmetic and one cooperative CTA per sign term. Its
60x60 matrix and truncated polynomial occupy approximately 22.5 KiB of shared
memory. Kernels operate on bounded chunks. After every chunk, the CUDA solver
atomically publishes the exact cumulative range completed so far; an interrupted
file can therefore be retained and the uncovered suffix launched separately.

## Build and validation

```bash
make six_by_thirty_hafnian
./six_by_thirty_hafnian --self-test
python3 -m unittest -v tests.hafnian.test_six_by_thirty_hafnian
```

Build CUDA on a target architecture, for example Blackwell:

```bash
nvcc -O3 -std=c++17 -arch=sm_120 -lineinfo \
  -o six_by_thirty_hafnian_gpu six_by_thirty_hafnian_gpu.cu
./six_by_thirty_hafnian_gpu --self-test
```

The self-tests verify the graph census and compare the formula with brute-force
perfect-matching counts on many small random graphs under two primes. During
development, CUDA ranges were also compared bit-for-bit with the independent
CPU implementation under multiple primes, ranges, launch widths, and chunk
sizes.

## Sharded execution

One CPU range:

```bash
./six_by_thirty_hafnian --run --prime 2147483647 \
  --begin 0 --end 1048576 --threads 16 --output results/p0-r0.result
```

One GPU range:

```bash
./six_by_thirty_hafnian_gpu --run --prime 2147483647 \
  --begin 0 --end 536870912 --threads 256 --chunk-terms 1048576 \
  --output results/p0.result
```

Ranges for each prime must form an exact, nonoverlapping cover of `[0,2^29)`.
The reducer checks geometry, algorithm version, payload digest, and coverage:

```bash
python3 tools/reduce_six_by_thirty_hafnian.py results/*.result
```

It first reconstructs the Glynn sum modulo each prime, converts it to the
perfect-matching and `T_4(6,30)` residues, then performs CRT. It prints
`exact=OK` only after the combined modulus exceeds `60!`.

The multi-GPU driver assigns primes dynamically and resumes from every
atomically published prefix:

```bash
python3 tools/run_six_by_thirty_hafnian_gpu.py \
  --binary ./six_by_thirty_hafnian_gpu \
  --gpus 0,1,2,3 --output hafnian-6x30-results
```

If a worker is interrupted, its last result records the exact covered endpoint.
The next invocation retains that segment and starts a new nonoverlapping suffix.

## Initial benchmark

On an RTX PRO 6000 Blackwell already occupied by another campaign, a 65,536
term exact sample took 0.0779 seconds (about 841,000 terms/s). The matching
16-thread CPU sample took 3.781 seconds. Both produced residue `1808785296`
modulo `2147483579`.

At that measured GPU rate, one complete prime takes about 10.6 minutes. Ten
primes require about 1.8 GPU-hours in total, or approximately 25--30 minutes on
four similar GPUs before CRT and independent validation.
