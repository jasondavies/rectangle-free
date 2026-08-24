# Exact `T_4(6,28)` defect-hafnian campaign

The 60 row-pair/colour tokens form the graph
`H = K4 x KG(6,2)`.  Relative to the saturated 30-column endpoint, a
28-column colouring has slack four.  Enumerate a pairwise-disjoint collection
`D` of non-size-two column supports with total excess at most four.  If `d` is
its size and `e` its excess, the remaining contribution is

```text
coefficient(D) * 2^(28-d) * m_(28-d)(H-D).
```

Canonical orbit propagation under `S6 x S4` reduces all defect collections to
36,398 residual queries.  A matching with `4-e` unmatched vertices is evaluated
as an augmented even-order hafnian; the GPU orders range from 48 to 64.

Every term contains `28! * 2^24`, so production reconstructs

```text
Q = T_4(6,28) / (28! * 2^24)
```

before restoring the common factor.  Exact per-query degree bounds require
three 31-bit primes for 36,395 queries and four primes for only three queries.

Build and test the maintained CPU components with:

```sh
make six-by-twenty-eight-hafnian-test
```

Build the CUDA worker for the target architecture, for example:

```sh
make six_by_twenty_eight_hafnian_gpu \
  NVCCFLAGS='-O3 -arch=sm_120 -std=c++17 -lineinfo'
```

Run or resume a multi-GPU campaign with:

```sh
python3 tools/run_six_by_twenty_eight_hafnian_gpu.py \
  --binary ./build/six_by_twenty_eight_hafnian_gpu \
  --output /path/to/results --gpus 0,1,2,3,4,5,6,7
```

The driver constructs one persistent task list per GPU, schedules expensive
queries first, reuses device allocations, and writes an authenticated range
checkpoint after every chunk.  Re-running the command resumes the exact
covered prefixes.  The reducer can also be invoked directly:

```sh
python3 tools/reduce_six_by_twenty_eight_hafnian.py \
  --directory /path/to/results
```

The production kernel always uses a conflict-free `N+1` shared-memory matrix
stride and occupancy-derived complete CTA waves.  On one RTX PRO 6000, the
exact workload projects to approximately 102 GPU-hours, excluding campaign
interruptions and final independent validation.

The finite-field Glynn/trace/Hessenberg kernel and launch policy are shared
with the 6x29 and 6x30 solvers.  Only the geometry-specific defect catalogs,
coefficients, and exact final reductions remain separate.
