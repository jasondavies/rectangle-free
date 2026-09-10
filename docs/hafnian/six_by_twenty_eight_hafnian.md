# Exact `T_4(6,28)` defect-hafnian campaign

The latest completed verification uses [shared-core contractions](common_core_campaign.md)
and [mixed whole-group/sign-range queues](mixed_campaign.md): **4.13 timed
GPU-hours on four RTX PRO 6000 GPUs**, **65 min solving / 72 min including
setup, validation and cleanup**. The full 36,398-query reduction matches the recorded result
(Experiment 500). Use the mixed workflow for this optimized campaign path.

The mathematics below also underlies the original independent-query solver.
Its commands and kernel timings are retained as a separate implementation,
not as the current shared-core campaign recipe. Their checkpoint formats and
sign summands are not interchangeable.

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

Every term contains `28! * 2^24`, so the independent-query solver reconstructs

```text
Q = T_4(6,28) / (28! * 2^24)
```

before restoring the common factor.  Exact per-query degree bounds require
three 31-bit primes for 36,395 queries and four primes for only three queries.

## Independent-query implementation

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
covered prefixes.  The production default is `2^24` terms per checkpoint:
one publication for the dominant order-48 queries and approximately 10--15
seconds of maximum recomputation for an interrupted order-64 query on the
measured RTX PRO 6000.  Override it with `--chunk-terms` when a provider has
different interruption or filesystem characteristics.  The reducer can also
be invoked directly:

```sh
python3 tools/reduce_six_by_twenty_eight_hafnian.py \
  --directory /path/to/results
```

Orders 48--58 use exact Gray-code rank-two updates and a fraction-free
generalized-Lanczos rebuild.  One batch inversion and warp product scans
replace one inversion per basis column; a whole checkpoint falls back to the
independent Gray-order Hessenberg kernel if any chain is non-cyclic.  Orders
60 and 64 always use that independent kernel.  On measured Ada 8.9 and
Blackwell 12.x GPUs, full-rank order-48--52 work instead uses an eight-term
hybrid resolvent chain: one tridiagonalization, two four-term truncated
resolvent determinants, and one structured rank-eight refresh evaluate all
eight terms.  Deficient, unaligned, larger-order, and other architectures keep
the established Gray backend.  Ada uses a separately measured 14-CTA launch
bound; Blackwell retains its order- and arithmetic-specific bounds.  Dense dot
products share one exact modular reduction across two or four residue
products. The historical independent-query workload projection was approximately
15 GPU-hours on one RTX PRO 6000, excluding interruptions and final independent
validation. It is not the cost of the completed shared-core verification above.

The independent finite-field Glynn/trace/Hessenberg fallback is shared with
the 6x29 and 6x30 solvers.  The Gray-chain core and the geometry-specific
defect catalog, coefficients, and exact final reduction are specific to 6x28.
