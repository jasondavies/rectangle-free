# Exact model-counting gate for rectangle-free grids

This is a bounded CPU research probe, not a production solver or a new 9×9
runtime estimate. `grid_model_count_probe.py` emits CNF and runs external
counters with wall-time, CPU-time and address-space limits. A timeout or
unrecognized output never becomes a numerical result.

**Measured outcome (Experiment 501):** both counters and both encodings pass
known counts through 4×4, but all twelve 5×5/6×6/7×7 cases time out at 60 seconds.
An isolated 5×5 run with a four-times-larger cache also times out at 180 seconds.
Do not scale or GPU-port this baseline on this evidence. See
[the experiment log](../../docs/experiments.md)
for limits, measurements and the scope of this negative result.

## Exact encodings

The two-bit encoding has two Boolean variables per cell. For each rectangle
and each of four colours, one eight-literal clause forbids all four cells
having that colour. Thus 9×9 has 162 variables and 5,184 clauses.

The one-hot control has four variables per cell, one exactly-one constraint
per cell (one positive clause and six binary exclusions), and one four-literal
exclusion per rectangle/colour. Thus 9×9 has 324 variables and 5,751 clauses.
Both encodings are parsimonious: there are no auxiliary assignments or
projection factors. They count labelled rows, columns and colours.

`--anchor` fixes the first cell to colour zero. Global colour symmetry makes
the four choices for that cell equinumerous, so the runner multiplies the
answer by four. It does **not** divide by 24 or quotient row/column symmetries.
The exhaustive tests check the restore factor and both encodings separately.

## Counters and exact cache keys

Experiment 501 uses these upstream repositories and pinned commits:

- [SharpSAT-TD](https://github.com/Laakeri/sharpsat-td),
  `0c234c11b77115bad89484e80e1d2fc95dd42317`.
- [Original sharpSAT](https://github.com/marcthurley/sharpSAT),
  `edfbde3424ce17d72d7f8d8f5b8681f2247f4932`.

SharpSAT-TD's upstream `CacheableComponent::equals` compares only a 128-bit
hash. Apply `sharpsat_td_exact_cache.patch` before building: it copies the full
component variable/clause key, compares it after the hash, and includes the
key storage in the cache accounting. Hash collisions then cause extra
comparisons, not count reuse. The original sharpSAT control already compares
packed component contents. This is not a formal audit of either whole solver;
small known-grid checks remain necessary. Use the unweighted arbitrary-integer
mode, not weighted double-precision counting.

Example setup with system GMP/GMPXX/MPFR development libraries:

```sh
git clone https://github.com/Laakeri/sharpsat-td.git build/sharpsat-td
git -C build/sharpsat-td checkout 0c234c11b77115bad89484e80e1d2fc95dd42317
git -C build/sharpsat-td apply ../../research/probes/sharpsat_td_exact_cache.patch
cmake -S build/sharpsat-td -B build/sharpsat-td/out \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build/sharpsat-td/out -j4
```

Build from source, not the repository's prebuilt binaries. The experiment
installed development headers/libraries only under ignored
`build/review-501/deps/`; no root installation was required. The old sharpSAT
source needs `-include cstdint` with the compiler used here (GCC 13.3).

The optional forced-collision test targets the patched external header:

```sh
g++ -O2 -std=c++11 -Ibuild/sharpsat-td/src \
  tests/sharpsat_td_cache_key_test.cpp \
  build/sharpsat-td/out/CMakeFiles/sharpSAT.dir/src/clhash/clhash.c.o \
  -o build/sharpsat-cache-key-test
build/sharpsat-cache-key-test
python3 tests/grid_model_count_probe_test.py
```

## Reproduce the bounded gate

```sh
python3 research/probes/grid_model_count_probe.py bench \
  --binary build/sharpsat-td/out/sharpSAT --solver td \
  --shapes 1x1 2x2 2x3 3x3 3x4 4x4 \
  --td-seconds 0.05 --seconds 20 --output build/model-count-small

python3 research/probes/grid_model_count_probe.py bench \
  --binary build/sharpsat-td/out/sharpSAT --solver td --anchor \
  --shapes 5x5 6x6 7x7 --td-seconds 1 \
  --seconds 60 --memory-gib 4 --cache-mib 1024 \
  --output build/model-count-bits
```

Repeat with `--encoding onehot` and a fresh output directory. For original
sharpSAT use `--solver original` and its corresponding binary. A sub-0.099
second TD budget uses upstream's trivial decomposition; the scaling gate
uses an actual one-second FlowCutter search. These are different test modes.

Each case retains the CNF, raw counter log and a JSON report with input/binary
SHA-256, invocation, limits, exit status, wall/CPU time, peak RSS and any completed
count. Completed counts are compared with `results.txt`; a mismatch aborts the
remaining suite. The output directory must be new. Experiment artifacts remain
under ignored `build/review-501/`; third-party source is not vendored.

## Structural caution

For at least two rows and columns, any two cells occur together in some
rectangle. In the two-bit encoding, its clause therefore connects their bits
in the primal graph (variables adjacent when they share a clause). After the
first-cell anchor is substituted, the remaining `2rn-2` variables still form
a clique, of treewidth `2rn-3`. Tests explicitly verify the complete edge set
through 9×9; at 5×5, 6×6 and 7×7 this gives 47, 69 and 95.

That obstructs a small *static* tree decomposition of this encoding. It does
not prove a lower bound for dynamic component caching, other encodings,
semantic compression or model counting in general. A bounded timeout also
does not establish the total runtime or rule out every #SAT-based algorithm.
