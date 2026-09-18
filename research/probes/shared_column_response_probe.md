# Shared last-column response: bounded CPU gate

This research probe tests contraction reuse across related outer masks. It is
not a production solver, a GPU benchmark, or a 9×9 runtime estimate. See
Experiment 502 in [the experiment log](../../docs/experiments.md).

## Exact identity

Fix all but the final outer-bit column, splitting the common core into halves
`A` and `B`. For eight rows/eight columns their widths are four and three.
Let `s` be the active rows of the final column, and define

```
F_s(W) = number of binary assignments on s whose token set avoids W

C([A|B|s]) = sum over disjoint U,V of D_A(U) D_B(V) F_s(U union V).
```

Contract `D_A` and `D_B` once. For each compatible support pair, evaluate the
requested last-column responses. Repeat independently for the complemented
core, querying the complementary active-row sets. Their products supply the
usual `C(G) C(G^c)` values; original outer coefficients would still have to be
retained in a production implementation.

The response vector has only `2^r` entries (256 at eight rows), irrespective of
the size of the seven-column distribution. The implementation never builds
that complete distribution. It either streams unions directly into responses,
or aggregates at most a configured number of union keys before flushing them.
Small output size alone does not bound the cost of constructing the vector.

### Exact symmetry and response evaluation

Both half-distributions use token-plane orbit representatives with their
original per-mask weights. The core join tests ordinary and swapped right
orientations, suppresses the duplicate for fixed right orbits, and restores
the left orbit size. This is valid because `F_s(W) = F_s(swap(W))`.

The optional union cache normalizes `W` under plane exchange and sums **total
contribution weights**, not per-mask weights. Confusing these conventions
would introduce factors of two.

For each union, form the two forbidden row-pair graphs. A depth-eight search
assigns each row one of: absent, colour zero, colour one. It rejects same-colour
conflicts immediately. A precomputed trie of demanded active-row masks prunes
unrequested extensions. Each surviving assignment contributes the union's
weight to its response slot, preserving multiplicities without approximation.

## Reproduce

```
make shared-column-response-test

build/shared_column_response_probe census \
  ../rectangle-free-data-v2/8x8-transpose/solve/s0000.orbits 10000000

# Best observed family: use its actual 28 extensions printed by the census.
build/shared_column_response_probe family 8 7 1974750805758 \
  43,46,47,51,58,59,60,62,170,174,178,184,186,188,190,195,210,211,226,227,234,236,238,240,242,248,250,252 \
  100000 10000000000 20

# Optimistic upper-reuse test, NOT observed campaign demand:
build/shared_column_response_probe family 8 7 1974750805758 \
  all 100000 20000000000 30
```

The observed family above comes from shard `s0768`. `family` takes rows,
core width, row-major core key (decimal or `0x` hex), comma-separated extensions
or `all`, union-cache entry cap, operation cap, and seconds per method/side.
Cache zero enables direct streaming. Core widths 1–7 and rows 2–8 are supported;
the probe does not implement nine-row masks. Distribution construction is
separately capped at one million full-support entries per map. A capped method
discards its answer vector and never reports exact parity.

JSON lines report both independent half-join and shared-response times,
construction time, predicate counts, response search nodes, cache size/flushes,
and equality checks. Both sides include building their needed halves; the
independent control reuses its fixed half across the whole family. Subtract
`build_seconds` when comparing CPU contraction-only work, but do not interpret
that subtraction as a GPU warm-cache measurement.

The padded 16×8 weight-class tile counter uses the production eight-row prefix
coordinates. It is a **model**, not a hardware instruction trace. Its accounting
time is reported and excluded from `seconds`. In particular, a BMMA scalar
sum does not give the contributing union masks required by the new method.

## Census scope and interpretation

The census reads a contiguous prefix of one checked-format `R8ORB01` or
`R8SQT01` file, bounded to ten million records. Delete physical column seven
and sort `(remaining row pattern, removed bit)` to obtain a deterministic
common row gauge. Record actual distinct requested extensions, not a presumed
256-fold reuse. This counts every retained record, including ones whose final
contribution is zero; it does not establish time-weighted coverage.

The measured four-file census used each entire shard separately. It does not
merge families across shards, canonicalize the core under column permutations,
or choose among all eight removable columns. Those could increase reuse, but
their grouping costs and effects on half-cache/layout reuse are unmeasured.
Therefore the result is not a campaign-wide upper bound on all shared-core
methods.

**Outcome:** exact CPU gains for selected high-reuse families, but no evidence
yet of an order-of-magnitude campaign improvement or a GPU advantage. Keep
the implementation isolated. Before a GPU port, measure production-work-weighted
coverage and the cost of broader core grouping. No external resources were
provisioned and no new grid result was recorded.
