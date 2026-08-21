# GPU code map

The active exact GPU implementation is intentionally split by responsibility:

- `twocolour_gpu_common.cuh`: orbit input, distribution recurrence,
  canonicalisation, and geometry-independent exact primitives.
- `twocolour_7x7_engine.cuh`: scalar full-mask expansion and join machinery
  used only by the complete known-result 7x7 regression solver.
- `twocolour_prefix_algebra.cuh`: fixed production prefix/suffix coordinates,
  mask transformations, and shared layout descriptors.
- `twocolour_prefix_core.cuh`: seven-row packed 7x5 cache construction and
  prefetch pipeline.
- `twocolour_weight_class_bmma.cuh`: direct weight-grouped layouts and
  the BMMA join shared by 7x9 and 8x8.
- `twocolour_canonical_device.cuh`: geometry-neutral canonical device cache
  and direct grouped-layout adapter shared by 7x9 and 8x8.
- `twocolour_8x8_prefix_solve.cu`: production 8x8 orchestration.
- `twocolour_7x9_engine.cuh`: reusable production 7x9 orchestration, resident and
  streamed cache operation, and the producer/join pipeline.
- `twocolour_7x9_cache_build.cu`: one-time construction and publication of the
  versioned, checksummed, memory-mapped 7x5 cache artifact.
- `twocolour_7x9_packed_solve.cu`: one-line executable wrapper around the 7x9
  engine.
- `twocolour_7x9_four_owner_solve.cu`: optional production 7x9 scheduler that
  keeps four left layouts resident and shares each union right layout.
- `gpu_result_checkpoint.hpp`: geometry-neutral manifest, provenance,
  validation, work-claim, and immutable checkpoint publication support shared
  by the 7x9 and 8x8 production solvers.
- `tools/aggregate_gpu_v3.py`: provider-neutral validation and exact reduction for
  new manifest-driven 7x9 and 8x8 v3 campaigns.
- `gpu_cuda_utils.cuh`: move-only device, pinned-host, stream, and event
  ownership utilities.
- `gpu_memory_policy.hpp`: shared automatic residency/batch policy and the
  `RECT_GPU_MEMORY_RESERVE_MIB` safety-reserve override.

The complete physical review surface is 3,069 lines for 8x8 and 4,626 lines
for the single-owner 7x9 path. The optional four-owner entry point gives a
5,213-line 7x9 surface; the independent 7x7 surface is 1,265 lines. Shared
checkpoint and SHA-256 support are counted in
each figure. This is distinct from the historical standalone CLIs in `legacy/gpu/`
and isolated research programs in `archive/gpu/` or `research/gpu/`.

Token-plane quotienting is the sole representation in every maintained
production distribution join. It stores one support representative under
global inner-colour complementation and restores both relative orientations in
the join. It is ordinary source code rather than a feature flag; there is no
maintained full-support GPU fallback.

`make gpu-production` builds the intentionally small production surface:
`twocolour_7x7_solve_gpu`, `twocolour_7x9_solve_gpu`,
`twocolour_7x9_four_owner_gpu`, `twocolour_7x9_cache_build`, and
`twocolour_8x8_solve_gpu`. Algorithmic
choices are fixed in those sources. The only remaining build parameters tune
resource use, such as the 7x9 prefetch allocation.

Quotient-sensitive scalar tuning knobs are not exposed by the production core.
The maintained scalar and BMMA implementations each contain only their
accepted quotient-aware kernel. Prefix coordinates and task chunking are fixed
source constants; the only build-time production capacity control is the 7x9
prefetch allocation.

Square production corpora are additionally quotiented by matrix transposition.
Their versioned magic values are `R7SQT01` and `R8SQT01`; production solvers
reject ordinary row/column-only corpora. `binary_orbit_augment`
`square-transpose-filter` produces the 7x7 corpus. The existing 8x8
`solve-transpose-filter` performs the same final operation after complement
pairing and sharding.

For 7x9, the complete packed 7x5 cache contains 2,370,316,739 quotient
representatives (17.66 GiB), down from 4,740,574,641 entries. Keeping it
resident is technically possible on a 46-GiB L40S, but a matched full-shard
gate found the existing 20-GiB streamed pipeline 2.56% faster because it builds
larger right batches; use `PACKED_PREFETCH_MIB=20480` for production L40S
builds. The 4,608-MiB default remains suitable for GPUs with less memory.

Device placement is automatic.  The 7x9 solver keeps the complete cache on the
GPU only when doing so leaves 32 GiB of recurring workspace in addition to the
safety reserve.  This selects residency on 96-GB RTX PRO 6000-class devices but
retains the measured streamed policy on 48-GB L40S devices.  Right batches then
expand to the largest safe 32-bit-offset layout.  Override only the safety
margin, when operationally necessary, with
`RECT_GPU_MEMORY_RESERVE_MIB`; cache mode is not a production flag.

The first 7x9 argument may be either the canonical `R7ORB01` 7x5 orbit file or
the versioned `R7PCK01` packed artifact emitted by
`twocolour_7x9_cache_build`. The former reconstructs the cache in about one
minute and is preferable on a cold, slow network or boot disk. The latter
memory-maps the 17.66-GiB payload and validates each touched 4-MiB block once;
it is preferable when staged on local NVMe or reused from the host page cache.
Both inputs have the same checkpoint identity, derived from the canonical
orbit-file SHA-256.

The 7x9 and 8x8 production solvers consume the same manifest form:

```text
ID ORBITS START END [FILTER_MOD FILTER_ID]
```

They publish self-identifying version-3 result files through the shared
checkpoint library. Each file binds the result to the solver binary,
compile-time algorithm configuration, canonical-cache contents, orbit-corpus
contents, and exact range/filter identity with SHA-256 digests, and includes a
checksum over its own payload. Publication uses a unique temporary file and a
non-overwriting hard link; a per-result advisory work claim prevents two
processes on one shared filesystem from solving the same item concurrently.
Ranges and ownership filters are intersected when both are present. Valid
results are checked before CUDA initialization, so a completed manifest exits
without reconstructing or uploading the canonical cache.

The 8x8 invocation is:

```text
twocolour_8x8_solve_gpu CANONICAL_SEED.orbits WORK.tsv RESULTS_DIR \
    [BATCH_EDGES=auto] [VERIFY_JOINS=4]
```

The default `auto` removes the historical 16,384-edge ceiling; device-memory
accounting remains the binding batch limit.  An explicit numeric edge ceiling
is retained for controlled profiling.

The explicit seed replaces the historical `PREFIX_CANONICAL_SEED` environment
variable and need not itself be a work item. The 8x8 result magic is
`RECT8X8_PREFIX_RESULT 3`; 7x9 retains `RECT7X9_PACKED_RESULT 3`. Version-2
7x9 files from the completed historical campaign remain readable by
`tools/aggregate_packed_7x9.py`, but production solvers will not treat them as
restart checkpoints because they lack provenance. `make
gpu_result_checkpoint_test` exercises the shared non-CUDA contract.

Use the same manifest that was supplied to a provider to inspect a partial v3
campaign:

```text
python3 tools/aggregate_gpu_v3.py 8x8 WORK.tsv RESULTS_DIR
```

For final acceptance, make the corpus locally accessible and enable all exact
gates:

```text
python3 tools/aggregate_gpu_v3.py 8x8 WORK.tsv RESULTS_DIR \
    --corpus-root CORPUS_DIR --full --verify-input-sha256 \
    --canonical-cache CANONICAL_SEED.orbits \
    --solver-binary twocolour_8x8_solve_gpu \
    --write-json campaign.json
```

The equivalent `7x9` command changes only the geometry and canonical-cache
arguments. Full validation proves range coverage and, for ownership-filtered
work, accepts only a complete set of owner IDs under one modulus on every
interval. It then enforces the geometry's global record, kernel, labelled
weight, and covered-weight invariants. For 7x9 it also requires the
independently known exact contribution. Result payload checksums are always verified;
`--verify-input-sha256`, `--canonical-cache`, and `--solver-binary` additionally
bind the downloaded outputs back to local immutable artifacts. Multiple exact
binaries are reportable for independent builds; use one `--solver-binary` per
variant or `--require-single-binary` for a homogeneous campaign.

The current 8x8 solver accepts only the versioned transpose-quotient magic
`R8SQT01`. The local production corpus was regenerated with the exact transpose
filter and is accompanied by a 1,024-entry SHA-256 list and provider-neutral
work manifest under `../rectangle-free-data-v2/8x8-transpose/`. Its legacy
`R8ORB01` predecessor is retained separately; legacy files must never be
relabelled. `binary_orbit_augment_8x8 solve-check` validates shards in parallel
and performs a deterministic exact reduction; set `OMP_NUM_THREADS` to the
desired checker concurrency. `tools/aggregate_8x8_results.py` remains solely for the
historical pre-transpose provider-result inventory. Run `make gpu-campaign-test`
for the non-CUDA campaign regression suite.

`make gpu-code-dump` regenerates `code-dump.txt` from the current maintained
surface for external review; the generated file is intentionally untracked.

Historical and regression targets are not dependencies of `gpu-production`.
Production code must not include files from `archive/gpu/` or `legacy/gpu/`.
The historical two-level hierarchy layout and kernel are wholly contained in
`legacy/gpu/twocolour_prefix_hierarchy.cuh`; the production prefix core has no
hierarchy feature gates, types, fields, builders, or kernels.
Conditionally compiled research hooks live under `research/gpu/`; ordinary
production preprocessing does not open them.  In particular, the rejected
Patricia/ZDD suffix-query implementation is isolated there behind
`PROFILE_SUFFIX_STRUCTURES`; the rejected suffix-bitplane kernels and index
builder are similarly isolated behind `PROTOTYPE_SUFFIX_BITPLANES`.
