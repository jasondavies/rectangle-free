# Mixed whole-group and sign-range campaigns

`tools/common_core_mixed_campaign.py` integrates independent-query sign
sharding with the complete shared-core campaign queue and grid reducer.
It is provider-neutral and opt-in: old running campaigns, manifests and
results are unchanged. There are no CUDA kernel changes.

Expensive independent queries are split into at most one sign task per GPU,
with each task responsible for every required prime over its half-open range.
Shared-core groups remain indivisible. Other groups are packed into contiguous
tasks. Longest-estimated tasks are assigned first to the least-loaded queue.

## Completed validation

Experiment 500 completed the full **6×28 verification** on four RTX PRO 6000
GPUs: **40 tasks**, **36,398 queries**, and **46,248 verified checkpoint ranges**.
The exact result matches [results.txt](../../results.txt). Measured compute was
**4.13 GPU-hours**, with **65 minutes solving / 72 minutes including setup,
validation and cleanup**. No worker restarts or spot replacements occurred;
all results were pulled and checked before the VM and disk were deleted.

## Prepare a measured manifest

Use a full baseline timing sample/projection and an audited replacement plan's
selected-order sample/projection. Missing timing strata or mismatched query
populations are rejected. No batched-checkpoint speedup is assumed, and the
old per-child overhead is retained for replaced work as a conservative allowance.

```sh
python3 tools/common_core_mixed_campaign.py build \
  --catalog CATALOG --plan CANDIDATE_PLAN --worker TESTED_CUDA_WORKER \
  --baseline-sample FULL_SAMPLE --baseline-projection FULL_PROJECTION \
  --replacement-sample REPLACEMENT_SAMPLE \
  --replacement-projection REPLACEMENT_PROJECTION \
  --workers 8 --target-shards 64 --output mixed.json
```

`--target-shards` is a sizing target, not a promised task count. The eight-GPU
6×28 plan generated in Experiment 499 produced 46 tasks: 28 whole-group tasks and 18
sign tasks covering the order-64 query and two order-60 queries. Its conservative
total is 4.44 GPU-hours, with queues at 30.8–34.7 minutes. These exclude repeated
startup, transfer/audit, final reduction, interruptions and multi-GPU contention;
they are not measured campaign wall times. The completed four-GPU campaign
used a separately generated 40-task manifest, not this eight-GPU assignment.

## Run and resume

```sh
CUDA_VISIBLE_DEVICES=0 python3 tools/common_core_mixed_campaign.py run \
  --catalog CATALOG --plan CANDIDATE_PLAN --manifest mixed.json \
  --worker TESTED_CUDA_WORKER --worker-id 0 --journal-dir journals
```

For the eight-worker example, run queue IDs 0–7 on separate GPUs. One persistent
worker process is retained
across a queue's tasks. Restart the same command to resume. Results live under
`journals/MANIFEST_SHA/task-NNNN.sqlite`; each file has an exclusive local claim.
Separate machines must have distinct queue assignments. `commands` emits an
equivalent shell command without launching it. `verify` validates inputs and
the bound worker binary without starting a solve.

Default CUDA chunks contain 32,768 signs and checkpoints contain 2²⁰ signs.
Transactions flush at 32 ranges or one second between requests, and on normal
exit. `--max-checkpoints 1` is a bounded rehearsal: it stops the queue after one
new checkpoint in a task, not after completing the campaign.

## Snapshot and reduce

The **old** `common_core_snapshot.py` accepts only old whole-group journals.
For mixed journals use the format-aware command below. It performs a SQLite
backup, verifies manifest/task ownership and checksums, forbids rollback of
previously published checkpoints, then atomically publishes a durable snapshot.
Copy this closed snapshot off-host, not an actively changing database file.

```sh
python3 tools/common_core_mixed_campaign.py snapshot \
  --catalog CATALOG --plan CANDIDATE_PLAN --manifest mixed.json \
  --source LIVE_TASK_SQLITE --output PUBLISHED_TASK_SQLITE
```

After downloading, use the same command with `--source DOWNLOADED_SQLITE
--verify-sha256 RECEIPT_SHA256` instead of `--output`.

```sh
python3 tools/common_core_mixed_campaign.py reduce \
  --catalog CATALOG --plan CANDIDATE_PLAN --manifest mixed.json \
  --journals CLOSED_TASK_SNAPSHOTS --require-complete
```

Pass one snapshot per task. Duplicate snapshots, mixed manifests/backends,
invalid hashes, and ranges outside a task's ownership are rejected. All sign
intervals and all required prime images are combined **before** normalization
and CRT. The outer coefficient and column multiplicity are applied once per
query, never once per task. Missing data yields a partial total; with
`--require-complete` it also exits unsuccessfully. No partial total is a final
grid result. The old reducer and standalone single-query shard results cannot
be silently mixed with this format.

## Validation and deployment boundary

The unit suite tests complete synthetic CRT/grid reductions against the existing
whole-group reducer, one-prime versus multi-prime children, overlap/gap rejection,
identity checks, rollback and restart. A bounded two-process CPU rehearsal tests
actual process killing, live snapshot/download verification, restore, resume and
direct residue recomputation. See `tests/hafnian/common_core_mixed_queue_test.py`.
The actual grid is not recalculated by these tests.

Experiment 498 separately checked the CUDA sign-range arithmetic. Experiment
500 passed a two-GPU mixed-task kill/backup/restore rehearsal: all 757 saved
and resumed ranges matched CPU recomputation, and the reducer correctly refused
to certify the incomplete grid. This is a recovery gate, not a full-grid result.

`tools/common_core_remote_campaign.py` accepts both the legacy and mixed
formats. For mixed work its private JSON config additionally needs `catalog`
and `plan`, alongside `manifest`, `worker`, `output`, Unix-time `deadline` and
optional `snapshot_seconds`. It starts one queue per visible GPU and publishes
validated closed snapshots with task-completion receipts. A sign task is complete
when its own interval covers every required prime; that is **not** sufficient to
complete its query without the other sign tasks.

Optional `queue_ids` selects a disjoint list of manifest queues for this machine,
mapped in order to its visible GPUs. Without it, all manifest queues are assigned.
The assignment length must equal the visible GPU count. Cross-machine queue
ownership is the external scheduler's responsibility.

The remote supervisor does not provision machines, pull results off-host, reduce
the full campaign, or delete cloud resources. An external supervisor must download
closed snapshots, verify their receipts and monotone coverage, run the mixed
reducer with `--require-complete`, and retain recoverable checkpoints on failure.
Never replace the binary, manifest dependencies or input artifacts of a running
campaign. Old provider-specific supervisors must still be adapted explicitly.
