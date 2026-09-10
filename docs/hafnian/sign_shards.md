# Independent-query sign sharding

`tools/common_core_sign_shards.py` splits one independent matching query into
disjoint half-open sign ranges. Each task computes every required prime image
over its assigned range, using the existing persistent CUDA worker unchanged.
This removes the whole-query scheduling bottleneck without reducing arithmetic.

This remains a standalone single-query tool. For integrated whole-group/sign-range
queues and grid reduction use the separate [mixed campaign workflow](mixed_campaign.md).
Neither tool provisions resources. Existing campaigns remain unchanged. Never
add standalone partial residues to a grid total or also count the same query
through a whole-group task.

## Usage

Build with the tested GPU binary and an audited catalog/plan. Group IDs are
plan-specific; 5417 is the order-64 query in the Experiment 494 6×28 plan.

```sh
python3 tools/common_core_sign_shards.py build \
  --catalog CATALOG --plan PLAN --worker WORKER \
  --group 5417 --parts 8 --output sign-manifest.json
```

Run tasks 0–7 independently, with one journal per task and one task per GPU.
`CUDA_VISIBLE_DEVICES` selects the GPU. Restart exactly the same command to
resume. Set `--max-checkpoints 1` for a bounded test, not a complete solve.

```sh
CUDA_VISIBLE_DEVICES=0 python3 tools/common_core_sign_shards.py run \
  --catalog CATALOG --plan PLAN --worker WORKER \
  --manifest sign-manifest.json --task 0 --journal task-0.sqlite
```

The default checkpoint is 2²⁰ signs, internally chunked into 32,768 signs;
durable transactions batch up to 32 ranges or one second between requests.
For this standalone format, stop its writer before copying the journal; the
old production snapshot tool does not accept it. Use the mixed workflow when
live publication is required. The exclusive per-journal
claim prevents two local writers. Distinct hosts must be assigned distinct tasks.

```sh
python3 tools/common_core_sign_shards.py reduce \
  --catalog CATALOG --plan PLAN --manifest sign-manifest.json \
  --journals task-0.sqlite task-1.sqlite task-2.sqlite task-3.sqlite \
             task-4.sqlite task-5.sqlite task-6.sqlite task-7.sqlite \
  --require-complete
```

This certifies **one matching count**, not `T_4(6,28)`. The reducer checks
manifest, catalog, plan, solver, controller, backend and configuration binding,
journal checksums, task ownership, all required primes and exact full-domain
coverage. Only after summing all ranges does it apply the full sign-domain and
dummy-factorial normalization, CRT and the certified matching bound. Missing
data produces `partial_query`; `--require-complete` additionally exits with an
error. Duplicate task snapshots are rejected rather than silently deduplicated.

Manifests bind controller source bytes: regenerate them after changing the
runner or shared controller. CPU-reference results use a separately bound
`--cpu-reference` manifest and cannot mix with GPU results.

## Validation and completed deployment

```sh
make BUILD_DIR=build/local-check build/local-check/hafnian_common_worker_host
HAFNIAN_TEST_WORKER=build/local-check/hafnian_common_worker_host \
  python3 tests/hafnian/common_core_sign_shards_test.py
```

The local tests cover uneven splitting, 32-bit boundary values, CRT, missing
primes, restarts, rollback, corrupt/duplicate/out-of-range results, and real
order-64 finite-field sign sums across four primes. Experiment 498 additionally
verified eight-way CUDA range summation under all four primes, and high/low
task checkpoint/restart with the real runner. Those tests used one GPU.
Experiment 500 subsequently passed a two-GPU mixed-task kill/backup/restore
gate, with all 757 saved/resumed ranges matching CPU recomputation.

The complete 6×28 verification then ran on four RTX PRO 6000 GPUs using the
integrated mixed workflow: 28 whole-group tasks and 12 sign tasks, covering
four intervals for each of three expensive independent queries. All 36,398
queries reproduced the existing exact result in 4.13 timed GPU-hours and
65 minutes of solving. This validates the shared interval executor in a full
campaign; standalone single-query journals still use their own format and
must not be mixed into a grid reduction.
