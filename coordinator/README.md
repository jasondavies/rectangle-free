# Coordinator Prototype

This is a tiny pull-based shard coordinator for distributed `partition_poly` runs.

It uses:
- `node:http` for the REST server
- `node:sqlite` for local state
- interleaved shards, not contiguous ranges

The shard model matches the solver CLI:
- a shard is one tuple `(task_start, task_end, task_stride, task_offset)`
- for a run with stride `S`, shard `r` owns all tasks `t` with `t % S == r`

That is the right shape for `7x7`:
- much lower overhead than one-task jobs
- much better balance than contiguous ranges
- easy retry and resharding

## Start the server

```bash
node coordinator/server.mjs serve --db coordinator/coordinator.sqlite --port 3000 --lease-seconds 86400
```

## Seed a run

Example `6x6` prototype:

```bash
node coordinator/server.mjs create-run \
  --db coordinator/coordinator.sqlite \
  --solver ./partition_poly \
  --rows 6 \
  --cols 6 \
  --prefix-depth 3 \
  --task-stride 2048 \
  --threads 32
```

Example `7x7` shape:

```bash
node coordinator/server.mjs create-run \
  --db coordinator/coordinator.sqlite \
  --solver ./partition_poly_7 \
  --rows 7 \
  --cols 7 \
  --prefix-depth 2 \
  --task-stride 16384 \
  --threads 32
```

Adaptive runs can be seeded too:

```bash
node coordinator/server.mjs create-run \
  --db coordinator/coordinator.sqlite \
  --solver ./partition_poly_7 \
  --rows 7 \
  --cols 7 \
  --prefix-depth 2 \
  --task-stride 16384 \
  --threads 32 \
  --adaptive-subdivide \
  --adaptive-threshold 128 \
  --adaptive-max-depth 3
```

## Run a worker

```bash
node coordinator/worker.mjs \
  --server http://127.0.0.1:3000 \
  --worker-id worker-a
```

To target one specific run:

```bash
node coordinator/worker.mjs \
  --server http://127.0.0.1:3000 \
  --worker-id worker-a \
  --run-id 3
```

To renew long leases automatically every 10 minutes:

```bash
node coordinator/worker.mjs \
  --server http://127.0.0.1:3000 \
  --worker-id worker-a \
  --run-id 3 \
  --heartbeat-seconds 600
```

The worker:
- fetches one shard lease
- runs the solver once for that shard
- reads the generated `.poly` file
- submits the result back into SQLite through the server

## Inspect run state

For a one-line summary of every run:

```bash
node coordinator/server.mjs list-runs --db coordinator/coordinator.sqlite
```

For one run’s queued/leased/done/failed breakdown plus a short shard summary:

```bash
node coordinator/server.mjs show-run --db coordinator/coordinator.sqlite --run-id 3
```

For every shard in one run:

```bash
node coordinator/server.mjs list-shards --db coordinator/coordinator.sqlite --run-id 3
```

## API

### `POST /fetch-work`

Request:

```json
{
  "worker_id": "worker-a"
}
```

To restrict leasing to one run:

```json
{
  "worker_id": "worker-a",
  "run_id": 3
}
```

Response when work exists:

```json
{
  "work_available": true,
  "shard": {
    "shard_id": 12,
    "run_id": 3,
    "shard_index": 11,
    "task_start": 0,
    "task_end": 385003,
    "task_stride": 16384,
    "task_offset": 11,
    "lease_token": "uuid",
    "lease_until": "2026-03-24T12:00:00.000Z",
    "solver": "./partition_poly_7",
    "solver_args": ["7", "7", "--prefix-depth", "2", "..."],
    "omp_threads": 32
  }
}
```

Response when empty:

```json
{
  "work_available": false
}
```

### `POST /submit-result`

Request body includes:
- `worker_id`
- `shard_id`
- `lease_token`
- `ok`
- timings
- solver stdout/stderr
- `result_poly` as the `.poly` file contents

### `POST /renew-lease`

Request:

```json
{
  "worker_id": "worker-a",
  "shard_id": 12,
  "lease_token": "uuid"
}
```

### Reset stuck leases

To force all leased shards back to the queue:

```bash
node coordinator/server.mjs reset-leases --db coordinator/coordinator.sqlite
```

To reset only one run:

```bash
node coordinator/server.mjs reset-leases --db coordinator/coordinator.sqlite --run-id 3
```

## Notes

- The server is intentionally simple and single-process.
- Lease expiry allows crashed workers to be retried.
- For very large production runs, start with far more shards than workers.
- A good initial `7x7` plan is `task_stride=16384` with one 32-thread process per machine.
