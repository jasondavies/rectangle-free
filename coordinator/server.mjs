#!/usr/bin/env node

import http from 'node:http';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { DatabaseSync } from 'node:sqlite';

const DEFAULT_DB_PATH = path.resolve('coordinator', 'coordinator.sqlite');
const DEFAULT_HOST = '0.0.0.0';
const DEFAULT_PORT = 3000;
const DEFAULT_LEASE_SECONDS = 3600;

function usage() {
  console.error(
    [
      'usage:',
      '  node coordinator/server.mjs serve [--db PATH] [--host HOST] [--port PORT] [--lease-seconds N]',
      '  node coordinator/server.mjs create-run --rows R --cols C --prefix-depth D --task-stride S [options]',
      '  node coordinator/server.mjs list-runs [--db PATH]',
      '  node coordinator/server.mjs show-run --run-id ID [--db PATH]',
      '  node coordinator/server.mjs list-shards --run-id ID [--db PATH]',
      '  node coordinator/server.mjs reset-leases [--run-id ID] [--db PATH]',
      '  node coordinator/server.mjs merge-run --run-id ID [--output FILE] [--db PATH]',
      '',
      'create-run options:',
      '  --task-start N           default 0',
      '  --task-end N             default total_tasks from solver setup',
      '  --solver PATH            default ./partition_poly_7',
      '  --threads N              default 32',
      '  --adaptive-subdivide     enable adaptive subdivision',
      '  --adaptive-threshold N   default 128',
      '  --adaptive-max-depth N   default 3',
      '  --name TEXT              optional run label',
      '',
      'REST API:',
      '  POST /fetch-work   body: { worker_id, run_id? }',
      '  POST /renew-lease body: { shard_id, lease_token, worker_id }',
      '  POST /submit-result body: { shard_id, lease_token, worker_id, ok, ... }',
      '  GET  /healthz',
    ].join('\n')
  );
}

function parseArgs(argv) {
  const args = {
    db: DEFAULT_DB_PATH,
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    leaseSeconds: DEFAULT_LEASE_SECONDS,
    taskStart: 0,
    taskEnd: null,
    solver: './partition_poly_7',
    threads: 32,
    adaptiveSubdivide: false,
    adaptiveThreshold: 128,
    adaptiveMaxDepth: 3,
    name: null,
    runId: null,
    output: null,
  };
  const positionals = [];
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    switch (arg) {
      case '--db':
        args.db = path.resolve(argv[++i]);
        break;
      case '--host':
        args.host = argv[++i];
        break;
      case '--port':
        args.port = Number(argv[++i]);
        break;
      case '--lease-seconds':
        args.leaseSeconds = Number(argv[++i]);
        break;
      case '--rows':
        args.rows = Number(argv[++i]);
        break;
      case '--cols':
        args.cols = Number(argv[++i]);
        break;
      case '--prefix-depth':
        args.prefixDepth = Number(argv[++i]);
        break;
      case '--task-stride':
        args.taskStride = Number(argv[++i]);
        break;
      case '--task-start':
        args.taskStart = Number(argv[++i]);
        break;
      case '--task-end':
        args.taskEnd = Number(argv[++i]);
        break;
      case '--solver':
        args.solver = argv[++i];
        break;
      case '--threads':
        args.threads = Number(argv[++i]);
        break;
      case '--adaptive-subdivide':
        args.adaptiveSubdivide = true;
        break;
      case '--adaptive-threshold':
        args.adaptiveThreshold = Number(argv[++i]);
        break;
      case '--adaptive-max-depth':
        args.adaptiveMaxDepth = Number(argv[++i]);
        break;
      case '--name':
        args.name = argv[++i];
        break;
      case '--run-id':
        args.runId = Number(argv[++i]);
        break;
      case '--output':
        args.output = path.resolve(argv[++i]);
        break;
      default:
        positionals.push(arg);
        break;
    }
  }
  return { command: positionals[0] ?? 'serve', args };
}

function ensureParentDir(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
}

function openDb(dbPath) {
  ensureParentDir(dbPath);
  const db = new DatabaseSync(dbPath);
  db.exec(`
    PRAGMA journal_mode = WAL;
    PRAGMA busy_timeout = 5000;

    CREATE TABLE IF NOT EXISTS runs (
      id INTEGER PRIMARY KEY,
      name TEXT,
      solver TEXT NOT NULL,
      rows INTEGER NOT NULL,
      cols INTEGER NOT NULL,
      prefix_depth INTEGER NOT NULL,
      task_start INTEGER NOT NULL,
      task_end INTEGER NOT NULL,
      task_stride INTEGER NOT NULL,
      total_tasks INTEGER NOT NULL,
      threads INTEGER NOT NULL,
      adaptive_subdivide INTEGER NOT NULL,
      adaptive_threshold INTEGER NOT NULL,
      adaptive_max_depth INTEGER NOT NULL,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS shards (
      id INTEGER PRIMARY KEY,
      run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
      shard_index INTEGER NOT NULL,
      task_start INTEGER NOT NULL,
      task_end INTEGER NOT NULL,
      task_stride INTEGER NOT NULL,
      task_offset INTEGER NOT NULL,
      status TEXT NOT NULL,
      worker_id TEXT,
      lease_token TEXT,
      lease_until TEXT,
      attempt_count INTEGER NOT NULL DEFAULT 0,
      started_at TEXT,
      completed_at TEXT,
      elapsed_seconds REAL,
      worker_seconds REAL,
      prefix_seconds REAL,
      ok INTEGER,
      result_poly TEXT,
      stdout TEXT,
      stderr TEXT,
      solver_exit_code INTEGER,
      last_error TEXT,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(run_id, shard_index)
    );

    CREATE INDEX IF NOT EXISTS shards_claim_idx
      ON shards(run_id, status, lease_until, shard_index);
  `);
  return db;
}

function nowIso() {
  return new Date().toISOString();
}

function addSeconds(iso, seconds) {
  return new Date(new Date(iso).getTime() + seconds * 1000).toISOString();
}

function json(res, statusCode, payload) {
  const body = JSON.stringify(payload, null, 2);
  res.writeHead(statusCode, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(body),
  });
  res.end(body);
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.setEncoding('utf8');
    req.on('data', (chunk) => {
      data += chunk;
      if (data.length > 10 * 1024 * 1024) {
        reject(new Error('request body too large'));
        req.destroy();
      }
    });
    req.on('end', () => {
      if (!data) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(data));
      } catch (err) {
        reject(err);
      }
    });
    req.on('error', reject);
  });
}

function buildSolverArgs(run, shard) {
  const args = [
    String(run.rows),
    String(run.cols),
    '--prefix-depth',
    String(run.prefix_depth),
    '--task-start',
    String(shard.task_start),
    '--task-end',
    String(shard.task_end),
    '--task-stride',
    String(shard.task_stride),
    '--task-offset',
    String(shard.task_offset),
  ];
  if (run.adaptive_subdivide) {
    args.push(
      '--adaptive-subdivide',
      '--adaptive-threshold',
      String(run.adaptive_threshold),
      '--adaptive-max-depth',
      String(run.adaptive_max_depth)
    );
  }
  return args;
}

function fetchWork(db, leaseSeconds, body) {
  const workerId = String(body.worker_id ?? '').trim();
  if (!workerId) {
    return { status: 400, payload: { error: 'worker_id is required' } };
  }
  const runId = body.run_id == null ? null : Number(body.run_id);
  const now = nowIso();
  const leaseUntil = addSeconds(now, leaseSeconds);
  const leaseToken = crypto.randomUUID();

  db.exec('BEGIN IMMEDIATE');
  try {
    const shard = runId == null
      ? db.prepare(`
          SELECT s.*, r.solver, r.rows, r.cols, r.prefix_depth, r.threads,
                 r.adaptive_subdivide, r.adaptive_threshold, r.adaptive_max_depth
          FROM shards s
          JOIN runs r ON r.id = s.run_id
          WHERE s.status IN ('queued', 'leased')
            AND (s.lease_until IS NULL OR s.lease_until < ?)
          ORDER BY s.run_id, s.shard_index
          LIMIT 1
        `).get(now)
      : db.prepare(`
          SELECT s.*, r.solver, r.rows, r.cols, r.prefix_depth, r.threads,
                 r.adaptive_subdivide, r.adaptive_threshold, r.adaptive_max_depth
          FROM shards s
          JOIN runs r ON r.id = s.run_id
          WHERE s.run_id = ?
            AND s.status IN ('queued', 'leased')
            AND (s.lease_until IS NULL OR s.lease_until < ?)
          ORDER BY s.shard_index
          LIMIT 1
        `).get(runId, now);

    if (!shard) {
      db.exec('COMMIT');
      return { status: 200, payload: { work_available: false } };
    }

    db.prepare(`
      UPDATE shards
      SET status = 'leased',
          worker_id = ?,
          lease_token = ?,
          lease_until = ?,
          attempt_count = attempt_count + 1,
          started_at = COALESCE(started_at, ?),
          updated_at = ?
      WHERE id = ?
    `).run(workerId, leaseToken, leaseUntil, now, now, shard.id);

    db.exec('COMMIT');

    const solverArgs = buildSolverArgs(shard, shard);
    return {
      status: 200,
      payload: {
        work_available: true,
        shard: {
          shard_id: shard.id,
          run_id: shard.run_id,
          shard_index: shard.shard_index,
          task_start: shard.task_start,
          task_end: shard.task_end,
          task_stride: shard.task_stride,
          task_offset: shard.task_offset,
          lease_token: leaseToken,
          lease_until: leaseUntil,
          solver: shard.solver,
          solver_args: solverArgs,
          omp_threads: shard.threads,
        },
      },
    };
  } catch (err) {
    db.exec('ROLLBACK');
    throw err;
  }
}

function submitResult(db, body) {
  const shardId = Number(body.shard_id);
  const leaseToken = String(body.lease_token ?? '');
  const workerId = String(body.worker_id ?? '');
  if (!Number.isInteger(shardId) || shardId <= 0) {
    return { status: 400, payload: { error: 'valid shard_id is required' } };
  }
  if (!leaseToken) {
    return { status: 400, payload: { error: 'lease_token is required' } };
  }
  if (!workerId) {
    return { status: 400, payload: { error: 'worker_id is required' } };
  }

  const shard = db.prepare('SELECT * FROM shards WHERE id = ?').get(shardId);
  if (!shard) {
    return { status: 404, payload: { error: 'unknown shard' } };
  }
  if (shard.lease_token !== leaseToken || shard.worker_id !== workerId) {
    return { status: 409, payload: { error: 'lease mismatch' } };
  }

  const ok = body.ok ? 1 : 0;
  const now = nowIso();
  db.prepare(`
    UPDATE shards
    SET status = ?,
        completed_at = ?,
        updated_at = ?,
        ok = ?,
        elapsed_seconds = ?,
        worker_seconds = ?,
        prefix_seconds = ?,
        result_poly = ?,
        stdout = ?,
        stderr = ?,
        solver_exit_code = ?,
        last_error = ?,
        lease_until = NULL
    WHERE id = ?
  `).run(
    ok ? 'done' : 'failed',
    now,
    now,
    ok,
    body.elapsed_seconds ?? null,
    body.worker_seconds ?? null,
    body.prefix_seconds ?? null,
    body.result_poly ?? null,
    body.stdout ?? null,
    body.stderr ?? null,
    body.solver_exit_code ?? null,
    body.last_error ?? null,
    shardId
  );

  return { status: 200, payload: { ok: true } };
}

function renewLease(db, leaseSeconds, body) {
  const shardId = Number(body.shard_id);
  const leaseToken = String(body.lease_token ?? '');
  const workerId = String(body.worker_id ?? '');
  if (!Number.isInteger(shardId) || shardId <= 0) {
    return { status: 400, payload: { error: 'valid shard_id is required' } };
  }
  if (!leaseToken) {
    return { status: 400, payload: { error: 'lease_token is required' } };
  }
  if (!workerId) {
    return { status: 400, payload: { error: 'worker_id is required' } };
  }

  const shard = db.prepare('SELECT * FROM shards WHERE id = ?').get(shardId);
  if (!shard) {
    return { status: 404, payload: { error: 'unknown shard' } };
  }
  if (shard.status !== 'leased') {
    return { status: 409, payload: { error: 'shard is not currently leased' } };
  }
  if (shard.lease_token !== leaseToken || shard.worker_id !== workerId) {
    return { status: 409, payload: { error: 'lease mismatch' } };
  }

  const now = nowIso();
  const leaseUntil = addSeconds(now, leaseSeconds);
  db.prepare(`
    UPDATE shards
    SET lease_until = ?,
        updated_at = ?
    WHERE id = ?
  `).run(leaseUntil, now, shardId);
  return { status: 200, payload: { ok: true, lease_until: leaseUntil } };
}

function statsForRun(db, runId) {
  return db.prepare(`
    SELECT
      COUNT(*) AS total,
      SUM(CASE WHEN status = 'queued' THEN 1 ELSE 0 END) AS queued,
      SUM(CASE WHEN status = 'leased' THEN 1 ELSE 0 END) AS leased,
      SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) AS done,
      SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed
    FROM shards
    WHERE run_id = ?
  `).get(runId);
}

function serve(args) {
  const db = openDb(args.db);
  const server = http.createServer(async (req, res) => {
    try {
      if (req.method === 'GET' && req.url === '/healthz') {
        json(res, 200, { ok: true });
        return;
      }
      if (req.method === 'POST' && req.url === '/fetch-work') {
        const body = await readJsonBody(req);
        const response = fetchWork(db, args.leaseSeconds, body);
        json(res, response.status, response.payload);
        return;
      }
      if (req.method === 'POST' && req.url === '/renew-lease') {
        const body = await readJsonBody(req);
        const response = renewLease(db, args.leaseSeconds, body);
        json(res, response.status, response.payload);
        return;
      }
      if (req.method === 'POST' && req.url === '/submit-result') {
        const body = await readJsonBody(req);
        const response = submitResult(db, body);
        json(res, response.status, response.payload);
        return;
      }
      if (req.method === 'GET' && req.url?.startsWith('/runs/')) {
        const runId = Number(req.url.split('/')[2]);
        const run = db.prepare('SELECT * FROM runs WHERE id = ?').get(runId);
        if (!run) {
          json(res, 404, { error: 'run not found' });
          return;
        }
        json(res, 200, { run, stats: statsForRun(db, runId) });
        return;
      }
      json(res, 404, { error: 'not found' });
    } catch (err) {
      json(res, 500, { error: String(err?.message ?? err) });
    }
  });

  server.listen(args.port, args.host, () => {
    console.log(`coordinator listening on http://${args.host}:${args.port}`);
    console.log(`database: ${args.db}`);
  });
}

function runSolverForTotalTasks(args) {
  const solverArgs = [
    args.solver,
    String(args.rows),
    String(args.cols),
    '--prefix-depth',
    String(args.prefixDepth),
    '--task-end',
    '0',
  ];
  if (args.adaptiveSubdivide) {
    solverArgs.push(
      '--adaptive-subdivide',
      '--adaptive-threshold',
      String(args.adaptiveThreshold),
      '--adaptive-max-depth',
      String(args.adaptiveMaxDepth)
    );
  }
  const proc = BunLikeSpawnSync(solverArgs);
  const match = proc.stdout.match(/Prefix depth:\s+\d+\s+\((\d+)\s+tasks\)/);
  if (!match) {
    throw new Error(`failed to parse total tasks from solver output\n${proc.stdout}\n${proc.stderr}`);
  }
  return Number(match[1]);
}

function BunLikeSpawnSync(argv) {
  const proc = spawnSync(argv[0], argv.slice(1), {
    encoding: 'utf8',
    env: {
      ...process.env,
      OMP_NUM_THREADS: '1',
      RECT_PROGRESS_STEP: '1000000',
    },
  });
  if (proc.error) throw proc.error;
  if (proc.status !== 0) {
    throw new Error(`command failed: ${argv.join(' ')}\n${proc.stdout}\n${proc.stderr}`);
  }
  return { stdout: proc.stdout ?? '', stderr: proc.stderr ?? '' };
}

function createRun(args) {
  if (!Number.isInteger(args.rows) || !Number.isInteger(args.cols) ||
      !Number.isInteger(args.prefixDepth) || !Number.isInteger(args.taskStride)) {
    throw new Error('create-run requires --rows, --cols, --prefix-depth, and --task-stride');
  }
  const db = openDb(args.db);
  const totalTasks = runSolverForTotalTasks(args);
  const taskStart = args.taskStart ?? 0;
  const taskEnd = args.taskEnd ?? totalTasks;
  const runInfo = db.prepare(`
    INSERT INTO runs (
      name, solver, rows, cols, prefix_depth, task_start, task_end, task_stride, total_tasks,
      threads, adaptive_subdivide, adaptive_threshold, adaptive_max_depth
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    args.name,
    args.solver,
    args.rows,
    args.cols,
    args.prefixDepth,
    taskStart,
    taskEnd,
    args.taskStride,
    totalTasks,
    args.threads,
    args.adaptiveSubdivide ? 1 : 0,
    args.adaptiveThreshold,
    args.adaptiveMaxDepth
  );
  const runId = Number(runInfo.lastInsertRowid);

  db.exec('BEGIN IMMEDIATE');
  try {
    const insertShard = db.prepare(`
      INSERT INTO shards (
        run_id, shard_index, task_start, task_end, task_stride, task_offset, status
      ) VALUES (?, ?, ?, ?, ?, ?, 'queued')
    `);
    for (let offset = 0; offset < args.taskStride; offset++) {
      insertShard.run(runId, offset, taskStart, taskEnd, args.taskStride, offset);
    }
    db.exec('COMMIT');
  } catch (err) {
    db.exec('ROLLBACK');
    throw err;
  }

  const run = db.prepare('SELECT * FROM runs WHERE id = ?').get(runId);
  const stats = statsForRun(db, runId);
  console.log(JSON.stringify({ run, stats }, null, 2));
}

function listRuns(args) {
  const db = openDb(args.db);
  const rows = db.prepare(`
    SELECT r.*,
           SUM(CASE WHEN s.status = 'queued' THEN 1 ELSE 0 END) AS queued,
           SUM(CASE WHEN s.status = 'leased' THEN 1 ELSE 0 END) AS leased,
           SUM(CASE WHEN s.status = 'done' THEN 1 ELSE 0 END) AS done,
           SUM(CASE WHEN s.status = 'failed' THEN 1 ELSE 0 END) AS failed
    FROM runs r
    LEFT JOIN shards s ON s.run_id = r.id
    GROUP BY r.id
    ORDER BY r.id DESC
  `).all();
  console.log(JSON.stringify(rows, null, 2));
}

function showRun(args) {
  if (!Number.isInteger(args.runId) || args.runId <= 0) {
    throw new Error('show-run requires --run-id');
  }
  const db = openDb(args.db);
  const run = db.prepare('SELECT * FROM runs WHERE id = ?').get(args.runId);
  if (!run) {
    throw new Error(`run ${args.runId} not found`);
  }
  const stats = statsForRun(db, args.runId);
  const shardStatus = db.prepare(`
    SELECT status, COUNT(*) AS count
    FROM shards
    WHERE run_id = ?
    GROUP BY status
    ORDER BY status
  `).all(args.runId);
  const leased = db.prepare(`
    SELECT shard_index, worker_id, lease_until, attempt_count
    FROM shards
    WHERE run_id = ? AND status = 'leased'
    ORDER BY shard_index
    LIMIT 16
  `).all(args.runId);
  const failed = db.prepare(`
    SELECT shard_index, worker_id, attempt_count, last_error
    FROM shards
    WHERE run_id = ? AND status = 'failed'
    ORDER BY shard_index
    LIMIT 16
  `).all(args.runId);
  const slowestDone = db.prepare(`
    SELECT shard_index, elapsed_seconds, worker_seconds, prefix_seconds, attempt_count
    FROM shards
    WHERE run_id = ? AND status = 'done'
    ORDER BY COALESCE(worker_seconds, elapsed_seconds, 0) DESC, shard_index
    LIMIT 10
  `).all(args.runId);
  console.log(JSON.stringify({
    run,
    stats,
    shard_status: shardStatus,
    leased_shards: leased,
    failed_shards: failed,
    slowest_done_shards: slowestDone,
  }, null, 2));
}

function listShards(args) {
  if (!Number.isInteger(args.runId)) {
    throw new Error('list-shards requires --run-id');
  }
  const db = openDb(args.db);
  const rows = db.prepare(`
    SELECT id, shard_index, task_stride, task_offset, status, worker_id, lease_until,
           attempt_count, elapsed_seconds, worker_seconds, prefix_seconds, ok
    FROM shards
    WHERE run_id = ?
    ORDER BY shard_index
  `).all(args.runId);
  console.log(JSON.stringify(rows, null, 2));
}

function resetLeases(args) {
  const db = openDb(args.db);
  const now = nowIso();
  const result = args.runId == null
    ? db.prepare(`
        UPDATE shards
        SET status = 'queued',
            worker_id = NULL,
            lease_token = NULL,
            lease_until = NULL,
            updated_at = ?,
            last_error = CASE
              WHEN status = 'leased' THEN 'manual lease reset'
              ELSE last_error
            END
        WHERE status = 'leased'
      `).run(now)
    : db.prepare(`
        UPDATE shards
        SET status = 'queued',
            worker_id = NULL,
            lease_token = NULL,
            lease_until = NULL,
            updated_at = ?,
            last_error = CASE
              WHEN status = 'leased' THEN 'manual lease reset'
              ELSE last_error
            END
        WHERE run_id = ? AND status = 'leased'
      `).run(now, args.runId);
  console.log(JSON.stringify({
    ok: true,
    run_id: args.runId ?? null,
    reset_count: result.changes,
  }, null, 2));
}

function writeTempPolyFiles(tempDir, rows) {
  const paths = [];
  for (const row of rows) {
    const filePath = path.join(tempDir, `shard_${row.shard_index}.poly`);
    fs.writeFileSync(filePath, row.result_poly, 'utf8');
    paths.push(filePath);
  }
  return paths;
}

function mergePolyBatch(solver, inputPaths, outputPath) {
  const argv = [solver, '--merge', ...inputPaths, '--poly-out', outputPath];
  BunLikeSpawnSync(argv);
}

function mergePolyFiles(solver, inputPaths, outputPath) {
  if (inputPaths.length === 0) {
    throw new Error('no shard result files to merge');
  }
  if (inputPaths.length === 1) {
    fs.copyFileSync(inputPaths[0], outputPath);
    return;
  }
  const tempDir = fs.mkdtempSync(path.join(process.cwd(), 'coordinator-merge-'));
  try {
    let current = inputPaths.slice();
    let round = 0;
    while (current.length > 1) {
      const next = [];
      for (let i = 0; i < current.length; i += 128) {
        const batch = current.slice(i, i + 128);
        const batchOut = path.join(tempDir, `round_${round}_batch_${Math.floor(i / 128)}.poly`);
        mergePolyBatch(solver, batch, batchOut);
        next.push(batchOut);
      }
      current = next;
      round++;
    }
    fs.copyFileSync(current[0], outputPath);
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

function mergeRun(args) {
  if (!Number.isInteger(args.runId) || args.runId <= 0) {
    throw new Error('merge-run requires --run-id');
  }
  const db = openDb(args.db);
  const run = db.prepare('SELECT * FROM runs WHERE id = ?').get(args.runId);
  if (!run) {
    throw new Error(`run ${args.runId} not found`);
  }
  const stats = statsForRun(db, args.runId);
  if ((stats.queued ?? 0) !== 0 || (stats.leased ?? 0) !== 0 || (stats.failed ?? 0) !== 0) {
    throw new Error(
      `run ${args.runId} is not mergeable: queued=${stats.queued ?? 0}, leased=${stats.leased ?? 0}, failed=${stats.failed ?? 0}`
    );
  }
  const rows = db.prepare(`
    SELECT shard_index, result_poly
    FROM shards
    WHERE run_id = ? AND status = 'done' AND ok = 1
    ORDER BY shard_index
  `).all(args.runId);
  if (rows.length !== stats.total) {
    throw new Error(`run ${args.runId} has only ${rows.length}/${stats.total} successful shard results`);
  }
  for (const row of rows) {
    if (!row.result_poly) {
      throw new Error(`shard ${row.shard_index} has no stored poly result`);
    }
  }
  const outputPath = args.output ?? path.resolve(`run_${args.runId}.poly`);
  ensureParentDir(outputPath);
  const tempDir = fs.mkdtempSync(path.join(process.cwd(), 'coordinator-run-'));
  try {
    const inputPaths = writeTempPolyFiles(tempDir, rows);
    mergePolyFiles(run.solver, inputPaths, outputPath);
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
  console.log(JSON.stringify({
    ok: true,
    run_id: args.runId,
    output: outputPath,
    shard_count: rows.length,
  }, null, 2));
}

function main() {
  const { command, args } = parseArgs(process.argv);
  if (command === 'serve') {
    serve(args);
    return;
  }
  if (command === 'create-run') {
    createRun(args);
    return;
  }
  if (command === 'list-runs') {
    listRuns(args);
    return;
  }
  if (command === 'show-run') {
    showRun(args);
    return;
  }
  if (command === 'list-shards') {
    listShards(args);
    return;
  }
  if (command === 'reset-leases') {
    resetLeases(args);
    return;
  }
  if (command === 'merge-run') {
    mergeRun(args);
    return;
  }
  usage();
  process.exit(1);
}

main();
