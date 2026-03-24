#!/usr/bin/env node

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';

function usage() {
  console.error(
    'usage: node coordinator/worker.mjs --server URL --worker-id ID [--poll-seconds N] [--once]'
  );
}

function parseArgs(argv) {
  const args = {
    server: null,
    workerId: `${os.hostname()}-${process.pid}`,
    pollSeconds: 10,
    once: false,
  };
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    switch (arg) {
      case '--help':
      case '-h':
        usage();
        process.exit(0);
        break;
      case '--server':
        args.server = argv[++i];
        break;
      case '--worker-id':
        args.workerId = argv[++i];
        break;
      case '--poll-seconds':
        args.pollSeconds = Number(argv[++i]);
        break;
      case '--once':
        args.once = true;
        break;
      default:
        throw new Error(`unknown argument: ${arg}`);
    }
  }
  if (!args.server) {
    throw new Error('--server is required');
  }
  return args;
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  const payload = await res.json();
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}: ${JSON.stringify(payload)}`);
  }
  return payload;
}

function parseNumber(re, text) {
  const match = text.match(re);
  return match ? Number(match[1]) : null;
}

function runShard(shard) {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'rectpoly-worker-'));
  const polyPath = path.join(tempDir, `shard_${shard.shard_id}.poly`);
  const cmd = [
    shard.solver,
    ...shard.solver_args,
    '--poly-out',
    polyPath,
  ];
  const started = Date.now();
  const proc = spawnSync(cmd[0], cmd.slice(1), {
    encoding: 'utf8',
    env: {
      ...process.env,
      OMP_NUM_THREADS: String(shard.omp_threads),
      RECT_PROGRESS_STEP: process.env.RECT_PROGRESS_STEP ?? '1000000',
    },
  });
  const elapsedSeconds = (Date.now() - started) / 1000;
  const stdout = proc.stdout ?? '';
  const stderr = proc.stderr ?? '';
  const resultPoly = fs.existsSync(polyPath) ? fs.readFileSync(polyPath, 'utf8') : null;
  fs.rmSync(tempDir, { recursive: true, force: true });
  return {
    ok: proc.status === 0,
    solver_exit_code: proc.status,
    elapsed_seconds: elapsedSeconds,
    worker_seconds: parseNumber(/Worker Complete in\s+([0-9.]+)\s+seconds\./, stdout),
    prefix_seconds: parseNumber(/Prefix generation:\s+([0-9.]+)\s+seconds/, stdout),
    stdout,
    stderr,
    result_poly: resultPoly,
    last_error: proc.status === 0 ? null : `solver exited with status ${proc.status}`,
  };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main() {
  const args = parseArgs(process.argv);
  while (true) {
    const fetched = await postJson(new URL('/fetch-work', args.server), {
      worker_id: args.workerId,
    });
    if (!fetched.work_available) {
      if (args.once) return;
      await sleep(args.pollSeconds * 1000);
      continue;
    }
    const { shard } = fetched;
    console.log(`running shard ${shard.shard_id} run=${shard.run_id} offset=${shard.task_offset}/${shard.task_stride}`);
    const result = runShard(shard);
    await postJson(new URL('/submit-result', args.server), {
      worker_id: args.workerId,
      shard_id: shard.shard_id,
      lease_token: shard.lease_token,
      ...result,
    });
    console.log(
      `submitted shard ${shard.shard_id}: ok=${result.ok} elapsed=${result.elapsed_seconds.toFixed(2)}s`
    );
    if (args.once) return;
  }
}

main().catch((err) => {
  console.error(String(err?.stack ?? err));
  process.exit(1);
});
