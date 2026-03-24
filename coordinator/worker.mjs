#!/usr/bin/env node

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { spawn } from 'node:child_process';

function usage() {
  console.error(
    'usage: node coordinator/worker.mjs --server URL --worker-id ID [--run-id ID] [--poll-seconds N] [--heartbeat-seconds N] [--once]'
  );
}

function parseArgs(argv) {
  const args = {
    server: null,
    workerId: `${os.hostname()}-${process.pid}`,
    runId: null,
    pollSeconds: 10,
    heartbeatSeconds: 300,
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
      case '--run-id':
        args.runId = Number(argv[++i]);
        break;
      case '--poll-seconds':
        args.pollSeconds = Number(argv[++i]);
        break;
      case '--heartbeat-seconds':
        args.heartbeatSeconds = Number(argv[++i]);
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
  if (args.runId != null && (!Number.isInteger(args.runId) || args.runId <= 0)) {
    throw new Error('--run-id must be a positive integer');
  }
  if (!Number.isFinite(args.heartbeatSeconds) || args.heartbeatSeconds < 0) {
    throw new Error('--heartbeat-seconds must be non-negative');
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

async function runShard(serverUrl, workerId, shard, heartbeatSeconds) {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'rectpoly-worker-'));
  const polyPath = path.join(tempDir, `shard_${shard.shard_id}.poly`);
  const cmd = [
    shard.solver,
    ...shard.solver_args,
    '--poly-out',
    polyPath,
  ];
  const started = Date.now();
  const proc = spawn(cmd[0], cmd.slice(1), {
    stdio: ['ignore', 'pipe', 'pipe'],
    env: {
      ...process.env,
      OMP_NUM_THREADS: String(shard.omp_threads),
      RECT_PROGRESS_STEP: process.env.RECT_PROGRESS_STEP ?? '1000000',
    },
  });
  let stdout = '';
  let stderr = '';
  proc.stdout.setEncoding('utf8');
  proc.stderr.setEncoding('utf8');
  proc.stdout.on('data', (chunk) => {
    stdout += chunk;
  });
  proc.stderr.on('data', (chunk) => {
    stderr += chunk;
  });

  let heartbeatTimer = null;
  if (heartbeatSeconds > 0) {
    heartbeatTimer = setInterval(async () => {
      try {
        await postJson(new URL('/renew-lease', serverUrl), {
          worker_id: workerId,
          shard_id: shard.shard_id,
          lease_token: shard.lease_token,
        });
      } catch (err) {
        console.error(`lease renew failed for shard ${shard.shard_id}: ${String(err?.message ?? err)}`);
      }
    }, heartbeatSeconds * 1000);
  }

  const exitCode = await new Promise((resolve, reject) => {
    proc.on('error', reject);
    proc.on('close', (code) => resolve(code));
  });
  if (heartbeatTimer) {
    clearInterval(heartbeatTimer);
  }
  const elapsedSeconds = (Date.now() - started) / 1000;
  const resultPoly = fs.existsSync(polyPath) ? fs.readFileSync(polyPath, 'utf8') : null;
  fs.rmSync(tempDir, { recursive: true, force: true });
  return {
    ok: exitCode === 0,
    solver_exit_code: exitCode,
    elapsed_seconds: elapsedSeconds,
    worker_seconds: parseNumber(/Worker Complete in\s+([0-9.]+)\s+seconds\./, stdout),
    prefix_seconds: parseNumber(/Prefix generation:\s+([0-9.]+)\s+seconds/, stdout),
    stdout,
    stderr,
    result_poly: resultPoly,
    last_error: exitCode === 0 ? null : `solver exited with status ${exitCode}`,
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
      run_id: args.runId,
    });
    if (!fetched.work_available) {
      if (args.once) return;
      await sleep(args.pollSeconds * 1000);
      continue;
    }
    const { shard } = fetched;
    console.log(
      `running shard db_id=${shard.shard_id} run=${shard.run_id} shard=${shard.shard_index} offset=${shard.task_offset}/${shard.task_stride}`
    );
    const result = await runShard(args.server, args.workerId, shard, args.heartbeatSeconds);
    await postJson(new URL('/submit-result', args.server), {
      worker_id: args.workerId,
      shard_id: shard.shard_id,
      lease_token: shard.lease_token,
      ...result,
    });
    console.log(
      `submitted shard db_id=${shard.shard_id} run=${shard.run_id} shard=${shard.shard_index}: ok=${result.ok} elapsed=${result.elapsed_seconds.toFixed(2)}s`
    );
    if (args.once) return;
  }
}

main().catch((err) => {
  console.error(String(err?.stack ?? err));
  process.exit(1);
});
