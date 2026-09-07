#!/usr/bin/env python3
"""Cost-balanced assignments and an explicit local persistent-queue runner.

Combine a full measured baseline with an audited selected-order replacement.
Retain baseline overhead per replaced child and use the paired storage replay
ratio for unchanged shared groups. Timings are forecasts, not solve results.
Build/verify/commands never launch workers; only the explicit run subcommand
does so. This tool does not provision cloud resources.
"""
import argparse
from array import array
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import shlex
import time

import common_core_campaign as cc

FORMAT = "common-core-manifest-v1"


def signature(cell):
    return (cell["core"], cell["pool"], *cell["schedule"])


def order(cell):
    return cell["core"] + (cell["pool"] - 3 if cell["pool"] else 0)


def load_projection(sample_path, projection_path, catalog):
    envelope = json.loads(sample_path.read_text())
    sample = envelope["payload"]
    cc.require(cc.digest(sample) == envelope["sha256"], "sample checksum mismatch")
    cc.require(sample["format"] == "common-core-steady-sample-v1" and
               sample["catalog"] == catalog.digest and sample["queries"] == catalog.count,
               "sample/catalog mismatch")
    projection = json.loads(projection_path.read_text())
    cc.require(projection["kind"] == "projection" and projection["sample"] == envelope["sha256"] and
               projection.get("selected_orders", []) == sample.get("selected_orders", []),
               "projection/sample mismatch")
    cells = {}
    cc.require(len(sample["bins"]) == len(projection["bins"]), "projection bin count")
    for source, measured in zip(sample["bins"], projection["bins"]):
        key = signature(source)
        cc.require(key == signature(measured) and source["groups"] == measured["groups"] and
                   source["groups"] > 0 and key not in cells, "projection census mismatch")
        cc.require(len(source["schedule"]) == 4 and source["schedule"][0] > 0 and
                   list(source["schedule"]) == sorted(source["schedule"], reverse=True), "invalid CRT schedule")
        for field in ("kernel_hours", "overhead_hours", "journal_hours"):
            cc.require(math.isfinite(measured[field]) and measured[field] >= 0, "invalid timing")
        cc.require(measured["journal_hours"] <= measured["overhead_hours"] + 1e-9,
                   "journal exceeds overhead")
        cells[key] = measured
    return sample, cells


def cost_model(baseline, replacement, orders, journal_ratio):
    cc.require(orders and len(set(orders)) == len(orders), "missing/duplicate replacement orders")
    cc.require(math.isfinite(journal_ratio) and 0 < journal_ratio <= 1, "invalid journal ratio")
    costs, population = {}, {}
    old_queries, old_overhead = Counter(), defaultdict(float)
    for key, cell in baseline.items():
        if order(cell) in orders:
            cc.require(cell["pool"] == 0 and cell["schedule"][0] == 1,
                       "replacement requires wholly independent baseline orders")
            old_queries[order(cell)] += cell["groups"]
            old_overhead[order(cell)] += cell["overhead_hours"]
            continue
        overhead = cell["overhead_hours"]
        if cell["pool"]:
            overhead -= cell["journal_hours"] * (1 - journal_ratio)
        costs[key] = 3600 * (cell["kernel_hours"] + overhead) / cell["groups"]
        population[key] = cell["groups"]
    new_queries = Counter()
    for key, cell in replacement.items():
        n = order(cell)
        cc.require(n in orders and old_queries[n] > 0 and key not in costs, "unexpected replacement stratum")
        children = cell["schedule"][0]
        new_queries[n] += cell["groups"] * children
        # Retain old host/checkpoint allowance per child, not the smaller
        # immediate-journal allowance measured for the grouped candidate.
        overhead = old_overhead[n] / old_queries[n] * children
        costs[key] = 3600 * (cell["kernel_hours"] / cell["groups"] + overhead)
        population[key] = cell["groups"]
    cc.require(new_queries == old_queries and set(new_queries) == set(orders),
               "replacement query populations differ")
    cc.require(all(math.isfinite(x) and x > 0 for x in costs.values()), "invalid modeled cost")
    return costs, population


def partition(costs, target_shards):
    """Near-target contiguous ranges; an oversized group remains indivisible."""
    cc.require(target_shards > 0 and len(costs) > 0, "invalid shard target")
    total = math.fsum(costs)
    target = total / target_shards
    tasks, start, seconds = [], 0, 0.

    def emit(end):
        nonlocal start, seconds
        tasks.append(dict(id=len(tasks), group_start=start, group_end=end, seconds=seconds))
        start, seconds = end, 0.

    for gid, value in enumerate(costs):
        cc.require(math.isfinite(value) and value > 0, "invalid group cost")
        if seconds and seconds + value > target and target - seconds < seconds + value - target:
            emit(gid)
        seconds += value
        if seconds >= target:
            emit(gid + 1)
    if start < len(costs):
        emit(len(costs))
    return tasks


def assign(tasks, workers):
    cc.require(workers > 0, "invalid worker count")
    queues = [dict(id=i, task_ids=[], seconds=0.) for i in range(workers)]
    for task in sorted(tasks, key=lambda t: (-t["seconds"], t["id"])):
        queue = min(queues, key=lambda w: (w["seconds"], w["id"]))
        task["worker"] = queue["id"]
        task["journal"] = f"task-{task['id']:04d}.sqlite"
        queue["task_ids"].append(task["id"])
        queue["seconds"] += task["seconds"]
    return queues


def validate(payload):
    cc.require(payload["format"] == FORMAT and payload["tasks"] and payload["workers"], "invalid manifest")
    tasks = payload["tasks"]
    end, journals = 0, set()
    for i, task in enumerate(tasks):
        cc.require(task["id"] == i and task["group_start"] == end and task["group_end"] > end,
                   "gap/overlap in group ownership")
        cc.require(math.isfinite(task["seconds"]) and task["seconds"] > 0, "invalid task cost")
        cc.require(task["journal"] == f"task-{i:04d}.sqlite" and task["journal"] not in journals,
                   "invalid/duplicate journal name")
        journals.add(task["journal"])
        end = task["group_end"]
    cc.require(end == payload["audit"]["groups"], "incomplete group coverage")
    seen = set()
    for i, worker in enumerate(payload["workers"]):
        cc.require(worker["id"] == i, "invalid worker ID")
        for tid in worker["task_ids"]:
            cc.require(0 <= tid < len(tasks) and tid not in seen and tasks[tid]["worker"] == i,
                       "duplicate/invalid task assignment")
            seen.add(tid)
        cc.require(math.isclose(worker["seconds"], math.fsum(tasks[t]["seconds"] for t in worker["task_ids"]),
                                abs_tol=1e-7), "worker cost mismatch")
    cc.require(len(seen) == len(tasks), "unassigned tasks")
    cc.require(math.isclose(payload["total_seconds"], math.fsum(t["seconds"] for t in tasks), abs_tol=1e-7),
               "total cost mismatch")


def build(args):
    started = time.monotonic()
    cc.require(not args.output.exists(), "output exists; use a fresh manifest path")
    catalog = cc.Catalog(args.catalog)
    plan = cc.Plan(args.plan, catalog)
    try:
        audit = plan.audit()
        print(cc.encoded(dict(status="audited", **audit)), flush=True)
        bs, baseline = load_projection(args.baseline_sample, args.baseline_projection, catalog)
        rs, replacement = load_projection(args.replacement_sample, args.replacement_projection, catalog)
        cc.require(not bs.get("selected_orders") and rs["plan"] == plan.digest and
                   rs["total_plan_groups"] == audit["groups"] and
                   rs["coefficient_sum"] == bs["coefficient_sum"] == audit["coefficient_sum"],
                   "incomplete baseline or replacement plan mismatch")
        replay = [json.loads(line) for line in args.storage_replay.read_text().splitlines()]
        cc.require(replay[-1].get("status") == "complete" and
                   replay[-1].get("exact_payload_parity") == "OK", "incomplete/unverified storage replay")
        ratio = replay[-1]["batched_seconds"] / replay[-1]["immediate_seconds"]
        model, expected = cost_model(baseline, replacement, rs["selected_orders"], ratio)
        population = Counter()
        seconds = array('d')
        for gid, parent, boundary, members in plan.groups():
            rows = [catalog[q] for q, _ in members]
            key = rows[0][0]
            unmatched = 2 * catalog.slack - ((key & cc.FULL).bit_count() - 2 * (key >> 60))
            core = 60 - ((key & cc.FULL) if len(members) == 1 else parent | boundary).bit_count() + unmatched
            sig = (core, boundary.bit_count(), *(sum(r[2] > pi for r in rows) for pi in range(4)))
            cc.require(sig in model, f"unmeasured stratum {sig}")
            population[sig] += 1
            seconds.append(model[sig])
            if (gid+1) % 1000000 == 0:
                print(cc.encoded(dict(status="costed", groups=gid+1, seconds=time.monotonic()-started)), flush=True)
        cc.require(population == expected and len(seconds) == audit["groups"], "measured/actual census mismatch")
        tasks = partition(seconds, args.target_shards)
        workers = assign(tasks, args.workers)
        sources = {name: cc.file_digest(getattr(args, name)) for name in
                   ("baseline_sample", "baseline_projection", "replacement_sample", "replacement_projection", "storage_replay")}
        payload = dict(format=FORMAT, catalog_path=str(args.catalog), plan_path=str(args.plan),
                       audit=audit, controller_sha256=cc.file_digest(cc.__file__),
                       worker_sha256=cc.file_digest(args.worker),
                       generator_sha256=cc.file_digest(__file__), sources=sources,
                       configuration=cc.CONFIG, primes=cc.PRIMES, target_shards=args.target_shards,
                       journal_ratio=ratio, total_seconds=math.fsum(seconds),
                       largest_group_seconds=max(seconds), tasks=tasks, workers=workers,
                       assumptions="Sampled RTX PRO 6000 costs; paired shared-journal ratio; old overhead per replaced child. Excludes repeated startup/audit, transfer, final reduction, interruptions and multi-GPU contention.")
        validate(payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as output:
            output.write(cc.encoded(dict(payload=payload, sha256=cc.digest(payload)))+'\n')
        print(cc.encoded(dict(status="complete", manifest=cc.digest(payload), tasks=len(tasks),
                              total_hours=payload["total_seconds"]/3600,
                              worker_hours=[w["seconds"]/3600 for w in workers],
                              largest_group_hours=max(seconds)/3600,
                              seconds=time.monotonic()-started)), flush=True)
    finally:
        plan.close()
        catalog.close()


def load_manifest(path):
    envelope = json.loads(path.read_text())
    payload = envelope["payload"]
    cc.require(cc.digest(payload) == envelope["sha256"], "manifest checksum mismatch")
    validate(payload)
    cc.require(payload["controller_sha256"] == cc.file_digest(cc.__file__) and
               payload["configuration"] == cc.CONFIG and payload["primes"] == list(cc.PRIMES),
               "controller/configuration changed; regenerate manifest")
    return payload, envelope['sha256']


def verify(args):
    payload, checksum = load_manifest(args.manifest)
    cc.require(cc.file_digest(args.worker) == payload['worker_sha256'], 'solver binary changed')
    catalog = cc.Catalog(Path(payload['catalog_path']))
    try:
        plan = cc.Plan(Path(payload['plan_path']), catalog)
        try:
            cc.require(catalog.digest == payload['audit']['catalog_sha256'] and
                       plan.digest == payload['audit']['plan_sha256'], 'manifest artifact binding mismatch')
        finally:
            plan.close()
    finally:
        catalog.close()
    print(cc.encoded(dict(status='manifest_verified', manifest=checksum)))


def commands(args):
    payload, checksum = load_manifest(args.manifest)
    cc.require(0 <= args.worker_id < len(payload["workers"]), "worker ID outside manifest")
    # One process audits once, indexes groups, and retains its worker across
    # tasks. A restart always reconstructs the audit/index from checked bytes.
    print("#!/bin/bash\nset -euo pipefail")
    print(shlex.join(['python3', 'tools/common_core_manifest.py', 'run',
                     '--manifest', str(args.manifest), '--worker-id', str(args.worker_id),
                     '--worker', str(args.worker), '--journal-dir', str(args.journal_dir)]))


def run_queue(args):
    payload, checksum = load_manifest(args.manifest)
    cc.require(0 <= args.worker_id < len(payload['workers']), 'worker ID outside manifest')
    cc.require(cc.file_digest(args.worker) == payload['worker_sha256'], 'solver binary changed')
    catalog = cc.Catalog(Path(payload['catalog_path']))
    plan = worker = None
    try:
        plan = cc.Plan(Path(payload['plan_path']), catalog)
        audit = plan.audit()
        cc.require(audit == payload['audit'], 'manifest artifact/audit binding mismatch')
        print(cc.encoded(dict(status='queue_audited', manifest=checksum, **audit)), flush=True)
        worker = cc.Worker(args.worker, args.cpu_reference)
        for tid in payload['workers'][args.worker_id]['task_ids']:
            # This checks file identities/stamps before reusing in-memory audit
            # state. No persisted sidecar can claim that an audit has passed.
            cc.require(plan.audit() == audit and cc.file_digest(args.worker) == payload['worker_sha256'] and
                       cc.file_digest(cc.__file__) == payload['controller_sha256'], 'queue inputs changed')
            task = payload['tasks'][tid]
            options = argparse.Namespace(**vars(args))
            options.group_start, options.group_end = task['group_start'], task['group_end']
            options.journal = args.journal_dir / checksum / task['journal']
            print(cc.encoded(dict(status='task_start', task=tid)), flush=True)
            if not cc.run(options, catalog, plan, audit, worker=worker):
                print(cc.encoded(dict(status='queue_checkpoint_limit_reached', task=tid)), flush=True)
                return
        print(cc.encoded(dict(status='queue_complete', manifest=checksum, worker=args.worker_id)), flush=True)
    finally:
        if worker:
            worker.close()
        if plan:
            plan.close()
        catalog.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build')
    for name in ('catalog', 'plan', 'baseline-sample', 'baseline-projection',
                 'replacement-sample', 'replacement-projection', 'storage-replay', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--target-shards', type=int, default=64)
    p.add_argument('--worker', type=Path, required=True, help='bind the tested solver binary')
    p = sub.add_parser('verify')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--worker', type=Path, required=True)
    p = sub.add_parser('commands')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--worker-id', type=int, required=True)
    p.add_argument('--worker', type=Path, required=True)
    p.add_argument('--journal-dir', type=Path, required=True)
    p = sub.add_parser('run', help='execute one audited queue with a persistent worker')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--worker-id', type=int, required=True)
    p.add_argument('--worker', type=Path, required=True)
    p.add_argument('--journal-dir', type=Path, required=True)
    p.add_argument('--chunk-terms', type=int, default=32768)
    p.add_argument('--checkpoint-terms', type=int, default=1 << 20)
    p.add_argument('--commit-ranges', type=int, default=32)
    p.add_argument('--commit-seconds', type=float, default=1.)
    p.add_argument('--max-checkpoints', type=int, default=0, help='stop the queue when a task hits this range limit')
    p.add_argument('--cpu-reference', action='store_true')
    args = parser.parse_args()
    if args.command == 'build':
        cc.require(args.workers > 0 and args.target_shards > 0, 'invalid worker/shard counts')
        build(args)
    elif args.command == 'commands':
        commands(args)
    elif args.command == 'run':
        run_queue(args)
    else:
        verify(args)


if __name__ == '__main__':
    main()
