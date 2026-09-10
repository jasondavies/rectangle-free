#!/usr/bin/env python3
"""Split one independent hafnian query into durable, exact sign-range tasks.

This opt-in tool does not alter whole-group campaign manifests. Its reduction
certifies one matching count, NOT a complete grid count. Do not add partial
range results to a grid total. Provisioning and queue assignment are external.
"""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time

import common_core_campaign as cc

FORMAT = "common-core-sign-shards-v1"


def split_domain(domain, parts):
    cc.require(type(parts) is int and 0 < parts <= domain, "invalid part count")
    return [[domain * i // parts, domain * (i + 1) // parts] for i in range(parts)]


def validate(payload):
    cc.require(payload["format"] == FORMAT, "unknown sign-shard format")
    meta = payload["meta"]
    cc.require(len(meta["primes"]) == len(meta["bounds"]) == 1 and
               1 <= meta["primes"][0] <= len(cc.PRIMES), "not an independent query")
    domain = meta["domain"]
    cc.require(type(domain) is int and 0 < domain <= 1 << 32 and not domain & (domain - 1),
               "invalid sign domain")
    tasks = payload["tasks"]
    cc.require(tasks and len(tasks) <= domain, "invalid task count")
    cursor = 0
    for i, task in enumerate(tasks):
        cc.require(task["id"] == i and type(task["begin"]) is int and type(task["end"]) is int and
                   task["begin"] == cursor < task["end"] <= domain,
                   "gap/overlap in sign ownership")
        cursor = task["end"]
    cc.require(cursor == domain, "incomplete sign ownership")
    cc.require(type(payload["group"]) is int and payload["group"] >= 0 and
               type(payload["query"]) is int and payload["query"] >= 0, "invalid query/group")
    cc.require(payload["backend"] in ("cuda", "cpu-reference"), "invalid backend")


def load(path):
    envelope = json.loads(path.read_text())
    payload = envelope["payload"]
    cc.require(envelope["sha256"] == cc.digest(payload), "manifest checksum mismatch")
    validate(payload)
    cc.require(payload["controller"] == cc.file_digest(cc.__file__) and
               payload["runner"] == cc.file_digest(__file__) and
               payload["configuration"] == cc.CONFIG and payload["primes"] == list(cc.PRIMES),
               "controller/configuration changed; regenerate manifest")
    return payload


@contextmanager
def artifacts(catalog_path, plan_path):
    catalog = cc.Catalog(catalog_path)
    plan = None
    try:
        plan = cc.Plan(plan_path, catalog)
        audit = plan.audit()
        yield catalog, plan, audit
    finally:
        if plan:
            plan.close()
        catalog.close()


def resolve(catalog, plan, gid):
    group = next(plan.groups(gid, gid + 1), None)
    cc.require(group is not None and group[0] == gid, "group outside plan")
    _, parent, boundary, members = group
    cc.require(len(members) == 1 and not parent and not boundary and not members[0][1],
               "sign sharding currently requires an independent query")
    return members[0][0], cc.expected_meta(catalog, parent, boundary, members)


def bound_artifacts(payload, catalog, plan, audit):
    cc.require(payload["catalog"] == catalog.digest and payload["plan"] == plan.digest and
               payload["audit"] == audit, "manifest artifact binding mismatch")
    query, meta = resolve(catalog, plan, payload["group"])
    cc.require(query == payload["query"] and meta == payload["meta"], "manifest query metadata mismatch")


def identity(payload, tid):
    cc.require(type(tid) is int and 0 <= tid < len(payload["tasks"]), "task outside manifest")
    return dict(format=FORMAT, manifest=cc.digest(payload), task=payload["tasks"][tid],
                group_start=payload["group"], group_end=payload["group"] + 1,
                solver_binary=payload["solver_binary"], backend=payload["backend"])


def checked_records(journal, payload, tid):
    cc.require(journal.identity == identity(payload, tid), "journal provenance/task mismatch")
    groups = list(journal.ordered_groups())
    cc.require(len(groups) <= 1, "unexpected journal group")
    if not groups:
        return [[] for _ in cc.PRIMES]
    gid, meta, records = groups[0]
    cc.require(gid == payload["group"] and meta == payload["meta"], "journal metadata mismatch")
    task = payload["tasks"][tid]
    for rows in records:
        for begin, end, _ in rows:
            cc.require(task["begin"] <= begin < end <= task["end"], "range outside task ownership")
    return records


def missing(rows, begin, end, limit):
    cc.require(limit > 0, "invalid checkpoint size")
    cursor = begin
    for a, b, _ in [*rows, (end, end, None)]:
        cc.require(cursor <= a <= b <= end, "range outside/overlapping task")
        while cursor < a:
            stop = min(a, cursor + limit)
            yield cursor, stop
            cursor = stop
        cursor = b


def solve(payload, tid, catalog, worker, journal, chunk, checkpoint, max_checkpoints=0):
    records = checked_records(journal, payload, tid)
    task, meta = payload["tasks"][tid], payload["meta"]
    return solve_interval(payload['group'], payload['query'], meta, task['begin'], task['end'],
                          records, catalog, worker, journal, chunk, checkpoint, max_checkpoints)


def solve_interval(gid, query, meta, begin, end, records, catalog, worker, journal,
                   chunk, checkpoint, max_checkpoints=0):
    """Execute a prevalidated independent interval; shared by both queue tools."""
    prepared = False
    completed = 0
    for pi in range(meta["primes"][0]):
        for start, stop in missing(records[pi], begin, end, checkpoint):
            journal.flush_if_due()
            if not prepared:
                reply = worker.request(f"prepare_single {catalog.slack} {catalog[query][0]}")
                expected = ["prepared", str(meta["domain"]), "1", str(meta["bounds"][0]), str(meta["primes"][0])]
                cc.require(reply == expected, "CPU/worker metadata mismatch")
                journal.put_group(gid, meta)
                prepared = True
            started = time.monotonic()
            reply = worker.request(f"run {pi} {start} {stop-start} {chunk}")
            cc.require(len(reply) == 4 and reply[0] == "result" and reply[2] == "1", "malformed worker result")
            seconds, value = float(reply[1]), int(reply[3])
            cc.require(math.isfinite(seconds) and seconds >= 0 and 0 <= value < cc.PRIMES[pi],
                       "invalid worker result")
            journal.put_range(gid, pi, start, stop, meta, [value], seconds, time.monotonic()-started)
            completed += 1
            print(cc.encoded(dict(checkpoint=completed,durable=not journal.db.in_transaction,
                                  group=gid,prime=cc.PRIMES[pi],begin=start,end=stop,
                                  compute_seconds=seconds)),flush=True)
            if max_checkpoints and completed >= max_checkpoints:
                return False
    return True


def build(args):
    with artifacts(args.catalog, args.plan) as (catalog, plan, audit):
        query, meta = resolve(catalog, plan, args.group)
        payload = dict(format=FORMAT, catalog=catalog.digest, plan=plan.digest, audit=audit,
                       controller=cc.file_digest(cc.__file__), runner=cc.file_digest(__file__),
                       solver_binary=cc.file_digest(args.worker), configuration=cc.CONFIG,
                       backend="cpu-reference" if args.cpu_reference else "cuda", primes=list(cc.PRIMES),
                       group=args.group, query=query, meta=meta,
                       tasks=[dict(id=i, begin=a, end=b) for i, (a, b) in enumerate(split_domain(meta["domain"], args.parts))])
        validate(payload)
        # Exclusive creation: a failed write fails checksum validation on load.
        with args.output.open('x') as output:
            output.write(cc.encoded(dict(payload=payload, sha256=cc.digest(payload))) + '\n')
        print(cc.encoded(dict(status="manifest_created", manifest=cc.digest(payload),
                              query=query, group=args.group, domain=meta["domain"], tasks=args.parts)))


def run(args, payload, catalog):
    cc.require(cc.file_digest(args.worker) == payload["solver_binary"], "worker binary changed")
    cc.require(0 < args.chunk_terms <= 1 << 20 and args.checkpoint_terms > 0 and args.max_checkpoints >= 0,
               "invalid chunk/checkpoint limit")
    ident = identity(payload, args.task)
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    with cc.claim(args.journal):
        journal = worker = None
        try:
            journal = cc.Journal(args.journal, ident)
            worker = cc.Worker(args.worker, payload["backend"] == "cpu-reference")
            with journal.batch(args.commit_ranges, args.commit_seconds):
                complete = solve(payload, args.task, catalog, worker, journal, args.chunk_terms,
                                 args.checkpoint_terms, args.max_checkpoints)
            print(cc.encoded(dict(status="task_complete" if complete else "checkpoint_limit_reached",
                                  task=args.task, committed_ranges=journal.committed_ranges)))
        finally:
            if worker:
                worker.close()
                worker.process.stdin.close()
                worker.process.stdout.close()
            if journal:
                journal.close()


def collect(payload, paths):
    records = [[] for _ in cc.PRIMES]
    seen = set()
    for path in paths:
        journal = cc.Journal(path)
        try:
            tid = journal.identity["task"]["id"]
            cc.require(tid not in seen, "duplicate task journal (choose one snapshot per task)")
            seen.add(tid)
            checked = checked_records(journal, payload, tid)
            for pi, rows in enumerate(checked):
                records[pi].extend(rows)
        finally:
            journal.close()
    return records


def reduce(args, payload, catalog):
    records = collect(payload, args.journals)
    value = cc.reduce_group(payload["meta"], records, [catalog[payload["query"]][0]], catalog.slack)[0]
    print(cc.encoded(dict(status="complete_query" if value is not None else "partial_query",
                          manifest=cc.digest(payload), query=payload["query"],
                          matching_count=None if value is None else str(value),
                          scope="One matching query only; not a grid count or campaign journal.")))
    cc.require(not args.require_complete or value is not None, "incomplete query; no count certified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "run", "reduce", "verify"):
        p = commands.add_parser(command)
        p.add_argument('--catalog', type=Path, required=True)
        p.add_argument('--plan', type=Path, required=True)
        if command != "build":
            p.add_argument('--manifest', type=Path, required=True)
        if command in ("build", "run"):
            p.add_argument('--worker', type=Path, required=True)
        if command == "build":
            p.add_argument('--group', type=int, required=True)
            p.add_argument('--parts', type=int, default=8)
            p.add_argument('--output', type=Path, required=True)
            p.add_argument('--cpu-reference', action='store_true')
        elif command == "run":
            p.add_argument('--task', type=int, required=True)
            p.add_argument('--journal', type=Path, required=True)
            p.add_argument('--chunk-terms', type=int, default=32768)
            p.add_argument('--checkpoint-terms', type=int, default=1 << 20)
            p.add_argument('--max-checkpoints', type=int, default=0)
            p.add_argument('--commit-ranges', type=int, default=32)
            p.add_argument('--commit-seconds', type=float, default=1.)
        elif command == "reduce":
            p.add_argument('--journals', type=Path, nargs='+', required=True)
            p.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    if args.command == 'build':
        build(args)
        return
    payload = load(args.manifest)
    with artifacts(args.catalog, args.plan) as (catalog, plan, audit):
        bound_artifacts(payload, catalog, plan, audit)
        if args.command == 'run':
            run(args, payload, catalog)
        elif args.command == 'reduce':
            reduce(args, payload, catalog)
        else:
            print(cc.encoded(dict(status="manifest_verified", manifest=cc.digest(payload))))


if __name__ == '__main__':
    main()
