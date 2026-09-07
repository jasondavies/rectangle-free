#!/usr/bin/env python3
"""Stratified production-worker timing; output is NOT a campaign result.

Sample uniformly within exact (core, pool, CRT schedule) strata. Persist only
bounded test ranges in explicitly benchmark-tagged journals. No extrapolation
across an unmeasured stratum, and no credit for missing independent work.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import common_core_campaign as cc


def emit(value):
    print(cc.encoded(value), flush=True)


def census(args):
    catalog = cc.Catalog(args.catalog)
    plan = cc.Plan(args.plan, catalog)
    rng = random.Random(args.seed)
    bins = {}
    seen = bytearray(catalog.count)
    queries = coefficient = 0
    started = time.monotonic()
    try:
        for gid, parent, boundary, members in plan.groups():
            rows = [catalog[q] for q, _ in members]
            key = rows[0][0]
            used = (key & cc.FULL).bit_count()
            unmatched = 2 * catalog.slack - (used - 2 * (key >> 60))
            core = 60 - ((key & cc.FULL) if len(members) == 1 else parent | boundary).bit_count() + unmatched
            pool = boundary.bit_count()
            schedule = tuple(sum(row[2] > pi for row in rows) for pi in range(4))
            signature = (core, pool, *schedule)
            for (qid, removed), row in zip(members, rows):
                cc.require(not seen[qid], "duplicate ownership")
                seen[qid] = 1
                queries += 1
                coefficient += row[1]
            order = 60-used+unmatched
            if args.orders and order not in args.orders:
                continue
            cell = bins.setdefault(signature, dict(core=core, pool=pool, schedule=schedule,
                                                    groups=0, samples=[]))
            cell["groups"] += 1
            slot = len(cell["samples"]) if len(cell["samples"]) < args.samples else rng.randrange(cell["groups"])
            if slot < args.samples:
                sample = dict(gid=gid, parent=parent, boundary=boundary,
                              members=members, rows=rows)
                if slot == len(cell["samples"]):
                    cell["samples"].append(sample)
                else:
                    cell["samples"][slot] = sample
            if gid % 1000000 == 0:
                emit(dict(scanned_groups=gid, seconds=time.monotonic()-started))
        cc.require(queries == catalog.count, "incomplete ownership")
        cc.require(bins, "no workload matches the requested orders")
        payload = dict(format="common-core-steady-sample-v1", catalog=catalog.digest,
                       plan=plan.digest, slack=catalog.slack, queries=queries,
                       coefficient_sum=coefficient, seed=args.seed, samples_per_bin=args.samples,
                       total_plan_groups=gid+1, selected_orders=args.orders,
                       census_seconds=time.monotonic()-started, bins=list(bins.values()))
        args.output.write_text(cc.encoded(dict(payload=payload, sha256=cc.digest(payload))) + "\n")
        emit(dict(bins=len(bins), groups=sum(c["groups"] for c in bins.values()), queries=queries,
                  sampled_groups=sum(len(c["samples"]) for c in bins.values()),
                  tail_signs=sum(c["groups"] * (1 << (c["core"]//2-1)) * sum(x>0 for x in c["schedule"])
                                 for c in bins.values() if c["pool"] == 0)))
    finally:
        plan.close()
        catalog.close()


class SampleCatalog:
    def __init__(self, sample, slack):
        self.slack = slack
        self.rows = {q: row for (q, _), row in zip(sample["members"], sample["rows"])}

    def __getitem__(self, q):
        return self.rows[q]


def load_sample(path):
    envelope = json.loads(path.read_text())
    cc.require(cc.digest(envelope["payload"]) == envelope["sha256"], "sample checksum mismatch")
    cc.require(envelope["payload"]["format"] == "common-core-steady-sample-v1", "sample format")
    return envelope["payload"], envelope["sha256"]


def range_begin(domain, count, gid, seed, chunk):
    """A reproducible aligned window, including high Gray-index ranges."""
    if not seed:
        return 0
    return random.Random((seed << 32) + gid).randrange((domain-count)//chunk+1)*chunk


def bench(args):
    payload, sample_hash = load_sample(args.sample)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    worker = cc.Worker(args.worker, False)
    emit(dict(kind="identity", sample=sample_hash, worker=worker.binary_hash,
              controller=cc.file_digest(cc.__file__), benchmark=cc.file_digest(__file__),
              startup_seconds=time.monotonic()-started, chunk=args.chunk, large=args.large,
              repeats=args.repeats, range_seed=args.range_seed))
    expected = {}
    try:
        # Each pass has its own journal; overlapping A/B ranges are never
        # presented as additive campaign coverage. Production refuses this format.
        for repeat in range(args.repeats):
            identity = dict(format="common-core-benchmark-only-v1", sample=sample_hash,
                            worker=worker.binary_hash, repeat=repeat,
                            group_start=0, group_end=payload.get("total_plan_groups", sum(c["groups"] for c in payload["bins"])))
            journal = cc.Journal(args.output / f"pass-{repeat}.sqlite", identity)
            try:
                cells = list(enumerate(payload["bins"]))
                random.Random(487 + repeat).shuffle(cells)
                for bid, cell in cells:
                    for sample in cell["samples"]:
                        cycle = time.monotonic()
                        catalog = SampleCatalog(sample, payload["slack"])
                        meta = cc.expected_meta(catalog, sample["parent"], sample["boundary"], sample["members"])
                        python_seconds = time.monotonic() - cycle
                        if cell["pool"] == 0:
                            message = f"prepare_single {catalog.slack} {sample['rows'][0][0]}"
                        else:
                            message = f"prepare {catalog.slack} {sample['parent']} {sample['boundary']} {len(sample['rows'])} "
                            message += " ".join(f"{row[0]} {removed}" for (_, removed), row in zip(sample["members"], sample["rows"]))
                        start = time.monotonic()
                        reply = worker.request(message)
                        prepare_seconds = time.monotonic() - start
                        want = ["prepared", str(meta["domain"]), str(len(sample["rows"]))]
                        for bound, count in zip(meta["bounds"], meta["primes"]):
                            want += [str(bound), str(count)]
                        cc.require(reply == want, "metadata parity failure")
                        start = time.monotonic()
                        journal.put_group(sample["gid"], meta)
                        group_journal_seconds = time.monotonic()-start
                        for pi in range(max(meta["primes"])):
                            # Warm once per active prime set, but measure warmup
                            # separately: production must pay it once per group.
                            measurements = []
                            begin = range_begin(meta["domain"], min(args.large, meta["domain"]),
                                                sample["gid"], args.range_seed, args.chunk)
                            for count in (min(32768, meta["domain"]), min(args.large, meta["domain"])):
                                start = time.monotonic()
                                reply = worker.request(f"run {pi} {begin} {count} {args.chunk}")
                                wall = time.monotonic()-start
                                active = cell["schedule"][pi]
                                cc.require(reply[0] == "result" and len(reply) == 3+active and int(reply[2]) == active,
                                           "malformed measured result")
                                seconds, residues = float(reply[1]), list(map(int, reply[3:]))
                                cc.require(seconds > 0 and all(0 <= x < cc.PRIMES[pi] for x in residues), "bad timing/residue")
                                sig = sample["gid"], pi, count
                                cc.require(sig not in expected or expected[sig] == residues, "repeat residue mismatch")
                                expected[sig] = residues
                                measurements.append(dict(begin=begin, count=count, gpu=seconds, wall=wall))
                            start = time.monotonic()
                            journal.put_range(sample["gid"], pi, begin, begin+count, meta, residues, seconds, wall)
                            # Exercise the same read/checksum path used on resume.
                            cc.require(len(list(journal.ranges(sample["gid"], pi, meta))) == 1, "checkpoint readback")
                            journal_seconds = time.monotonic()-start
                            emit(dict(kind="timing", repeat=repeat, bin=bid, gid=sample["gid"], pi=pi,
                                      domain=meta["domain"], python_seconds=python_seconds,
                                      prepare_seconds=prepare_seconds, group_journal_seconds=group_journal_seconds,
                                      journal_seconds=journal_seconds, measurements=measurements))
                        emit(dict(kind="group_done", repeat=repeat, gid=sample["gid"], wall=time.monotonic()-cycle))
            finally:
                journal.close()
        emit(dict(kind="done", seconds=time.monotonic()-started, repeat_residues="OK"))
    finally:
        worker.close()


def project(args):
    payload, sample_hash = load_sample(args.sample)
    records = [json.loads(line) for line in args.log.read_text().splitlines()]
    identities = [r for r in records if r.get("kind") == "identity"]
    cc.require(len(identities) == 1, "incomplete/mixed run identity")
    identity = identities[0]
    cc.require(identity["sample"] == sample_hash and sum(r.get("kind") == "done" for r in records) == 1, "incomplete/mixed run")
    timings = defaultdict(list)
    for r in records:
        if r.get("kind") == "timing":
            timings[r["bin"], r["gid"], r["pi"]].append(r)
    totals = defaultdict(float)
    journal_total = 0.
    cells_out = []
    for bid, cell in enumerate(payload["bins"]):
        estimates = []
        journal_estimates = []
        for sample in cell["samples"]:
            gpu = overhead = journal_cost = 0.
            for pi, active in enumerate(cell["schedule"]):
                if not active:
                    continue
                rows = timings[bid, sample["gid"], pi]
                cc.require(len(rows) == identity["repeats"] and
                           {r["repeat"] for r in rows} == set(range(identity["repeats"])),
                           "missing/duplicate stratum measurement")
                def estimate(r):
                    small, large = r["measurements"]
                    cc.require(r["domain"] == 1 << (cell["core"]//2-1) and
                               0 < small["count"] <= large["count"] <= r["domain"], "bad measured domain")
                    kernel = large["gpu"] * r["domain"] / large["count"]
                    # Treat excess first-call latency as a one-off initialization
                    # allowance; do not extrapolate it once per chunk.
                    setup = max(0., small["wall"] - large["gpu"]*small["count"]/large["count"])
                    checkpoints = (r["domain"] + args.checkpoint - 1)//args.checkpoint
                    journal = checkpoints*r["journal_seconds"]
                    host = setup + checkpoints*(max(0.,large["wall"]-large["gpu"])+r["journal_seconds"])
                    if pi == 0:
                        host += r["python_seconds"]+r["prepare_seconds"]+r["group_journal_seconds"]
                        journal += r["group_journal_seconds"]
                    return kernel, host, journal
                values = [estimate(r) for r in rows]
                gpu += statistics.mean(v[0] for v in values)
                overhead += statistics.mean(v[1] for v in values)
                journal_cost += statistics.mean(v[2] for v in values)
            estimates.append((gpu, overhead))
            journal_estimates.append(journal_cost)
        kind = "shared" if cell["pool"] else "tail"
        g = cell["groups"]
        kernel = g * statistics.mean(x[0] for x in estimates)/3600
        host = g * statistics.mean(x[1] for x in estimates)/3600
        journal = g * statistics.mean(journal_estimates)/3600
        journal_total += journal
        totals[kind+"_kernel_hours"] += kernel
        totals[kind+"_overhead_hours"] += host
        cells_out.append(dict(bin=bid, core=cell["core"], pool=cell["pool"], schedule=cell["schedule"], groups=g,
                              kernel_hours=kernel, overhead_hours=host,
                              journal_hours=journal,
                              sample_min_hours=g*min(sum(x) for x in estimates)/3600,
                              sample_max_hours=g*max(sum(x) for x in estimates)/3600))
    emit(dict(kind="projection", sample=sample_hash, checkpoint=args.checkpoint, **totals,
              selected_orders=payload.get("selected_orders", []),
              total_hours=sum(totals.values()),
              journal_hours_included=journal_total,
              sample_envelope_hours=[sum(c["sample_min_hours"] for c in cells_out),
                                     sum(c["sample_max_hours"] for c in cells_out)],
              caveat="sample envelope is not a confidence interval; excludes audit/reduction/provisioning and interruption losses",
              bins=cells_out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("census")
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=487)
    p.add_argument("--orders", type=int, nargs="*", default=[], help="project only these residual orders; still audit all ownership")
    p = sub.add_parser("bench")
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--worker", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunk", type=int, default=32768)
    p.add_argument("--large", type=int, default=262144)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--range-seed", type=int, default=0,
                   help="nonzero selects aligned windows throughout each Gray domain")
    p = sub.add_parser("project")
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--checkpoint", type=int, default=1 << 20)
    args = parser.parse_args()
    if args.command == "census":
        cc.require(args.samples > 0, "positive sample count required")
        cc.require(all(42 <= n <= 66 and n % 2 == 0 for n in args.orders), "invalid residual order")
        census(args)
    elif args.command == "bench":
        cc.require(0 < args.chunk <= 1 << 20 and args.large >= 32768 and args.repeats >= 2, "invalid timing parameters")
        bench(args)
    else:
        cc.require(args.checkpoint > 0, "positive checkpoint required")
        project(args)


if __name__ == "__main__":
    main()
