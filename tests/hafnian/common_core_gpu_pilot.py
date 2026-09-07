#!/usr/bin/env python3
"""Bounded real 6x28 shared+independent CUDA campaign integration pilot.

Run on a single GPU after the local tests. Uses existing ignored catalog/plan
artifacts; never generates a corpus or launches a whole grid campaign.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import common_core_campaign as cc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, default=ROOT / "build/hafnian_common_worker")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-check-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    worker = cc.Worker(args.worker, False)
    try:
        started = time.monotonic()
        reply = worker.request("check_tail")
        cc.require(reply == ["checked", "252"], "GPU tail/reference gate failed")
        print(cc.encoded(dict(tail_reference_ranges=252, primes=4, orders=[42,48,50,58,62,64,66],
                              seconds=time.monotonic()-started, exact="OK")), flush=True)
    finally:
        worker.close()
    if args.self_check_only:
        return
    catalog = cc.Catalog(ROOT / "build/common-core-6x28.catalog")
    plan = cc.Plan(ROOT / "build/common-core-6x28-order50-481.plan", catalog)
    try:
        audit = plan.audit()
        shared = next(gid for gid, _, _, m in plan.groups() if len(m) > 1)
        singles = [(cc.expected_meta(catalog, p, b, m)["domain"], gid)
                   for gid, p, b, m in plan.groups() if len(m) == 1]
        independent = min(singles)[1]
        for name, gid in (("shared", shared), ("independent", independent)):
            options = argparse.Namespace(group_start=gid, group_end=gid+1,
                chunk_terms=32768, checkpoint_terms=65536, max_checkpoints=1,
                journal=args.output / f"{name}.sqlite", cpu_reference=False, worker=args.worker)
            started = time.monotonic()
            cc.run(options, catalog, plan, audit)
            # Deliberate process stop after the first committed range; new
            # worker resumes the remaining ranges with a larger checkpoint.
            options.max_checkpoints=0
            options.checkpoint_terms=1<<20
            cc.run(options, catalog, plan, audit)
            elapsed = time.monotonic()-started
            cc.run(options, catalog, plan, audit)  # must not submit any work
            journal = cc.Journal(options.journal)
            count = journal.db.execute("SELECT count(*) FROM ranges").fetchone()[0]
            meta = journal.group(gid)
            for pi in range(max(meta["primes"])):
                cc.require(not list(cc.gaps(journal.ranges(gid,pi,meta),meta["domain"],1<<20)), "missing coverage")
            journal.close()
            print(cc.encoded(dict(pilot=name, group=gid, checkpoints=count,
                                  end_to_end_seconds=elapsed, restart="OK", coverage="OK")), flush=True)
        cc.reduce(argparse.Namespace(journals=[args.output/"shared.sqlite", args.output/"independent.sqlite"],
                  cpu_reference=False, query_results=True, require_complete=False), catalog, plan, audit)
    finally:
        plan.close()
        catalog.close()


if __name__ == "__main__":
    main()
