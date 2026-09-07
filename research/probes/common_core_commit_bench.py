#!/usr/bin/env python3
"""Storage-only A/B replay of the measured shared-group workload.

GPU timings advance a virtual clock; no GPU results are computed here. Copied
residue payloads exercise identical serialization but do NOT represent new
sign ranges. All output journals are marked benchmark-only and never reduced.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import common_core_campaign as cc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=2048)
    args = parser.parse_args()
    cc.require(args.groups > 0, "positive group count required")
    envelope = json.loads(args.sample.read_text())
    payload = envelope["payload"]
    cc.require(cc.digest(payload) == envelope["sha256"], "sample digest mismatch")
    records = [json.loads(line) for line in args.log.read_text().splitlines()]
    header = next(r for r in records if r.get("kind") == "identity")
    cc.require(header["sample"] == envelope["sha256"] and records[-1]["kind"] == "done", "incomplete/mixed timing source")
    timings = {(r["gid"], r["pi"]): r for r in records if r.get("kind") == "timing" and r["repeat"] == 0}
    source = cc.Journal(args.journal)
    cc.require(source.identity["format"] == "common-core-benchmark-only-v1" and
               source.identity["sample"] == envelope["sha256"] and source.identity["repeat"] == 0,
               "incompatible replay journal")
    templates = []
    cells = [c for c in payload["bins"] if c["pool"]]
    rng = random.Random(488)
    for cell in rng.choices(cells, weights=[c["groups"] for c in cells], k=args.groups):
        sample = rng.choice(cell["samples"])
        gid = sample["gid"]
        meta = source.group(gid)
        fields = []
        for pi in range(max(meta["primes"])):
            rows = list(source.ranges(gid, pi, meta))
            cc.require(len(rows) == 1, "invalid source coverage")
            r = timings[gid, pi]
            small, large = r["measurements"]
            fields.append((rows[0][2], large["gpu"]/large["count"],
                           max(0.,small["wall"]-large["gpu"]*small["count"]/large["count"])))
        r = timings[gid, 0]
        templates.append((meta, fields, r["prepare_seconds"]+r["python_seconds"]))
    source.close()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    logical_reference = None
    # A/B/B/A reduces sensitivity to monotone storage warm-up or drift.
    for trial, batched in enumerate((False, True, True, False)):
        identity = dict(format="common-core-commit-replay-only-v1", sample=envelope["sha256"],
                        source=cc.file_digest(args.journal), controller=cc.file_digest(cc.__file__),
                        group_start=0, group_end=args.groups, trial=trial, batched=batched)
        path = args.output / f"trial-{trial}.sqlite"
        journal = cc.Journal(path, identity)
        clock = [0.]
        commits = [0]
        journal.db.set_trace_callback(lambda sql: commits.__setitem__(0, commits[0]+int(sql == "COMMIT")))
        inserts = ranges = 0
        overhead = 0.
        def timed(fn, *values):
            nonlocal overhead
            start = time.perf_counter()
            result = fn(*values)
            elapsed = time.perf_counter()-start
            overhead += elapsed
            clock[0] += elapsed
            return result
        with patch.object(cc.time, "monotonic", side_effect=lambda: clock[0]):
            with journal.batch() if batched else nullcontext():
                for gid, (meta, fields, prep) in enumerate(templates):
                    timed(journal.flush_if_due)
                    clock[0] += prep
                    timed(journal.put_group, gid, meta)
                    inserts += 1
                    for pi, (values, per_sign, setup) in enumerate(fields):
                        clock[0] += setup
                        for begin in range(0, meta["domain"], 1 << 20):
                            end = min(meta["domain"], begin+(1 << 20))
                            timed(journal.flush_if_due)
                            gpu = per_sign*(end-begin)
                            clock[0] += gpu
                            timed(journal.put_range, gid, pi, begin, end, meta, values, gpu, gpu)
                            inserts += 1
                            ranges += 1
                timed(journal.flush)  # include final flush in measured overhead
        journal.close()
        # Reopen after commit; compare every logical field against the other
        # policy. Only provenance/hash fields are intentionally different.
        reader = cc.Journal(path)
        logical = []
        for gid, (meta, fields, _) in enumerate(templates):
            cc.require(reader.group(gid) == meta, "group changed during replay")
            for pi in range(len(fields)):
                rows = list(reader.ranges(gid, pi, meta))
                cc.require(not list(cc.gaps(rows, meta["domain"], 1 << 20)), "replay coverage gap")
                logical.append([gid, pi, rows])
        signature = cc.digest(logical)
        cc.require(logical_reference is None or signature == logical_reference, "A/B payload mismatch")
        logical_reference = signature
        reader.close()
        row = dict(trial=trial, batched=batched, groups=args.groups, range_rows=ranges,
                   insert_rows=inserts, transactions=commits[0], journal_seconds=overhead,
                   virtual_total_seconds=clock[0], logical_sha256=signature)
        results.append(row)
        print(cc.encoded(row), flush=True)
    a = statistics.mean(r["journal_seconds"] for r in results if not r["batched"])
    b = statistics.mean(r["journal_seconds"] for r in results if r["batched"])
    population = sum(c["groups"] for c in cells)
    print(cc.encoded(dict(status="complete", exact_payload_parity="OK", groups_population=population,
                         immediate_seconds=a, batched_seconds=b, journal_speedup=a/b,
                         journal_time_reduction_percent=100*(1-b/a),
                         projected_shared_journal_hours=[a/args.groups*population/3600,
                                                         b/args.groups*population/3600],
                         caveat="storage replay with virtual GPU time, not a measured GPU end-to-end speedup")), flush=True)


if __name__ == "__main__":
    main()
