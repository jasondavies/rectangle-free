#!/usr/bin/env python3
"""Audit preserved assignments and every new 52/54 embedding, on CPU only."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import common_core_campaign as cc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("catalog", "source", "candidate", "worker"):
        p.add_argument("--"+name, type=Path, required=True)
    args = p.parse_args()
    catalog = cc.Catalog(args.catalog)
    old, new = cc.Plan(args.source, catalog), cc.Plan(args.candidate, catalog)
    worker = cc.Worker(args.worker, True)
    try:
        target = set()
        candidates = iter(new.groups())
        preserved = 0
        for gid, parent, boundary, members in old.groups():
            if len(members) == 1:
                key = catalog[members[0][0]][0]
                n = 60+2*catalog.slack-2*((key & cc.FULL).bit_count()-(key>>60))
                if n in (52, 54):
                    target.add(members[0][0])
                    continue
            got = next(candidates)
            cc.require(got[1:] == (parent, boundary, members), "existing assignment changed")
            preserved += 1
        seen = set()
        shared = single = covered = 0
        for gid, parent, boundary, members in candidates:
            for qid, removed in members:
                cc.require(qid in target and qid not in seen, "new ownership not restricted to old target singletons")
                seen.add(qid)
            meta = cc.expected_meta(catalog, parent, boundary, members)
            if len(members) == 1:
                cc.require(parent == boundary == members[0][1] == 0, "invalid singleton")
                single += 1
                continue
            message = f"prepare {catalog.slack} {parent} {boundary} {len(members)} "
            message += " ".join(f"{catalog[q][0]} {r}" for q, r in members)
            reply = worker.request(message)
            expected = ["prepared", str(meta["domain"]), str(len(members))]
            for b, n in zip(meta["bounds"], meta["primes"]):
                expected += [str(b), str(n)]
            cc.require(reply == expected, "new group embedding/bound mismatch")
            shared += 1
            covered += len(members)
        cc.require(seen == target, "target coverage lost")
        print(cc.encoded(dict(preserved_groups=preserved, target_queries=len(target),
                             added_groups=shared, grouped_queries=covered, independent=single,
                             all_new_maps="OK", ownership="OK", source=old.digest, candidate=new.digest)))
    finally:
        worker.close()
        old.close()
        new.close()
        catalog.close()


if __name__ == "__main__":
    main()
