#!/usr/bin/env python3
"""Audit local common-core probes; never write production result files.

Sample coverage refers only to the union of logged sibling groups, not to
the full 6x27 query population. Complete verification uses independently
checksummed full-Glynn campaign results and their own sign normalization.
"""
import argparse
from collections import defaultdict
import importlib.util
from pathlib import Path


def records(path, prefix):
    for line in path.read_text().splitlines():
        if line.startswith(prefix + " "):
            yield dict(word.split("=", 1) for word in line.split()[1:])


def coverage(path):
    sectors = defaultdict(list)
    for item in records(path, "CORE27_GROUP"):
        if int(item["cap"]) not in (7, 9, 11):
            continue
        members = {int(v.split(":")[0]) for v in item["members"].strip(",").split(",")}
        assert len(members) == int(item["queries"])
        sectors[tuple(int(item[k]) for k in ("e", "d", "cap"))].append(members)
    for (e, d, cap), groups in sorted(sectors.items()):
        union = set().union(*groups)
        owned = set()
        assigned = []
        while groups:
            # Exact greedy set assignment, deterministic ties in log order.
            chosen = max(range(len(groups)), key=lambda i: len(groups[i] - owned))
            fresh = groups.pop(chosen) - owned
            if len(fresh) < 2:
                break
            assert not fresh & owned
            assigned.append(len(fresh))
            owned.update(fresh)
        print(f"CORE_SAMPLE_COVER e={e} d={d} cap={cap} union={len(union)} "
              f"grouped={len(owned)} groups={len(assigned)} singletons={len(union-owned)} "
              f"mean={len(owned)/len(assigned):.6f} "
              f"coverage={len(owned)/len(union):.6f} scope=logged_sibling_union_only")


def verify(path, directory):
    module_path = Path(__file__).resolve().parents[2] / "tools/reduce_six_by_twenty_eight_hafnian.py"
    spec = importlib.util.spec_from_file_location("campaign_reducer", module_path)
    reducer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reducer)
    items = list(records(path, "CORE_COMPLETE"))
    if not items:
        raise ValueError("no complete queries")
    seen = set()
    for item in items:
        query, prime = int(item["query_id"]), int(item["prime"])
        if (query, prime) in seen:
            raise ValueError("duplicate complete identity")
        seen.add((query, prime))
        paths = list(directory.glob(f"p{prime}-q{query:05d}-b*.result"))
        if not paths:
            raise ValueError(f"no saved results for {query}/{prime}")
        shards = [reducer.read_result(p) for p in paths]
        shards.sort(key=lambda f: int(f["begin"]))
        cursor, signed_sum = 0, 0
        for f in shards:
            if (int(f["occupied_tokens"]) != int(item["occupied"]) or
                    f["query_sha256"] != item["query_digest"] or
                    int(f["prime"]) != prime or int(f["query_id"]) != query):
                raise ValueError("query identity mismatch")
            end, begin, total = (int(f[k]) for k in ("end", "begin", "total_terms"))
            if begin != cursor or end <= begin or end > total:
                raise ValueError("range gap/overlap")
            if total != 1 << (int(f["vertices"]) // 2 - 1):
                raise ValueError("invalid full-Glynn domain")
            cursor = end
            signed_sum = (signed_sum + int(f["partial_glynn_sum"])) % prime
        if cursor != total:
            raise ValueError("incomplete independent reference")
        expected = signed_sum * pow(total, -1, prime) % prime
        if expected != int(item["augmented_hafnian"]):
            raise ValueError(f"complete hafnian mismatch query {query}: {expected} versus {item}")
        print(f"CORE_COMPLETE_PARITY query={query} prime={prime} residue={expected} exact=OK")
    print(f"CORE_COMPLETE_PARITY_SUMMARY queries={len(items)} checksums=OK full_ranges=OK exact=OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-coverage", type=Path)
    parser.add_argument("--complete", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--control-batch", type=Path,
                        help="emit a fresh production-control batch for these complete query IDs")
    args = parser.parse_args()
    if args.sample_coverage:
        coverage(args.sample_coverage)
    if args.complete:
        if args.control_batch:
            rows = list(records(args.complete, "CORE_COMPLETE"))
            if not rows:
                raise ValueError("no complete query IDs for control batch")
            with args.control_batch.open("x") as out:
                for row in rows:
                    q, p = int(row["query_id"]), int(row["prime"])
                    out.write(f"{q} {p} 0 0 control/p{p}-q{q:05d}-b0.result\n")
        elif not args.results:
            parser.error("--complete requires --results")
        if args.results:
            verify(args.complete, args.results)
