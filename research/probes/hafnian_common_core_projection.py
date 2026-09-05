#!/usr/bin/env python3
"""Project grouped kernel work only; never turn omitted fallback work into zero."""
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean


def records(path, prefix):
    for line in path.read_text().splitlines():
        if line.startswith(prefix + " "):
            yield dict(x.split("=", 1) for x in line.split()[1:])


def key(row):
    return tuple(int(row[x]) for x in ("order", "core", "pool", "active_queries", "prime_index"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("timings", nargs="+", type=Path)
    args = parser.parse_args()
    bins = {}
    for row in records(args.plan, "CORE_PLAN_BIN"):
        k = key(row)
        if k in bins:
            raise ValueError("duplicate plan bin")
        bins[k] = int(row["groups"])
    if not bins:
        raise ValueError("no plan bins")
    done = list(records(args.plan, "CORE_PLAN_DONE"))
    if len(done) != 1 or done[0]["exact_once"] != "OK":
        raise ValueError("missing complete plan summary")
    original = sum(g * k[3] * (1 << (k[0] // 2 - 1)) for k, g in bins.items())
    if original != int(done[0]["original_signs"]):
        raise ValueError("histogram loses original adaptive work")
    samples = defaultdict(list)
    identities = set()
    for path in args.timings:
        for row in records(path, "CORE_CUDA_SWEEP"):
            if row["exact"] != "OK":
                raise ValueError("unvalidated timing")
            k = key(row)
            if k not in bins or k[2] == 0:
                raise ValueError("timed group is not a grouped plan bin")
            identity = k, int(row["root"])
            if identity in identities:
                raise ValueError("duplicate timing sample")
            identities.add(identity)
            seconds = float(row["kernel_s"])
            if seconds <= 0 or int(row["signs"]) <= 0:
                raise ValueError("invalid timing")
            samples[k].append(seconds / int(row["signs"]))
    totals = defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0])
    missing = []
    for k, groups in sorted(bins.items()):
        n, c, pool, active, prime = k
        original = groups * active * (1 << (n // 2 - 1))
        if pool == 0:
            totals[n][4] += original
            continue
        if k not in samples:
            missing.append(k)
            continue
        values = samples[k]
        work = groups * (1 << (c // 2 - 1))
        for i, rate in enumerate((mean(values), min(values), max(values))):
            totals[n][i] += work * rate
        totals[n][3] += original
    for n, (seconds, low, high, covered, fallback) in sorted(totals.items()):
        print(f"CORE_PROJECT_ORDER order={n} grouped_gpu_hours={seconds/3600:.6f} "
              f"sample_min_hours={low/3600:.6f} sample_max_hours={high/3600:.6f} "
              f"covered_original_signs={covered} fallback_original_signs={fallback}")
    covered = sum(v[3] for v in totals.values())
    fallback = sum(v[4] for v in totals.values())
    print("CORE_PROJECT_TOTAL "
          f"grouped_gpu_hours={sum(v[0] for v in totals.values())/3600:.6f} "
          f"sample_min_hours={sum(v[1] for v in totals.values())/3600:.6f} "
          f"sample_max_hours={sum(v[2] for v in totals.values())/3600:.6f} "
          f"covered_original_fraction={covered/(covered+fallback):.9f} "
          f"fallback_original_signs={fallback} missing_bins={len(missing)} "
          "scope=grouped_kernels_only")
    if missing:
        raise ValueError(f"unmeasured grouped bins: {missing[:10]}")


if __name__ == "__main__":
    main()
