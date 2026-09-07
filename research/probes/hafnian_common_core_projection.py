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
    parser.add_argument("--pad-pool", type=int, choices=(13,),
                        help="hypothetical uniform padding of pool-11 groups; not an audited replacement plan")
    parser.add_argument("--order", type=int, help="project only this residual order after validating full histogram coverage")
    parser.add_argument("timings", nargs="+", type=Path)
    args = parser.parse_args()
    bins = {}
    for row in records(args.plan, "CORE_PLAN_BIN"):
        k = key(row)
        if args.pad_pool and k[2]:
            n, c, pool, g, prime = k
            if pool != 11 or c < 4:
                raise ValueError("padding projection requires only pool-11 grouped bins")
            k = n, c - 2, 13, g, prime
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
    if args.order:
        bins = {k: v for k, v in bins.items() if k[0] == args.order}
        if not bins:
            raise ValueError("selected order absent from plan")
    samples = defaultdict(list)
    identities = set()
    configurations = set()
    for path in args.timings:
        for configuration in records(path, "CORE_CUDA_CONFIG"):
            if int(configuration.get("profile", "0")):
                raise ValueError("instrumented phase timings cannot project campaign runtime")
            configurations.add(tuple(int(configuration.get(k, "0")) for k in (
                "hess", "boundary", "scratch", "warp_poly", "sparse_moments", "boundary_order", "threads",
                "inverse_chain", "live_moments", "sync_clear"))
                + (int(configuration.get("max_pool", "11")),))
            if len(configurations) > 1:
                raise ValueError("cannot combine different A/B configurations in one projection")
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
          f"selected_order={args.order or 'all'} "
          + ("scope=uniform_padding_hypothesis_grouped_kernels_only" if args.pad_pool else "scope=grouped_kernels_only"))
    if missing:
        raise ValueError(f"unmeasured grouped bins: {missing[:10]}")


if __name__ == "__main__":
    main()
