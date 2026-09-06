#!/usr/bin/env python3
"""Bounded, sequential GPU A/B driver; never provisions cloud resources.

Supply separately compiled control/hess/boundary/scratch/hb/all and optional
profile-control/profile-all binaries. Explicitly set all three CORE_OPT_*
macros for each binary: the default now enables all three optimisations.
Every solver run checks sampled signs
against its independent CPU formula. Instrumented runs are diagnostic only.
"""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["control", "hess", "boundary", "scratch", "hb", "all"])
    parser.add_argument("--full", action="store_true", help="all three prime sweeps, rather than bounded pilot")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    plan = args.plan.resolve()
    if not args.full:
        # Highest-population stratum per order/core/pool/query-size class;
        # deterministic representatives, not a campaign-time estimator.
        bins = {}
        groups = []
        for line in plan.read_text().splitlines():
            if line.startswith(("CORE_PLAN_BIN ", "CORE27_GROUP ")):
                row = dict(x.split("=", 1) for x in line.split()[1:])
                if int(row["prime_index"]) != 0:
                    continue
                if line.startswith("CORE_PLAN_BIN "):
                    bins[tuple(int(row[k]) for k in ("order", "core", "pool", "active_queries"))] = int(row["groups"])
                else:
                    n = 66 - 2 * int(row["e"]) - 2 * int(row["d"])
                    key = n, int(row["core"]), int(row["cap"]), int(row["queries"])
                    groups.append((key, line))
        chosen = {}
        for key, line in groups:
            n, c, q, g = key
            coarse = n, c, q, min(g, 4) if g < 8 else 8
            score = bins.get(key, 0)
            if coarse not in chosen or score > chosen[coarse][0]:
                chosen[coarse] = score, line
        if not chosen:
            raise ValueError("no pilot samples in plan")
        plan = args.output.resolve() / "pilot.log"
        plan.write_text("\n".join(v[1] for k, v in sorted(chosen.items())) + "\n")
        print(f"CORE_AB_PILOT cases={len(chosen)} scope=unweighted_diagnostic", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="8", OMP_DYNAMIC="FALSE")
    primes = (2147483647, 2147483629, 2147483587) if args.full else (2147483647,)
    for repeat in range(args.repeats):
        # Reverse order on alternating passes to expose warm-up/drift effects.
        variants = args.variants if repeat % 2 == 0 else args.variants[::-1]
        for name in variants:
            for prime in primes:
                target = args.output / f"{name}-r{repeat}-p{prime}.log"
                with target.open("x") as log:
                    subprocess.run([
                        str((args.binaries / name).resolve()), "--sweep", "--groups", str(plan),
                        "--prime", str(prime), "--count", "32768", "--threads", "256",
                    ], check=True, env=env, stdout=log, stderr=log)
                print(f"CORE_AB_COMPLETE variant={name} repeat={repeat} prime={prime} log={target}", flush=True)


if __name__ == "__main__":
    main()
