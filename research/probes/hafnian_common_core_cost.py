#!/usr/bin/env python3
"""Export measured per-shape sign costs for a heuristic, not a GPU forecast.

Interpolation is performed by the planner; a changed plan must be measured
again. Repeated timing sweeps are intentionally averaged, not deduplicated.
"""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("timings", type=Path, nargs="+")
    args = parser.parse_args()
    costs = defaultdict(list)
    sources = []
    for path in args.timings:
        raw = path.read_bytes()
        config = None
        seen = set()
        for line in raw.decode().splitlines():
            if line.startswith("CORE_CUDA_CONFIG "):
                config = dict(x.split("=", 1) for x in line.split()[1:])
                if any(int(config[k]) != 1 for k in ("hess", "boundary", "scratch")) or int(config["profile"]):
                    raise ValueError("cost model requires uninstrumented accepted-kernel timings")
                if any(int(config.get(k, 0)) for k in ("warp_poly", "sparse_moments", "boundary_order")):
                    raise ValueError("HCCOST01 cannot describe experimental kernel candidates")
                if int(config.get("threads", 256)) != 256 or int(config.get("max_pool", 11)) != 11:
                    raise ValueError("HCCOST01 requires the original 256-thread pool-11 configuration")
            elif line.startswith("CORE_CUDA_SWEEP "):
                if config is None:
                    raise ValueError("missing kernel configuration")
                row = dict(x.split("=", 1) for x in line.split()[1:])
                key = tuple(int(row[k]) for k in ("order", "core", "pool", "active_queries", "prime_index"))
                identity = key, int(row["root"])
                if identity in seen or row["exact"] != "OK":
                    raise ValueError("duplicate or unvalidated timing")
                seen.add(identity)
                value = float(row["kernel_s"]) / int(row["signs"])
                if not 0 < value <= .001:
                    raise ValueError("invalid/out-of-gate timing")
                costs[key].append(value * 10**12)
        if not seen:
            raise ValueError("empty timing source")
        sources.append(hashlib.sha256(raw).hexdigest())
    with args.output.open("x") as output:
        output.write("HCCOST01 hess=1 boundary=1 scratch=1\n")
        for digest in sources:
            output.write(f"# source_sha256 {digest}\n")
        for key, values in sorted(costs.items()):
            output.write(" ".join(map(str, (*key, round(mean(values)), len(values)))) + "\n")
    print(f"CORE_COST_TABLE bins={len(costs)} samples={sum(map(len, costs.values()))} scope=heuristic_model")


if __name__ == "__main__":
    main()
