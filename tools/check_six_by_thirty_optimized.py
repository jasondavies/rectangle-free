#!/usr/bin/env python3
"""Derive the endpoint from certified saved 6x29 results, without relabelling them."""
import argparse
import json
import math
from pathlib import Path

try:
    from . import reduce_six_by_twenty_nine_optimized as source
    from . import reduce_six_by_thirty_optimized as target
except ImportError:
    import reduce_six_by_twenty_nine_optimized as source
    import reduce_six_by_thirty_optimized as target


def check_from_6x29(directory):
    # The catalog test independently compares both CPU evaluators' reordered
    # matrices through exact term ranges. No v2 6x30 result files are forged
    # from these differently identified v2 6x29 records.
    old, new = source.CATALOG[3], target.CATALOG[0]
    for key in ("occupied", "vertices", "terms", "unmatched", "matching_bound_power"):
        if old[key] != new[key]:
            raise ValueError("source minor no longer matches endpoint minor")
    if old["coefficient"] != 540 or new["coefficient"] != 18:
        raise ValueError("unexpected symmetry multiplicities")
    result = source.reduce_results(sorted(directory.glob("*.result")))
    minor = int(result["matching_counts"]["3"])
    answer = 18 * minor * (1 << 30) * math.factorial(30)
    expected = 5813026373117572187494156438960699897545098374101961015296000000000
    if answer != expected:
        raise ValueError("endpoint identity differs from the historical 6x30 result")
    return {"status": "COMPLETE", "exact": True, "new_gpu_run": False,
            "source_geometry": "6x29", "source_query": 3,
            "source_catalog_sha256": source.CATALOG_SHA256,
            "source_solver_binary_sha256": result["solver_binary_sha256"],
            "minor": str(minor), "T_4(6,30)": str(answer)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-6x29", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(check_from_6x29(args.from_6x29), sort_keys=True))


if __name__ == "__main__":
    main()
