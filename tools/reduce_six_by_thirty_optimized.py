#!/usr/bin/env python3
"""Certified CRT of one 58-vertex minor, then restore 18 * 2^30 * 30!."""
import argparse
import json
from pathlib import Path

try:
    from .reduce_residual_hafnian import ResidualReducer
except ImportError:
    from reduce_residual_hafnian import ResidualReducer

WIDTH = 30
QUERY_COUNT = 1
FORMAT = "six-by-thirty-hafnian-v2"
ALGORITHM = "glynn-gray-resolvent-fixed-field-cuda-v7"
CATALOG_SHA256 = "cb4bc5458e88cc41cf4d8bed042906680accf24bdcb37d5d68d225a013f07744"
PRIMES = (2147483647, 2147483629, 2147483587)
CATALOG = [{
    "id": 0, "occupied": "1125904201809920", "defects": 0, "excess": 0,
    "unmatched": 0, "coefficient": 18, "vertices": 58, "terms": 1 << 28,
    "matching_bound_power": 85,
    "digest": "be1ea46dab7e26b5107bf1bc4aa392668c4cb0bd97c11f09cdd5bd504b61c338",
}]

_reducer = ResidualReducer(WIDTH, FORMAT, ALGORITHM, CATALOG_SHA256, CATALOG, PRIMES)
required_prime_count = _reducer.required_prime_count
read_result = _reducer.read_result
reduce_results = _reducer.reduce_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="*", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    paths = args.results + (sorted(args.directory.glob("*.result")) if args.directory else [])
    result = reduce_results(paths, args.allow_partial)
    print(json.dumps(result, sort_keys=True))
    if result["status"] == "COMPLETE":
        print(f"HAFNIAN_6X30_RESULT T_4(6,30)={result['T_4(6,30)']} exact=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
