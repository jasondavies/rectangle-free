#!/usr/bin/env python3
"""Exact per-query CRT for the 33-query, monomer-decomposed 6x29 campaign."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

try:
    from .reduce_six_by_twenty_nine_hafnian import crt
except ImportError:
    from reduce_six_by_twenty_nine_hafnian import crt

WIDTH = 29
QUERY_COUNT = 33
FORMAT = "six-by-twenty-nine-hafnian-v2"
ALGORITHM = "glynn-gray-resolvent-fixed-field-cuda-v7"
CATALOG_SHA256 = "b0738ab5e49081ace71fc0a63201ae4374e31ea035ab65bd4c1bc002b75319a6"
PRIMES = (2147483647, 2147483629, 2147483587)
CATALOG = json.loads(Path(__file__).with_name(
    "six_by_twenty_nine_optimized_catalog.json").read_text())


def required_prime_count(bound_power):
    modulus = 1
    for count, prime in enumerate(PRIMES, 1):
        modulus *= prime
        if modulus > 1 << bound_power:
            return count
    raise ValueError("prime schedule does not cover certified bound")


def read_result(path):
    raw = path.read_bytes()
    fields = {}
    payload = b""
    for line in raw.splitlines(keepends=True):
        key, separator, value = line.decode().rstrip("\n").partition(" ")
        if not separator or key in fields:
            raise ValueError(f"{path}: malformed/duplicate result field")
        fields[key] = value
        if key != "result_payload_sha256":
            payload += line
    if hashlib.sha256(payload).hexdigest() != fields.get("result_payload_sha256"):
        raise ValueError(f"{path}: payload digest mismatch")
    expected = {"format": FORMAT, "algorithm": ALGORITHM, "rows": "6",
                "columns": "29", "catalog_sha256": CATALOG_SHA256, "status": "complete"}
    if any(fields.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{path}: incompatible result provenance")
    query = int(fields["query_id"])
    if not 0 <= query < QUERY_COUNT:
        raise ValueError(f"{path}: invalid query")
    item = CATALOG[query]
    mapping = {"query_sha256": "digest", "occupied_tokens": "occupied",
               "defect_count": "defects", "excess": "excess",
               "unmatched_tokens": "unmatched", "defect_coefficient": "coefficient",
               "vertices": "vertices", "total_terms": "terms",
               "matching_bound_power": "matching_bound_power"}
    if any(fields.get(key) != str(item[value]) for key, value in mapping.items()):
        raise ValueError(f"{path}: query differs from certified catalog")
    if not re.fullmatch(r"[0-9a-f]{64}", fields.get("solver_binary_sha256", "")):
        raise ValueError(f"{path}: invalid solver digest")
    prime = int(fields["prime"])
    if (prime not in PRIMES or not 0 <= int(fields["partial_glynn_sum"]) < prime
            or not 0 <= int(fields["begin"]) < int(fields["end"]) <= item["terms"]):
        raise ValueError(f"{path}: invalid prime/range/residue")
    # Both successful Gray chains and exact whole-chunk fallbacks use global
    # Gray indices. Historical binary-order v1 pieces cannot be mixed in.
    if fields.get("matrix_stride") != str(item["vertices"] + 1):
        raise ValueError(f"{path}: incorrect matrix stride")
    if fields.get("gray_enabled") != "1" or fields.get("gray_chain") != "7":
        raise ValueError(f"{path}: unexpected term-order/backend metadata")
    return fields


def reduce_results(paths, allow_partial=False):
    jobs = defaultdict(list)
    binaries = set()
    for path in paths:
        fields = read_result(path)
        binaries.add(fields["solver_binary_sha256"])
        jobs[int(fields["query_id"]), int(fields["prime"])].append(
            (int(fields["begin"]), int(fields["end"]), int(fields["partial_glynn_sum"])))
    if len(binaries) > 1:
        raise ValueError("mixed solver binaries in verification campaign")
    counts = {}
    for item in CATALOG:
        residues = []
        for prime in PRIMES:
            cursor = signed_sum = 0
            for begin, end, value in sorted(jobs[item["id"], prime]):
                if begin != cursor:
                    raise ValueError(f"query {item['id']} prime {prime}: gap/overlap at {cursor}")
                cursor = end
                signed_sum = (signed_sum + value) % prime
            if cursor == item["terms"]:
                augmented = signed_sum * pow(item["terms"], -1, prime) % prime
                residues.append((prime, augmented * pow(math.factorial(item["unmatched"]), -1, prime) % prime))
        value, modulus = crt(residues)
        if modulus <= 1 << item["matching_bound_power"]:
            continue
        if value > 1 << item["matching_bound_power"]:
            raise ValueError(f"query {item['id']}: matching count exceeds certified bound")
        counts[item["id"]] = value
    if len(counts) != QUERY_COUNT:
        if allow_partial:
            return {"status": "PARTIAL", "resolved": len(counts), "queries": QUERY_COUNT}
        raise ValueError(f"only {len(counts)}/{QUERY_COUNT} queries have complete certified CRT coverage")
    packing = sum(item["coefficient"] * (1 << (29-item["defects"])) * counts[item["id"]]
                  for item in CATALOG)
    answer = packing * math.factorial(29)
    return {"status": "COMPLETE", "exact": True, "T_4(6,29)": str(answer),
            "catalog_sha256": CATALOG_SHA256, "solver_binary_sha256": next(iter(binaries)),
            "matching_counts": {str(k): str(v) for k, v in counts.items()}}


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
        print(f"HAFNIAN_6X29_RESULT T_4(6,29)={result['T_4(6,29)']} exact=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
