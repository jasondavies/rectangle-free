#!/usr/bin/env python3
"""Validate residual-hafnian shards and reconstruct the exact T_4(6,29)."""

from __future__ import annotations

import argparse
import hashlib
import math
from collections import defaultdict
from pathlib import Path


FORMAT = "six-by-twenty-nine-hafnian-v1"
ALGORITHM = "glynn-trace-hessenberg-residual-cuda-v1"
CATALOG_SHA256 = "a2c9f8d9ef2cf9e35502189a713d59b3175e585316227af3609b9a7c417611a8"
QUERY_COUNT = 29


def read_result(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    payload = b""
    for line_with_ending in path.read_text().splitlines(keepends=True):
        line = line_with_ending.rstrip("\r\n")
        key, separator, value = line.partition(" ")
        if not separator or key in fields:
            raise ValueError(f"{path}: malformed or duplicate field: {line!r}")
        fields[key] = value
        if key != "result_payload_sha256":
            payload += line_with_ending.encode()
    required = {
        "format", "algorithm", "rows", "columns", "catalog_sha256",
        "query_id", "query_sha256", "occupied_tokens", "defect_count",
        "excess", "unmatched_tokens", "defect_coefficient", "vertices",
        "solver_binary_sha256", "prime", "begin", "end", "total_terms",
        "partial_glynn_sum", "status", "result_payload_sha256",
    }
    if not required <= fields.keys():
        raise ValueError(f"{path}: missing fields {sorted(required-fields.keys())}")
    if fields["format"] != FORMAT or fields["algorithm"] != ALGORITHM:
        raise ValueError(f"{path}: incompatible format/algorithm")
    if (fields["rows"], fields["columns"], fields["catalog_sha256"]) != (
        "6", "29", CATALOG_SHA256
    ):
        raise ValueError(f"{path}: geometry or catalog mismatch")
    if fields["status"] != "complete":
        raise ValueError(f"{path}: incomplete status")
    if hashlib.sha256(payload).hexdigest() != fields["result_payload_sha256"]:
        raise ValueError(f"{path}: payload digest mismatch")
    return fields


def crt(residues: list[tuple[int, int]]) -> tuple[int, int]:
    value, modulus = 0, 1
    for prime, residue in residues:
        if math.gcd(modulus, prime) != 1:
            raise ValueError(f"moduli are not coprime at {prime}")
        correction = ((residue-value) % prime) * pow(modulus, -1, prime) % prime
        value += modulus*correction
        modulus *= prime
    return value, modulus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--allow-partial-primes", action="store_true")
    args = parser.parse_args()

    by_job: dict[tuple[int, int], list[tuple[int, int, int, Path]]] = defaultdict(list)
    metadata: dict[int, tuple[str, int, int, int, int, int, int]] = {}
    for path in args.results:
        fields = read_result(path)
        prime, query = int(fields["prime"]), int(fields["query_id"])
        begin, end = int(fields["begin"]), int(fields["end"])
        residue = int(fields["partial_glynn_sum"])
        vertices = int(fields["vertices"])
        total_terms = int(fields["total_terms"])
        expected_terms = 1 << (vertices//2-1)
        if not 0 <= query < QUERY_COUNT or vertices not in (54, 56, 58, 62):
            raise ValueError(f"{path}: invalid query/order")
        if total_terms != expected_terms or not 0 <= begin < end <= total_terms:
            raise ValueError(f"{path}: invalid term range")
        if not 0 <= residue < prime:
            raise ValueError(f"{path}: invalid residue")
        item = (
            fields["query_sha256"], int(fields["occupied_tokens"]),
            int(fields["defect_count"]), int(fields["excess"]),
            int(fields["unmatched_tokens"]), int(fields["defect_coefficient"]),
            vertices,
        )
        if query in metadata and metadata[query] != item:
            raise ValueError(f"{path}: inconsistent query metadata")
        metadata[query] = item
        by_job[prime, query].append((begin, end, residue, path))

    if set(metadata) != set(range(QUERY_COUNT)) and not args.allow_partial_primes:
        raise ValueError("result set does not contain all 29 query identities")
    sector_counts: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    for _, _, defects, excess, unmatched, coefficient, vertices in metadata.values():
        if unmatched != 2-excess or vertices != 62-2*defects-2*excess:
            raise ValueError("inconsistent defect/query dimensions")
        sector_counts[excess, defects][0] += 1
        sector_counts[excess, defects][1] += coefficient
    expected_sectors = {(0, 0): [1, 1], (1, 1): [2, 840],
                        (2, 1): [1, 1440], (2, 2): [25, 303660]}
    if dict(sector_counts) != expected_sectors and not args.allow_partial_primes:
        raise ValueError(f"defect sector mismatch: {dict(sector_counts)}")

    primes = sorted({prime for prime, _ in by_job})
    complete: list[tuple[int, int]] = []
    for prime in primes:
        contributions = 0
        complete_queries = 0
        for query in range(QUERY_COUNT):
            shards = sorted(by_job.get((prime, query), []))
            if not shards:
                continue
            cursor = 0
            signed_sum = 0
            total_terms = 1 << (metadata[query][6]//2-1)
            for begin, end, residue, path in shards:
                if begin != cursor:
                    raise ValueError(
                        f"prime {prime}, query {query}: gap/overlap at {cursor}, next {begin} ({path})"
                    )
                cursor = end
                signed_sum = (signed_sum+residue) % prime
            if cursor != total_terms:
                continue
            _, _, defects, excess, unmatched, coefficient, vertices = metadata[query]
            augmented_matching = signed_sum * pow(total_terms, -1, prime) % prime
            matching_count = augmented_matching * pow(math.factorial(unmatched), -1, prime) % prime
            contribution = coefficient % prime
            contribution = contribution * pow(2, 29-defects, prime) % prime
            contribution = contribution * matching_count % prime
            contributions = (contributions+contribution) % prime
            complete_queries += 1
        if complete_queries != QUERY_COUNT:
            if args.allow_partial_primes:
                print(
                    f"HAFNIAN_6X29_REDUCE prime={prime} complete_queries={complete_queries}/29 status=PARTIAL"
                )
                continue
            raise ValueError(f"prime {prime}: only {complete_queries}/29 complete queries")
        answer = contributions * math.factorial(29) % prime
        complete.append((prime, answer))
        print(
            f"HAFNIAN_6X29_REDUCE prime={prime} queries=29 packing_mod={contributions} "
            f"T_mod={answer} status=COMPLETE"
        )

    if not complete:
        if args.allow_partial_primes:
            print("HAFNIAN_6X29_CRT status=PARTIAL complete_primes=0")
            return 0
        raise ValueError("no completely covered prime")
    answer, modulus = crt(complete)
    # For each defect sector, bound the residual k-matchings by the number in
    # the complete graph on the same number of remaining original vertices.
    sectors = ((1, 0, 60, 29), (840, 1, 57, 28),
               (1440, 1, 56, 28), (303660, 2, 54, 27))
    packing_bound = sum(
        coefficient * (1 << (29-defects))
        * math.factorial(vertices)
        // (math.factorial(vertices-2*edges) * (1 << edges) * math.factorial(edges))
        for coefficient, defects, vertices, edges in sectors
    )
    bound = math.factorial(29) * packing_bound
    print(
        f"HAFNIAN_6X29_CRT primes={len(complete)} modulus_bits={modulus.bit_length()} "
        f"answer={answer} answer_bits={answer.bit_length()}"
    )
    if modulus > bound:
        print(f"HAFNIAN_6X29_RESULT T_4(6,29)={answer} exact=OK")
    else:
        print(
            f"HAFNIAN_6X29_RESULT status=MODULAR_ONLY modulus_bits={modulus.bit_length()} "
            f"required_complete_graph_bound_bits={bound.bit_length()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
