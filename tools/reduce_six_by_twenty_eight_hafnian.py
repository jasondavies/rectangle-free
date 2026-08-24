#!/usr/bin/env python3
"""Validate residual-hafnian shards and reconstruct exact T_4(6,28)."""

from __future__ import annotations

import argparse
import hashlib
import math
from collections import defaultdict
from pathlib import Path


FORMAT = "six-by-twenty-eight-hafnian-v2"
ALGORITHM = "glynn-gray-lanczos-residual-fixed-montgomery-cuda-v1"
CATALOG_SHA256 = "feb6a22408c51627ab8b8cdf91da1d4707f64f324fae02dd6ef082c774e68b2d"
QUERY_COUNT = 36_398
PRIMES = (
    2147483647, 2147483629, 2147483587, 2147483579,
)
COMMON_FACTOR = math.factorial(28) * (1 << 24)
# Certified in Experiment 406, divided upward by 28! * 2^24.
QUOTIENT_BOUND = 7_030_983_209_987_543_242_183_335_298_990_080


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
        "excess", "unmatched_tokens", "defect_coefficient",
        "matching_bound_power", "vertices", "matrix_stride",
        "solver_binary_sha256", "prime", "begin", "end", "total_terms",
        "partial_glynn_sum", "gray_enabled", "gray_chain", "gray_slots",
        "gray_grid_blocks", "gray_active_blocks_per_sm", "gray_chunks",
        "gray_failures", "gray_fallback_chunks", "status",
        "result_payload_sha256",
    }
    if not required <= fields.keys():
        raise ValueError(f"{path}: missing fields {sorted(required-fields.keys())}")
    if fields["format"] != FORMAT or fields["algorithm"] != ALGORITHM:
        raise ValueError(f"{path}: incompatible format/algorithm")
    if (fields["rows"], fields["columns"], fields["catalog_sha256"]) != (
        "6", "28", CATALOG_SHA256
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


def required_prime_count(bound_power: int) -> int:
    modulus = 1
    for count, prime in enumerate(PRIMES, 1):
        modulus *= prime
        if modulus > 1 << bound_power:
            return count
    raise ValueError(f"four primes do not cover matching bound 2^{bound_power}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="*", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    result_paths = list(args.results)
    if args.directory:
        result_paths.extend(sorted(args.directory.glob("p*-q*-b*.result")))
    if not result_paths:
        parser.error("provide result files or --directory")
    by_job: dict[tuple[int, int], list[tuple[int, int, int, Path]]] = defaultdict(list)
    metadata: dict[int, tuple[str, int, int, int, int, int, int, int]] = {}
    for path in result_paths:
        fields = read_result(path)
        prime, query = int(fields["prime"]), int(fields["query_id"])
        begin, end = int(fields["begin"]), int(fields["end"])
        residue = int(fields["partial_glynn_sum"])
        vertices = int(fields["vertices"])
        stride = int(fields["matrix_stride"])
        gray_enabled = int(fields["gray_enabled"])
        gray_chain = int(fields["gray_chain"])
        gray_slots = int(fields["gray_slots"])
        gray_grid_blocks = int(fields["gray_grid_blocks"])
        gray_active_blocks = int(fields["gray_active_blocks_per_sm"])
        gray_chunks = int(fields["gray_chunks"])
        gray_failures = int(fields["gray_failures"])
        gray_fallback_chunks = int(fields["gray_fallback_chunks"])
        total_terms = int(fields["total_terms"])
        expected_terms = 1 << (vertices//2-1)
        if not 0 <= query < QUERY_COUNT or vertices not in range(48, 65, 2):
            raise ValueError(f"{path}: invalid query/order")
        if stride not in (vertices, vertices+1):
            raise ValueError(f"{path}: invalid matrix stride")
        if gray_enabled not in (0, 1):
            raise ValueError(f"{path}: invalid Gray mode")
        expected_chain = {48: 6, 50: 7}.get(vertices, 0)
        if gray_enabled:
            if (gray_chain != expected_chain or gray_slots <= 0 or
                    gray_active_blocks <= 0 or gray_chunks <= 0):
                raise ValueError(f"{path}: invalid Gray chain geometry")
            if gray_slots != 2*gray_grid_blocks:
                raise ValueError(f"{path}: inconsistent Gray grid geometry")
            if gray_fallback_chunks > gray_chunks:
                raise ValueError(f"{path}: invalid Gray fallback count")
        elif any((gray_chain, gray_slots, gray_grid_blocks,
                  gray_active_blocks, gray_chunks, gray_failures,
                  gray_fallback_chunks)):
            raise ValueError(f"{path}: inactive Gray mode has nonzero state")
        if gray_failures and not gray_fallback_chunks:
            raise ValueError(f"{path}: Gray failures were not recomputed")
        if total_terms != expected_terms or not 0 <= begin < end <= total_terms:
            raise ValueError(f"{path}: invalid term range")
        if prime not in PRIMES or not 0 <= residue < prime:
            raise ValueError(f"{path}: invalid prime/residue")
        item = (
            fields["query_sha256"], int(fields["occupied_tokens"]),
            int(fields["defect_count"]), int(fields["excess"]),
            int(fields["unmatched_tokens"]), int(fields["defect_coefficient"]),
            vertices, int(fields["matching_bound_power"]),
        )
        if query in metadata and metadata[query] != item:
            raise ValueError(f"{path}: inconsistent query metadata")
        metadata[query] = item
        by_job[prime, query].append((begin, end, residue, path))

    if len(metadata) == QUERY_COUNT:
        sectors: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
        required_histogram: dict[int, int] = defaultdict(int)
        for _, _, defects, excess, unmatched, coefficient, vertices, bound in metadata.values():
            if unmatched != 4-excess or vertices != 64-2*defects-2*excess:
                raise ValueError("inconsistent defect/query dimensions")
            sectors[excess, defects][0] += 1
            sectors[excess, defects][1] += coefficient
            required_histogram[required_prime_count(bound)] += 1
        expected_sectors = {
            (0, 0): [1, 1], (1, 1): [2, 840], (2, 1): [1, 1440],
            (2, 2): [25, 303660], (3, 2): [36, 993600],
            (3, 3): [664, 62422320], (4, 1): [2, 480],
            (4, 2): [42, 800640], (4, 3): [2548, 291375360],
            (4, 4): [33077, 8126516160],
        }
        if dict(sectors) != expected_sectors:
            raise ValueError(f"defect sector mismatch: {dict(sectors)}")
        if dict(required_histogram) != {3: 36395, 4: 3}:
            raise ValueError(f"adaptive-prime histogram mismatch: {dict(required_histogram)}")
    elif not args.allow_partial:
        raise ValueError(f"result set contains only {len(metadata)}/{QUERY_COUNT} query identities")

    complete_residues: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (prime, query), shards in by_job.items():
        cursor, signed_sum = 0, 0
        total_terms = 1 << (metadata[query][6]//2-1)
        for begin, end, residue, path in sorted(shards):
            if begin != cursor:
                raise ValueError(
                    f"prime {prime}, query {query}: gap/overlap at {cursor}, "
                    f"next {begin} ({path})"
                )
            cursor = end
            signed_sum = (signed_sum+residue) % prime
        if cursor != total_terms:
            continue
        unmatched = metadata[query][4]
        augmented = signed_sum * pow(total_terms, -1, prime) % prime
        matching = augmented * pow(math.factorial(unmatched), -1, prime) % prime
        complete_residues[query].append((prime, matching))

    quotient = 0
    unresolved: list[int] = []
    prime_histogram: dict[int, int] = defaultdict(int)
    for query in range(QUERY_COUNT):
        if query not in metadata:
            unresolved.append(query)
            continue
        residues = sorted(
            complete_residues.get(query, []),
            key=lambda item: PRIMES.index(item[0]),
        )
        matching, modulus = crt(residues)
        bound_power = metadata[query][7]
        if modulus <= 1 << bound_power:
            unresolved.append(query)
            continue
        _, _, defects, excess, unmatched, coefficient, vertices, _ = metadata[query]
        if unmatched != 4-excess or vertices != 64-2*defects-2*excess:
            raise ValueError(f"query {query}: inconsistent defect dimensions")
        quotient += coefficient * (1 << (4-defects)) * matching
        prime_histogram[len(residues)] += 1

    print(
        "HAFNIAN_6X28_ADAPTIVE "
        f"resolved={QUERY_COUNT-len(unresolved)}/{QUERY_COUNT} "
        f"prime_histogram={dict(sorted(prime_histogram.items()))}"
    )
    if unresolved:
        if args.allow_partial:
            print(
                f"HAFNIAN_6X28_RESULT status=PARTIAL unresolved={len(unresolved)} "
                f"first_unresolved={unresolved[:10]}"
            )
            return 0
        raise ValueError(
            f"{len(unresolved)} queries lack certified CRT coverage; "
            f"first unresolved: {unresolved[:10]}"
        )
    if quotient > QUOTIENT_BOUND:
        raise ValueError("reconstructed quotient exceeds certified bound")
    answer = quotient * COMMON_FACTOR
    print(
        f"HAFNIAN_6X28_QUOTIENT value={quotient} bits={quotient.bit_length()} "
        f"bound_bits={QUOTIENT_BOUND.bit_length()} exact=OK"
    )
    print(f"HAFNIAN_6X28_RESULT T_4(6,28)={answer} exact=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
