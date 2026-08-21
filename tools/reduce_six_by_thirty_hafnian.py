#!/usr/bin/env python3
"""Validate finite-field hafnian shards and reconstruct T_4(6,30)."""

from __future__ import annotations

import argparse
import math
import hashlib
from collections import defaultdict
from pathlib import Path


TOTAL_TERMS = 1 << 29
FORMAT = "six-by-thirty-hafnian-v1"
ALGORITHMS = {"glynn-trace-hessenberg-v1", "glynn-trace-hessenberg-cuda-v1"}
GRAPH_SHA256 = "9563bf83042c9f9548261d3602279d3abec50417897b32f54e72b4816524a947"


def read_result(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    contents = path.read_text()
    lines = contents.splitlines(keepends=True)
    payload = b""
    for line_with_ending in lines:
        line = line_with_ending.rstrip("\r\n")
        key, separator, value = line.partition(" ")
        if not separator or key in fields:
            raise ValueError(f"{path}: malformed or duplicate field: {line!r}")
        fields[key] = value
        if key != "result_payload_sha256":
            payload += line_with_ending.encode()
    required = {
        "format", "algorithm", "rows", "colours", "vertices", "edges",
        "graph_sha256", "solver_binary_sha256", "prime", "begin", "end",
        "total_terms", "partial_glynn_sum", "status", "result_payload_sha256",
    }
    if not required <= fields.keys():
        raise ValueError(f"{path}: missing fields {sorted(required - fields.keys())}")
    if fields["format"] != FORMAT or fields["algorithm"] not in ALGORITHMS:
        raise ValueError(f"{path}: incompatible solver format")
    if (fields["rows"], fields["colours"], fields["vertices"], fields["edges"]) != (
        "6", "4", "60", "540"
    ):
        raise ValueError(f"{path}: geometry mismatch")
    if fields["graph_sha256"] != GRAPH_SHA256:
        raise ValueError(f"{path}: graph digest mismatch")
    if hashlib.sha256(payload).hexdigest() != fields["result_payload_sha256"]:
        raise ValueError(f"{path}: payload digest mismatch")
    if fields["status"] != "complete" or int(fields["total_terms"]) != TOTAL_TERMS:
        raise ValueError(f"{path}: incomplete result")
    return fields


def crt(residues: list[tuple[int, int]]) -> tuple[int, int]:
    value, modulus = 0, 1
    for prime, residue in residues:
        if math.gcd(modulus, prime) != 1:
            raise ValueError(f"moduli are not coprime at {prime}")
        correction = ((residue - value) % prime) * pow(modulus, -1, prime) % prime
        value += modulus * correction
        modulus *= prime
    return value, modulus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--allow-partial-primes", action="store_true")
    args = parser.parse_args()

    by_prime: dict[int, list[tuple[int, int, int, Path]]] = defaultdict(list)
    for path in args.results:
        fields = read_result(path)
        prime = int(fields["prime"])
        begin, end = int(fields["begin"]), int(fields["end"])
        residue = int(fields["partial_glynn_sum"])
        if not 0 <= begin < end <= TOTAL_TERMS or not 0 <= residue < prime:
            raise ValueError(f"{path}: invalid range or residue")
        by_prime[prime].append((begin, end, residue, path))

    complete: list[tuple[int, int]] = []
    for prime, shards in sorted(by_prime.items()):
        shards.sort()
        cursor = 0
        signed_sum = 0
        for begin, end, residue, path in shards:
            if begin != cursor:
                message = f"prime {prime}: gap/overlap at {cursor}, next {begin} ({path})"
                if args.allow_partial_primes:
                    print(f"HAFNIAN_REDUCE prime={prime} status=PARTIAL reason={message}")
                    break
                raise ValueError(message)
            cursor = end
            signed_sum = (signed_sum + residue) % prime
        if cursor != TOTAL_TERMS:
            if not args.allow_partial_primes:
                raise ValueError(f"prime {prime}: coverage ends at {cursor}, expected {TOTAL_TERMS}")
            continue
        perfect_matchings = signed_sum * pow(pow(2, 29, prime), -1, prime) % prime
        answer = perfect_matchings * pow(2, 30, prime) % prime
        for factor in range(2, 31):
            answer = answer * factor % prime
        complete.append((prime, answer))
        print(
            f"HAFNIAN_REDUCE prime={prime} shards={len(shards)} "
            f"perfect_matchings_mod={perfect_matchings} T_mod={answer} status=COMPLETE"
        )

    if not complete:
        if args.allow_partial_primes:
            print("HAFNIAN_CRT status=PARTIAL complete_primes=0")
            return 0
        raise ValueError("no completely covered prime")
    answer, modulus = crt(complete)
    print(
        f"HAFNIAN_CRT primes={len(complete)} modulus_bits={modulus.bit_length()} "
        f"answer={answer} answer_bits={answer.bit_length()}"
    )
    if modulus > math.factorial(60):
        print(f"HAFNIAN_RESULT T_4(6,30)={answer} exact=OK")
    else:
        print(
            "HAFNIAN_RESULT status=MODULAR_ONLY "
            f"need_modulus_above_60_factorial current_bits={modulus.bit_length()} "
            f"bound_bits={math.factorial(60).bit_length()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
