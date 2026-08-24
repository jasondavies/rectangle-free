#!/usr/bin/env python3
"""Resumable persistent multi-GPU campaign driver for T_4(6,28)."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from reduce_six_by_twenty_eight_hafnian import (
    CATALOG_SHA256,
    PRIMES,
    QUERY_COUNT,
    read_result,
    required_prime_count,
)


QUERY_PATTERN = re.compile(
    r"^HAFNIAN_6X28_QUERY id=(\d+) .* vertices=(\d+) "
    r"terms=(\d+) matching_bound_power=(\d+) digest=([0-9a-f]{64})$"
)


@dataclass(frozen=True)
class QueryInfo:
    query: int
    vertices: int
    terms: int
    bound_power: int
    digest: str


def load_catalog(binary: Path) -> list[QueryInfo]:
    completed = subprocess.run(
        [str(binary), "--list"], check=True, capture_output=True, text=True
    )
    expected_header = f"queries={QUERY_COUNT} digest={CATALOG_SHA256}"
    if expected_header not in completed.stdout:
        raise RuntimeError("solver reported an unexpected 6x28 catalog")
    queries: list[QueryInfo] = []
    for line in completed.stdout.splitlines():
        match = QUERY_PATTERN.match(line)
        if match:
            queries.append(QueryInfo(*(
                int(match.group(1)), int(match.group(2)), int(match.group(3)),
                int(match.group(4)), match.group(5),
            )))
    if len(queries) != QUERY_COUNT or [item.query for item in queries] != list(range(QUERY_COUNT)):
        raise RuntimeError("solver catalog listing is incomplete or unordered")
    return queries


def scan_results(directory: Path) -> dict[tuple[int, int], list[tuple[int, int, Path]]]:
    result: dict[tuple[int, int], list[tuple[int, int, Path]]] = {}
    for path in directory.glob("p*-q*-b*.result"):
        fields = read_result(path)
        identity = int(fields["prime"]), int(fields["query_id"])
        result.setdefault(identity, []).append(
            (int(fields["begin"]), int(fields["end"]), path)
        )
    return result


def completed_prefix(
    index: dict[tuple[int, int], list[tuple[int, int, Path]]],
    prime: int,
    query: int,
) -> tuple[int, list[Path]]:
    pieces = index.get((prime, query), [])
    pieces.sort()
    cursor = 0
    accepted: list[Path] = []
    for begin, end, path in pieces:
        if begin > cursor:
            break
        if begin < cursor:
            if end <= cursor:
                continue
            raise ValueError(f"prime {prime}, query {query}: overlap at {path}")
        cursor = end
        accepted.append(path)
    return cursor, accepted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary", type=Path, default=Path("./build/six_by_twenty_eight_hafnian_gpu")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0")
    # One result publication per common N=48 query, while retaining roughly
    # 10--15-second recovery granularity for the rare N=64 queries.
    parser.add_argument("--chunk-terms", type=int, default=1 << 24)
    parser.add_argument(
        "--blocks", type=int, default=0,
        help="Gray chain slots (fallback CTAs for non-Gray orders); 0 autotunes",
    )
    parser.add_argument(
        "--threads", type=int, default=0,
        help="fallback threads per CTA; Gray-enabled orders use 64; 0 autotunes",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus or len(set(gpus)) != len(gpus):
        parser.error("--gpus must contain distinct comma-separated device IDs")
    if args.chunk_terms <= 0 or not 0 <= args.threads <= 1024 or args.blocks < 0:
        parser.error("invalid kernel configuration")
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"binary not found: {binary}")
    args.output.mkdir(parents=True, exist_ok=True)
    logs = args.output / "logs"
    state = args.output / "state"
    logs.mkdir(exist_ok=True)
    state.mkdir(exist_ok=True)

    catalog = load_catalog(binary)
    result_index = scan_results(args.output)
    jobs: list[tuple[int, int, QueryInfo, int]] = []
    for query in catalog:
        for prime in PRIMES[:required_prime_count(query.bound_power)]:
            cursor, _ = completed_prefix(result_index, prime, query.query)
            if cursor < query.terms:
                jobs.append((query.terms-cursor, prime, query, cursor))
    jobs.sort(key=lambda item: item[0], reverse=True)

    assignments: list[list[tuple[int, int, QueryInfo, int]]] = [[] for _ in gpus]
    loads = [0] * len(gpus)
    for job in jobs:
        owner = min(range(len(gpus)), key=loads.__getitem__)
        assignments[owner].append(job)
        loads[owner] += job[0]
    print(
        f"HAFNIAN_6X28_CAMPAIGN_SCAN jobs={len(jobs)} "
        f"remaining_terms={sum(item[0] for item in jobs)} "
        f"gpu_loads={','.join(map(str, loads))}",
        flush=True,
    )

    def worker(index: int, gpu: str) -> None:
        batch = state / f"batch-gpu{gpu}.txt"
        lines: list[str] = []
        for _, prime, query, begin in assignments[index]:
            output = args.output / f"p{prime}-q{query.query:05d}-b{begin}.result"
            lines.append(f"{query.query} {prime} {begin} {query.terms} {output.resolve()}\n")
        temporary = batch.with_suffix(".tmp")
        temporary.write_text("".join(lines))
        temporary.replace(batch)
        if not lines or args.dry_run:
            return
        command = [
            str(binary), "--batch", str(batch),
            "--chunk-terms", str(args.chunk_terms),
            "--threads", str(args.threads),
        ]
        if args.blocks:
            command += ["--blocks", str(args.blocks)]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        log_path = logs / f"gpu{gpu}.log"
        print(
            f"HAFNIAN_6X28_WORKER_START gpu={gpu} jobs={len(lines)} load={loads[index]}",
            flush=True,
        )
        started = time.monotonic()
        with log_path.open("ab", buffering=0) as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, env=environment
            )
        if result.returncode:
            raise RuntimeError(f"GPU {gpu} failed; see {log_path}")
        print(
            f"HAFNIAN_6X28_WORKER_COMPLETE gpu={gpu} elapsed={time.monotonic()-started:.3f}",
            flush=True,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(worker, index, gpu) for index, gpu in enumerate(gpus)]
        for future in futures:
            future.result()
    if args.dry_run:
        return 0

    result_index = scan_results(args.output)
    for query in catalog:
        for prime in PRIMES[:required_prime_count(query.bound_power)]:
            cursor, _ = completed_prefix(result_index, prime, query.query)
            if cursor != query.terms:
                raise RuntimeError(
                    f"prime {prime}, query {query.query}: coverage {cursor}/{query.terms}"
                )
    reducer = Path(__file__).with_name("reduce_six_by_twenty_eight_hafnian.py")
    return subprocess.run(
        [sys.executable, str(reducer), "--directory", str(args.output)]
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
