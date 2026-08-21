#!/usr/bin/env python3
"""Resumable multi-GPU driver for the 29 residual hafnians of T_4(6,29)."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import queue
import subprocess
import sys
import time
from pathlib import Path

from reduce_six_by_twenty_nine_hafnian import read_result


PRIMES = (
    2147483647, 2147483629, 2147483587, 2147483579,
    2147483563, 2147483549, 2147483543, 2147483497,
    2147483489,
)
QUERY_VERTICES = (62, 58, 58, 56) + (54,) * 25


def total_terms(query: int) -> int:
    return 1 << (QUERY_VERTICES[query] // 2 - 1)


def completed_prefix(directory: Path, prime: int, query: int) -> tuple[int, list[Path]]:
    pieces: list[tuple[int, int, Path]] = []
    for path in directory.glob(f"p{prime}-q{query:02d}-b*.result"):
        fields = read_result(path)
        if (int(fields["prime"]), int(fields["query_id"])) != (prime, query):
            raise ValueError(f"{path}: filename/result identity mismatch")
        pieces.append((int(fields["begin"]), int(fields["end"]), path))
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
    parser.add_argument("--binary", type=Path, default=Path("./six_by_twenty_nine_hafnian_gpu"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--chunk-terms", type=int, default=1 << 20)
    parser.add_argument("--prime-count", type=int, default=len(PRIMES))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.prime_count <= len(PRIMES):
        parser.error(f"--prime-count must be in [1,{len(PRIMES)}]")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus or len(set(gpus)) != len(gpus):
        parser.error("--gpus must contain distinct comma-separated device IDs")
    binary = args.binary.resolve()
    if not args.dry_run and not binary.is_file():
        parser.error(f"binary not found: {binary}")
    args.output.mkdir(parents=True, exist_ok=True)
    logs = args.output / "logs"
    logs.mkdir(exist_ok=True)

    jobs: list[tuple[int, int, int]] = []
    for prime in PRIMES[:args.prime_count]:
        for query_id in range(len(QUERY_VERTICES)):
            cursor, paths = completed_prefix(args.output, prime, query_id)
            if cursor < total_terms(query_id):
                jobs.append((total_terms(query_id) - cursor, prime, query_id))
    jobs.sort(reverse=True)
    work: queue.Queue[tuple[int, int, int]] = queue.Queue()
    for job in jobs:
        work.put(job)
    print(
        f"HAFNIAN_6X29_CAMPAIGN_SCAN primes={args.prime_count} jobs={len(jobs)} "
        f"remaining_terms={sum(job[0] for job in jobs)}",
        flush=True,
    )

    def worker(gpu: str) -> None:
        while True:
            try:
                _, prime, query_id = work.get_nowait()
            except queue.Empty:
                return
            begin, _ = completed_prefix(args.output, prime, query_id)
            end = total_terms(query_id)
            output = args.output / f"p{prime}-q{query_id:02d}-b{begin}.result"
            log_path = logs / f"p{prime}-q{query_id:02d}-b{begin}-gpu{gpu}.log"
            command = [
                str(binary), "--run", "--query", str(query_id),
                "--prime", str(prime), "--begin", str(begin), "--end", str(end),
                "--chunk-terms", str(args.chunk_terms), "--output", str(output),
            ]
            print(
                f"HAFNIAN_6X29_CAMPAIGN_START gpu={gpu} prime={prime} query={query_id} "
                f"vertices={QUERY_VERTICES[query_id]} begin={begin} end={end}",
                flush=True,
            )
            if args.dry_run:
                work.task_done()
                continue
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            started = time.monotonic()
            with log_path.open("ab", buffering=0) as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
            if result.returncode:
                raise RuntimeError(
                    f"GPU {gpu}, prime {prime}, query {query_id} failed; see {log_path}"
                )
            fields = read_result(output)
            if int(fields["end"]) != end:
                raise RuntimeError(f"successful process left incomplete result: {output}")
            print(
                f"HAFNIAN_6X29_CAMPAIGN_COMPLETE gpu={gpu} prime={prime} query={query_id} "
                f"elapsed={time.monotonic()-started:.3f}",
                flush=True,
            )
            work.task_done()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(worker, gpu) for gpu in gpus]
        for future in futures:
            future.result()

    if args.dry_run:
        return 0
    result_files: list[Path] = []
    for prime in PRIMES[:args.prime_count]:
        for query_id in range(len(QUERY_VERTICES)):
            cursor, paths = completed_prefix(args.output, prime, query_id)
            if cursor != total_terms(query_id):
                raise RuntimeError(
                    f"prime {prime}, query {query_id}: coverage {cursor}/{total_terms(query_id)}"
                )
            result_files.extend(paths)
    reducer = Path(__file__).with_name("reduce_six_by_twenty_nine_hafnian.py")
    return subprocess.run([sys.executable, str(reducer), *map(str, result_files)]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
