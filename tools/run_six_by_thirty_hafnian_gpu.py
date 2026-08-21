#!/usr/bin/env python3
"""Resumable multi-GPU campaign driver for the exact 6x30 hafnian."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import queue
import subprocess
import sys
import time
from pathlib import Path

from reduce_six_by_thirty_hafnian import TOTAL_TERMS, read_result


PRIMES = (
    2147483647,
    2147483629,
    2147483587,
    2147483579,
    2147483563,
    2147483549,
    2147483543,
    2147483497,
    2147483489,
    2147483477,
)


def completed_prefix(directory: Path, prime: int) -> tuple[int, list[Path]]:
    pieces: list[tuple[int, int, Path]] = []
    for path in directory.glob(f"p{prime}-b*.result"):
        fields = read_result(path)
        if int(fields["prime"]) != prime:
            raise ValueError(f"{path}: filename/prime mismatch")
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
            raise ValueError(f"prime {prime}: overlapping result {path}")
        cursor = end
        accepted.append(path)
    return cursor, accepted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("./six_by_thirty_hafnian_gpu"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--prime-count", type=int, default=len(PRIMES))
    parser.add_argument("--chunk-terms", type=int, default=1 << 20)
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

    work: queue.Queue[tuple[int, int]] = queue.Queue()
    complete_paths: list[Path] = []
    for prime in PRIMES[: args.prime_count]:
        cursor, paths = completed_prefix(args.output, prime)
        complete_paths.extend(paths)
        print(f"HAFNIAN_CAMPAIGN_SCAN prime={prime} covered={cursor}/{TOTAL_TERMS} pieces={len(paths)}")
        if cursor < TOTAL_TERMS:
            work.put((prime, cursor))

    def worker(gpu: str) -> list[Path]:
        produced: list[Path] = []
        while True:
            try:
                prime, begin = work.get_nowait()
            except queue.Empty:
                return produced
            output = args.output / f"p{prime}-b{begin}.result"
            log_path = logs / f"p{prime}-b{begin}-gpu{gpu}.log"
            command = [
                str(binary), "--run", "--prime", str(prime),
                "--begin", str(begin), "--end", str(TOTAL_TERMS),
                "--chunk-terms", str(args.chunk_terms), "--output", str(output),
            ]
            print(
                f"HAFNIAN_CAMPAIGN_START gpu={gpu} prime={prime} "
                f"begin={begin} end={TOTAL_TERMS}"
            )
            if args.dry_run:
                print(" ".join(command))
                work.task_done()
                continue
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            started = time.monotonic()
            with log_path.open("ab", buffering=0) as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
            if result.returncode:
                raise RuntimeError(
                    f"GPU {gpu}, prime {prime} failed with status {result.returncode}; see {log_path}"
                )
            fields = read_result(output)
            if (int(fields["prime"]), int(fields["begin"]), int(fields["end"])) != (
                prime, begin, TOTAL_TERMS
            ):
                raise RuntimeError(f"incomplete result after successful process: {output}")
            produced.append(output)
            elapsed = time.monotonic() - started
            print(f"HAFNIAN_CAMPAIGN_COMPLETE gpu={gpu} prime={prime} elapsed={elapsed:.3f}")
            work.task_done()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(worker, gpu) for gpu in gpus]
        for future in futures:
            complete_paths.extend(future.result())

    if args.dry_run:
        return 0
    result_files: list[Path] = []
    for prime in PRIMES[: args.prime_count]:
        cursor, paths = completed_prefix(args.output, prime)
        if cursor != TOTAL_TERMS:
            raise RuntimeError(f"prime {prime} ends at {cursor}, expected {TOTAL_TERMS}")
        result_files.extend(paths)
    reducer = Path(__file__).with_name("reduce_six_by_thirty_hafnian.py")
    return subprocess.run([sys.executable, str(reducer), *map(str, result_files)]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
