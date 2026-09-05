#!/usr/bin/env python3
"""Shared persistent, resumable multi-GPU residual-hafnian scheduler."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class QueryInfo:
    query: int
    vertices: int
    terms: int
    bound_power: int
    digest: str


class Runner:
    def __init__(self, campaign, width, binary_name):
        self.campaign = campaign
        self.width = width
        self.binary_name = binary_name
        self.query_pattern = re.compile(
            rf"^HAFNIAN_6X{width}_QUERY id=(\d+) .* vertices=(\d+) "
            r"terms=(\d+) matching_bound_power=(\d+) digest=([0-9a-f]{64})$"
        )

    def load_catalog(self, binary: Path) -> list[QueryInfo]:
        completed = subprocess.run(
            [str(binary), "--list"], check=True, capture_output=True, text=True
        )
        expected_header = f"queries={self.campaign.QUERY_COUNT} digest={self.campaign.CATALOG_SHA256}"
        if expected_header not in completed.stdout:
            raise RuntimeError("solver reported an unexpected hafnian catalog")
        queries: list[QueryInfo] = []
        for line in completed.stdout.splitlines():
            match = self.query_pattern.match(line)
            if match:
                queries.append(QueryInfo(*(
                    int(match.group(1)), int(match.group(2)), int(match.group(3)),
                    int(match.group(4)), match.group(5),
                )))
        if len(queries) != self.campaign.QUERY_COUNT or [item.query for item in queries] != list(range(self.campaign.QUERY_COUNT)):
            raise RuntimeError("solver catalog listing is incomplete or unordered")
        return queries


    def scan_results(self, directory: Path) -> dict[tuple[int, int], list[tuple[int, int, Path]]]:
        result: dict[tuple[int, int], list[tuple[int, int, Path]]] = {}
        for path in directory.glob("p*-q*-b*.result"):
            fields = self.campaign.read_result(path)
            if self.width in (29, 30) and fields["solver_binary_sha256"] != self.binary_digest:
                raise ValueError(f"{path}: different verification solver binary")
            identity = int(fields["prime"]), int(fields["query_id"])
            result.setdefault(identity, []).append(
                (int(fields["begin"]), int(fields["end"]), path)
            )
        return result


    def completed_prefix(self,
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


    def main(self) -> int:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--binary", type=Path, default=Path("./build/"+self.binary_name)
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
        parser.add_argument(
            "--partition-index", type=int, default=0,
            help="weighted campaign partition to execute (default: 0)",
        )
        parser.add_argument(
            "--partition-weights", default="1",
            help="comma-separated relative capacities of disjoint campaign partitions",
        )
        parser.add_argument("--dry-run", action="store_true")
        args = parser.parse_args()
        gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
        if not gpus or len(set(gpus)) != len(gpus):
            parser.error("--gpus must contain distinct comma-separated device IDs")
        try:
            partition_weights = [
                float(item) for item in args.partition_weights.split(",") if item
            ]
        except ValueError:
            parser.error("--partition-weights must contain positive numbers")
        if (not partition_weights or any(weight <= 0 for weight in partition_weights)
                or not 0 <= args.partition_index < len(partition_weights)):
            parser.error("invalid campaign partition configuration")
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

        self.binary_digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        catalog = self.load_catalog(binary)
        result_index = self.scan_results(args.output)
        jobs: list[tuple[int, int, QueryInfo, int]] = []
        for query in catalog:
            for prime in self.campaign.PRIMES[:self.campaign.required_prime_count(query.bound_power)]:
                cursor, _ = self.completed_prefix(result_index, prime, query.query)
                if cursor < query.terms:
                    jobs.append((query.terms-cursor, prime, query, cursor))
        jobs.sort(key=lambda item: item[0], reverse=True)

        partition_jobs: list[list[tuple[int, int, QueryInfo, int]]] = [
            [] for _ in partition_weights
        ]
        partition_loads = [0] * len(partition_weights)
        for job in jobs:
            owner = min(
                range(len(partition_weights)),
                key=lambda index: partition_loads[index] / partition_weights[index],
            )
            partition_jobs[owner].append(job)
            partition_loads[owner] += job[0]
        jobs = partition_jobs[args.partition_index]
        print(
            f"HAFNIAN_6X{self.width}_PARTITION index={args.partition_index} "
            f"count={len(partition_weights)} jobs={len(jobs)} "
            f"load={partition_loads[args.partition_index]} "
            f"all_loads={','.join(map(str, partition_loads))}",
            flush=True,
        )

        assignments: list[list[tuple[int, int, QueryInfo, int]]] = [[] for _ in gpus]
        loads = [0] * len(gpus)
        for job in jobs:
            owner = min(range(len(gpus)), key=loads.__getitem__)
            assignments[owner].append(job)
            loads[owner] += job[0]
        print(
            f"HAFNIAN_6X{self.width}_CAMPAIGN_SCAN jobs={len(jobs)} "
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
                f"HAFNIAN_6X{self.width}_WORKER_START gpu={gpu} jobs={len(lines)} load={loads[index]}",
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
                f"HAFNIAN_6X{self.width}_WORKER_COMPLETE gpu={gpu} elapsed={time.monotonic()-started:.3f}",
                flush=True,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = [executor.submit(worker, index, gpu) for index, gpu in enumerate(gpus)]
            for future in futures:
                future.result()
        if args.dry_run:
            return 0

        result_index = self.scan_results(args.output)
        if len(partition_weights) > 1:
            for _, prime, query, _ in jobs:
                cursor, _ = self.completed_prefix(result_index, prime, query.query)
                if cursor != query.terms:
                    raise RuntimeError(
                        f"prime {prime}, query {query.query}: "
                        f"partition coverage {cursor}/{query.terms}"
                    )
            print(
                f"HAFNIAN_6X{self.width}_PARTITION_COMPLETE index={args.partition_index} "
                f"jobs={len(jobs)} exact=OK",
                flush=True,
            )
            return 0

        for query in catalog:
            for prime in self.campaign.PRIMES[:self.campaign.required_prime_count(query.bound_power)]:
                cursor, _ = self.completed_prefix(result_index, prime, query.query)
                if cursor != query.terms:
                    raise RuntimeError(
                        f"prime {prime}, query {query.query}: coverage {cursor}/{query.terms}"
                    )
        reducer = Path(self.campaign.__file__)
        return subprocess.run(
            [sys.executable, str(reducer), "--directory", str(args.output)]
        ).returncode
