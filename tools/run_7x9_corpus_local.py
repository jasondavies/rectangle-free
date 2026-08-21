#!/usr/bin/env python3
"""Build the filtered, 128-owner 7x9 solve corpus on one large host.

The campaign is restartable at generator, full-key reducer, and final solve-owner
boundaries.  Intermediate data is deliberately retained until the complete
corpus passes the global check.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


RANGES = 128
REDUCERS = 128
OWNERS = 128
MIN_FREE_BYTES = 700 * (1 << 30)
MIN_AVAILABLE_BYTES_16_WORKERS = 160 * (1 << 30)
PLAN_RE = re.compile(
    r"^solve_plan_7x9 range=(\d+) start=(\d+) end=(\d+) work=(\d+)$"
)


def available_memory() -> int:
    with open("/proc/meminfo", encoding="ascii") as source:
        for line in source:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def ensure_raw_files(directory: Path, prefix: str, count: int, width: int) -> None:
    for index in range(count):
        path = directory / f"{prefix}{index:0{width}d}"
        if not path.is_file() or path.stat().st_size % 16:
            raise RuntimeError(f"missing or malformed raw file: {path}")


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def publish_directory(temp: Path, final: Path) -> None:
    if final.exists():
        raise RuntimeError(f"refusing to replace completed directory: {final}")
    os.replace(temp, final)


def parallel_stage(name: str, items: list[int], workers: int, action) -> None:
    if not items:
        print(f"STAGE {name}: already complete", flush=True)
        return
    print(f"STAGE {name}: items={len(items)} workers={workers}", flush=True)
    started = time.monotonic()
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(action, item): item for item in items}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            future.result()
            completed += 1
            elapsed = time.monotonic() - started
            rate = completed / elapsed
            remaining = (len(items) - completed) / rate if rate else 0
            print(
                f"PROGRESS {name} completed={completed}/{len(items)} "
                f"last={item} elapsed={elapsed:.1f}s eta={remaining:.1f}s",
                flush=True,
            )
    print(f"STAGE {name}: seconds={time.monotonic() - started:.1f}", flush=True)


def build_plan(binary: Path, parent: Path, campaign: Path) -> list[tuple[int, int, int]]:
    plan_path = campaign / "ranges.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        ranges = [(int(row["start"]), int(row["end"]), int(row["work"])) for row in plan]
    else:
        result = subprocess.run(
            [str(binary), "solve-plan7x9", str(parent), str(RANGES)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
        (campaign / "plan.log").write_text(result.stdout, encoding="utf-8")
        rows: dict[int, tuple[int, int, int]] = {}
        for line in result.stdout.splitlines():
            match = PLAN_RE.match(line)
            if match:
                index, start, end, work = map(int, match.groups())
                rows[index] = (start, end, work)
        if sorted(rows) != list(range(RANGES)) or " OK" not in result.stdout:
            raise RuntimeError("the exact 7x9 planner did not pass all gates")
        ranges = [rows[index] for index in range(RANGES)]
        temp = plan_path.with_suffix(".json.tmp")
        temp.write_text(
            json.dumps(
                [
                    {"range": index, "start": start, "end": end, "work": work}
                    for index, (start, end, work) in enumerate(ranges)
                ],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temp, plan_path)
    if len(ranges) != RANGES or ranges[0][0] != 0 or ranges[-1][1] != 508_147_108:
        raise RuntimeError("invalid saved range plan")
    if any(ranges[index][1] != ranges[index + 1][0] for index in range(RANGES - 1)):
        raise RuntimeError("saved range plan is not contiguous")
    if sum(row[2] for row in ranges) != 32_521_414_912:
        raise RuntimeError("saved range plan has the wrong work total")
    return ranges


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent", type=Path, default=Path("/tmp/rect7x8-build/rect7x8-full.orbits")
    )
    parser.add_argument("--binary", type=Path, default=Path("./binary_orbit_augment_7x9"))
    parser.add_argument(
        "--campaign", type=Path, default=Path("../rectangle-free-data-v2/7x9-corpus")
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    parent = args.parent.resolve()
    binary = args.binary.resolve()
    campaign = args.campaign.resolve()
    workers = args.workers
    if not parent.is_file() or parent.stat().st_size != 8_130_353_748:
        raise RuntimeError(f"missing or incorrectly sized complete 7x8 corpus: {parent}")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"missing executable: {binary}")
    if workers < 1 or workers > 16:
        raise RuntimeError("workers must be between 1 and 16")

    campaign.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(campaign).free < MIN_FREE_BYTES:
        raise RuntimeError("less than 700 GiB is free for the restartable campaign")
    if workers == 16 and available_memory() < MIN_AVAILABLE_BYTES_16_WORKERS:
        raise RuntimeError("less than 160 GiB is currently available for 16 workers")

    for name in ("raw", "reduced", "solve", "logs/generate", "logs/reduce", "logs/final"):
        (campaign / name).mkdir(parents=True, exist_ok=True)
    ranges = build_plan(binary, parent, campaign)
    print(
        f"CAMPAIGN path={campaign} parent={parent} workers={workers} "
        f"free_gib={shutil.disk_usage(campaign).free / (1 << 30):.1f} "
        f"available_gib={available_memory() / (1 << 30):.1f}",
        flush=True,
    )

    def generate(index: int) -> None:
        final = campaign / "raw" / f"g{index:04d}"
        if final.exists():
            ensure_raw_files(final, f"g{index:04d}.b", REDUCERS, 3)
            return
        temp = final.with_name(final.name + ".tmp")
        if temp.exists():
            shutil.rmtree(temp)
        temp.mkdir()
        start, end, _ = ranges[index]
        prefix = temp / f"g{index:04d}"
        run_logged(
            [
                str(binary), "solve-extend7x8", str(parent), str(start), str(end),
                str(REDUCERS), str(prefix),
            ],
            campaign / "logs" / "generate" / f"g{index:04d}.log",
        )
        ensure_raw_files(temp, f"g{index:04d}.b", REDUCERS, 3)
        publish_directory(temp, final)

    pending = [index for index in range(RANGES) if not (campaign / "raw" / f"g{index:04d}").exists()]
    parallel_stage("generate", pending, workers, generate)

    def reduce_bucket(index: int) -> None:
        final = campaign / "reduced" / f"r{index:04d}"
        if final.exists():
            ensure_raw_files(final, "piece.s", OWNERS, 4)
            return
        temp = final.with_name(final.name + ".tmp")
        if temp.exists():
            try:
                ensure_raw_files(temp, "piece.s", OWNERS, 4)
                publish_directory(temp, final)
                return
            except RuntimeError:
                shutil.rmtree(temp)
        temp.mkdir()
        inputs = [
            campaign / "raw" / f"g{generator:04d}" / f"g{generator:04d}.b{index:03d}"
            for generator in range(RANGES)
        ]
        run_logged(
            [str(binary), "reduce-solve", "9", str(OWNERS), str(temp / "piece")]
            + [str(path) for path in inputs],
            campaign / "logs" / "reduce" / f"r{index:04d}.log",
        )
        ensure_raw_files(temp, "piece.s", OWNERS, 4)
        publish_directory(temp, final)

    pending = [
        index for index in range(REDUCERS)
        if not (campaign / "reduced" / f"r{index:04d}").exists()
    ]
    parallel_stage("reduce", pending, workers, reduce_bucket)

    def finalize(index: int) -> None:
        final = campaign / "solve" / f"s{index:04d}.orbits"
        if final.exists():
            subprocess.run(
                [str(binary), "solve-check-shard", str(OWNERS), str(index), str(final)],
                stdout=subprocess.DEVNULL,
                check=True,
            )
            return
        temp = final.with_suffix(".orbits.tmp")
        temp.unlink(missing_ok=True)
        inputs = [
            campaign / "reduced" / f"r{reducer:04d}" / f"piece.s{index:04d}"
            for reducer in range(REDUCERS)
        ]
        log = campaign / "logs" / "final" / f"s{index:04d}.log"
        run_logged(
            [
                str(binary), "solve-reduce-unique", str(OWNERS), str(index), str(temp)
            ]
            + [str(path) for path in inputs],
            log,
        )
        with log.open("a", encoding="utf-8") as output:
            subprocess.run(
                [str(binary), "solve-check-shard", str(OWNERS), str(index), str(temp)],
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )
        os.replace(temp, final)

    pending = [
        index for index in range(OWNERS)
        if not (campaign / "solve" / f"s{index:04d}.orbits").exists()
    ]
    parallel_stage("finalize", pending, workers, finalize)

    solve_files = [campaign / "solve" / f"s{index:04d}.orbits" for index in range(OWNERS)]
    check_log = campaign / "check.log"
    run_logged(
        [str(binary), "solve-check", str(OWNERS)] + [str(path) for path in solve_files],
        check_log,
    )
    if " OK" not in check_log.read_text(encoding="utf-8"):
        raise RuntimeError("complete solve-corpus check did not report OK")
    print(f"ALL_COMPLETE solve={campaign / 'solve'} check={check_log}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FATAL {error}", file=sys.stderr, flush=True)
        raise
