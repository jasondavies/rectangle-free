#!/usr/bin/env python3

import argparse
import heapq
import math
import os
import re
import random
import statistics
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable


TOTAL_TASKS_RE = re.compile(r"Prefix depth:\s+\d+\s+\((\d+)\s+tasks\)")
TOTAL_ELAPSED_RE = re.compile(r"Total elapsed including prefix generation:\s+([0-9.]+)\s+seconds\.")
WORKER_ELAPSED_RE = re.compile(r"Worker Complete in\s+([0-9.]+)\s+seconds\.")
PREFIX_ELAPSED_RE = re.compile(r"Prefix generation:\s+([0-9.]+)\s+seconds")


@dataclass
class SampleResult:
    offset: int
    worker_seconds: float
    total_seconds: float
    prefix_seconds: float
    task_count: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate full 7x7 runtime/core-hours from sampled interleaved shards."
    )
    parser.add_argument("--binary", default="./partition_poly_7")
    parser.add_argument("--mode", choices=["shards", "tasks", "blocks"], default="shards")
    parser.add_argument("--rows", type=int, default=7)
    parser.add_argument("--cols", type=int, default=7)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--prefix-depth", type=int, default=2)
    parser.add_argument("--stride", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--offsets", default="")
    parser.add_argument("--task-indices", default="")
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--adaptive-subdivide", action="store_true")
    parser.add_argument("--adaptive-threshold", type=int, default=128)
    parser.add_argument("--adaptive-max-depth", type=int, default=3)
    parser.add_argument("--cluster-machines", type=int, default=100)
    parser.add_argument("--cores-per-machine", type=int, default=64)
    parser.add_argument("--efficiency-low", type=float, default=0.70)
    parser.add_argument("--efficiency-high", type=float, default=0.90)
    parser.add_argument("--rect-progress-step", type=int, default=1000000)
    parser.add_argument("--bootstrap-runs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_base_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.binary,
        str(args.rows),
        str(args.cols),
        "--prefix-depth",
        str(args.prefix_depth),
    ]
    if args.adaptive_subdivide:
        cmd.extend(
            [
                "--adaptive-subdivide",
                "--adaptive-threshold",
                str(args.adaptive_threshold),
                "--adaptive-max-depth",
                str(args.adaptive_max_depth),
            ]
        )
    return cmd


def run_command(cmd: list[str], threads: int, progress_step: int, capture: bool = True) -> str:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["RECT_PROGRESS_STEP"] = str(progress_step)
    proc = subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=capture,
        env=env,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return proc.stdout


def query_total_tasks(args: argparse.Namespace) -> int:
    cmd = build_base_command(args) + ["--task-end", "0"]
    out = run_command(cmd, threads=1, progress_step=args.rect_progress_step)
    match = TOTAL_TASKS_RE.search(out)
    if not match:
        raise RuntimeError("failed to parse total task count from setup output")
    return int(match.group(1))


def parse_offsets(offsets_text: str) -> list[int]:
    offsets = []
    for piece in offsets_text.split(","):
        piece = piece.strip()
        if piece:
            offsets.append(int(piece))
    return offsets


def parse_task_indices(indices_text: str) -> list[int]:
    indices = []
    for piece in indices_text.split(","):
        piece = piece.strip()
        if piece:
            indices.append(int(piece))
    return indices


def choose_offsets(stride: int, samples: int) -> list[int]:
    if samples <= 0:
        raise ValueError("--samples must be positive")
    if samples > stride:
        samples = stride
    return [((2 * i + 1) * stride) // (2 * samples) for i in range(samples)]


def choose_task_indices(total_tasks: int, samples: int) -> list[int]:
    if samples <= 0:
        raise ValueError("--samples must be positive")
    if samples > total_tasks:
        samples = total_tasks
    return [((2 * i + 1) * total_tasks) // (2 * samples) for i in range(samples)]


def choose_block_starts(total_tasks: int, samples: int, block_size: int) -> list[int]:
    if block_size <= 0:
        raise ValueError("--block-size must be positive")
    if block_size >= total_tasks:
        return [0]
    starts = []
    for index in choose_task_indices(total_tasks, samples):
        start = index - block_size // 2
        if start < 0:
            start = 0
        if start + block_size > total_tasks:
            start = total_tasks - block_size
        starts.append(start)
    deduped = []
    seen = set()
    for start in starts:
        if start in seen:
            continue
        seen.add(start)
        deduped.append(start)
    return deduped


def parse_sample_output(offset: int, out: str) -> SampleResult:
    worker_match = WORKER_ELAPSED_RE.search(out)
    total_match = TOTAL_ELAPSED_RE.search(out)
    prefix_match = PREFIX_ELAPSED_RE.search(out)
    if not worker_match or not total_match or not prefix_match:
        raise RuntimeError(f"failed to parse timings for offset {offset}")
    return SampleResult(
        offset=offset,
        worker_seconds=float(worker_match.group(1)),
        total_seconds=float(total_match.group(1)),
        prefix_seconds=float(prefix_match.group(1)),
    )


def mean_and_sd(values: Iterable[float]) -> tuple[float, float]:
    vals = list(values)
    mean = statistics.fmean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, sd


def fmt_hours(seconds: float) -> str:
    return f"{seconds / 3600.0:.2f} h"


def simulate_makespan(task_seconds: list[float], worker_count: int, task_count: int,
                      bootstrap_runs: int, seed: int) -> tuple[float, float, float]:
    rng = random.Random(seed)
    samples = []
    for _ in range(bootstrap_runs):
        loads = [0.0] * worker_count
        heapq.heapify(loads)
        for _task_idx in range(task_count):
            duration = task_seconds[rng.randrange(len(task_seconds))]
            least = heapq.heappop(loads)
            heapq.heappush(loads, least + duration)
        samples.append(max(loads))
    samples.sort()
    mean = statistics.fmean(samples)
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return mean, lo, hi


def blocks_mode(args: argparse.Namespace, total_tasks: int) -> int:
    if args.block_size <= 0:
        raise SystemExit("--block-size must be positive")
    block_starts = (
        parse_task_indices(args.task_indices)
        if args.task_indices
        else choose_block_starts(total_tasks, args.samples, args.block_size)
    )
    if not block_starts:
        raise SystemExit("no block starts selected")
    bad_block_starts = [start for start in block_starts if start < 0 or start >= total_tasks]
    if bad_block_starts:
        raise SystemExit(
            f"block starts must lie in [0, {total_tasks}); bad starts: {bad_block_starts}"
        )

    base_cmd = build_base_command(args)
    print(f"Block size: {args.block_size}")
    print(f"Sample block starts: {', '.join(str(start) for start in block_starts)}")
    print("Threads per sampled block: 1")
    print()

    if args.dry_run:
        for start in block_starts:
            end = min(total_tasks, start + args.block_size)
            cmd = base_cmd + ["--task-start", str(start), "--task-end", str(end)]
            print("DRY RUN:", "OMP_NUM_THREADS=1 RECT_PROGRESS_STEP=%d %s" %
                  (args.rect_progress_step, " ".join(cmd)))
        return 0

    samples: list[SampleResult] = []
    for index, start in enumerate(block_starts, start=1):
        end = min(total_tasks, start + args.block_size)
        task_count = end - start
        cmd = base_cmd + ["--task-start", str(start), "--task-end", str(end)]
        print(f"[{index}/{len(block_starts)}] block [{start}, {end}): {' '.join(cmd)}")
        out = run_command(cmd, threads=1, progress_step=args.rect_progress_step)
        sample = parse_sample_output(start, out)
        sample.task_count = task_count
        samples.append(sample)
        net_worker = max(0.0, sample.total_seconds - sample.prefix_seconds)
        print(
            f"  worker={sample.worker_seconds:.2f}s total={sample.total_seconds:.2f}s "
            f"prefix={sample.prefix_seconds:.2f}s net_worker={net_worker:.2f}s"
        )

    worker_per_task = [
        max(0.0, sample.total_seconds - sample.prefix_seconds) / sample.task_count
        for sample in samples
    ]
    total_per_task = [sample.total_seconds / sample.task_count for sample in samples]
    prefix_values = [sample.prefix_seconds for sample in samples]
    mean_worker_per_task, sd_worker_per_task = mean_and_sd(worker_per_task)
    mean_total_per_task, sd_total_per_task = mean_and_sd(total_per_task)
    mean_prefix, sd_prefix = mean_and_sd(prefix_values)
    n = len(worker_per_task)
    ci95_worker = 1.96 * sd_worker_per_task / math.sqrt(n) if n > 1 else 0.0

    projected_core_seconds = mean_worker_per_task * total_tasks
    projected_core_hours = projected_core_seconds / 3600.0
    single_process_wall_seconds = mean_prefix + (projected_core_seconds / args.threads)
    sim_mean, sim_lo, sim_hi = simulate_makespan(
        worker_per_task,
        worker_count=args.threads,
        task_count=total_tasks,
        bootstrap_runs=args.bootstrap_runs,
        seed=args.seed,
    )
    cluster_cores = args.cluster_machines * args.cores_per_machine
    wall_seconds_fast = projected_core_hours * 3600.0 / (cluster_cores * args.efficiency_high)
    wall_seconds_slow = projected_core_hours * 3600.0 / (cluster_cores * args.efficiency_low)

    print()
    print("Block sample summary:")
    print(f"  mean worker/task : {mean_worker_per_task:.2f}s  sd={sd_worker_per_task:.2f}s")
    print(f"  mean total/task  : {mean_total_per_task:.2f}s  sd={sd_total_per_task:.2f}s")
    print(f"  mean prefix/run  : {mean_prefix:.2f}s  sd={sd_prefix:.2f}s")
    print(f"  rough 95% CI on worker/task mean: +/- {ci95_worker:.2f}s")
    print()
    print("Projected full run:")
    print(f"  projected core-hours            : {projected_core_hours:,.0f}")
    print(f"  ideal 32-thread wall + 1 setup  : {fmt_hours(single_process_wall_seconds)}")
    print(f"  simulated 32-thread worker wall : {fmt_hours(sim_mean)}")
    print(f"  simulated 32-thread 95% band    : {fmt_hours(sim_lo)} .. {fmt_hours(sim_hi)}")
    print(
        f"  cluster wall time @ {args.cluster_machines}x{args.cores_per_machine} cores"
        f" ({args.efficiency_low:.0%}-{args.efficiency_high:.0%} eff): "
        f"{fmt_hours(wall_seconds_fast)} .. {fmt_hours(wall_seconds_slow)}"
    )
    print()
    print("Assumptions:")
    print("  - each sampled block's net worker time approximates block_size iid task costs")
    print("  - per-process prefix setup is reported separately and excluded from projected core-hours")
    print("  - this mode is intended for adaptive 7x7 probing where full-shard sampling is too expensive")
    return 0


def main() -> int:
    args = parse_args()
    total_tasks = query_total_tasks(args)
    if args.stride <= 0:
        raise SystemExit("--stride must be positive")
    if args.stride > total_tasks:
        raise SystemExit(f"--stride must be <= total task count ({total_tasks})")
    if not (0.0 < args.efficiency_low <= 1.0 and 0.0 < args.efficiency_high <= 1.0):
        raise SystemExit("efficiency bounds must be in (0, 1]")
    if args.efficiency_low > args.efficiency_high:
        raise SystemExit("--efficiency-low must be <= --efficiency-high")

    print(f"Binary: {args.binary}")
    print(f"Grid: {args.rows}x{args.cols}")
    print(f"Total tasks: {total_tasks}")
    print(f"Mode: {args.mode}")
    print()

    if args.mode == "shards":
        offsets = parse_offsets(args.offsets) if args.offsets else choose_offsets(args.stride, args.samples)
        bad_offsets = [o for o in offsets if o < 0 or o >= args.stride]
        if bad_offsets:
            raise SystemExit(f"offsets must lie in [0, {args.stride}); bad offsets: {bad_offsets}")
        base_cmd = build_base_command(args) + ["--task-stride", str(args.stride)]
        print(f"Sampling stride: {args.stride}")
        print(f"Sample offsets: {', '.join(str(o) for o in offsets)}")
        print(f"Threads per run: {args.threads}")
        print()

        if args.dry_run:
            for offset in offsets:
                cmd = base_cmd + ["--task-offset", str(offset)]
                print("DRY RUN:", "OMP_NUM_THREADS=%d RECT_PROGRESS_STEP=%d %s" %
                      (args.threads, args.rect_progress_step, " ".join(cmd)))
            return 0

        samples: list[SampleResult] = []
        for index, offset in enumerate(offsets, start=1):
            cmd = base_cmd + ["--task-offset", str(offset)]
            print(f"[{index}/{len(offsets)}] offset {offset}: {' '.join(cmd)}")
            out = run_command(cmd, threads=args.threads, progress_step=args.rect_progress_step)
            sample = parse_sample_output(offset, out)
            samples.append(sample)
            print(
                f"  worker={sample.worker_seconds:.2f}s total={sample.total_seconds:.2f}s "
                f"prefix={sample.prefix_seconds:.2f}s"
            )

        mean_total, sd_total = mean_and_sd(sample.total_seconds for sample in samples)
        mean_worker, sd_worker = mean_and_sd(sample.worker_seconds for sample in samples)
        mean_prefix, sd_prefix = mean_and_sd(sample.prefix_seconds for sample in samples)
        n = len(samples)
        ci95_total = 1.96 * sd_total / math.sqrt(n) if n > 1 else 0.0

        projected_total_seconds = mean_total * args.stride
        projected_total_seconds_lo = max(0.0, (mean_total - ci95_total) * args.stride)
        projected_total_seconds_hi = (mean_total + ci95_total) * args.stride
        projected_core_hours = projected_total_seconds * args.threads / 3600.0

        cluster_cores = args.cluster_machines * args.cores_per_machine
        wall_seconds_fast = projected_core_hours * 3600.0 / (cluster_cores * args.efficiency_high)
        wall_seconds_slow = projected_core_hours * 3600.0 / (cluster_cores * args.efficiency_low)

        print()
        print("Sample summary:")
        print(f"  mean total  : {mean_total:.2f}s  sd={sd_total:.2f}s")
        print(f"  mean worker : {mean_worker:.2f}s  sd={sd_worker:.2f}s")
        print(f"  mean prefix : {mean_prefix:.2f}s  sd={sd_prefix:.2f}s")
        print(f"  rough 95% CI on shard total mean: +/- {ci95_total:.2f}s")
        print()
        print("Projected full run:")
        print(f"  full 32-thread shard fleet time : {fmt_hours(projected_total_seconds)}")
        print(
            f"  rough 95% CI                    : {fmt_hours(projected_total_seconds_lo)}"
            f" .. {fmt_hours(projected_total_seconds_hi)}"
        )
        print(f"  projected core-hours            : {projected_core_hours:,.0f}")
        print(
            f"  cluster wall time @ {args.cluster_machines}x{args.cores_per_machine} cores"
            f" ({args.efficiency_low:.0%}-{args.efficiency_high:.0%} eff): "
            f"{fmt_hours(wall_seconds_fast)} .. {fmt_hours(wall_seconds_slow)}"
        )
        print()
        print("Assumptions:")
        print("  - total cost is approximated by stride * mean(interleaved sampled shard time)")
        print("  - each shard is run as a separate process with the same setup overhead as the sample")
        print("  - cluster wall-time estimate is derived from projected core-hours, not perfect node-level packing")
        return 0

    if args.mode == "blocks":
        return blocks_mode(args, total_tasks)

    task_indices = (
        parse_task_indices(args.task_indices)
        if args.task_indices
        else choose_task_indices(total_tasks, args.samples)
    )
    bad_task_indices = [task_index for task_index in task_indices if task_index < 0 or task_index >= total_tasks]
    if bad_task_indices:
        raise SystemExit(
            f"task indices must lie in [0, {total_tasks}); bad indices: {bad_task_indices}"
        )
    base_cmd = build_base_command(args)
    print(f"Sample task indices: {', '.join(str(i) for i in task_indices)}")
    print("Threads per sampled task: 1")
    print()

    if args.dry_run:
        for task_index in task_indices:
            cmd = base_cmd + ["--task-start", str(task_index), "--task-end", str(task_index + 1)]
            print("DRY RUN:", "OMP_NUM_THREADS=1 RECT_PROGRESS_STEP=%d %s" %
                  (args.rect_progress_step, " ".join(cmd)))
        return 0

    samples = []
    for index, task_index in enumerate(task_indices, start=1):
        cmd = base_cmd + ["--task-start", str(task_index), "--task-end", str(task_index + 1)]
        print(f"[{index}/{len(task_indices)}] task {task_index}: {' '.join(cmd)}")
        out = run_command(cmd, threads=1, progress_step=args.rect_progress_step)
        sample = parse_sample_output(task_index, out)
        samples.append(sample)
        print(
            f"  worker={sample.worker_seconds:.2f}s total={sample.total_seconds:.2f}s "
            f"prefix={sample.prefix_seconds:.2f}s"
        )

    task_totals = [sample.total_seconds for sample in samples]
    mean_total, sd_total = mean_and_sd(task_totals)
    n = len(task_totals)
    ci95_total = 1.96 * sd_total / math.sqrt(n) if n > 1 else 0.0
    projected_core_seconds = mean_total * total_tasks
    projected_core_hours = projected_core_seconds / 3600.0
    ideal_32_seconds = projected_core_seconds / args.threads
    sim_mean, sim_lo, sim_hi = simulate_makespan(
        task_totals, worker_count=args.threads, task_count=total_tasks,
        bootstrap_runs=args.bootstrap_runs, seed=args.seed
    )
    cluster_cores = args.cluster_machines * args.cores_per_machine
    wall_seconds_fast = projected_core_hours * 3600.0 / (cluster_cores * args.efficiency_high)
    wall_seconds_slow = projected_core_hours * 3600.0 / (cluster_cores * args.efficiency_low)

    print()
    print("Task sample summary:")
    print(f"  mean task total : {mean_total:.2f}s  sd={sd_total:.2f}s")
    print(f"  rough 95% CI on task mean: +/- {ci95_total:.2f}s")
    print()
    print("Projected full run:")
    print(f"  projected core-hours            : {projected_core_hours:,.0f}")
    print(f"  ideal 32-thread wall            : {fmt_hours(ideal_32_seconds)}")
    print(f"  simulated 32-thread wall        : {fmt_hours(sim_mean)}")
    print(f"  simulated 32-thread 95% band    : {fmt_hours(sim_lo)} .. {fmt_hours(sim_hi)}")
    print(
        f"  cluster wall time @ {args.cluster_machines}x{args.cores_per_machine} cores"
        f" ({args.efficiency_low:.0%}-{args.efficiency_high:.0%} eff): "
        f"{fmt_hours(wall_seconds_fast)} .. {fmt_hours(wall_seconds_slow)}"
    )
    print()
    print("Assumptions:")
    print("  - sampled individual prefix tasks are representative of the full task distribution")
    print("  - core-hours estimate comes from mean sampled task time * total task count")
    print("  - 32-thread wall estimate comes from bootstrap list-scheduling on sampled task times")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
