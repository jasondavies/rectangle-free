#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PolyFileMeta:
    rows: int
    cols: int
    task_start: int
    task_end: int
    full_tasks: int


@dataclass
class Poly:
    coeffs: list[int]

    @property
    def deg(self) -> int:
        return len(self.coeffs) - 1

    @classmethod
    def zero(cls) -> "Poly":
        return cls([0])

    def eval(self, x: int) -> int:
        value = 0
        for coeff in reversed(self.coeffs):
            value = value * x + coeff
        return value

    def add(self, other: "Poly") -> "Poly":
        limit = max(len(self.coeffs), len(other.coeffs))
        coeffs = [0] * limit
        for idx in range(limit):
            a = self.coeffs[idx] if idx < len(self.coeffs) else 0
            b = other.coeffs[idx] if idx < len(other.coeffs) else 0
            coeffs[idx] = a + b
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        return Poly(coeffs)


def fail(message: str) -> "None":
    print(message, file=sys.stderr)
    raise SystemExit(1)


def parse_poly_file(path: Path) -> tuple[Poly, PolyFileMeta]:
    try:
        with path.open("r", encoding="ascii") as handle:
            lines = [line.rstrip("\r\n") for line in handle]
    except OSError:
        fail(f"Failed to open {path} for reading")

    if not lines:
        fail(f"Failed to read header from {path}")
    if lines[0] != "RECT_POLY_V1":
        fail(f"Invalid polynomial file header in {path}")

    rows = -1
    cols = -1
    task_start = 0
    task_end = 0
    full_tasks = -1
    coeffs: dict[int, int] = {}

    for line in lines[1:]:
        if line == "end":
            break
        if line.startswith("rows "):
            rows = parse_int(line[5:], str(path))
            continue
        if line.startswith("cols "):
            cols = parse_int(line[5:], str(path))
            continue
        if line.startswith("task_start "):
            task_start = parse_int(line[11:], str(path))
            continue
        if line.startswith("task_end "):
            task_end = parse_int(line[9:], str(path))
            continue
        if line.startswith("full_tasks "):
            full_tasks = parse_int(line[11:], str(path))
            continue
        if line.startswith("deg "):
            continue
        if line.startswith("coeff "):
            payload = line[6:]
            try:
                idx_text, value_text = payload.split(" ", 1)
            except ValueError:
                fail(f"Invalid coefficient line in {path}: {line}")
            idx = parse_int(idx_text, str(path))
            if idx < 0:
                fail(f"Invalid coefficient line in {path}: {line}")
            value_text = value_text.lstrip(" ")
            if not value_text:
                fail(f"Invalid coefficient line in {path}: {line}")
            coeffs[idx] = parse_int(value_text, str(path))
            continue
        fail(f"Unrecognised line in {path}: {line}")

    if rows < 0 or cols < 0 or full_tasks < 0:
        fail(f"Incomplete metadata in {path}")

    degree = 0
    for idx, value in coeffs.items():
        if value != 0 and idx > degree:
            degree = idx
    dense_coeffs = [0] * (degree + 1)
    for idx, value in coeffs.items():
        if idx < len(dense_coeffs):
            dense_coeffs[idx] = value

    return Poly(dense_coeffs), PolyFileMeta(
        rows=rows,
        cols=cols,
        task_start=task_start,
        task_end=task_end,
        full_tasks=full_tasks,
    )


def parse_int(text: str, label: str) -> int:
    if not text:
        fail(f"Missing integer for {label}")
    try:
        return int(text, 10)
    except ValueError:
        fail(f"Invalid integer for {label}: {text}")


def write_poly_file(path: Path, poly: Poly, meta: PolyFileMeta) -> None:
    try:
        with path.open("w", encoding="ascii") as handle:
            handle.write("RECT_POLY_V1\n")
            handle.write(f"rows {meta.rows}\n")
            handle.write(f"cols {meta.cols}\n")
            handle.write(f"task_start {meta.task_start}\n")
            handle.write(f"task_end {meta.task_end}\n")
            handle.write(f"full_tasks {meta.full_tasks}\n")
            handle.write(f"deg {poly.deg}\n")
            for idx, value in enumerate(poly.coeffs):
                handle.write(f"coeff {idx} {value}\n")
            handle.write("end\n")
    except OSError:
        fail(f"Failed to open {path} for writing")


def format_poly(poly: Poly) -> str:
    terms: list[str] = []
    for power in range(poly.deg, -1, -1):
        coeff = poly.coeffs[power]
        if coeff == 0:
            continue

        magnitude = abs(coeff)
        pieces: list[str] = []
        if magnitude != 1 or power == 0:
            pieces.append(str(magnitude))
            if power > 0:
                pieces.append("*")
        if power > 0:
            pieces.append("x")
            if power > 1:
                pieces.append(f"^{power}")
        term = "".join(pieces)
        if not terms:
            terms.append(f"-{term}" if coeff < 0 else term)
        else:
            sign = " - " if coeff < 0 else " + "
            terms.append(f"{sign}{term}")

    return "P(x) = 0" if not terms else "P(x) = " + "".join(terms)


def merge_shards(inputs: list[Path], poly_out_path: Path | None) -> int:
    if not inputs:
        fail("At least one input shard is required")

    merged = Poly.zero()
    merged_meta: PolyFileMeta | None = None
    task_seen: list[bool] | None = None
    covered_tasks = 0
    first_poly: Poly | None = None
    first_meta: PolyFileMeta | None = None

    for index, input_path in enumerate(inputs):
        current_poly, current_meta = parse_poly_file(input_path)

        if (
            current_meta.task_start < 0
            or current_meta.task_end < current_meta.task_start
            or current_meta.task_end > current_meta.full_tasks
        ):
            fail(f"Invalid task selection in shard: {input_path}")

        if index == 0:
            merged_meta = current_meta
            first_poly = current_poly
            first_meta = current_meta
            try:
                task_seen = [False] * merged_meta.full_tasks
            except MemoryError:
                fail("Failed to allocate merge task bitmap")
        else:
            assert merged_meta is not None
            if (
                current_meta.rows != merged_meta.rows
                or current_meta.cols != merged_meta.cols
                or current_meta.full_tasks != merged_meta.full_tasks
            ):
                fail(f"Incompatible polynomial shard: {input_path}")

        assert task_seen is not None
        for task in range(current_meta.task_start, current_meta.task_end):
            if task_seen[task]:
                fail(f"Overlapping shard task {task} in {input_path}")
            task_seen[task] = True
            covered_tasks += 1

        merged = merged.add(current_poly)

    if task_seen is None or merged_meta is None:
        fail("Failed to allocate merge task tracking")

    seen_indices = [index for index, seen in enumerate(task_seen) if seen]
    if seen_indices:
        min_task = seen_indices[0]
        max_task = seen_indices[-1] + 1
    else:
        min_task = 0
        max_task = 0

    merged_meta.task_start = min_task
    merged_meta.task_end = max_task
    contiguous_cover = all(task_seen[task] for task in range(min_task, max_task))

    if covered_tasks == merged_meta.full_tasks:
        merged_meta.task_start = 0
        merged_meta.task_end = merged_meta.full_tasks
        contiguous_cover = True

    if not contiguous_cover and len(inputs) == 1:
        assert first_poly is not None and first_meta is not None
        merged = first_poly
        merged_meta = first_meta
    elif not contiguous_cover and poly_out_path is not None:
        fail(
            f"Cannot write merged shard {poly_out_path}: input tasks are non-contiguous and incomplete"
        )

    print(f"Merged {len(inputs)} shard(s) for {merged_meta.rows}x{merged_meta.cols}")
    print(f"Covered tasks: {covered_tasks} / {merged_meta.full_tasks}")
    print()
    print("Chromatic Polynomial P(x):")
    print(format_poly(merged))
    print()
    print("Values:")
    print(f"P(4) = {merged.eval(4)}")
    print(f"P(5) = {merged.eval(5)}")

    if poly_out_path is not None:
        write_poly_file(poly_out_path, merged, merged_meta)
        print()
        print(f"Wrote merged polynomial to {poly_out_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge RECT_POLY_V1 shards using the same rules as partition_poly.c --merge.",
    )
    parser.add_argument("--poly-out", dest="poly_out", help="write the merged shard to this file")
    parser.add_argument("inputs", nargs="+", help="input shard files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return merge_shards([Path(path) for path in args.inputs], Path(args.poly_out) if args.poly_out else None)


if __name__ == "__main__":
    raise SystemExit(main())
