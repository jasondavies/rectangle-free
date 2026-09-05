#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import os
import re
import tempfile
import sys
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass
class PolyFileMeta:
    rows: int
    cols: int
    task_start: int
    task_end: int
    full_tasks: int
    version: int = 1
    algorithm: str = ""
    solver_source: str = ""
    mode: str = "polynomial"
    task_space: str = ""
    prefix_depth: int = 0
    reorder: int = 0


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
    """Strict V1/V2 reader. Legacy task identity remains unverified."""
    try:
        data = path.read_bytes()
        lines = data.decode("ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        fail(f"Cannot read {path}: {exc}")
    if not lines or lines[0] not in ("RECT_POLY_V1", "RECT_POLY_V2"):
        fail(f"Invalid header in {path}")
    version = int(lines[0][-1])
    if lines[-1] != "end" or lines.count("end") != 1:
        fail(f"Missing or misplaced end marker in {path}")
    if version == 2:
        if data != ("\n".join(lines) + "\n").encode("ascii"):
            fail(f"Noncanonical V2 line endings in {path}")
        if len(lines) < 3 or not re.fullmatch(r"sha256 [0-9a-f]{64}", lines[-2]):
            fail(f"Missing checksum in {path}")
        payload = ("\n".join(lines[:-2]) + "\n").encode("ascii")
        if hashlib.sha256(payload).hexdigest() != lines[-2][7:]:
            fail(f"Checksum mismatch in {path}")
        body = lines[1:-2]
    else:
        body = lines[1:-1]
    common = {"rows", "cols", "task_start", "task_end", "full_tasks", "deg"}
    extra = {"algorithm", "solver_source", "mode", "prefix_depth", "reorder", "task_space"}
    allowed = common | (extra if version == 2 else set())
    fields, coeffs = {}, {}
    for line in body:
        parts = line.split()
        if len(parts) == 3 and parts[0] == "coeff":
            idx = parse_int(parts[1], str(path))
            if idx < 0 or idx in coeffs:
                fail(f"Invalid or duplicate coefficient in {path}")
            coeffs[idx] = parse_int(parts[2], str(path))
        elif len(parts) == 2 and parts[0] in allowed and parts[0] not in fields:
            fields[parts[0]] = parts[1]
        else:
            fail(f"Unknown, duplicate or malformed field in {path}: {line}")
    if fields.keys() != allowed:
        fail(f"Incomplete metadata in {path}")
    nums = {k: parse_int(fields[k], str(path)) for k in common}
    degree = nums["deg"]
    if nums["rows"] <= 0 or nums["cols"] <= 0 or not 0 <= degree <= nums["rows"] * nums["cols"]:
        fail(f"Invalid geometry or degree in {path}")
    if len(coeffs) != degree + 1 or any(i not in coeffs for i in range(degree + 1)):
        fail(f"Missing or out-of-range coefficients in {path}")
    if not 0 <= nums["task_start"] <= nums["task_end"] <= nums["full_tasks"]:
        fail(f"Invalid task range in {path}")
    if nums["task_start"] == nums["task_end"] and any(coeffs.values()):
        fail(f"Nonzero contribution for an empty task range in {path}")
    kwargs = {}
    if version == 2:
        if fields["algorithm"] != "partition-structure-v2":
            fail(f"Unsupported algorithm in {path}")
        if fields["mode"] not in ("polynomial", "count4"):
            fail(f"Invalid mode in {path}")
        if fields["mode"] == "count4" and degree != 0:
            fail(f"Count4 shard has polynomial coefficients in {path}")
        for key in ("task_space", "solver_source"):
            if not re.fullmatch(r"[0-9a-f]{64}", fields[key]):
                fail(f"Invalid {key} in {path}")
        depth = parse_int(fields["prefix_depth"], str(path))
        reorder = parse_int(fields["reorder"], str(path))
        if depth not in (0, 2, 3, 4) or depth > nums["cols"] or reorder not in (0, 1):
            fail(f"Invalid task configuration in {path}")
        kwargs = dict(algorithm=fields["algorithm"], solver_source=fields["solver_source"],
                      task_space=fields["task_space"], mode=fields["mode"],
                      prefix_depth=depth, reorder=reorder)
    meta = PolyFileMeta(**{k: nums[k] for k in common - {"deg"}}, version=version, **kwargs)
    return Poly([coeffs[i] for i in range(degree + 1)]), meta


def parse_int(text: str, label: str) -> int:
    if not re.fullmatch(r"[+-]?[0-9]+", text):
        fail(f"Invalid integer for {label}: {text}")
    return int(text, 10)


def write_poly_file(path: Path, poly: Poly, meta: PolyFileMeta) -> None:
    lines = [f"RECT_POLY_V{meta.version}"]
    if meta.version == 2:
        lines += [f"algorithm {meta.algorithm}", f"solver_source {meta.solver_source}", f"mode {meta.mode}"]
    lines += [f"rows {meta.rows}", f"cols {meta.cols}"]
    if meta.version == 2:
        lines += [f"prefix_depth {meta.prefix_depth}", f"reorder {meta.reorder}", f"task_space {meta.task_space}"]
    lines += [f"task_start {meta.task_start}", f"task_end {meta.task_end}",
              f"full_tasks {meta.full_tasks}", f"deg {poly.deg}"]
    lines += [f"coeff {i} {value}" for i, value in enumerate(poly.coeffs)]
    payload = ("\n".join(lines) + "\n").encode("ascii")
    if meta.version == 2:
        payload += f"sha256 {hashlib.sha256(payload).hexdigest()}\n".encode("ascii")
    payload += b"end\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".tmp.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        fail(f"Cannot write {path}: {exc}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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


def merge_shards(inputs: list[Path], poly_out_path: Path | None, *, allow_legacy: bool = False) -> int:
    if not inputs:
        fail("At least one input shard is required")
    merged, meta, identity = Poly.zero(), None, None
    intervals = []
    for path in inputs:
        poly, current = parse_poly_file(path)
        if current.version == 1 and not allow_legacy:
            fail("Legacy shards lack task provenance; use --allow-legacy only for an audited historical campaign")
        current_identity = (current.version, current.rows, current.cols, current.full_tasks,
                            current.algorithm, current.solver_source, current.mode,
                            current.task_space, current.prefix_depth, current.reorder)
        if meta is None:
            meta, identity = current, current_identity
        elif current_identity != identity:
            fail(f"Incompatible polynomial shard: {path}")
        if current.task_start < current.task_end:
            intervals.append((current.task_start, current.task_end, str(path)))
        merged = merged.add(poly)
    assert meta is not None
    intervals.sort()
    covered, previous_end, contiguous = 0, None, True
    for start, end, path in intervals:
        if previous_end is not None:
            if start < previous_end:
                fail(f"Overlapping shard task {start} in {path}")
            if start != previous_end:
                contiguous = False
        covered += end - start
        previous_end = end
    if not contiguous and poly_out_path is not None:
        fail("Cannot write a merged shard with gaps in its task range")
    meta = replace(meta, task_start=intervals[0][0] if intervals else 0,
                   task_end=intervals[-1][1] if intervals else 0)
    if meta.version == 1:
        print("Warning: legacy task-space identity is unverified.", file=sys.stderr)
    print(f"Merged {len(inputs)} shard(s) for {meta.rows}x{meta.cols}")
    print(f"Covered tasks: {covered} / {meta.full_tasks}")
    if meta.mode == "count4":
        print(f"T_4 = {merged.coeffs[0]}")
    else:
        print(format_poly(merged))
        print(f"P(4) = {merged.eval(4)}")
        print(f"P(5) = {merged.eval(5)}")
    if poly_out_path is not None:
        write_poly_file(poly_out_path, merged, meta)
        print(f"Wrote merged polynomial to {poly_out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and merge V2 polynomial shards, or explicit historical V1.",
    )
    parser.add_argument("--allow-legacy", action="store_true", help="allow audited historical V1 shards")
    parser.add_argument("--poly-out", dest="poly_out", help="write the merged shard to this file")
    parser.add_argument("inputs", nargs="+", help="input shard files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return merge_shards([Path(path) for path in args.inputs], Path(args.poly_out) if args.poly_out else None, allow_legacy=args.allow_legacy)


if __name__ == "__main__":
    raise SystemExit(main())
