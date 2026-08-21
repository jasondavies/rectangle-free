#!/usr/bin/env python3
"""Summarise RECT_HARD_MISS_LOG files and their canonical-key overlap."""

from __future__ import annotations

import collections
import itertools
import struct
import sys
from pathlib import Path


HEADER = struct.Struct("<8sIIIII")
RECORD_PREFIX = struct.Struct("<BBHQ")
MAGIC = b"RHMISS1\0"


def read_log(path: Path):
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise SystemExit(f"{path}: too short for header")
    magic, version, maxn, adjword_bits, rows, cols = HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise SystemExit(f"{path}: bad magic {magic!r}")
    if version != 1:
        raise SystemExit(f"{path}: unsupported version {version}")
    if adjword_bits not in (16, 32, 64):
        raise SystemExit(f"{path}: unsupported AdjWord width {adjword_bits}")

    record_size = RECORD_PREFIX.size + 8 * maxn
    payload = data[HEADER.size :]
    if len(payload) % record_size != 0:
        raise SystemExit(
            f"{path}: payload size {len(payload)} is not a multiple of record size {record_size}"
        )

    counts: collections.Counter[tuple[int, tuple[int, ...]]] = collections.Counter()
    by_n: collections.Counter[int] = collections.Counter()
    by_degree: collections.Counter[tuple[int, int]] = collections.Counter()

    offset = HEADER.size
    total = 0
    while offset < len(data):
        n, max_degree, _reserved, _hash = RECORD_PREFIX.unpack_from(data, offset)
        offset += RECORD_PREFIX.size
        row_values = struct.unpack_from(f"<{maxn}Q", data, offset)
        offset += 8 * maxn
        key = (n, tuple(row_values[:n]))
        counts[key] += 1
        by_n[n] += 1
        by_degree[(n, max_degree)] += 1
        total += 1

    return {
        "path": path,
        "rows": rows,
        "cols": cols,
        "maxn": maxn,
        "total": total,
        "counts": counts,
        "by_n": by_n,
        "by_degree": by_degree,
    }


def print_one(log) -> None:
    counts = log["counts"]
    total = log["total"]
    unique = len(counts)
    duplicate_records = total - unique
    repeated_keys = sum(1 for v in counts.values() if v > 1)
    print(f"{log['path']}:")
    print(f"  grid: {log['rows']}x{log['cols']} maxn={log['maxn']}")
    print(f"  records: {total}")
    print(f"  unique: {unique}")
    print(f"  duplicate records: {duplicate_records}")
    print(f"  repeated unique keys: {repeated_keys}")
    if total:
        print(f"  duplicate record rate: {duplicate_records / total:.2%}")
    if log["by_n"]:
        parts = ", ".join(f"n={n}:{c}" for n, c in sorted(log["by_n"].items()))
        print(f"  records by n: {parts}")
    hot = log["counts"].most_common(5)
    if hot:
        hot_parts = ", ".join(f"n={key[0]} x{count}" for key, count in hot)
        print(f"  hottest keys: {hot_parts}")


def print_overlap(logs) -> None:
    for a, b in itertools.combinations(logs, 2):
        keys_a = set(a["counts"])
        keys_b = set(b["counts"])
        overlap = keys_a & keys_b
        overlap_records_a = sum(a["counts"][key] for key in overlap)
        overlap_records_b = sum(b["counts"][key] for key in overlap)
        print(f"{a['path']} vs {b['path']}:")
        print(f"  overlap unique keys: {len(overlap)}")
        if keys_a:
            print(f"  overlap of left: {len(overlap) / len(keys_a):.2%}")
        if a["total"]:
            print(f"  overlap records in left: {overlap_records_a / a['total']:.2%}")
        if keys_b:
            print(f"  overlap of right: {len(overlap) / len(keys_b):.2%}")
        if b["total"]:
            print(f"  overlap records in right: {overlap_records_b / b['total']:.2%}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(f"usage: {argv[0]} HARDMISS.bin [HARDMISS.bin ...]", file=sys.stderr)
        return 2

    logs = [read_log(Path(arg)) for arg in argv[1:]]
    for log in logs:
        print_one(log)
    if len(logs) > 1:
        print_overlap(logs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
