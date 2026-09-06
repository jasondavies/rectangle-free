#!/usr/bin/env python3
"""Pad sampled boundaries without changing children; not a production plan.

The CUDA/CPU formula must validate the changed core sign domain. Compare
normalized complete residues, not raw partial sums across the two layouts.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool", type=int, choices=(11, 13), default=13)
    args = parser.parse_args()
    output = []
    for line in args.groups.read_text().splitlines():
        if not line.startswith("CORE27_GROUP "):
            continue
        row = dict(x.split("=", 1) for x in line.split()[1:])
        boundary, parent = int(row["boundary"]), int(row["parent"])
        old = boundary.bit_count()
        if old > args.pool or not old % 2 or parent & boundary:
            raise ValueError("invalid source boundary")
        while boundary.bit_count() < args.pool:
            spare = ((1 << 60) - 1) & ~(parent | boundary)
            if not spare:
                raise ValueError("no spare token")
            boundary |= spare & -spare
        core = int(row["core"]) - (args.pool - old)
        if core < 0 or core % 2:
            raise ValueError("invalid padded core")
        row.update(boundary=str(boundary), cap=str(args.pool), core=str(core), tail=str(args.pool - 3))
        output.append("CORE27_GROUP " + " ".join(f"{k}={v}" for k, v in row.items()))
    if not output:
        raise ValueError("empty sample")
    with args.output.open("x") as file:
        file.write("# Padded sample only: same children, changed sign domains.\n")
        file.write("\n".join(output) + "\n")
    print(f"CORE_PADDING groups={len(output)} pool={args.pool} scope=sample_not_assignment")


if __name__ == "__main__":
    main()
