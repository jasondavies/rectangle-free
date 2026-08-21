#!/usr/bin/env python3
"""Exact gates for the dense-first direct-colour formulation.

For a valid four-colouring, order its four C4-free colour-class masks by
``(cardinality, mask)``.  If A and B are the first two classes, then

    |A| >= ceil(rc/4),
    |B| >= ceil((rc-|A|)/3).

For fixed A and B, the residual cells have an ordered binary completion C,D.
Requiring C,D <= B in the same order makes A,B the unique two densest classes.
The ordered C,D count is multiplied by 12: normally it contains both residual
orientations and an unordered four-block partition has 24 colour labellings;
when C=D=empty, the same factor is the exact 4*3 choice of labels for A,B.

The degree census applies the necessary C4-free two-path inequality on one
side.  The same list applies independently to rows and columns for a square.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import itertools
import json
import math


def c4_free(mask: int, rows: int, columns: int) -> bool:
    row_mask = (1 << columns) - 1
    seen_column_pairs = 0
    for row in range(rows):
        bits = (mask >> (row * columns)) & row_mask
        pair_mask = 0
        for first in range(columns):
            if not bits & (1 << first):
                continue
            for second in range(first + 1, columns):
                if bits & (1 << second):
                    pair = first * columns + second
                    pair_mask |= 1 << pair
        if seen_column_pairs & pair_mask:
            return False
        seen_column_pairs |= pair_mask
    return True


def class_key(mask: int) -> tuple[int, int]:
    return mask.bit_count(), mask


def direct_count(rows: int, columns: int) -> int:
    cells = rows * columns
    answer = 0
    for colouring in itertools.product(range(4), repeat=cells):
        classes = [0, 0, 0, 0]
        for cell, colour in enumerate(colouring):
            classes[colour] |= 1 << cell
        answer += all(c4_free(mask, rows, columns) for mask in classes)
    return answer


def dense_first_count(rows: int, columns: int) -> int:
    """Count exactly through canonical densest-two classes.

    Intended only for small validation grids: it examines all ternary choices
    A/B/residual and all binary residual completions.
    """
    cells = rows * columns
    full = (1 << cells) - 1
    minimum_first = (cells + 3) // 4
    free = [c4_free(mask, rows, columns) for mask in range(full + 1)]
    ordered_completions = 0
    for first in range(full + 1):
        if first.bit_count() < minimum_first or not free[first]:
            continue
        first_key = class_key(first)
        available = full ^ first
        second = available
        while True:
            if free[second] and first_key > class_key(second):
                residual = available ^ second
                third = residual
                second_key = class_key(second)
                while True:
                    fourth = residual ^ third
                    if (free[third] and free[fourth]
                            and second_key >= class_key(third)
                            and second_key >= class_key(fourth)):
                        ordered_completions += 1
                    if not third:
                        break
                    third = (third - 1) & residual
            if not second:
                break
            second = (second - 1) & available
    return 12 * ordered_completions


@lru_cache(maxsize=None)
def degree_partitions(total: int, slots: int, maximum: int) -> tuple[tuple[int, ...], ...]:
    if slots == 0:
        return ((),) if total == 0 else ()
    output: list[tuple[int, ...]] = []
    for degree in range(min(total, maximum), -1, -1):
        for suffix in degree_partitions(total - degree, slots - 1, degree):
            output.append((degree,) + suffix)
    return tuple(output)


def feasible_degree_partitions(vertices: int, edges: int) -> list[tuple[int, ...]]:
    pair_budget = math.comb(vertices, 2)
    return [
        degrees
        for degrees in degree_partitions(edges, vertices, vertices)
        if sum(math.comb(degree, 2) for degree in degrees) <= pair_budget
    ]


def degree_census(vertices: int, maximum_edges: int) -> list[dict[str, int]]:
    cells = vertices * vertices
    minimum = (cells + 3) // 4
    records = []
    for first_edges in range(minimum, maximum_edges + 1):
        sequences = feasible_degree_partitions(vertices, first_edges)
        records.append({
            "first_edges": first_edges,
            "minimum_second_edges": (cells - first_edges + 2) // 3,
            "one_side_degree_sequences": len(sequences),
            "minimum_two_paths": min(
                sum(math.comb(degree, 2) for degree in sequence)
                for sequence in sequences
            ),
            "maximum_two_paths": max(
                sum(math.comb(degree, 2) for degree in sequence)
                for sequence in sequences
            ),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    degree = subparsers.add_parser("degrees")
    degree.add_argument("--vertices", type=int, default=9)
    degree.add_argument("--maximum-edges", type=int, default=30)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--rows", type=int, default=3)
    validate.add_argument("--columns", type=int, default=3)
    arguments = parser.parse_args()

    if arguments.command == "degrees":
        print(json.dumps(
            degree_census(arguments.vertices, arguments.maximum_edges),
            indent=2,
        ))
    else:
        dense = dense_first_count(arguments.rows, arguments.columns)
        direct = direct_count(arguments.rows, arguments.columns)
        print(json.dumps({
            "rows": arguments.rows,
            "columns": arguments.columns,
            "dense_first_count": dense,
            "direct_count": direct,
            "match": dense == direct,
        }, indent=2))
        if dense != direct:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
