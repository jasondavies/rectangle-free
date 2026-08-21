#!/usr/bin/env python3
"""Sample the residual binary-completion gate after dense 9x9 classes A,B.

The B sampler is deliberately heuristic, not uniform.  Every emitted B is an
exact C4-free subset of the complement of A with the requested cardinality.
The residual test is exact: its two-colouring is a NAE-4-SAT instance, with
one constraint for every rectangle wholly contained in the residual mask.
This probe estimates whether residual infeasibility is remotely strong enough
to rescue explicit enumeration of the trillions of admissible B classes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import random


ROWS = 9
COLS = 9
FULL = (1 << 81) - 1


def pair_mask(row: int) -> int:
    output = 0
    next_pair = 0
    for first in range(COLS):
        for second in range(first + 1, COLS):
            if row & (1 << first) and row & (1 << second):
                output |= 1 << next_pair
            next_pair += 1
    return output


def row_choices(allowed: int) -> list[tuple[int, int, int]]:
    output = []
    subset = allowed
    while True:
        output.append((subset, pair_mask(subset), subset.bit_count()))
        if not subset:
            break
        subset = (subset - 1) & allowed
    return output


def sample_second(first_rows: list[int], target: int, rng: random.Random) -> int:
    choices = [row_choices(0x1ff ^ row) for row in first_rows]

    def search(row: int, used_pairs: int, remaining: int, result: int) -> int | None:
        if row == ROWS:
            return result if remaining == 0 else None
        candidates = [
            choice for choice in choices[row]
            if choice[2] <= remaining and not (choice[1] & used_pairs)
        ]
        rng.shuffle(candidates)
        # Prefer the degree needed to spread the remaining cells evenly, but
        # retain randomness within and around that degree.
        ideal = remaining / (ROWS - row)
        candidates.sort(key=lambda choice: abs(choice[2] - ideal) + rng.random())
        for cells, pairs, degree in candidates:
            future_maximum = 0
            for future in range(row + 1, ROWS):
                future_maximum += max(
                    (candidate[2] for candidate in choices[future]
                     if not ((used_pairs | pairs) & candidate[1])),
                    default=-100,
                )
            if remaining - degree > future_maximum:
                continue
            found = search(
                row + 1,
                used_pairs | pairs,
                remaining - degree,
                result | (cells << (row * COLS)),
            )
            if found is not None:
                return found
        return None

    found = search(0, 0, target, 0)
    if found is None:
        raise RuntimeError(f"failed to sample a B class of size {target}")
    return found


def rectangle_constraints(residual: int) -> tuple[list[int], int]:
    cells = [cell for cell in range(81) if residual & (1 << cell)]
    indices = {cell: index for index, cell in enumerate(cells)}
    constraints = []
    occurrence = [0] * len(cells)
    for first_row in range(ROWS):
        for second_row in range(first_row + 1, ROWS):
            for first_col in range(COLS):
                for second_col in range(first_col + 1, COLS):
                    corners = (
                        first_row * COLS + first_col,
                        first_row * COLS + second_col,
                        second_row * COLS + first_col,
                        second_row * COLS + second_col,
                    )
                    if all(residual & (1 << cell) for cell in corners):
                        support = sum(1 << indices[cell] for cell in corners)
                        constraints.append(support)
                        for cell in corners:
                            occurrence[indices[cell]] += 1
    order = sorted(range(len(cells)), key=occurrence.__getitem__, reverse=True)
    rank = {variable: position for position, variable in enumerate(order)}
    # Renumber to make the branching heuristic a cheap first-set-bit choice.
    remapped = []
    for support in constraints:
        target = 0
        while support:
            bit = support & -support
            variable = bit.bit_length() - 1
            target |= 1 << rank[variable]
            support -= bit
        remapped.append(target)
    return remapped, len(cells)


def binary_completion_exists(residual: int) -> tuple[bool, int, int]:
    constraints, variables = rectangle_constraints(residual)
    nodes = 0

    def solve(assigned: int, ones: int) -> bool:
        nonlocal nodes
        nodes += 1
        while True:
            forced_variable = -1
            forced_value = 0
            for support in constraints:
                known = support & assigned
                count = known.bit_count()
                if count == 4:
                    selected = (support & ones).bit_count()
                    if selected == 0 or selected == 4:
                        return False
                elif count == 3:
                    selected = (support & ones).bit_count()
                    if selected == 0 or selected == 3:
                        unknown = support & ~assigned
                        variable = unknown.bit_length() - 1
                        value = 1 if selected == 0 else 0
                        if forced_variable >= 0 and forced_variable == variable \
                                and forced_value != value:
                            return False
                        forced_variable = variable
                        forced_value = value
            if forced_variable < 0:
                break
            bit = 1 << forced_variable
            assigned |= bit
            if forced_value:
                ones |= bit
            else:
                ones &= ~bit
        if assigned.bit_count() == variables:
            return True
        unassigned = ((1 << variables) - 1) & ~assigned
        bit = unassigned & -unassigned
        return solve(assigned | bit, ones) or solve(assigned | bit, ones | bit)

    return solve(0, 0), len(constraints), nodes


def one_sample(arguments: tuple[list[int], int, int]) -> dict[str, int | bool]:
    first_rows, target, seed = arguments
    second = sample_second(first_rows, target, random.Random(seed))
    first = sum(row << (index * COLS) for index, row in enumerate(first_rows))
    residual = FULL ^ first ^ second
    satisfiable, rectangles, nodes = binary_completion_exists(residual)
    return {
        "second_edges": target,
        "residual_cells": residual.bit_count(),
        "residual_rectangles": rectangles,
        "satisfiable": satisfiable,
        "search_nodes": nodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--samples-per-size", type=int, default=100)
    parser.add_argument("--minimum", type=int, default=18)
    parser.add_argument("--maximum", type=int, default=29)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=392)
    options = parser.parse_args()
    first_rows = [int(field, 16) for field in options.first.split(',')]
    if len(first_rows) != ROWS:
        parser.error("--first requires nine hexadecimal rows")
    jobs = [
        (first_rows, size, options.seed + size * 1_000_003 + sample)
        for size in range(options.minimum, options.maximum + 1)
        for sample in range(options.samples_per_size)
    ]
    with ProcessPoolExecutor(max_workers=options.workers) as executor:
        results = list(executor.map(one_sample, jobs, chunksize=1))
    summary = []
    for size in range(options.minimum, options.maximum + 1):
        selected = [result for result in results if result["second_edges"] == size]
        satisfiable = sum(bool(result["satisfiable"]) for result in selected)
        summary.append({
            "second_edges": size,
            "samples": len(selected),
            "satisfiable": satisfiable,
            "satisfiable_fraction": satisfiable / len(selected),
            "mean_rectangles": sum(int(result["residual_rectangles"])
                                   for result in selected) / len(selected),
            "mean_search_nodes": sum(int(result["search_nodes"])
                                      for result in selected) / len(selected),
            "maximum_search_nodes": max(int(result["search_nodes"])
                                        for result in selected),
        })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
