#!/usr/bin/env python3
"""Full row/colour-symmetry gate for the universal direct-colour state.

This probe addresses a symmetry that ordered decision diagrams do not use.
Token states are four graph masks on the row set.  ``S_rows`` acts
simultaneously on every graph and ``S_4`` permutes the four graph planes.

The transfer stores total coefficient mass per state orbit.  Applying the
one-column operator to one representative is exact: invariance makes the
number of transitions from every member of a source orbit into a target orbit
identical.  Thus multiplying that representative transition count by the
source orbit's total mass gives the target orbit's exact total mass.

The Burnside census is a cheaper upper bound for three-column support orbits.
After quotienting rows, a valid grid is a compatible subset of the 64 possible
three-symbol row words.  We quotient these subsets by global colour and block-
column permutations.  Distinct word-set orbits can still produce the same
token support, hence the result is an upper bound rather than the transfer's
exact support-orbit count.
"""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
from itertools import permutations, product
import json
import math
import resource
import time
from typing import Iterable


State = tuple[int, int, int, int]


def row_pairs(rows: int) -> list[tuple[int, int]]:
    return [(first, second) for first in range(rows)
            for second in range(first + 1, rows)]


class StateCanonicalizer:
    def __init__(self, rows: int, cache_size: int | None = None) -> None:
        self.rows = rows
        pairs = row_pairs(rows)
        pair_index = {pair: index for index, pair in enumerate(pairs)}
        domain = 1 << len(pairs)
        self.images: list[list[int]] = []
        for permutation in permutations(range(rows)):
            bit_images = []
            for first, second in pairs:
                image = tuple(sorted((permutation[first], permutation[second])))
                bit_images.append(1 << pair_index[image])
            table = [0] * domain
            for mask in range(1, domain):
                bit = mask & -mask
                table[mask] = (
                    table[mask - bit] | bit_images[bit.bit_length() - 1]
                )
            self.images.append(table)

        @lru_cache(maxsize=cache_size)
        def cached(state: State) -> State:
            best: State | None = None
            for images in self.images:
                candidate = tuple(sorted(images[plane] for plane in state))
                if best is None or candidate < best:
                    best = candidate
            assert best is not None
            return best

        self._cached = cached

    def __call__(self, state: State) -> State:
        return self._cached(state)

    def cache_info(self):
        return self._cached.cache_info()


def one_column_supports(rows: int) -> list[tuple[State, int]]:
    pairs = row_pairs(rows)
    counts: Counter[State] = Counter()
    for colouring in product(range(4), repeat=rows):
        planes = [0, 0, 0, 0]
        for index, (first, second) in enumerate(pairs):
            if colouring[first] == colouring[second]:
                planes[colouring[first]] |= 1 << index
        counts[tuple(planes)] += 1
    return sorted(counts.items())


def disjoint(left: State, right: State) -> bool:
    return all(not (first & second) for first, second in zip(left, right))


def quotient_transfer(
    rows: int,
    columns: int,
    max_states: int,
    cache_size: int | None,
) -> list[dict[str, int | float]]:
    canonicalize = StateCanonicalizer(rows, cache_size)
    column_supports = one_column_supports(rows)
    states: dict[State, int] = {(0, 0, 0, 0): 1}
    records: list[dict[str, int | float]] = [{
        "columns": 0,
        "orbit_states": 1,
        "coefficient_sum": 1,
        "seconds": 0.0,
    }]
    for column_count in range(1, columns + 1):
        started = time.monotonic()
        following: dict[State, int] = {}
        for state, orbit_mass in states.items():
            target_counts: Counter[State] = Counter()
            for support, multiplicity in column_supports:
                if not disjoint(state, support):
                    continue
                target = canonicalize(tuple(
                    left | right for left, right in zip(state, support)
                ))
                target_counts[target] += multiplicity
            for target, transition_count in target_counts.items():
                following[target] = (
                    following.get(target, 0)
                    + orbit_mass * transition_count
                )
            if len(following) > max_states:
                raise RuntimeError(
                    f"state cap exceeded while constructing P^{column_count}: "
                    f"more than {max_states} target orbits"
                )
        states = following
        records.append({
            "columns": column_count,
            "orbit_states": len(states),
            "coefficient_sum": sum(states.values()),
            "seconds": round(time.monotonic() - started, 6),
            "canonical_cache_entries": canonicalize.cache_info().currsize,
        })
    return records


def three_column_codes() -> list[tuple[int, int, int]]:
    return [tuple((encoded >> (2 * position)) & 3 for position in range(3))
            for encoded in range(64)]


def compatible_codes(
    first: tuple[int, int, int],
    second: tuple[int, int, int],
) -> bool:
    if first == second:
        return False
    return all(
        sum(left == right == colour
            for left, right in zip(first, second)) < 2
        for colour in range(4)
    )


def fixed_subset_counts(
    colour_permutation: tuple[int, ...],
    column_permutation: tuple[int, ...],
    maximum_rows: int,
) -> list[int]:
    codes = three_column_codes()
    code_index = {code: index for index, code in enumerate(codes)}
    compatible = [[compatible_codes(first, second) for second in codes]
                  for first in codes]
    action = [
        code_index[tuple(
            colour_permutation[code[column_permutation[position]]]
            for position in range(3)
        )]
        for code in codes
    ]

    seen: set[int] = set()
    cycles: list[list[int]] = []
    for vertex in range(64):
        if vertex in seen:
            continue
        cycle = []
        current = vertex
        while current not in seen:
            seen.add(current)
            cycle.append(current)
            current = action[current]
        if len(cycle) > maximum_rows:
            continue
        if all(compatible[first][second]
               for index, first in enumerate(cycle)
               for second in cycle[index + 1:]):
            cycles.append(cycle)

    adjacency: list[int] = []
    for index, cycle in enumerate(cycles):
        mask = 0
        for other_index in range(index + 1, len(cycles)):
            if all(compatible[first][second]
                   for first in cycle for second in cycles[other_index]):
                mask |= 1 << other_index
        adjacency.append(mask)
    weights = tuple(len(cycle) for cycle in cycles)

    @lru_cache(maxsize=None)
    def count(candidate_mask: int, remaining: int) -> int:
        if remaining == 0:
            return 1
        if remaining < 0:
            return 0
        result = 0
        while candidate_mask:
            bit = candidate_mask & -candidate_mask
            candidate_mask -= bit
            index = bit.bit_length() - 1
            result += count(
                candidate_mask & adjacency[index],
                remaining - weights[index],
            )
        return result

    complete = (1 << len(cycles)) - 1
    return [count(complete, rows) for rows in range(maximum_rows + 1)]


def burnside_census(maximum_rows: int) -> list[dict[str, int]]:
    fixed_sums = [0] * (maximum_rows + 1)
    identity_counts: list[int] | None = None
    identity_colour = tuple(range(4))
    identity_columns = tuple(range(3))
    for colour_permutation in permutations(range(4)):
        for column_permutation in permutations(range(3)):
            counts = fixed_subset_counts(
                colour_permutation, column_permutation, maximum_rows
            )
            if (colour_permutation == identity_colour
                    and column_permutation == identity_columns):
                identity_counts = counts
            fixed_sums = [left + right
                          for left, right in zip(fixed_sums, counts)]
    assert identity_counts is not None
    group_order = math.factorial(4) * math.factorial(3)
    assert all(value % group_order == 0 for value in fixed_sums)
    return [{
        "rows": rows,
        "valid_unlabelled_row_sets": identity_counts[rows],
        "colour_column_orbits": fixed_sums[rows] // group_order,
    } for rows in range(maximum_rows + 1)]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    burnside = subparsers.add_parser("burnside")
    burnside.add_argument("--maximum-rows", type=int, default=9)
    transfer = subparsers.add_parser("transfer")
    transfer.add_argument("rows", type=int)
    transfer.add_argument("columns", type=int)
    transfer.add_argument("--max-states", type=int, default=1_000_000)
    transfer.add_argument("--cache-size", type=int)
    return result


def main() -> int:
    args = parser().parse_args()
    started = time.monotonic()
    try:
        if args.command == "burnside":
            if not 0 <= args.maximum_rows <= 9:
                raise SystemExit("maximum rows must be in 0..9")
            result: Iterable[dict[str, int | float]] = burnside_census(
                args.maximum_rows
            )
        else:
            if not 2 <= args.rows <= 5:
                raise SystemExit(
                    "reference transfer rows must be in 2..5; use the compiled "
                    "follow-up for larger row-permutation groups"
                )
            if args.columns < 0 or args.max_states < 1:
                raise SystemExit("columns must be nonnegative and cap positive")
            result = quotient_transfer(
                args.rows, args.columns, args.max_states, args.cache_size
            )
        for record in result:
            print(json.dumps(record))
        print(json.dumps({
            "kind": "symmetry_probe_summary",
            "command": args.command,
            "seconds": round(time.monotonic() - started, 6),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }))
        return 0
    except RuntimeError as error:
        print(json.dumps({
            "kind": "symmetry_probe_resource_limit",
            "reason": str(error),
            "seconds": round(time.monotonic() - started, 6),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
