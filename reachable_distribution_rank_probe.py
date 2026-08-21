#!/usr/bin/env python3
"""Exact modular rank and symmetry forecasts for universal half distributions.

The one-column vector p_S is represented in the squarefree token algebra.
The ``rank`` command streams all degree-k commutative products through exact
sparse Gaussian elimination modulo a prime.  Equality with the symmetric-power
bound proves that multiplication introduces no linear relations at that degree.

The ``forecast`` command decomposes Sym^k(V_1) under S_r.  Here

    V_1 = 1 + sum_{s=2}^r F[{s-subsets}],

so its character on a permutation g is 2^cycles(g) - fixed_points(g).
This is the no-higher-relations envelope against which measured ranks and
multiplicity blocks should be compared.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from math import comb, factorial, gcd, isqrt
import resource
import time
from typing import Iterable, Iterator


DEFAULT_PRIME = 1_000_003


def require_prime(value: str | int) -> int:
    value = int(value)
    if value < 2 or any(value % divisor == 0 for divisor in range(2, isqrt(value) + 1)):
        raise argparse.ArgumentTypeError(f"modulus is not prime: {value}")
    return value


def pair_table(rows: int) -> list[tuple[int, int]]:
    return [(first, second) for first in range(rows)
            for second in range(first + 1, rows)]


def one_column_distribution(
    rows: int, active: int, prime: int, pairs: list[tuple[int, int]]
) -> dict[int, int]:
    pair_count = len(pairs)
    result: dict[int, int] = {}
    assignment = active
    while True:
        mask = 0
        for pair, (first, second) in enumerate(pairs):
            if not (active >> first) & 1 or not (active >> second) & 1:
                continue
            first_colour = (assignment >> first) & 1
            if first_colour == (assignment >> second) & 1:
                mask |= 1 << (first_colour * pair_count + pair)
        coefficient = (result.get(mask, 0) + 1) % prime
        if coefficient:
            result[mask] = coefficient
        else:
            result.pop(mask, None)
        if assignment == 0:
            break
        assignment = (assignment - 1) & active
    return result


def squarefree_product(
    left: dict[int, int], right: dict[int, int], prime: int
) -> dict[int, int]:
    result: dict[int, int] = {}
    for left_mask, left_coefficient in left.items():
        for right_mask, right_coefficient in right.items():
            if left_mask & right_mask:
                continue
            mask = left_mask | right_mask
            coefficient = (
                result.get(mask, 0) + left_coefficient * right_coefficient
            ) % prime
            if coefficient:
                result[mask] = coefficient
            else:
                result.pop(mask, None)
    return result


@dataclass
class RankStats:
    rank: int = 0
    candidates: int = 0
    eliminations: int = 0
    stored_nonzeros: int = 0
    maximum_row: int = 0
    seconds: float = 0.0


class SparseRank:
    def __init__(self, prime: int) -> None:
        self.prime = prime
        self.pivots: dict[int, dict[int, int]] = {}
        self.eliminations = 0
        self.stored_nonzeros = 0
        self.maximum_row = 0

    def add(self, vector: dict[int, int]) -> bool:
        prime = self.prime
        while vector:
            pivot = max(vector)
            coefficient = vector[pivot]
            basis = self.pivots.get(pivot)
            if basis is None:
                if coefficient != 1:
                    inverse = pow(coefficient, prime - 2, prime)
                    vector = {
                        coordinate: value * inverse % prime
                        for coordinate, value in vector.items()
                    }
                self.pivots[pivot] = vector
                self.stored_nonzeros += len(vector)
                self.maximum_row = max(self.maximum_row, len(vector))
                return True
            self.eliminations += 1
            for coordinate, value in basis.items():
                reduced = (vector.get(coordinate, 0) - coefficient * value) % prime
                if reduced:
                    vector[coordinate] = reduced
                else:
                    vector.pop(coordinate, None)
        return False


def product_vectors(
    rows: int, degree: int, prime: int
) -> Iterator[dict[int, int]]:
    types = 1 << rows
    pairs = pair_table(rows)
    columns = [
        one_column_distribution(rows, active, prime, pairs)
        for active in range(types)
    ]
    if degree == 1:
        yield from columns
        return
    if degree == 2:
        for first in range(types):
            for second in range(first, types):
                yield squarefree_product(columns[first], columns[second], prime)
        return
    if degree == 3:
        for first in range(types):
            for second in range(first, types):
                pair = squarefree_product(columns[first], columns[second], prime)
                for third in range(second, types):
                    yield squarefree_product(pair, columns[third], prime)
        return
    raise ValueError("the exact streaming probe currently supports degrees 1..3")


def reachable_rank(
    rows: int, degree: int, prime: int = DEFAULT_PRIME, progress: int = 0
) -> RankStats:
    reducer = SparseRank(prime)
    started = time.monotonic()
    for candidates, vector in enumerate(product_vectors(rows, degree, prime), 1):
        reducer.add(vector)
        if progress and candidates % progress == 0:
            print(
                "RANK_PROGRESS"
                f" candidates={candidates} rank={len(reducer.pivots)}"
                f" eliminations={reducer.eliminations}"
                f" stored_nonzeros={reducer.stored_nonzeros}"
                f" seconds={time.monotonic() - started:.3f}",
                flush=True,
            )
    return RankStats(
        rank=len(reducer.pivots),
        candidates=candidates,
        eliminations=reducer.eliminations,
        stored_nonzeros=reducer.stored_nonzeros,
        maximum_row=reducer.maximum_row,
        seconds=time.monotonic() - started,
    )


def partitions(total: int, maximum: int | None = None) -> Iterator[tuple[int, ...]]:
    if total == 0:
        yield ()
        return
    maximum = total if maximum is None else min(maximum, total)
    for first in range(maximum, 0, -1):
        for rest in partitions(total - first, first):
            yield (first,) + rest


def class_denominator(cycles: tuple[int, ...]) -> int:
    result = 1
    for length in set(cycles):
        multiplicity = cycles.count(length)
        result *= length**multiplicity * factorial(multiplicity)
    return result


def border_strip_height(
    outer: tuple[int, ...], inner: tuple[int, ...]
) -> int | None:
    removed = {
        (row, column)
        for row, length in enumerate(outer)
        for column in range(length)
    } - {
        (row, column)
        for row, length in enumerate(inner)
        for column in range(length)
    }
    if not removed:
        return None
    for row, column in removed:
        if {
            (row, column), (row + 1, column),
            (row, column + 1), (row + 1, column + 1),
        } <= removed:
            return None
    seen = {next(iter(removed))}
    pending = list(seen)
    while pending:
        row, column = pending.pop()
        for neighbour in (
            (row - 1, column), (row + 1, column),
            (row, column - 1), (row, column + 1),
        ):
            if neighbour in removed and neighbour not in seen:
                seen.add(neighbour)
                pending.append(neighbour)
    if seen != removed:
        return None
    return len({row for row, _ in removed}) - 1


@lru_cache(maxsize=None)
def symmetric_group_character(
    shape: tuple[int, ...], cycles: tuple[int, ...]
) -> int:
    if not cycles:
        return int(sum(shape) == 0)
    target = sum(shape) - cycles[0]
    if target < 0:
        return 0
    total = 0
    for inner in partitions(target):
        if len(inner) > len(shape) or any(
            inner[index] > shape[index] for index in range(len(inner))
        ):
            continue
        height = border_strip_height(shape, inner)
        if height is not None:
            total += (-1) ** height * symmetric_group_character(inner, cycles[1:])
    return total


def power_cycle_type(cycles: tuple[int, ...], exponent: int) -> tuple[int, ...]:
    result: list[int] = []
    for length in cycles:
        pieces = gcd(length, exponent)
        result.extend([length // pieces] * pieces)
    return tuple(sorted(result, reverse=True))


def v1_character(cycles: tuple[int, ...]) -> int:
    return 2 ** len(cycles) - cycles.count(1)


def symmetric_power_character(cycles: tuple[int, ...], degree: int) -> int:
    complete = [1]
    for current in range(1, degree + 1):
        numerator = sum(
            v1_character(power_cycle_type(cycles, power))
            * complete[current - power]
            for power in range(1, current + 1)
        )
        if numerator % current:
            raise AssertionError("nonintegral symmetric-power character")
        complete.append(numerator // current)
    return complete[degree]


def irreducible_dimension(shape: tuple[int, ...]) -> int:
    hooks = 1
    for row, length in enumerate(shape):
        for column in range(length):
            below = sum(
                other > row and shape[other] > column
                for other in range(len(shape))
            )
            hooks *= length - column + below
    return factorial(sum(shape)) // hooks


def symmetry_forecast(rows: int, maximum_degree: int) -> list[dict[str, object]]:
    shapes = list(partitions(rows))
    order = factorial(rows)
    classes = [
        (cycles, order // class_denominator(cycles))
        for cycles in shapes
    ]
    results: list[dict[str, object]] = []
    previous: dict[tuple[int, ...], int] | None = None
    for degree in range(1, maximum_degree + 1):
        multiplicities: dict[tuple[int, ...], int] = {}
        for shape in shapes:
            numerator = sum(
                size * symmetric_power_character(cycles, degree)
                * symmetric_group_character(shape, cycles)
                for cycles, size in classes
            )
            if numerator % order:
                raise AssertionError("nonintegral irreducible multiplicity")
            multiplicities[shape] = numerator // order
        dimension = comb((1 << rows) - rows + degree - 1, degree)
        if sum(
            irreducible_dimension(shape) * multiplicity
            for shape, multiplicity in multiplicities.items()
        ) != dimension:
            raise AssertionError("representation dimensions do not sum")
        maximum_shape, maximum_multiplicity = max(
            multiplicities.items(), key=lambda item: item[1]
        )
        results.append({
            "degree": degree,
            "dimension": dimension,
            "sum_multiplicities": sum(multiplicities.values()),
            "maximum_multiplicity": maximum_multiplicity,
            "maximum_shape": maximum_shape,
            "commutant_entries": sum(value * value for value in multiplicities.values()),
            "cross_entries": None if previous is None else sum(
                previous[shape] * multiplicities[shape] for shape in shapes
            ),
            "multiplicities": multiplicities,
        })
        previous = multiplicities
    return results


def run_rank(args: argparse.Namespace) -> None:
    if args.rows < 2 or args.rows > 9:
        raise SystemExit("ROWS must be between 2 and 9")
    if args.degree < 1 or args.degree > 3:
        raise SystemExit("DEGREE must be between 1 and 3")
    if args.degree == 3 and args.rows > 6 and not args.allow_large:
        raise SystemExit("degree three above six rows requires --allow-large")
    stats = reachable_rank(args.rows, args.degree, args.prime, args.progress)
    first_rank = (1 << args.rows) - args.rows
    bound = comb(first_rank + args.degree - 1, args.degree)
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        "REACHABLE_RANK"
        f" rows={args.rows} degree={args.degree} prime={args.prime}"
        f" raw_candidates={stats.candidates} v1_rank={first_rank}"
        f" symmetric_bound={bound} rank={stats.rank}"
        f" deficiency={bound - stats.rank}"
        f" eliminations={stats.eliminations}"
        f" stored_nonzeros={stats.stored_nonzeros}"
        f" maximum_row={stats.maximum_row}"
        f" seconds={stats.seconds:.6f} max_rss_kib={maximum_rss}"
    )


def run_forecast(args: argparse.Namespace) -> None:
    if args.rows < 2 or args.rows > 12:
        raise SystemExit("ROWS must be between 2 and 12")
    if args.maximum_degree < 1 or args.maximum_degree > 8:
        raise SystemExit("MAX_DEGREE must be between 1 and 8")
    for result in symmetry_forecast(args.rows, args.maximum_degree):
        cross = result["cross_entries"]
        print(
            "SYMMETRY_FORECAST"
            f" rows={args.rows} degree={result['degree']}"
            f" dimension={result['dimension']}"
            f" sum_multiplicities={result['sum_multiplicities']}"
            f" maximum_multiplicity={result['maximum_multiplicity']}"
            f" maximum_shape={','.join(map(str, result['maximum_shape']))}"
            f" commutant_entries={result['commutant_entries']}"
            f" cross_entries={cross if cross is not None else 0}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    rank = subparsers.add_parser("rank")
    rank.add_argument("rows", type=int)
    rank.add_argument("degree", type=int)
    rank.add_argument("--prime", type=require_prime, default=DEFAULT_PRIME)
    rank.add_argument("--progress", type=int, default=0)
    rank.add_argument("--allow-large", action="store_true")
    rank.set_defaults(action=run_rank)
    forecast = subparsers.add_parser("forecast")
    forecast.add_argument("rows", type=int)
    forecast.add_argument("maximum_degree", type=int)
    forecast.set_defaults(action=run_forecast)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
