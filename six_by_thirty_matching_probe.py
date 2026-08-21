#!/usr/bin/env python3
"""Exact CPU gate for the extremal T_4(6, 30) perfect-matching reduction.

For q colours and r=q+2 rows, every extremal column has colour-class shape
2,2,1,...,1.  Its two collision tokens form an edge of

    H(q, r) = K_q x KG(r, 2).

At the maximum width q*C(r,2)/2, the rectangle-free colouring count is the
number of perfect matchings of H, multiplied by the column order and by the
(q-2)! assignments of singleton colours in every column.

The target is q=4, r=6.  This probe supplies a symmetry-canonical recursive
CPU counter, exact small-case validation, and a bounded target census.  It is
intentionally a falsifiable CPU gate rather than a production GPU hafnian.
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence


class ResourceLimit(RuntimeError):
    pass


@dataclass
class MatchingStats:
    states: int = 0
    branches: int = 0
    canonical_requests: int = 0
    canonical_row_images: int = 0
    started: float = 0.0


class PairTokenGeometry:
    def __init__(self, colours: int, rows: int):
        if colours < 2:
            raise ValueError("at least two colours are required")
        if rows != colours + 2:
            raise ValueError("the two-token extremal reduction requires rows=colours+2")
        self.colours = colours
        self.rows = rows
        self.pairs = tuple(itertools.combinations(range(rows), 2))
        self.pair_index = {pair: index for index, pair in enumerate(self.pairs)}
        self.pair_count = len(self.pairs)
        self.full_pair_mask = (1 << self.pair_count) - 1

        self.disjoint_masks = []
        for left in self.pairs:
            mask = 0
            for index, right in enumerate(self.pairs):
                if not set(left).intersection(right):
                    mask |= 1 << index
            self.disjoint_masks.append(mask)

        # A colour permutation merely permutes lanes, so sorting the lane masks
        # quotients S_q exactly.  These compact byte tables quotient S_r.
        self.row_tables = []
        chunks = (self.pair_count + 7) // 8
        for permutation in itertools.permutations(range(rows)):
            image = []
            for i, j in self.pairs:
                mapped = tuple(sorted((permutation[i], permutation[j])))
                image.append(self.pair_index[mapped])
            chunk_tables = []
            for chunk in range(chunks):
                table = [0] * 256
                for byte in range(256):
                    transformed = 0
                    for bit in range(8):
                        source = 8 * chunk + bit
                        if source < self.pair_count and byte & (1 << bit):
                            transformed |= 1 << image[source]
                    table[byte] = transformed
                chunk_tables.append(tuple(table))
            self.row_tables.append(tuple(chunk_tables))
        self.row_tables = tuple(self.row_tables)

    @property
    def vertices(self) -> int:
        return self.colours * self.pair_count

    @property
    def degree(self) -> int:
        return (self.colours - 1) * math.comb(self.rows - 2, 2)

    @property
    def edges(self) -> int:
        return self.vertices * self.degree // 2

    @property
    def maximum_columns(self) -> int:
        return self.vertices // 2

    @property
    def column_weight(self) -> int:
        return math.factorial(self.colours - 2)

    def initial_state(self) -> tuple[int, ...]:
        return (self.full_pair_mask,) * self.colours

    def token_mask(self, column: Sequence[int]) -> int:
        """Return the full q*C(r,2)-bit collision-token mask."""
        mask = 0
        for pair_index, (i, j) in enumerate(self.pairs):
            if column[i] == column[j]:
                mask |= 1 << (column[i] * self.pair_count + pair_index)
        return mask

    def exact_column_weights(self) -> dict[int, int]:
        weights: dict[int, int] = defaultdict(int)
        for column in itertools.product(range(self.colours), repeat=self.rows):
            weights[self.token_mask(column)] += 1
        return dict(weights)


class SymmetryMatchingCounter:
    def __init__(
        self,
        geometry: PairTokenGeometry,
        max_states: int = 0,
        max_seconds: float = 0.0,
        progress_every: int = 0,
    ):
        self.geometry = geometry
        self.max_states = max_states
        self.max_seconds = max_seconds
        self.progress_every = progress_every
        self.stats = MatchingStats(started=time.monotonic())

        @lru_cache(maxsize=None)
        def canonical_cached(state: tuple[int, ...]) -> tuple[int, ...]:
            self.stats.canonical_requests += 1
            best: tuple[int, ...] | None = None
            for chunk_tables in self.geometry.row_tables:
                lanes = []
                for lane in state:
                    transformed = 0
                    value = lane
                    for table in chunk_tables:
                        transformed |= table[value & 0xFF]
                        value >>= 8
                    lanes.append(transformed)
                candidate = tuple(sorted(lanes))
                if best is None or candidate < best:
                    best = candidate
            self.stats.canonical_row_images += len(self.geometry.row_tables)
            assert best is not None
            return best

        self._canonical_cached = canonical_cached

        @lru_cache(maxsize=None)
        def solve_cached(state: tuple[int, ...]) -> int:
            self.stats.states += 1
            if self.max_states and self.stats.states > self.max_states:
                raise ResourceLimit(f"state cap {self.max_states:,} exceeded")
            if self.stats.states & 1023 == 0:
                elapsed = time.monotonic() - self.stats.started
                if self.max_seconds and elapsed > self.max_seconds:
                    raise ResourceLimit(f"time cap {self.max_seconds:g}s exceeded")
            if self.progress_every and self.stats.states % self.progress_every == 0:
                elapsed = time.monotonic() - self.stats.started
                print(
                    f"MATCHING_PROGRESS states={self.stats.states} "
                    f"branches={self.stats.branches} elapsed={elapsed:.3f}s",
                    flush=True,
                )

            if not any(state):
                return 1

            # Minimum residual degree is an exact branching heuristic.
            pivot_colour = pivot_pair = -1
            pivot_neighbours: list[tuple[int, int]] = []
            minimum_degree = self.geometry.vertices + 1
            for colour, lane in enumerate(state):
                bits = lane
                while bits:
                    bit = bits & -bits
                    pair = bit.bit_length() - 1
                    neighbours = []
                    compatible = self.geometry.disjoint_masks[pair]
                    for other_colour, other_lane in enumerate(state):
                        if other_colour == colour:
                            continue
                        candidates = other_lane & compatible
                        while candidates:
                            other_bit = candidates & -candidates
                            neighbours.append((other_colour, other_bit.bit_length() - 1))
                            candidates ^= other_bit
                    if not neighbours:
                        return 0
                    if len(neighbours) < minimum_degree:
                        minimum_degree = len(neighbours)
                        pivot_colour, pivot_pair = colour, pair
                        pivot_neighbours = neighbours
                        if minimum_degree == 1:
                            break
                    bits ^= bit
                if minimum_degree == 1:
                    break

            children: Counter[tuple[int, ...]] = Counter()
            pivot_bit = 1 << pivot_pair
            for other_colour, other_pair in pivot_neighbours:
                child = list(state)
                child[pivot_colour] &= ~pivot_bit
                child[other_colour] &= ~(1 << other_pair)
                children[self.canonical(tuple(child))] += 1

            self.stats.branches += len(pivot_neighbours)
            return sum(multiplicity * solve_cached(child)
                       for child, multiplicity in children.items())

        self._solve_cached = solve_cached

    def canonical(self, state: Iterable[int]) -> tuple[int, ...]:
        return self._canonical_cached(tuple(sorted(state)))

    def solve(self) -> int:
        initial = self.canonical(self.geometry.initial_state())
        return self._solve_cached(initial)

    def solve_state(self, state: Iterable[int]) -> int:
        return self._solve_cached(self.canonical(state))

    def report(self, status: str, value: int | None = None) -> str:
        elapsed = time.monotonic() - self.stats.started
        canonical_info = self._canonical_cached.cache_info()
        solve_info = self._solve_cached.cache_info()
        fields = [
            f"status={status}",
            f"colours={self.geometry.colours}",
            f"rows={self.geometry.rows}",
            f"vertices={self.geometry.vertices}",
            f"degree={self.geometry.degree}",
            f"edges={self.geometry.edges}",
            f"states={self.stats.states}",
            f"branches={self.stats.branches}",
            f"canonical_states={canonical_info.currsize}",
            f"canonical_hits={canonical_info.hits}",
            f"solve_hits={solve_info.hits}",
            f"row_images={self.stats.canonical_row_images}",
            f"elapsed={elapsed:.6f}",
        ]
        if value is not None:
            fields.extend((f"perfect_matchings={value}", f"bits={value.bit_length()}"))
        return "MATCHING " + " ".join(fields)


def ordered_column_dp(geometry: PairTokenGeometry, columns: int) -> int:
    """Small exact oracle over full token unions and ordered physical columns."""
    weights = geometry.exact_column_weights()
    states = {0: 1}
    for _ in range(columns):
        following: dict[int, int] = defaultdict(int)
        for used, count in states.items():
            for token_set, weight in weights.items():
                if not used & token_set:
                    following[used | token_set] += count * weight
        states = following
    return sum(states.values())


def small_regression() -> None:
    geometry = PairTokenGeometry(2, 4)
    counter = SymmetryMatchingCounter(geometry)
    perfect_matchings = counter.solve()
    expected = (
        math.factorial(geometry.maximum_columns)
        * geometry.column_weight ** geometry.maximum_columns
        * perfect_matchings
    )
    direct = ordered_column_dp(geometry, geometry.maximum_columns)
    if perfect_matchings != 1 or direct != expected:
        raise AssertionError(
            f"small regression failed: PM={perfect_matchings} "
            f"direct={direct} expected={expected}"
        )
    print(
        "MATCHING_REGRESSION colours=2 rows=4 columns=6 "
        f"perfect_matchings={perfect_matchings} T={direct} exact=OK"
    )


def colour_sector_count() -> int:
    """Number of labelled (a,b,c) sectors for four colours and 15 vertices/lane."""
    return sum(1 for a in range(16) for b in range(16 - a)
               for c in (15 - a - b,) if c >= 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--colours", type=int, default=4)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--max-states", type=int, default=100_000)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--skip-regression", action="store_true")
    args = parser.parse_args()

    if not args.skip_regression:
        small_regression()

    geometry = PairTokenGeometry(args.colours, args.rows)
    print(
        "MATCHING_GRAPH "
        f"colours={geometry.colours} rows={geometry.rows} "
        f"pair_vertices={geometry.pair_count} vertices={geometry.vertices} "
        f"degree={geometry.degree} edges={geometry.edges} "
        f"maximum_columns={geometry.maximum_columns} "
        f"column_weight={geometry.column_weight} "
        f"symmetry_order={math.factorial(args.colours)*math.factorial(args.rows)}"
    )
    if args.colours == 4 and args.rows == 6:
        print(f"MATCHING_COLOUR_SECTORS count={colour_sector_count()} exact=OK")

    counter = SymmetryMatchingCounter(
        geometry,
        max_states=args.max_states,
        max_seconds=args.max_seconds,
        progress_every=args.progress_every,
    )
    try:
        perfect_matchings = counter.solve()
    except ResourceLimit as error:
        print(counter.report("LIMIT"))
        print(f"MATCHING_LIMIT reason={error}")
        return 3

    print(counter.report("COMPLETE", perfect_matchings))
    columns = geometry.maximum_columns
    result = (
        math.factorial(columns)
        * geometry.column_weight ** columns
        * perfect_matchings
    )
    print(
        f"MATCHING_RESULT T_{args.colours}({args.rows},{columns})={result} "
        f"result_bits={result.bit_length()} exact=OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
