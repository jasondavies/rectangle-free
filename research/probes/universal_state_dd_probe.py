#!/usr/bin/env python3
"""Exact weighted decision-diagram gate for the universal four-colour state.

For one grid column let

    P = sum_x e_{I(x)},  x in [4]^rows,

in the squarefree algebra on colour/row-pair tokens.  Then ``P ** columns``
is the universal state for that many columns and the sum of its coefficients
is exactly T_4(rows, columns).

This probe represents the *single* state P^k as a reduced edge-weighted
multi-valued decision diagram.  It therefore tests nonlinear sharing without
constructing a basis for the ambient reachable vector space.  Arithmetic is
exact modulo a prime.  Hard node and operation caps make representation
explosion a controlled negative result rather than an out-of-memory failure.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from math import isqrt
import resource
import time
from typing import Iterable, Iterator


DEFAULT_PRIME = 1_000_003
TERMINAL = -1
ZERO = (0, TERMINAL)
ONE = (1, TERMINAL)
Ref = tuple[int, int]


class ResourceLimit(RuntimeError):
    pass


def require_prime(value: str | int) -> int:
    value = int(value)
    if value < 2 or any(value % divisor == 0
                        for divisor in range(2, isqrt(value) + 1)):
        raise argparse.ArgumentTypeError(f"modulus is not prime: {value}")
    return value


def pair_table(rows: int) -> list[tuple[int, int]]:
    return [(first, second) for first in range(rows)
            for second in range(first + 1, rows)]


def balanced_pairs(vertices: tuple[int, ...]) -> list[tuple[int, int]]:
    if len(vertices) < 2:
        return []
    middle = len(vertices) // 2
    left = vertices[:middle]
    right = vertices[middle:]
    return (
        balanced_pairs(left)
        + balanced_pairs(right)
        + [(first, second) for first in left for second in right]
    )


def ordered_pairs(rows: int, order: str) -> list[tuple[int, int]]:
    pairs = pair_table(rows)
    if order == "lex":
        return pairs
    if order == "reverse":
        return list(reversed(pairs))
    if order == "balanced":
        result = balanced_pairs(tuple(range(rows)))
        assert sorted(result) == pairs
        return result
    raise ValueError(f"unknown pair order: {order}")


def colourings(rows: int) -> Iterator[tuple[int, ...]]:
    for encoded in range(4 ** rows):
        value = encoded
        colouring = []
        for _ in range(rows):
            colouring.append(value & 3)
            value >>= 2
        yield tuple(colouring)


def column_pattern(
    colouring: tuple[int, ...],
    pairs: list[tuple[int, int]],
    mode: str,
) -> bytes:
    bundled = []
    for first, second in pairs:
        if colouring[first] == colouring[second]:
            bundled.append(1 << colouring[first])
        else:
            bundled.append(0)
    if mode == "bundled":
        return bytes(bundled)
    if mode == "colour-major":
        return bytes(
            int(bool(mask & (1 << colour)))
            for colour in range(4)
            for mask in bundled
        )
    raise ValueError(f"unknown site mode: {mode}")


@dataclass(frozen=True)
class Node:
    level: int
    children: tuple[Ref, ...]


class WeightedMDD:
    """Reduced modular MDD with a scalar weight on every reference."""

    def __init__(
        self,
        site_count: int,
        alphabet: int,
        prime: int,
        max_nodes: int,
        max_operations: int,
    ) -> None:
        self.site_count = site_count
        self.alphabet = alphabet
        self.prime = prime
        self.max_nodes = max_nodes
        self.max_operations = max_operations
        self.nodes: list[Node] = []
        self.unique: list[dict[tuple[Ref, ...], int]] = [
            {} for _ in range(site_count)
        ]
        self.add_cache: dict[tuple[int, Ref, Ref], Ref] = {}
        self.convolution_cache: dict[tuple[int, int, int], Ref] = {}
        self.total_cache: dict[int, int] = {}
        self.operations = 0
        self.transitions = [
            (left, right, left | right)
            for left in range(alphabet)
            for right in range(alphabet)
            if not left & right
        ]

    def _operation(self) -> None:
        self.operations += 1
        if self.max_operations and self.operations > self.max_operations:
            raise ResourceLimit(
                f"operation cap exceeded ({self.max_operations:,})"
            )

    def make_node(self, level: int, children: Iterable[Ref]) -> Ref:
        children = tuple(children)
        assert len(children) == self.alphabet
        pivot = next((coefficient for coefficient, _ in children if coefficient), 0)
        if not pivot:
            return ZERO
        inverse = pow(pivot, self.prime - 2, self.prime)
        normalized = tuple(
            ((coefficient * inverse) % self.prime, node)
            if coefficient else ZERO
            for coefficient, node in children
        )
        existing = self.unique[level].get(normalized)
        if existing is None:
            if self.max_nodes and len(self.nodes) >= self.max_nodes:
                raise ResourceLimit(f"node cap exceeded ({self.max_nodes:,})")
            existing = len(self.nodes)
            self.nodes.append(Node(level, normalized))
            self.unique[level][normalized] = existing
        return pivot, existing

    def scale(self, reference: Ref, scalar: int) -> Ref:
        if not reference[0] or not scalar:
            return ZERO
        return reference[0] * scalar % self.prime, reference[1]

    def add(self, level: int, left: Ref, right: Ref) -> Ref:
        if not left[0]:
            return right
        if not right[0]:
            return left
        if left[1] == right[1]:
            coefficient = (left[0] + right[0]) % self.prime
            return (coefficient, left[1]) if coefficient else ZERO
        if level == self.site_count:
            coefficient = (left[0] + right[0]) % self.prime
            return (coefficient, TERMINAL) if coefficient else ZERO
        if right < left:
            left, right = right, left
        key = (level, left, right)
        cached = self.add_cache.get(key)
        if cached is not None:
            return cached
        self._operation()
        left_node = self.nodes[left[1]]
        right_node = self.nodes[right[1]]
        assert left_node.level == right_node.level == level
        result = self.make_node(
            level,
            (
                self.add(
                    level + 1,
                    self.scale(left_child, left[0]),
                    self.scale(right_child, right[0]),
                )
                for left_child, right_child
                in zip(left_node.children, right_node.children)
            ),
        )
        self.add_cache[key] = result
        return result

    def convolve(self, level: int, left: Ref, right: Ref) -> Ref:
        if not left[0] or not right[0]:
            return ZERO
        scalar = left[0] * right[0] % self.prime
        if level == self.site_count:
            return scalar, TERMINAL
        first, second = sorted((left[1], right[1]))
        key = (level, first, second)
        base = self.convolution_cache.get(key)
        if base is None:
            self._operation()
            left_node = self.nodes[first]
            right_node = self.nodes[second]
            assert left_node.level == right_node.level == level
            output = [ZERO] * self.alphabet
            for left_symbol, right_symbol, union in self.transitions:
                contribution = self.convolve(
                    level + 1,
                    left_node.children[left_symbol],
                    right_node.children[right_symbol],
                )
                output[union] = self.add(
                    level + 1, output[union], contribution
                )
            base = self.make_node(level, output)
            self.convolution_cache[key] = base
        return self.scale(base, scalar)

    def total(self, reference: Ref) -> int:
        if not reference[0]:
            return 0
        return reference[0] * self._unit_total(reference[1]) % self.prime

    def _unit_total(self, node_id: int) -> int:
        if node_id == TERMINAL:
            return 1
        cached = self.total_cache.get(node_id)
        if cached is not None:
            return cached
        node = self.nodes[node_id]
        result = sum(
            coefficient * self._unit_total(child)
            for coefficient, child in node.children
        ) % self.prime
        self.total_cache[node_id] = result
        return result

    def profile(self, reference: Ref) -> dict[str, object]:
        if not reference[0]:
            return {"reachable_nodes": 0, "widths": [0] * self.site_count}
        pending = [reference[1]]
        seen: set[int] = set()
        widths = [0] * self.site_count
        while pending:
            node_id = pending.pop()
            if node_id == TERMINAL or node_id in seen:
                continue
            seen.add(node_id)
            node = self.nodes[node_id]
            widths[node.level] += 1
            pending.extend(
                child for coefficient, child in node.children
                if coefficient and child != TERMINAL
            )
        return {
            "reachable_nodes": len(seen),
            "maximum_width": max(widths, default=0),
            "widths": widths,
        }


def build_one_column(
    rows: int,
    order: str,
    mode: str,
    prime: int,
    max_nodes: int,
    max_operations: int,
) -> tuple[WeightedMDD, Ref, int]:
    pairs = ordered_pairs(rows, order)
    counts: dict[bytes, int] = {}
    for colouring in colourings(rows):
        pattern = column_pattern(colouring, pairs, mode)
        counts[pattern] = (counts.get(pattern, 0) + 1) % prime
    site_count = len(pairs) if mode == "bundled" else 4 * len(pairs)
    alphabet = 16 if mode == "bundled" else 2
    diagram = WeightedMDD(
        site_count, alphabet, prime, max_nodes, max_operations
    )

    current: dict[bytes, Ref] = {
        pattern: (coefficient, TERMINAL)
        for pattern, coefficient in counts.items()
        if coefficient
    }
    for level in range(site_count - 1, -1, -1):
        grouped: dict[bytes, list[Ref]] = {}
        for pattern, reference in current.items():
            prefix = pattern[:-1]
            children = grouped.get(prefix)
            if children is None:
                children = [ZERO] * alphabet
                grouped[prefix] = children
            children[pattern[-1]] = reference
        current = {
            prefix: diagram.make_node(level, children)
            for prefix, children in grouped.items()
        }
    assert len(current) == 1 and b"" in current
    return diagram, current[b""], len(counts)


def result_record(
    rows: int,
    columns: int,
    order: str,
    mode: str,
    diagram: WeightedMDD,
    reference: Ref,
    seconds: float,
    support: int | None = None,
) -> dict[str, object]:
    profile = diagram.profile(reference)
    result: dict[str, object] = {
        "rows": rows,
        "columns": columns,
        "order": order,
        "mode": mode,
        "modulus": diagram.prime,
        "coefficient_sum": diagram.total(reference),
        "allocated_nodes": len(diagram.nodes),
        "operations": diagram.operations,
        "seconds": round(seconds, 6),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        **profile,
    }
    if support is not None:
        result["one_column_support"] = support
    return result


def run_probe(args: argparse.Namespace) -> int:
    started = time.monotonic()
    try:
        diagram, one_column, support = build_one_column(
            args.rows,
            args.order,
            args.mode,
            args.prime,
            args.max_nodes,
            args.max_operations,
        )
        print(json.dumps(result_record(
            args.rows, 1, args.order, args.mode, diagram, one_column,
            time.monotonic() - started, support,
        )), flush=True)

        powers: dict[int, Ref] = {1: one_column}
        if args.strategy == "sequential":
            schedule = [(columns, columns - 1, 1)
                        for columns in range(2, args.columns + 1)]
        else:
            if args.columns != 9:
                raise ValueError("the three-block strategy requires columns=9")
            schedule = [(2, 1, 1), (3, 2, 1), (6, 3, 3), (9, 6, 3)]
        for columns, left_power, right_power in schedule:
            power_started = time.monotonic()
            powers[columns] = diagram.convolve(
                0, powers[left_power], powers[right_power]
            )
            print(json.dumps(result_record(
                args.rows, columns, args.order, args.mode, diagram,
                powers[columns], time.monotonic() - power_started,
            )), flush=True)
        return 0
    except ResourceLimit as error:
        print(json.dumps({
            "rows": args.rows,
            "order": args.order,
            "mode": args.mode,
            "status": "resource_limit",
            "reason": str(error),
            "allocated_nodes": len(diagram.nodes) if "diagram" in locals() else 0,
            "operations": diagram.operations if "diagram" in locals() else 0,
            "seconds": round(time.monotonic() - started, 6),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }), flush=True)
        return 3


def brute_t4(rows: int, columns: int) -> int:
    column_states = list(colourings(rows))
    total = 0

    def visit(depth: int, used: tuple[int, ...]) -> None:
        nonlocal total
        if depth == columns:
            total += 1
            return
        for colouring in column_states:
            next_used = list(used)
            valid = True
            for pair, (first, second) in enumerate(pair_table(rows)):
                if colouring[first] != colouring[second]:
                    continue
                token = 1 << colouring[first]
                if next_used[pair] & token:
                    valid = False
                    break
                next_used[pair] |= token
            if valid:
                visit(depth + 1, tuple(next_used))

    visit(0, (0,) * len(pair_table(rows)))
    return total


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("rows", type=int)
    result.add_argument("columns", type=int, help="largest sequential power P^k")
    result.add_argument(
        "--order", choices=("lex", "reverse", "balanced"), default="lex"
    )
    result.add_argument(
        "--mode", choices=("bundled", "colour-major"), default="bundled"
    )
    result.add_argument("--prime", type=require_prime, default=DEFAULT_PRIME)
    result.add_argument("--max-nodes", type=int, default=1_000_000)
    result.add_argument("--max-operations", type=int, default=10_000_000)
    result.add_argument(
        "--strategy", choices=("sequential", "three-block"),
        default="sequential",
        help="three-block computes P^3, P^6, P^9 by a balanced 3+3+3 path",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    if not 2 <= args.rows <= 9:
        raise SystemExit("rows must be in 2..9")
    if args.columns < 1:
        raise SystemExit("columns must be positive")
    return run_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
