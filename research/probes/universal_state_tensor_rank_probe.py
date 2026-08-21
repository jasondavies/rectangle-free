#!/usr/bin/env python3
"""Certified tensor-train rank lower bounds for the universal three-column state.

The universal state ``F3=P^3`` is first built as the checked weighted MDD from
``universal_state_dd_probe``.  At every site cut, reachable MDD nodes represent
the residual coefficient functions.  Prefix assignments reaching different
nodes have disjoint support, so the tensor matricization rank equals the rank
of those residual functions.

This probe evaluates a deterministic subfamily of residuals on deterministic
accepted suffixes and performs exact modular elimination.  The resulting rank
is a certified lower bound: it is the rank of a literal submatrix of the full
integer coefficient tensor reduced modulo a prime.  When all residual nodes
and all suffix assignments fit under the caps, the reported rank is exact.
"""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.probes import universal_state_dd_probe as dd


def reachable_by_level(diagram: dd.WeightedMDD, reference: dd.Ref) -> list[list[int]]:
    result = [[] for _ in range(diagram.site_count)]
    pending = [reference[1]] if reference[0] else []
    seen: set[int] = set()
    while pending:
        node_id = pending.pop()
        if node_id == dd.TERMINAL or node_id in seen:
            continue
        seen.add(node_id)
        node = diagram.nodes[node_id]
        result[node.level].append(node_id)
        pending.extend(
            child for coefficient, child in node.children
            if coefficient and child != dd.TERMINAL
        )
    for nodes in result:
        nodes.sort()
    return result


def evenly_spaced(values: list[int], count: int) -> list[int]:
    if len(values) <= count:
        return values
    return [values[(index * len(values) + len(values) // 2) // count]
            for index in range(count)]


def mix64(value: int) -> int:
    value &= (1 << 64) - 1
    value ^= value >> 30
    value = value * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
    value ^= value >> 27
    value = value * 0x94D049BB133111EB & ((1 << 64) - 1)
    return value ^ (value >> 31)


def accepted_suffix(
    diagram: dd.WeightedMDD,
    node_id: int,
    salt: int,
) -> tuple[int, ...]:
    symbols: list[int] = []
    while node_id != dd.TERMINAL:
        node = diagram.nodes[node_id]
        choices = [
            (symbol, child)
            for symbol, (coefficient, child) in enumerate(node.children)
            if coefficient
        ]
        if not choices:
            raise AssertionError("reachable MDD node has no accepting child")
        choice = mix64(salt ^ (node_id * 0x9E3779B97F4A7C15) ^ node.level)
        symbol, child = choices[choice % len(choices)]
        symbols.append(symbol)
        node_id = child
        salt = mix64(salt + 0xD1B54A32D192ED03)
    return tuple(symbols)


def evaluate_suffix(
    diagram: dd.WeightedMDD,
    node_id: int,
    symbols: tuple[int, ...],
) -> int:
    coefficient = 1
    for symbol in symbols:
        if node_id == dd.TERMINAL:
            return 0
        edge, node_id = diagram.nodes[node_id].children[symbol]
        coefficient = coefficient * edge % diagram.prime
        if not coefficient:
            return 0
    return coefficient if node_id == dd.TERMINAL else 0


def modular_rank(matrix: np.ndarray, prime: int) -> int:
    matrix = np.asarray(matrix, dtype=np.int64).copy()
    rows, columns = matrix.shape
    rank = 0
    for column in range(columns):
        candidates = np.flatnonzero(matrix[rank:, column])
        if not len(candidates):
            continue
        pivot = rank + int(candidates[0])
        if pivot != rank:
            matrix[[rank, pivot]] = matrix[[pivot, rank]]
        inverse = pow(int(matrix[rank, column]), prime - 2, prime)
        matrix[rank, column:] = matrix[rank, column:] * inverse % prime
        targets = np.flatnonzero(matrix[rank + 1:, column]) + rank + 1
        if len(targets):
            factors = matrix[targets, column].copy()
            matrix[targets, column:] = (
                matrix[targets, column:]
                - factors[:, None] * matrix[rank, column:]
            ) % prime
        rank += 1
        if rank == rows:
            break
    return rank


def suffix_columns(
    diagram: dd.WeightedMDD,
    nodes: list[int],
    level: int,
    column_cap: int,
    exact_domain_cap: int,
) -> tuple[list[tuple[int, ...]], bool]:
    remaining = diagram.site_count - level
    domain = diagram.alphabet ** remaining
    if domain <= exact_domain_cap:
        return list(product(range(diagram.alphabet), repeat=remaining)), True
    wanted = min(column_cap, domain)
    result: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    attempt = 0
    maximum_attempts = max(1024, wanted * 64)
    while len(result) < wanted and attempt < maximum_attempts:
        node_id = nodes[attempt % len(nodes)]
        pattern = accepted_suffix(
            diagram, node_id,
            mix64((attempt + 1) * 0xA24BAED4963EE407),
        )
        if pattern not in seen:
            seen.add(pattern)
            result.append(pattern)
        attempt += 1
    return result, False


def cut_rank_record(
    diagram: dd.WeightedMDD,
    all_nodes: list[int],
    level: int,
    rank_cap: int,
    exact_domain_cap: int,
) -> dict[str, object]:
    started = time.monotonic()
    nodes = evenly_spaced(all_nodes, rank_cap)
    columns, complete_domain = suffix_columns(
        diagram, nodes, level, rank_cap, exact_domain_cap
    )
    matrix = np.empty((len(nodes), len(columns)), dtype=np.int64)
    for row, node_id in enumerate(nodes):
        matrix[row] = [evaluate_suffix(diagram, node_id, value)
                       for value in columns]
    rank = modular_rank(matrix, diagram.prime)
    exact = len(nodes) == len(all_nodes) and complete_domain
    return {
        "level": level,
        "left_sites": level,
        "right_sites": diagram.site_count - level,
        "reachable_residuals": len(all_nodes),
        "sampled_residuals": len(nodes),
        "sampled_suffixes": len(columns),
        "rank": rank,
        "status": "exact" if exact else "certified_lower_bound",
        "seconds": round(time.monotonic() - started, 6),
    }


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    try:
        diagram, one_column, support = dd.build_one_column(
            args.rows, args.order, args.mode, args.prime,
            args.max_nodes, args.max_operations,
        )
        square = diagram.convolve(0, one_column, one_column)
        cube = diagram.convolve(0, square, one_column)
        profile = diagram.profile(cube)
        print(json.dumps({
            "kind": "tensor_rank_header",
            "rows": args.rows,
            "columns": 3,
            "order": args.order,
            "mode": args.mode,
            "modulus": args.prime,
            "one_column_support": support,
            "coefficient_sum": diagram.total(cube),
            "reachable_nodes": profile["reachable_nodes"],
            "maximum_width": profile["maximum_width"],
            "allocated_nodes": len(diagram.nodes),
            "operations": diagram.operations,
            "build_seconds": round(time.monotonic() - started, 6),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }), flush=True)
        levels = reachable_by_level(diagram, cube)
        selected_levels = (
            set(range(diagram.site_count))
            if args.levels is None else set(args.levels)
        )
        for level, nodes in enumerate(levels):
            if level not in selected_levels:
                continue
            if not nodes:
                continue
            record = cut_rank_record(
                diagram, nodes, level, args.rank_cap,
                args.exact_domain_cap,
            )
            record.update({
                "kind": "tensor_cut_rank",
                "rows": args.rows,
                "order": args.order,
                "mode": args.mode,
                "modulus": args.prime,
            })
            print(json.dumps(record), flush=True)
        return 0
    except dd.ResourceLimit as error:
        print(json.dumps({
            "kind": "tensor_rank_resource_limit",
            "rows": args.rows,
            "order": args.order,
            "mode": args.mode,
            "modulus": args.prime,
            "reason": str(error),
            "seconds": round(time.monotonic() - started, 6),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }), flush=True)
        return 3


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("rows", type=int)
    result.add_argument(
        "--order", choices=("lex", "reverse", "balanced"), default="balanced"
    )
    result.add_argument(
        "--mode", choices=("bundled", "colour-major"), default="bundled"
    )
    result.add_argument("--prime", type=dd.require_prime, default=dd.DEFAULT_PRIME)
    result.add_argument("--rank-cap", type=int, default=512)
    result.add_argument("--exact-domain-cap", type=int, default=4096)
    result.add_argument(
        "--levels", type=lambda value: tuple(int(item) for item in value.split(",")),
        help="comma-separated cut levels to evaluate (default: every cut)",
    )
    result.add_argument("--max-nodes", type=int, default=1_000_000)
    result.add_argument("--max-operations", type=int, default=10_000_000)
    return result


def main() -> int:
    args = parser().parse_args()
    if not 2 <= args.rows <= 9:
        raise SystemExit("rows must be in 2..9")
    if args.rank_cap < 1 or args.exact_domain_cap < 1:
        raise SystemExit("rank and domain caps must be positive")
    if args.prime > 3_000_000_000:
        raise SystemExit("prime is too large for overflow-safe int64 elimination")
    if args.levels is not None and any(level < 0 for level in args.levels):
        raise SystemExit("cut levels must be nonnegative")
    site_count = args.rows * (args.rows - 1) // 2
    if args.mode == "colour-major":
        site_count *= 4
    if args.levels is not None and any(level >= site_count for level in args.levels):
        raise SystemExit(f"cut levels must be smaller than {site_count}")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
