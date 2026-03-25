#!/usr/bin/env python3

import argparse
from fractions import Fraction
import os
from pathlib import Path
import re
import subprocess
import sys
import time


DEFAULT_PRIMES = [
    18446744073709551557,
    18446744073709551533,
    18446744073709551521,
    18446744073709551437,
    18446744073709551359,
    18446744073709551293,
    18446744073709551253,
]

EVAL_RE = re.compile(r"^P\((\d+)\) mod (\d+) = (\d+)$")


def choose_default_solver(rows: int) -> str:
    if rows == 7 and Path("./partition_poly_7").exists():
        return "./partition_poly_7"
    return "./partition_poly"


def parse_args() -> argparse.Namespace:
    argv = sys.argv[1:]
    if "--" in argv:
        split = argv.index("--")
        solver_args = argv[split + 1 :]
        argv = argv[:split]
    else:
        solver_args = []
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct a full chromatic polynomial by modular evaluations and exact interpolation."
        )
    )
    parser.add_argument("rows", type=int)
    parser.add_argument("cols", type=int)
    parser.add_argument("--solver", default=None, help="Solver binary to run")
    parser.add_argument("--threads", type=int, default=1, help="OMP_NUM_THREADS for each evaluation")
    parser.add_argument("--prefix-depth", type=int, default=2)
    parser.add_argument(
        "--degree",
        type=int,
        default=None,
        help="Polynomial degree; defaults to rows*cols",
    )
    parser.add_argument(
        "--prime",
        dest="primes",
        action="append",
        type=int,
        default=[],
        help="Prime modulus to use; repeatable",
    )
    parser.add_argument(
        "--poly-out",
        default=None,
        help="Write the reconstructed polynomial to a text file",
    )
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="Print the reconstructed values P(q) for q=0..degree",
    )
    args = parser.parse_args(argv)
    args.solver_args = solver_args
    return args


def run_eval(
    solver: str,
    rows: int,
    cols: int,
    prefix_depth: int,
    q: int,
    prime: int,
    threads: int,
    solver_args: list[str],
) -> int:
    cmd = [
        solver,
        str(rows),
        str(cols),
        "--prefix-depth",
        str(prefix_depth),
        "--eval-q",
        str(q),
        "--mod",
        str(prime),
        *solver_args,
    ]
    env = {"OMP_NUM_THREADS": str(threads)}
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, **env},
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        raise RuntimeError(f"solver failed for q={q}, mod={prime}")
    for line in reversed(proc.stdout.splitlines()):
        match = EVAL_RE.match(line.strip())
        if match:
            line_q = int(match.group(1))
            line_mod = int(match.group(2))
            value = int(match.group(3))
            if line_q != q or line_mod != prime:
                raise RuntimeError(
                    f"unexpected evaluation line for q={q}, mod={prime}: {line.strip()}"
                )
            return value
    raise RuntimeError(f"failed to parse evaluation for q={q}, mod={prime}")


def choose_primes(primes: list[int], degree: int) -> tuple[list[int], int, bool]:
    selected = list(primes) if primes else list(DEFAULT_PRIMES)
    bound = max(1, degree**degree)
    modulus = 1
    used: list[int] = []
    for prime in selected:
        if prime <= degree:
            raise ValueError(f"prime {prime} must be greater than degree {degree}")
        used.append(prime)
        modulus *= prime
        if modulus > bound:
            return used, bound, True
    return used, bound, modulus > bound


def crt_combine(current: int, modulus: int, residue: int, prime: int) -> int:
    delta = (residue - (current % prime)) % prime
    inv = pow(modulus % prime, -1, prime)
    step = (delta * inv) % prime
    return current + modulus * step


def reconstruct_values(
    solver: str,
    rows: int,
    cols: int,
    prefix_depth: int,
    degree: int,
    primes: list[int],
    threads: int,
    solver_args: list[str],
) -> tuple[list[int], int]:
    values = [0] * (degree + 1)
    modulus = 1
    for prime_index, prime in enumerate(primes, start=1):
        t0 = time.time()
        prime_values = []
        for q in range(degree + 1):
            value = run_eval(solver, rows, cols, prefix_depth, q, prime, threads, solver_args)
            prime_values.append(value)
        if modulus == 1:
            values = prime_values
            modulus = prime
        else:
            for idx, residue in enumerate(prime_values):
                values[idx] = crt_combine(values[idx], modulus, residue, prime)
            modulus *= prime
        dt = time.time() - t0
        print(
            f"Prime {prime_index}/{len(primes)} {prime}: reconstructed values modulo {modulus} in {dt:.2f}s",
            flush=True,
        )
    return values, modulus


def forward_differences(values: list[int]) -> list[int]:
    diffs = []
    current = list(values)
    while current:
        diffs.append(current[0])
        current = [current[i + 1] - current[i] for i in range(len(current) - 1)]
    return diffs


def binomial_basis_next(basis: list[Fraction], k: int) -> list[Fraction]:
    denom = Fraction(1, k + 1)
    out = [Fraction(0)] * (len(basis) + 1)
    for idx, coeff in enumerate(basis):
        out[idx] += coeff * Fraction(-k, 1) * denom
        out[idx + 1] += coeff * denom
    return out


def interpolate_integer_values(values: list[int]) -> list[int]:
    degree = len(values) - 1
    diffs = forward_differences(values)
    basis = [Fraction(1)]
    poly = [Fraction(0) for _ in range(degree + 1)]
    for k, delta in enumerate(diffs):
        for idx, coeff in enumerate(basis):
            poly[idx] += Fraction(delta) * coeff
        if k < degree:
            basis = binomial_basis_next(basis, k)
    ints: list[int] = []
    for coeff in poly:
        if coeff.denominator != 1:
            raise ValueError(f"non-integral coefficient encountered: {coeff}")
        ints.append(coeff.numerator)
    return trim_coeffs(ints)


def trim_coeffs(coeffs: list[int]) -> list[int]:
    out = list(coeffs)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def eval_poly(coeffs: list[int], x: int) -> int:
    total = 0
    power = 1
    for coeff in coeffs:
        total += coeff * power
        power *= x
    return total


def format_poly(coeffs: list[int]) -> str:
    pieces = []
    for degree in range(len(coeffs) - 1, -1, -1):
        coeff = coeffs[degree]
        if coeff == 0:
            continue
        sign = "-" if coeff < 0 else "+"
        abs_coeff = abs(coeff)
        if degree == 0:
            term = f"{abs_coeff}"
        elif degree == 1:
            term = "x" if abs_coeff == 1 else f"{abs_coeff}*x"
        else:
            term = f"x^{degree}" if abs_coeff == 1 else f"{abs_coeff}*x^{degree}"
        if not pieces:
            pieces.append(f"-{term}" if coeff < 0 else term)
        else:
            pieces.append(f" {sign} {term}")
    return "".join(pieces) if pieces else "0"


def write_big_poly_file(
    path: str,
    rows: int,
    cols: int,
    coeffs: list[int],
    modulus: int,
    primes: list[int],
    solver: str,
    solver_args: list[str],
) -> None:
    with open(path, "w", encoding="ascii") as handle:
        handle.write("RECT_BIG_POLY_V1\n")
        handle.write(f"rows {rows}\n")
        handle.write(f"cols {cols}\n")
        handle.write(f"deg {len(coeffs) - 1}\n")
        handle.write(f"modulus_product {modulus}\n")
        handle.write(f"solver {solver}\n")
        for arg in solver_args:
            handle.write(f"solver_arg {arg}\n")
        for prime in primes:
            handle.write(f"prime {prime}\n")
        for degree, coeff in enumerate(coeffs):
            handle.write(f"coeff {degree} {coeff}\n")
        handle.write("end\n")


def main() -> int:
    args = parse_args()
    degree = args.degree if args.degree is not None else args.rows * args.cols
    solver = args.solver or choose_default_solver(args.rows)
    primes, bound, exact = choose_primes(args.primes, degree)
    if not exact:
        print(
            f"warning: product of supplied primes does not exceed value bound {bound}",
            file=sys.stderr,
        )

    print(f"Solver: {solver}")
    print(f"Grid: {args.rows}x{args.cols}")
    print(f"Degree: {degree}")
    print(f"Threads per evaluation: {args.threads}")
    print(f"Extra solver args: {args.solver_args if args.solver_args else '[]'}")
    print(f"Primes: {len(primes)}")

    values, modulus = reconstruct_values(
        solver=solver,
        rows=args.rows,
        cols=args.cols,
        prefix_depth=args.prefix_depth,
        degree=degree,
        primes=primes,
        threads=args.threads,
        solver_args=args.solver_args,
    )

    coeffs = interpolate_integer_values(values)
    for q, value in enumerate(values):
        if eval_poly(coeffs, q) != value:
            raise ValueError(f"interpolation check failed at q={q}")

    print("\nChromatic Polynomial P(x):")
    print(f"P(x) = {format_poly(coeffs)}")
    print("\nValues:")
    print(f"P(4) = {eval_poly(coeffs, 4)}")
    print(f"P(5) = {eval_poly(coeffs, 5)}")

    if args.show_values:
        print("\nReconstructed values:")
        for q, value in enumerate(values):
            print(f"P({q}) = {value}")

    if args.poly_out:
        write_big_poly_file(
            args.poly_out,
            args.rows,
            args.cols,
            coeffs,
            modulus,
            primes,
            solver,
            args.solver_args,
        )
        print(f"\nWrote reconstructed polynomial to {args.poly_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
