#!/usr/bin/env python3
"""Exact small-instance gate for a shared core and polynomial-hafnian boundary.

This is an algebra test, not a performance implementation or a GPU projection.
For each core sign matrix R, form
  f(z) = det(I-z A_CC R)^(-1/2),
  K(z) = A_BB + z A_BC R (I-z A_CC R)^(-1) A_CB.
For a boundary subset S, average sign_product * [z^(|C|/2)]
f(z) haf(K[S,S]) over the 2^(|C|/2-1) fixed-first-sign assignments.
The result is haf(A[C union S]). Boundary signs are NOT enumerated.
"""
import argparse
from functools import lru_cache
from itertools import combinations
import random
import time


def product(a, b, degree, prime):
    out = [0] * (degree + 1)
    for i, x in enumerate(a):
        if x:
            for j in range(degree + 1 - i):
                out[i+j] += x * b[j]
    return tuple(v % prime for v in out)


def multiply(a, b, prime):
    return [[sum(x*y for x, y in zip(row, column)) % prime
             for column in zip(*b)] for row in a]


def grouped_minors(adjacency, core, subsets, prime):
    """All vertices 0..core-1 are fixed; subsets index the remaining vertices."""
    if core < 0 or core % 2 or core > len(adjacency):
        raise ValueError('core must have nonnegative even order')
    boundary = len(adjacency) - core
    if any(len(s) % 2 or len(set(s)) != len(s) or
           any(v < 0 or v >= boundary for v in s) for s in subsets):
        raise ValueError('boundary subsets must be even and valid')
    degree = core // 2
    if prime <= 2 * degree:
        raise ValueError('prime too small for the coefficient recurrence')
    totals = [0] * len(subsets)
    terms = 1 << max(0, degree - 1)
    for signs in range(terms):
        signs_by_pair = [1] + [1 if signs & (1 << i) else -1
                              for i in range(max(0, degree - 1))]
        sign_product = 1
        for s in signs_by_pair:
            sign_product *= s
        # Use M=R A_CC; Sylvester makes its determinant equal to A_CC R.
        m = [[signs_by_pair[i//2] * adjacency[i ^ 1][j] % prime
              for j in range(core)] for i in range(core)]
        vectors = [[signs_by_pair[i//2] * adjacency[i ^ 1][core+j] % prime
                    for j in range(boundary)] for i in range(core)]
        k = [[[adjacency[core+i][core+j] % prime] + [0] * degree
              for j in range(boundary)] for i in range(boundary)]
        power = [[int(i == j) for j in range(core)] for i in range(core)]
        traces = [0] * (degree + 1)
        for d in range(1, degree + 1):
            for i in range(boundary):
                for j in range(boundary):
                    k[i][j][d] = sum(adjacency[core+i][v] * vectors[v][j]
                                     for v in range(core)) % prime
            power = multiply(m, power, prime)
            traces[d] = sum(power[i][i] for i in range(core)) % prime
            if d < degree:
                vectors = multiply(m, vectors, prime)
        f = [1] + [0] * degree
        for d in range(1, degree + 1):
            f[d] = (sum(traces[j] * f[d-j] for j in range(1, d+1))
                    * pow(2*d, -1, prime)) % prime

        @lru_cache(None)
        def boundary_hafnian(mask):
            if not mask:
                return (1,) + (0,) * degree
            first = (mask & -mask).bit_length() - 1
            rest = mask ^ (1 << first)
            candidates = rest
            answer = [0] * (degree + 1)
            while candidates:
                bit = candidates & -candidates
                second = bit.bit_length() - 1
                contribution = product(k[first][second],
                                       boundary_hafnian(rest ^ bit), degree, prime)
                answer = [(a+b) % prime for a, b in zip(answer, contribution)]
                candidates ^= bit
            return tuple(answer)

        for i, subset in enumerate(subsets):
            mask = sum(1 << v for v in subset)
            polynomial = boundary_hafnian(mask)
            value = sum(f[d] * polynomial[degree-d] for d in range(degree+1))
            totals[i] = (totals[i] + sign_product * value) % prime
    inverse_terms = pow(terms, -1, prime)
    return [value * inverse_terms % prime for value in totals]


def reference_minors(adjacency, core, subsets, prime):
    @lru_cache(None)
    def haf(mask):
        if not mask:
            return 1
        first = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << first)
        candidates = rest
        answer = 0
        while candidates:
            bit = candidates & -candidates
            second = bit.bit_length() - 1
            answer += adjacency[first][second] * haf(rest ^ bit)
            candidates ^= bit
        return answer % prime
    return [haf((1 << core)-1 | sum(1 << (core+v) for v in subset))
            for subset in subsets]


def self_test():
    started = time.monotonic()
    rng = random.Random(475)
    checks = 0
    # Same shared-parent/three-deletion structure as the census. Include
    # singular/zero cores, signed weights, no core, and multiple minor orders.
    for core, boundary in [(0, 7), (2, 5), (4, 7), (4, 9), (6, 5), (8, 7)]:
        subsets = list(combinations(range(boundary), boundary-3))
        subsets += [(), tuple(range(2))]
        n = core + boundary
        for mode in ('complete', 'empty_core', 'weighted'):
            a = [[0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    value = 1 if mode == 'complete' else rng.randrange(-2, 4)
                    if mode == 'empty_core' and j < core:
                        value = 0
                    a[i][j] = a[j][i] = value
            for prime in (1000003, 1000033):
                actual = grouped_minors(a, core, subsets, prime)
                expected = reference_minors(a, core, subsets, prime)
                if actual != expected:
                    raise AssertionError((core, boundary, mode, prime, actual, expected))
                checks += len(subsets)
    print(f'COMMON_CORE_IDENTITY minors={checks} primes=2 '
          f'boundary_signs=eliminated exact=OK seconds={time.monotonic()-started:.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', required=True)
    parser.parse_args()
    self_test()
