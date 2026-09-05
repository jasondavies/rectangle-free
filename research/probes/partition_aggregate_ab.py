#!/usr/bin/env python3
"""Sequential ABBA CPU timings with full-polynomial equality, never just P(4)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', default='build/partition_poly_8')
    parser.add_argument('--candidate', default='build/partition_residual_ab')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('solver_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.threads < 1 or args.cycles < 1:
        parser.error('threads and cycles must be positive')
    command = args.solver_args
    if command[:1] == ['--']:
        command = command[1:]
    if not command or any(x.startswith('--poly-out') for x in command):
        parser.error('pass geometry/solver arguments after --, without --poly-out')
    args.output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'OMP_NUM_THREADS': str(args.threads)}
    binaries = {'A': str(Path(args.control).resolve()),
                'B': str(Path(args.candidate).resolve())}
    binary_hashes = {k: hashlib.sha256(Path(v).read_bytes()).hexdigest()
                     for k, v in binaries.items()}
    reference = None
    measurements = []
    for index, label in enumerate('ABBA' * args.cycles):
        start = time.monotonic()
        result = subprocess.run([binaries[label], *command], env=env,
                                capture_output=True, text=True)
        wall = time.monotonic() - start
        (args.output/f'{index:02d}-{label}.log').write_text(result.stdout + result.stderr)
        result.check_returncode()
        polynomials = [line for line in result.stdout.splitlines() if line.startswith('P(x) =')]
        if len(polynomials) != 1:
            raise RuntimeError('Expected exactly one complete polynomial')
        if reference is None:
            reference = polynomials[0]
        if polynomials[0] != reference:
            raise RuntimeError(f'Full polynomial mismatch in run {index} ({label})')
        reported = re.search(r'Total elapsed including prefix generation: ([\d.]+)', result.stdout)
        row = dict(variant=label, wall_seconds=wall,
                   solver_seconds=float(reported[1]) if reported else None,
                   polynomial_sha256=hashlib.sha256(reference.encode()).hexdigest(),
                   residual_stats=[line for line in result.stdout.splitlines()
                                   if line.startswith('RESIDUAL_AB ')])
        measurements.append(row)
        print(json.dumps(row), flush=True)
        (args.output/'measurements.json').write_text(json.dumps(dict(
            commands=binaries, binary_sha256=binary_hashes, solver_args=command,
            threads=args.threads, environment={k: v for k, v in env.items()
                                             if k.startswith(('OMP_', 'RECT_'))},
            runs=measurements), indent=2) + '\n')
    means = {k: statistics.mean(r['wall_seconds'] for r in measurements if r['variant'] == k)
             for k in binaries}
    print(json.dumps(dict(mean_wall_seconds=means,
                         wall_reduction_percent=100*(1-means['B']/means['A']),
                         full_polynomial_parity=True)), flush=True)


if __name__ == '__main__':
    main()
