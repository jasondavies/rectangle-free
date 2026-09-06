#!/usr/bin/env python3
"""CPU exactness gate for every independent common-core CUDA A/B combination.

The cooperative body runs on OpenMP threads, not CUDA. This detects arithmetic
and arena-lifetime mistakes but is not a substitute for CUDA race checking.
Optional --groups adds actual large-core/pool-11 cases from a saved plan log.
"""
import argparse
import itertools
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path)
    parser.add_argument("--candidates", action="store_true", help="test warp/sparse candidates on the accepted H/B/S kernel")
    args = parser.parse_args()
    out = ROOT / "build/common-core-variants-test"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_WAIT_POLICY="ACTIVE", OMP_DYNAMIC="FALSE")
    variants = ([(1, 1, 1, w, s, 0, 11) for w, s in itertools.product((0, 1), repeat=2)]
                + [(1, 1, 1, 1, 1, 16, pool) for pool in (11, 13)]) if args.candidates else [
        (*v, 0, 0, 0, 11) for v in itertools.product((0, 1), repeat=3)]
    for hess, boundary, scratch, warp, sparse, order_trials, pool in variants:
        name = f"h{hess}b{boundary}s{scratch}w{warp}m{sparse}o{order_trials}q{pool}"
        exe = out / name
        with (out / f"{name}.log").open("w") as log:
            subprocess.run([
                os.environ.get("CXX", "g++"), "-x", "c++", "-O2", "-std=c++17",
                "-fopenmp", "-fsanitize=undefined", "-fno-sanitize-recover=all",
                "-DCORE_HOST_EMULATION", f"-DCORE_OPT_HESS={hess}",
                f"-DCORE_OPT_BOUNDARY={boundary}", f"-DCORE_OPT_SCRATCH={scratch}",
                f"-DCORE_OPT_WARP_POLY={warp}", f"-DCORE_OPT_SPARSE_MOMENTS={sparse}",
                f"-DCORE_BOUNDARY_ORDER={order_trials}", f"-DCORE_MAX_POOL={pool}",
                str(ROOT / "research/gpu/hafnian_common_core_gpu.cu"), "-o", str(exe),
            ], check=True, stdout=log, stderr=log)
            subprocess.run([str(exe), "--self-test"], check=True, env=env,
                           stdout=log, stderr=log)
            if pool == 13:
                for prime in (2147483647, 2147483629, 2147483587, 2147483579):
                    for order in (42, 44, 46, 48):
                        subprocess.run([
                            str(exe), "--groups", str(ROOT / "tests/hafnian/common_core_pool13.groups"),
                            "--order", str(order), "--prime", str(prime), "--cap", "13",
                            "--count", "4", "--threads", "4",
                        ], check=True, env=env, stdout=log, stderr=log)
            if args.groups:
                for prime in (2147483647, 2147483629, 2147483587, 2147483579):
                    for order in (42, 44, 46, 48):
                        subprocess.run([
                            str(exe), "--groups", str(args.groups.resolve()),
                            "--order", str(order), "--prime", str(prime),
                            "--count", "4", "--threads", "4",
                        ], check=True, env=env, stdout=log, stderr=log)
        print(f"CORE_VARIANT_TEST variant={name} ubsan=OK exact=OK", flush=True)


if __name__ == "__main__":
    main()
