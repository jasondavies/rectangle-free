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
    args = parser.parse_args()
    out = ROOT / "build/common-core-variants-test"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_WAIT_POLICY="ACTIVE", OMP_DYNAMIC="FALSE")
    for hess, boundary, scratch in itertools.product((0, 1), repeat=3):
        name = f"h{hess}b{boundary}s{scratch}"
        exe = out / name
        with (out / f"{name}.log").open("w") as log:
            subprocess.run([
                os.environ.get("CXX", "g++"), "-x", "c++", "-O2", "-std=c++17",
                "-fopenmp", "-fsanitize=undefined", "-fno-sanitize-recover=all",
                "-DCORE_HOST_EMULATION", f"-DCORE_OPT_HESS={hess}",
                f"-DCORE_OPT_BOUNDARY={boundary}", f"-DCORE_OPT_SCRATCH={scratch}",
                str(ROOT / "research/gpu/hafnian_common_core_gpu.cu"), "-o", str(exe),
            ], check=True, stdout=log, stderr=log)
            subprocess.run([str(exe), "--self-test"], check=True, env=env,
                           stdout=log, stderr=log)
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
