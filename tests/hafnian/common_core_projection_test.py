#!/usr/bin/env python3
"""Reject instrumented or mixed-kernel logs before campaign projection."""
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    with tempfile.TemporaryDirectory(prefix="core-projection-") as directory:
        root = Path(directory)
        plan = root / "plan"
        plan.write_text(
            "CORE_PLAN_BIN order=8 core=4 pool=7 active_queries=2 prime_index=0 groups=1\n"
            "CORE_PLAN_BIN order=8 core=4 pool=7 active_queries=2 prime_index=1 groups=1\n"
            "CORE_PLAN_DONE original_signs=32 exact_once=OK\n")
        first, second = root / "first", root / "second"
        def write(path, prime, hess=0, profile=0):
            path.write_text(
                f"CORE_CUDA_CONFIG hess={hess} boundary=0 scratch=0 profile={profile}\n"
                f"CORE_CUDA_SWEEP order=8 core=4 pool=7 active_queries=2 prime_index={prime} root=1 signs=2 kernel_s=1 exact=OK\n")
        def run():
            return subprocess.run([
                sys.executable, str(ROOT / "research/probes/hafnian_common_core_projection.py"),
                "--plan", str(plan), str(first), str(second),
            ], text=True, capture_output=True)
        write(first, 0)
        write(second, 1)
        assert run().returncode == 0
        write(second, 1, profile=1)
        result = run()
        assert result.returncode != 0 and "instrumented phase" in result.stderr
        write(second, 1, hess=1)
        result = run()
        assert result.returncode != 0 and "different A/B" in result.stderr
    print("CORE_PROJECTION_TEST valid=OK profile_rejected=OK mixed_rejected=OK")


if __name__ == "__main__":
    main()
