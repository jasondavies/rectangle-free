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
        def write(path, prime, hess=0, profile=0, extra=""):
            path.write_text(
                f"CORE_CUDA_CONFIG hess={hess} boundary=0 scratch=0 profile={profile} {extra}\n"
                f"CORE_CUDA_SWEEP order=8 core=4 pool=7 active_queries=2 prime_index={prime} root=1 signs=2 kernel_s=1 exact=OK\n")
        def run(*extra):
            return subprocess.run([
                sys.executable, str(ROOT / "research/probes/hafnian_common_core_projection.py"),
                "--plan", str(plan), *extra, str(first), str(second),
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
        for field, value in (("warp_poly", 1), ("sparse_moments", 1), ("boundary_order", 16),
                             ("threads", 128), ("max_pool", 13), ("inverse_chain", 1),
                             ("live_moments", 1), ("sync_clear", 1)):
            write(second, 1, extra=f"{field}={value}")
            result = run()
            assert result.returncode != 0 and "different A/B" in result.stderr
        plan.write_text(
            "CORE_PLAN_BIN order=12 core=4 pool=11 active_queries=2 prime_index=0 groups=1\n"
            "CORE_PLAN_BIN order=12 core=4 pool=11 active_queries=2 prime_index=1 groups=1\n"
            "CORE_PLAN_DONE original_signs=128 exact_once=OK\n")
        for prime, path in enumerate((first, second)):
            path.write_text(
                "CORE_CUDA_CONFIG hess=1 boundary=1 scratch=1 profile=0 max_pool=13\n"
                f"CORE_CUDA_SWEEP order=12 core=2 pool=13 active_queries=2 prime_index={prime} root=1 signs=1 kernel_s=1 exact=OK\n")
        result = run("--pad-pool", "13", "--order", "12")
        assert result.returncode == 0 and "scope=uniform_padding_hypothesis" in result.stdout
        assert run("--order", "12").returncode != 0  # Padding cannot be implicit.
        assert run("--pad-pool", "13", "--order", "14").returncode != 0
    print("CORE_PROJECTION_TEST valid=OK profile_rejected=OK mixed_rejected=OK")


if __name__ == "__main__":
    main()
