#!/usr/bin/env python3
"""Versioned timing export must not mix kernels, GPUs, or old/new fields."""
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "research/probes/hafnian_common_core_cost.py"
CONFIG = ("hess=1 boundary=1 scratch=1 profile=0 warp_poly=1 sparse_moments=1 "
          "boundary_order=16 max_pool=11 threads=128 inverse_chain=1 live_moments=1 sync_clear=1")

with tempfile.TemporaryDirectory(prefix="core-cost-") as directory:
    base = Path(directory)
    first, second, out = (base / x for x in ("first", "second", "costs"))
    def write(path, config=CONFIG, device="RTX_PRO_6000 sm=12.0", order=50):
        path.write_text(f"CORE_CUDA_CONFIG {config}\nCORE_CUDA_DEVICE name={device}\n"
                        f"CORE_CUDA_SWEEP order={order} core={order-8} pool=11 active_queries=2 "
                        "prime_index=0 root=123 signs=32768 kernel_s=0.01 exact=OK\n")
    def run(version, *paths):
        if out.exists():
            out.unlink()
        return subprocess.run([sys.executable, str(TOOL), "--format", str(version),
                               "--output", str(out), *map(str, paths)], text=True, capture_output=True)
    write(first)
    assert run(2, first).returncode == 0
    assert out.read_text().startswith("HCCOST02 ") and "50 42 11 2 0" in out.read_text()
    assert run(1, first).returncode != 0
    for config, device in ((CONFIG.replace("inverse_chain=1", "inverse_chain=0"), "RTX_PRO_6000 sm=12.0"),
                           (CONFIG, "L40S sm=8.9"), (CONFIG.replace("profile=0", "profile=1"), "RTX_PRO_6000 sm=12.0")):
        write(second, config, device)
        assert run(2, first, second).returncode != 0
        assert not out.exists()
    write(second, order=52)
    assert run(2, second).returncode != 0
    write(second, CONFIG.replace("threads=128", "threads=256"))
    assert run(2, second).returncode != 0
print("CORE_COST_TEST v2=OK order50=OK mixed_rejected=OK old_model_rejected=OK")
