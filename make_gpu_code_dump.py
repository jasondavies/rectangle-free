#!/usr/bin/env python3
"""Create a deterministic concatenation of the maintained GPU solver surface."""

from pathlib import Path
import subprocess
import sys


FILES = [
    "gpu_cuda_utils.cuh",
    "gpu_memory_policy.hpp",
    "twocolour_gpu_common.cuh",
    "twocolour_7x7_engine.cuh",
    "twocolour_prefix_algebra.cuh",
    "twocolour_prefix_core.cuh",
    "twocolour_weight_class_bmma.cuh",
    "twocolour_canonical_device.cuh",
    "gpu_result_checkpoint.hpp",
    "sha256.hpp",
    "twocolour_7x7_gpu.cu",
    "twocolour_7x9_engine.cuh",
    "twocolour_7x9_cache_build.cu",
    "twocolour_7x9_packed_solve.cu",
    "twocolour_7x9_four_owner_solve.cu",
    "twocolour_8x8_prefix_solve.cu",
]


def main() -> None:
    root = Path(__file__).resolve().parent
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "code-dump.txt"
    if not output.is_absolute():
        output = root / output
    revision = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True
    ).strip()
    sections = [
        "PRODUCTION GPU CODE DUMP",
        f"Repository base snapshot: {revision}",
        "Generated from the current working tree by make_gpu_code_dump.py.",
        "",
        "Included files:",
        *(f"  - {name}" for name in FILES),
        "",
    ]
    separator = "=" * 88
    for name in FILES:
        path = root / name
        sections.extend(
            [separator, f"FILE: {name}", separator, "", path.read_text(), ""]
        )
    output.write_text("\n".join(sections))
    print(f"WROTE {output} files={len(FILES)}")


if __name__ == "__main__":
    main()
