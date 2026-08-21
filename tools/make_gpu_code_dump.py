#!/usr/bin/env python3
"""Create a deterministic concatenation of the maintained GPU solver surface."""

from pathlib import Path
import subprocess
import sys


FILES = [
    "src/gpu/gpu_cuda_utils.cuh",
    "src/gpu/gpu_memory_policy.hpp",
    "src/gpu/twocolour_gpu_common.cuh",
    "src/gpu/twocolour_7x7_engine.cuh",
    "src/gpu/twocolour_prefix_algebra.cuh",
    "src/gpu/twocolour_prefix_core.cuh",
    "src/gpu/twocolour_weight_class_bmma.cuh",
    "src/gpu/twocolour_canonical_device.cuh",
    "src/gpu/gpu_result_checkpoint.hpp",
    "src/common/sha256.hpp",
    "src/gpu/twocolour_7x7_gpu.cu",
    "src/gpu/twocolour_7x9_engine.cuh",
    "src/gpu/twocolour_7x9_cache_build.cu",
    "src/gpu/twocolour_7x9_packed_solve.cu",
    "src/gpu/twocolour_7x9_four_owner_solve.cu",
    "src/gpu/twocolour_8x8_prefix_solve.cu",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "code-dump.txt"
    if not output.is_absolute():
        output = root / output
    revision = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True
    ).strip()
    sections = [
        "PRODUCTION GPU CODE DUMP",
        f"Repository base snapshot: {revision}",
        "Generated from the current working tree by tools/make_gpu_code_dump.py.",
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
