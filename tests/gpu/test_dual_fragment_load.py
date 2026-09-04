"""Test the actual host/device fragment loaders without requiring a GPU."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
NVCC = shutil.which(os.environ.get("NVCC", "nvcc"))


@unittest.skipUnless(NVCC, "nvcc is required for fragment-loader tests")
class DualFragmentLoadTest(unittest.TestCase):
    def test_geometries(self):
        for rows, pairs in ((6, 3), (6, 4), (6, 7), (7, 5), (8, 7)):
            with self.subTest(rows=rows, pairs=pairs), tempfile.TemporaryDirectory(
                prefix="dual-fragment-"
            ) as directory:
                binary = Path(directory) / "check"
                build = subprocess.run(
                    [NVCC, "-O2", "-std=c++17", "-arch=sm_89",
                     "-Xcompiler=-fopenmp", "-I" + str(ROOT / "src/gpu"),
                     f"-DGRID_ROWS={rows}", f"-DTWCOLOUR_PREFIX_PAIR_COUNT={pairs}",
                     str(ROOT / "tests/gpu/dual_fragment_load_test.cu"),
                     "-o", str(binary)],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    timeout=120,
                )
                self.assertEqual(build.returncode, 0, build.stdout)
                run = subprocess.run(
                    [str(binary)], text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, timeout=30,
                )
                self.assertEqual(run.returncode, 0, run.stdout)
                self.assertIn("host checks passed", run.stdout)


if __name__ == "__main__":
    unittest.main()
