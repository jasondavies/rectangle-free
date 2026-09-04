"""Compile-only configuration regressions; skips when nvcc is unavailable."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
NVCC = shutil.which(os.environ.get("NVCC", "nvcc"))


@unittest.skipUnless(NVCC, "nvcc is required for compile-only prefix tests")
class PrefixConfigurationTest(unittest.TestCase):
    def compile(self, rows, pairs, chunk=16):
        with tempfile.TemporaryDirectory(prefix="prefix-contract-") as directory:
            return subprocess.run(
                [NVCC, "-std=c++17", "-arch=sm_89", "-c",
                 "-I" + str(ROOT / "src/gpu"),
                 f"-DGRID_ROWS={rows}", f"-DTWCOLOUR_PREFIX_PAIR_COUNT={pairs}",
                 f"-DTWCOLOUR_PREFIX_TASK_CHUNK={chunk}",
                 str(ROOT / "tests/gpu/prefix_configuration_test.cu"),
                 "-o", str(Path(directory) / "fixture.o")],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=90,
            )

    def test_supported_widths(self):
        for rows, pairs in ((6, 1), (6, 5), (7, 5), (7, 7), (8, 7)):
            with self.subTest(rows=rows, pairs=pairs):
                result = self.compile(rows, pairs)
                self.assertEqual(result.returncode, 0, result.stdout)

    def test_unsupported_widths(self):
        for rows, pairs, message in (
            (7, 4, "more suffix bits"),
            (8, 8, "at most seven prefix pairs"),
            (6, 9, "at most seven prefix pairs"),
        ):
            with self.subTest(rows=rows, pairs=pairs):
                result = self.compile(rows, pairs)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(message, result.stdout)

    def test_invalid_task_chunks(self):
        for chunk, message in (
            (0, "positive uint32_t"),
            (-1, "positive uint32_t"),
            (2**32, "positive uint32_t"),
            (2**31, "including final warp claims"),
        ):
            with self.subTest(chunk=chunk):
                result = self.compile(8, 7, chunk)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(message, result.stdout)


if __name__ == "__main__":
    unittest.main()
