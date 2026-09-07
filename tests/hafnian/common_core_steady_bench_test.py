#!/usr/bin/env python3
"""Projection must preserve every measured stratum and all CRT images."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("steady", ROOT / "research/probes/common_core_steady_bench.py")
steady = importlib.util.module_from_spec(spec)
spec.loader.exec_module(steady)


class ProjectionTest(unittest.TestCase):
    def fixture(self, directory):
        cells = [dict(core=36, pool=11, schedule=[2, 2, 0, 0], groups=10, samples=[dict(gid=0)]),
                 dict(core=36, pool=0, schedule=[1, 0, 0, 0], groups=20, samples=[dict(gid=10)])]
        payload = dict(format="common-core-steady-sample-v1", bins=cells)
        checksum = steady.cc.digest(payload)
        sample, log = Path(directory)/"sample.json", Path(directory)/"run.jsonl"
        sample.write_text(json.dumps(dict(payload=payload, sha256=checksum)))
        records = [dict(kind="identity", sample=checksum, repeats=2)]
        for repeat in range(2):
            for bid, gid, prime in [(0, 0, 0), (0, 0, 1), (1, 10, 0)]:
                records.append(dict(kind="timing", bin=bid, gid=gid, pi=prime, repeat=repeat,
                                    domain=131072, python_seconds=.1, prepare_seconds=.2,
                                    group_journal_seconds=.3, journal_seconds=.4,
                                    measurements=[dict(count=32768,gpu=1.,wall=1.5),
                                                  dict(count=65536,gpu=2.,wall=2.5)]))
        records.append(dict(kind="done"))
        log.write_text("\n".join(map(json.dumps,records)))
        return argparse.Namespace(sample=sample, log=log, checkpoint=65536), records

    def test_weighted_shared_and_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(directory)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                steady.project(args)
            result = json.loads(output.getvalue())
            self.assertAlmostEqual(result["shared_kernel_hours"]*3600, 80)
            self.assertAlmostEqual(result["tail_kernel_hours"]*3600, 80)
            self.assertAlmostEqual(result["shared_overhead_hours"]*3600, 52)
            self.assertAlmostEqual(result["tail_overhead_hours"]*3600, 58)
            self.assertAlmostEqual(result["total_hours"]*3600, 270)

    def test_missing_tail_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, rows = self.fixture(directory)
            args.log.write_text("\n".join(json.dumps(r) for r in rows if r.get("bin") != 1))
            with self.assertRaisesRegex(ValueError, "missing/duplicate"):
                steady.project(args)

    def test_incomplete_run_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, rows = self.fixture(directory)
            args.log.write_text("\n".join(map(json.dumps, rows[:-1])))
            with self.assertRaisesRegex(ValueError, "incomplete/mixed"):
                steady.project(args)

    def test_gray_windows(self):
        domain, count, chunk = 1 << 32, 262144, 32768
        begin = steady.range_begin(domain, count, 11, 487, chunk)
        self.assertEqual(begin % chunk, 0)
        self.assertLessEqual(begin+count, domain)
        self.assertEqual(begin, steady.range_begin(domain, count, 11, 487, chunk))
        self.assertEqual(steady.range_begin(domain, count, 11, 0, chunk), 0)
        self.assertEqual(steady.range_begin(count, count, 11, 487, chunk), 0)


if __name__ == "__main__":
    unittest.main()
