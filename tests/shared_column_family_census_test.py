#!/usr/bin/env python3
"""Regression checks for canonical deletion families and workload sampling."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

BINARY = Path(sys.argv.pop(1) if len(sys.argv) > 1 else 'build/shared_column_family_census').resolve()


class FamilyCensusTest(unittest.TestCase):
    def run_probe(self, *args, ok=True):
        result = subprocess.run([str(BINARY), *map(str, args)], text=True,
                                capture_output=True, timeout=60)
        self.assertEqual(result.returncode == 0, ok, result.stderr)
        return result.stdout

    def test_exact_self_tests(self):
        self.assertIn('SHARED_COLUMN_FAMILY_TEST exact=OK', self.run_probe('--self-test'))

    def test_global_parent_fanout_not_sample_occupancy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.tsv'
            # Empty grid plus a single active cell. Both have the zero core;
            # its complete family has nine orbits (0..8 cells in one column).
            path.write_text('0\t0\t0\t10\t2\n0\t1\t1\t10\t3\n')
            result = [json.loads(line) for line in self.run_probe('parents', path, 2, 1).splitlines()]
            samples = [r for r in result if r['type'] == 'parent_sample']
            self.assertEqual([r['fanout'] for r in samples], [9, 9])
            self.assertEqual(result[-1]['parents'], 1)
            self.assertEqual(result[-1]['weighted_tiles'], [0, 0, 50, 0])
            self.assertEqual(result[-1]['weighted_records'], [0, 0, 20, 0])
            path.write_text('0\t0\t0\t10\t2\n0\t1\t1')
            self.run_probe('parents', path, 2, 1, ok=False)

    def test_complete_assigned_family_and_caps(self):
        result = [json.loads(line) for line in
                  self.run_probe('family', 0, 100, 100000000, 10).splitlines()]
        self.assertEqual(result[0]['fanout'], 9)
        self.assertEqual(result[-1]['type'], 'family_complete')
        self.assertEqual(result[-1]['checked_answers'], 54)
        self.assertEqual(result[-1]['capped_method_sides'], 0)
        self.assertEqual(len([r for r in result if 'method' in r and 'status' in r]), 8)
        capped = [json.loads(line) for line in
                  self.run_probe('family', 0, 100, 1, 10).splitlines()]
        self.assertEqual(capped[-1]['checked_answers'], 0)
        self.assertEqual(capped[-1]['capped_method_sides'], 8)
        for row in capped:
            if 'status' in row:
                self.assertIsNone(row['nonzero_answers'])
        for parent, cache, work, seconds in [(1 << 56, 100, 100, 10),
                                            (0, 0, 100, 10), (0, 100, 0, 10),
                                            (0, 100, 100, 'nan')]:
            self.run_probe('family', parent, cache, work, seconds, ok=False)

    def test_production_representatives(self):
        result = [json.loads(line) for line in self.run_probe('model-parent', 0).splitlines()]
        self.assertEqual(result[-1]['type'], 'parent_model_complete')
        self.assertEqual(result[-1]['fanout'], 9)
        self.assertEqual(len(set(r['production_key'] for r in result[:-1])), 9)
        self.assertEqual(result[-1]['production_tiles'], sum(r['production_tiles'] for r in result[:-1]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'keys.tsv'
            path.write_text('0 0 0 1 0\n')
            self.assertEqual(json.loads(self.run_probe('check-production-keys', path))['keys'], 1)
            path.write_text(f'0 0 {(1 << 64)-1} 1 0\n')
            self.run_probe('check-production-keys', path, ok=False)

    def test_orbit_population_and_reader_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.orbits'
            header = struct.pack('<8sIQ', b'R8SQT01\0', 8, 2)
            path.write_bytes(header + struct.pack('<QQQQ', 0, 1, 1, 1))
            output = [json.loads(s) for s in self.run_probe(path, 100, 2, 503).splitlines()]
            self.assertEqual(output[0]['records'], 2)
            samples = [r for r in output if r['type'] == 'sample']
            self.assertEqual(len(samples), 2)
            self.assertEqual(sum(r['stratum_records'] for r in samples), 2)
            self.run_probe(f'{path},{path}', 100, 2, 503, ok=False)
            path.write_bytes(header + struct.pack('<QQQQ', (1 << 64)-1, 1, 1, 1))
            self.run_probe(path, 100, 2, 503, ok=False)
            path.write_bytes(header)
            self.run_probe(path, 100, 2, 503, ok=False)


if __name__ == '__main__':
    unittest.main()
