#!/usr/bin/env python3
import importlib.util
import itertools
import json
import random
import struct
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction

BINARY = Path(sys.argv.pop(1) if len(sys.argv) > 1 else 'build/reuse_budget_cut_census').resolve()
SPEC = importlib.util.spec_from_file_location('reuse_cut', Path(__file__).resolve().parents[1] / 'research/probes/reuse_budget_cut.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReuseCutTest(unittest.TestCase):
    def test_sampling_keeps_complete_left_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.orbits'
            records = []
            for left, count in [(1, 8), (2, 64), (3, 256)]:
                for right in range(count):
                    key = 0
                    for row in range(8):
                        shift = (7-row)*4
                        key = (key << 8) | ((left >> shift) & 15) | (((right >> shift) & 15) << 4)
                    records.append(struct.pack('<QQ', key, 1))
            path.write_bytes(struct.pack('<8sIQ', b'R8SQT01\0', 8, len(records)) + b''.join(records))
            out = Path(directory) / 'selected'
            MODULE.sample([path], out, 505, 1)
            metadata = json.loads((out / 'selection.json').read_text())
            self.assertEqual(sorted(g['records'] for g in metadata['groups']), [8, 64, 256])
            self.assertEqual(len((out / 'records.tsv').read_text().splitlines()), 328)
            with self.assertRaises(FileExistsError):
                MODULE.sample([path], out, 505, 1)

    def fixture(self):
        layouts = {k: dict(bytes=100, entries=10) for k in [0, 1, 2, 10, 11, 12]}
        rows = [dict(choices=[dict(left=i, right=10+i, tiles=100),
                              dict(left=2, right=10+i, tiles=20)]) for i in range(2)]
        return layouts, rows

    def test_shared_opening_is_charged_once(self):
        layouts, rows = self.fixture()
        result = MODULE.optimize(layouts, rows, Fraction('1.5'))
        self.assertEqual(result['choices'], [1, 1])
        self.assertEqual(result['tiles'], 40)
        self.assertEqual(result['left_reserved_bytes'], 300)
        self.assertEqual(result['left_reserved_keys'], 3)
        self.assertEqual(result['left_keys'], 1)

    def test_count_and_byte_caps(self):
        layouts, rows = self.fixture()
        result = MODULE.optimize(layouts, rows, Fraction('1.25'))
        self.assertEqual(result['choices'], [0, 0])  # floor(2 * 1.25) = 2 IDs
        layouts[2]['bytes'] = 101
        result = MODULE.optimize(layouts, rows, Fraction('1.5'))
        self.assertEqual(result['choices'], [0, 0])
        layouts[2]['bytes'] = 100
        layouts[12]['bytes'] = 101
        for row in rows:
            row['choices'][1]['right'] = 12
        self.assertEqual(MODULE.optimize(layouts, rows, Fraction('1.5'))['choices'], [0, 0])

    def test_memory_only_diagnostic(self):
        layouts, rows = self.fixture()
        layouts[2]['bytes'] = 20
        self.assertEqual(MODULE.optimize(layouts, rows, Fraction('1.25'))['choices'], [0, 0])
        self.assertEqual(MODULE.optimize(layouts, rows, Fraction('1.25'), Fraction(1000000))['choices'], [1, 1])

    def test_existing_facilities_are_free_and_no_cost_regressions(self):
        layouts, rows = self.fixture()
        rows[0]['choices'][1]['left'] = 1
        result = MODULE.optimize(layouts, rows, Fraction(1))
        self.assertEqual(result['choices'], [1, 0])
        rows[0]['choices'][1]['tiles'] = 101
        self.assertEqual(MODULE.optimize(layouts, rows, Fraction(1))['choices'], [0, 0])

    def test_cpp_cut_inverse_and_export(self):
        result = subprocess.run([str(BINARY), '--self-test'], text=True, capture_output=True, timeout=30, check=True)
        self.assertIn('exact=OK', result.stdout)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.tsv'
            out = Path(directory) / 'output.jsonl'
            path.write_text('0 0 0 1\n')
            run = subprocess.run([str(BINARY), str(path), 'four'], text=True, capture_output=True, timeout=30, check=True)
            out.write_text(run.stdout)
            layouts, rows = MODULE.load(out)
            self.assertEqual(len(rows[0]['choices']), 8)
            self.assertEqual(rows[0]['choices'][0]['left'], 0)
            self.assertEqual(rows[0]['choices'][0]['right'], 0)
            baseline = rows[0]['choices'][0]['tiles']
            self.assertTrue(all(c['tiles'] == baseline for c in rows[0]['choices']))
            self.assertEqual(MODULE.optimize(layouts, rows, Fraction(1))['tiles'], baseline)
            selected = Path(directory) / 'selected.json'
            subprocess.run([sys.executable, str(SPEC.origin), 'solve', str(out),
                            str(selected), '--memory-only'], capture_output=True, check=True)
            self.assertEqual(json.loads(selected.read_text())['budgets'][0]['tiles'], baseline)
            out.write_text('\n'.join(run.stdout.splitlines()[:-1]) + '\n')
            with self.assertRaises(ValueError):
                MODULE.load(out)
            path.write_text('0 0 0')
            self.assertNotEqual(subprocess.run([str(BINARY), str(path), 'four'], capture_output=True).returncode, 0)

    def test_optimistic_bound(self):
        layouts, rows = self.fixture()
        self.assertEqual(MODULE.savings_upper_bound(layouts, rows, Fraction(1)), 0)
        self.assertEqual(MODULE.savings_upper_bound(layouts, rows, Fraction('1.5')), 160)
        self.assertEqual(MODULE.savings_upper_bound(layouts, rows, Fraction(1), True), 160)
        layouts[2]['bytes'] = 200
        self.assertEqual(MODULE.savings_upper_bound(layouts, rows, Fraction('1.5')), 80)

    def test_bounds_against_exhaustive_small_instances(self):
        rng = random.Random(505)
        for _ in range(12):
            layouts = {k: dict(bytes=rng.randrange(1, 6)*10, entries=1) for k in range(8)}
            rows = [dict(choices=[dict(left=i, right=4+i, tiles=100)] +
                         [dict(left=rng.randrange(4), right=rng.randrange(4, 8),
                               tiles=rng.randrange(150)) for _ in range(3)]) for i in range(3)]
            baseline = MODULE.statistics(layouts, rows, [0]*3)
            for growth in [Fraction(1), Fraction('1.5'), Fraction(2)]:
                for retirement in [False, True]:
                    best = 0
                    for choices in itertools.product(range(4), repeat=3):
                        feasible = True
                        for side in ['left', 'right']:
                            keys = {r['choices'][c][side] for r, c in zip(rows, choices)}
                            if not retirement:
                                keys.update(r['choices'][0][side] for r in rows)
                            if len(keys) > int(baseline[side+'_keys']*growth) or sum(layouts[k]['bytes'] for k in keys) > int(baseline[side+'_bytes']*growth):
                                feasible = False
                        if feasible:
                            cost = sum(r['choices'][c]['tiles'] for r, c in zip(rows, choices))
                            best = max(best, baseline['tiles']-cost)
                    self.assertGreaterEqual(MODULE.savings_upper_bound(layouts, rows, growth, retirement), best)


if __name__ == '__main__':
    unittest.main()
