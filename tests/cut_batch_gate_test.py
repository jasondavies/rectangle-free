#!/usr/bin/env python3
"""Output-batch accounting and independent cut reconstruction regressions."""
from pathlib import Path
import sys
import struct
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'research/probes'))
import cut_batch_gate as gate


class BatchGateTest(unittest.TestCase):
    def fixture(self):
        layouts = {k:dict(bytes=10*k, entries=k) for k in [1, 2, 3]}
        records = [dict(choices=[dict(left=1, right=k, tiles=i+7)])
                   for i, k in enumerate([3, 1, 3, 2, 1])]
        return layouts, records

    def test_right_grouping_coverage_and_no_duplicate_builds(self):
        layouts, records = self.fixture()
        p = gate.batch_plan(layouts, records, [0]*5, 30, 3)
        self.assertEqual(p['batch_count'], 2)
        self.assertEqual(p['records'], 5)
        self.assertEqual(p['tiles'], sum(range(7, 12)))
        self.assertEqual(p['right_build_bytes'], 60)
        self.assertEqual(p['right_build_entries'], 6)
        self.assertEqual(p['right_build_layouts'], 3)
        self.assertEqual(p['peak_right_output_bytes'], 30)
        large = gate.batch_plan(layouts, records, [0]*5, 60, 5)
        self.assertEqual(large['batch_count'], 1)
        self.assertEqual(large['tiles'], p['tiles'])

    def test_caps_empty_and_invalid_inputs(self):
        layouts, records = self.fixture()
        for byte_cap, edge_cap in [(29, 5), (60, 1), (0, 5), (60, 0)]:
            with self.assertRaises(ValueError):
                gate.batch_plan(layouts, records, [0]*5, byte_cap, edge_cap)
        with self.assertRaises(ValueError):
            gate.batch_plan(layouts, records, [0], 60)
        with self.assertRaises(ValueError):
            gate.batch_plan(layouts, records, [1]*5, 60)
        empty = gate.batch_plan({}, [], [], 1)
        self.assertEqual(empty['records'], 0)
        self.assertEqual(empty['peak_right_output_bytes'], 0)

    def test_independent_owner_builds(self):
        layouts, records = self.fixture()
        a = gate.batch_plan(layouts, records[:2], [0]*2, 60)
        b = gate.batch_plan(layouts, records[2:], [0]*3, 60)
        self.assertEqual(a['right_build_bytes']+b['right_build_bytes'], 100)
        self.assertEqual(gate.batch_plan(layouts, records, [0]*5, 60)['right_build_bytes'], 60)

    def test_exact_candidate_reconstruction(self):
        # One row-major bit; transposition must move it to a different row.
        key = 1 << 58
        self.assertEqual(gate.half(key, 15), 1 << 30)
        choices = [dict(left=1 << 30, right=0, columns=15, transpose=0, reverse=0),
                   dict(left=0, right=1 << 20, columns=15, transpose=1, reverse=1)]
        record = dict(source=0, index=13, key=str(key), weight='19', choices=choices)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'in.tsv'
            path.write_text(f'0 13 {key} 19\n')
            gate.validate_records([record], path)
            record['weight'] = '20'
            with self.assertRaises(ValueError):
                gate.validate_records([record], path)


    def test_population_census(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'corpus.orbits'
            path.write_bytes(struct.pack('<8sIQ', b'R8SQT01\0', 8, 4) +
                             b''.join(struct.pack('<QQ', k, 1) for k in [1, 2, 16, 32]))
            p = gate.population_census([path], [(1, 1), (2, 4)])[0]
            self.assertEqual((p['records'], p['groups'], p['maximum_fanout']), (4, 3, 2))
            self.assertEqual([b['records'] for b in p['bands']], [2, 2])
            self.assertEqual([b['groups'] for b in p['bands']], [2, 1])
            self.assertEqual(len(p['sha256']), 64)


if __name__ == '__main__':
    unittest.main()
