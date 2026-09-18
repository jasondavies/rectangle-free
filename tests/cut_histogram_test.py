#!/usr/bin/env python3
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/probes'))
import reuse_budget_cut as cut


class HistogramTest(unittest.TestCase):
    def test_complete_demand_eligibility_boundaries(self):
        run=subprocess.run([str(ROOT/'build/cut_histogram_census'),'--demand-self-test'],
                           capture_output=True,text=True,timeout=60,check=True)
        self.assertIn('exact=OK',run.stdout)

    def test_indexed_and_subset_sum_tile_costs(self):
        run=subprocess.run([str(ROOT/'build/cut_histogram_census'),'--tile-self-test'],
                           capture_output=True,text=True,timeout=60,check=True)
        self.assertIn('exact=OK',run.stdout)

    def test_reference_bucket_class_parity(self):
        run=subprocess.run([str(ROOT/'build/cut_histogram_census'),'--self-test'],
                           capture_output=True,text=True,timeout=60,check=True)
        self.assertIn('exact=OK',run.stdout)

    def test_cold_warm_export_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            inp=p/'in.tsv';filt=p/'filter.tsv'
            inp.write_text(f'0 0 0 1\n0 1 {0x1030507090b0d0f} 9\n')
            filt.write_text('0 0 0 2 3 4 5\n0 1 0 2 3 4 5\n')
            baseline=subprocess.run([str(ROOT/'build/reuse_budget_cut_census'),str(inp),'all',str(filt)],
                                    text=True,capture_output=True,timeout=60,check=True)
            (p/'reference').write_text(baseline.stdout)
            expected=cut.load(p/'reference')
            for mode in ['cached','projected','indexed','zeta','grouped','planned','demand']:
                args=[str(ROOT/'build/cut_histogram_census'),str(inp),str(filt),str(p/mode),mode]
                run=subprocess.run(args,text=True,capture_output=True,timeout=60,check=True)
                metrics=list(map(json.loads,run.stdout.splitlines()))
                self.assertGreater(metrics[0]['builds'],0)
                self.assertEqual(metrics[1]['builds'],0)
                for phase in ['cold','warm']:
                    self.assertEqual(cut.load(p/f'{mode}.{phase}.jsonl'),expected)
                self.assertNotEqual(subprocess.run(args,capture_output=True).returncode,0)

    def test_grouped_restores_input_order(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            keys=[0x1030507090b0d0f,0x0123456789abcdef]*4
            inp=p/'in.tsv';filt=p/'filter.tsv'
            inp.write_text(''.join(f'0 {i} {key} {i+1}\n' for i,key in enumerate(keys)))
            filt.write_text(''.join(f'0 {i} 0 2 3 4 5\n' for i in range(len(keys))))
            expected=None
            for mode in ['zeta','grouped','planned','demand']:
                subprocess.run([str(ROOT/'build/cut_histogram_census'),str(inp),str(filt),str(p/mode),mode],
                               capture_output=True,timeout=60,check=True)
                for phase in ['cold','warm']:
                    actual=cut.load(p/f'{mode}.{phase}.jsonl')
                    if expected is None:expected=actual
                    self.assertEqual(actual,expected)
                    self.assertEqual([r['index'] for r in actual[1]],list(range(len(keys))))


if __name__=='__main__':
    unittest.main()
