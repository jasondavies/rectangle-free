#!/usr/bin/env python3
"""Research-only shortlist/cache and filtered-export regressions."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'research/probes'))
import cut_shortlist as short
import reuse_budget_cut as cut


class ShortlistTest(unittest.TestCase):
    def run_binary(self, name, *args, success=True):
        p = subprocess.run([str(ROOT/'build'/name), *map(str, args)],
                           capture_output=True, text=True, timeout=60)
        if success:
            self.assertEqual(p.returncode, 0, p.stderr)
        else:
            self.assertNotEqual(p.returncode, 0)
        return p

    def test_counts_cache_shortlists_and_filtered_export(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            inp = d/'input.tsv'
            inp.write_text(f'0 0 0 1\n1 17 {0x1030507090b0d0f} 5\n')
            cold = self.run_binary('cut_support_counts', inp, d/'cache')
            warm = self.run_binary('cut_support_counts', inp, d/'cache2', d/'cache')
            a, b = [list(map(json.loads, p.stdout.splitlines())) for p in [cold, warm]]
            self.assertEqual(a[:-1], b[:-1])
            self.assertGreater(a[-1]['new_builds'], 0)
            self.assertEqual(b[-1]['new_builds'], 0)
            self.run_binary('cut_support_counts', inp, d/'cache', success=False)
            counts = d/'counts.jsonl'
            counts.write_text(cold.stdout)
            supports, rows = short.load_counts(counts)
            exact = self.run_binary('reuse_budget_cut_census', inp, 'all')
            path = d/'exact.jsonl'
            path.write_text(exact.stdout)
            layouts, original = cut.load(path)
            for mode in ['product', 'reuse', 'mixed']:
                for n in [2, 4]:
                    slots = short.shortlist(supports, rows, n, mode)
                    self.assertEqual(slots, short.shortlist(supports, rows, n, mode))
                    for keep in slots:
                        self.assertEqual(keep[0], 0)
                        self.assertEqual(len(set(keep)), 1+2*n)
                        for i in keep[1:]:
                            self.assertIn(i ^ 1, keep)
                    result = short.evaluate(supports, rows, path, slots)
                    self.assertLessEqual(result['tiles'], result['baseline_tiles'])
            slots = short.shortlist(supports, rows, 4, 'reuse')
            filt = d/'filter.tsv'
            filt.write_text(''.join(' '.join(map(str,[r['source'],r['index'],*s]))+'\n'
                                   for r,s in zip(rows, slots)))
            out = self.run_binary('reuse_budget_cut_census', inp, 'all', filt)
            (d/'filtered.jsonl').write_text(out.stdout)
            fl, fr = cut.load(d/'filtered.jsonl')
            for r, expected, keep in zip(fr, original, slots):
                self.assertEqual(r['choices'], [expected['choices'][i] for i in keep])
            for key in fl:
                self.assertEqual(fl[key], layouts[key])
            for malformed in ['0 0 1\n', '0 0 0 0\n', '0 0 0 140\n', '0 0 0\n']:
                filt.write_text(malformed)
                self.run_binary('reuse_budget_cut_census', inp, 'all', filt, success=False)
            counts.write_text('\n'.join(cold.stdout.splitlines()[:-1])+'\n')
            with self.assertRaises(ValueError):
                short.load_counts(counts)
            (d/'bad-cache').write_text('R8SUPPORT1\n0 1\n0 1\n')
            self.run_binary('cut_support_counts', inp, d/'bad-output', d/'bad-cache', success=False)

    def test_large_complete_group_sampling(self):
        import struct
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            data = []
            for right in range(1200):
                key = sum(((right >> (4*r)) & 15) << (8*r+4) for r in range(8))
                data.append(struct.pack('<QQ', key, 1))
            corpus = d/'in.orbits'
            corpus.write_bytes(struct.pack('<8sIQ', b'R8SQT01\0', 8, len(data))+b''.join(data))
            cut.sample([corpus], d/'sample', 506, 1, [(1024, 4095)])
            self.assertEqual(len((d/'sample/records.tsv').read_text().splitlines()), 1200)


if __name__ == '__main__':
    unittest.main()
