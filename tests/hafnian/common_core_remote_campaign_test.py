#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import common_core_campaign as cc
from common_core_remote_campaign import complete_snapshot,atomic_json

class CoverageTest(unittest.TestCase):
    def test_coverage_requires_all_groups_primes_and_contiguous_ranges(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'result.sqlite'
            task=dict(group_start=0,group_end=2)
            j=cc.Journal(path,dict(format=cc.FORMAT,**task))
            meta=dict(domain=4,bounds=[35],primes=[2])
            try:
                self.assertFalse(complete_snapshot(path,task))
                for gid in (0,1):
                    j.put_group(gid,meta)
                    j.put_range(gid,0,0,4,meta,[1],0.,0.)
                    j.put_range(gid,1,2,4,meta,[2],0.,0.)
                self.assertFalse(complete_snapshot(path,task))
                j.put_range(0,1,0,2,meta,[3],0.,0.)
                self.assertFalse(complete_snapshot(path,task))
                j.put_range(1,1,0,2,meta,[3],0.,0.)
                self.assertTrue(complete_snapshot(path,task))
                with self.assertRaisesRegex(ValueError,'ownership'):
                    complete_snapshot(path,dict(group_start=0,group_end=3))
            finally:j.close()

    def test_atomic_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'nested'/'receipt.json'
            for value in ({'version':1},{'version':2}):
                atomic_json(path,value)
                self.assertEqual(json.loads(path.read_text()),value)
            self.assertFalse(path.with_suffix('.json.tmp').exists())

if __name__=='__main__':unittest.main()
