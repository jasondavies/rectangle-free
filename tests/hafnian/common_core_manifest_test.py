#!/usr/bin/env python3
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
import common_core_manifest as m


class ManifestTest(unittest.TestCase):
    def cell(self, core, pool, count, groups, kernel, overhead, journal):
        return dict(core=core, pool=pool, schedule=[count, count, 0, 0], groups=groups,
                    kernel_hours=kernel, overhead_hours=overhead, journal_hours=journal)

    def model(self):
        old = [self.cell(36,11,2,10,1,.2,.1), self.cell(52,0,1,6,6,.2,.1),
               self.cell(54,0,1,4,4,.3,.2)]
        new = [self.cell(44,11,3,2,.03,.01,.005), self.cell(46,11,3,1,.02,.01,.005),
               self.cell(54,0,1,1,.1,.01,.005)]
        return {m.signature(c):c for c in old}, {m.signature(c):c for c in new}

    def test_model_keeps_old_overhead(self):
        old,new=self.model()
        costs,pop=m.cost_model(old,new,[52,54],.1)
        self.assertAlmostEqual(sum(costs[k]*n for k,n in pop.items())/3600,1.76)

    def test_missing_replacement_rejected(self):
        old,new=self.model()
        new.pop(next(iter(new)))
        with self.assertRaisesRegex(ValueError,'populations'):
            m.cost_model(old,new,[52,54],.1)

    def test_invalid_ratio_rejected(self):
        for value in [0,-1,1.1,float('nan')]:
            with self.assertRaises(ValueError):
                m.cost_model(*self.model(),[52,54],value)

    def payload(self, costs, workers=8, shards=64):
        tasks=m.partition(costs,shards)
        queues=m.assign(tasks,workers)
        return dict(format=m.FORMAT,tasks=tasks,workers=queues,
                    total_seconds=sum(costs),audit=dict(groups=len(costs)))

    def test_partition_and_assignment(self):
        rng=random.Random(490)
        values=[rng.uniform(.01,3) for _ in range(1000)]+[120,80,240]
        p=self.payload(values)
        m.validate(p)
        self.assertEqual(p,self.payload(values))
        for task in p['tasks']:
            self.assertAlmostEqual(task['seconds'],sum(values[task['group_start']:task['group_end']]))
        loads=[w['seconds'] for w in p['workers']]
        self.assertLessEqual(max(loads)-min(loads),max(t['seconds'] for t in p['tasks']))
        for w in p['workers']:
            ordered=[p['tasks'][i]['seconds'] for i in w['task_ids']]
            self.assertEqual(ordered,sorted(ordered,reverse=True))

    def test_oversized_group_indivisible(self):
        p=self.payload([1]*20+[100]+[1]*20,workers=4,shards=16)
        m.validate(p)
        heavy=next(t for t in p['tasks'] if t['group_start']<=20<t['group_end'])
        self.assertEqual((heavy['group_start'],heavy['group_end']),(20,21))

    def test_small_population(self):
        m.validate(self.payload([1],8,64))

    def test_reject_bad_ownership(self):
        p=self.payload([1]*100,2,4)
        for field,value in [('group_start',1),('group_end',0),('journal','../bad'),('seconds',float('nan'))]:
            bad=copy.deepcopy(p)
            bad['tasks'][0][field]=value
            with self.assertRaises(ValueError):m.validate(bad)
        bad=copy.deepcopy(p)
        bad['workers'][1]['task_ids'].append(bad['workers'][0]['task_ids'][0])
        with self.assertRaises(ValueError):m.validate(bad)
        bad=copy.deepcopy(p)
        bad['workers'][0]['task_ids'].pop()
        with self.assertRaises(ValueError):m.validate(bad)

    def test_projection_identity(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            c=self.cell(52,0,1,6,1,.2,.1)
            p=dict(format='common-core-steady-sample-v1',catalog='abc',queries=6,bins=[c],selected_orders=[52])
            s=dict(payload=p,sha256=m.cc.digest(p))
            q=dict(kind='projection',sample=s['sha256'],selected_orders=[52],bins=[c])
            sp,qp=d/'sample',d/'projection'
            sp.write_text(json.dumps(s));qp.write_text(json.dumps(q))
            catalog=argparse.Namespace(digest='abc',count=6)
            m.load_projection(sp,qp,catalog)
            q['bins'][0]['groups']=7
            qp.write_text(json.dumps(q))
            with self.assertRaisesRegex(ValueError,'census'):m.load_projection(sp,qp,catalog)
            q['sample']='wrong'
            qp.write_text(json.dumps(q))
            with self.assertRaisesRegex(ValueError,'mismatch'):m.load_projection(sp,qp,catalog)

    def test_commands_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            p=self.payload([1]*100,2,4)
            p.update(controller_sha256=m.cc.file_digest(m.cc.__file__), configuration=m.cc.CONFIG,
                     primes=list(m.cc.PRIMES), catalog_path='catalog with spaces', plan_path='plan',
                     worker_sha256='not-the-worker-hash')
            manifest=d/'manifest.json'
            manifest.write_text(m.cc.encoded(dict(payload=p,sha256=m.cc.digest(p))))
            args=argparse.Namespace(manifest=manifest,worker_id=0,worker=d/'worker',journal_dir=d/'journals')
            output=io.StringIO()
            with contextlib.redirect_stdout(output):m.commands(args)
            text=output.getvalue()
            self.assertIn('set -euo pipefail',text)
            self.assertIn('common_core_manifest.py run',text)
            self.assertIn('--worker-id 0',text)
            self.assertEqual(text.count('common_core_manifest.py'),1)
            args.worker.write_bytes(b'not an executable')
            with self.assertRaisesRegex(ValueError,'solver binary changed'):m.verify(args)
            p['controller_sha256']='wrong'
            manifest.write_text(m.cc.encoded(dict(payload=p,sha256=m.cc.digest(p))))
            with self.assertRaisesRegex(ValueError,'controller'):m.commands(args)
            manifest.write_text(m.cc.encoded(dict(payload=p,sha256='wrong')))
            with self.assertRaisesRegex(ValueError,'checksum'):m.commands(args)


if __name__=='__main__':
    unittest.main()
