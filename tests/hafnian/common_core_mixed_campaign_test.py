#!/usr/bin/env python3
"""Synthetic exact counts exercise mixed ownership, queue recovery and CRT.

No synthetic residue is a mathematical grid result. Real-worker bounded
checks are separate; these tests cover the entire scheduling/reduction flow.
"""
import argparse
from contextlib import redirect_stdout
import copy
import io
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import common_core_campaign as cc
import common_core_manifest as cm
import common_core_mixed_campaign as mix

META = [dict(domain=8,bounds=[65,25],primes=[3,1]), dict(domain=16,bounds=[108],primes=[4])]
VALUES = [(1 << 80)+123, (1 << 61)+55, 17]


class Catalog:
    slack=2
    count=3
    digest='catalog'
    rows=[(0,3,4),((1 << 60)|7,5,3),((1 << 60)|13,7,1)]
    def __getitem__(self,q):return self.rows[q]


class Plan:
    digest='plan'
    def groups(self,start=0,end=None):
        yield from [(0,1,62,[(1,14),(2,28)]),(1,0,0,[(0,0)])][start:end]
    def audit(self):
        return dict(groups=2,queries=3,singletons=1,coefficient_sum=15,
                    catalog_sha256='catalog',plan_sha256='plan')


def meta(catalog,parent,boundary,members):
    return copy.deepcopy(META[1 if len(members)==1 else 0])


class Worker:
    made=[]
    fail_after=None
    def __init__(self,binary,reference):
        self.process=argparse.Namespace(stdin=io.StringIO(),stdout=io.StringIO())
        self.calls=[];self.runs=0;Worker.made.append(self)
    def close(self):pass
    def request(self,command):
        self.calls.append(command)
        if command.startswith('prepare'):
            self.which=1 if command.startswith('prepare_single') else 0
            m=META[self.which]
            return ['prepared',str(m['domain']),str(len(m['bounds']))]+[
                str(x) for pair in zip(m['bounds'],m['primes']) for x in pair]
        _,pi,begin,count,chunk=command.split();pi=int(pi);self.runs+=1
        if self.fail_after is not None and self.runs > self.fail_after:
            raise RuntimeError('simulated worker death')
        members=[0] if self.which else [1,2]
        active=[q for q in members if Catalog()[q][2] > pi]
        values=[]
        for q in active:
            key=Catalog()[q][0];unmatched=4-((key&cc.FULL).bit_count()-2*(key>>60))
            values.append(VALUES[q]*math.factorial(unmatched)*int(count)%cc.PRIMES[pi])
        return ['result','0',str(len(values)),*map(str,values)]


class MixedTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.binary=self.root/'worker';self.binary.write_bytes(b'test-only')
        tasks=mix.make_tasks([2.,100.],{1:16},3,6);queues=cm.assign(tasks,3)
        self.payload=dict(format=mix.FORMAT,backend='cpu-reference',sources=mix.sources(),
                          configuration=cc.CONFIG,primes=list(cc.PRIMES),audit=Plan().audit(),
                          solver_binary=cc.file_digest(self.binary),tasks=tasks,workers=queues,total_seconds=102.)
        self.args=argparse.Namespace(worker=self.binary,worker_id=0,journal_dir=self.root/'journals',
                                    chunk_terms=4,checkpoint_terms=3,max_checkpoints=0,
                                    commit_ranges=2,commit_seconds=100.)
        Worker.made=[];Worker.fail_after=None
        self.patch_meta=patch.object(cc,'expected_meta',side_effect=meta);self.patch_meta.start()
        self.patch_worker=patch.object(cc,'Worker',Worker);self.patch_worker.start()
    def tearDown(self):
        self.patch_worker.stop();self.patch_meta.stop();self.temp.cleanup()
    def path(self,tid):return self.args.journal_dir/cc.digest(self.payload)/self.payload['tasks'][tid]['journal']
    def run_queue(self,i,limit=0):
        self.args.worker_id=i;self.args.max_checkpoints=limit
        with redirect_stdout(io.StringIO()):
            mix.run_queue(self.args,self.payload,Catalog(),Plan(),Plan().audit())
    def solve(self):
        for i in range(3):self.run_queue(i)
        return [self.path(i) for i in range(len(self.payload['tasks']))]
    def reduce(self,paths,require=False):
        args=argparse.Namespace(journals=paths,query_results=False,require_complete=require)
        with redirect_stdout(io.StringIO()):
            return mix.reduce_campaign(args,self.payload,Catalog(),Plan(),Plan().audit())

    def test_partition_is_exact_and_balanced(self):
        mix.validate(self.payload);mix.bind(self.payload,Catalog(),Plan(),Plan().audit())
        self.assertEqual(len(self.payload['tasks']),4)
        self.assertEqual(sum(t['end']-t['begin'] for t in self.payload['tasks'] if t['kind']=='signs'),16)
        self.assertLess(max(q['seconds'] for q in self.payload['workers']),40.)
        self.assertAlmostEqual(sum(t['seconds'] for t in self.payload['tasks']),102.)

    def test_random_partition_coverage_and_cost(self):
        rng=random.Random(499)
        for _ in range(100):
            costs=[rng.uniform(.01,50.) for _ in range(rng.randrange(1,100))]
            domains={i:1 << rng.randrange(1,33) for i in range(len(costs)) if rng.random()<.3}
            tasks=mix.make_tasks(costs,domains,8,32);queues=cm.assign(tasks,8)
            p=dict(format=mix.FORMAT,backend='cuda',audit=dict(groups=len(costs)),tasks=tasks,
                   workers=queues,total_seconds=sum(costs));mix.validate(p)
            self.assertAlmostEqual(sum(t['seconds'] for t in tasks),sum(costs))
            for t in tasks:
                if t['kind']=='signs':self.assertIn(t['group_start'],domains)
        one=mix.make_tasks([100.],{0:1},8,64)
        self.assertEqual(one[0]['kind'],'groups')

    def test_overlap_gap_domain_queue_corruption_rejected(self):
        for mutation in ('whole_overlap','range_overlap','range_gap','range_end','domain','queue','path'):
            p=copy.deepcopy(self.payload)
            if mutation=='whole_overlap':p['tasks'][2].update(kind='groups')
            elif mutation=='range_overlap':p['tasks'][2]['begin']-=1
            elif mutation=='range_gap':p['tasks'][2]['begin']+=1
            elif mutation=='range_end':p['tasks'][-1]['end']-=1
            elif mutation=='domain':p['tasks'][2]['domain']=32
            elif mutation=='queue':p['workers'][1]['task_ids'].append(p['workers'][0]['task_ids'][0])
            else:p['tasks'][0]['journal']='../bad'
            with self.assertRaises(ValueError,msg=mutation):mix.validate(p)

    def test_cannot_split_shared_group_or_wrong_domain(self):
        p=copy.deepcopy(self.payload)
        for t in p['tasks']:
            if t['kind']=='signs':t['domain']*=2;t['begin']*=2;t['end']*=2
        mix.validate(p)
        with self.assertRaisesRegex(ValueError,'domain differs'):mix.bind(p,Catalog(),Plan(),Plan().audit())
        p=copy.deepcopy(self.payload)
        p['tasks'][0].update(kind='signs',begin=0,end=8,domain=8)
        mix.validate(p)
        with self.assertRaisesRegex(ValueError,'independent'):mix.bind(p,Catalog(),Plan(),Plan().audit())

    def test_complete_reduction_coefficients_once_and_idempotence(self):
        paths=self.solve();summary=self.reduce(list(reversed(paths)),True)
        expected=math.factorial(28)*sum(co*(1 << (28-(key>>60)))*value
                       for (key,co,primes),value in zip(Catalog.rows,VALUES))
        self.assertEqual(summary['partial_labelled_count'],str(expected))
        self.assertEqual(summary['complete_queries'],3)
        self.assertEqual(summary['status'],'complete')
        before=[cc.file_digest(p) for p in paths]
        self.solve()
        self.assertEqual([cc.file_digest(p) for p in paths],before)
        self.assertTrue(all(not w.calls for w in Worker.made[-3:]))
        self.assertEqual(len(Worker.made),6)  # one process per queue, not per task

    def test_partial_prime_completes_only_eligible_shared_children(self):
        paths=self.solve()
        # Remove prime 2 of the shared group: its one-prime child stays complete.
        j=cc.Journal(paths[0],mix.identity(self.payload,0))
        with j.db:j.db.execute('DELETE FROM ranges WHERE pi=2')
        j.close();result=self.reduce(paths)
        self.assertEqual(result['complete_queries'],2)
        self.assertEqual(result['missing_shared_queries'],1)
        with self.assertRaisesRegex(ValueError,'incomplete'):self.reduce(paths,True)

    def test_parity_with_existing_whole_group_reducer(self):
        paths=self.solve();mixed=self.reduce(paths,True)
        old=self.root/'legacy.sqlite'
        ident=dict(format=cc.FORMAT,plan=Plan.digest,catalog=Catalog.digest,configuration=cc.CONFIG,
                   primes=list(cc.PRIMES),backend='cpu-reference',solver_binary='synthetic-fixture',
                   controller='synthetic-fixture',group_start=0,group_end=2)
        journal=cc.Journal(old,ident)
        try:
            for path in paths:
                reader=cc.Journal(path)
                try:
                    for gid,m,images in reader.ordered_groups():
                        journal.put_group(gid,m)
                        for pi,image in enumerate(images):
                            for a,b,values in image:journal.put_range(gid,pi,a,b,m,values,0.,0.)
                finally:reader.close()
        finally:journal.close()
        args=argparse.Namespace(journals=[old],cpu_reference=True,query_results=False,require_complete=True)
        with redirect_stdout(io.StringIO()) as output:cc.reduce(args,Catalog(),Plan(),Plan().audit())
        legacy=json.loads(output.getvalue())
        for key in ('status','complete_queries','missing_shared_queries','pending_independent_queries','partial_labelled_count'):
            self.assertEqual(mixed[key],legacy[key])

    def test_missing_range_task_and_duplicate_snapshot(self):
        paths=self.solve();partial=self.reduce(paths[:-1])
        self.assertEqual(partial['complete_queries'],2)
        self.assertEqual(partial['pending_independent_queries'],1)
        with self.assertRaisesRegex(ValueError,'duplicate task'):self.reduce(paths+[paths[-1]])

    def test_stop_and_resume_across_queues(self):
        for i in range(3):self.run_queue(i,limit=1)
        paths=list((self.args.journal_dir/cc.digest(self.payload)).glob('*.sqlite'))
        self.assertEqual(self.reduce(paths)['status'],'partial')
        paths=self.solve();self.assertEqual(self.reduce(paths,True)['complete_queries'],3)

    def test_worker_death_keeps_committed_rows_and_resumes(self):
        Worker.fail_after=3
        with self.assertRaisesRegex(RuntimeError,'worker death'):self.run_queue(0)
        tid=self.payload['workers'][0]['task_ids'][0]
        j=cc.Journal(self.path(tid))
        self.assertEqual(j.db.execute('SELECT count(*) FROM ranges').fetchone()[0],2)
        j.close();Worker.fail_after=None
        self.assertEqual(self.reduce(self.solve(),True)['complete_queries'],3)

    def test_valid_checksum_outside_interval_rejected(self):
        self.path(1).parent.mkdir(parents=True)
        j=cc.Journal(self.path(1),mix.identity(self.payload,1));j.put_group(1,META[1])
        j.put_range(1,0,0,6,META[1],[0],0.,0.);j.close()
        with self.assertRaisesRegex(ValueError,'outside sign'):self.reduce([self.path(1)])

    def test_foreign_manifest_and_corrupt_residue_rejected(self):
        paths=self.solve();self.payload['solver_binary']='different'
        with self.assertRaisesRegex(ValueError,'provenance'):self.reduce(paths)
        self.payload['solver_binary']=cc.file_digest(self.binary)
        j=cc.Journal(paths[0],mix.identity(self.payload,0))
        with j.db:j.db.execute("UPDATE ranges SET hash='wrong'")
        j.close()
        with self.assertRaisesRegex(ValueError,'checksum'):self.reduce(paths)

    def test_changed_binary_rejected_before_spawn(self):
        self.binary.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'binary changed'):self.run_queue(0)
        self.assertEqual(Worker.made,[])

    def test_snapshot_restore_and_monotonicity(self):
        self.run_queue(0,limit=1)
        tid=self.payload['workers'][0]['task_ids'][0];source=self.path(tid)
        dest=self.root/'published.sqlite'
        check=lambda path:mix.check_snapshot(path,self.payload,Catalog(),Plan())
        r=mix.snapshot_io.snapshot(source,dest,check)
        self.assertEqual(r['ranges'],1)
        self.assertEqual(cc.file_digest(dest),r['sha256'])
        self.run_queue(0)
        completed=mix.snapshot_io.snapshot(source,dest,check)
        self.assertGreater(completed['ranges'],1)
        with self.assertRaisesRegex(ValueError,'timeout'):
            mix.snapshot_io.snapshot(source,dest,check,timeout=1e-12)
        self.assertEqual(cc.file_digest(dest),completed['sha256'])
        j=cc.Journal(source,mix.identity(self.payload,tid))
        with j.db:j.db.execute('DELETE FROM ranges WHERE pi=0')
        j.close()
        with self.assertRaisesRegex(ValueError,'discard/change'):
            mix.snapshot_io.snapshot(source,dest,check)
        self.assertEqual(cc.file_digest(dest),completed['sha256'])
        with cc.claim(dest):
            with self.assertRaises(BlockingIOError):mix.snapshot_io.snapshot(source,dest,check)

    def test_manifest_checksum_and_source_binding(self):
        p=copy.deepcopy(self.payload);path=self.root/'manifest.json'
        path.write_text(cc.encoded(dict(payload=p,sha256=cc.digest(p))))
        self.assertEqual(mix.load(path),p)
        p['sources']['mixed_runner']='changed'
        path.write_text(cc.encoded(dict(payload=p,sha256=cc.digest(p))))
        with self.assertRaisesRegex(ValueError,'controller/configuration'):mix.load(path)
        path.write_text(cc.encoded(dict(payload=p,sha256='wrong')))
        with self.assertRaisesRegex(ValueError,'checksum'):mix.load(path)


if __name__=='__main__':unittest.main()
