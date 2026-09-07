#!/usr/bin/env python3
"""CPU-only indexing, streaming verification, CRT and persistent-queue tests."""
import argparse
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import random
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
import common_core_campaign as cc
import common_core_manifest as cm


def artifacts(directory):
    # Synthetic 29-query catalog for index/ownership tests, not a defect census.
    key = 0
    images = cc.prime_count(cc.bound_power(key, 1))
    data = b'HCCAT001' + struct.pack('<QQ', 1, 29) + struct.pack('<QQB',key,1,images)*29
    cd = hashlib.sha256(data).hexdigest()
    cp, pp = directory/'catalog', directory/'plan'
    cp.write_bytes(data+cd.encode())
    data = b'HCPLAN01'+struct.pack('<QQQQ',1,11,0,29)+cd.encode()
    for gid in range(29):
        data += struct.pack('<QQQQQ',0,0,1,gid,0)
    data += struct.pack('<QQ',(1<<64)-1,29)
    pp.write_bytes(data+hashlib.sha256(data).hexdigest().encode())
    return cp,pp


class IOTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_index_and_audit_cache(self):
        cp,pp=artifacts(self.root)
        c=cc.Catalog(cp);p=cc.Plan(pp,c)
        try:
            with self.assertRaisesRegex(ValueError,'completed audit'):
                list(p.groups(20,22))
            expected=list(p.groups())
            audit=p.audit()
            with patch.object(p,'u64',wraps=p.u64) as reads:
                self.assertEqual(list(p.groups(25,27)),expected[25:27])
                self.assertLess(reads.call_count,20)
            with patch.object(p,'groups',side_effect=AssertionError('rescanned')):
                self.assertEqual(p.audit(),audit)
            audit['groups']=0
            self.assertEqual(p.audit()['groups'],29)
            self.assertEqual(list(p.groups(29,29)),[])
            with self.assertRaises(ValueError):list(p.groups(27,30))
            replacement=self.root/'replacement'
            replacement.write_bytes(pp.read_bytes())
            replacement.replace(pp)
            with self.assertRaisesRegex(ValueError,'changed'):p.audit()
        finally:p.close();c.close()

    def test_fresh_process_never_trusts_cached_audit(self):
        cp,pp=artifacts(self.root)
        c=cc.Catalog(cp);p=cc.Plan(pp,c);p.audit();p.close()
        q=cc.Plan(pp,c)
        try:
            self.assertIsNone(q._offsets)
            self.assertIsNone(q._audit)
            q.audit()
            self.assertEqual(len(q._offsets),29)
        finally:q.close();c.close()

    def journal(self, name):
        identity=dict(format='io-test-only',group_start=0,group_end=20)
        j=cc.Journal(self.root/name,identity)
        meta=dict(domain=8,bounds=[65,25],primes=[3,1])
        with j.batch(100,100):
            for gid in (0,2,10):
                j.put_group(gid,meta)
                for pi in range(3):
                    values=[1,2] if pi==0 else [3]
                    j.put_range(gid,pi,0,3,meta,values,0.,0.)
                    j.put_range(gid,pi,3,8,meta,values,0.,0.)
            j.put_group(11,meta)  # valid metadata without completed work
        return j

    def test_ordered_payloads_equal_point_reads(self):
        j=self.journal('stream.sqlite')
        try:
            wanted=[]
            for gid in (0,2,10,11):
                meta=j.group(gid)
                wanted.append((gid,meta,[list(j.ranges(gid,pi,meta)) for pi in range(4)]))
            queries=[]
            j.db.set_trace_callback(queries.append)
            self.assertEqual(list(j.ordered_groups()),wanted)
            self.assertEqual(len(queries),2)
        finally:j.close()

    def test_stream_rejects_bad_checksum_and_unrequested_prime(self):
        j=self.journal('bad.sqlite')
        try:
            meta=j.group(0)
            j.put_range(0,3,0,3,meta,[],0.,0.)
            with self.assertRaisesRegex(ValueError,'unrequested'):list(j.ordered_groups())
            j.db.execute('DELETE FROM ranges WHERE pi=3');j.db.commit()
            j.db.execute("UPDATE ranges SET hash='bad' WHERE gid=2 AND pi=0 AND begin=0");j.db.commit()
            with self.assertRaisesRegex(ValueError,'checksum'):list(j.ordered_groups())
        finally:j.close()

    def test_cached_arithmetic(self):
        rng=random.Random(491)
        for count in range(1,5):
            for _ in range(50):
                residues=[rng.randrange(p) for p in cc.PRIMES[:count]]
                x,mod=0,1
                for r,p in zip(residues,cc.PRIMES):
                    x += mod*((r-x)*pow(mod,-1,p)%p);mod*=p
                self.assertEqual(cc.crt(residues),(x,mod))
        cc.normalization_inverse.cache_clear()
        for _ in range(2):
            for n in (0,10,32):
                for u in range(7):
                    for pi,p in enumerate(cc.PRIMES):
                        self.assertEqual(cc.normalization_inverse(1<<n,u,pi),pow((1<<n)*math.factorial(u),-1,p))
        self.assertEqual(cc.normalization_inverse.cache_info().hits,84)

    def test_merge_partial_journals_and_reject_overlap(self):
        cp,pp=artifacts(self.root)
        c=cc.Catalog(cp);p=cc.Plan(pp,c);audit=p.audit()
        try:
            meta=cc.expected_meta(c,0,0,[(0,0)])
            paths=[]
            identity=dict(format=cc.FORMAT,catalog=c.digest,plan=p.digest,configuration=cc.CONFIG,
                primes=list(cc.PRIMES),backend='cpu-reference',controller='test',solver_binary='test',
                group_start=0,group_end=29)
            for i in range(2):
                path=self.root/f'partial-{i}.sqlite';paths.append(path)
                j=cc.Journal(path,identity)
                try:
                    j.put_group(0,meta)
                    for pi in range(max(meta['primes'])):
                        j.put_range(0,pi,i*(meta['domain']//2),(i+1)*(meta['domain']//2),meta,[0],0.,0.)
                finally:j.close()
            options=argparse.Namespace(journals=paths,cpu_reference=True,query_results=False,require_complete=False)
            with contextlib.redirect_stdout(io.StringIO()) as output:cc.reduce(options,c,p,audit)
            result=json.loads(output.getvalue())
            self.assertEqual((result['complete_queries'],result['pending_independent_queries']),(1,28))
            options.journals=[paths[0],paths[0]]
            with self.assertRaisesRegex(ValueError,'overlapping'):cc.reduce(options,c,p,audit)
        finally:p.close();c.close()

    def test_queue_one_worker_and_bounded_resume(self):
        cp,pp=artifacts(self.root)
        c=cc.Catalog(cp);p=cc.Plan(pp,c);audit=p.audit();p.close();c.close()
        binary=self.root/'fake-worker';binary.write_bytes(b'fake worker for protocol tests only')
        tasks=[dict(id=0,group_start=0,group_end=10,seconds=1.),dict(id=1,group_start=10,group_end=29,seconds=1.)]
        workers=cm.assign(tasks,1)
        payload=dict(format=cm.FORMAT,tasks=tasks,workers=workers,total_seconds=2.,audit=audit,
                     catalog_path=str(cp),plan_path=str(pp),controller_sha256=cc.file_digest(cc.__file__),
                     worker_sha256=cc.file_digest(binary),configuration=cc.CONFIG,primes=list(cc.PRIMES))
        manifest=self.root/'manifest.json'
        manifest.write_text(cc.encoded(dict(payload=payload,sha256=cc.digest(payload))))
        instances=[]
        class Worker:
            def __init__(self, path, cpu):
                self.binary_hash=cc.file_digest(path);self.backend='cpu-reference';self.closed=False
                instances.append(self)
            def request(self,line):
                a=line.split()
                if a[0]=='prepare_single':
                    self.count=cc.prime_count(cc.bound_power(int(a[2]),1))
                    return ['prepared',str(1<<30),'1',str(cc.bound_power(int(a[2]),1)),str(self.count)]
                return ['result','0','1','0']
            def close(self):self.closed=True
        args=argparse.Namespace(manifest=manifest,worker_id=0,worker=binary,journal_dir=self.root/'journals',
             cpu_reference=True,chunk_terms=32,checkpoint_terms=1<<30,commit_ranges=32,commit_seconds=1.,max_checkpoints=1)
        with patch.object(cc,'Worker',Worker),contextlib.redirect_stdout(io.StringIO()) as output:
            cm.run_queue(args)
        self.assertIn('queue_checkpoint_limit_reached',output.getvalue())
        self.assertNotIn('"task":1',output.getvalue())
        args.max_checkpoints=0
        with patch.object(cc,'Worker',Worker),contextlib.redirect_stdout(io.StringIO()) as output:
            cm.run_queue(args)
        self.assertIn('queue_complete',output.getvalue())
        self.assertEqual(len(instances),2)  # one per process-equivalent, not per task
        self.assertTrue(all(w.closed for w in instances))
        journals=list((args.journal_dir/cc.digest(payload)).glob('*.sqlite'))
        self.assertEqual(len(journals),2)
        c=cc.Catalog(cp);p=cc.Plan(pp,c)
        try:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                cc.reduce(argparse.Namespace(journals=journals,cpu_reference=True,query_results=False,
                          require_complete=True),c,p,p.audit())
            self.assertEqual(json.loads(output.getvalue())['complete_queries'],29)
        finally:p.close();c.close()


if __name__=='__main__':unittest.main()
