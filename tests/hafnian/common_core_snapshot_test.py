#!/usr/bin/env python3
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import common_core_campaign as cc
import common_core_snapshot as snap


class SnapshotTest(unittest.TestCase):
    def test_live_snapshot_and_failure_preserves_previous(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);source=d/'live.sqlite';dest=d/'snapshot.sqlite'
            identity=dict(format=cc.FORMAT,group_start=0,group_end=1)
            meta=dict(domain=1000,bounds=[10],primes=[1])
            j=cc.Journal(source,identity);j.put_group(0,meta);j.close()
            errors=[];ready=threading.Event()
            def write():
                try:
                    j=cc.Journal(source,identity)
                    try:
                        for i in range(60):
                            j.put_range(0,0,i,i+1,meta,[i],0.,0.)
                            ready.set();time.sleep(.002)
                    finally:j.close()
                except BaseException as e:errors.append(e)
            t=threading.Thread(target=write);t.start();ready.wait(5)
            r=snap.snapshot(source,dest)
            self.assertGreater(r['ranges'],0)
            t.join();self.assertEqual(errors,[])
            r=snap.snapshot(source,dest);self.assertEqual(r['ranges'],60)
            checksum=cc.file_digest(dest)
            self.assertEqual(snap.verify_snapshot(dest,checksum)['ranges'],60)
            with self.assertRaisesRegex(ValueError,'checksum'):
                snap.verify_snapshot(dest,'0'*64)
            j=cc.Journal(source,identity)
            with j.db:j.db.execute('DELETE FROM ranges WHERE begin=0')
            j.close()
            with self.assertRaisesRegex(ValueError,'discard/change'):snap.snapshot(source,dest)
            self.assertEqual(cc.file_digest(dest),checksum)
            self.assertEqual(list(d.glob('.snapshot.sqlite.*')),[])
            with self.assertRaises(ValueError):snap.snapshot(source,source)
            with cc.claim(dest):
                with self.assertRaises(BlockingIOError):snap.snapshot(source,dest)
            self.assertEqual(cc.file_digest(dest),checksum)
            with self.assertRaisesRegex(ValueError,'timeout'):snap.snapshot(source,dest,1e-12)
            self.assertEqual(cc.file_digest(dest),checksum)

    def test_bad_payload_not_published(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);source=d/'live.sqlite';dest=d/'snapshot.sqlite'
            j=cc.Journal(source,dict(format=cc.FORMAT,group_start=0,group_end=1))
            m=dict(domain=1,bounds=[1],primes=[1]);j.put_group(0,m);j.put_range(0,0,0,1,m,[0],0.,0.)
            with j.db:j.db.execute("UPDATE ranges SET hash='bad'")
            j.close()
            with self.assertRaisesRegex(ValueError,'checksum'):snap.snapshot(source,dest)
            self.assertFalse(dest.exists())


if __name__=='__main__':unittest.main()
