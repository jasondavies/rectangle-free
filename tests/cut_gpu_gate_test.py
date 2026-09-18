#!/usr/bin/env python3
import hashlib
from pathlib import Path
import random
import struct
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/probes'))
import cut_gpu_gate as gpu
from cut_batch_gate import half


class CutGpuGateTest(unittest.TestCase):
    def test_assemble(self):
        rng=random.Random(512)
        for _ in range(100):
            a,b=rng.getrandbits(32),rng.getrandbits(32)
            key=gpu.assemble(a,b)
            self.assertEqual((half(key,15),half(key,240)),(a,b))

    def test_orbit_encoding(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'probe'
            gpu.write_orbits(p,[(0,3),(0x123,7)])
            magic,width,n=struct.unpack('<8sIQ',p.read_bytes()[:20])
            self.assertEqual((magic,width,n),(b'R8SQT01\0',8,2))
            self.assertEqual(struct.unpack('<QQQQ',p.read_bytes()[20:]),(0,3,0x123,7))
            with self.assertRaises(FileExistsError):gpu.write_orbits(p,[])

    def test_result_checksum(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'r'
            payload='id x\ncontribution 123456\n'
            text='RECT8X8_PREFIX_RESULT 3\n'+payload+'result_payload_sha256 '+hashlib.sha256(payload.encode()).hexdigest()+'\n'
            p.write_text(text)
            self.assertEqual(gpu.parse_result(p)['contribution'],'123456')
            p.write_text(text.replace('123456','123457'))
            with self.assertRaises(ValueError):gpu.parse_result(p)


if __name__=='__main__':unittest.main()
