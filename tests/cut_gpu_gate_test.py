#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/probes'))
import cut_gpu_gate as gpu
from cut_batch_gate import half


class CutGpuGateTest(unittest.TestCase):
    def fixture(self, root):
        config = dict(rounds=2, panels=[dict(id=0, records=1,
                                           labelled_weight=3, covered_weight=6)])
        files = ['solver', 'inputs/seed.orbits', 'inputs/p00-baseline.orbits',
                 'inputs/p00-adaptive.orbits']
        for name in files:
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
        config['files'] = {name:gpu.digest(root/name) for name in files}
        (root/'config.json').write_text(json.dumps(config))
        for variant in ('baseline', 'adaptive'):
            for prefix in ('check', 'r0', 'r1'):
                identifier = f'{prefix}-p00-{variant}'
                path = f'inputs/p00-{variant}.orbits'
                fields = dict(id=identifier, path=path, start=0, end=0,
                              filter_mod=0, filter_id=0, geometry='8x8',
                              token_plane_quotient=1, transpose_quotient=1,
                              solver_binary_sha256=config['files']['solver'],
                              solver_configuration_sha256='a'*64,
                              canonical_cache_sha256=config['files']['inputs/seed.orbits'],
                              orbit_corpus_sha256=config['files'][path], records=1,
                              kernels=1, labelled_weight=3, covered_weight=6,
                              contribution=123456, verified=2 if prefix=='check' else 0,
                              direct_comparisons=10, minimum_free_bytes=100,
                              total_seconds=2 if variant=='baseline' else 1,
                              gpu_seconds=.5, left_layout_seconds=.1,
                              right_layout_seconds=.1, canonical_resolve_seconds=.1,
                              load_seconds=.1, cache_factory_seconds=0,
                              canonical_upload_seconds=0, validation_seconds=0)
                target = root/('check-results' if prefix=='check' else 'results')/f'{identifier}.result'
                target.parent.mkdir(exist_ok=True)
                self.write_result(target, fields)
        return root/'results/r0-p00-adaptive.result'

    @staticmethod
    def write_result(path, fields):
        payload = ''.join(f'{k} {v}\n' for k,v in fields.items())
        path.write_text('RECT8X8_PREFIX_RESULT 3\n'+payload+'result_payload_sha256 '+
                        hashlib.sha256(payload.encode()).hexdigest()+'\n')

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
            p=self.fixture(Path(d))
            self.assertEqual(gpu.parse_result(p, 'r0-p00-adaptive',
                                             'inputs/p00-adaptive.orbits')['contribution'],123456)
            p.write_text(p.read_text().replace('123456','123457'))
            with self.assertRaises(ValueError):gpu.summarize(Path(d))

    def test_summary(self):
        with tempfile.TemporaryDirectory() as d:
            self.fixture(Path(d))
            result=gpu.summarize(Path(d))
            self.assertTrue(result['exact'])
            self.assertEqual(result['cpu_join_checks'],4)
            self.assertEqual(result['recurring_time_reduction_percent'],50)

    def test_prepare_single_panel_and_standalone_bundle(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'source'; panel=source/'owner-00'
            panel.mkdir(parents=True)
            (panel/'records.tsv').write_text('0 0 0 3\n')
            rows=[dict(type='layout',key=0,entries=1,bytes=64),
                  dict(type='record',source=0,index=0,key='0',weight='3',
                       choices=[dict(left=0,right=0,tiles=1,columns=15,transpose=0,reverse=0)]),
                  dict(type='complete',records=1,layouts=1)]
            (panel/'cost.warm.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (panel/'result.json').write_text(json.dumps(dict(
                input_sha256=gpu.digest(panel/'records.tsv'),adaptive=dict(choices=[0]))))
            binary=root/'fake-solver'; binary.write_bytes(b'not executed')
            bundle=root/'bundle'; gpu.prepare(source,bundle,binary)
            config=json.loads((bundle/'config.json').read_text())
            self.assertIn('gpu_result_v3.py',config['files'])
            self.assertEqual(len(config['panels']),1)
            # Isolated mode proves the remote gate does not need repository imports.
            run=subprocess.run([sys.executable,'-I','-c',
                'import runpy,sys; sys.path.insert(0,sys.argv[1]); '
                'runpy.run_path(sys.argv[1]+"/gate.py",run_name="bundle_test")',str(bundle)],
                cwd=root,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)

    def test_summary_rejects_invalid_fields(self):
        for field, value in [('start',1), ('end',1), ('filter_mod',2), ('filter_id',1),
                             ('transpose_quotient',0), ('token_plane_quotient',0),
                             ('solver_configuration_sha256','b'*64),
                             ('solver_binary_sha256','b'*64), ('contribution',0),
                             ('covered_weight',0), ('records',0),
                             ('total_seconds','nan'), ('gpu_seconds','inf'),
                             ('left_layout_seconds',-1), ('cache_factory_seconds',1)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as d:
                root=Path(d); p=self.fixture(root)
                fields=dict(line.split(maxsplit=1) for line in p.read_text().splitlines()[1:-1])
                fields[field]=value
                self.write_result(p,fields)
                with self.assertRaises(ValueError):gpu.summarize(root)

    def test_missing_result(self):
        with tempfile.TemporaryDirectory() as d:
            self.fixture(Path(d)).unlink()
            with self.assertRaises(FileNotFoundError):gpu.summarize(Path(d))

    def test_duplicate_field_and_missing_timing(self):
        for defect in ('duplicate','missing'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as d:
                root=Path(d); p=self.fixture(root)
                fields=dict(line.split(maxsplit=1) for line in p.read_text().splitlines()[1:-1])
                if defect=='missing':
                    del fields['gpu_seconds']; self.write_result(p,fields)
                else:
                    payload=''.join(f'{k} {v}\n' for k,v in fields.items())+'id duplicate\n'
                    p.write_text('RECT8X8_PREFIX_RESULT 3\n'+payload+'result_payload_sha256 '+
                                 hashlib.sha256(payload.encode()).hexdigest()+'\n')
                with self.assertRaises(ValueError):gpu.summarize(root)


if __name__=='__main__':unittest.main()
