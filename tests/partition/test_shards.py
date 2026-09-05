import contextlib
import hashlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

from tools.merge_poly import Poly, PolyFileMeta, parse_poly_file, write_poly_file, merge_shards

ROOT = Path(__file__).resolve().parents[2]


class Shards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'a.poly'
        self.meta = PolyFileMeta(4, 3, 0, 1, 21, version=2,
                                 algorithm='partition-structure-v2', solver_source='a'*64,
                                 task_space='b'*64, prefix_depth=2, reorder=1)
        self.poly = Poly([0, -42, 17])

    def put(self, name='a.poly', **kwargs):
        path = self.path.with_name(name)
        write_poly_file(path, self.poly, replace(self.meta, **kwargs))
        return path

    def reject(self, data):
        self.path.write_bytes(data)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_poly_file(self.path)

    def test_roundtrip_and_checksum(self):
        self.put()
        self.assertEqual(parse_poly_file(self.path), (self.poly, self.meta))
        data = self.path.read_bytes()
        prefix, tail = data.rsplit(b'sha256 ', 1)
        self.assertEqual(tail.splitlines()[0].decode(), hashlib.sha256(prefix).hexdigest())
        self.reject(data.replace(b'-42', b'-43'))

    def test_malformed_legacy(self):
        self.put(version=1)
        good = self.path.read_bytes()
        for data in (good.replace(b'end\n', b''), good+b'junk\n',
                     good.replace(b'coeff 1 -42\n', b''),
                     good.replace(b'coeff 1 -42\n', b'coeff 1 -42\ncoeff 1 2\n'),
                     good.replace(b'deg 2', b'deg 3'),
                     good.replace(b'rows 4', b'rows 4\nrows 4'),
                     good.replace(b'task_start 0', b'task_start -1'),
                     good.replace(b'coeff 1 -42', b'coeff 1 -4_2')):
            with self.subTest(data=data): self.reject(data)

    def test_v2_requires_checksum_and_exact_lines(self):
        self.put()
        good = self.path.read_bytes()
        self.reject(good.replace(b'\n', b'\r\n'))
        self.reject(good.split(b'sha256 ')[0] + b'end\n')

    def test_reject_identity_changes(self):
        a = self.put()
        for changes in ({'solver_source': 'c'*64}, {'task_space': 'c'*64},
                        {'reorder': 0}, {'prefix_depth': 3}, {'version': 1}):
            b = self.put('b.poly', task_start=1, task_end=2, **changes)
            with self.subTest(changes=changes), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                merge_shards([a,b], None, allow_legacy=True)

    def test_legacy_is_explicit(self):
        a = self.put(version=1)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            merge_shards([a], None)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            merge_shards([a], None, allow_legacy=True)

    def test_interval_merge(self):
        a, b = self.put(), self.put('b.poly', task_start=1, task_end=2)
        out = self.path.with_name('out.poly')
        with contextlib.redirect_stdout(io.StringIO()): merge_shards([b,a], out)
        p, m = parse_poly_file(out)
        self.assertEqual(p, self.poly.add(self.poly))
        self.assertEqual((m.task_start,m.task_end), (0,2))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            merge_shards([a,a], out)
        b = self.put('b.poly', task_start=2, task_end=3)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            merge_shards([a,b], out)

    def test_huge_task_space_uses_intervals(self):
        a = self.put(full_tasks=10**15)
        with contextlib.redirect_stdout(io.StringIO()): merge_shards([a], None)

    def test_failed_replace_preserves_file(self):
        self.put()
        original = self.path.read_bytes()
        with patch('tools.merge_poly.os.replace', side_effect=OSError('injected')):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                write_poly_file(self.path, Poly([123]), self.meta)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.glob('*.tmp.*')), [])

    def test_all_historical_files(self):
        for path in (ROOT/'poly').rglob('*.poly'): parse_poly_file(path)

    def test_actual_solver_shards(self):
        binary = ROOT/os.environ.get('PARTITION_TEST_BINARY', 'build/partition_poly_7')
        if not binary.exists(): self.skipTest('build partition_poly_7 first')
        def run(name, *args):
            path = self.path.with_name(name)
            subprocess.run([str(binary), '4', '3', '--poly-out', str(path), *args],
                           env={**os.environ, 'OMP_NUM_THREADS':'1'}, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return path
        full = run('full.poly')
        source = subprocess.check_output(
            ['python3', str(ROOT/'tools/partition_source_id.py')], text=True).strip()
        self.assertEqual(parse_poly_file(full)[1].solver_source, source)
        a = run('a.poly', '--task-end','1')
        b = run('b.poly','--task-start','1')
        out = self.path.with_name('merged.poly')
        with contextlib.redirect_stdout(io.StringIO()): merge_shards([a,b],out)
        self.assertEqual(parse_poly_file(out)[0],parse_poly_file(full)[0])
        other = run('other.poly','--no-reorder')
        self.assertEqual(parse_poly_file(other)[0],parse_poly_file(full)[0])
        self.assertNotEqual(parse_poly_file(other)[1].task_space,parse_poly_file(full)[1].task_space)
        if os.name == 'posix' and Path('/dev/full').exists():
            # An unwritable/nonexistent destination must fail, not report success.
            result = subprocess.run([str(binary),'4','3','--task-end','0','--poly-out',
                                     str(self.path.parent/'absent'/'out.poly')],
                                    env={**os.environ,'OMP_NUM_THREADS':'1'},
                                    stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            self.assertNotEqual(result.returncode,0)


if __name__ == '__main__': unittest.main()
