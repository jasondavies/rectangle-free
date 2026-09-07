#!/usr/bin/env python3
"""CPU-only sign ownership, durable restart, provenance and CRT regression."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import common_core_campaign as cc
import common_core_sign_shards as ss


class Catalog:
    slack = 2
    def __getitem__(self, q):
        return ((1 << 60) | 7, 1, 4)  # three unmatched vertices


class FakeWorker:
    def __init__(self, meta, value):
        self.meta, self.value, self.calls = meta, value, []

    def request(self, message):
        self.calls.append(message)
        if message.startswith('prepare_single'):
            return ['prepared', str(self.meta['domain']), '1', str(self.meta['bounds'][0]), '4']
        _, pi, begin, count, chunk = message.split()
        value = int(count) * self.value * 6 % cc.PRIMES[int(pi)]
        return ['result', '0', '1', str(value)]


class SignShards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.payload = dict(format=ss.FORMAT, meta=dict(domain=16, bounds=[100], primes=[4]),
                            group=5, query=9, backend='cpu-reference', solver_binary='test',
                            tasks=[dict(id=i, begin=a, end=b) for i, (a, b) in enumerate(ss.split_domain(16, 3))])
        self.value = (1 << 80) + 123

    def tearDown(self):
        self.temp.cleanup()

    def complete(self, tid, stop=0):
        path = self.root / f'{tid}.sqlite'
        journal = cc.Journal(path, ss.identity(self.payload, tid))
        worker = FakeWorker(self.payload['meta'], self.value)
        try:
            with journal.batch(2, 100):
                finished = ss.solve(self.payload, tid, Catalog(), worker, journal, 4, 3, stop)
        finally:
            journal.close()
        return path, worker, finished

    def test_irregular_partition_and_large_unsigned_boundary(self):
        self.assertEqual(ss.split_domain(16, 3), [[0, 5], [5, 10], [10, 16]])
        ss.validate(self.payload)
        for domain in (1, 16, 1 << 31, 1 << 32):
            ranges = ss.split_domain(domain, min(domain, 7))
            self.assertEqual(sum(b-a for a, b in ranges), domain)
            self.assertEqual(ranges[-1][1], domain)
            for before, after in zip(ranges, ranges[1:]):
                self.assertEqual(before[1], after[0])
        for parts in (0, -1, 17, True):
            with self.assertRaises(ValueError):
                ss.split_domain(16, parts)

    def test_manifest_gap_overlap_incomplete_rejected(self):
        for change in ('gap', 'overlap', 'end', 'prime'):
            payload = copy.deepcopy(self.payload)
            if change in ('gap', 'overlap'):
                payload['tasks'][1]['begin'] += 1 if change == 'gap' else -1
            elif change == 'end':
                payload['tasks'][-1]['end'] -= 1
            else:
                payload['meta']['primes'] = [0]
            with self.assertRaises(ValueError):
                ss.validate(payload)

    def test_exact_crt_shuffled_tasks_and_restart_no_work(self):
        paths = [self.complete(i)[0] for i in (2, 0, 1)]
        records = ss.collect(self.payload, paths)
        self.assertEqual(cc.reduce_group(self.payload['meta'], records, [Catalog()[9][0]], 2), [self.value])
        _, worker, finished = self.complete(0)
        self.assertTrue(finished)
        self.assertEqual(worker.calls, [])

    def test_bounded_restart_and_missing_task(self):
        path, first, finished = self.complete(1, stop=1)
        self.assertFalse(finished)
        records = ss.collect(self.payload, [path])
        self.assertEqual(cc.reduce_group(self.payload['meta'], records, [Catalog()[9][0]], 2), [None])
        _, next_worker, finished = self.complete(1)
        self.assertTrue(finished)
        self.assertNotIn(first.calls[-1], next_worker.calls)

    def test_missing_prime_never_certified(self):
        paths = [self.complete(i)[0] for i in range(3)]
        journal = cc.Journal(paths[2], ss.identity(self.payload, 2))
        with journal.db:
            journal.db.execute('DELETE FROM ranges WHERE pi=3')
        journal.close()
        records = ss.collect(self.payload, paths)
        self.assertEqual(cc.reduce_group(self.payload['meta'], records, [Catalog()[9][0]], 2), [None])

    def test_duplicate_snapshot_and_wrong_manifest_rejected(self):
        path = self.complete(0)[0]
        with self.assertRaisesRegex(ValueError, 'duplicate task'):
            ss.collect(self.payload, [path, path])
        payload = copy.deepcopy(self.payload)
        payload['solver_binary'] = 'different'
        with self.assertRaisesRegex(ValueError, 'provenance'):
            ss.collect(payload, [path])

    def test_valid_checksum_outside_ownership_rejected(self):
        path = self.root / 'bad.sqlite'
        journal = cc.Journal(path, ss.identity(self.payload, 1))
        journal.put_group(5, self.payload['meta'])
        journal.put_range(5, 0, 0, 1, self.payload['meta'], [0], 0., 0.)
        journal.close()
        with self.assertRaisesRegex(ValueError, 'outside task'):
            ss.collect(self.payload, [path])

    def test_corrupt_payload_rejected(self):
        path = self.complete(0)[0]
        journal = cc.Journal(path, ss.identity(self.payload, 0))
        with journal.db:
            journal.db.execute("UPDATE ranges SET hash='corrupt'")
        journal.close()
        with self.assertRaisesRegex(ValueError, 'checksum'):
            ss.collect(self.payload, [path])

    def test_exception_rolls_back_uncommitted_range(self):
        path = self.root / 'rollback.sqlite'
        journal = cc.Journal(path, ss.identity(self.payload, 0))
        with self.assertRaises(RuntimeError):
            with journal.batch(32, 100):
                journal.put_group(5, self.payload['meta'])
                journal.put_range(5, 0, 0, 3, self.payload['meta'], [0], 0., 0.)
                raise RuntimeError('interrupted')
        self.assertEqual(ss.checked_records(journal, self.payload, 0), [[], [], [], []])
        journal.close()

    def test_resume_rejects_another_task(self):
        path = self.complete(0)[0]
        journal = cc.Journal(path)
        try:
            with self.assertRaisesRegex(ValueError, 'provenance'):
                ss.checked_records(journal, self.payload, 1)
        finally:
            journal.close()

    @unittest.skipUnless(os.environ.get('HAFNIAN_TEST_WORKER'), 'set HAFNIAN_TEST_WORKER for real finite-field checks')
    def test_real_order64_worker_boundaries_all_primes(self):
        worker = cc.Worker(Path(os.environ['HAFNIAN_TEST_WORKER']), True)
        try:
            self.assertEqual(worker.request('prepare_single 2 0'), ['prepared', str(1 << 31), '1', '108', '4'])
            checks = 0
            for pi, prime in enumerate(cc.PRIMES):
                # Odd divisions, prime changes, high sign bits and domain end.
                for begin in (0, (1 << 28) - 3, (1 << 30) - 3, (1 << 31) - 7):
                    whole = int(worker.request(f'run {pi} {begin} 7 4')[3])
                    left = int(worker.request(f'run {pi} {begin} 3 2')[3])
                    right = int(worker.request(f'run {pi} {begin+3} 4 2')[3])
                    self.assertEqual(whole, (left + right) % prime)
                    checks += 1
            self.assertEqual(checks, 16)
        finally:
            worker.close()
            worker.process.stdin.close()
            worker.process.stdout.close()


if __name__ == '__main__':
    unittest.main()
