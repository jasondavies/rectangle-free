#!/usr/bin/env python3
"""Exactness, bounded-failure and orbit-reader tests for the CPU-only probe."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

BINARY = Path(sys.argv.pop(1) if len(sys.argv) > 1 else 'build/shared_column_response_probe').resolve()


class SharedColumnTest(unittest.TestCase):
    def run_probe(self, *args, ok=True):
        result = subprocess.run([str(BINARY), *map(str, args)], capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(result.returncode == 0, ok, result.stderr)
        return result.stdout

    def test_exhaustive_counts_and_quotient(self):
        self.assertIn('exact=OK', self.run_probe('--self-test'))

    def test_bounded_failure_is_not_a_count(self):
        output = self.run_probe('family', 3, 2, 63, 'all', 1, 1, 1)
        results = [json.loads(line) for line in output.splitlines()]
        for result in results:
            if 'method' in result:
                self.assertNotEqual(result['status'], 'complete')
                self.assertIsNone(result['nonzero_answers'])
        self.assertFalse(any('parity' in r for r in results))

    def test_reader_and_family_row_gauge(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.orbits'
            keys = []
            for last in (3, 5):
                keys.append(sum((r | (((last >> r) & 1) << 7)) << (8*r)
                                for r in range(8)))
            payload = struct.pack('<8sIQ', b'R8SQT01\0', 8, len(keys))
            payload += b''.join(struct.pack('<QQ', key, 1) for key in keys)
            path.write_bytes(payload)
            records = [json.loads(s) for s in self.run_probe('census', path, 100).splitlines()]
            self.assertEqual(records[0]['records'], 2)
            self.assertEqual(records[0]['raw_families'], 1)
            self.assertEqual(records[0]['row_gauge_families'], 1)
            self.assertEqual(records[0]['demand_histogram'], {'2': 1})
            self.assertEqual(records[1]['extensions'], [3, 5])
            path.write_bytes(payload[:-1])
            self.run_probe('census', path, 100, ok=False)
            path.write_bytes(b'BADMAGIC' + payload[8:])
            self.run_probe('census', path, 100, ok=False)

    def test_geometry_and_resource_guards(self):
        for rows, n, key, cache in ((9, 7, 0, 1), (3, 0, 0, 1),
                                    (3, 2, 64, 1), (3, 2, 0, 1000001)):
            self.run_probe('family', rows, n, key, 'all', cache, 1000, 1, ok=False)


if __name__ == '__main__':
    unittest.main()
