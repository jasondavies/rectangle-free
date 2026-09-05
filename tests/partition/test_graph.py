import hashlib
import os
import subprocess
import unittest


class GraphTests(unittest.TestCase):
    def binary(self):
        return os.environ.get('PARTITION_GRAPH_TEST_BINARY', 'build/partition_graph_test')

    def test_cache_branches_and_sha(self):
        result = subprocess.run([self.binary()], capture_output=True, text=True, check=True)
        vectors = 0
        for line in result.stdout.splitlines():
            if line.startswith('SHA '):
                _, count, digest = line.split()
                self.assertEqual(digest, hashlib.sha256(bytes(range(int(count)))).hexdigest())
                vectors += 1
        self.assertEqual(vectors, 140)
        self.assertIn('PARTITION_GRAPH_TEST exact=OK', result.stdout)

    def test_validation_build_rejects_coefficient_overflow(self):
        result = subprocess.run([self.binary(), '--overflow'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('signed integer overflow', result.stderr)
