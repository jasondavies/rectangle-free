import hashlib
import itertools
import math
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from tools import reduce_six_by_thirty_optimized as reducer


ROOT = Path(__file__).resolve().parents[2]
MINOR = 1133887175503385561722350
ANSWER = 5813026373117572187494156438960699897545098374101961015296000000000


def result_text(prime, begin=0, end=1 << 28, **changes):
    item = reducer.CATALOG[0]
    fields = dict(format=reducer.FORMAT, algorithm=reducer.ALGORITHM,
                  rows=6, columns=30, catalog_sha256=reducer.CATALOG_SHA256,
                  query_id=0, query_sha256=item["digest"],
                  occupied_tokens=item["occupied"], defect_count=0, excess=0,
                  unmatched_tokens=0, defect_coefficient=18, vertices=58,
                  matching_bound_power=85, solver_binary_sha256="a" * 64,
                  prime=prime, begin=begin, end=end, total_terms=1 << 28,
                  matrix_stride=59, gray_enabled=1, gray_chain=7,
                  status="complete", partial_glynn_sum=MINOR*(end-begin) % prime)
    fields.update(changes)
    payload = "".join(f"{k} {v}\n" for k, v in fields.items())
    return payload + "result_payload_sha256 " + hashlib.sha256(payload.encode()).hexdigest() + "\n"


class EndpointMinorTests(unittest.TestCase):
    def test_bound_and_work(self):
        self.assertEqual(reducer.required_prime_count(85), 3)
        self.assertEqual(3 * reducer.CATALOG[0]["terms"], 805306368)
        self.assertEqual(18 * MINOR * (1 << 30) * math.factorial(30), ANSWER)

    def test_synthetic_shards_and_old_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for prime in reducer.PRIMES:
                for begin, end in ((0, 123), (123, 1 << 28)):
                    path = Path(directory) / f"p{prime}-b{begin}.result"
                    path.write_text(result_text(prime, begin, end))
                    paths.append(path)
            result = reducer.reduce_results(paths)
            self.assertEqual(int(result["T_4(6,30)"]), ANSWER)
            self.assertEqual(result["matching_counts"], {"0": str(MINOR)})
            self.assertEqual(reducer.reduce_results(paths[:-1], True)["status"], "PARTIAL")
            with self.assertRaisesRegex(ValueError, "gap/overlap"):
                reducer.reduce_results(paths + [paths[0]])
            path = Path(directory) / "bad.result"
            for changed in ({"format": "six-by-thirty-hafnian-v1"},
                            {"columns": 29}, {"defect_coefficient": 540},
                            {"matching_bound_power": 1}, {"gray_enabled": 0},
                            {"query_id": 3}, {"query_sha256": "a" * 64}):
                with self.subTest(changed=changed):
                    path.write_text(result_text(reducer.PRIMES[0], **changed))
                    with self.assertRaises(ValueError):
                        reducer.read_result(path)
            paths[0].write_text(result_text(reducer.PRIMES[0], 0, 123,
                                           solver_binary_sha256="b" * 64))
            with self.assertRaisesRegex(ValueError, "mixed solver"):
                reducer.reduce_results(paths)

    def test_laplace_identity_on_small_endpoint_graph(self):
        # Independent direct matching enumeration on K4 x KG(4,2), not a
        # hafnian/CRT implementation. A fixed vertex has three equivalent mates.
        pairs = list(itertools.combinations(range(4), 2))
        tokens = list(itertools.product(range(4), pairs))
        neighbours = [sum(1 << j for j, (d, f) in enumerate(tokens)
                          if c != d and not set(e).intersection(f))
                      for c, e in tokens]

        def pm(mask):
            if not mask:
                return 1
            bit = mask & -mask
            remaining = mask ^ bit
            candidates = neighbours[bit.bit_length()-1] & remaining
            count = 0
            while candidates:
                mate = candidates & -candidates
                candidates ^= mate
                count += pm(remaining ^ mate)
            return count

        full = (1 << len(tokens))-1
        minors = [pm(full ^ 1 ^ (1 << j)) for j in range(len(tokens))
                  if neighbours[0] & (1 << j)]
        self.assertEqual(len(minors), 3)
        self.assertEqual(len(set(minors)), 1)
        self.assertEqual(pm(full), 3 * minors[0])

    @unittest.skipUnless((ROOT / "build/six_by_thirty_optimized_cpu").exists()
                         and (ROOT / "build/six_by_twenty_nine_optimized_cpu").exists(),
                         "build independent minor CPU evaluators first")
    def test_cpu_ranges_match_saved_6x29_minor_identity(self):
        for prime in reducer.PRIMES:
            for begin, end in ((0, 16), (12345, 12473), ((1 << 28)-17, 1 << 28)):
                residues = []
                for binary, query in (("six_by_thirty_optimized_cpu", 0),
                                      ("six_by_twenty_nine_optimized_cpu", 3)):
                    output = subprocess.check_output(
                        [str(ROOT / "build" / binary), "--query", str(query),
                         "--prime", str(prime), "--begin", str(begin),
                         "--end", str(end), "--threads", "1"], text=True)
                    if query == 0:
                        self.assertIn(reducer.CATALOG_SHA256, output)
                        self.assertIn(reducer.CATALOG[0]["digest"], output)
                    residues.append(int(re.search(r" residue=(\d+) ", output)[1]))
                self.assertEqual(*residues)


if __name__ == "__main__":
    unittest.main()
