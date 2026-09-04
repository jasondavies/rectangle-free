import hashlib
import math
from pathlib import Path
import tempfile
import unittest

from tools import reduce_six_by_twenty_nine_optimized as reducer


def result_text(query, prime, begin=0, end=None, **changes):
    item = reducer.CATALOG[query]
    if end is None:
        end = item["terms"]
    fields = dict(format=reducer.FORMAT, algorithm=reducer.ALGORITHM,
                  rows=6, columns=29, catalog_sha256=reducer.CATALOG_SHA256,
                  query_id=query, query_sha256=item["digest"],
                  occupied_tokens=item["occupied"], defect_count=item["defects"],
                  excess=item["excess"], unmatched_tokens=item["unmatched"],
                  defect_coefficient=item["coefficient"], vertices=item["vertices"],
                  matching_bound_power=item["matching_bound_power"],
                  solver_binary_sha256="a"*64, prime=prime, begin=begin, end=end,
                  total_terms=item["terms"], matrix_stride=item["vertices"]+1,
                  gray_enabled=1, gray_chain=7, status="complete",
                  partial_glynn_sum=((query+100)*math.factorial(item["unmatched"])
                                    * (end-begin)) % prime)
    fields.update(changes)
    payload = "".join(f"{k} {v}\n" for k, v in fields.items())
    return payload + "result_payload_sha256 " + hashlib.sha256(payload.encode()).hexdigest()+"\n"


class OptimizedReductionTests(unittest.TestCase):
    def test_census(self):
        self.assertEqual(len(reducer.CATALOG), 33)
        self.assertEqual(sorted(x["coefficient"] for x in reducer.CATALOG[:5]),
                         [90, 180, 240, 540, 720])
        self.assertEqual(sum(x["coefficient"] for x in reducer.CATALOG[:5]), math.comb(60, 2))
        self.assertEqual(sum(3*x["terms"] for x in reducer.CATALOG), 11072962560)
        self.assertTrue(all(reducer.required_prime_count(x["matching_bound_power"]) == 3
                            for x in reducer.CATALOG))

    def test_exact_split_resume_and_reduction(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for item in reducer.CATALOG:
                for prime in reducer.PRIMES:
                    # Deliberately unaligned Gray boundaries must cover exactly.
                    for begin, end in ((0, 123), (123, item["terms"])):
                        path = Path(directory)/f"{item['id']}-{prime}-{begin}.result"
                        path.write_text(result_text(item["id"], prime, begin, end))
                        paths.append(path)
            result = reducer.reduce_results(paths)
            expected = math.factorial(29)*sum(x["coefficient"]*(1 << (29-x["defects"]))
                        * (x["id"]+100) for x in reducer.CATALOG)
            self.assertEqual(int(result["T_4(6,29)"]), expected)
            self.assertEqual(reducer.reduce_results(paths[:-1], True)["status"], "PARTIAL")
            with self.assertRaisesRegex(ValueError, "gap/overlap"):
                reducer.reduce_results(paths+[paths[0]])

    def test_rehashed_bad_metadata_rejected(self):
        changes = ({"matching_bound_power": 1}, {"defect_coefficient": 91},
                   {"query_sha256": "b"*64}, {"gray_chain": 0},
                   {"begin": 10, "end": 9}, {"prime": 17},
                   {"format": "six-by-twenty-nine-hafnian-v1"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"bad.result"
            for change in changes:
                with self.subTest(change=change):
                    # Avoid duplicate keyword arguments in the prime case.
                    prime = change.get("prime", reducer.PRIMES[0])
                    extra = {k: v for k, v in change.items() if k != "prime"}
                    path.write_text(result_text(0, prime, **extra))
                    with self.assertRaises(ValueError):
                        reducer.read_result(path)

    def test_dummy_minor_identity_on_small_graphs(self):
        # Independent recursive perfect-matchings oracle, not a hafnian formula.
        def pm(a, remaining):
            if not remaining:
                return 1
            v = min(remaining)
            return sum(pm(a, remaining-{v,w}) for w in remaining-{v} if a[v][w])
        for seed in range(12):
            n = 8
            a = [[0]*10 for _ in range(10)]
            for i in range(n):
                for j in range(i+1, n):
                    a[i][j] = a[j][i] = int(((i*13+j*7+seed) % 5) != 0)
            for i in range(n):
                for dummy in (8, 9):
                    a[i][dummy] = a[dummy][i] = 1
            minors = sum(pm(a, set(range(n))-{i,j})
                         for i in range(n) for j in range(i+1, n))
            self.assertEqual(pm(a, set(range(10))), 2*minors)


if __name__ == "__main__":
    unittest.main()
