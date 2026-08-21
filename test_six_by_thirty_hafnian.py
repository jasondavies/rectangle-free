import hashlib
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from reduce_six_by_thirty_hafnian import (
    ALGORITHMS,
    FORMAT,
    GRAPH_SHA256,
    TOTAL_TERMS,
    crt,
    read_result,
)


def result_text(**overrides: object) -> str:
    fields: list[tuple[str, object]] = [
        ("format", FORMAT),
        ("algorithm", sorted(ALGORITHMS)[0]),
        ("rows", 6),
        ("colours", 4),
        ("vertices", 60),
        ("edges", 540),
        ("graph_sha256", GRAPH_SHA256),
        ("solver_binary_sha256", "a" * 64),
        ("prime", 2147483647),
        ("begin", 0),
        ("end", TOTAL_TERMS),
        ("total_terms", TOTAL_TERMS),
        ("partial_glynn_sum", 123),
        ("threads", 256),
        ("elapsed_seconds", "1.25"),
        ("status", "complete"),
    ]
    values = dict(fields)
    values.update(overrides)
    payload = "".join(f"{key} {values[key]}\n" for key, _ in fields)
    return payload + f"result_payload_sha256 {hashlib.sha256(payload.encode()).hexdigest()}\n"


class HafnianReductionTests(unittest.TestCase):
    def test_result_payload_and_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "valid.result"
            path.write_text(result_text())
            fields = read_result(path)
            self.assertEqual(int(fields["partial_glynn_sum"]), 123)

            path.write_text(result_text().replace("partial_glynn_sum 123", "partial_glynn_sum 124"))
            with self.assertRaisesRegex(ValueError, "payload digest mismatch"):
                read_result(path)

    def test_crt(self):
        target = 9876543210123456789
        primes = (2147483647, 2147483629, 2147483587)
        reconstructed, modulus = crt([(prime, target % prime) for prime in primes])
        self.assertGreater(modulus, target)
        self.assertEqual(reconstructed, target)

    def test_end_to_end_reducer(self):
        primes = (
            2147483647, 2147483629, 2147483587, 2147483579, 2147483563,
            2147483549, 2147483543, 2147483497, 2147483489, 2147483477,
        )
        perfect_matchings = 123456789
        expected = math.factorial(30) * (1 << 30) * perfect_matchings
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for prime in primes:
                path = Path(directory) / f"p{prime}.result"
                signed_sum = perfect_matchings * pow(2, 29, prime) % prime
                path.write_text(result_text(prime=prime, partial_glynn_sum=signed_sum))
                paths.append(path)
            reducer = Path(__file__).with_name("reduce_six_by_thirty_hafnian.py")
            completed = subprocess.run(
                [sys.executable, str(reducer), *map(str, paths)],
                check=True, capture_output=True, text=True,
            )
            self.assertIn(f"HAFNIAN_RESULT T_4(6,30)={expected} exact=OK", completed.stdout)


if __name__ == "__main__":
    unittest.main()
