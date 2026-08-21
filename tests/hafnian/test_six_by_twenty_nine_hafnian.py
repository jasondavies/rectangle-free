import hashlib
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.reduce_six_by_twenty_nine_hafnian import (
    ALGORITHM,
    CATALOG_SHA256,
    FORMAT,
    crt,
    read_result,
)


COEFFICIENTS = [
    1, 480, 360, 1440,
    4320, 38880, 17280, 51840, 25920, 25920, 540, 2160, 38880,
    1080, 8640, 4320, 4320, 25920, 360, 12960, 8640, 8640, 4320,
    4320, 4320, 2880, 1440, 4320, 1440,
]
PRIMES = (
    2147483647, 2147483629, 2147483587, 2147483579,
    2147483563, 2147483549, 2147483543, 2147483497,
    2147483489,
)


def query_metadata(query: int) -> tuple[int, int, int, int]:
    if query == 0:
        return 0, 0, 2, 62
    if query <= 2:
        return 1, 1, 1, 58
    if query == 3:
        return 1, 2, 0, 56
    return 2, 2, 0, 54


def result_text(prime: int, query: int, signed_sum: int) -> str:
    defects, excess, unmatched, vertices = query_metadata(query)
    terms = 1 << (vertices // 2 - 1)
    fields = [
        ("format", FORMAT),
        ("algorithm", ALGORITHM),
        ("rows", 6),
        ("columns", 29),
        ("catalog_sha256", CATALOG_SHA256),
        ("query_id", query),
        ("query_sha256", hashlib.sha256(f"query-{query}".encode()).hexdigest()),
        ("occupied_tokens", query),
        ("defect_count", defects),
        ("excess", excess),
        ("unmatched_tokens", unmatched),
        ("defect_coefficient", COEFFICIENTS[query]),
        ("vertices", vertices),
        ("solver_binary_sha256", "a" * 64),
        ("prime", prime),
        ("begin", 0),
        ("end", terms),
        ("total_terms", terms),
        ("partial_glynn_sum", signed_sum),
        ("status", "complete"),
    ]
    payload = "".join(f"{key} {value}\n" for key, value in fields)
    return payload + f"result_payload_sha256 {hashlib.sha256(payload.encode()).hexdigest()}\n"


class TwentyNineReductionTests(unittest.TestCase):
    def test_sector_fixture(self):
        self.assertEqual(len(COEFFICIENTS), 29)
        self.assertEqual(sum(COEFFICIENTS[4:]), 303660)

    def test_payload_authentication(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result"
            path.write_text(result_text(PRIMES[0], 0, 123))
            self.assertEqual(read_result(path)["partial_glynn_sum"], "123")
            path.write_text(path.read_text().replace("partial_glynn_sum 123", "partial_glynn_sum 124"))
            with self.assertRaisesRegex(ValueError, "payload digest mismatch"):
                read_result(path)

    def test_crt(self):
        target = 123456789012345678901234567890
        value, modulus = crt([(prime, target % prime) for prime in PRIMES[:4]])
        self.assertGreater(modulus, target)
        self.assertEqual(value, target)

    def test_end_to_end_reducer(self):
        packing = 0
        for query, coefficient in enumerate(COEFFICIENTS):
            defects, _, _, _ = query_metadata(query)
            packing += coefficient * (1 << (29 - defects)) * (query + 2)
        expected = math.factorial(29) * packing
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for prime in PRIMES:
                for query in range(29):
                    _, _, unmatched, vertices = query_metadata(query)
                    terms = 1 << (vertices // 2 - 1)
                    matching_count = query + 2
                    augmented = matching_count * math.factorial(unmatched) % prime
                    signed_sum = augmented * terms % prime
                    path = Path(directory) / f"p{prime}-q{query}.result"
                    path.write_text(result_text(prime, query, signed_sum))
                    paths.append(path)
            reducer = (
                Path(__file__).parents[2]
                / "tools"
                / "reduce_six_by_twenty_nine_hafnian.py"
            )
            completed = subprocess.run(
                [sys.executable, str(reducer), *map(str, paths)],
                check=True, capture_output=True, text=True,
            )
            self.assertIn(
                f"HAFNIAN_6X29_RESULT T_4(6,29)={expected} exact=OK",
                completed.stdout,
            )


if __name__ == "__main__":
    unittest.main()
