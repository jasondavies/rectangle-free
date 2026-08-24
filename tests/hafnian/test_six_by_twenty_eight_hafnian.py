from __future__ import annotations

import hashlib
import math
import tempfile
import unittest
from pathlib import Path

from tools.reduce_six_by_twenty_eight_hafnian import (
    CATALOG_SHA256,
    COMMON_FACTOR,
    FORMAT,
    ALGORITHM,
    PRIMES,
    QUOTIENT_BOUND,
    crt,
    read_result,
    required_prime_count,
)


def result_text() -> str:
    fields = [
        ("format", FORMAT), ("algorithm", ALGORITHM), ("rows", 6),
        ("columns", 28), ("catalog_sha256", CATALOG_SHA256),
        ("query_id", 0), ("query_sha256", "a" * 64),
        ("occupied_tokens", 0), ("defect_count", 0), ("excess", 0),
        ("unmatched_tokens", 4), ("defect_coefficient", 1),
        ("matching_bound_power", 101), ("vertices", 64),
        ("matrix_stride", 65), ("solver_binary_sha256", "b" * 64),
        ("prime", PRIMES[0]), ("begin", 0), ("end", 16),
        ("total_terms", 1 << 31), ("partial_glynn_sum", 123),
        ("status", "complete"),
    ]
    payload = "".join(f"{key} {value}\n" for key, value in fields)
    return payload + f"result_payload_sha256 {hashlib.sha256(payload.encode()).hexdigest()}\n"


class TwentyEightReductionTests(unittest.TestCase):
    def test_factored_bound_and_prime_counts(self):
        self.assertEqual(COMMON_FACTOR, math.factorial(28) * (1 << 24))
        self.assertEqual(QUOTIENT_BOUND.bit_length(), 113)
        self.assertEqual(required_prime_count(62), 3)
        self.assertEqual(required_prime_count(101), 4)

    def test_crt(self):
        target = 123456789012345678901234567890
        value, modulus = crt([(prime, target % prime) for prime in PRIMES])
        self.assertGreater(modulus, target)
        self.assertEqual(value, target)

    def test_payload_authentication(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result"
            path.write_text(result_text())
            self.assertEqual(read_result(path)["partial_glynn_sum"], "123")
            path.write_text(path.read_text().replace("partial_glynn_sum 123", "partial_glynn_sum 124"))
            with self.assertRaisesRegex(ValueError, "payload digest mismatch"):
                read_result(path)


if __name__ == "__main__":
    unittest.main()
