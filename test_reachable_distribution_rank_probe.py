import unittest
from math import comb, factorial

import reachable_distribution_rank_probe as probe


class ReachableRankTest(unittest.TestCase):
    def test_small_exact_ranks(self) -> None:
        self.assertEqual(probe.reachable_rank(2, 2, 3).rank, 3)
        self.assertEqual(probe.reachable_rank(3, 3, 3).rank, 25)
        self.assertEqual(probe.reachable_rank(4, 2, 3).rank, 78)
        self.assertEqual(probe.reachable_rank(4, 3, 1_000_003).rank, 304)

    def test_symmetry_forecast_dimensions(self) -> None:
        rows = 5
        results = probe.symmetry_forecast(rows, 3)
        first_rank = (1 << rows) - rows
        self.assertEqual(
            [item["dimension"] for item in results],
            [comb(first_rank + degree - 1, degree) for degree in range(1, 4)],
        )
        self.assertEqual(results[0]["maximum_shape"], (5,))

    def test_character_orthogonality(self) -> None:
        rows = 5
        shapes = list(probe.partitions(rows))
        order = factorial(rows)
        classes = [
            (cycles, order // probe.class_denominator(cycles))
            for cycles in shapes
        ]
        for left in shapes:
            for right in shapes:
                inner = sum(
                    size * probe.symmetric_group_character(left, cycles)
                    * probe.symmetric_group_character(right, cycles)
                    for cycles, size in classes
                )
                self.assertEqual(inner, order if left == right else 0)


if __name__ == "__main__":
    unittest.main()
