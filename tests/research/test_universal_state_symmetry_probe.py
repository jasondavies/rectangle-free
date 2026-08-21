import unittest

from research.probes import universal_state_symmetry_probe as probe


class UniversalStateSymmetryProbeTest(unittest.TestCase):
    def test_three_column_burnside_census(self) -> None:
        records = probe.burnside_census(4)
        self.assertEqual(
            [record["valid_unlabelled_row_sets"] for record in records],
            [1, 64, 1944, 37404, 513360],
        )
        self.assertEqual(
            [record["colour_column_orbits"] for record in records],
            [1, 3, 28, 317, 3826],
        )

    def test_exact_quotient_transfer(self) -> None:
        records = probe.quotient_transfer(3, 3, 1000, None)
        self.assertEqual(
            [record["orbit_states"] for record in records],
            [1, 3, 8, 18],
        )
        self.assertEqual(records[-1]["coefficient_sum"], 228984)

        four_rows = probe.quotient_transfer(4, 4, 10_000, None)
        self.assertEqual(
            [record["orbit_states"] for record in four_rows],
            [1, 5, 30, 190, 1182],
        )
        self.assertEqual(four_rows[-1]["coefficient_sum"], 2545607472)


if __name__ == "__main__":
    unittest.main()
