import unittest

from research.probes import dense_colour_class_probe as probe
from research.probes import dense_residual_completion_sample as residual_probe


class DenseColourClassProbeTest(unittest.TestCase):
    def test_dense_first_identity(self) -> None:
        self.assertEqual(probe.dense_first_count(2, 2), 252)
        self.assertEqual(probe.dense_first_count(2, 3), 3912)
        self.assertEqual(probe.dense_first_count(3, 3), 228984)

    def test_nine_by_nine_degree_gate(self) -> None:
        records = probe.degree_census(9, 30)
        self.assertEqual(records[0], {
            "first_edges": 21,
            "minimum_second_edges": 20,
            "one_side_degree_sequences": 198,
            "minimum_two_paths": 15,
            "maximum_two_paths": 36,
        })
        self.assertEqual(records[-1], {
            "first_edges": 30,
            "minimum_second_edges": 17,
            "one_side_degree_sequences": 1,
            "minimum_two_paths": 36,
            "maximum_two_paths": 36,
        })

    def test_sampled_residual_is_exactly_completable(self) -> None:
        first_rows = [
            0x181, 0x086, 0x118, 0x142, 0x124,
            0x0c8, 0x0b0, 0x02b, 0x055,
        ]
        second = residual_probe.sample_second(
            first_rows, 18, residual_probe.random.Random(392)
        )
        self.assertEqual(second.bit_count(), 18)
        first = sum(row << (index * 9) for index, row in enumerate(first_rows))
        self.assertFalse(first & second)
        residual = residual_probe.FULL ^ first ^ second
        satisfiable, _, _ = residual_probe.binary_completion_exists(residual)
        self.assertTrue(satisfiable)


if __name__ == "__main__":
    unittest.main()
