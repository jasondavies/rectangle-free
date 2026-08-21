import unittest

from research.probes import dense_colour_class_probe as probe


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


if __name__ == "__main__":
    unittest.main()
