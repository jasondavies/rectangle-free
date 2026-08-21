import unittest

import numpy as np

from research.probes import universal_state_tensor_rank_probe as probe


class UniversalStateTensorRankTest(unittest.TestCase):
    def test_modular_rank(self) -> None:
        prime = 1_000_003
        self.assertEqual(probe.modular_rank(np.zeros((3, 4), dtype=np.int64), prime), 0)
        self.assertEqual(probe.modular_rank(np.eye(4, dtype=np.int64), prime), 4)
        dependent = np.array([[1, 2, 3], [2, 4, 6], [0, 1, 1]], dtype=np.int64)
        self.assertEqual(probe.modular_rank(dependent, prime), 2)

    def test_small_exact_cut_ranks(self) -> None:
        diagram, one_column, _ = probe.dd.build_one_column(
            3, "balanced", "bundled", probe.dd.DEFAULT_PRIME,
            100_000, 1_000_000,
        )
        cube = diagram.convolve(
            0, diagram.convolve(0, one_column, one_column), one_column
        )
        levels = probe.reachable_by_level(diagram, cube)
        for level, nodes in enumerate(levels):
            record = probe.cut_rank_record(
                diagram, nodes, level, 10_000, 10_000
            )
            self.assertEqual(record["status"], "exact")
            self.assertLessEqual(record["rank"], len(nodes))


if __name__ == "__main__":
    unittest.main()
