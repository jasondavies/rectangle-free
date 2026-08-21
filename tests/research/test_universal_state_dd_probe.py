import unittest

from research.probes import universal_state_dd_probe as probe


class UniversalStateDecisionDiagramTest(unittest.TestCase):
    def test_small_totals_against_brute_force(self) -> None:
        for rows, columns in ((2, 3), (3, 2), (3, 3)):
            expected = probe.brute_t4(rows, columns)
            for order in ("lex", "reverse", "balanced"):
                for mode in ("bundled", "colour-major"):
                    with self.subTest(
                        rows=rows, columns=columns, order=order, mode=mode
                    ):
                        diagram, one_column, _ = probe.build_one_column(
                            rows,
                            order,
                            mode,
                            probe.DEFAULT_PRIME,
                            100_000,
                            1_000_000,
                        )
                        state = one_column
                        for _ in range(1, columns):
                            state = diagram.convolve(0, state, one_column)
                        self.assertEqual(diagram.total(state), expected)

    def test_one_column_total(self) -> None:
        diagram, state, _ = probe.build_one_column(
            5, "balanced", "bundled", probe.DEFAULT_PRIME,
            100_000, 1_000_000,
        )
        self.assertEqual(diagram.total(state), 4 ** 5)


if __name__ == "__main__":
    unittest.main()
