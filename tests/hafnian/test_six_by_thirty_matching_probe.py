import math
import random
import unittest

from research.probes.six_by_thirty_matching_probe import (
    PairTokenGeometry,
    SymmetryMatchingCounter,
    colour_sector_count,
    ordered_column_dp,
)


class SixByThirtyMatchingProbeTests(unittest.TestCase):
    def test_target_graph(self):
        geometry = PairTokenGeometry(4, 6)
        self.assertEqual(geometry.pair_count, 15)
        self.assertEqual(geometry.vertices, 60)
        self.assertEqual(geometry.degree, 18)
        self.assertEqual(geometry.edges, 540)
        self.assertEqual(geometry.maximum_columns, 30)
        self.assertEqual(geometry.column_weight, 2)
        self.assertEqual(colour_sector_count(), 136)

    def test_two_colour_extremal_identity(self):
        geometry = PairTokenGeometry(2, 4)
        perfect_matchings = SymmetryMatchingCounter(geometry).solve()
        direct = ordered_column_dp(geometry, geometry.maximum_columns)
        reduced = (
            math.factorial(geometry.maximum_columns)
            * geometry.column_weight ** geometry.maximum_columns
            * perfect_matchings
        )
        self.assertEqual(perfect_matchings, 1)
        self.assertEqual(direct, 720)
        self.assertEqual(reduced, direct)

    def test_canonicalization_preserves_full_state(self):
        geometry = PairTokenGeometry(3, 5)
        counter = SymmetryMatchingCounter(geometry)
        full = geometry.initial_state()
        self.assertEqual(counter.canonical(full), full)

    def test_target_random_induced_subgraphs(self):
        geometry = PairTokenGeometry(4, 6)
        counter = SymmetryMatchingCounter(geometry)

        def plain(state):
            state = tuple(state)
            if not any(state):
                return 1
            colour = next(c for c, lane in enumerate(state) if lane)
            bit = state[colour] & -state[colour]
            pair = bit.bit_length() - 1
            result = 0
            for other_colour, lane in enumerate(state):
                if other_colour == colour:
                    continue
                candidates = lane & geometry.disjoint_masks[pair]
                while candidates:
                    other_bit = candidates & -candidates
                    child = list(state)
                    child[colour] ^= bit
                    child[other_colour] ^= other_bit
                    result += plain(child)
                    candidates ^= other_bit
            return result

        rng = random.Random(0x630)
        for vertices in (2, 4, 6, 8, 10):
            for _ in range(8):
                selected = rng.sample(range(60), vertices)
                lanes = [0] * 4
                for vertex in selected:
                    lanes[vertex // 15] |= 1 << (vertex % 15)
                self.assertEqual(counter.solve_state(lanes), plain(lanes))


if __name__ == "__main__":
    unittest.main()
