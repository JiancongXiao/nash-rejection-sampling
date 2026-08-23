from __future__ import annotations

import math
import random
import unittest

from nashrs.rejection import accept_gibbs_proposal, acceptance_probability


class RejectionTest(unittest.TestCase):
    def test_boundary_probabilities(self) -> None:
        self.assertEqual(acceptance_probability(0.0, 0.2), 1.0)
        self.assertAlmostEqual(acceptance_probability(1.0, 0.5), math.exp(-2.0))

    def test_correct_acceptance_direction_empirically(self) -> None:
        rng = random.Random(11)
        n = 100_000
        rate = sum(accept_gibbs_proposal(0.7, 0.5, rng) for _ in range(n)) / n
        self.assertAlmostEqual(rate, math.exp(-1.4), delta=0.006)

    def test_higher_g_is_less_likely(self) -> None:
        low_rng = random.Random(9)
        high_rng = random.Random(9)
        low = sum(accept_gibbs_proposal(0.2, 0.5, low_rng) for _ in range(20_000))
        high = sum(accept_gibbs_proposal(0.8, 0.5, high_rng) for _ in range(20_000))
        self.assertGreater(low, high)


if __name__ == "__main__":
    unittest.main()

