from __future__ import annotations

import math
import random
import unittest

from nashrs.geometric_resampling import (
    effective_sample_size,
    geometric_importance_weights,
    resample_indices,
)


class GeometricResamplingTest(unittest.TestCase):
    def test_identical_policies_give_uniform_weights(self) -> None:
        weights = geometric_importance_weights([-1.0, -2.0], [-1.0, -2.0], 0.25)
        self.assertEqual(weights, [0.5, 0.5])
        self.assertAlmostEqual(effective_sample_size(weights), 2.0)

    def test_matches_geometric_target_over_mixture_proposal(self) -> None:
        current = [math.log(0.8), math.log(0.2)]
        reference = [math.log(0.4), math.log(0.6)]
        weights = geometric_importance_weights(current, reference, 0.5)
        expected = []
        for p, q in zip((0.8, 0.2), (0.4, 0.6)):
            expected.append(math.sqrt(p * q) / (0.5 * p + 0.5 * q))
        total = sum(expected)
        self.assertAlmostEqual(weights[0], expected[0] / total)
        self.assertAlmostEqual(weights[1], expected[1] / total)

    def test_resampling_is_reproducible(self) -> None:
        first = resample_indices([0.2, 0.8], 10, random.Random(7))
        second = resample_indices([0.2, 0.8], 10, random.Random(7))
        self.assertEqual(first, second)

    def test_rejects_arithmetic_errors(self) -> None:
        with self.assertRaises(ValueError):
            geometric_importance_weights([], [], 0.5)
        with self.assertRaises(ValueError):
            geometric_importance_weights([0.0], [0.0], 1.1)


if __name__ == "__main__":
    unittest.main()
