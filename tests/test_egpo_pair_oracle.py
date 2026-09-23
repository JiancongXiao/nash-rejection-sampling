import unittest

from nashrs.egpo_pair_oracle import oriented_probability


class EGPOPairOracleTest(unittest.TestCase):
    def test_probability_is_reoriented_after_swap(self):
        self.assertAlmostEqual(oriented_probability([0.8, 0.2], False), 0.8)
        self.assertAlmostEqual(oriented_probability([0.8, 0.2], True), 0.2)

    def test_two_classes_are_required(self):
        with self.assertRaises(ValueError):
            oriented_probability([1.0], False)


if __name__ == "__main__":
    unittest.main()
