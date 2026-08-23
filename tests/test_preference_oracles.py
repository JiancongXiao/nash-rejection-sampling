import unittest

from nashrs.preference_oracles import BTLComponent, MixtureBTLPreferenceOracle


class LengthReward:
    def __init__(self, sign=1.0):
        self.sign = sign

    def score(self, prompts, responses):
        del prompts
        return [self.sign * len(response) for response in responses]


class PreferenceOracleTest(unittest.TestCase):
    def test_btl_is_antisymmetric(self):
        oracle = MixtureBTLPreferenceOracle([BTLComponent(LengthReward())])
        forward = oracle.compare(["p"], ["long"], ["x"])[0]
        reverse = oracle.compare(["p"], ["x"], ["long"])[0]
        self.assertAlmostEqual(forward + reverse, 1.0)
        self.assertGreater(forward, 0.5)

    def test_mixture_combines_component_probabilities(self):
        oracle = MixtureBTLPreferenceOracle(
            [
                BTLComponent(LengthReward(1.0), weight=3.0),
                BTLComponent(LengthReward(-1.0), weight=1.0),
            ]
        )
        probability = oracle.compare(["p"], ["long"], ["x"])[0]
        self.assertGreater(probability, 0.5)

    def test_rejects_invalid_components(self):
        with self.assertRaises(ValueError):
            MixtureBTLPreferenceOracle([])
        with self.assertRaises(ValueError):
            BTLComponent(LengthReward(), temperature=0.0)


if __name__ == "__main__":
    unittest.main()
