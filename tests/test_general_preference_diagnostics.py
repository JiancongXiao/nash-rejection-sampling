import unittest

from nashrs.general_preference_diagnostics import (
    aggregate_diagnostics,
    diagnose_score_mixture,
)


class GeneralPreferenceDiagnosticsTest(unittest.TestCase):
    def test_identical_components_are_exactly_btl(self):
        result = diagnose_score_mixture(
            [[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            [1.0, 1.0],
            [1.0, 1.0],
        )
        self.assertEqual(result["component_disagreement_rate"], 0.0)
        self.assertEqual(result["cycle_rate"], 0.0)
        self.assertAlmostEqual(result["bt_logit_rmse"], 0.0)

    def test_conflicting_components_report_disagreement_and_non_btl_residual(self):
        result = diagnose_score_mixture(
            [[4.0, 1.0, 0.0, -2.0], [0.0, 3.0, -1.0, 2.0]],
            [0.6, 0.4],
            [1.0, 1.0],
        )
        self.assertGreater(result["component_disagreement_rate"], 0.0)
        self.assertGreater(result["bt_logit_rmse"], 0.0)

    def test_aggregate_averages_prompt_diagnostics(self):
        aggregate = aggregate_diagnostics([
            {
                "component_disagreement_rate": 0.2,
                "cycle_rate": 0.0,
                "bt_logit_rmse": 0.1,
                "bt_probability_rmse": 0.02,
            },
            {
                "component_disagreement_rate": 0.4,
                "cycle_rate": 0.2,
                "bt_logit_rmse": 0.3,
                "bt_probability_rmse": 0.04,
            },
        ])
        self.assertEqual(aggregate["prompts"], 2)
        self.assertAlmostEqual(aggregate["component_disagreement_rate"], 0.3)


if __name__ == "__main__":
    unittest.main()
