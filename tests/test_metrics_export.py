import unittest

from nashrs.metrics_export import parse_step_lines


class MetricsExportTest(unittest.TestCase):
    def test_parses_steps_and_accumulates_costs(self):
        lines = [
            "noise",
            "(PPOTrainer pid=1) INFO x] ✨ Global step 1: "
            "{'reward': 0.4, 'kl': 0.01, 'nashrs/acceptance_rate': 0.5, "
            "'nashrs/preference_model_calls': 8.0, 'nashrs/accepted': 2.0, "
            "'nashrs/acceptance_trials': 3.0, 'nashrs/proposals': 4.0, "
            "'nashrs/gpu_hours': 0.1}",
            "(PPOTrainer pid=1) INFO x] ✨ Global step 2: "
            "{'reward': 0.6, 'kl': 0.02, 'nashrs/acceptance_rate': 0.75, "
            "'nashrs/preference_model_calls': 6.0, 'nashrs/gpu_hours': 0.2}",
        ]
        records = parse_step_lines(lines, samples_per_step=8)
        self.assertEqual([record["step"] for record in records], [1, 2])
        self.assertEqual(records[0]["nashrs/step_total_preference_model_calls"], 64.0)
        self.assertEqual(
            records[1]["nashrs/cumulative_total_preference_model_calls"], 112.0
        )
        self.assertAlmostEqual(
            records[1]["nashrs/cumulative_sum_sample_gpu_hours"], 2.4
        )
        self.assertEqual(records[1]["reward"], 0.6)
        self.assertEqual(records[1]["nashrs/acceptance_rate"], 0.75)
        self.assertAlmostEqual(records[0]["nashrs/aggregate_acceptance_rate"], 2 / 3)
        self.assertEqual(records[0]["nashrs/aggregate_proposal_efficiency"], 0.5)

    def test_rejects_nonpositive_step_size(self):
        with self.assertRaises(ValueError):
            parse_step_lines([], samples_per_step=0)

    def test_exports_common_comparison_costs(self):
        lines = [
            "✨ Global step 1: "
            "{'comparison/preference_model_calls': 4.0, "
            "'comparison/generated_tokens': 10.0, "
            "'comparison/opponent_generations': 2.0, "
            "'comparison/gpu_hours': 0.25}"
        ]
        record = parse_step_lines(lines, samples_per_step=5)[0]
        self.assertEqual(
            record["experiment/cumulative_total_preference_model_calls"], 20.0
        )
        self.assertEqual(record["experiment/cumulative_total_generated_tokens"], 50.0)
        self.assertEqual(record["experiment/cumulative_total_opponent_generations"], 10.0)
        self.assertEqual(record["experiment/cumulative_sum_sample_gpu_hours"], 1.25)


if __name__ == "__main__":
    unittest.main()
