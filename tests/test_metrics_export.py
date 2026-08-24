import unittest

from nashrs.metrics_export import parse_step_lines


class MetricsExportTest(unittest.TestCase):
    def test_parses_steps_and_accumulates_costs(self):
        lines = [
            "noise",
            "(PPOTrainer pid=1) INFO x] ✨ Global step 1: "
            "{'reward': 0.4, 'kl': 0.01, 'nashrs/acceptance_rate': 0.5, "
            "'nashrs/preference_model_calls': 8.0, 'nashrs/gpu_hours': 0.1}",
            "(PPOTrainer pid=1) INFO x] ✨ Global step 2: "
            "{'reward': 0.6, 'kl': 0.02, 'nashrs/acceptance_rate': 0.75, "
            "'nashrs/preference_model_calls': 6.0, 'nashrs/gpu_hours': 0.2}",
        ]
        records = parse_step_lines(lines)
        self.assertEqual([record["step"] for record in records], [1, 2])
        self.assertEqual(records[1]["nashrs/cumulative_preference_model_calls"], 14.0)
        self.assertAlmostEqual(records[1]["nashrs/cumulative_gpu_hours"], 0.3)
        self.assertEqual(records[1]["reward"], 0.6)
        self.assertEqual(records[1]["nashrs/acceptance_rate"], 0.75)


if __name__ == "__main__":
    unittest.main()
