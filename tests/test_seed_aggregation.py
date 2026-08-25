from __future__ import annotations

import unittest

from nashrs.seed_aggregation import aggregate_seed_runs


class SeedAggregationTest(unittest.TestCase):
    def test_hierarchical_aggregation_is_deterministic(self):
        runs = [
            {
                "run_seed": 1,
                "prompts": 2,
                "reward_delta": 0.1,
                "direct_win_rate": 0.6,
                "candidate_average_win_rate": 0.6,
                "candidate_worst_case_win_rate": 0.6,
                "candidate_exploitability": 0.0,
                "candidate_mean_tokens": 10.0,
                "candidate_truncation_rate": 0.0,
                "win_probabilities": [0.55, 0.65],
                "reward_deltas": [0.05, 0.15],
            },
            {
                "run_seed": 2,
                "prompts": 2,
                "reward_delta": 0.2,
                "direct_win_rate": 0.7,
                "candidate_average_win_rate": 0.7,
                "candidate_worst_case_win_rate": 0.7,
                "candidate_exploitability": 0.0,
                "candidate_mean_tokens": 12.0,
                "candidate_truncation_rate": 0.5,
                "win_probabilities": [0.65, 0.75],
                "reward_deltas": [0.15, 0.25],
            },
        ]
        first = aggregate_seed_runs(runs, bootstrap_samples=100, seed=9)
        second = aggregate_seed_runs(runs, bootstrap_samples=100, seed=9)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["across_seed"]["direct_win_rate"]["mean"], 0.65)
        self.assertEqual(first["seeds_with_win_rate_above_half"], 2)

    def test_rejects_duplicate_seeds(self):
        run = {
            "run_seed": 1,
            "prompts": 1,
            "reward_delta": 0.0,
            "direct_win_rate": 0.5,
            "candidate_average_win_rate": 0.5,
            "candidate_worst_case_win_rate": 0.5,
            "candidate_exploitability": 0.0,
            "candidate_mean_tokens": 1.0,
            "candidate_truncation_rate": 0.0,
            "win_probabilities": [0.5],
            "reward_deltas": [0.0],
        }
        with self.assertRaises(ValueError):
            aggregate_seed_runs([run, dict(run)], bootstrap_samples=10)


if __name__ == "__main__":
    unittest.main()
