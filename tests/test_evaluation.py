from __future__ import annotations

import unittest

from nashrs import evaluate_pairwise


class FirstWinsOracle:
    def compare(self, prompts, left, right):
        del left, right
        return [0.8] * len(prompts)


class EvaluationTest(unittest.TestCase):
    def test_pairwise_metrics(self) -> None:
        result = evaluate_pairwise(
            ["p1", "p2"],
            {"nash-rs": ["a", "b"], "baseline": ["c", "d"]},
            FirstWinsOracle(),
        )
        self.assertEqual(result.preference_model_calls, 2)
        self.assertAlmostEqual(result.average_win_rate("nash-rs"), 0.8)
        self.assertAlmostEqual(result.worst_case_win_rate("nash-rs"), 0.8)
        self.assertAlmostEqual(result.empirical_exploitability("nash-rs"), 0.0)
        self.assertAlmostEqual(result.empirical_exploitability("baseline"), 0.3)


if __name__ == "__main__":
    unittest.main()

