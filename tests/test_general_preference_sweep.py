from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "examples" / "sweep_general_preference_mixture.py"
SPEC = importlib.util.spec_from_file_location("sweep_general_preference_mixture", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class GeneralPreferenceSweepTest(unittest.TestCase):
    def test_evaluates_cached_component_scores(self):
        records = [
            {"component_scores": [[3.0, 0.0, -1.0], [0.0, 2.0, -1.0]]},
            {"component_scores": [[1.0, -1.0, 0.0], [-2.0, 0.0, 2.0]]},
        ]
        result = MODULE.evaluate_candidate(records, [0.5, 0.5], [1.0, 1.0])
        self.assertEqual(result["prompts"], 2)
        self.assertEqual(result["weights"], [0.5, 0.5])
        self.assertGreater(result["component_disagreement_rate"], 0.0)

    def test_rank_prioritizes_cycles_then_btl_residual(self):
        cyclic = {
            "cycle_rate": 0.1,
            "bt_probability_rmse": 0.01,
            "weights": [0.8, 0.2],
        }
        acyclic = {
            "cycle_rate": 0.0,
            "bt_probability_rmse": 0.2,
            "weights": [0.5, 0.5],
        }
        self.assertGreater(MODULE.rank_key(cyclic), MODULE.rank_key(acyclic))


if __name__ == "__main__":
    unittest.main()
