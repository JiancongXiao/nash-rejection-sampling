from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "examples" / "evaluate_ppo_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("evaluate_ppo_checkpoints", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CheckpointEvaluationTest(unittest.TestCase):
    def test_discovers_checkpoints_in_step_order(self):
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            base = root / "base"
            checkpoints = root / "checkpoints"
            final = root / "final"
            for path in (
                base,
                checkpoints / "global_step16_hf",
                checkpoints / "global_step8_hf",
                final,
            ):
                path.mkdir(parents=True)
                (path / "config.json").write_text("{}")
            models = MODULE.discover_models(base, checkpoints, final)
            self.assertEqual([name for name, _ in models], ["base", "step_8", "step_16"])

    def test_loads_jsonl_prompts(self):
        with tempfile.TemporaryDirectory() as root_text:
            path = Path(root_text) / "prompts.jsonl"
            path.write_text(
                json.dumps({"prompt": "first"})
                + "\n"
                + json.dumps({"prompt": "second"})
                + "\n"
            )
            self.assertEqual(MODULE.load_prompts(path), ["first", "second"])


if __name__ == "__main__":
    unittest.main()
