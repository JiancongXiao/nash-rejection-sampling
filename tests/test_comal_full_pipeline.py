import argparse
import json
from pathlib import Path
import tempfile
import time
import unittest

from examples.main.comal_full_pipeline import (
    prepare_splits,
    read_jsonl,
    record_iteration,
    split_pairs,
)


class ComalFullPipelineTest(unittest.TestCase):
    def test_prompt_and_pair_splits_preserve_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompts = root / "prompts.jsonl"
            prompts.write_text(
                "".join(json.dumps({"prompt": f"p{index}"}) + "\n" for index in range(12))
            )
            split_root = root / "splits"
            prepare_splits(
                argparse.Namespace(prompts=prompts, count=12, splits=6, output=split_root)
            )
            self.assertEqual(
                sum(len(read_jsonl(split_root / f"train_{index}.jsonl")) for index in range(6)),
                12,
            )

            pairs = root / "pairs.jsonl"
            pairs.write_text(
                "".join(json.dumps({"prompt": f"p{index}"}) + "\n" for index in range(10))
            )
            pair_root = root / "pair_split"
            split_pairs(
                argparse.Namespace(input=pairs, output=pair_root, validation_size=3)
            )
            self.assertEqual(len(read_jsonl(pair_root / "train.jsonl")), 9)
            self.assertEqual(len(read_jsonl(pair_root / "test.jsonl")), 1)

    def test_iteration_record_is_contiguous_and_accounted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "train.log").write_text("learning rate: 0.0000005000\nloss: 0.25\n")
            (root / "generation.json").write_text(
                json.dumps({"generated_tokens": 123, "num_samples": 5})
            )
            (root / "preference.json").write_text(
                json.dumps({"preference_model_calls": 20, "preference_components": 2})
            )
            metrics = root / "metrics.jsonl"
            record_iteration(
                argparse.Namespace(
                    iteration=0,
                    metrics=metrics,
                    training_log=root / "train.log",
                    generation_summary=root / "generation.json",
                    preference_summary=root / "preference.json",
                    started_at=time.time() - 1,
                    num_gpus=8,
                    learning_rate=5e-7,
                )
            )
            record = read_jsonl(metrics)[0]
            self.assertEqual(record["step"], 1)
            self.assertEqual(record["generated_tokens"], 123)
            self.assertEqual(record["preference_model_calls"], 20)
            self.assertGreater(record["gpu_hours"], 0)


if __name__ == "__main__":
    unittest.main()
