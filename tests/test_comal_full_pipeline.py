import argparse
import json
from pathlib import Path
import tempfile
import time
from types import ModuleType
import unittest

from examples.main.comal_get_logprobs_compat import (
    install_missing_tulu_symbols,
    limit_gpuids_to_dataset_rows,
    selected_model_type,
)
from examples.main.comal_full_pipeline import (
    prepare_splits,
    read_jsonl,
    record_iteration,
    split_pairs,
)


class ComalFullPipelineTest(unittest.TestCase):
    def test_single_gpu_job_preserves_effective_global_batch(self):
        root = Path(__file__).resolve().parents[1]
        job = (root / "cluster/hopper/main/main_3b_comal_full.pbs").read_text()
        submitter = (
            root / "cluster/hopper/main/submit_3b_comal_single.sh"
        ).read_text()
        accelerate = (
            root / "configs/accelerate_comal_single.yaml"
        ).read_text()

        self.assertIn('NUM_GPUS="${NASHRS_COMAL_NUM_GPUS:-2}"', job)
        self.assertIn(
            'ACCUMULATE_STEP="$((GLOBAL_BATCH_SIZE / (NUM_GPUS * LOCAL_BATCH_SIZE)))"',
            job,
        )
        self.assertIn("NASHRS_COMAL_NUM_GPUS=1", submitter)
        self.assertIn("ngpus=1:ncpus=12:mem=225GB", submitter)
        self.assertIn("distributed_type: 'NO'", accelerate)
        self.assertIn("mixed_precision: bf16", accelerate)

    def test_qwen_compatibility_guards_only_missing_tulu_symbols(self):
        data_utils = ModuleType("data_utils")
        qwen_dataset = object()
        data_utils.PreferenceBaseQwenDataset = qwen_dataset

        installed = install_missing_tulu_symbols(data_utils)

        self.assertEqual(
            installed,
            ("PreferenceBaseTuluDataset", "collate_preference_base_tulu"),
        )
        self.assertIs(data_utils.PreferenceBaseQwenDataset, qwen_dataset)
        self.assertTrue(hasattr(data_utils, "PreferenceBaseTuluDataset"))
        self.assertTrue(hasattr(data_utils, "collate_preference_base_tulu"))
        self.assertEqual(selected_model_type([]), "qwen")
        self.assertEqual(selected_model_type(["--model_type", "qwen"]), "qwen")

    def test_logprob_workers_do_not_exceed_tiny_split_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "test.jsonl"
            data.write_text(json.dumps({"prompt": "one row"}) + "\n")
            argv = [
                "compat.py",
                "--input_dir",
                str(data),
                "--output_dir",
                str(data),
                "--gpuids",
                "0",
                "1",
                "--model_type",
                "qwen",
            ]

            removed = limit_gpuids_to_dataset_rows(argv)

            self.assertEqual(removed, ("1",))
            gpu_index = argv.index("--gpuids")
            self.assertEqual(argv[gpu_index + 1 : gpu_index + 2], ["0"])
            self.assertEqual(argv[gpu_index + 2], "--model_type")

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
