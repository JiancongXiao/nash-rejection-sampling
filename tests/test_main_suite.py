from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from nashrs.main_metrics import (
    validate_run_step_records,
    validate_step_records,
    write_run_manifest,
    write_smoke_manifest,
)
from nashrs.main_suite import (
    MAIN_METHODS,
    get_main_method,
    validate_optimizer_steps,
    validate_smoke_steps,
)
from examples.main.finalize_smoke import normalize_record


class MainSuiteTest(unittest.TestCase):
    def test_openrlhf_metric_names_are_normalized(self) -> None:
        record = normalize_record(
            {
                "step": 1,
                "policy_loss": 0.25,
                "actor_lr": 1e-6,
                "nashrs/preference_model_calls": 8.0,
                "nashrs/generated_tokens": 128.0,
                "nashrs/gpu_hours": 0.01,
            }
        )
        self.assertEqual(record["loss"], 0.25)
        self.assertEqual(record["learning_rate"], 1e-6)
        self.assertEqual(record["preference_model_calls"], 8.0)
        self.assertEqual(record["generated_tokens"], 128.0)
        self.assertEqual(record["gpu_hours"], 0.01)

    def test_openrlhf_main_cost_includes_actor_rollout_tokens(self) -> None:
        record = normalize_record(
            {
                "step": 1,
                "policy_loss": 0.25,
                "actor_lr": 1e-6,
                "comparison/preference_model_calls": 4.0,
                "comparison/generated_tokens": 120.0,
                "comparison/gpu_hours": 0.01,
                "response_length": 48.5,
                "nashrs/samples_in_step": 2,
            },
            method="nash_rs",
        )
        self.assertEqual(record["generated_tokens"], 217.0)

    def test_main_has_seven_separate_method_native_entries(self) -> None:
        self.assertEqual(
            MAIN_METHODS,
            (
                "reward_ppo",
                "self_play",
                "nash_md",
                "nash_rs",
                "mpo",
                "egpo",
                "comal",
            ),
        )
        for name in MAIN_METHODS:
            spec = get_main_method(name)
            self.assertTrue(spec.experiment_id.startswith("main/"))
            self.assertNotIn("controlled", spec.entrypoint)

    def test_smoke_step_guard(self) -> None:
        self.assertEqual(validate_smoke_steps(2), 2)
        self.assertEqual(validate_smoke_steps(4), 4)
        for invalid in (0, 1, 5):
            with self.assertRaises(ValueError):
                validate_smoke_steps(invalid)

    def test_non_smoke_step_budget_is_positive_and_unbounded(self) -> None:
        self.assertEqual(validate_optimizer_steps(1), 1)
        self.assertEqual(validate_optimizer_steps(400), 400)
        with self.assertRaises(ValueError):
            validate_optimizer_steps(0)

    def test_metrics_and_parameter_update_are_required(self) -> None:
        records = [
            {
                "step": step,
                "loss": 1.0 / step,
                "learning_rate": 1e-6,
                "preference_model_calls": 4,
                "generated_tokens": 32,
                "gpu_hours": 0.01,
            }
            for step in (1, 2)
        ]
        self.assertEqual(len(validate_step_records(records, expected_steps=2)), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest = write_smoke_manifest(
                path,
                method="egpo",
                records=records,
                expected_steps=2,
                parameter_update={"changed_elements": 7, "l2_update_norm": 0.1},
            )
            self.assertEqual(manifest["suite"], "main")
            self.assertTrue(json.loads(path.read_text())["parameter_update_verified"])
        with self.assertRaises(ValueError):
            write_smoke_manifest(
                Path("unused.json"),
                method="comal",
                records=records,
                expected_steps=2,
                parameter_update={"changed_elements": 0, "l2_update_norm": 0.0},
            )

    def test_pilot_manifest_supports_more_than_four_steps(self) -> None:
        records = [
            {
                "step": step,
                "loss": 1.0 / step,
                "learning_rate": 1e-6,
                "preference_model_calls": 2,
                "generated_tokens": 10,
                "gpu_hours": 0.001,
            }
            for step in range(1, 17)
        ]
        self.assertEqual(
            len(validate_run_step_records(records, expected_steps=16)), 16
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_run_manifest(
                Path(directory) / "run_manifest.json",
                method="nash_rs",
                run_kind="pilot",
                records=records,
                expected_steps=16,
                parameter_update={"changed_elements": 2, "l2_update_norm": 0.01},
            )
        self.assertEqual(manifest["run_kind"], "pilot")
        self.assertEqual(manifest["accounting_totals"]["generated_tokens"], 160.0)

    def test_scaling_gate_is_distinct_from_pilot_and_full(self) -> None:
        records = [
            {
                "step": 1,
                "loss": 0.5,
                "learning_rate": 1e-6,
                "preference_model_calls": 2,
                "generated_tokens": 20,
                "gpu_hours": 0.002,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_run_manifest(
                Path(directory) / "run_manifest.json",
                method="nash_rs",
                run_kind="scaling_gate",
                records=records,
                expected_steps=1,
                parameter_update={"changed_elements": 1, "l2_update_norm": 0.001},
            )
        self.assertEqual(manifest["run_kind"], "scaling_gate")


if __name__ == "__main__":
    unittest.main()
