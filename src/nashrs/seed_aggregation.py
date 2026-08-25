"""Aggregate fixed-test metrics across independent training seeds."""

from __future__ import annotations

import json
from pathlib import Path
import random
import statistics
from typing import Sequence


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_seed_run(run_dir: Path, baseline: str, candidate: str) -> dict:
    eval_dir = run_dir / "independent_test_128"
    summary = json.loads((eval_dir / "summary.json").read_text())
    methods = summary["methods"]
    if baseline not in methods or candidate not in methods:
        raise ValueError(f"{run_dir} does not contain {baseline!r} and {candidate!r}")
    baseline_index = methods.index(baseline)
    candidate_index = methods.index(candidate)
    details = [
        json.loads(line)
        for line in (eval_dir / "per_prompt_metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(details) != summary["prompts"]:
        raise ValueError(f"per-prompt count does not match summary in {run_dir}")
    win_probabilities = [
        row["pairwise_probability"][candidate][baseline] for row in details
    ]
    reward_deltas = [
        row["scalar_reward"][candidate] - row["scalar_reward"][baseline]
        for row in details
    ]
    return {
        "run_dir": str(run_dir),
        "run_seed": summary.get("run_seed"),
        "prompts": len(details),
        "candidate_reward": summary["mean_scalar_reward"][candidate],
        "baseline_reward": summary["mean_scalar_reward"][baseline],
        "reward_delta": statistics.mean(reward_deltas),
        "direct_win_rate": summary["pairwise_matrix"][candidate_index][baseline_index],
        "candidate_average_win_rate": summary["average_win_rate"][candidate],
        "candidate_worst_case_win_rate": summary["worst_case_win_rate"][candidate],
        "candidate_exploitability": summary["empirical_exploitability"][candidate],
        "candidate_mean_tokens": summary["generation"][candidate]["mean_response_tokens"],
        "candidate_truncation_rate": summary["generation"][candidate]["truncation_rate"],
        "win_probabilities": win_probabilities,
        "reward_deltas": reward_deltas,
    }


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def aggregate_seed_runs(
    runs: Sequence[dict], bootstrap_samples: int = 10_000, seed: int = 20260825
) -> dict:
    if not runs:
        raise ValueError("at least one seed run is required")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    seeds = [run["run_seed"] for run in runs]
    if any(value is None for value in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("run seeds must be present and unique")
    prompt_counts = {run["prompts"] for run in runs}
    if len(prompt_counts) != 1:
        raise ValueError("all seed runs must use the same prompt count")

    scalar_fields = (
        "reward_delta",
        "direct_win_rate",
        "candidate_average_win_rate",
        "candidate_worst_case_win_rate",
        "candidate_exploitability",
        "candidate_mean_tokens",
        "candidate_truncation_rate",
    )
    across_seed = {
        field: _mean_std([float(run[field]) for run in runs])
        for field in scalar_fields
    }

    rng = random.Random(seed)
    bootstrap = {"direct_win_rate": [], "reward_delta": []}
    for _ in range(bootstrap_samples):
        sampled_runs = [rng.choice(runs) for _ in runs]
        sampled_wins = []
        sampled_rewards = []
        for run in sampled_runs:
            count = run["prompts"]
            indices = [rng.randrange(count) for _ in range(count)]
            sampled_wins.extend(run["win_probabilities"][index] for index in indices)
            sampled_rewards.extend(run["reward_deltas"][index] for index in indices)
        bootstrap["direct_win_rate"].append(statistics.mean(sampled_wins))
        bootstrap["reward_delta"].append(statistics.mean(sampled_rewards))

    serializable_runs = [
        {key: value for key, value in run.items() if key not in {"win_probabilities", "reward_deltas"}}
        for run in runs
    ]
    return {
        "num_seeds": len(runs),
        "seeds": seeds,
        "prompts_per_seed": next(iter(prompt_counts)),
        "per_seed": serializable_runs,
        "across_seed": across_seed,
        "hierarchical_bootstrap_samples": bootstrap_samples,
        "hierarchical_confidence_intervals_95": {
            metric: [_percentile(values, 0.025), _percentile(values, 0.975)]
            for metric, values in bootstrap.items()
        },
        "seeds_with_win_rate_above_half": sum(
            run["direct_win_rate"] > 0.5 for run in runs
        ),
    }
