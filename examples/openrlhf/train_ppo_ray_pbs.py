"""Launch OpenRLHF PPO with PBS-visible GPUs registered explicitly in Ray."""

from __future__ import annotations

import os
from pathlib import Path
import runpy

import ray
import torch


def main() -> None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    gpu_count = torch.cuda.device_count()
    print(f"PBS CUDA_VISIBLE_DEVICES: {visible_devices}", flush=True)
    print(f"PyTorch visible GPUs: {gpu_count}", flush=True)
    if gpu_count < 1:
        raise SystemExit("No PBS GPU is visible inside the container")

    ray_root = Path(os.environ["NASHRS_RAY_TMPDIR"])
    ray_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    ray.init(
        num_gpus=gpu_count,
        include_dashboard=False,
        _temp_dir=str(ray_root),
    )
    resources = ray.cluster_resources()
    print(f"Ray cluster resources: {resources}", flush=True)
    if resources.get("GPU", 0) < 1:
        ray.shutdown()
        raise SystemExit("Ray did not register the PBS GPU")

    # OpenRLHF reuses this initialized Ray instance and parses its normal CLI.
    runpy.run_module("openrlhf.cli.train_ppo_ray", run_name="__main__")


if __name__ == "__main__":
    main()
