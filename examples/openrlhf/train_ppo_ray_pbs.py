"""Launch OpenRLHF PPO with PBS-visible GPUs registered explicitly in Ray."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import subprocess


def normalize_pbs_gpu_uuid() -> tuple[str, str]:
    """Translate PBS GPU UUIDs to the same devices' physical indices for vLLM.

    CUDA accepts UUIDs natively, but vLLM 0.15 converts every entry in
    CUDA_VISIBLE_DEVICES with int(). The mapping preserves PBS's device choice;
    it does not select an additional or different GPU.
    """

    original = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    entries = [entry.strip() for entry in original.split(",") if entry.strip()]
    if not entries or all(entry.isdigit() for entry in entries):
        return original or "<unset>", original or "<unset>"

    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    inventory = {}
    for line in output.splitlines():
        index, uuid = (part.strip() for part in line.split(",", maxsplit=1))
        inventory[uuid] = index

    normalized = []
    for entry in entries:
        if entry.isdigit():
            normalized.append(entry)
            continue
        matches = [index for uuid, index in inventory.items() if uuid.startswith(entry)]
        if len(matches) != 1:
            raise SystemExit(
                f"Could not uniquely map PBS GPU identifier {entry!r}: {inventory}"
            )
        normalized.append(matches[0])
    mapped = ",".join(normalized)
    os.environ["CUDA_VISIBLE_DEVICES"] = mapped
    return original, mapped


def main() -> None:
    original_devices, visible_devices = normalize_pbs_gpu_uuid()
    # Import after normalization so Ray, PyTorch, vLLM, and engine subprocesses
    # all inherit the numeric form required by vLLM 0.15.
    import ray
    import torch

    pbs_assignment = os.environ.get("NASHRS_PBS_GPU_ASSIGNMENT", original_devices)
    print(f"PBS GPU assignment: {pbs_assignment}", flush=True)
    print(f"vLLM-compatible CUDA_VISIBLE_DEVICES: {visible_devices}", flush=True)
    gpu_count = torch.cuda.device_count()
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
