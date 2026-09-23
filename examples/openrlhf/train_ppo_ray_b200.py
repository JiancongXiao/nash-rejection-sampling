"""Portable single-node eight-GPU OpenRLHF/Ray launcher for B200."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import signal
import subprocess


def normalize_visible_devices() -> None:
    """Map scheduler-provided GPU UUIDs to indices required by vLLM 0.15."""

    original = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    entries = [entry.strip() for entry in original.split(",") if entry.strip()]
    if not entries or all(entry.isdigit() for entry in entries):
        return
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
    )
    inventory = {}
    for line in output.splitlines():
        index, uuid = (value.strip() for value in line.split(",", 1))
        inventory[uuid] = index
    normalized = []
    for entry in entries:
        matches = [index for uuid, index in inventory.items() if uuid.startswith(entry)]
        if len(matches) != 1:
            raise SystemExit(f"cannot uniquely map GPU {entry!r}: {inventory}")
        normalized.append(matches[0])
    if len(set(normalized)) != 8:
        raise SystemExit(f"expected eight unique scheduler GPUs: {normalized}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(normalized)


def main() -> None:
    normalize_visible_devices()
    import ray
    import torch

    gpu_count = torch.cuda.device_count()
    if gpu_count != 8:
        raise SystemExit(f"expected exactly eight visible GPUs, found {gpu_count}")

    runtime_env = None
    if os.environ.get("NASHRS_PATCH_OPENRLHF_LORA_SYNC") == "1":
        from nashrs.openrlhf_lora_sync import install_openrlhf_lora_vllm_sync_patch

        install_openrlhf_lora_vllm_sync_patch()
        runtime_env = {
            "worker_process_setup_hook": install_openrlhf_lora_vllm_sync_patch
        }

    ray_root = Path(os.environ["NASHRS_RAY_TMPDIR"])
    ray_root.mkdir(parents=True, exist_ok=True)
    timeout = int(os.environ.get("NASHRS_RAY_STARTUP_TIMEOUT", "300"))

    def fail_startup(_signum, _frame):
        raise TimeoutError(f"ray.init did not return within {timeout} seconds")

    signal.signal(signal.SIGALRM, fail_startup)
    signal.alarm(timeout)
    try:
        ray.init(
            address="local",
            num_cpus=min(96, os.cpu_count() or 1),
            num_gpus=8,
            include_dashboard=False,
            object_store_memory=int(
                os.environ.get("NASHRS_RAY_OBJECT_STORE_BYTES", str(16 * 1024**3))
            ),
            _node_ip_address="127.0.0.1",
            _temp_dir=str(ray_root),
            runtime_env=runtime_env,
        )
    finally:
        signal.alarm(0)
    resources = ray.cluster_resources()
    print("Ray cluster resources:", resources, flush=True)
    if int(resources.get("GPU", 0)) != 8:
        ray.shutdown()
        raise SystemExit("Ray did not register all eight GPUs")
    try:
        runpy.run_module("openrlhf.cli.train_ppo_ray", run_name="__main__")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
