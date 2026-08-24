"""Verify that an OpenRLHF smoke run changed at least one actor tensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import model_info, snapshot_download
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-revision")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tensor_locations(root: Path) -> dict[str, Path]:
    locations = {}
    for path in sorted(root.rglob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                locations.setdefault(key, path)
    return locations


def load_tensor(path: Path, key: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).float()


def main() -> None:
    args = parse_args()
    local_base = Path(args.base_model)
    if local_base.exists():
        base_root = local_base
        revision = args.base_revision or local_base.name
    else:
        revision = args.base_revision or model_info(args.base_model).sha
        base_root = Path(snapshot_download(args.base_model, revision=revision))
    base = tensor_locations(base_root)
    trained = tensor_locations(args.checkpoint)
    preferred = "model.layers.0.self_attn.q_proj.weight"
    common = sorted(set(base).intersection(trained))
    if not common:
        raise SystemExit("No common safetensors keys found")
    ordered = ([preferred] if preferred in common else []) + [
        key for key in common if key != preferred
    ]
    changed = None
    for checked, key in enumerate(ordered, start=1):
        before = load_tensor(base[key], key)
        after = load_tensor(trained[key], key)
        difference = after - before
        changed_elements = int(torch.count_nonzero(difference))
        if changed_elements:
            changed = {
                "tensor": key,
                "l2_update_norm": float(torch.linalg.vector_norm(difference)),
                "max_abs_update": float(difference.abs().max()),
                "changed_elements": changed_elements,
                "num_elements": difference.numel(),
                "tensors_checked": checked,
            }
            break
    if changed is None:
        raise SystemExit(
            f"PPO smoke checkpoint is identical across all {len(common)} common tensors"
        )
    result = {
        "base_model": args.base_model,
        "base_revision": revision,
        "checkpoint": str(args.checkpoint),
        **changed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
