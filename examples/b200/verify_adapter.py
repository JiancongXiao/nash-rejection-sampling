"""Fail unless a saved LoRA adapter contains a finite, nonzero update."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_file = args.adapter / "adapter_model.safetensors"
    config_file = args.adapter / "adapter_config.json"
    if not model_file.is_file() or not config_file.is_file():
        raise FileNotFoundError(f"incomplete adapter: {args.adapter}")
    import torch

    tensors = changed = elements = 0
    squared = 0.0
    maximum = 0.0
    with safe_open(model_file, framework="pt", device="cpu") as stream:
        for key in stream.keys():
            value = stream.get_tensor(key).float()
            if not torch.isfinite(value).all():
                raise ValueError(f"non-finite adapter tensor: {key}")
            tensors += 1
            count = int(torch.count_nonzero(value))
            changed += count
            elements += value.numel()
            squared += float(value.double().square().sum())
            maximum = max(maximum, float(value.abs().max()))
    result = {
        "adapter": str(args.adapter),
        "tensors": tensors,
        "nonzero_elements": changed,
        "elements": elements,
        "l2_norm": squared**0.5,
        "max_abs": maximum,
    }
    if tensors == 0 or changed == 0 or result["l2_norm"] == 0.0:
        raise RuntimeError("adapter contains no learned update")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
