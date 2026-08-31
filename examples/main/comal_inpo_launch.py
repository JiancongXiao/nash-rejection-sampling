"""Launch the authors' INPO trainer with configurable Qwen/checkpoint paths."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import types


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--tau-eta-ratio", type=float, default=1.0 / 3.0)
    parser.add_argument("--accumulate-step", type=int, default=4)
    args = parser.parse_args()

    comal_root = args.source_root / "COMAL"
    sys.path.insert(0, str(comal_root))
    source_path = comal_root / "inpo.py"
    source = source_path.read_text()
    old = 'output = output[0]["logits"]'
    if source.count(old) != 1:
        raise RuntimeError("unexpected COMAL inpo.py forward-output contract")
    source = source.replace(old, "output = output.logits")
    module = types.ModuleType("comal_official_inpo_compat")
    module.__file__ = str(source_path)
    exec(compile(source, str(source_path), "exec"), module.__dict__)

    run_args = argparse.Namespace(
        log=True,
        model_pt="",
        epoch=args.epochs,
        eta=args.eta,
        tau_eta_ratio=args.tau_eta_ratio,
        dataset=args.dataset,
        exp_name=args.output,
        pretrained=args.pretrained,
        batch_size=1,
        accumulate_step=args.accumulate_step,
        model_type=args.tokenizer,
        seed=args.seed,
        max_lr=5e-7,
        warmup_ratio=0.1,
        max_len=640,
        gradient_checkpointing=True,
        use_flash_attention=False,
        mixed_precision=True,
        allow_tf32=True,
        empty_cache=False,
        eval_interval=500,
        report_freq=10,
        grad_norm=0,
        device="auto",
        ref_free=False,
    )
    module.run(run_args)


if __name__ == "__main__":
    main()
