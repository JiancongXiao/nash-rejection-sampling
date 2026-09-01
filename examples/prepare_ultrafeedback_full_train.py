"""Build the leakage-free full UltraFeedback training prompt set."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from examples.prepare_ultrafeedback_prompts import (
    file_digest,
    normalize_prompt,
    prompt_digest,
    write_jsonl,
)


def read_prompt_digests(path: Path) -> set[str]:
    digests: set[str] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = normalize_prompt(str(row.get("prompt", "")))
        if not prompt:
            raise ValueError(f"empty prompt at {path}:{line_number}")
        recorded = row.get("prompt_sha256")
        computed = prompt_digest(prompt)
        if recorded is not None and recorded != computed:
            raise ValueError(f"incorrect prompt digest at {path}:{line_number}")
        digests.add(computed)
    return digests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="openbmb/UltraFeedback")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--selection-prompts", type=Path, required=True)
    parser.add_argument("--test-prompts", type=Path, required=True)
    parser.add_argument("--previous-train-prompts", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    selection = read_prompt_digests(args.selection_prompts)
    test = read_prompt_digests(args.test_prompts)
    if selection & test:
        raise ValueError("selection and test prompt sets overlap")
    held_out = selection | test

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    dataset = load_dataset(
        args.dataset,
        split="train",
        revision=args.revision,
    )
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)

    rows: list[dict] = []
    accepted_all: set[str] = set()
    rejected_empty = 0
    rejected_duplicate = 0
    rejected_length = 0
    excluded_held_out = 0
    for index in indices:
        example = dataset[index]
        prompt = normalize_prompt(str(example.get("instruction", "")))
        if not prompt:
            rejected_empty += 1
            continue
        digest = prompt_digest(prompt)
        if digest in accepted_all:
            rejected_duplicate += 1
            continue
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if len(token_ids) > args.max_prompt_tokens:
            rejected_length += 1
            continue
        accepted_all.add(digest)
        if digest in held_out:
            excluded_held_out += 1
            continue
        rows.append(
            {
                "prompt": prompt,
                "prompt_sha256": digest,
                "source": example.get("source"),
                "token_count": len(token_ids),
            }
        )

    if excluded_held_out != len(held_out):
        missing = len(held_out) - excluded_held_out
        raise RuntimeError(f"{missing} held-out prompts were not found in the source")

    previous_train_count = 0
    if args.previous_train_prompts is not None:
        previous = read_prompt_digests(args.previous_train_prompts)
        output_digests = {row["prompt_sha256"] for row in rows}
        missing = previous - output_digests
        if missing:
            raise RuntimeError(
                f"full training set is missing {len(missing)} previous training prompts"
            )
        previous_train_count = len(previous)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "train_full.jsonl"
    write_jsonl(output, rows)
    manifest = {
        "dataset": args.dataset,
        "dataset_revision": args.revision,
        "source_rows": len(dataset),
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "max_prompt_tokens": args.max_prompt_tokens,
        "training_rows": len(rows),
        "training_file": {
            "path": str(output),
            "sha256": file_digest(output),
        },
        "held_out": {
            "selection_rows": len(selection),
            "test_rows": len(test),
            "excluded_rows": excluded_held_out,
            "selection_sha256": file_digest(args.selection_prompts),
            "test_sha256": file_digest(args.test_prompts),
        },
        "previous_training_rows_verified": previous_train_count,
        "rejected": {
            "empty": rejected_empty,
            "duplicate": rejected_duplicate,
            "over_token_limit": rejected_length,
        },
        "disjoint_from_held_out_by_prompt_sha256": True,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
