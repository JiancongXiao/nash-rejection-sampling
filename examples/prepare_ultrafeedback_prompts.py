"""Create deterministic, disjoint prompt splits from UltraFeedback."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def normalize_prompt(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines()).strip()


def prompt_digest(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="openbmb/UltraFeedback")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--selection-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--max-prompt-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    requested = args.train_size + args.selection_size + args.test_size
    if min(args.train_size, args.selection_size, args.test_size) <= 0:
        raise ValueError("all split sizes must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    dataset = load_dataset(
        args.dataset,
        split="train",
        revision=args.revision,
    )
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)

    accepted: list[dict] = []
    seen: set[str] = set()
    rejected_empty = 0
    rejected_duplicate = 0
    rejected_length = 0
    for index in indices:
        example = dataset[index]
        prompt = normalize_prompt(str(example.get("instruction", "")))
        if not prompt:
            rejected_empty += 1
            continue
        digest = prompt_digest(prompt)
        if digest in seen:
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
        seen.add(digest)
        accepted.append(
            {
                "prompt": prompt,
                "prompt_sha256": digest,
                "source": example.get("source"),
                "token_count": len(token_ids),
            }
        )
        if len(accepted) == requested:
            break

    if len(accepted) != requested:
        raise RuntimeError(f"needed {requested} valid prompts, found {len(accepted)}")

    split_rows = {
        "train_2048": accepted[: args.train_size],
        "selection_256": accepted[
            args.train_size : args.train_size + args.selection_size
        ],
        "test_1024": accepted[args.train_size + args.selection_size :],
    }
    digest_sets = {
        name: {row["prompt_sha256"] for row in rows}
        for name, rows in split_rows.items()
    }
    names = list(digest_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if digest_sets[left] & digest_sets[right]:
                raise RuntimeError(f"prompt leakage between {left} and {right}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in split_rows.items():
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        files[name] = {
            "path": str(path),
            "rows": len(rows),
            "sha256": file_digest(path),
        }

    manifest = {
        "dataset": args.dataset,
        "dataset_revision": args.revision,
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "max_prompt_tokens": args.max_prompt_tokens,
        "files": files,
        "rejected": {
            "empty": rejected_empty,
            "duplicate": rejected_duplicate,
            "over_token_limit": rejected_length,
        },
        "disjoint_by_prompt_sha256": True,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
