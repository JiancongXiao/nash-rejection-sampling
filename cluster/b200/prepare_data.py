"""Download pinned inputs and freeze the full prompt-only PKU split."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def sha256(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def main() -> None:
    cache = Path(os.environ["NASHRS_CACHE_ROOT"]) / "huggingface" / "hub"
    data_root = Path(os.environ["NASHRS_DATA_ROOT"])
    data_root.mkdir(parents=True, exist_ok=True)
    seed = int(os.environ.get("NASHRS_SEED", "47"))
    prompt_limit = int(os.environ.get("NASHRS_PROMPT_MAX_LEN", "128"))

    policy_id = os.environ["NASHRS_POLICY_MODEL_ID"]
    policy_revision = os.environ["NASHRS_POLICY_REVISION"]
    pm_id = os.environ["NASHRS_PREFERENCE_MODEL_ID"]
    pm_revision = os.environ["NASHRS_PREFERENCE_REVISION"]
    pku_revision = os.environ["NASHRS_PKU_REVISION"]
    policy = Path(snapshot_download(policy_id, revision=policy_revision, cache_dir=cache))
    preference = Path(snapshot_download(pm_id, revision=pm_revision, cache_dir=cache))
    dataset = Path(
        snapshot_download(
            "PKU-Alignment/PKU-SafeRLHF",
            repo_type="dataset",
            revision=pku_revision,
            cache_dir=cache,
            allow_patterns=["data/*/*.jsonl"],
            max_workers=4,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(policy)
    if not tokenizer.chat_template:
        raise RuntimeError("Tulu SFT checkpoint must provide its native chat template")

    def read(split: str) -> dict[str, dict]:
        records: dict[str, dict] = {}
        files = sorted(dataset.glob(f"data/*/{split}.jsonl"))
        if len(files) != 3:
            raise RuntimeError(f"expected all three PKU sources for {split}: {files}")
        for file in files:
            for line in file.read_text().splitlines():
                row = json.loads(line)
                prompt = " ".join(row["prompt"].split())
                records.setdefault(
                    prompt,
                    {"prompt": prompt, "source": str(file.relative_to(dataset))},
                )
        return records

    train_records, test_records = read("train"), read("test")
    overlap = set(train_records) & set(test_records)
    for prompt in overlap:
        train_records.pop(prompt)

    def eligible(records: dict[str, dict]) -> list[dict]:
        rows = []
        for row in records.values():
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
            if len(ids) <= prompt_limit:
                rows.append(row)
        random.Random(seed).shuffle(rows)
        return rows

    train, test = eligible(train_records), eligible(test_records)
    eval_count = int(os.environ.get("NASHRS_EVAL_PROMPTS", "256"))
    if not train or len(test) < eval_count:
        raise RuntimeError(f"insufficient eligible prompts: train={len(train)} test={len(test)}")

    files = {}
    for name, rows in (("train_full", train), ("test_256", test[:eval_count])):
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        path = data_root / f"{name}.jsonl"
        if path.exists() and path.read_text() != content:
            raise RuntimeError(f"refusing to alter frozen data file: {path}")
        path.write_text(content)
        files[name] = {"path": str(path), "count": len(rows), "sha256": sha256(content)}

    manifest = {
        "policy": {"id": policy_id, "revision": policy_revision, "path": str(policy)},
        "preference_model": {"id": pm_id, "revision": pm_revision, "path": str(preference)},
        "dataset": "PKU-Alignment/PKU-SafeRLHF",
        "dataset_revision": pku_revision,
        "seed": seed,
        "max_chat_prompt_tokens": prompt_limit,
        "excluded_train_test_overlap": len(overlap),
        "files": files,
        "note": "Prompt-only NLHF training; original PKU responses and labels are not targets.",
    }
    (data_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
