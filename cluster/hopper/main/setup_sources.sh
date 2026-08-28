#!/bin/bash
set -euo pipefail

SOURCE_ROOT="${NASHRS_MAIN_SOURCE_ROOT:-/scratch/jiancongxiao/external/nashrs-main}"
mkdir -p "$SOURCE_ROOT"

checkout() {
  local name="$1"
  local url="$2"
  local revision="$3"
  local target="$SOURCE_ROOT/$name"
  if [[ ! -d "$target/.git" ]]; then
    git clone "$url" "$target"
  fi
  local current
  current="$(git -C "$target" rev-parse HEAD)"
  if [[ "$current" != "$revision" ]]; then
    git -C "$target" fetch --depth 1 origin "$revision"
    # External trees are disposable pinned dependencies, never user worktrees.
    git -C "$target" checkout --detach "$revision"
  fi
  current="$(git -C "$target" rev-parse HEAD)"
  if [[ "$current" != "$revision" ]]; then
    echo "Failed to pin $name: expected $revision, found $current" >&2
    exit 3
  fi
  echo "$name $current"
}

checkout COMAL https://github.com/yale-nlp/COMAL.git \
  30e9446fbc5212d9e6886a27d09d1266e50a45c0
checkout EGPO https://github.com/zhourunlong/EGPO.git \
  f295bf3750627374c7ddb5d4459390321de2836e
checkout safe-rlhf https://github.com/PKU-Alignment/safe-rlhf.git \
  e8cca16665ef2340ac92c6514f05519310251581

PYTHONPATH="$PWD/src" python3 examples/main/doctor.py \
  --source-root "$SOURCE_ROOT" --strict
