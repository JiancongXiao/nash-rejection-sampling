#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="${NASHRS_CONFIG_FILE:-$SCRIPT_DIR/config.env}"
# shellcheck disable=SC1090
source "$CONFIG_FILE"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT"
export HF_HOME="$NASHRS_CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

MANIFEST="$NASHRS_DATA_ROOT/manifest.json"
MODEL="$($NASHRS_VENV_ROOT/native-trl-0.13.0/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["policy"]["path"])' "$MANIFEST")"
PROMPTS="$($NASHRS_VENV_ROOT/native-trl-0.13.0/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["files"]["test_256"]["path"])' "$MANIFEST")"
TRAIN_ROOT="$NASHRS_RESULT_ROOT/seed${NASHRS_SEED}"
OUTPUT="$TRAIN_ROOT/evaluation"
for method in nash_rs nash_md egpo mpo; do
  grep -q '"status": "complete"' "$TRAIN_ROOT/$method/run_manifest.json" || {
    echo "Training is not complete for $method" >&2; exit 3;
  }
done
mkdir -p "$OUTPUT"
GPU_COUNT="$($NASHRS_VENV_ROOT/native-trl-0.13.0/bin/python -c 'import torch; print(torch.cuda.device_count())')"
[[ "$GPU_COUNT" == 8 ]] || { echo "Expected 8 visible GPUs, found $GPU_COUNT" >&2; exit 4; }

"$NASHRS_VENV_ROOT/native-trl-0.13.0/bin/torchrun" \
  --standalone --nnodes=1 --nproc-per-node=8 \
  -m examples.b200.evaluate_suite \
  --base-model "$MODEL" --result-root "$TRAIN_ROOT" \
  --prompts "$PROMPTS" \
  --preference-config "$REPO_ROOT/configs/pku_tulu3_8b_preference.json" \
  --output "$OUTPUT" --max-prompts "$NASHRS_EVAL_PROMPTS" \
  --max-new-tokens "$NASHRS_GENERATE_MAX_LEN" \
  --bootstrap-samples "$NASHRS_BOOTSTRAP_SAMPLES" --bootstrap-seed 20260923

test -s "$OUTPUT/summary.json"
echo "Evaluation completed: $OUTPUT/summary.json"
