#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="${NASHRS_CONFIG_FILE:-$SCRIPT_DIR/config.env}"
METHOD="${METHOD:-${1:-}}"
case "$METHOD" in nash_rs|nash_md|egpo|mpo) ;; *) echo "METHOD must be nash_rs, nash_md, egpo, or mpo" >&2; exit 2;; esac
# shellcheck disable=SC1090
source "$CONFIG_FILE"

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT"
export HF_HOME="$NASHRS_CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

MANIFEST="$NASHRS_DATA_ROOT/manifest.json"
[[ -s "$MANIFEST" ]] || { echo "Run cluster/b200/prepare.sh first" >&2; exit 3; }
MODEL="$($NASHRS_VENV_ROOT/native-trl-0.13.0/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["policy"]["path"])' "$MANIFEST")"
PROMPTS="$($NASHRS_VENV_ROOT/native-trl-0.13.0/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["files"]["train_full"]["path"])' "$MANIFEST")"
PROMPT_BUDGET="$($NASHRS_VENV_ROOT/native-trl-0.13.0/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["files"]["train_full"]["count"])' "$MANIFEST")"
PREFERENCE_CONFIG="$REPO_ROOT/configs/pku_tulu3_8b_preference.json"
OUTPUT="$NASHRS_RESULT_ROOT/seed${NASHRS_SEED}/$METHOD"
mkdir -p "$OUTPUT"

if [[ -s "$OUTPUT/run_manifest.json" ]] && \
   grep -q '"status": "complete"' "$OUTPUT/run_manifest.json"; then
  echo "$METHOD is already complete: $OUTPUT"
  exit 0
fi

GPU_COUNT="$($NASHRS_VENV_ROOT/native-trl-0.13.0/bin/python -c 'import torch; print(torch.cuda.device_count())')"
[[ "$GPU_COUNT" == 8 ]] || { echo "Expected 8 visible GPUs, found $GPU_COUNT" >&2; exit 4; }
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv > "$OUTPUT/gpu_inventory.csv"
cp "$CONFIG_FILE" "$OUTPUT/config.env"
cp "$MANIFEST" "$OUTPUT/data_manifest.json"
git -C "$REPO_ROOT" rev-parse HEAD > "$OUTPUT/git_commit.txt"

COMMON=(
  --method "$METHOD"
  --model "$MODEL"
  --prompts "$PROMPTS"
  --preference-config "$PREFERENCE_CONFIG"
  --source-root "$NASHRS_SOURCE_ROOT"
  --output "$OUTPUT"
  --prompt-budget "$PROMPT_BUDGET"
  --global-batch "$NASHRS_GLOBAL_BATCH"
  --train-micro-batch "$NASHRS_NATIVE_MICRO_BATCH"
  --generate-batch "$NASHRS_NATIVE_GENERATE_BATCH"
  --learning-rate "$NASHRS_ACTOR_LR"
  --max-new-tokens "$NASHRS_GENERATE_MAX_LEN"
  --seed "$NASHRS_SEED"
  --checkpoint-steps "$NASHRS_CHECKPOINT_STEPS"
  --lora-rank "$NASHRS_LORA_RANK"
  --lora-alpha "$NASHRS_LORA_ALPHA"
  --lora-dropout "$NASHRS_LORA_DROPOUT"
)

if [[ "$METHOD" != nash_rs ]]; then
  RESUME=()
  if [[ "$METHOD" == mpo ]]; then
    [[ -s "$OUTPUT/trainer_checkpoint/state.pt" ]] && RESUME=(--resume)
  elif compgen -G "$OUTPUT/trainer/checkpoint-*" >/dev/null; then
    RESUME=(--resume)
  fi
  "$NASHRS_VENV_ROOT/native-trl-0.13.0/bin/torchrun" \
    --standalone --nnodes=1 --nproc-per-node=8 \
    -m examples.b200.train_native_ddp "${COMMON[@]}" "${RESUME[@]}"
else
  OPENRLHF_ENV="$NASHRS_VENV_ROOT/openrlhf-0.9.3-vllm-0.15.0"
  LOG="$OUTPUT/train.log"
  export NASHRS_RAY_TMPDIR="${TMPDIR:-/tmp}/nashrs-ray-${SLURM_JOB_ID:-${PBS_JOBID:-$$}}"
  export NASHRS_PATCH_OPENRLHF_LORA_SYNC=1
  export NASHRS_METHOD=nash_rs NASHRS_SEED NASHRS_TAU NASHRS_B1 NASHRS_B2
  export NASHRS_PROPOSAL_BATCH_SIZE NASHRS_MAX_PROPOSALS
  export NASHRS_OPPONENT_MODEL="$MODEL"
  export NASHRS_PREFERENCE_COMPONENTS_JSON
  NASHRS_PREFERENCE_COMPONENTS_JSON="$($OPENRLHF_ENV/bin/python -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["components"], separators=(",",":")))' "$PREFERENCE_CONFIG")"
  export NASHRS_AGENT_FUNC_PATH="$REPO_ROOT/examples/openrlhf/nash_rs_agent.py"
  export NASHRS_RUN_ROOT_OVERRIDE="$OUTPUT"
  SAVE_ARGS=(--save_hf_ckpt)
  [[ -d "$OUTPUT/deepspeed" ]] && SAVE_ARGS+=(--load_checkpoint)
  set -o pipefail
  "$OPENRLHF_ENV/bin/python" "$REPO_ROOT/examples/openrlhf/train_ppo_ray_b200.py" \
    --ref_num_nodes 1 --ref_num_gpus_per_node 8 \
    --critic_num_nodes 1 --critic_num_gpus_per_node 8 \
    --actor_num_nodes 1 --actor_num_gpus_per_node 8 \
    --vllm_num_engines 8 --vllm_tensor_parallel_size 1 \
    --colocate_all_models --vllm_gpu_memory_utilization 0.35 \
    --vllm_enable_sleep --deepspeed_enable_sleep --gradient_checkpointing \
    --pretrain "$MODEL" --critic_pretrain "$MODEL" \
    --agent_func_path "$NASHRS_AGENT_FUNC_PATH" \
    --save_path "$OUTPUT/actor_adapter" --ckpt_path "$OUTPUT/deepspeed" \
    "${SAVE_ARGS[@]}" \
    --micro_train_batch_size 1 --train_batch_size "$NASHRS_GLOBAL_BATCH" \
    --micro_rollout_batch_size 1 --rollout_batch_size "$NASHRS_GLOBAL_BATCH" \
    --vllm_generate_batch_size 8 --n_samples_per_prompt 1 \
    --seed "$NASHRS_SEED" --num_episodes 1 --max_epochs 1 \
    --max_samples "$PROMPT_BUDGET" --prompt_max_len "$NASHRS_PROMPT_MAX_LEN" \
    --generate_max_len "$NASHRS_GENERATE_MAX_LEN" --zero_stage 2 \
    --param_dtype bf16 --lora_rank "$NASHRS_LORA_RANK" \
    --lora_alpha "$NASHRS_LORA_ALPHA" --lora_dropout "$NASHRS_LORA_DROPOUT" \
    --actor_learning_rate "$NASHRS_ACTOR_LR" --critic_learning_rate 1e-5 \
    --lr_warmup_ratio 0 --lr_scheduler constant --init_kl_coef 0.01 \
    --prompt_data "$PROMPTS" --input_key prompt --apply_chat_template \
    --attn_implementation sdpa --enforce_eager --logging_steps 1 \
    --save_steps "$NASHRS_CHECKPOINT_STEPS" --max_ckpt_num 1 2>&1 | tee "$LOG"
  "$OPENRLHF_ENV/bin/python" "$REPO_ROOT/examples/export_openrlhf_metrics.py" \
    --log "$LOG" --output "$OUTPUT/step_metrics.jsonl" \
    --samples-per-step "$NASHRS_GLOBAL_BATCH"
  "$OPENRLHF_ENV/bin/python" -m examples.b200.verify_adapter \
    "$OUTPUT/actor_adapter" --output "$OUTPUT/parameter_update.json"
  "$OPENRLHF_ENV/bin/python" - "$OUTPUT/run_manifest.json" <<PY
import json, sys
json.dump({
  "status": "complete", "method": "nash_rs", "model": "$MODEL",
  "prompts": "$PROMPTS", "prompt_budget": int("$PROMPT_BUDGET"),
  "world_size": 8, "global_batch": int("$NASHRS_GLOBAL_BATCH"),
  "seed": int("$NASHRS_SEED"), "tau": float("$NASHRS_TAU"),
  "b1": int("$NASHRS_B1"), "b2": int("$NASHRS_B2"),
  "proposal_batch_size": int("$NASHRS_PROPOSAL_BATCH_SIZE"),
  "preference_model_batch_size": int("$NASHRS_PM_BATCH_SIZE"),
  "max_proposals": int("$NASHRS_MAX_PROPOSALS"),
  "lora": {"rank": int("$NASHRS_LORA_RANK"), "alpha": int("$NASHRS_LORA_ALPHA"), "dropout": float("$NASHRS_LORA_DROPOUT")}
}, open(sys.argv[1], "w"), indent=2, sort_keys=True)
PY
fi

"$NASHRS_VENV_ROOT/native-trl-0.13.0/bin/python" -m examples.b200.verify_adapter \
  "$OUTPUT/actor_adapter" --output "$OUTPUT/adapter_verification.json"
echo "$METHOD full PKU training completed: $OUTPUT"
