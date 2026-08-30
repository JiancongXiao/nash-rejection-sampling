#!/bin/bash

set -euo pipefail

SEED="${1:-47}"
if [[ "$SEED" != "47" ]]; then
  echo "The first full 3B gate is pre-registered for seed 47" >&2
  exit 2
fi
METHODS=(reward_ppo self_play nash_md nash_rs mpo egpo)
PROMPT_BUDGET=2048
MAX_NEW_TOKENS=512
PROMPTS="/scratch/jiancongxiao/datasets/ultrafeedback_nashrs_2k_v1/train_2048.jsonl"
MODEL_CACHE="/scratch/jiancongxiao/cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUBMISSION_ROOT="/scratch/jiancongxiao/results/main-native-3b-full-submissions"
MANIFEST="$SUBMISSION_ROOT/$STAMP.tsv"
mkdir -p "$SUBMISSION_ROOT"

for required in "$PROMPTS" "$MODEL_CACHE" configs/main_native_full_3b_seed47.json; do
  if [[ ! -e "$required" ]]; then
    echo "Missing full 3B input: $required" >&2
    exit 3
  fi
done
if [[ "$(wc -l < "$PROMPTS" | tr -d '[:space:]')" != "$PROMPT_BUDGET" ]]; then
  echo "Full 3B prompt file must contain exactly $PROMPT_BUDGET rows" >&2
  exit 4
fi
PROMPT_SHA256="$(sha256sum "$PROMPTS" | awk '{print $1}')"
printf 'seed\tmethod\tsegment\tjob_id\tprompt_budget\tmax_new_tokens\tprompt_sha256\n' > "$MANIFEST"

for method in "${METHODS[@]}"; do
  job="$(qsub \
    -N "f3b_${method:0:8}" \
    -v "NASHRS_MAIN_METHOD=$method,NASHRS_SEED=$SEED" \
    cluster/hopper/main/main_3b_full.pbs)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$SEED" "$method" full "$job" "$PROMPT_BUDGET" "$MAX_NEW_TOKENS" "$PROMPT_SHA256" \
    | tee -a "$MANIFEST"
done

previous=""
for start in 0 6 12 18; do
  dependency=()
  if [[ -n "$previous" ]]; then
    dependency=( -W "depend=afterok:$previous" )
  fi
  job="$(qsub \
    -N "f3b_comal_${start}" \
    "${dependency[@]}" \
    -v "NASHRS_SEED=$SEED,NASHRS_COMAL_ITER_START=$start,NASHRS_COMAL_NUM_ITERS=6" \
    cluster/hopper/main/main_3b_comal_full.pbs)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$SEED" comal "$start-$((start + 5))" "$job" "$PROMPT_BUDGET" "$MAX_NEW_TOKENS" "$PROMPT_SHA256" \
    | tee -a "$MANIFEST"
  previous="$job"
done

echo "Full 3B submission manifest: $MANIFEST"
echo "Prompt SHA256: $PROMPT_SHA256"
