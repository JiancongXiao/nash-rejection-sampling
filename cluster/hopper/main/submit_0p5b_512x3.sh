#!/bin/bash

set -euo pipefail

METHODS=(reward_ppo self_play nash_md nash_rs mpo egpo comal)
SEEDS=(47 101 211)
PROMPT_BUDGET="${NASHRS_MAIN_PROMPT_BUDGET:-512}"
MAX_NEW_TOKENS="${NASHRS_MAIN_GENERATE_MAX_LEN:-512}"
DATA_ROOT="/scratch/jiancongxiao/datasets/ultrafeedback_nashrs_2k_v1"
SOURCE_PROMPTS="$DATA_ROOT/train_2048.jsonl"
PROMPTS="$DATA_ROOT/train_${PROMPT_BUDGET}_main_scaling_gate.jsonl"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUBMISSION_ROOT="/scratch/jiancongxiao/results/main-native-0p5b-512-submissions"
MANIFEST="$SUBMISSION_ROOT/$STAMP.tsv"
mkdir -p "$SUBMISSION_ROOT"

if [[ ! -f "$SOURCE_PROMPTS" ]]; then
  echo "Missing source prompt data: $SOURCE_PROMPTS" >&2
  exit 2
fi
if (( $(wc -l < "$SOURCE_PROMPTS") < PROMPT_BUDGET )); then
  echo "Source prompt data has fewer than $PROMPT_BUDGET rows" >&2
  exit 3
fi
if [[ ! -f "$PROMPTS" ]]; then
  temporary_prompts="${PROMPTS}.tmp.$$"
  head -n "$PROMPT_BUDGET" "$SOURCE_PROMPTS" > "$temporary_prompts"
  mv "$temporary_prompts" "$PROMPTS"
fi
if [[ "$(wc -l < "$PROMPTS" | tr -d '[:space:]')" != "$PROMPT_BUDGET" ]]; then
  echo "Prompt pool does not contain exactly $PROMPT_BUDGET rows: $PROMPTS" >&2
  exit 4
fi
PROMPT_SHA256="$(sha256sum "$PROMPTS" | awk '{print $1}')"

printf 'seed\tmethod\tjob_id\tprompt_budget\tmax_new_tokens\tprompt_sha256\n' > "$MANIFEST"

for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
    short_method="${method:0:8}"
    job_id="$(qsub \
      -N "m512_${short_method}" \
      -v "NASHRS_MAIN_METHOD=$method,NASHRS_SEED=$seed,NASHRS_MAIN_PROMPT_BUDGET=$PROMPT_BUDGET,NASHRS_MAIN_GENERATE_MAX_LEN=$MAX_NEW_TOKENS,NASHRS_MAIN_PROMPTS=$PROMPTS" \
      cluster/hopper/main/main_0p5b_512.pbs)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$seed" "$method" "$job_id" "$PROMPT_BUDGET" "$MAX_NEW_TOKENS" "$PROMPT_SHA256" \
      | tee -a "$MANIFEST"
  done
done

echo "Submission manifest: $MANIFEST"
echo "Prompt pool: $PROMPTS"
echo "Prompt SHA256: $PROMPT_SHA256"
