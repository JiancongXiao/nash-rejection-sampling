#!/bin/bash

set -euo pipefail
if ! command -v qsub >/dev/null 2>&1; then
  source /etc/profile >/dev/null 2>&1 || true
  module load pbs
fi

SEED=47
METHODS=(nash_rs nash_md mpo egpo)
SOURCE="/scratch/jiancongxiao/datasets/ultrafeedback_nashrs_full_v1/train_full.jsonl"
PROMPTS="/scratch/jiancongxiao/datasets/ultrafeedback_nashrs_full_v1/train_8192_main_7b_seed47.jsonl"
MODEL_CACHE="/scratch/jiancongxiao/cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct"
SUBMISSION_ROOT="/scratch/jiancongxiao/results/main-native-llama3p1-8b-lora-8192-submissions"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="$SUBMISSION_ROOT/$STAMP.tsv"
mkdir -p "$SUBMISSION_ROOT"

if [[ ! -f "$SOURCE" ]] || (( $(wc -l < "$SOURCE") < 8192 )); then
  echo "UltraFeedback training pool has fewer than 8192 rows: $SOURCE" >&2
  exit 2
fi
if [[ ! -f "$PROMPTS" ]]; then
  temporary="${PROMPTS}.tmp.$$"
  head -n 8192 "$SOURCE" > "$temporary"
  mv "$temporary" "$PROMPTS"
fi
if [[ "$(wc -l < "$PROMPTS" | tr -d '[:space:]')" != "8192" ]]; then
  echo "Frozen Llama prompt file must contain exactly 8192 rows" >&2
  exit 3
fi
PROMPT_SHA256="$(sha256sum "$PROMPTS" | awk '{print $1}')"

if compgen -G "$MODEL_CACHE/snapshots/*/config.json" >/dev/null; then
  CACHE_JOB=""
else
  CACHE_JOB="$(qsub cluster/hopper/main/cache_llama3p1_8b.pbs)"
fi

printf 'seed\tmethod\tjob_id\tcache_job\tprompt_budget\tprompt_sha256\tlora_rank\tlora_alpha\tlora_dropout\ttau\n' > "$MANIFEST"
for method in "${METHODS[@]}"; do
  dependency=()
  if [[ -n "$CACHE_JOB" ]]; then dependency=(-W "depend=afterok:$CACHE_JOB"); fi
  job="$(qsub "${dependency[@]}" \
    -N "l8b_${method:0:7}" \
    -v "NASHRS_MAIN_METHOD=$method,NASHRS_SEED=$SEED,NASHRS_TAU=0.5" \
    cluster/hopper/main/main_llama3p1_8b_lora_8192.pbs)"
  printf '%s\t%s\t%s\t%s\t8192\t%s\t16\t32\t0.05\t%s\n' \
    "$SEED" "$method" "$job" "${CACHE_JOB:--}" "$PROMPT_SHA256" \
    "$([[ "$method" == "nash_rs" ]] && echo 0.5 || echo n/a)" | tee -a "$MANIFEST"
done

echo "Cache job: ${CACHE_JOB:-already cached}"
echo "Submission manifest: $MANIFEST"
echo "Prompt SHA256: $PROMPT_SHA256"

