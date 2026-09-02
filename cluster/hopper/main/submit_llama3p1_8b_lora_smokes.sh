#!/bin/bash

set -euo pipefail
if ! command -v qsub >/dev/null 2>&1; then
  source /etc/profile >/dev/null 2>&1 || true
  module load pbs
fi

SEED="${1:-47}"
METHODS=(nash_rs nash_md mpo egpo)
MODEL_CACHE="/scratch/jiancongxiao/cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct"
TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
SUBMISSION_ROOT="/scratch/jiancongxiao/results/main-native-llama3p1-8b-lora-smoke-submissions"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="$SUBMISSION_ROOT/$STAMP.tsv"
mkdir -p "$SUBMISSION_ROOT"

if compgen -G "$MODEL_CACHE/snapshots/*/config.json" >/dev/null; then
  CACHE_JOB=""
else
  if [[ -z "${HF_TOKEN:-}" && ! -s "$TOKEN_PATH" ]]; then
    echo "Llama 3.1 is gated and is not cached." >&2
    echo "Accept Meta's license and place a Hugging Face read token at: $TOKEN_PATH" >&2
    exit 2
  fi
  CACHE_JOB="$(qsub cluster/hopper/main/cache_llama3p1_8b.pbs)"
fi

printf 'seed\tmethod\tjob_id\tcache_job\toptimizer_steps\tlora_rank\tlora_alpha\tlora_dropout\ttau\n' > "$MANIFEST"
for method in "${METHODS[@]}"; do
  dependency=()
  if [[ -n "$CACHE_JOB" ]]; then dependency=(-W "depend=afterok:$CACHE_JOB"); fi
  case "$method" in
    nash_rs) short=nrs ;;
    nash_md) short=nmd ;;
    mpo) short=mpo ;;
    egpo) short=egpo ;;
  esac
  job="$(qsub "${dependency[@]}" \
    -N "l8s_$short" \
    -v "NASHRS_MAIN_METHOD=$method,NASHRS_SEED=$SEED,NASHRS_TAU=0.5" \
    cluster/hopper/main/main_llama3p1_8b_lora_smoke.pbs)"
  printf '%s\t%s\t%s\t%s\t2\t16\t32\t0.05\t%s\n' \
    "$SEED" "$method" "$job" "${CACHE_JOB:--}" \
    "$([[ "$method" == "nash_rs" ]] && echo 0.5 || echo n/a)" | tee -a "$MANIFEST"
done

echo "Cache job: ${CACHE_JOB:-already cached}"
echo "Submission manifest: $MANIFEST"

