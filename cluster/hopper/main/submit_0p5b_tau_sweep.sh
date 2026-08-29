#!/bin/bash

set -euo pipefail

TAUS=(0.25 0.5 1.0 2.0 4.0)
SEED="${NASHRS_SEED:-47}"
PROMPT_BUDGET="${NASHRS_MAIN_PROMPT_BUDGET:-512}"
MAX_NEW_TOKENS="${NASHRS_MAIN_GENERATE_MAX_LEN:-512}"
DATA_ROOT="/scratch/jiancongxiao/datasets/ultrafeedback_nashrs_2k_v1"
PROMPTS="$DATA_ROOT/train_${PROMPT_BUDGET}_main_scaling_gate.jsonl"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUBMISSION_ROOT="/scratch/jiancongxiao/results/main-native-0p5b-tau-sweep-submissions"
MANIFEST="$SUBMISSION_ROOT/$STAMP.tsv"
mkdir -p "$SUBMISSION_ROOT"

if [[ ! -f "$PROMPTS" ]]; then
  echo "Missing fixed prompt pool: $PROMPTS" >&2
  exit 2
fi
if [[ "$(wc -l < "$PROMPTS" | tr -d '[:space:]')" != "$PROMPT_BUDGET" ]]; then
  echo "Prompt pool does not contain exactly $PROMPT_BUDGET rows: $PROMPTS" >&2
  exit 3
fi
PROMPT_SHA256="$(sha256sum "$PROMPTS" | awk '{print $1}')"

printf 'tau\tseed\tjob_id\tprompt_budget\tmax_new_tokens\tprompt_sha256\tresult_namespace\n' > "$MANIFEST"

for tau in "${TAUS[@]}"; do
  tau_slug="${tau/./p}"
  namespace="main-native-0p5b-tau-sweep-seed${SEED}/tau_${tau_slug}"
  job_id="$(qsub \
    -N "tau_${tau_slug}" \
    -v "NASHRS_MAIN_METHOD=nash_rs,NASHRS_SEED=$SEED,NASHRS_TAU=$tau,NASHRS_MAIN_PROMPT_BUDGET=$PROMPT_BUDGET,NASHRS_MAIN_GENERATE_MAX_LEN=$MAX_NEW_TOKENS,NASHRS_MAIN_PROMPTS=$PROMPTS,NASHRS_RESULTS_NAMESPACE_OVERRIDE=$namespace" \
    cluster/hopper/main/main_0p5b_512.pbs)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$tau" "$SEED" "$job_id" "$PROMPT_BUDGET" "$MAX_NEW_TOKENS" \
    "$PROMPT_SHA256" "$namespace" | tee -a "$MANIFEST"
done

echo "Submission manifest: $MANIFEST"
echo "Prompt pool: $PROMPTS"
echo "Prompt SHA256: $PROMPT_SHA256"
