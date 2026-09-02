#!/bin/bash

set -euo pipefail
QSUB="$(command -v qsub || true)"
if [[ -z "$QSUB" ]]; then QSUB="/cm/shared/apps/pbspro/current/bin/qsub"; fi
if [[ ! -x "$QSUB" ]]; then
  echo "qsub is unavailable; load PBS first" >&2
  exit 3
fi
if (( $# != 5 )); then
  echo "usage: $0 SEED NASH_RS NASH_MD MPO EGPO" >&2
  exit 2
fi

SEED="$1"
shift
METHODS=(nash_rs nash_md mpo egpo)
JOB_IDS=("$@")
ROOT="/scratch/jiancongxiao/results/main-native-llama3p1-8b-lora-8192-seed${SEED}"
INPUT_ROOT="$ROOT/evaluation-inputs"
MODELS_JSON="$INPUT_ROOT/models-four.json"
mkdir -p "$INPUT_ROOT"

paths=()
for index in "${!METHODS[@]}"; do
  method="${METHODS[$index]}"
  job_number="${JOB_IDS[$index]%%.*}"
  run="$ROOT/$method/${job_number}.hopper-m-02"
  actor="$run/actor"
  manifest="$run/run_manifest.json"
  if [[ ! -f "$actor/config.json" || ! -f "$manifest" ]]; then
    echo "Missing completed $method run: $run" >&2
    exit 4
  fi
  if ! jq -e '.parameter_update_verified == true' "$manifest" >/dev/null; then
    echo "Parameter update was not verified for $method: $manifest" >&2
    exit 5
  fi
  paths+=("$actor")
done

jq -n \
  --arg nash_rs "${paths[0]}" \
  --arg nash_md "${paths[1]}" \
  --arg mpo "${paths[2]}" \
  --arg egpo "${paths[3]}" \
  '{nash_rs: $nash_rs, nash_md: $nash_md, mpo: $mpo, egpo: $egpo}' \
  > "$MODELS_JSON"

job="$($QSUB \
  -N eval_l8b \
  -v "NASHRS_MAIN_MODELS_JSON=$MODELS_JSON,NASHRS_SEED=$SEED" \
  cluster/hopper/main/evaluate_llama3p1_8b_8192.pbs)"
echo "Llama 8B four-method evaluation: $job"
echo "Models: $MODELS_JSON"

