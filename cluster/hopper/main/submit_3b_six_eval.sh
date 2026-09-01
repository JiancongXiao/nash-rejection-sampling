#!/bin/bash

set -euo pipefail

QSUB="$(command -v qsub || true)"
if [[ -z "$QSUB" ]]; then
  QSUB="/cm/shared/apps/pbspro/current/bin/qsub"
fi
if [[ ! -x "$QSUB" ]]; then
  echo "qsub is unavailable; load PBS or check $QSUB" >&2
  exit 3
fi

if (( $# != 7 )); then
  echo "usage: $0 SEED REWARD SELF_PLAY NASH_MD NASH_RS MPO EGPO" >&2
  exit 2
fi

SEED="$1"
shift
METHODS=(reward_ppo self_play nash_md nash_rs mpo egpo)
JOB_IDS=("$@")
ROOT="/scratch/jiancongxiao/results/main-native-3b-full-seed${SEED}"
INPUT_ROOT="$ROOT/evaluation-inputs"
mkdir -p "$INPUT_ROOT"
MODELS_JSON="$INPUT_ROOT/models-six.json"

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
  --arg reward_ppo "${paths[0]}" \
  --arg self_play "${paths[1]}" \
  --arg nash_md "${paths[2]}" \
  --arg nash_rs "${paths[3]}" \
  --arg mpo "${paths[4]}" \
  --arg egpo "${paths[5]}" \
  '{reward_ppo: $reward_ppo, self_play: $self_play, nash_md: $nash_md,
    nash_rs: $nash_rs, mpo: $mpo, egpo: $egpo}' > "$MODELS_JSON"

job="$($QSUB \
  -N eval3b_six \
  -v "NASHRS_MAIN_MODELS_JSON=$MODELS_JSON,NASHRS_SEED=$SEED,NASHRS_EVAL_EXPECTED_MODEL_COUNT=6" \
  cluster/hopper/main/evaluate_3b_full.pbs)"

echo "3B six-method evaluation: $job"
echo "Models: $MODELS_JSON"
