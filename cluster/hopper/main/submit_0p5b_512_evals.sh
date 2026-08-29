#!/bin/bash

set -euo pipefail

METHODS=(reward_ppo self_play nash_md nash_rs mpo egpo comal)
SEEDS=(47 101 211)
TRAINING_MANIFEST="${1:-/scratch/jiancongxiao/results/main-native-0p5b-512-submissions/20260828T140954Z.tsv}"
EVAL_ROOT="/scratch/jiancongxiao/results/main-native-0p5b-512-evaluations"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EVAL_MANIFEST="$EVAL_ROOT/$STAMP.tsv"
mkdir -p "$EVAL_ROOT"

if [[ ! -f "$TRAINING_MANIFEST" ]]; then
  echo "Missing training submission manifest: $TRAINING_MANIFEST" >&2
  exit 2
fi

declare -A JOBS
while IFS=$(printf '\t') read -r seed method job_id prompt_budget max_tokens digest; do
  if [[ "$seed" == "seed" ]]; then
    continue
  fi
  JOBS["$seed,$method"]="$job_id"
done < "$TRAINING_MANIFEST"

printf 'seed\teval_job_id\tmodels_json\n' > "$EVAL_MANIFEST"
for seed in "${SEEDS[@]}"; do
  RUN_ROOT="/scratch/jiancongxiao/results/main-native-0p5b-512-seed${seed}"
  MODELS_JSON="$RUN_ROOT/evaluation_models.json"
  jq -n \
    --arg reward_ppo "$RUN_ROOT/reward_ppo/${JOBS[$seed,reward_ppo]}/actor" \
    --arg self_play "$RUN_ROOT/self_play/${JOBS[$seed,self_play]}/actor" \
    --arg nash_md "$RUN_ROOT/nash_md/${JOBS[$seed,nash_md]}/actor" \
    --arg nash_rs "$RUN_ROOT/nash_rs/${JOBS[$seed,nash_rs]}/actor" \
    --arg mpo "$RUN_ROOT/mpo/${JOBS[$seed,mpo]}/actor" \
    --arg egpo "$RUN_ROOT/egpo/${JOBS[$seed,egpo]}/actor" \
    --arg comal "$RUN_ROOT/comal/${JOBS[$seed,comal]}/actor" \
    '{reward_ppo: $reward_ppo, self_play: $self_play, nash_md: $nash_md,
      nash_rs: $nash_rs, mpo: $mpo, egpo: $egpo, comal: $comal}' \
    > "$MODELS_JSON"

  for method in "${METHODS[@]}"; do
    actor="$(jq -r --arg method "$method" '.[$method]' "$MODELS_JSON")"
    manifest="$RUN_ROOT/$method/${JOBS[$seed,$method]}/run_manifest.json"
    if [[ ! -f "$actor/config.json" || ! -f "$manifest" ]]; then
      echo "Seed $seed method $method is incomplete: $actor" >&2
      exit 3
    fi
    if [[ "$(jq -r '.parameter_update_verified' "$manifest")" != "true" ]]; then
      echo "Seed $seed method $method did not verify its update" >&2
      exit 4
    fi
  done

  eval_job="$(qsub \
    -N "e512_s${seed}" \
    -v "NASHRS_MAIN_MODELS_JSON=$MODELS_JSON,NASHRS_SEED=$seed" \
    cluster/hopper/main/evaluate_0p5b_512.pbs)"
  printf '%s\t%s\t%s\n' "$seed" "$eval_job" "$MODELS_JSON" | tee -a "$EVAL_MANIFEST"
done

echo "Evaluation submission manifest: $EVAL_MANIFEST"
