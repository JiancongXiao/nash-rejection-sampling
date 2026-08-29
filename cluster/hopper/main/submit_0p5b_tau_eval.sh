#!/bin/bash

set -euo pipefail

SEED="${1:-47}"
SWEEP_ROOT="/scratch/jiancongxiao/results/main-native-0p5b-tau-sweep-v2-seed${SEED}"
BASELINE_ROOT="/scratch/jiancongxiao/results/main-native-0p5b-512-seed${SEED}"
SUBMISSION_ROOT="/scratch/jiancongxiao/results/main-native-0p5b-tau-sweep-v2-evaluations"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MODELS_JSON="$SWEEP_ROOT/evaluation_models.json"
MANIFEST="$SUBMISSION_ROOT/$STAMP.tsv"
PROMPTS="/scratch/jiancongxiao/datasets/ultrafeedback_nashrs_2k_v1/selection_256.jsonl"
mkdir -p "$SUBMISSION_ROOT"

if [[ "$SEED" != 47 ]]; then
  echo "The current tau sweep only has seed 47, got seed $SEED" >&2
  exit 2
fi

jq -n \
  --arg reward_ppo "$BASELINE_ROOT/reward_ppo/588763.hopper-m-02/actor" \
  --arg self_play "$BASELINE_ROOT/self_play/588764.hopper-m-02/actor" \
  --arg nash_md "$BASELINE_ROOT/nash_md/588765.hopper-m-02/actor" \
  --arg mpo "$BASELINE_ROOT/mpo/588767.hopper-m-02/actor" \
  --arg egpo "$BASELINE_ROOT/egpo/588768.hopper-m-02/actor" \
  --arg comal "$BASELINE_ROOT/comal/588769.hopper-m-02/actor" \
  --arg tau_0p25 "$SWEEP_ROOT/tau_0p25_retry1024/589883.hopper-m-02/actor" \
  --arg tau_0p5 "$SWEEP_ROOT/tau_0p5/589745.hopper-m-02/actor" \
  --arg tau_0p75 "$SWEEP_ROOT/tau_0p75/589753.hopper-m-02/actor" \
  --arg tau_1p0 "$BASELINE_ROOT/nash_rs/588766.hopper-m-02/actor" \
  --arg tau_2p0 "$SWEEP_ROOT/tau_2p0/589747.hopper-m-02/actor" \
  --arg tau_4p0 "$SWEEP_ROOT/tau_4p0/589748.hopper-m-02/actor" \
  '{reward_ppo: $reward_ppo, self_play: $self_play, nash_md: $nash_md,
    mpo: $mpo, egpo: $egpo, comal: $comal,
    nash_rs_tau_0p25: $tau_0p25, nash_rs_tau_0p5: $tau_0p5,
    nash_rs_tau_0p75: $tau_0p75, nash_rs_tau_1p0: $tau_1p0,
    nash_rs_tau_2p0: $tau_2p0, nash_rs_tau_4p0: $tau_4p0}' \
  > "$MODELS_JSON"

if [[ "$(jq 'length' "$MODELS_JSON")" != 12 ]]; then
  echo "Expected 12 models in $MODELS_JSON" >&2
  exit 3
fi
while IFS= read -r actor; do
  if [[ ! -f "$actor/config.json" ]]; then
    echo "Missing evaluation actor: $actor" >&2
    exit 4
  fi
done < <(jq -r '.[]' "$MODELS_JSON")

PROMPT_SHA256="$(sha256sum "$PROMPTS" | awk '{print $1}')"
RESULTS_NAMESPACE="main-native-0p5b-tau-sweep-v2-seed${SEED}/selection-evaluation"
JOB_ID="$(qsub \
  -N "tau_eval_s${SEED}" \
  -v "NASHRS_MAIN_MODELS_JSON=$MODELS_JSON,NASHRS_SEED=$SEED,NASHRS_EVAL_EXPECTED_MODEL_COUNT=12,NASHRS_EVAL_RESULTS_NAMESPACE=$RESULTS_NAMESPACE,NASHRS_EVAL_PROMPTS=$PROMPTS,NASHRS_EVAL_MAX_PROMPTS=256,NASHRS_EVAL_MAX_NEW_TOKENS=512" \
  cluster/hopper/main/evaluate_0p5b_512.pbs)"

printf 'seed\teval_job_id\tmodel_count\tprompt_count\tprompt_sha256\tmodels_json\tresults_namespace\n' > "$MANIFEST"
printf '%s\t%s\t12\t256\t%s\t%s\t%s\n' \
  "$SEED" "$JOB_ID" "$PROMPT_SHA256" "$MODELS_JSON" "$RESULTS_NAMESPACE" \
  | tee -a "$MANIFEST"
echo "Tau evaluation submission manifest: $MANIFEST"
