#!/bin/bash
set -euo pipefail

SEED="${1:-47}"
PREFLIGHT="${2:-0}"
SOURCE_JOB="${NASHRS_EXISTING_NASH_RS_JOB:-586852.hopper-m-02}"
mkdir -p "/scratch/jiancongxiao/results/openrlhf-7method-ultrafeedback-2k-seed${SEED}"

submit() {
  local name="$1"
  shift
  qsub -N "$name" -v "NASHRS_SEED=$SEED,NASHRS_PREFLIGHT=$PREFLIGHT,$*" \
    cluster/hopper/openrlhf_7method_ultrafeedback_2k.pbs
}

echo "Reusing completed Nash-RS seed-$SEED training job: $SOURCE_JOB"
echo "reward_ppo $(submit cmp_reward "NASHRS_METHOD=reward_ppo")"
echo "self_play_ppo $(submit cmp_self "NASHRS_METHOD=self_play_ppo")"
echo "nash_md_ppo $(submit cmp_nmd "NASHRS_METHOD=nash_md_ppo")"
echo "mpo_ppo $(submit cmp_mpo "NASHRS_METHOD=mpo_ppo")"
echo "comal_ppo $(submit cmp_comal "NASHRS_METHOD=comal_ppo")"
PREDICTION_JOB="$(submit cmp_egpred "NASHRS_METHOD=egpo_ppo,NASHRS_EGPO_PHASE=prediction")"
if [[ "$PREFLIGHT" == "1" ]]; then
  PREDICTOR_NAMESPACE="openrlhf-7method-preflight-seed${SEED}"
else
  PREDICTOR_NAMESPACE="openrlhf-7method-ultrafeedback-2k-seed${SEED}"
fi
PREDICTOR_PATH="/scratch/jiancongxiao/results/$PREDICTOR_NAMESPACE/egpo_ppo_prediction/$PREDICTION_JOB/actor"
CORRECTION_JOB="$(
  qsub \
    -N cmp_egcorr \
    -W "depend=afterok:$PREDICTION_JOB" \
    -v "NASHRS_SEED=$SEED,NASHRS_PREFLIGHT=$PREFLIGHT,NASHRS_METHOD=egpo_ppo,NASHRS_EGPO_PHASE=correction,NASHRS_OPPONENT_MODEL=$PREDICTOR_PATH" \
    cluster/hopper/openrlhf_7method_ultrafeedback_2k.pbs
)"
echo "egpo_prediction $PREDICTION_JOB"
echo "egpo_correction $CORRECTION_JOB (afterok:$PREDICTION_JOB)"
