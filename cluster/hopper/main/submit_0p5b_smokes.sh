#!/bin/bash
set -euo pipefail

SEED="${1:-47}"
METHODS=(reward_ppo self_play nash_md nash_rs mpo egpo comal)

for method in "${METHODS[@]}"; do
  steps=2
  if [[ "$method" == "reward_ppo" || "$method" == "nash_rs" ]]; then
    steps=4
  fi
  job="$(qsub \
    -N "main_${method//_/-}" \
    -v "NASHRS_MAIN_METHOD=$method,NASHRS_MAIN_STEPS=$steps,NASHRS_SEED=$SEED" \
    cluster/hopper/main/main_0p5b_smoke.pbs)"
  echo "$method $job steps=$steps"
done
