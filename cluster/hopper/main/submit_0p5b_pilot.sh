#!/bin/bash

set -euo pipefail

SEED="${1:-47}"
STEPS="${2:-16}"
for method in reward_ppo self_play nash_md nash_rs mpo egpo comal; do
  qsub -v "NASHRS_MAIN_METHOD=$method,NASHRS_MAIN_STEPS=$STEPS,NASHRS_SEED=$SEED" \
    cluster/hopper/main/main_0p5b_pilot.pbs
done
