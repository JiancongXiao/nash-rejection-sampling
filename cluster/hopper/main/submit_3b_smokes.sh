#!/bin/bash
set -euo pipefail

SEED="${1:-47}"
MODEL_CACHE="/scratch/jiancongxiao/cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct"
METHODS=(reward_ppo self_play nash_md nash_rs mpo egpo comal)
DEPENDENCY=()

module load pbs >/dev/null 2>&1 || true
if ! compgen -G "$MODEL_CACHE/snapshots/*/config.json" >/dev/null; then
  cache_job="$(qsub cluster/hopper/main/cache_qwen2p5_3b.pbs)"
  DEPENDENCY=( -W "depend=afterok:$cache_job" )
  echo "model_cache $cache_job"
else
  echo "model_cache already_ready"
fi

submission_root="/scratch/jiancongxiao/results/main-native-3b-smoke-submissions"
mkdir -p "$submission_root"
submission_file="$submission_root/seed${SEED}-$(date +%Y%m%d-%H%M%S).tsv"
printf "method\tjob_id\tsteps\tmodel\n" > "$submission_file"

for method in "${METHODS[@]}"; do
  steps=2
  if [[ "$method" == "reward_ppo" || "$method" == "nash_rs" ]]; then
    steps=4
  fi
  job="$(qsub \
    "${DEPENDENCY[@]}" \
    -N "q3_${method//_/-}" \
    -v "NASHRS_MAIN_METHOD=$method,NASHRS_MAIN_STEPS=$steps,NASHRS_SEED=$SEED" \
    cluster/hopper/main/main_3b_smoke.pbs)"
  printf "%s\t%s\t%s\t%s\n" "$method" "$job" "$steps" "Qwen/Qwen2.5-3B-Instruct" | tee -a "$submission_file"
done

echo "submission_file $submission_file"
