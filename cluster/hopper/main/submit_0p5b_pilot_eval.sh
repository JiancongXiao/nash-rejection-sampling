#!/bin/bash

set -euo pipefail

if [[ "$#" -ne 8 ]]; then
  echo "usage: $0 SEED reward_ppo self_play nash_md nash_rs mpo egpo comal" >&2
  exit 2
fi
SEED="$1"
shift
METHODS=(reward_ppo self_play nash_md nash_rs mpo egpo comal)
ROOT="/scratch/jiancongxiao/results/main-native-pilot-0p5b-seed${SEED}"
MODELS_JSON="$ROOT/evaluation_models.json"
mkdir -p "$ROOT"

python3 - "$MODELS_JSON" "$ROOT" "${METHODS[@]}" -- "$@" <<'PY'
import json
import sys

output, root, *values = sys.argv[1:]
separator = values.index("--")
methods = values[:separator]
job_ids = values[separator + 1 :]
if len(methods) != len(job_ids):
    raise SystemExit("method and job-id counts differ")
models = {
    method: f"{root}/{method}/{job_id}.hopper-m-02/actor"
    for method, job_id in zip(methods, job_ids)
}
with open(output, "w") as handle:
    json.dump(models, handle, indent=2)
    handle.write("\n")
print(json.dumps(models, indent=2))
PY

qsub -v "NASHRS_MAIN_MODELS_JSON=$MODELS_JSON,NASHRS_SEED=$SEED" \
  cluster/hopper/main/evaluate_0p5b_pilot.pbs
