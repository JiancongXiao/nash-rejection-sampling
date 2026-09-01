#!/bin/bash

set -euo pipefail

if ! command -v qsub >/dev/null 2>&1; then
  source /etc/profile >/dev/null 2>&1 || true
  module load pbs
fi

SEED="${1:-47}"
if [[ "$SEED" != "47" ]]; then
  echo "The first full UltraFeedback run is pre-registered for seed 47" >&2
  exit 2
fi
METHODS=(reward_ppo self_play nash_md nash_rs mpo egpo)
SEGMENT_PROMPTS="${NASHRS_FULL_SEGMENT_PROMPTS:-8192}"
PROMPTS="/scratch/jiancongxiao/datasets/ultrafeedback_nashrs_full_v1/train_full.jsonl"
DATA_MANIFEST="/scratch/jiancongxiao/datasets/ultrafeedback_nashrs_full_v1/manifest.json"
MODEL_CACHE="/scratch/jiancongxiao/cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EXPERIMENT_ROOT="/scratch/jiancongxiao/results/main-native-3b-ultrafeedback-full-seed${SEED}-${STAMP}"
SUBMISSION_ROOT="/scratch/jiancongxiao/results/main-native-3b-ultrafeedback-full-submissions"
SUBMISSION_MANIFEST="$SUBMISSION_ROOT/$STAMP.tsv"
mkdir -p "$SUBMISSION_ROOT" "$EXPERIMENT_ROOT"

for required in "$PROMPTS" "$DATA_MANIFEST" "$MODEL_CACHE"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing full UltraFeedback input: $required" >&2
    exit 3
  fi
done
if (( SEGMENT_PROMPTS <= 0 || SEGMENT_PROMPTS % 2 != 0 )); then
  echo "Segment prompt count must be a positive even integer" >&2
  exit 4
fi

TOTAL_PROMPTS="$(wc -l < "$PROMPTS" | tr -d '[:space:]')"
MANIFEST_PROMPTS="$(python3 - "$DATA_MANIFEST" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["training_rows"])
PY
)"
if [[ "$TOTAL_PROMPTS" != "$MANIFEST_PROMPTS" ]]; then
  echo "Prompt row count $TOTAL_PROMPTS disagrees with manifest $MANIFEST_PROMPTS" >&2
  exit 5
fi
PROMPT_SHA256="$(sha256sum "$PROMPTS" | awk '{print $1}')"
printf 'seed\tmethod\tsegment_index\tsegment_start\tsegment_end\tfinal\tjob_id\tdepends_on\tprompt_sha256\tresult_root\n' > "$SUBMISSION_MANIFEST"

for method in "${METHODS[@]}"; do
  method_root="$EXPERIMENT_ROOT/$method"
  mkdir -p "$method_root"
  start=0
  segment_index=0
  previous_job=""
  while (( start < TOTAL_PROMPTS )); do
    end=$((start + SEGMENT_PROMPTS))
    if (( end >= TOTAL_PROMPTS )); then
      end="$TOTAL_PROMPTS"
      final=1
    else
      final=0
    fi
    job_name="uf3_${method:0:4}_${segment_index}"
    variables="NASHRS_MAIN_METHOD=$method,NASHRS_SEED=$SEED,NASHRS_MAIN_PROMPTS=$PROMPTS,NASHRS_MAIN_PROMPT_BUDGET=$end,NASHRS_SEGMENT_START=$start,NASHRS_SEGMENT_IS_FINAL=$final,NASHRS_RESULT_ROOT_OVERRIDE=$method_root"
    submit=(qsub -N "$job_name" -v "$variables")
    if [[ -n "$previous_job" ]]; then
      submit+=(-W "depend=afterok:$previous_job")
    fi
    submit+=(cluster/hopper/main/main_3b_full.pbs)
    job="$("${submit[@]}")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$SEED" "$method" "$segment_index" "$start" "$end" "$final" \
      "$job" "${previous_job:--}" "$PROMPT_SHA256" "$method_root" \
      | tee -a "$SUBMISSION_MANIFEST"
    previous_job="$job"
    start="$end"
    segment_index=$((segment_index + 1))
  done
done

python3 - "$EXPERIMENT_ROOT/experiment.json" "$DATA_MANIFEST" <<PY
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
data_manifest = json.load(open(sys.argv[2]))
path.write_text(json.dumps({
    "suite": "Main: Nash-RS vs method-native faithful baselines",
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "full_parameter_training": True,
    "seed": int("$SEED"),
    "methods": "${METHODS[*]}".split(),
    "excluded_method": "comal",
    "training_prompts": int("$TOTAL_PROMPTS"),
    "prompt_sha256": "$PROMPT_SHA256",
    "data_manifest": data_manifest,
    "segment_prompts": int("$SEGMENT_PROMPTS"),
    "resume_preserves_optimizer_state": True,
    "generation_max_tokens": 512,
    "nash_rs_tau": 0.5,
    "nash_rs_b1": 2,
    "nash_rs_b2": 2,
}, indent=2, sort_keys=True) + "\n")
PY

echo "Experiment root: $EXPERIMENT_ROOT"
echo "Submission manifest: $SUBMISSION_MANIFEST"
echo "Full training prompts: $TOTAL_PROMPTS"
echo "COMAL was not submitted or modified."
