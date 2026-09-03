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

MODE="${1:-smoke}"
SEED="${2:-47}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "usage: $0 [smoke|full] [seed]" >&2
  exit 2
fi
if [[ "$SEED" != "47" ]]; then
  echo "The first full 3B gate is pre-registered for seed 47" >&2
  exit 2
fi

SUBMISSION_ROOT="/scratch/jiancongxiao/results/main-native-3b-comal-single-submissions"
mkdir -p "$SUBMISSION_ROOT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="$SUBMISSION_ROOT/${MODE}-${STAMP}.tsv"
printf 'seed\tmethod\trun_kind\tsegment\tjob_id\tngpus\tglobal_batch\n' > "$MANIFEST"

if [[ "$MODE" == "smoke" ]]; then
  job="$("$QSUB" \
    -N "c3b_1g_smoke" \
    -l "walltime=04:00:00" \
    -l "select=1:ngpus=1:ncpus=12:mem=225GB" \
    -v "NASHRS_SEED=$SEED,NASHRS_COMAL_RUN_KIND=smoke,NASHRS_COMAL_ITER_START=0,NASHRS_COMAL_NUM_ITERS=1,NASHRS_COMAL_NUM_GPUS=1,NASHRS_MAIN_PROMPT_BUDGET=64,NASHRS_MAIN_GENERATE_MAX_LEN=128" \
    cluster/hopper/main/main_3b_comal_full.pbs)"
  printf '%s\tcomal\tsmoke\t0\t%s\t1\t32\n' "$SEED" "$job" | tee -a "$MANIFEST"
else
  previous=""
  for start in $(seq 0 2 22); do
    dependency=()
    if [[ -n "$previous" ]]; then
      dependency=( -W "depend=afterok:$previous" )
    fi
    job="$("$QSUB" \
      -N "f3b_c1g_s${start}" \
      -l "walltime=24:00:00" \
      -l "select=1:ngpus=1:ncpus=12:mem=225GB" \
      "${dependency[@]}" \
      -v "NASHRS_SEED=$SEED,NASHRS_COMAL_RUN_KIND=full,NASHRS_COMAL_ITER_START=$start,NASHRS_COMAL_NUM_ITERS=2,NASHRS_COMAL_NUM_GPUS=1" \
      cluster/hopper/main/main_3b_comal_full.pbs)"
    printf '%s\tcomal\tfull\t%s-%s\t%s\t1\t32\n' \
      "$SEED" "$start" "$((start + 1))" "$job" | tee -a "$MANIFEST"
    previous="$job"
  done
fi

echo "COMAL 3B single-GPU submission manifest: $MANIFEST"
