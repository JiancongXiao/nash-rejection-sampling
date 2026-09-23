#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="${NASHRS_CONFIG_FILE:-$SCRIPT_DIR/config.env}"
MODE="auto"
DRY_RUN=0
while (( $# )); do
  case "$1" in
    --scheduler) MODE="$2"; shift 2;;
    --config) CONFIG_FILE="$(realpath "$2")"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
[[ -f "$CONFIG_FILE" ]] || { echo "Missing config: $CONFIG_FILE" >&2; exit 3; }
# shellcheck disable=SC1090
source "$CONFIG_FILE"
export NASHRS_CONFIG_FILE="$CONFIG_FILE"
mkdir -p "$REPO_ROOT/logs" "$NASHRS_RESULT_ROOT/submissions"

if [[ "$MODE" == auto ]]; then
  if command -v sbatch >/dev/null; then MODE=slurm
  elif command -v qsub >/dev/null; then MODE=pbs
  else MODE=local
  fi
fi
case "$MODE" in slurm|pbs|local) ;; *) echo "scheduler must be auto, slurm, pbs, or local" >&2; exit 2;; esac

METHODS=(nash_rs nash_md egpo mpo)
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUBMISSION="$NASHRS_RESULT_ROOT/submissions/$STAMP.json"
declare -a IDS=()

if [[ "$MODE" == slurm ]]; then
  declare -a TRAIN_ARGS=() EVAL_ARGS=()
  if [[ -n "${NASHRS_SLURM_TRAIN_ARGS:-}" ]]; then
    read -r -a TRAIN_ARGS <<< "$NASHRS_SLURM_TRAIN_ARGS"
  fi
  if [[ -n "${NASHRS_SLURM_EVAL_ARGS:-}" ]]; then
    read -r -a EVAL_ARGS <<< "$NASHRS_SLURM_EVAL_ARGS"
  fi
  for method in "${METHODS[@]}"; do
    command=(sbatch --parsable)
    if (( ${#TRAIN_ARGS[@]} )); then command+=("${TRAIN_ARGS[@]}"); fi
    command+=(--job-name="tulu-pku-$method" \
      --export="ALL,METHOD=$method,NASHRS_CONFIG_FILE=$CONFIG_FILE" \
      "$SCRIPT_DIR/slurm_train.sbatch")
    if (( DRY_RUN )); then printf '%q ' "${command[@]}"; echo; IDS+=("DRY-$method")
    else job_id="$("${command[@]}")"; IDS+=("${job_id%%;*}"); fi
  done
  dependency="$(IFS=:; echo "${IDS[*]}")"
  command=(sbatch --parsable)
  if (( ${#EVAL_ARGS[@]} )); then command+=("${EVAL_ARGS[@]}"); fi
  command+=(--job-name=tulu-pku-eval --dependency="afterok:$dependency" \
    --export="ALL,NASHRS_CONFIG_FILE=$CONFIG_FILE" "$SCRIPT_DIR/slurm_eval.sbatch")
  if (( DRY_RUN )); then printf '%q ' "${command[@]}"; echo; EVAL_ID=DRY-eval
  else EVAL_ID="$("${command[@]}")"; fi
elif [[ "$MODE" == pbs ]]; then
  declare -a TRAIN_ARGS=() EVAL_ARGS=()
  if [[ -n "${NASHRS_PBS_TRAIN_ARGS:-}" ]]; then
    read -r -a TRAIN_ARGS <<< "$NASHRS_PBS_TRAIN_ARGS"
  fi
  if [[ -n "${NASHRS_PBS_EVAL_ARGS:-}" ]]; then
    read -r -a EVAL_ARGS <<< "$NASHRS_PBS_EVAL_ARGS"
  fi
  for method in "${METHODS[@]}"; do
    command=(qsub)
    if (( ${#TRAIN_ARGS[@]} )); then command+=("${TRAIN_ARGS[@]}"); fi
    command+=(-N "tulu-pku-${method:0:7}" \
      -v "METHOD=$method,NASHRS_CONFIG_FILE=$CONFIG_FILE" "$SCRIPT_DIR/pbs_train.pbs")
    if (( DRY_RUN )); then printf '%q ' "${command[@]}"; echo; IDS+=("DRY-$method")
    else IDS+=("$("${command[@]}")"); fi
  done
  dependency="$(IFS=:; echo "${IDS[*]}")"
  command=(qsub)
  if (( ${#EVAL_ARGS[@]} )); then command+=("${EVAL_ARGS[@]}"); fi
  command+=(-N tulu-pku-eval -W "depend=afterok:$dependency" \
    -v "NASHRS_CONFIG_FILE=$CONFIG_FILE" "$SCRIPT_DIR/pbs_eval.pbs")
  if (( DRY_RUN )); then printf '%q ' "${command[@]}"; echo; EVAL_ID=DRY-eval
  else EVAL_ID="$("${command[@]}")"; fi
else
  if (( DRY_RUN )); then
    for method in "${METHODS[@]}"; do echo "METHOD=$method bash $SCRIPT_DIR/train_8gpu.sh"; done
    echo "bash $SCRIPT_DIR/evaluate_8gpu.sh"
    IDS=(DRY-nash_rs DRY-nash_md DRY-egpo DRY-mpo); EVAL_ID=DRY-eval
  else
    for method in "${METHODS[@]}"; do METHOD="$method" bash "$SCRIPT_DIR/train_8gpu.sh"; IDS+=("local-$method"); done
    bash "$SCRIPT_DIR/evaluate_8gpu.sh"; EVAL_ID=local-eval
  fi
fi

python3 - "$SUBMISSION" "$MODE" "$EVAL_ID" "${IDS[@]}" <<'PY'
import json, pathlib, sys
path, scheduler, evaluation, *training = sys.argv[1:]
payload = {
    "scheduler": scheduler,
    "training": dict(zip(("nash_rs", "nash_md", "egpo", "mpo"), training)),
    "evaluation": evaluation,
}
pathlib.Path(path).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
echo "Submission manifest: $SUBMISSION"
