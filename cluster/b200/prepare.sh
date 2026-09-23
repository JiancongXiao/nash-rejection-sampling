#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${NASHRS_CONFIG_FILE:-$SCRIPT_DIR/config.env}"
# shellcheck disable=SC1090
source "$CONFIG_FILE"
export HF_HOME="$NASHRS_CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
"$NASHRS_VENV_ROOT/native-trl-0.13.0/bin/python" "$SCRIPT_DIR/prepare_data.py"
