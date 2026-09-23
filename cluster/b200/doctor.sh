#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="${NASHRS_CONFIG_FILE:-$SCRIPT_DIR/config.env}"
# shellcheck disable=SC1090
source "$CONFIG_FILE"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT"
for required in \
  "$NASHRS_VENV_ROOT/openrlhf-0.9.3-vllm-0.15.0/bin/python" \
  "$NASHRS_VENV_ROOT/native-trl-0.13.0/bin/python" \
  "$NASHRS_DATA_ROOT/manifest.json"; do
  [[ -e "$required" ]] || { echo "Missing: $required" >&2; exit 2; }
done
"$NASHRS_VENV_ROOT/native-trl-0.13.0/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("visible GPUs", torch.cuda.device_count())
assert torch.cuda.device_count() == 8
for i in range(8):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
    assert torch.cuda.get_device_capability(i)[0] >= 10
PY
NCCL_DEBUG=WARN "$NASHRS_VENV_ROOT/native-trl-0.13.0/bin/torchrun" \
  --standalone --nproc-per-node=8 -m examples.b200.nccl_smoke
echo "Eight-GPU B200/NCCL doctor passed"
