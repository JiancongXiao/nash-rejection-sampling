#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="${NASHRS_CONFIG_FILE:-$SCRIPT_DIR/config.env}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing $CONFIG_FILE; copy config.env.example to config.env first." >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"

PYTHON_BIN="${NASHRS_PYTHON_BIN:-python3.12}"
command -v "$PYTHON_BIN" >/dev/null || {
  echo "Python 3.12 is required; set NASHRS_PYTHON_BIN to its executable." >&2
  exit 3
}
mkdir -p "$NASHRS_VENV_ROOT" "$NASHRS_CACHE_ROOT/pip" "$NASHRS_SOURCE_ROOT"

unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT UV_CONSTRAINT
export PIP_CACHE_DIR="$NASHRS_CACHE_ROOT/pip"

OPENRLHF_ENV="$NASHRS_VENV_ROOT/openrlhf-0.9.3-vllm-0.15.0"
NATIVE_ENV="$NASHRS_VENV_ROOT/native-trl-0.13.0"

if [[ ! -x "$OPENRLHF_ENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$OPENRLHF_ENV"
  "$OPENRLHF_ENV/bin/python" -m pip install --upgrade pip setuptools wheel
  "$OPENRLHF_ENV/bin/python" -m pip install "vllm==0.15.0"
  if [[ "${NASHRS_SKIP_FLASH_ATTN_BUILD:-0}" != "1" ]]; then
    MAX_JOBS="${MAX_JOBS:-16}" "$OPENRLHF_ENV/bin/python" -m pip install \
      "flash-attn==2.8.3" --no-build-isolation
  fi
  "$OPENRLHF_ENV/bin/python" -m pip install "openrlhf==0.9.3"
  "$OPENRLHF_ENV/bin/python" -m pip uninstall -y \
    nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base \
    nvidia-cutlass-dsl-libs-core nvidia-cutlass-dsl-libs-cu12 || true
  "$OPENRLHF_ENV/bin/python" -m pip install "nvidia-cutlass-dsl==4.3.4"
fi
"$OPENRLHF_ENV/bin/python" -m pip install -e "$REPO_ROOT"
"$OPENRLHF_ENV/bin/python" -m pip check

if [[ ! -x "$NATIVE_ENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$NATIVE_ENV"
  "$NATIVE_ENV/bin/python" -m pip install --upgrade pip setuptools wheel
  # Install the same CUDA/PyTorch family selected by the vLLM wheel.
  TORCH_VERSION="$($OPENRLHF_ENV/bin/python -c 'import torch; print(torch.__version__.split("+")[0])')"
  "$NATIVE_ENV/bin/python" -m pip install \
    "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cu128
  "$NATIVE_ENV/bin/python" -m pip install -r \
    "$SCRIPT_DIR/requirements-b200-native.txt"
fi
"$NATIVE_ENV/bin/python" -m pip install -e "$REPO_ROOT"
"$NATIVE_ENV/bin/python" -m pip check

EGPO_ROOT="$NASHRS_SOURCE_ROOT/egpo"
if [[ ! -d "$EGPO_ROOT/.git" ]]; then
  git clone https://github.com/zhourunlong/EGPO.git "$EGPO_ROOT"
fi
git -C "$EGPO_ROOT" fetch --tags origin
# This is the repository revision pinned by nashrs.main_suite.EGPO_SOURCE.
EGPO_COMMIT="$($NATIVE_ENV/bin/python - <<'PY'
from nashrs.main_suite import EGPO_SOURCE
print(EGPO_SOURCE.commit)
PY
)"
git -C "$EGPO_ROOT" checkout --detach "$EGPO_COMMIT"

"$OPENRLHF_ENV/bin/python" - <<'PY'
import importlib.metadata as md
import torch
for name in ("openrlhf", "vllm", "deepspeed", "ray", "transformers"):
    print(f"{name}: {md.version(name)}")
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit(f"setup validation requires one node with 8 visible GPUs; found {torch.cuda.device_count()}")
for index in range(8):
    major, minor = torch.cuda.get_device_capability(index)
    print(index, torch.cuda.get_device_name(index), (major, minor))
    if major < 10:
        raise SystemExit("expected NVIDIA Blackwell/B200 (compute capability >= 10.0)")
PY

"$NATIVE_ENV/bin/python" -m pytest "$REPO_ROOT/tests" -q
"$OPENRLHF_ENV/bin/python" -m pip freeze > "$NASHRS_EXPERIMENT_ROOT/openrlhf.freeze.txt"
"$NATIVE_ENV/bin/python" -m pip freeze > "$NASHRS_EXPERIMENT_ROOT/native.freeze.txt"
echo "B200 environments are ready under $NASHRS_VENV_ROOT"
