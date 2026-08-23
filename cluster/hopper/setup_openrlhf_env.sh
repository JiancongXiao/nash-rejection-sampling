#!/bin/bash

set -euo pipefail

: "${NASHRS_VENV:?NASHRS_VENV must point to a scratch virtual environment}"

if [[ -e "$NASHRS_VENV" && ! -x "$NASHRS_VENV/bin/python" ]]; then
  echo "Refusing to reuse incomplete environment: $NASHRS_VENV" >&2
  exit 2
fi

if [[ ! -x "$NASHRS_VENV/bin/python" ]]; then
  python -m venv "$NASHRS_VENV"
fi

source "$NASHRS_VENV/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Install vLLM first so its validated CUDA 12.8 PyTorch stack is present before
# flash-attn is compiled. OpenRLHF is installed last to resolve the remaining
# Ray, DeepSpeed, Transformers, and training dependencies.
python -m pip install "vllm==0.15.0"
MAX_JOBS="${MAX_JOBS:-8}" python -m pip install \
  "flash-attn==2.8.3" --no-build-isolation
python -m pip install "openrlhf==0.9.3"

python -m pip install -e .
python -m pip check

python - <<'PY'
import importlib.metadata as metadata
import torch

packages = ("openrlhf", "vllm", "deepspeed", "ray", "transformers", "flash-attn")
for package in packages:
    print(f"{package}: {metadata.version(package)}")
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the OpenRLHF environment")
print("gpu:", torch.cuda.get_device_name(0))
PY

mkdir -p "${NASHRS_ENV_MANIFEST_DIR}"
python -m pip freeze > \
  "${NASHRS_ENV_MANIFEST_DIR}/openrlhf-0.9.3-vllm-0.15.0.freeze.txt"

PYTHONPATH=src python -m unittest discover -s tests -v

echo "OpenRLHF environment ready: $NASHRS_VENV"
