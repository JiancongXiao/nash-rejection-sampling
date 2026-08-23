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

if [[ -n "${TMPDIR:-}" ]]; then
  mkdir -p "$TMPDIR"
fi

# NVIDIA NGC images export a global pip constraint file for their preinstalled
# stack. A clean virtual environment must not inherit those pins: for example,
# the 25.03 image pins pydantic 2.10.6 while vLLM 0.15 requires >=2.12.
for constraint_var in PIP_CONSTRAINT PIP_BUILD_CONSTRAINT UV_CONSTRAINT; do
  if [[ -n "${!constraint_var:-}" ]]; then
    echo "Ignoring container-level $constraint_var=${!constraint_var}"
    unset "$constraint_var"
  fi
done

python -m pip install --upgrade pip "setuptools==80.9.0" wheel

# Install vLLM first so its validated CUDA 12.8 PyTorch stack is present before
# flash-attn is compiled. OpenRLHF is installed last to resolve the remaining
# Ray, DeepSpeed, Transformers, and training dependencies.
python -m pip install "vllm==0.15.0"
MAX_JOBS="${MAX_JOBS:-8}" python -m pip install \
  "flash-attn==2.8.3" --no-build-isolation
python -m pip install "openrlhf==0.9.3"

# vLLM's dependency resolver may install a newer split CUTLASS DSL family.
# FlashInfer 0.6.1 requires CUTLASS DSL >=4.3.4, so pin that minimum and remove
# the split family before installing the monolithic CUDA 12 wheel. The H200
# path uses vLLM's bundled FlashAttention 3 extension; flash-attn's optional
# FA4/CuTe modules target Blackwell and are not imported by this validation.
python -m pip uninstall -y \
  nvidia-cutlass-dsl \
  nvidia-cutlass-dsl-libs-base \
  nvidia-cutlass-dsl-libs-core \
  nvidia-cutlass-dsl-libs-cu12
python -m pip install "nvidia-cutlass-dsl==4.3.4"

python -m pip install -e .
python -m pip check

python - <<'PY'
import importlib.metadata as metadata
import torch

packages = (
    "openrlhf",
    "vllm",
    "deepspeed",
    "ray",
    "transformers",
    "flash-attn",
    "nvidia-cutlass-dsl",
)
for package in packages:
    print(f"{package}: {metadata.version(package)}")
from vllm.vllm_flash_attn import is_fa_version_supported

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the OpenRLHF environment")
print("gpu:", torch.cuda.get_device_name(0))
if torch.cuda.get_device_capability(0)[0] != 9:
    raise SystemExit("Expected an SM90 Hopper GPU for the FA3 validation")
if not is_fa_version_supported(3):
    raise SystemExit("vLLM's bundled FlashAttention 3 extension is unavailable")
print("vLLM FlashAttention 3: available")
PY

mkdir -p "${NASHRS_ENV_MANIFEST_DIR}"
python -m pip freeze > \
  "${NASHRS_ENV_MANIFEST_DIR}/openrlhf-0.9.3-vllm-0.15.0.freeze.txt"

PYTHONPATH=src python -m unittest discover -s tests -v

echo "OpenRLHF environment ready: $NASHRS_VENV"
