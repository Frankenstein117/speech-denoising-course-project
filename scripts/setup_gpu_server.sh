#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  apt-get update
  apt-get install -y python3 python3-venv git wget unzip libsndfile1
else
  sudo apt-get update
  sudo apt-get install -y python3 python3-venv git wget unzip libsndfile1
fi

python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is required.")
print(sys.version)
PY

python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .

.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Check the server image and NVIDIA driver.")
print("gpu", torch.cuda.get_device_name(0))
PY

