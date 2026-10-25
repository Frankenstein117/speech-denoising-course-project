#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

.venv/bin/python -m src.enhance_rnnoise \
  data/processed/VoiceBank-DEMAND/noisy_testset_wav \
  outputs/enhanced/rnnoise \
  --executable work/bin/rnnoise_demo

.venv/bin/python -m src.enhance_dccrn \
  data/processed/VoiceBank-DEMAND/noisy_testset_wav \
  outputs/enhanced/dccrn \
  --config models/dccrn/config.yaml \
  --checkpoint models/dccrn/48epoch.pth

.venv/bin/python -m src.evaluate_stage \
  --clean-dir data/processed/VoiceBank-DEMAND/clean_testset_wav \
  --candidate-dir outputs/enhanced/rnnoise \
  --stage rnnoise \
  --output outputs/metrics/model_comparison/rnnoise_per_file.csv

.venv/bin/python -m src.evaluate_stage \
  --clean-dir data/processed/VoiceBank-DEMAND/clean_testset_wav \
  --candidate-dir outputs/enhanced/dccrn \
  --stage dccrn \
  --output outputs/metrics/model_comparison/dccrn_per_file.csv

MPLBACKEND=Agg MPLCONFIGDIR="$root/work/mpl" XDG_CACHE_HOME="$root/work/cache" \
  .venv/bin/python -m src.compare_models
