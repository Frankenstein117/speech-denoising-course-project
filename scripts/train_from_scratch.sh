#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

.venv/bin/python -m src.evaluate_stage \
  --clean-dir data/processed/VoiceBank-DEMAND/clean_testset_wav \
  --candidate-dir data/processed/VoiceBank-DEMAND/noisy_testset_wav \
  --stage noisy \
  --output outputs/metrics/scratch/noisy.csv

.venv/bin/python -m src.train_scratch --config configs/train_dccrn_scratch.yaml
.venv/bin/python -m src.enhance_scratch \
  data/processed/VoiceBank-DEMAND/noisy_testset_wav \
  outputs/enhanced/dccrn_scratch \
  --checkpoint outputs/training/dccrn_scratch/best.pt
.venv/bin/python -m src.evaluate_stage \
  --clean-dir data/processed/VoiceBank-DEMAND/clean_testset_wav \
  --candidate-dir outputs/enhanced/dccrn_scratch \
  --stage dccrn_scratch \
  --output outputs/metrics/scratch/dccrn_scratch.csv

.venv/bin/python -m src.train_scratch --config configs/train_rnnoise_scratch.yaml
.venv/bin/python -m src.enhance_scratch \
  data/processed/VoiceBank-DEMAND/noisy_testset_wav \
  outputs/enhanced/rnnoise_scratch \
  --checkpoint outputs/training/rnnoise_scratch/best.pt
.venv/bin/python -m src.evaluate_stage \
  --clean-dir data/processed/VoiceBank-DEMAND/clean_testset_wav \
  --candidate-dir outputs/enhanced/rnnoise_scratch \
  --stage rnnoise_scratch \
  --output outputs/metrics/scratch/rnnoise_scratch.csv

.venv/bin/python -m src.train_scratch --config configs/train_wave_unet_scratch.yaml
.venv/bin/python -m src.enhance_scratch \
  data/processed/VoiceBank-DEMAND/noisy_testset_wav \
  outputs/enhanced/wave_unet_scratch \
  --checkpoint outputs/training/wave_unet_scratch/best.pt
.venv/bin/python -m src.evaluate_stage \
  --clean-dir data/processed/VoiceBank-DEMAND/clean_testset_wav \
  --candidate-dir outputs/enhanced/wave_unet_scratch \
  --stage wave_unet_scratch \
  --output outputs/metrics/scratch/wave_unet_scratch.csv

.venv/bin/python -m src.compare_scratch_models

