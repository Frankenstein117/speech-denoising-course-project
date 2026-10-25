#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

continue_model() {
  local config="$1"
  local name="$2"
  local checkpoint="outputs/training/$name/last.pt"

  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  fi

  .venv/bin/python -m src.train_scratch \
    --config "$config" \
    --resume "$checkpoint" \
    --epochs 100 \
    --early-stopping-patience 12

  .venv/bin/python -m src.enhance_scratch \
    data/processed/VoiceBank-DEMAND/noisy_testset_wav \
    "outputs/enhanced/$name" \
    --checkpoint "outputs/training/$name/best.pt"

  .venv/bin/python -m src.evaluate_stage \
    --clean-dir data/processed/VoiceBank-DEMAND/clean_testset_wav \
    --candidate-dir "outputs/enhanced/$name" \
    --stage "$name" \
    --output "outputs/metrics/scratch/$name.csv"
}

continue_model configs/train_dccrn_scratch.yaml dccrn_scratch
continue_model configs/train_rnnoise_scratch.yaml rnnoise_scratch
continue_model configs/train_wave_unet_scratch.yaml wave_unet_scratch

.venv/bin/python -m src.compare_scratch_models
