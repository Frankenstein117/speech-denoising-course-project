#!/bin/zsh
set -euo pipefail

project_dir="${0:A:h:h}"
cd "$project_dir"
source .venv/bin/activate
export DPDFNET_MODEL_DIR="$project_dir/models"

python -m src.enhance \
  data/processed/VoiceBank-DEMAND/noisy_testset_wav \
  outputs/enhanced/voicebank_demand \
  --model dpdfnet2_48khz_hr \
  --attenuation-db 12

python -m src.evaluate \
  --clean-dir data/processed/VoiceBank-DEMAND/clean_testset_wav \
  --noisy-dir data/processed/VoiceBank-DEMAND/noisy_testset_wav \
  --enhanced-dir outputs/enhanced/voicebank_demand \
  --output-dir outputs/metrics/voicebank_demand

example="p232_001.wav"
python -m src.visualize_results \
  --metrics outputs/metrics/voicebank_demand/per_file_metrics.csv \
  --clean "data/processed/VoiceBank-DEMAND/clean_testset_wav/$example" \
  --noisy "data/processed/VoiceBank-DEMAND/noisy_testset_wav/$example" \
  --enhanced "outputs/enhanced/voicebank_demand/$example"

