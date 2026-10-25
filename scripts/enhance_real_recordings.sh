#!/bin/zsh
set -euo pipefail

project_dir="${0:A:h:h}"
cd "$project_dir"
source .venv/bin/activate
export DPDFNET_MODEL_DIR="$project_dir/models"
mkdir -p data/real_recordings outputs/enhanced/real_recordings

python -m src.enhance \
  data/real_recordings \
  outputs/enhanced/real_recordings \
  --model dpdfnet2_48khz_hr \
  --attenuation-db 12

