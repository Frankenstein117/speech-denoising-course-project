#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
raw="$root/data/raw"
processed="$root/data/processed/VoiceBank-DEMAND"
base="https://datashare.ed.ac.uk/bitstream/handle/10283/2791"
mkdir -p "$raw" "$processed"

files=(
  clean_trainset_28spk_wav.zip
  noisy_trainset_28spk_wav.zip
  clean_testset_wav.zip
  noisy_testset_wav.zip
)

for name in "${files[@]}"; do
  if [[ ! -f "$raw/$name" ]]; then
    wget -c "$base/$name" -O "$raw/$name"
  fi
  directory="${name%.zip}"
  if [[ ! -d "$processed/$directory" ]]; then
    unzip -q "$raw/$name" -d "$processed"
  fi
done

echo "VoiceBank+DEMAND is ready in $processed"

