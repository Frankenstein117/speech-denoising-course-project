#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
raw="$root/data/raw"
processed="$root/data/processed/VoiceBank-DEMAND"
mkdir -p "$raw" "$processed"

downloads=(
  "clean_trainset_28spk_wav.zip|https://datashare.ed.ac.uk/bitstreams/245452b6-6235-44b6-a6f9-e7eb19797769/download"
  "noisy_trainset_28spk_wav.zip|https://datashare.ed.ac.uk/bitstreams/ecb5a102-bb00-46d3-8af5-40c79823b837/download"
  "clean_testset_wav.zip|https://datashare.ed.ac.uk/bitstreams/dec213d3-bf57-4777-9663-c24bdce92d5e/download"
  "noisy_testset_wav.zip|https://datashare.ed.ac.uk/bitstreams/13c1bfbf-14a6-41db-9b41-8f7310f01ad5/download"
)

for item in "${downloads[@]}"; do
  IFS='|' read -r name url <<< "$item"
  if ! unzip -tq "$raw/$name" >/dev/null 2>&1; then
    wget -c --retry-connrefused --waitretry=5 --timeout=30 "$url" -O "$raw/$name"
  fi
  unzip -tq "$raw/$name" >/dev/null
  directory="${name%.zip}"
  if [[ ! -d "$processed/$directory" ]]; then
    unzip -q "$raw/$name" -d "$processed"
  fi
done

echo "VoiceBank+DEMAND is ready in $processed"

