#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
raw="$root/data/raw"
processed="$root/data/processed/VoiceBank-DEMAND"
mkdir -p "$raw" "$processed"

files=(
  clean_trainset_28spk_wav.zip
  noisy_trainset_28spk_wav.zip
  clean_testset_wav.zip
  noisy_testset_wav.zip
)
mirror_base="${VOICEBANK_MIRROR_BASE:-https://hf-mirror.com/datasets/mrfakename/noisy-speech-files/resolve/main}"
fallback_base="https://huggingface.co/datasets/mrfakename/noisy-speech-files/resolve/main"

download() {
  local url="$1"
  local destination="$2"
  curl --fail --location --continue-at - \
    --retry 5 --retry-delay 5 --connect-timeout 30 \
    --output "$destination" "$url"
}

for name in "${files[@]}"; do
  if ! unzip -tq "$raw/$name" >/dev/null 2>&1; then
    if ! download "$mirror_base/$name?download=true" "$raw/$name"; then
      echo "Mirror failed; retrying from Hugging Face."
      download "$fallback_base/$name?download=true" "$raw/$name"
    fi
  fi
  unzip -tq "$raw/$name" >/dev/null
  directory="${name%.zip}"
  if [[ ! -d "$processed/$directory" ]]; then
    unzip -q "$raw/$name" -d "$processed"
  fi
done

echo "VoiceBank+DEMAND is ready in $processed"

