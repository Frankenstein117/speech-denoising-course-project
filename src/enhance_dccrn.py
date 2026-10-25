from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from src.metrics import as_mono_float, resample


MODEL_SAMPLE_RATE = 16_000


def load_model(config: Path, checkpoint: Path):
    from espnet2.bin.enh_inference import SeparateSpeech

    return SeparateSpeech(
        train_config=config,
        model_file=checkpoint,
        device="cpu",
        dtype="float32",
        normalize_output_wav=False,
    )


def enhance_file(source: Path, destination: Path, model) -> float:
    """Enhance one file with ESPnet's pretrained DNS-2020 DCCRN model."""
    audio, sample_rate = sf.read(source, always_2d=False)
    audio = as_mono_float(audio)
    model_audio = resample(audio, sample_rate, MODEL_SAMPLE_RATE).astype(np.float32)

    started = time.perf_counter()
    enhanced = model(model_audio[None, :], fs=MODEL_SAMPLE_RATE)[0][0]
    elapsed = time.perf_counter() - started

    enhanced = np.asarray(enhanced, dtype=np.float64)[: len(model_audio)]
    enhanced = resample(enhanced, MODEL_SAMPLE_RATE, sample_rate)[: len(audio)]
    if len(enhanced) < len(audio):
        enhanced = np.pad(enhanced, (0, len(audio) - len(enhanced)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, np.clip(enhanced, -1.0, 1.0), sample_rate, subtype="PCM_16")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhance WAV files with pretrained DCCRN.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, default=Path("models/dccrn/config.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/dccrn/48epoch.pth"))
    args = parser.parse_args()

    model = load_model(args.config, args.checkpoint)
    if args.input.is_file():
        elapsed = enhance_file(args.input, args.output, model)
        print(f"enhanced={args.output} elapsed_seconds={elapsed:.3f}")
        return

    files = sorted(args.input.glob("*.wav"))
    if not files:
        raise SystemExit(f"No WAV files found in {args.input}")
    total_elapsed = 0.0
    for source in tqdm(files, desc="DCCRN"):
        total_elapsed += enhance_file(source, args.output / source.name, model)
    print(f"files={len(files)} inference_seconds={total_elapsed:.3f}")


if __name__ == "__main__":
    main()
