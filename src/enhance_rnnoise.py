from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from src.metrics import as_mono_float, resample


FRAME_SIZE = 480
MODEL_SAMPLE_RATE = 48_000


def enhance_file(source: Path, destination: Path, executable: Path) -> float:
    """Enhance one file with the official RNNoise command-line demo."""
    audio, sample_rate = sf.read(source, always_2d=False)
    audio = as_mono_float(audio)
    model_audio = resample(audio, sample_rate, MODEL_SAMPLE_RATE)
    original_model_length = len(model_audio)

    # RNNoise consumes 10 ms frames. The demo intentionally discards its first
    # delayed frame, so one zero frame is appended to flush a length-matched output.
    padded_length = int(np.ceil(original_model_length / FRAME_SIZE) * FRAME_SIZE)
    padded = np.pad(model_audio, (0, padded_length - original_model_length + FRAME_SIZE))
    pcm = np.round(np.clip(padded, -1.0, 1.0) * 32767.0).astype("<i2")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rnnoise-") as temporary:
        raw_input = Path(temporary) / "input.pcm"
        raw_output = Path(temporary) / "output.pcm"
        pcm.tofile(raw_input)
        started = time.perf_counter()
        subprocess.run([str(executable), str(raw_input), str(raw_output)], check=True)
        elapsed = time.perf_counter() - started
        enhanced = np.fromfile(raw_output, dtype="<i2").astype(np.float64) / 32768.0

    enhanced = enhanced[:original_model_length]
    enhanced = resample(enhanced, MODEL_SAMPLE_RATE, sample_rate)[: len(audio)]
    if len(enhanced) < len(audio):
        enhanced = np.pad(enhanced, (0, len(audio) - len(enhanced)))
    sf.write(destination, enhanced, sample_rate, subtype="PCM_16")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhance WAV files with official RNNoise.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--executable", type=Path, default=Path("work/bin/rnnoise_demo"))
    args = parser.parse_args()

    if args.input.is_file():
        elapsed = enhance_file(args.input, args.output, args.executable.resolve())
        print(f"enhanced={args.output} elapsed_seconds={elapsed:.3f}")
        return

    files = sorted(args.input.glob("*.wav"))
    if not files:
        raise SystemExit(f"No WAV files found in {args.input}")
    total_elapsed = 0.0
    for source in tqdm(files, desc="RNNoise"):
        total_elapsed += enhance_file(
            source, args.output / source.name, args.executable.resolve()
        )
    print(f"files={len(files)} inference_seconds={total_elapsed:.3f}")


if __name__ == "__main__":
    main()
