from __future__ import annotations

import argparse
import time
from pathlib import Path

import dpdfnet
import soundfile as sf
from tqdm import tqdm


SUPPORTED = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def enhance_file(source: Path, destination: Path, model: str, attenuation_db: float) -> float:
    """Enhance one recording and return elapsed seconds."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    audio, sample_rate = sf.read(source, always_2d=False)
    started = time.perf_counter()
    enhanced = dpdfnet.enhance(
        audio,
        sample_rate=sample_rate,
        model=model,
        attn_limit_db=attenuation_db,
    )
    elapsed = time.perf_counter() - started
    sf.write(destination, enhanced, sample_rate, subtype="PCM_16")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhance one audio file or a directory.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="dpdfnet2_48khz_hr")
    parser.add_argument("--attenuation-db", type=float, default=12.0)
    args = parser.parse_args()

    if args.input.is_file():
        elapsed = enhance_file(args.input, args.output, args.model, args.attenuation_db)
        print(f"enhanced={args.output} elapsed_seconds={elapsed:.3f}")
        return

    files = sorted(path for path in args.input.iterdir() if path.suffix.lower() in SUPPORTED)
    if not files:
        raise SystemExit(f"No supported audio files found in {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    total_elapsed = 0.0
    for source in tqdm(files, desc="Enhancing"):
        total_elapsed += enhance_file(
            source, args.output / f"{source.stem}.wav", args.model, args.attenuation_db
        )
    print(f"files={len(files)} inference_seconds={total_elapsed:.3f}")


if __name__ == "__main__":
    main()

