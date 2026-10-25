from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from src.metrics import as_mono_float, resample
from src.scratch_models import build_model


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint["model"], **checkpoint.get("model_kwargs", {}))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, int(checkpoint.get("sample_rate", 16_000))


def enhance_file(source: Path, destination: Path, model, model_rate: int, device) -> None:
    audio, source_rate = sf.read(source, always_2d=False)
    audio = as_mono_float(audio)
    model_audio = resample(audio, source_rate, model_rate).astype(np.float32)
    with torch.inference_mode():
        tensor = torch.from_numpy(model_audio).unsqueeze(0).to(device)
        enhanced = model(tensor).squeeze(0).cpu().numpy()
    enhanced = resample(enhanced, model_rate, source_rate)[: len(audio)]
    if len(enhanced) < len(audio):
        enhanced = np.pad(enhanced, (0, len(audio) - len(enhanced)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, np.clip(enhanced, -1, 1), source_rate, subtype="PCM_16")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhance speech with a model trained from scratch.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_rate = load_checkpoint(args.checkpoint, device)
    if args.input.is_file():
        enhance_file(args.input, args.output, model, model_rate, device)
        return
    files = sorted(args.input.glob("*.wav"))
    for source in tqdm(files, desc="enhance"):
        enhance_file(source, args.output / source.name, model, model_rate, device)


if __name__ == "__main__":
    main()

