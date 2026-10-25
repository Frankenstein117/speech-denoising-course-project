from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal
from torch.utils.data import Dataset


def paired_files(clean_dir: Path, noisy_dir: Path) -> list[tuple[Path, Path]]:
    pairs = [(path, noisy_dir / path.name) for path in sorted(clean_dir.glob("*.wav"))]
    pairs = [(clean, noisy) for clean, noisy in pairs if noisy.exists()]
    if not pairs:
        raise RuntimeError(f"No paired WAV files in {clean_dir} and {noisy_dir}")
    return pairs


def split_by_speaker(
    pairs: list[tuple[Path, Path]], validation_speakers: int = 2, seed: int = 42
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    speakers = sorted({clean.stem.split("_")[0] for clean, _ in pairs})
    rng = random.Random(seed)
    rng.shuffle(speakers)
    validation = set(speakers[:validation_speakers])
    train = [pair for pair in pairs if pair[0].stem.split("_")[0] not in validation]
    valid = [pair for pair in pairs if pair[0].stem.split("_")[0] in validation]
    return train, valid


def _load(path: Path, sample_rate: int) -> np.ndarray:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if source_rate != sample_rate:
        divisor = np.gcd(source_rate, sample_rate)
        audio = signal.resample_poly(audio, sample_rate // divisor, source_rate // divisor)
    return np.asarray(audio, dtype=np.float32)


class PairedAudioDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        sample_rate: int = 16_000,
        segment_seconds: float = 2.0,
        random_crop: bool = True,
    ):
        self.pairs = pairs
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_seconds)
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        clean_path, noisy_path = self.pairs[index]
        clean = _load(clean_path, self.sample_rate)
        noisy = _load(noisy_path, self.sample_rate)
        length = min(len(clean), len(noisy))
        clean, noisy = clean[:length], noisy[:length]
        if length >= self.segment_samples:
            maximum = length - self.segment_samples
            start = random.randint(0, maximum) if self.random_crop and maximum else 0
            clean = clean[start : start + self.segment_samples]
            noisy = noisy[start : start + self.segment_samples]
        else:
            padding = self.segment_samples - length
            clean = np.pad(clean, (0, padding))
            noisy = np.pad(noisy, (0, padding))
        return torch.from_numpy(noisy.copy()), torch.from_numpy(clean.copy())

