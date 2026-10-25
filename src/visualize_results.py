from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf


def plot_summary(metrics_csv: Path, output: Path) -> None:
    data = pd.read_csv(metrics_csv)
    metrics = ["snr_db", "ssnr_db", "pesq_wb", "pesq_nb", "stoi"]
    long = data.melt(id_vars=["file", "stage"], value_vars=metrics, var_name="metric")
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4))
    for metric, axis in zip(metrics, axes):
        sns.boxplot(data=long[long.metric == metric], x="stage", y="value", ax=axis)
        axis.set_title(metric.upper())
        axis.set_xlabel("")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_example(clean: Path, noisy: Path, enhanced: Path, output: Path) -> None:
    records = [("Clean", clean), ("Noisy", noisy), ("Enhanced", enhanced)]
    fig, axes = plt.subplots(3, 2, figsize=(14, 9))
    for row, (label, path) in enumerate(records):
        audio, sr = sf.read(path)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        librosa.display.waveshow(audio, sr=sr, ax=axes[row, 0], color="#2166ac")
        spectrum = librosa.amplitude_to_db(abs(librosa.stft(audio, n_fft=2048)), ref=np.max)
        librosa.display.specshow(spectrum, sr=sr, x_axis="time", y_axis="hz", ax=axes[row, 1])
        axes[row, 0].set_title(f"{label} waveform")
        axes[row, 1].set_title(f"{label} spectrogram")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--noisy", type=Path, required=True)
    parser.add_argument("--enhanced", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    plot_summary(args.metrics, args.output_dir / "metric_comparison.png")
    plot_example(args.clean, args.noisy, args.enhanced, args.output_dir / "example_comparison.png")


if __name__ == "__main__":
    main()

