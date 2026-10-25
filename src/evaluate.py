from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.metrics import as_mono_float, full_reference_metrics


def score_directory(clean_dir: Path, candidate_dir: Path, stage: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    clean_files = sorted(clean_dir.glob("*.wav"))
    for clean_path in tqdm(clean_files, desc=f"Scoring {stage}"):
        candidate_path = candidate_dir / clean_path.name
        if not candidate_path.exists():
            continue
        clean, clean_sr = sf.read(clean_path)
        candidate, candidate_sr = sf.read(candidate_path)
        if clean_sr != candidate_sr:
            raise ValueError(f"Sample-rate mismatch for {clean_path.name}: {clean_sr} vs {candidate_sr}")
        values = full_reference_metrics(
            as_mono_float(clean), as_mono_float(candidate), clean_sr
        )
        rows.append({"file": clean_path.name, "stage": stage, **values})
    if not rows:
        raise RuntimeError(f"No paired WAV files found for {stage}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate noisy and enhanced paired speech.")
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--noisy-dir", type=Path, required=True)
    parser.add_argument("--enhanced-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/metrics"))
    args = parser.parse_args()

    noisy = score_directory(args.clean_dir, args.noisy_dir, "noisy")
    enhanced = score_directory(args.clean_dir, args.enhanced_dir, "enhanced")
    per_file = pd.concat([noisy, enhanced], ignore_index=True)
    metric_columns = ["snr_db", "ssnr_db", "pesq_wb", "pesq_nb", "stoi"]
    summary = per_file.groupby("stage")[metric_columns].agg(["mean", "std", "median"])
    improvement = (
        enhanced.set_index("file")[metric_columns] - noisy.set_index("file")[metric_columns]
    ).mean()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_file.to_csv(args.output_dir / "per_file_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary_metrics.csv")
    improvement.rename("mean_improvement").to_csv(args.output_dir / "mean_improvement.csv")
    print(summary.round(4))
    print("\nMean improvement:\n", improvement.round(4))


if __name__ == "__main__":
    main()

