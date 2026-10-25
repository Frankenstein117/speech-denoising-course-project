from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs/metrics/scratch"
STAGES = ["noisy", "dccrn_scratch", "rnnoise_scratch", "wave_unet_scratch"]
METRIC_COLUMNS = ["snr_db", "ssnr_db", "pesq_wb", "pesq_nb", "stoi"]


def main() -> None:
    frames = []
    for stage in STAGES:
        path = METRICS / f"{stage}.csv"
        frame = pd.read_csv(path)
        frame["stage"] = stage
        frames.append(frame)
    all_results = pd.concat(frames, ignore_index=True)
    summary = all_results.groupby("stage")[METRIC_COLUMNS].mean().reindex(STAGES)
    improvement = summary.drop(index="noisy").subtract(summary.loc["noisy"])
    METRICS.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(METRICS / "all_scratch_models_per_file.csv", index=False)
    summary.to_csv(METRICS / "all_scratch_models_mean_metrics.csv")
    improvement.to_csv(METRICS / "all_scratch_models_mean_improvement.csv")
    payload = {
        "training": "random initialization; no pretrained weights",
        "mean_metrics": summary.to_dict(orient="index"),
        "mean_improvement_over_noisy": improvement.to_dict(orient="index"),
    }
    (METRICS / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.round(4))
    print("\nMean improvement over noisy:\n", improvement.round(4))


if __name__ == "__main__":
    main()

