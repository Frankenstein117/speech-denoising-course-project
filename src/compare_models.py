from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
METRICS = ["snr_db", "ssnr_db", "pesq_wb", "pesq_nb", "stoi"]
LABELS = {
    "noisy": "Noisy input",
    "rnnoise": "RNNoise (PPT)",
    "dccrn": "DCCRN (PPT)",
    "dpdfnet2": "DPDFNet2",
}


def markdown_table(frame: pd.DataFrame) -> str:
    columns = ["model", *frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---", *(["---:"] * len(frame.columns))]) + " |",
    ]
    for index, row in frame.iterrows():
        lines.append(
            "| "
            + " | ".join([str(index), *(f"{float(value):.4f}" for value in row)])
            + " |"
        )
    return "\n".join(lines)


def load_scores() -> pd.DataFrame:
    baseline = pd.read_csv(OUTPUTS / "metrics/voicebank_demand/per_file_metrics.csv")
    baseline = baseline[baseline.stage.isin(["noisy", "enhanced"])].copy()
    baseline.loc[baseline.stage == "enhanced", "stage"] = "dpdfnet2"
    rnnoise = pd.read_csv(OUTPUTS / "metrics/model_comparison/rnnoise_per_file.csv")
    dccrn = pd.read_csv(OUTPUTS / "metrics/model_comparison/dccrn_per_file.csv")
    return pd.concat([baseline, rnnoise, dccrn], ignore_index=True)


def make_plots(summary: pd.DataFrame, improvement: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    colors = ["#8d99ae", "#2a9d8f", "#e9c46a", "#457b9d"]
    order = ["noisy", "rnnoise", "dccrn", "dpdfnet2"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
    for axis, metric in zip(axes, METRICS):
        values = [summary.loc[stage, metric] for stage in order]
        axis.bar([LABELS[x] for x in order], values, color=colors)
        axis.set_title(metric.replace("_db", " (dB)").upper())
        axis.tick_params(axis="x", rotation=34)
        axis.bar_label(axis.containers[0], fmt="%.3f", fontsize=8, padding=2)
        axis.margins(y=0.15)
    fig.suptitle("VoiceBank+DEMAND: mean objective metrics (824 utterances)")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "figures/model_comparison_metrics.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
    model_order = ["rnnoise", "dccrn", "dpdfnet2"]
    model_colors = colors[1:]
    for axis, metric in zip(axes, METRICS):
        values = [improvement.loc[stage, metric] for stage in model_order]
        bars = axis.bar([LABELS[x] for x in model_order], values, color=model_colors)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(metric.replace("_db", " (dB)").upper())
        axis.tick_params(axis="x", rotation=34)
        axis.bar_label(bars, fmt="%+.3f", fontsize=8, padding=2)
        axis.margins(y=0.2)
    fig.suptitle("Mean change relative to noisy input")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "figures/model_comparison_improvement.png", dpi=200)
    plt.close(fig)


def main() -> None:
    data = load_scores()
    comparison_dir = OUTPUTS / "metrics/model_comparison"
    figures_dir = OUTPUTS / "figures"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(comparison_dir / "all_models_per_file.csv", index=False)

    summary = data.groupby("stage")[METRICS].mean().loc[
        ["noisy", "rnnoise", "dccrn", "dpdfnet2"]
    ]
    improvement = summary.loc[["rnnoise", "dccrn", "dpdfnet2"]].subtract(
        summary.loc["noisy"], axis="columns"
    )
    summary.to_csv(comparison_dir / "all_models_mean_metrics.csv")
    improvement.to_csv(comparison_dir / "all_models_mean_improvement.csv")
    make_plots(summary, improvement)

    sample = "p232_001.wav"
    sample_dir = OUTPUTS / "comparison_audio" / Path(sample).stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "clean.wav": ROOT / "data/processed/VoiceBank-DEMAND/clean_testset_wav" / sample,
        "noisy.wav": ROOT / "data/processed/VoiceBank-DEMAND/noisy_testset_wav" / sample,
        "rnnoise.wav": OUTPUTS / "enhanced/rnnoise" / sample,
        "dccrn.wav": OUTPUTS / "enhanced/dccrn" / sample,
        "dpdfnet2.wav": OUTPUTS / "enhanced/voicebank_demand" / sample,
    }
    for name, source in sources.items():
        shutil.copy2(source, sample_dir / name)

    result = {
        "dataset": "VoiceBank+DEMAND official paired test set",
        "paired_files": 824,
        "input_sample_rate_hz": 48000,
        "models": {
            "rnnoise": {"listed_in_teacher_ppt": True, "native_sample_rate_hz": 48000},
            "dccrn": {
                "listed_in_teacher_ppt": True,
                "native_sample_rate_hz": 16000,
                "parameters": 3671053,
            },
            "dpdfnet2": {"listed_in_teacher_ppt": False, "native_sample_rate_hz": 48000},
        },
        "mean_metrics": summary.round(4).to_dict(orient="index"),
        "mean_improvement": improvement.round(4).to_dict(orient="index"),
        "cpu_inference_seconds": {"rnnoise": 120.806, "dccrn": 184.772, "dpdfnet2": 275.635},
    }
    with (OUTPUTS / "model_comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    report = f"""# 语音去噪模型复现对照

## 实验设置

- 测试集：VoiceBank+DEMAND 官方配对测试集，共 824 条。
- 输入：48 kHz 单声道含噪语音；以同名干净语音为参考。
- 模型：RNNoise、DCCRN（均在老师 PPT 中），以及此前完成的 DPDFNet2。
- 复现方式：公开预训练权重推理，不是从头训练。
- DCCRN 原生采样率为 16 kHz，推理前后统一重采样；RNNoise 与 DPDFNet2 原生为 48 kHz。
- 所有输出都恢复为 48 kHz，并使用同一自动时延对齐和同一组评价代码。

## 平均指标

{markdown_table(summary.rename(index=LABELS))}

## 相对含噪输入的平均变化

{markdown_table(improvement.rename(index=LABELS))}

## 结果解读

- DPDFNet2 在 SNR、分段 SNR、宽带 PESQ 和 STOI 上最好；DCCRN 的窄带 PESQ 最好。
- DCCRN 五项指标均优于含噪输入，是这两个 PPT 模型中整体效果较好的一项。
- RNNoise 提高了 SNR、分段 SNR、PESQ-WB 和 PESQ-NB，但 STOI 平均下降。这说明它在这套离线语料上压噪较强，同时会损伤一部分语音可懂度线索。
- 运行时间只反映本机 CPU 和当前批处理封装。RNNoise 的核心优势是低延迟实时处理，不能仅凭离线客观指标判断其工程价值。

## 可复查文件

- `metrics/model_comparison/all_models_per_file.csv`：每条语音的全部指标。
- `metrics/model_comparison/all_models_mean_metrics.csv`：四组平均结果。
- `metrics/model_comparison/all_models_mean_improvement.csv`：三个模型相对含噪输入的增量。
- `comparison_audio/p232_001/`：同一条语音的干净、含噪和三个增强版本。
- `figures/model_comparison_metrics.png`、`figures/model_comparison_improvement.png`：对照图。
"""
    (OUTPUTS / "model_comparison_report.md").write_text(report, encoding="utf-8")
    print(summary.round(4))
    print("\nImprovement:\n", improvement.round(4))


if __name__ == "__main__":
    main()
