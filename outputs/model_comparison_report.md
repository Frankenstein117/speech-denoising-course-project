# 语音去噪模型复现对照

## 实验设置

- 测试集：VoiceBank+DEMAND 官方配对测试集，共 824 条。
- 输入：48 kHz 单声道含噪语音；以同名干净语音为参考。
- 模型：RNNoise、DCCRN（均在老师 PPT 中），以及此前完成的 DPDFNet2。
- 复现方式：公开预训练权重推理，不是从头训练。
- DCCRN 原生采样率为 16 kHz，推理前后统一重采样；RNNoise 与 DPDFNet2 原生为 48 kHz。
- 所有输出都恢复为 48 kHz，并使用同一自动时延对齐和同一组评价代码。

## 平均指标

| model | snr_db | ssnr_db | pesq_wb | pesq_nb | stoi |
| --- | ---: | ---: | ---: | ---: | ---: |
| Noisy input | 8.3949 | 1.5353 | 1.9668 | 2.9456 | 0.9211 |
| RNNoise (PPT) | 12.5943 | 5.9782 | 2.1067 | 3.0470 | 0.8895 |
| DCCRN (PPT) | 14.5955 | 6.7916 | 2.6256 | 3.5454 | 0.9349 |
| DPDFNet2 | 16.9152 | 7.8232 | 2.7007 | 3.4968 | 0.9394 |

## 相对含噪输入的平均变化

| model | snr_db | ssnr_db | pesq_wb | pesq_nb | stoi |
| --- | ---: | ---: | ---: | ---: | ---: |
| RNNoise (PPT) | 4.1994 | 4.4429 | 0.1399 | 0.1013 | -0.0316 |
| DCCRN (PPT) | 6.2007 | 5.2563 | 0.6589 | 0.5998 | 0.0138 |
| DPDFNet2 | 8.5203 | 6.2879 | 0.7340 | 0.5511 | 0.0183 |

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
