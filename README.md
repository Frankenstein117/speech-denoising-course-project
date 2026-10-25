# 语音去噪系统课程实验

本项目实现语音去噪模型的从零训练与统一评价。训练代码包含一个DCCRN结构的复数谱卷积循环网络、一个受RNNoise启发的因果循环谱掩码网络，以及一个直接处理波形的Wave-U-Net。三个模型均从随机初始化开始训练，不读取公开预训练权重。评价流程在VoiceBank+DEMAND配对测试集上计算SNR、SSNR、PESQ-WB、PESQ-NB和STOI，并输出逐文件指标、统计汇总和可视化图表。

## 仓库内容

- `src/`：去噪、评价和可视化程序
- `scripts/`：完整实验与三模型对比脚本
- `configs/`：实验参数
- `tests/`：指标实现测试
- `outputs/metrics/`：本次实验的指标结果
- `outputs/figures/`：模型对比图
- `outputs/语音去噪系统实验报告.docx`：完整实验报告

数据集、增强后的WAV、Python虚拟环境和训练检查点体积较大，不提交到普通Git仓库。

## 环境安装

建议使用Python 3.10～3.12。GPU服务器已有兼容的PyTorch时，安装过程会保留满足版本要求的现有环境：

```bash
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Windows 激活虚拟环境时使用：

```powershell
.venv\Scripts\activate
```

## GPU服务器配置

推荐RTX 4090 24GB或同级GPU、100GB以上磁盘，以及带CUDA驱动的Ubuntu 24.04或PyTorch镜像。克隆仓库后运行：

```bash
chmod +x scripts/*.sh
./scripts/setup_gpu_server.sh
```

## 准备数据

可以直接下载并解压VoiceBank+DEMAND官方的28人训练集和测试集：

```bash
./scripts/download_voicebank_demand.sh
```

完成后的目录为：

```text
data/processed/VoiceBank-DEMAND/
├── clean_trainset_28spk_wav/
├── noisy_trainset_28spk_wav/
├── clean_testset_wav/
└── noisy_testset_wav/
```

两个目录中的干净语音和含噪语音应同名配对。本次实验共评价 824 对语音。

## 从随机初始化训练并测试

依次训练三个模型、选择验证集最优权重、处理824条测试语音并计算指标：

```bash
chmod +x scripts/*.sh
./scripts/train_from_scratch.sh
```

训练集按说话人划分训练部分与验证部分，官方测试集只用于最终评价。最优检查点分别保存在`outputs/training/dccrn_scratch/best.pt`、`outputs/training/rnnoise_scratch/best.pt`和`outputs/training/wave_unet_scratch/best.pt`。

运行测试：

```bash
python -m unittest discover -s tests
```

主要结果保存在 `outputs/metrics/`，图表保存在 `outputs/figures/`。

## 既有预训练基线结果

仓库中现有图表和报告记录的是早期预训练基线，便于与自主训练结果比较。自主训练结果应以`outputs/metrics/scratch/`中服务器运行后产生的CSV为准，并据此更新报告。
