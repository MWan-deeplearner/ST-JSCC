[**English**](#english) | [**中文**](#chinese)

---

<h1 id="english">ST-JSCC</h1>

**ST-JSCC: Synergizing Structural and Textural Dependencies for Robust and Efficient Image Transmission**

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

> Official PyTorch implementation of ST-JSCC — a novel joint source-channel coding framework that synergizes structural (Swin Transformer) and textural (convolution) dependencies for robust wireless image transmission over noisy channels.

## 📖 Overview

ST-JSCC introduces a hybrid encoder-decoder architecture for deep joint source-channel coding (JSCC). By combining the **Structural branch** built on Swin Transformer blocks with a **Textural branch** via learned importance-aware feature modulation, the model achieves state-of-the-art image reconstruction quality under both **AWGN** and **Rayleigh fading** channels.

### Supported Models

| Model | Description |
|-------|-------------|
| **STJSCC** | Full ST-JSCC with structural + textural synergy |
| ConvJSCC | Convolution-based JSCC baseline |
| ResJSCC | Residual convolution JSCC baseline |
| SwinJSCC | Swin Transformer JSCC baseline |
| LAJSCC | Linear-attention JSCC |
| FAJSCC | Importance-aware frequency-adaptive JSCC |
| LICRFJSCC | Learned image compression + residual fusion JSCC |

### Supported Channel Types

- **AWGN** — Additive White Gaussian Noise
- **Rayleigh** — Rayleigh fading channel (with CSI estimation option)
- **None** — Noiseless channel (for debugging)

### Evaluation Metrics

- PSNR, SSIM, MS-SSIM, LPIPS
- FLOPs, Parameters (M), Inference Runtime (ms), GPU Memory (MB)

---

## 🗂️ Project Structure

```
ST-JSCC/
├── model/                      # Model definitions
│   ├── JSCC.py                 # All JSCC variants (STJSCC, ConvJSCC, FAJSCC, etc.)
│   ├── STComponent.py          # ST-JSCC core: STBlock, LinearAttention, GFFN
│   ├── Encoder.py              # Encoder architectures
│   ├── Decoder.py              # Decoder architectures
│   ├── FAComponent.py          # Frequency-adaptive components
│   ├── LAComponent.py          # Linear-attention components
│   ├── ConvComponent.py        # Convolution components
│   ├── ResConvComponent.py     # Residual convolution components
│   ├── SWComponent.py          # Swin Transformer components
│   ├── LICRComponent.py        # Learned image compression components
│   ├── common_component.py     # Shared modules (GDN, Channel, PatchEmbed, etc.)
│   └── model_maker.py          # Model factory
├── utils/                      # Utilities
│   ├── data_maker.py           # Dataset & dataloader (DIV2K, Kodak, Flickr30k, etc.)
│   ├── loss_maker.py           # Loss functions (MSE, SSIM, MS-SSIM, FA loss)
│   ├── optimizer_maker.py      # Optimizer factory
│   ├── train_utils.py          # Training loop & Trainer class
│   ├── test_utils.py           # Evaluation & metrics computation
│   ├── torch_msssim.py         # MS-SSIM implementation
│   ├── utils.py                # Helper functions
│   └── vis_feature.py          # Feature visualization
├── configs/                    # Hydra configuration
│   ├── train.yaml              # Training config
│   ├── model_eval.yaml         # Evaluation config
│   └── data_info/              # Dataset-specific configs
│       ├── DIV2K.yaml
│       └── Flickr30k.yaml
├── plot/                       # Plotting scripts (paper figures)
│   ├── quantitative.py
│   ├── ablation_gl.py
│   ├── ablation_elu_alpha.py
│   ├── generalization.py
│   └── ...
├── saved_models/               # Pre-trained model checkpoints (.pt)
├── result_dicts/               # Cached evaluation results (.pkl)
├── train.py                    # Training entry point
├── test.py                     # Evaluation entry point
├── main.pdf                    # Paper PDF
├── main.tex                    # Paper LaTeX source
├── requirements.txt            # Python dependencies
└── LICENSE                     # MIT license
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create and activate conda environment (recommended)
conda create -n stjscc python=3.12
conda activate stjscc

# Install PyTorch (adjust cuda version as needed)
# See https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 2. Prepare Datasets

Download the datasets and organize them as follows:

```
<data_dir>/
├── DIV2K/
│   ├── DIV2K_train_HR/         # Training images (800 HR images)
│   └── DIV2K_valid_HR/         # Validation images (100 HR images)
├── Kodak/                      # Kodak24 test set
├── Flickr30k/
│   └── flickr30k-images/       # Flickr30k training images
├── Set5/
│   └── HR/
├── Set14/
│   └── HR/
```

Update the `data_dir` path in `configs/train.yaml` and `configs/model_eval.yaml` to point to your dataset directory.

> **Note:** DIV2K can be downloaded from [https://data.vision.ee.ethz.ch/cvl/DIV2K/](https://data.vision.ee.ethz.ch/cvl/DIV2K/). Kodak24 is available at [http://r0k.us/graphics/kodak/](http://r0k.us/graphics/kodak/).

### 3. Training

```bash
# Train ST-JSCC on DIV2K with AWGN channel, PSNR metric
python train.py \
    model_name=STJSCC \
    chan_type=AWGN \
    rcpp=12 \
    performance_metric=PSNR \
    data_info=DIV2K.yaml \
    device=cuda:0

# Train on Rayleigh fading channel
python train.py \
    model_name=STJSCC \
    chan_type=Rayleigh \
    rcpp=12 \
    performance_metric=PSNR

# Train SSIM-optimized model
python train.py \
    model_name=STJSCC \
    chan_type=AWGN \
    rcpp=12 \
    performance_metric=SSIM

# Train other baseline models
python train.py model_name=ConvJSCC chan_type=AWGN rcpp=12
python train.py model_name=SwinJSCC chan_type=AWGN rcpp=12
python train.py model_name=FAJSCC chan_type=AWGN rcpp=12
```

**Key Hyperparameters** (configured in `configs/train.yaml`):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `rcpp` | Reverse channel per pixel (bandwidth ratio) | `12` |
| `total_max_epoch` | Total training epochs | `500` |
| `learning_rate` | Learning rate | `1e-4` |
| `batch_size` | Batch size (per GPU) | `32` (DIV2K) |
| `chan_type` | Channel type: `AWGN`, `Rayleigh` | `Rayleigh` |
| `performance_metric` | Optimization target: `PSNR`, `SSIM` | `PSNR` |

### 4. Evaluation

```bash
# Evaluate a trained model on Kodak dataset
python test.py \
    model_name=STJSCC \
    chan_type=AWGN \
    rcpp=12 \
    test_data=Kodak \
    data_info=DIV2K.yaml \
    device=cuda:0

# Evaluate on other test sets
python test.py model_name=STJSCC test_data=DIV2K chan_type=AWGN
python test.py model_name=STJSCC test_data=Set5 chan_type=AWGN
python test.py model_name=STJSCC test_data=Set14 chan_type=AWGN
```

The evaluation outputs: PSNR, SSIM, MS-SSIM, LPIPS across multiple SNR levels (1, 4, 7, 10 dB by default), plus FLOPs, parameter count, runtime, and GPU memory usage. Results are cached as `.pkl` files in `result_dicts/`.

### 5. Using Pre-trained Models

Pre-trained models are provided in `saved_models/`. Set `save_dir` in the config to point to this directory and run evaluation directly — no retraining needed.

---

## 📊 Model Zoo

| Model | Channel | rcpp | Params | SSIM (Kodak) | Download |
|------|------|------|--------|--------------|------|
| STJSCC | AWGN | 12 | 1.01   | 0.7147       | `saved_models/STJSCC_DIV2K_AWGN_rcpp012_PSNR.pt` |
| STJSCC | Rayleigh | 12 | 1.01   | 0.6792       | `saved_models/STJSCC_DIV2K_Rayleigh_rcpp012_PSNR.pt` |
| STJSCC-32ch | AWGN | 12 | 0.45   | 0.6894       | `saved_models/STJSCC_32_channel_DIV2K_AWGN_rcpp012_PSNR.pt` |
| STJSCC-64ch | AWGN | 12 | 1.79   | 0.7348       | `saved_models/STJSCC_64_channel_DIV2K_AWGN_rcpp012_PSNR.pt` |
| STJSCC-4blk | AWGN | 12 | 0.83   | 0.7082       | `saved_models/STJSCC_4_blocks_DIV2K_AWGN_rcpp012_PSNR.pt` |
| STJSCC-6blk | AWGN | 12 | 1.19   | 0.7237       | `saved_models/STJSCC_6_blocks_DIV2K_AWGN_rcpp012_PSNR.pt` |

---

## 📈 Reproducing Paper Figures

Scripts in the `plot/` directory reproduce the figures from the paper:

```bash
cd plot
python quantitative.py       # Main quantitative comparison (PSNR vs SNR)
python ablation_gl.py        # Ablation study figures
python generalization.py     # Generalization performance
```

> **Note:** Run `test.py` first to generate the required result `.pkl` files before running plot scripts.

---

## 🔧 Dependencies

See [`requirements.txt`](./requirements.txt) for the full list. Core dependencies:

- **PyTorch** ≥ 1.10 (1.13 recommended)
- **torchvision** ≥ 0.14
- **Hydra** (hydra-core, omegaconf) — configuration management
- **fvcore** — FLOPs computation
- **lpips** — perceptual quality metric
- **timm** — Swin Transformer components
- **pytorch_wavelets** — DWT/IDWT operations
- **einops** — tensor operations
- **matplotlib**, **numpy**, **Pillow**

---

## 📝 Citation

If you find this work useful in your research, please cite:

```bibtex
@article{he2025stjscc,
  title   = {ST-JSCC: Synergizing Structural and Textural Dependencies
             for Robust and Efficient Image Transmission},
  author  = {Rulong He and Mingyang Wan and Haoming Luo and others},
  journal = {TBD},
  year    = {2025}
}
```

---

## 👤 Contact

**Corresponding Author:** Mingyang Wan

- 📧 Email: mingyang_wan@163.com
- 🔗 Issues: For questions or bug reports, please [open an issue]([https://github.com/MWan-deeplearner/ST-JSCC/issues](https://github.com/MWan-deeplearner/ST-JSCC/issues))

---

## 📄 License

This project is released under the [MIT License](./LICENSE).

---

## 🙏 Acknowledgements

Code references and inspirations:

- [SwinJSCC](https://github.com/semcomm/SwinJSCC) — Swin Transformer-based JSCC baseline
- [GDN (Generalized Divisive Normalization)](https://arxiv.org/abs/1611.01704)
- [DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

---

---

<h1 id="chinese">ST-JSCC（中文）</h1>

**ST-JSCC：融合结构与纹理依赖的鲁棒高效图像传输**

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

> ST-JSCC 的官方 PyTorch 实现 —— 一种新颖的联合信源信道编码框架，通过融合结构（Swin Transformer）与纹理（卷积）依赖关系，实现噪声信道下的鲁棒无线图像传输。

## 📖 简介

ST-JSCC 为深度联合信源信道编码（Deep JSCC）提出了一种混合编码器-解码器架构。通过将基于 Swin Transformer 块的**结构分支**与基于可学习重要性感知特征调制的**纹理分支**相结合，该模型在 **AWGN** 和 **Rayleigh 衰落**信道下均达到了领先的图像重建质量。

### 支持的模型

| 模型 | 说明 |
|------|------|
| **STJSCC** | 完整 ST-JSCC（结构+纹理协同） |
| ConvJSCC | 基于卷积的 JSCC 基线 |
| ResJSCC | 基于残差卷积的 JSCC 基线 |
| SwinJSCC | 基于 Swin Transformer 的 JSCC 基线 |
| LAJSCC | 线性注意力 JSCC |
| FAJSCC | 频率自适应重要性感知 JSCC |
| LICRFJSCC | 学习图像压缩 + 残差融合 JSCC |

### 支持的信道类型

- **AWGN** — 加性高斯白噪声
- **Rayleigh** — Rayleigh 衰落信道（可选 CSI 估计）
- **None** — 无噪声信道（调试用）

### 评估指标

- PSNR, SSIM, MS-SSIM, LPIPS
- FLOPs, 参数量 (M), 推理时间 (ms), GPU 显存 (MB)

---

## 🗂️ 项目结构

```
ST-JSCC/
├── model/                      # 模型定义
│   ├── JSCC.py                 # 所有 JSCC 变体
│   ├── STComponent.py          # ST-JSCC 核心组件
│   ├── Encoder.py / Decoder.py # 编解码器
│   ├── FAComponent.py          # 频率自适应组件
│   ├── LAComponent.py          # 线性注意力组件
│   ├── common_component.py     # 共享模块 (GDN, Channel 等)
│   └── model_maker.py          # 模型工厂
├── utils/                      # 工具函数
│   ├── data_maker.py           # 数据集与数据加载
│   ├── loss_maker.py           # 损失函数
│   ├── train_utils.py          # 训练循环
│   ├── test_utils.py           # 评估与指标计算
│   └── ...
├── configs/                    # Hydra 配置文件
│   ├── train.yaml              # 训练配置
│   ├── model_eval.yaml         # 评估配置
│   └── data_info/              # 数据集配置
├── plot/                       # 论文图表绘制脚本
├── saved_models/               # 预训练模型 (.pt)
├── result_dicts/               # 评估结果缓存 (.pkl)
├── train.py                    # 训练入口
├── test.py                     # 评估入口
├── main.pdf / main.tex         # 论文 PDF 及 LaTeX 源码
├── requirements.txt            # Python 依赖
└── LICENSE                     # MIT 许可证
```

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建 conda 环境（推荐）
conda create -n stjscc python=3.12
conda activate stjscc

# 安装 PyTorch（根据 CUDA 版本调整）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 准备数据集

下载数据集并按如下结构组织：

```
<data_dir>/
├── DIV2K/
│   ├── DIV2K_train_HR/         # 训练集（800 张高清图）
│   └── DIV2K_valid_HR/         # 验证集（100 张高清图）
├── Kodak/                      # Kodak24 测试集
├── Flickr30k/
│   └── flickr30k-images/       # Flickr30k 训练集
├── Set5/HR/
├── Set14/HR/
```

将 `configs/train.yaml` 和 `configs/model_eval.yaml` 中的 `data_dir` 路径修改为你的数据集目录。

> DIV2K 可从 [https://data.vision.ee.ethz.ch/cvl/DIV2K/](https://data.vision.ee.ethz.ch/cvl/DIV2K/) 下载。Kodak24 可从 [http://r0k.us/graphics/kodak/](http://r0k.us/graphics/kodak/) 获取。

### 3. 训练

```bash
# 在 DIV2K 上训练 ST-JSCC，AWGN 信道，PSNR 指标
python train.py \
    model_name=STJSCC \
    chan_type=AWGN \
    rcpp=12 \
    performance_metric=PSNR \
    data_info=DIV2K.yaml \
    device=cuda:0

# 在 Rayleigh 衰落信道下训练
python train.py \
    model_name=STJSCC \
    chan_type=Rayleigh \
    rcpp=12 \
    performance_metric=PSNR

# 训练 SSIM 优化模型
python train.py \
    model_name=STJSCC \
    chan_type=AWGN \
    rcpp=12 \
    performance_metric=SSIM

# 训练其他基线模型
python train.py model_name=ConvJSCC chan_type=AWGN rcpp=12
python train.py model_name=SwinJSCC chan_type=AWGN rcpp=12
python train.py model_name=FAJSCC chan_type=AWGN rcpp=12
```

**主要超参数**（在 `configs/train.yaml` 中配置）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `rcpp` | 每像素信道数的倒数（带宽比） | `12` |
| `total_max_epoch` | 总训练轮数 | `500` |
| `learning_rate` | 学习率 | `1e-4` |
| `batch_size` | 每 GPU 批次大小 | `32` (DIV2K) |
| `chan_type` | 信道类型: `AWGN`, `Rayleigh` | `Rayleigh` |
| `performance_metric` | 优化目标: `PSNR`, `SSIM` | `PSNR` |

### 4. 评估

```bash
# 在 Kodak 数据集上评估训练好的模型
python test.py \
    model_name=STJSCC \
    chan_type=AWGN \
    rcpp=12 \
    test_data=Kodak \
    data_info=DIV2K.yaml \
    device=cuda:0

# 在其他测试集上评估
python test.py model_name=STJSCC test_data=DIV2K chan_type=AWGN
python test.py model_name=STJSCC test_data=Set5 chan_type=AWGN
python test.py model_name=STJSCC test_data=Set14 chan_type=AWGN
```

评估输出包括：多个 SNR 级别（默认为 1, 4, 7, 10 dB）下的 PSNR、SSIM、MS-SSIM、LPIPS，以及 FLOPs、参数量、推理时间和 GPU 显存占用。结果缓存于 `result_dicts/` 目录。

### 5. 使用预训练模型

`saved_models/` 目录提供了预训练模型。将配置文件中的 `save_dir` 指向该目录即可直接评估，无需重新训练。

---

## 📊 模型库

| 模型 | 信道 | rcpp | 参数量 | SSIM (Kodak) | 下载 |
|------|------|------|--------|--------|------|
| STJSCC | AWGN | 12 | 1.01   | 0.7147 | `saved_models/STJSCC_DIV2K_AWGN_rcpp012_PSNR.pt` |
| STJSCC | Rayleigh | 12 | 1.01   | 0.6792 | `saved_models/STJSCC_DIV2K_Rayleigh_rcpp012_PSNR.pt` |
| STJSCC-32ch | AWGN | 12 | 0.45   | 0.6894 | `saved_models/STJSCC_32_channel_DIV2K_AWGN_rcpp012_PSNR.pt` |
| STJSCC-64ch | AWGN | 12 | 1.79   | 0.7348 | `saved_models/STJSCC_64_channel_DIV2K_AWGN_rcpp012_PSNR.pt` |
| STJSCC-4blk | AWGN | 12 | 0.83   | 0.7082 | `saved_models/STJSCC_4_blocks_DIV2K_AWGN_rcpp012_PSNR.pt` |
| STJSCC-6blk | AWGN | 12 | 1.19   | 0.7237 | `saved_models/STJSCC_6_blocks_DIV2K_AWGN_rcpp012_PSNR.pt` |

---

## 📈 复现论文图表

`plot/` 目录中的脚本可复现论文中的图表：

```bash
cd plot
python quantitative.py       # 主定量对比图 (PSNR vs SNR)
python ablation_gl.py        # 消融实验图
python generalization.py     # 泛化性能图
```

> 运行绘图脚本前需先运行 `test.py` 生成所需的 `.pkl` 结果文件。

---

## 🔧 依赖

详见 [`requirements.txt`](./requirements.txt)。核心依赖：

- **PyTorch** ≥ 1.10（推荐 1.13）
- **torchvision** ≥ 0.14
- **Hydra** (hydra-core, omegaconf) — 配置管理
- **fvcore** — FLOPs 计算
- **lpips** — 感知质量指标
- **timm** — Swin Transformer 组件
- **pytorch_wavelets** — DWT/IDWT 操作
- **einops** — 张量操作
- **matplotlib**, **numpy**, **Pillow**

---

## 📝 引用

如果您的研究使用了本工作，请引用：

```bibtex
@article{he2025stjscc,
  title   = {ST-JSCC: Synergizing Structural and Textural Dependencies
             for Robust and Efficient Image Transmission},
  author  = {Rulong He and Mingyang Wan and Haoming Luo and others},
  journal = {Signal Processing},
  year    = {2026}
}
```

---

## 👤 联系方式

**通讯作者：** 万明扬 (Mingyang Wan)

- 📧 邮箱：mingyang_wan@163.com
- 🔗 问题反馈：请 [提交 Issue]([https://github.com/MWan-deeplearner/ST-JSCC/issues](https://github.com/MWan-deeplearner/ST-JSCC/issues))

---

## 📄 许可证

本项目基于 [MIT License](./LICENSE) 发布。

---

## 🙏 致谢

代码参考与灵感来源：

- [SwinJSCC](https://github.com/semcomm/SwinJSCC) — 基于 Swin Transformer 的 JSCC 基线
- [GDN (Generalized Divisive Normalization)](https://arxiv.org/abs/1611.01704)
- [DIV2K 数据集](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
