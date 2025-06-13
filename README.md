# ARM: Asymmetric Reinforcing against Multimodal Representation Bias

This repository contains the official implementation of our paper:  
**"Asymmetric Reinforcing against Multimodal Representation Bias"**  
Developed based on the [MMPretrain](https://github.com/open-mmlab/mmpretrain) framework.

![image](https://github.com/user-attachments/assets/f4078619-4d9d-4eb2-8330-cd1787b7ea8f)

## 🔍 Introduction

The strength of multimodal learning lies in its ability to integrate information from various sources, providing rich and comprehensive insights. However, in real-world scenarios, multi-modal systems often face the challenge of dynamic modality contributions, the dominance of different modalities may change with the environments, leading to suboptimal performance in multimodal learning. Current methods mainly enhance weak modalities to balance multimodal representation bias, which inevitably optimizes from a partialmodality perspective, easily leading to performance descending for dominant modalities. To address this problem, we propose an Asymmetric Reinforcing method against Multimodal representation bias (ARM). Our ARM dynamically reinforces the weak modalities while maintaining the ability to represent dominant modalities through conditional mutual information. Moreover, we provide an in-depth analysis that optimizing certain modalities could cause information loss and prevent leveraging the full advantages of multimodal data. By exploring the dominance and narrowing the contribution gaps between modalities, we have significantly improved the performance of multimodal learning, making notable progress in mitigating imbalanced multimodal learning.

<!--## 🛠️ Features

- Dynamically computes **conditional mutual information** as fusion weights.
- Explicitly narrows **marginal contribution gaps** across modalities.
- Compatible with **MMPretrain-style pipelines**.
- Supports training on custom multimodal datasets.
-->

## 📂 Project Structure

```
arm/
├── configs/              # Configuration files for training
├── mmpretrain/           # Core implementation of ARM
├── tools/                # Training and evaluation scripts
├── work_dirs/            # Results and training logs
└── README.md
```

## 🚀 Getting Started

### 1. Environment Setup

Install [MMPretrain](https://github.com/open-mmlab/mmpretrain) and dependencies:

> Ensure `mmpretrain v1.2.0` is installed and available in your environment.

### 2. Dataset Preparation

Please refer to [OGM-GE](https://github.com/GeWu-Lab/OGM-GE_CVPR2022).

### 3. Training

Run the training using the provided config:

```bash
python tools/train.py configs/arm/resnet18_ks.py
```

### 4. Evaluation

```bash
python tools/test.py configs/arm/resnet18_ks.py checkpoints/best.pth
```

## 📦 Pretrained Models and Training Logs

We provide pretrained models and training logs for reproducibility:

| Dataset     | Backbone | Config | Checkpoint | Training Log |
|-------------|----------|--------|------------|--------------|
|  Kinetics-Sounds | ResNet18 | [resnet18_ks.py](configs/arm/resnet18_ks.py) | [Download](work_dirs/) | [Log](work_dirs/train_log.log) |

> 📝 *Note: Please ensure to use the same config and environment for consistent results.*

## 📄 Citation

If you find our work useful, please consider citing:

```bibtex
@article{your2025arm,
  title={Asymmetric Reinforcing against Multimodal Representation Bias},
  author={Your Name and Others},
  journal={ArXiv},
  year={2025}
}
```

## 🤝 Acknowledgements

This project is built upon the excellent [MMPretrain](https://github.com/open-mmlab/mmpretrain) framework by OpenMMLab and [OGM-GE](https://github.com/GeWu-Lab/OGM-GE_CVPR2022) by GeWu-Lab.
