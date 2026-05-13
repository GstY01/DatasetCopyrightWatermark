# Dataset Copyright Watermark (数据集版权水印工具包)

本项目是一个基于后门攻击方法的数据集版权保护工具包，支持对多种数据集（如 CIFAR-10、CIFAR-100、自定义数据集）植入版权水印，并提供水印验证功能。

## 功能特点

- 支持 **12 种后门攻击方法** 作为水印植入方式
- 内置支持 **CIFAR-10、CIFAR-100、GTSRB** 等常用数据集
- 支持 **自定义数据集**（需按 ImageFolder 格式组织）
- 完整的 **水印植入** 和 **版权验证** 流程
- 灵活的配置和参数调整
- 详细的日志和输出信息

## 支持的攻击方法

| 攻击方法 | 说明 |
|---------|------|
| BadNets | 固定位置添加补丁 |
| Blended | 图案混合 |
| Blind | 盲水印 |
| BppAttack | 比特率攻击 |
| FTrojan | 特征特洛伊 |
| LC | 低频扰动 |
| LF | 标签翻转 |
| ReFool | 反射欺骗 |
| SIG | 信号注入 |
| SSBA | 静态后门 |
| TrojanNN | 神经网络特洛伊 |
| WaNet | 弹性扭曲 |

## 环境要求

- Python 3.8+
- PyTorch 1.10.0
- TorchVision 0.11.0
- NumPy 1.23.5
- PyYAML 6.0
- tqdm
- Matplotlib
- scikit-learn

## 安装

```bash
# 克隆项目
git clone https://github.com/yourname/DatasetCopyrightWatermark.git
cd DatasetCopyrightWatermark

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 基本使用

运行主程序，它将演示完整的水印植入和验证流程：

```bash
python main.py
```

### 2. 代码示例

#### 水印植入

```python
import torch
from data.dataset_loader import load_cifar10
from core.embed import WatermarkEmbed

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
dataset = load_cifar10(train=True)

# 初始化水印植入器
embedder = WatermarkEmbed(
    attack_name="badnets",  # 选择攻击方法
    dataset=dataset,
    dataset_name="cifar10",
    device=device
)

# 执行水印植入
data_path, key_path = embedder.run(
    poison_rate=0.1,    # 投毒比例
    target_label=0      # 目标标签
)
```

#### 版权验证

```python
from core.verify import CopyrightVerify

# 初始化验证器
verifier = CopyrightVerify(
    key_path="./output/keys/cifar10_badnets_key.yaml",
    device=device
)

# 验证版权
is_infringed, asr = verifier.check_infringement(
    suspect_data_path="./output/poisoned/cifar10_badnets/poisoned_dataset.pt",
    threshold=0.8
)
```

#### 使用自定义数据集

```python
from data.dataset_loader import load_custom_dataset

# 加载自定义数据集（需按 ImageFolder 格式组织）
custom_dataset = load_custom_dataset("./data/my_custom_dataset")

# 植入水印
embedder = WatermarkEmbed("wanet", custom_dataset, "custom", device)
data_path, key_path = embedder.run(poison_rate=0.15)
```

## 项目结构

```
DatasetCopyrightWatermark/
├── attacks/              # 攻击方法实现
│   ├── base_attack.py   # 基类
│   ├── badnets.py
│   ├── blended.py
│   ├── ...
│   └── wanet.py
├── configs/             # 配置文件
│   ├── badnets.yaml
│   ├── blended.yaml
│   └── ...
├── core/                # 核心模块
│   ├── embed.py        # 水印植入
│   ├── verify.py       # 版权验证
│   ├── attack_adapter.py  # 攻击适配器
│   └── dataset_loader.py  # 数据集加载
├── data/                # 数据相关
│   ├── dataset_loader.py
│   └── __init__.py
├── utils/               # 工具模块
│   ├── logger.py
│   ├── io.py
│   └── __init__.py
├── output/              # 输出目录（自动生成）
│   ├── poisoned/       # 投毒数据集
│   └── keys/           # 水印密钥
├── main.py             # 主程序
├── requirements.txt    # 依赖
└── README.md           # 说明文档
```

## 配置文件

每种攻击方法都有对应的配置文件在 `configs/` 目录下，您可以根据需要调整参数。

## 输出说明

程序运行后会在 `output/` 目录生成以下内容：

- `output/poisoned/{dataset}_{attack}/`: 投毒后的数据集
  - `poisoned_dataset.pt`: 带水印的数据集
  - `clean_dataset.pt`: 干净的数据集
- `output/keys/{dataset}_{attack}_key.yaml`: 水印密钥（包含水印类型、参数等信息）

## 注意事项

1. **密钥保管**: 请妥善保管生成的水印密钥（`.yaml` 文件），这是验证版权的唯一凭证
2. **投毒比例**: 建议投毒比例在 0.05~0.2 之间，过高会影响数据集正常使用
3. **目标标签**: 不同的版权可以使用不同的目标标签作为标识



## 项目声明 Project Statement

本项目的作者及单位：

```
The author and affiliation of this project:
项目名称（Project Name）：Dataset Copyright Watermark
项目作者（Author）：Weibin Chen, Yongdong Wu
作者单位（Affiliation）：暨南大学网络空间安全学院（College of Cyber Security, Jinan University）
```

若你使用本项目用于论文的实验，你可以引用本项目，latex版本引用如下：

If you use this project for the experiment of the paper, you can cite this project, the latex version is cited as follows:

```
@misc{datasetcopyrightwatermark,
  author       = {Chen, Weibin},
  title        = {Dataset Copyright Watermark: A Backdoor Attack Based Dataset Copyright Protection Toolkit},
  year         = {2026},
  howpublished = {\url{https://github.com/GstY01/DatasetCopyrightWatermark}}
}
```

word版本引用如下：

```
The word version is quoted as follows:
W. Chen, Dataset Copyright Watermark: A Backdoor Attack Based Dataset Copyright Protection Toolkit, https://github.com/GstY01/DatasetCopyrightWatermark (2026).
```

当你公开了基于本项目的代码时，你必须注明原项目作者及出处：

When you disclose the code based on this project, you must indicate the original project author and source:

```
Author: Weibin Chen
Project: `https://github.com/GstY01/DatasetCopyrightWatermark`
```
