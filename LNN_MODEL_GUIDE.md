# 🧠 液态神经网络 (LNN) 完整指南

## 概述

本仓库包含了液态神经网络 (Liquid Neural Networks, LNN) 的完整实现，包括所有论文中提到的变体模型。这些模型专为时间序列预测、机器人控制、边缘设备部署等场景设计。

如果你还不熟悉 LNN 的背景、公式推导、训练流程和部署方案，建议先读：[LNN 原理入门：给高中生的液态神经网络指南](docs/guides/LNN_PRINCIPLES_FOR_BEGINNERS.md)。

## 📦 已实现的模型

### 核心模型

| 模型 | 文件 | 描述 | 适用场景 |
|------|------|------|----------|
| **LTC** | `lnn/core/ltc.py` | 原始液态时间常数网络 | 需要高精度的场景 |
| **CfC** | `lnn/core/cfc.py` | 闭式连续时间网络 | 速度与精度平衡的首选 |

### 变体模型 (Variants)

所有变体模型位于: `lnn/core/variants.py`

| 模型 | 描述 | 最佳适用 |
|------|------|----------|
| **StrictCfC** | 严格约束的CfC | 高精度需求 |
| **HybridCfC** | 混合门控机制的CfC | 平衡性能 |
| **CTLTC** | 连续时间LTC | 长序列依赖 |
| **LiquidS4** | LNN+S4结合 | 超长序列建模 |
| **LRC** | 液态电阻电容网络 | 生物合理性 |
| **CfC-DT** | 支持显式时间步的CfC | 不规则时间采样数据 |
| **Euler-LTC-DT** | 使用Euler方法的简化LTC | 边缘设备/微控制器 |

## 🎯 LNN 对比大模型的核心优势

### 1. 🏆 参数效率 (Parameter Efficiency)

**LNN**: 仅需数千参数 (32-128 隐藏单元足够)
- CfC: ~3,300 参数
- LTC: ~2,300 参数

**对比大模型**:
- GPT-2: ~124M 参数 (40,000x 更大!)
- LLaMA-7B: ~7B 参数 (2,000,000x 更大!)

### 2. ⚡ 计算速度

**LNN 优势**:
- CfC: O(N) 时间复杂度，无矩阵指数运算
- 无需自注意力机制
- 可在边缘设备实时运行

### 3. 🧠 连续时间建模

**自然建模物理系统**:
- 处理非均匀采样数据 (CfC-DT)
- 更好的训练域外推能力
- 时间常数具有物理意义

### 4. 🎯 泛化能力

- 对分布偏移鲁棒
- 小数据即能有效训练
- 对输入扰动敏感性低

### 5. 🔌 边缘设备部署

- 微控制器可运行 (Euler-LTC-DT)
- 内存占用极低
- 节能推理

### 6. 📊 可解释性

- 连续动力学更易解释
- 时间常数可理解
- 调试和分析更简单

## 🚀 快速开始

### 安装

```bash
cd LNN
pip install -e .
```

### 基本使用

```python
import torch
from lnn.core.cfc import CfCNetwork
from lnn.core.variants import (
    CfCDTNetwork,
    EulerLTCDTNetwork,
)

# 创建 CfC 模型 (推荐首选)
model = CfCNetwork(
    input_size=1,      # 输入特征数
    hidden_size=32,    # 隐藏单元数
    output_size=1,     # 输出维度
    num_layers=1,      # 层数 (通常1层足够)
    return_sequences=True  # 返回完整序列
)

# 前向传播
x = torch.randn(32, 100, 1)  # batch, seq_len, features
output = model(x)  # Shape: (32, 100, 1)
```

### 处理不规则时间数据

```python
from lnn.core.variants import CfCDTNetwork

model = CfCDTNetwork(1, 32, 1)
x = torch.randn(32, 100, 1)
dt = torch.randn(32, 100, 1)  # 显式时间步
output = model(x, dt=dt)
```

## 📜 可用脚本

| 脚本 | 用途 |
|------|------|
| `scripts/verify_all_models.py` | 验证所有模型实现 |
| `scripts/demo_lnn_advantages.py` | 演示 LNN 优势 |
| `scripts/experiment_all_variants.py` | 完整训练对比实验 |
| `scripts/tutorial_how_to_use.py` | 完整教程 |

## 📊 运行验证

```bash
# 验证所有模型
python scripts/verify_all_models.py

# 演示 LNN 优势
python scripts/demo_lnn_advantages.py
```

## 🎓 何时使用 LNN vs 大模型

### ✅ 使用 LNN

- 时间序列预测和预报
- 机器人控制与自主系统
- IoT传感器数据处理
- 实时边缘应用
- 物理系统建模
- 节能AI

### ✅ 使用大模型

- 自然语言理解与生成
- 复杂模式识别 (图像、音频)
- 知识密集型任务
- 创意生成与内容创作
- 通用智能任务

## 📁 项目结构

```
LNN/
├── lnn/
│   ├── core/              # 核心模型
│   │   ├── ltc.py        # LTC网络
│   │   ├── cfc.py        # CfC网络
│   │   ├── variants.py   # 所有变体模型 🌟
│   │   ├── trainer.py    # 训练器
│   │   └── __init__.py
│   ├── data/             # 数据工具
│   └── utils/            # 工具函数
├── scripts/              # 实用脚本 🌟
├── docs/                 # 文档 🌟
├── tests/                # 测试
└── papers/               # 相关论文
```

## 🔬 优化建议

查看: `docs/OPTIMIZATION_STRATEGIES.md`

### 快速参考

| 场景 | 推荐模型 |
|------|----------|
| 最佳精度 | LTC, CTLTC, LRC |
| 最佳速度 | CfC, HybridCfC, Euler-LTC-DT |
| 长序列 | LiquidS4 |
| 不规则时间 | CfC-DT |
| 边缘部署 | Euler-LTC-DT |
| 平衡性能 | CfC, HybridCfC |

### 超参数建议

```python
hidden_size: 32-128    # LNN不需要大隐藏层
num_layers: 1-2         # 通常1层足够
lr: 0.001-0.003        # 较小学习率更稳定
batch_size: 32-128
```

## 📚 参考文献

1. Liquid Time-constant Networks (原始LTC)
2. Closed-form Continuous-time Neural Networks (CfC)
3. Liquid Resistive-Capacitive Networks (LRC)
4. S4: Structured State Spaces for Sequence Modeling (LiquidS4灵感)

## 🎉 完成情况

✅ 所有9个模型已实现并验证  
✅ 完整API一致性  
✅ 验证脚本可用  
✅ 完整文档  
✅ 优化策略文档  
✅ 教程和示例  

## 📞 下一步

1. 运行: `python scripts/demo_lnn_advantages.py` - 查看优势演示
2. 阅读: `docs/OPTIMIZATION_STRATEGIES.md` - 优化指南
3. 阅读: `docs/IMPLEMENTATION_SUMMARY.md` - 实现总结
4. 尝试: 在你自己的时间序列数据上训练!

---

**祝使用愉快!** 🚀
