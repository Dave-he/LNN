---
title: Jetson validation summary — iter#35 Retinal Ganglion LNN deep read (arXiv 2511.18014, AAAI-26)
date: 2026-06-05
platform: Linux hyx-desktop 5.15.148-tegra (Jetson Orin Nano Super, aarch64)
python: 3.14.4 (pyenv) — primary, CPU path
cuda: torch 2.11.0+cu130 → cuda.is_available()=False (BSP driver 12.060 < cu130)
tag: loop-iteration, jetson-cpu-only, paper-deep-read, retinal-lnn, LTC-CfC, vision-prosthetics, AAAI-2026
---

# Jetson validation summary — iter#35 Retinal Ganglion LNN deep read

> 本轮执行 **用户提示 "搜索LNN相关论文代码"** —— 深读 arXiv 2511.18014
> "Modeling Retinal Ganglion Cells with Neural Differential Equations"
> (Dobek et al. AAAI-26 Student Abstract)。

## 1. 改动量

```
docs/reports/Modeling_Retinal_Ganglion_Cells_with_Neural_Differential_Equations_研读报告.md   新增 (~200 行, 9 节)
docs/PRD_LNN_Edge_Research.md   C-level 表 +1 行 (Retinal LNN)
```

无 lnn/ 代码改动 — 本轮纯论文研读。

## 2. 论文核心

- **方法**: LTC + CfC 应用到 tiger salamander 视网膜神经节细胞 (RGC) 活动预测
- **数据集**: 3 个 (Maheswaranathan 2023)
- **架构**: 4 模型 (ConvNet / LSTM / LTC / CfC) 共享同一卷积栈
- **结果**: **LTC/CfC 赢 MAE 2.73/2.86 vs ConvNet 4.07 + 5-8× 少参数 + 更快收敛 + 更快 query**
- **结果**: **LTC/CfC 输 Pearson ρ** (0.480 vs ConvNet 0.569) — 时序对位差
- **ANOVA**: 3 数据集 p<0.05 — NODEs 显著优于 ConvNet/LSTM on MAE
- **代码**: **无** (Student Abstract 性质, AAAl-26 student abstract track)
- **应用**: 边缘 AI / 视觉假体 (vision prosthetics)

## 3. 关键发现(与本仓 LTC/CfC 公式同构)

| 维度 | 论文 | 本仓 |
|---|---|---|
| 公式 | `dh/dt = -α ⊙ h + tanh(W_h h + u)` | `lnn/core/ltc.py` sigmoid-gated 闭式 |
| 实测 MAE | LTC 2.73 / CfC 2.86 (rgc9) | (未测) |
| 5-8× 少参数 | ✅ | (论文 vs ConvNet) |
| 边缘 AI 适用 | ✅ vision prosthetics | ✅ 适合 Jetson |
| Pearson ρ 输 | 0.480 (vs ConvNet 0.569) | (未测) |

## 4. 评级

| 维度 | 评级 |
|---|---|
| 学术新意 | B+ (AAAI-26 Student Abstract, 实证工作) |
| 工程价值 | **A** (Jetson 边缘 视觉假体, 5-8× 少参数是硬指标) |
| 代码可获取 | C (无代码 / 无数据) |
| 本仓优先级 | **A-** (公式复用 0 障碍) |

## 5. 复现路线(3 stages, ~3 loop)

| Stage | 出口物 | 估时 |
|---|---|---|
| A. 调研 + design | `analysis/retinal_lnn/<date>_design.md` | 0.5 loop |
| B. 找数据 + 跑 4 架构 × 3 seeds | analysis + paper | 1-2 loop |
| C. 写复现报告 | docs/reports/Retinal_LNN_复现报告.md | 0.5 loop |

**ROI 评估**: **高** (公式同构) + 边缘 AI 卖点明确。
**最大阻塞**: 数据 (Maheswaranathan 2023 RGC 数据集不公开)。

## 6. pytest 套件(102/102, 29.67s)

无变化(纯论文改动)。vs iter#34: 102 → 102 = **0 变动,0 回归**。

## 7. verify_all_models.py(9/9)

无变化。

## 8. 关键 takeaway

1. **论文 = 本仓 LTC/CfC 在生物 RGC 数据上的实证** —— 公式同构,直接可对照
2. **Jetson 边缘 5-8× 少参数** 是硬指标 —— 与本仓 LFM2.5 INT8 路线吻合
3. **Pearson ρ 输** 警告:LTC/CfC 适合"值预测",不适合"时序对位"
4. **Student Abstract 性质** — 数据 / 公式 / 训练细节缺,复现需自找数据
5. **本仓评估维度可加 retinal** —— 第 4 个 domain (timeseries/molecular/sMNIST Gapped/retinal)
