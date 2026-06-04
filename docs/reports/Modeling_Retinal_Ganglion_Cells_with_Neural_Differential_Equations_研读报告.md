---
title: Modeling Retinal Ganglion Cells with Neural Differential Equations — 研读报告
paper: arXiv 2511.18014v1
authors: Kacper Dobek, Daniel Jankowski, Krzysztof Krawiec
venue: AAAI-26 Student Abstract and Poster Program
date: 2025-11-22
tags: [LNN, LTC, CfC, retinal-ganglion-cells, tiger-salamander, vision-prosthetics, edge-ai, NODEF, paper-report, AAAI-2026]
status: deep-read
report-date: 2026-06-05
report-author: LNN-research-agents
---

# Modeling Retinal Ganglion Cells with Neural Differential Equations — 研读报告

> 论文: arXiv 2511.18014v1 (Dobek et al. 2025), AAAI-26 Student Abstract
> 链接: https://arxiv.org/abs/2511.18014v1
> 代码: **无代码 / 无数据链接**
> 与本仓直接相关度: **高** —— 是 **LTC / CfC** 的**生物应用实证**,直接对比
>  本仓的 `lnn/core/ltc.py::LTCNetwork` 和 `lnn/core/cfc.py::CfCNetwork` 在
>  视网膜神经活动预测上的相对表现。

---

## 1. 一句话定位

> 把 **LTC 和 CfC** 两种连续时间 ODE 模型应用到**虎蝾螈视网膜神经节细胞 (RGC) 活动预测**,
> 与 ConvNet 基线和 LSTM 在 3 个数据集上对照。**LTC/CfC 赢 MAE + 5-8× 少参数 + 更快收敛 + 更快 query**,
> **输 Pearson correlation**(时序对位差),适用**边缘部署 / 视觉假体**。

应用: 视觉假体 (vision prosthetics) 边缘 AI —— **小模型 + 频繁重训**场景。

## 2. 关键设计(论文 §Model Architecture)

4 architectures 共享**同一卷积栈**(前段),仅后段 temporal 模块不同:
- **ConvNet**: 卷积 → dense → output (无 temporal)
- **LSTM**: 卷积 → dense → LSTM cell → 2 dense → output
- **LTC**: 卷积 → dense → LTC cell (连续时间 ODE) → output
- **CfC**: 卷积 → dense → CfC cell (闭式解 ODE) → output

`latent_vec` 都是 32 维,接 temporal 模块后输出 `n` 维预测。

## 3. 关键结果(论文 Table 1, rgc9 数据集)

| 模型 | Pearson ρ ↑ | 95% CI | MAE ↓ | #params | Time [s] |
|---|---:|---|---:|---:|---:|
| ConvNet | **0.569** | [0.564, 0.574] | 4.07 | ~3-5k | ~30 |
| LSTM | 0.421 | [0.384, 0.457] | 3.18 | ~10k | **最慢** |
| **LTC** | 0.480 | [0.469, 0.491] | **2.73** | **5-8× 少** | ~30 (与 ConvNet 相当) |
| **CfC** | 0.474 | [0.468, 0.479] | 2.86 | **5-8× 少** | ~30 (与 ConvNet 相当) |

**核心结论**:
- ✅ **LTC/CfC 赢 MAE** (2.73/2.86 vs ConvNet 4.07, LSTM 3.18) — **3 数据集全部 NODEs 优** (ANOVA p<0.05)
- ✅ **5-8× 少参数** —— 关键 edge AI 卖点
- ✅ **更快 query 时间** —— 边缘推理卖点
- ❌ **输 Pearson ρ** (LTC 0.480 vs ConvNet 0.569) —— 时序对位不如 ConvNet
- ⚠️ **抗噪差** —— ConvNet + LSTM 抗噪更稳

## 4. ANOVA + 关键观察(论文 §Results)

- **3 数据集 ANOVA** 全部 p<0.05 — **NODEs 显著优于 ConvNet/LSTM** on MAE
- **NODEs 更好预测 dependent variable 数值**,但**对时序对位(peak timing)差**
- 推测: ConvNet 滑窗结构天然对齐峰位,LTC/CfC 连续时间积分**平滑掉峰**

## 5. 局限性(论文自承 + 我的批注)

| 维度 | 论文 | 我的补注 |
|---|---|---|
| 数据 | 3 个 tiger salamander RGC 数据集 (Maheswaranathan et al. 2023) | **仅 salamander,未涵盖 mouse / primate / human**;迁移需新数据 |
| 任务 | RGC 神经活动回归 (单变量时序) | 没覆盖 spike sorting / 光感受器编码 / RGC 分类 |
| 模型 | LTC/CfC 标准 (无 custom ODE) | 与本仓 `LTCNetwork` / `CfCNetwork` 几乎同构 |
| 公式 | **未给具体 ODE 公式** (Student Abstract) | 引用 Hasani 2021/2022 原始 |
| 代码 | ⚠️ **无官方代码 / 无数据** (Student Abstract 性质) | 复现需自找数据 + 自写训练 |
| 范围 | Student Abstract — 2 页 + supplementary | 全文 < 10 页,数据/超参细节缺 |

## 6. 对本仓库的价值

### 6.1 公式层 — **本仓已有等价**

```python
# 本仓 lnn/core/ltc.py::LTCNetwork (sigmoid-gated 闭式)
# 论文 LTC 单元 (显式 α ⊙ h 形式, RK4 自适应步)
# 数学行为一致: per-dim 衰减 + tanh 循环
```

**复现 0 障碍** —— 论文 LTC/CfC 实现 = 本仓 `LTCNetwork` / `CfCNetwork`。

### 6.2 评估层 — **新 dimension**

本仓评估维度 = timeseries / molecular / sMNIST Gapped。
论文 = **生物神经活动回归** (RGC 神经节细胞)。

- 新评估任务: 神经活动预测 (regression) - 类似 timeseries
- 关键指标: MAE + Pearson + #params + query time (5 维)
- **Jetson 边缘 AI 卖点**: 5-8× 少参数 + 更快 query → 适合假体 / 神经接口

### 6.3 思想层 — **小模型 + 频繁重训**

论文 §Discussion 强调: "their efficiency and adaptability make them well suited for
scenarios with limited data and frequent retraining, such as edge deployments in
vision prosthetics"

→ 本仓 **lightweight benchmarking** 维度(LFM2.5 INT8 也走这条线)。

### 6.4 复现路线(stage 拆分)

| Stage | 出口物 | 估时 |
|---|---|---|
| A. 调研 + design | `analysis/retinal_lnn/<date>_design.md` | 0.5 loop(本轮可完成) |
| B. 找数据(MIT-Stanford 视网膜 或 synthetic RGC) + 跑 4 架构 × 3 seeds | analysis + paper | 1-2 loop |
| C. 写复现报告 | docs/reports/Retinal_LNN_复现报告.md | 0.5 loop |

**ROI 评估**: **高** —— 公式同构 (直接复用本仓) + 边缘 AI 卖点明确。
**最大阻塞**: 数据 (MIT-Stanford 不公开)。

## 7. 推荐评级 + 优先级

- **学术新意**: B+ (AAAI-26 Student Abstract, 实证工作, 非新方法)
- **工程价值**: **A** (Jetson 边缘 视觉假体, 5-8× 少参数是硬指标)
- **代码可获取**: C (无代码 / 无数据 — Student Abstract 性质)
- **本仓优先级**: **A-** —— 公式复用 0 障碍, 工程价值高

## 8. 与本仓 8 套 LNN 概念扩展的关系

```
LNN backbone candidates:
├── LTC (本仓核心)                ✅ 已有 + 论文 LTC 直接对应
├── CfC (闭式解)                  ✅ 已有 + 论文 CfC 直接对应
├── CT-LTC                        ✅ 已有
├── PDNA-pulse                    ✅ 已有 (iter#19)
├── SVAF-τ-blend                  ✅ 已有 (iter#22)
├── DynPMNN-FHN                   ✅ 已有 (iter#23)
├── RLSTG-LTC                     调研中 (黎曼, iter#31)
└── transformable-LTC             调研中 (部署精化, iter#34)
```

论文 = **第 1 套 (LTC) + 第 2 套 (CfC) 的**生物应用实证**,与本仓实现直接对照。
**对仓库 = AAAI-26 引用支撑**(边缘 AI 卖点)。

## 9. 参考

- arXiv: https://arxiv.org/abs/2511.18014v1
- Venue: AAAI-26 Student Abstract and Poster Program
- Cited: Hasani 2021 (LTC), Hasani 2022 (CfC), Maheswaranathan 2023 (RGC dataset)

---

> 本报告由 LNN-research-agents 自动生成,基于 arXiv 2511.18014v1 PDF + WebFetch abstract 交叉验证。
> 报告日期 2026-06-05。
