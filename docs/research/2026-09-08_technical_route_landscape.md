---
title: "LNN 最新论文技术路线图 — 2026-02 至 2026-08 跨 25 篇 + 开源生态"
date: 2026-09-08
source: "papers/daily/2026-09-08_lnn_research.json (25 篇) + analysis/repo_watchlist/2026-09-08_lnn_open_source_watchlist.md (11 repo + 17 HF) + 9/7-9/8 已 deep-dive 两篇"
related_digest: "docs/daily/2026-09-08_LNN_research_digest.md"
lineage_reports:
  - "docs/research/2026-09-07_stochastic_liquid_deformation_cfc_sde_report.md"
  - "docs/research/2026-09-08_dropin_cfc_deformation_field_report.md"
tag: technical-landscape
---

# LNN 最新论文技术路线图 (2026-02 → 2026-09)

> 基于 2026-09-08 自动 digest 命中 25 篇候选 (arXiv 2026-02-25 ~ 2026-08-27) + 当日 11 个 GitHub 仓库 + 17 个 HF 模型 + 9/7 与 9/8 两篇已 deep-dive 的报告。
> 横向抽取 9 条技术路线, 纵向串出 6 个横向主题, 并映射到本仓 `liquid_*` 路径已有工作。

## 一、9 条技术路线

### 路线 1 · 连续时间架构替换 (Continuous-Time Architectural Substitution)

**主线**: 用 ODE/CfC 闭式替换传统 MLP / Transformer / RNN 模块, 形成"drop-in 连续时间场"。

| 论文 | 时间 | 核心 |
|---|---|---|
| **2606.07670** (Drop-in CfC Deformation Field) | 2026-06-04 | D-3DGS MLP → CfC cells, σ(gate) 两状态插值, 闭式无 solver |
| **2608.28702** (CfC → SDE) | 2026-08-27 | 同架构再加 Gaussian 噪声, σ→0 退化为 2606.07670 |

**Lineage 闭环** (本仓 9/8 deep-dive 已展开):
```
MLP deformation field (D-3DGS)
    → CfC stack (2606.07670, 9/8 deep-dived)
    → CfC stack + Gaussian noise (2608.28702, 9/7 deep-dived)
```

**关键设计模式**: σ(gate) 插值机制 — 把"两个候选状态"通过 sigmoid 时间门平滑插值, 让 loss landscape 结构性具备 t 的连续性, 而不是靠优化器偶然发现。

### 路线 2 · ODE-SDE 衍生 (ODE/SDE Derivatives & Physics-driven)

把 ODE/CfC 推广到物理建模、随机微分方程、振动动力学等连续系统。

| 论文 | 核心 | 应用域 |
|---|---|---|
| **2606.15571** Liquid RFM | 随机特征方法解时变 PDE | 时间偏微分方程 |
| **2603.00153** Pulse-Driven Neural Architecture | 可学习振动动力学 | 鲁棒连续时间序列 |
| **2605.08176** Physics-Modeled Neural Networks | 物理建模嵌入网络 | 物理一致性 |
| **2605.24047** EMMA | 多模态数据提取多物理参数 | 物理参数回归 |

**共同特征**: 把 ODE 当一等公民, 不只是序列模型的一个 trick, 而是显式建模物理过程的工具。

### 路线 3 · 时序预测 / 序列建模 (Time-Series Forecasting & Sequence Modeling)

LNN 的老本行, 但**新趋势是图结构 + 跨域**。

| 论文 | 核心 | 任务 |
|---|---|---|
| **2606.21295** Topological Neural Dynamics | 神经元级别 (非层级) 序列建模 | 替代层级 RNN/LSTM/Transformer |
| **2607.01986** Liquid Latent State Dynamics | 可解释的液体潜变量 | 涡轮发动机退化 (C-MAPSS) |
| **2606.15807** Memory-Augmented Graph LTC | 图结构 + 跨域迁移 | 交通状态预测 |
| **2604.24788** LNN for Natural Gas Price | LNN 直接回归 | 天然气现货价格时序 |

**关键演化**: 从"序列预测" → "跨域序列预测" (2606.15807) → "拓扑序列建模" (2606.21295) — 三个方向都在解同一痛点: 传统 RNN/LSTM 的层级归纳偏置不够灵活。

### 路线 4 · 多模态融合 / 视觉 (Multi-modal Fusion & Vision)

| 论文 | 核心 | 任务 |
|---|---|---|
| **2606.26849** Liquid Fusion of Heterogeneous Representations | 异质表征融合 | 通用 SOD (显著目标检测) |
| **2606.20491** GazeLNN | LNN 注意力预测 | 自动驾驶主动感知 |
| **2607.12909** Real-time Fall Detection | 视觉 + LNN | 低功耗边缘平台 |
| **2603.00459** Continuous-Time Mask Refinement | 可解释 mask 精修 | 医学图像分割 |

**主旋律**: LNN 在视觉任务上**没成为主流 backbone** (CNN/Transformer 仍占主导), 而是用在**时间连续性关键**的子任务上 (注意力、refinement、姿态稳定)。

### 路线 5 · 决策 / 规划 / 控制 (Decision / Planning / Control)

| 论文 | 核心 | 任务 |
|---|---|---|
| **2608.03041** PLAN (Parallel Liquid-Inspired) | 并行液体近似网络 | FJSP (柔性作业车间) |
| **2604.18274** LiquidTAD | 并行液体时序松弛 | 时序动作检测 |
| **2607.08283** TFP (Temporally Conditioned Memory-Fusion) | 暂存记忆融合 | 视觉运动策略 (VLA) |
| **2603.27058** Liquid + Mixture Density Heads | 混合密度头 | 模仿学习 |

**本仓直接相关**: PLAN (2608.03041) — 我们有 r302-r304 PLAN-CfC bench 系列, 这是同期工作, 路线一致。

### 路线 6 · 听觉 / 音频 / 音乐 (Audio / Music)

| 论文 | 核心 | 任务 |
|---|---|---|
| **2606.19579** FlowFake | 液体网络检测 audio deepfake | 跨数据集泛化 |
| **2604.10815** MeloTune | 端上情绪学习 + P2P mood coupling | 主动音乐推荐 |

**亮点**: FlowFake 提出"训练集 → 未知合成器"泛化问题, LNN 的时间稳定性是关键; MeloTune 是端上 + 多设备协同的实验性场景。

### 路线 7 · 安全 / 对抗 (Security / Adversarial)

| 论文 | 核心 | 任务 |
|---|---|---|
| **2604.02149** AEGIS | 热力学 SSM + 信息熵 | 零日网络逃逸检测 |

**单点探索**: 用 thermodynamic state space + entropy-guided immune system 做零日检测, 液态概念没有显式出场 (score=1), 但方向相关。

### 路线 8 · 理论 / 拓扑基础 (Theory & Topology)

| 论文 | 核心 | 贡献 |
|---|---|---|
| **2606.21295** Topological Neural Dynamics | 神经元级 (非层级) 框架 | 挑战 RNN/LSTM/Transformer 共同结构前提 |
| **2603.00153** Pulse-Driven Neural Architecture | 可学习振动 | 鲁棒连续时间序列处理 |

**与路线 3 重叠**: 这两篇同时是序列建模创新和理论基础。

### 路线 9 · 对比 / 评测 / 混合 (Comparative & Hybrid)

| 论文 | 核心 | 任务 |
|---|---|---|
| **2605.27467** Comparative Analysis LNN vs LSTM | 临床场景对比 | 序列模式识别鲁棒性/效率/临床实用性 |
| **2604.03955** Symbolic-Vector Attention Fusion | 液体 + 符号-向量注意力融合 | 集体智能 |
| **2604.07219** Robust Hybrid Beamforming | LNN + 液晶天线 (literal liquid crystal) | 混合波束赋形 |

**亮点**: 2604.07219 是**字面意义上的"液体+液体"** — 液晶 (硬件) + LNN (算法) 做混合波束, 跨学科协同。

## 二、6 个横向主题 (Cross-cutting Themes)

### 主题 A · "时间门 σ(gate) 平滑插值"作为通用元模式

不止 2606.07670 / 2608.28702 — 路线 3 的 Graph LTC (2606.15807), 路线 5 的 LiquidTAD (2604.18274), 路线 8 的 Pulse-Driven (2603.00153) 都用了**类输入门 + 两状态插值**的元结构。
**含义**: 这是一个 LNN 社区的事实标准, 而不是个别 trick。
**对本仓意义**: 本仓 `liquid_*` 路径下所有 CfC 变体都应保留 σ(gate) 抽象作为统一接口, 便于将来跨论文共享闭式解。

### 主题 B · "From continuous-time to discrete deployment"

理论 ODE/CfC → 实际边缘部署: 这条链在多个论文里反复出现。
- 路线 4 视觉: 2607.12909 (low-power edge fall detection)
- 路线 6 音频: 2604.10815 MeloTune (on-device music)
- 开源 11 repo 几乎全部围绕"在 230M ~ 8B LFM2.5 上做部署优化"
- HF 17 模型里 **LFM2-2.6B-Longevity (3645 下载)**, **LFM2.5-1.2B-Instruct (365942 下载)**, **LFM2.5-2.6B (120565 下载)** 才是真正落地的部分。

**含义**: arXiv 论文 "理论上" + 开源/HF 生态 "实际上" 共同推进 LNN 从科研原型走向产品化。

### 主题 C · "Liquid + X" 混合模式

- Liquid + MoE (2606.12240)
- Liquid + Memory (2607.08283)
- Liquid + Symbolic/Vector (2604.03955)
- Liquid + Liquid Crystal (2604.07219)
- Liquid + Mixture Density (2603.27058)

**含义**: 纯 LNN 已经很难独立 SOTA, **LNN 作为归纳偏置模块嵌入更大模型** 才是当前主流。这一点对本仓 `liquid_*` 路径特别重要 — `BlendGatedCfC` (本仓 r293)、`PLAN-CfC` (r302-r304) 已经在走 "CfC + X" 混合。

### 主题 D · "LNN 在视觉任务上的位置感"

LNN 没在 ImageNet / 检测 / 分割 backbone 上击败 CNN/Transformer, 它赢的是**时间连续性 + 可解释 + 小数据**:
- 路线 4 全是子任务, 不是 backbone
- 路线 5 决策/规划: 用作时序记忆, 不是感知主干

**含义**: 在"端侧 + 小数据 + 时序" 这个三角区里, LNN 是强候选; 在"大模型 + 感知主干" 这个象限里, LNN 还不是主角。

### 主题 E · "闭式 vs 求解器" 的工程张力

- 路线 1 强调 "无 solver" (2606.07670)
- 路线 2 用 random feature 闭式 (2606.15571)
- 路线 3 部分仍依赖 ODE solver (拓扑神经动力学 2606.21295, Graph LTC 2606.15807)

**含义**: 工业界推闭式 (推理快), 学界继续探索 ODE 解 (理论深)。本仓 `liquid_*` 路径当前以**闭式为主** (CfC), 与路线 1 / 路线 2 一致。

### 主题 F · "LFM2 / LFM2.5 = 唯一真正落地的 Liquid"

看 HF 下载量:
- LFM2.5-1.2B-Instruct (365942 下载, 668 likes)
- LFM2.5-VL-3B (27071 下载, 202 likes)
- LFM2.5-VL-3B-GGUF (46465 下载)
- LFM2-2.6B-Longevity (3645 下载)

**含义**: 学术圈继续在 CfC / LTC / NCP 上做探索, 但**产品圈的实际产品是 LFM2 / LFM2.5** (LiquidAI 推出)。两者并行推进, 不冲突: 学术给基础理论, LFM 给产品 SKU。

## 三、对本仓 `liquid_*` 路径的映射

### 已有工作对应表

| 本仓工作 | 对应路线 | 对应论文 |
|---|---|---|
| r287 Binary Gated Pulse / r290-r295 decorrelation default | 路线 8 振动动力学 | 2603.00153 Pulse-Driven |
| r293 BlendGated (honest negative) | 主题 C Liquid + X | (内部混合) |
| r302-r304 PLAN-CfC integration | 路线 5 决策/规划 | 2608.03041 PLAN FJSP |
| r305 MidpointCfC predictor-corrector | 路线 2 ODE 衍生 | (内部 ODE 解) |
| r294-r295 small-λ in-cell decorrelation | 主题 A σ(gate) 元模式 | 2606.07670, 2606.15807 |

### 自然下一步 (gap)

1. **Noise-injection bench (对应 9/7 CfC→SDE)**: 本仓没有 `bench_*noise*.py`, 是空白。可以加 `bench_cfc_sde_noise_injection.py`, 在 CfC cell 时间门上加 Gaussian 噪声, 验证 σ→0 退化的稳定性。这与 r290-r295 decorrelation 工作同属"在时间门上做正则化" 主题, 但走的是不同理论依据 (SDE 极限 vs Barlow-Twins decorrelation)。

2. **Graph LTC + cross-domain (对应 2606.15807)**: 本仓 irregular TS 已有 bench_irregular_dt.py, 但没有 graph LTC + 跨域迁移。可以加 `bench_graph_ltc_cross_domain.py`, 在合成的多域时序上验证跨域迁移能力。

3. **Liquid + Symbolic 混合 (对应 2604.03955)**: 本仓 Liquid + X 已有 BlendGatedCfC, 但 Liquid + Symbolic attention 是新维度, 可探索。

### 不属于本仓主线但值得跟踪

- 路线 6 音频 (2604.10815 MeloTune, 2606.19579 FlowFake) — 离本仓 `liquid_*` 路径较远, 但与端侧部署主题相关
- 路线 7 安全 (2604.02149 AEGIS) — 不是真 LNN, 是 SSM, 只保持观察
- 路线 9 LNN vs LSTM 对比 (2605.27467) — 综述性, 不驱动本仓新工作

## 四、Summary: 一句话总结 9 月路线图

**理论路线 (1+2+3+8)**: 从"ODE 闭式" → "SDE 推广" → "图 LTC / 拓扑神经动力学" 三层递进, σ(gate) 平滑插值是贯穿主线。
**应用路线 (4+5+6)**: LNN 作为"时间归纳偏置模块"嵌入视觉/决策/音频子任务, 而不是 backbone。
**生态路线 (主题 B+F)**: LFM2/LFM2.5 才是真正落地, 学术探索 + 产品化并行, LiquidAI 是双轨桥梁。
**本仓路线**: 在路线 1 (CfC 闭式场) + 路线 2 (ODE 衍生) 上已有积累 (r287-r305), 自然下一步是 noise-injection bench 与 graph LTC + cross-domain bench, 进一步闭合 "σ(gate) 元模式" 的实验覆盖。