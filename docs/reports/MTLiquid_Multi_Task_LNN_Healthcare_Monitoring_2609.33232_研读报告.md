---
title: MTLiquid — Multi-Task Liquid Neural Networks for Healthcare Monitoring — 研读报告
paper: https://arxiv.org/abs/2609.33232v1
arxiv_id: 2609.33232
date: 2026-09-30
tags: [LNN, CfC, multi-task-learning, healthcare, irregular-time-series, mortality-prediction, sepsis-detection, edge-deployment]
status: deep-read
report-author: LNN-research-agents
source: docs/daily/2026-09-30_LNN_research_digest.md arXiv 候选 (2609.33232, 9/27 提交, eBRAIN Lab NYU Abu Dhabi)
---

# MTLiquid — 研读报告

> 论文: [MTLiquid: Enabling Efficient Multi-Task Learning using Liquid Neural Networks for Lightweight Healthcare Monitoring Systems](https://arxiv.org/abs/2609.33232v1)
> 提交: 2026-09-27 (v1, 8 页 / 1 图 / 2 表) | 作者: Rachmad Vidya Wicaksana Putra, Fahad Abdul Rauf, Muhammad Shafique (eBRAIN Lab, NYU Abu Dhabi)
> 报告日期: 2026-09-30 | 工具: `skills/paper-analyzer` (arXiv 模式, 直接读 HTML 全文 + abstract)
> 强关键词命中: **liquid neural networks, closed-form continuous-time (CfC), continuous-time processing, multi-task learning, irregular time series, mortality prediction, sepsis detection** (score = 8, 是 9/30 digest 中唯一 score > 0 的 LNN 强关键词候选)

---

## 1. 一句话定位

> **MTLiquid = 共享 CfC backbone + per-task input projection + per-task output head + uncertainty-weighted loss + capped-cycling data scheduler**, 是首个把 CfC 应用到 ICU 多任务 (mortality + sepsis) 联合监测的工作. 用 **0.40 MB / 105K 参数的单模型同时预测 P12 (mortality) AUROC 0.8418 与 P19 (sepsis) AUROC 0.9379**, 与两个独立训练的 CfC 单任务模型 (合计 188K 参数 / 0.71 MB) 性能持平, 同时相对独立单任务 LSTM (合计 6.83 MB) 缩小 **94.1%** 模型大小, 训练功耗比 LSTM/RNN backbone 低 **43.1% / 36.9%**, 直接面向可穿戴 / 边缘 ICU 监测场景.

---

## 2. 元数据

| 字段 | 值 |
|---|---|
| arXiv id | 2609.33232v1 |
| 标题 | MTLiquid: Enabling Efficient Multi-Task Learning using Liquid Neural Networks for Lightweight Healthcare Monitoring Systems |
| 作者 | Rachmad Vidya Wicaksana Putra, Fahad Abdul Rauf, Muhammad Shafique |
| 单位 | eBRAIN Lab, New York University (NYU) Abu Dhabi |
| 类别 (cs.LG) | cs.LG (primary), cs.NE (cross-list) |
| 提交日期 | 2026-09-27 05:21 UTC |
| 页数 / 图表 | 8 pages, 1 figure, 2 tables |
| License | arXiv non-exclusive license |
| DOI | https://doi.org/10.48550/arXiv.2609.33232 |
| 资助 | NYUAD Center for CyberSecurity (CCS) — Tamkeen under NYUAD Research Institute Award G1104 |
| 代码 | 未公开 (论文未给出仓库链接; 仅描述 PyTorch 实现 + 单卡 RTX 4090 Ti) |

---

## 3. 核心问题

### 3.1 问题陈述

ICU 病人监护产生两类典型多任务需求:
1. **in-hospital mortality prediction (P12)** — Silva et al. 2012 PhysioNet Challenge, 48h 内 37 个不规则采样生理信号 → 预测院内死亡
2. **sepsis early detection (P19)** — Reyna et al. 2020 PhysioNet Challenge, 40 个通道小时级信号 → 预测 sepsis onset

两者都需要在 **边缘可穿戴设备 / 床旁低功耗设备** 上连续运行, 受严格 memory / power budget 约束. 当前 SOTA 是每个任务训练一个独立模型, 部署时需多个模型共存, 资源消耗线性增长.

### 3.2 现有方案的 3 类局限 (论文 §1)

1. **离散时间 RNN/LSTM**: 即使加 elapsed-time 输入 (Che et al. 2018 GRU-D, Cao et al. 2018 BRITS, Shukla & Marlin 2021), 离散 step 让 cell 不感知真实物理时间间隔, 处理 P12 / P19 这类 channel-dependent missing 数据时性能次优.
2. **LNN / LTC**: 连续时间 ODE 神经元物理合理, 但每步需要 ODE solver → 训练 / 推理 latency 显著高.
3. **CfC (Liquid)**: Hasani et al. 2022 给出 ODE 闭式近似, 速度提升 ~10×, 但现有工作**只关注单任务**, LNN 用于 multi-task learning 的可行性从未被研究过.

### 3.3 关键挑战 (论文 §1.2 列出)

1. 网络结构必须对 ICU 连续时间任务高效 (mortality + sepsis);
2. 多任务机制不能比单任务有显著精度损失;
3. 数据集大小极不均衡 (P12 vs P19) 时需要补偿机制.

### 3.4 MTLiquid 的对应答案

1. **架构**: 共享 CfC backbone + per-task input projection (降到 d=64 公共嵌入) + per-task linear head; backbone 只有 105K 参数 / 0.40 MB;
2. **训练策略**: Kendall et al. 2018 task-uncertainty weighting (learned σ_k per task) + capped-cycling (R=3) data loader policy.

---

## 4. 方法论与核心思路

### 4.1 整体架构 (论文 Figure 1, Algorithm 1+2)

```
task-k input  (X^(k), M^(k), Δt^(k))   k ∈ {P12, P19}
        │
        ▼
[per-task GRU-D forward-fill imputation]   Eq. (3)
        │  (^x_{t,d} = m·x + (1-m)·^x_{t-1})
        ▼
[per-task input projection]   W_k ∈ R^{64 × D_k^in}    Eq. (无编号)
        │  projected sequence z ∈ R^{64}
        ▼
[SHARED CfC backbone (f_shared)]    Eqs. (4)(5)(6) + Algorithm 1
        │  2-layer SiLU MLP → 2 candidate branches (ff_1, ff_2, tanh)
        │  + time-gate branches (t_a, t_b, linear) → sigmoid σ
        │  h_t = (1-σ_t)·ff_1 + σ_t·ff_2
        ▼
[per-task linear head]   ^y^(k) = W_{head,k} h_T + b_{head,k}
```

### 4.2 输入处理: Forward-fill imputation (Eq. 3)

与 GRU-D 一致, 缺失值由"最近一次观测" + 自身 mask bit 拼接后输入:

$$
\hat{x}_{t,d} = m_{t,d}\,x_{t,d} + (1-m_{t,d})\,\hat{x}_{t-1,d}
$$

P12 (41 通道) 输入维度 82, P19 (60 通道) 输入维度 120.

### 4.3 CfC 闭式状态更新 (Eq. 1 → 2)

LTC 神经元 ODE (Eq. 1):

$$
\frac{dx(t)}{dt} = -\Big[\tfrac{1}{\tau} + f(x,I;\theta)\Big] x(t) + f(x,I;\theta)\,A
$$

分段常数输入假设下的闭式近似 (Eq. 2):

$$
h(t) = \sigma(-f(x,I;\theta_f)\,t)\odot g(x,I;\theta_g) + \big[1-\sigma(-f(x,I;\theta_f)\,t)\big]\odot h(x,I;\theta_h)
$$

其中 $\sigma(\cdot)$ 是 sigmoid, $\odot$ Hadamard 积, $t$ 是 elapsed time (作为输入而非 fixed step). 单次前向即可计算, 比 ODE-solver 加速约 10×.

### 4.4 MTLiquid 的具体 CfC 实例化 (Eq. 4-6)

论文把 $g(\cdot), h(\cdot)$ 显式拆成两条 tanh 候选分支 + 一条 sigmoid 时间门控分支:

$$
g_t = \mathrm{ff}_1(x_t^{(k)}, h_{t-1}), \quad h_t^{\text{cand}} = \mathrm{ff}_2(x_t^{(k)}, h_{t-1}) \quad \text{(Eq. 4)}
$$

$$
t_{\text{interp}} = \sigma\!\Big(-\underbrace{t_a(x_t^{(k)}, h_{t-1})}_{f(\cdot;\theta_f)} \cdot \Delta t_t^{(k)} + t_b(x_t^{(k)}, h_{t-1})\Big) \quad \text{(Eq. 5)}
$$

$$
h_t = g_t \odot t_{\text{interp}} + h_t^{\text{cand}} \odot (1 - t_{\text{interp}}) \quad \text{(Eq. 6)}
$$

Algorithm 1 把这条管线写成 11 行伪代码: 2-layer SiLU backbone → ff_1, ff_2 (tanh) + t_a, t_b (linear) → sigmoid 时间门 → 闭式插值.

> **关键点**: 所有 4 个分支 (ff_1, ff_2, t_a, t_b) 在 P12 与 P19 之间**完全共享** — 这是"single continuous-time state-transition function 跨任务复用"的实质证据, 直接验证"liquidity 跨采样分布鲁棒".

### 4.5 训练策略 (Algorithm 2)

两个独立创新:

**Loss weighting (Eq. 7)** — Kendall, Gal, Cipolla 2018 task uncertainty:

$$
\mathcal{L} = \sum_{k=1}^{K} \frac{\mathcal{L}_k}{2\sigma_k^2} + \log \sigma_k
$$

每个任务一个 learned log-variance 参数, 与网络权重联合优化. $\log\sigma_k$ 正则化项阻止任务被无限 down-weight. $\mathcal{L}_k$ 本身是 class-weighted CE, 权重为该任务正例频率倒数, 处理标签不平衡.

**Capped-cycling data loader policy** — 每 batch 同时从 $D_{P12}$ 与 $D_{P19}$ 取一对, 短侧 loader 循环复用但封顶 R=3 次, 长侧每个 epoch length-match + 每 pass reshuffle. 既避免 starvation, 又防止短任务过拟合.

### 4.6 与 baseline 的对照设计

- 单任务 ablation: 同一网络骨架, 分别在 P12 / P19 上单独训;
- backbone ablation: 在相同 scaffold (per-task projection + shared recurrent cell + per-task heads) 下, 把 CfC 替换为 LSTM / RNN;
- 共同超参: hidden=256, d=64, batch=128, 57 epochs, Adam, exponential LR decay (γ=0.9), 3 seeds (42, 123, 777), AUROC + AUPRC 主指标, 同时报告 params / MB / multi-task 训练功耗.

---

## 5. 核心公式 (LaTeX)

> 论文全部 7 个核心公式, 这里给出与原始定义等价的 LaTeX.

**(1) LTC neuron ODE** — 连续时间液态神经元动力学的原始 ODE:

$$
\frac{dx(t)}{dt} = -\Big[\frac{1}{\tau} + f(x(t), I(t); \theta)\Big] x(t) + f(x(t), I(t); \theta)\,A
$$

**(2) CfC closed-form hidden update** — ODE 的闭式近似 (核心 liquid cell):

$$
h(t) = \sigma(-f(x, I; \theta_f)\, t) \odot g(x, I; \theta_g) + \big[1 - \sigma(-f(x, I; \theta_f)\, t)\big] \odot h(x, I; \theta_h)
$$

**(3) Forward-fill imputation (GRU-D style)** — 处理 irregular missing data:

$$
\hat{x}_{t,d} = m_{t,d}\, x_{t,d} + (1 - m_{t,d})\, \hat{x}_{t-1, d}, \quad m_{t,d} \in \{0, 1\}
$$

**(4) Two candidate branches (paper Eq. 4)**:

$$
g_t = \mathrm{ff}_1(x_t^{(k)}, h_{t-1}), \qquad h_t^{\text{cand}} = \mathrm{ff}_2(x_t^{(k)}, h_{t-1})
$$

**(5) Time-interpolation gate (paper Eq. 5)** — MTLiquid 显式化 liquid gate:

$$
t_{\text{interp}} = \sigma\!\Big(-\underbrace{t_a(x_t^{(k)}, h_{t-1})}_{f(\cdot;\theta_f)} \cdot \Delta t_t^{(k)} + t_b(x_t^{(k)}, h_{t-1})\Big)
$$

**(6) CfC state interpolation (paper Eq. 6)**:

$$
h_t = g_t \odot t_{\text{interp}} + h_t^{\text{cand}} \odot (1 - t_{\text{interp}})
$$

**(7) Task-uncertainty multi-task loss (paper Eq. 7)** — Kendall, Gal, Cipolla 2018:

$$
\mathcal{L} = \sum_{k=1}^{K} \frac{\mathcal{L}_k}{2\sigma_k^2} + \log \sigma_k
$$

其中每个 $\mathcal{L}_k$ 是 class-weighted cross-entropy (权重 = 正例频率倒数).

---

## 6. 关键成果与贡献

### 6.1 主表 (Table 1) — 单任务 vs 多任务性能

| Setting | Model | Task | Params | Size (MB) | AUROC | AUPRC |
|---|---|---|---:|---:|---:|---:|
| **Single-task (one model per task)** | CfC | P12 | **92,930** | **0.35** | 0.8409 ± 0.0032 | 0.5238 ± 0.0108 |
| | CfC | P19 | **95,362** | **0.36** | **0.9472 ± 0.0083** | **0.7532 ± 0.0039** |
| | LSTM | P12 | 875,010 | 3.34 | 0.8234 ± 0.0163 | 0.4624 ± 0.0299 |
| | LSTM | P19 | 913,922 | 3.49 | 0.9226 ± 0.0008 | 0.7460 ± 0.0083 |
| | RNN | P12 | 219,138 | 0.84 | 0.7882 ± 0.0080 | 0.4608 ± 0.0070 |
| | RNN | P19 | 228,866 | 0.87 | 0.9381 ± 0.0075 | 0.7358 ± 0.0386 |
| **Multi-task (one model for both)** | **MTLiquid** | **P12** | **105,350** | **0.40** | **0.8418 ± 0.0060** | **0.5132 ± 0.0043** |
| | **MTLiquid** | **P19** | " | " | **0.9379 ± 0.0024** | **0.7662 ± 0.0079** |
| | LSTM (multi) | P12 | 870,150 | 3.32 | 0.8335 ± 0.0070 | 0.4986 ± 0.0098 |
| | LSTM (multi) | P19 | " | " | 0.9173 ± 0.0028 | 0.7602 ± 0.0112 |
| | RNN (multi) | P12 | 228,102 | 0.87 | 0.8389 ± 0.0050 | 0.5004 ± 0.0045 |
| | RNN (multi) | P19 | " | " | 0.9317 ± 0.0015 | 0.7371 ± 0.0091 |

**核心数字**:

- MTLiquid 在 P12 / P19 上分别比单任务 CfC baseline **高 0.09 pp / 低 0.93 pp AUROC** (本质持平);
- P19 AUPRC 反超单任务 +1.3 pp (0.7662 vs 0.7532);
- 参数数 105K vs 两个独立 CfC 合计 188K → **44.1% 减少**; vs 两个独立 LSTM 合计 1.79M → **94.1% 减少**; vs 两个独立 RNN 合计 448K → **76.5% 减少**.

### 6.2 Ablation (Table 2) — 多任务训练组件的边际贡献

| Configuration | P12 AUROC | P12 AUPRC | P19 AUROC | P19 AUPRC |
|---|---:|---:|---:|---:|
| Baseline (no loss weighting, zero-fill, default cycle) | 0.7175 | 0.2974 | 0.9140 | 0.7193 |
| + Loss weighting (Kendall et al.) | 0.7328 | 0.3162 | 0.9322 | 0.7281 |
| + GRU-D forward-fill imputation | 0.8010 | 0.4324 | 0.9348 | 0.7547 |
| Loader ablation: `--truncate_to_shorter` | 0.8415 | 0.5027 | **0.8799** | 0.6987 |
| **Capped cycling, R=3 (our MTLiquid)** | **0.8418** | **0.5132** | **0.9379** | **0.7662** |

**最有意思的结论**:
- forward-fill 是 P12 的最大单一增益 (+6.8 pp AUROC, +11.6 pp AUPRC), 因为 P12 的 missingness 更严重;
- naive `--truncate_to_shorter` 在 P12 上看着还行 (0.8415) 但把 P19 打塌 (-5.8 pp AUROC), capped-cycling 把 P19 救回 (0.8799 → 0.9379);
- **数据调度 + imputation 联合决定**多任务成败, 不是单一 tricks.

### 6.3 功耗

- 单任务 CfC: 与 LSTM 相当 (P12 ±0.3%, P19 ±2.9%), 比 RNN 低 (P12 -26.1%, P19 -18.3%);
- 多任务 MTLiquid: 比 LSTM backbone **低 43.1%**, 比 RNN backbone **低 36.9%**, 直接受益于"单次 sigmoid gate vs LSTM 4-gate per step" 的算力差.

### 6.4 贡献清单

1. **方法贡献** — 首个把 CfC 应用到 ICU 多任务连续时间监测的工作; 给出 end-to-end 架构 (per-task projection + shared CfC + per-task head) + 训练策略 (uncertainty weighting + capped cycling) 的完整 recipe;
2. **实验贡献** — 在 PhysioNet P12 (2012) + P19 (2019) 两个标准 benchmark 上验证, 105K 参数 / 0.40 MB 单模型同时接近两套独立单任务 baseline;
3. **理论贡献** — 实证证明"单一连续时间状态转移函数可同时服务异质采样分布的多任务", 把 LNN 从单任务推向多任务的可行路径;
4. **部署贡献** — 量化展示 -94.1% 模型 size + -43% 训练功耗, 为可穿戴 / 床旁低功耗 ICU 监测硬件提供具体数字.

---

## 7. 局限性与未来展望

### 7.1 作者承认的局限 (论文未单列 Limitations 节, 但分散在文中)

1. **数据基准受限**: 仅在 PhysioNet P12 + P19 两个 ICU benchmark 验证, 未涉及急诊、围术期、心脏 ICU 等其他监护场景;
2. **任务数受限**: 实验只演示 K=2 多任务 (P12 + P19), K>2 (例如加 decompensation, length-of-stay, AKI) 时 shared CfC 的容量与 gating 是否还鲁棒未知;
3. **数据集大小差异**: P12 与 P19 大小不同通过 capped cycling 处理, 但未给出对"极端不均衡" (如 1:100) 的鲁棒性数据;
4. **超参 R=3 是 personal-tunable 的**: capped-cycling 的 R 上限与 max-repeat factor 是经验值, 没有做 sweep;
5. **与更现代 MTL 平衡策略未对比**: 仅用 Kendall uncertainty weighting, 没对比 GradNorm (Chen et al. 2018b), PCGrad (Yu et al. 2020), 或 Pareto MTL (Sener & Koltun 2018) — 论文 §3.3 提到这些但没 ablation;
6. **未给真实部署数据**: 论文报 training 功耗, 未给 inference latency / 边缘硬件 (MCU / Jetson) 实测;
7. **未公开代码与数据预处理流水线**: paper-only, 复现需自行实现 GRU-D forward-fill + Algorithm 1+2.

### 7.2 可延展方向

- **任务数扩展**: 加 decompensation, AKI, LoS 等 ICU 任务, 验证 shared CfC 在 K≥5 时的 scaling;
- **MTL 平衡策略 ablation**: uncertainty weighting vs GradNorm vs PCGrad vs Pareto 在 shared CfC 上 head-to-head;
- **边缘部署验证**: 0.40 MB 模型 → ONNX / TF-Lite / CoreML, 跑 Cortex-M / Jetson Orin Nano 实测 inference latency / energy;
- **隐私 / 联邦学习**: ICU 数据隐私敏感, shared CfC 的小 footprint 可能天然适合 on-device personalization + federated fine-tune;
- **与 SOTA 强 baseline 对齐**: 与 RETAIN (Choi et al. 2016), MIMIC-IV 上的 Transformer-based MT-ICU (e.g., MET 2024) head-to-head;
- **可解释性**: P12/P19 上的 shared hidden state 能否可视化出"危险信号模式" (类似 Liquid Foundation Model 早期 paper 的 interpretability claim).

---

## 8. 与本项目 (LNN 仓库) 的关系

### 8.1 仓库内既有相关研读 / 复现

| 关联项 | 路径 / 内容 |
|---|---|
| **Liquid Foundation Model 系列** | `LFM2-2.6B` 等 LFM2.5 部署研读 (9/24-9/29 digest) — 与 MTLiquid 共享"liquid cell for edge"哲学但走 LLM 路线 |
| **CfC / LTC 既有研读** | `FlowFake_LTC_2606.19579_研读报告.md`, `AerACM_CfC_..._研读报告.md`, `AEGIS_TVD-HL-SSM_2604.02149_研读报告.md` 等 — 都是单任务 CfC, MTLiquid 是首个系统化多任务版本 |
| **多任务既有 work** | `AwareLiquid_M1_MT-LNN_研读报告.md` (2026-09-14) — 走 attention-free LLM 路线, 与 MTLiquid 同样强调 shared backbone, 但目标模态不同 |
| **复现脚本候选** | `scripts/bench_cfc.py`, `scripts/bench_learned_beta_cfc.py`, `scripts/experiment_timeseries.py` 等 — 已有 CfC 训练骨架, 改造为 multi-task variant 难度低 |
| **本项目 CfC 调参栈** | r287-r307 (MDN-CfC head, CmAPSS, irregular-time) 已有大量 ablation, 可直接借鉴作为 MTLiquid 的起点 |

### 8.2 借鉴与可落地项

1. **架构借鉴**: per-task projection → shared CfC → per-task head 是非常干净的 multi-task 模板, 可直接套用到本仓的 irregular time-series 任务 (CmAPSS 退化 + Henry Hub 预测 + 强化学习 policy 多任务);
2. **训练策略借鉴**: Kendall uncertainty weighting + capped-cycling 是简单可加的 module, 加进 `replicate_paper_experiment.py` 的 reward 多任务框架;
3. **部署可行性**: 0.40 MB / 105K 参数 / 43% 训练功耗的节省, 与本仓 edge deployment / Jetson / iOS export pipeline (`export_lnn_for_ios.py`, `export_lnn_tensorrt.py`, `jetson_lnn_benchmark.py`) 直接对接, 可作为"医院 IoT 边缘节点" 落地 demo;
4. **复现优先级**: 中. 论文未公开代码, 但 algorithm 1+2 + 4 个公式 + 2 张表足以独立实现; 数据 (PhysioNet P12 + P19) 公开, 实现难度 ~5-10 人天, 与本仓既有 CfC bench 栈可复用.

### 8.3 一句话定位 (与 §1 互补)

MTLiquid 是 **"LNN 单任务 → 多任务"** 的关键桥梁工作: 它没有引入新 cell, 而是证明了 **既有 CfC 闭式 cell 已经天然具备跨任务复用能力**, 关键工程技巧是输入边界 / 输出边界的 task-specific 化 + 数据调度平衡. 这对本仓而言意味着: 任何想把 CfC 从单任务扩展到多任务的尝试, **不再需要解决"连续时间多任务"开放问题**, 只需按本论文的 scaffold 实施即可. 这对资源受限的 ICU / 工业监测 / 边缘 AI 场景有直接借鉴价值.

---

## 9. 引用与外部链接

- arXiv abstract: https://arxiv.org/abs/2609.33232
- arXiv HTML (experimental): https://arxiv.org/html/2609.33232v1
- arXiv PDF: https://arxiv.org/pdf/2609.33232
- DOI: https://doi.org/10.48550/arXiv.2609.33232
- PhysioNet P12 (Silva et al. 2012 Challenge): https://physionet.org/content/challenge-2012/
- PhysioNet P19 (Reyna et al. 2020 Challenge): https://physionet.org/content/challenge-2019/
- CfC 原论文 (Hasani et al. 2022, Nature Machine Intelligence): Liquid Time-Constant Networks
- LTC 原论文 (Hasani et al. 2021): https://arxiv.org/abs/2006.04439
- Kendall, Gal, Cipolla 2018 (Multi-Task Learning Using Uncertainty to Weigh Losses): https://arxiv.org/abs/1705.07115
- GRU-D (Che et al. 2018): https://www.nature.com/articles/s41598-018-24271-9

---

## 10. 复现检查清单 (供后续 reproduce 阶段)

- [ ] 数据: PhysioNet P12 (challenge-2012) + P19 (challenge-2019) 公开下载
- [ ] 预处理: GRU-D forward-fill (Eq. 3) + mask concat, 维度 82 (P12) / 120 (P19)
- [ ] 网络: per-task Linear projection (D_k^in → 64) + shared 2-layer SiLU backbone (256) + ff_1, ff_2 (tanh, 256→64) + t_a, t_b (linear, 256→1) → sigmoid gate → per-task Linear head (64→1)
- [ ] 训练: Adam, γ=0.9 LR decay, 57 epochs, batch=128, 3 seeds (42, 123, 777), capped-cycling R=3
- [ ] 损失: class-weighted CE per task + Kendall uncertainty weighting (Eq. 7)
- [ ] 指标: AUROC + AUPRC on P12 / P19, 同时报 params / MB / 训练功耗
- [ ] 期待数字: P12 AUROC ≈ 0.84, P19 AUROC ≈ 0.94, params ≈ 105K, size ≈ 0.40 MB
