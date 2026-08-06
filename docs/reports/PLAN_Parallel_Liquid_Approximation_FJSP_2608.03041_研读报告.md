---
title: PLAN 论文研读报告 — Parallel Liquid-Inspired Approximation Network for FJSP
arxiv: 2608.03041v1
date: 2026-08-06
tags: [LNN, FJSP, edge-inference, parallelization, PPO, DRL, paper-analyzer]
status: deep-read
---

# PLAN: Parallel Liquid-Inspired Approximation Network for FJSP — 研读报告

> 论文 arXiv ID: [2608.03041v1](https://arxiv.org/abs/2608.03041v1) | PDF: <https://arxiv.org/pdf/2608.03041v1>
> 来源: docs/daily/2026-08-06_LNN_research_digest.md arXiv 候选 (1/25, 全新未消化)
> 分析日期: 2026-08-06 | 工具: `skills/paper-analyzer`

---

## 📄 Title & Authors

- **完整标题**: *PLAN: Parallel Liquid-Inspired Approximation Network for Efficient Representation Learning in Flexible Job Shop Scheduling*
- **作者**: Dhivya Dharshini Kannan, Wei Zhang, Jieyi Bi, Yingpeng Du, Tianjun Wei, Jie Zhang, Zuming Liu, Anupam Trivedi
- **发表**: arXiv 预印本 2608.03041v1 (2026-08-04 提交)
- **第一作者与本仓库关联**: **Dhivya Dharshini Kannan 同时是 DLNet (arxiv 2601.06227, ICPR 2026) 的第一作者** — 同一作者在 LNN 边缘压缩方向有连续工作线 (Arduino Nano 33 BLE Sense int8 部署 → 现在推到 FJSP 调度推理加速)。

## 🎯 Core Problem

灵活作业车间调度 (FJSP) 是经典 NP-hard 组合优化。DRL 解法用 attention-centric backbone (transformer / heterogeneous graph transformer) 在中小规模上 SoTA,但:

1. **参数爆炸**: 注意力权重随状态空间 O(N²) 增长,大规模 FJSP (200×5+) 内存与延迟都不可接受
2. **推理延迟**: 顺序 attention 在生产排程实时决策中成为瓶颈
3. **LNN 的悖论**: 液态神经网络参数高效、状态演化自适应,但**内在顺序动力学**使其无法利用 GPU/TPU 并行性 → 训练慢、推理虽小但 wall-clock 仍不及小 transformer

PLAN 目标: 保留 LNN 的参数高效 + 状态自适应,同时把顺序 ODE 动力学**重写为可并行张量操作**,实现"又小又快又好"。

## 💡 Methodology

### 1. MDP 形式化
- 状态: 实体特征 (作业 / 机器 / 工序)
- 转移: 动作执行后更新
- 奖励: makespan 改善
- 风险敏感 (stochastic FJSP): Value-at-Risk 目标

### 2. 液态启发的并行状态动力学 (核心创新)

经典连续 LNN:

$$\frac{dh_t}{dt} = -\frac{h_t}{\tau} + \sigma(W_h h_t + W_x x_t)$$

Euler 离散化(传统 sequential, 步间依赖):

$$h_{t+1} = h_t + \Delta t\left(-\frac{h_t}{\tau} + \sigma(W_h h_t + W_x x_t)\right)$$

**PLAN 关键重写**(去掉对前一时刻隐藏状态的依赖,转成可批量并行):

$$H = \hat{H} + \Delta t \cdot \frac{\tanh(W\hat{H} + U_X) - \hat{H}}{\tau}$$

其中 $\hat{H}$ 是**初始上下文隐藏状态**(由 context aggregator 单次计算),$H$ 是整序列并行输出的更新后状态。
→ 单个 matmul + 单个 tanh 完成整 batch 时序演化,无顺序瓶颈。

### 3. Context-Aware Learning
- 特征投影 → 隐空间 → 轻量多头注意力聚合 → 融合 → 初始化 $\hat{H}$
- "上下文聚合"和"状态演化"**结构解耦**: 上下文一次算清,状态一次并行更新,避免每步重算

### 4. Parallel Liquid Approximation
- 取消 sequential recurrence, single-step liquid correction 以 batched ops 执行
- "Approximation" 一词: 牺牲了严格 ODE 演化(用单步 Euler 修正逼近),换取完全并行

### 5. SPM-PLAN (stochastic FJSP 变体)
- 紧凑 stochastic processing module
- 用 inducing vector cross-attention (类似 SPT/SVGP 的诱导点思想) 建模不确定加工时间
- 避免完整 scenario attention 的 O(N²) 代价

### 6. 训练: PPO
- Clipped surrogate objective
- Actor / Critic 双网络
- GAE advantage estimate

## 📊 Key Results & Contributions

| 场景 | makespan 改善 | 推理延迟改善 | 参数占比 |
|---|---:|---:|---:|
| Deterministic FJSP (SD1/SD2) | -1.2% | -13.2% | 22-47% |
| Stochastic FJSP (SD3) | -1.4% | -31.7% | 22-47% |
| Multi-faceted dynamic FJSP | -2.3% | -26.9% | 22-47% |
| 最大单点 (200×5 largest) | -10.2% | **-69.2%** | 22-47% |

**核心贡献**:
1. **形式化 LNN sequential 动力学为并行张量操作** — 给"小 + 快 + 准"的 LNN 改进提供了通用 recipe (不只 FJSP 适用)
2. **plug-and-play backbone**: 替换 transformer / heterogeneous graph transformer,SPM-PLAN 处理随机性
3. **多场景验证**: deterministic / stochastic / multi-faceted dynamic,数据集 SD1/SD2/SD3,Brandimarte, Hurink,实例规模 10×5 到 200×5
4. **消融**: LNN-ODE / LNN-Euler / attention-only / 完整 PLAN — PLAN 两组件结合最优

## ⚠️ Limitations & Future Work

- 论文未显式列 limitations section(节选缺失),但根据方法与实验可推断:
  - 离散调度步长必须与 PLAN 的并行 liquid 更新对齐 — 高度规则化时间步的场景最优,真实工业中高度不规则事件流(机器随机故障 / 急件插单)还需适配
  - 单步 Euler 修正本质是**一阶近似**,对长时间窗口(>数百步)累积误差需验证
  - 多目标优化(能耗 / 完工时间 / 设备磨损 Pareto)未涉及
- 未来方向: 更复杂动态约束、多目标、跨域迁移(把 PLAN backbone 用到其它 LNN 任务如异常检测 / 设备预测性维护)

## 🔗 与本仓库 LNN 体系的关联 (2026-08-06 视角)

| 本仓库已有工作 | PLAN 借鉴点 / 增量 |
|---|---|
| `docs/reports/DLNet_Dual_Stage_Distillation_Pareto_LNN_2601.06227_研读报告.md` (N20 内部) | 同第一作者的 Arduino int8 边缘压缩线,PLAN 是其向"调度推理加速"方向的延伸 |
| `scripts/jetson_lnn_benchmark.py` | PLAN 的"22-47% baseline 参数 + 69.2% 延迟降低"模式可直接套到 Jetson Orin Nano int8 benchmark 设计:把 PARALLEL_LIQUID_APPROXIMATION 当作 backbone,量化后测 Orin Nano 吞吐 |
| `scripts/lfm25_benchmark.py` | LFM2.5-1.2B 的 GLU/RMSNorm block 同样存在 sequential 依赖,PLAN 的"结构解耦 + 并行近似"思想可启发 LFM2.5 边缘推理优化(改 sequential recurrence 为 batched approx) |
| `docs/reports/LRFM_N2_Frozen_LTC_Features_vs_Trained_CfC_2026-08-05.md` | PLAN 的"上下文 + 状态解耦"和 N2 闭包的"frozen feature + linear readout"思路同源,都是把"重"的部分冻结/并行,把"轻"的部分留给任务层 |

## 🧪 建议下一步 (Jetson Orin Nano 视角)

1. **复现 PLAN block**: 在 `models/` 新增 `parallel_liquid_approx.py`,实现 $H = \hat{H} + \Delta t \cdot \tanh(W\hat{H}+U_X - \hat{H}) / \tau$ 单元 + context aggregator
2. **Jetson benchmark**: 在 SD1 10×5 / 30×5 规模跑 PARALLEL_LIQUID_APPROXIMATION vs SEQUENTIAL_LNN vs TINY_TRANSFORMER,记录 latency / throughput / energy (Orin Nano 15W TDP)
3. **int8 量化**: 仿 DLNet 的 dual-stage distillation + Pareto 选择,在 Jetson Orin Nano TensorRT 上测 4/8/16-bit 不同精度吞吐
4. **跨任务迁移**: 试把 PLAN backbone 用到本仓库 `experiment_concept_drift.py` (概念漂移检测) — 时序不规则场景下并行近似的鲁棒性

## 📎 引用与下载

- arXiv: <https://arxiv.org/abs/2608.03041v1>
- PDF: <https://arxiv.org/pdf/2608.03041v1>
- 暂无官方代码仓(论文未列),可作为 re-implementation 候选
