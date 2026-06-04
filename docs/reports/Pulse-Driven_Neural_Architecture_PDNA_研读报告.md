---
title: Pulse-Driven Neural Architecture (PDNA) — 研读报告
paper: arXiv 2603.00153v1
author: Paras Sharma (single author)
date: 2026-02-25
tags: [LNN, CfC, pulse-module, oscillatory-dynamics, gap-robustness, smnist, ablations]
status: deep-read
report-date: 2026-06-04
report-author: LNN-research-agents
---

# Pulse-Driven Neural Architecture (PDNA) — 研读报告

> 论文: arXiv:2603.00153v1 [cs.NE] 25 Feb 2026, Paras Sharma
> 体量: ~13 页
> 链接: https://arxiv.org/abs/2603.00153v1
> 代码: https://github.com/Parassharmaa/pdna (**公开**)
> 与本仓直接相关度: **高** — **直接 build on CfC**,只加 2 个 additive residual 模块
> (pulse + self-attend),本仓 `lnn/core/cfc_cell.py` 完全可以接上。

---

## 1. 一句话定位

> 在 **CfC backbone** 之上加两个 gated residual 模块:(1) **pulse 模块**输出
> `α · A · sin(ωt + φ(h))` 的可学习振荡,(2) **self-attend 模块**输出
> `β · Wself · σ(h)` 的逐点 recurrent self-attention。在 **sequential MNIST**
> + **Gapped evaluation protocol**(测试时挖洞)上,5 seed ablation 证明:
> pulse variant 在 multi-gap 下 acc +4.62 pp(Cohen's d = 0.87),
> **结构性优势**而非"任意非零扰动就够"(noise control 输 baseline)。

论文解决的是"输入中断时连续时间网络的鲁棒性"——这点与生物神经系统的
"内部 clock 持续运行"对应,带一点 neuroscience 角度(Lisman 2005,
Buzsáki 2006)。

## 2. 关键公式(可复现核心,本仓直接可用)

### 2.1 CfC backbone(本仓已有)

```
h(t) = σ(-f(x,h;θf) · t) ⊙ g(x,h;θg) + (1 - σ(-f(x,h;θf) · t)) ⊙ h0     (2)
```

其中 `t` 是 CfC cell integration step 内的 elapsed time,不是 wall-clock。

### 2.2 Pulse 模块(本仓新增 ~15 行)

```
pulse(t, h) = A · sin(ω · t + φ(h))                                     (3)
h_pulse = h_cfc + α · pulse(t, h_cfc)                                   (4)
```

参数:
- `A ∈ R^d` — 逐维度可学习振幅
- `ω ∈ R^d` — 逐维度可学习频率,**log-uniform 初始化 [0.1, 10.0]** 鼓励频率多样性
- `φ(h) = W_φ h + b_φ` — state-dependent phase 线性投影
- `α` — 标量门,**初始化 0.01**(让模型先学 CfC baseline,再慢慢注入 pulse)

`t ∈ {0, 1, ..., T-1}` 是 per-sequence 线性时间步索引,**一个 integer per row**。

### 2.3 Self-Attend 模块(本仓新增 ~10 行)

```
self_attend(h) = Wself · σ(h)                                          (5)
h_out = h_pulse + β · self_attend(h_pulse)                             (6)
```

`β` 也是 0.01 初始化。**注意是 pointwise**(每个 timestep 独立),不是 sequence-wise
self-attention——所以 GPU 仍然 parallel。

### 2.4 完整 PDNA forward

```
h = CfC(x)                                        # backbone
if pulse:    h = h + α · A · sin(ω·t + φ(h))       # gated oscillation
if attend:   h = h + β · Wself · σ(h)             # gated self-proj
y = Classifier(h[:, -1, :])                       # last-step
```

## 3. Gapped Evaluation Protocol(论文的方法论贡献)

测试时把 input 序列**挖洞** —— 这是 PDNA 的"任务创新":

| Gap Level | Gap Size | 描述 |
|---|---:|---|
| Gap 0% | 0% | 标准评估(无洞) |
| Gap 5% | 5% | 中间连续挖掉 |
| Gap 15% | 15% | 中间连续挖掉 |
| Gap 30% | 30% | 中间连续挖掉(过深 → 接近 chance) |
| **Multi-gap** | 20% (scattered) | **4 个洞均匀分布**(更生态有效) |

**关键设计原则**:**训练时不用 gapped 数据**,只在测试时挖洞 → 这把"架构 robustness"
与"数据增强"严格隔离。

- Contiguous gap 居中放在 T/2 位置
- Multi-gap 4 个洞均匀分布
- 信息丢失因 digit class 而异(中间行 "1" 信息少,中间行 "8" 信息多)

## 4. 5 变体 ablation(论文 Table 1)

| Variant | Pulse | Self-Attend | 用途 |
|---|---|---|---|
| A | ✗ | ✗ | Baseline CfC(对照) |
| B | **random** | ✗ | **Noise control**(同 magnitude 随机扰动) |
| C | ✓ | ✗ | 单独的 oscillation |
| D | ✗ | ✓ | 单独的 self-attention |
| E | ✓ | ✓ | Full PDNA |

**B 是 critical control**:如果 noise control 与 pulse 持平或更好,说明"任何非零扰动
就够了";结果显示 noise **hurt** performance,所以 pulse 的优势是 **structural
而非 dynamic**。

## 5. 实验设置

- **任务**: Sequential MNIST(28 timesteps × 28 features)
- **Hidden size**: 128(所有变体一致)
- **优化器**: AdamW + cosine annealing + 3-epoch linear warmup
- **LR**: 5e-4
- **Batch size**: 512
- **Max epochs**: 40, early stopping patience 8 (val acc)
- **Gradient clipping**: max norm 1.0
- **Seeds**: 5 (42, 123, 456, 789, 1337)
- **Dropout**: 0.1 on classifier head
- **硬件**: 单卡 RTX A4000 16GB
- **总 runs**: 5 变体 × 5 seeds = 25 runs

## 6. 主要结果(论文 Table 3 + Table 4)

### 6.1 干净数据精度(无 gap)

| Variant | sMNIST (%) |
|---|---:|
| A. Baseline CfC | 97.82 ± 0.12 |
| B. CfC + Noise | 97.78 ± 0.20 |
| **C. CfC + Pulse** | **97.96 ± 0.14** |
| D. CfC + SelfAttend | 97.89 ± 0.21 |
| E. Full PDNA | 97.93 ± 0.16 |

**关键观察**:所有变体干净精度 ~98%(pulse variant 略高)。Augmentation **不干扰**
标准学习,只在 gap 条件下才显出价值。

### 6.2 Gap 鲁棒性(论文 Table 4)

| Variant | Gap 0% | Gap 5% | Gap 15% | Gap 30% | **Multi** |
|---|---:|---:|---:|---:|---:|
| A. Baseline CfC | 97.82 | 94.88 | 48.35 | 28.51 | 88.24 |
| B. CfC + Noise | 97.78 | 94.60 | 49.56 | 29.78 | 88.01 |
| **C. CfC + Pulse** | 97.96 | **95.82** | 48.27 | 29.58 | **92.86** |
| D. CfC + SelfAttend | 97.89 | 95.49 | 52.24 | 28.46 | 91.02 |
| E. Full PDNA | 97.93 | 95.28 | 49.43 | 29.71 | 91.96 |

**关键发现**:
- **Multi-gap**: pulse **92.86%** vs baseline **88.24%** = **+4.62 pp**(5/5 seeds)
- **Gap-5%**: pulse **+0.93 pp** over baseline (p=0.034), **+1.22 pp** over noise (p=0.013)
- **Gap-30%**: 全部 ~28-30% 接近 chance → "信息恢复容量上限"假说(本架构尺度内)
- **degradation 方差**: pulse-augmented 3.05-3.57% vs baseline 5.02% → 更稳定

### 6.3 统计检验

- pulse vs baseline multi-gap:**5/5 seeds 全胜**
- self-attend vs baseline multi-gap:**5/5 seeds 全胜**
- pulse vs noise multi-gap:**+4.85 pp (p=0.079, d=1.05)**
- pulse vs noise gap-5%:**+1.22 pp (p=0.013, 显著)**

## 7. Pulse 训练动力学(论文值得复现观察)

- **α 初始化** 0.01,**训练后增长到 ~0.66** — 模型主动利用 pulse 机制
- **ω 频谱** 学到范围 **[0.06, 10.02]**,跨两个数量级 — 类比神经科学的
  multi-frequency band(Buzsáki 2006 gamma/alpha/theta)
- **计算开销**: +38% parameters, **+5% wall-time** — 实战可部署

## 8. 局限与未来工作(论文自报)

1. **Task scope**: 只在 sMNIST 评估。**psMNIST(784 步)、sCIFAR-10(1024 步)、
   Long Range Arena** 是直接后续
2. **统计效能**: n=5 seeds,部分比较未到 p<0.05;但 Cohen's d > 1.0 + 5/5 win rate
   说明 effect 是真的
3. **Parallel processing**: CfC 是并行的(全序列同时算),所以 pulse 是 **post-hoc
   augmentation to full hidden-state tensor**——**不是真正的 continuous-time
   dynamic evolving between input steps**。**sequential ODE-based architecture**
   (sequential LTC)才能让 pulse 在 gap 期间真正演化状态——作者**明确承认**这预期
   会得到更强结果
4. **Degradation floor**: Gap 30% 全员 chance;**multi-gap 更具区分力**
5. **Non-additive composition**: Full PDNA (E) **不严格优于** single component
   (C, D);可能因为两个模块竞争同一 hidden state 维度,或 CfC parallel 限制
   它们的相互作用
6. **未来方向**: (a) **把 pulse 作为真 ODE term 集成进 sequential LTC**,
   (b) 长程任务,(c) 把 learned frequency spectrum 当 unsupervised 表达,
   (d) **gap training**(训练时也挖洞)

## 9. 与本仓的契合度

| 维度 | 评估 |
|---|---|
| **算法复用** | CfC backbone 完全复用;**pulse + self-attend 模块 ~25 行核心代码** |
| **数据可获得性** | **MNIST 是 default torchvision**;**无需下载** |
| **Jetson 部署** | 5 变体 × 5 seeds × 40 epoch sMNIST — **单 A4000 16GB 可跑**(论文) |
| **CPU 部署** | hidden=128 sMNIST 训练可走 CPU(慢一点但 5 seed × 5 变体能跑) |
| **代码可用性** | **公开** — github.com/Parassharmaa/pdna,**单命令复现** |
| **与本仓已有** | 互补 LiquidTAD(Long seq / video)与本项目(短 seq / gap robustness) |
| **统计严谨度** | 5 seed + paired t-test + Cohen's d + 5/5 win rate(虽然 n=5 偏小) |
| **可争议点** | pulse 是 post-hoc augmentation 真正 continuous-time dynamic 需 sequential LTC;non-additive composition 暗示 C 和 D 互不协同 |

### 9.1 本仓复现优先级

- **P1 (次低阻,本周可做)**: PDNA PulseHead 作为 `lnn/core/cfc_cell.py` 的
  `--pulse-head {none, sinusoidal, hierarchical_decay}` 选项,在 `lnn_experiments`
  的 sMNIST Gapped protocol 上跑 5 seed × 4 backbone ablation
- 关键 ablation: CfC baseline vs CfC+pulse vs CfC+self-attend vs Full PDNA,
  multi-gap accuracy + variance + Cohen's d
- 与本仓 `HierarchicalDecayLiquidTADHead` 形成 **连续时间架构的两个
  augmentation 系列**(Hierarchical decay-rate sharing vs learnable oscillation)

### 9.2 复现计划(stage 拆分,与本仓节奏一致)

**Stage A**(iter#19 候选): `lnn/core/cfc_cell.py` 加 `PDNAPulseHead`(~15 行)
+ `tests/test_pdna_pulse.py` 5 个 unit test(sinusoid shape / α gate
init 0.01 / ω diversity / output magnitude / gradient flow)
**Stage B**(iter#20 候选): `scripts/experiment_pdna_smoke.py` sMNIST Gapped protocol
跑 5 seed × 4 backbone(CfC / CfC+pulse / CfC+self-attend / Full PDNA) +
backbone matrix 加 sMNIST 行
**Stage C**(iter#21+ 候选): Long Range Arena 长程任务(视 Jetson RAM)

## 10. 与本仓已有研读的关系

| 已有报告 | 与 PDNA 的连接 |
|---|---|
| `LiquidTAD_..._研读报告.md` | 都是连续时间架构增强(LiquidS4 block vs CfC + pulse),**不同的时序鲁棒性维度**(长序列 vs gap robustness) |
| `MeloTune_CfC_Proactive_Music_Curation_研读报告.md` | 同 CfC backbone,不同 augmentation(Mesh Memory Protocol vs pulse) |
| `Liquid_Networks_MDH_Imitation_Learning_研读报告.md` | 同 CfC backbone(MDH 是 head),**LNN head 形态系列** |
| `Symbolic-Vector_Attention_Fusion_SVAF_研读报告.md` | **同周姊妹方向**: SVAF 是"per-agent state 协议",PDNA 是"per-hidden-dim oscillation"—— 都是 continuous-time 架构"在协议层 vs 在 activation 层"的两种 augmentation 哲学 |
| `Physics-Modeled_Neural_Networks_DynPMNN_研读报告.md` | 不同的连续时间隐喻: DynPMNN 是 FitzHugh-Nagumo 神经动力学,PDNA 是正弦振荡 |
| `Comparative_Analysis_of_LNN_and_LSTM_研读报告.md` | 都是 CfC 的相对优势实验;但本仓 11+ 轮 loop 已经知道 **synthetic 任务上 LNN 不通杀** —— PDNA 的 gap-robustness 维度可能是 LNN 真正赢的新证据点 |

## 11. 关键 takeaway(对本项目)

1. **CfC backbone + 25 行 augmentation = +4.62 pp multi-gap acc** 是本仓可立即
   复现的最小成本高信号实验
2. **pulse 模块的参数化** (A · sin(ωt + φ(h)))与本仓 `HierarchicalDecayLiquidBlock`
   的 decay-rate sharing 在哲学上互补 — 一个是 **frequency 维度**的多尺度,
   一个是 **decay 维度**的多尺度
3. **Gapped Evaluation Protocol** 是论文的方法论贡献 — 本仓可以把它接到
   `experiment_long_sequence.py` 跑 Long Sequence + Gap 联合评估
4. **noise control vs pulse** 的设计(§4 Variant B)是 ablation 严谨度的范例
   — 任何报"我加了 X 就 +Y%"的工作都应该有 matched-magnitude control
5. **承认 post-hoc augmentation 与真 ODE 的区别**(论文 §8 limitation iii)
   —— 与 SVAF 论文"intelligence is in the temporal separation"呼应,
   **两个独立工作都把 per-neuron τ / per-neuron ω 当 first-class architectural
   primitive** — 这是 LNN 文献的方法论趋同

## 12. 元数据

- **论文公开度**: arXiv standard,可访问
- **代码公开度**: **公开**(github.com/Parassharmaa/pdna) — 罕见,本轮唯一
  代码公开的 LNN 候选
- **数据可获得性**: **MNIST torchvision**,无需下载
- **复现成本估计**: 5 变体 × 5 seeds × 40 epoch sMNIST on A4000 = ~30-60 min;
  CPU 路径 hidden=64 + 5 seed × 3 backbone 估计 2-4 hr
- **与 PRD §10 关系**: 加进第三波 backlog 候选,**优先级 P1**(代码公开 +
  数据零成本 + 与本仓 LNN/CfC 复用度高 + 统计严谨)

---

> 本报告由 LNN-research-agents 自动生成,基于 arXiv 2603.00153v1 PDF + WebFetch
> 摘要交叉验证。报告日期 2026-06-04,与项目 daily digest 同步。
