---
title: Blockage Prediction in Directional mmWave Links Using LTC — 跨场景泛化 + 低功率筛选的隐蔽偏差
paper: https://arxiv.org/abs/2306.04997v1
arxiv_id: 2306.04997
date: 2026-10-26
tags: [LNN, LTC, NCP, mmWave, 6G, blockage-prediction, time-series, negative-result-audit]
status: deep-read
report-author: LNN-research-agents
source: docs/daily/2026-10-26_LNN_research_digest.md arXiv 候选 (2306.04997, 2023-06-08 提交, MIT / Aalborg University)
---

# LTC for mmWave Blockage Prediction — 研读报告

> 论文: [Blockage Prediction in Directional mmWave Links Using Liquid Time Constant Network](https://arxiv.org/abs/2306.04997v1)
> 提交: 2023-06-08 07:35:39 UTC (v1, 2 pages, IRMMW 2023 会议预印本) | 作者: Martin H. Nielsen (MIT + Aalborg University), Chia-Yi Yeh (MIT), Ming Shen (Aalborg University), Muriel Médard (MIT)
> 报告日期: 2026-10-26 | 工具: `skills/paper-analyzer` (arXiv 模式, 读 PDF 全文 + abstract)
> 强关键词命中: **liquid time constant, LTC, NCP, time-series prediction** (selector score 25)

---

## 1. 一句话定位

> **这是本仓迄今研读过的"最像工业界真实用法"的一篇 LNN 论文, 也是最需要打折的一篇。** 它的做法值得学: **只用接收信号功率 (RSS) 单路输入, 室内受控遮挡训练 → 室外非受控遮挡测试**, 完全不做场景特定重训, 这是把 LTC 的"跨场景泛化"卖点用在了刀刃上。**但两处方法论处理必须打折**: (a) 评测时**剔除了接收功率低于 0.4 (归一化) 的样本** —— 这恰恰是**最可能发生遮挡的时刻**, 因此 97.85% 这个头条数字是在一个**被筛掉困难样本后的子集**上算的; (b) 摘要与正文对**同一个 baseline 给了两个不同的引用编号** ([5] vs [2]), 而 "97.85%" 这个头条值在正文的场景区间 (73.95%–99.6%) 里找不到对应位置。**它证明的是"LTC 能从功率曲线的形状里学到预遮挡特征并迁移", 不是"LTC 在 mmWave 遮挡预测上达到 97.85% 准确率"。**

---

## 2. 元数据

| 字段 | 值 |
|---|---|
| arXiv id | 2306.04997v1 |
| 标题 | Blockage Prediction in Directional mmWave Links Using Liquid Time Constant Network |
| 作者 | Martin H. Nielsen (1,2), Chia-Yi Yeh (1), Ming Shen (2), Muriel Médard (1) |
| 单位 | 1 = Massachusetts Institute of Technology (MIT), Cambridge MA; 2 = Aalborg University, Denmark |
| 提交日期 | 2023-06-08 07:35:39 UTC |
| 页数 | **2 pages** (arXiv comment: "2 pages, pre-print for IRMMW 2023 conference") |
| 类别 | eess.SP (PDF 页边标注: `arXiv:2306.04997v1 [eess.SP] 8 Jun 2023`) |
| License | arXiv non-exclusive license |
| DOI | https://doi.org/10.48550/arXiv.2306.04997 |
| 会议 | IRMMW 2023 (IEEE International Radar Millimeter-Wave Conference) 预印本 |
| 数据集 | 公开 60 GHz 定向收发机遮挡数据集 (来自 Wu et al. 2022 [5] / Alrabeiah & Alkhateeb [2]) —— 含室内受控遮挡场景 + 室外非受控遮挡场景 (车辆经过) |
| 代码 | **无** |
| 本地 PDF | [`papers/2306.04997/2306.04997.pdf`](../../papers/2306.04997/2306.04997.pdf) |
| 本地全文 | [`papers/2306.04997/2306.04997.txt`](../../papers/2306.04997/2306.04997.txt) |

---

## 3. 核心问题

### 3.1 问题陈述

mmWave / THz 通信靠**定向波束**传输, 带宽高但极度脆弱: 一辆车或一个人穿过波束路径, 链路就断了。要做 6G/THz 的可靠通信, 必须**提前预测遮挡何时发生**。

### 3.2 现有方案的 3 类局限 (论文 §I)

1. **需要额外观测资源** —— 现有 SOTA 方法依赖 (a) 其他频段 (sub-6 GHz) 的信息 [2], 或 (b) 摄像头画面 [3]。**在真实部署里, 这意味着要额外硬件 (摄像头) 或额外射频链路, 成本与功耗都上去了。**
2. **需要先验场景信息** —— 基于元学习 (meta-learning) 的方法 [4] 需要关于周围环境的先验统计观测, **泛化到新环境代价高昂**。
3. **in-band 主动预测随预测时距衰减** —— Wu et al. 2022 [5] 提出用**预遮挡功率签名 (pre-blockage power signature)** 做 in-band 主动预测, 但论文指出: "**the prediction accuracy decreases significantly when predicting even just slightly further into the future**"。

### 3.3 论文的定位

> "if we can predict **without these additional resources or hardware**" —— **只用接收信号功率一条链路, 不加任何硬件, 不需要场景先验, 一次训练直接迁移到新场景。**

### 3.4 问题被重新表述为什么形式

问题从"如何预测遮挡"变成"**如何在单路 RSS 输入、无场景先验、跨训练/测试场景的条件下, 预测未来 $T_k$ 个时隙的遮挡状态**"。这是一个干净的、可以用一个二分类头 + 一个精度指标完整刻画的监督学习问题 —— **这是本文比很多 LNN 论文做得好的地方: 问题边界清楚, 评测可做。**

---

## 4. 方法论与核心思路

### 4.1 总体架构 (论文 Fig. 1)

```
RSS 功率序列 r[t]  (64 个 beam index 上的功率随时间变化)
        │
        ▼
[LTC 网络: 8 个神经元]  论文 §IV 描述的具体划分
   · 2 个 input neuron        ← 接收信号
   · 4 个 inter neuron        ← 中间/隐状态
   · 2 个 control neuron      ← 控制 (时间常数相关)
   · 1 个 output neuron       ← 输出 0/1
        │  (由 NCP 稀疏连接构建)
        ▼
输出神经元 → 未来 K 个时隙的遮挡预测 x̂[t+K]
```

**这是一个极小的网络 (8 神经元)**, 且用 **NCP (Neural Circuit Policies, Lechner et al. 2020)** 的稀疏连接方式构建。

### 4.2 NCP 的作用 (论文 §III)

论文对 NCP 的定位是三重的:

1. **资源效率** —— "create a very sparse and efficient network for training";
2. **未见数据的适应性** —— "allow the LTC network to be adaptable to previously unseen data";
3. **对非理想样本的鲁棒性** —— "provide robustness to nonideal data samples, which enables **offline training using pre-blockage signature**"。

**第 3 点是本文方法论的枢纽**: NCP 的稀疏结构让"用**一种场景** (室内受控遮挡) 学到的**预遮挡签名**, 迁移到另一种场景 (室外非受控遮挡)"成为可能 —— 因为网络只需要识别功率曲线的**局部形状特征**, 而不需要拟合场景的全局统计。

### 4.3 训练设置 (论文 §IV)

| 项 | 值 |
|---|---|
| 网络 | 8-neuron LTC (2 input / 4 inter / 2 control / 1 output), NCP 稀疏连接 |
| 输入 | 64 个 beam index 上的功率时序 |
| 输出 | 0 或 1 (输出神经元) |
| 训练数据 | **室内**数据集的 pre-blockage signature (受控遮挡) |
| 训练轮数 | 40 epochs |
| 优化器 | Adam |
| 损失 | cross-entropy |
| 学习率 | 0.02 |
| 测试 | **室外**数据集 (非受控遮挡, 车辆经过), 场景 17–22 |
| 评测过滤 | **排除接收功率 < 0.4 (归一化) 的预测** |

### 4.4 与 LTC (Hasani 2021) / NCP / CfC 的关系

- **与 LTC (Hasani 2021)**: **完全依赖**。本文是 LTC 在一个**具体的、有真实硬件数据支撑的工程问题**上的应用, 且明确引用了 LTC 的三个原始卖点: "strong expressiveness, stability, and performance in handling time series"; "requires fewer neurons and generalizes easier than conventional RNN networks"。

  > ⚠️ **注意一个值得留意的差别**: 本文用的是 **LTC + fused ODE solver** (真正的 ODE 求解), **不是 CfC**。在一个"要低延迟、要部署到通信设备"的场景里, **CfC (免 solver, 1–5 个数量级更快) 本应是更优选择, 论文却没有提到 CfC**。这是本文在"部署可行性"维度上的一处明显盲区。

- **与 NCP (Lechner et al. 2020)**: **深度耦合** —— NCP 不只是"稀疏化", 它是**跨场景泛化能力的载体** (§4.2 第 3 点)。

- **与 CfC**: **未涉及**。

### 4.5 问题形式化 (论文 §II)

论文给了一个干净的形式化, 这是本文值得学的部分:

$$
S_{ob} = \{r[t - T_{ob} + 1], \cdots, r[t-1], r[t]\}
$$

$$
\hat{x}[t+K], \quad \forall K \in \{1, \cdots, T_k\}
$$

$$
\max \; P\big(\hat{x}[t+K] = x[t+K] \mid S_{ob}\big), \quad \forall K \in \{1, \cdots, T_k\}
$$

其中 $r[t]$ 是接收信号功率, $x[t]\in\{0,1\}$ 是遮挡状态 (1 = 遮挡), $t \in \mathbb{Z}$ 是采样索引。**目标函数是"未来 K 个时隙的遮挡预测成功率"**, 这比笼统的"准确率"要清楚得多。

---

## 5. 核心公式提取

**本文没有新的公式** —— 它是 LTC 的应用, 核心动力学来自 Hasani 2021。下表是本报告对其所用公式的 grounding:

| 编号 | 公式 | 含义 | 来源 |
|---|---|---|---|
| Eq. 1 | $S_{ob} = \{r[t-T_{ob}+1], \cdots, r[t-1], r[t]\}$ | 观测窗口: 过去 $T_{ob}$ 个时隙的接收功率 | 本文 §II |
| Eq. 2 | $\hat{x}[t+K],\ \forall K \in \{1,\cdots,T_k\}$ | 预测目标: 未来 $T_k$ 个时隙的遮挡状态 | 本文 §II |
| Eq. 3 | $\max P(\hat{x}[t+K] = x[t+K] \mid S_{ob})$ | 优化目标: 条件遮挡预测成功率 | 本文 §II |
| **LTC** (Eq. 5) | $\frac{dx(t)}{dt} = -\left[\frac{1}{\tau} + f(x(t), I(t), t, \theta)\right]\odot x(t) + f(x(t), I(t), t, \theta)\odot A$ | LTC 神经元 ODE — **本文使用的核心动力学, 未作改动** | Hasani et al. 2021, arXiv:2006.04439 |
| **NCP** | 稀疏连接的连续时间回路策略 (inter-neuron / command-neuron 的有界稀疏拓扑) | 提供跨场景泛化与对非理想样本的鲁棒性 | Lechner et al. 2020, *Nature Machine Intelligence* 2:642–652 |
| **深度定义** | $L = \dfrac{\text{平均积分步数}}{\text{每个输入样本}}$ | LTC 的 computational depth (本文未测, 但决定了部署可行性) | Hasani 2021 §Definition of Computational Depth (L) |

---

## 6. 关键成果与贡献 —— 含证据强度审计

### 6.1 头条数字与正文区间对不上 ⚠️

论文的**三处**头条声称都用了 97.85%:

- 摘要: "the proposed LTC can predict with **upwards of 97.85% accuracy** without prior knowledge of the outdoor scenario or retraining/tuning";
- §I: "we show accuracy in blockage prediction **above 97.85% for all outdoor scenarios** for the immediate future timeslot ($t+1$)";
- §I: "accuracy in blockage prediction **above 97.85%** for all outdoor scenarios".

但**正文的 §IV 给出的区间是**:

> "The LTC network accuracy scores range from **73.95% to 99.6%**, with the lowest score being significantly better than the state-of-the-art."

**73.95% 与 "above 97.85% for all outdoor scenarios" 直接矛盾。** 一处可能的调和是: 73.95% 可能是**所有时距 (t+1/t+5/t+10) 与所有场景的最低值**, 而 97.85% 特指 t+1; 但**论文没有给出 t+1 的逐场景数值表** (Fig. 2 是柱状图, 无数据表), 因此**无法从论文正文核验 97.85% 这个数**。

**处置: 本报告把 97.85% 标注为"作者自报、无法从正文核验"。** 本仓引用的应是可核验的区间: **73.95% – 99.6%**。

### 6.2 基线引用不一致 ⚠️

论文对**同一个 baseline** 在两处给了不同引用:

- §I: "a 12% to 39% accuracy improvement in predicting blockage in the near future ($t+5$ and $t+10$ timeslots) compared to **the baseline [5]**";
- §IV / Fig. 2 caption: "using our approach (red) and **the baseline approach [2]** (blue)"; "The LTC network accuracy scores range from 73.95% to 99.6%, with the lowest score being significantly better than **the state-of-the-art**";
- §IV: "our proposed model outperforms the baseline approach given in **[2]**".

**[5] = Wu et al. 2022** (in-band pre-blockage power signature, 同一数据集来源), **[2] = Alrabeiah & Alkhateeb 2020** (sub-6 GHz 辅助的深度学习遮挡预测)。**这两者是不同的方法, 且 Fig. 2 的曲线究竟对应哪一个, 论文说不清。** 由于 [2] 用的是**额外的 sub-6 GHz 观测** (即它有本文没有的信息优势), 而 [5] 才是**单路 in-band** 的公平对手 —— 这个引用错乱直接**影响 "12%–39% 提升" 这个数字的意义**。

**处置: "12%–39% 提升"的对照组不明确, 引用时必须标注。**

### 6.3 🔴 最严重的方法论问题: 评测时剔除了低功率样本

论文 §IV:

> "Since not all 64 beams receive a strong enough signal, **we exclude the predictions where the received power is lower than 0.4 normalized received power.**"

**这一句话对结论的影响是决定性的。** 物理上:

- 遮挡发生时, 接收功率**大幅下降**;
- **被剔掉的正是 "接收功率 < 0.4 归一化" 的样本, 即最可能正在发生遮挡、或刚发生遮挡的时刻**;
- 而遮挡预测任务的**真正价值恰恰在于预测这些低功率时刻会不会变成持续遮挡**。

**换句话说, 这个筛选把"深遮挡"这个最具业务意义的子集从评测中移除, 换来一个更高的准确率。** 这不是一个中性的工程细节 —— 它**系统性地抬高了报告的准确率**。

**因此**: 97.85% / 73.95%–99.6% 这些数字**只对"接收功率高于 0.4 归一化"的样本子集有效**。论文没有报告**在完整样本集上的准确率**, 也没有报告被剔除样本的比例。**本报告不引用这些数字作为"LTC 的遮挡预测准确率", 只引用为"在筛除低功率样本后的条件准确率"。**

### 6.4 Fig. 1 题注与实际频段不符 (次要)

Fig. 1 题注: "Proposed architecture and system model for **THz** blockage prediction"; §IV 结尾: "attractive option for pre-blockage prediction in **sub-terahertz** communication systems"。但**数据集是 60 GHz** (§IV: "a directional transmitter and a directional receiver at 60 GHz")。

60 GHz 是 mmWave, 不是 THz (THz 通常指 100 GHz 以上)。**"THz" 的措辞超出了实测数据的支持范围**, 虽然论文引言里同时讨论了 mmWave 和 THz 两个频段, 显然是从引言的表述带到了结论。

### 6.5 逐场景数字缺失

论文正文只给了三个概括性数字: **73.95%–99.6%** (总区间)、**>80% (t+5, 部分场景)**、**>70% (t+10, 部分场景)**。**6 个室外场景 (17–22) × 3 个时距 (t+1/t+5/t+10) = 18 个数据点, 一个具体的表格都没有**, 全部信息只在 Fig. 2 的柱状图里。**"部分场景" (in some scenarios) 这个限定语也没有说明是哪些场景。**

### 6.6 贡献清单 (本仓视角)

1. **✅ 真正有价值的工程洞察**: **RSS 单路输入 + 室内→室外的跨场景迁移** 是 LNN 泛化能力的一个**非玩具 (non-toy)** 验证。作者自己的总结是准确的: "the neural network can be quickly deployed and **does not need to be trained for individual environments**"。
2. **✅ 问题形式化清晰** (§II Eq. 1–3): 观测窗口 / 预测目标 / 优化目标三段式, 是本仓研读过的 LNN 应用论文里少见的规范。
3. **✅ NCP 的角色被正确识别**: 不是"稀疏化好看", 而是**跨场景泛化的载体** —— 这与本仓 `lnn/core/sncp_policy_lite.py` (crowdnav LTC + actor-critic) 的立意同向。
4. **⚠️ 头条数字不可核验**: 97.85% 与正文 73.95%–99.6% 矛盾, 且无逐场景表 (§6.1)。
5. **⚠️ 对照组引用错乱**: "12%–39% 提升"的 baseline 是 [5] 还是 [2] 说不清 (§6.2)。
6. **🔴 低功率样本剔除使准确率系统性偏高** (§6.3) —— **这是本报告最重要的折扣**。
7. **🔴 未考虑 CfC**: 部署到通信设备时 CfC 免 solver 的优势直接对口, 论文完全没提。
8. **🟠 无延迟 / 吞吐 / 功耗数字**: 8 神经元 + fused ODE solver 的每步推理开销完全未测 —— 而"低延迟通信"是论文动机之一。
9. **🟠 无代码, 无逐场景数据表** → 按 AGENTS.md 约束 3, 标注**未复现**。

---

## 7. 局限性与未来展望

### 7.1 作者自陈局限

论文几乎没有自陈局限 (§IV 结尾只有一句): "For future work, we will explore how this pre-blockage prediction can help improve reliability and latency in THz communication systems." 即**未来方向是把预测接到实际通信系统里**, 而非承认本文方法的不足。

**作者未提及的三个自认局限, 按重要性排序**: (a) 低功率样本剔除; (b) 97.85% 与 73.95% 的区间矛盾; (c) baseline 引用不一致。

### 7.2 隐含局限 (本仓视角)

1. **🔴 评测样本筛选引入系统性高估** (§6.3) —— 剔除低功率样本 = 剔除深遮挡时刻。
2. **🔴 头条数字无法从正文核验** (§6.1) —— 97.85% vs 73.95%–99.6% 矛盾, 无逐场景表。
3. **🔴 对照组引用不一致** (§6.2) —— 影响 "12%–39%" 的可解释性。
4. **🟠 场景数量少 (6 个)、且都来自同一数据集的同一采集设置** —— 泛化性结论的适用范围限于 "受控 → 非受控" 这一种迁移类型, 不等于"任意环境可迁移"。
5. **🟠 训练集用了测试场景的"同源"数据** —— 室内与室外来自同一采集框架 [5], 因此"跨场景"是**同一实验系统内的域迁移**, 不是跨设备/跨厂商/跨频段的迁移。
6. **🟠 只用单路 RSS, 没有与"功率阈值 + 迟滞"这类经典信号处理规则对照** —— 遮挡检测里的经典做法 (功率下降 + 持续时间判决) 极其廉价且鲁棒; **论文没有把它列为基线**, 这使 "12%–39% 提升" 的实际意义打折 —— 因为一个几行代码的规则可能已经拿到大部分收益。
7. **🟠 未与 CfC / 其他连续时间模型对照** —— 无法判断收益来自"LTC 动力学"还是"NCP 稀疏性"还是"8 神经元的小网络"。
8. **🟠 未测推理延迟/功耗** —— 通信设备场景的核心指标缺失。
9. **🟡 只有 t+1/t+5/t+10 三个离散时距**, 未给出预测时距与精度的连续曲线。
10. **🟡 无代码**。

### 7.3 未来方向 (本仓建议)

- **复现本报告 §6.3 的高估假设**: 在本仓 `bench_irregular_dt.py` 或一个新写的 mmWave 风格 bench 上, 对同一批预测**分别在"全样本"与"剔除低功率样本"两个子集上算准确率**, 量化这类筛选带来的偏差幅度。**这是成本最低、且对本仓报告规约最有价值的一次复现** —— 因为本仓报告普遍不做这种样本筛选, 但知道"筛选能造出多少虚高"有助于解读外部论文。
- **补一条 baseline**: 论文缺的那个"功率阈值 + 迟滞"规则基线, 值得在本仓自己数据上跑一遍, 作为**任何"时序模型 vs 简单规则"类声称的标尺**。

---

## 8. 对本仓的意义

### 8.1 可作 `lnn/core/variants.py` 的什么变体

**无新变体**, 但有一处**结构模式**值得对齐: **本仓 `lnn/core/sncp_policy_lite.py` 的 SNCP 架构** (LTC + NCP 稀疏连接 + actor-critic) 与本文的 **LTC + NCP 稀疏连接 + 二分类头** 是**同一个骨架**, 只是任务不同 (人群导航 vs 链路遮挡)。**本文给本仓 SNCP 那条线补了一个独立应用场景的佐证** —— "LTC + NCP 这个组合在真实工程时序任务上确实有效", 且**有效的方式正是稀疏化带来的跨域迁移**, 而不是精度内卷。

### 8.2 可嫁接到哪个 bench 脚本

**建议新增一条 bench: `scripts/bench_ltc_mmwave_style_transfer.py`** —— 合成一个"功率轨迹 → 二分类"的任务, 结构照本文:

- 输入: 长度 T 的标量/多 beam 功率序列;
- 模型: 小 LTC (可复用 `lnn/core/ltc.py` 的 `LTCNetwork`) + 稀疏连接;
- **两个域**: 域 A = 受控遮挡 (固定形状的下降沿), 域 B = 非受控 (随机时机的下降沿);
- **训练在 A, 测试在 B** —— 度量跨域迁移;
- **两个指标口径**: 全样本准确率 **与** "剔除低功率样本后"准确率 —— **把 §6.3 的偏差直接量化成本仓自己的数字**。

**这一步的成本远低于复现真实 mmWave 数据集 (需要外部硬件数据 + 许可), 而产出的是本仓自己的、可复核的"跨域迁移 + 评测口径"证据。**

### 8.3 提供哪个新 ablation axis

**新增消融轴: "跨域迁移 (domain transfer)"** —— 本仓现有消融沿 τ / β / 分支数 / 时间条件化 / 遗忘门 (§见 LNN-Fly 报告的 "Δt 条件化 × 遗忘门" 建议), **尚未把"训练域 ≠ 测试域"作为一个独立的轴**。本文说明这是一个真实且 LNN 可能占优的场景 (因为连续时间动力学对输入形状的依赖更弱)。

**配套的新增消融轴: "评测样本筛选的口径"** —— 任何时序分类任务都可以顺手算两个数 (全样本 vs 剔除困难样本), 差值就是本报告 §6.3 揭示的那类虚高。**建议本仓今后所有报告出现准确率时, 注明是否做过样本筛选。**

### 8.4 与本仓既有结论的关系

- **与 LNN-Fly (2606.28827) 同向且互补**: LNN-Fly 证明"连续时间形式不够, 必须对真实 Δt 做条件化" (同条件对照, 12/20 vs 5/20); 本文证明"连续时间 + 稀疏化可以在**不重训**的前提下跨到新场景" —— **两者都是本仓 §0 "边缘部署 / 时间归纳偏置组件" 定位的正向支撑**, 且都不是主基准横扫 (符合 NR-06)。
- **与本仓既有 telecom 研读的关系**: 本仓已有 `docs/reports/LNN_Next-Generation_AI_for_Telecom_研读报告.md` (自第一性原理的电信 LNN 综述) 与 `docs/reports/Liquid_Crystal_Antennas_LNN_6G_Beamforming_研读报告.md` / `Liquid_Crystal_Antennas_Hybrid_BF_LNN_2604.07219_研读报告.md` (液晶天线 + LNN 波束赋形)。**本文的独特价值是它是其中唯一一篇有真实硬件测量数据支撑的** (60 GHz 定向收发机), 另两篇偏架构提案。
- **与 NR-06 的关系**: 本文**不构成**任何"LNN 在标准主基准上 SOTA"的证据 —— 它是一个长尾工业任务 (6G 链路预测), 与本仓 §0 "LNN 真价值所在: 时间连续性关键子任务" 一栏一致 (GazeLNN / Fall Detection / FlowFake / Pulse-Driven 同属这一类)。

---

## 9. Verdict

- **"LTC 能在只加接收功率、不需场景先验、不重训的条件下跨场景预测遮挡" → TARGET-POSITIVE (方法论层面)** —— 这是本文的核心贡献, 且被 NCP 稀疏结构解释清楚, 与本仓 SNCP 那条线同构。**引用这一条是安全的。**
- **"LTC 在 mmWave 遮挡预测上达到 97.85% 准确率" → TARGET-NEGATIVE-WITH-NUANCE** —— 该数字与正文 73.95%–99.6% 矛盾, 无逐场景表可核验, 且**是在剔除低功率样本后的条件准确率**。**引用时必须带全部三条限定。**
- **"比 SOTA 提升 12%–39%" → TARGET-DEPENDENT-WITH-NUANCE** —— baseline 引用在 [5] 与 [2] 之间摇摆, 而两者信息条件不同; 且缺"功率阈值+迟滞"这个廉价规则基线。**可作为线索, 不可作为结论。**
- **作为本仓"跨域迁移"消融轴的来源 → TARGET-POSITIVE** —— §8.2 给出了一条成本可控、产出本仓自有数字的 bench 设计。
- **作为"评测样本筛选会造出虚高"的方法论警示 → TARGET-POSITIVE** —— §6.3 的分析可直接写进本仓的报告规约。

---

## 10. 引用与 grounding

| 本报告的断言 | grounding 路径 |
|---|---|
| LTC 动力学 Eq. 5 | `papers/foundational/hasani_2021_ltc.txt`; `docs/LNN_深度研读报告.md` §1.2 (L2) |
| LTC computational depth 定义 | `papers/foundational/hasani_2021_ltc.txt` §Definition of Computational Depth (L) |
| NCP (稀疏 + 跨场景适应) | Lechner et al. 2020, *Nature Machine Intelligence* 2:642–652 (论文 ref [7]); 本仓落地: `lnn/core/sncp_policy_lite.py`, `docs/reports/SNCP-PPO_Crowdnav_LTC_深度研读报告.md` |
| CfC 免 solver / 1–5 数量级 | `papers/foundational/lechner_2022_cfc.txt`; `docs/LNN_深度研读报告.md` §1.2 (L3) |
| "连续时间形式不够, 必须 Δt 条件化" | `docs/reports/LNN-Fly_Continuous_Time_UAV_Navigation_2606.28827_研读报告.md` |
| "LNN 未在标准主基准显著超过 transformer" | `docs/LNN_深度研读报告.md` §0 + `analysis/negative_results/README.md` NR-06 |
| 本仓既有 telecom 研读 | `docs/reports/LNN_Next-Generation_AI_for_Telecom_研读报告.md`; `docs/reports/Liquid_Crystal_Antennas_LNN_6G_Beamforming_研读报告.md`; `docs/reports/Liquid_Crystal_Antennas_Hybrid_BF_LNN_2604.07219_研读报告.md` |
| 文本审计脚本 | `/tmp/lnn_mmwave_audit.py` (2026-10-26) |
