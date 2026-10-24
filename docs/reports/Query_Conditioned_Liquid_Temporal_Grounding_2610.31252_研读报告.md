---
title: "Query-Conditioned Liquid Temporal Grounding for Long-Form Audio"
date: 2026-10
tags: [LNN, CfC, temporal-audio-grounding, FiLM, query-conditioning, audio-language-model, long-form-audio, candidate-pool, boundary-refinement]
arxiv_id: 2610.31252v1
---

# 研读报告：Query-Conditioned Liquid Temporal Grounding — 用 FiLM + CfC 把长音频定位改写成"多候选生成 + 冻结 LALM 选择"

## 1. 元数据

- **论文标题**：Query-Conditioned Liquid Temporal Grounding for Long-Form Audio
- **作者**：Yuelan Cheng¹, Yonggang Zhu³, Yongqiang Chen¹, Peiwei Chang¹, Yingyi Chen², Wenwu Wang¹
- **机构**：¹Centre for Vision, Speech and Signal Processing, University of Surrey, UK；²College of Information and Electrical Engineering, China Agricultural University, China；³School of Artificial Intelligence, Beijing University of Posts and Telecommunications, China
- **发表时间**：2026-10-22 (arXiv: 2610.31252v1, cs.SD)
- **代码**：文中未给出公开仓库地址
- **本地 PDF**：[papers/2610.31252/2610.31252.pdf](../../papers/2610.31252/2610.31252.pdf)（771 KB, 5 页，已落盘到仓库）
- **本地文本**：`../../papers/2610.31252/2610.31252.txt`（22.3 KB，5 页）
- **关联概念**：CfC (Hasani 2022) / Closed-form continuous-time / FiLM (Perez 2018) / 冻结大音频语言模型 (LALM) / 候选池 + 语义选择 / 时序音频定位 (TAG) / 长音频理解 / 边界精修
- **本仓相关性**：**中等偏正面** —— 本文把 CfC 用作"查询条件化时序建模"的组件层，与本仓 §0"液态作为时间归纳偏置组件"定位一致；但任务是音频定位而非时序预测，且核心增益来自**候选池 + 冻结 LALM 选择**这套架构，**CfC 本身的贡献经消融被证明是关键的但非充分**（见 §5.2）。

## 2. 核心问题

**长音频时序定位 (Temporal Audio Grounding, TAG)** 要求给定一段自然语言 query，返回录音中对应事件的时间区间 [s, e]。短音频形态（目标事件占据录音相当比例）已被音频-语言预训练方法较好解决，但**长音频形态（可达 30 分钟）引入三重困难**：

1. **目标极稀疏**：目标事件往往只占录音的很小一部分，其余是大量无关或**声学/语义上相似**的内容（论文 Fig. 1b 的例子：30 分钟录音里定位"演讲者快速连举 Rosa Parks / 甘地盐行 / 妇女参政运动口号 / 肯尼迪登月计划四个历史例证"的那一小段）。
2. **query 多约束**：query 可能同时包含说话人、事件、属性与时序关系等多个语义约束，需要在**大时间搜索空间**上做联合匹配。
3. **现有方案退化**：已有研究明确报告音频时长增加时性能显著下降，且精确时序定位对当前音频语言模型仍是难点。

论文指出已有做法的不足：显式时序编码、grounding 监督、global-to-local 推理，都**没有让时序建模直接由 query 引导**。

论文把问题重述为：**如何在不改动冻结音频语言骨干的前提下，用一个被 query 条件化驱动的连续时间模块，在长录音上保留多个时序假设，并从中挑出语义最匹配的一个。**

## 3. 方法论与核心思路

### 3.1 总体架构（Fig. 2）

```
音频 → 冻结骨干 (MOSS-Audio / Qwen3-Omni) 编码 → H
query → 冻结骨干编码 → Q (token 级), q (池化)
       ↓
   FiLM 查询条件化 → H̃
       ↓
   CfC 连续时间建模 → x_t → z_t → Z
       ↓ (并行)
   帧级 音频-查询 匹配信号 p^m
       ↓
   融合 F = LN([H̃ ; Z ; p^m]) → 三个轻量头 → p_s, p_e, p_r
       ↓
   候选区间提取 (无参数) → 候选池 C = {c_i}
       ↓
   冻结 LALM 逐候选打 Yes/No 兼容性分 → Top-1 候选 τ̃
       ↓
   轻量边界精修 → 最终区间 [ŝ, ê]
```

设计哲学：**不直接从全录音回归单个区间**，而是先保留多个 query 相关时序假设（候选池），再由**冻结的大音频语言模型**做语义裁决，最后做局部边界精修。骨干全程冻结，只训练轻量 grounding 模块 —— **单张 A100 即可训练**。

### 3.2 查询条件化：FiLM（Eq. 1）

音频编码器输出高分辨率表征，经时间投影与池化得到 $H = [h_1, \dots, h_T]$；query 编码器输出 token 级 $Q = [q_1, \dots, q_L]$ 与池化表征 $q$。由于 $H$ 初始与 query 无关，先用特征线性调制注入 query 语义：

$$
\tilde{h}_t = \big(1 + f_\gamma(q)\big) \odot h^*_t + f_\beta(q) \tag{1}
$$

其中 $f_\gamma$、$f_\beta$ 是可学习投影，$h^*_t$ 是投影后的帧特征。**同一组调制参数在所有时间步共享** —— 这既让模型强调 query 相关特征维度，又保留时间顺序结构。

### 3.3 连续时间时序建模：CfC（Eq. 2）

查询条件化序列 $\tilde{H} = [\tilde{h}_1, \dots, \tilde{h}_T]$ 交给闭式连续时间网络：

$$
x_t = \mathrm{CfCCell}(\tilde{h}_t, x_{t-1};\, \Delta t) \tag{2}
$$

状态经归一化与正则化为 $z_t = \mathrm{Dropout}(\mathrm{LN}(x_t))$，得 $Z = [z_1, \dots, z_T]$；对时长为 $D$ 的录音，$\Delta_t = D/(T-1)$。

论文对"液态"的定义是明确且克制的：**"liquid" 指 CfC 的输入自适应连续时间动力学，使时序演化依赖查询条件化的表征**。关键在于 **FiLM 在 CfC 之前** —— query 语义直接影响长程时序信息的传播，而不只是在下游候选打分时才起作用。

### 3.4 帧级匹配信号与候选生成（Eq. 3–4）

并行地，从**原始**时序表征 $H$ 计算帧级音频-查询匹配信号：音频帧与 query token 交互得 token 级匹配分 $S_i^{tok}$，与池化 query 的全局相似度 $S_i^{glb}$ 组合：

$$
p_i^m = (1 - w_g) S_i^{tok} + w_g S_i^{glb} \tag{3}
$$

匹配信号与 FiLM 条件化特征、CfC 输出融合为 $F = \mathrm{LN}([\tilde{H}; Z; p^m])$，经三个轻量头预测 start / end / 帧级 relevance 分布 $p_s, p_e, p_r$。

候选区间由 relevance 与 matching 的高响应区生成：

$$
u = w_r \bar{p}_r + w_m \bar{p}_m \tag{4}
$$

种子取 $u$ 之上固定分位 $\rho_s = 0.85$，向满足增长分位 $\rho_g$（默认 0.65）的邻域生长，合并邻近区域，生成多尺度时序变体，最后用 $p_s$、$p_e$ 的局部响应调整边界，得到候选集 $C = \{c_i = (s_i, e_i)\}_{i=1}^M$。**整个候选提取过程无可训练参数**，只作用于模型预测出的时序证据。

### 3.5 训练目标（Eq. 5）

$$
\mathcal{L}_{gen} = \lambda_{bnd}\mathcal{L}_{bnd} + \lambda_{rel}\mathcal{L}_{rel} + \lambda_{match}\mathcal{L}_{match} + \lambda_{gIoU}\mathcal{L}_{gIoU} + \lambda_{ehn}\mathcal{L}_{ehn} \tag{5}
$$

- $\mathcal{L}_{bnd}$：用软边界目标监督 start/end 分布
- $\mathcal{L}_{rel}$：focal 风格类平衡 BCE，监督帧级 relevance，应对长录音的前景-背景不平衡
- $\mathcal{L}_{match}$：硬负样本损失，要求 GT 区间内响应超过区间外的强响应
- $\mathcal{L}_{gIoU}$：一维 GIoU 损失，监督预测区间
- $\mathcal{L}_{ehn}$：端点硬负样本损失，压制远离 GT 端点的伪响应

损失权重 $(0.3, 1.0, 0.5, 0.5, 0.3)$。**论文明确指出**：这些损失**不直接监督候选级排序**，语义候选选择交给冻结 LALM 完成 —— 这是理解其架构分工的关键。

### 3.6 LALM 候选选择与边界精修

对每个候选区间 $c_i = (s_i, e_i)$，冻结 LALM 独立评估对应音频裁剪与原始 query $Q$，用 Yes/No 兼容性 prompt，对 Yes/No 两个 next-token logit 做双向 softmax，取 Yes 概率作为兼容性分，选最高分者。

对 Qwen3-Omni 系统再接一个轻量局部精修器：在候选边界外扩 5 秒的窗口上，冻结 Qwen3-Omni 音频编码器给语义表征，轻量 Conv1D 编码器在 mel 频谱上给高分辨率声学特征；两者都经 FiLM 查询条件化、融合、在候选的 start/center/end 附近池化，得到局部边界表征。预测**有界残差修正与门控**：

$$
\Delta b = \delta_{max}\tanh(r_b), \qquad \alpha_b = \sigma(g_b), \quad b \in \{s, e\}, \qquad \delta_{max} = 5\ \text{秒}
$$

$$
\hat{b} = \operatorname{clip}\big(\tilde{b} + \alpha_b \Delta b,\ 0,\ D\big) \tag{}
$$

训练损失 $\mathcal{L}_{ref} = \lambda_{reg}\mathcal{L}_{reg} + \lambda_{IoU}\mathcal{L}_{IoU}$；需要超过 $\delta_{max}$ 修正的候选在训练时**被排除**（其目标超出局部精修器的修正范围）。

### 3.7 与 CfC / LTC / GRU 的关系

| 维度 | 本文的用法 | 与本仓的关系 |
|---|---|---|
| CfC | 作为**查询条件化后的时序建模器**，位于 FiLM 之后、候选生成之前 | 与本仓 `lnn/core/variants.py` 中 CfC 作为可插拔时序单元的位置一致；本论文提供的是"FiLM 前置到 CfC"这一**条件化顺序**的证据 |
| LTC | 未直接使用（CfC 已含 LTC 血统的连续时间机制） | 无新信息 |
| GRU | **作为消融对照组**：把 CfC 换成标准 GRU | 这是本文最有价值的对照之一，见 §5.2 |
| 骨干 (LALM) | **冻结**，只做 Yes/No 打分 | 与本仓"轻量模块 + 冻结骨干"的边缘推理路线同构 |

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. 1 | $$ \tilde{h}_t = \big(1 + f_\gamma(q)\big)\odot h^*_t + f_\beta(q) $$ | FiLM 查询条件化；调制参数跨时间步共享 |
| Eq. 2 | $$ x_t = \mathrm{CfCCell}(\tilde{h}_t, x_{t-1};\,\Delta t) $$ | 查询条件化序列上的闭式连续时间更新 |
| Eq. 3 | $$ p_i^m = (1-w_g)S_i^{tok} + w_g S_i^{glb} $$ | 帧级 token 级匹配与全局相似度的融合 |
| Eq. 4 | $$ u = w_r\bar{p}_r + w_m\bar{p}_m $$ | 归一化 relevance 与 matching 曲线的融合，用于选种子 |
| Eq. 5 | $$ \mathcal{L}_{gen} = \lambda_{bnd}\mathcal{L}_{bnd}+\lambda_{rel}\mathcal{L}_{rel}+\lambda_{match}\mathcal{L}_{match}+\lambda_{gIoU}\mathcal{L}_{gIoU}+\lambda_{ehn}\mathcal{L}_{ehn} $$ | 时序证据预测器的复合训练目标 |
| Eq. 4' | $$ \hat{b} = \operatorname{clip}\big(\tilde{b}+\alpha_b\Delta b,\,0,\,D\big),\quad \Delta b=\delta_{max}\tanh(r_b),\ \alpha_b=\sigma(g_b) $$ | 边界精修的有界残差 + 门控，$\delta_{max}=5$ 秒 |

## 5. 关键成果与贡献

### 5.1 主结果（Table 1，LAT-Bench 英文 TAG 子集，全部为百分比）

| 模型 / 设置 | mIoU | R@0.3 | R@0.5 | R@0.7 |
|---|---|---|---|---|
| *官方报告值（Listening with Time）* | | | | |
| LAT-Audio Main Model | 50.00 | 68.10 | 54.10 | 34.50 |
| Gemini-2.5-Pro Main Model | 45.30 | 65.20 | 53.90 | 27.70 |
| Gemini-3.0-Pro Main Model | 41.00 | 51.40 | 44.80 | 22.90 |
| Qwen3-Omni SW | 26.20 | 41.90 | 25.80 | 15.50 |
| Qwen3-Omni QA-SFT | 39.20 | 54.20 | 41.70 | 24.50 |
| *本文实验* | | | | |
| MOSS-Audio Direct | 15.71 | 22.77 | 14.79 | 8.69 |
| MOSS-Audio SW | 26.39 | 36.15 | 14.79 | 4.23 |
| MOSS-Audio QA-SFT | 17.75 | 27.00 | 18.08 | 9.86 |
| **MOSS-Audio Ours** | **38.52** | **57.98** | **40.61** | **19.25** |
| Qwen3-Omni Direct | 15.80 | 26.40 | 16.00 | 7.00 |
| Qwen3-Omni Ours (ρ_g=0.65) | 42.72 | 65.02 | 42.49 | 19.95 |
| Qwen3-Omni Ours (ρ_g=0.65) + Ref. | 43.20 | 65.26 | 42.96 | 21.13 |
| Qwen3-Omni Ours (Union) | 43.45 | 65.26 | 45.54 | 21.13 |
| **Qwen3-Omni Ours (Union) + Ref.** | **44.13** | **65.02** | **46.95** | **22.07** |

**关键读法（论文自己也很坦白）**：
- 相对自家基线，Direct → Ours 是**巨大提升**（Qwen3-Omni mIoU 15.80 → 42.72，+26.9）。
- 相对**官方报告的 QA-SFT**，本文在 mIoU 上超过（42.72 vs 39.20；Union+Ref 44.13 vs 39.20），**但 R@0.7 明显更低**（19.95 vs 24.50）—— 即"整体交叠更好、严格阈值更差"，论文原文承认"remains below QA-SFT at R@0.7"。
- 论文**没有**与最强基线 LAT-Audio（50.00 mIoU）在同口径下比较 —— LAT-Audio 绝对值仍最高，论文定位为"轻量适配冻结骨干"而非刷榜。

### 5.2 消融：FiLM 与 CfC 的互补性（Table 3，同用 ρ_g=0.65）

| FiLM | 时序模块 | mIoU | R@0.3 | R@0.5 | R@0.7 |
|---|---|---|---|---|---|
| *MOSS-Audio* | | | | | |
| No | None | 36.29 | 54.92 | 36.38 | 18.31 |
| Yes | None | 33.85 | 51.88 | 34.04 | 17.14 |
| No | CfC | 37.21 | 54.67 | 37.79 | 19.71 |
| Yes | GRU | 32.00 | 46.95 | 32.86 | 17.84 |
| **Yes** | **CfC** | **38.52** | **57.98** | **40.61** | 19.25 |
| *Qwen3-Omni* | | | | | |
| No | None | 35.43 | 53.52 | 37.10 | 15.25 |
| Yes | None | 35.31 | 53.29 | 37.79 | 17.14 |
| No | CfC | 38.37 | 59.62 | 39.44 | 14.79 |
| Yes | GRU | 31.29 | 48.83 | 30.05 | 12.68 |
| **Yes** | **CfC** | **42.72** | **65.02** | **42.49** | 19.95 |

**三条可读的结论**：
1. **CfC 明显优于 GRU**：Qwen3-Omni 上 42.72 vs 31.29 mIoU（+11.4）。论文据此断言"收益不能简单归因于引入了递推时序建模"，而是来自 CfC 的连续时间、输入自适应动力学。
2. **FiLM 单独用无效甚至有害**：Qwen3-Omni 上 FiLM-Only（35.31）低于 No-FiLM-No-CfC（35.43），MOSS-Audio 上 33.85 < 36.29。**只有与 CfC 耦合时才产生正增益**（38.37 → 42.72）。论文明确写道"query conditioning becomes more effective when coupled with CfC-based liquid temporal modeling"。
3. **FiLM+CfC 并非在所有指标上最优**：MOSS-Audio 的 R@0.7 上 No-FiLM+CfC（19.71）反而高于 FiLM+CfC（19.25）。论文只强调 mIoU 最高，**这一点被弱化处理，值得读者留意**。

### 5.3 候选池分析（Table 2，精修前 Top-1 vs Oracle）

| 骨干 | 候选池 | 选择 | mIoU | R@0.3 | R@0.5 | R@0.7 |
|---|---|---|---|---|---|---|
| MOSS-Audio | Ours | Top-1 | 38.52 | 57.98 | 40.61 | 19.25 |
| MOSS-Audio | Ours | **Oracle** | **46.72** | 68.31 | 50.94 | 26.76 |
| Qwen3-Omni | ρ_g=0.65 | Top-1 | 42.72 | 65.02 | 42.49 | 19.95 |
| Qwen3-Omni | ρ_g=0.65 | **Oracle** | **51.04** | 74.41 | 55.63 | 30.52 |
| Qwen3-Omni | ρ_g=0.80 | Top-1 | 41.47 | 62.44 | 43.90 | 18.31 |
| Qwen3-Omni | ρ_g=0.80 | **Oracle** | 50.66 | 74.18 | 54.93 | 31.22 |
| Qwen3-Omni | Union | Top-1 | 43.45 | 65.26 | 45.54 | 21.13 |
| Qwen3-Omni | Union | **Oracle** | **55.20** | **78.40** | **61.97** | **38.26** |

**Oracle（从同一候选池里选与 GT 交叠最大的候选）与 Top-1 之间存在 8–12 mIoU 的持续差距** —— 即**候选池里往往已经藏着更好的时序假设，但冻结 LALM 的语义选择没能挑出来**。论文在结论中把这一点列为首要未来方向（"The Top-1–oracle gap motivates future work on stronger candidate selection"）。Union 池同时提升 Top-1 与 Oracle，说明两个提案配置提供的是**互补而非冗余**的假设。

### 5.4 贡献清单

1. **架构级贡献**：把长音频 TAG 从"单区间回归"改写为"多候选生成 + 冻结 LALM 语义选择 + 局部边界精修"的三段式，把"选哪个"从时序模块的负担中解耦出来。
2. **条件化顺序的证据**：FiLM 置于 CfC **之前**（而非之后），使 query 语义直接参与长程时序传播；消融显示 FiLM 单独无效、与 CfC 耦合才有效（+4.35 mIoU on Qwen3-Omni）。
3. **CfC vs GRU 的干净对照**：+11.4 mIoU（Qwen3-Omni），支持"连续时间输入自适应动力学"而非"递推本身"是收益来源。
4. **候选池的上下界刻画**：给出 Top-1 与 Oracle 的系统性差距，把"候选选择"确立为下一个瓶颈。
5. **轻量性**：骨干全程冻结，仅训练 grounding 模块，单张 A100 可训练，3 epoch / lr 1e-4 / 1024 temporal bins。

## 6. 局限性与未来展望

### 6.1 作者自陈局限

- **Top-1–oracle 差距**：候选选择仍是关键瓶颈（明确写入结论）。
- **R@0.7 低于 QA-SFT**：精修后的系统在严格 IoU 阈值下仍不及 QA-SFT 基线。
- **只有 Qwen3-Omni 系统有边界精修**，MOSS-Audio 没有 —— 两个骨干的配置不对等，跨骨干比较时需注意。
- **训练集与评测集来自同一系列**（LAT Chronicle 训练 → LAT-Bench 评测，3,737 对 query-segment，9:1 划分），存在同源分布偏好。
- **英文子集**：仅在 LAT-Bench 英文 TAG 子集上评测。
- **冻结骨干的隐性代价**：每个候选都要调一次 LALM（前向推理成本随候选数 M 线性增长），论文未报告候选数 M 的取值，也未报告因此带来的推理开销。

### 6.2 隐含局限（本仓视角）

- **"轻量"是相对的**：冻结 30B 的 Qwen3-Omni 与 8B 的 MOSS-Audio，候选选择阶段每个候选一次 LALM 前向。与本仓"边缘部署、毫秒级延迟"的目标相差甚远 —— **本仓若要借鉴，只能借鉴"FiLM 前置到 CfC"这一条件化顺序，不能借鉴整套流水线**。
- **Δt 的使用是平凡的**：$\Delta_t = D/(T-1)$ 是均匀分箱，**没有处理不规则采样**。而这恰恰是 LNN 相对固定步长 RNN 的核心卖点之一 —— 本文并未展示 CfC 在变步长下的优势，$\Delta_t$ 只是一个占位符。
- **关键指标未被同等对待**：Table 3 中 MOSS-Audio 的 R@0.7 上 FiLM+CfC（19.25）**低于** No-FiLM+CfC（19.71）；论文的叙述完全聚焦 mIoU。**在本仓语境下这是重要折扣** —— 本仓关注的时序任务往往更看重严格指标而非平均交叠。
- **GRU 消融的公平性存疑**：GRU 变体与 CfC 变体的参数量/超参是否匹配未说明；若 GRU 用了不同的时间步数或训练预算，"CfC 优于 GRU"的因果强度会打折。
- **与本仓主战场的关系**：本仓关注连续时间建模的**时序预测/表征**能力，本文关注**定位（grounding）**。CfC 在本文中承担的其实是一个"条件化时序聚合器"，其增益与本仓关心的"预测精度/表征质量"没有直接对应关系。
- **可复现性缺口**：无代码、无 LAT-Bench 上除官方 subset 外的完整配置（3 epoch / lr 1e-4 / 1024 bins 已给，但候选数 M、w_r/w_m/w_g、LALM prompt 模板等超参未给）。按 AGENTS.md 约束 3，全部数字标注为"作者自报、我方未复现"。

### 6.3 未来方向

- 更强的候选选择器（论文自陈的首要方向）；可考虑用轻量打分器替代 LALM，压缩推理成本；
- 变步长 / 不规则采样的 $\Delta_t$ 建模（真正发挥 CfC 的优势）；
- 跨语言、跨数据集（不只 LAT 系列）的泛化验证；
- 把边界精修推广到所有骨干（当前只有 Qwen3-Omni 有）。

## 7. 对本仓的意义

**可作 `lnn/core/variants.py` 的什么变体。** 最直接的是 **"FiLM 前置条件化 + CfC"** 这一组合：本仓若在 CfC 前加一层由外部条件（多变量协变量、任务 query、上下文 embedding）驱动的 FiLM 式调制，是一个成本极低、且本文提供同骨干消融支撑的变体。注意本文的结论是 **FiLM 单独无效、必须与 CfC 耦合** —— 若本仓单独加 FiLM 而不动时序模块，预期无收益。

**可嫁接到哪个 bench 脚本。** 本仓的 `bench_cfc_n_tau.py`（多时间常数）、`bench_multi_tau_cfc.py`、`bench_decoupled_cfc.py` 都能接一个条件化输入头，检验"条件化 × τ"交互。更贴近本文结构的是**新增"候选池 + 选择器"这一评测范式**：本仓多数 bench 是单次回归，没有"生成多个候选再选"这一层，Oracle 与 Top-1 的差距度量在本仓还没有对应指标。

**提供哪个新 ablation axis。** **"连续时间 vs 离散递推（GRU/LSTM）"作为一等消融轴**，且必须在**固定参数量与固定训练预算**下做（本文该对照的参数匹配未说明）。本仓现有 `bench_cfc_vs_ssm.py` 已在做类似对比，但需按本文教训补上**参数匹配声明**。

**Verdict**：
- **作为"CfC 优于 GRU"的证据：TARGET-DEPENDENT-WITH-NUANCE** —— 支持性较强但非决定性：+11.4 mIoU 的差距明显，但消融的参数匹配未说明，且本仓任务（预测/表征）≠本文任务（定位/选择）。可作为**引用性证据**，不可作为**结论性证据**。
- **作为"FiLM 条件化顺序"的证据：TARGET-POSITIVE（限于该用法）** —— "FiLM 必须前置到 CfC、且单独无效"是本文最有可操作性的结论，且是同骨干同配置下的消融，可信度较高。
- **作为本仓边缘部署路线的参考：TARGET-NEGATIVE-WITH-NUANCE** —— 整套"冻结 30B LALM + 逐候选前向"与本仓毫秒级边缘推理的目标**完全相反**；若有人据此主张"LNN 可以做长音频定位"，应指出其推理成本与本仓场景不匹配。
- **作为"LNN 主题饱和"的反证：支持"未饱和"** —— 本文的 CfC 用法（前置条件化 + 候选池）在本仓现有研读中**尚无同类**，说明 LNN 的应用面仍有未覆盖的模式。

> **复现状态**：**未复现**。LAT-Bench 为公开 benchmark，但本文的配置与超参不完整，且无代码。按 AGENTS.md 约束 3 标注为"作者自报、我方未复现"。
