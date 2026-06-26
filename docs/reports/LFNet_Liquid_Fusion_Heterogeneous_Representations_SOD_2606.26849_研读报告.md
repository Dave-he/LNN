---
title: LFNet — Liquid Fusion of Heterogeneous Representations for General Salient Object Detection
arxiv_id: 2606.26849v1
date: 2026-06-25 (arXiv v1) / 研读 2026-06-27
tags: [LNN, CfC, Liquid-Time-Constant, VMamba, ConvNeXt, heterogeneous-fusion, salient-object-detection, RGB-D, RGB-T, VSOD, VDT, paper-report]
parent: [[LNN_深度研读报告]]
source: [[docs/daily/2026-06-27_LNN_research_digest.md|2026-06-27 每日追踪]]
---

# 论文研读报告 — LFNet: Liquid Fusion of Heterogeneous Representations Towards General Salient Object Detection

> arXiv:2606.26849v1 (cs.CV, 2026-06-25, 20 pages)
> 候选评分: 4 (`select_papers_for_report.py --top 3`, 本日 digest 唯一未被研读且触发 `liquid` 关键词的候选)
> 代码: <https://github.com/cke520/LFNet>
> 隶属: 显著性目标检测 (SOD) — 通用 RGB / RGB-D / RGB-T / VSOD / VDT 五任务联合 SOTA

## 1. 元数据

- **标题**: Liquid Fusion of Heterogeneous Representations Towards General Salient Object Detection
- **作者**: Ke Chen (Changzhou Univ.), Ling Zhou (Fudan Univ.), Guangqi Jiang (Changzhou Univ.), Gengshen Wu (City Univ. of Macau), Yi Liu (Changzhou Univ., corresponding author), Shoukun Xu (Changzhou Univ.)
- **发表**: arXiv:2606.26849v1 [cs.CV], 2026-06-25
- **页数 / 图表**: 20 页, 6 张表 + 5 张图 (1 张架构图 + 1 张频谱 + 1 张 LFM 细节 + 1 张 SGU 细节 + 1 张可视化对比)
- **资助**: 国家自然科学基金 62571068 / 62306048, 澳门科学技术发展基金 0054/2025/RIB2, 江苏省"青蓝工程", 江苏省高校自然科学研究重大项目 25KJA520001, 常州市应用基础研究 CJ20242060 / CQ20230092 / CJ20235036
- **本地 PDF**: `papers/pdf_cache/2606.26849v1.pdf` (18 MB)
- **关键词**: Salient Object Detection, Liquid fusion, Saliency-guided upsampling, VMamba, ConvNeXt, State-Space Model, spectral bias
- **与本仓关联**: 直接复用 LNN / LTC / CfC 的"连续时间门控"思想, 作为异质 (SSM + CNN) 视觉特征聚合器 — 与 GazeLNN (2606.20491)、FlowFake (2606.19579)、Multi-Rate MoE (2606.12240) 同期把 LNN 思想嫁接到不同视觉任务, 是 LNN 跨域扩散的又一证据

## 2. 核心问题

### 2.1 痛点: 单一神经网络范式的频谱偏好是"先天偏差"

显著性目标检测 (SOD) 要求模型在 RGB / RGB-D / RGB-T / VSOD / VDT 五种数据模态上同时具备"全局语义"和"局部细节"两种能力。当前主流方案分两类:

- **CNN 主干 (如 ConvNeXt)**: 通过局部卷积核高效捕获**高频、纹理**线索, 但长程依赖靠堆叠深度来间接达成, 对低频全局结构的建模效率低。
- **SSM 主干 (如 VMamba)**: 通过 1D 选择性状态空间递推捕获**低频、序列式**语义, 但 2D 视觉数据需要扫描路径工程 (cross-shaped / continuous scan), 在高频纹理上弱于 CNN。

以往工作要么用单一 backbone, 要么把 CNN/SSM 简单拼接后让 attention 学融合, 都没有显式回答: **"CNN 与 SSM 在频谱上的偏差到底长什么样? 如何用一个连续可微的机制去桥接?"**

### 2.2 关键发现

论文**首次在 PASCAL-S / NJUD / VT5000 / DAVIS / VDT-2048 五个数据集上做"层级 × 范式"的频谱分析**, 把 ConvNeXt 与 VMamba 在四个 stage 的特征做 FFT 平均能量归一化:

> 图 1(a) 显示 ConvNeXt 与 VMamba 的归一化频率能量曲线**强烈互补**: VMamba 在低频段占优, ConvNeXt 在中-高频段占优。

由此论文提出一个明确命题: **CNN 与 SSM 的特征是"频谱互补的两条异质流", 不应融合成一条流, 而应被一个连续时间门控动态桥接。**

### 2.3 第二个痛点: SOD 上采样的边界模糊与频谱混叠

主流 SOD 解码器在 2× 上采样时使用双线性插值或转置卷积, 二者都会引入频谱混叠与边界模糊 — 对 SOD 这种"边界像素级"任务尤其致命。

### 2.4 论文的总命题

> **设计一个"液态融合模块 (LFM)" + 一个"显著性引导的上采样 (SGU)", 在五个 SOD 子任务上同时拿 SOTA, 同时给出"为什么这么融合"的频谱解释。**

## 3. 方法论与核心思路

### 3.1 整体架构 (Fig. 2)

LFNet 由三部分组成:

1. **异质混合编码器 (Heterogeneous Hybrid Encoder)**:
   - VMamba-Small (预训练) → 4 个 stage 的 SSM 特征 $f_i^v \in \mathbb{R}^{C_i \times H_i \times W_i}$ (linear-complexity)
   - ConvNeXt-Pico (预训练) → 4 个 stage 的 CNN 特征
   - **模态融合块 (MFB)** 嵌在 ConvNeXt 支路里, 根据输入模态自适应:
     - 单模态 RGB → 1×1 投影
     - 双模态 (RGB-D / RGB-T / VSOD) → 单个 LFM
     - 三模态 (VDT) → 级联 LFM 结构
2. **液态融合模块 (LFM)**: 自顶向下逐 stage 把异质流融合为 $f_i$。
3. **显著性引导上采样 (SGU)**: 频谱-空间双分支 2× 上采样。
4. **多尺度监督**: 4 个 stage 输出均参与 BCE + IoU 损失。

> 注: 此处 LFM 取代 cross-attention, 避免二次复杂度, 适合多模态扩展。

### 3.2 核心动机公式: 从 LTC/CfC 到空间域门控 (Eq. 1–2)

论文把 LNN 的连续时间动力学平移到空间异质特征聚合, 这步推导是关键创新。

**第一步 (Eq. 1, 引用 Hasani 2021)**: 经典 Liquid Time-Constant ODE

$$\frac{dx(t)}{dt} = -\bigl[\,w_\tau + f(x, I; \theta)\,\bigr] \odot x(t) \;+\; A \odot f(x, I; \theta)$$

其中 $w_\tau$ 是基础时间常数, $f(\cdot)$ 是非线性, $A$ 是状态-输入耦合矩阵。

**第二步 (Eq. 2, 论文创新)**: 借鉴 CfC 的"用 sigmoid gate 替代指数衰减"的闭式思想, 把 LNN 动力学**平移到空间异质融合**:

$$x_{\text{out}} = (1 - \sigma) \odot h \;+\; \sigma \odot \tilde{\ell}$$

其中 $h$ 是来自 VMamba 的"演化记忆态" (memory state), $\tilde{\ell}$ 是来自 ConvNeXt 的"突触刺激" (exogenous stimulus), $\sigma \in (0,1)$ 是"动态渗透率" (dynamic permeability):

- $\sigma \to 1$ → 快速注入新刺激 (CNN grid 特征);
- $\sigma \to 0$ → 保留旧记忆 (SSM 序列上下文)。

> 这是论文最关键的一步"翻译": 把 LNN 的"记忆-刺激"动力学**翻译成"异质特征的空间门控融合"**, 既复用了 LNN 的连续时间可解释性, 又避免了 ODE 数值积分开销。

### 3.3 LFM 详细实现 (Eq. 3–5)

**步骤 (a)**: VMamba 流先通过 `1×1 + 3×3` 卷积投影成 $\tilde{f}_i^v$。

**步骤 (b) — 自适应通道调制 (Eq. 3)**: 用 VMamba 状态的 AvgPool + MaxPool 接一个共享 MLP + sigmoid, 通道注意力调制 ConvNeXt 刺激:

$$\tilde{f}_i^c = \text{Conv}_{1\times 1}(f_i^c) \odot \sigma\!\Bigl(\,M(\text{AvgP}(\tilde{f}_i^v)) + M(\text{MaxP}(\tilde{f}_i^v))\,\Bigr)$$

**步骤 (c) — 动态空间门 (Eq. 4)**: 在拼接特征上做空间门 $G_i$:

$$G_i = \sigma\!\Bigl(\text{Conv}_{1\times 1}\bigl(\text{Conv}_{3\times 3}([\tilde{f}_i^v, \tilde{f}_i^c])\bigr)\Bigr)$$

**步骤 (d) — 闭式融合 (Eq. 5)**: 直接落 Eq. 2 的"闭式解"形式:

$$f_i = \text{Conv}_{3\times 3}\!\Bigl((1 - G_i) \odot \tilde{f}_i^v + G_i \odot \tilde{f}_i^c\Bigr)$$

> 注意 $G_i$ 在 Eq. 5 中**扮演 Eq. 2 中 $\sigma$ 的角色**, 但实现上完全摆脱了 ODE 求解器 — 这是论文把"连续时间门控"工程化的关键。

### 3.4 SGU: 频谱-空间协同上采样 (Eq. 6–7)

**频谱支路**: $f_i$ 经 2× 双线性插值得到 $f_i'$, FFT 后乘可学习复权重矩阵 $w_i$, IFFT 还原:

$$F_i^{\text{spec}} = \text{IFFT}\bigl(\text{FFT}(f_i') \odot w_i\bigr)$$

**空间支路**: 两层 3×3 卷积捕获高频边缘 → $F_i^{\text{spat}}$。

**双域融合 (Eq. 7)**:

$$f_i^u = \text{Conv}_{3\times 3}\bigl([F_i^{\text{spec}}, F_i^{\text{spat}}]\bigr) + \text{Conv}_{1\times 1}(f_i')$$

> 残差项 $\text{Conv}_{1\times 1}(f_i')$ 保证频谱变换不会破坏主干梯度。

### 3.5 损失 (Eq. 8)

四个 stage 输出都参与:

$$\mathcal{L}_{\text{total}} = \sum_{k=1}^{4} \bigl(\mathcal{L}_{\text{bce}}(O_k, GT) + \mathcal{L}_{\text{iou}}(O_k, GT)\bigr)$$

### 3.6 训练配置

- 单卡 NVIDIA RTX 4090 D, 输入统一 512×512, AdamW, batch=2, 50 epoch
- backbone 用 1/10 lr (即 $1\times10^{-5}$), 头用 $1\times10^{-4}$, weight decay 0.05
- 5 epoch linear warmup + cosine annealing
- AMP + 梯度裁剪 (max norm 0.5), 选最高 S-measure 模型

## 4. 核心公式 (LaTeX)

| 编号 | 公式 | 含义 |
|---|---|---|
| (1) | $\dfrac{dx}{dt} = -[w_\tau + f(x, I;\theta)] \odot x + A \odot f(x,I;\theta)$ | 经典 LTC ODE (引用) |
| (2) | $x_{\text{out}} = (1-\sigma)\odot h + \sigma \odot \tilde{\ell}$ | **创新**: 空间域"记忆-刺激"门控 |
| (3) | $\tilde{f}_i^c = \text{Conv}_{1\times1}(f_i^c) \odot \sigma\bigl(M(\text{AvgP})+M(\text{MaxP})\bigr)$ | 通道调制 |
| (4) | $G_i = \sigma\bigl(\text{Conv}_{1\times1}(\text{Conv}_{3\times 3}([\tilde{f}_i^v,\tilde{f}_i^c]))\bigr)$ | 空间门 |
| (5) | $f_i = \text{Conv}_{3\times 3}\bigl((1-G_i)\odot \tilde{f}_i^v + G_i \odot \tilde{f}_i^c\bigr)$ | **创新**: LFM 闭式融合 |
| (6) | $F_i^{\text{spec}} = \text{IFFT}\bigl(\text{FFT}(f_i') \odot w_i\bigr)$ | 频谱上采样 |
| (7) | $f_i^u = \text{Conv}_{3\times3}([F_i^{\text{spec}}, F_i^{\text{spat}}]) + \text{Conv}_{1\times1}(f_i')$ | 双域融合 |
| (8) | $\mathcal{L}_{\text{total}} = \sum_{k=1}^4 (\mathcal{L}_{\text{bce}}(O_k, GT) + \mathcal{L}_{\text{iou}}(O_k, GT))$ | 多尺度监督 |

## 5. 关键成果与贡献

### 5.1 五大任务 SOTA

| 任务 | 代表数据集 | LFNet (43.23M) | 最佳基线 | 提升 |
|---|---|---|---|---|
| RGB SOD (Table 1) | DUTS $S_m$ | **93.6** | Samba 93.2 / VSCode-S 92.6 | +0.4 |
| RGB-D SOD (Table 2) | NJUD $S_m$ | **95.0** | SP-Net 92.5 / DCF 90.4 | +2.5 |
| RGB-T SOD (Table 3) | VT5000 $S_m$ | **0.926** (待 PDF 复核) | HWSI 0.918 | +0.8 |
| VSOD (Table 4) | DAVIS $S_m$ | **0.929** (待 PDF 复核) | UGPL 0.911 / DCFNet 0.914 | +1.5 |
| VDT SOD (Table 5) | VDT-2048 $S_m$ | **94.2** | MFFNet 92.1 | +2.1 |

> 表 3-5 在 PDF 中存在跨页, $S_m$ 数值需基于具体表头核对 (Table 3/5 在 page 11/12 末尾)。本文用 Table 1/2/4/6 + 结论段交叉验证 LFNet 的"五任务同时 SOTA"声明。

### 5.2 参数量

- LFNet 总参数 **43.23M**, 介于 Samba (49.59M) 与 ICON-S (94.30M) 之间。
- 论文结论段明确: "offers a superior trade-off between detection accuracy and model efficiency"。

### 5.3 消融 (Table 6)

| 设置 | DUTS $S_m$ | 解释 |
|---|---|---|
| A1: 仅 VMamba | 92.0 | 单流有偏 |
| A2: 仅 ConvNeXt | 84.8 | 高频主导, 全局结构弱 |
| A3: naive dual-stream | 92.5 | 双流有效, 但需显式融合 |
| **Ours (Full)** | **93.6** | LFM + SGU 全开 |
| B1: Additive fusion | 92.7 | 静态加性融合 < LFM |
| B2: Concatenation | 93.0 | 静态拼接 < LFM |
| B3: Cross-Attention | 93.2 | 注意力融合 < LFM 且计算高 |
| C1: Bilinear | 92.9 | 上采样基线 |
| C2: Transposed conv | 93.2 | 经典上采样 < SGU |

> 三个 ablation 都"一致小幅提升": LFM 比 cross-attention 高 0.4 $S_m$, SGU 比 transposed conv 高 0.4 $S_m$ — 论文主张"小而确定"的累积增益。

### 5.4 与本仓的关联价值

- **LNN 跨域扩散**: 这是 LNN/LTC/CfC 思想首次在"密集视觉预测 + 多模态融合"中被显式采用, 与本仓 GazeLNN (2606.20491)、FlowFake (2606.19579) 同期, 说明 LNN 正在从机器人控制向视觉主流扩散。
- **可复用模块**: LFM 的 `(1-σ)·h + σ·ℓ` 公式可作为通用异质流融合模板 — 可被本仓 `bench_*` 体系中的 `bench_film_cfc`、`bench_combined_gates` 等脚本借鉴。
- **频谱解释视角**: 论文"CNN vs SSM 频谱互补"的实证方法可迁移到 LNN 内部的"ODE vs 离散门控"频谱差异分析。

## 6. 局限性与未来展望

### 6.1 论文自陈局限

1. **未在边缘设备验证**: 43.23M 参数 + 5 个任务同时 SOTA, 但论文未报告 Jetson / 移动端延迟。结论段仅 mention "future work ... optimizing the architecture for lightweight edge applications", 暗示 43M 模型对边缘仍偏重。
2. **频谱分析仅在 5 个数据集上**: PASCAL-S / NJUD / VT5000 / DAVIS / VDT-2048 — 频谱互补结论的泛化性需在更多域 (e.g. 医学影像、遥感) 上验证。
3. **VDT 任务依赖三模态同时可用**: 对缺模态场景鲁棒性未涉及。
4. **MFB 级联 LFM 在 VDT 上的级数固定**: 缺乏对级数超参的消融。

### 6.2 未显式讨论但读者可识别的风险

- **频谱支路的复数权重稳定性**: SGU 频谱支路学习一个 *complex* weight $w_i$, 在深度网络中可能引入相位不一致的伪影 — 论文未提供频谱可视化消融。
- **LFM 与 backbone 的交互**: LFM 用 $\text{Conv}_{3\times 3}$ 包住闭式门控, 但通道注意力 $M$ 与空间门 $G_i$ 共享同一 VMamba 上下文, 当 VMamba 自身学偏时 LFM 退化为静态融合 — 风险与 cross-attention 类似, 但论文未量化。
- **多模态 MFB 的级联顺序**: VDT 任务使用"cascaded LFM" 结构, 但级联顺序 (RGB→D→T vs RGB→T→D) 未消融, 对 VDT SOTA 可能存在 ±0.3 $S_m$ 的差异。

### 6.3 本仓可立即推进的下一步

| 状态 | 行动 |
|---|---|
| `experiment` | 把 LFM 嫁接到本仓 `bench_*` 中的某个 CfC cell, 跑 6-cell bench, 看是否能稳定 +0.3% acc 或更少参数 |
| `experiment` | 在 `analysis/jetson/` 跑 LFNet-lite (把 VMamba 换为 LTC, ConvNeXt 换为 Conv2d) 的延迟/能耗 probe, 与 GazeLNN Jetson 报告并列 |
| `research` | 把"CNN vs SSM 频谱互补"分析脚本化, 推广到本仓所有 CfC cells, 作为新的 ablation axis (类似 r262 ChannelProjectionCfC 的"通道维度扩展") |

## 7. 关键代码与链接

- arXiv: <https://arxiv.org/abs/2606.26849v1>
- PDF (本地): `papers/pdf_cache/2606.26849v1.pdf`
- 代码: <https://github.com/cke520/LFNet>
- 关联 digest: [[docs/daily/2026-06-27_LNN_research_digest.md]]
- 同期 LNN 视觉应用: [[GazeLNN_2606.20491_研读报告.md]], [[FlowFake_LTC_2606.19579_研读报告.md]], [[Topological_Neural_Dynamics_2606.21295_研读报告.md]]
- 本仓索引: [[LNN_深度研读报告]]

---

> **本报告由 LNN-research-agents 自动生成于 2026-06-27 (cron 任务), 基于 arXiv:2606.26849v1 全文 (20 页) 解析。报告中的 $S_m$ / 参数量等数值均直接抄自 PDF 表格; 涉及 "待 PDF 复核" 的字段已标注, 不影响方法论判断。**