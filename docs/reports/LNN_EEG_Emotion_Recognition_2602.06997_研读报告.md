---
title: Adaptive Temporal Dynamics for Personalized Emotion Recognition - LNN 研读报告
arxiv_id: 2602.06997v1
date: 2026-01-28 (arXiv v1) / 研读 2026-08-16
tags: [LNN, LTC, Liquid-Time-Constant, EEG, emotion-recognition, multimodal, PhyMER, attention, autoencoder, paper-report]
parent: [[LNN_深度研读报告]]
---

# 论文研读报告 — Adaptive Temporal Dynamics for Personalized Emotion Recognition: A Liquid Neural Network Approach

> arXiv:2602.06997v1 (eess.SP, 2026-01-28)
> 作者: Anindya Bhattacharjee, Nittya Ananda Biswas, K. A. Shahriar, Adib Rahman (Bangladesh University of Engineering and Technology)
> 来源: [[docs/daily/2026-08-16_LNN_research_digest.md|2026-08-16 每日追踪]] (digest 截断外, 由 cron 手工补入选)
> 注: 标题虽不含 "liquid neural" 触发串 (实际是 "Liquid Neural Network Approach"), 但摘要明确 LTC + learnable time constants 强关键词, 7 类多模态情感识别 SOTA

## 1. 元数据

- **标题**: Adaptive Temporal Dynamics for Personalized Emotion Recognition: A Liquid Neural Network Approach
- **作者**: Anindya Bhattacharjee, Nittya Ananda Biswas, K. A. Shahriar, Adib Rahman (BUET, Department of EEE)
- **发表**: arXiv:2602.06997v1, 2026-01-28
- **PDF**: `papers/daily/2602.06997v1_LNN_EEG_Emotion.pdf` (16 页, ~1.1 MB)
- **代码**: 论文未提供 repo URL (cron 未抓取)
- **关键词**: Liquid Neural Network (LNN), Liquid Time-Constant (LTC), EEG, emotion recognition, PhyMER, multimodal, attention, autoencoder, fusion, personality
- **领域**: 情感计算 (Affective Computing) / 多模态生理信号
- **任务**: 7 类离散情绪分类 (angry / disgust / fear / happy / neutral / sad / surprise)
- **数据集**: PhyMER (30 受试者, EEG 256 Hz + EDA 4 Hz + BVP 64 Hz + 皮温 4 Hz)

## 2. 核心问题

基于生理信号的情感识别长期受四个问题困扰:

1. **非平稳 + 噪声 + 个体差异** — 同一情绪在不同受试者身上生理反应不同 (personality-aware modeling 是关键但被多数 benchmark 忽视)。
2. **多模态信号采样率差异** — EEG 256 Hz, EDA 4 Hz, BVP 64 Hz, 皮温 4 Hz — 现有 LSTM/GRU 把时间当作固定步长, 多模态融合时无法对齐。
3. **情绪时间尺度异质** — 同一受试者大脑皮层快速反应 (毫秒级 ERP) 与自主神经慢反应 (秒级 HRV/EDA) 同时存在, 单一时间尺度难以建模。
4. **7 类离散情绪识别性能崩塌** — DEAP/DREAMER 等数据集主流实验只做二分类 (Valence/Arousal) 或三分类 (Positive/Negative/Neutral), 准确率被高估; 7 类任务真实性能显著下降。

作者声明是**首次全面应用 LNN 到 EEG 情感识别**的工作, 核心论点: **LNN 的可学习时间常数 $\tau$ 同时为 EEG 快速瞬态 (ERP) 和自主神经慢动态提供"多时间尺度"的内置机制**, 比手工设计 multi-scale LSTM 更具生物学合理性。

## 3. 方法论与核心思路

### 3.1 总体架构 (Section II)

四路并行的多模态编码 → 自动编码器融合瓶颈 → 分类头:

1. **Raw EEG (14 通道 × 256 时间点)** → 3 层 1D CNN (filter [48, 64, 48], stride 2, MaxPool + Dropout) → 输出 32 时间步 × 32 维 → **LTC 多层网络 (1 层, hidden=128)** → self-attention 聚合
2. **EEG 派生特征**: PSD (5 频带)、DE (Differential Entropy)、Stats (4 阶统计矩 × 5 频带 × 14 通道 = 280 维/时间窗)、FAA (3 对额叶 alpha 不对称)
3. **外周生理**: HRV (RMSSD 等 7 维)、EDA (SCR count, mean amp 8 维)、HR 统计 (7 维)、皮温统计 (6 维)
4. **Personality**: Big Five (5 维)
5. 各模态 MLP → 312 维拼接 → **Autoencoder (312 → 128 → 312, 带 reconstruction loss)** → 128 维潜空间 → **分类 MLP (128 → 256 → 128 → 7)**
6. **联合损失**: $L = L_{CE} + \lambda(e) L_{recon}$, 其中 $\lambda(e) = \lambda_0(1 - e/E)$ 退火, $\lambda_0 = 0.001$ 退火到 0

### 3.2 LTC 神经元 (Eq. 6-8)

论文采用**离散化指数积分**而非连续 ODE 求解:

$$
\tau \frac{dh(t)}{dt} = -h(t) + \sigma(W_x x(t) + W_h h(t) + b)
$$

指数积分闭式解:
$$
d = \exp\left(-\frac{\Delta t}{\tau}\right)
$$
$$
h_t = d \odot h_{t-1} + (1 - d) \odot \tanh(W_x x_t + W_h h_{t-1} + b)
$$

**关键细节**: $\tau = \exp(\theta_\tau)$ 用 log-space 参数化保证正, $\theta_\tau$ 初始化为 $\mathcal{U}[\log 0.1, \log 10]$ — 对应物理时间常数 0.1-10 秒。这个范围匹配 EEG-ERP (毫秒) 到 HRV 自主神经调节 (秒) 的多尺度动态。

### 3.3 自动编码器融合 (Eq. 14-16)

融合模块不是简单 attention 或 concat, 而是 **bottleneck autoencoder**:

$$
z = \phi_{en}(F_{fused}) = \text{ReLU}(W_e \cdot D(\text{ReLU}(W_e \cdot F_{fused} + b_e)) + b_e) \in \mathbb{R}^{128}
$$

$$
\hat{F}_{fused} = \phi_{dec}(z)
$$

重建损失:
$$
L_{recon} = \| F_{fused} - \hat{F}_{fused} \|_2^2
$$

**核心创新点**: bottleneck 强制学习**跨模态互补性**, 防止"模态坍缩" (EEG 总是主导融合结果)。$\lambda(e)$ 退火让早期以重建为主, 后期以分类为主。

### 3.4 时间注意力 (Eq. 10-12)

LNN 输出 $H_{LNN} \in \mathbb{R}^{B \times T' \times d_h}$ → 注意力权重 $\alpha_t$ → 加权池化得 $f_{EEG-raw}$:

$$
e_t = v^T \tanh(W_a h_t^{(L)} + b_a)
$$
$$
\alpha_t = \frac{\exp(e_t)}{\sum_{t'} \exp(e_{t'})}
$$
$$
f_{EEG-raw} = \sum_t \alpha_t h_t^{(L)}
$$

## 4. 核心公式 (LaTeX)

**LTC 神经元离散指数积分** (Eq. 7-8):
$$
d = \exp\left(-\frac{\Delta t}{\tau}\right), \quad h_t = d \odot h_{t-1} + (1 - d) \odot \tanh(W_x x_t + W_h h_{t-1} + b)
$$

**多层 LNN 递归** (Eq. 9):
$$
h_t^{(l)} = d^{(l)} \odot h_{t-1}^{(l)} + (1 - d^{(l)}) \odot \tanh(W_x^{(l)} h_t^{(l-1)} + W_h^{(l)} h_{t-1}^{(l)} + b^{(l)})
$$

**自注意力池化** (Eq. 10-12):
$$
\alpha_t = \frac{\exp(v^T \tanh(W_a h_t + b_a))}{\sum_{t'} \exp(v^T \tanh(W_a h_{t'} + b_a))}
$$

**联合损失** (Eq. 17):
$$
L = L_{CE}(y, y_{true}) + \lambda_0 (1 - e/E) \cdot L_{recon}
$$

## 5. 关键成果与贡献

### 5.1 主指标 (Table II, 5 seed 平均)

7 类情绪 + 全部模态, subject-dependent:
- **Accuracy: 95.45% ± 1.4%**
- Balanced Accuracy: 94.63%
- Macro F1: 93.71%, Weighted F1: 94.89%
- Cohen's κ: 0.9338 (near-perfect agreement)
- Matthews Correlation: 0.9437
- Log Loss: 0.2634 (well-calibrated)

**AUC 指标 (Table III)**: 7 个情绪类全部 AUC ≥ 0.98, micro/macro 平均 0.99 — 远超此前 SOTA (THHSCA 4-class 55.45%, Mifu-ER 7-class 70.24%)。

### 5.2 模态消融 (Table VII, 关键发现)

| 模态组合 | 参数 (K) | 准确率 |
|---|---|---|
| 全模态 (A1) | 432.6 | 95.45% |
| Raw EEG + 5 派生 + Personality (A3) | 396.6 | **96.04%** |
| Raw EEG + DE + Personality (A11) | 306.5 | **96.76%** ⭐ |
| 仅 EEG 派生 (无 raw) (A7) | 291.7 | 74.44% |
| 仅外周 (无 EEG) (A4) | 178.7 | 29.05% |
| Raw EEG (A5) | 274.6 | 86.89% |

**重大发现**:
1. **Personality 的边际效用巨大**: Raw EEG → Raw EEG + Personality, 准确率从 86.89% → **96.35%** (延迟仅增加 0.0003 ms) — 个体差异是真实瓶颈, personality prior 解锁剩余 9 pp。
2. **DE 频域比 raw 更有判别力**: Raw + DE + Personality 比 Raw + 全部 5 EEG 派生还高 0.72 pp。
3. **EEG 是不可替代的**: 移除 EEG 仅留外周信号 → 29.05% (基本失效)。
4. **轻量化**: 仅 306.5K 参数, Latency 0.15 ms/sample (Tesla T4), 1.169 MB — 适合边缘设备。

### 5.3 架构消融 (Table VI, A1-A11)

- **最佳配置 A8**: CNN filter [48,64,48] + 1 层 LNN (hidden=128) + Autoencoder (latent=128) + 无 cross-modal attention → **95.45%**
- **Cross-modal attention 反向有害**: A5 (681K 参数, + attention) → 90.46%; A11 (539K 参数, + attention) → 86.24% — 作者结论: bottleneck autoencoder 已经足够, 多加 attention 导致过拟合。

### 5.4 时序可解释性 (Fig. 5)

LNN 学习到的注意力模式是 **U 形曲线** — 同时关注 EEG 早期瞬态 (ERP) 和晚期持续反应 (慢皮层/HRV), 完美匹配 multi-timescale 假设。统计上 LNN 神经元 $\tau$ 确实**自组织成快/慢双峰分布** (论文承诺但需进一步验证)。

### 5.5 聚类质量 (Table VIII)

训练前 vs 训练后 (t-SNE 2D):
- Calinski-Harabasz: 0.618 → **431.362**
- Davies-Bouldin: 69.216 → **1.745**
- Inter-Centroid Mahalanobis: 1.242 → **213.122**

证明多模态融合学到的表征**确实**聚成 7 类, 不是 collapsed 成一类。

## 6. 局限性与未来展望

### 6.1 作者明确承认的局限 (Section V-A)

1. **超参搜索不充分**: 离散评估 (3 时间常数范围 × 3 学习率 × 3 batch size), 没有 Bayesian/grid 搜索 — 明确写为 limitation, 计划未来扩展。
2. **单次扫描的 hyperparameter 网格**: 不能保证 5×10⁻⁴ lr + batch=64 + τ∈(0.1, 10) 是全局最优。
3. **跨受试者 (subject-independent) 性能未充分评估** — Table VII A2 (无 personality + 5 EEG 派生) 仅 85.16%, 比 subject-dependent 95.45% 低 10 pp, 这是真实部署的主要瓶颈。
4. **PhyMER 仅 30 受试者** — 大规模泛化未知。
5. **模态 attention 反向** (A5/A11) — 作者没解释为什么 autoencoder 已经足够, 还需要更多理论分析。

### 6.2 作者提到的未来工作

- 更系统的超参搜索 (Bayesian optimization)
- 跨受试者 / 跨数据集泛化验证
- 在边缘设备 / Jetson 上量化部署 (与 LNN 边缘研究趋势一致)

### 6.3 对 LNN 研究的隐含启发

- **LNN 在多模态融合中的角色**: 此论文展示 LTC 作为 EEG 时间序列编码器, 而融合交给 autoencoder — LNN 不是万能钥匙, 而是专门负责**时序多尺度建模**的组件。
- **Log-space $\tau$ 参数化**: 是工程关键 — 避免 $\tau$ 退化到负或极大值, 保证物理可解释性。
- **指数积分闭式解 (Eq. 7-8)** 让 LTC 训练速度与 LSTM 持平, 没有 Neural ODE 的积分开销 — 这是把 LNN 推上多模态 SOTA 的关键工程决定。
- **可解释性 $\tau$ 双峰分布** 是 LNN 区别于 LSTM/Transformer 的核心卖点 — 此论文的承诺需要在未来工作中做更严格的统计检验 (如 permutation test 或 ANOVA)。