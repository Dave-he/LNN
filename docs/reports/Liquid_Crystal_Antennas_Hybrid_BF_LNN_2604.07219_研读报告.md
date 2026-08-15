---
title: Robust Hybrid Beamforming with Liquid Crystal Antennas and LNN - 研读报告
arxiv_id: 2604.07219v1
date: 2026-04-08 (arXiv v1) / 研读 2026-08-16
tags: [LNN, ODE, hybrid-beamforming, sub-THz, MU-MIMO, 6G, liquid-crystal-antenna, manifold-optimization, robustness, paper-report]
parent: [[LNN_深度研读报告]]
---

# 论文研读报告 — Robust Hybrid Beamforming with Liquid Crystal Antennas and Liquid Neural Networks

> arXiv:2604.07219v1 (cs.IT, 2026-04-08)
> 作者: Xinquan Wang, Mingjun Ying, Hongren Chen, Guanyue Qian, Xingchen Liu, Peijie Ma, Dipankar Shakya, Christos Argyropoulos, Theodore S. Rappaport (NYU WIRELESS + Penn State)
> 来源: [[docs/daily/2026-08-16_LNN_research_digest.md|2026-08-16 每日追踪]] (digest 截断外, 由 cron 手工补入选)
> 注: 录用至 IEEE VTC2026-Spring (Nice, 2026-06), NYU WIRELESS 工业 affiliate 项目, 与该组前序工作 [F. Zhu et al. GLOBECOM 2024] 同脉络

## 1. 元数据

- **标题**: Robust Hybrid Beamforming with Liquid Crystal Antennas and Liquid Neural Networks
- **作者**: Xinquan Wang, Mingjun Ying, Hongren Chen, Guanyue Qian, Xingchen Liu, Peijie Ma, Dipankar Shakya, Christos Argyropoulos, Theodore S. Rappaport (NYU WIRELESS, NYU Tandon; Pennsylvania State University)
- **发表**: arXiv:2604.07219v1, 2026-04-08; 录用至 IEEE VTC2026-Spring (Nice, France, 2026-06), pp. 1–6
- **资助**: NYU WIRELESS Industrial Affiliates Program, NYU Tandon ECE PhD Fellowship, NSF Grant 2234123
- **PDF**: `papers/daily/2604.07219v1_Liquid_Crystal_Antennas_LNN.pdf` (6 页, ~1.0 MB)
- **代码**: 论文未提供 (cron 未抓取)
- **关键词**: Sub-THz, 6G, hybrid beamforming, liquid crystal antenna, LNN, ODE, manifold optimization, MU-MIMO, NYURay, channel estimation
- **场景**: Brooklyn MetroTech Commons, NYC; 108 GHz 载频; 1.8 km × 1.2 km 覆盖区; BS 高度 20 m, UE 高度 1.5 m
- **仿真硬件**: NVIDIA RTX 4090, PyTorch 2.5.1

## 2. 核心问题

sub-THz (≥100 GHz) 6G 通信有两大根本障碍:

1. **硬件瓶颈**: 100 GHz 以上低损耗 per-element phase shifter 难以制造, 传统相控阵需要相位控制网络 (有损/昂贵), lens / metasurface 天线码本粗或校准开销大。
2. **信道估计瓶颈**: 高频 Doppler spread 大 → 信道相干时间短 → 估计精度下降; 同时 sub-THz 对 UE 位置和遮挡极度敏感。

由此产生两难: **既能 sub-THz 频率规模化波束赋形, 又能鲁棒应对不完美信道估计**。

论文贡献拆为 3 件互补的事:
- **硬件方案**: 用电压调谐的 **Liquid Crystal (LC) 天线**做模拟 BF — 48 单元线性阵列, GT3-23001 LC 材料, 105-108 GHz 频段, 单元素 6.87 dB 增益, 5° 波束宽度, 无需相位偏移器
- **算法方案**: 用 ODE-based **Liquid Neural Network (LNN)** 做数字 BF, 通过 **manifold optimization** 把搜索空间从 $\mathbb{C}^{M \times K}$ 压缩到 $\mathbb{C}^{N \times K}$
- **场景验证**: 不是统计信道模型, 而是 **NYURay** ray-tracing 模拟器 (在 142 GHz 实测校准过) 在 108 GHz Brooklyn 城市场景生成 site-specific 信道

论文的核心诊断: **离散时间架构 (GRU) 把 channel snapshot 当独立样本处理**, 而 sub-THz 信道是连续时间演化过程 (UE 移动 + 环境动态), ODE-based LNN 提供**自然归纳偏置**。

## 3. 方法论与核心思路

### 3.1 系统模型 (Section II)

下行 sub-THz MU-MIMO: BS 端 $M=48$ 个 LC 天线单元, 服务 $K=4$ 个 UE, 每个 UE $N_k=4$ 接收天线 ($M \gg N_k$)。

两阶段混合 BF:
- **模拟阶段 (RF 域)**: 从 $n_p = 19$ 个预优化 radiation pattern codebook 中选 $p$, 范围 $-45° \sim +45°$, 5° 步长
- **数字阶段 (基带)**: LNN 输出 base matrix $X \in \mathbb{C}^{N \times K}$, 通过 manifold projection $W = \hat{H}^H X$ 得到 precoder

优化目标 (Eq. 3-5):
$$
\max_{W, p} R = \sum_{k=1}^K \log_2 \det(I + \gamma_k)
$$
$$
\text{s.t. } \text{Tr}(WW^H) \le P, \quad p \in \{1, \dots, n_p\}
$$

### 3.2 LC 天线 (Section III-A)

每个 LC 单元 (Fig. 1):
- 48 单元线性阵列
- LC 材料: GT3-23001 (厚度 12 μm)
- 通过偏置电压控制 LC 分子取向 → 改变 effective permittivity → 改变辐射方向
- Holographic BF 合成 19 个 radiation pattern codebook
- 单元素 6.87 dB 增益, 90° 视场, 5° 波束宽度

### 3.3 LNN (Section III-B)

论文使用的 LNN (Eq. 6-9) 推导自标准一阶 ODE:

$$
\frac{dx(t)}{dt} = -(1 + f(i(t))) x(t) + a f(i(t))
$$

积分因子法解得 (Eq. 7):
$$
x(t) = (x(0) - a) \odot e^{-o_\tau t - \int_0^t f(i(s), \theta_f) ds} + a
$$

**简化**: 移除积分项 (论文称 follow Hasani 2022), 用 sigmoid 替换指数衰减 (因 $e^{-\alpha t}$ 收敛到 bias $a$ 太快):

$$
x(t) = \sigma(-f(x, i; \theta_f) t) \odot g(x, i; \theta_g) + (1 - \sigma(-f(x, i; \theta_f) t)) \odot h(x, i; \theta_h)
$$

其中 $g, h$ 是 learnable NN heads, $f$ 也是 NN head, 共同实现**输入依赖的连续时间动力学**。

### 3.4 Manifold Optimization (Section III-C)

核心观察: massive MIMO 中 $M \gg N$, 因此最优 precoder $W^*$ 必然落在 estimated channel $\hat{H}$ 的 row space 内:

$$
W = \hat{H}^H X, \quad X \in \mathbb{C}^{N \times K}
$$

**搜索空间从 $\mathbb{C}^{M \times K}$ 压到 $\mathbb{C}^{N \times K}$** — 在 $M=48, N=4$ 时压缩 12×。这是论文复用 LNN 的关键: LNN 只需要输出小矩阵 $X$, 不直接输出大矩阵 $W$。

### 3.5 前向 / 反向传播

**前向** (Eq. 10):
- 输入: 归一化 $\hat{H}_n = \hat{H} / \sigma$ (防止 underflow)
- 3 层全连接 liquid neurons
- 输出: $X = \text{LNN}(\hat{H}_n)$
- 投影: $W = \sqrt{P / \text{Tr}(\hat{H}^H X (\hat{H}^H X)^H)} \cdot \hat{H}^H X$

**反向**: 用 **log loss**:
$$
L = -\sum_{k=1}^K \log(\max(\epsilon, R_k))
$$

log 损失强制**公平 SE 分配**给所有用户, 防止 LNN 退化为只服务最强信道的单用户方案。Adam 优化, $\alpha = 0.01$。

## 4. 核心公式 (LaTeX)

**LNN 闭式连续时间单元** (Eq. 9):
$$
x(t) = \sigma(-f(x, i; \theta_f) t) \odot g(x, i; \theta_g) + (1 - \sigma(-f(x, i; \theta_f) t)) \odot h(x, i; \theta_h)
$$

**Manifold Projection 约束** (Section III-C):
$$
W = \hat{H}^H X, \quad X \in \mathbb{C}^{N \times K}
$$

**最终 precoder (含功率归一化)** (Eq. 10):
$$
W = \sqrt{\frac{P}{\text{Tr}(\hat{H}^H X (\hat{H}^H X)^H)}} \cdot \hat{H}^H X
$$

**Log loss for fair SE distribution**:
$$
L = -\sum_{k=1}^K \log(\max(\epsilon, R_k))
$$

## 5. 关键成果与贡献

### 5.1 主结果 (Fig. 3, CEE = -10 dB, P = 30 dBm)

| 方法 | SE (bps/Hz) | vs 3GPP 基线 |
|---|---|---|
| **LNN (LC)** | **~8.8** | 1.9× |
| GRU (LC) | ~5.5 | — |
| LAGD (LC) | ~5.4 | — |
| LNN (3GPP) | ~4.6 | 1× |
| GRU (3GPP) | ~3.5 | — |
| LAGD (3GPP) | ~3.0 | — |

**主要增益**:
- **LNN vs LAGD**: **+88.6% SE** (CEE = -10 dB, P = 30 dBm)
- **LC vs 3GPP**: **1.9× SE** (硬件层)
- **LNN + LC**: 双重正增益

### 5.2 鲁棒性 (Fig. 4, P = 30 dBm, CEE 从 -20 dB 到 0 dB)

| 方法 | SE @ CEE=-20dB | SE @ CEE=0dB | 衰减 |
|---|---|---|---|
| **LNN (LC)** | 8.8 | 6.0 | **-31.7%** |
| LAGD (LC) | 5.4 | 2.4 | **-55.4%** |

**核心结论**: 在信道估计极度不完美时 (CEE 0 dB, 即噪声功率 = 信号功率), LNN 性能衰减速度只有 LAGD 的一半 — 这是 LNN 真正落地的关键证据。

### 5.3 鲁棒性的两个机制 (Section IV-C)

1. **Sigmoid gating in Eq. 9**: 隐式约束 hidden state 更新幅度, 防止 channel 估计噪声通过 LNN 放大 — 相比 LAGD 的迭代梯度 (噪声累积放大)
2. **Manifold projection $W = \hat{H}^H X$**: 强制 precoder 留在 channel row space 内, 隐式正则化, 防止过拟合到估计噪声

### 5.4 与该组前序工作的联系

- Ref [36]: F. Zhu et al. GLOBECOM 2024 — "Robust continuous-time beam tracking with liquid neural network" (LNN 第一次用于 beam tracking)
- Ref [37]: X. Wang et al. IEEE WCL 2024 — "Robust beamforming with gradient-based liquid neural network" (LNN + gradient-based manifold meta learning)
- Ref [39]: F. Zhu et al. ZTE Communications 2025 — 综述性文章

**论文明确把 LNN 推到 sub-THz + 真实 LC 硬件 + 大规模 ray-tracing 验证** — 这是 LNN 在 6G 物理层落地的关键工程拼图。

## 6. 局限性与未来展望

### 6.1 作者明确承认的局限

1. **未做 per-user SE 方差分析 + SE 累积分布 (CDF)** — 仅报 average SE, 没有用户公平性的统计验证。明确写为 "deferred to future work"。
2. **LC 与 3GPP 天线对比未控制 aperture 与 element count** — 1.9× SE 增益可能部分来自硬件差异而非纯 LC 优势。明确写为 "left to future work"。
3. **缺乏 field measurement 验证** — 仅基于 NYURay 模拟 (虽然用 142 GHz 实测校准过 ray-tracing)。
4. **K=4 用户固定** — 不同 user count 下的 scalability 未知。
5. **LNN 是 3 层全连接, 不是真正的深层或 NCP 稀疏连接** — 与 Hasani 原版 Liquid Time-Constant Network 略有简化 (Eq. 9 直接用 sigmoid, 没有 ODE 求解器)。
6. **仿真仅在 1 个 urban 场景 (Brooklyn MetroTech Commons)** — 室内、工厂、室外开阔地等场景未验证。

### 6.2 作者提到的未来工作

- 改变 user count 与 antenna count 的 scalability 分析
- Per-user SE 分布分析 (公平性)
- LC 天线的 field measurement 验证 (与 [17][18] 已有天线集成)

### 6.3 对 LNN / 6G 研究的隐含启发

- **LNN 在 6G 物理层的角色**: 不是替代信号处理专家, 而是**鲁棒性增强器** — 把 sigmoid gating 的隐式 boundedness 转化为对不完美 CSI 的容忍度, 这与 CF (cross-form continuous-time) 在控制 / 视频 / 音频中的"参数高效 + 鲁棒"定位完全一致。
- **Manifold optimization + LNN**: 这是个值得关注的组合范式 — 把 LNN 的输出空间压到物理意义清晰的子流形, 减少搜索难度同时保留学习能力。可推广到 RIS、波束赋形、信道估计。
- **log loss for multi-user fairness**: 把通信领域的多用户公平问题转换为 LNN 训练损失, 比单独 post-processing 阈值更优雅, 也可推广到 multi-task learning。
- **真实硬件 + 真实 ray-tracing 验证**: 论文最强的贡献是把 LNN 从 benchmark 推到工程现实, 与本仓库 `analysis/jetson/` 边缘推理验证的工作异曲同工 — LNN 在 6G 物理层落地的可复现模板。