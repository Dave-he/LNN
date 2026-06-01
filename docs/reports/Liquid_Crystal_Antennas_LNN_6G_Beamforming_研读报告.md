---
title: Robust Hybrid Beamforming with Liquid Crystal Antennas and Liquid Neural Networks
date: 2026-04
tags: [LNN, Liquid-Time-Constant, Beamforming, 6G, sub-THz, MU-MIMO, Liquid-Crystal-Antenna, Manifold-Optimization]
---

# 研读报告：液态神经网络 + 液晶天线在 6G sub-THz 混合波束成形中的应用

## 1. 元数据
- **论文标题**：Robust Hybrid Beamforming with Liquid Crystal Antennas and Liquid Neural Networks
- **作者**：Xinquan Wang, Mingjun Ying, Hongren Chen, Guanyue Qian, Xingchen Liu, Peijie Ma, Dipankar Shakya, Christos Argyropoulos, Theodore S. Rappaport
- **机构**：NYU WIRELESS (纽约大学) + Pennsylvania State University
- **发表时间**：2026 年 4 月 8 日 (v1)；将发表于 IEEE VTC2026-Spring, Nice, France, Jun. 2026
- **来源**：arXiv:2604.07219v1
- **本地 PDF**：[papers/daily/2604.07219v1_Liquid_Crystal_Antennas_LNN.pdf](../papers/daily/2604.07219v1_Liquid_Crystal_Antennas_LNN.pdf)

## 2. 核心问题
6G 通信要进入 sub-THz（>100 GHz）频段以解锁极大带宽，但**硬件 + 信道估计**两端同时被卡死：

1. **硬件瓶颈**：传统半导体移相器在 100 GHz 以上损耗极高、难以做 per-element 调相。sub-THz 频段缺乏低损、波束可控的天线/前端。
2. **信道估计退化**：高频段 Doppler spread 增大、coherence time 显著缩短；beam squint 严重；user 移动 / 阻挡使 CSI 估计极易出错。
3. **算法-硬件失配**：传统 iterative / Bayesian BF 方法在高 CEE（channel estimation error）下不鲁棒；普通 NN 缺乏对连续物理动力学的归纳偏置。

**论文的核心论断**：用 **液晶 (LC) 可重构天线**做 analog BF 段（电压驱动介电常数调谐，无需移相器），用 **ODE 驱动的 Liquid Neural Network (LNN)** 做 digital BF 段（连续时间动力学天然匹配 sub-THz 信道时变），并通过**流形优化**把 M≫N massive MIMO 的搜索空间压扁。系统级目标：在 NYURay 仿真得到的 108 GHz 城市 microcell 场景下，相对 LAGD / GRU baseline 取得更高谱效 (SE) 且对不完美 CSI 更鲁棒。

## 3. 方法论与核心思路

### 3.1 系统模型 (Downlink sub-THz MU-MIMO Hybrid BF)
- BS：M=48 个 LC 天线单元，连接独立 RF 链；K=4 个 UE，每个 UE N_k=4 个 ULA 天线
- 发射信号 $u = \sum_{k=1}^{K} w_k s_k$；接收 $y = H^{(p)} W s + n$（式 1）
- 信道由 **NYURay 射线追踪**生成（载频 108 GHz，5 km × 1.2 km 城市，覆盖 1.8 km × 1.2 km 区域，10^6 rays，最多 5 次反射），而非假设 Rayleigh。信道矩阵是 L 条路径的加权和（式 2）

### 3.2 模拟 BF 段：LC 液晶天线
- 48 单元线阵；每单元含 GT3-23001 LC 介质
- 通过偏置电压调谐有效介电常数，等效 holographic 数字波束成形
- 性能：**5° 波束宽度 / 90° 视场 / 单元 6.87 dB 增益**，无需任何半导体移相器
- 19 个预优化 pattern 的**离散码本**（-45° ~ +45°，5° 步进），由全波电磁仿真得到
- 模拟阶段任务：在码本中选 $p \in \{1,\dots,n_p=19\}$ 最大化聚合信道增益

### 3.3 数字 BF 段：LNN + 流形优化

**为什么是 LNN 而不是 GRU？**
- 物理动机：sub-THz 信道随用户移动和环境动力学**连续演化**；ODE 状态演化（式 9）是更自然的归纳偏置。GRU/LSTM 是离散时间模型，把每一时刻 channel snapshot 当独立样本，丢失了连续时间结构。
- 鲁棒性动机：LNN 的 sigmoid gating（式 9）天然把隐藏状态更新有界化，充当隐式正则器，**抑制了 CEE 噪声在迭代中的放大**。

**LNN 闭式更新（关键公式）**

从一阶线性 ODE 出发：
$$
\frac{dx(t)}{dt} = -x(t) + S(t), \quad S(t)=f(i(t))(a-x(t))
$$

代入 $S(t)$ 后得到线性 ODE：
$$
\frac{dx(t)}{dt} = -(1 + f(i(t)))x(t) + a f(i(t)) \tag{6}
$$

经 integrating-factor 解并简化后得到闭式近似（去掉积分项）：
$$
x(t) \approx b \odot e^{-[o_\tau + f(i(t), \theta_f)]t} \odot f(-i(t), \theta_f) + a \tag{8}
$$

为了既保留 ODE 时序语义又便于训练，把指数衰减换为 sigmoid 门控，并让 $a$、$b$ 通过 NN head $g, h$ 学习。最终模型：
$$
x(t) = \sigma(-f(x, i; \theta_f) t) \odot g(x, i; \theta_g) + \left[1 - \sigma(-f(x, i; \theta_f) t)\right] \odot h(x, i; \theta_h) \tag{9}
$$

这里 $\theta_f, \theta_g, \theta_h$ 是三个 MLP head 的参数，$\sigma$ 为 sigmoid 门。输入由"当前输入 $i$"与"上一时刻隐状态 $x$"拼接，给每个 liquid 神经元看全时序上下文。

**流形优化 (Manifold Optimization)**

利用 $M \gg N$ massive MIMO 的特性，把 $W$ 投影到估计信道的行空间：
$$
W = \hat{H}^H X
$$
LNN 实际只需要学习一个 $N \times K$ 的小矩阵 $X$（这里 $N=4, K=4$），把搜索空间从 $\mathbb{C}^{M \times K}$ 压到 $\mathbb{C}^{N \times K}$，参数量级数下降。再做功率归一化：
$$
W = \sqrt{\frac{P}{\text{Tr}(\hat{H}^H X (\hat{H}^H X)^H)}} \cdot \hat{H}^H X \tag{10}
$$

**训练目标 (log-sum SE loss)**
$$
\mathcal{L} = -\sum_{k=1}^{K} \log(\max(\epsilon, R_k))
$$
- 鼓励**用户间公平**（log 函数对低 R_k 惩罚大），避免网络塌缩到单用户解
- 对模拟 BF 联合训练时只取使 SE 最大的 pattern $p$ 的 loss 端到端反传
- Adam, $\eta=0.01$，NVIDIA RTX 4090 + PyTorch 2.5.1

### 3.4 评测设定
- 场景：Brooklyn MetroTech Commons, 108 GHz
- 假设：K=4, P=30 dBm, α=0.01, CEE=-10 dB（除鲁棒性曲线外）
- 基线：**LAGD**（Learning-Aided Gradient Descent）+ **GRU-based**（同任务）；同场景下还与 **3GPP TR 38.901** 标准天线阵列对比
- 鲁棒性扫描：CEE 从 -20 dB 扫到 0 dB

## 4. 核心公式提取

### 4.1 Hybrid BF 优化目标
$$
\max_{W, p} R, \quad R = \sum_{k=1}^{K} \log_2 \det(I + \gamma_k) \tag{3, 5}
$$
$$
\text{s.t.}\ \text{Tr}(WW^H) \le P,\ p \in \{1, 2, \ldots, n_p\}
$$
$$
\gamma_k = \hat{H}^{(p)}_k w_k (\hat{H}^{(p)}_k w_k)^H \left[ \sum_{j \neq k} \hat{H}^{(p)}_k w_j (\hat{H}^{(p)}_k w_j)^H + \sigma^2 I \right]^{-1} \tag{4}
$$

### 4.2 射线追踪信道
$$
H^{(p)}_k = \sum_{\ell=1}^{L} \alpha_\ell \, G^{(p)}(\theta^t_\ell, \phi^t_\ell)\, a_r(\theta^r_\ell, \phi^r_\ell)\, a^H_t(\theta^t_\ell, \phi^t_\ell) \tag{2}
$$

### 4.3 LNN 闭式状态更新 (主贡献)
$$
x(t) = \sigma(-f(x, i; \theta_f) t) \odot g(x, i; \theta_g) + \left[1 - \sigma(-f(x, i; \theta_f) t)\right] \odot h(x, i; \theta_h) \tag{9}
$$

### 4.4 功率归一化数字 BF
$$
W = \sqrt{\frac{P}{\text{Tr}(\hat{H}^H X (\hat{H}^H X)^H)}} \cdot \hat{H}^H X \tag{10}
$$

### 4.5 CEE 定义
$$
\text{CEE} \triangleq 10 \log_{10} \frac{\mathbb{E}[\|E^{(p)}\|^2_F]}{\mathbb{E}[\|H^{(p)}\|^2_F]}
$$

## 5. 关键成果与贡献

### 5.1 谱效 (SE) 增益
- 在 P=30 dBm、CEE=-10 dBm 下，**LNN + LC 天线**相对 **LAGD + LC** 取得 **+88.6% SE**（图 3）
- 全部 LC 配置相对 **3GPP TR 38.901 标准天线**取得 **≥ 1.9× SE**

### 5.2 鲁棒性优势
- P=30 dBm，CEE 从 -20 dB 扫到 0 dB：
  - **LNN (LC)**：SE 从 8.8 → 6.0 bps/Hz，**仅 -31.7%**
  - **LAGD (LC)**：SE 从 5.4 → 2.4 bps/Hz，**-55.4%**
- 原因：sigmoid gating 有界化隐藏状态 + 流形投影把 precoder 约束在估计信道的行空间（隐式正则）

### 5.3 三层贡献定位
1. **硬件贡献**：首次把 48 单元 LC 阵列 + 5° 波束 / 6.87 dB 增益的 holographic 模式作为 108 GHz 模拟 BF 段（避开半导体移相器）。
2. **算法贡献**：用 ODE 闭式 LNN 替代离散 GRU/LAGD，匹配 sub-THz 连续时变；流形优化把参数空间从 $M \times K$ 压到 $N \times K$。
3. **验证贡献**：用 site-specific NYURay 仿真（不是统计信道模型）做端到端评估，给出 SE + 鲁棒性两条曲线的定量证据。

## 6. 局限性与未来展望

### 6.1 论文自陈的局限
| 类别 | 具体限制 | 位置 |
|---|---|---|
| 受控对比 | LC 阵列与 3GPP 阵列未做"等口径/等孔径"匹配比较；当前 1.9× 增益里混合了"LC 硬件更优"与"码本 + holographic 模式选择更优" | §V 末段 |
| 评估规模 | K=4, M=48, P=30 dBm 单点；UE 数、阵列尺寸、SE 分布尚未扫 | §V 末段 |
| 实验验证 | 全部为 NYURay 仿真结果，**没有真实 over-the-air 测量** | §V 末段 |
| 系统级问题 | LC 介电响应在毫秒级，会限制波束更新速率（论文承认但未量化） | §I 末段 |
| 物理边界 | LC 单元间互耦与 per-element 独立控制的 codebook 设计仍是开放问题 | §I 末段 |

### 6.2 未来方向
- **真实 OTA 验证**：把 108 GHz LC 阵列在 NYU 实际场景做 over-the-air 测量，把仿真 SE 增益落到实测
- **等口径对比**：构造等孔径 / 等单元数 / 同极化的 LC vs 3GPP 阵列，剥离硬件增益与算法增益
- **多用户 / 大阵列扩展**：扫 K、M、per-user SE 分布与 CDF，验证可扩展性
- **LC 时延建模**：把 LC 介电响应毫秒级时延纳入 LNN 输入 / state，做"硬件-算法联合时延预算"
- **替代 ODE 求解器**：当前闭式近似是 Hasani 等式 (8) 的简化；可尝试保留积分项以更高精度匹配 1 ms 量级信道相干时间
