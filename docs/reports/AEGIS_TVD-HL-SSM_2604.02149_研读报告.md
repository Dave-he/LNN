---
title: "AEGIS: Adversarial Entropy-Guided Immune System — Thermodynamic State Space Models for Zero-Day Network Evasion Detection"
arxiv_id: "2604.02149v1"
authors: "Vickson Ferrel"
date_published: "2026-04-02"
date_reported: "2026-06-06"
tags: [LNN, LTC, Liquid-State-Space, Hyperbolic, Poincaré, SSM, Mamba, Network-Security, Adversarial-Detection, Encrypted-Traffic]
---

# AEGIS: Adversarial Entropy-Guided Immune System — 研读报告

## 1. 元数据

- **论文标题**：AEGIS: Adversarial Entropy-Guided Immune System — Thermodynamic State Space Models for Zero-Day Network Evasion Detection
- **arXiv ID**：[2604.02149v1](https://arxiv.org/abs/2604.02149v1)
- **作者**：Vickson Ferrel (Universiti Malaysia Sarawak; Vixero Technology Enterprise)
- **发表时间**：2026-04-02
- **核心类别**：cs.CR, cs.LG
- **页数 / 图表**：10 页, 3 图 3 表
- **标签**：`LNN` `LTC` `Liquid-State-Space` `Hyperbolic` `Poincaré` `Mamba-3` `eBPF` `Zero-Day` `Encrypted-Traffic`

## 2. 核心问题

TLS 1.3 大规模部署后，传统 Deep Packet Inspection (DPI) 失效，业界普遍转向以 ET-BERT 为代表的**欧氏 Transformer 内容阅读器**对加密流量做分类。然而，此类模型存在三重复合结构脆弱性：

1. **字节级对抗形态变异（Adversarial Pre-Padding）**：Jing et al. 证明对 ET-BERT 注入前缀随机字节噪声后，准确率从 >99% 暴跌至 **25.68%**。
2. **加密拟态（Cryptographic Mimicry）**：VLESS Reality 等协议可动态伪造合法 TLS 1.3 WebSocket 证书，证书层检测完全失效。
3. **意图式分类的内置缺陷**：AMOI 等对抗框架可将恶意 C2 流量锚定到良性分布的"胖中部（Fat Middle）"，使"商业 VPN vs 恶意混淆"这种意图分辨失去意义。

因此，**核心问题**被作者重新定义为：

> 当一个**灰盒-白盒对手**同时具备内容操控、加密拟态、流量-时序形态变异、以及部署零日协议的能力时，能否在**严格的零信任范式**下，对所有持续性加密代理 / 混淆隧道做出一致可解释的异常检测，并同时抵御 Manifold Shattering（流形粉碎）这种白盒优化攻击？

## 3. 方法论与核心思路

作者提出 **AEGIS = TVD-HL-SSM**（Thermodynamic Variance-Guided Hyperbolic Liquid State Space Model），把"读字节"的范式彻底替换为"读流物理量"的范式，骨架由四块构成：

### 3.1 6 维流物理量提取（完全抛弃 payload）

对每条流 $F$，提取 1,000 包因果窗口内的 6 维连续时序向量：
$$
x_i = [S_i, \Delta t_i, D_i, W_i, F_i, P_i] \in \mathbb{R}^{6}
$$

- $S_i$：包体大小
- $\Delta t_i = T_i - T_{i-1}$：**微秒级 IAT**（核心热力学指标）
- $D_i \in \{-1, +1\}$：出/入方向
- $W_i$：TCP 接收窗
- $F_i = \texttt{flags}/255$：协议状态位连续化
- $P_i$：载荷比（直接捕获对抗注入的形态开销）

> **关键设计**：彻底丢弃 payload，使"对抗前缀注入"对特征空间毫无作用。

### 3.2 双曲 Poincaré 流形投影

把欧氏 $x_i$ 映射到 Poincaré 圆盘 $\mathbb{D}^n$：
$$
\phi(x_i) = \frac{W_p x_i}{1 + \|W_p x_i\| + \epsilon}
$$

两点距离用等距不变量 Riemann 度量：
$$
d_c(u, v) = \text{arcosh}\!\left(1 + \frac{2\|u-v\|^2}{(1-\|u\|^2)(1-\|v\|^2)}\right)
$$

**为什么非欧**：僵尸网络拓扑与混淆路由呈指数级层级分支，欧氏空间嵌入会引发**结构失真 + 梯度爆炸**；双曲流形天然适配这种 hierarchical exponential branching。

### 3.3 Liquid Time-Constants（LTC）建模 IAT 衰减

状态 $h(t)$ 满足连续时间 ODE：
$$
\frac{dh(t)}{dt} = -\frac{h(t)}{\tau(\Delta t_i)} + f(h(t), x(t), t, \theta)
$$

时间常数本身被参数化为 IAT 的函数：
$$
\tau(\Delta t_i) = \text{softplus}(\tau_\theta) \cdot \exp(-\Delta t_i / \tau_\theta) + \epsilon
$$

**热力学直觉**：自动化代理（Xray-core 等）无法完美模拟人类驱动 IAT 分布的随机性；LTC 让神经状态衰减率与包间时隙成比例，从而**结构上**捕获到无状态代理合成器无法伪造的方差。

### 3.4 Mamba-3 线性 SSM 核（O(N) 取代 O(N²)）

为支撑 1,000 包窗口，用 Zero-Order Hold 离散化选择式 SSM：
$$
\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B
$$
$$
h_k = \bar{A} h_{k-1} + \bar{B} x_k, \quad y_k = C h_k
$$

通过 TileLang JIT 编译，在 RTX 4090 上达到 64,000 包批并行，**理论算力上限 40 Mpps**。

### 3.5 热力学方差检测（TVD，反 Manifold Shattering）

对手可以优化扰动让恶意时序分布与良性"对齐"（Manifold Shattering），绕过普通 Liquid SSM。为此 AEGIS 在隐藏态序列上计算 Shannon 熵：
$$
P(x_i) = \frac{\exp(h_i)}{\sum_{j=1}^{N} \exp(h_j)}, \quad
H(X) = -\sum_{i=1}^{N} P(x_i) \log_2 P(x_i)
$$

判定：若 $H(X) < \mathbb{E}[H_{\text{benign}}] - \tau_{\text{threshold}}$ 则标记为 **Class-1 热力学异常**。并以辅助损失压住变异：
$$
\mathcal{L}_{\text{thermo}} = \lambda \cdot \left(\mathbb{E}[H(X_{\text{benign}})] - H(X)\right), \quad \lambda = 0.1
$$

良性流量展现**自然随机**熵；自动化零日渗漏则展现**刚性结构**熵。

### 3.6 零拷贝 IPC + eBPF 内核旁路

- **C++ Harvester**：用 libbpf + XDP 在网卡层拦截，DMA 写入 `/dev/shm`
- **PyTorch Executioner**：`torch.frombuffer()` 直接 mmap 物理页，**绕过 Python GIL**
- 64,000 包 swarms 跨 PCIe 异步消费，**I/O 完全被推理掩盖**

## 4. 核心公式（LaTeX 汇总）

| # | 公式 | 含义 |
|---|---|---|
| 1 | $x_i = [S_i, \Delta t_i, D_i, W_i, F_i, P_i]$ | 6 维流物理量 |
| 2 | $\hat{x} = \frac{\log(x+1) - \mu_{\log}}{\sigma_{\log} + \epsilon}$ | log-Z-score 归一化 |
| 3 | $\phi(x_i) = \frac{W_p x_i}{1 + \|W_p x_i\| + \epsilon}$ | Poincaré 投影 |
| 4 | $d_c(u,v) = \text{arcosh}\!\left(1 + \frac{2\|u-v\|^2}{(1-\|u\|^2)(1-\|v\|^2)}\right)$ | 双曲距离 |
| 5 | $\frac{dh}{dt} = -\frac{h}{\tau(\Delta t_i)} + f(h, x, t, \theta)$ | LTC 连续时间动力学 |
| 6 | $\tau(\Delta t_i) = \text{softplus}(\tau_\theta) \exp(-\Delta t_i/\tau_\theta) + \epsilon$ | IAT 驱动的时变时间常数 |
| 7 | $\bar{A} = e^{\Delta A},\ \bar{B} = (\Delta A)^{-1}(e^{\Delta A}-I)\Delta B$ | ZOH 离散化 |
| 8 | $H(X) = -\sum_i P(x_i) \log_2 P(x_i)$ | 序列级 Shannon 熵 |
| 9 | $\mathcal{L}_{\text{thermo}} = \lambda\big(\mathbb{E}[H_{\text{benign}}] - H(X)\big)$ | 熵辅助训练损失 |
| 10 | $\mathrm{FL}(p_t) = -\alpha(1-p_t)^\gamma \log p_t$ | Focal Loss（$\gamma=2,\alpha=0.75$） |

## 5. 关键成果与贡献

### 5.1 性能（在 4 层对抗语料上）

| 指标 | 数值 |
|---|---|
| F1-score | **0.9952** |
| True Positive Rate | **99.50%** (57,551 / 57,838) |
| False Positive Rate | 0.2141% (265 / 良性集) |
| 单条推理延迟 | **262.27 µs**（RTX 4090） |
| 批量算力上限 | 40 Mpps（理论） |
| 训练收敛点 | Epoch 10, val loss 0.0052 |
| 序列窗口 | N = 1,000 包 / 6 维 |
| 训练总序列 | 908,037 条（400 GB 4 层语料） |
| 熵阈值 $\tau_{\text{threshold}}$ | 0.12（动态标定） |

### 5.2 范式级贡献

1. **TVD-HL-SSM 架构**：首次将 **Poincaré 双曲嵌入 + LTC + Mamba-3 选择式 SSM** 串成端到端流水线，并通过 TileLang JIT 编译，规避 O(N²) Transformer 的显存瓶颈。
2. **热力学熵检测**：给出对抗 Manifold Shattering 的解析解——把"流形对齐"转化为"熵刚性"，并以可微辅助损失直接注入训练。
3. **零信任对抗语料**：400 GB / 4 层（跨太平洋骨干 + IoT 僵尸 + 零日 rootkit + VLESS Reality），明示"无法复现 F1"以保证 Tier IV 闭环。
4. **硬件感知部署范式**：C++ eBPF XDP Harvester + 零拷贝共享内存 + torch.frombuffer 跨进程映射，证明 LSTM/CfC 范式之外的**液态架构可以行速率落地**。

### 5.3 与既有范式对比（论文 Table I）

| 架构 | 特征空间 | 抗对抗前缀 | 抗时序变异 |
|---|---|---|---|
| ET-BERT | 欧氏字节 | **脆弱** | N/A |
| 传统 SSM | 欧氏时间 | 免疫 | **脆弱** |
| **AEGIS** | **双曲流物理** | 免疫 | **抗（熵机制）** |

## 6. 局限性与未来展望

### 6.1 作者承认 / 可推断的局限

1. **Tier IV 数据不公开**：精确 F1-score 不可第三方复现，仅放出 Tier I-III + 架构流水，限制了横向 benchmark 的严谨性。
2. **0.2141% FPR 在超大骨干流量上仍可能引发告警疲劳**：阈值 $\tau_{\text{threshold}}=0.12$ 需动态校准，跨运营商、跨地域的可迁移性未充分论证。
3. **N=1,000 窗口与 64,000 包 swarms 的双尺度耦合**：缓冲延迟对真实 line-rate 部署的影响未在多机 / 多 NIC 环境下做端到端评测。
4. **AMOI / Ayaka AH-MSI 是作者内部框架**：白盒攻击强度尚未经过第三方独立复现，"Manifold Shattering 99.5% 绕过率"这一前置结论的外部可验证性偏弱。
5. **LTC 的 $\tau$ 演化**仅在 1,000 包窗口内被检查，**长时间漂移（小时级）** 的热力学异常识别能力未明确。
6. **半监督 / 自监督潜能未展开**：对抗零日仍需部分标注，语料扩张成本高。

### 6.2 未来展望（基于论文 + 行业趋势外推）

- **多模态液态审计**：把 6 维流物理量扩展到 DPI 安全 + 用户行为 + 应用层语义（HTTP/3, QUIC）多模态融合。
- **自适应 τ 阈值**：用元学习（meta-learning）让 $\tau_{\text{threshold}}$ 在不同网络基线分布下自动迁移。
- **联邦零信任协同**：跨 ISP / 跨云的 LTC 联邦学习，共享热力学统计量而不出原始流。
- **Jetson / 边缘移植**：在 ARM + NPU 上用量化 LTC + 小型 Mamba 替代 Mamba-3，部署在 5G UPF 与卫星地面站。
- **可解释性**：把 $H(X)$ 曲线、$\tau(\Delta t_i)$ 演化可视化，作为 SOC 分析师的取证工具。

## 7. 对 LNN 知识库的桥接价值

AEGIS 是 LNN 在**网络安全 / 零信任基础设施**中的**非典型但强有力**的应用样本：

- 首次把 **LTC（[Hasani et al., 2021]）+ Hyperbolic SSM** 在**真实对抗**场景下并联；
- 给出"流物理 + 双曲几何 + 熵正则"的**复合抗对抗模板**，可平移到金融反欺诈、卫星流量异常检测、IoT 僵尸网络早期预警；
- "零拷贝 IPC + eBPF" 的部署经验，与本项目 Jetson 边缘部署（`scripts/jetson_lnn_benchmark.py`）的工程哲学高度同构，值得后续在边缘 LNN 端侧推理中复用。

## 8. 引用与参考

- AEGIS 原文：arXiv:2604.02149v1
- 关键参考：[8] Hasani et al., *Liquid Time-Constant Networks*；[9] Mamba-3 选择式 SSM；[4] Jing et al., *Adversarial Pre-Padding*；[6] AMOI / Ayaka AH-MSI；[18] Lin et al., *Focal Loss*；[19] Loshchilov & Hutter, *AdamW*。
