---
title: LTC-Fall — Physics-Informed Dual-LTC Fall Detection on Low-Power Edge Platforms
date: 2026-07-18
tags: [LNN, LTC, Liquid-Time-Constant, Fall-Detection, Edge-AI, Stability-Manifold, Counterfactual-Inference, Biomechanics, Lyapunov]
---

# 研读报告：LTC-Fall — 基于双 LTC 动力学解耦 + 稳定性流形的视觉跌倒检测（边缘实时）

## 1. 元数据
- **论文标题**：Real-time fall detection based on vision for low-power edge platforms
- **作者**：Wenjun Xia (一作), Zhicheng Peng, Haopeng Li, Zhengdi Zhang (通讯)
- **机构**：Jiangsu University, Zhenjiang, China
- **发表时间**：2026-07-14 (arXiv:2607.12909v1, q-bio.NC / cs.AI / cs.CV)
- **代码**：未公开（论文未给出 GitHub 链接）
- **本地 PDF**：[papers/daily/2607.12909v1.pdf](../../papers/daily/2607.12909v1.pdf)（548 KB, 8 页）
- **关联概念**：Liquid Time-Constant (LTC) / 神经 ODE / 倒立摆模型 / Lyapunov 稳定性 / 边缘部署 / 反事实推理 / Time-to-Collision

## 2. 核心问题

视觉跌倒检测（vision-based fall detection）在老年人监护与智能监控中是核心场景，但现有深度学习方案普遍陷入"**姿态分类静态认知陷阱 (posture classification cognitive trap)**"：

1. **静态姿态匹配 vs 动态失稳过程**：主流 2D-CNN / 3D-CNN / skeleton-RNN / GCN 把跌倒视为"摔倒到地面"等离散姿态模板匹配，把复杂的连续物理事件简化为"模式匹配问题"，完全剥离人体运动背后的连续生物力学机制。
2. **类跌倒动作的混淆**：日常动作中**主动高度下降**（rapid squatting, controlled sitting）与**被动失稳**（real fall）在视觉上高度重叠，系统无法区分，导致大量假报警 (false alarm)，严重削弱连续监护系统的实用价值。
3. **边缘部署算力约束 vs 模型规模**：传统 pipeline 依赖大型 backbone (MobileNetV3 ~2.5M, ResNet-18 ~11M, Transformer-Base ~86M) + 重型 temporal model (LSTM 隐藏 128 约 200K)，无法在边缘 MCU 上做实时闭环 (30 FPS → 33.3 ms/帧预算)。
4. **可解释性缺失**：基于置信度的分类器无法告诉用户"系统为什么认为这是跌倒"，对医疗监护场景信任度不足。

由此问题被重新表述为：**"跌倒不是一个视觉分类问题，而是一个连续物理失稳过程；必须从'姿态分类'转向'稳定性判定'，并在边缘 MCU 上以极小参数完成实时推理。"** 这正是 LTC (continuous-time ODE, 输入相关时间常数) 的天然应用场景——但既有工作尚未将 LTC 引入视觉跌倒检测的 biomechanics 视角。

## 3. 方法论与核心思路

### 3.1 总体架构：三层流水线

```
Perception Layer (YOLOv11n-pose)
    ↓ 17 COCO 关键点 + 时间归一化 + Support Polygon 6D 几何特征
Dynamics Decoupling Layer (核心创新)
    ├── LTC-CoM Subsystem (16 hidden, τ_A 大, 模拟大惯性)
    ├── LTC-BoS Subsystem (16 hidden, τ_B 小, 模拟高频敏捷)
    └── Coupling Interaction Module (Hadamard 乘性耦合)
Stability Manifold Determination Layer
    ├── Lyapunov-inspired 距离 D_M(H(t))
    ├── Stability Score S(t) = σ(-D_M + λ_margin)
    ├── Directional Velocity Check (向外判定)
    ├── Counterfactual Inference (反事实恢复轨迹)
    └── TTC Estimation (Time-to-Collision)
```

**上下文关系**：
- **与 LTC (Hasani 2021)**：采用输入相关时间常数 τ ∈ ℝ^d_{h>0} 的 ODE 形式，让"时间常数"显式编码物理惯性差异（大 τ = 大惯性 = CoM；小 τ = 高频敏捷 = BoS）。
- **与 CfC (Hasani 2022)**：作者明确选择原 ODE 形式而非闭式近似，原因在 §3.2 — 需要保留 ODE 数值积分在 irregular sampling 下的物理一致性，且 tanh 激活的 zero-centered 有界性保证 ODE 全局稳定。
- **与 LNN/LFM2 (Liquid Foundation Models)**：本工作专注 ODE-based continuous-time 而非 closed-form；与 LFM2 在视觉任务上的连续时间建模可形成对比。
- **与传统视觉分类 pipeline**：将"分类"重定义为"**相空间稳定性判定**"，输出不仅是标签还有 Lyapunov 函数值、反事实轨迹、TTC 三层可解释信号。

### 3.2 核心：双 LTC 子系统的动力学解耦 (Eq. 1–3)

人体平衡可建模为**倒立摆 (inverted pendulum)**，稳定性由质心 (CoM, 大惯性) 与支撑面 (BoS, 高频敏捷) 的连续耦合决定。论文显式构造两个独立 LTC 子系统，各自遵循同型 ODE：

$$\tau \odot \frac{dh(t)}{dt} = -h(t) + \tanh\!\bigl(W^{(i)} x(t) + W^{(h)} h(t)\bigr) \quad \text{(Eq. 1)}$$

- $\tau \in \mathbb{R}^{d_h}$：**逐维**可学习时间常数向量，初始化 $\mathcal{U}[0.1, 2.0]$；
- $W^{(i)} \in \mathbb{R}^{d_h \times d_x}, W^{(h)} \in \mathbb{R}^{d_h \times d_h}$：输入与循环权重矩阵；
- $d_h = 16$（每个子系统隐藏单元数）；
- 前向用 **Euler 法**数值积分，固定步长 $\Delta t = 1/30$ s 与视频帧率一致。

**LTC-CoM** (Eq. 2)：输入 $x_{com} \in \mathbb{R}^6$ 为归一化髋中心坐标 + 一阶差分（速度）+ 二阶差分（加速度）；$\tau_A$ 初始化偏大 → 数学上模拟质心大机械惯性，保证轨迹平滑不能瞬变，对应真实引力约束。

**LTC-BoS** (Eq. 3)：输入 $x_{bos} \in \mathbb{R}^6$ 为支撑多边形 6D 几何特征 (面积 / 质心 / 宽 / 高 / 边界裕度)；$\tau_B$ 初始化偏小 → 数学上反射高频敏捷 (补偿步伐、撑地等)。

**两个 ODE 项的物理意义**：
- 恢复 / 阻尼项 $-h(t)$：肌肉刚度 + 关节摩擦的指数衰减，无外部刺激时动量回归零能量平衡；
- 饱和驱动力 $\tanh(\cdot)$：**zero-centered, 有界 ∈ [-1,1]** 的双曲正切对应生理饱和效应（最大肌肉力矩），保证 ODE 数值全局稳定。

**LTC vs LSTM 的必要性** (作者论证)：LSTM 离散门控强制按帧更新，无法建模肌肉收缩与重心转移造成的**变时延**且对不规则采样敏感；LTC 作为真正微分方程求解器自然匹配机械连续惯性动力学。

### 3.3 耦合交互模块 (Eq. 4)

生物力学现实：CoM 与 BoS 高度耦合——CoM 严重偏移触发 BoS 调整（步伐），BoS 崩塌（打滑）加速 CoM 下落。论文用乘性耦合机制构造 joint hidden state：

$$h_{joint} = \text{Concat}\!\Bigl(P_A h_A + M \odot \sigma(\beta) \odot (P_A h_A \otimes P_B h_B),\ P_B h_B\Bigr) \quad \text{(Eq. 4)}$$

- $P_A : \mathbb{R}^{d_A} \to \mathbb{R}^{d_j}, P_B : \mathbb{R}^{d_B} \to \mathbb{R}^{d_j}$：线性投影，$d_j = 32$；
- $M \in \mathbb{R}^{d_j}$：可学习交互掩码；$\beta$：可学习门控标量，由 $\sigma(\cdot)$ 调制；
- $\odot$：Hadamard 积；$\otimes$：外积。

**物理行为**：正常站立时交叉交互保持微弱；异常事件（如打滑）"支撑失效"信号被急剧放大并注入 CoM 子系统，加速 joint state 跌出稳定区域。

### 3.4 稳定性流形判定 (Eq. 5–6)

将 joint hidden state $H(t) = h_{joint}(t) \in \mathbb{R}^{d_A + d_B}$ 映射到相空间，显式定义稳定区域 $\Omega_{stable}$，中心 $H_0 \in \mathbb{R}^{d_{joint}}$ 为可学习参数（初始化为零），代表正常状态的**经验平衡点**。协方差逆 $\Sigma^{-1}$ 初始化为单位阵并固定训练，得到简化 Mahalanobis-like 距离：

$$D_M(H(t)) = \sqrt{(H(t) - H_0)^T \cdot \text{diag}(\Sigma^{-1}) \cdot (H(t) - H_0) + \epsilon} \quad \text{(Eq. 5)}$$

- $\epsilon = 10^{-8}$：数值稳定项；
- 对角近似 $\Sigma^{-1}$：边缘算力约束下的折衷（完整协方差留作未来工作）；
- $D_M$ 与 **Lyapunov 直接法**深刻联系：$D_M$ 作为候选 Lyapunov 函数 $V(x)$，状态落在 $\Omega_{stable}$ 吸引域内 → 系统局部稳定。

连续稳定性评分：

$$S(t) = \sigma(-D_M(H(t)) + \lambda_{margin}) \quad \text{(Eq. 6)}$$

- $\lambda_{margin}$：流形边界的物理容差阈值（可学习）。

**Boundary Crossing Detection**——必须同时满足两条：
1. **空间条件** $D_M(H(t)) > \lambda_{margin}$；
2. **速度条件** $\cos\langle \frac{dH(t)}{dt}, H(t) - H_0 \rangle > 0$（即法向速度向量**背离**稳定中心）。

第二条件精确过滤假阳性：用户主动下蹲时虽然 $D_M$ 暂越阈值，但内部恢复动力学会把速度向量重新指向 $H_0$，自动归类为受控动作。

### 3.5 反事实推理与 TTC (Eq. 7–8)

**Counterfactual Inference** (Eq. 7)：注入"最优恢复力" $I_{recovery}$，ODE 求解器前向积分投射虚拟未来轨迹：

$$H_{cf}(t + \Delta t) = H(t) + \int_t^{t+\Delta t} f_{joint}(H(\tau), I_{recovery}, \theta)\, d\tau$$

物理意义：作为动态"压力测试"。若 $H_{cf}$ 折回 $\Omega_{stable}$ → 受控动作；若 $H_{cf}$ 仍发散 → 不可逆跌倒，提供最终报警的不可反驳物理基础。

**TTC Estimation** (Eq. 8)：

$$\text{TTC} = 0.5 \cdot \frac{D_M(H(t))}{\|v(t)\|} + 0.5 \cdot \text{MLP}_{ttc}(H(t), D_M(H(t)))$$

结合基于物理的几何投影与可学习细化项，对低速接近边界的鲁棒性更好。

**训练策略**：仅对主分类器施加 cross-entropy 损失；反事实 + TTC 头通过共享梯度反向传播隐式训练，**无专用辅助损失**。

### 3.6 参数规模分析

| 模块 | 隐藏维度 | 参数数 | 占比 |
|---|---:|---:|---:|
| LTC-CoM | 16 | 400 (16 τ + 112 W_in + 272 W_h) | 2.49% |
| LTC-BoS | 16 | 400 | 2.49% |
| Coupling Interaction | 32 | 1,121 | 6.97% |
| Stability Manifold + Velocity + Counterfactual | — | 14,167 | 88.06% |
| **总计** | **64 核心神经元** | **16,088 (16.1K)** | **100%** |
| float32 内存 | | 0.06 MB | |

对比：LSTM 基线 (hidden 128) ≈ 200K；MobileNetV3-Small ≈ 2.5M；ResNet-18 ≈ 11M；Transformer-Base ≈ 86M。LTC-Fall 仅为 MobileNetV3 的 **0.6%**。

**端到端推理延迟**：LTC-Fall 时序模块 **20–46 ms/帧**（30 FPS 视频理论预算 33.3 ms/帧），为前端 YOLOv11n-pose 感知与底层 MCU 调度留出充足 CPU / 内存裕度。

## 4. 核心公式提取

| 编号 | 公式 | 物理 / 工程意义 |
|:---:|---|---|
| (1) | $\tau \odot \frac{dh(t)}{dt} = -h(t) + \tanh(W^{(i)} x(t) + W^{(h)} h(t))$ | 双 LTC 子系统统一 ODE，$\tau$ 编码子系统惯性差异 |
| (2) | $\tau_A \frac{dh_A}{dt} = -h_A + \tanh(W^{(i)}_A x_{com} + W^{(h)}_A h_A)$ | CoM (大惯性) 子系统 |
| (3) | $\tau_B \frac{dh_B}{dt} = -h_B + \tanh(W^{(i)}_B x_{bos} + W^{(h)}_B h_B)$ | BoS (高频敏捷) 子系统 |
| (4) | $h_{joint} = \text{Concat}(P_A h_A + M \odot \sigma(\beta) \odot (P_A h_A \otimes P_B h_B), P_B h_B)$ | 物理乘性耦合，异常时放大交互 |
| (5) | $D_M(H(t)) = \sqrt{(H-H_0)^T \text{diag}(\Sigma^{-1}) (H-H_0) + \epsilon}$ | Lyapunov 候选函数，对角协方差近似 |
| (6) | $S(t) = \sigma(-D_M(H(t)) + \lambda_{margin})$ | 连续稳定性评分 ∈ [0,1] |
| (7) | $H_{cf}(t+\Delta t) = H(t) + \int_t^{t+\Delta t} f_{joint}(H(\tau), I_{recovery}, \theta)\, d\tau$ | 反事实恢复轨迹 |
| (8) | $\text{TTC} = 0.5 \cdot D_M(H) / \|v(t)\| + 0.5 \cdot \text{MLP}_{ttc}(H, D_M)$ | 物理 + 学习混合 TTC 估计 |

## 5. 关键成果与贡献

### 5.1 主结果（消融实验 Table II, 3 seed 均值 ± 标准差）

| 配置 | 参数 | Acc (%) | F1 (%) |
|---|---:|---:|---:|
| **(a) LTC-Fall (Full)** | **16.0K** | **96.63 ± 1.26** | **91.02 ± 4.02** |
| (b) w/o Decoupling | 3.7K | 91.92 ± 0.00 | 78.47 ± 1.48 |
| (c) w/o Coupling | 3.0K | 95.29 ± 2.08 | 87.75 ± 6.31 |
| (d) w/o Stability Manifold | 18.6K | 95.29 ± 1.72 | 87.41 ± 5.54 |
| (e) w/o Counterfactual | 10.8K | 95.28 ± 1.76 | 87.41 ± 5.54 |
| (f) LSTM Baseline (hidden 16) | 8.1K | 93.60 ± 0.48 | 82.81 ± 1.63 |

**关键消融解读**：
- **动力学解耦 (a vs b, ΔF1 = -12.55%)**：去掉 CoM/BoS 显式分离，单 LTC 处理拼接 12D 输入时性能**断崖式下跌**，证明物理解耦对"区分主动高度下降 vs 被动失稳"是结构性必需。
- **耦合交互 (a vs c, ΔF1 = -3.27%)**：直接 concat 两子系统输出而无 Hadamard 交互，质量中心偏移 → 支撑调整的物理反馈贡献非平凡。
- **稳定性流形 (a vs d, ΔF1 = -3.61%)**：用普通 2 层 MLP 替换 Lyapunov 流形（参数反而 +2.6K），失去几何距离约束 + 方向速度检查，崩解为不可解释黑盒边界，更易对受控动作触发假阳。
- **辅助头 (a vs e, ΔF1 = -3.61%)**：去掉反事实 + TTC 头 → 中等 F1 下降；虽无专用损失，共享梯度反向传播提供有益正则化，把潜表征推向物理合理的稳定性轨迹。
- **LTC vs LSTM (a vs f)**：相同隐藏维度下 LTC 比 LSTM **F1 +3.03%, Acc +3.03%**，验证 ODE 数值积分对**不规则生物力学瞬态**的建模优于离散门控机制。

### 5.2 评价策略：Precision @ Fixed Recall

作者**主动放弃传统 overall accuracy**，改用严格的"固定召回率下的精度"协议——先固定 Recall > 98%（"生命安全红线"），再比较 Precision。这与跌倒检测零漏检的代价敏感性匹配（漏检 = 错过黄金救援窗口，假阳 = 二次人工核验可控成本）。在该协议下：

- 黑盒分类器在受控动作下触发"宁可错报"模式 → Precision 暴跌至不可接受水平；
- LTC-Fall 通过稳定性流形 + 反事实推理维持高 Precision，**根本性解决行业假阳困境**。

### 5.3 边缘部署指标

- **参数**：16,088 (16.1K) → float32 内存 0.06 MB；
- **核心神经元总数**：64；
- **时序模块推理延迟**：20–46 ms/帧（30 FPS 预算 33.3 ms/帧）→ **实时通过**；
- **CPU / 内存裕度**：为前端 YOLOv11n-pose 与 MCU 调度留足带宽，**稳定非阻塞闭环执行**。

### 5.4 贡献清单

1. **首次将 LTC 神经网络引入视觉跌倒检测的 biomechanics 视角**——把视觉任务从"姿态分类"重定义为"连续物理失稳过程"；
2. **双 LTC 解耦架构**——CoM (大惯性 τ) + BoS (高频敏捷 τ) 显式分离，物理意义上对应倒立摆两大动力学子系统；
3. **稳定性流形 + 方向速度检查**——Lyapunov 候选函数 + 法向速度背离条件，过滤受控动作假阳；
4. **反事实恢复推理 + TTC 估计**——给出"不可逆跌倒"的不可反驳物理基础，并支持预警；
5. **16.1K 参数 + 64 神经元的极限轻量化**——20–46 ms/帧边缘 MCU 实时推理，float32 仅 0.06 MB。

## 6. 局限性与未来展望

### 6.1 作者自陈局限
- **三状态未完整闭环**：架构设计为三状态 (Normal / Falling / Fallen) 时间转移，本研究**仅在 Normal vs Falling 二分类数据集**上验证核心稳定性判别能力，完整三状态转移留作未来工作。
- **数据集规模与单一性**：单数据集 29 FPS、1280×720，未做跨数据集 (cross-dataset) 与多视角泛化验证。
- **协方差对角近似**：$\Sigma^{-1}$ 取对角形式为边缘算力妥协，未估计完整协方差。
- **Recall @ 98% 固定**：未给出 ROC / PR 全曲线，仅展示固定点上的 Precision，难以评估低召回区的能力。
- **真实边缘硬件未量化**：未给出 MCU (Raspberry Pi Pico / ESP32 / Cortex-M7) 上的实测功耗、内存峰值、电池续航等具体数字。

### 6.2 隐含局限（与本仓视角）
- **感知层依赖 YOLOv11n-pose**：感知错误（遮挡、多人、远距离小目标）会污染下游 LTC 输入；论文未给出 YOLO 失败的鲁棒性测试。
- **物理先验过度耦合**：把 CoM / BoS 解耦为正交互补子系统的假设在双人 / 多人场景、外骨骼辅助、坐姿等情形下可能失效——但论文未讨论。
- **欧拉积分步长固定**：固定 $\Delta t = 1/30$ s 假设采样均匀；不规则采样（如事件相机、不同步多相机）下未验证。
- **反事实 $I_{recovery}$ 的构造依赖人工设定**：作为"动态压力测试"的核心，反事实力的具体参数化形式未见消融，可能影响不可逆判定阈值。
- **缺乏与 CfC (closed-form) 的直接对照**：作者明确选 ODE 而非闭式近似，但未给"如果用 CfC 是否同等效果"的消融，无法判断是 LTC 本身还是 ODE 数值积分带来增益。
- **F1 标准差偏大**：完整模型 F1 标准差 ±4.02 表明跨 seed 不稳定，3 seed 不足以建立统计显著结论。
- **隐式训练的辅助头**：反事实 + TTC 无专用损失，依赖主分类器梯度反向传播；其实际贡献可能源于正则化而非真正的物理可解释信号。

### 6.3 未来方向
- **三状态完整建模**：Normal → Falling → Fallen 时间转移的端到端训练，加入姿势恢复 / 倒地后行为识别；
- **跨数据集与多视角泛化**：在 UR-Fall / Le2i / Multicam / UP-Fall 等公开基准上验证，特别是视角变化、遮挡、多人场景；
- **完整协方差估计**：用 EM 或可学习低秩近似估计 $\Sigma^{-1}$，提升 Lyapunov 距离的几何精确性；
- **真实硬件落地测试**：在 Raspberry Pi 4 / Jetson Nano / Cortex-M7 MCU 上给出功耗、延迟、内存峰值的实测数据；
- **与 CfC 的对照消融**：明确 ODE 数值积分 vs 闭式近似的边际贡献；
- **多人 / 复杂场景扩展**：处理多人同时跌倒、外骨骼辅助、轮椅等特殊情形；
- **不确定性估计**：在 S(t) 基础上引入贝叶斯或 MC-Dropout 不确定性，给出置信区间；
- **与因果推理结合**：把"反事实恢复"形式化为 do-calculus 框架，提升物理因果一致性。

## 7. 对本仓的意义

### 7.1 工程模板价值
- **双 ODE 子系统 + Hadamard 耦合**是**可复用的 ODE 模块化模板**：本仓 `lnn/core/variants.py` 可新增 `DualLTCSubsystem(coeff_init=(τ_A, τ_B))`，把物理子系统的"惯性差异"显式编码为初始 τ 分布，是 LNN 与领域物理先验结合的范例。
- **Lyapunov 流形分类器**（Eq. 5–6）可封装为 `StabilityManifoldClassifier(H0, Σ_inv, λ_margin)`，与现有 `bench_lyapunov_stable_cfc` 系列脚本直接对照。
- **反事实推理 + TTC**是**可解释 AI 在连续时间模型上的范式**，可作为 `lnn/core/diagnostics.py` 的新模块。
- **Euler 积分 + 固定步长** 与本仓 `EulerLTCNetwork` 实现一致；可参考其 ODE 数值稳定性论证（$\tanh$ zero-centered 保障）补充到文档。

### 7.2 评测协议价值
- **Precision @ Fixed Recall**是面向"安全关键 + 代价敏感"任务的**通用评测框架**，可作为 `analysis/lnn_diagnostics/` 新增协议，特别适用于 elderly care、医疗监测、自动驾驶等场景。
- **16.1K 参数 + 0.06 MB + 20–46 ms/帧**为**边缘 LNN 部署**提供量化基线，可与 `scripts/jetson_lnn_benchmark.py` 现有 GRU / CfC 基准对照。

### 7.3 与本仓现有工作的关系
- 与 `bench_lyapunov_stable_cfc` / `bench_iss_stable_cfc` / `bench_frozen_multibasin_lyapunov_cfc` 直接对位：均把稳定性分析与 LNN 结合，但本工作把 stability 作为**判别信号**而非**正则化项**。
- 与 `bench_frozen_random_basin_cfc` / `bench_frozen_sampled_multitau_cfc` 的多时间常数思路呼应——但本文 τ 是**物理子系统差异**而非**多尺度信号分解**。
- 与 `bench_adaptive_time_constant_cfc` / `bench_sinusoidal_time_emb_cfc` 对位：均涉及 τ 的可学习 / 输入相关设计，但本文明确把 τ 与"子系统惯性差异"绑定。
- 与 `bench_soft_neuron_attention_cfc` / `bench_attn_weights_dump` 等注意力机制正交——本文用 ODE + Lyapunov 几何替代注意力，与本仓"非注意力路线"系列互补。

### 7.4 Verdict
- **TARGET-POSITIVE** — **首次把 LTC 引入视觉跌倒检测 biomechanics**，与本仓 LNN 主题高度对齐；动力学解耦 + Lyapunov 流形 + 反事实推理是新颖且工程可落地的设计；
- **TARGET-POSITIVE** — **边缘实时部署** (16.1K, 20–46 ms/帧) 是 LNN 在边缘 AI 场景的强证据；
- **TARGET-POSITIVE** — **可解释性范式**（Lyapunov 距离 + 反事实 + TTC）值得沉淀到本仓诊断模块；
- **TARGET-NEGATIVE-WITH-NUANCE** — 二分类实验 + 3 seed + F1 标准差偏大，统计证据偏弱；
- **TARGET-NEGATIVE-WITH-NUANCE** — 未开源代码、未公开数据集、协方差对角近似，完整复现需自行实现；
- **TARGET-DEPENDENT-WITH-NUANCE** — 三状态建模、跨数据集、CfC 对照留作未来工作，本仓可作为跟进方向。