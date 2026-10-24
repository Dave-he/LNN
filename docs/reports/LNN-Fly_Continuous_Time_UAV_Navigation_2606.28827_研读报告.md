---
title: "LNN-Fly: Continuous-Time UAV Navigation for Robust Obstacle Avoidance under Timing Mismatch"
date: 2026-10
tags: [LNN, CfC, continuous-time-policy, UAV-navigation, obstacle-avoidance, sim-to-real, deployment-robustness, adaptive-forgetting, timing-mismatch]
arxiv_id: 2606.28827v1
---

# 研读报告：LNN-Fly — 用显式 Δt 条件化 + 输入驱动自适应遗忘，把 CfC 变成可部署的避障策略

## 1. 元数据

- **论文标题**：LNN-Fly: Continuous-Time UAV Navigation for Robust Obstacle Avoidance under Timing Mismatch
- **作者**：Yulin Huang, Shaojie Chen, Di Feng, Jiahao Wang, Ping Liu†, Jianxiao Zou（† 通讯，liup81@uestc.edu.cn；前四人为硕士研究生）
- **机构**：深圳先进技术研究院 / 电子科技大学（UESTC），中国深圳
- **发表时间**：2026-06-27 (arXiv: 2606.28827v1, cs.RO)；IEEE 会议投稿格式（2026）
- **代码**：文中未给出公开仓库地址
- **本地 PDF**：[papers/2606.28827/2606.28827.pdf](../../papers/2606.28827/2606.28827.pdf)（4.6 MB, 8 页，已落盘到仓库）
- **本地文本**：`../../papers/2606.28827/2606.28827.txt`（39.9 KB，8 页）
- **关联概念**：CfC (Hasani 2022) / Closed-form continuous-time / 连续时间导航策略 / Δt 条件化 / 输入驱动自适应遗忘门 / 动态规划启发的结构化收缩 / 可微物理训练 / sim-to-real / 边缘推理
- **本仓相关性**：**正面但有条件** —— 这是本仓目前读到的**第一篇把 CfC 用在真实机器人部署闭环、且报告了真机时序失配数据**的论文，与本仓"边缘部署 / 时间归纳偏置组件"路线（§0 定位）同向；但其任务面（UAV 避障）与本仓主战场（时序预测 / 连续时间建模基准）不同。

## 2. 核心问题

端到端 UAV 导航在仿真里能做得很灵活，但**部署后避障行为会退化**。论文把退化拆成两个耦合的挑战：

**Challenge 1 — 强且可迁移的避障能力。** 现有端到端策略大多优化速度或轨迹跟踪，把避障当作"控制好了的自然结果"。但在密集障碍场景里这个假设站不住：物理模型、传感器处理、飞控栈的任何小偏差都会在迁移后累积成碰撞。工业巡检 / 室内安防 / 搜救 / 野火监测这些目标应用里，**安全与鲁棒性优先于最大速度**。

**Challenge 2 — 真实控制回路的时序失配。** 真实板载系统很少维持固定控制周期：不规则传感器采样、频率下降、调度抖动在资源受限的边缘机上很常见。而**大量学习式序列策略训练和部署时都假设控制更新等间隔到达**。当有效 Δt 在部署时变化，**隐状态演化与物理运动就会错位**，产生不安全指令。这在近距离避障时尤其致命 —— 哪怕几毫秒的控制不规则都可能直接导致碰撞。

论文把问题重述为：**如何设计一个轻量连续时间策略 π_θ(h_k, y_k, Δt_k)，使其在仿真器/传感器/飞控栈/时间轴全部偏移的条件下，仍然保持可靠的避障行为。**

## 3. 方法论与核心思路

### 3.1 总体架构

LNN-Fly 是一个 **encoder → CfC 连续时间核心 → policy head** 的紧凑三段式（Fig. 2）：

| 模块 | 规格 |
|---|---|
| LiDAR 编码器 | 60×15 角度分辨率扇区化 → reshape `[B,1,15,60]` → 4 层 CNN (1→12→24→36) → 2 层 MLP → 64 维 |
| 状态编码器 | 10 维状态向量 → MLP (10→48→48) |
| 融合 | 64 + 48 = **112 维**拼接向量 |
| 时序核心 | **CfC 单元，hidden size 32**，输入显式携带 δ_k = log(Δt_k + ε) |
| 输出头 | 32→32→6，输出 6 维指令 u_k = [a_k^T, c_k^T]^T（3 维期望加速度 + 3 维速度反馈修正项，用于无里程计时补偿速度漂移） |

与"把 CfC 当通用时序滤波器"的既有做法不同，本文的立场是：**连续时间递推应当是任务结构化的组件**，隐状态的演化必须同时受制于（a）经过的物理时间、（b）当前障碍上下文。

### 3.2 动态规划启发的神经液态动力学（Eq. 1–6）

作者把隐状态 h(t) 解释为一个**应与"观测条件化的局部目标"保持一致的紧凑摘要**，定义局部一致性能量：

$$
E(t) = \frac{1}{2}\|h(t) - \kappa(x_t, h_t)\|_2^2 \tag{1}
$$

其中 κ(x_t, h_t) 是从当前观测与潜在上下文预测出的**学习到的局部目标**。论文明确声明：**动态规划类比仅用作收缩结构的设计动机，模型不优化 Bellman 目标、也不做值迭代** —— 这是与真正的 DP/RL 的关键区别，不可误读。

对该能量做下降动力学，并附加输入驱动的泄漏项，得到：

$$
\frac{dh}{dt} = \omega(x,h) \odot \underbrace{\left[\kappa(x,h) - h\right]}_{\text{Structured Target Contraction}} - \lambda(x) \odot h \underbrace{}_{\text{Stimulus-Driven Discounting}} \tag{2}
$$

这个参数化把**每个隐通道限制为向局部目标的收缩形式**，而不是无约束的递推重写 —— 这是本文与"标准 CfC"的结构性差异。

在 $[t_k, t_{k+1})$ 上用分段常数近似（κ ≈ κ_k, ω ≈ ω_k, λ ≈ λ_k），Eq. 2 化为线性 ODE：

$$
\dot{h}(t) = -(\omega_k + \lambda_k) \odot h(t) + \omega_k \odot \kappa_k \tag{3}
$$

其闭式离散更新为：

$$
h_{k+1} = \alpha_k \odot h_k + (1 - \alpha_k) \odot \frac{\omega_k}{\omega_k + \lambda_k + \epsilon_s} \odot \kappa_k \tag{4}
$$

$$
\alpha_k = \exp\left(-(\omega_k + \lambda_k)\Delta t_k\right) \tag{5}
$$

**Δt 的注入点在此**：网络参数以 $\delta_k = \log(\Delta t_k + \epsilon)$ 为条件，δ_k 同时参与 κ_k、ω_k、λ_k 的生成。于是有效神经时间常数 $\tau_k = (\omega_k + \lambda_k)^{-1}$ **同时随输入特征与真实经过的物理时间自适应**，使递推更新在变速率控制下保持同步。

### 3.3 输入驱动的自适应时间尺度调节（Eq. 9–10）

遗忘率不采用手工调度规则，而是由当前观测直接驱动：

$$
\lambda_{raw} = W_\lambda x_k + b_\lambda + \delta_k \cdot \theta_\lambda \tag{9}
$$

$$
\lambda_k = \operatorname{clip}\left(\operatorname{softplus}(\lambda_{raw}) + \lambda_{floor},\ 0,\ \lambda_{max}\right) \tag{10}
$$

其中 $W_\lambda, b_\lambda, \theta_\lambda$ 可学习；λ_floor = 0 保留良性条件下的近似记忆，λ_max 为有限上界以避免"过度遗忘"不稳定。**注意**：λ_floor 与 λ_max 不单独优化，只作为自适应遗忘的保守上下限 —— 论文称除 λ_max 过小外，中等幅度改动性能稳定。

**涌现行为（Fig. 3）**：在开阔区域或对同一障碍的持续规避中，学到的最优 λ_k 较小 → 保留潜在动量、避免振荡性过度反应；在突发危险或部署扰动下，邻近度线索推高 λ_k → 缩短有效时间常数、压制陈旧隐状态历史。障碍清除后 λ_k 出现尖峰，快速刷新状态以适配下一飞行段。

### 3.4 可微物理训练（Eq. 11）

策略输出的加速度与修正项经重力补偿形成最终控制 u_k，用**同一个采样控制间隔 Δt_k** 做可微动力学 rollout（$s_{k+1} = \Phi(s_k, u_k, \Delta t_k)$），保证隐状态演化与物理传播在训练期时序一致。复合损失：

$$
\mathcal{L}_{total} = w_1 \mathcal{L}_{track} + w_2 \mathcal{L}_{safe} + w_3 \mathcal{L}_{smooth} + w_4 \mathcal{L}_{reg} \tag{11}
$$

- $\mathcal{L}_{track}$：惩罚目标速度、朝向、高度的偏差
- $\mathcal{L}_{safe}$：用平滑障碍函数 $\phi(d) = \operatorname{softplus}(d_{safe} - d)$ 强制避障，按接近速度加权
- $\mathcal{L}_{smooth}$：惩罚控制量（加速度）及高阶导（jerk、snap），保证轨迹可执行
- $\mathcal{L}_{reg}$：对遗忘率分布做正则（L1、均值目标、软区间惩罚），维持长期记忆稳定性同时允许敏捷反应

### 3.5 与 CfC / LTC / GRU 的关系

| 维度 | Standard CfC (Hasani 2022) | LNN-Fly | 差异实质 |
|---|---|---|---|
| 隐状态更新 | 通用闭式连续时间更新 | **强制向局部目标 κ 的收缩形式**（Eq. 2） | 结构性归纳偏置，限制隐通道的自由度 |
| 时间注入 | 隐式（作为 cell 的 interval 参数） | **显式 δ_k = log(Δt+ε) 条件化**，同时调制 κ/ω/λ | Δt 不只是"缩放"，而是进入参数生成 |
| 遗忘机制 | 由 cell 内部参数隐式决定 | **外置的输入驱动自适应遗忘门 λ_k**，带 clip 上下限 | λ 可解释、可视化（Fig. 3），且被 L_reg 直接监督 |
| 训练目标 | 通用监督/自监督序列损失 | **可微物理 rollout + 安全屏障 + 平滑/jerk 约束** | 任务结构化，BPTT 穿过受约束转移（Eq. 7–8） |

与 LTC (Hasani 2021) 的关系：LTC 是带输入门/遗忘门的离散化 ODE cell，LNN-Fly 显式把"遗忘"从 cell 内部提到**输入驱动、任务相关、可正则化**的层面；与 GRU 的关系：Eq. 4 的形式与 GRU 的凸组合更新同构（$\alpha_k h_k + (1-\alpha_k)\tilde{h}_k$），差别在于 $\alpha_k$ 由**显式时间条件**导出、且状态被约束为向 κ 收缩而非任意候选。

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. 1 | $$ E(t) = \tfrac{1}{2}\lVert h(t) - \kappa(x_t, h_t)\rVert_2^2 $$ | 局部一致性能量；h 应向观测条件化局部目标 κ 靠拢 |
| Eq. 2 | $$ \frac{dh}{dt} = \omega(x,h)\odot\big[\kappa(x,h) - h\big] - \lambda(x)\odot h $$ | 结构化目标收缩 + 刺激驱动折扣，构成本文的核心动力学 |
| Eq. 3 | $$ \dot h(t) = -(\omega_k+\lambda_k)\odot h(t) + \omega_k\odot\kappa_k $$ | 分段常数近似下的线性 ODE，可解析求解 |
| Eq. 4 | $$ h_{k+1} = \alpha_k\odot h_k + (1-\alpha_k)\odot\tfrac{\omega_k}{\omega_k+\lambda_k+\epsilon_s}\odot\kappa_k $$ | 闭式离散转移（论文编号 Eq. 5/6 之一） |
| Eq. 5 | $$ \alpha_k = \exp\big(-(\omega_k+\lambda_k)\Delta t_k\big) $$ | 时间感知门控因子；显式依赖真实 Δt |
| Eq. 9 | $$ \lambda_{raw} = W_\lambda x_k + b_\lambda + \delta_k\cdot\theta_\lambda,\quad \delta_k=\log(\Delta t_k+\epsilon) $$ | 输入驱动的遗忘率原始值，显式含时间项 |
| Eq. 10 | $$ \lambda_k = \operatorname{clip}\big(\operatorname{softplus}(\lambda_{raw})+\lambda_{floor},\,0,\,\lambda_{max}\big) $$ | 自适应遗忘门的软区间约束 |
| Eq. 11 | $$ \mathcal{L}_{total} = w_1\mathcal{L}_{track}+w_2\mathcal{L}_{safe}+w_3\mathcal{L}_{smooth}+w_4\mathcal{L}_{reg} $$ | 可微物理训练的复合损失 |
| Eq. 7 | $$ \frac{\partial \mathcal{L}_{total}}{\partial\theta} = \sum_k \frac{\partial \mathcal{L}_{total}}{\partial h_{k+1}}\cdot\frac{\partial h_{k+1}}{\partial(\kappa_k,\omega_k,\lambda_k)}\cdot\frac{\partial(\kappa_k,\omega_k,\lambda_k)}{\partial\theta} $$ | BPTT 穿过受约束转移（隐式一致性偏置的来源） |
| Eq. 8 | $$ \frac{\partial h_{k+1}}{\partial h_k} \approx \mathrm{Diag}(\alpha_k) + \frac{\partial h_{k+1}}{\partial\kappa_k}\frac{\partial\kappa_k}{\partial h_k} + \frac{\partial h_{k+1}}{\partial\omega_k}\frac{\partial\omega_k}{\partial h_k} $$ | 隐状态 Jacobian 的一阶分解，主项是显式时间收缩因子 |

## 5. 关键成果与贡献

### 5.1 主结果：时序失配下的成功率（Table I）

30 Hz 训练、Scenario II @ 4.0 m/s、20 次试验（`-` = 灾难性失败）：

| 频率 | 模型 | N=1 成功率 | N=3 | N=5 |
|---|---|---|---|---|
| 30 Hz | **LNN-Fly** | **20/20** | **20/20** | **18/20** |
| 30 Hz | LNN-Fly (Fixed Δt) | 20/20 | 18/20 | 19/20 |
| 30 Hz | Standard CfC | 20/20 | 19/20 | 17/20 |
| 30 Hz | LSTM | 20/20 | 14/20 | 11/20 |
| 8 Hz | **LNN-Fly** | 18/20 | 14/20 | **13/20** |
| 8 Hz | LNN-Fly (Fixed Δt) | 15/20 | 11/20 | 9/20 |
| 8 Hz | Standard CfC | 18/20 | 15/20 | 13/20 |
| 8 Hz | LSTM | 15/20 | 8/20 | 5/20 |
| 5 Hz | **LNN-Fly** | 12/20 | 9/20 | 灾难性失败 |
| 5 Hz | LNN-Fly (Fixed Δt) | 5/20 | 灾难性失败 | 灾难性失败 |
| 5 Hz | Standard CfC | 10/20 | 7/20 | 灾难性失败 |
| 5 Hz | LSTM | 灾难性失败 | 灾难性失败 | 灾难性失败 |

**最关键的一行是 LNN-Fly vs LNN-Fly (Fixed Δt)**：5 Hz + N=1 时 12/20 vs 5/20。这直接证明论文的核心论断 —— **仅有连续时间（ODE）形式是不够的，必须对真实经过的 Δt 做条件化**，否则内部状态在 6 倍训练间隔下会与物理迅速失步。这是本文相对 Standard CfC 的实质增量。

### 5.2 频率抖动鲁棒性（Table II，感知固定 N=3，50% 占空比 30 Hz ↔ f_low）

| 抖动模式 | 模型 | d_min (m) ↑ | d_below ↓ | 成功率 ↑ |
|---|---|---|---|---|
| 30↔8 Hz（中等） | **LNN-Fly** | 0.331 | 0.111 | **17/20** |
| 30↔8 Hz | LNN-Fly (Fixed Δt) | 0.271 | 0.156 | 12/20 |
| 30↔8 Hz | Standard CfC | 0.309 | 0.109 | 16/20 |
| 30↔8 Hz | LSTM | 0.267 | 0.126 | 9/20 |
| 30↔5 Hz（严重） | **LNN-Fly** | 0.326 | 0.145 | **14/20** |
| 30↔5 Hz | LNN-Fly (Fixed Δt) | 0.232 | 0.179 | 7/20 |
| 30↔5 Hz | Standard CfC | 0.295 | 0.136 | 14/20 |
| 30↔5 Hz | LSTM | — | — | 灾难性失败 |

严重抖动下 LNN-Fly 相对 Fixed Δt 变体 **14/20 vs 7/20，翻倍**。

### 5.3 消融：自适应遗忘 λ 的作用

四个变体在 10,000 训练步下对比（LNN-Fly / Diff-Fly / Standard CfC / LNN-Fly w/o λ），报告无碰撞成功率（安全）与平均/最大速度（敏捷）：

- **LNN-Fly vs LNN-Fly w/o λ**：移除 λ **保留了大部分安全性，但速度显著下降** —— 说明遗忘门的作用是"在危险附近快速刷新状态，同时保留结构化连续时间核心"，而非贡献安全性本身。
- **LNN-Fly w/o λ vs Standard CfC**：进一步支持结构化 DP 启发式核心设计的价值。
- **Diff-Fly 在训练期保持竞争力**，但迁移结果（§5.1）显示其优势在时序/仿真器偏移下不稳定。

### 5.4 延迟统计（Table III）

| 阶段 | 平台 | 中位数 (ms) | P95 (ms) |
|---|---|---|---|
| LiDAR 预处理 | 桌面工作站 | 2.88 | 3.55 |
| LiDAR 预处理 | 板载计算机 | 3.93 | 8.70 |
| 策略推理 | 桌面工作站 (GPU) | **0.514** | 0.591 |
| 策略推理 | 板载 CPU @ 10 Hz | 2.48 | 28.31 |
| 策略推理 | 板载 CPU @ 30 Hz | 2.57 | 27.23 |

**P95 的 27–28 ms 尾延迟**出现在板载 CPU 上，论文归因为"真实飞行的调度与通信抖动"。这一点值得本仓注意：**中位数 2.5 ms 与 P95 28 ms 之间有一个数量级以上的差距**，说明真机部署的尾部行为与仿真测得的平均行为差别很大。

### 5.5 真机验证

- 室外森林：4.0 m/s 穿越密集障碍
- 室内窄走廊：**10 Hz 与 30 Hz 双频率均稳定，2.0 m/s，20 次飞行 100% 成功、零碰撞**
- 零样本迁移：可微点质量四旋翼仿真器 → Gazebo+PX4 SITL → 真机，无微调
- 仿真基准：35×10×6 m Gazebo+PX4 场景，Scenario I/II/III 障碍密度递增，每速度 50 次试验；Scenario III @ 5.0 m/s 时 EGO-Planner V2 掉到 18% 成功率

### 5.6 贡献清单

1. **timing-aware 连续时间策略**：把显式 Δt 条件化与输入驱动自适应遗忘组合进一个轻量 CfC 变体，并用 Fixed Δt 对照组**实证证明**时间条件化本身的必要性。
2. **部署导向的训练与评测流程**：可微物理 rollout 中注入 LiDAR 噪声/漂移、部分观测、异步感知、控制周期抖动、执行延迟；评测覆盖仿真器迁移、时序失配、真机三档。
3. **零样本真机迁移**：在 LiDAR 扇区化编码下把策略接入真机，20/20 无碰撞，并给出桌面/板载双平台的延迟分位数。
4. **机制可视化**：Fig. 3 给出 λ_k 沿轨迹的颜色演化（持续规避时低、障碍清除后尖峰），把"隐式时间尺度"变成可读的诊断信号。

## 6. 局限性与未来展望

### 6.1 作者自陈局限

- **任务面窄**：只做 UAV 避障，策略规模极小（hidden size 32、6 维动作），未验证可扩展性。
- **λ_floor / λ_max 不单独优化**，只是保守上下限；论文只称"除 λ_max 过小外性能稳定"，未给出系统敏感性扫描。
- **"动态规划启发"仅是设计动机**，不优化 Bellman 目标、不做值迭代 —— 论文自己反复强调这一点。
- **真机测试规模小**：室内 20 次飞行、单一窄走廊；室外只报 4.0 m/s 一个速度点；**未做跨机体/跨传感器泛化**。
- **P95 尾延迟 27 ms 未做归因分离**，只笼统归于调度与通信抖动。
- **专家路由诊断"未在此报告"**：作为 MoE 无关的普通 CfC 单元，但论文提到监控了 expert load balancing / router health（该表述来自另一篇 LFM2 语境，出现在 tokenizer 论文而非本文，此处不适用）。

### 6.2 隐含局限（本仓视角）

- **P95 未闭环**：28 ms 的 P95 已接近 30 Hz 控制周期（33.3 ms）的 85%。论文把它归为"环境抖动"，但**从控制理论看，这是一个潜在的相位裕度问题**；没有把尾延迟纳入训练扰动分布（训练注入的是控制周期抖动，不是推理尾延迟）。
- **场景分布窄**：障碍均为**静态**圆柱（radius 区间给定、coverage 7–10%）。没有动态障碍、移动人群、可通行区域变化。
- **"Structured Target Contraction" 的收益未单独归因**：Eq. 2 的收缩约束与 Eq. 5 的 α 门控在实现上耦合，论文只做了"w/o λ"这一个消融，**没有做 "w/o 结构化收缩（但保留 Δt 条件化）"的对照** —— 也就是说，"DP 启发的收缩结构"这一创新点**至今没有干净的因果证据**。
- **成功率的统计力度**：20 次试验下 20/20 vs 19/20 的差异不具统计意义；论文多处对比（如 14/20 vs 14/20）实际是持平。
- **与本仓关系**：本仓 §0 把 LNN 定位为"时间归纳偏置组件"，本文正好提供了一个**组件级的负面佐证 —— 组件本身（连续时间）不够，必须配时间条件化**。这与本仓 NR-01/NR-02 的教训方向一致（"结构看起来更好"≠"实测更好"）。
- **可复现性缺口**：无代码仓库、无超参细节（λ_floor/λ_max 的具体取值未给）、无训练数据说明。**按 AGENTS.md 约束 3，本文所有数字均标注为"厂商自报、我方未复现"。**

### 6.3 未来方向

- 把推理尾延迟纳入训练扰动分布，做"感知+推理+通信"全链路时序建模；
- 动态/移动障碍与多机协同下的扩展；
- 跨机体、跨 LiDAR 传感器的 sim-to-real 泛化；
- 显式分离"结构化收缩"与"Δt 条件化"的因果贡献（补做缺失的消融）。

## 7. 对本仓的意义

**可作 `lnn/core/variants.py` 的什么变体。** 直接可落地两件事：(1) **一个显式 Δt 条件化的 CfC 变体** —— 本仓现有 CfC 实现若未把 Δt 作为一等输入（而非仅作为 cell 的时间标量），这是低成本高价值的修正；(2) **一个输入驱动的自适应遗忘门** —— 本仓 `bench_cfc_n_tau.py` / `bench_multi_tau_cfc.py` 系列已经在扫时间常数，把"由输入驱动的 λ_k"接入，可直接检验本仓已有的多时间常数结论是否只在固定 λ 下成立。

**可嫁接到哪个 bench 脚本。** 本仓的 `bench_cfc_vs_ssm.py`、`bench_irrregular_dt.py`（不规则采样时间序列）是最直接的落点 —— 这两个脚本的时序扰动与本文的 Δt 失配场景同构。可以做一个低成本受控实验：**在同一不规则采样基准上，对比 (a) 现有 CfC、(b) Δt 条件化 CfC、(c) Fixed Δt 的 CfC**，验证本仓自己数据上是否复现本文的核心论断（连续时间形式本身不够）。这比直接复现 UAV 任务便宜两个数量级。

**提供哪个新 ablation axis。** 新增 axis：**"Δt 条件化 × 遗忘门" 二维消融**。本仓现有消融主要沿 τ / β / 分支数展开，尚未把"时间条件化"作为独立变量。这条 axis 恰好对应本文最有价值的因果结论，且成本低。

**Verdict**：
- **目标场景（UAV 避障 / 边缘机器人时序失配）：TARGET-POSITIVE** —— 若目标部署栈存在真实的控制周期抖动或采样不规则，本文提供的"显式 Δt 条件化 + 输入驱动遗忘"是有实证支撑（本文自有对照组）的改进方向，值得按上述低成本 bench 复现其核心消融。
- **本仓主战场（时序预测 / 连续时间建模基准）：TARGET-DEPENDENT-WITH-NUANCE** —— 论文的收益来自**部署期时序失配**，若本仓基准的时间采样本身规整，预期增益有限；真正值得复用的是"Δt 条件化"这一个变量，而非整套 UAV 流水线。
- **作为"液态架构价值"证据：TARGET-NEGATIVE-WITH-NUANCE** —— 本文恰好说明**连续时间形式本身不是收益来源**（Fixed Δt 变体在 5 Hz 下崩溃），这与本仓 NR-01/NR-02 的教训同向，可作为"LNN 需要任务结构化设计而非通用时序滤波器"的新增论据。

> **复现状态**：**未复现**。本文所有数字均为作者自报（20 次试验规模，无代码、无独立复现来源）。引用时须按 AGENTS.md 约束 3 标注"未验证"。
