---
title: "Physics-Informed Calibration of Aeromagnetic Compensation in Magnetic Navigation Systems using Liquid Time-Constant Networks"
date: 2024-01
tags: [LNN, LTC, CfC, NCP, magnetic-navigation, MagNav, aeromagnetic-compensation, Tolles-Lawson, state-estimation, edge-AI, SandboxAQ]
---

# 研读报告：物理信息 + LTC 的航空磁补偿标定 — 用连续时间网络从强磁场噪声里剥出微弱磁异常信号

## 1. 元数据

- **论文标题**：Physics-Informed Calibration of Aeromagnetic Compensation in Magnetic Navigation Systems using Liquid Time-Constant Networks
- **作者**：Favour Nerrise（Stanford University 电气工程系 / SandboxAQ AI Ph.D. Resident）, Andrew (Sosa) Sosanya, Patrick Neary
- **机构**：Stanford University; SandboxAQ（量子安全导航公司，驻留项目产出）
- **发表时间**：2024-01-17 (arXiv: 2401.09631v1, cs.LG)；标注 NeurIPS 2023
- **代码**：文中未给出公开仓库链接
- **本地 PDF**：[papers/2401.09631/2401.09631.pdf](../../papers/2401.09631/2401.09631.pdf)
- **本地文本**：[papers/2401.09631/2401.09631.txt](../../papers/2401.09631/2401.09631.txt)
- **数据集**：USAF–MIT Signal Enhancement for Magnetic Navigation Challenge（6 条飞行，5 个标量 + 3 个矢量磁力计，10 Hz 采样）
- **关联概念**：LTC（Hasani 2021）/ CfC（Hasani 2022）/ NCP / AutoNCP / Tolles–Lawson 标定模型 / 状态空间估计 / 磁异常导航 / 边缘设备

## 2. 核心问题

磁异常导航（MagNav）是 GPS 之外 increasingly viable 的备份方案——它不受干扰欺骗（jamming/spoofing）、不受天气影响、无监督定位，低空可达 10 m RMS 定位误差。但它的前提是**磁力计读数必须足够干净**：航空器自身的磁性材料（永磁场、感应场、涡电流场，Tolles–Lawson 三类误差源）加上机载电子设备、通信系统、以及传感器随温度变化的磁性，共同产生的干扰场**远强于**地壳磁异常这个真正有导航价值的弱信号。磁力计输出是地球主场与机载场的非线性组合，导致导航估计随时间漂移。

经典解法是 **Tolles–Lawson (T-L) 标定**：用硬/软铁模型把干扰场参数化为少数几个系数，实时估计这些系数并从磁力计读数里减掉。但这个框架有一个被作者点明的根本假设裂缝：**T-L 的误差源是静态的**。而现实中电子设备、通信系统、温度相关的传感器磁性都会产生**时变磁场**，直接违反静态假设——这就是为什么经典 T-L 在真实飞行中会残留系统误差。

另一条路线是纯神经网络：已有工作（DeepMAD 等）证明 NN 可以改善标定，但作者指出这些方法的通病是——**不利用系统的固有物理定律**、计算代价高、且在高频时序数据上不鲁棒；而经典 Kalman 滤波虽在线性好，却因底层系统高度非线性且对扰动敏感而受限。

本文把问题重新表述为：**能否用一个"物理信息"的两段式学习框架——先用连续时间网络把混有干扰的磁力计信号解混成纯净的磁异常信号，再用学到的残差反过来更新 T-L 模型的系数，从而同时突破 T-L 的静态假设限制与纯 NN 的黑盒/算力问题？**

## 3. 方法论与核心思路

### 3.1 总体架构

```
机载磁力计原始读数 (10 Hz, 65 特征/飞行)
    │
    ├─► 预处理: z-score 标准化 → 机内磁力计对机外 (Mag 1) 残差化
    │                              → 特征选择 (Spearman / LASSO / XGBoost)
    │
    ▼
[LTC 或 LTC-CfC + NCP(AutoNCP) 线路]
    输入 I(t) → 感知层 → 液体层 (时间连续门控) → 输出层
    │  基线: T-L / LSTM / MLP / CNN
    ▼
预测的滤波信号 ŷ(t)  ≈  纯净磁异常信号 (IGRF 已扣除)
    │
    ▼
第二段: 由残差反推并更新 Tolles–Lawson 模型的系数 → 改进机载磁补偿标定
```

**关键点在于第一段输出的用途**：LTC 网络不只是"做个滤波"，它的输出被用来**更新一个物理标定模型的状态估计**。这就是标题里 "physics-informed" 的落点——网络提供的是 T-L 框架里缺失的时变/非线性部分。

### 3.2 LTC 的连续时间状态方程

LTC 把 RNN 的隐状态演化写成 ODE：

$$
\frac{dx}{dt} = \frac{w}{\tau} + f(x, I, \theta)\,x(t) + A f(x, I, \theta) \tag{1}
$$

其中 $A$ 是偏置向量，$f$ 是以 $\theta$ 参数化的神经网络，$w/\tau$ 是时间常数项。**与标准离散 RNN 的区别在于**：RNN 靠"一步一步的历史依赖"近似连续动力学，LTC 直接在连续时间上建模，**其响应快慢（时间常数）随输入自适应**。

### 3.3 CfC：绕开数值 ODE 求解器

作者明确指出 ODE 类模型的代价：需要数值求解器，训练与推理都慢，数据复杂度越高越糟（自动驾驶、金融时序、物理仿真）。本文采用 **CfC**（Hasani 2022）——用闭式连续深度解替代求解器：

$$
x(t) = \underbrace{\sigma\!\big(-f(x,I;\theta_f)\,t\big)}_{\text{时间连续门控}} \odot g(x,I;\theta_g) + \underbrace{\Big[1-\sigma\!\big(-[f(x,I;\theta_f)]t\big)\Big]}_{\text{时间连续门控}} \odot h(x,I;\theta_h) \tag{2}
$$

其中 $f, g, h$ 是三个 NCP 网络头，$\odot$ 为 Hadamard 积。对时变数据集，$t$ 由时间戳或顺序确定；对采样时序无关的应用，$t$ 在两个超参之间均匀采样。**两个 sigmoid 门分别承担"保持旧状态"与"注入新状态"**，这正是 CfC 相对 LTC 免于数值求解的结构来源。

### 3.4 NCP / AutoNCP 线路

作者沿用 Neural Circuit Policy 范式（模仿 *C. elegans* 神经系统的稀疏结构）：四类递归连接——sensory（接收输入）、inter-neurons（中间）、command（决策）、motor（驱动输出）；突触可兴奋/抑制。与标准网络不同，**每个神经元的动力学都由 ODE 支配**。本文使用 **AutoNCP** 接线：64 个 inter 与 command 神经元，每个试验 1 个输出神经元，用 **PPO** 训练。

### 3.5 数据与特征工程

- **数据集**：USAF–MIT Signal Enhancement for MagNav Challenge，6 条飞行，每条 65 个特征，含 INS、航电、电气、雷达读数。
- **地面真值**：经 T-L 标定后，机尾 "stinger" 上的标量磁力计 **Mag 1** 作为目标信号 $\hat y$。
- **预处理三步**：(1) 全部特征 z-score 标准化；(2) **机内磁力计相对机外 Mag 1 做残差化**，以聚合更多信息；(3) **特征选择**（Spearman 相关 + LASSO 正则 + XGBoost 三种方法联合），最终保留磁力计 4/5 读数、飞行器位置与姿态、INS 速度、高度，以及若干电气量（$V_{BAT1}, V_{BAT2}, CUR_{ACLo}, CUR_{FLAP}, CUR_{TANK}, CUR_{IHTR}$）。
- **抗泄漏设计**：**把每条飞行当作一个样本**，避免同一条飞行内回环片段造成过拟合；用 K 折交叉验证重采样。

### 3.6 实验设置

- 硬件：NVIDIA Tesla T4 GPU，Python 3.10，PyTorch，Optuna 做超参/结构搜索与调优。
- 优化：Adam（$lr=10^{-3}$），MSE 损失，**dropout 50%**，训练 300 epoch，batch size 64。
- 评价指标：RMSE（nT）：

$$
\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(y_i-\hat y_i\right)^2} \tag{3}
$$

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. 1 | $$\frac{dx}{dt}=\frac{w}{\tau}+f(x,I,\theta)x(t)+Af(x,I,\theta)$$ | **LTC 连续时间状态方程**（$w/\tau$ 为时间常数项，$A$ 为偏置） |
| Eq. 2 | $$x(t)=\sigma(-f(x,I;\theta_f)t)\odot g(x,I;\theta_g)+[1-\sigma(-[f(x,I;\theta_f)]t)]\odot h(x,I;\theta_h)$$ | **CfC 闭式状态更新**（双 sigmoid 时间连续门控，无 ODE 求解器） |
| Eq. 3 | $$\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2}$$ | 标定误差度量（nT） |

## 5. 关键成果与贡献

### 5.1 主结果（Table 1，航磁补偿标定误差 RMSE / nT）

| 模型 | Flt1003 [nT] | Flt1007 [nT] |
|---|---:|---:|
| Tolles–Lawson（经典状态估计） | 58.85 | 45.13 |
| LSTM | 41.79 | 42.18 |
| MLP | 30.47 | 26.23 |
| CNN | 26.05 | 30.56 |
| LTC | 20.31 | 22.89 |
| **LTC-CfC** | **18.20** | **19.14** |

- **LTC 相对 T-L 基线降低最多 58% 的补偿误差**（Flt1003：58.85 → 20.31，约 −65%；摘要口径为"up to 58% reduction"）。
- **LTC-CfC 相对 T-L 降低 64%**（两条飞行平均，摘要口径；逐条看 Flt1003 为 −69.1%、Flt1007 为 −57.6%）。
- **LTC-CfC 同时优于 LTC**（18.20 vs 20.31；19.14 vs 22.89）——这正是 CfC"免数值求解器、训练推理更快"这一设计目标的直接体现；由于数据是 10 Hz 等间隔采样，$\sigma(-f\cdot t)$ 中的 $t$ 可取等间隔。
- **相对非 ODE 基线的差距显著**：最好的纯 NN 基线（CNN）为 26.05 / 30.56，LTC-CfC 比它低 30%（Flt1003）到 37%（Flt1007）。
- 作者称该方法"在真实飞行数据上成功检测出弱异常场，且具备显著精度阈值"（Fig. 3 给出 IGRP 修正后 Mag 1 真值与预测信号的对照）。

### 5.2 贡献清单

1. **把物理标定与学习滤波串成两段式框架**：网络输出不直接当结果，而是用于**更新 T-L 状态估计模型的系数**，让经典物理模型获得时变/非线性纠偏能力——这是"physics-informed"的具体落点，也是本文区别于纯 NN 标定工作（DeepMAD 等）的关键。
2. **在真实飞行数据上验证 LTC/CfC 的物理量回归能力**：连续时间网络此前多用于机器人导航与自动驾驶控制，本文验证了它在**纳秒级量级的微弱磁场信号分离**上的适用性，且误差比经典 T-L 低一个数量级（45–59 nT → 18–19 nT）。
3. **暴露了经典 T-L 静态假设的失效**：明确论证时变磁源（电子设备、通信系统、温度相关磁性）违反 T-L 的静态误差源假设，这是作者对该领域的一个实质性诊断。
4. **AutoNCP + 50% dropout 的稀疏线路**在仅有 6 条飞行的小样本下工作（每飞行即一样本 + K 折 CV），给出了一种小样本高维时序的可行配方。

## 6. 局限性与未来展望

### 6.1 作者自陈局限

- **仅两条飞行用于测试**：Table 1 只报告 Flt1003 与 Flt1007，数据集共 6 条飞行；其余 4 条的用途（训练/验证）未在正文中详述，**测试覆盖有限**。
- **计划中但未完成的工作**：作者明确写道"我们计划利用训练中额外学到的基于物理的特征，并在新数据集上做外部验证"——即**外部验证尚未进行**。这是引用其数字时必须标注的前置条件。
- **只给出 RMSE 点估计**：无置信区间、无多次运行的方差报告，**无法判断 18.20 vs 20.31（LTC-CfC vs LTC）这一差距是否超出 run-to-run 噪声**。这一点对本仓尤其关键：本仓历史负结果 NR-08（"SDE-CfC 改进确定性 CfC 形变场"被判为持平而非改进）正是因为 +0.09 dB 落在 σ≈0.05 dB 的方差内。**本文的 LTC-CfC vs LTC 差距（~10%）相对其绝对量级较大，但仍未做重复实验验证。**
- **未给出实时延迟/算力指标**：虽然动机之一是"纯 NN 计算代价高"，但论文**没有报告推理延迟、参数量或内存占用**，"实时可用"只停留在展望表述（"希望看到该工作被应用于实时飞机导航系统"）。

### 6.2 隐含局限（本仓视角）

1. **"物理信息"的实现程度有限**：本文的物理性体现在**用网络输出更新 T-L 系数**这一间接路径上，网络本身仍是纯数据驱动的黑盒。严格说这属于 **physics-guided**（用物理模型约束学习流程）而非 **physics-informed**（把物理先验嵌入网络架构/损失）。摘要中的措辞偏强，引用时建议降级表述。
2. **误差降幅的"基线选择"问题**：64% 的降幅是相对**经典 T-L**。T-L 本身是一个线性模型，而参数量与容量都远大于它的 LSTM/MLP/CNN 也在基线之列——换句话说，"提升"中有相当部分来自"用更复杂的模型替代线性模型"，而非"连续时间动力学"这一特定机制。**论文没有做 LTC-vs-GRU / LTC-vs-CfC 之外的机制消融**，因此不能断言"连续时间"是收益来源。
3. **两个信号源概念容易混淆**：论文标题的"physics-informed"与 AGENTS.md 约束表中 NR-05 的"LFM2.5 不是纯液态模型"属于同一类风险——**不要因为模型名叫 LTC/CfC 就推断其机制完全被验证**。本文的 NCP 线路（AutoNCP）与 ODE 动力学共同参与了结果，**无法分离出"CfC 闭式解"单独的贡献**。
4. **与本仓已有复现资产的关系**：本仓已有 `scripts/experiment_physics_lnn.py` 与 `scripts/experiment_autoncp.py`，而本文的配方（LTC/CfC + AutoNCP + dropout 50% + 物理模型耦合）**与本仓既有实现高度重合**，具备低成本复现条件。但受限于 MagNav 数据集的获取与 NASA/USAF 许可，**完整复现（含真实飞行数据）需外部数据**。
5. **磁场信号的非平稳性未被建模**：论文自己指出温度相关磁性会变，但模型未引入温度作为显式条件或状态变量——LTC 的自适应时间常数部分地覆盖了这一点，但缺少验证。

### 6.3 未来方向

- 利用训练中额外学到的基于物理的特征；
- 在新数据集上做外部验证；
- 落地到实时飞机导航系统。

## 7. 对本仓的意义

**可作为 `lnn/core/variants.py` 的新变体**：**LTC 与 LTC-CfC 并列的双变体**（Eq.1 的 ODE 形式 vs Eq.2 的闭式形式）可作为一组标准的"求解器代价 vs 精度"对照轴加入本仓变体族。这与本仓已有的 CfC 变体形成互补——本仓此前的 CfC 多用于序列分类，本文提供了**在纳秒级物理量回归上的定量对照**。

**可嫁接的 bench 脚本**：
- `scripts/experiment_physics_lnn.py` —— 与本文的"用 LTC 做物理量回归 + 更新下游物理模型"范式最贴近；
- `scripts/experiment_autoncp.py` —— 本文使用 AutoNCP（64 inter + 64 command 神经元、50% dropout），可直接复用本仓既有的 NCP 实现；
- `scripts/experiment_all_variants.py` —— 用于跑 LTC vs CfC vs LSTM 的统一对照。

**提供的新 ablation axis**：(a) **LTC-CfC vs LTC 的差距是否超出 run-to-run 方差**——这直接对应本仓 NR-08 的教训（±0.05 dB 量级的方差门槛），本文的 ~10% 相对差距需要多次运行确认；(b) AutoNCP 稀疏度（dropout 比例）与精度的关系；(c) ODE 求解器 vs 闭式解在**训练/推理时间**上的真实差距（本文只报告精度，未报告时间，本仓可补上这一维）；(d) **物理模型耦合层**（T-L 式状态估计）作为独立模块的贡献——把"物理模块可学系数"与"纯 NN 滤波"拆开对照，直接回答 §6.2 第 2 点的机制归因问题。

**Verdict**：
- **目标正面（TARGET-POSITIVE）**：作为 **"物理信息 + 连续时间网络"在真实飞行数据上的可行性与量级证据**（误差降低 ~60%），以及 AutoNCP 稀疏线路在小样本物理回归上的可用配方，对本仓 `experiment_physics_lnn.py` / `experiment_autoncp.py` 有直接的方法论借鉴。
- **目标依赖（TARGET-DEPENDENT-WITH-NUANCE）**：其精度优势**无法归因到"连续时间"机制本身**——因为缺少与 GRU / LSTM 的机制级对照（仅有容量相当的 LSTM 基线），且"physics-informed"实为 physics-guided。引用时应表述为"**连续时间网络系列（含 CfC）在该任务上优于线性 T-L 与常规 CNN/LSTM**"，而非"LTC 机制本身带来该提升"。
- **负向指标须并列**：无置信区间、无重复实验、无外部验证、无时延指标——**其"64% 降幅"这一数字在外部验证完成前应标注为单数据集、两条测试飞行上的点估计**。
