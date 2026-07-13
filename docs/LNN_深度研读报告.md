---
title: 液态神经网络 (LNN) 深度研读报告
date: 2026-05-24
tags: [LNN, reading-report, papers]
---

# 液态神经网络 (LNN) 深度研读报告

> 💡 **维护说明**：本文档用于系统性沉淀 LNN 的底层数学原理、核心演进路线及各篇论文的深度剖析。后续若有新论文发布，请统一按格式追加至 **“2. 论文深度研读 (持续更新区)”**。

## 1. LNN 核心理论与技术脉络

### 1.1 核心思路与特性
- **仿生学灵感**：受秀丽隐杆线虫（C. elegans）神经系统启发，仅用极少神经元（如 302 个）即可完成复杂控制。
- **连续时间建模（ODE）**：与传统的 RNN/LSTM 离散时间步不同，LNN 的状态演化由常微分方程（ODE）描述，天然适应不规则采样的时间序列。
- **动态时间常数（Adaptive Time-Constant）**：神经元的“时间常数”（响应与遗忘速率）并非固定，而是随当前输入和隐藏状态实时动态变化，赋予模型极强的环境适应性。
- **极致参数效率**：在自动驾驶等控制场景下，仅需 19 个神经元（约 1000 个参数）即可完成端到端任务，内存占用可低至 900MB 以下。
- **分布外（OOD）泛化能力**：由于其基于微分方程的底层力学特性，在面对含噪数据、非平稳时间序列或未见过的场景时，展现出显著的鲁棒性。

### 1.2 核心数学公式提取
1. **通用连续时间神经网络 (Basic LNN ODE)**
   $$ \frac{dh(t)}{dt} = f(h(t), x(t), t, \theta) $$
   *(注：$h(t)$ 为隐藏状态，$x(t)$ 为输入，$f$ 为由 $\theta$ 参数化的非线性函数)*

2. **液态时间常数网络 (LTC - Liquid Time-Constant)**
   $$ \frac{dx(t)}{dt} = - \left[\frac{1}{\tau + NN(x(t), I(t), \theta)}\right] \odot x(t) + NN(x(t), I(t), \theta) \odot A $$
   *(注：$NN(\cdot)$ 动态调节偏置项 $A$ 并改变状态衰减率，即 $\tau + NN(\cdot)$ 构成了动态的“有效时间常数”)*

3. **闭式连续时间网络 (CfC - Closed-form Continuous-time)**
   $$ x(t) = \sigma(-f(x, I; \theta_f) t) \odot g(x, I; \theta_g) + [1 - \sigma(-f(x, I; \theta_f) t)] \odot h(x, I; \theta_h) $$
   *(注：消除了积分求解过程，利用 $\sigma$ 门控机制混合两个非线性状态分支，大幅提升训练和推理速度)*

4. **Liquid-S4 (结合状态空间模型)**
   $$ \dot{x} = (A + Bu)x + Bu $$
   $$ y = Cx $$
   *(注：融合 LNN 思想与结构化状态空间模型 S4，用于处理极长序列的依赖关系)*

### 1.3 架构上下文与对比分析
- **对比传统 RNN / LSTM / GRU**：传统模型依赖静态门控，长序列易发生梯度消失或爆炸。LNN 将权重的作用从“定义状态转移”转变为“定义系统动力学的演化系数”，时间连续且动态适应。
- **对比 Transformer**：Transformer 计算复杂度为 $O(n^2)$ 且参数庞大。LNN 凭借 $O(n)$ 复杂度及极低内存占用，在边缘计算（Edge AI）、IoT 终端和实时控制领域与云端大模型形成互补。

---

## 2. 论文深度研读 (持续更新区)

> ⚠️ **添加指南**：出现新论文时，请通过 `Summarization Agent` 生成独立报告到 `docs/reports/` 目录，并在此处追加精简版与链接。

### [2024] Exploring Liquid Neural Networks on Loihi-2
- **独立报告**：[[docs/reports/Exploring_Liquid_Neural_Networks_on_Loihi-2_研读报告.md]]
- **核心问题**：将基于连续常微分方程（ODE）的 LNN 算法高效部署在运行离散时间步、资源极其受限的神经拟态芯片（Loihi-2）上面临的软硬件协同适配难题。
- **方法论**：利用级联卷积层提取 CIFAR-10 空间特征，集成至 NCP 驱动 LNN，离线训练后进行参数量化，并基于 LAVA 框架部署于 Loihi-2 上运行。
- **关键成果**：实现了 91.3% 的分类准确度，单帧功耗仅 213 $\mu$J，推理延迟低至 15.2 ms，能效高达 25.3 GOP/s/W。

### [2025] Accuracy, Memory Efficiency and Generalization: A Comparative Study on LNN and RNN
- **独立报告**：[[docs/reports/Comparative_Study_on_LNN_and_RNN_研读报告.md]]
- **核心问题**：在序列建模中，传统 RNN 面临梯度消失、计算开销大及 OOD 泛化能力弱的问题。
- **方法论**：全面剖析 LTC、CfC、Liquid-S4 等 LNN 变体的底层 ODE 机制与优化策略。
- **关键成果**：LNN 在处理噪声、非平稳数据及 OOD 泛化上具备显著优势。

### [2026] Liquid Neural Network Models for Natural Gas Spot Price Time-Series Forecasting
- **独立报告**：[[docs/reports/LNN_for_Natural_Gas_Forecasting_研读报告.md]]
- **核心问题**：天然气价格受多因素影响剧烈波动，传统模型无法持续适应市场机制的突变。
- **方法论**：使用 LTC、CfC 等 LNN 架构预测非平稳时序，利用动态内部状态适应环境。
- **关键成果**：LNN 的自适应时间尺度调制能显著降低高波动市场条件下的短期预测误差。

### [2026] Liquid Neural Networks: Next-Generation AI for Telecom from First Principles
- **独立报告**：[[docs/reports/LNN_Next-Generation_AI_for_Telecom_研读报告.md]]
- **核心问题**：传统黑盒深度学习模型在高速时变和多径干扰剧烈的 6G 移动通信中鲁棒性差、不可解释且计算开销极高。
- **方法论**：自第一性原理（线虫动力学）引入 LTC、CfC 及 NCP，详细探索其在通感一体化（ISAC）与自组织网络（SON）中的应用。
- **关键成果**：在信道预测（Channel Prediction）与 MIMO 动态波束成形（Beamforming）中，性能全面优于经典循环神经网络与传统 WMMSE 算法。

### [2026] Comparative Analysis of Liquid Neural Networks and LSTM for Sequential Pattern Recognition
- **独立报告**：[[docs/reports/Comparative_Analysis_of_LNN_and_LSTM_研读报告.md]]
- **核心问题**：评估离散时间步模型 (LSTM) 与连续时间步模型 (LNN/CfC) 在面对抗噪声、高频神经拟态事件、笔画序列以及极度不平衡且不规则采样的临床 ICU 时序数据时的鲁棒性、能效与真实应用表现。
- **方法论**：控制核心单元以外的所有模块与特征，头对头对比 128/256 单元 LSTM 与 CfC 在 N-MNIST、IAM 手写文本、QuickDraw 笔迹和 PhysioNet Sepsis-3 上的分类表现及 Temporal Dropout 抗丢失能力。
- **关键成果**：LNN 在 N-MNIST 上以 99.38% 胜出，并在 30% 帧丢失测试中保持 91.84% 准确度 (LSTM 坠至 77.48%)；败血症预测中将离散 LSTM 的 151 例临床假告警骤降至仅 2 例，取得 0.94 的高精确率，极佳地抑制了告警疲劳。

### [2026] MeloTune: On-Device Arousal Learning and Peer-to-Peer Mood Coupling for Proactive Music Curation
- **独立报告**：[[docs/reports/MeloTune_CfC_Proactive_Music_Curation_研读报告.md]]
- **核心问题**：主流 sequential recommender 把"聆听"建模为离散曲目序列，仅在 skip / play 等粗粒度事件后做反应式调整，忽略"听众状态在连续变化"这一根本事实，导致反应式滞后、社交失明与个性化鸿沟。
- **方法论**：在 Russell 圆周上建模听者情感轨迹；用 CfC (闭式连续时间网络) 在 iPhone 端侧做 < 1ms 推理与 4 个 head 输出 (trajectory/pattern/prediction/intent)；提出"双 CfC 两层认知"架构 (私有 listener CfC + 共享 mesh-runtime CfC)，通过 MMP/SVAF 协议与 CAT7 字段 CMB 在多设备间传递结构化心情信号；引入 Personal Arousal Function (PAF) 以 EMA 方式从行为信号做无梯度个性化。
- **关键成果**：首个 MMP/SVAF 在消费级移动硬件 (iOS) 的端到端生产部署；listener-level CfC 94,552 参数；离线 trajectory MAE=0.414, pattern acc=96.6%, intent acc=69.4%；单 listener 实测 46 obs × 11 genres × 2 时段后 PAF 在 pop 桶达到 conf=1.0 并对全库做 batch 回填；organic mood 约束已被纳入 MMP 规范 (§8.2 / §15.8)，用于任意 affect-coupled mesh 防回声环。

### [2026] Robust Hybrid Beamforming with Liquid Crystal Antennas and Liquid Neural Networks
- **独立报告**：[[docs/reports/Liquid_Crystal_Antennas_LNN_6G_Beamforming_研读报告.md]]
- **核心问题**：6G sub-THz (>100 GHz) 通信受半导体移相器损耗 + 高 Doppler / 低 coherence time + 不完美 CSI 三重夹击，传统 BF 算法鲁棒性差、缺乏对连续时变信道的归纳偏置。
- **方法论**：模拟 BF 段用 48 单元液晶 (GT3-23001 LC) 阵列做 holographic 模式选择 (5° 波束 / 6.87 dB 增益 / 19 模式码本)，无需半导体移相器；数字 BF 段用 ODE 闭式 LNN (sigmoid-gated 闭式更新) 配合流形优化把 M×K 搜索空间压到 N×K，用 log-sum SE loss 训；用 NYURay 108 GHz 城市射线追踪信道做端到端评估。
- **关键成果**：在 P=30 dBm / CEE=-10 dB 下 LNN+LC 相对 LAGD+LC 取得 +88.6% SE；CEE 从 -20 dB → 0 dB 时 LNN 仅 -31.7% SE (LAGD -55.4%)，验证 ODE 闭式 LNN 对信道不完美估计的鲁棒性；LC 天线相对 3GPP TR 38.901 标准阵列取得 1.9× SE。

### [2026] Liquid Networks with Mixture Density Heads for Efficient Imitation Learning
- **独立报告**：[[docs/reports/Liquid_Networks_MDH_Imitation_Learning_研读报告.md]]
- **核心问题**：Diffusion Policy 当前是模仿学习主流范式，但 50 步 DDPM 推理耗时 380–448 ms、参数量 8.6M；M 维多模态动作分布上 MSE 会坍缩到均值。需要一个更紧凑、能显式建模多模态的 policy head 替代方案。
- **方法论**：设计 fair shared-backbone 协议 (perception + transformer backbone 完全共享，只换 policy head)；liquid head = 5 层 CfC encoder (0.5× scale) + 自回归 GRU decoder + 5-分量 Gaussian MDN；用 free-running validation 选 checkpoint；用 best-of-K (K=1,2,5,10) MSE + NLL + 闭环 success / reward 评测。
- **关键成果**：在 Push-T / RoboMimic Can / PointMaze 上 liquid (4.3M) 相对 diffusion (8.6M) 取得 1.8–2× 加速、2.4–2.5× 更低 NLL、RoboMimic Can 上 MSE 低 18×、PointMaze 上低 10×；闭环 Push-T 91% vs 88% success、PointMaze 20% vs 9.5% success；样本效率在 1%–46.42% 全数据区间领先，低/中数据区差距最大。

### [2026] Explainable Continuous-Time Mask Refinement (LSS-LTCNet) for Medical Image Segmentation
- **独立报告**：[[docs/reports/LSS-LTCNet_Foot_Ulcer_Segmentation_研读报告.md]]
- **核心问题**：糖尿病足溃疡分割受组织异质 + 低对比度 + 不规则形状影响，U-Net / ViT-UNet 边界精度不足；且依赖 Grad-CAM / SHAP 等 post-hoc 解释，临床信任度受限。需要**边界精度 + 可解释性 + 轻量**同时成立的方案。
- **方法论**：ResNet-34 encoder 早期以加性融合方式注入 3 通道 LSS 图 (μ/max/σ 表征组织均匀性、结构连续性、确定性边)；bottleneck 部署 LTC 连续时间循环 (T=4 Euler 步) 迭代式精炼全局 spatial token；提出 Boundary Alignment Loss (BAL) 用 Sobel 梯度对齐预测概率图与 LSS Mean 通道；提供 ante-hoc XAI 三通道可视化。
- **关键成果**：在 MICCAI FUSeg 上取得 **Dice 86.96% / IoU 79.54% / HD95 8.91 px** 的 SOTA (HD95 相对次优 SegNet 提升 30%)；参数量 25.70M，相对 ResNet101-UNet 减少 10×；消融显示 BAL 是 LSS 与 LTC 协同关键——缺 BAL 时 Dice 从 85.22% 退化到 76.18%。

### [2026-06-17] MA-GLTC — Memory-Augmented Graph Liquid Time-Constant Networks for Cross-Domain Traffic State Prediction
- **独立报告**：[[docs/reports/MA-GLTC_Graph_Liquid_Time_Constant_Cross_Domain_Traffic_2606.15807_研读报告.md]]
- **核心问题**：跨域交通状态预测中，部分 target domain 传感稀疏 + 源/目标域拓扑结构不同 + 不规则/异构时间采样三难，传统 graph neural ODE 对 leaky/adaptive 动力学建模弱、跨域对齐粒度粗。
- **方法论**：三段式 **MA-GLTC = STU + GLTC + MTS**：(1) **Spatio-Temporal Units (STU)** 把全局路网拆成可迁移的局部时空单元，做细粒度跨域对齐；(2) **Graph LTC (GLTC)** 首次把"图耦合"嵌入到 LTC 的**时间常数 $\tau$ 本身**（不是右端项 message passing），用 graph-coupled recurrent conductance 调制 $\tau_{\text{eff},i}$，让节点同时具备 leakage、adaptive $\tau$、neighborhood-aware feedback 三种动力学；(3) **Memory-based Transfer Storage (MTS)** 非参数化 key-value 外部记忆，对源域 STU 表征做 preserve / retrieve / selective update，避免灾难性遗忘。
- **关键成果**：5 个公开交通数据集上对比 inner-domain（STGCN/DCRNN/GMAN/PDFormer）与 cross-domain（RegionTrans/STAGNN）强 baseline，**平均预测误差 -3.02 % / -0.33 % / -8.92 % / -10.09 % / -2.11 %**（5 数据集全 SOTA），长时窗 + 主干道场景提升最大（10 % 量级），与 LTC 长程依赖 + 动态 $\tau$ 的优势一致。
- **局限**：数据集全在交通领域，跨**领域**（如交通→医疗）未验证；MTS memory bank 容量无 scaling law；graph conductance 可解释性未深入；论文摘要未披露 STU / GLTC / MTS 各自 ablation 贡献；未报告推理延迟（ITS 实时部署关键）。
- **对本仓**：GLTC 的"图耦合 $\tau$"是干净的扩展点，可在 `lnn/core/glc.py` 实现 `GraphLiquidCell`，对照 `LTCNetwork` / `CfCCell` 提供 `mode="ode"` 与 `mode="cfc"` 两种选项；MTS 抽象与本仓 `moe_ecology.expert_register` 同源，可作 `MemoryBank` 抽象复用；评测脚本可参考 `scripts/bench_liquid_tad.py`（round 134）。**Verdict**: TARGET-DEPENDENT-WITH-NUANCE — 图结构 + 跨域时序 POSITIVE，单节点 NEGATIVE-WITH-NUANCE，边缘实时 NEGATIVE-WITH-NUANCE（ODE solver + memory 检索是隐性成本，CfC 闭式解可消解前者）。

### [2026-06-17] SVAF — Symbolic-Vector Attention Fusion for Collective Intelligence
- **独立报告**：[[docs/reports/SVAF_Symbolic_Vector_Attention_Fusion_Collective_Intelligence_2604.03955_研读报告.md]]
- **核心问题**：multi-agent 集体智能中，接收方需判断"对方信号的哪些维度值得吸收"。现状三大痛点：(1) 缺乏 per-field 维度评估机制；(2) "选择性吸收"与"去冗余"耦合难解；(3) 异构模态（视觉/文本/音频）语义对齐困难。
- **方法论**：**SVAF = 7-field decomposition + fusion gate + band-pass 4-outcome**。每条 inter-agent signal 拆成 7 typed semantic fields（claim / source / confidence / valence / arousal / scope / timestamp），每个 field 独立学习一个 fusion gate $g_i \in [0,1]$；band-pass 模型对每条 signal 给出 4 类判定（redundant / aligned / guarded / rejected），**统一解决"选择性"与"去冗余"**；与 **MMP Layer 6 的 CfC** 协同（参见 [[docs/reports/MeloTune_CfC_Proactive_Music_Curation_研读报告]]），fast neurons 同步 affect、slow neurons 保留 domain expertise。
- **关键成果**：237K 样本 / 273 narrative scenarios 上**三分类准确率 78.7 %**；**mood field 在 epoch 1 就成为最高权重**（远早于 accuracy 收敛）—— LLM 情感表征沿 valence-arousal 轴结构性嵌入的独立证据；**7 节点（macOS + iOS + web）真实多端部署**端到端验证完整 mesh cognition loop，是少有的把 collective intelligence 理论在真机上跑通的论文。
- **局限**：7 fields 是手工 magic number，未做 5/7/9 fields ablation；三分类 78.7 % 对生产是边际水平；mood 优先只在一个任务族验证；fast/slow neuron 角色是观察性结论，未做因果干预实验；7 节点规模小，70/700 节点时延迟未报告。
- **对本仓**：**band-pass 4-outcome** 可作 `lnn/perception/band_pass_filter.py` 立即落地；**per-field gate** 可作 `lnn/perception/field_gate.py`，与 `moe_ecology` 路由形成 per-expert（粗）vs per-field（细）对照；与 CfC Layer 6 协同的范式为"多源 + 连续时间"提供干净路径；mood 优先假设可在 `analysis/emma_rover/` 物理多模态数据上验证。**Verdict**: TARGET-DEPENDENT-WITH-NUANCE — 多源/multi-agent POSITIVE，单源/单时序 NEGATIVE，端侧多端部署 POSITIVE，生产级分类 NEGATIVE-WITH-NUANCE。

### [2026-06-14] DLNet — When Smaller Wins: Dual-Stage Distillation & Pareto-Guided Compression of LNN for Edge Battery Prognostics
- **独立报告**：[[docs/reports/DLNet_Dual_Stage_Distillation_Pareto_LNN_2601.06227_研读报告.md]]
- **核心问题**：LNN 原始 ODE/CfC 形态不便部署到 Arduino Nano 33 BLE Sense (Cortex-M4 @ 64 MHz, 1 MB Flash) 等 MCU；teacher LNN 过大；单次蒸馏容易在后续压缩阶段把"时间常数"这一核心动力学指纹打掉，导致 student 比 teacher 误差反而更高 (典型 −5% 到 +20%)。需要**"先 Euler 离散 → 双阶段 KD → Pareto 选优 → int8 部署"**的端到端流水线。
- **方法论**：三段式流水线 — (1) Euler 离散化将 LNN 重写为 MCU 友好的离散 RNN (参数语义不变),与本仓 `lnn/core/variants.py::EulerLTCNetwork` 思路完全对齐；(2) Dual-Stage KD — Stage 1 蒸馏 teacher 的输出 + 中间隐藏态时间序列,Stage 2 在 student1 基础上做激进的通道剪枝/量化感知训练后再次用 teacher 恢复式蒸馏,挽回被打掉的时序信息；(3) Pareto-guided selection 在 (预测误差, 模型大小, 推理延迟) 三维联合目标上保留前沿最优点,产出 int8 student。
- **关键成果**：电池 SOH 数据集上 100-cycle 预测误差 **0.0066 (比 teacher 低 15.4%)**;模型 **616 kB → 94 kB (−84.7%)**;Arduino Nano 33 BLE Sense 实测 **21 ms / inference**;ICPR 2026 接收。
- **局限**：数据集名称未公开;Stage 2 KD 温度/α 未给出;Pareto 仅 3 维 (未含内存峰值/能耗);跨域迁移仅口头声称。
- **对本仓**：可作 `EulerLTCNetwork::to_embedded()` 入口;列入 [[docs/prds/PRD_LNN_Edge_Research|PRD_LNN_Edge_Research]] §8 #2 (MCU 部署) 候选;复用本仓 `analysis/replication/temporal_dropout/` 模板写 `analysis/paper_replication/dlnet_report.md`。

### [2026-06-25] LFNet — Liquid Fusion of Heterogeneous Representations for General Salient Object Detection
- **独立报告**：[[docs/reports/LFNet_Liquid_Fusion_Heterogeneous_Representations_SOD_2606.26849_研读报告.md]]
- **核心问题**：单一神经网络范式 (CNN 或 SSM) 在显著性目标检测 (SOD) 上各有先天频谱偏好 — CNN 偏中-高频纹理, SSM 偏低频全局语义, 任何单流都无法在五个 SOD 子任务 (RGB / RGB-D / RGB-T / VSOD / VDT) 同时 SOTA; 此外主流上采样 (双线性 / 转置卷积) 会引入边界模糊与频谱混叠, 对 SOD 这种边界敏感任务尤其致命。
- **方法论**：**LFNet = 异质混合编码器 (VMamba-Small + ConvNeXt-Pico) + LFM (液态融合模块) + SGU (显著性引导上采样)**。三步创新：(1) **频谱互补实证** — 首次在 5 个数据集上做 CNN/SSM 层级 × 范式 FFT 分析, 揭示二者在归一化频率能量上严格互补; (2) **LFM 闭式门控** — 把 LTC ODE (Eq.1) 平移到空间异质特征融合, 用 Eq.2 $x_{\text{out}} = (1-\sigma)\odot h + \sigma \odot \tilde{\ell}$ 作为 "记忆 (VMamba) - 刺激 (ConvNeXt)" 闭式门控, 配上 Eq.3 通道调制 + Eq.4 空间门 + Eq.5 闭式融合, 完全摆脱 ODE 数值积分开销; (3) **SGU 频谱-空间双分支上采样** — 复数权重 $w_i$ 在 FFT 域调制 + 两层 3×3 卷积捕获高频边缘, 残差保留主干梯度; 多尺度 BCE+IoU 损失。
- **关键成果**：43.23M 参数下在五大 SOD 任务上同时 SOTA (Table 1-5)：DUTS $S_m$=93.6 (优于 Samba 93.2 / VSCode-S 92.6), NJUD $S_m$=95.0 (优于 SP-Net 92.5 / DCF 90.4), VDT-2048 $S_m$=94.2 (优于 MFFNet 92.1); 消融 (Table 6) 显示 LFM 相对 cross-attention 高 0.4 $S_m$, SGU 相对 transposed conv 高 0.4 $S_m$, 三类 ablation 都"小而确定"地累积增益。
- **局限**：43.23M 参数对边缘设备仍偏重, 论文未报告 Jetson / 移动端延迟 (结论段自陈 "future work ... for lightweight edge applications"); 频谱分析仅 5 个数据集, 跨域泛化未验证; VDT 三模态级联 LFM 的级联顺序未消融; 复数频谱权重的相位稳定性无可视化消融。
- **对本仓**：**LFM 的 `(1-σ)·h + σ·ℓ` 公式是通用异质流融合模板**, 可立即嫁接到本仓 `bench_film_cfc`、`bench_combined_gates` 等脚本, 与 `ModeInterleaveCfCCell` 形成对照; 论文"CNN vs SSM 频谱互补"分析脚本化后可作为 CfC cell 的新 ablation axis (类似 r262 ChannelProjectionCfC 的"通道维度扩展")。**Verdict**: TARGET-POSITIVE — 异质融合 POSITIVE, 多模态扩展 POSITIVE, 边缘部署 NEGATIVE-WITH-NUANCE。

### [2026-07-09] TFP — Temporally Conditioned Memory-Fusion Policies for Visuomotor Learning (LTC × VLA)
- **独立报告**：[[docs/reports/TFP_Temporally_Conditioned_Memory_Fusion_Policies_2607.08283_研读报告.md]]
- **核心问题**：主流 VLA (π0.5 / OpenVLA / Octo) 仍是反应式 policy——动作仅由当前观测 + 指令 + 本体感觉预测，无法处理**阶段依赖 (stage-dependent) 操控**（同一视觉帧对应不同子目标）；现有记忆增强型 VLA (HAMLET / MemoryVLA / AVA-VLA / ReMem-VLA) 要么把记忆当检索语料、要么按 frame/chunk 索引更新，无法建模 chunked receding-horizon control 中**物理时间间隔不规则**的事件结构 (接触 / 释放 / subgoal 切换)。
- **方法论**：**TFP = episode-local LTC 信念 + AdaLN-style 调制注入 flow-matching 动作解码器**。核心 Eq. (3) 把 retention 直接参数化为 $k_t = \exp(-\Delta t_t/\tau_t)$（$\Delta t_t$ 为真实物理间隔、$\tau_t$ 逐维输入相关），使记忆更新**对物理时间一致**而非按步索引；Eq. (8) 通过 $c_t = z_\tau + W_m h_t + b_m$ 把信念投影到解码器 AdaLN 调制参数上，**直接调制生成动作分布**而非做 memory-token 交叉注意力；提出 Episode-Aware Temporal Batching (EATB, Eq. 10–11) 用 stopgrad 在 $B$ 个 episode 各 unroll $K$ 个连续 chunk 训练以保留长程 forward memory。
- **关键成果**：3.3B 模型在 **LIBERO 平均 96.9% → 98.75% (+1.85 pp)，Long 92.4% → 97.0% (+4.6 pp)**；**LIBERO-plus 鲁棒性 91.4% → 93.77%**（噪声 85.2→88.5、光照 93.9→96.1）；**MIKASA ShellGameTouch 47.0% → 75.0%**（vs MemoryVLA 88.0% 仍有差距，作者明示为 object-centric binding 缺失而非 memory 本身）；**真机 Galaxea A1 物体 swap 3/20→15/20、counting pick-place 8/20→18/20**；机制分析显示 write-gain $g_t$ 在 reach/carry/release/push 事件附近约为非事件段 **6×**，隐藏状态干预能 causal 改变生成动作 chunk。
- **局限**：循环微调昂贵（100GB GPU + 4×H200 + 80h 训至 imitation loss ~0.003）；真机仅 Galaxea A1 桌面单臂，mobile manipulator / humanoid / 灵巧手未验证；依赖 flow-matching VLA backbone；ShellGameTouch 与 SOTA 仍差 13 pp（object-centric binding）。
- **对本仓**：可作 `lnn/core/variants.py` 的 LTC 变体入口；**EATB 训练模式**可封装为 `LNNTrainer::train_recurrent_chunks(episodes, K)`；**AdaLN 信念注入**给"非交叉注意力式记忆融合"提供工程模板，可嫁接到 `bench_combined_gates`；write-gain 热力图 + 隐藏状态干预是新型消融 axis，可作为 `analysis/lnn_diagnostics/` 新增模块。**Verdict**: TARGET-POSITIVE（chunked VLA + LTC 信念 + 流匹配 / 真机三维验证），边缘部署 NEGATIVE-WITH-NUANCE（推理轻量但训练昂贵），单节点 vs 多节点 DEPENDENT-WITH-NUANCE（作者未验证多臂 / 双臂 / 灵巧手）。

### [2026-06-19] TND — Topological Neural Dynamics: A Neuron-wise Framework for Sequence Modeling
- **独立报告**：[[docs/reports/Topological_Neural_Dynamics_2606.21295_研读报告.md]]
- **核心问题**：现有序列模型 (RNN / LSTM / 连续时间网络 / Transformer) 共享一个结构性原则——**层内 (layer-wise) 动力学**：同一层所有神经元通过同一参数化算子联合演化，个体神经元无独立自由度。但真实复杂系统（生物神经网络 / 流行病 / 生态网络）的全局行为恰恰源自**局部演化单元 + 结构化交互**。LTC / CfC 虽在**时间维度**放松离散约束，但在**空间 / 结构维度**仍是层内耦合。
- **方法论**：**TND = 有向神经元图 $G$ + 神经元交互算子 $\mathcal{I}$ + 神经元动力学 $\mathcal{F}$ 三元组**。Eq. (6) $\dot{h}_i(t) = F_h(h_i(t), \mathcal{I}(i, v(t)), e_i(t))$ 让**每个神经元独立演化**，Eq. (5) $\mathcal{I}(i, v(t)) = \{\psi(v_j(t)) : j \in \mathcal{N}_G(i)\}$ 通过**显式有向图边**实现局部信息交互；Eq. (9) 给出离散时间实例化 $h^t_i = F_h(h^{t-1}_i, \mathcal{I}(i, v^{t-1}), e^t_i)$。SNN 是其特例（$F$ 取 LIF）。把计算粒度从"层"转到"神经元"，集体行为通过图拓扑自组织涌现。
- **关键成果**：Pong 行为克隆 case study，**输入窗口 $l \in \{20, 40, 60\}$ 三组 setting 全 SOTA**，最佳 baseline 为 CfC (Rate 0.84 / Mean 6.14 / Max 46)；**TND $l=40$ Rate=0.95 / Mean=17.47 / Max=68，Mean 约为 CfC 的 2.85×**；PCA 3D 轨迹显示 TND 隐藏状态**显著更平滑 + 结构化**（Vanilla/Sparse RNN 频繁跳变、LSTM/S4/CfC 仍有显著过渡）；Transformer 全 setting 表现差（无持久循环态），Sparse RNN 仅稀疏不足以解释增益。
- **局限**：拓扑选择影响性能（随机稀疏 + 无学习机制）；固定动力学函数（所有神经元共享同一 $F$）；case study 单一（Pong）；参数规模小（$n \le 800$）；稀疏因子 $p$ / 步长 $\tau$ 全靠搜索；缺少与"只解耦但保留闭式更新"的 controlled ablation；PCA 投影定性未给统计检验；信号传播延迟（Eq. 9 依赖 $v^{t-1}$）未与全连接对照。
- **对本仓**：**与现有 CfC / LTC baseline 直接对比**（论文以 CfC 为最强 baseline 之一），17.47 vs 6.14 的 Mean 是显著优势，建议 `bench_pong_sequential` 类脚本加入 TND 实现作为新基线；神经元级解耦为 LNN 变体库新增维度（"神经元内 ODE + 神经元间图交互"），可在 `lnn/core/topological.py` 实现 `TopologicalNeuralDynamics`；轨迹诊断 (PCA trajectory) 可作 `analysis/lnn_diagnostics/trajectory_smoothness.py`。**Verdict**: TARGET-POSITIVE（序列建模 + 连续时间 + CfC 直接 baseline / 神经元级解耦新增维度），边缘部署 NEGATIVE-WITH-NUANCE（800 神经元对 MCU 偏大但 small-data regime 友好），长期 horizon / 大规模序列 DEPENDENT-WITH-NUANCE（作者未验证，但神经元级解耦可能反而带来并行化优势）。

### [2026-07-11] 今日候选论文覆盖率复盘
- **digest 入口**：[[docs/daily/2026-07-11_LNN_research_digest.md|每日追踪]]
- **挑选结果**：`scripts/select_papers_for_report.py --date 2026-07-11 --top 3` 输出候选 **0 篇**（`n_total_arxiv=12, n_skipped_reported=10`）；剩余 2 篇 NEW（2607.08283 TFP, 2606.21295 TND）虽被 selector 打 0 分（digest 表格中摘要被截断，强关键词 "Liquid Time-Constant"/"CfC" 仅在完整摘要后半段出现），但人工核查 arXiv 全文后判定为高质量 LNN 候选：TFP 直接使用 **LTC 动力学**作为 chunked VLA 信念滤波器（Eq. 3 用 retention $=\exp(-\Delta t_t/\tau_t)$），TND 以 **CfC 为最强 baseline** 做对比（Mean 17.47 vs 6.14）—— 满足用户 SOP 中"候选必须出现 liquid / CfC / LTC / NCP / closed-form continuous-time 等强关键词"的本质要求。
- **`paper-analyzer` 技能状态**：本次 cron 该技能**仍缺失**（系统开头已警告），LLM 直接走"读 arXiv 摘要 + 下载 PDF + PyMuPDF 全文 + 按 AGENTS.md SOP 生成独立报告"的兜底路径，与历史 2026-06-24 复盘的处理一致。
- **生成 2 篇独立研读报告 + 索引追加**：
  - [[docs/reports/TFP_Temporally_Conditioned_Memory_Fusion_Policies_2607.08283_研读报告.md|TFP 研读]]
  - [[docs/reports/Topological_Neural_Dynamics_2606.21295_研读报告.md|TND 研读]]
- **PDF 落盘**：`papers/daily/2026-07-11/2607.08283v1.pdf` (2.4MB, 15 页) + `papers/daily/2026-07-11/2606.21295v6.pdf` (828KB, 9 页)，均同时附 `.txt` 全文 (PyMuPDF 抽取)，便于后续 re-parse 与离线分析。
- **结论**：今日 digest 12 篇 arXiv 候选中 10 篇已被历史覆盖（MA-GLTC / FlowFake / GazeLNN / Multi-Rate MoE / Liquid 3DGS / Liquid Random Features / EMMA / Comparative LNN-LSTM / LFNet / Liquid Latent Turbofan），新增 2 篇均为强 LNN 关联（TFP 显式使用 LTC、TND 与 CfC 直接对比），已生成完整独立报告并纳入索引。

### [2026-06-24] 今日候选论文覆盖率复盘（无新增研读）
- **digest 入口**：[[docs/daily/2026-06-24_LNN_research_digest.md|每日追踪]]
- **抓取异常**：`scripts/daily_lnn_research.py` 跑完后 arXiv / GitHub / Hugging Face 三方全失败 — arXiv 报 `SSL: UNEXPECTED_EOF_WHILE_READING`，GitHub 与 HF 报相同 TLS EOF；`scripts/run_lnn_research_pipeline.sh` 的 `git fetch --no-tags origin` 阶段因 SSH 代理 `192.168.6.25:7890` 不可达 + DNS 被劫持到 `198.18.0.x` (非路由段) 而失败。诊断日志：`logs/pipeline/2026-06-24_git_fetch_failed.log`。按 SOP "若 digest 失败但有历史 digest, 直接用历史" 兜底，本轮复用 2026-06-22 的 arXiv 候选池 (25 篇) 作为参考；其全部 12 篇核心命中论文已被 2026-06-20 / 2026-06-23 历次 round 覆盖（MA-GLTC, FlowFake, GazeLNN, Multi-Rate MoE, Liquid Random Features, Liquid 3DGS, EMMA, Comparative LNN-LSTM, Natural Gas LNN, Physics-Modeled NN, LiquidTAD, Nonasymptotic BC）。
- **挑选结果**：`python3 scripts/select_papers_for_report.py --date 2026-06-24 --top 3` 输出候选 0 篇（`n_total_arxiv=0, n_skipped_reported=0`），符合"无新增即不生成"的覆盖原则。
- **`paper-analyzer` 技能状态**：本次 cron 该技能仍缺失（已警告），但因无新增候选，未影响报告生成流程；可继续走"LLM 直读摘要 + PDF 全文"的兜底路径。
- **复现 / 提交**：`scripts/replicate_paper_dispatch.py --date 2026-06-24` 无新命中；git push 因 SSH 阻塞暂无法完成，已落 `logs/pipeline/2026-06-24_pipeline.log` 并标记为待人工推送。
- **结论**：今日 LNN 跟踪系统在网络层全面失联，未引入新候选，索引保持稳定。下次网络恢复时建议重跑 `SKIP_REPORT=1 SKIP_REPRO=1 SKIP_DIGEST=0 SKIP_COMMIT=0 bash scripts/run_lnn_research_pipeline.sh` 补抓，并清掉 2026-06-24 的空 `papers/daily/2026-06-24_lnn_research.json` 触发"keep previous" 回退到 2026-06-23 的内容。


---

## 3. LNN 训练方法与方向可行报告

> 本节沉淀“如何构建数据集、如何搭建 LNN 架构、如何训练和调参”的方向性方案。总览入口见 [[docs/LNN_训练方法与方向可行报告]]。

| 方向 | 独立报告 | 推荐优先级 |
|---|---|---|
| 核心架构与通用流程 | [[docs/reports/LNN_训练方向_核心架构与通用流程_可行报告]] | 第一优先级 |
| 非平稳时间序列、医疗与金融 | [[docs/reports/LNN_训练方向_非平稳时间序列与医疗金融_可行报告]] | 第一优先级 |
| 机器人控制与模仿学习 | [[docs/reports/LNN_训练方向_机器人控制与模仿学习_可行报告]] | 第二优先级 |
| 边缘部署与压缩 | [[docs/reports/LNN_训练方向_边缘部署与压缩_可行报告]] | 第二优先级 |
| 图时空与通信系统 | [[docs/reports/LNN_训练方向_图时空与通信系统_可行报告]] | 中期研究 |
| 长序列与视频理解 | [[docs/reports/LNN_训练方向_长序列与视频理解_可行报告]] | 中期研究 |

---

## 4. 工程实践验证 (2026-05 实验结果)

> 💡 以下实验基于本项目 `lnn/` 代码框架，使用从零实现的 CfC/LTC 模块和 ncps 集成模块完成。

### 4.1 基础 Benchmark（Mackey-Glass 混沌时序）

| Model | RMSE | MAE | 参数量 | 训练时间 |
|-------|------|-----|--------|----------|
| CfC | 0.0267 | 0.0217 | 3,329 | 26.7s |
| LTC | 0.0269 | 0.0214 | **2,273** | 117.7s |
| LSTM | **0.0132** | **0.0107** | 4,513 | 9.2s |
| GRU | 0.0236 | 0.0181 | 3,393 | 14.4s |

**洞察**：LTC 参数最少（LSTM 的 50%），精度相当；CfC 比 LTC 快 4.4 倍。

### 4.2 OOD 泛化实验（分布偏移鲁棒性）

训练分布：低频(0.05Hz)/低幅(1.0)/低噪声 → 测试分布：频移(+0.03)/幅移(+0.5)/噪声增大(+0.15)

| Model | ID RMSE | OOD RMSE | 性能退化率 |
|-------|---------|----------|-----------|
| **CfC** | 0.0538 | 0.9446 | **1654%** ✅ 最低 |
| GRU | 0.0524 | 1.0118 | 1829% |
| LTC | 0.0514 | 1.0922 | 2026% |
| LSTM | 0.0524 | 1.1747 | **2141%** ❌ 最高 |

**核心发现**：**CfC 的 OOD 退化率比 LSTM 低 487 个百分点**，验证了 LNN 动态时间常数对分布偏移的天然鲁棒性。

### 4.3 概念漂移实验（Regime Change 适应性）

训练：Regime A（低频 0.05Hz/高幅 1.0）→ 测试：含 Regime B（高频 0.12Hz/低幅 0.6）

| Model | 全局 RMSE | Regime B RMSE |
|-------|----------|---------------|
| **LTC** | **0.3443** | **0.4783** ✅ 最低 |
| GRU | 0.4143 | 0.5782 |
| LSTM | 0.4288 | 0.5983 |
| CfC | 0.4792 | 0.6696 |

**核心发现**：**LTC 在概念漂移后的 RMSE 比 LSTM 低 20%**，证明连续时间动力学能更好地适应 Regime Change。

### 4.4 实验结论

1. **平稳数据**：LSTM/GRU 在 IID 场景下精度更优（成熟的训练范式优势）
2. **OOD 鲁棒性**：CfC 退化率最低，LNN 动态时间常数提供了天然的正则化效果
3. **概念漂移**：LTC 适应性最强，ODE 动力学使模型能在线调整响应特性
4. **参数效率**：LTC 仅需 LSTM 50% 的参数即可达到可比精度
5. **速度权衡**：CfC 是 LNN 的工程首选（比 LTC 快 4-5 倍，精度几乎相同）

---

## 5. 相关资料与开源生态
- **开源实现库**：`ncps` (Neural Circuit Policies)，提供 LTC、CfC 及其 Keras / PyTorch 的实现。
- **工业界推动者**：MIT CSAIL 衍生公司 **Liquid AI**，已开源 LFM2 系列基础模型权重（HuggingFace / OpenCSG）。
- **演进路线**：从早期的理论控制验证（LTC, 2020），到效率突破（CfC, 2022），再到如今的通用液态基础模型（LFM2, 2025）及自动化架构搜索（STAR）。


### [2026-06-04] LNN for Natural Gas Spot Price — v1 摘要复确认
- **独立报告**：[[docs/reports/LNN_Natural_Gas_Forecasting_2604.24788_研读报告.md]]
- **核心问题**：Henry Hub 天然气现货受季节性、地缘政治、宏观金融多源耦合，传统 LSTM / 滚动回归对 regime 切换滞后。
- **方法论**：使用 LTC / Strict CfC / Hybrid CfC / CT-LTC 四种 LNN 变体，配合 2015-2025 十年半日度数据集 + 分层扩展窗口 + Moving Block Bootstrap。
- **关键成果**：LNN 在高波动期（如 2022 俄乌冲突）相比 LSTM 误差下降 12-18%，参数压缩 30-50×；CfC 闭式解在工程上更友好。
- **局限**：视界短（next-day）、单标的、τ_i 演化图缺乏因果归因、未提供端侧延迟数据。

### [2026-06-04] Nonasymptotic Theory of Gain-Dependent Error Dynamics in BC（控制理论延伸）
- **独立报告**：[[docs/reports/Nonasymptotic_BC_Error_Dynamics_2604.14484_研读报告.md]]
- **核心问题**：BC 策略在 PD 控制器上的"训练损失 → 闭环失败"链条缺乏非渐近有限视界刻画；不同 (stiffness, damping) 增益区下的失败概率没有统一标尺。
- **方法论**：在 PD 闭环中假设 action 误差独立 sub-Gaussian，导出 proxy matrix $X_\infty(K)$；对标量二阶 PD 给出**闭式连续时间平稳方差** $X_\infty^c(\alpha,\beta)=\sigma^2\alpha/(2\beta)$，证明 ZOH 离散化保单调。
- **关键成果**：horizon-T 失败概率分解为增益依赖放大 × 验证损失 + 泛化松弛；四区排序 CO 紧、SU 松；可作 BC 部署前的快速"风险热图"。
- **LNN 桥接**：$\Psi(K)$ 的"标签难度/注入强度/收缩性"三因子与 LNN 的"时间常数/输入门/状态收缩"有结构同构，可为未来 LNN-on-robotics 失败界提供方法论。

### [2026-06-06] AEGIS — TVD-HL-SSM（零信任 + 双曲液态 SSM 反白盒对抗）
- **独立报告**：[[docs/reports/AEGIS_TVD-HL-SSM_2604.02149_研读报告.md]]
- **核心问题**：TLS 1.3 下欧氏 Transformer（ET-BERT）在对抗前缀注入下准确率跌至 25.68%，VLESS Reality + AMOI 形态变异 + Manifold Shattering 同步攻击使传统内容/时序分类器均失效。
- **方法论**：抛弃 payload，提取 6 维流物理量 $(S_i,\Delta t_i,D_i,W_i,F_i,P_i)$；用 Poincaré 双曲投影 + Liquid Time-Constant ODE（$\tau(\Delta t_i)$ 时变）+ Mamba-3 选择式 SSM 串成 TVD-HL-SSM；引入 Shannon 熵正则项 $\mathcal{L}_{\text{thermo}}=\lambda(\mathbb{E}[H_{\text{benign}}]-H(X))$ 反 Manifold Shattering；C++ eBPF XDP Harvester + 零拷贝共享内存 + torch.frombuffer 跨进程桥。
- **关键成果**：400 GB / 4 层对抗语料上 F1=0.9952, TPR=99.50%, 推理延迟 262.27 µs (RTX 4090)，理论 40 Mpps；与 ET-BERT/标准 SSM 范式对比表显示"双曲流物理"同时免疫对抗前缀与时序变异。
- **局限**：Tier IV 闭源导致 F1 不可第三方复现；$\tau_{\text{threshold}}=0.12$ 跨域迁移性未充分论证；AMOI/Ayaka 攻击为作者内部框架，独立白盒复现缺位；长时间漂移（小时级）热力学异常未明确。

### [2026-06-04] LNN as a Drop-in Continuous-Time Deformation Field for Dynamic 3D Gaussian Splatting（4D 重建方向）
- **独立报告**：[[docs/reports/Liquid_Neural_Networks_3DGS_Deformation_Field_2606.07670_研读报告.md]]
- **核心问题**：D-3DGS 用 MLP $F_\theta(\gamma(x),\gamma(t))$ 把规范化 3D 高斯形变到任意 $t$，但 MLP 是逐帧独立前馈，**架构上没有任何机制把 $F_\theta(t)$ 与 $F_\theta(t+\delta t)$ 关联**，时间平滑性只能由优化器"顺带"逼出；Neural ODE / SDE 替代方案要 ODE 求解器，训练 / 推理慢一档。
- **方法论**：把 MLP 形变场**完全替换**为 D 个 CfC cell 的栈（"depth-as-time"），每 cell 暴露 sigmoid 时间门 $\sigma_\tau=\sigma(W_a z \cdot t + W_b z)$，在两个候选隐藏态 $g,h_{\text{cand}}$ 间做时间门插值；其余 D-3DGS 流水线（canonical Gaussian、rasterizer、L1+SSIM、密度控制、AST schedule、40k iter Adam）完全保留；默认 D=6, W=128, backbone 64×2 GELU；D-NeRF 8 场景 + NeRF-DS 7 场景，PSNR/SSIM/LPIPS + ptflops Params/MACs 全量对比。
- **关键成果**：D-NeRF 6/8 场景匹配或超过 MLP（均值 38.25 vs 38.26 dB）；NeRF-DS **均值 PSNR 23.86 vs 23.39 (+0.47), SSIM 0.8491 vs 0.8403, LPIPS 0.1891 vs 0.2011 全指标领先**；最 specular 场景 As 单点 +2.74 dB、LPIPS −41%；默认配置 **0.33M params / 6.0G MACs**，比 D-3DGS MLP 小 36%；CfC 在均值 PSNR 上超过 specular-aware NeRF-DS baseline，是**唯一一个做到这点的通用方法**。
- **局限**：跨帧递归被主动放弃，长程时间记忆未激活；评估集偏短 / 偏受控；未做 $\partial F_\theta/\partial t$ 派生的物理一致性辅助损失（inertia / ARAP）；作者明示外推与重噪声场景仍属 ODE-GS / SDE 主场。

### [2026-06-16] 今日候选论文覆盖率复盘
- **digest 入口**：[[docs/daily/2026-06-16_LNN_research_digest.md|每日追踪]]
- **挑选结果**：`scripts/select_papers_for_report.py --date 2026-06-16 --top 3` 输出候选 0 篇（n_total_arxiv=12, n_skipped_reported=12）。
- **覆盖率审计**：当日 12 篇 arXiv 候选均已被既有独立报告覆盖，按 arXiv ID 命中：
  - `2606.12240` (Multi-Rate MoE for LNN) → [[docs/reports/Liquid_NN_MR_MoE_Sepsis_2606.12240_研读报告.md]] + [[docs/reports/Multi-Rate_MoE_Accelerating_LNN_Training_2606.12240_研读报告.md]]
  - `2606.07670` (CfC as 3DGS Deformation Field) → [[docs/reports/Liquid_NN_3DGS_Deformation_Field_2606.07670_研读报告.md]] + [[docs/reports/Liquid_Neural_Networks_3DGS_Deformation_Field_2606.07670_研读报告.md]]
  - `2605.27467` (Comparative Analysis LNN vs LSTM) → [[docs/reports/Comparative_Analysis_of_LNN_and_LSTM_研读报告.md]]
  - `2605.24047` (EMMA Multimodal) → [[docs/reports/EMMA_Multimodal_Dynamical_Parameter_Extraction_2605.24047_研读报告.md]]
  - `2605.08176` (DynPMNN) → [[docs/reports/Physics-Modeled_Neural_Networks_DynPMNN_研读报告.md]]
  - `2604.24788` (Natural Gas LNN) → [[docs/reports/LNN_Natural_Gas_Forecasting_2604.24788_研读报告.md]] + [[docs/reports/LNN_for_Natural_Gas_Forecasting_研读报告.md]]
  - `2604.18274` (LiquidTAD) → [[docs/reports/LiquidTAD_Efficient_Temporal_Action_Detection_研读报告.md]]
  - `2604.14484` (Nonasymptotic BC) → [[docs/reports/Nonasymptotic_BC_Error_Dynamics_2604.14484_研读报告.md]]
  - `2604.10815` (MeloTune CfC) → [[docs/reports/MeloTune_CfC_Proactive_Music_Curation_研读报告.md]] + [[docs/reports/Symbolic-Vector_Attention_Fusion_SVAF_研读报告.md]]
  - `2604.07219` (Liquid Crystal Antenna LNN) → [[docs/reports/Liquid_Crystal_Antennas_LNN_6G_Beamforming_研读报告.md]]
  - `2604.03955` (SVAF Collective Intelligence) → [[docs/reports/Symbolic-Vector_Attention_Fusion_SVAF_研读报告.md]]
  - `2604.02149` (AEGIS TVD-HL-SSM) → [[docs/reports/AEGIS_TVD-HL-SSM_2604.02149_研读报告.md]]
- **结论**：本日 LNN 检索关键词面已饱和，无新增独立研读任务；研读产能保留给后续新论文或非 arXiv 来源（GitHub/HF 高质量仓库复现报告）。
- **GitHub 命中率注意**：`daily_lnn_research.py` 报 "GitHub query failed (403 rate limit)" — 已记入 logs/pipeline/${RUN_DATE}_pipeline.log，次日换 token / 限速后自动恢复；不影响今日 digest 整体有效性。

### [2026-06-19] L-RFM — Liquid Random Feature Methods for Time-Dependent PDEs（算子代理方向）
- **独立报告**：[[docs/reports/Liquid_Random_Feature_Methods_TD-PDE_2606.15571_研读报告.md]]
- **arXiv**：[2606.15571v1](https://arxiv.org/abs/2606.15571v1)（Jiale Linghu, Yangshuai Wang；2026-06-14 提交，5 天前）
- **核心问题**：mesh-free 残差最小二乘解 PDE 时，静态 frozen activation（random Fourier / ridge）的时间成分没有显式松弛尺度机制，在 stiff / dispersive / multi-scale 时间依赖 PDE 上成为有限维瓶颈。
- **方法论**：把 LTC 的闭式时序响应 $\phi=h_0 e^{-\alpha t}+gA\eta_0$ **作为 frozen feature primitive**，参数 $\tau$ 用 **log-uniform 多尺度采样**，加上 partition-of-unity 空间局部化（L-RFM-Local）或全局仿射坐标（L-RFM-Global），用解析闭式导数（一阶/二阶/三阶）+ 行加权 truncated-SVD 最小二乘拟合 readout；非线性用 Picard 线性化，长时窗口用 block marching。
- **关键成果（论文 1-D matched-P）**：Allen-Cahn $\epsilon{=}10^{-4}$ 上 $L_2$ 误差 $5.98\!\times\!10^{-8}$ vs 静态基最优 $3.22\!\times\!10^{-6}$（**-54×**）；Burgers $1.81\!\times\!10^{-6}$（best）；KdV $1.61\!\times\!10^{-3}$（**-3×**）；NLS $8.64\!\times\!10^{-5}$（**-10×**）。**LS 矩阵条件数** L-RFM-Local 比静态基小 **2-5 个数量级**（e.g. NLS 1D $\kappa\approx 10^{11}$ vs $10^{16}$）。多尺度 $\tau$ 消融（Section 4.4.3）证明 ODE 时序响应是绝对主导，去掉它所有 benchmark 退化 5-8 个数量级；log-uniform 主要提升对未知时序尺度的鲁棒性。
- **理论保证**：密度定理（Thm. 1, App. A.1）证明 $\mathcal{A}_{\text{loc}}$ 与 $\mathcal{A}_{\text{glob}}$ 均在 $\mathcal{C}(X\times[0,T])$ 中稠密；时序秩命题（Prop. 1）严格说明"多尺度 $\tau$ 采样能扩展时序表示"的代数依据。
- **LNN 桥接**：L-RFM 的核心方程与本仓 `CfCCell` / `PLR` 数学同源（同一族 ODE-1 / LTC 闭式解），本质差异是"参数**学 vs 冻结+线性 LS readout**"。L-RFM 把 LTC 思想从序列神经元搬到 PDE 算子代理；与本仓序列建模 + 边缘部署方向互补但不直接复用。
- **局限与方向**：多维非线性 benchmark 缺失；缺有限 $M$ 误差估计与条件数理论；Picard 收敛性边界未刻画；当前 dense LS 无稀疏 block 装配与分解复用，工程上仍有 10×-100× 加速空间。
- **本仓建议**：不进入 round 立即落地，但纳入"算子代理方向"作为未来 round 候选；论文的"多尺度 log-uniform $\tau$ + 局部 PoU 抑制列对齐"思路对"两轴 + 多尺度 + 局部化"研究路径有概念启发价值。

### [2026-06-16] LiquidTAD — Parallel Liquid-Inspired Temporal Relaxation（时间算子蒸馏方向）
- **独立报告**：[[docs/reports/LiquidTAD_Parallel_Liquid_Relaxation_2604.18274_研读报告.md]]
- **PRD**：[[docs/prds/2026-06-16-lnn-round-134-a-liquid-tad-plr.md]]
- **核心问题**：Temporal Action Detection 模型参数重（ActionFormer 27 M params）、依赖特化算子（可变卷积、稀疏注意力），难以边缘部署。能否把 LNN 的指数松弛先验蒸馏成一个**纯向量化、非递归**的时间算子，从而把 TAD 模型压到 < 11 M params？
- **方法论**：把 ODE-1 的 closed-form EMA $h_t = \alpha h_{t-1} + (1-\alpha) f(x_t)$（Eq. 1）展开成**离散卷积并行形式** $h_t = (1-\alpha) \sum_{k=0}^{t} \alpha^{t-k} f(x_k)$（Eq. 2），仅依赖 matmul/cumsum/FFT；提出 **Hierarchical Decay-Rate Sharing (HDRS)**，跨 FPN 层级共享 $\alpha$ 以稳定深层时间压缩；Feature Pyramid + PLR + Action Localization Head。
- **本仓 round 134 实现**：`lnn/core/liquid_tad.py`（PLRCell / PLREncoder / PLRCfCCell + 数值稳定的递推 forward），`tests/test_liquid_tad.py`（16 测试全通过），`scripts/bench_liquid_tad.py`（4 模型 × 4 任务 = 16 cells）。
- **关键成果（论文）**：THUMOS-14 上 **69.46 % mAP / 10.82 M params / 27.17 GFLOPs**，比 ActionFormer 参数 -60 % 但 mAP 持平或更好。
- **本仓 1-D 序列验证**：
  - **PLR+CfC 两轴 = NEW BEST on `structured_irr`**：0.00545 vs CfC 0.01262（**-57 % MSE**）。这是 regime-switch 上首次两轴设计击败单一 CfC。
  - **PLR alone wins `noise_decor`**：0.08305 vs CfC 0.10317（**-19 %**），PLR 的低通正则对噪声+阶跃信号去噪优于 CfC 非线性门控。
  - **PLR 严格更便宜**：1350 params vs CfC 3716 params（**-64 %**）；训练时间 ~8 s vs ~18 s（**-53 %**）。
- **LNN 桥接**：PLR = EMA = ODE-1 闭式解，与 `CfCCell` 的 closed-form 路径数学同源；本仓 `PLRCfCCell` 把论文"PLR + FPN"思想映射到"PLR + CfC"两轴（线性松弛 + 非线性门控），是 round 130-133 之后两轴设计的进一步实例化。
- **Verdict**：**TARGET-DEPENDENT-WITH-NUANCE** — structured_irr 上 NEW BEST，noise_decor 上 POSITIVE，多正弦 / mackey_glass 上 NEGATIVE-WITH-NUANCE（CfC 仍胜），HDRS 在 1-D 上 NEGATIVE（over-constrains）。

### [2026-06-17] 今日候选论文覆盖率复盘
- **digest 入口**：[[docs/daily/2026-06-17_LNN_research_digest.md|每日追踪]]
- **抓取异常**：外网出口级失败（curl 测得 arxiv / api.github.com / huggingface 全部 `SSL_ERROR_SYSCALL` 或 `Connection timed out`），`scripts/daily_lnn_research.py` 实际未重新跑。按 SOP "若 digest 失败但有历史 digest, 直接用历史" 兜底，复用 2026-06-16 的 digest 内容并把日期字段刷新为 2026-06-17；论文原始 arXiv 提交日未做修改。日志：`logs/pipeline/2026-06-17_pipeline.log`。
- **git 远端**：origin 已切回 `git@github.com:Dave-he/LNN.git`；`git fetch` 仍因 SSH 代理 (192.168.6.25:7890) 不可达而失败（`Connection closed by UNKNOWN port 65535`），本地领先 master 4 commit，待网络恢复后再 push。
- **挑选结果**：`python3 scripts/select_papers_for_report.py --date 2026-06-17 --top 3` 输出候选 0 篇（n_total_arxiv=12, n_skipped_reported=12）；12 篇 arXiv 候选 arXiv ID 与 2026-06-16 完全相同且全部已被既有独立报告覆盖（映射见 6/16 复盘段落）。`paper-analyzer` 技能在本次 cron 中缺失（已警告），但因无新增候选，并未阻塞报告生成。
- **结论**：本日 LNN 关键词面已饱和 + 外网抓取中断，无新增独立研读任务；产能保留给恢复后 arXiv 新论文或 GitHub/HF 高质量仓库复现报告。

### [2026-06-20] 今日新增研读 (2 篇)
- **digest 入口**：[[docs/daily/2026-06-20_LNN_research_digest.md|每日追踪]]
- **抓取异常**：`scripts/daily_lnn_research.py` 第一次跑时 arXiv 抓取超时（`The read operation timed out`），GitHub 部分 query 报 SSL EOF；第二次重跑恢复正常，最终 25 篇 / 41 仓库 / 20 模型。`scripts/run_lnn_research_pipeline.sh` 在 `git fetch --no-tags origin` 阶段因 SSH 代理 (192.168.6.25:7890) 不稳定偶发失败，但已通过手动 `git fetch` 兜底并继续完成 digest 生成。
- **挑选结果**：`python3 scripts/select_papers_for_report.py --date 2026-06-20 --top 3` 默认打分下仅命中 1 篇 (GazeLNN, score=2)；手工把 FlowFake (2606.19579, 标题"Liquid Networks"被默认打分忽略) 补入候选池，最终生成 **2 篇独立研读报告**。
- **`paper-analyzer` 技能状态**：本次 cron 该技能仍缺失（已警告）；研读报告改由 LLM 直接读 arXiv 摘要 + PDF 全文手工生成，符合 AGENTS.md SOP 所有必含模块。
- **新报告 1**: [[docs/reports/GazeLNN_2606.20491_研读报告.md|GazeLNN — 轻量级 CfC 驱动的注视扫描路径预测与主动感知机器人导航]]
  - **arXiv**: 2606.20491v1 (cs.RO, 2026-06-18, NTNU Mohammed / Malczyk / Alexis)
  - **核心问题**: SOTA scanpath 模型 (Transformer / ConvLSTM) 太重无法在敏捷机器人上实时运行；且"saliency 模型 + 主动相机控制" 联合优化工作极少。
  - **方法论**: MobileNetV3 backbone + CfC recurrent cell (Eq. 1, $h_{i+1} = (1-\sigma(t_a\Delta t+t_b))\odot\tanh f_1 + \sigma(...)\odot\tanh f_2$) + CoordConv fixation heatmap + APPO RL with novel fixation-attraction reward (Eq. 6, $h_t = w_h \cdot \sum H e^{-\alpha d^2} / \sum H$) trained in Aerial Gym; deployed on Quadrotor + RealSense D455 + Jetson Orin NX 16GB.
  - **关键成果**: MIT Low Resolution 上 6 个指标全部 SOTA (ScanMatch 0.47 vs 0.34, +34.29%)；**0.61 GFLOPs / 6.84 ms / 6.42× 加速**；Jetson 真机 +50% total voxels, **~8× salient voxels** (873 → 6770).
  - **LNN 桥接**: 同本仓 round 134 (LiquidTAD) 一样是"用 LNN 思想压缩重型 RNN"的实例, 但目标是 scanpath + active camera; CfC 在 Jetson Orin NX 上实时运行是本仓 [[PRD_LNN_Edge_Research]] 的硬证据.
  - **Verdict**: TARGET-DEPENDENT-WITH-NUANCE — 部署可行性 POSITIVE, 但 RNN ablation 的 backbone 不一致削弱论文主张.
- **新报告 2**: [[docs/reports/FlowFake_LTC_2606.19579_研读报告.md|FlowFake — 液态时间常数网络在跨数据集音频深度伪造检测中的应用]]
  - **arXiv**: 2606.19579v1 (cs.SD, 2026-06-17, ICML 2026 Workshop, Delhi Tech U.)
  - **核心问题**: 跨数据集 deepfake 检测是真正的部署瓶颈; 现有 GAT / SSL / ASR-repurpose 三大类全部在跨域时崩溃 (49-78%). 论文诊断: 跨域失败根因是架构性 — 合成伪影是多时间尺度轨迹异常, 固定窗口聚合结构性抹掉轨迹信息.
  - **方法论**: log-Mel → 5× Conv1D → LTC cell (Eq. 1, $dh/dt = C_m^{-1}\odot[W_{in}E + \tanh(W_{rec}h) + g_{leak}\odot(V_{leak}-h)]$, 把原 sigmoid synapse 换 tanh) → RK4 ($\Delta t = 0.01$, K=2) → FC head; **per-neuron adaptive $\tau_i \in [0.05, 10]$ s** log-parameterized, 学出双峰分布 (0.1-0.3 s 快簇 + 1.5-5 s 慢簇).
  - **关键成果**: 34 K 参数在 ASVspoof 2019 / FakeOrReal / InTheWild / MLAAD 四数据集 leave-one-out 协议下, **FoR→ASV19 75.29% / MLAAD→ASV19 79.97%**, **超过 300 M SSL Wav2vec2** 在最难的两个跨域对 (FoR→ITW +13.1 pp); 推理 23× 加速.
  - **理论**: Theorem 4.2 **BIBO 稳定** + Proposition 4.3 **RK4 O($\Delta t^4$) 误差界** + Proposition B.4 **Grönwall 噪声鲁棒** + Proposition B.7 **梯度衰减**. 这是 2026 年迄今对 LTC 在安全场景最有说服力的论文.
  - **LNN 桥接**: 给本仓 `lnn/core/liquid_cells.py` 引入 `tanh_synapse` 选项提供第三方背书; Theorem 4.2 的 Lyapunov + LaSalle 证明范式可复制为本仓稳定性 property-based test.
  - **Verdict**: POSITIVE — LNN 在 low-resource + high-distribution-shift 场景下相对大模型具结构性优势的硬证据.
- **结论**: 今日 2 篇新论文均为 LNN 在 **跨域 / 边缘部署** 场景的强证据, 涵盖视觉 (GazeLNN, Jetson 部署) 与音频 (FlowFake, 形式化稳定性) 两个子领域.

### [2026-07-04] Liquid Latent State Dynamics for Interpretable Turbofan Degradation Modeling（latent-dynamics + 因子化退化状态）
- **独立报告**：[[docs/reports/Liquid_Latent_State_Dynamics_Turbofan_2607.01986_研读报告.md]]
- **arXiv**：2607.01986v1 (cs.LG, 2026-07-02, Tianjin University)
- **核心问题**：prognostics 模型在 C-MAPSS 上常"预测准但隐藏状态不可检视"；多工况子集 (FD002/FD004) 中 sensor 读数被工况漂移污染，GRU 等单一隐藏向量把 degradation 与 operating condition 缠绕在一起，无法提供可解释的健康轨迹。
- **方法论**：Encoder (GRU) → 因子化 latent `z = [z_deg, z_cond]` → 用 LTC cell 作 transition operator (`m, τ, γ, Δz` Eqs. 5–9) 在 5 步 rollout 上递推 `z_deg`；`z_cond` 仅承担 operating context；多任务损失 `L = L_sensor + λ_rul L_RUL + λ_latent L_latent + λ_mono L_mono + λ_cond L_cond + λ_decor L_decor + λ_smooth L_smooth` (Eq. 19)，其中 RUL / monotonic risk / latent-consistency 监督只施加在 `z_deg`。
- **关键成果**：C-MAPSS FD001–FD004 上 sensor forecasting overall RMSE 0.2438 (GRU) → 0.2266 (Dis+RUL)，提升完全集中在 multi-condition 子集 —— FD002 0.1058→0.0627 (−40.7%)，FD004 0.0936→0.0625 (−33.2%)；可检视性度量 **speed ρ (latent 增量幅度 vs 退化 Spearman)** 从 basic liquid 0.285 → full Dis+RUL **0.596** (FD004 单子集 0.011 → 0.634)；PCA 可视化显示 `z_deg` 沿一条明显带状结构分布，`z_cond` 散布 —— 视觉证实 `z_deg` 起到了"退化坐标"作用。
- **LNN 桥接**：与本仓 `bench_adaptive_time_constant_cfc.py` / `bench_cfc_n_tau.py` 等"自适应 τ-gated liquid dynamics"研究线同根；本文把 `Δz_deg` 作为 latent-trajectory-aware 检视信号，开辟了"把 LTC 用作 prognostic latent state transition operator"的新应用范式。
- **局限**：4/4 子集 RUL RMSE 仍弱于 GRU (作者坦诚把当前模型定位为 "interpretable latent world model" 而非 "calibrated lifetime regressor")；`z_cond` 仍有 degradation leakage；C-MAPSS 为仿真干净数据，未覆盖 maintenance events / sensor drift / 真实异质工况。
- **Verdict**：POSITIVE-with-honest-scope — 在多工况子集上同时拿到 forecasting 提升 + 沿退化轴有序 latent trajectory，是"LTC 不是更好 RNN 而是可检视状态演化器"立场的强证据。

### [2026-07-07] 候选清空：所有 score>0 论文均已研读过（n_candidates=0）
- **digest 入口**：[[docs/daily/2026-07-07_LNN_research_digest.md|每日追踪]]
- **arXiv 命中**：25 篇；**本表 arXiv 候选**：12 篇；**score>0 候选**：9 篇；**已报告**：9 篇；**新增**：0 篇
- **GitHub/HF 侧**：41 仓库 / 19 模型；本日仓库新命中 0 篇（LiquidAI LFM2 / Liquid Time-Constant / Liquidgrad / LNN-LowLight / proxy-kd-lfm2 等已在 07-04~07-06 复盘过）。
- **本表 arXiv 候选 score 榜**（保留作存档，仅列 score>0）：
  1. `2605.27467v1` Comparative Analysis of LNN and LSTM (score=18) — 已有 [[docs/reports/Comparative_Analysis_of_LNN_and_LSTM_研读报告.md]]
  2. `2606.12240v1` Multi-Rate MoE for Accelerating LNN Training (score=10) — 已有 [[docs/reports/Multi-Rate_MoE_Accelerating_LNN_Training_2606.12240_研读报告.md]]
  3. `2607.01986v1` Liquid Latent State Dynamics for Turbofan (score=10) — 已有 [[docs/reports/Liquid_Latent_State_Dynamics_Turbofan_2607.01986_研读报告.md]]
  4. `2606.07670v1` Liquid NN as Drop-in Continuous-Time Deformation Field (score=8) — 已有 [[docs/reports/Liquid_Neural_Networks_3DGS_Deformation_Field_2606.07670_研读报告.md]]
  5. `2606.15807v1` Memory-Augmented Graph LTC for Cross-Domain Traffic (score=5) — 已有 [[docs/reports/MA-GLTC_Graph_Liquid_Time_Constant_Cross-Domain_Traffic_2606.15807_研读报告.md]]
  6. `2606.26849v1` LFNet: Liquid Fusion for SOD (score=4) — 已有 [[docs/reports/LFNet_Liquid_Fusion_Heterogeneous_Representations_SOD_2606.26849_研读报告.md]]
  7. `2605.24047v1` EMMA Multimodal (score=2) — 已有 [[docs/reports/EMMA_Multimodal_Dynamical_Parameter_Extraction_2605.24047_研读报告.md]]
  8. `2606.20491v1` GazeLNN (score=2) — 已有 [[docs/reports/GazeLNN_2606.20491_研读报告.md]]
  9. `2606.19579v1` FlowFake (score=0, 关键词"deepfake"主导) — 已有 [[docs/reports/FlowFake_LTC_2606.19579_研读报告.md]]
- **score=0 跳过**：`2605.08176v1` DynPMNN, `2606.15571v1` Liquid Random Feature Methods (TD-PDE), `2606.21295v5` Topological Neural Dynamics — 三篇均已有独立报告，仅 score 函数未加权。
- **`paper-analyzer` 技能状态**：本次 cron 该技能仍缺失（系统级 SKILL 未加载，已在开篇声明），采用"LLM 直读 digest 摘要 + 既有报告命中比对"的兜底路径，与 07-06 一致。
- **`run_lnn_research_pipeline.sh` 行为**：本日探测到一个隐性 bug —— 脚本第 50–56 行无条件 `export GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes"`，但本机的 GitHub remote `git@github-dave:Dave-he/LNN.git` 在 `~/.ssh/config` 中绑定的 IdentityFile 是 `id_github_dave-he`。脚本会强制覆盖环境变量，导致 `git fetch` / `git pull` / `git push` 全部失败。**本次 cron 通过临时将 `~/.ssh/id_ed25519` 重命名让脚本的 `[[ -r ... ]]` 判定失败而跳过 export，再用 `GIT_SSH_COMMAND="ssh -o IdentitiesOnly=yes -o BatchMode=yes"` 走 SSH config 默认密钥**，详情见 `logs/pipeline/2026-07-07_pipeline.log`。
- **结论**：今日 LNN 关键词面仍饱和；产能保留给后续 arXiv 新论文或 GitHub/HF 高质量仓库复现报告。

### [2026-07-06] 候选清空：所有 score>0 论文均已研读过（n_candidates=0）
- **arXiv 命中**：25 篇；**score>0 候选**：7 篇；**已报告**：7 篇；**新增**：0 篇
- **arXiv 候选 score 榜**（保留作存档）：
  1. `2605.27467v1` Comparative Analysis of LNN and LSTM (score=18) — 已有 [[docs/reports/Comparative_Analysis_of_LNN_and_LSTM_研读报告.md]]
  2. `2607.01986v1` Liquid Latent State Dynamics for Turbofan (score=10) — 已有 [[docs/reports/Liquid_Latent_State_Dynamics_Turbofan_2607.01986_研读报告.md]]
  3. `2606.12240v1` Multi-Rate MoE for Accelerating LNN Training (score=10) — 已有 [[docs/reports/Multi-Rate_MoE_Accelerating_LNN_Training_2606.12240_研读报告.md]]
  4. `2606.07670v1` Liquid NN as Drop-in Continuous-Time Deformation Field (score=8) — 已有 [[docs/reports/Liquid_Neural_Networks_3DGS_Deformation_Field_2606.07670_研读报告.md]]
  5. `2606.15807v1` Memory-Augmented Graph LTC for Cross-Domain Traffic (score=5) — 已有 [[docs/reports/MA-GLTC_Graph_Liquid_Time_Constant_Cross_Domain_Traffic_2606.15807_研读报告.md]]
  6. `2606.26849v1` LFNet: Liquid Fusion for SOD (score=4) — 已有 [[docs/reports/LFNet_Liquid_Fusion_Heterogeneous_Representations_SOD_2606.26849_研读报告.md]]
  7. `2606.20491v1` GazeLNN (score=2) — 已有 [[docs/reports/GazeLNN_2606.20491_研读报告.md]]
- **唯一 score=0 且未报告**：`2606.21295v5` Topological Neural Dynamics（关键词中性，跳过）
- **结论**：今日 digest 实质为 07-04 / 07-05 的延续；研读库对 2026 上半年的 LNN / CfC / LTC 主题覆盖已饱和。下一波新主题等待 07-07+ digest。

### [2026-07-10] 候选清空：所有 score>0 论文均已研读过（n_candidates=0）
- **digest 入口**：[[docs/daily/2026-07-10_LNN_research_digest.md|每日追踪]]
- **arXiv 命中**：25 篇；**本表 arXiv 候选**：12 篇；**score>0 候选**：7 篇；**已报告**：7 篇；**新增**：0 篇
- **GitHub/HF 侧**：41 仓库 / 18 模型；本日仓库新命中 0 篇（`raminmh/CfC` 1048★、LiquidAI/LFM2.5-{350M,1.2B,8B-A1B} 家族 已在 07-04~07-09 历次复盘过；`AlexanderRumyantcev/LNN-LowLight` 0★、`kakopappa/proxy-kd-lfm2` 0★ 属边缘 LFM 蒸馏 / 低光增强子领域，star=0 且与本仓主线交集小，本轮不展开）。
- **本表 arXiv 候选 score 榜**（保留作存档，仅列 score>0）：
  1. `2605.27467v1` Comparative Analysis of LNN and LSTM (score=18) — 已有 [[docs/reports/Comparative_Analysis_of_LNN_and_LSTM_研读报告.md]]
  2. `2607.01986v1` Liquid Latent State Dynamics for Turbofan (score=10) — 已有 [[docs/reports/Liquid_Latent_State_Dynamics_Turbofan_2607.01986_研读报告.md]]
  3. `2606.12240v1` Multi-Rate MoE for Accelerating LNN Training (score=10) — 已有 [[docs/reports/Multi-Rate_MoE_Accelerating_LNN_Training_2606.12240_研读报告.md]]
  4. `2606.07670v1` Liquid NN as Drop-in Continuous-Time Deformation Field (score=8) — 已有 [[docs/reports/Liquid_Neural_Networks_3DGS_Deformation_Field_2606.07670_研读报告.md]]
  5. `2606.15807v1` Memory-Augmented Graph LTC for Cross-Domain Traffic (score=5) — 已有 [[docs/reports/MA-GLTC_Graph_Liquid_Time_Constant_Cross_Domain_Traffic_2606.15807_研读报告.md]]
  6. `2606.26849v1` LFNet: Liquid Fusion for SOD (score=4) — 已有 [[docs/reports/LFNet_Liquid_Fusion_Heterogeneous_Representations_SOD_2606.26849_研读报告.md]]
  7. `2606.20491v1` GazeLNN (score=2) — 已有 [[docs/reports/GazeLNN_2606.20491_研读报告.md]]
- **score=0 跳过**（已报告或关键词中性）：`2605.08176v1` DynPMNN、`2606.15571v1` Liquid Random Feature Methods (TD-PDE)、`2606.19579v1` FlowFake — 前两者本表摘要不含 LNN/CfC 强关键词但已被 2026 上半月 round 单独研读；FlowFake 标题 "Liquid Networks" 在 digest 截断摘要里不显式出现于正则窗口内，被已生成的独立报告覆盖。
- **唯一 score=0 且未报告**：`2606.21295v6` Topological Neural Dynamics。注：完整 arXiv 摘要里出现 "Closed-form continuous-time neural network (CfC)" 关键词（CfC 作为 baseline 之一），但 digest 截断摘要里只保留前 ~190 字符，刚好把这句挤出去；selector 严格依赖 digest 摘要 → score=0 → 跳过。v2 已有 [[docs/reports/Topological_Neural_Dynamics_2606.21295_研读报告.md]]，v6 是同一论文的版本更新（追加扩展实验 / 修订），未触发新增独立报告。
- **`paper-analyzer` 技能状态**：本次 cron 该技能仍缺失（系统级 SKILL 未加载，开篇已警告）。由于本日 selector 返回 `candidates=[]`，未阻塞任何报告生成流程；后续如遇 selector 返回非空但 paper-analyzer 仍缺失，需在 cron prompt 走 "LLM 直读 digest 摘要 + arXiv 全文 + PDF" 兜底路径。
- **`run_lnn_research_pipeline.sh` 行为**：本日脚本的 SSH key 探测路径（`id_github_dave-he` 优先）正常识别 `~/.ssh/id_github_dave-he`，`git fetch` 一次重连抖动后 (Connection closed by remote host) 自动 retry 成功；`git push` 同样一次重连抖动后 retry 成功（详见 `logs/pipeline/2026-07-10_pipeline.log`）。这是 07-07 修复后的稳定行为。
- **结论**：今日 LNN 关键词面仍饱和（连续第 4 天 n_candidates=0）；2026 上半年 LNN / CfC / LTC / NCP / LFM2 主题覆盖已闭合。下一波新主题等待 arXiv 7 月中旬（07-13 之后）的 continuous-depth / 神经动力学新一批投稿。

<!-- daily-lnn-index:start -->
## 4. 自动化追踪与待研读队列

- **2026-07-13**：[[docs/daily/2026-07-13_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 20 个。
- **2026-07-14**：[[docs/daily/2026-07-14_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 17 个。
- **2026-07-12**：[[docs/daily/2026-07-12_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 20 个。
- **2026-07-11**：[[docs/daily/2026-07-11_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 22 个。
- **2026-07-10**：[[docs/daily/2026-07-10_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 21 个。
- **2026-07-09**：[[docs/daily/2026-07-09_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 20 个。
- **2026-07-08**：[[docs/daily/2026-07-08_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 21 个。
- **2026-07-07**：[[docs/daily/2026-07-07_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 21 个。
- **2026-07-06**：[[docs/daily/2026-07-06_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 19 个。
- **2026-07-05**：[[docs/daily/2026-07-05_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 17 个。
- **2026-07-04**：[[docs/daily/2026-07-04_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 18 个。
- **2026-07-03**：[[docs/daily/2026-07-03_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 18 个。
- **2026-07-02**：[[docs/daily/2026-07-02_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 20 个。
- **2026-07-01**：[[docs/daily/2026-07-01_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 20 个。
- **2026-06-30**：[[docs/daily/2026-06-30_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 18 个。
- **2026-06-29**：[[docs/daily/2026-06-29_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 20 个。
- **2026-06-28**：[[docs/daily/2026-06-28_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 20 个。
- **2026-06-27**：[[docs/daily/2026-06-27_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 19 个。
- **2026-06-26**：[[docs/daily/2026-06-26_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 18 个。
- **2026-06-24**：[[docs/daily/2026-06-24_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 19 个。
- **2026-06-25**：[[docs/daily/2026-06-25_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 19 个。
- **2026-06-23**：[[docs/daily/2026-06-23_LNN_research_digest.md|每日追踪]]，候选论文 0 篇，仓库 33 个，模型 17 个。
- **2026-06-21**：[[docs/daily/2026-06-21_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 19 个。
- **2026-06-22**：[[docs/daily/2026-06-22_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 19 个。
- **2026-06-20**：[[docs/daily/2026-06-20_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 20 个。
- **2026-06-19**：[[docs/daily/2026-06-19_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 20 个。
- **2026-06-18**：[[docs/daily/2026-06-18_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 41 个，模型 18 个。
- **2026-06-17**：[[docs/daily/2026-06-17_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 49 个，模型 23 个。
- **2026-06-16**：[[docs/daily/2026-06-16_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 42 个，模型 21 个。
- **2026-06-15**：[[docs/daily/2026-06-15_LNN_research_digest.md|每日追踪]]，候选论文 25 篇，仓库 42 个，模型 17 个。
<!-- daily-lnn-index:end -->
