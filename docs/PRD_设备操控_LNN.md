---
title: LNN 用于设备操控 — 原理 / 适配性 / 落地案例 / 挑战与方案
date: 2026-06-09
tags: [PRD, LNN, device-control, robotics, drone, industrial, battery, edge]
status: living-document
authors: [heyongxian]
related: [[PRD_LNN_Edge_Research]] §10 #10-22, [[VERIFICATION_RESULTS]] §2, [[SNCP-PPO_Crowdnav_LTC_深度研读报告]], [[EntroLnn_Entropy-Guided_Transformable_LNN_研读报告]]
---

# 设备操控的液态神经网络 (LNN) 专章

> 本专章把仓库 LNN 应用面**从判别/生成/预测/调度正式收口到「设备操控」闭环**,
> 围绕「动态 + 不确定 + 实时性」三大约束,按"原理 → 适配性 → 落地案例 →
> 挑战与方案"四章展开,**在每章末尾给出本仓的复现 / 验证入口**,便于
> 读者直接对照现有 `lnn/core/*.py` 与 `scripts/*` 跑最小可复现实验。

## 0. 动机 — 为什么这一章重要

设备操控(device control) — 机器人、工业控制、无人机、电池 BMS、主动悬架、
协作臂、手术机器人 — 是 LNN 论文**最密集、ROI 最高**的应用面:

- 2021 Hasani et al. (Nature Machine Intelligence) — 6-DoF 自驾小车 LNN 端到端
  视觉 + 控制,比 RNN 基线在分布外扰动下稳定性显著提升
- 2023–2025 RT-LNN / Liquid-IMU 系列 — 四足机器人 legged locomotion,
  MIT/Hyundai/MIT-mini-cheetah 上验证
- 2024 EMMA / Hester / Tanna (IEEE 10826128) — 边缘 CfC vs ODE-LTC Pareto,
  1–5 数量级加速 (Jetson 实测,见本仓 `docs/VERIFICATION_RESULTS.md`)
- 2024–2025 Liquid S4 / Liquid-S5 — 工业控制 / BMS 状态估计
- 2025 Liu et al. (MDPI Sensors 25/10/3090) — Jetson AGX 上 LNN < 10 mW,
  适合电池供电无人系统
- 2026 EntroLnn (Li et al. SAC '26, arXiv 2601.06195) — 锂电池 SoH 在线精化,
  公式与本仓 `LTCNetwork` 几乎同构(见 [[EntroLnn_Entropy-Guided_Transformable_LNN_研读报告]])

仓库的 **35 轮迭代**已沉淀 9 类 LNN 变体、Jetson Orin Nano Super 真机 Pareto
sweep、SNCP-PPO 群导航 curriculum ablation、3-seed multi-seed
verification、EntroLnn 同构研读 — 缺一个"专门面向设备操控"的总览文档。
本 PRD 补这一段。

---

## 1. 原理 — LNN 凭什么能做设备操控

### 1.1 数学骨架:神经 ODE + 输入驱动 + 低秩 wiring

LNN 在数学上是一类**输入驱动的常微分方程 (input-driven ODE)** 神经网络,
三种主流形态都共享同一骨架:

| 形态 | 论文 | 公式 | 本仓对应 |
|---|---|---|---|
| **LTC** (Liquid Time-Constant) | Hasani 2021 NatMI | `dh/dt = -α(h, x) ⊙ h + f(h, x)` | `lnn/core/ltc.py::LTCNetwork` |
| **CfC** (Closed-form Continuous) | Hasani 2022 NatMI | 闭式解近似 ODE 隐式,免去 RK4 | `lnn/core/cfc.py::CfCNetwork` |
| **NCP** (Neural Circuit Policy) | Lechner 2020 NatMI | 受秀丽线杆虫神经连接启发的稀疏 wiring | `lnn/ncps_integration/ncps_models.py` |
| **PDNA-pulse** | Sharmaa 2026 arXiv 2603.00153 | LTC + per-dim pulse gate α | `lnn/core/cfc.py::PDNAPulseHead` |
| **Liquid-S4** | Hasani 2022 arXiv | S4 + 输入依赖 τ | (调研中) |

与 RNN/LSTM 的关键差异:

1. **时间常数是输入依赖的**:`α(h, x) = σ(W_α h + U_α x + b_α)` —
   控制器对**当前观测的"惊讶度"**自适应调整记忆窗口。
   LSTM 的时间常数固定为 `tanh`/`sigmoid` 的组合,缺乏显式可调尺度。
2. **连续深度 (depth) 是可微的**:ODE 求解(RK4 / Euler / Dormand-Prince)与
   残差网络比,训练可解释性、稳定性(类 Lyapunov 论证)更好。
3. **稀疏 wiring 友好**:NCP 的稀疏性(`<5%` 连接)直接降低 Jetson 类边缘
   设备的内存/能耗预算,这与典型 Transformer 的 dense attention 形成对比。

### 1.2 设备操控为什么需要这三件事

- **动态** (dynamic):系统的状态方程 `ẋ = f(x, u, t)` 本身就是 ODE。
  用神经 ODE 直接拟合,参数意义对齐真实物理时间常数(秒级 → 毫秒级),
  便于**模型预测控制 (MPC) 的 horizon 选择**。
- **不确定** (uncertain):扰动、负载变化、参数漂移。LNN 的输入依赖 α 让
  控制器自动"放大/缩小"记忆,面对分布外扰动比固定 τ 的 LSTM 鲁棒。
- **实时性** (real-time):CfC 用闭式解跳过 RK4,`O(1)` 推理步(对比 ODE-LTC 的
  `O(K)` 步迭代)。本仓 `docs/VERIFICATION_RESULTS.md` §1 给出 Jetson
  Orin Nano Super CPU path 上 CfCStyle 40897 步/秒、GRU 280985 步/秒、
  LTC 12561 步/秒的实测数据(1-seed, hidden=8, seq=32)。

### 1.3 形式化:稳定性与万能逼近

- **稳定性**:Hasani 2021 证明,当 `α(h, x) ⊙ h` 项主导且 `f(h, x)` 在原点
  Lipschitz,有界输入必有界输出(类非线性系统稳定判据)。
- **万能逼近**:输入依赖 α 的 LTC 仍能逼近任意紧集上的连续函数
  (Cybenko 1989 类定理 + Hasani 2021 推论)。
- **可微 MPC**:ODE 的可微性让模型预测控制可以**端到端对控制器反向传播**,
  L4 级自动驾驶仿真在 CARLA/CommonRoad 上验证过。

---

## 2. 适配性 — LNN 与设备操控约束的匹配表

设备操控的四大约束(动态 / 不确定 / 实时 / 资源受限)对 LNN 各项优势的
匹配关系,以及**本仓可复现入口**:

| 设备操控约束 | LNN 关键特性 | 本仓实现 | 复现入口 |
|---|---|---|---|
| **动态** (ms–s 时间尺度) | 输入依赖时间常数 `α(h, x)` | `LTCNetwork.ode_step` (`lnn/core/ltc.py`) | `scripts/experiment_timeseries.py` |
| **不确定** (扰动、负载) | ODE 隐式正则 + Lyapunov 稳定 | `CfCCell` 闭式解 (`lnn/core/cfc.py`) | `scripts/experiment_concept_drift.py` |
| **实时** (<10 ms 推理) | CfC `O(1)` 闭式解 | `CfCStyle` (in-house 闭式实现) | `scripts/jetson_lnn_benchmark.py` |
| **资源受限** (Jetson 8GB RAM) | 稀疏 NCP wiring, 1k–2k 参数可达 | `NCPSAutoNCP` (`lnn/ncps_integration/`) | `scripts/scan_emma_rover_hidden_size.py` |
| **可解释** (安全审计) | 神经元级时间常数可视化 | `τ_visualize` (in LTCNetwork) | `scripts/visualize_emma_rover_attention.py` |
| **持续学习** (模型更新) | Transformable LNN (EntroLnn) | 与本仓 LTCNetwork 同构 | (调研已落,代码无,见研读报告) |
| **多模态** (视觉+IMU+力) | Liquid Encoder 拼装 | `LNNImitationPolicy` (`lnn/core/control.py`) | `scripts/experiment_graph_lnn_molecule.py` (类比) |
| **闭环训练** (RL/IL) | PPO + LTC actor-critic | `SNCPPolicyLite` (`lnn/core/sncp_policy_lite.py`) | `scripts/experiment_sncp_ppo_lite.py` |

**对照 RNN/LSTM 的相对优势**(本仓已实证):

- **参数效率**:`LNNImitationPolicy` 在 Tox21-style 任务上 LTC 比 GRU
  少 28% 参数且 AUC 并列 0.754(见 `analysis/molecular/`)。
- **稳定性**:Mackey-Glass 6-seed median MSE 0.0182 (LTC) vs 0.003 (GRU) —
  **诚实负面**:在某些短序列静态任务上 GRU 仍更准,但 LTC 参数 50% 少。
- **Jetson 推理**:GRU 速度 6.9× CfC, CfC 比 LTC 快 3.3× (CPU path, h=8, T=32)
  — **设备操控选型时必须按 latency budget 选, 不要默认 LTC**。
- **Pareto 前沿** (iter#35 3-seed): PDNAPulse h=16 T=32 (1474 params) 
  0.4224 ± 0.0257 胜 CfCStyle h=16 T=32 0.4658 ± 0.0078 by -9.4%。

**关键结论**:LNN 不是 RNN 的"无条件替代品",**是设备操控的 Pareto 维度**。
`论文 claim` 与"本仓实测"之间需用 `docs/VERIFICATION_RESULTS.md` 第 1 节
4-model Pareto 表对账,见 §1.2 末尾的实测数据。

---

## 3. 落地案例 — 4 个最佳设备操控案例

每个案例含:① 任务定义 ② LNN 选型 ③ 本仓可复现入口 ④ 量化目标 ⑤ 已知失败模式。

### 3.1 案例 A:四足机器人 locomotion (Tier-A 复现入口)

**任务**:12-DoF 四足(mini-cheetah 类)在户外草地/沙地/楼梯上 trotting
+gait transition + 受外力扰动 (push 3 N·s) 后 0.5 s 内恢复平衡。

**LNN 选型**:**Liquid-S4 (time-scale mix)** + 输入依赖 τ,3-layer Liquid-S4
hidden=128,act_dim=12。理由:S4 的对角状态空间 + 连续时间常数便于
MPC horizon (50 ms) 调参;τ 可视化后运维可解释。

**本仓可复现入口** (类比,非真机):
- `scripts/experiment_sncp_ppo_lite.py` — PointMassNavLite 与四足 nav 子任务
  同构(actor-critic + LTC),可改 hidden=128、act_dim=12 复现
- `lnn/core/sncp_policy_lite.py::SNCPPolicyLite` — 复用 actor-critic 架构,
  把 2D Gaussian 换成 12-DoF Squashed-Gaussian
- `lnn/data/robotics.py` — SyntheticImitationDataset 已有 6-state imitation
  loader,可作 IL warm-start 数据源

**量化目标**(对照 MIT mini-cheetah Liquid 论文):
- 4-gait transition success ≥ 95% (paper: 96.7%)
- push 3 N·s recovery time ≤ 0.5 s (paper: 0.42 s)
- Jetson Orin Nano CPU path 推理 < 5 ms (50 ms control loop 内)

**已知失败模式**(诚实负面):
- 1 seed lucky:iter#35 3-seed 已撤回 iter#34 1-seed "全局冠军",**真实 3-seed
  CV 6–22%**(见 [[VERIFICATION_RESULTS]] §1),所以量产选型必须 ≥3 seed。
- 高方差警示:GRU h=8 T=32 std=0.134、PDNAPulse h=8 T=32 std=0.120 —
  **这两个 config 不应该作 production 候选**(见 iter#35 §3 解读)。

### 3.2 案例 B:四旋翼无人机视觉-惯性导航 (Tier-A 复现入口)

**任务**:室内无 GPS、稀疏 landmark 下,4K 视觉 + IMU 100 Hz 融合定位
(VIO), 端到端输出 (x, y, z, yaw) 控制指令。挑战:15 min 续航
(< 2 W 平均功耗)、分布外光照突变。

**LNN 选型**:**CfC (闭式) + IMU 时间序列 + 视觉 CNN feature 抽头**。
理由:室内控制 loop 200 Hz (5 ms),CfC `O(1)` 闭式解 + 4k CNN feature 抽头
典型 hidden=64–128,推理 < 1 ms 容易;功耗 < 10 mW 量级(Liu 2025 MDPI 实测)。

**本仓可复现入口**:
- `lnn/data/emma_drone_synth_regression.py` — 已有 EMMA drone synth
  multimodal regression 数据生成器
- `scripts/benchmark_emma_drone_synth.py` — 跑 EMMA drone 任务,
  把 multimodal_physreg 换成 LNN encoder 看稀疏 wiring 效果
- `lnn/core/multimodal.py` + `multimodal_physreg.py` — 多模态 fusion
  模板(Liquid encoder 拼装)

**量化目标**(对照 Liu 2025 MDPI Sensors 25/10/3090):
- 定位误差 ≤ 5 cm @ 室内 50 m 飞行距离
- 功耗 ≤ 10 mW (Jetson AGX, LNN only)
- 光照突变恢复 ≤ 200 ms

**已知失败模式**:
- 多模态 fusion 用 attention 拼装时 LNN encoder 数 ≥ 3 会突破 Jetson CPU 预算
  (见本仓 `scripts/jetson_lnn_benchmark.py` Pareto sweep)
- 闭式 CfC 在长 horizon (T > 64) 上 ODE 假设开始发散,本仓 iter#35 候选
  之一是 T=64/128 压力测试 + nan_count guard

### 3.3 案例 C:工业控制 — 倒立摆 / 电机伺服 / HVAC (Tier-B 复现入口)

**任务**:1-DoF 倒立摆 + DC 电机 + 编码器反馈,200 Hz 控制 loop;
扩展到多电机协调(机械臂关节)、HVAC 风阀 / 制冷压缩机。

**LNN 选型**:**Liquid-S4 with NCP sparse wiring**,hidden=16–32,
τ ∈ [10 ms, 1 s] 多尺度。理由:工业控制要 PID-like 解释性(LNN τ 可视化
代 PID 增益)+ 极小参数(< 1k) 以便部署到 STM32 / Cortex-M4 (无 GPU)。

**本仓可复现入口**:
- `lnn/core/control.py::LNNImitationPolicy` — 复用 LNN + 模仿学习骨架
- `lnn/data/robotics.py` — SyntheticImitationDataset 可生成 6-state 倒立摆
  数据
- `scripts/comprehensive_lnn_experiment.py` — 综合基准,看 LNN 在小参数下的稳定性

**量化目标**:
- 阶跃响应超调 ≤ 5%、调节时间 ≤ 0.5 s
- 抗负载扰动 (1 kg 突变) 恢复 ≤ 0.8 s
- 推理 < 100 µs(Cortex-M4 @ 168 MHz)

**已知失败模式**:
- 本仓 iter#7 / iter#9 诚实负面:小预算 + 固定 lr 下 GRU/LSTM 显著优于
  CfC/LTC;LTC 在 concept_drift 上 catastrophic (MSE 高 +1301% vs LSTM) —
  工业控制**绝不能用纯小预算硬切 protocol**,必须 gradual multi-regime
  + lr warmup(见 iter#10 phase-B 教训)
- 必须跨 ≥3 seed 验证,1 seed lucky 会过度乐观(见 iter#11 N=5 教训)

### 3.4 案例 D:电池 SoH (State of Health) 在线精化 (Tier-A 复现入口)

**任务**:锂电池组(124 cell LFP 18650)充放电循环中,**在线精化** SoH 估计
(CC-CV 充电曲线的 Q-QdV 特征 → RUL 预测)。挑战:① 部署到 BMS MCU(< 64 KB RAM)
② 训练数据少(参考电池 2234 周期,目标电池仅 200 周期)③ 老化漂移 + 工况变化。

**LNN 选型**:**EntroLnn-style Transformable LNN**(静态 LNN 训练 + 部署时
动态精化参数),与本仓 `LTCNetwork` 公式同构(见
[[EntroLnn_Entropy-Guided_Transformable_LNN_研读报告]])。hidden=16–32,
seq_len=128 (128 周期 CC-CV 充电曲线)。

**本仓可复现入口**:
- `lnn/core/ltc.py::LTCNetwork` — 公式与 EntroLnn Eq. 10 几乎同构
  (`dh/dt = -α ⊙ h + tanh(W_h h + ū)`)
- `scripts/experiment_natural_gas_lnn.py` — 类比季节性+漂移的时序预测
  骨架(真实形态合成数据 + 多 backbone 对照)
- 调研报告归档于 `docs/reports/EntroLnn_Entropy-Guided_Transformable_LNN_研读报告.md`

**量化目标**(对照 EntroLnn 论文):
- CFT (Capacity Fade Tracking) MAE ≤ 0.005 (paper: 0.004577)
- EoL (End of Life) 预测周期数误差 ≤ 18 (paper: 18 cycles)
- MCU 推理 < 1 ms (Cortex-M7)

**已知失败模式**:
- "transformable" 在线精化需要 reference + target 双阶段训练数据;
  本仓 `experiment_natural_gas_lnn.py` 是单阶段,需扩展两阶段接口
- 电池数据 MIT-Stanford CC BY 4.0 可下载,但需要预处理对齐本仓时序
  format (mackey_glass → natural_gas → battery 三阶真实形态递进)

### 3.5 4 案例对比表(选型 quick reference)

| 维度 | 案例 A 四足 | 案例 B 无人机 | 案例 C 工业控制 | 案例 D 电池 SoH |
|---|---|---|---|---|
| **控制 loop** | 50 ms | 5 ms | 5 ms | 100 ms (offline) |
| **观测维度** | 12 DoF + IMU | 4K + IMU | 编码器 + 电流 | 电压 + 电流 + T |
| **首选 LNN** | Liquid-S4 | CfC | NCP-sparse LTC | Transformable LTC |
| **hidden** | 128 | 64 | 16–32 | 16–32 |
| **推理预算** | 5 ms (CPU) | 1 ms (CPU) | 100 µs (MCU) | 1 ms (MCU) |
| **训练数据** | 仿真 + 真机 IL | 仿真 + 真机 VIO | 仿真 + 历史日志 | 124 cell × 2234 周期 |
| **本仓入口** | SNCP-PPO 类比 | EMMA drone 模板 | LNNImitationPolicy | LTCNetwork 公式同构 |
| **T 风险** | 1-seed lucky | T>64 ODE 发散 | 小预算硬切 | 双阶段数据 |
| **3-seed?** | 必须 | 必须 | 必须 | 必须 |

---

## 4. 挑战与方案

### 4.1 训练成本与样本效率

**挑战**:LNN 的 ODE 求解(RK4 / DP)在 GPU 训练时比 Transformer self-attention
慢 3–5×;小数据集(< 1k 样本)易过拟合。

**方案**(本仓已落地或候选):
- **用 CfC 而非 LTC**:闭式解省去 RK4 步迭代,本仓
  `scripts/jetson_lnn_benchmark.py` 实测 CfC 比 LTC 快 3.3×(CPU path)。
- **迁移学习**:参考电池(大样本)→ 目标电池(小样本)EntroLnn-style
  transformable 框架,本仓 `experiment_natural_gas_lnn.py` 是单阶段,
  需扩展两阶段接口。
- **数据增强**:CC-CV 充电曲线加时间拉伸 (0.9x–1.1x) + 高斯噪声 (σ=0.5%) +
  片段 drop,本仓 `lnn/data/timeseries.py` 已有 `_make_data` 钩子。
- **预训练 + 微调**:在 mackey_glass / concept_drift 上预训练 LTC,
  然后在目标任务上 fine-tune(类似 NLP 的 pretrain-finetune)。

### 4.2 边缘部署的工程挑战

**挑战**:Jetson Orin Nano Super 8GB RAM、aarch64、driver 12060 < torch 2.11
+cu130 需求,实际只能跑 CPU path;Cortex-M4 64 KB RAM 几乎不可能跑 PyTorch。

**方案**(本仓已实证):
- **CPU path Pareto**:见 `docs/VERIFICATION_RESULTS.md` §1 4-model Pareto 表。
  PDNAPulse h=16 T=32 (1474 params) 0.4224 ± 0.0257 MSE 是 3-seed 冠军。
- **量化**:LNNIM 论文 (Liu 2025) INT8 量化后 < 10 mW,本仓未来工作
  (iter#36+ 候选)。
- **ONNX 导出 + TensorRT**:CfC 闭式解无 ODE 算子,ONNX 友好;LTC 需
  `tf2onnx` 自定义算子(见本仓 daily research 跟踪 ncps #21 + gap_sdk #253)。
- **Cortex-M4 部署**:用 `ncps` 的 TFLite Micro 后端,模型 < 50 KB
  (hidden=16, 量化 INT8)。

### 4.3 实时性 vs 精度的 Pareto

**挑战**:CfC 闭式解快但精度略低于 LTC;LTC 精度高但 RK4 慢 3×。
单一 backbone 难以同时满足 ms 推理 + sub-cm 定位。

**方案**(本仓已沉淀):
- **多 backbone 选型表**:用 `scripts/jetson_lnn_benchmark.py` 跑
  Pareto sweep,看你的 latency budget 落在 Pareto 哪一段。
- **混合架构**:粗定位用 GRU(快)+ 细定位用 CfC(精)+ 仲裁器(> 0.5 m 用
  CfC,否则 GRU),参考本仓 `lnn/core/ensemble.py`。
- **早退 (early-exit)**:CfC 在前 8 step 已稳定就提前 break,T 减少到 8–16
  (本仓 iter#36 候选: T=64/128 压力测试)。

### 4.4 安全性 / 形式化验证

**挑战**:L4 级自动驾驶、医疗机器人、手术刀需要**形式化保证**(Lyapunov
stability、reachability),深度学习黑盒不可接受。

**方案**(行业前沿,本仓未复现):
- **Lyapunov-stable LNN**:Hasani 2021 给出 ODE 稳定性的充分条件
  (α 主导 + Lipschitz),可在训练后做数值验证。
- **可达集分析**:用 neural ODE verification tool (Venus / ReachNN*)
  验证状态空间可达集上界,本仓未来工作。
- **运行时监控**:LNN 神经元级 τ 可视化后,τ 异常突增 → 切换到安全模式
  (PID fallback)。本仓 `lnn/core/ltc.py` 的 `time_constants()` 可作为
  运行时探针。

### 4.5 可解释性与运维

**挑战**:LNN τ 矩阵随时间变化,运维要可视化、漂移检测、A/B 实验。

**方案**:
- **τ 可视化**:本仓 `scripts/visualize_emma_rover_attention.py` 是模板,
  可移植到任意 LNN 任务。
- **漂移检测**:用 CUSUM / Page-Hinkley 监控 τ 矩阵的 Frobenius norm,
  突增 → 触发 re-training。`analysis/sncp_ppo_lite/` 有类似监控数据。
- **A/B 实验框架**:在 `scripts/jetson_lnn_benchmark.py` 的 `--seeds` 多 seed
  跑 baseline,差异 > 2σ 视为显著。本仓 iter#35 已落地 `aggregate_seeds`。

### 4.6 持续学习 (transformable LNN)

**挑战**:设备老化、工况变化,模型必须在线精化但不能灾难性遗忘 (catastrophic
forgetting)。

**方案**(EntroLnn 风格,本仓调研已落):
- **Transformable LNN**:静态 LNN 训练 + 部署时动态 LNN 在线精化,
  公式与本仓 `LTCNetwork` 几乎同构(无 0 障碍)。
- **EWC (Elastic Weight Consolidation)**:在 LNN 损失上加 Fisher 信息矩阵
  约束重要参数漂移,本仓未来工作(iter#37+ 候选)。
- **Replay buffer**:新数据与旧数据按 1:9 比例混合训练,本仓
  `lnn/core/trainer.py` 可扩展。

### 4.7 跨任务通用 backbone 选型

**挑战**:没有"通杀 backbone",必须在每个任务上画 ranking(本仓 iter#12
+ 多次复现都印证)。

**方案**(本仓已沉淀):
- **`scripts/build_backbone_matrix.py`**:自动扫 `analysis/timeseries_ablation/`、
  `analysis/molecular/`、`analysis/sncp_ppo_lite/`、`analysis/jetson/`,
  pivot 出 task × backbone 矩阵,10+ rows × 11 backbones × 4 domains
  (timeseries/molecular/smnist_gap/lra_pathfinder/natural_gas/...).
- **task-conditional ranking**:LSTM 3 wins / GRU 1 win / CfC/LTC 0 wins
  (4 任务),**绝不默认 LNN 通杀**。

---

## 5. 总结与下一步

### 5.1 一句话总结

> **LNN 不是"更准的 RNN",而是"实时 + 鲁棒 + 可解释"的设备操控 Pareto 选项**。
> 选型必须按任务(控制 loop、latency budget、可解释需求)挑 backbone,
> 必须跨 ≥3 seed 验证,必须看 Jetson Pareto sweep 而非 GPU 玩具数。

### 5.2 本仓复现 checklist(读者按此跑通)

```bash
# 1) 4-model 3-seed Pareto sweep(Jetson 真机, 1 min)
python scripts/jetson_lnn_benchmark.py --pareto --seeds 3

# 2) 设备操控 4 case 引用 harness
python scripts/experiment_device_control_cases.py --case all --quick

# 3) 本地部署模拟(不触发真机, 仅 sim:// + filesystem)
python scripts/local_deployment_sim.py --case all --target local_cpu_smoke --quick --steps 8

# 4) SNCP-PPO 群导航 curriculum ablation
python scripts/experiment_sncp_ppo_lite.py --curriculum --ppo-updates-per-stage 20

# 5) 跨 task backbone 矩阵
python scripts/build_backbone_matrix.py --include-jetson

# 6) 周线自动 CI
# .github/workflows/lnn_weekly_verify.yml 周一 03:07 UTC 跑全套
```

### 5.3 后续 iter 候选(由 iter#36 跟踪)

1. **设备操控真机实验桥接**:`experiment_device_control_cases.py`
   4 case 加真机 stub(MIT mini-cheetah simulator / Crazyflie / Gazebo / BMS)
   - 2026-06-24 增量:已先落地 `scripts/local_deployment_sim.py` 作为
     **本地部署模拟**边界层,只生成 `sim://` manifest / audit / budget gate,
     不触发真机接口;使用说明见 [[docs/recipes/local_deployment_simulation]]。
2. **T=64/128 压力测试 + nan_count guard**:防止长 horizon ODE 发散
   (案例 B 已知失败模式)
3. **EntroLnn 公式落地**:`LTCNetwork` 加 `transformable_mode`,实现参考电池
   训练 + 目标电池在线精化(案例 D)
4. **Lyapunov 数值验证工具**:在训练后加 `verify_stability(model, X)`,
   形式化 α 主导 + Lipschitz 条件(挑战 4.4)
5. **Cortex-M4 部署 stub**:用 `ncps` TFLite Micro 后端,案例 C/D 真部署
6. **Jetson CUDA 路径**:等 driver 升级或换 Jetson-packaged torch wheel,
   重跑 Pareto sweep 拿 GPU 数据(挑战 4.2)

### 5.4 关联文档

- [[PRD_LNN_Edge_Research]] §10 #10-22 — 本专章在 PRD 表格中的条目
- [[VERIFICATION_RESULTS]] §1 — Jetson 4-model 3-seed Pareto 实测表
- [[VERIFICATION_RESULTS]] §2 — 设备操控 4 case harness 数据(本 iter 新增)
- [[SNCP-PPO_Crowdnav_LTC_深度研读报告]] — 案例 A 群导航公式复用参考
- [[EntroLnn_Entropy-Guided_Transformable_LNN_研读报告]] — 案例 D 公式同构证明
- [[LNN_深度研读报告]] — 9 变体实现摘要与论文引用

---

## 附 A:本仓 LNN 设备操控相关代码 map

```
lnn/core/
  ├── ltc.py                 # LTCNetwork — 案例 A/C/D 基础
  ├── cfc.py                 # CfCNetwork + PDNAPulseHead — 案例 B/A 基础
  ├── control.py             # LNNImitationPolicy — 案例 C IL 入口
  ├── sncp_policy_lite.py    # SNCPPolicyLite — 案例 A 群导航 actor-critic
  ├── multimodal.py          # 案例 B 多模态 fusion 模板
  ├── multimodal_physreg.py  # 案例 B 物理参数回归
  └── liquid_neuron.py       # 神经元级 τ 可视化(挑战 4.5)

lnn/data/
  ├── robotics.py            # SyntheticImitationDataset — 案例 C 倒立摆数据
  ├── emma_drone_synth_regression.py  # 案例 B 无人机多模态
  ├── emma_rover_regression.py        # 案例 A 四足类比
  └── natural_gas_generator.py        # 案例 D 季节性+漂移类比

scripts/
  ├── experiment_device_control_cases.py  # ★ 本 iter 新增 — 4 case 引用 harness
  ├── local_deployment_sim.py              # 本地部署模拟 — sim:// manifest + audit + budget gate
  ├── experiment_sncp_ppo_lite.py          # 案例 A 群导航
  ├── benchmark_emma_drone_synth.py        # 案例 B 无人机
  ├── scan_emma_rover_hidden_size.py       # 案例 A hidden 扫描
  ├── jetson_lnn_benchmark.py              # Jetson Pareto(本专章核心数据源)
  ├── build_backbone_matrix.py             # 跨 task 矩阵(挑战 4.7)
  └── verify_all_models.py                 # 9 变体 1-cliff smoke
```

## 附 B:本专章使用本仓 35 轮迭代的关键数据

- iter#7 / iter#9: 1-seed 教训 — 不能信 single-seed 数字
- iter#10 phase-B: 1-seed 逆转,CfC 胜 LSTM 需 gradual + warmup
- iter#11: N=5 教训 — 1 seed lucky 会过估 14.7%, 真实 3-seed 差 −9.4%
- iter#26 / iter#27: SNCP-PPO curriculum ablation, 设备操控 RL 入口
- iter#33: Jetson 2-model Pareto, CfC 0.470 vs GRU, 12.3% 精度差
- iter#34: Jetson 4-model, PDNAPulse h=8 T=32 0.401 1-seed 冠军(后撤回)
- iter#35: Jetson 4-model 3-seed, PDNAPulse h=16 T=32 0.4224 ± 0.0257 真实冠军
- iter#31: RLSTG 黎曼流形扩展(调研)
- iter#34: EntroLnn 公式同构(案例 D 入口)
- iter#35: Retinal LNN 视神经假体(案例 B 视觉扩展调研)
