---
title: Virtual Temperature Sensors in Power Transformers Using Neural ODEs (arXiv 2608.13260) — 研读报告
date: 2026-09-28
tags: [LNN, paper, neural-ode, virtual-sensor, power-transformer, industrial-monitoring, research-brief, unverified-extraction]
arxiv_id: 2608.13260
pdf: papers/daily/2608.13260.pdf
status: extracted-from-pdf-text
grounding: 本地 PDF 首 600 字符已用 pdftotext 验证 (Berk Hadzhamolla, Alexander Johannes Stasik, Signe Riemer-Sørensen — U. Oslo + SINTEF + NMBU, 2026-08-13, cs.LG)
---

# Virtual Temperature Sensors in Power Transformers Using Neural ODEs

> ⚠️ **Grounding 警告**: 本文基于本地 PDF 第 1 页 (pdftotext -l 1) 提取. 标题与机构已读, **全文未 grounding**. 严禁复述未验证数字.

## 元数据
- **来源**: arXiv:2608.13260v1 [cs.LG] 13 Aug 2026
- **作者**: Berk Hadzhamolla (U. Oslo), Alexander Johannes Stasik, Signe Riemer-Sørensen (SINTEF AS + NMBU)
- **本地 PDF**: [papers/daily/2608.13260.pdf](../papers/daily/2608.13260.pdf)
- **应用**: 电力变压器热点温度软测量 (virtual / soft sensor)

## 核心问题
- **痛点**: 电力变压器内部热点 (hot-spot) 温度是寿命预测关键, 但**直接测量需要埋传感器** (成本高 + 维护难)
- 现有方法: 物理模型 (IEC 60076) 简单但精度差; 数据驱动 (MLP/LSTM) 缺物理一致性
- 作者思路: Neural ODE 把**物理演化** + **数据修正** 联合学

## 方法论与核心思路
- **物理骨架**: 变压器内部温度演化近似指数衰减 + 时变功率输入
  $$T_{\text{hs}}(t) \approx T_{\text{amb}} + (T_{\text{hs}}(0) - T_{\text{amb}}) e^{-t/\tau} + \int_0^t K(\tau, P(t')) P(t') dt'$$
- **Neural ODE 修正**: 学一个**残差速度场** $v_\theta(T, t, P, T_{\text{amb}})$ 加到物理速度场上
  $$\dot{T}_{\text{pred}}(t) = \underbrace{\dot{T}_{\text{phys}}(t)}_{\text{物理骨架}} + v_\theta(T_{\text{pred}}(t), t, P(t), T_{\text{amb}})$$
- 推理: 跑 Neural ODE 沿时间积分, 输出 hot-spot 温度序列
- 与 LNN 关系:
  - 与本仓 [[Physics-Modeled_Neural_Networks_DynPMNN_研读报告]] (DynPMNN) 高度同源 — 都是"物理 + NN 残差"
  - 与 LTC 的 ODE-on-state 思路相似, 但 LTC 是通用 ODE, 本文是带物理约束的 ODE
- **上下文关系**: 紧邻 [[LNN_for_Natural_Gas_Forecasting_研读报告]] (能源时序)

## 核心公式提取 (基于 PDF 标题 + abstract 推断)
- 主方程 (推断):
  $$\dot{T}_{\text{hs}}(t) = \underbrace{\frac{T_{\text{oil}}(t) - T_{\text{hs}}(t)}{\tau_{\text{wind}}}}_{\text{对流传热}} + \underbrace{\frac{P(t)}{\rho V c_t}}_{\text{焦耳加热}} + v_\theta(\cdot)$$
- Neural ODE 解形式:
  $$T_{\text{hs}}(t) = T_{\text{hs}}(t_0) + \int_{t_0}^t \big[ f_{\text{phys}}(T_{\text{hs}}(s), P(s)) + v_\theta(T_{\text{hs}}(s), s) \big] ds$$
- 训练目标:
  $$\mathcal{L} = \frac{1}{N} \sum_i \| T_{\text{hs}}^{(i)} - \hat{T}_{\text{hs}}^{(i)} \|^2 + \lambda_{\text{phys}} \underbrace{\| v_\theta \|^2}_{\text{残差正则}}$$

## 关键成果与贡献 (待 PDF grounding)
- ⚠️ 数字未验证, **仅声明方向**:
- 在公开 / 仿真数据集 (SINTEF 仿真?) 上展示 hot-spot 预测精度
- 与纯物理模型 (IEC 60076) 相比, 显著降低 RMSE
- **不得** 在未读全文前复述具体 RMSE / MAE 数字

## 局限性与未来展望
- (基于 abstract 推断, 未 grounding):
- 依赖公开 / 仿真数据, 真实变压器可能 domain gap
- 物理骨架形式需要 prior 知识 (变压器热模型)

## 本仓具体实现路径 (in-house, 合成数据)

### 适配度
- **高**: 与本仓 DynPMNN 高度同源, 可复用物理骨架 + NN 残差架构
- **可与本仓现有 transformer 热模型合并**: 走 in-house 仿真, 不接真实电网

### 实施步骤
1. **数据生成器** (`lnn/data/transformer_thermal_synth.py` 新建):
   - 合成变压器热模型: 油温 / 热点 / 环境温度 / 功率曲线
   - 控制物理参数 (τ_loss, C_array, thermal_time_const)
   - 加 Gaussian 噪声模拟实测
   - 边界: 仅合成数据, 不接真实电网 / 真实传感器
2. **模型** (在 `lnn/core/physics_node_transformer.py` 新建, 与 DynPMNN 共享骨架):
   - 物理 backbone: $\dot{T}_{\text{phys}}(t) = (T_{\text{oil}} - T_{\text{hs}})/\tau + P/\rho V c$
   - Neural residual: 2-layer MLP (32 hidden), 输入 (T_hs, T_oil, P, T_amb)
   - Solver: torchdiffeq `dopri5`
3. **Loss**:
   - 主: T_hs 预测 MSE
   - 物理正则: λ=1e-3 残差 norm
   - 长期 rollout (T=24h) 累积误差监控
4. **实验队列** (`analysis/transformer_thermal_node/`):
   - 3 物理参数组合 × 5 seed = 15 run
   - 对照: 纯 IEC 60076 / 纯 MLP / 纯 NODE / DynPMNN baseline
   - 关键指标: 1-step MSE / 24h rollout RMSE / 残差 norm
5. **诚实负结果预防**:
   - 若 24h rollout RMSE > IEC 60076 → 进 `analysis/negative_results/`
   - **不得**宣称"Neural ODE 软测量在所有工况下 SOTA"
6. **合规边界**:
   - 仅合成数据 (`lnn/data/transformer_thermal_synth.py`)
   - 沿用 2026-06-09 用户偏好 critical 级 (不操控设备, 不接真实电网)

### 关联 grounding
- 邻近: [[Physics-Modeled_Neural_Networks_DynPMNN_研读报告]] (物理 + NN 残差)
- 邻近: [[LNN_for_Natural_Gas_Forecasting_研读报告]] (能源时序)
- 邻近: [[LNN_Natural_Gas_Forecasting_2604.24788_研读报告]] (同源天然气)
- 邻近: [[LNN_Training_Paradigm_2026_Summer_Cross_Section]]
- 约束: [[docs/LNN_深度研读报告]] §0

## ⚠️ 必须后续 grounding 的项
1. **必须读 PDF 全文**, 验证 abstract 数字与数据集范围
2. 验证 IEC 60076 baseline 的具体精度对比
3. 验证真实 vs 仿真数据集的 domain gap
4. 升级前所有数字声明须标 "待 PDF grounding"

## 维护说明
- 本报告为 **extracted-from-pdf-text 状态**, 待全文 grounding 后升级.
- 升级路径: pdftotext 全文 → 验证 §关键成果数字 → 升级 `status: standard`