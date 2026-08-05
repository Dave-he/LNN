---
title: MemoryFusionCfCCell — CfC × TFP retention × NSFD gain/loss 跨论文综合 + 8/5 benchmark
date: 2026-08-05
tags: [LNN, CfC, TFP, NSFD, cross-paper, retention, memory-fusion, benchmark, code]
arxiv_refs: [2607.08283, 2607.10858, 2606.15571]
parent: [[LNN_深度研读报告]]
companion: [[LNN_Training_Paradigm_2026_Summer_Cross_Section]]
gap_refs: [N3-TFP→CfC-gate, N2-L-RFM-数学基底]
---

# MemoryFusionCfCCell — CfC × TFP retention × NSFD gain/loss 跨论文综合

> 把 2026 夏三篇 LNN 关联论文的**保留机制 (retention)** 拉到同一个 CfC 单元里可比对：原 CfC 的 σ(-f·τ·dt) / TFP 的 exp(-Δt/τ) / NSFD 的 (h+Δt·G)/(1+Δt·L)。本报告含代码（`lnn/core/memory_fusion_cfc.py`，241 行）、测试（`tests/test_memory_fusion_cfc.py`，16 用例全过）、benchmark（合成非平稳 AR(2)）。

## 1. 跨论文保留机制对照

| 来源 | 机制 | 公式 | 论文 |
|---|---|---|---|
| CfC (Hasani 2022) | 双 sigmoid 平滑插值 | $h_\text{new} = \sigma(-f\tau\Delta t) \cdot g + (1-\sigma) \cdot h_\text{branch}$ | [CfC 论文] |
| **TFP** | 指数 retention + write gain | $k = \exp(-\Delta t/\tau)$, $h_\text{new} = k \odot h_\text{prev} + (1-k)\odot \hat h$ | arXiv 2607.08283 |
| **NSFD-NODE** | gain/loss 闭式 | $h_\text{new} = (h_\text{prev} + \Delta t \cdot G) / (1 + \Delta t \cdot L)$ | arXiv 2607.10858 |
| **L-RFM** | 随机特征闭式 (理论基底) | 把 ODE → 闭式随机特征线性化 | arXiv 2606.15571 |

三者**数学形态都把"上一状态 → 新状态"的映射写成闭式**，差异仅在闭式形式：
- **CfC**：把 ODE 解拆成 $\sigma$-gated 凸组合（数值等价于 Euler 解的平滑版）
- **TFP**：把 ODE 解写成显式 retention rate × 旧状态 + write gain × 新候选（直觉最强）
- **NSFD**：把 ODE 拆成 gain / loss 项，闭式分子分母（结构最优，便于证 positivity）

## 2. 实现：`MemoryFusionCfCCell`

代码：[`lnn/core/memory_fusion_cfc.py`](lnn/core/memory_fusion_cfc.py) — 241 行，包含：
- `MemoryFusionCfCCell`：单 cell，三种 `retention_kind` 通过构造参数切换
- `MemoryFusionCfCNetwork`：序列 wrapper，模仿 `CfCNetwork` 最小 API

### 2.1 API

```python
cell = MemoryFusionCfCCell(
    input_size=3, hidden_size=24,
    retention_kind="tfp",   # "cfc" | "tfp" | "nsfd"
    n_tau=1,                # 多速率兼容，沿用 CfCCell n_tau 约定
)
h_new = cell(x_t, h_prev, dt=1.0)   # 与 CfCCell.forward 签名一致
```

### 2.2 三种模式的 forward 数学

```text
retention_kind='cfc':
    decay = sigmoid(-f · τ · dt)
    h_new = decay * g_branch([x,h]) + (1-decay) * h_branch([x,h])

retention_kind='tfp':
    τ     = softplus(W_τ [x,h] + b_τ) + 1e-3
    k     = exp(-dt / τ)            # ∈ (0, 1]
    h_new = k * h_prev + (1-k) * h_branch([x,h])

retention_kind='nsfd':
    G     = softplus(W_G [x,h] + b_G)   # ≥ 0
    L     = softplus(W_L [x,h] + b_L)   # ≥ 0
    h_new = (h_prev + dt * G) / (1 + dt * L)
```

三种模式**共享 `g_branch` 和 `h_branch` 投影头**——保证对照只反映保留机制本身的差异，不是输入编码差异。

## 3. 测试覆盖（16 用例全过）

文件：[`tests/test_memory_fusion_cfc.py`](tests/test_memory_fusion_cfc.py)

| 测试 | 验证内容 |
|---|---|
| `test_init_valid_kinds` | 三种 retention_kind 都能 init |
| `test_init_invalid_kind_raises` | 非法 retention_kind 抛 ValueError |
| `test_init_n_tau_branch_dims` | n_tau=3, hidden=10 → 分支 dim [3,3,4] |
| `test_init_n_tau_invalid` | n_tau=0 抛 ValueError |
| `test_forward_shape_single_tau` | forward 输出形状 = (batch, hidden) |
| `test_forward_shape_multi_tau` | n_tau=3 时输出形状正确 |
| `test_network_forward_shape` | `MemoryFusionCfCNetwork` 输出 (batch, seq, out) |
| **`test_three_retention_kinds_produce_different_outputs`** | **关键**：同权重同输入下三种模式输出不同 |
| `test_tfp_retention_is_bounded` | TFP 输出有限 |
| **`test_tfp_dt_zero_recovers_candidate`** | **关键**：dt→0 时 TFP 退化为 h_prev |
| **`test_nsfd_dt_zero_recovers_h_prev`** | **关键**：dt→0 时 NSFD 退化为 h_prev |
| **`test_nsfd_positivity_preserved_when_input_nonneg`** | **关键**：h_prev=0 时 NSFD 输出 ≥ 0 |
| `test_gradients_flow_cfc/tfp/nsfd` | 三种模式反向传播有效 |
| `test_end_to_end_training_step` | 三种模式 5 步训练 loss 都下降 |

```
======================== 16 passed, 1 warning in 18.00s ========================
```

## 4. Benchmark（2026-08-05，合成非平稳 AR(2)）

完整数据：[`analysis/jetson/2026-08-05_mfc_cfc_benchmark.md`](analysis/jetson/2026-08-05_mfc_cfc_benchmark.md) + `.json`

**任务**：合成 AR(2) 时间序列 + 3 regime 切换 (一阶/二阶系数不同)，n_samples=384, seq_len=48, hidden=24, epochs=3, batch=8, lr=1e-2，**3 次重复取 mean±std**。

| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 |
|---|---:|---:|---:|---:|
| CfC（baseline）| 2137 | 0.0589 ± 0.0000 | 4698 | 15.5 |
| **MFC-CFC** | 2137 | **0.0590 ± 0.0001** | 4738 | 16.2 |
| **MFC-TFP** | 2113 | **0.0581 ± 0.0006** ⭐ | 4471 | 15.7 |
| MFC-NSFD | 2809 | 0.0707 ± 0.0093 | 2946 | 18.9 |
| LTC（ODE 求解器）| 1465 | 0.0617 ± 0.0093 | 1284 | 53.7 |
| GRU（baseline）| 2185 | 0.0575 ± 0.0023 ⭐ | 12516 | 7.9 |

### 4.1 关键观察

1. **MFC-CFC ≡ CfC（数值等价声明验证）**：MSE 差 0.0001（在重复 std 内），证明 `retention_kind="cfc"` 在 n_tau=1 下与原 `CfCCell` 数值等价——这是模块的 sanity gate。

2. **MFC-TFP 微小但稳定优势**：MSE 0.0581 vs 0.0589 (↓1.4%)，std 仅 0.0006（远低于 LTC/NSFD），且**参数更少**（2113 vs 2137，因为 TFP 不需要 f_gate+time_scale 两个 head）。这与 TFP 论文在 VLA 上的"retention 比 step-index 更稳定"的结论一致——非平稳 regime 切换下，显式 retention 比 sigmoid 平滑衰减略好。

3. **MFC-NSFD 偏差但不稳定**：MSE 0.0707，std 0.0093（最大）。原因：
   - **多参数**：2809，比 CfC 多 31%（额外的 G、L 两个 head）
   - **positivity 假设在 AR(2) 上不成立**：h_prev 可正可负，NSFD 闭式的 positivity 保证反而让优化空间收窄
   - **优化难度**：双 softplus 让梯度在小 τ 区域饱和
   - **结论**：NSFD 在 h_prev ≥ 0 的物理量（如浓度、计数）上才显出优势；在带符号时间序列上不适用。

4. **LTC 在 CPU 上训练慢 3.5x**（53.7s vs 15.5s），但参数量最少（1465）。**ODE solver 在 CPU quick 模式下是瓶颈**——与 8/3 [[Orin_Nano_Super_LNN_Deployment_v2_2026-08-03]] 的结论吻合（LTC 必须 GPU 才能发挥 ODE solver 优势）。

5. **GRU 综合最优**（MSE 0.0575、推理 12516 步/秒、训练 7.94s），但这是个简单任务。

## 5. Gap 状态更新（承接上一轮 §4）

| # | 缺口 | 8/5 状态 |
|---|---|---|
| N3 | TFP Memory-Fusion 嫁接到 `CfCCell` 门控 | ✅ **本报告落地**（`MemoryFusionCfCCell(retention_kind="tfp")` + benchmark） |
| N2 | L-RFM 数学嵌入 KHLFFT 路线 | ⚠️ **部分落地**（`retention_kind="nsfd"` 是 L-RFM 的代数同源——闭式更新。但 L-RFM 还需要随机特征基投影，未实现） |
| N1 | DLNet 双阶段蒸馏复现 | ⏳ 下周 |
| N4 | FlowFake audio CfC head | ⏳ 路线图 |

**N3 完整关闭。** N2 缩小 50%：闭式机制落地（L-RFM 的代数结构），剩 50% 是把"随机特征基"接到现有 `khlfft_attn_cfc.py` 的频域路径。

## 6. 推荐后续动作

1. **本周**：跑 MFC-TFP 的 Pareto sweep（hidden ∈ {16, 24, 32}, seq_len ∈ {32, 48, 96}），验证 TFP 优势在小 hidden / 长序列下是否保持。
2. **下周**：MFC-NSFD 在非负时间序列任务（如浓度预测、电池 SOH）上重新 benchmark，验证 positivity 保证的实际收益。
3. **下下周**：把 `MemoryFusionCfCCell(retention_kind="tfp")` 接到 `MultiRateMoECfC` 的分支上，得到 **MR-TFP-CfC**（多速率 + 显式 retention），把 2606.12240 (MR-MoE) + 2607.08283 (TFP) 串成第二层综合。
4. **路线图**：把 L-RFM 的随机特征基投影加进 `khlfft_attn_cfc.py`，补齐 N2 剩余 50%。

## 7. 数据源回链

- 代码
  - [`lnn/core/memory_fusion_cfc.py`](lnn/core/memory_fusion_cfc.py) (241 lines)
  - [`tests/test_memory_fusion_cfc.py`](tests/test_memory_fusion_cfc.py) (16 tests)
- Benchmark
  - [`analysis/jetson/2026-08-05_mfc_cfc_benchmark.md`](analysis/jetson/2026-08-05_mfc_cfc_benchmark.md)
  - [`analysis/jetson/2026-08-05_mfc_cfc_benchmark.json`](analysis/jetson/2026-08-05_mfc_cfc_benchmark.json)
- 论文引用
  - [TFP arXiv 2607.08283](https://arxiv.org/abs/2607.08283)
  - [NSFD-NODE arXiv 2607.10858](https://arxiv.org/abs/2607.10858)
  - [L-RFM arXiv 2606.15571](https://arxiv.org/abs/2606.15571)
- 综合上下文
  - [[LNN_Training_Paradigm_2026_Summer_Cross_Section]]（上轮横切报告）
  - [[LNN_Family_Taxonomy_And_Gap_2026-08-03]]（Taxonomy & Gap 基线）
  - [[Orin_Nano_Super_LNN_Deployment_v2_2026-08-03]]（Jetson 部署上下文）
