---
title: SNCP-PPO Crowdnav (LTC + PPO 机器人人群导航) 深度研读报告
date: 2026-06-08
tags: [LNN, LTC, PPO, CrowdNav, Robotics, TurtleBot3, Reinforcement-Learning, Social-Force, Curriculum]
source: https://github.com/heimdilon/sncp-ppo-crowdnav
---

# 研读报告：SNCP-PPO Crowdnav — LTC + PPO 在机器人人群导航的应用

## 1. 元数据

- **仓库**：heimdilon/sncp-ppo-crowdnav
- **作者**：Heimdilon (TurtleBot3 Waffle 部署 + 5-phase curriculum 训练)
- **发布时间**：2026-06-07 (GitHub)
- **本地 PDF / 代码**：N/A (代码仓库, 6 commit, v6 训练收敛)
- **Stars / Language**：0 stars / Python 93.3% + Jupyter Notebook 6.7%
- **配套**：sncp_ppo_colab.ipynb (Colab 一键跑) + waffle_ros/ (ROS 部署节点)

## 2. 核心问题

传统机器人人群导航 (CrowdNav) 用 Social Force / ORCA 等模型化方法,
**对动态非合作行人鲁棒但难泛化**;深度 RL (PPO/SAC) 端到端学策略,
**易灾难性遗忘 + 难处理多行人变体**。本仓库提出 **SNCPPolicy**:
- **LTC (Liquid Time-Constant)** 作 recurrent encoder
- **PPO + GAE + clipped value loss** 训 actor-critic
- **5-phase curriculum** 渐进加行人 (1→2→3→4→5)
- **Generalist metric**: `min(success_rate)` 跨 holdout 场景

## 3. 关键架构 (`sncp_ppo/models.py::SNCPPolicy`)

### 3.1 输入 (robot-local 坐标系)
- `robot_node`: 7 维 (位置/速度/朝向/goal vector ...)
- `spatial_edges`: H×2 (per-pedestrian 相对位置)
- `temporal_edges`: 2 维 (机器人自身 [v, w])

### 3.2 Encoders
- **Robot MLP**: 7 → 64 → 128
- **Temporal LTC encoder**: 2 → 32 → 256 (处理 [v, w] 时序)
- **Per-pedestrian spatial LTC encoder**: 2 → 32 → 256 (per-pedestrian 时序)

### 3.3 Fusion
- **Attention pooling**: 行人与机器人运动状态 (Q vs K) → `u_att`
- **Node LTC**: 融合 `concat[v_m, m_rr, u_att]` (640→32→256) → 共享 trunk `sf`
- **Actor head**: `[v ∈ [0, 0.26], w ∈ [-1.8, 1.8]]` (2-dim action)
- **Critic head**: V(s)

### 3.4 关键 trick
> 行人速度被**故意省略**于 observation 中 ——
> 策略必须从 3 个 LTC hidden state 通过 BPTT 推断行人运动。
> 这强制 LTC 学到真正的时序表征,而不是依赖 shortcut 速度信号。

## 4. PPO Setup

- **Optimizer**: Adam, `--lr 1e-4` + cosine `--lr_end_factor 0.1`
- **GAE with truncation bootstrap** (V(s_final) 在 timeout 时计算)
- **重要细节**: action / log-prob 存**未截断**的 Normal(μ, σ) sample;
  只有发给 env 的 action 被截断 ——
  保留 `exp(new_logp - old_logp)` ratio identity
- **Recurrent hidden state per-step** 存 + update 时 re-feed (SB3/CleanRL 近似)
- **Clipped value loss** (同 PPO 论文)
- **Best checkpoint**: `min(success_rate)` 跨 holdout 场景 ——
  **泛化指标** 而非单一场景成功率
- **5-phase curriculum**: 1→2→3→4→5 行人 (边界默认 10%/25%/50%/75% episodes)
- **CSV log**: `logs/training_<timestamp>.csv` (per-episode + per-scenario holdout)

## 5. 奖励函数 (`crowd_sim/crowd_env.py`)

| 组件 | 公式 | 范围 |
|---|---|---|
| Goal | +20 成功; +10·Δd dense; 小角度 pen. | [-10, +20] |
| Collision | -20 接触 | [-20, 0] |
| Comfort (I_sp) | -0.5·I_sp / N (per-human capped at 10/d_hr) | bounded < 0 |
| Standstill | -0.5 if v < 0.05 m/s | [-0.5, 0] |

Comfort **除以 num_humans** 保持 phase-invariant
(防止 curriculum 切换时 value function shock)。

## 6. 量化结果 (v6, 3000 episodes, seed=42, ~2h40m on Colab T4)

100 deterministic eval episodes per scenario, seed=100:

| Scenario | Pedestrians | Success | Collision | Timeout | Avg Reward |
|---|---:|---:|---:|---:|---:|
| easy | 1 | **100%** | 0% | 0% | 85.2 |
| easy_plus | 2 | **100%** | 0% | 0% | 85.7 |
| medium | 3 | **100%** | 0% | 0% | 86.4 |
| hard | 5 | **86%** | 14% | 0% | 75.4 |
| extreme* | 5 | 26% | 74% | 0% | 13.6 |

*`extreme` 是 OOD (random spawns vs 训练 circle pattern)*

**关键发现**: v3 在 dense 训练后**灾难性遗忘**到 1-pedestrian ~6% 成功率;
v6 保持 easy/easy_plus/medium 100%, 到达 hard 86% —— **不需要 curriculum replay**。

## 7. 复现 ROI 评估

| 维度 | 评估 |
|---|---|
| **公式复用** | 高 — LTC = 本仓 `lnn/core/ltc.py::LTCNetwork` (已实现) |
| **数据依赖** | 低 — 无需真实行人数据集,社会力模型生成 |
| **训练预算** | 中 — Colab T4 2h40m;Orin Nano CPU 估 6-10x = 16-26h,不可行 |
| **Jetson 部署** | 已有 `waffle_ros/` 节点 — 可直接 forward-only 部署 |
| **跨仓库价值** | **P1** — 仓库首例 LTC + PPO RL 应用,补全 LNN 三大应用域 (时序回归/视觉/RL) 第三块 |

## 8. 最小可复现单元 (本轮 iter#26 实现)

不依赖 TurtleBot3 / Social Force,做 2D point-mass + 静态障碍 + 1-2 行人 ghost:

- `lnn/core/sncp_policy_lite.py::SNCPPolicyLite` (~150 行):
  - 用本仓 `LTCNetwork` 作 temporal encoder
  - MLP 作 spatial encoder
  - 简化 attention pooling
  - Actor head → 2D velocity (v, w) Gaussian
  - Critic head → V(s)
- `tests/test_sncp_policy_lite.py`: shape / action 范围 / value scalar / recurrent state
- `scripts/experiment_sncp_ppo_lite.py`: 50-episode PPO 2D nav smoke

**目的**: 验证 LTC 编码器在 PPO actor-critic 中可端到端训练,
不追求达到原仓库 86% hard 成功率。

## 9. 与仓库现有 LNN backbones 的关系

| 仓库 | 用途 | 是否可加 SNCP-PPO Lite |
|---|---|---|
| `lnn/core/ltc.py` | 通用 LTC cell | ✅ 直接复用 |
| `lnn/core/cfc.py` | CfC (closed-form) | ⚠️ 不能用 (需要 recurrence,不是 closed-form) |
| `lnn/core/dynpmnn.py` | FHN-ODE | ⚠️ 太慢,RL inner loop 不合适 |
| `lnn/core/riemannian_ltc.py` | Riemannian LTC | ⚠️ smoke only (iter#38) |

**结论**: SNCP-PPO Lite 仅用 `LTCNetwork`,**与现有 backbone 无冲突**。

## 10. 建议后续

- **iter#27 stage B**: 跑 200-episode curriculum (1→2 行人) + 3 seed × {LTC encoder / GRU encoder} ablation
- **iter#28 stage C**: 把 SNCP-PPO Lite ingest 到 `scripts/build_backbone_matrix.py`,
  新增 `crowdnav_lite` 行
- **iter#29 真实 ROS 部署**: 与 `waffle_ros/` 对接,本仓 Jetson forward-only benchmark
  (≤200ms latency 目标)

## 11. 来源

- 主仓库: https://github.com/heimdilon/sncp-ppo-crowdnav
- 论文: 无 (v6 训练收敛 + Colab 复现 = 实证 paper-equivalent)
- 本地复现入口: `lnn/core/sncp_policy_lite.py` + `scripts/experiment_sncp_ppo_lite.py` (本轮 iter#26 stage A)
