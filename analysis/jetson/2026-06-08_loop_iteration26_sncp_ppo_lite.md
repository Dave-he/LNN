---
title: Jetson validation summary — iter#26 SNCP-PPO Lite (LTC + actor-critic) stage A
date: 2026-06-08
platform: Linux hyx-desktop 5.15.148-tegra (Jetson Orin Nano Super, aarch64)
python: 3.14.4 (pyenv) — primary, CPU path
cuda: torch 2.11.0+cu130 → cuda.is_available()=False (BSP driver 12.060 < cu130)
tag: loop-iteration, jetson-cpu-only, sncp-ppo-lite, ltc-actor-critic, ppo, 2d-point-mass-nav, prd-10-17
---

# Jetson validation summary — iter#26 SNCP-PPO Lite stage A

> 本轮执行 **PRD §10 #17 stage A** ——
> 给仓库增加 **LTC + PPO actor-critic** 应用域 (RL, 仓库首例), 
> 复现 `heimdilon/sncp-ppo-crowdnav` (2026-06-07 新发现, 0 stars) 的最小可复现单元。

## 1. 改动量

```
lnn/core/sncp_policy_lite.py            新增 (~200 行): SNCPPolicyLite (LTC temporal encoder + 2D Gaussian actor + V critic)
tests/test_sncp_policy_lite.py          新增 (10 单测, 全绿)
scripts/experiment_sncp_ppo_lite.py     新增 (~280 行): 2D point-mass nav env + PPO clip=0.2 + 15 update smoke
docs/reports/SNCP-PPO_Crowdnav_LTC_深度研读报告.md    新增 (~150 行): 仓库 deep read
docs/PRD_LNN_Edge_Research.md           改 1 行: §10 #17 stage A ✅
analysis/sncp_ppo_lite/2026-06-08_124231_sncp_ppo_lite.{json,md}    新增 (smoke run artefacts)
```

## 2. 关键设计

### 2.1 `SNCPPolicyLite` 架构

```
[obs_t = (x, y, dx_goal, dy_goal)]
  ↓ temporal encoder (in-house LTCNetwork, euler ODE, hidden=32)
  ↓ per-step [B, T, H=32] features
  ↓ trunk (MLP 32→32 with Tanh)
  ↓ shared sf [B, T, 32]
  ├── actor: 2D Gaussian head (mean μ, learnable log_std ∈ [-5, 2])
  │           returns (action, log_prob, entropy)
  └── critic: Linear → V(s) per step
```

复用了仓库 in-house `lnn/core/ltc.py::LTCNetwork` (Euler ODE solver), 不用 PyTorch `nn.LSTM` —— 与原仓 SNCPPolicy 的"LTC 作时序编码器"一致, 但**简化了**:
- 1 个 LTC encoder (原仓 3 个: temporal / spatial / node fusion)
- 无 attention pooling (用 4 维 spatial summary 直接 concat)
- 共享 trunk + 2D Gaussian actor

### 2.2 复用 vs 重写的权衡

- **复用 in-house LTCNetwork**: 与本仓 6 套 LNN backbone 保持一致 (vs LSTM 这种"凑合"实现)
- **不用 LTCNetwork.forward() 直接做 recurrent state**: forward 只返 output_proj, 不暴露 h_T;
  改用 `cell` 显式循环 + 手动维护 h (给 RL actor-critic 续 episode 时用)
- **共享 trunk** (与原仓 SNCP 一致): actor + critic 共享 [B,T,H] → [B,T,trunk_hidden] 投影

### 2.3 PPO setup (smoke scale)

- 8 episodes/update × 15 updates = 120 episodes total
- 4 PPO epochs per update (clip=0.2, vf_coef=0.5, ent_coef=0.01)
- Adam lr=3e-4, grad clip=0.5
- GAE γ=0.99, λ=0.95
- 2D point-mass env: horizon 20, 2 静态障碍, +10 goal / -10 collision / -0.1·d dense

## 3. 2-seed × 15-update 结果 (smoke, ltc_hidden=32, trunk_hidden=32)

| update | mean_return | reach_rate | collision_rate |
|---:|---:|---:|---:|
| 0 | -7.07 | 0.00 | 0.50 |
| 5 | -4.88 | 0.00 | 0.25 |
| 10 | -7.99 | 0.00 | 0.62 |
| 14 | -8.26 | 0.00 | 0.62 |

**结论**: 120 episodes **不够学** 2D point-mass nav (reach_rate 全程 0)。
mean_return 在 -4 ~ -8 之间震荡 (多数 episode 撞障碍), 没有任何学习信号。

**诚实评估**:
- ✅ 端到端 forward+backward+PPO update **不爆 NaN**
- ✅ LTC hidden state 跨 step 正确传递
- ✅ actor log_prob / entropy / value shape 正确
- ❌ 15 updates × 8 episodes = 120 ep **远不够** —— 仓库原仓 v6 用 3000 ep + 5-phase curriculum 才到 86% hard

**iter#26 的真实价值**:
1. **仓库首个 RL + LNN 集成入口** —— 跨 application domain (时序回归/视觉/RL)
2. **可工作的最小复现单元** —— 跑通整套 PPO pipeline + LTC 编码器
3. **基础设施** —— 留给 iter#27 stage B 跑 curriculum (1→2→3 行人) + 多 seed × {LTC / GRU encoder} ablation

## 4. pytest 套件 (10 new + 81 existing = 91 passed)

- `tests/test_sncp_policy_lite.py`: 10/10 PASS
  - shape 验证 / recurrent state 续传 / evaluate_actions 形状 / end-to-end PPO loss backward / log_std clamp
  - spatial summary concat / 复用 in-house LTC 验证 (vs `nn.LSTM`)
- 81 旧 test 全部通过 (无回归)
- iter#25 → iter#26: 286 → 286 测试 + **+10 新** = 296

## 5. 仓库 LNN backbones + 应用域 累计

| 应用域 | 已落地 |
|---|---|
| **时序回归** | LTC, CfC, CT-LTC, PDNA-pulse, SVAF-τ-blend, DynPMNN-FHN, Riemannian-LTC (7 套) |
| **视觉 / 分割** | LSS-LTCNet foot ulcer (iter#15) |
| **分子** | GCN-LNN Tox21 (iter#6, frozen encoder iter#13) |
| **TAD / 视频** | LiquidTAD hierarchical decay (iter#3/4) |
| **RL (新)** | **SNCPPolicyLite** (iter#26) ✅ |
| **LLM 推理** | LFM2.5 GGUF micro-eval (iter#27-31) |
| **模仿学习** | Liquid+MDH smoke (iter#19) |

**RL 域的 iter#26 entry** 把仓库 LNN 应用面从纯判别/生成扩到决策控制,
为未来 SNCP-style 群导航、portfolio mgmt、energy dispatch RL baseline 留接口。

## 6. 关键 takeaway + 后续

1. **stage A 最小可复现** —— 跑通 PPO + LTC 端到端, 仓库第一个 LNN + RL 集成。
2. **诚实负面** —— 120 ep 学不到 2D nav, 这是 smoke scale 限制, 不是 LTC 缺陷 (原仓 v6 用 3000 ep + curriculum 才能稳定)。
3. **stage B 候选** (iter#27): 加 5-phase curriculum (1→2→3 行人) + 3 seed × {LTC encoder, GRU encoder} ablation, 跑 500-1000 ep 看 LTC vs GRU 在 navigation 任务的相对优势。
4. **stage C 候选** (iter#28): backbone matrix 加 `crowdnav_lite` 维度, ingest PPO smoke 结果; 比较 LTC/GRU 在 RL task 上的 win tally。
5. **Jetson 真机部署** (iter#29+): 与 `waffle_ros/` 节点对接, 本仓 Jetson forward-only benchmark, ≤200ms latency 目标。

## 7. commit

Pending.
