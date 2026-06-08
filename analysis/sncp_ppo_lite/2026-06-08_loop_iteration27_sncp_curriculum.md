---
title: 2026-06-08 Loop iteration 27 — PRD §10 #17 stage B: SNCP-PPO 1→2→3 行人 curriculum ablation
date: 2026-06-08
tags: [LNN, loop, sncp-ppo, curriculum, pedestrian, ltc-actor-critic, ppo, 2d-point-mass-nav, prd-10-17, iter27, stage-b, smoke-implementation]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-08 Loop iteration 27 — PRD §10 #17 stage B

> `/loop 1h` 第 27 次触发 (PRD iter 计数 #27)。
> 紧接 iter#26 (SNCPPolicyLite + iter#26 smoke) 后,本轮执行 **PRD §10 #17 stage B**:
> 给 `PointMassNavLite` 加 `--n-pedestrians` + 移动行人 + `--curriculum` 1→2→3 行人 sequential 训练。
>
> 1. **新脚本逻辑** `scripts/experiment_sncp_ppo_lite.py` (~80 行) — `PointMassNavLite` 行人 + curriculum mode
> 2. **新测试文件** `tests/test_sncp_pedestrian_env.py` (6 单测, 全绿, 含 1 个 CLI smoke 端到端)
> 3. **curriculum smoke** 1 seed × 3 stages × 20 PPO updates (480 episodes total) — 跑通
> 4. **0 回归** pytest 16/16 SNCP tests (10 旧 + 6 新) + verify 9/9
> 5. **commit + rebase + push origin/master** (本轮目标)

## 1. 关键实现

### 1.1 `PointMassNavLite` 加 `--n-pedestrians` (零填 padding 跨阶段 obs_dim 恒定)

```python
# scripts/experiment_sncp_ppo_lite.py
class PointMassNavLite:
    BASE_OBS_DIM: int = 4   # [pos.x, pos.y, goal_dx, goal_dy]
    MAX_PED_SLOTS: int = 5  # 5 predefined PEDESTRIAN_ORIGINS
    PEDESTRIAN_ORIGINS: list = [
        (-0.2, 0.3), (0.5, -0.3), (0.7, 0.5), (-0.4, -0.2), (0.3, 0.7)
    ]

    def __init__(self, seed: int = 0, n_pedestrians: int = 0) -> None:
        # obs_dim 是常数: 4 + 2*5 = 14 — 跨 curriculum 阶段恒定
        # 这样单一 policy 可以在 1→2→3 行人阶段复用
        self.obs_dim = self.BASE_OBS_DIM + 2 * self.MAX_PED_SLOTS
        ...

    def _ped_positions(self, t: int) -> list[tuple[float, float]]:
        # 每个 ped 沿 (cx, cy) + 0.2 半径圆形轨道行走
        # omega 交替正负 (i%2==0 → +0.12, 否则 −0.12)
        ...

    def _current_obs(self) -> torch.Tensor:
        base = [pos.x, pos.y, goal_dx, goal_dy]
        active_peds = self._ped_positions(self._step_count)
        ped_obs = []
        for px, py in active_peds:
            ped_obs.extend([self.pos[0] - px, self.pos[1] - py])
        # Zero-pad 未启用的 ped 槽位到 MAX_PED_SLOTS
        pad_n = self.MAX_PED_SLOTS - self.n_pedestrians
        ped_obs.extend([0.0] * (2 * pad_n))
        return torch.tensor(base + ped_obs)
```

### 1.2 Curriculum 模式 (sequential 1→2→3 stages, 参数跨阶段复用)

```python
# scripts/experiment_sncp_ppo_lite.py::main
parser.add_argument("--curriculum", action="store_true", ...)
parser.add_argument("--ped-curriculum-list", default="1,2,3", ...)
parser.add_argument("--ppo-updates-per-stage", type=int, default=20, ...)

# Pre-size policy to max(stages) obs_dim — 单一 policy 跨所有阶段
first_env = PointMassNavLite(seed=args.seed, n_pedestrians=max(stages))
policy = SNCPPolicyLite(temporal_input_size=first_env.obs_dim, ...)

# Stage loop: 1→2→3 顺序训练
for stage_idx, n_ped in enumerate(stages):
    env = PointMassNavLite(seed=args.seed + stage_idx, n_pedestrians=n_ped)
    for upd in range(args.ppo_updates_per_stage):
        obs_buf, ... = collect_rollout(policy, env, ...)
        ppo_stats = ppo_update(policy, optimiser, obs_buf, ...)
        # 记录 per-stage 历史
```

## 2. 6 单测 (tests/test_sncp_pedestrian_env.py)

| 测试 | 验证 |
|---|---|
| `test_obs_dim_constant_across_n_pedestrians` | obs_dim = 14 对 n ∈ {0, 1, 2, 3, 5} 全部一致 |
| `test_n_zero_backward_compat` | n=0 保留 2 静态障碍 + ped 槽位全填 0 (iter#26 兼容) |
| `test_pedestrians_move_deterministically` | 同 seed → 同 trajectory,跨时间步位置不同 |
| `test_obs_includes_ped_relative_positions_and_pads` | active slots 真实相对位置,剩余 slots 填 0 |
| `test_invalid_n_pedestrians_rejected` | n<0 与 n>MAX_PED_SLOTS 抛 ValueError |
| `test_curriculum_cli_runs_end_to_end` | CLI `--curriculum 1,2,3` 端到端,产出 JSON+MD |

## 3. Curriculum smoke 结果 (1 seed × 3 stages × 20 PPO updates)

| stage | n_ped | last-5 mean_return | last-5 reach_rate | last-5 collision_rate |
|---:|---:|---:|---:|---:|
| 0 | 1 | −3.94 | 0.00 | 0.12 |
| 1 | 2 | −4.48 | 0.00 | 0.17 |
| 2 | 3 | −3.61 | 0.00 | 0.07 |

- **诚实负面**: 3/3 stages reach_rate=0。240 episodes/stage 仍不足学 pedestrian avoidance
- **infra 完整**: curriculum mode 跑通 (sequential 1→2→3 + 跨阶段参数复用 + 零填 obs_dim 恒定)
- **下一阶段**: ~3000 ep/stage + 5-phase 才是对比信号 (原仓 Colab T4 跑 3000 ep 2h40m)

## 4. PRD §10 #17 → stage A+B ✅

| 字段 | 更新 |
|---|---|
| Status | stage A → **stage A+B ✅ (iter#26/27)** |
| 输出物新增 | `tests/test_sncp_pedestrian_env.py` |
| 行内备注 | curriculum infrastructure 完整,240 ep/stage 不够,需 3000 ep 对照 |

## 5. 验证

- `pytest tests/test_sncp_pedestrian_env.py tests/test_sncp_policy_lite.py -v` → **16/16 绿** (10 iter#26 + 6 iter#27)
- `python3.14 scripts/verify_all_models.py` → **9/9 ✓**
- 仓库全测 (multimodal 5 失败为 iter#25 验证的 pre-existing 与本仓本轮无关)
- iter#27 总改动: scripts/experiment_sncp_ppo_lite.py +~80 行 / tests/test_sncp_pedestrian_env.py +155 行 / docs/PRD_LNN_Edge_Research.md 改 1 行
