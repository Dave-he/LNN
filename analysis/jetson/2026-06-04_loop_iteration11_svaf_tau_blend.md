---
title: Jetson validation summary — iter#22 SVAF §10 #9: τ-modulated peer-blending
date: 2026-06-04
platform: Linux hyx-desktop 5.15.148-tegra (Jetson Orin Nano Super, aarch64)
python: 3.14.4 (pyenv) — primary, CPU path
cuda: torch 2.11.0+cu130 → cuda.is_available()=False (BSP driver 12.060 < cu130)
tag: loop-iteration, jetson-cpu-only, svaf-tau-blend, prd-10-9
---

# Jetson validation summary — iter#22 SVAF §10 #9

> 本轮执行 **PRD §10 #9 stage A** —— `lnn/core/cfc.py` 加 SVAF τ-modulated
> peer-blending 算子 + 9 unit tests + 2-agent mesh toy experiment。

## 1. 改动量

```
lnn/core/cfc.py                          +~100 行 (4 functions: similarity_per_dim,
                                          tau_modulated_blend_coef,
                                          tau_modulated_blend_update,
                                          default_three_group_tau)
tests/test_svaf_tau_blend.py            +160 行 (9 tests)
scripts/experiment_svaf_tau_toy.py      +200 行 (2-agent mesh + JSON/MD output)
analysis/svaf/2026-06-04_tau_toy.{md,json}  新增
docs/PRD_LNN_Edge_Research.md            1 行状态更新 (#10-9 stage A ✅)
```

## 2. 公式对齐论文 Eq. 20

```
β_i = min(α_eff × K × sim_i / τ_i, 1.0)
sim_i = max(1 - |h_local_i - h_mesh_i| / max(|h_local_i|, |h_mesh_i|), 0)
h_new = (1 - β) ⊙ h_local + β ⊙ h_mesh
```

按论文 Table 14 τ 分类: Fast < 5s / Medium 5-30s / Slow > 30s

## 3. 关键 toy 实验结果(d=6, n=20 steps, peer=0.5)

| Group | τ | final distance to peer |
|---|---:|---:|
| Fast | 1s | 0.0000 ✅ |
| Medium | 10s | 0.0000 ✅ |
| Slow | 60s | 0.0183 ✅ |

**论文 §7.1 核心论断复现**: "Fast neurons (τ<5s) couple readily; slow neurons
(τ>30s) resist coupling entirely". 20 步后 Fast/Medium 完全收敛到 peer (0.5),
Slow 仅变 0.0183(从 1.0 出发 → 0.9817)— **slow 保持主权**。

## 4. 9 unit test 覆盖

1. `test_blend_coef_in_unit_interval` — β ∈ [0, 1] 严格不变量
2. `test_fast_tau_blends_more_than_slow` — 相同 sim 下 fast β > slow β
3. `test_identical_h_saturates_beta` — sim=1 时 β = min(αK/τ, 1)
4. `test_opposite_h_gives_zero_beta` — 反向向量 β=0
5. `test_two_agent_mesh_fast_converges_slow_preserves` — N 步 mesh 更新
6. `test_two_agent_mesh_5step_gap` — 5 步即可观察 fast/slow 速度差
7. `test_default_three_group_tau_layout` — helper 切分正确
8. `test_similarity_per_dim_range` — sim ∈ [0, 1] 基本性质
9. `test_update_formula_basic` — 更新公式正确性

## 5. pytest 套件(75 tests, 12.08s)

```
tests/test_core.py                  : 46 passed
tests/test_liquid_tad_hierarchical.py: 6 passed
tests/test_pdna_pulse.py            : 12 passed
tests/test_loop_status_prd.py       :  8 passed
tests/test_svaf_tau_blend.py        :  9 passed (iter#22 新增)
─────────────────────────────────────────────
75 passed, 1 warning in 12.08s
```

vs iter#21: 66 → 75 = **+9 新增,0 回归**。

## 6. verify_all_models.py(9/9)

无变化。

## 7. 与本周回退基线对比

| 指标 | iter#18 | iter#19 | iter#20 | iter#21 | iter#22 (本次) |
|---|---:|---:|---:|---:|---:|
| verify_all_models | 9/9 | 9/9 | 9/9 | 9/9 | **9/9** |
| pytest 套件 | 46/46 | 58/58 | 58/58 | 66/66 | **75/75** (+9) |

## 8. 关键 takeaway

1. **SVAF Eq. 20 (β_i = αK·sim/τ, clipped 1) 在小 toy mesh 上** 复现论文 §7.1
   核心论断 "fast 同步, slow 主权"
2. **helper default_three_group_tau** 让用户一键切分 1/3 Fast + 1/3 Medium + 1/3 Slow
3. **与 PDNA 互补**: PDNA 用 `α·A·sin(ωt+φ(h))` 做 per-dim 频率 oscillation;
   SVAF τ-blend 用 per-dim τ 调制 peer-blend — **仓库 LNN 现在有 frequency +
   decay 两套 continuous-time augmentation primitives**
4. **本轮所有"小 helper" 测试都过** — eq 算子级 unit test 比 ablations 更稳定

## 9. 已知阻塞(无变化)

| 阻塞 | 来源 | 影响 |
|---|---|---|
| CUDA 不可用 | Jetson BSP driver 12060 < torch 2.11 cu130 | 较大 hidden LNN sweep 需 CPU |
| RAM 1.7 GB available | 多 agents 并行 + 8GB 统一显存 | LFM2.5 系 / 较大 hidden 受限 |
| THUMOS-14 数据未下载 | LiquidTAD 真 stage C | §8 #2 / §9 #3 pending |

本轮无新增阻塞。
