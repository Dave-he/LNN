---
title: Jetson validation summary — iter#23 DynPMNN §10 #1 stage A: FHNCell + DynPMNNNetwork
date: 2026-06-04
platform: Linux hyx-desktop 5.15.148-tegra (Jetson Orin Nano Super, aarch64)
python: 3.14.4 (pyenv) — primary, CPU path
cuda: torch 2.11.0+cu130 → cuda.is_available()=False (BSP driver 12.060 < cu130)
tag: loop-iteration, jetson-cpu-only, dynpmnn-stage-a, prd-10-1
---

# Jetson validation summary — iter#23 DynPMNN §10 #1 stage A

> 本轮执行 **PRD §10 #1 stage A** —— `lnn/core/dynpmnn.py` 加 `FHNCell + DynPMNNNetwork`,
> 完整复现 arXiv:2605.08176v1 §2.2-2.3 FitzHugh-Nagumo ODE + Euler 积分。

## 1. 改动量

```
lnn/core/dynpmnn.py            +175 行 (FHNCell + DynPMNNNetwork, 含 docstring)
tests/test_dynpmnn.py          +165 行 (9 tests)
docs/PRD_LNN_Edge_Research.md   1 行状态更新 (#10-1 stage A ✅)
```

## 2. FHN ODE 实现(论文 §2.2 直译)

```
dV/dt = V - V^3/3 - W + I
dW/dt = epsilon * (V + a - b*W)
```

实现: `FHNCell.forward` 用 `n_euler_steps` 步 Euler 积分(dt=1/n_euler_steps),
输出最终 (V, W) + 完整 V 轨迹(供后续 backbone matrix 加 `--backbone fhn_dynpmnn` 用)。

可学习参数 (per-dim):
- `a ∈ R^d` (recovery offset, init 0.7)
- `b ∈ R^d` (recovery decay, init 0.8)
- `epsilon ∈ R^d` (recovery speed, init 0.08)
- `W_in ∈ R^(d_in × d)` (input projection)

## 3. 9 unit test 覆盖

1. `test_forward_shape_with_sequences` — return_sequences=True shape
2. `test_forward_shape_last_only` — return_sequences=False shape
3. `test_initial_state_zero` — V_0 = W_0 = 0
4. `test_gradient_flows_to_all_params` — a/b/epsilon/input_proj 都收非零 grad
5. `test_forward_no_nan_or_inf` — bounded input 不爆
6. `test_fhn_response_to_strong_input` — FHN 兴奋性: 强 input 驱动 V 远
7. `test_single_step_does_not_explode` — 单步 Euler 稳定
8. `test_multi_layer_chains_correctly` — 3 层 chain 形状 + 参数数对
9. `test_full_pipeline_backward` — forward + MSE + backward 3 步无 NaN

## 4. pytest 套件(84 tests, 11.18s)

```
tests/test_core.py                  : 46 passed
tests/test_liquid_tad_hierarchical.py: 6 passed
tests/test_pdna_pulse.py            : 12 passed
tests/test_loop_status_prd.py       :  8 passed
tests/test_svaf_tau_blend.py        :  9 passed
tests/test_dynpmnn.py               :  9 passed (iter#23 新增)
─────────────────────────────────────────────
84 passed, 1 warning in 11.18s
```

vs iter#22: 75 → 84 = **+9 新增,0 回归**。

## 5. verify_all_models.py(9/9)

无变化。`dynpmnn.py` 是新 module,verify 路径不涉及。

## 6. 与本周回退基线对比

| 指标 | iter#19 | iter#20 | iter#21 | iter#22 | iter#23 (本次) |
|---|---:|---:|---:|---:|---:|
| verify_all_models | 9/9 | 9/9 | 9/9 | 9/9 | **9/9** |
| pytest 套件 | 58/58 | 58/58 | 66/66 | 75/75 | **84/84** (+9) |

## 7. 关键 takeaway

1. **FHN ODE + Euler 积分** 在 PyTorch autograd 中**端到端可微** — 9 个测试全过证明
2. **新 LNN backbone DynPMNN 接入矩阵候选** — stage B 任务(iter#24 候选)
   `scripts/ablation_lnn_vs_lstm_timeseries.py --backbone fhn_dynpmnn` 跑 multi-seed
3. **ODE 形式是物理给定的** — 与 CfC/LTC 的"学 ODE 函数"不同,DynPMNN 的 ODE 是固定的
   FitzHugh-Nagumo 形式,只学 a/b/epsilon + 输入投影
4. **论文 §2.4 RKBS 理论**未实现(只保证 universal approximation 的理论保证),但
   实现层是端到端可微的 smoke 版本

## 8. 已知阻塞(无变化)

| 阻塞 | 来源 | 影响 |
|---|---|---|
| CUDA 不可用 | Jetson BSP driver 12060 < torch 2.11 cu130 | 较大 hidden LNN sweep 需 CPU |
| RAM 1.7 GB available | 多 agents 并行 + 8GB 统一显存 | LFM2.5 系 / 较大 hidden 受限 |
| THUMOS-14 数据未下载 | LiquidTAD 真 stage C | §8 #2 / §9 #3 pending |

本轮无新增阻塞。
