---
title: Jetson validation summary — iter#19 PDNA stage A: PDNAPulseHead + 12 unit tests
date: 2026-06-04
platform: Linux hyx-desktop 5.15.148-tegra (Jetson Orin Nano Super, aarch64)
python: 3.14.4 (pyenv) — primary, CPU path
cuda: torch 2.11.0+cu130 → cuda.is_available()=False (BSP driver 12.060 < cu130)
tag: loop-iteration, jetson-cpu-only, pdna-stage-a, prd-10-10
---

# Jetson validation summary — iter#19 PDNA stage A

> 本轮首次落地 **代码改动**(前 4 轮 iter#14-18 都是 paper 研读 + PRD 维护)。
> 修改: `lnn/core/cfc.py` 加 `PDNAPulseHead` (~80 行,含 docstring)
> + `tests/test_pdna_pulse.py` (12 个 unit test)。

## 1. 改动量

```
lnn/core/cfc.py              +80 行 (PDNAPulseHead class)
tests/test_pdna_pulse.py    +196 行 (12 tests)
```

## 2. PDNAPulseHead 关键参数

| 参数 | 默认值 | 来源 |
|---|---|---|
| `hidden_size` | (required) | backbone hidden dim |
| `use_self_attend` | `True` | 论文 Variant E (full PDNA);`False` 单独用 pulse |
| `omega_low/high` | 0.1 / 10.0 | 论文 §3.2 log-uniform init |
| `alpha_init` | 0.01 | 论文 §3.2 让 backbone 先学 |
| `beta_init` | 0.01 | 论文 §3.3 self-attend gate |

## 3. 12 个 unit test 覆盖

| # | 测试 | 验证内容 |
|---:|---|---|
| 1 | `test_output_shape_preserved` | output [B,T,d] 与 input 同形 |
| 2 | `test_output_shape_with_explicit_t` | 显式 t 输入时 shape 保持 |
| 3 | `test_alpha_init_0_01` | α 严格 0.01 (论文硬编码) |
| 4 | `test_beta_init_0_01` | β 严格 0.01 (论文硬编码) |
| 5 | `test_gates_are_learnable_parameters` | α/β 是 `nn.Parameter` 且 `requires_grad` |
| 6 | `test_omega_log_uniform_init_range` | 64 sample 下 ω max/min > 5x |
| 7 | `test_omega_in_expected_window` | 128 sample 下所有 ω ∈ [0.05, 20.0] |
| 8 | `test_pulse_signal_magnitude_is_small_at_init` | init 时 `|h_pulse - h| < 0.05` |
| 9 | `test_pulse_amplitude_per_dim` | A 初始化为 ones (per-dim) |
| 10 | `test_gradient_flows_to_all_params` | 8 个参数全部收到非零 grad |
| 11 | `test_gradient_flows_without_self_attend` | use_self_attend=False 路径不报错 |
| 12 | `test_pdna_head_in_cfc_pipeline` | 端到端:CfCNetwork + PDNAPulseHead + Linear 投影 |

## 4. pytest 套件(58 tests,15.24s)

```
tests/test_core.py                  : 46 passed
tests/test_liquid_tad_hierarchical.py: 6 passed (iter#3 沿用)
tests/test_pdna_pulse.py            : 12 passed (iter#19 新增)
─────────────────────────────────────────────
58 passed, 1 warning in 15.24s
```

vs iter#18 的 46/46 → iter#19 的 58/58 = **+12 新增,0 回归**。

## 5. verify_all_models.py(9 变体 smoke)

```
LTC                 : ✓ 通过
CfC                 : ✓ 通过
StrictCfC           : ✓ 通过
HybridCfC           : ✓ 通过
CTLTC               : ✓ 通过
LiquidS4            : ✓ 通过
LRC                 : ✓ 通过
CfC-DT              : ✓ 通过
Euler-LTC-DT        : ✓ 通过
```

与 iter#14-18 一致 — `PDNAPulseHead` 改动**不触碰** `CfCCell` / `CfCNetwork`,
**不破坏**任何已注册的 verify 用例。

## 6. 与本周回退基线对比

| 指标 | iter#15 | iter#16 | iter#17 | iter#18 | iter#19 (本次) |
|---|---:|---:|---:|---:|---:|
| verify_all_models | 9/9 ✅ | 9/9 ✅ | 9/9 ✅ | 9/9 ✅ | **9/9 ✅** |
| pytest 套件 | 46/46 | 46/46 | 46/46 | 46/46 | **58/58** (+12) |

**0 回归,12 新增**。本轮首次让仓库 LNN 架构实现数量 +1。

## 7. 下游可启动项

PRD §10 #10 stage A 已完成。**stage B**(iter#20 候选):
- `scripts/experiment_pdna_smoke.py` 跑 sMNIST Gapped protocol
- 5 seed × 4 backbone (CfC / CfC+pulse / CfC+self-attend / Full PDNA)
- backbone matrix 加 `smnist_gap` 行

## 8. 已知阻塞(无变化)

| 阻塞 | 来源 | 影响 |
|---|---|---|
| CUDA 不可用 | Jetson BSP driver 12060 < torch 2.11 cu130 | iter#2 修通的 py3.10+torch2.10 路径需要空载窗口 |
| RAM 1.7 GB available | 多 agents 并行 + 8GB 统一显存 | LFM2.5-1.2B / 较大 hidden LNN sweep 受限 |
| THUMOS-14 数据未下载 | LiquidTAD stage C 真实数据 | 暂用 toy 长视频 |

本轮无新增阻塞。
