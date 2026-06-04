---
title: Jetson validation summary — iter#37 RLSTG §10 stage B: Riemannian LTC implementation
date: 2026-06-05
platform: Linux hyx-desktop 5.15.148-tegra (Jetson Orin Nano Super, aarch64)
python: 3.14.4 (pyenv) — primary, CPU path
cuda: torch 2.11.0+cu130 → cuda.is_available()=False (BSP driver 12.060 < cu130)
tag: loop-iteration, jetson-cpu-only, rlstg-stage-b, tangent-space, hyperbolic, geoopt
---

# Jetson validation summary — iter#37 RLSTG stage B

> 本轮执行 **PRD §10 RLSTG stage B** —— 在 iter#36 design doc 基础上实现
> `lnn/core/riemannian_ltc.py` (TangentSpaceLTC + RiemannianLTC + RiemannianLTCNetwork),
> 加 9 个 unit test。**完整可工作** smoke 实现, 复现 arXiv 2601.14115 §3.2-3.3 核心模式。

## 1. 改动量

```
lnn/core/riemannian_ltc.py        +205 行 (TangentSpaceLTC + RiemannianLTC + RiemannianLTCNetwork)
tests/test_riemannian_ltc.py      +175 行 (9 tests)
```

依赖安装: `pip install geoopt` (geoopt 0.5.1)

## 2. 关键设计点

### 2.1 公式直译

```python
# Tangent-space LTC (论文 Eq. 10)
h_{t+1} = h_t + dt * (-α ⊙ h_t + tanh(W_h h_t + W_u u_t + b))

# Riemannian wrapper (论文 Eq. 12)
x_{t+1} = expmap0( dt * LTC(tangent: logmap0(Linear(u))) )
```

### 2.2 geoopt 0.5.1 局限 + 缓解

| 局限 | 缓解 |
|---|---|
| 完整 `expmap` / `logmap` 需 parallel transport, **autograd 不支持** | 退到 **origin-only** `expmap0` / `logmap0` (closed form, autograd 通过) |
| `logmap0(v)` 当 `v[0] != 0` 时返回 NaN (tangent 必须在 origin subspace `v[0]=0`) | 显式 `u_amb[..., 0] = 0.0` 后再 logmap0 |
| `expmap0(v)` 对大 `||v||` 会爆 (cosh/sinh overflow) | tangent norm clip 到 `max_tangent_norm=1.0` (默认) |
| 初始随机 weight 太大 → 第一步 NaN | `input_proj.weight` 用 `std=0.1` 初始化, `dt=0.001` 默认 |

### 2.3 9 unit test 覆盖

| Test | 验证内容 |
|---|---|
| `test_init_state_on_manifold` | `init_state` 满足 `⟨x, x⟩_L = -1` |
| `test_forward_shape_riemannian_ltc` | 单层 forward shape |
| `test_forward_shape_riemannian_ltc_network` | 网络 forward shape (return_sequences True/False) |
| `test_gradient_flows_to_input_proj_and_tangent_ltc` | grad 流到 input_proj + tangent_ltc.W_h + alpha |
| `test_no_nan_for_small_inputs` | 小 input 不爆 NaN |
| `test_invalid_manifold_raises` | 传 bogus manifold name raise |
| `test_end_to_end_loss_backward_step` | forward + MSE + backward + 3 步 optim step 不爆 |
| `test_tangent_space_ltc_basic` | TangentSpaceLTC.forward shape + 小 dt 时近似 identity |
| `test_multi_step_stable` | 10 步 forward 不爆 NaN |

## 3. pytest 套件(111/111, 26.67s)

```
102 旧 + 9 新 (test_riemannian_ltc) = 111 passed
```

vs iter#36: 102 → 111 = **+9 新增,0 回归**。

## 4. verify_all_models.py(9/9)

无变化。

## 5. 已暴露 limitation (stage C/D 候选)

1. **origin-only expmap**: 每步都从 origin 开始, 不连续 tangent space 推进 — 与论文 full expmap 不同
2. **gradient norm ~167**: Riemannian 的固有 curvy 反传, 训练时可能需 gradient clip
3. **无 paper baseline 对照**: stage C 需跑 synthetic hyperbolic graph + 4 backbone (cfc/ltc/gru/RLSTG) ablation
4. **geoopt 0.5.1 限制**: full expmap 需 parallel transport, 需升级 geoopt 或手动实现

## 6. 关键 takeaway

1. **stage B 完整可工作** — 9/9 测试通过, 端到端 forward + loss + backward + step 不爆
2. **geoopt 0.5.1 局限已知** — 退到 origin-only 是合理 stage B 妥协
3. **§10 RLSTG 复现路线 stage B 完成** — 剩 stage C (synthetic graph ablation) + stage D (复现报告)
4. **LNN 概念扩展仓库** — 实际有了**第 8 套**可 import 模块: `from lnn.core.riemannian_ltc import RiemannianLTC, RiemannianLTCNetwork`
