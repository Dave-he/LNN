---
title: Jetson validation summary — iter#36 RLSTG stage A design
date: 2026-06-05
platform: Linux hyx-desktop 5.15.148-tegra (Jetson Orin Nano Super, aarch64)
python: 3.14.4 (pyenv) — primary, CPU path
cuda: torch 2.11.0+cu130 → cuda.is_available()=False (BSP driver 12.060 < cu130)
tag: loop-iteration, jetson-cpu-only, rlstg-stage-a, design-doc, planning
---

# Jetson validation summary — iter#36 RLSTG stage A design

> 本轮执行 **RLSTG §10 复现路线 stage A** —— 设计文档, 决定复现深度, 列出
> dependencies 与风险。**纯设计文档, 无 lnn/ 代码改动**。

## 1. 改动量

```
analysis/riemannian_lnn/2026-06-05_rlstg_stage_a_design.md   新增 (~150 行, 10 节)
```

## 2. 复现深度决策

| 论文块 | 复现? | 理由 |
|---|---|---|
| §3.2 Tangent-space LTC | ✅ 完整 | 公式与本仓 LTC 几乎同构 |
| §3.3 Riemannian ODE solver | ✅ 完整 | geoopt 装上即可 |
| §4 理论 | ❌ 跳过 | 仓库无 formal proof 传统 |
| §5 ENRON link prediction | ⚠️ 部分 | 数据集无 → synthetic graph fallback |

## 3. 关键依赖

- `geoopt` (PyTorch-compatible manifold ops) — `pip install geoopt`
  - fallback: `torchdyn` (Neural ODE + Riemannian 集成)
- Hyperboloid 流形 (`manifold.Hyperboloid`)

## 4. 文件结构(stage B 计划)

```
lnn/core/riemannian_ltc.py          (新增, ~150 行)
  ├── class TangentSpaceLTC         (复用 LTC 公式, tangent space)
  ├── class RiemannianLTC           (wrap + exp/log map)
  └── class RiemannianLTCNetwork    (堆叠多层)

tests/test_riemannian_ltc.py        (~150 行, 8 tests)
scripts/experiment_rlstg_smoke.py   (~200 行, synthetic graph ablation)
```

## 5. 关键公式(本仓实现)

```python
class TangentSpaceLTC(nn.Module):
    def forward(self, h, u, dt=0.1):
        return h + dt * (-self.alpha * h + torch.tanh(self.W_h @ h + u))

class RiemannianLTC(nn.Module):
    def __init__(self, manifold=Hyperboloid()):
        self.manifold = manifold
        self.tangent_ltc = TangentSpaceLTC(...)
    def forward(self, x, u, dt=0.1):
        v = self.tangent_ltc(x, u, dt)
        return self.manifold.expmap(x, v)
```

## 6. 8 unit test 计划

| Test | 验证内容 |
|---|---|
| `test_exp_log_consistency_small` | 小向量 `exp(x, log(x, v)) ≈ v` |
| `test_exp_log_numerical_stability_large` | 大向量不爆 |
| `test_riemannian_ltc_forward_shape` | 输入输出同 manifold shape |
| `test_gradient_flows_through_geometry` | grad 流到 manifold curvature |
| `test_hyperbolic_distance_grows` | 不同 class 间 hyperbolic distance 增加 |
| `test_multi_step_ode_stable` | 10 步迭代无 NaN/Inf |
| `test_end_to_end_loss_backward` | forward + MSE + backward + step |
| `test_invalid_manifold_raises` | 传非 manifold 类型 raise |

## 7. 计划 schedule

| iter | 工作 | 估时 |
|---|---|---|
| #36 (本轮) | stage A design doc | 0.5 loop ✅ |
| #37 | stage B: riemannian_ltc.py + 8 unit tests | 2-3 loop |
| #38 | stage C: synthetic graph + 3-seed × 4-backbone ablation | 1-2 loop |
| #39 | stage D: 写复现报告 | 0.5 loop |

## 8. 风险 + 缓解

| 风险 | 缓解 |
|---|---|
| `geoopt` 与 torch 2.11.0+cu130 兼容 | 先 `pip install` 试, 失败回退 `torchdyn` |
| `expmap` 在 Hyperboloid 上数值不稳 | double precision fallback |
| ENRON 数据无 | 限报告 synthetic, 不 claim ENRON 复现 |
| 装 `geoopt` 占 Jetson RAM | 评估后再装 |

## 9. pytest 套件(102/102, 76.91s)

无变化(纯设计文档)。vs iter#35: 102 → 102 = **0 变动,0 回归**。

## 10. verify_all_models.py(9/9)

无变化。

## 11. 关键 takeaway

1. **stage A 是 0.5 loop 的小手术** —— 提前理清复现深度,避免后续返工
2. **复现深度 = 1+2 完整 + 3 跳过 + 4 部分** —— 实用主义
3. **关键依赖 geoopt** —— 需要在 stage B 启动时先装,兼容性先验
4. **RLSTG 是 7 套 LNN 概念扩展之一** —— 完成 stage B/C 后, 仓库跨 8 套架构对比
