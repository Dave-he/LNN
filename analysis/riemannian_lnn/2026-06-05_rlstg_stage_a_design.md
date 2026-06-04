---
title: RLSTG stage A design — Riemannian Liquid Spatio-Temporal Graph Network
paper: arXiv 2601.14115v1 (Lu et al. WWW '26)
date: 2026-06-05
tags: [RLSTG, riemannian-LTC, tangent-space, hyperbolic, design-doc, stage-A, iter36]
status: design-doc
---

# RLSTG stage A design (iter#36)

> 论文深读见 [[Riemannian_Liquid_Spatio-Temporal_Graph_Network_RLSTG_研读报告]]
> 目标: 决定复现深度, 设计 tangent-space wrapper, 列出 dependencies 与风险。

## 1. 复现深度决策

论文有 4 块:
1. **§3.2 Tangent-space LTC** — 公式 + 实现 (本仓可复用 90%)
2. **§3.3 Riemannian ODE solver** — 稳定求解器 (需 `geoopt` 或 `torchdyn`)
3. **§4 理论** — stability / universal approximation 推广 (本仓无 formal proof 传统,**不做**)
4. **§5 实验** — ENRON link prediction (数据集/代码无, 需 fallback)

| 块 | 复现? | 理由 |
|---|---|---|
| 1. Tangent-space LTC | ✅ 完整 | 公式与本仓 LTC 几乎同构 |
| 2. Riemannian ODE solver | ✅ 完整 | geoopt 装上即可 |
| 3. 理论证明 | ❌ 跳过 | 仓库无 formal proof 传统, ROI 低 |
| 4. ENRON link prediction | ⚠️ 部分 | 数据集无 → 用 synthetic graph fallback |

**决策**: 完整复现 1+2,部分 4,跳过 3。

## 2. Dependencies

- `geoopt` (PyTorch-compatible manifold ops) — `pip install geoopt` 即可
  - 或: `torchdyn` (Neural ODE + Riemannian 集成)
- 选 `geoopt` (更轻, 只用 `manifold.Hyperboloid` + `exp_map` / `log_map`)

## 3. 文件结构(stage B 之后)

```
lnn/core/riemannian_ltc.py          (新增, ~150 行)
  ├── class TangentSpaceLTC         (复用本仓 LTCNetwork 作为 tangent space 求解)
  ├── class RiemannianLTC           (wrap TangentSpaceLTC, 加 exp/log map)
  ├── class RiemannianLTCNetwork    (堆叠多层)
  └── helper: hyperbolic_distance (验证嵌入合理性)

tests/test_riemannian_ltc.py        (新增, ~150 行, 8 tests)
  ├── exp/log 一致性 (小向量)
  ├── exp/log 数值稳定 (大向量)
  ├── RiemannianLTC forward shape
  ├── gradient 流到 a/b/ε + 几何参数
  ├── hyperbolic embedding 训练后 distance 应 > 0
  ├── 多步 ODE 数值稳定
  ├── 端到端 forward + loss + backward
  └── invalid manifold 仍 raise

scripts/experiment_rlstg_smoke.py   (新增, ~200 行)
  ├── synthetic hyperbolic graph dataset (无 ENRON)
  ├── 4 baselines: cfc/ltc/gru/RLSTG
  ├── 3 seeds × 4 backbones
  └── 输出到 analysis/synthetic_graph/

analysis/riemannian_lnn/
  ├── 2026-06-05_stage_a_design.md (本文件)
  ├── <date>_stage_b_code.md
  └── <date>_stage_c_synthetic.md
```

## 4. 关键公式(本仓实现版本)

论文 Eq. 9 / Eq. 10 简化为:

```python
class TangentSpaceLTC(nn.Module):
    """τ ⊙ h + tanh(W_h h + u) 在 tangent space."""
    def forward(self, h, u, dt=0.1):
        # h, u ∈ T_{x} M, 在切空间操作
        return h + dt * (-self.alpha * h + torch.tanh(self.W_h @ h + u))
```

```python
class RiemannianLTC(nn.Module):
    """tangent-space LTC + exp/log map."""
    def __init__(self, manifold: geoopt.manifolds.Hyperboloid):
        self.manifold = manifold
        self.tangent_ltc = TangentSpaceLTC(...)
    
    def forward(self, x, u, dt=0.1):
        # x 在流形上, u 在 T_x M
        v = self.tangent_ltc(x, u, dt)  # 在切空间
        x_new = self.manifold.expmap(x, v)  # 推回流形
        return x_new
```

## 5. 测试策略

### 5.1 单元测试(8 tests)

| Test | 验证内容 |
|---|---|
| `test_exp_log_consistency_small` | 小向量 `exp(x, log(x, v)) ≈ v` (Taylor 展开精度) |
| `test_exp_log_numerical_stability_large` | 大向量不爆 (Hyperboloid 数值范围 [-1, ∞)) |
| `test_riemannian_ltc_forward_shape` | 输入输出同 manifold shape |
| `test_gradient_flows_through_geometry` | grad 流到 manifold curvature + 隐藏权重 |
| `test_hyperbolic_distance_grows` | 嵌入学习后, 不同 class 间 hyperbolic distance 增加 |
| `test_multi_step_ode_stable` | 10 步迭代无 NaN/Inf |
| `test_end_to_end_loss_backward` | forward + MSE + backward + step |
| `test_invalid_manifold_raises` | 传非 manifold 类型 raise |

### 5.2 集成测试(本仓惯例)

- 与 `lnn/core/graph.py::GraphLNNPredictor` 集成: 加 `riemannian_ltc` recurrent_type
- 跑 synthetic hyperbolic graph: 3 seeds × 4 backbones
- backbone matrix 自动 ingest

## 6. ROI 评估

- **优点**: 公式与本仓 LTC 同构, stage B 实现门槛低
- **缺点**:
  - `geoopt` 需装 (新依赖, 装在 pyenv 3.14.4 兼容性未验)
  - ENRON 数据无,需 fallback synthetic
  - 论文理论证明 (~12 页) 跳过
- **估值**: 2-3 loop 完整 stage B+C, ROI 中等
- **决策**: 进入 stage B 需 iter#37 启动,本 iter#36 完 stage A

## 7. stage A 任务清单(本 iter#36 完成)

- ✅ 决定复现深度(1+2 完整, 3 跳过, 4 部分)
- ✅ 列出 dependencies (`geoopt`)
- ✅ 设计文件结构
- ✅ 设计关键公式(LTC 在 tangent space + exp/log wrapper)
- ✅ 列出 8 个 unit test + 1 集成测试
- ✅ ROI 评估

## 8. 风险 + 缓解

| 风险 | 缓解 |
|---|---|
| `geoopt` 与 torch 2.11.0+cu130 兼容 | 先 `pip install` 试,失败回退 `torchdyn` |
| `expmap` 在 Hyperboloid 上数值不稳 | 用 double precision fallback |
| synthetic graph 不真 | 限报告 synthetic, 不 claim ENRON 复现 |
| 装 `geoopt` 占 Jetson RAM | 评估后再装, 可回退 |

## 9. 计划 schedule

| iter | 工作 | 估时 |
|---|---|---|
| #36 (本轮) | stage A design doc | 0.5 loop(本轮) |
| #37 | stage B: `lnn/core/riemannian_ltc.py` + 8 unit tests | 2-3 loop |
| #38 | stage C: synthetic hyperbolic graph + 3-seed × 4-backbone ablation + matrix ingest | 1-2 loop |
| #39 | stage D: 写复现报告 | 0.5 loop |
| 合计 | 4-5 loop 完整 | — |

## 10. 参考

- 论文: arXiv 2601.14115v1, Lu et al. WWW '26
- 项目页: https://rlstg.github.io
- 依赖: `geoopt` (https://github.com/geoopt/geoopt)
- 本仓现存: `lnn/core/ltc.py`, `lnn/core/cfc.py`, `lnn/core/dynpmnn.py` (6 套 LNN 概念扩展)
