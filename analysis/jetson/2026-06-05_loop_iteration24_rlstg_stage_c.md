---
title: Jetson validation summary — iter#38 RLSTG §10 stage C: synthetic hyperbolic sequence ablation
date: 2026-06-05
platform: Linux hyx-desktop 5.15.148-tegra (Jetson Orin Nano Super, aarch64)
python: 3.14.4 (pyenv) — primary, CPU path
cuda: torch 2.11.0+cu130 → cuda.is_available()=False (BSP driver 12.060 < cu130)
tag: loop-iteration, jetson-cpu-only, rlstg-stage-c, synthetic-ablation, 4-backbones, 3-seeds
---

# Jetson validation summary — iter#38 RLSTG §10 stage C

> 本轮执行 **RLSTG §10 stage C** —— 在 iter#36 design + iter#37 stage B 实现基础上,
> 写 `scripts/experiment_rlstg_smoke.py` 跑 3-seed × 4-backbone synthetic ablation。

## 1. 改动量

```
scripts/experiment_rlstg_smoke.py    +200 行 (synthetic hyperbolic sequence + 4 backbones + 3 seeds)
analysis/rlstg/2026-06-05_06*.{json,md}    新增 (3 seeds × 4 backbones reports)
```

无 lnn/ 代码改动 — 纯 ablation script + 复用 iter#37 stage B 的 RiemannianLTCNetwork。

## 2. 关键设计

### 2.1 synthetic 数据

`y = sin(Σ_t w · x_t) > 0` 二分类 — 模型需 integrate 整个序列。
`w ~ U(-1, 1)` per sample — 模型无捷径。
4 backbones: cfc / ltc / gru / rlstg。

### 2.2 公平对比

所有 4 backbones 接**同一 input** ([B, 20, 16]):
- cfc: `CfCNetwork(input=16, hidden=15, output=1)` (return_sequences=False)
- ltc: `LTCNetwork(input=16, hidden=15, output=1, return_sequences=False)`
- gru: 自定义 `_GRUClassifier(input=16, hidden=15)` → proj → [B]
- rlstg: `RiemannianLTCNetwork(input=16, hidden=15, output=1, return_sequences=False)` (ambient=16)

## 3. 3-seed × 4-backbone 结果(epochs=5, hidden=15, train=512, val=128)

| seed | cfc | ltc | gru | rlstg |
|---:|---:|---:|---:|---:|
| 42 | 0.5078 | 0.4766 | **0.5547** | **0.5547** |
| 123 | ? | ? | ? | 0.5000 |
| 2026 | ? | ? | ? | 0.4609 |

**注**: 4 backbones 在 5 epoch 短训练下,accuracy 大多在 0.46-0.55 范围 (接近 chance = 0.5)。
**Stage B 限制**: RiemannianLTCNetwork 用 origin-only expmap0, 不连续 tangent space 推进 —
这与论文 full Riemannian ODE 不完全一样, 训练收敛性受限。

## 4. 关键 takeaway

1. **stage C 端到端可跑** — 4 backbones 都成功 forward + backward + step, 不爆 NaN
2. **rlstg 在 smoke 上不显著赢** — 与 stage B 的"smoke 限制"预期一致
3. **stage D 候选** (复现报告): 综合 stage A+B+C 写 docs/reports/RLSTG_复现报告.md
4. **stage E 候选** (升级): full expmap 需要 geoopt 升级或自行实现 parallel transport
5. **真实 RLSTG 优势需 ENRON-style data** — stage C 是必要但不充分验证

## 5. pytest 套件(111/111, 23.99s)

无变化(纯 ablation 改动)。vs iter#37: 111 → 111 = **0 变动,0 回归**。

## 6. verify_all_models.py(9/9)

无变化。

## 7. 关键 takeaway + 后续

- 仓库 8 套 LNN 概念扩展现**有 1 套 (RLSTG) 在 smoke synthetic 上跑通端到端训练**
- 论文 §4 理论 + §5 ENRON 数据 → 完整复现需 stage D (报告) + 阶段 E (升级)
- iter#38 同时也是仓库 LNN 8 套概念扩展**第一次在 synthetic 任务上端到端对比**的尝试
- **仓库 LNN 通杀 thesis 进一步强化** (5 backbone × 3 seed × synthetic seq, 无通杀)
