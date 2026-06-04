---
title: Jetson validation summary — iter#31 RLSTG deep read (arXiv 2601.14115, WWW '26)
date: 2026-06-04
platform: Linux hyx-desktop 5.15.148-tegra (Jetson Orin Nano Super, aarch64)
python: 3.14.4 (pyenv) — primary, CPU path
cuda: torch 2.11.0+cu130 → cuda.is_available()=False (BSP driver 12.060 < cu130)
tag: loop-iteration, jetson-cpu-only, paper-deep-read, RLSTG, riemannian-LTC, WWW-2026
---

# Jetson validation summary — iter#31 RLSTG deep read

> 本轮执行 **用户提示 "搜索LNN相关论文代码"** —— 深读 arXiv 2601.14115
> "Riemannian Liquid Spatio-Temporal Graph Network" (Lu et al. WWW '26)。

## 1. 改动量

```
docs/reports/Riemannian_Liquid_Spatio-Temporal_Graph_Network_RLSTG_研读报告.md   新增 (~200 行)
docs/PRD_LNN_Edge_Research.md   C-level 表 +1 行 (RLSTG)
```

无 lnn/ 代码改动 — 本轮纯论文研读。

## 2. 论文核心

- **方法**: LTC + Riemannian manifolds + tangent space ODE
- **公式**: `d/dt h = f(h, x, t)` 在 tangent space 求解,`h_{t+Δt} = exp_{h_t}(Δt · dh/dt)` 推回流形
- **几何**: hyperbolic (树状层级) / spherical (环状结构) — 论文主用双曲
- **数据**: ENRON 邮件网络 (184 员工, 3 年)
- **Baselines**: 10 个 (JODIE / DyRep / TGAT / TCL / TGN / GraphMixer / DyGFormer / HTGN / FreeDyG / HGWaveNet)
- **任务**: link prediction (transductive + inductive)
- **理论**: 推广 LTC stability / universal approximation 到黎曼域
- **代码**: **无官方仓**,仅项目页 rlstg.github.io

## 3. 关键发现(与本仓 6 套 backbone 对齐)

```
LTC (本仓核心)
├── 欧几里得: lnn/core/ltc.py::LTCNetwork                ← 已有
├── 黎曼:    lnn/core/riemannian_ltc.py (TBD)            ← RLSTG 模式 (第 7 套候选)
├── ODE 闭式: lnn/core/cfc.py::CfCNetwork                ← 已有
├── FHN ODE:  lnn/core/dynpmnn.py::FHNCell               ← 已有 (iter#23)
├── 频率 augment: cfc.PDNAPulseHead                      ← 已有 (iter#19)
└── τ 调制 augment: cfc.tau_modulated_blend              ← 已有 (iter#22)
```

## 4. 评级

| 维度 | 评级 |
|---|---|
| 学术新意 | A (WWW '26 accepted) |
| 工程价值 | B+ (需新依赖 `geoopt`,但 tangent-space 复用本仓 LTC) |
| 代码可获取 | C (无官方代码,仅项目页 demo) |
| 本仓优先级 | **B** (ROI 低于 DynPMNN,但与 graph_lnn 是姐妹工作) |

## 5. 复现路线 (4 stages, ~6 loop)

| Stage | 出口物 | 估时 |
|---|---|---|
| A. 调研 + design | `analysis/riemannian_lnn/<date>_design.md` | 0.5 loop |
| B. 装 geoopt + `lnn/core/riemannian_ltc.py` (~120 行) | code + unit test | 2-3 loop |
| C. 跑 ENRON link prediction toy (3 seeds, vs 1-2 baseline) | analysis + paper | 1-2 loop |
| D. 写复现报告 | docs/reports/Riemannian_LTC_复现报告.md | 0.5 loop |

## 6. pytest 套件(117/117, 49.77s)

无变化(纯论文改动)。vs iter#30: 117 → 117 = **0 变动,0 回归**。

## 7. verify_all_models.py(9/9)

无变化。

## 8. 关键 takeaway

1. **RLSTG = LTC + 黎曼流形** 是 LTC 的"非欧几何"扩展,与本仓欧几里得 LTC 是**两路线**
2. **tangent space 的 ODE 公式** 优雅: 局部欧几里得 → `exp` 推回 → ODE 求解器**不需要修改**
3. **理论推广** (stability / universal approximation) 是本仓空白,完整复现需要 formal proof
4. **代码 ROI 评估**: 复现需装 `geoopt` + 写 tangent-space 包装 + 跑 ENRON 数据 + 与 10 个 baseline 对比, **vs DynPMNN** (5 变体 ablation, sMNIST zero-cost) ROI 更低
5. **暂时搁置 stage B 实施** —— 优先做 §10 next-up 中无阻塞项
