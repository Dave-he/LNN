---
title: Jetson validation summary — iter#34 EntroLnn deep read (arXiv 2601.06195, SAC '26)
date: 2026-06-05
platform: Linux hyx-desktop 5.15.148-tegra (Jetson Orin Nano Super, aarch64)
python: 3.14.4 (pyenv) — primary, CPU path
cuda: torch 2.11.0+cu130 → cuda.is_available()=False (BSP driver 12.060 < cu130)
tag: loop-iteration, jetson-cpu-only, paper-deep-read, EntroLnn, battery-SoH, transformable-LNN, SAC-2026
---

# Jetson validation summary — iter#34 EntroLnn deep read

> 本轮执行 **用户提示 "搜索LNN相关论文代码"** —— 深读 arXiv 2601.06195
> "EntroLnn — Entropy-Guided Transformable Liquid Neural Networks" (Li et al. SAC '26)。

## 1. 改动量

```
docs/reports/EntroLnn_Entropy-Guided_Transformable_LNN_研读报告.md   新增 (~200 行, 8 节)
docs/PRD_LNN_Edge_Research.md   C-level 表 +1 行 (EntroLnn)
```

无 lnn/ 代码改动 — 本轮纯论文研读。

## 2. 论文核心

- **方法**: LTC + battery SoH 实时精化("transformable" 静态+动态 LNN)
- **公式**: `dh/dt = -α ⊙ h + tanh(W_h h + ū)` (Eq. 10) — **与本仓 `LTCNetwork` 几乎同构**
- **数据**: MIT-Stanford 124 LFP 18650 电池, 寿命 500-2000+ 周期
- **任务**: CFT 实时精化 + EoL 预测
- **结果**: **MAE 0.004577 for CFT** + **18 cycles for EoL**
- **代码**: **无官方代码仓** (CC BY 4.0)

## 3. 关键发现(与本仓 LTC 同构)

```
论文:  dh/dt = -α ⊙ h + tanh(W_h h + ū)   (Eq. 10, α 是 per-dim 衰减)
本仓:  lnn/core/ltc.py::LTCCell (sigmoid-gated 闭式)
```

| 维度 | 论文 (EntroLnn) | 本仓 (LTCNetwork) |
|---|---|---|
| Per-dim 衰减 α | 显式 `α ⊙ h` | sigmoid-gated `time_scale` |
| 隐藏状态维度 | 64 | 可配置 (default 24-48) |
| ODE 求解器 | RK4 adaptive step | 闭式解 (无 solver) |
| "transformable" | 部署期梯度精化 | **未实现** — iter#34 设计参考 |

## 4. 评级

| 维度 | 评级 |
|---|---|
| 学术新意 | A- (SAC '26, "transformable" 概念新意) |
| 工程价值 | A- (Jetson 边缘 battery monitor) |
| 代码可获取 | C (无官方代码) |
| 本仓优先级 | **B+** (公式同构 ROI 高) |

## 5. 复现路线(4 stages, ~6 loop)

| Stage | 出口物 | 估时 |
|---|---|---|
| A. 调研 + design | `analysis/entrolnn/<date>_design.md` | 0.5 loop |
| B. 装 RK4 + 写 `lnn/core/transformable_ltc.py` (~150 行) | code + unit test | 2-3 loop |
| C. 跑 MIT-Stanford 或 fallback synthetic data 3 seeds | analysis + paper | 1-2 loop |
| D. 写复现报告 | docs/reports/EntroLnn_复现报告.md | 0.5 loop |

## 6. pytest 套件(102/102, 28.22s)

无变化(纯论文改动)。vs iter#33: 102 → 102 = **0 变动,0 回归**。

## 7. verify_all_models.py(9/9)

无变化。

## 8. 关键 takeaway

1. **EntroLnn 公式与本仓 LTC 几乎同构** —— 论文 Eq. 10 是显式 α ⊙ h 形式,本仓是 sigmoid-gated 闭式
2. **"transformable" 思想可借鉴** —— 同一网络架构 + 部署期在线精化参数,本仓 `experiment_graph_lnn_molecule.py --frozen-encoder` 是姐妹模式
3. **Jetson 边缘 battery monitor 是真实应用** —— INA219 + 温度传感器 → EntroLnn 简化版 → SoH %
4. **仓库 7 → 8 套 LNN 概念扩展**: LTC / CfC / CT-LTC / PDNA-pulse / SVAF-τ-blend / DynPMNN-FHN / RLSTG-LTC / **transformable-LTC**
5. **暂时搁置 stage B 实施** —— 优先做 §10 next-up 中无阻塞项
