---
title: 2026-06-04 Loop iteration 7 — Comparative LNN vs LSTM v2 (4 backbones × 3 seeds, Mackey-Glass)
date: 2026-06-04
tags: [LNN, LSTM, GRU, CfC, LTC, comparison, ablation, multiseed, loop, validation]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 7 — Comparative LNN vs LSTM **v2**

> `/loop 1h` 第 7 次触发,PRD §8 #5 落地。
> 在与 iter#4 / iter#6 相同的 multi-seed 模板下,
> 对 CfC / LTC / GRU / LSTM 四个 backbone 在 Mackey-Glass 上做对照。
>
> **结论先行**: 在 samples=1200 / hidden=24 / 8 epochs 的中等 smoke 规模下,
> **GRU 反而是最准的(MSE 0.00336)** — 这与"LNN 必赢 LSTM"的简化叙事不一致;
> LNN 类参数效率好 (LTC −50%, CfC −27% vs LSTM) 但 MSE 高 +40~+50%,
> 训练慢 ~2× ~ 7×。
> 把这条**诚实负面信号**记入 [[Comparative_Analysis_of_LNN_and_LSTM_研读报告]]
> v2 作为复现协议的边界条件。

## 1. 实验设计

### 1.1 任务

经典 Mackey-Glass 混沌时间序列(τ=17, β=0.2, γ=0.1, n=10);
1200 样本,70/15/15 train/val/test;`seq_len=32`,horizon=1。
所有 4 个 backbone 用同一份数据、同一份 seed、同一份训练预算
(epochs=8, batch=32, AdamW lr=3e-3)。

### 1.2 模型

| Backbone | 实现 | hidden | output |
|---|---|---|---|
| `cfc` | `lnn.core.cfc.CfCNetwork` | 24 | 1 |
| `ltc` | `lnn.core.ltc.LTCNetwork` (默认 RK4) | 24 | 1 |
| `gru` | `nn.GRU + Linear` | 24 | 1 |
| `lstm` | `nn.LSTM + Linear` | 24 | 1 |

### 1.3 驱动脚本

新增 `scripts/ablation_lnn_vs_lstm_timeseries.py`(~270 行),
follow iter#4/#6 pattern: 4 backbones × N seeds → mean±std 汇总。
单脚本一次跑完 12 个 trial,直接写 JSON + MD 到
`analysis/timeseries_ablation/`。

### 1.4 环境

Jetson Orin Nano Super, **CPU 路径** (CUDA NvMap ENOMEM 持续,iter#2 解释),
pyenv 3.14.4 + torch 2.11.0+cu130。

## 2. 原始结果 (12 trials)

数据来自 `analysis/timeseries_ablation/2026-06-04_024244_lnn_vs_lstm.{json,md}`。

| seed | backbone | params | test MSE | test MAE | train s | inf samples/s |
|---:|---|---:|---:|---:|---:|---:|
| 42 | ltc | 1,321 | 0.00439 | 0.05428 | 132.83 | 127 |
| 7 | ltc | 1,321 | 0.00504 | 0.05922 | 134.62 | 135 |
| 123 | ltc | 1,321 | 0.00532 | 0.05468 | 122.38 | 144 |
| 42 | gru | 1,969 | 0.00283 | 0.04314 | 18.27 | 798 |
| 7 | gru | 1,969 | 0.00357 | 0.04778 | 17.16 | 666 |
| 123 | gru | 1,969 | 0.00368 | 0.04896 | 17.74 | 952 |
| 42 | lstm | 2,617 | 0.00447 | 0.05090 | 18.57 | 973 |
| 7 | lstm | 2,617 | 0.00303 | 0.04539 | 19.01 | 988 |
| 123 | lstm | 2,617 | 0.00295 | 0.04288 | 19.09 | 901 |

(CfC 的 3 seed 数据见 JSON;mean±std 汇总在下表。)

## 3. 跨 seed 汇总 (mean ± std)

| Backbone | params | Test MSE | Test MAE | Train s | Inf samples/s |
|---|---:|---:|---:|---:|---:|
| `cfc` | 1,921 | 0.00521 ± 0.00057 | 0.05706 ± 0.00430 | 43.50 ± 0.28 | 445 ± 13 |
| `ltc` | **1,321** | 0.00491 ± 0.00048 | 0.05606 ± 0.00274 | 129.94 ± 6.61 | 136 ± 8 |
| **`gru`** | 1,969 | **0.00336 ± 0.00046** | **0.04662 ± 0.00308** | **17.72 ± 0.56** | 805 ± 143 |
| `lstm` | 2,617 | 0.00348 ± 0.00085 | 0.04639 ± 0.00410 | 18.89 ± 0.28 | **954 ± 46** |

### 3.1 相对 LSTM baseline

| Backbone | Δparams | Δtest_mse | Δtest_mae | Δtrain_s | Δinf_throughput |
|---|---:|---:|---:|---:|---:|
| `cfc` | **−26.60%** | +49.50% | +22.99% | +130.21% | −53.40% |
| `ltc` | **−49.52%** | +41.04% | +20.85% | +587.77% | −85.79% |
| `gru` | −24.76% | **−3.61%** | +0.50% | **−6.20%** | −15.58% |

## 4. 解读 — 诚实地承认

### 4.1 在此规模下,LNN 类没赢 LSTM

历史报告 [[Comparative_Analysis_of_LNN_and_LSTM_研读报告]] 摘了原论文
的"LNN 在临床序列上比 LSTM 鲁棒"结论。
本轮在 Mackey-Glass 上的 12 trial 结果**反例**:

- CfC test MSE 比 LSTM **高 49.5%** (0.00521 vs 0.00348);
- LTC test MSE 比 LSTM **高 41.0%** (0.00491 vs 0.00348);
- GRU test MSE 比 LSTM **低 3.6%**(在 std 内,基本持平);
- 训练时间: LTC 比 LSTM 慢 **5.9×**(ODE RK4 解算开销),
  CfC 比 LSTM 慢 **2.3×**(解析闭式但仍然 per-step 多算 exp/log)。

### 4.2 为什么差距这么大?

候选解释:

1. **任务不匹配**: Mackey-Glass 是平稳混沌时序,
   "时间常数自适应"和"非规则采样"在此任务上没用。
   原论文的优势体现在 *non-stationary clinical signals* 上。
2. **规模太小**: hidden_size=24 / 8 epochs / 1200 样本是 smoke 级别;
   原论文用 hidden=64-128, ep=100+, real ICU 数据。
3. **未做随机搜索**: 4 个 backbone 共用同一 lr=3e-3,
   对 LSTM 是 "正合适",对 CfC/LTC 可能没有自适应。

### 4.3 为什么仍要把这些结果归档?

因为**项目 PRD §6 要求"可复现 + 可追溯"**,
silently overrepresenting LNN 优势是反 PRD 行为。
本轮的负面信号正是 PRD §8 #5 v2 想要的:**显式标注复现协议的边界条件**,
避免读者把原论文 clinical 结论照搬到通用时序回归任务上。

### 4.4 LTC 参数效率是真正的亮点

LTC 用 **1,321** 个参数(LSTM 的 50.5%)就把 MAE 拉到与 LSTM 同档(0.0561 vs 0.0464,差 21%);
在**模型存储 / 显存预算**是硬约束的场景(微控制器、嵌入式)上,
LTC 单凭参数效率就值得选 — 即使 MSE 高一点。
本轮 [[2026-06-04_loop_iteration6_graph_lnn_tox21_smoke]] 在分子任务上的发现也吻合这条:
**LTC 用 28% 更少参数得到同等 AUC,且方差最低**。

### 4.5 跨 iter 一致性

| iter | 任务 | LTC 表现 |
|---|---|---|
| iter#6 (Tox21-styled molecular) | 二分类 | **赢**: −28% 参数 + +3.5pp acc + 方差最低 |
| iter#7 (Mackey-Glass time series) | 一步回归 | **平**: −50% 参数 + MSE 高 41% |

LTC 在 *小静态图 + 适度训练预算* 上是赢家;
在 *标准 1D 时间序列 + 同等训练预算* 上输给 GRU/LSTM。
这条 "task-conditional ranking" 与远程 EMMA agent 在 commits
`5518b20 / cf14d21 / 7575a9d` 反复发现的 "regime-conditional encoder family"
**完全同源** — 不存在"一个 backbone 通杀所有任务"。

## 5. PRD §8 进展更新

| # | 任务 | 状态 |
|---|---|---|
| 1 | Jetson CUDA wheel | ✅ iter#2 |
| 2 | LiquidTAD 复现 | A ✅ + B ✅ + C-lite ✅ |
| 3 | LFM2.5-1.2B INT4 | pending |
| 4 | EMMA | pending(远程 agent 在做) |
| 5 | **Comparative LNN vs LSTM v2** | **✅ 本轮(诚实负面信号)** |
| 6 | GCN-CfC | ✅ 调研 + ✅ follow-up A |
| 7 | Pareto sweep | ✅ iter#2 |
| 8 | Loop 去重 | pending |

## 6. 衍生工作

| 任务 | 推入 |
|---|---|
| 在 hidden=64 / epochs=50 / samples=4000 重做本对比,看 LNN 优势是否随规模出现 | PRD §8 #5 phase-B |
| 把 lr 改成 per-backbone 自适应(用 `lr_scheduler` + warmup) | NEXT_STEPS |
| 复现原论文 PhysioNet 临床序列子集 | PRD §8 #5 phase-C |
| 把"task-conditional ranking" 表写进 [[OPTIMIZATION_STRATEGIES]] | docs |

## 7. 参考产物

- 源代码: `scripts/ablation_lnn_vs_lstm_timeseries.py`(本轮新增,~270 行)
- JSON+MD: `analysis/timeseries_ablation/2026-06-04_024244_lnn_vs_lstm.{json,md}`
- 上一轮: [[2026-06-04_loop_iteration6_graph_lnn_tox21_smoke]]
- 原研读: [[Comparative_Analysis_of_LNN_and_LSTM_研读报告]]
- PRD: [[PRD_LNN_Edge_Research]] §8 #5
- 相关现有脚本: `scripts/benchmark_comparison.py`(单 seed, 本仓老版本)
