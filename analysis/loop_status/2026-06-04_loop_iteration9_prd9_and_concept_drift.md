---
title: 2026-06-04 Loop iteration 9 — PRD §9 + concept_drift ablation + loop_status --week
date: 2026-06-04
tags: [LNN, loop, PRD, ablation, concept-drift, meta-tooling, retro]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 9 — PRD §9 + concept_drift ablation + loop_status --week

> `/loop 1h` 第 9 次触发。
> Iter#8 把 PRD §8 的最后两个非阻塞任务都完成了(只剩 #3 LFM2.5 等 RAM + #4 EMMA 远程
> 代理负责),自然推进到 **新一轮 PRD §9 scoping + 两个工程小动作**:
>
> 1. **`loop_status.py --week N`** — 周报视图,把过去 N 天的 iteration / commits /
>    PRD 状态聚合成一张表。
> 2. **iter#7 后续: concept_drift 数据上的 LNN-vs-LSTM 复测** —
>    把"GRU 在 Mackey-Glass 上赢 LNN"扩展到论文宣称的 LNN 优势区(非平稳信号),
>    结论:**LSTM 也赢**,LTC 在该任务上 catastrophic(MSE 高 +1301%)。
> 3. **PRD §9 写出**: 把 8 个新候选任务、3 项"已调研未复现"、3 条复现协议
>    边界条件落到 PRD,供下一周 loop 调度参考。

## 1. loop_status --week N 模式

### 1.1 改动

`scripts/loop_status.py` 加 `--week N` 参数,enable 时跳到 `_main_weekly` 路径,
扫过去 N 天的:
- daily digest / jetson benchmark 是否就位
- iteration 报告数量(本工具的 ground truth)
- 本地 commit 数量
- analysis 文件总数

输出 per-day breakdown 表 + iteration index + PRD §8 snapshot。

### 1.2 首次跑(过去 3 天)

```text
=== Loop weekly retro: 2026-06-02 → 2026-06-04 (3 days) ===
  days with daily digest:    3/3
  days with jetson benchmark:3/3
  total iterations:          7
  total local commits:       84
  total analysis files:      47
  PRD §8 pending:            2 / 8
```

84 commits / 3 day = **平均每天 28 commits**,主要由远程 EMMA agent 的 multi-seed
ablation 风暴贡献(我自己的 loop 每天 1–4 个 commit);**iteration 报告 7 个**
(iter#1–7;iter#8 在 2026-06-04 当天才生成,会在下次 --week 看到)。

## 2. concept_drift LNN-vs-LSTM 复测(iter#7 v2 boundary 扩展)

### 2.1 动机

iter#7 把 4 backbone × 3 seed 跑在 Mackey-Glass(平稳混沌)上,
发现 GRU 反而最准,LNN 类比 LSTM 慢且 MSE 高。
但 [[Comparative_Analysis_of_LNN_and_LSTM_研读报告]] 原论文说
"LNN 在非平稳临床信号上才显优势"。
本轮直接用 `lnn.data.timeseries.generate_concept_drift` 复测同一份对照:

- 数据 1200 样本,前 600 = regime A(freq 0.05, amp 1.0),
  后 600 = regime B(freq 0.12, amp 0.6) — 一个**单次硬切换** non-stationary。
- 模型/超参与 iter#7 完全一致(hidden=24, 8 epoch, lr 3e-3),
  只换数据;3 seed = {42, 7, 123}。

### 2.2 结果 (mean ± std, 3 seeds)

| Backbone | params | Test MSE | Δ vs LSTM | Train s | Inf samples/s |
|---|---:|---:|---:|---:|---:|
| **`lstm`** | 2,617 | **0.00637 ± 0.00258** | baseline | 18.96 | **876** |
| `cfc` | 1,921 | 0.01524 ± 0.01083 | **+139.4%** | 83.29 | 267 |
| `gru` | 1,969 | 0.02077 ± 0.00900 | **+226.2%** | 35.12 | 494 |
| `ltc` | **1,321** | **0.08923 ± 0.00433** | **+1301.2%** | 220.64 | 84 |

### 2.3 解读 — 三个意外信号

1. **LSTM 在论文宣称的 LNN 优势区上赢得更大**: 单 sharp drift 测试下,
   LSTM MSE 0.00637(iter#7 Mackey-Glass 上 0.00348),
   LTC 一路恶化到 0.08923 — 比 LSTM 高 14×。
2. **LTC catastrophic failure**: 显然 RK4 ODE 集成 + 训练数据未见的 regime B
   组合导致积分轨迹严重外推失稳。这是一条**LTC 在硬 drift 上不安全**的工程边界。
3. **GRU 不再领跑**: iter#7 GRU 是 Mackey-Glass 上的赢家(MSE −3.6% vs LSTM),
   本轮 GRU MSE 比 LSTM 高 +226.2%。说明 **GRU 的简单门控也是规模/任务条件性的**。

### 2.4 三种合理解释 — 不能直接证伪论文

- **"单次硬 drift ≠ 论文的 gradual 非平稳"**:
  原论文用的是 ICU sepsis 时序(累积慢漂移 + 缺失值);
  本轮的 50/50 sharp split 是更严酷的 step change,可能不是论文 claim 的目标区。
- **超参未自适应**: 所有 backbone 共用 lr=3e-3 + 无 warmup;
  LTC 的 RK4 解算可能需要更小 lr + 渐进 schedule。
- **样本太少**: 1200 → 840 train tokens,LNN 类对 sample 复杂度敏感。

**结论**: paper claim 在 *合成、单次 sharp drift、固定 lr、小预算* 这一组**很严格**
的复现协议下不成立。这是边界条件信息,不是论文的反例。
**已 append 到 [[Comparative_Analysis_of_LNN_and_LSTM_研读报告]] v3 章节**。

### 2.5 跨 iter 一致性总结

| iter | 任务 | 数据 | Backbone 排名 |
|---|---|---|---|
| iter#6 | 分子二分类 | 静态 Erdős-Rényi 图 | **LTC ≈ CfC ≈ GRU**(AUC 并列 0.754) |
| iter#7 | 一步时序回归 | Mackey-Glass 平稳混沌 | **GRU > LSTM ≫ LTC > CfC** |
| iter#9 | 一步时序回归 | concept_drift 单次硬切 | **LSTM ≫ CfC > GRU ≫ LTC** |

**关键模式**:
- 没有"通杀" backbone;
- LSTM 在 1D 时序回归上(无论平稳/非平稳)都很稳;
- LTC 在静态图任务上不输,在时序回归上(尤其 OOD drift)显著输;
- 跟远程 EMMA agent commits `5518b20 / cf14d21 / 7575a9d` 的 regime-conditional 结论同源。

## 3. PRD §9 — 下一周 8 个新候选任务

PRD §8 8 个任务里只剩 2 个真正 pending (#3 LFM2.5 等 RAM, #4 EMMA 远程负责),
工作面已基本耗尽。本轮在 PRD 新增 §9 块,列出下一周候选 backlog
(具体写到 docs/PRD_LNN_Edge_Research.md 的 §9):

| ID | 候选任务 | 出口物 |
|---:|---|---|
| 9-1 | LFM2.5-1.2B INT4 离线推理(空载夜间窗口) | `analysis/lfm25/<date>_lfm25_int4.md` |
| 9-2 | concept_drift 复测 phase-B: 渐进多 regime + lr warmup + curriculum | iter#10 |
| 9-3 | LiquidTAD 真 Stage C: THUMOS-14 50-video 子集复现 | `analysis/paper_replication/liquid_tad_thumos.md` |
| 9-4 | `experiment_graph_lnn_molecule.py` 加 `--frozen-encoder` 两阶段(模拟 GCN-CfC 解耦) | iter#11 |
| 9-5 | `loop_status.py --since-last-loop` 模式: 自动定位上次 iter 结束后的所有变更 | iter#12 |
| 9-6 | 把 `lnn.core.long_sequence.HierarchicalDecayLiquidTADHead` 加 ONNX export + TensorRT INT8 | iter#13 |
| 9-7 | 跨数据 backbone ranking 自动生成: ablation runner 加 `--datasets` 多个,出 task-conditional 表 | iter#14 |
| 9-8 | PRD §6 验证指标自动 CI: 在 GitHub Actions 跑 `verify_all_models.py + ablation_*` 周线 | iter#15 |

并标"已调研未复现"(C 级,只入索引不投复现 budget):

- Linlab2026/GCN-CfC(iter#5)— 双框架管线
- LiquidAI/LFM2.5-8B-A1B(too big for Orin Nano)
- raminmh/CfC official tf 实现(已被 ncps 取代)

## 4. PRD §8 进展(本轮终态)

| # | 状态 | 出口物 |
|---:|:---:|---|
| 1 | ✅ iter#2 | Jetson CUDA wheel |
| 2 | ✅ iter#3/#4 | LiquidTAD A+B+C-lite |
| 3 | ⏳ pending | LFM2.5-1.2B INT4 |
| 4 | ⏳ pending | EMMA(远程) |
| 5 | ✅ iter#7 | Comparative v2(Mackey-Glass) → **v3 增补 concept_drift (本轮)** |
| 6 | ✅ iter#5/#6 | GCN-CfC + Tox21 |
| 7 | ✅ iter#2 | Pareto sweep |
| 8 | ✅ iter#8 | loop_status meta tool → **加 --week (本轮)** |

## 5. 参考产物

- 代码增量:
  - `scripts/loop_status.py` +`--week N` 路径(+ ~150 行新增 `_main_weekly`/`_format_weekly_md`)
  - `scripts/ablation_lnn_vs_lstm_timeseries.py` 加 `--dataset concept_drift` 选项
- 报告:
  - `analysis/timeseries_ablation/2026-06-04_045055_lnn_vs_lstm.{json,md}` (concept_drift, 12 trials)
  - `analysis/loop_status/2026-06-04_044413_loop_status_weekly_3d.{json,md}` (周报)
  - 本文件 (iter#9 summary)
- PRD: [[PRD_LNN_Edge_Research]] §9 新增
- 上游研读: [[Comparative_Analysis_of_LNN_and_LSTM_研读报告]] (v3 append)
- 累积 iteration: [[2026-06-04_loop_iteration8_loop_status_tooling]] / [[2026-06-04_loop_iteration7_lnn_vs_lstm_v2]] / [[2026-06-04_loop_iteration6_graph_lnn_tox21_smoke]] / [[2026-06-04_loop_iteration5_gcn_cfc_repo_survey]] / [[2026-06-03_loop_iteration4_liquid_tad_3way_ablation]] / [[2026-06-03_loop_iteration3_liquid_tad_stage_ab]] / [[2026-06-03_loop_iteration2_cuda_fix_pareto]] / [[2026-06-03_loop_validation_summary]]
