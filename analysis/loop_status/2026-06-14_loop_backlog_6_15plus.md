---
title: 2026-06-14 Loop backlog — 6/15+ 可执行项备忘
date: 2026-06-14
tags: [LNN, loop, backlog, planning, prd-10-pending, iter40-41-candidate]
parent: [[PRD_LNN_Edge_Research]]
related:
  - [[DLNet_Dual_Stage_Distillation_Pareto_LNN_2601.06227_研读报告]]
  - [[Liquid_NN_MR_MoE_Sepsis_2606.12240_研读报告]]
  - [[Liquid_Neural_Networks_Latest_Papers_Summary]]
---

# 2026-06-14 Loop backlog — 6/15+ 可执行项备忘

> 由 2026-06-14 daily 完成后的"前几日待优化"盘点触发(今日 5 项 task 中 4 项已完成 / 1 项取消)。
> 本备忘列出 6/15 起按 ROI 与依赖排序的可执行 backlog,供 `/loop 1h` 调度参考。
> **非正式 iter**, 不涉及 `lnn/` 代码改动, 仅文档与决策。

## 1. 高优先级 (P0, 6/15-6/16 可启动)

### 1.1 PRD §10 #23 MR-MoE 异 τ + MoE 脓毒症复现 stage A (iter#40)

- **现状**: iter#39 (2026-06-12) 已落地研读报告 `Liquid_NN_MR_MoE_Sepsis_2606.12240_研读报告.md`, PRD §10 #23 候选已建;**无 stage A 代码落地**。
- **阻塞**: PhysioNet/CinC 2019 Sepsis 数据需下载 (~250 MB, 40344 ICU stay),`wget` 走 https://physionet.org/files/challenge-2019/1.0.0/。
- **可启动最小子任务**:
  - (a) `lnn/core/cfc.py::CfCCell` 加 `n_tau: int = 1, tau_scales: tuple = (0.1, 1.0, 10.0)` 配置项 + 8 个单测覆盖(沿用 iter#39 描述)
  - (b) `lnn/core/mr_moe.py::MRMoEHead` 骨架 (~150 行) + K=3/异 τ/双注意力单测
- **预期产出**: 1 commit (`feat(cfc): n_tau config + MRMoEHead skeleton`) + 8 unit tests
- **ROI**: A- (公式同构 95%, 1 行 config 即可吃下多 τ 收益)

### 1.2 PRD §10 #24 DLNet Dual-Stage KD + Pareto LNN 边缘电池 stage A (iter#41)

- **现状**: 2026-06-14 研读已落地 `DLNet_Dual_Stage_Distillation_Pareto_LNN_2601.06227_研读报告.md`, PRD §10 #24 候选今日刚建;**无 stage A 代码落地**。
- **阻塞**: Arduino / Jetson 真机不在手 → stage G (硬件验证) 阻塞; stage A-F 全部可在 CPU 跑。
- **可启动最小子任务**:
  - (a) `lnn/core/variants.py::EulerLTCNetwork::to_embedded()` (~50 行, 收编 onnx/QAT/export 入口) + 3 单测
  - (b) `lnn/core/distill.py::DualStageDistiller` (~150 行, MSE + hidden-state KD loss + Stage 2 恢复式蒸馏 + Pareto 前沿采样) + 7 单测
  - (c) `scripts/replicate_dlnet.py` (~280 行, NASA PCoE B0005/B0006 训练 teacher → Stage 1 → Stage 2 + Pareto 选优) + 5 单测 + 3-seed quick 烟测
- **预期产出**: 1 commit (`feat(distill): DLNet dual-stage + Pareto LNN battery SOH stage A`) + 15 unit tests + 1 analysis/dlnet JSON+MD
- **ROI**: A (公式同构 95% + ICPR 2026 接收 + 真实硬件闭环)

### 1.3 temporal_dropout stage B 多 seed 复现

- **现状**: 2026-06-14 重跑仅 1 seed;报告 §6 局限与未来工作已标注 N=1 风险。
- **可启动最小子任务**:
  - (a) `scripts/replicate_temporal_dropout.py` 加 `--seeds 42 2026 777` CLI, 聚合 mean ± std
  - (b) `analysis/replication/temporal_dropout/3seed_aggregate.{json,md}` + 退化曲线加 std 误差带
  - (c) `tests/test_temporal_dropout_3seed.py` 1 个 regression test
- **预期产出**: 1 commit (`chore(replication): temporal_dropout stage B 3-seed`) + 1 regression test
- **ROI**: B+ (iter#11/35 N=1 教训兑现 + 论文 claim 稳健性背书)

## 2. 中优先级 (P1, 6/17-6/19 窗口)

### 2.1 imitation_lnn 手工跑多 backbone 矩阵 (无自动 cron)

- **现状**: 6/12-6/14 三天 `analysis/control/imitation_lnn` 全部 `cfc+mdn cpu 12 epoch test_mse 0.0011`, 单调;无自动 cron, 是用户早晨手工跑(见 6 个文件 mtime 06:40-08:36)。
- **建议手工调用矩阵** (一次性跑, 写入 `analysis/control/matrix/`):
  - 6 个组合: `{cfc, ltc, autoncp} × {mdn, mse}` + default cpu 12 epoch + seed=42
  - 复用 `scripts/experiment_imitation_lnn.py` 已就位的 `--recurrent` 与 `--head` CLI
  - 总耗时: 6 × 2.65s = 16s
- **预期产出**: 6 个 JSON + 6 个 MD + 1 个 `matrix_summary.{json,md}`
- **ROI**: B (让 imitation cron 跨日有 backbone matrix 信号)

### 2.2 LFM2.5-1.2B INT8 推理 RAM blocker 重探

- **现状**: PRD §10 #7 阻塞 (8GB RAM 跑 1.7GB 模型需空载窗口);6-11 GGUF 集中日已 3 条独立路径。
- **可探查**:
  - (a) 当前 `models/` 目录是否有现成 LFM2.5 GGUF 权重?
  - (b) 当前 RAM 实际使用情况 (`vm_stat | grep "Pages free"`)?
  - (c) 若空载窗口存在, 跑 `scripts/run_llm_micro_eval.py --backend llama-cli --gguf-path <path>` 1 次 7/7 sanity
- **预期产出**: `analysis/lfm25/<date>_lfm25_int8_retry.{json,md}`
- **ROI**: B (若成功解除 blocker, 释放 §10 完成度 +1)

### 2.3 SVAF τ-modulated peer-blending stage B (PRD §10 #9)

- **现状**: iter#22 (2026-06-04) 落地 stage A;**stage B pending (P2)**。
- **可启动最小子任务**:
  - toy 2-agent mesh + τ_i ∈ {1, 10, 60} 三组神经元,N 步耦合后看 spectral diff
  - 验证"fast τ 同步 / slow τ 主权"现象
- **预期产出**: `analysis/svaf/2026-06-04_tau_toy.md` + ~100 行 core code
- **ROI**: C+ (P2 mini-task)

## 3. 低优先级 (P2, 6/20+ 长尾)

### 3.1 PRD §8 #2 LiquidTAD 真 Stage C THUMOS-14 50-video

- **阻塞**: 需 CUDA 空载窗口 + THUMOS-14 50-video 子集准备
- **ROI**: B (真实 TAD benchmark)

### 3.2 PRD §10 #22 case B 压力测试 (T=64/128 + nan_count guard)

- **现状**: iter#36 已落地 case B 无人机 4-DoF 回归;case B 风险标注"T>64 ODE 发散"
- **可启动最小子任务**:
  - (a) `scripts/experiment_device_control_cases.py::run_case_drone` 加 `--seq-len {64, 128}` + nan_count guard
  - (b) 失败时自动 rollback 到 seq_len=32 + 写入 `analysis/device_control/drone_<date>_T128_failed.md`
- **ROI**: C (补完 4-case 中唯一已知风险点)

### 3.3 PRD §10 #22 case C 扩展 (NCP sparse 真激活)

- **现状**: iter#36 case C 工业控制 1-DoF 倒立摆 IL 用 LTC recurrent,NCP sparse wiring 留为 follow-up
- **可启动最小子任务**:
  - (a) `scripts/experiment_device_control_cases.py::run_case_industrial` 加 `--recurrent-type autoncp`
  - (b) 配套单测覆盖 sparse wiring 行为
- **ROI**: C (完成 NCP 全 4-case 覆盖)

## 4. 已闭环 / 不需要 backlog

- ✅ 6/14 digest + watchlist + papers tracker + 3 日 imitation control (892c879)
- ✅ DLNet 研读 + LNN_深度研读报告索引 + Latest_Papers_Summary §2.10 + PRD §10 #24 (今日优化)
- ✅ 6/14 digest 加 DLNet 研读链接 (今日优化)
- ✅ temporal_dropout 重跑 timing + 报告 §6 局限说明 (今日优化)

## 5. 决策原则 (供 6/15+ 选择任务时参考)

1. **iter 模式 (深度) vs daily 模式 (广度)**: 上午 iter (P0 #1-3), 下午 daily (P1 #2.1 手工矩阵)。
2. **1-seed 教训**: 任何 1 seed 即下结论的任务都必须有 stage B 多 seed 计划 (§1.3, §2.2 类)。
3. **硬件 blocker**: 不阻塞 iter 流程;RAM/CUDA 空载窗口来时插入。
4. **诚实负面预算**: 每个新复现任务预留 1 个"负面结论"出口,不强制成功。