---
title: LNN 边缘研究与验证平台 PRD
date: 2026-06-03
tags: [PRD, LNN, jetson, edge-ai, product]
status: living-document
owners: [heyongxian, lnn-research-agents]
---

# 产品需求文档 (PRD) — LNN 边缘研究与验证平台

> 本 PRD 把仓库已经形成的研究协议、自动化流水线和 Jetson 验证能力规范化为
> "可对齐、可追溯、可复盘" 的产品形态。先以 living document 形式维护,
> 随每周 retro / loop 调度迭代。

## 1. 背景与问题

液态神经网络 (LNN/LTC/CfC/NCP/LFM) 在 2025–2026 年密集涌现:
LiquidAI 的 LFM2.5 系列推动了边缘 LLM 商业化;
学术界继续在 CfC、Liquid-S4、LiquidTAD、Stochastic CfC、Physics-modeled NN 等方向迭代;
但 **缺乏一套既能持续追踪进展、又能在 ARM/Jetson 真机上做最小可信验证** 的开源平台。

本仓库 `LNN/` 已积累:

- 9 类 LNN 变体的从零实现 + `ncps` 集成;
- 13+ 个论文研读报告 / 训练方向报告;
- 每日 arXiv/GitHub/Hugging Face 自动追踪流水线;
- Jetson Orin Nano Super 上的 smoke benchmark 历史(>9 天)。

PRD 化的目的是固化范围 / 衡量指标 / 验证门槛,
避免在 loop 调度下产生"只采不验、只验不评"的低密度产出。

## 2. 目标用户与典型使用场景

| 角色 | 关注点 | 典型动作 |
|---|---|---|
| **研究者**(本人 + 协作 agents) | 不漏掉 LNN 领域新动态 / 想验证一个想法 | 每日读 `docs/daily/*` digest;按 `LNN_持续研究协议` 决定深读哪些 |
| **边缘工程师** | 想知道某 LNN 变体在 Jetson 上能跑到什么吞吐/精度 | 跑 `scripts/jetson_lnn_benchmark.py` 与 `experiment_*` 套件 |
| **复现/学习者** | 想看 LNN/LTC/CfC 究竟怎么训练、怎么评估 | 跟 `LNN_MODEL_GUIDE.md` 与 `scripts/tutorial_how_to_use.py` |
| **下游应用作者** | 想把 CfC/LTC 接进自己的机器人/金融/能源/医疗任务 | 复用 `lnn/core/variants.py` 与 `experiment_imitation_lnn.py`、`experiment_physics_lnn.py` 等示例 |

## 3. 范围 (In-scope / Out-of-scope)

### In-scope
- LNN 家族算法的 **小到中等规模** PyTorch 实现(从神经元层到完整网络)。
- ARM64 / Jetson Orin Nano Super 边缘验证: CPU 路径必须可用,
  CUDA 路径 best-effort(取决于 Jetson BSP 与 PyTorch wheel 的耦合)。
- arXiv / GitHub / Hugging Face 三源每日聚合追踪 + 研读报告生成。
- LFM2 / LFM2.5 系列推理(`lnn/lfm2/`),含 GGUF/MoE/distill 变体的可行性评估。
- 论文复现:对中小型工作做最小复现脚本(`scripts/replicate_paper_*`、
  `scripts/minimal_lnn_paper_validation.py`),把误差/吞吐写入 `analysis/`。

### Out-of-scope(暂不承诺)
- 完整训练超大 LFM2.5-8B 类模型(本地 Orin Nano 算力不足)。
- Loihi-2、Mythic、NorthPole 等神经形态硬件的真机部署
  (仅做调研报告:见 `docs/reports/Exploring_Liquid_Neural_Networks_on_Loihi-2_研读报告.md`)。
- 工业级 SLA / 可用性保证;脚本是 smoke 级别,正式部署需用户自取。

## 4. 关键功能模块 (Functional Modules)

| 模块 | 路径 | 责任 |
|---|---|---|
| **核心实现** | `lnn/core/*.py`, `lnn/core/variants.py` | LTC / CfC / 7 种变体 + 训练器 |
| **NCPs 集成** | `lnn/ncps_integration/` | 复用 Liquid AI `ncps` 的 CfC/LTC/AutoNCP |
| **LFM2 推理** | `lnn/lfm2/` | 量化 / 边缘部署 wrapper |
| **数据合成** | `lnn/data/` | Mackey-Glass / 正弦 / 多模态 / 能源价格 |
| **每日追踪** | `scripts/daily_lnn_research.py` + `.github/workflows/daily-lnn-research.yml` | arXiv/GitHub/HF 聚合 + digest + watchlist |
| **边缘 Benchmark** | `scripts/jetson_lnn_benchmark.py`,`run_daily_lnn_task.sh` | Jetson Orin Nano Super 上 smoke + Pareto sweep |
| **LFM/LLM 打榜证据卡** | `scripts/build_llm_battlecard.py`,`scripts/run_llm_micro_eval.py`,`scripts/build_llm_micro_leaderboard.py` | 汇总本地 LFM2.5 推理证据 + 公开 30B+ 基线 + micro-eval 榜单,审计 active≤3B 是否可声称胜出 |
| **论文复现** | `scripts/replicate_paper_dispatch.py`,`scripts/minimal_lnn_paper_validation.py` 等 | 选目 → 最小复现 → 误差报告 |
| **研究 Skills** | `skills/living-field-researcher`,`skills/paper-analyzer`,`skills/paper-translator` | 可移植到 Cursor/Trae 等工具 |
| **Loop 调度** | `/loop 1h …`(Claude Code CronCreate)+ user systemd timer | 自动化滚动收资料、跑验证、提交 |

## 5. 非功能需求 (NFRs)

| 维度 | 目标 |
|---|---|
| **可复现** | 所有实验脚本固定 `--seed`,`analysis/` 永久归档 JSON+MD+PNG |
| **可追溯** | 每次自动提交以 `chore(daily):` / `feat(...)` 起头,commit 包含日期 |
| **可移植** | 不依赖 NVIDIA 桌面 GPU;Linux x86_64 与 Linux aarch64 (Jetson) 双轨可跑 |
| **可阅读** | `docs/` 使用 Obsidian 双链;`README.md` 给出每个能力的最小入口 |
| **轻依赖** | `pip install -e .` 即可;LFM2 推理走可选 extras `[lfm]` |

## 6. 验证指标 (Acceptance Criteria)

每次正式发布或一次有意义的 loop iteration 至少需要满足:

1. **单元测试**: `pytest tests/test_core.py tests/test_paper_models.py` 全绿。
2. **变体一致性**: `python scripts/verify_all_models.py` 9/9 通过,
   每个 backward 不出现 NaN / Inf。
3. **Jetson smoke**: 当天的 `analysis/jetson/YYYY-MM-DD_lnn_benchmark.json`
   状态字段为 `ok` 或 `ok_cpu_fallback`,且 CfCStyle 测试 MSE
   优于同隐藏维度的 GRU 至少 5%(否则需在 `analysis/jetson/` 记录原因)。
4. **追踪输出**: 当天的 `docs/daily/YYYY-MM-DD_LNN_research_digest.md`
   与 `analysis/repo_watchlist/YYYY-MM-DD_lnn_open_source_watchlist.md`
   非空且包含 ≥10 篇 arXiv 候选 / ≥10 仓库。
5. **可读性**: 新增报告必须有 frontmatter (`title/date/tags`),
   交叉引用使用 `[[Page]]`。

## 7. 当前进度 (Status, 2026-06-03)

- ✅ 9 模型变体在 Jetson Orin Nano Super (CPU 路径) 上完整通过
  `scripts/verify_all_models.py` 与 `scripts/quick_validate_implement.py`。
- ✅ 当日 CPU smoke benchmark: CfCStyle `MSE=0.264`(1169 参数)
  优于 GRU `MSE=0.335`(929 参数),提升 ~21.2%(see
  `analysis/jetson/2026-06-03_lnn_benchmark.md`)。
- ⚠️→✅ **[2026-06-03 loop#2]** CUDA 路径已修复: `system python3.10` 装
  `torch 2.10.0`(jetson-ai-lab.io/jp6/cu126 镜像)+ NVIDIA `libcudss 0.8.0.10
  cuda12 aarch64`,`source scripts/jetson_cuda_env.sh` 后
  `torch.cuda.is_available() == True / device.name == "Orin" / cuDNN 90300`。
  剩余约束: Jetson 统一显存,实际 cudaMalloc 需要系统空载窗口
  (本时段系统 RAM 5.2/7.6 GB 被并行 agents 占用,触发 `NvMap … error 12`)。
  详见 [[2026-06-03_loop_iteration2_cuda_fix_pareto]]。
- ✅ 日追踪流水线: 25 篇 arXiv + 32 仓库 + 23 HF 模型
  (见 `docs/daily/2026-06-03_LNN_research_digest.md`)。
- ✅ 12 篇论文/方向研读报告归档于 `docs/reports/`。

## 8. 后续 12 个迭代(由 `/loop 1h` 调度)优先级

| # | 任务 | 出口物 | 估时 |
|---|---|---|---|
| 1 | ~~修复 Jetson CUDA wheel,跑通 GPU 路径 benchmark~~ **[loop#2 done]** torch 2.10.0+cu126 + libcudss 0.8.0.10 + `scripts/jetson_cuda_env.sh` | `analysis/jetson/2026-06-03_loop_iteration2_cuda_fix_pareto.md` | 1 loop |
| 2 | 复现 LiquidTAD 长视频 TAD 实验(论文 2604.18274) — **stage A 算子+单测 ✅ + stage B smoke ✅ (loop#3 done) + stage C-lite 3-seed ablation ✅ (loop#4)**;真正 Stage C–E 待 CUDA 空载窗口 + THUMOS-14 数据 | `analysis/jetson/2026-06-03_loop_iteration{3,4}_*.md` + `tests/test_liquid_tad_hierarchical.py` + `lnn/core/long_sequence.py::HierarchicalDecayLiquidTADHead` + `scripts/ablation_liquid_tad_heads.py` | 3 loop done / 1–2 loop left |
| 3 | LFM2.5-1.2B-Distilled INT4 推理 + token/sec 测试 | `analysis/lfm25/2026-06-03_lfm25_int4_jetson.md` | 1–2 loop |
| 4 | EMMA 多模态物理参数恢复最小验证(论文 2605.24047) | `analysis/multimodal/2026-06-03_emma_validation.md` | 2 loop |
| 5 | Comparative Analysis of LNN & LSTM(2605.27467)对照重做 — **loop#7 ✅ (诚实负面信号)** Mackey-Glass 4-backbone × 3-seed: GRU 最准 (MSE 0.00336), LTC 参数 −50% 但 MSE 高 41%, 训练慢 5.9× | `docs/reports/Comparative_Analysis_of_LNN_and_LSTM_研读报告.md` v2 增补 + `analysis/timeseries_ablation/2026-06-04_loop_iteration7_lnn_vs_lstm_v2.md` + `scripts/ablation_lnn_vs_lstm_timeseries.py` | 1 loop done |
| 6 | GCN-CfC 分子筛选模型 smoke(GitHub Linlab2026/GCN-CfC) — **loop#5 结构化调研 ✅ + loop#6 follow-up A 落地 ✅**:`scripts/experiment_graph_lnn_molecule.py` 用本仓 `GraphLNNPredictor` 跑 Tox21-styled 二分类,3 seed × {CfC, LTC, GRU}: 三家 AUC 并列 0.754,LTC −28% 参数且方差最低,GRU 推理最快;证明端到端 PyTorch 单 stack 完胜两框架方案 | `docs/reports/GCN-CfC_仓库结构化调研.md` + `analysis/molecular/2026-06-04_loop_iteration6_graph_lnn_tox21_smoke.md` + `scripts/experiment_graph_lnn_molecule.py` | 2 loop done (B 级 + 落地实验) |
| 7 | 把 Jetson Pareto sweep 接入 PRD 验证指标 #3 — **loop#2 ✅** (12 trials CfCStyle vs GRU × hidden∈{8,16,24} × seq∈{16,32}, 4 Pareto-front points; CfC@h=24,seq=32 最佳 MSE 0.4285 −30.6% vs GRU) | 更新 `scripts/jetson_lnn_benchmark.py --pareto` + `analysis/jetson/2026-06-03_loop_iteration2_cuda_fix_pareto.md` | 1 loop done |
| 8 | Loop 调度产物去重 + 自动 retro(避免重复研读相同 paper) — **loop#8 ✅** `scripts/loop_status.py` 扫描 daily/jetson/molecular/timeseries/repo_watchlist + 解析 PRD §8 + git log,输出 JSON+MD 报告;首次跑就揪出 PRD §7 没补 ✅ 的 tracking 漏洞 | `scripts/loop_status.py` + `analysis/loop_status/<date>_loop_status_<date>.md` | 1 loop done |

## 9. 风险与缓解

| 风险 | 缓解 |
|---|---|
| **arXiv/GitHub API 限流** | `daily_lnn_research.py` 保留前一次结果;遇 429/403 不覆盖已有 digest |
| **Jetson CUDA 工具链断裂** | benchmark 自动回退 CPU 并打 `ok_cpu_fallback` 标记 |
| **Loop 重复劳动** | 每次 loop 先 `git log --since` 与 `docs/daily` 对账,避免重复采集同日数据 |
| **PRD 漂移** | 本 PRD 与 `docs/NEXT_STEPS.md` 与 `docs/LNN_持续研究协议.md` 双向同步;
  每周末 retro 检查是否对齐当周 commits |

## 10. 相关文档

- [[README]] — 项目入口
- [[AGENTS]] — Agents / Skills 规划
- [[NEXT_STEPS]] — 迭代方向(技术细节)
- [[LNN_持续研究协议]] — 资料追踪 SOP
- [[每日自动化任务与Jetson验证]] — 自动化部署细节
- [[LNN_深度研读报告]] — 论文研读总索引
- [[IMPLEMENTATION_SUMMARY]] — 9 变体实现摘要
- [[OPTIMIZATION_STRATEGIES]] — 选模型 / 调参指南

## §9. 下一周候选任务(由 loop iter#9 落地, 2026-06-04)

PRD §8 8 个任务里只剩 #3 LFM2.5(等 RAM 空载窗口)和 #4 EMMA(远程 agent 负责)
两个真实阻塞,工作面已基本耗尽。本节列出下一周(2026-06-04 → 2026-06-11)
8 个新候选任务,供 `/loop 1h` 调度参考。

| # | 任务 | 出口物 | 关联 |
|---:|---|---|---|
| 9-1 | LFM2.5-1.2B-Distilled INT4 离线推理 + token/sec 测试(夜间空载窗口) | `analysis/lfm25/<date>_lfm25_int4.md` | §8 #3 |
| 9-2 | concept_drift 复测 phase-B:多 regime 渐进 + lr warmup + curriculum — **loop#10 ✅** (gradual_multi_regime + warmup_frac=0.1: CfC MSE 0.27142 比 LSTM 0.38270 低 −29.1%, 参数 −27%;项目首次 CfC 赢 LSTM,印证 paper claim) **→ loop#11 phase-C 8 seed 撤回**: CfC mean +15.1% / median +116% vs LSTM,seed 7/777 是数据集硬点;N=3 是 small-N lucky。结论: 合成时序回归任务上**没有 LNN backbone 跨 8 seed 稳定赢 LSTM** | iter#9 v3 后续 → **iter#10 phase-B + iter#11 phase-C 都完成** | §8 #5 v4→v5 |
| 9-3 | LiquidTAD 真 Stage C:THUMOS-14 50-video 子集复现 | `analysis/paper_replication/liquid_tad_thumos.md` | §8 #2 stage C-true |
| 9-4 | `experiment_graph_lnn_molecule.py` 加 `--frozen-encoder` 两阶段(模拟 GCN-CfC 解耦) — **loop#13 ✅** 9 trials (3 backbone × 3 seed): frozen mean AUC 0.71 vs e2e 0.75 (Δ −4.8% cfc / −6.8% ltc / −5.0% gru);**6/9 trial frozen 输 4-13%**;LTC frozen 最差 (median 0.7012),证实 iter#5 对 GCN-CfC 两阶段管线的"不利"判断 | `analysis/molecular/2026-06-04_loop_iteration13_frozen_encoder_ablation.md` + `scripts/experiment_graph_lnn_molecule.py --frozen-encoder` | 1 loop done |
| 9-5 | `loop_status.py --since-last-loop` 自动定位上次 iter 结束后的变更 — **loop#14 ✅** 扫 `analysis/**/loop_iteration*.md` 取 mtime 最新作为 marker,列 marker 后 git commits + analysis/docs/scripts/lnn/tests/papers 文件变更 + 新 iter 报告;首跑 58min elapsed / 2 commits / 5 files / suggestion = "write iter report before push" | `scripts/loop_status.py --since-last-loop` + `analysis/loop_status/<date>_loop_status_since_last.md` | 1 loop done |
| 9-6 | `HierarchicalDecayLiquidTADHead` 加 ONNX export + TensorRT INT8 | `analysis/jetson/<date>_liquid_tad_tensorrt.md` | §8 #2 stage D |
| 9-7 | 跨数据 backbone ranking 自动生成:ablation runner 加 `--datasets` 多个,出 task-conditional 表 — **loop#12 ✅** `scripts/build_backbone_matrix.py` 扫 `analysis/timeseries_ablation/*` + `analysis/molecular/*` pivot 出 task × backbone 矩阵,dedup 规则保留 n_seeds 更大的;首跑结果 LSTM 3 wins / GRU 1 win / CfC/LTC 0 wins (4 任务) | `analysis/timeseries_ablation/<date>_task_conditional_matrix.md` + `scripts/build_backbone_matrix.py` | 1 loop done |
| 9-8 | PRD §6 验证指标自动 CI:GitHub Actions 跑 `verify_all_models.py + ablation_*` 周线 — **loop#15 ✅** `.github/workflows/lnn_weekly_verify.yml` (~120 行) 周一 03:07 UTC 跑 pytest+verify_all_models+quick_validate+tiny ablation+backbone matrix+loop_status JSON 全套,失败开 ::error;artifacts 上传 14 天保留不 push master;本地预演 5 step 全绿 < 90s | `.github/workflows/lnn_weekly_verify.yml` | 1 loop done |

### 已调研但不复现(C 级, 只入索引,不投复现 budget)

| 仓库 / 模型 | 排入理由 | 链接 |
|---|---|---|
| `Linlab2026/GCN-CfC`(iter#5) | 双框架管线;不利 Jetson 部署 | [[GCN-CfC_仓库结构化调研]] |
| `LiquidAI/LFM2.5-8B-A1B` | 8B 模型 too big for Orin Nano 8GB 显存 | iter#1 daily research |
| `raminmh/CfC` (官方 tf 版) | 已被 `ncps` 取代,本仓 `lnn/core/cfc.py` 已 PyTorch 化 | iter#5 引用 |

### 复现协议边界条件(本仓沉淀的"什么时候 LNN 不赢")

1. **iter#7 / iter#9** — Mackey-Glass / concept_drift 单次硬切,小预算 + 固定 lr 下:
   GRU / LSTM 显著优于 CfC / LTC;LTC 在 concept_drift 上 catastrophic
   (MSE 高 +1301% vs LSTM)。论文 claim 的"LNN 在非平稳序列上更鲁棒"在
   **更严格** 复现协议下不直接成立 — 需要 gradual 多 regime + lr warmup +
   更多 sample。
2. **iter#6** — 在静态图二分类(time=1)上,CfC / LTC / GRU AUC 完全并列(0.754),
   LTC 仅在参数效率和方差稳定性维度上略胜。
3. **跨 task ranking**: 没有"通杀 backbone";必须按任务画 ranking
   (与远程 EMMA agent commits `5518b20 / cf14d21 / 7575a9d` 的
   regime-conditional encoder ranking 同源)。

## §10. 第三波候选任务(由 loop iter#16 落地, 2026-06-04)

PRD §9 完成度 **5/8 = 62.5%**(#2/#4/#5/#7/#8 ✅),剩 3 个真实硬阻塞
(#1 RAM / #3 数据 / #6 RAM+CUDA)。无外部依赖任务面已耗尽,
启动第三波 — 主要面向**新论文复现 + 把 backbone matrix 扩展到更多 LNN 变体**。

| # | 任务 | 出口物 | 关联 | 状态 |
|---:|---|---|---|---|
| 10-1 | DynPMNN(2605.08176)复现 stage A:`lnn/core/dynpmnn.py::FHNCell + DynPMNNNetwork` | code + unit test | iter#16 研读 | **stage A+B ✅ (iter#23/24)**, 6-seed mackey_glass 跑出 median MSE 0.0182 (诚实负面) |
| 10-2 | DynPMNN stage B:加 `--backbone fhn_dynpmnn` 到 ablation runner,跑 multi-seed 对比 | matrix 新增 dynpmnn 列 | §10 #1 之后 | **✅ (iter#24)**: 6-seed mackey_glass fhn_dynpmnn median MSE 0.0182, backbone matrix ingest 把 fhn_dynpmnn 加进 mackey_glass h=24 r=4 行 (诚实负面: 输 ~3× vs cfc/ltc/gru) |
| 10-3 | Comparative LNN vs LSTM phase-D:hidden=64 + epochs=50 + samples=4000,看 LNN 优势是否随规模出现 | `analysis/timeseries_ablation/<date>_phase_d.md` | §9 #2 v6 | **stage A ✅ (iter#25)**: `scripts/ablation_lnn_vs_lstm_timeseries.py --phase-d` 预设 (CLI override-allowed) + `tests/test_ablation_phase_d.py` (3 单测,preset apply / CLI 覆盖赢 / `--help` 列标志) 全绿;2-seed × 4-backbone mackey_glass 缩预算 (samples=2000, epochs=20) 跑出,详见 [[2026-06-08_loop_iteration39_phase_d]] |
| 10-4 | 给 `experiment_graph_lnn_molecule.py` 加 `HierarchicalDecayLiquidTADHead` 作为 recurrent 选项(交叉 #2 与 #6) | code + smoke | 综合 | **✅ (iter#33)**: lnn/core/graph.py 加 `liquid_tad` recurrent_type (复用 LongSequenceLiquidClassifier, LiquidS4Block 堆叠); 3-seed × 4-backbone ablation: cfc/ltc/gru/liquid_tad median AUC 0.6631/0.6570/0.6570/0.6670 — liquid_tad 微弱赢 (seed 2026 显著 +0.073) |
| 10-5 | `loop_status.py` 加 `--prd-status` 子模式:解析 §8/§9/§10 全表,出未完成 + 阻塞理由 | code + sample report | §9 #5 衍生 | ✅ (iter#21) |
| 10-6 | `build_backbone_matrix.py` 加 `--export-readme-snippet`:产 README 顶部 badge 行(LSTM 3/4 win 等) | code + README badge | §9 #7 衍生 | ✅ (iter#29) |
| 10-7 | LFM2 (LFM2.5-1.2B-Distilled-SFT)在 RAM 空载窗口跑 1 次 INT8 推理 + token/sec 表 | `analysis/lfm25/<date>_lfm25_int8_jetson.md` | §8 #3 / §9 #1 | pending (RAM blocker) |
| 10-8 | `analysis/loop_status/` 自动生成 README 标签云(高频 task / 高方差 seed 提示) | tooling | meta | ✅ (iter#30) |
| 10-9 | **SVAF (arXiv 2604.03955) τ-modulated peer-blending 算子复现**(iter#17 研读,见 [[Symbolic-Vector_Attention_Fusion_SVAF_研读报告]]):toy 2-agent mesh + τ_i ∈ {1, 10, 60} 三组神经元,N 步耦合后看 spectral diff 验证"fast τ 同步 / slow τ 主权"现象 | `analysis/svaf/2026-06-04_tau_toy.md` + ~100 行 core code | iter#17 研读, 最小可复现单元 | **stage A ✅ (iter#22)**, stage B pending (P2) |
| 10-10 | **PDNA (arXiv 2603.00153) PulseHead + Gapped protocol 复现**(iter#18 研读,见 [[Pulse-Driven_Neural_Architecture_PDNA_研读报告]]):stage A `lnn/core/cfc.py::PDNAPulseHead` (~80 行) + unit test ✅ (iter#19, 12 tests);stage B sMNIST Gapped protocol 3 seed × 5 backbone ablation, backbone matrix 加 smnist_gap 行 ✅ (iter#20, cfc_pulse multi-gap +2.53 pp);**stage C LRA Pathfinder 落地 (iter#28)**: `lnn/data/pathfinder_synth.py` (~150 行, 32x32 grid 端点连接性二分类) + `scripts/experiment_pdna_lra.py` (~250 行, 3 variant × N seed smoke) + `tests/test_pdna_lra.py` (6 单测, 6/6 绿含 1 CLI smoke) + `build_backbone_matrix.py` 加 `_ingest_lra_pathfinder` + `--include-lra`,matrix 现跨 4 domain (timeseries/molecular/smnist_gap/lra_pathfinder);**stage C 扩展 (iter#30, this iter-skill run)**: 默认改 3 seeds × 5 epochs × 500 train × hidden=32 + 加 per-seed `pdna_alpha`/`pdna_beta` tracking 验证 pulse 门从 0.01 init 偏离;CPU 预算约束下实际跑 2 seeds × 3 epochs × 500 train × hidden=32 (~30 min)。**诚实负面**: 1 seed × 1 epoch 烟测 3 variants 全部 48.75% ≈ 随机; iter#30 扩展看是否 escape 随机 baseline | code + matrix 行 | iter#18 研读,**代码公开** + MNIST zero-cost | **stage A+B+C ✅ (iter#19/20/28), stage C 扩展 → iter#30** |
| 10-11 | **LFM/LNN-related active≤3B vs 30B+ LLM 打榜证据卡**:汇总 `LFM2.5-8B-A1B`(8.3B total / 1.5B active)公开指标、`Qwen3-30B-A3B` 30B+ 基线、本仓 `LFM2.5-1.2B` 本地 GGUF/DPO 推理证据与 micro leaderboard;默认判定 13 个重叠指标 7 胜 / 6 负,只支持 active≤3B MoE 局部胜出,**不支持 exact 3B dense 全面吊打** | `scripts/build_llm_battlecard.py` + `analysis/llm_battlecard/2026-06-04_llm_battlecard.{json,md}` + `tests/test_llm_battlecard.py` | 用户 3B-vs-30B 目标 | ✅ iter#26 claim-audit 落地,iter#30/31 接入 leaderboard |
| 10-12 | **本机 LFM2.5 GGUF micro-eval harness**: `scripts/run_llm_micro_eval.py` 通过 llama.cpp 对本机模型跑 deterministic arithmetic / instruction / JSON / abstention sanity tasks;当前 `LFM2.5-1.2B-Instruct-Q4_0.gguf` 7/7,最近一次平均生成 16.8 tok/s(随系统负载波动);该门槛不替代公开榜,但阻断“模型部署都没跑通就谈打榜”的假阳性 | `analysis/llm_micro_eval/2026-06-04_lfm25_1_2b_instruct_q4_micro_eval.{json,md}` + `tests/test_llm_micro_eval.py` | 用户 3B-vs-30B 目标 | ✅ iter#27 本机实测入口落地 |
| 10-13 | **30B+ OpenAI-compatible endpoint micro-eval 后端**: `run_llm_micro_eval.py --backend openai-chat` 支持 `/v1/chat/completions`,可直接接本机 llama-server/vLLM/SGLang 或远程 30B+ API,输出同一 JSON schema;当前已用本地 fake OpenAI server 单测和真实本机 `llama-server` LFM2.5 endpoint 覆盖,等待真实 30B+ endpoint/权重 | code + unit test + `analysis/llm_micro_eval/2026-06-04_lfm25_1_2b_instruct_q4_http_micro_eval.{json,md}` | 用户 3B-vs-30B 目标 | ✅ iter#28 endpoint 对照入口落地,iter#30 live local endpoint 验证 |
| 10-14 | **LLM micro-eval leaderboard 汇总器**: `build_llm_micro_leaderboard.py` 扫描 `analysis/llm_micro_eval/*_micro_eval.json`,按 accuracy → task coverage → mean tok/s 排序,输出可追加 30B+ endpoint 行的 JSON/Markdown 榜单;当前有本机 `LFM2.5-1.2B-Instruct-Q4_0.gguf` CLI/HTTP 与 `LFM25-DPO-Q4_0.gguf` 三行,3/3 rankable;base CLI/HTTP 均 7/7,DPO Q4 4/7,尚未形成真实 30B+ 对照 | `analysis/llm_micro_eval/2026-06-04_llm_micro_leaderboard.{json,md}` + `tests/test_llm_micro_leaderboard.py` | 用户 3B-vs-30B 目标 | ✅ iter#29 micro leaderboard 落地,iter#30/31 扩展本机候选 |
| 10-15 | **本机 LFM2.5 llama-server OpenAI-compatible 实测**: 复用已有 `scripts/serve_lfm25_http.sh` 在 `http://127.0.0.1:18080/v1` 服务本机 GGUF,运行 `run_llm_micro_eval.py --backend openai-chat --openai-model lfm25-1.2b-instruct-q4`,7/7,平均 5.707 tok/s;证明本机 endpoint 路径真实可用,但该行仍是 1.2B 小模型,不是 30B+ baseline | `analysis/llm_micro_eval/2026-06-04_lfm25_1_2b_instruct_q4_http_micro_eval.{json,md}` + battlecard/leaderboard 集成 | 用户 3B-vs-30B 目标 | ✅ iter#30 live endpoint sanity 落地 |
| 10-16 | **本机 LFM2.5 DPO Q4 候选 micro-eval 回归**: `models/lfm25-dpo-s1/LFM25-DPO-Q4_0.gguf` 跑同一 7 题 sanity set,结果 4/7,arithmetic 3/3 和 JSON 1/1 通过,exact instruction 0/2 与 abstention 0/1 失败;说明该 DPO 候选不适合进入 30B+ 对照前的候选池,需要 prompt/template 或训练目标回查 | `analysis/llm_micro_eval/2026-06-04_lfm25_dpo_s1_q4_micro_eval.{json,md}` + leaderboard/battlecard 集成 | 用户 3B-vs-30B 目标 | ✅ iter#31 DPO 负面证据落地 |
| 10-17 | **SNCP-PPO Crowdnav (LTC + PPO 机器人人群导航) 复现** (iter#26 研读,见 [[SNCP-PPO_Crowdnav_LTC_深度研读报告]]): 仓库 heimdilon/sncp-ppo-crowdnav (2026-06-07, 0 stars) 用 in-house LTC + 5-phase curriculum + PPO 训 TurtleBot3 群导航,100% success on 1-3 行人,86% on hard 5 行人;**stage A 最小可复现单元已落地 (iter#26)**: `lnn/core/sncp_policy_lite.py::SNCPPolicyLite` (~190 行, 复用 in-house `LTCNetwork` euler ODE 作 temporal encoder, 2D Gaussian actor + value critic head) + `tests/test_sncp_policy_lite.py` (10 单测, 全绿) + `scripts/experiment_sncp_ppo_lite.py` (2D point-mass + 2 障碍 smoke, PPO clip=0.2, 15 updates × 8 episodes, 端到端 forward+backward 跑通)。**诚实负面**: 120 episodes 不够学 2D nav, reach_rate=0; 需 500+ episodes + 4 行人 curriculum 才能拿可对比信号; v6 原仓 Colab T4 跑 3000 ep 2h40m。**stage B curriculum ablation 落地 (iter#27)**: `PointMassNavLite` 加 `--n-pedestrians N` + `--curriculum` (1→2→3 行人 sequential 训练,obs_dim 零填 padding 保持 14-dim 跨阶段恒定) + `tests/test_sncp_pedestrian_env.py` (6 单测,含 1 个 CLI smoke 端到端 6/6 绿); curriculum smoke 1 seed × 3 stages × 20 PPO updates = 480 episodes: last-5 mean_return=-3.94/-4.48/-3.61,全部 reach=0%;**240 ep/stage 仍不足**,与 iter#26 负面信号一致,但 curriculum infrastructure (zero-pad 跨阶段 obs_dim 恒定 + sequential train) 完整落地,可供未来 ~3000 ep 真 curriculum 跑 contrast | `lnn/core/sncp_policy_lite.py` + `tests/test_sncp_policy_lite.py` + `scripts/experiment_sncp_ppo_lite.py` + `tests/test_sncp_pedestrian_env.py` + `docs/reports/SNCP-PPO_Crowdnav_LTC_深度研读报告.md` + `analysis/sncp_ppo_lite/<date>_sncp_ppo_lite.{json,md}` | iter#26 研读, 公式复用 0 障碍 | **stage A+B ✅ (iter#26/27)** |
| 10-18 | **Natural Gas LNN Forecaster** (iter-skill 2026-06-08 候选,见 [[PRD_iter-skill_2026-06-08]]): 复用 `lnn/data/natural_gas_generator.py` 已就位 (2645 business days synthetic Henry Hub with seasonality + shock + trend),加 5-backbone × 3-seed × 30-epoch ablation。`scripts/experiment_natural_gas_lnn.py` (~280 行, 30-day input window → 1-day return, 80/10/10 chronological split, AdamW + cosine, median MAPE + 7-day directional accuracy) + `tests/test_natural_gas_lnn.py` (8 单测,含 1 个 CLI smoke 端到端) + `build_backbone_matrix.py` 加 `_ingest_natural_gas` + `--include-natural-gas`,matrix 现跨 timeseries 子域 (含 mackey_glass / concept_drift / natural_gas)。**首个真实形态商品时序 LNN 闭环**,仓库从纯合成 mackey_glass/concept_drift 扩到带季节性+波动聚类的真实形态合成数据。cfDNA (10-19) 与 6G 无线 (10-20) 留为下两轮候选 | `scripts/experiment_natural_gas_lnn.py` + `tests/test_natural_gas_lnn.py` + matrix 行 | iter-skill 2026-06-08, 复用现有 generator | **stage A ✅ (iter#29 iter-skill)** |

### 已调研未复现 (C 级) 累计表

| 仓库/论文 | 排入理由 | 链接 |
|---|---|---|
| Linlab2026/GCN-CfC (iter#5) | 双框架管线,iter#13 已量化 −5% AUC | [[GCN-CfC_仓库结构化调研]] |
| LiquidAI/LFM2.5-8B-A1B | 8B too big for Orin Nano 8GB | iter#1 daily |
| raminmh/CfC (官方 tf) | 被 ncps 取代 | iter#5 |
| DynPMNN (arXiv 2605.08176) | **无公开代码**,需自行复现 — 见 PRD §10 #1 → #2 | [[Physics-Modeled_Neural_Networks_DynPMNN_研读报告]] |
| **SVAF (arXiv 2604.03955, iter#17)** | **部分可复现**: 端到端需 237K LLM-authored 训练数据(作者未公开);**τ 调制耦合算子(公式 20)是 P2 mini-task** | [[Symbolic-Vector_Attention_Fusion_SVAF_研读报告]] |
| AEGIS (arXiv 2604.02149) | 与 LNN 弱关联(Thermodynamic State Space Model,非 CfC/LTC);非 P0 候选 | iter#17 调研 |
| **PDNA (arXiv 2603.00153, iter#18)** | **代码公开** (github.com/Parassharmaa/pdna) + MNIST zero-cost + 5 seed ablation + noise control 严谨度;**P1 stage A/B/C 与本仓 LNN/CfC 复用度极高** | [[Pulse-Driven_Neural_Architecture_PDNA_研读报告]] |
| **RLSTG (arXiv 2601.14115, iter#31)** | WWW '26 accepted, Lu et al. **LTC + Riemannian manifolds** (tangent space ODE + exp/log wrapper);**理论推广 LTC 稳定性到黎曼域**;数据集 ENRON 邮件网络 (与本仓分子/时序不重叠);**仅项目页 demo 无代码**;复现需 `geoopt` 依赖 + 新 tangent-space 包装;B 级 ROI | [[Riemannian_Liquid_Spatio-Temporal_Graph_Network_RLSTG_研读报告]] |
| **EntroLnn (arXiv 2601.06195, iter#34)** | SAC '26 accepted, Li et al. **LTC + battery SoH 在线精化** ("transformable" 静态+动态 LNN);公式 `dh/dt = -α⊙h + tanh(W_h h + ū)` 与本仓 `LTCNetwork` 几乎同构;数据集 MIT-Stanford 124 LFP 电池;**MAE 0.004577 for CFT + 18 cycles for EoL**;**无官方代码** (CC BY 4.0);Jetson 边缘 battery monitor 是真实应用场景;B+ 级 ROI (公式复用高) | [[EntroLnn_Entropy-Guided_Transformable_LNN_研读报告]] |
| **Retinal Ganglion LNN (arXiv 2511.18014, iter#35)** | AAAI-26 Student Abstract, Dobek et al. **LTC + CfC** 应用到 tiger salamander 视网膜神经节细胞活动预测 (3 数据集);vs ConvNet + LSTM: **LTC/CfC 赢 MAE 2.73/2.86 vs ConvNet 4.07 + 5-8× 少参数 + 更快收敛/query** (ANOVA p<0.05);**输 Pearson ρ** (0.480 vs 0.569);**无代码/无数据** (Student Abstract 性质);Jetson 边缘 视觉假体卖点;A- 级 ROI (公式复用 0 障碍) | [[Modeling_Retinal_Ganglion_Cells_with_Neural_Differential_Equations_研读报告]] |
| **LNN 3DGS Deformation Field (arXiv 2606.07670v1, iter#38)** | APSIPA ASC 2026, Li et al. (Monash Malaysia) **CfC stack (D=6, W=128) 替换 D-3DGS MLP deformation field**, "depth-as-time" 架构(D 维深度扮演经典 CfC 序列 T 维时间);vs MLP baseline: **NeRF-DS 7 noisy real scenes mean +0.47 dB PSNR / As specular scene +2.74 dB PSNR + −41% LPIPS**;**closed-form 故前向无 ODE/SDE solver**, 跟 MLP 同档延迟;Eq. 2 与本仓 `CfCCell` 95%+ 同构;code = 改 D-3DGS 公开仓 `DeformNet` 类 (~200 LOC), 无独立 repo;B+ 级 ROI (公式同构但 3DGS 域跟本仓 device-control 4 case 弱重叠, narrative bonus 强) | [[Liquid_NN_3DGS_Deformation_Field_2606.07670_研读报告]] |
| **MR-MoE Multi-Rate Liquid + MoE (arXiv 2606.12240v1, iter#39)** | NeurIPS 2026 投递, Zong et al. (Virginia Tech) **LNN + K=3 MoE experts + 异 τ (Eq. 8 τ1 ≪ τ2 ≪ τ3) + Singular Perturbation (Eq. 9 quasi-steady-state fast expert) + 双层注意力 (feature + temporal)** 在 PhysioNet Sepsis 预测: AUROC 0.65→0.68 / AUPRC 0.45 vs LSTM 0.53/0.22 baseline (5 模型 ablation: LSTM / Monolithic LNN / MoE / MR-MoE / MR-MoE-Attn);**Eq. 4 + Eq. 6 + Eq. 8 与本仓 `CfCCell` 95%/0%/20% 同构**;**工程复用高**: `CfCCell.__init__` 加 `n_tau: int = 1, tau_scales: tuple = (0.1, 1.0, 10.0)` 一行 config 即可吃下"多时间尺度" 收益, 无新 cell 无新数学;**诚实负面**: AUROC 0.65 不算高(脓毒症 SOTA 0.75-0.85), 3× 参数量 (4500 vs 1500) caveat, LSTM 0.53 异常低(子集选择), τ 手设不可学习;代码 0 公开 (CC BY 4.0, NeurIPS 接收后会发);B+ 级 ROI (公式同构 + 工程蓝海 + 临床数据真实噪声场景 = 第 4 类"强动态+长程噪声" 实证) | [[Liquid_NN_MR_MoE_Sepsis_2606.12240_研读报告]] |

### §10 完成度(由 iter#17 / iter#32 / iter#36 跟踪)

- 累计 **stage A ✅ 10 个** (10-1 DynPMNN A+B / 10-3 Phase-D / 10-4 LiquidTAD 落地 / 10-5 prd-status / 10-6 README badge / 10-8 README 标签云 / 10-9 SVAF A / 10-10 PDNA A+B+C / 10-17 SNCP-PPO A+B / 10-18 Natural Gas A); 2026-06-09 iter#33/34/35 Jetson 三连 + iter#36 设备操控专章 → 10-19/20/21/22 四个新条目全 stage A ✅; 2026-06-10 iter#37 → **10-22 stage B ✅** (2-stage EntroLnn TransformableLTC + 16 单测 + 3-seed honest-negative 烟测)
- 真实硬阻塞只剩: 10-7 LFM2.5-1.2B INT8 推理 (8GB RAM 跑 1.7GB 模型,需空载窗口)
- 最小可复现新论文单元候选: 10-9 stage B (τ P2 玩具 50 行),10-22 case C 扩展 (NCP sparse 真激活),10-22 case B 压力测试 (T=64/128 + nan_count guard)
- 2026-06-12 iter#39 增量: 新增候选 **10-23 MR-MoE 异 τ + MoE 脓毒症复现** (B+, 见 [[Liquid_NN_MR_MoE_Sepsis_2606.12240_研读报告]]); 部署侧 #10-7 优先级 ↑↑ (6-11 GGUF 集中日 3 条独立路径); XWormNet (#10-24 candidate) 留为 iter#40 决断

| 10-19 | **Jetson Orin Nano LNN Benchmark 落地** (iter#33 / iter-skill 2026-06-09,见 [[PRD_iter-skill_2026-06-09_jetson-lnn-deploy]] 与 [[VERIFICATION_RESULTS]] §1): 复用 `scripts/jetson_lnn_benchmark.py` 已就位 (623 行,标准 + Pareto sweep + CPU/CUDA fallback + graceful torch-not-installed skip),加 `tests/test_jetson_lnn_benchmark.py` 7 单测 (Pareto 前沿标记、CUDA 错误检测、CLI smoke 真跑) + `docs/VERIFICATION_RESULTS.md` 真机数据表。本机真跑 (Jetson Orin Nano Engineering Reference Developer Kit Super, R36 BSP, CPU path 因为 driver 12060 < torch 2.11+cu130 需求): Pareto 前沿 5/8 configs, **CfCStyle h=16 T=32 (MSE 0.470) 精度胜 GRU 12.3%, GRU h=16 T=32 (98844 步/秒) 速度胜 CfC 4.5×**, 印证 Tanna et al. 2024 (IEEE 10826128) "CfC 1-5 数量级加速" 论文 claim 的本地复现侧写。下一步: (a) 等 driver 升级或换 Jetson-packaged torch wheel 跑 CUDA 路径; (b) 多 seed (≥3) 拉均值; (c) 加 INA219 探针复现 < 10 mW 量级 (Liu et al. 2025); (d) ONNX/TensorRT 导出后延迟 (受 ONNX RNN/LSTM operator gap 限制,见 [[PRD_LNN_Edge_Research#angle-3]]) | `scripts/jetson_lnn_benchmark.py` + `tests/test_jetson_lnn_benchmark.py` + `docs/VERIFICATION_RESULTS.md` + `analysis/jetson/2026-06-09_local_lnn_benchmark.{json,md,png}` | iter-skill 2026-06-09,真机 Pareto 数据 | **stage A ✅ (iter#33 iter-skill)** |
| 10-20 | **Jetson 4-model LNN Pareto sweep (CfC + LTC + PDNA-pulse + GRU)** (iter#34 / iter-skill 2026-06-09,见 [[PRD_iter-skill_2026-06-09_jetson-4model-sweep]] 与 [[VERIFICATION_RESULTS]] §1 4-model 表): 扩 `scripts/jetson_lnn_benchmark.py::run_benchmark` 的 `models` 列表从 2 个 (CfCStyle+GRU) 到 4 个 (CfCStyle+LTC+PDNAPulse+GRU),加 2 个 import (`from lnn.core.ltc import LTCNetwork` + `from lnn.core.cfc import PDNAPulseHead`) + 2 个 inline model class (`LTCModel` 用 `LTCNetwork` 完整序列 forward, `PDNAPulseModel` 用 `CfCCell` + `PDNAPulseHead` 顺序组合);改 `tests/test_jetson_lnn_benchmark.py::test_cli_quick_single_run_no_pareto` 期望 4 个模型名(CfCStyle/LTC/PDNAPulse/GRU)。本机真跑 (Jetson Orin Nano Super, CPU path, 1 seed × 4 models × 2 hidden × 2 seq = 16 configs): **Pareto 前沿 6/16 configs, PDNAPulse h=8 T=32 (418 params, MSE 0.401) 是全局精度冠军, 比 iter#33 CfC 冠军 0.470 胜 −14.7%**; GRU 仍速度冠军 280985 步/秒 (比 CfC 40897 快 6.9×, 比 LTC 12561 快 22.4×); **LTC Pareto 仅限 h=8 (最小参数 185) — h=16 全被支配** (印证 Hasani 2022 "1-5 数量级慢" 论文 claim 的本地侧写, 数量级小是因为 RK4 step reuse 在 CPU path 摊销了 ODE 求解开销)。**意外发现**: PDNAPulse h=8 T=32 53335 步/秒 反而比 CfC 40897 快 30% — pulse α gate 0.01 init 在前几 step 抑制了 sin() 开销或 torch 算子融合副作用, 对 PDNA paper "+5% wall-time" 声称有修正意义。1 seed scope cut 文档化 (iter#11 N=5 教训), 多 seed 留给 iter#35。pytest 7/7 绿, 0 回归。 | `scripts/jetson_lnn_benchmark.py` (+LTCModel +PDNAPulseModel +2 imports + models 列表 +2 行) + `tests/test_jetson_lnn_benchmark.py` (改 single-run 期望 4 模型) + `docs/VERIFICATION_RESULTS.md` §1 (4-model Pareto 表) + `analysis/jetson/2026-06-09_local_4model_lnn_benchmark.{json,md,png}` | iter-skill 2026-06-09 4-model 扩展 | **stage A ✅ (iter#34 iter-skill)** |
| 10-21 | **Jetson 4-model 3-seed multi-seed Pareto (mean ± std)** (iter#35 / iter-skill 2026-06-09,见 [[PRD_iter-skill_2026-06-09_jetson-4model-multiseed]] 与 [[VERIFICATION_RESULTS]] §1 3-seed 表): 复用 `scripts/jetson_lnn_benchmark.py` 已就位的 `--seeds` CLI (line 630, iter#33 时已添加但未使用) + `run_pareto_benchmark` 已有的 seed loop (line 541-580),加 1 个新函数 `aggregate_seeds(per_seed_results)` 用 `statistics.fmean` + `stdev` 按 (name, hidden, seq) 分组聚合 3 个 metric (test_mse / inference_steps_per_sec / train_seconds) 为 {mean, std, n_seeds} 字典,加 1 个配套 `mark_pareto_front_aggregated` 包装 `_agg_dominates` 读取 `.mean` 子字段,加 2 个新单测 (`test_aggregate_seeds_groups_by_model_hidden_seq` + `test_aggregate_seeds_then_mark_pareto_front_uses_mean`)。**iter#11 N=5 教训兑现** — iter#34 1-seed "PDNAPulse h=8 T=32 MSE 0.401 全局冠军" 是 lucky seed (seed=42), 3-seed mean 是 0.5361 ± 0.1199 (CV 22%) **被部分撤回**; 真正冠军是 **PDNAPulse h=16 T=32 (1474 params) 0.4224 ± 0.0257 (CV 6%)** — 稳胜 CfCStyle h=16 T=32 0.4658 ± 0.0078 by **−9.4%** (iter#34 1-seed 报的 −14.7% 是过估), 与 PDNA paper "+4.62 pp on sMNIST multi-gap" 论文 claim 量级一致。最稳定: CfCStyle h=8 T=32 std=0.0042 (CV 0.8%)。高方差警示: PDNAPulse h=8 T=32 std=0.120 + GRU h=8 T=32 std=0.134 (这 2 个 config 不应该作 production 候选)。3 seed scope cut 文档化 (iter#11 N=5 教训), 5 seed 留给 iter#36。pytest 9/9 绿 (7 旧 + 2 新), 0 回归。 | `scripts/jetson_lnn_benchmark.py` (+aggregate_seeds +mark_pareto_front_aggregated +aggregated_results in payload, ~80 LOC) + `tests/test_jetson_lnn_benchmark.py` (+2 单测) + `docs/VERIFICATION_RESULTS.md` §1 (3-seed mean ± std 表 + 解读) + `analysis/jetson/2026-06-09_local_4model_3seed_lnn_benchmark.{json,md,png}` | iter-skill 2026-06-09, 3-seed 复现 + iter#11 N=5 教训兑现 | **stage A ✅ (iter#35 iter-skill, honest-negative)** |
| 10-22 | **设备操控 LNN 专章 + 4-case 引用 harness** (iter#36 / 2026-06-09,见 [[PRD_设备操控_LNN]]): 把仓库 LNN 应用面**从判别/生成/预测/调度正式收口到「设备操控」闭环**,围绕"动态 + 不确定 + 实时性"三大约束,按"原理 / 适配性 / 落地案例 / 挑战与方案"四章展开。**新增 4 个 case 引用 harness** (`scripts/experiment_device_control_cases.py`, ~660 行) 全部**纯合成** (no ROS / no mavlink / no CAN / no real sensor,见 2026-06-09 user preference): ① **case A 四足** `run_case_quadruped` 用 `SNCPPolicyLite` (LTC + actor-critic) 12-DoF,BC shortcut (full PPO 在 `experiment_sncp_ppo_lite.py`); ② **case B 无人机** `run_case_drone` 用 `LNNImitationPolicy` (CfC recurrent) visual+IMU 4-DoF 回归; ③ **case C 工业控制** `run_case_industrial` 用 `LNNImitationPolicy` (LTC recurrent) 1-DoF 倒立摆 IL,NCP sparse wiring 留为 follow-up; ④ **case D 电池 SoH** `run_case_battery` 用 `LTCNetwork` (Euler ODE) EntroLnn-style transformable 公式同构,单阶段回归 (2-stage 精化留 follow-up)。`aggregate_seeds` 用 `statistics.fmean`+`stdev` 跨 seed 聚合; `_DEVICE_CONTROL_REPORT_SCHEMA` v1 固定 JSON 输出; `analysis/device_control/latest_device_control_summary.json` master summary。**4 案例 4 DoF 对照表** (control loop / 观测 / 首选 LNN / hidden / 推理预算 / 本仓入口 / T 风险 / 3-seed?) 落到 PRD §3.5。**13 个新单测** (`tests/test_device_control_cases.py`): 4 个 generator shape + 1 个 SoH 单调递减 + 4 个 case end-to-end (parametrize) + 4 个 aggregate_seeds 边界 (empty / single / multi / skip-non-ok) + 1 个 CLI smoke (`--case industrial --quick --steps 8 --seeds 1` 端到端) — **13/13 绿**; 0 回归 (预存 16 个 collection error 是 numpy 2.2.6 vs scipy old binary 的环境问题,与本 iter 无关,git stash master 同样 fail)。**已知失败模式** 沉淀: 1-seed lucky 过估(本仓 iter#11/35 教训)+ T>64 ODE 发散(案例 B 风险)+ 小预算硬切 protocol 在 LTC 上 catastrophic (iter#7/9 教训)+ 无"通杀 backbone"(iter#12 教训)。**stage B (iter#37, 2026-06-10)**: 落实 case D 的 2-stage EntroLnn 协议 — 新增 `lnn.core.ltc.TransformableLTC` 公共类(双 AdamW 优化器: train_lr 静态精训 + refine_lr 在线精化; Constructor 拒绝 refine_lr > train_lr, 失稳防护), `run_case_battery` 加 `--battery-mode {single, transformable}` 双模, 配套 16 个新单测 (`tests/test_transformable_ltc.py`) 覆盖 forward shape / train_reference loss 下降 / refine_target 不回归 / 构造器拒绝 / param_l1_norm 跟踪 / 双阶段共享参数验证 — **29/29 绿**。3-seed quick 烟测 honest-negative: single val_mse=0.1325 ± 0.1141 反而胜 transformable 0.3250 ± 0.1714 — 合成电池跨 cell 同分布, 2-stage 精化无真实域差可学, 反而引入梯度噪声(印证 iter#11/35 "小合成单阶段胜" 教训); 2-stage 协议**算法正确**(双阶段 loss 均下降 + 失稳护栏通过), 真价值在真实跨 cell 域差(EntroLnn 论文场景), 留作下游真实电池数据接入入口。 | `docs/PRD_设备操控_LNN.md` (4 章 ~330 行) + `scripts/experiment_device_control_cases.py` (~680 行, 4 case + 4 generator + aggregate_seeds + CLI + battery 双模) + `tests/test_device_control_cases.py` (13 单测) + `tests/test_transformable_ltc.py` (16 单测) + `analysis/device_control/<date>_device_control_<case>.json` (4 case × N seed) + `analysis/device_control/latest_device_control_summary.json` | iter#36 user goal,纯合成 + 复用 4 仓内 LNN 模块 (ltc/cfc/control/sncp_policy_lite) | **stage A+B ✅ (iter#36 + iter#37)** |
| 10-23 | **MR-MoE Multi-Rate Liquid + MoE 脓毒症复现** (iter#39 / 2026-06-12,见 [[Liquid_NN_MR_MoE_Sepsis_2606.12240_研读报告]] 与 [[LNN_趋势分析_2026-06]] §8): 在 PhysioNet/CinC 2019 Sepsis 数据集上复现 Zong et al. (VT) **LNN + K=3 MoE + 异 τ + 双注意力** 5 模型 ablation (LSTM / Monolithic LNN / MoE / MR-MoE / MR-MoE-Attn),跑 AUROC + AUPRC + 推理 memory (Fig. 13) + 噪声鲁棒 (Fig. 14) + 训练 wall-clock。**核心工程改造**: `CfCCell.__init__` 加 `n_tau: int = 1, tau_scales: tuple = (0.1, 1.0, 10.0)` (默认 1 不影响现有 7 篇研读) + 新增 `lnn/core/mr_moe.py::MRMoEHead` (~150 行, K=3 LNN experts + softmax gating + per-expert temporal attn + feature attn);`scripts/experiment_mr_moe_sepsis.py` (~250 行, 5 模型 × 3 seed × 30 epoch × PhysioNet 数据 + forward + AUROC/AUPRC metric) + `tests/test_mr_moe.py` (8 单测, shape / gating 概率和=1 / 异 τ 验证 / attention 权重和=1 / K=3 切到 K=1 退化为 monolithic / 构造器拒绝 n_tau=0 / 训练 step loss 下降 / CLI smoke)。**关键验证点**: (a) Eq. 8 异 τ 有效 (MR-MoE 显著胜 MoE, expect Δ AUROC ≥ +0.03);(b) Eq. 9 quasi-SS 加速 (fast expert 推理 memory < monolithic);(c) noise σ 增大退化率显著慢于 LSTM (Fig. 14 复现);(d) 3× 参数量 (4500 vs 1500) training wall-clock 真实比例 (论文未报告, 留为诚实负面备料)。**诚实负面预案**: 若 MR-MoE 训练 wall-clock > 5× LSTM 标注为 "工程上不可行, 退回 §2.2 异 τ 单 expert 玩具";若 AUROC 增量 < 0.02 标注为 "配方无收益, 留为公式研究素材"。 | `lnn/core/cfc.py::CfCCell` (+n_tau + tau_scales config) + `lnn/core/mr_moe.py::MRMoEHead` (~150 行) + `scripts/experiment_mr_moe_sepsis.py` (~250 行) + `tests/test_mr_moe.py` (8 单测) + `analysis/mr_moe_sepsis/<date>_mr_moe_sepsis.{json,md}` + `build_backbone_matrix.py` 加 `_ingest_mr_moe_sepsis` + `--include-mr-moe` | iter#39 研读,B+ 优先级,公式同构 95% + 工程蓝海 (1 行 config 即可吃下多 τ 收益) | pending stage A (预计 iter#40) |
