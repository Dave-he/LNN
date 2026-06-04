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
| 10-2 | DynPMNN stage B:加 `--backbone fhn_dynpmnn` 到 ablation runner,跑 multi-seed 对比 | matrix 新增 dynpmnn 列 | §10 #1 之后 | pending |
| 10-3 | Comparative LNN vs LSTM phase-D:hidden=64 + epochs=50 + samples=4000,看 LNN 优势是否随规模出现 | `analysis/timeseries_ablation/<date>_phase_d.md` | §9 #2 v6 | pending |
| 10-4 | 给 `experiment_graph_lnn_molecule.py` 加 `HierarchicalDecayLiquidTADHead` 作为 recurrent 选项(交叉 #2 与 #6) | code + smoke | 综合 | pending |
| 10-5 | `loop_status.py` 加 `--prd-status` 子模式:解析 §8/§9/§10 全表,出未完成 + 阻塞理由 | code + sample report | §9 #5 衍生 | ✅ (iter#21) |
| 10-6 | `build_backbone_matrix.py` 加 `--export-readme-snippet`:产 README 顶部 badge 行(LSTM 3/4 win 等) | code + README badge | §9 #7 衍生 | pending |
| 10-7 | LFM2 (LFM2.5-1.2B-Distilled-SFT)在 RAM 空载窗口跑 1 次 INT8 推理 + token/sec 表 | `analysis/lfm25/<date>_lfm25_int8_jetson.md` | §8 #3 / §9 #1 | pending (RAM blocker) |
| 10-8 | `analysis/loop_status/` 自动生成 README 标签云(高频 task / 高方差 seed 提示) | tooling | meta | pending |
| 10-9 | **SVAF (arXiv 2604.03955) τ-modulated peer-blending 算子复现**(iter#17 研读,见 [[Symbolic-Vector_Attention_Fusion_SVAF_研读报告]]):toy 2-agent mesh + τ_i ∈ {1, 10, 60} 三组神经元,N 步耦合后看 spectral diff 验证"fast τ 同步 / slow τ 主权"现象 | `analysis/svaf/2026-06-04_tau_toy.md` + ~100 行 core code | iter#17 研读, 最小可复现单元 | **stage A ✅ (iter#22)**, stage B pending (P2) |
| 10-10 | **PDNA (arXiv 2603.00153) PulseHead + Gapped protocol 复现**(iter#18 研读,见 [[Pulse-Driven_Neural_Architecture_PDNA_研读报告]]):stage A `lnn/core/cfc.py::PDNAPulseHead` (~80 行) + unit test ✅ (iter#19, 12 tests);stage B sMNIST Gapped protocol 3 seed × 5 backbone ablation, backbone matrix 加 smnist_gap 行 ✅ (iter#20, cfc_pulse multi-gap +2.53 pp);stage C Long Range Arena 长程任务 | code + matrix 行 | iter#18 研读,**代码公开** + MNIST zero-cost | **stage A+B ✅, stage C pending (P1)** |
| 10-11 | **LFM/LNN-related active≤3B vs 30B+ LLM 打榜证据卡**:汇总 `LFM2.5-8B-A1B`(8.3B total / 1.5B active)公开指标、`Qwen3-30B-A3B` 30B+ 基线、本仓 `LFM2.5-1.2B` 本地 GGUF/DPO 推理证据;默认判定 13 个重叠指标 7 胜 / 6 负,只支持 active≤3B MoE 局部胜出,**不支持 exact 3B dense 全面吊打** | `scripts/build_llm_battlecard.py` + `analysis/llm_battlecard/2026-06-04_llm_battlecard.{json,md}` + `tests/test_llm_battlecard.py` | 用户 3B-vs-30B 目标 | ✅ iter#26 claim-audit 落地 |
| 10-12 | **本机 LFM2.5 GGUF micro-eval harness**: `scripts/run_llm_micro_eval.py` 通过 llama.cpp 对本机模型跑 deterministic arithmetic / instruction / JSON / abstention sanity tasks;当前 `LFM2.5-1.2B-Instruct-Q4_0.gguf` 7/7,最近一次平均生成 16.8 tok/s(随系统负载波动);该门槛不替代公开榜,但阻断“模型部署都没跑通就谈打榜”的假阳性 | `analysis/llm_micro_eval/2026-06-04_lfm25_1_2b_instruct_q4_micro_eval.{json,md}` + `tests/test_llm_micro_eval.py` | 用户 3B-vs-30B 目标 | ✅ iter#27 本机实测入口落地 |
| 10-13 | **30B+ OpenAI-compatible endpoint micro-eval 后端**: `run_llm_micro_eval.py --backend openai-chat` 支持 `/v1/chat/completions`,可直接接本机 llama-server/vLLM/SGLang 或远程 30B+ API,输出同一 JSON schema;当前已用本地 fake OpenAI server 单测覆盖,等待真实 30B+ endpoint/权重 | code + unit test | 用户 3B-vs-30B 目标 | ✅ iter#28 endpoint 对照入口落地 |
| 10-14 | **LLM micro-eval leaderboard 汇总器**: `build_llm_micro_leaderboard.py` 扫描 `analysis/llm_micro_eval/*_micro_eval.json`,按 accuracy → task coverage → mean tok/s 排序,输出可追加 30B+ endpoint 行的 JSON/Markdown 榜单;当前只有本机 `LFM2.5-1.2B-Instruct-Q4_0.gguf` 一行,1/1 rankable,7/7,16.843 tok/s,尚未形成真实 30B+ 对照 | `analysis/llm_micro_eval/2026-06-04_llm_micro_leaderboard.{json,md}` + `tests/test_llm_micro_leaderboard.py` | 用户 3B-vs-30B 目标 | ✅ iter#29 micro leaderboard 落地 |

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

### §10 完成度(由 iter#17 跟踪)

- **0/9 完成**(全部 pending,iter#17 加 10-9 SVAF)
- 最小可启动无阻塞项: 10-5 (loop_status 已有),10-8 (loop_status 已有)
- 最小可复现新论文单元: 10-9 (τ 调制算子 50 行)
