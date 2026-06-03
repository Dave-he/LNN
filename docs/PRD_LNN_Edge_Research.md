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
| 5 | Comparative Analysis of LNN & LSTM(2605.27467)对照重做 | `docs/reports/Comparative_Analysis_of_LNN_and_LSTM_研读报告.md` 增补 v2 | 1 loop |
| 6 | GCN-CfC 分子筛选模型 smoke(GitHub Linlab2026/GCN-CfC) | `analysis/repo_watchlist/2026-06-03_gcn_cfc.md` | 1–2 loop |
| 7 | 把 Jetson Pareto sweep 接入 PRD 验证指标 #3 | 更新 `scripts/jetson_lnn_benchmark.py` + 本 PRD | 1 loop |
| 8 | Loop 调度产物去重 + 自动 retro(避免重复研读相同 paper) | `scripts/select_papers_for_report.py` 改造 | 1 loop |

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
