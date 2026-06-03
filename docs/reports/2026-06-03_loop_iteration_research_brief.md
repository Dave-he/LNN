---
title: 2026-06-03 Loop 迭代 — LNN 论文与仓库简评
date: 2026-06-03
tags: [LNN, loop, brief, paper-review, repo-watchlist]
parent: [[LNN_深度研读报告]]
---

# 2026-06-03 Loop 迭代 — LNN 论文与仓库简评

> 由 `/loop 1h` 调度自动触发;本次重点对最近 10 天内新出现的
> LNN/CfC/LTC/LFM 相关论文与仓库做 ≤200 字的"是否值得深读 / 是否值得复现"判断,
> 输入到 [[NEXT_STEPS]] 与 [[PRD_LNN_Edge_Research]] 的优先级队列。

## 1. arXiv 候选(top-7)

### 1.1 Comparative Analysis of Liquid Neural Networks and LSTM — Robustness, Efficiency, Clinical Utility (2605.27467, 2026-05-26)
- 作者: Ye Kyaw Thu 等。
- 看点: 把 CfC vs LSTM 放在 **临床序列模式识别** 任务上,
  控制鲁棒性 / 效率 / 临床效用三轴。
- 适配本仓库: 已有 [[Comparative_Analysis_of_LNN_and_LSTM_研读报告]];
  建议下一个 loop 补 *PhysioNet 子集 + Jetson CPU 推理时延* 列。
- 复现成本: 低 — 数据少,模型小,适合 Orin Nano CPU 跑通。
- 推荐: **B+(下一个 loop 跟进)**。

### 1.2 EMMA — Extracting Multiple physical parameters from Multimodal Data (2605.24047, 2026-05-21)
- 物理参数从 video+audio+image time-series 多模态联合恢复。
- 与本仓 `experiment_multimodal_lnn.py` 设计对路:
  EMMA 的图像/音频路径可以接到 CfC 时间主轴。
- 复现成本: 中等(数据需自合成),GPU 强烈推荐。
- 推荐: **A(列入 PRD §8 任务 #4)**。

### 1.3 Physics-Modeled Neural Networks (2605.08176, 2026-05-05)
- 提出 DynPMNNs: 每个隐层是一个 ODE,
  与 LTC 共享"连续时间 = 网络"假设。
- 与 [[LNN_训练方向_物理建模与多模态科学发现_可行报告]] 重合。
- 复现成本: 低(synthetic damped oscillator 即可)。
- 推荐: **A−** — 可在 `scripts/experiment_physics_lnn.py` 加 baseline 对照。

### 1.4 LiquidTAD — Efficient Temporal Action Detection (2604.18274, 2026-04-20, v2)
- 并行 Liquid-inspired relaxation 显著降低 TAD 参数量与计算。
- 仓 `scripts/experiment_long_sequence.py --mode tad` 已经覆盖了 smoke;
  下一步要在 THUMOS14 子集做正式对照。
- 复现成本: 中(数据准备最费工)。
- 推荐: **A(列入 PRD §8 任务 #2)**。

### 1.5 Liquid Networks with Mixture Density Heads for Imitation Learning (2603.27058, 2026-03-28)
- 已有 `scripts/experiment_imitation_lnn.py --head mdn`,
  本次 loop 直接以这条命令做一次 smoke (待下一个 loop 跑)。
- 推荐: **B+(执行准备就绪)**。

### 1.6 Robust Hybrid Beamforming with Liquid Crystal Antennas + LNNs (2604.07219, 2026-04-08)
- 与 [[LNN_训练方向_图时空与通信系统_可行报告]] 主题相关,
  但所需 6G/sub-THz 信道仿真器不在本仓范围。
- 推荐: **C(仅记录)**。

### 1.7 Explainable Continuous-Time Mask Refinement for Medical Image Segmentation (2603.00459, 2026-02-28)
- CT-LTC 思路用于医学分割边界精炼。
- 与 `lnn/core/variants.py::CTLTC` 共享一类 motif,
  可借此评估 CTLTC 在视觉/分割上的可移植性。
- 推荐: **B(下下个 loop 评估)**。

## 2. GitHub 候选(top-5)

| 仓库 | 评级 | 摘要 / 下一步 |
|---|---|---|
| [`Linlab2026/GCN-CfC`](https://github.com/Linlab2026/GCN-CfC) | **A** | CfC 接 GCN 做分子筛选,与 `experiment_graph_lnn.py` 思路重合;下一个 loop 在本仓 smoke 跑通即可。 |
| [`Alexng2024/EDSSM`](https://github.com/Alexng2024/EDSSM) | **A−** | 事件驱动闭式连续状态空间模型,显式 `e^{AΔt}`,与 CfC-DT 一脉相承;值得做对照 benchmark。 |
| [`infinition/LSTN`](https://github.com/infinition/LSTN) | **B** | Rust 语言层面尝试液态语言模型;Jetson 部署友好(无 CUDA 依赖),适合做"非 PyTorch 路径"调研。 |
| [`parhat1/cfdna-tau-repository`](https://github.com/parhat1/cfdna-tau-repository) | **B** | LNN + cfDNA 癌症检测,within-cohort AUC=0.91,LOSO 0.40 — 提醒我们要做泛化评估而非单一 cohort。 |
| [`everest-an/O1`](https://github.com/everest-an/O1) | **B−** | MT-LNN / brain-inspired liquid network 原型,常数 memory + O(1) cache;架构思路有借鉴,但项目还很早期。 |

## 3. Hugging Face 模型(top-5)

| 模型 | 评级 | 说明 |
|---|---|---|
| `LiquidAI/LFM2.5-8B-A1B` | **A**(参考) | 48k+ 下载;本仓 Jetson Orin Nano 8GB 显存装不下 8B 全精度,但适合做 KV cache / RAM 估算的对照。 |
| `LiquidAI/LFM2.5-8B-A1B-MLX-4bit` | **A** | 4bit MLX 量化;虽然 MLX 是 Apple Silicon,但量化方案 + 配方可迁移到 Jetson 上的 GGUF 流程。 |
| `reaperdoesntknow/LFM2.5-1.2B-Distilled-SFT` | **A+** | **首选** 进入 Jetson 推理 — 1.2B 蒸馏 + SFT,显存与 Orin Nano 匹配;PRD §8 任务 #3。 |
| `coder3101/LFM2.5-VL-450M-heretic` | **B+** | LFM2.5 VL 450M,适合做多模态边缘推理对照,等待官方文档稳定后再评估。 |
| `w-ahmad/LFM2.5-8B-A1B-GGUF-MoQ` | **B+** | GGUF + MoQ 混合量化,作为 LFM2.5 边缘 llama.cpp 路径备选。 |

## 4. 本 loop 决策

- **立即执行(本次 commit 内)**: 第 1–2 节的简评归档 + 今日 benchmark refresh(已完成)。
- **下一 loop 入栈**: PRD §8 任务 #1(Jetson CUDA wheel)与 #3(LFM2.5-1.2B INT 推理)。
- **重要观察**: arXiv API 在本时段返回 429,GitHub Search 返回 403 限流;
  `daily_lnn_research.py` 已正确保留前一轮结果,但 loop 频率 1h 偏高,
  下次可考虑把 loop 间隔扩到 2–3h,或把"采集"与"评估"拆成两条 cron。

## 5. 相关索引

- [[PRD_LNN_Edge_Research]]
- [[NEXT_STEPS]]
- [[Liquid_Neural_Networks_Latest_Papers_Summary]]
- [[每日自动化任务与Jetson验证]]
- [[analysis/jetson/2026-06-03_lnn_benchmark|今日 Jetson Benchmark]]
- [[analysis/repo_watchlist/2026-06-03_lnn_open_source_watchlist|今日 Watchlist]]
