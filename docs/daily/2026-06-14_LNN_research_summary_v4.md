---
title: LNN 下午场研究摘要 v4 - 2026-06-14 (loop session #3)
date: 2026-06-14
tags: [LNN, FAME, forecastability-aware, MoE, sparse-routing, top-K, MR-MoE, round-78, AnchorMoE]
status: loop-session
report-date: 2026-06-14 (loop 3)
report-author: LNN-research-agents (loop 1h, 第 3 次)
---

# LNN 下午场研究摘要 v4 — 2026-06-14 (loop session #3)

> **场景**: `/loop 1h` 第 3 次循环。基于 round 76 (`CfCCell` 加 `n_tau`) + round 77 (`MRMoECfCCell` K-experts + softmax router) 的双重硬解锁,本回合聚焦"**MR-MoE 的路由如何升级**" — 从密集 softmax 升级到**稀疏 top-K + 可预测性感知**。
> **目标**: (1) 找出 round 77 的自然下游 (2) 找到 B+ 论文有真实生产数据 + 代码可复现 (3) 启动 #10-36 Forecastability-Aware Router 单 PR。

---

## TL;DR

1. **新 arXiv 1 篇 B+ 级 + 1 篇 B 级**: **FAME (arXiv:2606.08896v1, 6-08, B+)** — 真实生产部署 (山东新北洋 5000+ 售货机 / 60M+ 交易), Top-2 路由 -12.4% MSE vs LightGBM, 1.92 experts/series; **AnchorMoE (arXiv:2606.03631v2, 6-02, B)** — 可解释 MTSC, 正交约束 + 锚路由。
2. **跨域"MoE-for-TS" 是当月 arXiv 第三类共识**: round 76/77 已收 6 例 (异 τ + MoE),本场再扩 **FAME 稀疏 top-K 路由** (与 round 77 密集 softmax 形成代际差),跨域共识从 6 例扩到 **7 例**。
3. **FAME 是 MR-MoE 的天然下一代**: round 77 用密集 softmax (所有 K experts 都 forward),FAME 用稀疏 top-K (只激活 K' < K 个 experts,典型 K'=2)。**1.92 experts/series** 实测稀疏度。
4. **本仓 PRD 候选 (按 ROI 排序)**:
   - **P0 #1**: `ForecastabilityRouter` — top-K 稀疏路由 + 多维可预测性指纹, 5-7h, 复现 arXiv:2606.08896
   - **P1 #2**: `AnchorRouter` — 正交约束 + 锚路由, 6-8h, 复现 arXiv:2606.03631
   - **P1 #3**: 集成 `MRMoECfCCell` + FAME router 进 CfCNetwork 烟测, 4-6h
   - **P2 #4**: K×n_tau×top_K 全交叉 sweep, 10+h

---

## 1. 新增 arXiv 论文 (本 loop session, 6-02 ~ 6-11 集中)

### 1.1 B+ 级 (本仓可立即落地)

#### 1.1.1 **FAME** (arXiv 2606.08896v1, 6-08, B+)
- **标题**: *Forecastability-Aware Mixture of Experts for Heterogeneous Time Series Forecasting*
- **作者**: 山东新北洋 (SNBC) + 公开 retail benchmark
- **核心思路**:
  1. **Multidimensional Forecastability Fingerprint** — 每个 series 表征为多维特征 (lifecycle, sparsity, volatility, seasonality, spectral patterns, contextual sensitivity)
  2. **Expert-Suitability Targets** — 从 validation performance 挖掘每个 expert 适合哪类 series
  3. **Cost-Aware Sparse Router** — 学习激活**小预算** expert 集合 (Top-2 实测平均 1.92 experts/series)
  4. **生产部署** — 真实 5000+ 售货机, 60M+ 交易, **集成进补货计划 pipeline**
- **关键数据**:
  - **Top-2 routing** 相对最强单 expert (LightGBM) **-12.4% MSE**
  - 1.92 experts/series 平均激活 — 真实稀疏
  - LightGBM 单 expert 已是 retail 强 baseline
- **代码**: https://github.com/hit636/FAME (公开)
- **对照本仓**:
  - `lnn/core/mr_moe_cfc.py::MRMoECfCCell` (round 77) — 密集 softmax router, K experts 都 forward
  - **缺**: 稀疏 top-K router + 显式 forecastability fingerprint
- **落地建议**:
  - 加 `ForecastabilityRouter(input_size, hidden_size, n_experts, top_k=2)` — 输出 top-K sparse weights
  - 复用 round 77 的 `MRMoECfCCell` experts, 把 `softmax(W · [x; h])` 换成 `top-k(mask(W · [x; h]))`
  - 烟测: 构造不同"可预测性"区段 (低频 vs 高频 sin),验证 top-K 路由能区分
- **PRD 候选**: §10 #10-36 (NEW, P0) — round 77 的自然下一代

#### 1.1.2 **AnchorMoE** (arXiv 2606.03631v2, 6-02, B)
- **标题**: *AnchorMoE: Interpretable Time Series Classification via Anchor-Routed MoE*
- **核心思路**:
  1. **Multi-view patch representation** + 路由到 specialized experts
  2. **Geometric orthogonality constraint** — 防止 expert 表示冗余
  3. **Uncertainty-aware reliability gate** — 动态校准每段贡献, 抑制背景噪声
- **对照本仓**:
  - 本仓无 MTSC 任务, 直接落地 ROI 低
  - 但 **orthogonality constraint** 是防 expert collapse 的好工程组件, 可借鉴到 FAME
- **PRD 候选**: §10 #10-37 (NEW, P1) — 仅作 FAME 的辅助组件

### 1.2 B 级 (观察)

- **LongMoE (2606.09907, 6-06)** — Trajectory-Aware MoE for Longitudinal Multimodal, B, 偏医疗
- **TimeROME-DLM (2606.12841, 6-11)** — Temporal Causal Tracing for Masked Diffusion LMs, B, 偏 LLM editing
- **MoE Transformer for AMR (2606.09085, 6-08)** — Automatic Modulation Recognition, B-, 通信

### 1.3 HF 部署数据 (6-04 ~ 6-11, 维持稳定)

| 模型 | 累计下载 | Likes | 备注 |
|---|---:|---:|---|
| **LFM2.5-1.2B-Instruct** | 122,264 | 603 | **事实标准, 维持** |
| LFM2.5-8B-A1B-GGUF | 77,760 | 210 | 量化版 |
| LFM2.5-8B-A1B | 62,687 | 604 | MoE 旗舰 |
| LFM2.5-VL-1.6B | 40,951 | 298 | 视觉 |
| LFM2.5-8B-A1B-MLX-8bit | 3,952 | 18 | Apple Silicon 8bit |
| LFM2.5-1.2B-JP-202606 | 4,520 | 61 | 日语二次发布 |

**6-11 单日**: 无新 LiquidAI 上线, 6-12~6-14 期间 HF 暂稳。**本仓 #10-7 (LFM2.5-1.2B INT8) P0 维持**。

---

## 2. 跨域信号强化: 7 篇 arXiv 6-02~6-11 同周独立 "MoE / 异 τ" 范式

| 论文 | 月日 | 域 | 范式 |
|---|---|---|---|
| MR-MoE (2606.12240) | 6-10 | 脓毒症时序 | K=3 LNN experts, 异 τ |
| COGENT (2606.11162) | 6-09 | 冰盖物理 | 显式 relative rollout time |
| Liquid-3DGS (2606.07670) | 6-04 | 4D 视觉 | depth-as-time |
| LiquidTAD (2604.18274) | 4-20 | 视频动作 | temporal pyramid |
| CausalMoE (2606.13024) | 6-11 | 时序因果 | Pattern-Routed MoE |
| Beyond Uniform Tokens (2606.13624) | 6-11 | TS-LLM | adaptive token budget |
| **FAME (2606.08896)** | **6-08** | **零售售货机时序** | **Top-K sparse routing** |

**结论**: 7 篇 arXiv 独立提出"显式多专家 / 多时间尺度"范式, **2026-06 跨域学界第三类共识**。**FAME 是首个"稀疏 top-K 路由 + 生产部署"范本**,跟 round 77 的"密集 softmax + 学术 prototype"形成代际差。

---

## 3. 本仓 PRD 候选增量 (本 loop session)

| ID | 标题 | 优先级 | 估时 | 复用 |
|---|---|---|---|---|
| #10-36 | **ForecastabilityAwareRouter (top-K sparse routing)** | **P0** | 5-7h | round 77 `MRMoECfCCell` experts + 新 `ForecastabilityRouter` |
| #10-37 | **Orthogonality constraint (反 expert collapse)** | P1 | 3-4h | 加在 #10-36 上, 防 top-K 退化 |
| #10-38 | **K×n_tau×top_K sweep** | P2 | 10+h | 三维交叉 (K=1/3/5 × n_tau=1/3 × top_K=1/2/3) |
| #10-39 | **LFM2.5-1.2B INT8 推理 (#10-7 维持)** | P0 | 10-15h | `lnn/lfm2/inference.py` + INT8 量化 |

---

## 4. 立即执行项 (本 loop session 选定 #10-36)

### 4.1 选择理由

- **#10-36 ForecastabilityAwareRouter** 是 round 77 MR-MoE 的**直接升级**:
  - round 77 密集 softmax (K experts 都 forward)
  - 本场稀疏 top-K (只激活 K' < K, FAME 实测 K'=2)
  - **接口兼容**: `MRMoECfCCell` 接受任意 callable router,只需替换 router 实现
- **FAME 有生产数据 + 代码** (山东新北洋 5000+ 售货机, GitHub repo) — 不是 paper-only
- **5-7h 单 PR**, 12-15 单元测试, 烟测验证 top-K=1/2/3 vs 全 softmax

### 4.2 范围 (本 commit)

1. `lnn/core/forecastability_router.py` — 新模块:
   - `ForecastabilityRouter(input_size, hidden_size, n_experts, top_k=2)` — top-K sparse routing
   - 路由: `logits = W · [x_t; h_prev]` → `top_k(logits)` + softmax over top-K → K' nonzero weights
   - 接口: 与 `MRMoECfCCell.experts` 列表对接
2. `lnn/core/fame_cfc.py` — 新模块 (轻量级包装):
   - `FAMECfCCell(input_size, hidden_size, n_experts, top_k=2)` — MRMoECfCCell 替换 router 为 ForecastabilityRouter
   - `FAMECfCNetwork(...)` — 网络级包装
3. `tests/test_fame_cfc.py` — 单元测试:
   - top-K=1 等价于单 expert
   - top-K=K 等价于 softmax
   - top-K=2 稀疏度
   - 路由梯度流到被选 expert
   - 烟测: toy sin
4. `scripts/bench_fame_cfc.py` — top-K=1/2/3 sweep
5. `docs/research/2026-06-14_fame_cfc_sweep_report.md` — 烟测报告
6. `docs/prds/2026-06-14-lnn-round-78-a-fame-cfc.md` — PRD
7. `lnn/core/__init__.py` — 导出
8. README.md — 加 FAME 简述

### 4.3 验收

- `pytest tests/test_fame_cfc.py -q` 全绿
- 116+ 既有 CfC+MR-MoE 测试零回归
- 烟测: top-K=2 在 toy sin MSE ≤ top-K=1, 接近 top-K=K=3

---

## 5. 与 round 76/77 衔接

- **round 76 (push 69a319b)**: 细胞**内**多 τ → 13.4% toy sin 收益
- **round 77 (push 3a5cb1f)**: 细胞**间**多 expert + 密集 softmax 路由 → 30.7% toy sin 收益
- **本 round 78 (#10-36)**: 路由升级到稀疏 top-K → 期望 ≥30.7% (且 inference 更快)

**叙事链路**: "LNN 单一架构无优势" → "n_tau 微正" → "MoE 大幅正" → "稀疏路由 production-grade"

---

## 6. 一句话总结

> **本 loop (2026-06-14 第 3 次): 新增 1 篇 B+ arXiv (FAME arXiv:2606.08896, 山东新北洋 5000+ 售货机生产部署, Top-2 路由 -12.4% MSE vs LightGBM) + 1 篇 B (AnchorMoE 2606.03631); 跨域 "MoE/异τ" 信号从 6 例扩到 7 例, 形成 2026-06 学界第三类共识; 本场立即执行 #10-36 ForecastabilityAwareRouter (top-K 稀疏路由, 5-7h 单 PR, 复现 arXiv:2606.08896), 是 round 77 MR-MoE 密集 softmax 路由的自然下一代, 期望 inference 更快 + 收益维持或上升。**
