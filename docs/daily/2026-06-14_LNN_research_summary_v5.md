---
title: LNN 下午场研究摘要 v5 - 2026-06-14 (loop session #4)
date: 2026-06-14
tags: [LNN, MoE, routing, Causal-Audit, FAME, round-79, K-n_tau-top_K-sweep]
status: loop-session
report-date: 2026-06-14 (loop 4)
report-author: LNN-research-agents (loop 1h, 第 4 次)
---

# LNN 下午场研究摘要 v5 — 2026-06-14 (loop session #4)

> **场景**: `/loop 1h` 第 4 次循环。基于 round 76/77/78 三次硬解锁 (`CfCCell` n_tau + `MRMoECfCCell` dense + `FAMECfCCell` top-K sparse),本回合聚焦"**三维 (K, n_tau, top_K) 哪个组合最优**" — 用 27-cell sweep 给出**数据驱动的答案**。
> **目标**: (1) 找出 round 76/77/78 累计的"细胞内多τ + 细胞间多 expert + 稀疏路由"栈的最优配置 (2) 引入 Causal Audit 论文的"observational ≠ causal"警告 (3) 启动 #10-38 三维 sweep benchmark。

---

## TL;DR

1. **新 arXiv 1 篇 B 级 (反向证据)**: **Causal Audit (arXiv:2606.10703v1, 6-09, B)** — 跨 3 个高冗余 MoE 架构 (OLMoE-1B-7B / Qwen1.5-MoE-A2.7B / DeepSeek-V2-Lite), 60 个 metric-layer 组合 **无观测指标能预测 expert causal importance** (Cohen's d < 0.17)。**这警告我们 FAME top-K 路由的"激活频率"可能不是 causal importance 的可靠代理**。
2. **Routing Foresight (arXiv:2606.11867v1, 6-10, C+)** — RL 后训练 micro-step 负载均衡, 跟本仓 toy 烟测相关性低。
3. **HF 部署数据**: LFM2.5-1.2B-Instruct **122,264 dl / 603 likes** 维持事实标准; 6-11 后无新 LiquidAI 上线。
4. **本仓 PRD 候选 (按 ROI 排序)**:
   - **P0 #1**: **#10-38 K×n_tau×top_K 27-cell sweep** — 3-4h, **回答 round 76-78 三次硬解锁的最优组合**, 是当前最 natural 的数据驱动下一步
   - **P1 #2**: #10-37 Orthogonality constraint — 3-4h, 加在 ForecastabilityRouter 上, 防 top-K 退化 (Causal Audit 论文支持)
   - **P2 #3**: #10-7 LFM2.5-1.2B INT8 部署 — 维持 P0, 10-15h, 但 loop session 时间不够

---

## 1. 新增 arXiv 论文 (本 loop session)

### 1.1 B 级 (反向证据 — 必须记录)

#### 1.1.1 **Causal Audit of Expert Importance** (arXiv 2606.10703v1, 6-09, B)
- **标题**: *From Observation to Intervention: A Causal Audit of Expert Importance in Mixture-of-Experts Models*
- **核心结论** (反观测代理):
  - 跨 3 个高冗余 MoE: **OLMoE-1B-7B-0924 / Qwen1.5-MoE-A2.7B / DeepSeek-V2-Lite**
  - 60 个 metric-layer 组合, **无任何观测指标 (utilization rate, activation norm, routing weight distribution) 能预测 expert causal importance** (multiple-comparison correction 后, Cohen's d < 0.17)
  - Token-level routing weight control 排除 power 不足, 只在 OLMoE 最后一层 MoE 恢复 1 个 Bonferroni-显著信号 (d=+0.231, p=0.0013)
  - **结论**: 现有 pruning 成功**不是因为找到了可移除 expert**, 而是因为 early-layer redundancy 使 selection criteria 互替
- **对照本仓**:
  - `lnn/core/fame_cfc.py::FAMECfCCell` 的 `last_g` / `last_top_idx` 是**观测**信号, 不能直接作为 causal importance
  - `lnn/core/forecastability_router.py` 的 entropy 报告也是**观测**信号
- **落地建议**:
  - #10-38 sweep 报告里**显式注明**: "FAME top-K routing 是 observational signal, 不代表 causal expert importance per arXiv:2606.10703"
  - #10-37 Orthogonality constraint 作为**工程对策** (虽然不直接解决 causal importance 问题, 但能**强制 expert 表征多样性**)
- **PRD 候选**: §10 #10-37 (P1, 跟本 loop 协同)

### 1.2 C+ 级 (观察, 低 ROI)

#### 1.2.1 **Harnessing Routing Foresight** (arXiv 2606.11867v1, 6-10, C+)
- 标题: *Harnessing Routing Foresight for Micro-step-level MoE load balancing in RL Post-training*
- 核心: 解决 RL 后训练 micro-step 级别 load fluctuation
- 对照本仓: 本仓无 RL 后训练,无 micro-step 概念, ROI 低
- PRD 候选: 不入

### 1.3 HF 部署数据 (6-04 ~ 6-11, 维持)

| 模型 | 累计下载 | Likes | 备注 |
|---|---:|---:|---|
| **LFM2.5-1.2B-Instruct** | 122,264 | 603 | 事实标准, 维持 |
| LFM2.5-8B-A1B-GGUF | 77,760 | 210 | 量化版 |
| LFM2.5-8B-A1B | 62,687 | 604 | MoE 旗舰 |
| LFM2.5-VL-1.6B | 40,951 | 298 | 视觉 |
| LFM2.5-1.2B-Instruct-GGUF | 36,603 | 178 | 1.2B 量化版, 边缘部署首选 |
| LFM2.5-1.2B-Thinking | 18,509 | 361 | 推理增强 |
| LFM2.5-1.2B-JP-202606 | 4,520 | 61 | 日语 |
| LFM2.5-8B-A1B-MLX-8bit | 3,952 | 18 | Apple Silicon 8bit |

**6-12~6-14 单日**: 无新 LiquidAI 上线。**本仓 #10-7 (LFM2.5-1.2B INT8) P0 维持,但本 loop 时间预算不够,留给下个独立 session**。

---

## 2. Round 76-78 累计栈回顾 + 本场策略

### 2.1 三次硬解锁 (push 顺序)

| Round | 模块 | push | 单特征贡献 (toy sin) |
|---|---|---|---|
| 76 | `CfCCell(n_tau)` | 69a319b | n_tau=3 vs n_tau=1: -13.4% |
| 77 | `MRMoECfCCell(K=3 dense)` | 3a5cb1f | K=3 vs K=1: -30.7% |
| 78 | `FAMECfCCell(K=3 top_k=2)` | e6e98cf | top_k=2 vs top_k=3: 持平精度 + **3.7× 更稳** |

### 2.2 自然下一步问题

三次硬解锁都在**同一架构栈**内加维度, 累计维度:
- K (number of experts) ∈ {1, 3, 5}
- n_tau (per-expert multi-rate) ∈ {1, 3}
- top_K (sparse activation) ∈ {1, 2, 3, K}

**未解答**: K × n_tau × top_K 的**最优组合**是什么? 例如:
- K=3, n_tau=3, top_k=2 (round 76+77+78 累计) — 是 9 effective τ groups + 1 expert skip
- K=5, n_tau=3, top_k=3 — 15 effective τ groups + 2 expert skip
- K=1, n_tau=3, top_k=1 — 单 expert 但 3 τ groups (round 76 only)

**#10-38 sweep** 给出**数据驱动**答案, 而不是单点 cherry-pick。

---

## 3. 本仓 PRD 候选 (本 loop session)

| ID | 标题 | 优先级 | 估时 | 复用 |
|---|---|---|---|---|
| #10-38 | **K×n_tau×top_K 27-cell sweep** | **P0** | 3-4h | round 76/77/78 全部接口 (3 × 2 × 4 = 24 cell, 加 3 baseline = 27) |
| #10-37 | Orthogonality constraint on router | P1 | 3-4h | 加在 `ForecastabilityRouter` 上, 防 top-K 退化 (Causal Audit 论文支持) |
| #10-7 | LFM2.5-1.2B INT8 推理 | P0 (维持) | 10-15h | 留给独立 session |

---

## 4. 立即执行项 (本 loop session 选定 #10-38)

### 4.1 选择理由

- **#10-38 sweep** 是 round 76-78 三次硬解锁的**自然汇聚**:
  - 不用新加代码, 全部用现有 `CfCCell` / `MRMoECfCCell` / `FAMECfCCell` / `FAMECfCNetwork`
  - 3-4h 单 PR, 27-cell sweep 给出 K×n_tau×top_K 全景
  - 烟测报告 + 表格 + 图表 (markdown table)
  - 立即给出**最优组合**给后续真实数据复现
- 跟 **Causal Audit 论文 (2606.10703)** 协同: 报告里**显式注明** top-K 是 observational signal, 留待未来 causal-aware routing 跟进

### 4.2 范围 (本 commit)

1. `scripts/sweep_kntau_topk.py` — 新脚本:
   - 27 cell = K ∈ {1, 3, 5} × n_tau ∈ {1, 3} × top_k ∈ {1, 2, 3, K}
   - 每个 cell 跑 3 seed, 报告 mean ± std MSE
   - 同时报告 avg activated_per_step, router entropy, n_effective_tau
2. `docs/research/2026-06-14_kntau_topk_sweep_report.md` — sweep 报告:
   - 完整 27-cell 表格
   - 找出最优 cell
   - 跟 round 76-78 单点结果对比, 验证单点不是 cherry-pick
   - **显式注明**: "FAME top-K routing 是 observational signal, 不代表 causal expert importance per arXiv:2606.10703" (避免 over-claim)
3. `docs/prds/2026-06-14-lnn-round-79-a-kntau-topk-sweep.md` — PRD
4. README.md — 加 sweep 表格链接 (可选)

### 4.3 验收

- `python scripts/sweep_kntau_topk.py --epochs 30 --seeds 0 1 2` 跑完 27 cell
- 报告里给出**最优 cell** (按 mean loss) + 跟 round 76-78 单点比较
- 全程不需新加任何 cell/network 代码

---

## 5. 与 round 76-78 衔接

| Round | 改动 | 单点最优 (toy sin) |
|---|---|---|
| 0 | 单 CfCCell | 0.0525 |
| 76 | + n_tau=3 | 0.0463 |
| 77 | + K=3 dense | 0.0364 |
| 78 | + K=3 top_k=2 | 0.0366 (更稳) |
| **79 (本场)** | **3D sweep 27 cell** | **找出全局最优** |

**叙事链路**: "LNN 单一架构无优势" → "n_tau 微正" → "MoE 大幅正" → "稀疏 MoE 持平+更稳" → "3D sweep 全景 (本场)"

---

## 6. 一句话总结

> **本 loop (2026-06-14 第 4 次): 新增 1 篇 B arXiv (Causal Audit 2606.10703, 反向证据: 观测 routing 不能预测 causal importance) + 1 篇 C+ (Routing Foresight); HF LFM2.5 6-12~6-14 维持稳定; 本场立即执行 #10-38 K×n_tau×top_K 27-cell sweep (3-4h 单 PR, 复用 round 76/77/78 全部接口, 无新代码), 是 round 76-78 三次硬解锁的自然汇聚, 用数据驱动回答"细胞内多τ + 细胞间多 expert + 稀疏路由"栈的最优组合, 报告里显式注明 Causal Audit 反向证据避免 over-claim。**
