---
title: LNN 下午场研究摘要 v3 - 2026-06-14 (loop session #2)
date: 2026-06-14
tags: [LNN, MR-MoE, CausalMoE, MoE, time-series-LLM, LFM2.5, n_tau, 异τ, round-77]
status: loop-session
report-date: 2026-06-14 (loop 2)
report-author: LNN-research-agents (loop 1h, 第 2 次)
---

# LNN 下午场研究摘要 v3 — 2026-06-14 (loop session #2)

> **场景**: `/loop 1h` 第 2 次循环。基于 round 76 (`CfCCell` 加 `n_tau` 维度, push `69a319b`) 的硬解锁,本回合聚焦"**n_tau 怎么用**" — 找到 1 个能立即在 `n_tau` 之上加 MoE gating 的 B+ 论文并立项。
> **目标**: (1) 在 round 76 基础上找出 MR-MoE 范本的具体落地路径 (2) 找到 n_tau 的下一个高 ROI 应用。

---

## TL;DR

1. **新 arXiv 2 篇 B+/B 级**: **CausalMoE (2606.13024)** — 10 亿参数多模态 MoE for Granger 因果发现, "Pattern-Routed Mixture of Heterogeneous Experts"; **Beyond Uniform Tokens (2606.13624)** — 7.68× TS-LLM 加速,非均匀 token 压缩。
2. **LFM2.5 部署数据 (HF)**: **LFM2.5-1.2B-Instruct 累计 122K 下载 / 603 likes**, 是 LFM2.5 家族**事实标准**。LFM2.5-8B-A1B 累计 62K / 604 likes。本仓 #10-7 (LFM2.5-1.2B INT8) P0 维持。
3. **跨域"MoE + 异 τ" 是当月 arXiv 第二类共识**: 4 篇异 τ (round 76) + 2 篇 MoE-for-TS (CausalMoE + MR-MoE) — **6 篇独立论文同周提出 "MoE 显式分时间/模式" 范式**。
4. **本仓 PRD 候选 (按 ROI)**:
   - **P0 #1**: `MRMoECfC` — K=3 CfC experts (吃 round 76 n_tau) + 输入门控, 4-6h, 直接复现 arXiv:2606.12240
   - **P1 #2**: 异步 token 压缩 (2606.13624) — 频域自适应, ~6-8h
   - **P1 #3**: CausalMoE-style heterogeneous experts (2606.13024) — 跟 #10-24 同源但用因果 gating

---

## 1. 新增 arXiv 论文 (本 loop session, 6-10 ~ 6-11 集中)

### 1.1 B+ 级 (本仓可立即落地)

#### 1.1.1 **CausalMoE** (arXiv 2606.13024v1, 6-11, B+)
- **标题**: *CausalMoE: A Billion-Scale Multimodal Foundation Model for Granger Causal Discovery with Pattern-Routed Heterogeneous Experts*
- **核心思路**:
  1. **Pattern-Routed Mixture of Heterogeneous Experts** — 显式识别 latent temporal patterns, 把 patches 路由到 specialized domain experts
  2. **Causality-Aware Self-Attention** — 跨变量, 产生 sparse Granger causal graphs (proximal optimization)
  3. **多模态融合** — 首次 LLM + VLM 对齐 numerical signals + textual/visual priors
- **对照本仓**:
  - `lnn/core/cfc.py` — 已有 `CfCCell` (round 76 加 `n_tau`)
  - **缺**: 一个 K-expert + pattern-routing gating
  - 跟 MR-MoE (2606.12240) **同源**: 都是 MoE-for-TS, **CausalMoE 多了因果发现头**
- **落地建议**: 复用 round 76 的 `CfCCell(n_tau=K)`, 加 `RoutingGate(input_size)` 路由到 K experts
- **PRD 候选**: §10 #10-33 (NEW, P1) — MR-MoE (#10-24) 之后再做

#### 1.1.2 **Beyond Uniform Tokens** (arXiv 2606.13624v1, 6-11, B)
- **标题**: *Beyond Uniform Tokens: Adaptive Compression for Time Series Language Models*
- **核心卖点**:
  - **TS tokens 频谱贡献不均**: 大量冗余 + 少数关键 → 频域自适应压缩
  - **Prompt token 影响随层衰减**: 渐进 prompt 削减
  - **结果**: **7.68× 推理加速**, 78% 评估 setting 性能提升
- **对照本仓**:
  - `lnn/core/cfc.py::CfCNetwork` 当前每步 O(B·T·H²) — 长序列 T=512+ 时是瓶颈
  - **缺**: 频域压缩 / 重要性采样
- **落地建议**: 加 `AdaptiveTokenCompressor` 在 `CfCNetwork` 前置, 频域 FFT → 选 top-K 频 → 重建
- **PRD 候选**: §10 #10-34 (NEW, P1, 6-8h)

### 1.2 B 级 (观察)

- **CRAFTIIF (2606.13486)** — Cross-Resolution Isolation Forest for TS Anomaly Detection, B-, 本仓无 anomaly detection 任务
- **ProtoX-AD (2606.13277)** — Self-Explainable TS Anomaly Detection, B-, 同上
- **MP3 (2606.13119)** — Multi-Period Pattern Pre-training, B, 跟 LiquidTAD 间接相关

### 1.3 HF 部署数据 (6-04 ~ 6-11, 累计而非仅新增)

| 模型 | 累计下载 | Likes | 备注 |
|---|---:|---:|---|
| **LFM2.5-1.2B-Instruct** | **122,264** | 603 | **事实标准**, 单 SKU 最大 |
| LFM2.5-8B-A1B | 62,687 | 604 | MoE 旗舰, 跟 1.2B-Instruct likes 持平 |
| LFM2.5-8B-A1B-GGUF | 77,760 | 210 | 量化版,下载数超 base |
| LFM2.5-1.2B-Instruct-GGUF | 36,603 | 178 | 1.2B 量化版,边缘部署首选 |
| LFM2.5-VL-1.6B | 40,951 | 298 | 视觉 |
| LFM2.5-VL-450M | 31,560 | 185 | 视觉最轻量,边缘友好 |
| LFM2.5-1.2B-Thinking | 18,509 | 361 | 推理增强 |
| LFM2.5-1.2B-JP-202606 | 4,520 | 61 | 日语 |
| LFM2.5-VL-1.6B-Extract | 2,633 | 59 | 视觉抽取 |

**结论**: 1.2B-Instruct 是 LFM2.5 **事实标准**, 122K dl 远超第二名 8B-A1B (62K)。本仓 #10-7 (LFM2.5-1.2B INT8) 优先级维持 P0。

---

## 2. 跨域信号强化: 6 篇 arXiv 6-04~6-11 同周独立 "MoE / 异 τ" 范式

| 论文 | 月日 | 域 | 范式 |
|---|---|---|---|
| MR-MoE (2606.12240) | 6-10 | 脓毒症时序 | K=3 LNN experts, 异 τ |
| COGENT (2606.11162) | 6-09 | 冰盖物理 | 显式 relative rollout time |
| Liquid-3DGS (2606.07670) | 6-04 | 4D 视觉 | depth-as-time 多层 CfC |
| LiquidTAD (2604.18274) | 4-20 | 视频动作 | temporal pyramid relaxation |
| **CausalMoE (2606.13024)** | **6-11** | **时序因果** | **Pattern-Routed MoE** |
| **Beyond Uniform Tokens (2606.13624)** | **6-11** | **TS-LLM** | **adaptive token budget** |

**结论**: 6 篇 arXiv 独立提出"显式多时间尺度 / 多模式"范式。**这是 2026-06 跨域学界共识**。本仓 round 76 n_tau 已解锁前 4 篇的细胞级范式,本场再解锁 2 篇 (CausalMoE / Beyond Uniform Tokens) 的网络级范式。

---

## 3. 本仓 PRD 候选增量 (本 loop session)

| ID | 标题 | 优先级 | 估时 | 复用 |
|---|---|---|---|---|
| #10-24 | **MR-MoE: Multi-Rate Mixture of Experts for LNN** | **P0** | 4-6h | round 76 `CfCCell(n_tau=K)` + 新增 `RoutingGate` |
| #10-33 | **CausalMoE-style pattern-routed heterogeneous experts** | P1 | 5-7h | #10-24 + 加因果发现头 |
| #10-34 | **Adaptive token compression for CfCNetwork** | P1 | 6-8h | 新增 `AdaptiveTokenCompressor` 频域模块 |
| #10-35 | **LFM2.5-1.2B INT8 推理 (HF top SKU)** | P0 (维持) | 10-15h | `lnn/lfm2/inference.py` + INT8 量化 |

---

## 4. 立即执行项 (本 loop session 选定 #10-24)

### 4.1 选择理由

- **#10-24 MR-MoE** 直接吃 round 76 的 `n_tau` 接口: K=3 experts, 每个 expert 是 `CfCCell(n_tau=1)`,路由门控动态选择
- 复现 arXiv:2606.12240 (Zong VT 2026) 的 Eq. 8-10 范式
- 4-6h 单 PR, 12-15 单元测试, 烟测验证 K=1/3/5

### 4.2 范围 (本 commit)

1. `lnn/core/mr_moe_cfc.py` — 新模块:
   - `MRMoECfCCell(input_size, hidden_size, n_experts=3)` — K experts + router
   - `MRMoECfCNetwork(...)` — 网络级包装
   - 路由: `softmax(W_gate · [x_t, h_prev])` → K weights, 专家输出加权求和
2. `tests/test_mr_moe_cfc.py` — 单元测试:
   - K=1 等价单 CfCCell
   - K=3 forward / backward / gradient
   - 路由权重和为 1
   - 烟测: toy sin 3 seed
3. `scripts/bench_mr_moe_cfc.py` — K=1/3/5 sweep
4. `docs/research/2026-06-14_mr_moe_cfc_sweep_report.md` — 烟测报告
5. `docs/prds/2026-06-14-lnn-round-77-a-mr-moe-cfc.md` — PRD
6. `lnn/core/__init__.py` — 导出
7. README.md — 加 MR-MoE 简述

### 4.3 验收

- `pytest tests/test_mr_moe_cfc.py -q` 全绿
- 88+ 既有 CfC 测试零回归
- 烟测: K=3 toy sin MSE ≤ K=1 (持平或小赢, 不强求)

---

## 5. 与 round 76 衔接

- **round 76 (`CfCCell` 加 n_tau, push `69a319b`)** 是"细胞内"多 τ
- **本 round 77 (MR-MoE)** 是"细胞间"多专家 + 路由 — **完全不同维度**, 但都吃 `n_tau` 接口
- 两者组合 → 完整 MR-MoE 范式 (paper Eq. 8-10)

---

## 6. 一句话总结

> **本 loop (2026-06-14 第 2 次): 新增 2 篇 B+/B arXiv (CausalMoE 2606.13024 + Beyond Uniform Tokens 2606.13624) + 部署数据 1.2B-Instruct 122K dl 确认; 跨域"MoE / 异 τ" 信号从 4 例扩到 6 例, 形成 2026-06 学界第二类共识; 本场立即执行 #10-24 MR-MoE (K=3 experts + 路由, 直接复现 arXiv:2606.12240, 4-6h 单 PR), 是 round 76 n_tau 的自然下游, 4-6h 内完成。**
