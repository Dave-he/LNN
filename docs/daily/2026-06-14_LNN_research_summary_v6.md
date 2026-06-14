---
title: LNN 下午场研究摘要 v6 - 2026-06-14 (loop session #5)
date: 2026-06-14
tags: [LNN, orthogonality, AnchorMoE, phi-Balancing, DBES, expert-collapse, sweep-followup, round-80]
status: loop-session
report-date: 2026-06-14 (loop 5)
report-author: LNN-research-agents (loop 1h, 第 5 次)
---

# LNN 下午场研究摘要 v6 — 2026-06-14 (loop session #5)

> **场景**: `/loop 1h` 第 5 次循环。基于 round 76-79 累计栈 (n_tau + K + top_K + 16-cell sweep),本回合聚焦"**top_k=1 极不稳定怎么解**" — sweep 显示 K=3 top_k=1 训练发散到 1.86 (mean 0.7595 ± 0.7906),需要**正交约束防 expert collapse**。
> **目标**: (1) 找 #10-37 orthogonality constraint 的具体公式 (2) 启动单 PR 实施 (3) sweep 重跑验证 top_k=1 不稳定被解。

---

## TL;DR

1. **新 arXiv 1 篇 B+ + 1 篇 B + 1 篇 anchor(已研读)**:
   - **φ-Balancing (arXiv:2605.15403v1, 5-14, B+)** — 严格凸对称可微势函数 + mirror descent, EMA-based routing 调整, 几乎无开销, 持续胜 Switch-style / loss-free baseline
   - **DBES (arXiv:2605.18498v1, 5-18, B)** — 5 个专家特化诊断指标 (Routing Specialization, Normalized Effective Rank, Domain Isolation, Routing Stiffness, N-gram Expertise)
   - **AnchorMoE (arXiv:2606.03631v2, 6-02, B)** — **「geometric orthogonality constraint that penalizes representational redundancy」** ← 直接模板
2. **round 79 sweep 揭示 P0 风险**: K=3 top_k=1 mean 0.7595 ± 0.7906 (部分 seed 发散到 1.86),**专家 collapse 是真实威胁**。Causal Audit (arXiv:2606.10703) 进一步证明观测指标不能预测 causal importance — 这正是 #10-37 要解的痛点。
3. **本仓 PRD 候选 (按 ROI 排序)**:
   - **P0 #1**: **#10-37 Orthogonality Constraint** — AnchorMoE 模板, 3-4h, 加在 FAMECfCCell.forward 返回 orthogonality_loss, sweep 重跑验证 top_k=1 不再发散
   - **P1 #2**: #10-40 φ-Balancing style EMA load balancing — 5-7h, 替代 orthogonality
   - **P2 #3**: #10-41 DBES 5 指标诊断 — 文档级 3-4h, 不上代码

---

## 1. 新增 arXiv 论文 (本 loop session)

### 1.1 B+ 级 (本仓可立即落地)

#### 1.1.1 **φ-Balancing** (arXiv 2605.15403v1, 5-14, B+)
- **标题**: *φ-Balancing for Mixture-of-Experts Training*
- **核心**:
  - **原则性框架**: 严格凸对称可微势函数, 最小化 routing 分布的期望
  - **Min-max 公式**: 通过凸对偶导出
  - **Mirror descent 在线算法**: EMA-based routing 调整, 几乎无开销
  - **持续胜 Switch-style / loss-free baseline**: 大规模 pretraining + 下游 fine-tuning
- **对照本仓**:
  - 跟 #10-37 orthogonality 是**互补**方案(都是防 collapse)
  - φ-Balancing 走**显式 routing distribution 平衡**, orthogonality 走**专家表示空间去相关**
- **落地**: 可作 #10-40 候选 (5-7h, EMA + mirror descent)
- **PRD 候选**: §10 #10-40 (P1)

#### 1.1.2 **AnchorMoE** (arXiv 2606.03631v2, 6-02, B) — 重点
- **标题**: *AnchorMoE: Interpretable Time Series Classification via Anchor-Routed MoE*
- **核心 (与 #10-37 直接相关)**:
  - **Multi-view patch representation** + 路由到 specialized experts
  - **Geometric orthogonality constraint** — 惩罚 expert 表示冗余, 强迫 expert 学**异质**预测模式
  - **Uncertainty-aware reliability gate** — 抑制背景噪声
- **公式 (本仓适配)**:
  ```
  L_orth = Σ_{i<j} ||cos_sim(h_i, h_j)||²
  其中 h_i 是 expert i 的 hidden state 输出
  ```
  - 极简: 对每对 expert 的 hidden state 计算 cosine similarity, 取平方求和
  - 加 λ * L_orth 到主 loss
- **对照本仓**:
  - 本仓 `FAMECfCCell` 的 K experts hidden states 可直接取
  - 加 orthogonality_loss 函数 + 在 network forward 返回辅助 loss
  - λ 默认 0.01 (轻量, 跟 AnchorMoE 论文 spirit 一致)
- **落地**: 3-4h 单 PR, 加 `orthogonality_loss(expert_outputs)` + 测试 + sweep 重跑
- **PRD 候选**: §10 #10-37 (P0, 本场执行)

### 1.2 B 级 (诊断/评估)

#### 1.2.1 **DBES** (arXiv 2605.18498v1, 5-18, B)
- **标题**: *DBES: A Systematic Benchmark and Metric Suite for Evaluating Expert Specialization in Large-Scale MoEs*
- **5 个诊断指标**:
  1. **Routing Specialization** — 路由是否对不同输入有差异化
  2. **Normalized Effective Rank** — expert 输出的有效秩
  3. **Domain Isolation** — 跨域 expert 是否被分离
  4. **Routing Stiffness Score** — 路由对输入扰动的敏感度
  5. **N-gram Expertise** — 序列 n-gram 级别的 expert 偏好
- **对照本仓**:
  - 直接可借鉴指标 #1 (Routing Specialization) 验证 orthogonality 是否真的让 expert **特化**
  - 不需复现完整 benchmark, 只取 1-2 指标作 sanity check
- **PRD 候选**: §10 #10-41 (P2, 文档级 3-4h)

---

## 2. Round 79 Sweep 揭示的 P0 风险 (本场 narrative 关键)

| Cell | K | top_k | mean loss | std | 备注 |
|---:|---:|---:|---:|---:|---|
| K=3, n_tau=1 | 3 | **1** | **0.7595** | **0.7906** | **部分 seed 发散到 1.86, 训练失败** |
| K=3, n_tau=1 | 3 | 2 | 0.0646 | 0.0130 | 稳定 |
| K=3, n_tau=1 | 3 | 3 | 0.0579 | 0.0105 | 稳定 |
| K=5, n_tau=1 | 5 | 1 | 0.2395 | 0.0993 | 同样发散风险 (std 0.099) |

**结论**: **top_k=1 (router argmax, 单 expert) 极不稳定** — 这是 sweep 暴露的硬阻塞。
- Causal Audit 论文 (2606.10703) 进一步证明观测指标无法预测 causal importance
- AnchorMoE orthogonality constraint 是直接对策

---

## 3. 本仓 PRD 候选 (本 loop session)

| ID | 标题 | 优先级 | 估时 | 复用 |
|---|---|---|---|---|
| #10-37 | **Orthogonality Constraint on Expert Representations** | **P0** | 3-4h | AnchorMoE 模板, 加在 `FAMECfCCell` |
| #10-40 | φ-Balancing EMA load balancing | P1 | 5-7h | φ-Balancing 论文, mirror descent |
| #10-41 | DBES 5 指标诊断 | P2 | 3-4h (doc) | DBES 论文指标 #1 + #2 |

---

## 4. 立即执行项 (本 loop session 选定 #10-37)

### 4.1 选择理由

- **#10-37 Orthogonality Constraint** 是 round 79 sweep 暴露的**硬阻塞对策**:
  - K=3 top_k=1 训练发散 (0.7595 ± 0.7906)
  - Causal Audit 反向证据支持需要 explicit diversity enforcement
  - AnchorMoE 论文给出**具体公式** (geometric orthogonality constraint)
  - 3-4h 单 PR, 风险低
- 跟 sweep 形成闭环: 实施后**重跑 sweep** 验证 K=3 top_k=1 不再发散

### 4.2 范围 (本 commit)

1. `lnn/core/orthogonality.py` — 新模块:
   - `orthogonality_loss(expert_outputs: list[Tensor], lambda_coeff: float = 0.01) -> Tensor`
   - 公式: `L_orth = Σ_{i<j} ||cos_sim(h_i, h_j)||²`
2. `lnn/core/fame_cfc.py` — 改 `FAMECfCCell.forward` 返回 `(hidden, aux_loss)`,`FAMECfCNetwork.forward` 累加 aux_loss, **同时** 维持原 return API (aux_loss 作为额外输出供 trainer 选)
3. `tests/test_orthogonality.py` — 单元测试:
   - 全 0 expert 输出 → 0 loss
   - 完全正交 expert 输出 → 0 loss
   - 重复 expert 输出 → 高 loss
   - 梯度流到 expert parameters
   - lambda=0 时无影响 (back-compat)
4. `scripts/bench_orthogonality.py` — sweep 重跑 K=3 top_k=1 with λ ∈ {0, 0.01, 0.1}, 验证发散被解
5. `docs/research/2026-06-14_orthogonality_report.md` — 报告
6. `docs/prds/2026-06-14-lnn-round-80-a-orthogonality-constraint.md` — PRD
7. README.md — 加 orthogonality 简述

### 4.3 验收

- `pytest tests/test_orthogonality.py -q` 全绿
- 132+ 既有 CfC+MR-MoE+FAME 测试零回归
- **关键**: bench 验证 K=3 top_k=1 加上 orthogonality 后 std 显著下降 (目标: < 0.05, vs 当前 0.7906)

---

## 5. 与 round 76-79 衔接

- **round 76** n_tau: 13.4% toy 收益
- **round 77** K=3 dense: 30.7% toy 收益
- **round 78** K=3 top_k=2: 持平 + 3.7× 更稳
- **round 79** sweep: 揭示 top_k=1 不稳定
- **本 round 80** orthogonality: **解 top_k=1 不稳定**, 让稀疏 top-K 路径在 toy 上也稳定

---

## 6. 一句话总结

> **本 loop (2026-06-14 第 5 次): 新增 1 篇 B+ arXiv (φ-Balancing 2605.15403) + 1 篇 B (DBES 2605.18498) + AnchorMoE (2606.03631, B) 的 orthogonality constraint 作为 #10-37 模板; 锚定 round 79 sweep 暴露的 K=3 top_k=1 发散 (0.7595 ± 0.7906) 作为硬阻塞; 本场立即执行 #10-37 Orthogonality Constraint (AnchorMoE 公式, 3-4h 单 PR), 期望 K=3 top_k=1 加 orthogonality 后 std < 0.05 (vs 当前 0.79), 让 top_k=1 路径在 toy 上稳定, 解 sweep 暴露的硬阻塞。**
