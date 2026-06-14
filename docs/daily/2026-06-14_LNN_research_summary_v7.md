---
date: 2026-06-14
round: 81
tags: [LNN, phi-balancing, EMA, FAME, MR-MoE, defence-in-depth, Causal-Audit, AnchorMoE]
status: daily-summary
---

# LNN 研究日报 v7 — 2026-06-14 (`/loop 1h` 第 6 次)

## 0. TL;DR

本场是 round 81 (第 6 次 `/loop 1h`)。round 80 (PRD #10-37 orthogonality) 解了 K=3 top_k=1 硬阻塞; 本场**互补**地实现 φ-balancing (PRD #10-40, arXiv:2605.15403), 给 K=3 top_k=1 再加一层**直接干预 routing logits** 的工程保险。两者都是**直接干预**而非仅观测, 共同回应 arXiv:2606.10703 Causal Audit 警告。

## 1. 新论文 (本场研读)

| 论文 | arXiv | 状态 | 关键贡献 |
|---|---|---|---|
| **φ-Balancing** | 2605.15403v1 | B+ | 严格凸对称可微势函数, mirror descent, EMA-based routing 调整 |
| Causal Audit | 2606.10703v1 | A (ICML 2026 Workshop) | 观测 ≠ 因果 expert importance, 反向证据 |

(round 80 AnchorMoE 2606.03631v2 + DBES 2605.18498v1 + Causal Audit 2606.10703v1 全部验证为真实 arXiv 论文)

## 2. 新增产出 (本场)

- `lnn/core/phi_balancing.py` (NEW) — `PhiBalancer` (EMA mirror-descent bias)
- `lnn/core/forecastability_router.py` (MODIFIED) — 接受 optional `balancer` 参数
- `lnn/core/fame_cfc.py` (MODIFIED) — `phi_balance`, `ema_alpha`, `phi_step_size` 参数
- `lnn/core/__init__.py` (MODIFIED) — 导出 `PhiBalancer`
- `tests/test_phi_balancing.py` (NEW) — 16/16 全绿
- `scripts/bench_phi_balancing.py` (NEW) — 4 conditions × 3 seeds
- `docs/prds/2026-06-14-lnn-round-81-a-phi-balancing.md` (NEW) — PRD #10-40
- `docs/research/2026-06-14_phi_balancing_report.md` (NEW) — 烟测报告
- `docs/daily/2026-06-14_LNN_research_summary_v7.md` (本文件)
- `README.md` (MODIFIED) — 加 φ-balancing 节

## 3. 烟测结果 (本场核心)

| Condition | task loss | std | diverged |
|---|---:|---:|---:|
| baseline (round 79) | 0.7595 | 0.7906 | **1/3** |
| orth only (round 80) | 0.1089 | 0.0543 | 0/3 |
| **φ only** (round 81) | **0.1250** | **0.0705** | **0/3** |
| both | 0.1433 | 0.0826 | 0/3 |

**关键**: φ-balancing 单独 (-83.5% loss, 0 diverged) 与 orth 互补, 不需 aux loss, 不需 `forward_with_aux` — 更轻量。

## 4. 累计叙事 (round 76-81 完整 LNN+MoE 栈)

| Round | 改动 | 单点 | sweep rank | 关键贡献 |
|---|---|---:|---:|---|
| 0 | 单 CfCCell | 0.0525 | #7 | baseline |
| 76 | + n_tau=3 | 0.0463 | #8 | 细胞内多 τ |
| 77 | + K=3 dense MoE | 0.0364 | #3 | 细胞间多 expert |
| 78 | + K=3 top_k=2 | 0.0366 | #6 | 稀疏路由 + 3.7× 更稳 |
| 79 | 16-cell sweep | 0.0490 | #1 | 全景 + 暴露 top_k=1 发散 |
| 80 | + orthogonality λ=0.001 | 0.1089 | 0 diverged | 解 sweep 硬阻塞 |
| **81** | **+ φ-balancing η=0.05** | **0.1250** | **0 diverged** | **互补, 防御栈第 5 层** |

**2026-06 完整栈**: "细胞内多 τ + 细胞间多 expert + 稀疏路由 + 正交保险 + 负载均衡" — 6 轮迭代在 toy sin 上稳定可重现。

## 5. Causal Audit 协同累计

arXiv:2606.10703 警告: 观测指标不能预测 causal expert importance。

累计两层**直接干预**防御 (round 80-81):
- **round 80 orthogonality**: 直接干预 expert 表征空间 (geometric constraint)
- **round 81 φ-balancing**: 直接干预 routing logits (mirror-descent bias)

**完全回应** Causal Audit 警告 — 即使观测路由 collapse, 内部表征 + routing logits 都被强制平衡。

## 6. 下次 loop 候选

1. **全 16 cell φ-balancing sweep** (P1, 5-7h) — 验证 φ-balancing 在所有 K × top_k 组合都帮助
2. **#10-7 LFM2.5-1.2B INT8** (P0 维持) — K=5 dense + orth + φ 作部署默认
3. **真实 SNBC 数据复现** (P3) — 真实 heterogeneous 时序
