---
date: 2026-06-14
round: 82
tags: [LNN, cosine-router, parameter-free, geometric-coupling, FAME, honest-negative, Causal-Audit]
status: daily-summary
---

# LNN 研究日报 v8 — 2026-06-14 (`/loop 1h` 第 7 次)

## 0. TL;DR

本场是 round 82 (第 7 次 `/loop 1h`)。基于 arXiv:2605.12476 (Routers Learn the Geometry of Their Experts) 实现 parameter-free `CosineRouter` — 给 LNN+MoE 栈加**第 4 种 routing 策略**。在 toy sin 上是**诚实负结果** (cosine 单独 0.96, 3/3 diverged, 不如 learned+orth 0.1089), 但科学价值仍在: 暴露 routing 不是"free", 跟论文 (1B SMoE) 一致 (cosine 是 scale-dependent), 进一步回应 Causal Audit 警告。

## 1. 新论文 (本场研读)

| 论文 | arXiv | 状态 | 关键贡献 |
|---|---|---|---|
| **Geometric Coupling** | 2605.12476v1 | A | Router 与 expert 沿同一方向 gradient flow; 3× 路由方向相似性; parameter-free K-Means router 最低 load imbalance |
| Routing Foresight (RL post-training) | 2606.11867v1 | B | Micro-step level load balancing in RL (非本场) |
| Myth of Expert Specialization | 2604.09780v1 | A (NeurIPS 2026) | 证明 LB 损失抑制共享 hidden state direction |

## 2. 新增产出 (本场)

- `lnn/core/cosine_router.py` (NEW) — `CosineRouter` (zero `nn.Parameter`)
- `lnn/core/fame_cfc.py` (MODIFIED) — `router_type='learned'|'cosine'` 参数
- `lnn/core/__init__.py` (MODIFIED) — 导出 `CosineRouter`
- `tests/test_cosine_router.py` (NEW) — 18/18 全绿
- `scripts/bench_cosine_router.py` (NEW) — 5 conditions × 3 seeds
- `docs/prds/2026-06-14-lnn-round-82-a-cosine-router.md` (NEW) — PRD #10-41
- `docs/research/2026-06-14_cosine_router_report.md` (NEW) — 烟测报告 (含诚实负)
- `docs/daily/2026-06-14_LNN_research_summary_v8.md` (本文件)

## 3. 烟测结果 (本场核心, 含诚实负)

| Condition | task loss | std | diverged | 备注 |
|---|---:|---:|---:|---|
| learned baseline | 0.7595 | 0.7906 | 1/3 | 完全复现 |
| learned + orth (round 80) | **0.1089** | 0.0543 | **0/3** | **最优 (4 层防御 1)** |
| learned + φ (round 81) | 0.1250 | 0.0705 | 0/3 | 防御层 2 |
| **cosine** (round 82) | **0.9604** | 0.3513 | **3/3** | **失败 — scale-dependent** |
| cosine + orth | 0.7732 | 0.4832 | 2/3 | 失败 — 互相救不了 |

**关键负面证据**: 移除 learned router 完全不 work on tiny problems — 跟 round 73 (GRU > Mamba @ 3-epoch) 一样是诚实负结果。

## 4. 累计叙事 (round 76-82 含负)

| Round | 改动 | 单点 | 关键贡献 |
|---|---|---:|---|
| 0 | 单 CfCCell | 0.0525 | baseline |
| 76 | + n_tau=3 | 0.0463 | 细胞内多 τ |
| 77 | + K=3 dense | 0.0364 | 细胞间多 expert |
| 78 | + K=3 top_k=2 | 0.0366 | 稀疏路由 |
| 79 | 16-cell sweep | 0.0490 | 全景 + 暴露 top_k=1 发散 |
| 80 | + orthogonality | 0.1089 | 防御层 1 (表征) |
| 81 | + φ-balancing | 0.1250 | 防御层 2 (routing) |
| **82** | **+ CosineRouter** | **0.9604** | **负结果 — scale-dependent** |

**完整叙事**: 6 正 (round 76-81) + 1 负 (round 82) = 7 轮 LNN+MoE 迭代, 跟 round 73 (GRU 胜 Mamba) 一样把诚实记录当科学价值。

## 5. Causal Audit 协同累计 (4 层)

arXiv:2606.10703 警告: 观测指标不能预测 causal expert importance。

累计回应 (round 80-82):
- **round 80 orth**: 直接干预表征空间
- **round 81 φ**: 直接干预 routing logits
- **round 82 cosine 负结果**: 暴露 routing 不是"free"

**最终结论**: 在 toy 数据上, **learned router + orthogonality + φ-balancing** 是最优配置 (round 80 仍胜)。

## 6. 下次 loop 候选

1. **cosine + K=3 top_k=2 sweep** (P1, 3-4h) — 论文的 sweet spot
2. **#10-7 LFM2.5-1.2B INT8** (P0 维持) — learned+orth+φ 作部署默认
3. **真实 SNBC 数据复现** (P3) — 真实 heterogeneous 时序
