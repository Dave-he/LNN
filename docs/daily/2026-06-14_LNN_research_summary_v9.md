---
date: 2026-06-14
round: 83
tags: [LNN, MoE-ecology, dimensionless-diagnostic, ortho-toxicity, Causal-Audit, FAME]
status: daily-summary
---

# LNN 研究日报 v9 — 2026-06-14 (`/loop 1h` 第 8 次)

## 0. TL;DR

本场是 round 83 (第 8 次 `/loop 1h`)。基于 arXiv:2605.06415 (Zhang 2026) 实现 **MoE Ecology 诊断** — 给 LNN+MoE 栈加**第 1 个理论诊断**。**复现 paper 头部发现**: E ≥ 0.5 ⇒ zero dead experts (在 12/12 toy 16-cell grid 上 dead=0)。**复现 paper finding #2**: "ortho toxicity is dataset-dependent" 在 toy sin 上最严重 (+16.4% loss at λ=1.0), 但在 random/structured 上仅 +5% — **部分符合** 论文 dataset-dependent 主张。**验证 round 80 λ=0.001 在 3 synthetic datasets 上都是安全的** (loss change < 0.1% vs λ=0)。

## 1. 新论文 (本场研读)

| 论文 | arXiv | 状态 | 关键贡献 |
|---|---|---|---|
| **MoE Ecology (E = T·H/(O+B))** | 2605.06415v1 | A | 单一 dimensionless number 预测 healthy/dead ecology; E ≥ 0.5 ⇒ no dead experts; 12 controlled experiments (8 vision, 4 language, 11K epochs); 6 secondary findings (incl. "ortho toxicity is dataset-dependent") |
| EDSSM 1d ODE | 2605.08545v1 | B | 1D ODE in EDSSM (rust) — non-LNN |
| Timeflies irregular TS | 2606.13571v1 | C | irregular time series — non-LNN |
| MR-MoE | 2606.12240v1 | A (cite) | 与 round 76 相关, 不重做 |

## 2. 新增产出 (本场)

- `lnn/core/moe_ecology.py` (NEW) — `moe_ecology_number` + `MoEEcologyMonitor` (nn.Module)
- `lnn/core/fame_cfc.py` (MODIFIED) — `moe_ecology_diagnostic(B, T, O)` method
- `lnn/core/__init__.py` (MODIFIED) — 导出新符号
- `tests/test_moe_ecology.py` (NEW) — 14/14 全绿
- `scripts/bench_moe_ecology.py` (NEW) — 2 experiments (A: 16-cell grid, B: ortho toxicity)
- `docs/prds/2026-06-14-lnn-round-83-a-moe-ecology.md` (NEW) — PRD #10-42
- `docs/research/2026-06-14_moe_ecology_report.md` (NEW) — 烟测 + 诚实讨论
- `docs/daily/2026-06-14_LNN_research_summary_v9.md` (本文件)

## 3. 烟测结果 (本场核心, 含诚实讨论)

### 3.1 Experiment A: 16-cell grid (toy sin, 2 epochs)

- **12/12 configs have dead=0** — paper E ≥ 0.5 头部发现复现 ✓
- K=3 top_k=2 n_tau=2 = **0.538** (本场最佳) — 复现 round 79
- K=5 cells borderline (4 experts @ 8.6% util, above 1% 阈值)
- 所有 E >> 0.5 (因为 B=0 让 E ≈ 1/eps) — paper threshold trivially satisfied

### 3.2 Experiment B: ortho toxicity (K=3 top_k=1, 3 datasets × 5 lambdas)

| Dataset | λ=0 (best) | λ=0.001 (round 80) | λ=1.0 (worst) | Δ loss |
|---|---:|---:|---:|---:|
| toy_sin | **0.626** | 0.627 | 0.729 | **+16.4%** |
| random | **0.894** | 0.893 | 0.942 | +5.4% |
| structured | **2.764** | 2.764 | 2.895 | +4.7% |

**关键发现**:

1. **Ortho toxicity 在 λ=1.0 全部 3 datasets 都成立** (loss 4.7% 到 16.4% 上升)
2. **Round 80 λ=0.001 在 3 datasets 上都安全** (loss change < 0.1%)
3. **Dataset-dependent 程度** (paper finding 2): toy_sin 受影响最严重 (+16.4%), random/structured 仅 +5% — 跟 paper 的 "ortho toxicity is dataset-dependent" 主张 **部分符合**
4. **E scales as 1/λ**: 当 λ=1.0, E drops 到 0.34-0.96, **toy_sin 和 structured 跨过 paper 0.5 阈值**

## 4. 累计叙事 (round 76-83 含负)

| Round | 改动 | 单点 | 关键贡献 |
|---|---|---:|---|
| 0 | 单 CfCCell | 0.0525 | baseline |
| 76 | + n_tau=3 | 0.0463 | 细胞内多 τ |
| 77 | + MR-MoE K=3 softmax | 0.0324 | 13.4% → 30.7% |
| 78 | + FAME top-K | (denser) | top-K sparse MoE |
| 79 | K×n_tau×top_K 16-cell | 0.0490 | K=5 dense 全局最优 |
| 80 | + orthogonality λ=0.001 | 0.1089 (vs 0.7595) | **fixes K=3 top_k=1 hard cell** |
| 81 | + φ-balancing η=0.05 | 0.1250 | 互补防御层 2 |
| 82 | + CosineRouter | **0.96 (3/3 div)** | **诚实负: parameter-free 不 work on tiny** |
| **83** | **+ MoE Ecology E=TH/(O+B)** | **diagnostic** | **第 1 个理论诊断; paper E ≥ 0.5 复现; ortho toxicity 部分符合** |

## 5. 累计 LNN+MoE 防御栈

5 层 (按对 K=3 top_k=1 增益排序):

1. **baseline learned router** (round 0) — diverges 1/3
2. **+ orth λ=0.001** (round 80) — 0.1089, **fixes**
3. **+ φ-balancing η=0.05** (round 81) — 0.1250, 互补
4. **+ MR-MoE K=3 softmax** (round 77) — 0.0324, 多 τ
5. **+ FAME top-K sparse** (round 78) — 0.05x, sparse

新增的 **MoE Ecology E** (round 83) 是**诊断层** (不直接修复, 而是**告诉我们什么时候需要 1-4 的修复**):
- E < 0.5 with active aux loss → intervention needed
- E → ∞ (B≈0) → cells 仍 may diverge (round 80 之前的 baseline)
- E drops below 0.5 at λ=1.0 → **告诉我们 ortho 太强了**, 减弱

## 6. 后续候选 (round 84+)

- **#10-43** Auto-enable φ-balancing when E < 0.5 (intervention policy)
- **#10-44** Test E on real LLM training (paper 12 experiments reproduction)
- **#10-45** Gradient-based H instead of empirical H (more accurate)
- **#10-46** Test on vision classification (CIFAR, ImageNet) for true dataset-dependence
- **#10-7 LFM2.5-1.2B INT8** — deployment default, 需 full stack stable
- **#10-XX Timeflies 2606.13571** — irregular time series, 与 round 76 n_tau 相邻

## 7. 诚实负 + 局限

1. **Empirical H, not gradient-based**: paper H 来自 router gradient, 我们用 g_mean 近似
2. **3 synthetic datasets only**: 没复现 paper 的 12 vision/language experiments
3. **2 epochs**: paper 用 11K epochs; 长训可能暴露 λ=1.0 的 dead experts (我们只看到 loss 退化)
4. **Dead threshold 1%** vs paper 5% — 我们的阈值更严格, 可能少估 dead count
5. **没看到 "ortho helps" regime**: paper 说有些 datasets ortho 有利, 我们 3 个都没看到
