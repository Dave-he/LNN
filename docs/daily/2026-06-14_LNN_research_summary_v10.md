---
date: 2026-06-14
round: 84
tags: [LNN, ecology-gated-balancing, auto-intervention, FAME, Causal-Audit, ortho-toxicity, honest-negative]
status: daily-summary
---

# LNN 研究日报 v10 — 2026-06-14 (`/loop 1h` 第 9 次)

## 0. TL;DR

本场是 round 84 (第 9 次 `/loop 1h`)。**关闭 round 83 诊断层的回环**：实现 `EcologyGatedBalancer` — 当 live E < 0.5 时, **自动启用 φ-balancing**。给 round 83 的被动诊断 (E) 加**牙齿**, 让它从 monitor 升级为 **autonomous cell-health manager**。**诚实负结果**: 在 orth λ=1.0 的 ortho-toxicity regime, gated φ **跟 always-φ 一样不能恢复** (gate 正确触发, 但 φ 本身不够强), 这是 round 83 finding #2 "ortho toxicity is dataset-dependent" 的延伸。Gate 在 E 健康的 random dataset 上**正确不触发** (无 false positive)。

## 1. 新论文 (本场研读)

| 论文 | arXiv | 状态 | 关键贡献 |
|---|---|---|---|
| E = T·H/(O+B) | 2605.06415v1 | A (round 83) | MoE Ecology 头部发现 (本场基线) |
| φ-Balancing | 2605.15403v1 | A (round 81) | φ-balancing EMA mirror-descent (本场干预) |
| Causal Audit | 2606.10703v1 | A (round 79+) | Observational ≠ causal (本场 gate 的局限) |
| AnchorMoE orth | 2606.03631v1 | A (round 80) | 本场 gate 触发的背景 (orth 退化为 λ=1.0) |

## 2. 新增产出 (本场)

- `lnn/core/ecology_gated_balancing.py` (NEW) — `EcologyGatedBalancer` (无 hysteresis)
- `lnn/core/fame_cfc.py` (MODIFIED) — `ecology_gated_balancing` flag + gate wiring + `_step_idx`
- `lnn/core/__init__.py` (MODIFIED) — 导出 `EcologyGatedBalancer`
- `tests/test_ecology_gated_balancing.py` (NEW) — 13/13 全绿
- `scripts/bench_ecology_gated.py` (NEW) — 3 conditions × 3 datasets, orth λ=1.0 强制 E < 0.5
- `docs/prds/2026-06-14-lnn-round-84-a-ecology-gated-balancing.md` (NEW) — PRD #10-43
- `docs/research/2026-06-14_ecology_gated_balancing_report.md` (NEW) — 烟测 + 诚实负
- `docs/daily/2026-06-14_LNN_research_summary_v10.md` (本文件)

## 3. 烟测结果 (本场核心, 含诚实负)

3 conditions × 3 datasets, **orth λ=1.0 强制 E < 0.5** 触发 gate:

| Condition | Dataset | Loss | E_last | Dead | Gate |
|---|---|---:|---:|---:|---:|
| A baseline (no φ) | toy_sin | 0.7347 | 0.00 | 2 | -1 |
| B always-φ (η=0.05) | toy_sin | 0.7286 | 0.28 | 1 | -1 |
| **C gated-φ (auto)** | toy_sin | 0.7302 | 0.00 | 2 | **16** ✓ |
| A baseline | random | 0.9420 | 0.96 | 0 | -1 |
| B always-φ | random | 0.9431 | 0.97 | 0 | -1 |
| **C gated-φ** | random | 0.9420 | 0.96 | 0 | **-1 (no false pos)** ✓ |
| A baseline | structured | 2.8953 | 0.00 | 2 | -1 |
| B always-φ | structured | 2.9057 | 0.00 | 2 | -1 |
| **C gated-φ** | structured | 2.8953 | 0.00 | 2 | **16** ✓ |

**关键发现**:

1. **Gate 触发逻辑正确**: toy_sin (E=0) 和 structured (E=0) 都正确触发 at step 16
2. **No false positive**: random (E=0.96 > 0.5) gate 正确不触发
3. **诚实负 (核心)**: 在 λ=1.0 ortho-toxicity regime, gated φ 跟 always-φ 一样**不能恢复** — gate 正确触发, φ 自动附加, 但 orth 损失太强, φ 救不回来
4. **Round 83 finding #2 延伸**: ortho toxicity 不是 φ 单独能解的, 需要**更强干预** (例如 auto-disable orth 而非 auto-enable φ)

## 4. 累计叙事 (round 76-84 含负)

| Round | 改动 | 单点 | 关键贡献 |
|---|---|---:|---|
| 0 | 单 CfCCell | 0.0525 | baseline |
| 76 | + n_tau=3 | 0.0463 | 细胞内多 τ |
| 77 | + MR-MoE K=3 softmax | 0.0324 | 13.4% → 30.7% |
| 78 | + FAME top-K | (denser) | top-K sparse MoE |
| 79 | K×n_tau×top_K 16-cell | 0.0490 | K=5 dense 全局最优 |
| 80 | + orthogonality λ=0.001 | 0.1089 (vs 0.7595) | **fixes K=3 top_k=1 hard cell** |
| 81 | + φ-balancing η=0.05 | 0.1250 | 互补防御层 2 |
| 82 | + CosineRouter | 0.96 (3/3 div) | 诚实负: parameter-free 不 work on tiny |
| 83 | + MoE Ecology E | diagnostic | 第 1 个理论诊断; E≥0.5 复现; ortho toxicity 部分符合 |
| **84** | **+ Ecology-Gated φ** | **policy** | **第 1 个自动决策层; gate 正确; 诚实负: λ=1.0 orth 救不回来** |

## 5. 累计 LNN+MoE 自主栈 (5 + 1 + 1 = 7 层)

5 层防御 (round 76-81):
1. baseline learned router
2. + orth λ=0.001
3. + φ-balancing η=0.05
4. + MR-MoE K=3 softmax
5. + FAME top-K sparse

1 层诊断 (round 83):
6. **MoE Ecology E** — tells us *when* we need 1-5

1 层决策 (round 84):
7. **Ecology-Gated φ** — auto-enable 3 when E < 0.5

**未完成**: gate 选择 φ (而不是 orth disable), 在 λ=1.0 regime 救不回。下一轮候选: gate 选择**更强制**的干预 (orth disable / routing reset)。

## 6. 后续候选 (round 85+)

- **#10-44** Auto-disable orth when E < 0.5 (替代或附加于 auto-φ)
- **#10-45** Gradient-based H (替代 empirical H)
- **#10-46** Test on vision classification (paper 12 experiments)
- **#10-47** Causal importance-based gate (回应 Causal Audit)
- **#10-7 LFM2.5-1.2B INT8** — deployment, 需 full stack stable
- **#10-XX Timeflies 2606.13571** — irregular time series

## 7. 诚实负 + 局限

1. **Gated φ 不比 always-φ 更强** — 它只决定**何时**启用 φ
2. **No auto-disable** — 一旦触发, 永远不退 (设计选择, 可改进)
3. **E 是 observational** — 用 empirical mixture weights, 不是 gradient-based H
4. **Orth toxicity 是 dataset-dependent** — round 83 已确认, 本场 bench 只测 λ=1.0 (definitely toxic), 没测 transition region (λ=0.01-0.1)
5. **2 epochs** — 长训可能暴露不同的 gate 行为
6. **Eval mode 不自动 attach balancer** — by design (eval 不应改变 model)
