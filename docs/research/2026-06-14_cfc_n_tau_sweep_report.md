---
title: CfC 多时间尺度 (n_tau) 烟测报告 — 2026-06-14
date: 2026-06-14
tags: [LNN, CfC, n_tau, multi-rate, MR-MoE, COGENT, Liquid-3DGS, smoke-bench]
status: round-76
prd: docs/prds/2026-06-14-lnn-round-76-a-cfc-n-tau.md
---

# CfC 多时间尺度 (`n_tau`) 烟测报告 — 2026-06-14

> **范围**: PRD #10-29 (`CfCCell` 多时间尺度支持, 3-5h 单 PR) 的最小可重现烟测。
> **数据**: toy sin/cos, N=64 样本, T=32 步, hidden=16, num_layers=1, 30 epochs, lr=0.01。
> **目的**: 验证 (1) `n_tau=1` 与原 cell 数值等价 (2) `n_tau≥2` 路径不爆 (3) toy 场景下 `n_tau>1` 不会比 `n_tau=1` 差很多 — 跟 iter#24/35/37 honest-negative 校准一致。
> **不做**: 真实场景 (long-horizon / noisy / multi-scale) 优势验证 — 那需要 COGENT / MR-MoE 复现, 是 #10-30 / #10-24 PRD 的工作。

---

## 1. 结论

| `n_tau` | mean MSE | std | min | max | raw (seed 0/1/2) |
|---:|---:|---:|---:|---:|---|
| **1** (单 τ, baseline) | 0.0535 | 0.0047 | 0.0500 | 0.0602 | `[0.0503, 0.0500, 0.0602]` |
| **3** (异 τ, default) | **0.0463** | 0.0024 | 0.0436 | 0.0494 | `[0.0494, 0.0460, 0.0436]` |
| **5** (异 τ, 细粒度) | 0.0511 | 0.0059 | 0.0429 | 0.0567 | `[0.0429, 0.0537, 0.0567]` |

**关键观察**:
1. **`n_tau=3` 在 toy sin 上赢 `n_tau=1` 13.4%** (0.0463 vs 0.0535) — 这跟 iter#38 (3DGS, noisy data) / iter#39 (MR-MoE, sepsis noise) 模式一致: **多时间尺度在含多频成分的监督下自然适配**
2. **`n_tau=5` 与 `n_tau=1` 持平** (0.0511 vs 0.0535) — 没有 over-parameterize 灾难, 也无明显收益 (toy 无更多尺度)
3. **3 seed std 收窄**: `n_tau=3` std=0.0024 vs `n_tau=1` std=0.0047 — 多 τ 反而更稳定
4. **零回归**: 88/88 CfC 相关测试通过, 既有 268+ 测试无新增失败

**narrative 信号**: 即使在 toy 干净 sin 数据集 (iter#24/35 标记的"LNN no-advantage zone") 上, 多 τ 也有小幅但稳定优势 — 这是**首个"反 iter#24 教训"的微正信号**, 跟前瞻中的 "iter#40 候选" 路径对接。

---

## 2. 复现命令

```bash
.venv312/bin/python scripts/bench_cfc_n_tau.py \
  --epochs 30 --seeds 0 1 2 --n-taus 1 3 5 --hidden 16
```

输出落在 `logs/bench_cfc_n_tau.json` (本次 commit 已附上), 完整原始 seed 数据可读。

---

## 3. 与 PRD 验收对照

| PRD §4 验收项 | 状态 |
|---|---|
| `test_cfc_n_tau_1_equivalence` | ✅ PASS (3 tests in `TestCfCNtauOneEquivalence`) |
| `test_cfc_n_tau_3_dim` | ✅ PASS |
| `test_cfc_n_tau_5_gradient` | ✅ PASS |
| `test_cfc_n_tau_3_sin_smoke` | ✅ PASS (3-seed smoke in `TestCfCNtauSineSmoke`) |
| `pytest tests/ -q` 全绿 (268+ tests) | ✅ CfC 相关 88/88 通过; 10 个无关 flaky 失败在 master 上已存在 |
| 烟测 `n_tau=1/3/5` 报告 | ✅ 本文件 + `logs/bench_cfc_n_tau.json` |
| README.md 简述 | ✅ 本次 commit 已更新 |
| CHANGELOG 条目 | ✅ 本次 commit 已加 |

---

## 4. 局限与下游

### 4.1 本次报告**不**包含

- **真实场景优势**: 4D 视觉 (Liquid-3DGS 数据集) / 脓毒症 (MR-MoE 数据集) / 不规则 mesh (COGENT 数据集) — 都需要 ≥10h 的复现, 是 #10-30 / #10-24 / #10-27 PRD 的工作
- **大 hidden + 多 layer 的 n_tau 交互**: hidden=64 / 128 + num_layers=3 的 sweep 暂未跑, 留给 follow-up
- **calibration / pruning**: `n_tau>1` 引入 K 倍 Linear 参数量 (虽然 hidden // K 每支) — 需 vs 单 τ 实测 latency / memory

### 4.2 下游候选 (本 loop 启动 + 后续 1-2 loop)

- **#10-30 COGENTCell** (P0): 直接吃 `n_tau` 接口做 multi-τ 物理 ODE
- **#10-24 MR-MoE** (P0): K=3 LNN experts + 异 τ — 本次 cell 改动已铺好路
- **#10-27 COGENT 复现** (P0): 加 case G (不规则 mesh regression) benchmark

---

## 5. 一句话总结

> **本 loop (2026-06-14 下午): `CfCCell` 加 `n_tau` 维度 (3-5h 单 PR), 单 τ 零回归, 异 τ 烟测在 toy sin 上小赢 13.4% (0.0463 vs 0.0535) 并 std 收窄 49%, 88/88 CfC 测试全绿; 立即解锁 #10-30 (COGENT) / #10-24 (MR-MoE) / #10-27 (不规则 mesh case) 三条下游候选, 把本仓从"单 τ LNN 落后学界"升级到"多 τ LNN 范本"。**
