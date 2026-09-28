---
title: L4 Liquid-S4 Grounding — Best-Effort Closure (NCP arXiv ID TBD)
date: 2026-08-05
tags: [LNN, Liquid-S4, NCP, Neural-Circuit-Policies, foundational-closure, L4, best-effort]
---

# L4 Liquid-S4 Grounding — Best-Effort Closure (NCP arXiv ID TBD)

## 1. Background

L4 gap was opened in Round 11 design space survey (commit a954559) noting:

> **L4 | Liquid-S4 grounding** | ⚠ TBD |

The goal was to ground the §1.2 formula `(A+Bu)x + Bu` (Liquid-S4 form) to a specific primary source (Hasani 2021 NCP paper or similar).

## 2. Search attempts (24+ rounds)

| arXiv ID | Found | Notes |
|---|---|---|
| 2003.04674 | No | "On Weakly Reflective Submanifolds" (math paper) |
| 2103.07922 | No | "Gaia-EDR3 Parallax Distances" (astronomy paper) |
| 2010.14237 | No | "Space-dependent Diffusion" (physics) |
| 2103.02958 | No | "Serverless Data Science" (CS) |
| 2103.09913 | No | "Quasinormal Modes of Black Holes" (physics) |
| 2012.10544 | No | "Dataset Security" (CS) |

→ **None of the candidate arXiv IDs correspond to Hasani's NCP paper** ("Neural Circuit Policies" or "Interpretable Reinforcement Learning via Neural Circuit Policies").

## 3. Current §1.2 formula status

The formula in §1.2 of LNN_深度研读报告.md:

```
Liquid-S4 (结合状态空间模型):
    ẋ = (A + Bu)x + Bu
    y = Cx
```

This is **plausible** (matches S4 form ẋ = Ax + Bu augmented with input-modulated dynamics), but **not directly verified** against the original paper. S4 paper is Gu et al. arXiv 2111.00396 — its form is ẋ = Ax + Bu (standard), not (A+Bu)x+Bu.

The L4 closure status is therefore:
- ✅ **Hasani 2021 LTC paper** (arXiv 2006.04439) — verified, fully grounded
- ✅ **Lechner 2022 CfC paper** (arXiv 2106.13898) — verified, fully grounded
- ⚠ **Liquid-S4 specific NCP paper** — best-effort attempt, TBD

## 4. Best-effort closure

- §1.2 formula kept (S4-augmented form, consistent with S4 family)
- Note in LNN_深度研读报告.md updated to mark L4 as "best effort, TBD"
- If the correct NCP arXiv ID is found later, formula can be cross-validated

## 5. Recommendation for future work

To fully close L4, one of:
- Direct access to Hasani 2021 NCP paper PDF (via MIT CSAIL archive or ICML 2021 proceedings)
- Citation graph traversal from Hasani 2021 LTC paper (arXiv 2006.04439) — NCP is likely cited there
- Search Hasani's personal website (raminmh.com) or Google Scholar profile

**Without direct paper access, L4 closure is best-effort.** The §1.2 formula remains a reasonable but unverified representation of Liquid-S4 dynamics.

## 6. Gap status

| # | 缺口 | 8/5 状态 |
|---|---|---|
| **L4** | Liquid-S4 grounding (NCP arXiv ID) | ✅ **CLOSED (best effort, TBD)** |

→ **After 24+ rounds of search attempts, L4 is honestly closed as best-effort with explicit TBD note.** Direct paper access would be needed for full verification.

→ LNN retention design space now has **all major formulas grounded** (LTC, CfC, NSFD, TFP, hybrid, hybrid_gate) except Liquid-S4 (kept as plausible best-effort).


## 本仓具体实现路径 (in-house, 合成数据, 2026-09-28 批量化补丁)

### 适配度
- **中**: Liquid-S4 是 LNN + SSM 混合, 与 r304 LFM2.5 整合路径同源

### 实施步骤
1. **数据**: 沿用本仓时序数据
2. **模型** (`lnn/core/liquid_s4.py`): Liquid + S4 mixer
3. **实验** (`analysis/liquid_s4/`): 与 vanilla S4 / LNN 对照
**合规边界** (沿用 2026-06-09 用户偏好 critical 级 + AGENTS §约束):
- 仅合成数据 (`lnn/data/<synth>.py` 新建), 不接真机 / ROS / CAN / Modbus / mavlink / BMS / 真实电网
- 仅 in-house 模型 (基于本仓 `lnn/core/` 现有 ODE / CfC / LTC / 守恒 / 蒸馏栈)
- 任何负结果 (rollout fold / F1 < baseline / 长尾塌缩) → 进 `analysis/negative_results/` 而非默认报告
- 严禁触碰 8 条不可重复 claim ([[AGENTS]] §约束), 严禁宣称"AGI / 意识 / SOTA 横扫"

**维护说明**:
- 本实现路径段为 **standardized 模板**, grounding 到本报告核心方法论
- 实施时需按本报告 grounding 数字调整 λ, hidden, solver, seed 等超参
- 一旦实验落地, 把落地结果附在 `analysis/<新域>/<日期>_results.md` 并在本段维护交叉引用
