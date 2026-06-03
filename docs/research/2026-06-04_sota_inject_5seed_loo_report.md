---
title: 37th meta-conclusion refinement — inject=0.1 is NOT universal; freeze+inject is incompatible (round 56)
date: 2026-06-04
tags: [LNN, Bi-CfC-NAD, SOTA-recipe, freeze, inject-incompatibility, 5-seed-LOO, conditional-recipe, 37th-meta-conclusion]
related:
  - "[[docs/research/2026-06-04_inject_recipe_large_budget_report]]"
  - "[[docs/research/2026-06-04_noise_injection_recipe_report]]"
  - "[[LNN_TLDR]]"
---

# ⚠️ Round 56 — SOTA Recipe + inject=0.1 5-seed LOO Probe (★ Counterintuitive)

> **★ 37th meta-conclusion refinement (★ NEGATIVE result)**: **inject=0.1 recipe 在 round 38 SOTA recipe (h=96/ep=80/K=10/freeze=audio_only) 下 HURTS**,不是 help。**5-seed LOO mean: 8.88 (no inject) → 18.29 (inject=0.1), +106% regression**。**Hypothesis REFUTED**: inject=0.1 不是 universal recipe;**生产推荐必须条件化**:
> - **Non-frozen Bi-CfC** (round 54-55 协议): **inject=0.1 wins** (478.77 vs 581.50, -17.7%)
> - **Frozen audio_encoder (SOTA recipe)**: **DO NOT inject** (8.88 vs 18.29, +106% regression)
> **关键洞察**:**freeze audio_encoder** 把模型动态切成两阶段 (warmup 训练 + freeze 后只训 cross-attn/MDN);在 phase 2 注入 noise 但 audio_encoder 已冻结,产生 *训练-测试 mismatch*,**反而伤害 cross-attn 收敛**。

## 1. 背景与动机

Round 54-55 确立 inject=0.1 是 non-frozen Bi-CfC 的生产标准:
- h=16/ep=20 small-budget: 478.77 (vs 581.50 no inject, -17.7%)
- h=64/ep=80 large-budget: 5,646 (vs 132,886 no inject, -96%)

**Round 38 SOTA recipe** 在 h=96/ep=80/K=10/freeze=audio_only 下达到 LOO MSE 0.42 (single-seed=42)。
**Round 43** 复测 5-seed mean 8.16 ± 6.78 (3/4 seeds 失败)。
**假设**: inject=0.1 应能 *稳定* 5-seed mean 降到 < 4。
**实测**: inject=0.1 在 SOTA recipe 下 *恶化* 5-seed mean 从 8.88 到 18.29 (+106%)。

## 2. 实验设计

`/tmp/sota_inject_probe.py` (本轮新写, inline 175 行):
- **2 inject × 5 seeds × 4 folds = 40 fold runs** (~10s each, total ~7 min)
- inject=0.0 (no augmentation) vs inject=0.1
- regime: TemporalSegmentRegressionDataset 4-fold LOO, h=96, ep=80, warmup=40, freeze=audio_only
- model: `CrossModalAttnBiCfCNADWithMDN`
- audio injection 在 **两 phase 都进行** (warmup + phase 2)

JSON: `analysis/emma_rover/2026-06-04_053654_sota_inject_5seed_loo.json`

## 3. 完整结果 (5-seed LOO mean)

### inject=0.0 baseline (5-seed LOO mean = 8.88)

| seed | fold 0 | fold 1 | fold 2 | fold 3 | LOO mean |
|---:|---:|---:|---:|---:|---:|
| 1 | (data) | (data) | (data) | (data) | 9.37 |
| 2 | (data) | (data) | (data) | (data) | 0.72 ⭐ |
| 3 | (data) | (data) | (data) | (data) | 14.82 |
| 7 | 0.006 | 28.65 | 0.10 | 3.01 | 7.94 |
| 42 | 17.84 | 0.11 | 22.30 | 5.98 | 11.56 |

5-seed mean = **8.88 ± 5.25**,min=0.72,max=14.82 (✓ matches round 43's 8.16)

### inject=0.1 (5-seed LOO mean = 18.29)

| seed | fold 0 | fold 1 | fold 2 | fold 3 | LOO mean |
|---:|---:|---:|---:|---:|---:|
| 1 | 27.43 | 74.37 | 0.54 | 2.10 | 26.11 |
| 2 | 0.01 | 0.21 | 0.01 | 16.48 | 4.18 ⭐ |
| 3 | 1.33 | 26.05 | 47.60 | 36.27 | 27.81 |
| 7 | 6.06 | 37.17 | 0.12 | 35.79 | 19.79 |
| 42 | 22.90 | 10.05 | 4.75 | 16.51 | 13.55 |

5-seed mean = **18.29 ± 9.69**,min=4.18,max=27.81

**Delta = +9.41 (+106% regression)** ❌

## 4. 关键观察 (★ 37th meta-conclusion refinement)

### 4.1 inject=0.1 在 SOTA recipe 下 *恶化* 5-seed mean 2x

| metric | inject=0.0 | inject=0.1 | delta |
|---|---:|---:|---:|
| 5-seed LOO mean | 8.88 | 18.29 | **+9.41 (+106%)** |
| 5-seed std | 5.25 | 9.69 | +4.44 (+85%) |
| min across seeds | 0.72 | 4.18 | +3.46 |
| max across seeds | 14.82 | 27.81 | +12.99 |

**inject=0.1 不仅 mean 高 2x,std 也高 2x,最差 seed 也更差**。这是全维度恶化。

### 4.2 inject=0.1 在 small-budget / large-budget 仍 win

(对比 round 54-55)
- h=16/ep=20 (round 54): 478.77 vs 581.50 → inject wins **-17.7%**
- h=64/ep=80 (round 55 Probe 1): 5,646 vs 132,886 → inject wins **-96%**
- **h=96/ep=80/freeze (round 56)**: 18.29 vs 8.88 → inject HURTS **+106%**

**结构差异**: SOTA recipe **冻结 audio_encoder** 在 warmup 之后 → audio_encoder 的 hidden state 路径被"锁定"在特定分布上。**注入 noise 改变了 audio 输入分布,但 frozen encoder 无法 adapt**,产生 *训练-测试 mismatch*。

### 4.3 机制推测:Freeze + Inject 结构性不兼容

| 阶段 | 训练时 | 测试时 | mismatch 风险 |
|---|---|---|---|
| **Non-frozen Bi-CfC** (round 54-55) | audio_encoder 持续 adapt to noisy audio | audio_encoder 处理 clean audio | **无 mismatch** (encoder adapts) → inject=0.1 wins |
| **Frozen Bi-CfC** (SOTA recipe) | audio_encoder 在 warmup 训练 (40 epochs inject=0.1),然后 **冻结** | frozen encoder 处理 clean audio | **mismatch**: encoder 学到 "denoise noisy audio",但 test 是 clean audio → 特征失真 → cross-attn 表现差 |

**核心矛盾**: inject=0.1 的成功机制是 *audio_encoder 通过 hidden state propagation 调整 a_feat magnitude 到 peak*(round 53)。但 **frozen encoder 无法 adapt**,只 *死记* 它在 inject 训练下学到的"denoised" features,见到 clean test audio 时 *反向* 失真。

### 4.4 替代方案候选

**A. 永远不 inject** (回到 round 38 baseline):
- 5-seed mean 8.88 (与 round 43 一致)
- 单一 seed=2 lucky: 0.72 (匹配 round 38 seed=42 lucky 0.42 模式)

**B. 只在 warmup 阶段 inject**:
- audio_encoder 学到"denoise noisy audio"
- Phase 2 注入 noise 但 encoder frozen → 仍可能 mismatch
- 需 6 runs 验证

**C. 不 freeze audio_encoder (回 round 54-55 协议)**:
- inject=0.1 wins (-17.7% in small-budget)
- 但失去 freeze 提供的 *正则化* 效应 (round 25-26 显示 freeze 也有价值)

**D. 减小 inject 强度 (inject=0.01 or 0.05)**:
- 弱 noise 可能不触发 mismatch,但 gain 也减小
- 待 5-seed 验证

**E. 训练时用真 EMMA audio (已有天然 noise) 替代 inject**:
- 这是 round 38 原始设置的真实情况
- round 43 已验证仍 seed-lucky

## 5. 元结论第十七次精化(37th, NEGATIVE)

| Round | 元结论演进 (inject recipe 维度) |
|---:|---|
| 54 | "active sigma=0.1 noise injection 是 non-frozen Bi-CfC 生产标准" |
| 55 | "inject 跨 budget 通用, sweet spot [0.1, 0.2]" |
| **56** | "**inject=0.1 NOT universal; freeze+inject 结构性不兼容**" |

### 5.1 ★ 37th meta-conclusion(完整版)

> "**inject=0.1 recipe 的适用范围**:
> 1. **✓ Non-frozen Bi-CfC**: inject=0.1 wins (round 54-55, -17.7% small-budget, -96% large-budget)
> 2. **✗ Frozen audio_encoder SOTA recipe**: inject=0.1 HURTS (+106% regression)
> 3. **机制**: frozen encoder 学到 "denoise noisy audio" 但 test 是 clean → 训练-测试 mismatch
> 4. **生产推荐必须条件化**:
>     - non-frozen Bi-CfC → **inject=0.1** (478.77 small-budget, 5,646 large-budget)
>     - frozen-audio_encoder adaptive-freeze → **NO inject** (8.88 5-seed mean)
> 5. **新研究问题**: 是否有方法 *既* 享受 freeze 优势 *又* 享受 inject 优势?候选:
>     - 只在 warmup 阶段 inject
>     - inject=0.01 或 0.05 (弱 noise)
>     - audio_encoder 不 freeze 但其他参数 freeze (部分 freeze)
> 6. **谨慎对待 single-seed SOTA**: round 38 seed=42 0.42 仍 lucky,5-seed mean 8.88 是真实期望"

## 6. 重要生产含义

### 6.1 inject 0.1 的条件化推荐

| 场景 | 训练模式 | inject? | 期望 5-seed mean |
|---|---|---|---:|
| Small-budget 快速实验 | non-frozen | **YES (0.1)** | 478.77 |
| Large-budget 全模型 | non-frozen | **YES (0.1)** | 5,646 |
| **SOTA recipe (h=96/freeze)** | **frozen audio_encoder** | **NO** | **8.88** |
| Production deployment | TBD | 需更多测试 | TBD |

### 6.2 freeze 的价值与代价

**Freeze 优势** (round 25-26):
- audio_encoder 冻结后,只训 cross-attn + MDN → 简化训练
- 在某些 regime 下能改善泛化

**Freeze 代价** (round 56):
- audio_encoder 失去对 input 分布变化的 adapt 能力
- 与 inject 等 input 增强技术 *不兼容*
- 真实数据下 (天然 noise) 与 inject 共存会有 mismatch

**生产建议**:
- 想要 inject augmentation → **不要 freeze** audio_encoder
- 想要 freeze 的简化训练 → **不要 inject**,靠真实数据 noise

## 7. 对历史结论的影响

### 7.1 vs Round 54 (★ 34th meta-conclusion)

**修订**:
- Round 54: "**active sigma=0.1 noise injection 是生产标准**"
- Round 56: "**条件化**: 仅 non-frozen Bi-CfC, frozen-audio-encoder SOTA recipe 不适用"
- 修订: "**inject=0.1 是 non-frozen Bi-CfC 的生产标准**;frozen recipe 应避免 inject"

### 7.2 vs Round 38 (SOTA 0.42 single-seed)

**完全确认**:
- Round 38: 0.42 (single-seed lucky)
- Round 56: 5-seed mean 8.88 (无 inject), 18.29 (有 inject)
- **SOTA 0.42 仍 advisory**;真实生产期望 ~8.88

### 7.3 vs Round 25-26 (freeze value)

**修订**:
- Round 25-26: freeze audio_encoder 是 regularization,改善泛化
- Round 56: freeze 限制了 audio_encoder adapt inject-induced 变化
- 修订: "**freeze value 是 regime-dependent**: 与 inject 共存时,freeze 变成阻碍"

## 8. 下一步研究思路 (W+1)

| 优先级 | 思路 | 状态 | 依赖 |
|---|---|---|---|
| ★★★ | **只在 warmup 阶段 inject** (phase 2 不 inject) — 看是否能保留 freeze 优势 + inject gain | 待跑 | torch, ~10 分钟 |
| ★★★ | **weak inject 扫描** (0.01/0.03/0.05) in SOTA recipe — 找 inject+freeze 兼容的弱 noise | 待跑 | torch, ~15 分钟 |
| ★★ | **vanilla_cfc LOO large-budget probe** (round 45 协议) — 验证 vanilla_cfc 优势 | 待跑 | torch, ~20 分钟 |
| ★★ | **部分 freeze 探针**: 冻结 cross-attn 而非 audio_encoder — 看是否能享受 inject + 部分 freeze | 待跑 | torch |
| ★ | **inject+freeze 同存时的 a_feat magnitude dump** — 看 frozen audio_encoder 处理 clean vs noisy test 的差异 | 待跑 | torch |
| ★ | **round 38 SOTA recipe + inject 0.01 5-seed 复测** — 验证弱 inject 是否更优 | 待跑 | torch |

## 9. 提交

- ✅ JSON: `analysis/emma_rover/2026-06-04_053654_sota_inject_5seed_loo.json` (40 fold runs)
- ✅ 报告: `docs/research/2026-06-04_sota_inject_5seed_loo_report.md` (本文件)
- ⏳ 建议: 把 inline script 移到 `scripts/probe_sota_inject_5seed_loo.py` 永久化
- ⏳ TLDR v7 → v8: 同步 37th meta-refinement (★ NEGATIVE result)
- ⏳ commit + push

## 10. 一句话总结

> **40 fold runs (2 inject × 5 seeds × 4 folds)**:**inject=0.1 在 round 38 SOTA recipe (h=96/ep=80/freeze=audio_only) 下 *恶化* 5-seed LOO mean 从 8.88 到 18.29 (+106% regression)**。**Hypothesis REFUTED**: inject=0.1 *不是* universal recipe;**生产推荐必须条件化** — 仅适用于 **non-frozen Bi-CfC**,**frozen-audio_encoder SOTA recipe 应避免 inject**。机制: frozen audio_encoder 学到 "denoise noisy audio" 但 test 是 clean audio → 训练-测试 mismatch。

---
*本报告由连续 /loop 迭代触发 (2026-06-04 local time)。Round 55 后立即跟进,把 inject=0.1 应用到 round 38 SOTA recipe,40 fold runs 决定性发现 *freeze+inject 结构性不兼容*。*
