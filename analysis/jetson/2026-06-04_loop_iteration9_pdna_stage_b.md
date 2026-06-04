---
title: Jetson validation summary — iter#20 PDNA stage B: 3-seed × 5-variant sMNIST Gapped ablation
date: 2026-06-04
platform: Linux hyx-desktop 5.15.148-tegra (Jetson Orin Nano Super, aarch64)
python: 3.14.4 (pyenv) — primary, CPU path
cuda: torch 2.11.0+cu130 → cuda.is_available()=False (BSP driver 12.060 < cu130)
tag: loop-iteration, jetson-cpu-only, pdna-stage-b, smnist-gapped-ablation, prd-10-10
---

# Jetson validation summary — iter#20 PDNA stage B

> 本轮执行 **PRD §10 #10 stage B** —— 5-variant × 3-seed ablation on
> Sequential MNIST with the paper's Gapped evaluation protocol。
> 上一轮 (iter#19) 已落地 `PDNAPulseHead` 类 + 12 unit test。

## 1. 改动量

```
scripts/experiment_pdna_smoke.py     +330 行 (5 变体 × N seed sMNIST Gapped)
scripts/build_backbone_matrix.py     +55 行 (_ingest_smnist_gap + --include-smnist-gap flag)
analysis/pdna/2026-06-04_pdna_stage_b_summary.{md,json}    新增
analysis/pdna/2026-06-04_pdna_*_seed*.json (15 个)         新增
analysis/backbone_matrix/2026-06-04_144554_backbone_matrix.{md,json}   新增
```

## 2. 实验设计

- **数据**: Sequential MNIST (28 timesteps × 28 features, 10 类)
  - 训练: 8000 / 60000 随机子集 (smoke scale)
  - 测试: 1500 / 10000 随机子集
- **架构**: 5 变体
  | 变体 | Backbone | PDNAPulseHead 模式 | 目的 |
  |---|---|---|---|
  | A. baseline_cfc | CfCNetwork | 无 | 论文对照 |
  | B. cfc_noise | CfCNetwork | NoiseHead (matched α=0.01) | 论文 critical control |
  | C. cfc_pulse | CfCNetwork | pulse only (β=0) | 单独 oscillation |
  | D. cfc_selfattend | CfCNetwork | attend only (α=0) | 单独 self-attention |
  | E. full_pdna | CfCNetwork | pulse + attend | Full PDNA |
- **训练**: hidden=64 (paper 128), 5 epochs (paper 40), AdamW + cosine annealing, batch=128, lr=5e-4
- **Gapped protocol**:
  - 0% / 5% / 15% / 30% (contiguous, centered at T/2)
  - **multi20** (4 evenly spaced gaps, total 20%) — 论文 headline 指标

## 3. 关键结果(论文 Table 4 对应)

| 变体 | n_params | Gap 0% | Gap 5% | Multi-gap (⭐) |
|---|---:|---:|---:|---:|
| baseline_cfc ⚠N<5 (n=3) | 18570 | 37.98±4.03 | 34.13±2.66 | 35.51±6.40 |
| cfc_noise ⚠N<5 (n=3) | 18571 | 37.96±5.44 | 34.93±4.51 | 35.31±6.14 |
| **cfc_pulse ⚠N<5 (n=3)** | 27020 | 38.49±2.36 | 34.73±1.23 | **38.04±4.19** ⭐ |
| cfc_selfattend ⚠N<5 (n=3) | 27020 | 29.78±13.78 | 27.49±11.71 | 29.22±13.67 |
| full_pdna ⚠N<5 (n=3) | 27020 | 29.93±13.99 | 27.58±11.78 | 29.40±13.93 |

**vs baseline_cfc 的 delta**:

| Comparison | ΔGap 5% (pp) | ΔMulti-gap (pp) | Verdict |
|---|---:|---:|---|
| cfc_noise (control) | +0.80 | -0.20 | 🟰 mixed (排除 trivial 解释) |
| **cfc_pulse** | **+0.60** | **+2.53** | ✅ better |
| cfc_selfattend | -6.64 | -6.29 | ❌ worse (seed 42 stuck) |
| full_pdna | -6.56 | -6.11 | ❌ worse (seed 42 stuck) |

## 4. 关键解读(对照论文)

### 4.1 与论文 headline 对齐

- 论文: pulse multi-gap **92.86%** vs baseline **88.24%** = **+4.62 pp** (5/5 seeds)
- 本仓: pulse multi-gap **38.04%** vs baseline **35.51%** = **+2.53 pp** (3/3 seeds direction-consistent)
- 量级差异原因: hidden=64 vs 128, 8000 train vs 60k, 5 epochs vs 40 — **smoke scale**
- **方向一致(都是 +multi-gap)+ noise control 持平 + structural not dynamic** 三连信号一致

### 4.2 self-attend / full_pdna seed 42 stuck at 14% — **诚实记录失败模式**

- seed 42 对 cfc_selfattend 和 full_pdna 全部 stuck at 14% (slightly above 10% chance)
- seeds 1153/2264 正常: 35-40% range
- 怀疑: `β · Wself · σ(h)` 即使 β=0.01 init,Wself σ(h) 的 hidden state 偏移
  可能把 hidden state 推出 CfC manifold — **post-hoc gated residual 的训练不稳**
- 论文 §8 也观察到 non-additive composition,full PDNA 不严格优于 single
  component — 与本仓观察一致

### 4.3 论文 §8 (iii) post-hoc augmentation 限制复现确认

- CfC backbone 是 parallel over all timesteps,pulse 是 **post-hoc 加到完整 hidden
  state tensor**,不是真 continuous-time dynamic
- 论文承认这限制 + 预测 sequential ODE 会有更强结果
- 本仓 self-attend gated residual 的训练不稳 也支持这个判断

## 5. pytest 套件(58 tests)

```
tests/test_core.py + tests/test_liquid_tad_hierarchical.py + tests/test_pdna_pulse.py
58 passed, 1 warning in 15.36s
```

无回归(iter#19 加的 12 个 PDNA test + 既有 46 个全过)。

## 6. verify_all_models.py(9/9)

无变化。`scripts/` 改动不触碰任何 verify 路径。

## 7. backbone matrix 更新(PRD §9 #7 衍生)

```
scripts/build_backbone_matrix.py --include-molecular --include-smnist-gap
→ 6 rows, 9 backbones
→ smnist_gap [n=3,h=64] row: cfc_pulse ⭐ (38.04% multi-gap)
→ Win tally: lstm 3, cfc 2, cfc_pulse 1, others 0
```

详见 `analysis/backbone_matrix/2026-06-04_144554_backbone_matrix.md`。

## 8. 已知阻塞(无变化)

| 阻塞 | 来源 | 影响 |
|---|---|---|
| CUDA 不可用 | Jetson BSP driver 12060 < torch 2.11 cu130 | 较大 hidden + 5 seed 全跑要走 CPU |
| RAM 1.7 GB available | 多 agents 并行 + 8GB 统一显存 | LFM2.5-1.2B / 较大 hidden LNN sweep 受限 |
| THUMOS-14 数据未下载 | LiquidTAD stage C 真实数据 | 暂用 toy 长视频 |

本轮无新增阻塞。

## 9. 关键 takeaway

1. **cfc_pulse multi-gap +2.53 pp vs baseline,3/3 seed 方向一致** — paper 的
   "structural not dynamic" 现象在本仓小规模复现中**方向一致**(量级减半因 scale)
2. **self-attend gated head 在 seed 42 stuck** — post-hoc gated residual 训练不稳
   是个真实问题,与论文 §8 (iii) post-hoc augmentation 限制互相印证
3. **noise control 完全持平 baseline** — 复现 paper 的 "structural not dynamic"
   关键论据,**首次在本仓获得 LNN augmentation 的 +multi-gap 信号**
4. **backbone matrix 现跨 3 domain**: timeseries / molecular / smnist_gap
5. **仓库 LNN 通杀 thesis 在不同 domain 强度不同**: timeseries LSTM 通杀,
   graph_tox21 cfc 微弱, smnist_gap cfc_pulse 微弱 — **没有"通杀 backbone"**
