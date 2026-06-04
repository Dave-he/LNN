---
title: 2026-06-04 Loop iteration 18 — PDNA 研读 + PRD §10 #10 落地
date: 2026-06-04
tags: [LNN, loop, PRD-10, paper-report, PDNA, CfC, pulse-module, gap-robustness, sMNIST]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 18 — PDNA 研读 + PRD §10 #10 落地

> `/loop 1h` 第 18 次触发。
> 紧接 iter#17 (SVAF) 后,本轮挑 daily digest 缓存里**唯一直接 build on CfC**
> 且**代码公开**的候选: PDNA (Pulse-Driven Neural Architecture, arXiv 2603.00153)。
>
> 1. **Daily research 复用**: arXiv 仍 429,GitHub 仍 403(本日所有限流),缓存保留
> 25/51/24
> 2. **挑 PDNA 做深读** — Paras Sharma 2026-02 single-author,13 页,
>    **代码公开** (github.com/Parassharmaa/pdna),**MNIST 零数据成本**
> 3. **PRD §10 加 #10** (P1 优先级): stage A 25 行 PDNAPulseHead + unit test,
>    stage B sMNIST Gapped protocol 5 seed × 4 backbone ablation, stage C LRA 长程
> 4. **Jetson 验证零回归**: 9/9 verify + 46/46 pytest
> 5. **commit + rebase + push origin/master**

## 1. Daily research 状态

```text
- digest: docs/daily/2026-06-04_LNN_research_digest.md (复用 iter#16 缓存)
- arXiv API: 429 (no new)
- GitHub API: 403 (rate limit, keep previous)
- HF API: 24 models (cached)
```

## 2. PDNA 论文深读

完整报告: [[Pulse-Driven_Neural_Architecture_PDNA_研读报告]]
(264 行, 12 节)

### 2.1 一句话定位

> 在 **CfC backbone** 之上加两个 gated residual 模块:
> (1) **pulse** `α · A · sin(ωt + φ(h))`,
> (2) **self-attend** `β · Wself · σ(h)`.
> 在 sequential MNIST + Gapped evaluation protocol 上,
> 5 seed ablation 证明 pulse variant 在 multi-gap 下 acc +4.62 pp
> (Cohen's d=0.87),**结构性优势**(noise control 输 baseline)。

### 2.2 核心数据点

- **任务**: Sequential MNIST(28 timesteps × 28 features)
- **架构**: CfC backbone (hidden=128) + 2 个 gated additive residual
- **训练**: 5 seed × 5 变体 = 25 runs on RTX A4000 16GB
- **α 训练动力学**: 0.01 → **~0.66**(主动利用)
- **ω 频谱**: 学到范围 **[0.06, 10.02]**,跨两个数量级
- **计算开销**: +38% params, **+5% wall-time**
- **Multi-gap 精度**:
  - A. Baseline CfC: **88.24%**
  - B. CfC + Noise: 88.01%(输)
  - **C. CfC + Pulse: 92.86%** (+4.62 pp vs A, **5/5 seeds**)
  - D. CfC + SelfAttend: 91.02%
  - E. Full PDNA: 91.96%(**非 additively superior**)
- **统计检验**: paired t-test, pulse vs noise gap-5% **p=0.013** 显著

### 2.3 5 变体 ablation(论文 Table 1)

| Variant | Pulse | Self-Attend | 用途 |
|---|---|---|---|
| A | ✗ | ✗ | Baseline CfC |
| B | random | ✗ | **Noise control**(matched magnitude) |
| C | ✓ | ✗ | 单独的 oscillation |
| D | ✗ | ✓ | 单独的 self-attention |
| E | ✓ | ✓ | Full PDNA |

**B 是 critical control** — 排除"任何非零扰动就够"的 trivial 解释。

### 2.4 与本仓的契合度

| 维度 | 评估 |
|---|---|
| 算法复用 | CfC backbone 完全复用;**pulse + self-attend ~25 行核心代码** |
| 数据可获得性 | **MNIST torchvision,零数据成本** |
| Jetson 部署 | sMNIST + hidden=128 — 单 A4000 16GB 论文可跑,CPU 路径 hidden=64 + 5 seed × 3 backbone 估计 2-4 hr |
| 代码可用性 | **公开** — github.com/Parassharmaa/pdna |
| 统计严谨度 | 5 seed + paired t-test + Cohen's d + 5/5 win rate(虽然 n=5 偏小) |
| 复现优先级 | **P1**(代码公开 + 数据零成本 + 与本仓 LNN/CfC 复用度高) |

## 3. PRD §10 扩展

```
| 10-10 | PDNA (arXiv 2603.00153) PulseHead + Gapped protocol 复现 |
        | stage A: lnn/core/cfc_cell.py::PDNAPulseHead (~25 行) + unit test |
        | stage B: sMNIST Gapped protocol 5 seed × 4 backbone ablation |
        | stage C: Long Range Arena 长程任务 |
        | 状态: pending (P1) — 代码公开 + MNIST 零成本 + 复用度高 |
```

§10 完成度: **0/9 → 0/10**(新增 #10 但 pending)
C 级"已调研未复现"表新增 1 行(PDNA)。

## 4. Jetson 验证(0 回归)

```
verify_all_models.py    : 9/9 ✅
pytest test_core+test_liquid_tad_hierarchical : 46/46 ✅
```

详见 [[2026-06-04_loop_iteration7_pdna_paper_deep_read]] (analysis/jetson/)。

## 5. 提交与推送

iter#18 改动:
- 新增 `docs/reports/Pulse-Driven_Neural_Architecture_PDNA_研读报告.md` (264 行)
- 修订 `docs/PRD_LNN_Edge_Research.md` (§10 加 #10 P1, C 级表加 PDNA 行)
- 新增 `analysis/jetson/2026-06-04_loop_iteration7_pdna_paper_deep_read.md` (验证摘要)
- 新增 `analysis/loop_status/2026-06-04_loop_iteration18_pdna_paper_and_prd10_10.md` (本报告)

**PRD §6 verify 协议执行**: ✅ (验证 0 回归,见 §4)
**PRD §9 状态**: 5/8 = 62.5%(无变化)
**PRD §10 状态**: 0/10 = 0%(新增 #10 但仍 pending)

## 6. 下轮 (iter#19) 候选

按 PRD §10 P 排序 + 阻塞条件:

1. **§10 #10 PDNA stage A** (PDNAPulseHead + unit test): **无阻塞,本周可启动** —
   25 行核心代码 + 5 个 unit test,预估 1-2 hr 工作量
2. **§10 #5 (loop_status --prd-status)**: 无阻塞,可与 #10 并行
3. **§10 #8 (loop_status README 标签云)**: 无阻塞,衍生 tooling
4. **§10 #9 (SVAF τ 调制算子)**: 无阻塞但 P2
5. §10 #6 (backbone matrix --export-readme-snippet): 无阻塞
6. §10 #3 (Comparative phase-D): hidden=64 需求高 RAM,需空载窗口
7. §10 #1/#2 DynPMNN 复现:与 #10 PDNA 互不冲突,但建议 #10 跑通后再开
