---
title: 2026-06-04 Loop iteration 16 — DynPMNN 研读 + PRD §10 第三波 backlog
date: 2026-06-04
tags: [LNN, loop, PRD-10, paper-report, DynPMNN, FitzHugh-Nagumo, RKBS]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 16 — DynPMNN 研读 + PRD §10 scoping

> `/loop 1h` 第 16 次触发。
> PRD §9 5/8 done,剩下 3 个全部硬阻塞(RAM / 数据 / CUDA-stable);
> 内部工程任务面已耗尽。本轮回到用户原话核心:
> **"搜索 LNN 相关论文代码 + 补充论文报告 + 把发现 scope 到 PRD"**:
>
> 1. **刷新今日 daily research**: arXiv 此时段仍 429 但仓库与 HF 抓到
>    25 论文 / 51 repo / **24 HF 模型**;
> 2. **挑 DynPMNN (arXiv 2605.08176v1) 做结构化研读** — 仓库现有
>    `lnn/core/physics.py` 与 `experiment_physics_lnn.py` 几乎对接 OK,
>    1 loop 可启动复现;
> 3. **PRD 新增 §10**: 8 个新候选任务,其中 #1/#2 直接对应 DynPMNN 复现。

## 1. 今日 daily research 刷新

```text
- digest: docs/daily/2026-06-04_LNN_research_digest.md
- data: papers/daily/2026-06-04_lnn_research.json
- repo watchlist: analysis/repo_watchlist/2026-06-04_lnn_open_source_watchlist.md
- papers/repos/models: 25/51/24
- arXiv API: 429 (keep previous result set)
- GitHub: fewer items this run, keep previous
```

新增 HF 模型 1 个 (vs iter#15 23 → 24);arXiv 仍是上一缓存。

## 2. DynPMNN — Physics-Modeled Neural Networks 研读

完整报告: [[Physics-Modeled_Neural_Networks_DynPMNN_研读报告]]

### 2.1 一句话总结

每个隐藏层定义为 **FitzHugh-Nagumo 神经元 ODE 的积分轨迹** 而不是
静态激活,**Euler-type schemes** 嵌进 PyTorch 计算图端到端可训,
理论建立在 Reproducing Kernel Banach Spaces 框架上。

### 2.2 与本仓 LNN 谱系的相对位置

| 工作 | 时间维度处理 | 函数形式 |
|---|---|---|
| 经典 MLP | 离散一步 | sigmoid/tanh 静态 |
| LTC (本仓 `lnn/core/ltc.py`) | 连续(可学习时间常数) | $-h/\tau + drive$ |
| CfC (本仓 `lnn/core/cfc.py`) | 连续(闭式解) | 解析公式 |
| Neural ODE | 连续(任意 $\dot h = f(h, t)$) | 学 $f$ |
| **DynPMNN (本轮研读)** | **连续(FHN 模型)** | **物理给定方程,只学参数** |

DynPMNN 与 LTC 思想最近,但**ODE 函数形式固定**:多了物理约束、少了灵活性。

### 2.3 复现可行性

| 维度 | 评估 |
|---|---|
| 代码可获取 | ❌ 无公开实现 |
| 数据规模 | 论文只 California Housing(过窄) |
| 计算复杂度 | Euler 几步,接近 LTC 量级 |
| 本仓对接 | ✅ `lnn/core/physics.py::PhysicsInformedLNN` + `experiment_physics_lnn.py` 几乎天然对接 |
| 估时 | A 实现 + 单测 1 loop;B 接 ablation 1 loop;C smoke + 报告 1-2 loop |

→ 入 **PRD §10 #1 / #2**。

## 3. PRD §10 — 第三波候选 backlog

| # | 任务 | 关键产出 |
|---:|---|---|
| **10-1** | DynPMNN 复现 stage A | `lnn/core/dynpmnn.py::FHNCell + DynPMNNNetwork` + unit test |
| **10-2** | DynPMNN stage B | ablation runner 加 `--backbone fhn_dynpmnn`,matrix 新列 |
| 10-3 | Comparative LNN vs LSTM phase-D(hidden=64+ep=50+samples=4000) | 看规模是否扭转 iter#11 negative |
| 10-4 | `experiment_graph_lnn_molecule.py` 加 `HierarchicalDecayLiquidTADHead` recurrent 选项 | 交叉 PRD §2 与 §6 |
| 10-5 | `loop_status.py --prd-status` 子模式 | 解析 §8/§9/§10 表出未完成 + 阻塞理由 |
| 10-6 | `build_backbone_matrix.py --export-readme-snippet` | 自动产 README 顶部 badge 行 |
| 10-7 | LFM2.5-1.2B-Distilled-SFT INT8 推理(等空载窗口) | §9 #1 / §8 #3 |
| 10-8 | `analysis/loop_status/` 自动产 README 标签云 | 高频 task / 高方差 seed 提示 |

并把 DynPMNN 加进 **PRD §9 "已调研未复现"C 级表**,
理由:无公开代码,需自行复现。

### 3.1 与 §8 / §9 的衔接

- §8: 12 个 / 8 个 ✅ + 2 个真实阻塞
- §9: 8 个 / 5 个 ✅ + 3 个真实阻塞
- §10: 8 个新候选,其中 #1/#2 直接 unblock #9 类 "LNN claim 验证" 死结
  — DynPMNN 加进 backbone matrix 可能给我们 11 轮 loop 来都没拿到的
  "LNN backbone 跨 N=5 seed 稳定赢 LSTM" 信号(也可能再次负面)。

## 4. 衍生

| 任务 | 推入 |
|---|---|
| 把 DynPMNN 加到 README LNN 谱系图 | docs / iter#17 顺手 |
| 跑一次 `--prd-status` 看 §10 进展(待 #10-5 完成) | PRD §10 自循环 |
| 给 §10 #1-#2 立即排时间窗,iter#17 开始 | NEXT_STEPS |

## 5. 参考产物

- 新论文研读: [[Physics-Modeled_Neural_Networks_DynPMNN_研读报告]] (~150 行)
- PRD §10 新增: 8 个候选 + DynPMNN 入 C 级表
- 上一轮: [[2026-06-04_loop_iteration15_weekly_verify_ci]]
- 累积 iter chain:
  - iter#15 weekly CI
  - iter#14 since-last-loop
  - iter#13 frozen-encoder
  - iter#12 backbone matrix
  - iter#11 retraction
  - ...
- 数据源: `docs/daily/2026-06-04_LNN_research_digest.md` (25 papers / 51 repos / 24 HF)
- PRD: [[PRD_LNN_Edge_Research]] §10 #1 / #2
