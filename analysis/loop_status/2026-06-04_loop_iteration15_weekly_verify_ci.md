---
title: 2026-06-04 Loop iteration 15 — weekly CI workflow (PRD §9 #8)
date: 2026-06-04
tags: [LNN, loop, PRD-9, CI, github-actions, verification, automation]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 15 — `.github/workflows/lnn_weekly_verify.yml`

> `/loop 1h` 第 15 次触发。
> 完成 PRD §9 #8 weekly CI — 把 PRD §6 的 5 条验证指标固化成
> GitHub Actions 工作流,每周一 03:07 UTC 自动跑 + 手动触发。
> **每周一上班前** repo 是否 releasable 自动看 commit status,
> 不再靠记忆。
>
> 同时,本轮在本地预演 5 个 CI step,**全绿** (53 tests / 9 variants /
> tiny ablation 3 trials / matrix rebuild)。

## 1. workflow 设计

文件: `.github/workflows/lnn_weekly_verify.yml`,与已存在的
`daily-lnn-research.yml`(每日抓 arXiv/GH/HF)互补:

| 维度 | daily-lnn-research.yml | lnn_weekly_verify.yml(本轮) |
|---|---|---|
| 频率 | 22:30 UTC 每日 | **03:07 UTC 周一** |
| 任务 | 资料抓取 + push 到 master | **代码验证 + 上传 artifact 不 push** |
| 出口 | docs/daily/, papers/daily/, analysis/repo_watchlist/ | pytest/verify_all_models/quick_validate/tiny ablation/matrix rebuild |
| 失败影响 | 当天 digest 缺一份 | **PRD §6 红色徽章 — 仓库不应 release** |

### 1.1 12 个 step

1. checkout (depth=1)
2. setup-python 3.11 + pip cache
3. `pip install -e ".[dev]"` (pytest + ruff + ipython + jupyter)
4. env forensics (`python -V`, `pip show torch torchdiffeq`)
5. **`pytest`** `tests/test_core.py + test_paper_models.py + test_liquid_tad_hierarchical.py`
6. **`verify_all_models.py`** (9 LNN 变体一致性)
7. **`quick_validate_implement.py`** (同 9 变体,前向 timing)
8. (optional) tiny **ablation_lnn_vs_lstm_timeseries.py** smoke (~ 2 min CPU)
9. (optional) **`build_backbone_matrix.py --include-molecular`** 更新矩阵
10. **`loop_status.py`** weekly + since-last-loop JSON dump
11. bundle artefacts to /tmp/lnn_weekly_artifacts/
12. `actions/upload-artifact@v4` (retention 14 天)

### 1.2 设计原则

| 原则 | 落实 |
|---|---|
| **不污染 master** | artifact upload,不 commit;失败时 master 仍干净 |
| **wallclock < 10 min** | 实测本地总和 < 90s;预留 10 min timeout 应对 ubuntu-latest 抖动 |
| **off-:00 cron** | `7 3 * * 1` 而非 `0 3 * * 1`,RTK fleet 友好(参考 RTK.md 约定) |
| **手动 dispatch + 可关 ablation** | `workflow_dispatch.inputs.include_ablation` 让排错时跳过 expensive 步骤 |
| **PRD §6 显式 surface** | `summary` job 把 verify 结果转成 `::notice` / `::error`,GitHub UI 直接看红绿 |

## 2. 本地预演 5 个 step

| step | 输出 | wallclock |
|---|---|---:|
| pytest 三套 | **53 passed** in 18.14s (paper_models 加进来后从 46 涨到 53) | 18 s |
| verify_all_models | 9/9 ✓ | < 30 s |
| quick_validate_implement | 9/9 ✓ | < 30 s |
| tiny ablation (cfc/gru/lstm × seed 42, samples=400/h=16/ep=4) | 3 trials,~6s | 6 s |
| build_backbone_matrix --include-molecular | 矩阵更新:新 `mackey_glass [h=16,r=4]` 行出现,**graph_tox21 n_seeds 从 3 涨到 6**(iter#13 frozen 数据自动吸收) | < 2 s |

总和 **< 90 秒**,远低于 ubuntu-latest 的 10 min budget。

### 2.1 矩阵自动更新发现

本轮 ablation 跑完后立刻 rebuild 矩阵,出现了**新的行**:

```text
mackey_glass [h=24]                       (n=3)  winner: lstm   ← iter#7
mackey_glass [h=16,r=4]                   (n=1)  winner: cfc    ← 本轮 CI smoke
concept_drift [h=24]                      (n=3)  winner: lstm   ← iter#9
gradual_multi_regime [warmup=0.1,h=24,r=4](n=8)  winner: lstm   ← iter#11
graph_tox21 [seeds:6]                     (n=6)  winner: cfc    ← iter#6 e2e + iter#13 frozen 合并
```

注意 `mackey_glass [h=16,r=4]` 是单 seed(n=1)smoke,**CfC 赢** —
但只 1 seed 不算数(matrix 工具的解读模板已经强调这点)。
重要的是矩阵自动捕捉了 hidden_size 维度的细分,
**未来 CI 每次跑就在矩阵里多一行新 config**。

## 3. PRD §9 进展

| # | 状态 | 备注 |
|---:|:---:|---|
| 9-1 | ⏳ | LFM2.5 — 等 RAM(本时段 available 仍约 1.7 GB) |
| 9-2 | ✅ iter#10/#11 | gradual + warmup + 8-seed |
| 9-3 | ⏳ | LiquidTAD Stage C-true(待 THUMOS-14 数据) |
| 9-4 | ✅ iter#13 | frozen-encoder −5% AUC |
| 9-5 | ✅ iter#14 | loop_status --since-last-loop |
| 9-6 | ⏳ | ONNX + TensorRT INT8(待 CUDA 稳定) |
| 9-7 | ✅ iter#12 | backbone matrix |
| **9-8** | **✅ iter#15** | **lnn_weekly_verify.yml** |

§9 完成度 **5/8 = 62.5%**,剩 3 个真实硬阻塞(全部要么 RAM 要么外部数据)。
本轮也是 §9 里**最后一个无外部依赖的任务**,
说明仓库已经把所有可控范围内的任务做完了。

## 4. 衍生

| 任务 | 推入 |
|---|---|
| Add ruff lint step(代码格式守门) | NEXT_STEPS |
| 把 weekly_verify 的 README badge 加进 README.md 顶部 | docs |
| 给 `lnn_weekly_verify.yml` 加 `concurrency: weekly-verify-${{ github.ref }}` cancel-in-progress | NEXT_STEPS |
| weekly verify 失败时自动开 issue | NEXT_STEPS |
| 把 matrix MD post 到 GitHub release notes | NEXT_STEPS |

## 5. 参考产物

- 新 workflow: `.github/workflows/lnn_weekly_verify.yml` (~120 行)
- 本轮 CI 在本地预演产物: `analysis/timeseries_ablation/2026-06-04_103221_lnn_vs_lstm.{json,md}` +
  `analysis/backbone_matrix/2026-06-04_103223_backbone_matrix.{json,md}`
- 上一轮: [[2026-06-04_loop_iteration14_since_last_loop]]
- 上一个 CI workflow: `.github/workflows/daily-lnn-research.yml`
- PRD: [[PRD_LNN_Edge_Research]] §6 (验证指标) + §9 #8
- 本仓 5 个验证脚本(被 CI 调起):
  - `scripts/verify_all_models.py`
  - `scripts/quick_validate_implement.py`
  - `scripts/ablation_lnn_vs_lstm_timeseries.py`
  - `scripts/build_backbone_matrix.py`
  - `scripts/loop_status.py`
