---
title: 2026-06-04 Loop iteration 8 — Loop coverage tooling (PRD §8 #8)
date: 2026-06-04
tags: [LNN, loop, meta-tooling, PRD, automation, dedup]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 8 — Loop coverage tooling (PRD §8 #8)

> `/loop 1h` 第 8 次触发。完成 PRD §8 #8 "Loop 调度产物去重 + 自动 retro" 的
> 落地: `scripts/loop_status.py` 一个 ~300 行的元工具,
> 把过去 7 轮 loop 我每次都重新做的"先扫一遍今天已经有什么产物"动作固化下来,
> 下一轮 loop 第一行 bash 就能拿到全景。
>
> **首跑就揪出实战 bug**: PRD §8 #7 (Pareto sweep) iter#2 已做但 PRD 表
> 没标 ✅,跟踪状态漂移。这是工具的第一个真实价值。

## 1. 工具能力

`scripts/loop_status.py [--date YYYY-MM-DD] [--no-write] [--json]`

扫描以下来源(按 `{date}` 前缀文件名匹配):

| 来源 | 含义 |
|---|---|
| `docs/daily/{date}_LNN_research_digest.md` | 今日论文/仓库/HF 数字 digest 是否就位 |
| `papers/daily/{date}_lnn_research.json` | digest 对应的结构化数据 |
| `analysis/repo_watchlist/{date}_lnn_open_source_watchlist.md` | 当日开源仓库扫描 |
| `analysis/jetson/{date}_lnn_benchmark.{json,md}` | 当日 Jetson smoke benchmark |
| `analysis/{jetson,molecular,long_sequence,timeseries_ablation}/{date}_loop_iteration*_*.md` | iteration 报告(按编号排序) |
| `analysis/{...}/{date}_*.{json,md}` | 其他今日 analysis 产物 |
| `git log --since/--until` | 当日本地 commits |
| `docs/PRD_LNN_Edge_Research.md` §8 表 | PRD task 完成状态(关键词 ✅ / loop# / 完成 等) |

输出: 一份 JSON + 一份 Markdown 报告写到 `analysis/loop_status/`,
另外 stdout 打印一屏摘要(含"下一步建议")。

## 2. 设计要点

### 2.1 默认 "pending",显式才 "completed"

PRD §8 状态判断: 没有完成标记的行默认 pending,有 ✅ / `loop#N` /
`完成 ✅` / `DONE` / `已 ✅` / `done.` / `完成]` 等任一标记才算 completed。
这条选择是**安全优先**: 过早标 completed 会让我们漏掉 follow-up,
过晚标只是多读一次 PRD,代价低。

### 2.2 文件名约定就是 ground truth

iteration 报告检测靠 `{date}_loop_iteration(\d+)_.+\.md` 正则,
跨多个 analysis 子目录扫描。这是过去 7 轮 loop 自然形成的命名习惯,
不强制就立刻失效 — 工具用注释提醒"改名会漏"。

### 2.3 git 只看本地

`git log --since` 只看 local repo,远程 EMMA agent 的提交需手动 fetch。
工具在 §4 注释里显式说明,避免给"今天什么都没做"的错觉。

## 3. 首跑揪出的 tracking 漏洞

第一次跑(parser bug fix 之前 + 之后)输出对比:

| 跑次 | 报告的 pending | 真实 pending | 备注 |
|---|---:|---:|---|
| 初次 (parser too lenient) | 0 / 8 | 4 | "loop#" / "完成" 关键字误匹配 |
| 修正后 (default-to-pending) | 4 / 8 | 2 | #7 / #8 在 PRD 表里没标 ✅ |
| 补完 PRD §7+§8 ✅ 后 | **2 / 8** | **2** | #3 (LFM2.5) + #4 (EMMA),都真实阻塞 |

**两个有价值的发现**:
1. **过去 7 轮 loop 我在 iter#2 完成了 PRD §8 #7 Pareto sweep,但忘了回头打 ✅** —
   工具揪出了这条沉默漂移。直接补 PRD 行。
2. **当前 8 个 PRD 任务里只剩 2 个真正 pending**:
   - #3 LFM2.5-1.2B INT4 推理 — 阻塞于 Jetson 共享显存 RAM ≥ 2GB 空载窗口;
   - #4 EMMA 多模态 — 已由远程 EMMA agent 负责(避让冲突)。
   → 下次 loop 应该考虑 **PRD §9 / 扩展现有 iteration**,而不是死磕 #3/#4。

## 4. 今日(2026-06-04)状态全景(由工具输出)

```text
=== Loop status 2026-06-04 ===
  fixed artefacts: 5/5 present
  iteration reports today: 3       (#5 GCN-CfC survey, #6 graph_lnn Tox21, #7 LNN-vs-LSTM v2)
  other analysis files today: 11
  local git commits today: 7
  PRD §8 pending: 2 / 8            (#3 LFM2.5, #4 EMMA)
```

## 5. PRD §8 任务全景(本轮终态)

| # | 状态 | 出口物 / 备注 |
|---:|:---:|---|
| 1 | ✅ iter#2 | `scripts/jetson_cuda_env.sh` + libcudss |
| 2 | ✅ iter#3/#4 | LiquidTAD A+B+C-lite |
| 3 | ⏳ pending | LFM2.5-1.2B INT4 — 等 RAM 空载窗口 |
| 4 | ⏳ pending | EMMA — 远程 agent 负责 |
| 5 | ✅ iter#7 | Comparative LNN vs LSTM v2(诚实负面信号) |
| 6 | ✅ iter#5/#6 | GCN-CfC 调研 + Tox21-styled smoke |
| 7 | ✅ iter#2 | Pareto sweep (**本轮补上 PRD ✅ 漂移**) |
| 8 | ✅ iter#8 | **本轮** — `scripts/loop_status.py` |

## 6. 衍生工作

| 任务 | 推入 |
|---|---|
| `--week N` 模式: 扫过去 N 天,跨日聚合 | NEXT_STEPS |
| Skill 模式: 把 `loop_status` 包成 Vercel Skills,跨工具可用 | NEXT_STEPS |
| Pre-loop hook: 让 `scripts/run_daily_lnn_task.sh` 在最前先 `loop_status.py --json` | PRD §10 (新) |
| 把 "已调研但不复现" 加进 PRD §9 新表 | PRD §9 |

## 7. 参考产物

- 源代码: `scripts/loop_status.py` (本轮新增, ~300 行)
- 首跑报告: `analysis/loop_status/2026-06-04_033255_loop_status_2026-06-04.{json,md}`
- 直接受益人: 下一次 `/loop` 触发的我自己
- 累积 iteration 报告(本工具扫描的对象):
  - [[2026-06-03_loop_validation_summary]]
  - [[2026-06-03_loop_iteration2_cuda_fix_pareto]]
  - [[2026-06-03_loop_iteration3_liquid_tad_stage_ab]]
  - [[2026-06-03_loop_iteration4_liquid_tad_3way_ablation]]
  - [[2026-06-04_loop_iteration5_gcn_cfc_repo_survey]]
  - [[2026-06-04_loop_iteration6_graph_lnn_tox21_smoke]]
  - [[2026-06-04_loop_iteration7_lnn_vs_lstm_v2]]
- PRD: [[PRD_LNN_Edge_Research]]
