---
title: 2026-06-04 Loop iteration 14 — loop_status --since-last-loop (PRD §9 #5)
date: 2026-06-04
tags: [LNN, loop, PRD-9, meta-tooling, dedup, automation]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 14 — `loop_status.py --since-last-loop` (PRD §9 #5)

> `/loop 1h` 第 14 次触发。
> iter#8 写了 `loop_status.py` 的 single-day 视图,iter#12 加了 `--week N`
> 周报,本轮补齐第三种视图: **`--since-last-loop`**,直接定位"最近一次
> iteration 报告之后发生了什么"。这是 /loop 调度时最常需要的视图 —
> 比 `--date today` 更精准(不会被早晨的 systemd 跑混淆),
> 比 `--week 3` 更聚焦(只看到上一个 loop 收尾后的增量)。

## 1. 工具能力

`scripts/loop_status.py --since-last-loop` 干 4 件事:

1. **找 marker**: 扫 `analysis/{jetson,molecular,long_sequence,timeseries_ablation,loop_status,backbone_matrix}/` 下所有
   `YYYY-MM-DD_loop_iteration\d+_*.md`,取 mtime 最新的那一份;
2. **找 commits**: `git log --since=<marker_mtime> --pretty=format:%h|%an|%ad|%s` 列出
   marker 之后的所有本地 commit;
3. **找文件变更**: 在 `analysis/ docs/ scripts/ lnn/ tests/ papers/` 下 walk,
   收集 mtime > marker_mtime 的文件;
4. **找新 iteration 报告**: filter 3 的结果,匹配 `loop_iteration\d+_.+\.md`
   (确认本 loop 是否已经写出了新的 marker);
5. **给建议**: 三种情况
   - `nothing changed since marker` → 可能是 stale marker
   - `N new iteration report(s)` → 已经写了,可以推送
   - `N commits + M file changes, no new iter report yet` → 写 iter 报告再 push。

输出 JSON + Markdown 报告到 `analysis/loop_status/`,
另外 stdout 一屏摘要供下次 /loop 第一行 bash 读。

## 2. 首跑结果(在 iter#13 之后 58 min 触发)

```text
=== Loop status — since last iteration marker ===
  marker: analysis/molecular/2026-06-04_loop_iteration13_frozen_encoder_ablation.md
  elapsed: 58 min
  commits since: 2          ← e7f38a4 (本仓 iter#13) + 85fbd24 (远程 EMMA agent)
  files modified: 5
  new iteration reports: 0
  suggestion: write a new iteration report before pushing
```

文件变更明细:
- `docs/PRD_LNN_Edge_Research.md` ← iter#13 PRD §9 #4 标记
- `scripts/loop_status.py` ← 本轮新增 --since-last-loop
- `analysis/emma_rover/2026-06-04_083748_phase2_10seed.json` ← 远程 EMMA agent
- `docs/research/2026-06-04_phase2_10seed_report.md` ← 远程 EMMA agent
- `scripts/probe_phase2_10seed.py` ← 远程 EMMA agent

**工具自动区分了本仓 commit 与远程 EMMA agent commit** — 不需要人盯 git log。

## 3. 设计要点

### 3.1 marker 必须是 mtime,不能是 filename 里的日期

iteration 报告文件名里的日期是 *report* 日期(可能批量补写),
而我们要的是 *该 iter 真正结束* 的时间。
`stat.st_mtime` 一定是写文件那一刻,
比 filename 解析靠谱。

### 3.2 walk 多个根而不只是 analysis/

iter#13 改了 `docs/PRD_LNN_Edge_Research.md` 和 `scripts/experiment_graph_lnn_molecule.py`,
如果只 walk `analysis/` 会漏掉这两条关键变更。
所以工具默认扫 `analysis/ + docs/ + scripts/ + lnn/ + tests/ + papers/`。

### 3.3 顶部 60 行截断 + 总数提示

mtime 改动文件很容易超 50(尤其 `analysis/repo_watchlist/*` 每天刷新),
MD 表头显式列总数 + "...(+N more)" 注释,
完整列表在 JSON 里。

### 3.4 fallback: 没找到 marker → 退到今日 00:00

第一次跑 /loop 或仓库刚 clone 时 `analysis/` 是空的;
不应该让工具 crash,而是 fallback 到当日午夜,
给出"没找到 marker,从今天 0 点开始数"的视图。

## 4. 三种视图对比表

| 视图 | trigger | 用途 |
|---|---|---|
| **single-day** | 默认 / `--date YYYY-MM-DD` | 看某一天的所有产物 / 决定 PRD 下一任务 |
| **weekly retro** | `--week N` | 周报 / 跨日聚合 |
| **since-last-loop** ⭐ | `--since-last-loop` | /loop 触发时第一行 bash;只看上次 iter 之后的增量 |

iter#8 + iter#12 + iter#14 三轮把 `loop_status.py` 从 ~300 行扩到
**~720 行**,covers 三种"什么时候"切片。

## 5. PRD §9 进展

| # | 状态 | 备注 |
|---:|:---:|---|
| 9-1 | ⏳ | LFM2.5 — 等 RAM(本时段 available 1.7 GB,仍不足) |
| 9-2 | ✅ iter#10/#11 | gradual + warmup + 8-seed |
| 9-3 | ⏳ | LiquidTAD Stage C-true(待 THUMOS-14 数据) |
| 9-4 | ✅ iter#13 | frozen-encoder 量化 GCN-CfC 两阶段 −5% AUC |
| **9-5** | **✅ iter#14** | **`loop_status --since-last-loop`** |
| 9-6 | ⏳ | ONNX + TensorRT INT8(待 CUDA 稳定 + RAM) |
| 9-7 | ✅ iter#12 | backbone matrix |
| 9-8 | ⏳ | weekly CI |

§9 完成度 **4/8**(50%),3 个真实阻塞(#1 RAM, #3 数据, #6 RAM+CUDA),
1 个未启动(#8 CI 需 GitHub Actions 配置)。

## 6. 衍生

| 任务 | 推入 |
|---|---|
| 加 `--remote` 模式,fetch + 看远程 commits(目前只看本地) | NEXT_STEPS |
| 加 `--diff-stat` 模式,对每个文件给 +/- 行数 | NEXT_STEPS |
| 把"3 视图入口"写进 README 顶部 quickstart | docs |
| pre-commit hook: 写 iter 报告前自动跑一次 `--since-last-loop` 给读者参考 | NEXT_STEPS |

## 7. 参考产物

- 代码增量: `scripts/loop_status.py` (+ ~190 行 `--since-last-loop` 路径)
- 首跑 JSON+MD: `analysis/loop_status/2026-06-04_093200_loop_status_since_last.{json,md}`
- 上一轮: [[2026-06-04_loop_iteration13_frozen_encoder_ablation]]
- 同一工具的两个早先视图: [[2026-06-04_loop_iteration8_loop_status_tooling]] (single-day) /
  iter#12 的 `--week`
- PRD: [[PRD_LNN_Edge_Research]] §9 #5
