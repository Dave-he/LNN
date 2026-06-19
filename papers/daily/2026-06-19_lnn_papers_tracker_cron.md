# 每日 LNN 论文追踪 — 2026-06-19

## 运行结果

**今日 (UTC 2026-06-19 01:20Z) 未发现最近 24 小时内新提交的 LNN / CfC / LTC / Neural ODEs 相关论文。**

## 执行明细

1. **工作目录**：`/Users/hyx/workspace/LNN` (分支: `master`，远程: `origin`)。
2. **数据源**：`https://export.arxiv.org/api/query?search_query=all:"Liquid Neural Networks" OR all:"CfC" OR all:"LTC" OR all:"Neural ODEs"&start=0&max_results=100&sortBy=submittedDate&sortOrder=descending`
   - 该 endpoint 在 2026-06-19 01:01Z ~ 01:18Z 期间因速率限制返回 `HTTP 429 / RemoteDisconnected` / 空响应，多次重试均失败；最终在 01:18Z 成功取回 226 KB Atom XML。
3. **筛选规则**：`<published>` 时间戳距今 ≤ 24h (cutoff = `2026-06-18T01:20:45Z`)。
4. **结果**：
   - API 返回 100 条候选（按 `published` 倒序）。
   - 最近的条目: `2606.19579` ("FlowFake: Liquid Networks for Audio Deepfake Detection")，`published = 2026-06-17T20:32:32Z`，距 cutoff **-4.8h**。
   - 严格符合 24h 窗口的条目数: **0**。

## 跳过提交说明

依据任务规则「若<发表日期>不在最近 24h 内则跳过下载与提交」，本次运行未执行 PDF 下载、未执行 `git commit`、未执行 `git push`。

## 最近 LNN 相关候选（供下个 cron 关注，已在外层 daily digest 中跟踪）

| arXiv ID | <published> (UTC) | 标题 | 距今 |
|---|---|---|---|
| 2606.19579 | 2026-06-17T20:32:32Z | FlowFake: Liquid Networks for Audio Deepfake Detection | ~28.8h 前 |
| 2606.19109 | 2026-06-17T14:23:41Z | Locally Stable Neural ODEs with Characterized Region of Attraction | ~35h 前 |
| 2606.18315 | 2026-06-16T11:23:30Z | Ghost Attractor Networks: Basin-Structured Dynamical Decoders for Closed-Loop Sequential Generation | ~62h 前 |

> 注：本仓库默认分支为 `master`（非 `main`），远程仓库 `origin → git@github.com:Dave-he/LNN.git`。