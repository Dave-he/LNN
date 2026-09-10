---
date: 2026-09-10
source: cron daily LNN tracker
tags: [LNN, CfC, LTC, Neural-ODEs, arxiv, cron]
---

# 2026-09-10 LNN 每日论文追踪 — Cron 执行报告

## 1. 查询

- 查询 URL（按 cron 配置）：
  `http://export.arxiv.org/api/query?search_query=all:"Liquid Neural Networks" OR all:"CfC" OR all:"LTC" OR all:"Neural ODEs"&start=0&max_results=100&sortBy=submittedDate&sortOrder=descending`
- 实际下载：因本机 TUN/代理（`utun4`, 198.18.0.1）将 `http://export.arxiv.org` 的 301→HTTPS 重定向打断，curl 退码 16。改为直连 HTTPS 端点 `https://export.arxiv.org/api/query?...` 成功（236 KB XML）。
- 总条目：100；过去 7 天内：1。

## 2. 24 小时窗口筛选结果

| 时间窗口 | 条目数 |
|---|---|
| 最近 24h（UTC 2026-09-09 01:05 → 2026-09-10 01:05） | **1** |
| 最近 7 天（参考） | 1 |

### 候选条目（top 1，published 在 24h 内）

| arXiv ID | Published (UTC) | 标题 |
|---|---|---|
| 2609.10350v1 | 2026-09-09T15:46:16Z | Cyber-Financial Contagion: Modeling the Propagation of an AI Vendor Compromise Through the Banking System |

> ⚠️ **主题相关性提示**：标题与摘要（cs.AI / cs.CY / cs.LG；包含自定义方法名 `CFC-Prop`、`CFC-GNN`）表明这是一篇**网络金融传染**领域论文，仅因作者命名模型时使用了 "CFC" 字符串而被本次关键词查询回中，与 LNN/LTC/CfC/Neural ODE 主题**无关**。建议在后续 cron 逻辑中将查询关键词收紧到 `ti:` / `abs:` 字段或显式加入 `(Liquid OR CfC OR LTC OR "Neural ODE") AND ti:` 限定，避免误中子串匹配。

## 3. PDF 下载结果

| arXiv ID | 下载结果 | 说明 |
|---|---|---|
| 2609.10350v1 | ❌ 失败 | arXiv 返回 `File unavailable for 2609.10350v1`（HTML 错误页，非 PDF）。该 ID 出现在 API feed 但 PDF 端点尚未发布/已被撤稿。 |

下载命令：
```
curl -L --max-time 120 -A "Mozilla/5.0 (LNN-daily-cron)" \
  -o papers/daily/2609.10350v1.pdf \
  https://arxiv.org/pdf/2609.10350v1.pdf
```
返回 HTML（7668 字节，`<h1>File unavailable for 2609.10350v1</h1>`），已删除残留 HTML 文件，避免污染 `papers/daily/`。

## 4. Git 操作

- 状态：因本次**无成功新增 PDF**，按 cron 任务规范 **跳过 `git add / commit / push`**。
- 远程：默认分支为 `master`（非 `main`）。原 `git fetch origin` 失败（`Connection closed by UNKNOWN port 65535`），因为本机 TUN 拦截了到 `github.com:22` 的 SSH 连接 —— 即便有内容要提交，今天也无法 push。
- 工作区残留未跟踪文件 `uv.lock` 非本次任务产生，未纳入本次 commit（按 cron 任务说明只处理 `papers/daily/*.pdf`）。

## 5. 下一步建议

1. 收紧查询：将 `all:` 改为 `ti:` 或 `abs:`，或显式排除模型名误中（增加 `NOT "CFC-Prop" NOT "CFC-GNN"`）。
3. 增加 PDF 健康检查（`file <path>` 必须是 `PDF document`，否则删除）。
3. 当主分支非 `main` 时让 push 命令可配置（当前 cron 脚本 `scripts/run_daily_lnn_task.sh` 写死 `main`，建议改为 `git push origin HEAD` 或读取脚本配置）。
4. 在 cron 中加入 SSH 失败的兜底（fetch 失败时记录并跳过 push，而不是中断整体任务）。