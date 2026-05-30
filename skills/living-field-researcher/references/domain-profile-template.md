# Domain Research Protocol Template

Copy or adapt this into `docs/<DOMAIN>_研究协议.md` when a repository lacks an explicit protocol.

```markdown
---
title: <DOMAIN> 持续研究协议
date: YYYY-MM-DD
tags: [research-protocol, living-review, <domain-slug>]
---

# <DOMAIN> 持续研究协议

## 1. 范围与研究问题

- 领域名称：
- 领域 slug：
- 当前阶段：
- 核心研究问题：
- 暂不纳入范围：

## 2. 关键词与查询式

| 组别 | 关键词 / 查询式 | 用途 | 噪声风险 |
|---|---|---|---|
| 核心概念 |  | 发现主线论文 |  |
| 方法别名 |  | 覆盖同义术语 |  |
| 应用场景 |  | 发现落地论文 |  |
| 代码/模型 |  | 发现仓库和模型 |  |

## 3. 数据源

| 数据源 | 访问方式 | 频率 | 输出位置 | 备注 |
|---|---|---|---|---|
| arXiv | API / search | daily | `papers/daily/` |  |
| GitHub | Search API | daily/weekly | `analysis/repo_watchlist/` |  |
| Hugging Face | API / search | daily/weekly | `analysis/repo_watchlist/` |  |
| 其他 |  |  |  |  |

## 4. 纳入与排除规则

纳入：
-

排除：
-

## 5. 评分与优先级

| 信号 | 说明 | 权重 |
|---|---|---:|
| 主题相关性 | 标题/摘要/标签是否命中核心概念 |  |
| 新颖性 | 是否提出新方法、新数据、新 benchmark 或新理论 |  |
| 影响潜力 | 作者/机构、引用、Star、下载、社区讨论 |  |
| 可复现性 | 是否有代码、数据、明确指标 |  |
| 项目契合度 | 是否能进入当前 repo 的报告、代码或实验队列 |  |

优先级标签：
- `read_now`：
- `watch`：
- `repo_analyze`：
- `experiment`：
- `ignore`：

## 6. 输出约定

- 每日摘要：
- 原始数据：
- 单篇研读报告：
- 全局索引：
- 开源观察：
- 实验结果：

## 7. 迭代节奏

- 每日：
- 每周：
- 每月：

## 8. 当前种子集合

| 类型 | 标题/名称 | 链接 | 重要性 | 后续动作 |
|---|---|---|---|---|
| paper |  |  |  |  |
| repo |  |  |  |  |
| dataset/model |  |  |  |  |
```
