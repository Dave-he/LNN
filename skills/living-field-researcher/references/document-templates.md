# Document Templates

Use these compact templates when creating new artifacts from scratch. Adapt names to the repository's existing conventions.

## Daily Digest

```markdown
---
title: <DOMAIN> 每日研究追踪 - YYYY-MM-DD
date: YYYY-MM-DD
tags: [<domain-slug>, daily, automation]
---

# <DOMAIN> 每日研究追踪 - YYYY-MM-DD

> 自动生成或半自动生成：记录本次搜索范围、候选项和建议动作。

## 搜索范围

| 数据源 | 查询式/参数 | 候选数 | 备注 |
|---|---|---:|---|
|  |  |  |  |

## 高优先级候选

| 状态 | 类型 | 名称 | 理由 | 链接 |
|---|---|---|---|---|
| read_now | paper |  |  |  |
| repo_analyze | repo |  |  |  |
| experiment | benchmark/model |  |  |  |

## 全量候选

### Papers

| 日期 | 标题 | 作者 | 摘要 | 链接 |
|---|---|---|---|---|

### Repositories / Models

| 更新 | 名称 | 指标 | 说明 | 链接 |
|---|---|---:|---|---|

## 建议动作

-
```

## Paper Reading Report

```markdown
---
title: <PAPER_TITLE> 研读报告
date: YYYY-MM-DD
tags: [<domain-slug>, paper, reading-report]
---

# <PAPER_TITLE> 研读报告

## 元数据

- 标题：
- 作者：
- 时间 / venue：
- 链接：
- 标签：

## 核心问题

## 方法论与核心思路

## 核心公式提取

$$

$$

## 关键成果与贡献

## 局限性与未来展望

## 复现线索

- 代码：
- 数据：
- 指标：
- 依赖：
- 本仓库下一步：
```

## Repository Analysis

```markdown
---
title: <REPO_NAME> 代码调研
date: YYYY-MM-DD
tags: [<domain-slug>, repo-analysis]
---

# <REPO_NAME> 代码调研

## 元数据

- 仓库：
- Star / Fork：
- 主要语言：
- 最近更新：
- License：

## 项目定位

## 代码结构

## 环境与复现成本

## 与本仓库关系

## 风险与缺口

## 建议动作
```

## Concept Note

```markdown
---
title: <CONCEPT_AS_CLAIM>
date: YYYY-MM-DD
tags: [<domain-slug>, concept]
---

# <CONCEPT_AS_CLAIM>

## 核心观点

## 支撑证据

- [[<report-1>]]
- [[<report-2>]]

## 反例或限制

## 关联概念

- [[<neighbor-concept>]]

## 可验证问题

-
```

## Weekly Synthesis

```markdown
---
title: <DOMAIN> 周度研究综述 - YYYY-WW
date: YYYY-MM-DD
tags: [<domain-slug>, weekly, synthesis]
---

# <DOMAIN> 周度研究综述 - YYYY-WW

## 本周新增

## 关键变化

## 值得精读

## 值得复现

## 关键词与协议调整

## 下周动作
```
