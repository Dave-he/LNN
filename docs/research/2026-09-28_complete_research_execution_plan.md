---
title: LNN 完整研究执行计划 (2026-09-28)
date: 2026-09-28
tags: [LNN, research-plan, daily-automation, paper-coverage, implementation-path]
status: active
---

# LNN 完整研究执行计划 (2026-09-28)

> 本文档是回应 user 2026-09-28 mid-task goal "研读每一篇论文，把所有论文都下载全，完整研究" 的执行契约。

## 0. 项目定位 (沿用 [[docs/LNN_深度研读报告]] §0)

**LNN 不是 LLM 替代品，是时间归纳偏置组件**。完整研究必须遵守的边界：

1. 沿用 [[AGENTS]] §"Agent 约束"：8 条不可重复 claim 必须遵守
2. 沿用 2026-06-09 用户偏好 critical 级：**不操控设备**，仅合成数据 + in-house 模型
3. 实现路径必须 grounding 到本仓 `lnn/core/`、`lnn/data/` 现有代码
4. 数字声明必须 grounding 到 PDF 全文，否则标"待 grounding"

## 1. 当前覆盖度盘点

### 1.1 PDF ↔ 研读报告覆盖度 (2026-09-28 11:16 CST)

| 指标 | 数量 | 备注 |
|---|---:|---|
| 本地 PDF | **51** | `papers/` 全量 |
| 研读报告 | **108** | `docs/reports/` (含 4 篇今日新增) |
| 已有 arxiv id 覆盖 | **28 个独立 arxiv id** | 见 §1.3 |
| 未对应研读报告的 PDF | **8** (今日减至 **4**) | 见 §1.2 |
| 已 grounding 数字的报告 | ~50% | 多数报告含 grounding；~50% 仅 abstract 级别 |

### 1.2 今日新增 (2026-09-28) — 填补空缺

| arxiv id | 标题 (短) | 报告路径 | grounding 状态 |
|---|---|---|---|
| 2606.10596 | Embedding Hybrid Systems into Continuous Latent Vector Fields | `docs/reports/Embedding_Hybrid_Systems_..._2606.10596_研读报告.md` | extracted-from-tracker (PDF 扫描图，需 OCR) |
| 2606.22075 | Frequency-Domain Neural ODEs for Modeling Non-Linear Dynamical Systems | `docs/reports/Frequency_Domain_Neural_ODEs_..._2606.22075_研读报告.md` | extracted-from-pdf-text (首段) |
| 2606.23550 | Approximating velocity fields with planted attractors via Neural-ODEs | `docs/reports/Planted_Attractors_Neural_ODE_..._2606.23550_研读报告.md` | extracted-from-pdf-text (首段) |
| 2608.13260 | Virtual Temperature Sensors in Power Transformers Using Neural ODEs | `docs/reports/Virtual_Temperature_Sensors_..._2608.13260_研读报告.md` | extracted-from-pdf-text (首段) |

**仍存在空缺 (4 个 PDF)**：
- `papers/foundational/hasani_2021_ltc.pdf` — 在 `docs/reports/LNN_Mathematical_Foundations_Comprehensive_2026-08-05.md` 已 grounding
- `papers/foundational/lechner_2022_cfc.pdf` — 同上
- `papers/foundational/hasani_2021_ncp_wrong-arxiv-id.pdf` — **非有效 arxiv id**，仅 reference
- `papers/foundational/_dup_hasani_2021_ltc_aaai.pdf` — **重复文件**，与 hasani_2021_ltc.pdf 同源

**结论**：核心空缺已补；剩余 4 个均为 foundational/duplicate，可忽略或后续清理。

### 1.3 已覆盖 arxiv id 列表 (28 个)

包含但不限于：2606.15571 (LRFM), 2604.10815 (MeloTune), 2606.26849 (LFNet), 2607.10858 (NSFD), 2606.19579 (FlowFake), 2606.20491 (GazeLNN), 2607.08283 (TFP), 2607.01986 (Turbofan), 2606.12240 (MR-MoE), 2602.06997 (EEG), 2604.07219 (Crystal Antennas), 2606.21295 (Topological), 2607.12909 (Fall Detection), 2608.03041 (PLAN), 2608.28702 (SDE-CfC), 2603.27058 (MDN-CfC), 2609.10715 (NCP), 2609.19674 (Conservation), 2606.07670 (Drop-in CfC), 2604.24788 (Natural Gas), 2605.27467 (Comparative LNN-LSTM), 2605.24047 (EMMA), 2603.00459 (LSS-LTCNet), 2604.14484 (BC Error Dynamics) 等。

## 2. arXiv 抓取修复 (本次关键变更)

### 2.1 问题诊断

- arXiv `/api/query` 端点对 host IP 持续 **HTTP 406** (Varnish throttle)
- 单 term / 短 query / 长 query 都返回 406
- `export.arxiv.org/rss/cs.LG` 等 category RSS 端点返回 **HTTP 200**（但周日 0 item，周末 arXiv 不发新论文）
- `arxiv.org/abs/<id>` / `/list/cs.LG/recent` 也返回 200
- 9/23 commit `23d9c7b` 加 `Accept: */*` header 未根治

### 2.2 修复方案 (本 session 落地)

修改 `scripts/daily_lnn_research.py`：

1. **新增 `ARCHIVE_RSS_CATEGORIES`** 配置：`["cs.LG", "cs.AI", "cs.NE", "cs.RO", "cs.SY"]`
2. **新增 `fetch_arxiv_rss(max_results)` 函数**：
   - 遍历 category RSS feed (`https://export.arxiv.org/rss/<category>`)
   - 解析 RSS `<item>` 节点 → 用 `keyword_score` 过滤
   - 去重 by arxiv id
   - 每个 paper 加 `"source": f"rss:{category}"` 标签
3. **修改 main 调用流程**：
   - 先 try `/api/query`，失败 / 0 paper → 触发 RSS fallback
   - RSS fallback 也失败 → 标 degraded
4. **errors 字段扩展**：digest 数据源状态自动展示 fallback 状态

### 2.3 预期效果

- 周一 (2026-09-29) 04:30 CST 自动跑：若 API 恢复 → normal path；否则 → RSS fallback 拿 7-15 篇
- 周末：API + RSS 双 0 → graceful degraded，无 nan

## 3. 完整研究 SOP (后续每日增量)

### 3.1 每日 digest SOP (已自动化)

```
04:30 CST systemd timer →
  daily_lnn_research.py →
      • fetch_arxiv (or fetch_arxiv_rss fallback)
      • fetch_github_repos
      • fetch_huggingface_models
      • download_pdfs (top-K)
      • render_daily_digest + render_repo_watchlist
      • sync_with_origin (race 方案 E)
  → commit + push
```

### 3.2 研读报告 SOP (人工/半自动触发)

按 [[AGENTS]] §2 "Summarization Agent" SOP：
1. digest 出现新 arxiv → 立即下载 PDF → 写独立研读报告
2. 命名: `docs/reports/<论文文件名>_研读报告.md`
3. 必备 6 模块: 元数据 / 核心问题 / 方法论 / 核心公式 / 关键成果 / 局限与展望
4. **新增第 7 模块 (本次落地)**: **本仓具体实现路径**
   - 必须 grounding 到 `lnn/core/`, `lnn/data/`, `analysis/<新域>/`
   - 标注诚实负结果预防策略
   - 标注合规边界 (不操控设备)

### 3.3 已研读报告"实现路径"补全策略 (重要)

108 个报告中，约 50 个早期报告**无"具体实现路径"段**。完整研究要求：

- **优先级 A (本仓核心路径, ~20 篇)**: r301-r307 系列 + CfC/LTC 主力 + LFM2.5 整合 → 立即补
- **优先级 B (长尾工业应用, ~15 篇)**: EEG / Fall Detection / Turbofan / Power Transformer 等 → 本月内补
- **优先级 C (综述/分类/跨学科, ~15 篇)**: 综合 survey → 季度补

**当前 session 落地**: 已补 4 篇新报告的"具体实现路径"段 (2606.10596/22075/23550/2608.13260)。剩余报告的"实现路径"补全 = **后续工作流增量任务**。

### 3.4 实施边界 (再次强调)

| 项 | 允许 | 禁止 |
|---|---|---|
| 数据 | 合成 (torch.randn / 数值 ODE) + 公开 benchmark (MNIST, WikiText) | 真机 (机器人 / 电网 / 工业总线) |
| 模型 | in-house `lnn/core/*` | 第三方 plugin (私有 API) |
| 设备 | Jetson Orin Nano (已部署) | ADB / devicectl / 真实传感器 |
| 实验 | in-house simulator + 合成 input | ROS / mavlink / CAN / Modbus / BMS |

## 4. 后续增量任务清单 (按优先级)

### P0 (24h 内)
- [x] 修 arXiv 抓取 (RSS fallback) — **本 session 落地**
- [x] 补 4 个未研读 arxiv 报告 — **本 session 落地**
- [ ] 9/29 周一 04:30 自动跑：验证 RSS fallback 是否在有 paper 的日子生效

### P1 (本周)
- [ ] 给 r301-r307 系列报告补"具体实现路径"段 (7 篇)
- [ ] 给 LFM2.5 整合相关报告补实现路径 (3 篇)
- [ ] 清理 `papers/foundational/` 重复文件 (`_dup_hasani_2021_ltc_aaai.pdf`)

### P2 (本月)
- [ ] 给长尾工业应用报告补实现路径 (~15 篇)
- [ ] 写《2026-10 LNN 路线图》: 基于已 grounding 的报告，列出本仓下 1 季度研发优先级

### P3 (季度)
- [ ] 给综述/分类报告补实现路径 (~15 篇)
- [ ] OCR 扫描图 PDF (`2606.10596v1` 等) 提取全文 grounding
- [ ] 重构本仓 docs 结构为 Zettelkasten (参见 [[living-field-researcher]] skill)

## 5. 用户偏好与约束遵循证明

✅ **不操控设备**: 所有实现路径均限定在 in-house simulator + 合成数据 (已写入 4 篇新报告)
✅ **AGENTS.md claim 边界**: 所有报告未触及 8 条不可重复 claim
✅ **诚实负结果**: r306/r307 + MT-LNN retracted 4 条主张延续标注
✅ **目标 grounding**: 一切数字声明需标 grounding 状态 (extracted-from-tracker / extracted-from-pdf-text / standard)

## 6. 相关文档

- [[docs/LNN_深度研读报告]] §0 项目定位
- [[AGENTS]] §"Agent 约束" 与 "工作流预期设计"
- [[docs/research/2026-09-24_technical_route_landscape]] §五 诚实约束
- [[skills/living-field-researcher]] SKILL.md (持续研究方法论)
- [[MEMORY|lnn-2026-09-28-goal]] (ICM 本 session goal 记录)
- [[MEMORY|lnn-daily-automation]] (race 修复历史)

## 7. 维护说明

- 本文档由 `living-field-researcher` skill 派生
- 每月审视一次 (与 [[AGENTS]] §约束 §维护说明 一致)
- 新增 arxiv id 必须同步更新 §1.3 覆盖列表
- RSS fallback 修复后，下一次 406 alert 必须标 `[FIXED] RSS fallback`