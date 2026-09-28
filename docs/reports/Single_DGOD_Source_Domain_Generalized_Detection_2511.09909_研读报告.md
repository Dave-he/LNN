---
title: Single-Domain Generalized Object Detection (arXiv 2511.09909) — 研读报告 (摘要)
date: 2026-09-28
tags: [LNN, paper, object-detection, single-dgod, domain-generalization, cross-domain]
arxiv_id: 2511.09909
pdf: papers/arxiv_pdf/2511.09909.pdf
status: pdf-grounded (abstract), out-of-LNN-scope
---

# Single-Domain Generalized Object Detection

> **Grounding 状态**: abstract 已 grounding. **本论文不直接属于 LNN / CfC / LTC / Neural ODE 主线**, 记录在本仓仅作 digest 命中留档.

## 元数据
- **来源**: arXiv:2511.09909 [cs.CV] (Single-DGOD)
- **本地 PDF**: [papers/arxiv_pdf/2511.09909.pdf](../papers/arxiv_pdf/2511.09909.pdf) (6.7MB)

## 核心问题
- 单源域训练 + 多未知域泛化的目标检测 (Single-DGOD)
- 现有方法: 离散数据增强 + 静态扰动, 难以覆盖真实跨域差异

## 与 LNN 关系
- **弱**: 目标检测是 vision 任务, 本文方法是 domain augmentation 而非连续时间动力学
- **不进入主线**: 不与 r301-r307 冲突, 也不与之同源
- **记录价值**: 仅作 digest 留档, 后续 digest 过滤逻辑可据此 exclude vision-only 增强探针

## 维护说明
- 摘要级 grounding, 不进入 LNN 主线分析
- 后续若 digest 再次命中此类 vision-only 论文, 直接排除避免噪音