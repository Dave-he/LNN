---
title: Operator-Consistent Physics-Informed Learning for Wafer Thermal Reconstruction in Lithography (arXiv 2510.09207) — 研读报告
date: 2026-09-28
tags: [LNN, paper, pinn, physics-informed, thermal-reconstruction, lithography]
arxiv_id: 2510.09207
pdf: papers/arxiv_pdf/2510.09207.pdf
status: pdf-grounded (abstract)
---

# Operator-Consistent Physics-Informed Learning for Wafer Thermal Reconstruction

> **Grounding 状态**: abstract + title 已 grounding 到 `papers/arxiv_pdf/2510.09207.pdf`.

## 元数据
- **来源**: arXiv:2510.09207v2 [math-ph] 27 Oct 2025
- **作者**: Ze Tao, Fujun Liu, Yuxi Jin, Ke Xu, Minghui Sun 等
- **本地 PDF**: [papers/arxiv_pdf/2510.09207.pdf](../papers/arxiv_pdf/2510.09207.pdf) (14.2MB)

## 核心问题
- **痛点**: 半导体 wafer thermal 重建需要 operator-level 一致性 (e.g. mass / energy conservation), 传统 PINN 难保证
- **现有方法**: 经典 PINN 在 weak-form 物理约束下重建 thermal field, 但与 operator 不直接对齐
- **作者思路**: 显式 enforce operator-consistent (e.g. PDE operator) 物理约束

## 方法论与核心思路
- 与本仓 `analysis/lrfm/` (Liquid Random Feature Methods) + `analysis/multimodal_physreg/` **高度同源**
- 不与 r301-r307 冲突, 是物理一致性方向延伸
- 与 [[Virtual_Temperature_Sensors_Neural_ODE_Power_Transformer_2608.13260_研读报告]] 共同构成工业 thermal 监测域

## 本仓具体实现路径
1. **数据** (`lnn/data/wafer_thermal_synth.py`): 合成 2D wafer thermal field (heat equation)
2. **模型** (`lnn/core/operator_consistent_pinn.py`): Liquid RFM backbone + operator-consistent loss
3. **实验** (`analysis/wafer_pinn/`): 5 seed × 3 noise level
4. **合规**: 仅合成数据, 不接真实半导体设备

## 维护说明
- abstract-grounded; 待全文 §实验 grounding