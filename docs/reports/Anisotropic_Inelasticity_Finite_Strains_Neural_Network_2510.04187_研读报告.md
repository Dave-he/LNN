---
title: A Complement to Neural Networks for Anisotropic Inelasticity at Finite Strains (arXiv 2510.04187) — 研读报告
date: 2026-09-28
tags: [LNN, paper, neural-network, anisotropic-inelasticity, finite-strain, mechanics]
arxiv_id: 2510.04187
pdf: papers/arxiv_pdf/2510.04187.pdf
status: pdf-grounded (abstract)
---

# A Complement to Neural Networks for Anisotropic Inelasticity at Finite Strains

> **Grounding 状态**: abstract + title 已 grounding 到 `papers/arxiv_pdf/2510.04187.pdf`.

## 元数据
- **来源**: arXiv:2510.04187v1 [cs.CE] 5 Oct 2025
- **作者**: Hagen Holthusen (FAU Erlangen), Ellen Kuhl (Stanford)
- **本地 PDF**: [papers/arxiv_pdf/2510.04187.pdf](../papers/arxiv_pdf/2510.04187.pdf) (10.5MB)

## 核心问题
- **痛点**: 经典 Neural Network 在**各向异性 (anisotropic) 有限应变 (finite strain) 非弹性**模拟中无法保证物理一致性 (各向异性张量结构, 应力-应变关系)
- **现有方法**: 数据驱动 NN 拟合 stress-strain curve, 但 anisotropy / objectivity 难以 grounding
- **作者思路**: 用 Neural Network **complement** 经典本构模型, NN 学 anisotropic part, 经典模型提供 physical skeleton

## 方法论与核心思路
- **Hybrid 架构**: 经典 anisotropic inelasticity model + 神经 residual 学 corrections
- 与本仓 `analysis/multimodal_physreg/` (Physics-Modeled) + `lnn/core/dynpmnn` 域**高度同源**
- 不与 r301-r307 冲突, 是物理一致性方向延伸

## 本仓具体实现路径
1. **数据** (`lnn/data/finite_strain_anisotropy_synth.py` 新建): 合成各向异性 stress-strain 曲线
2. **模型** (`lnn/core/anisotropic_nn_complement.py`): 经典 backbone + NN 残差
3. **实验** (`analysis/anisotropic_complement/`): 5 seed × 3 anisotropy level
4. **合规**: 仅合成数据, 不接真机 / 工业仿真

## 维护说明
- abstract-grounded; 待全文 §实验 grounding