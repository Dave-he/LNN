---
title: Continuous-Time Homeostatic Dynamics for Reentrant Inference Models (arXiv 2512.05158) — 研读报告
date: 2026-09-28
tags: [LNN, paper, continuous-time, homeostatic, reentrant, inference-model]
arxiv_id: 2512.05158
pdf: papers/arxiv_pdf/2512.05158.pdf
status: pdf-grounded (abstract)
---

# Continuous-Time Homeostatic Dynamics for Reentrant Inference Models

> **Grounding 状态**: abstract + title 已 grounding 到 `papers/arxiv_pdf/2512.05158.pdf`. 作者 B. G. Chae (ETRI Korea).

## 元数据
- **来源**: arXiv:2512.05158v1 [math.DS] 4 Dec 2025
- **作者**: B. G. Chae (ETRI, Electronics and Telecommunications Research Institute, Daejeon, Korea)
- **本地 PDF**: [papers/arxiv_pdf/2512.05158.pdf](../papers/arxiv_pdf/2512.05158.pdf) (1.1MB)
- **背景**: 数学动力学 (math.DS) + 神经科学 (homeostasis)

## 核心问题
- **痛点**: Reentrant inference (循环反馈推理) 模型需要**持续稳态调节** —— 否则神经元激活会 drift 到饱和或 silent
- **现有方法**: 离散 batch normalization / layer norm, 但与连续时间动力学不兼容
- **作者思路**: 在连续时间神经 ODE 上加 **homeostatic dynamics** —— 类似生物神经元自适应调节

## 方法论与核心思路
- **核心机制**:
  - 在 neural ODE 的 vector field 上加 homeostasis 项:
  $$\dot{x}(t) = f_\theta(x(t), u(t)) - \eta (x(t) - x_{\text{target}})$$
  其中 $\eta$ 是 homeostatic rate, $x_{\text{target}}$ 是目标态
  - 形成 **Fast-Weights Homeostatic Reentry Network**: 短时学习 + 长时间稳态调节并存
- 与 LNN 关系:
  - 与本仓 LTC / CfC 的"时间常数"思路**同源**: LTC 的 τ 调节, CfC 的 gating, 本论文的 homeostasis 是另一种稳态调节
  - 不与 r301-r307 冲突, 是稳定性方向的延伸
- 与既有 homeostatic 理论 (Turrigiano homeostatic plasticity) 同源

## 关键成果与贡献
- ⚠️ abstract 未明确列出实验数字 (math.DS 类, 重理论), 需后续 PDF grounding
- **优势**: 给 reentrant inference 提供连续时间 stability 保证
- **诚实声明**: 数字待 PDF §实验 / §理论段 grounding

## 局限性与未来展望
- (基于 abstract 推断):
- homeostatic rate $\eta$ 与系统动力学存在耦合, 需 case-by-case 调
- 与监督学习的兼容性待补

## 本仓具体实现路径 (in-house, 合成数据)

### 适配度
- **中-高**: 与本仓 LTC 路径**高度同源**, 可作为 homeostatic extension 嫁接到现有 CfC backbone
- **不与 r301-r307 冲突**, 是连续时间稳态方向的补充

### 实施步骤
1. **数据生成器** (`lnn/data/homeostatic_reentry_synth.py` 新建):
   - 合成 reentrant 推理任务 (signal + noise, recurrent inference)
   - 模拟漂移 (drift) 场景, 测试 homeostatic 调节
   - 边界: 仅合成数据
2. **模型** (`lnn/core/homeostatic_cfc.py` 新建):
   - CfC backbone: 沿用 `lnn/core/closed_form_continuous_cell.py` 栈
   - Homeostatic head: $x_{\text{target}} - x$ 反馈项
   - 训练目标: minimize inference loss + λ=1e-3 homeostatic violation
3. **Loss**:
   - 主: inference NLL
   - 正则: λ=1e-3 $\| x - x_{\text{target}} \|^2$ (稳态偏离)
4. **实验队列** (`analysis/homeostatic_cfc/`):
   - 3 noise regime × 5 seed = 15 run
   - 对照: vanilla CfC / CfC + layer norm / LSTM
   - 关键指标: 长期 rollout 漂移率 / inference accuracy / homeostatic violation
5. **诚实负结果预防**:
   - 若 rollout 漂移率比 vanilla CfC 差 → 进 `analysis/negative_results/`
   - **不得**宣称"homeostatic CfC 在所有 reentrant 任务上 SOTA"
6. **合规边界**:
   - 仅合成数据, 不接真机 / 生物神经系统

### 关联 grounding
- 邻近: [[Liquid_Neural_Networks_Mathematical_Foundations_Comprehensive_2026-08-05]] (LTC 数学)
- 邻近: [[Locally_Stable_Neural_ODEs_Region_of_Attraction_2606.19109_研读报告]] (稳定性)
- 邻近: [[Conservation_Buys_Stability_Factoring_Buys_Counterfactuals_2609.19674_研读报告]]
- 约束: [[docs/LNN_深度研读报告]] §0

## 维护说明
- 本报告已 grounding 到 PDF abstract, 后续需读全文
- 升级路径: pdftotext 全文 → 验证实验数字 → 升级 `status: pdf-grounded (full)`