---
title: Locally Stable Neural ODEs with Characterized Region of Attraction (arXiv 2606.19109) — 研读报告
date: 2026-09-28
tags: [LNN, paper, neural-ode, stability, region-of-attraction, lyapunov]
arxiv_id: 2606.19109
pdf: papers/arxiv_pdf/2606.19109.pdf
status: pdf-grounded (abstract)
---

# Locally Stable Neural ODEs with Characterized Region of Attraction

> **Grounding 状态**: abstract 已 grounding 到 `papers/arxiv_pdf/2606.19109.pdf`. 作者来自 KTH + Stockholm + Toulouse.

## 元数据
- **来源**: arXiv:2606.19109v1 [math.OC] 17 Jun 2026
- **作者**: Alice Harting, Karl Henrik Johansson (KTH), Sophie Tarbouriech, Matthieu Barreau
- **本地 PDF**: [papers/arxiv_pdf/2606.19109.pdf](../papers/arxiv_pdf/2606.19109.pdf) (1.8MB)
- **背景**: 控制系统理论 (math.OC) + ML 交叉

## 核心问题
- **痛点**: 经典 Neural ODE universal approximation 学到的 vector field 缺乏**稳定性保证** —— 在某些初值下可能发散 / chaotic
- **现有方法**: 在特定 ODE 系统上 stability analysis 成熟, 但 Neural ODE 上无法直接应用
- **作者思路**: 把 Lyapunov 函数 + Neural ODE 联合学习, 给出 **locally exponentially stable dynamics** 的形式化保证

## 方法论与核心思路
- **核心机制**:
  - 学到的 vector field $f_\theta(x)$ 在 region $\mathcal{R}$ 内满足 Lyapunov 不等式: $\nabla V(x) \cdot f_\theta(x) \le -\alpha V(x)$, $\forall x \in \mathcal{R}$
  - $V(x)$ 是同学习的 Lyapunov 函数 (e.g. 二次型 + 神经修正)
  - Region $\mathcal{R}$ 是 region of attraction 的**形式化内逼近**
- 形式 (基于 abstract 推断):
  $$\dot{x}(t) = f_\theta(x(t)), \quad V(x) > 0 \text{ on } \mathcal{R}$$
  $$\nabla V(x) \cdot f_\theta(x) \le -\alpha V(x) \text{ on } \mathcal{R}$$
- 与 LNN 关系:
  - 与本仓 `analysis/sncp_ppo_lite/` 控制稳定性域相关
  - 与 [[Conservative_LNN]] (Conservation 系列) 互补: 本方法是稳定性 (Lyapunov), 守恒是守恒律
  - 不与 r301-r307 冲突, 是稳定性方向的补充

## 关键成果与贡献
- ⚠️ abstract 提及 "universally approximates locally exponentially stable dynamics" —— 这是**理论性 claim**, 非数字
- **优势**: 给出 Neural ODE stability 的形式化保证, 这是控制系统理论 ↔ ML 难得的桥梁
- **诚实声明**: 具体数值实验待 PDF 全文 grounding

## 局限性与未来展望
- (基于 abstract 推断):
- region of attraction 大小是 trade-off (大 → 弱保证, 小 → 强保证)
- 与 Lyapunov 数值求解的耦合紧

## 本仓具体实现路径 (in-house, 合成数据)

### 适配度
- **中**: 与本仓 `analysis/sncp_ppo_lite/` (控制稳定性) + `lnn/core/conservation_*` (守恒) 路径相关
- **可作为 stability regularization 嫁接到现有 CfC / LTC backbone**

### 实施步骤
1. **数据生成器** (`lnn/data/stable_neural_ode_synth.py` 新建):
   - 合成 locally stable ODE 系统 (e.g. damped pendulum, stable linear systems)
   - 已知 ground-truth Lyapunov 函数 (for eval)
   - 边界: 仅合成数据
2. **模型** (`lnn/core/stable_neural_ode.py` 新建):
   - Vector field: 神经 MLP
   - Lyapunov head: $V(x)$ 同学习
   - Loss:
     - 主: dynamics fit (data fit)
     - Lyapunov penalty: $\max(0, \nabla V \cdot f + \alpha V)$
     - Region characterization: 通过 grid sampling
3. **Loss** (`lnn/core/trainer.py` 加 multi-task):
   - 主: λ₁=1.0 data fit
   - Lyapunov: λ₂=10 (penalty)
   - Region: λ₃=0.1 (鼓励大 region)
4. **实验队列** (`analysis/stable_neural_ode/`):
   - 3 系统 (damped pendulum / 2D stable linear / Van der Pol modified) × 5 seed
   - 对照: vanilla Neural ODE / Physics-Informed Neural ODE
   - 关键指标: max region size / Lyapunov violation rate / rollout MSE
5. **诚实负结果预防**:
   - 若 Lyapunov violation > 0 → 进 negative_results
   - **不得**宣称"所有训练后的 Neural ODE 都稳定"
6. **合规边界**:
   - 仅合成数据, 不接真机 / 控制系统

### 关联 grounding
- 邻近: [[Conservation_Buys_Stability_Factoring_Buys_Counterfactuals_2609.19674_研读报告]]
- 邻近: [[Structure_Preserving_Neural_ODEs_NSFD_2607.10858_研读报告]]
- 邻近: [[SNCP-PPO_Crowdnav_LTC_深度研读报告]] (稳定性 + LNN)
- 约束: [[docs/LNN_深度研读报告]] §0

## 维护说明
- 本报告已 grounding 到 PDF abstract, 后续需读全文
- 升级路径: pdftotext 全文 → 验证实验数字 → 升级 `status: pdf-grounded (full)`