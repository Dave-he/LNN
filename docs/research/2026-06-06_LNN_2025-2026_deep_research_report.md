---
title: LNN 2025-2026 深度研究 + arXiv 目录审计 (round 72)
date: 2026-06-06
tags: [LNN, CfC, LTC, DSS, S4, Mamba, arXiv-catalog, verification, bibliographic-hygiene, round-72]
related:
  - "[[docs/research/2026-06-04_LNN_research_report]]"
  - "[[docs/液态神经网络最新进展与开源项目调研]]"
  - "[[docs/LNN_持续研究协议]]"
---

# 🌊 LNN 2025-2026 深度研究报告 — round 72

> 触发: `/iter 液态神经网络最新进展研究与验证` (2026-06-06)
> 本轮调用 `deep-research` skill,5 路 fan-out 搜索 → 18 sources → 14 claims → 3-vote 事实核查 → 最终 8 confirmed / 6 killed。
> **核心结论**: 现有 LNN 论文目录存在严重 arXiv ID 错配,且 2025-2026 SOTA 证据缺位,需 (a) 修目录 (b) 重锚 CfC 性能声明 (c) CfC vs DSS/S4 head-to-head 消融 (d) 边缘硬件实测。

## 1. 研究方法

**5 路 fan-out**:
1. CfC 与神经 ODE 训练优化 (固定点 / implicit / 自适应步长)
2. LTC 拓扑与稀疏化 (NCP, sparse wiring, time-constants)
3. LNN vs SSM/Mamba/KAN 对比与混合架构
4. Edge/TinyML 部署与硬件能效 (Jetson, MCU, Loihi-2)
5. 可复现 SOTA 基准与已知反例 (mackey-glass, sMNIST, drone, HAR, telecom, finance)

**3-vote adversarial verify** per claim (need 2/3 refutes to kill).
**资源消耗**: 67 agent calls / ~2.8M tokens / ~44 min wall-clock。

## 2. 关键发现 — arXiv ID 错配

deep-research 在事实核查阶段发现上游研究目录存在 **3 个 arXiv ID 错配**,这些 ID 指向与 LNN 无关的论文:

| 误用 ID | 实际指向的论文 | 真实归属 | 正确 ID (如可识别) |
|---|---|---|---|
| `2003.06236` | 碳氢化合物蒙特卡洛模拟 | (与 LNN 无关) | `2106.13898` (Hasani et al. CfC, Nature MI 4:992-1003, 2022, DOI 10.1038/s42256-022-00556-7) |
| `2203.14343` | Gupta/Gu/Berant "Diagonal State Spaces" (2022) | 误指 Lockhart et al. "Adaptive Solvers for Neural ODEs" | (待补) |
| `2002.08071` | Massaroli et al. "Dissecting Neural ODEs" (NeurIPS 2020) | 误指 Lienen & Günnemann "torchode" | (待补) |

**注意**: 仓库内 `docs/LNN_持续研究协议.md:138` 和 `docs/液态神经网络最新进展与开源项目调研.md:60+126` 已使用正确 ID `2106.13898` 引用 CfC,说明错配是孤立的、可能存在于某个外部 catalog 或 daily JSON 的元数据中。

## 3. 6 个研究方向上的 2025-2026 状态

### 3.1 CfC 与神经 ODE 训练优化 (固定点 / implicit / 自适应步长)

- **存活的主要实证声明**: CfC 论文 (2106.13898, Nature MI 2022) **不是精确闭式解**,而是对 LTC 积分的**紧密有界近似**。这一性质是该论文的稳定数学结论,不会随时间变化。
- **被拒绝的声明**: "1-5 orders of magnitude speedup"、 "no solver needed"、 "scales remarkably well" — 这些措辞**因 ID 错配而被杀**,**而非被直接证据反驳**。在 re-anchor 到正确 2106.13898 之前,不可下游引用。
- **2025-2026 状态**: 缺乏直接证据,需重新检索 Lockhart et al. adaptive solvers (正确 ID 待补) 及其在 CfC 上的应用。

### 3.2 LTC 拓扑与稀疏化 (NCP, sparse wiring)

- 上游搜索的 6 个角度中, 0 results (edge-hardware) + 1 result (hybrid) + 6 results (LTC topology) = 7 sources,大部分标注 "unreliable"。
- 现有 repo 已有 `lnn/core/dynpmnn.py` (FHNCell + DynPMNNNetwork) 与 `tests/test_dynpmnn.py`,且 round 24 已做过 6-seed mackey_glass benchmark (honest negative result)。
- **2025-2026 状态**: 本仓内部已有 autonomous NCP 与 pruning 工作,外部 2025-2026 引用度不高,稀疏化方向属于"持续打磨"而非"突破窗口"。

### 3.3 Edge/TinyML 部署与硬件能效

- **deep-research 0 results**。这是最值得关注的盲区 — 现有 daily pipeline 未覆盖 Jetson / Cortex-M / Loihi-2 / FPGA / Apple Silicon 上的 LNN 实证能耗/时延。
- 仓库内有 `scripts/jetson_lnn_benchmark.py` 与 `analysis/jetson/`,但**未与 SSM/Mamba/Transformer 在同硬件上做 head-to-head**。
- **2025-2026 状态**: **是本轮最有价值的新增实验方向**。Pareto 曲线 (能耗/时延 vs 序列长度 vs 参数数) 至今缺位。

### 3.4 LNN vs SSM/Mamba/KAN 对比与混合架构

- 仅 1 result,标注 "unreliable"。
- **唯一存活的实证声明**: DSS (Diagonal State Spaces, Gupta/Gu/Berant 2022) **purely diagonal** 状态矩阵在 LRA 上 avg 81.88 vs S4 80.21,Speech Commands 98.2 vs 98.1。
- ⚠️ **关键限制**: 这是 2022 历史对比,**不含 Mamba (2023) / RetNet (2023) / Mamba-2 (2024) / 任何 2025-2026 SOTA**。可作为 fixed baseline 引用,不可作为前沿声明。
- **2025-2026 状态**: 大量 Mamba-vs-Transformer 比较存在,但 Mamba-vs-CfC 极少。这是 **第二代头对头基准缺口**。

### 3.5 多尺度 / 层级 / 模块化 / 联邦 / 终身学习

- 0 dedicated results from deep-research。
- 仓库内已有 Riemannian LTC (arXiv 2601.14115)、RLSTG、HierarchicalDecayLiquidTADHead、EntroLnn 等工作 (round 65-71)。
- **2025-2026 状态**: 仓库内部已较丰富,外部引用度待补。

### 3.6 可复现 SOTA 与已知反例

- 6 sources (5 novel),最终 8 confirmed / 6 killed。
- **重要发现**: **没有任何 2025-2026 论文在标准基准 (sMNIST / permuted-MNIST / seq CIFAR / Mackey-Glass / drone / HAR) 上提供完整可复现的 CfC vs Mamba/Mamba-2/RetNet/Transformer head-to-head**。上游 daily research front-loaded 了 2020-2022 foundational work。
- **2025-2026 状态**: **这是 deep-research 最强烈的 actionable signal** — 本仓有完整 ncp 库 + CfC 复现 + 多 backbones,可独立补齐这个空白。

## 4. 复现性 / 负结果

- 现有 catalog 强倾向于"vendor claims"(CfC 比 ODE 快 1-5 个数量级),弱化诚实反例 (round 24 mackey_glass 6-seed negative result, round 21 GRU encoder +3.9% vs Bi-CfC +35.2%)。
- 论文目录的 negative result 比例 < 5%,远低于真实文献中应有的负结果密度。
- **可执行修正**: 在 daily research JSON 中加入 "reproduction_status" 字段,强制区分 (a) vendor claim (b) third-party reproduced (c) failed to reproduce。

## 5. 工程化下一步思路 (按价值排序)

1. **[P0] 修复 arXiv ID 错配 + 添加 reproduction_status 字段** — 修目录,纯数据工程,半天可完成。可在 daily pipeline 中加 lint 步骤, 校验 ID 与 title 对应。
2. **[P1] CfC vs DSS/S4 head-to-head on sMNIST / permuted-MNIST / seq-CIFAR / Mackey-Glass** — 3-seed × 4-backbone, 跑 `scripts/build_backbone_matrix.py` 增量。直接填补 3.6 缺口。**预期 delta**: 5-10pp accuracy 在长序列任务上,且能给出**明确的负结果** (Mamba/SSM 是否在小数据集上击败 CfC)。
3. **[P2] Jetson Orin Nano 上 CfC vs Mamba vs Transformer 能耗/时延 Pareto 曲线** — 跑 `scripts/jetson_lnn_benchmark.py` 增量,加入 Mamba/Transformer baseline。**预期 delta**: 给出 LNN 在何种 (seq_len, hidden) 范围内 energy-efficient 优势,反之亦然。
4. **[P3] Lockhart et al. adaptive solvers 正确 ID + CfC 上应用复现** — 需要先补全 ID,再在 `lnn/core/` 写一个 `cfc_adaptive.py` 包装。
5. **[P4] 跑 notebook-style 论文目录去重** — 用 `analysis/llm_micro_eval/` 类似流水线,每周扫一次 arXiv LNN/LTC/CfC,自动发现新 ID 错配。

## 6. 提交 + 推送

- 本报告 + 1-2 个 PRD (round-72-a/b) + 实现/测试/commit。
- 全套 `pytest tests/` 在 round 71 已 **稳定 137/137** (含 14+ 个新单测), round 72 须零回归。

---
*本报告由 `deep-research` skill 自动生成, 67 agent calls, 8 confirmed / 6 killed 3-vote verification。*
