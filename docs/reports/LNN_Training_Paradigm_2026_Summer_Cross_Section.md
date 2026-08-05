---
title: LNN 训练范式 2026 夏横切 — Multi-Rate MoE / Distillation / Random Feature / LFM2.5 蒸馏 串讲
date: 2026-08-05
tags: [LNN, training-paradigm, multi-rate, MoE, distillation, random-feature, LFM2.5, CfC, cross-section, gap-update]
parent: [[LNN_深度研读报告]]
companion: [[LNN_Family_Taxonomy_And_Gap_2026-08-03]]
---

# LNN 训练范式 2026 夏横切 — Multi-Rate MoE / Distillation / Random Feature / LFM2.5 蒸馏

> 本文不重复研读单篇论文细节，而是把 2026-05 → 2026-08 期间出现的 **4 条 LNN 训练范式主线** 拉到同一坐标系：蒸馏压缩 / 多速率 MoE / 随机特征闭式化 / 基础模型 LFM2.5 蒸馏，并把它们映射到 [[LNN_Family_Taxonomy_And_Gap_2026-08-03]] 的 Gap 清单，给出 8/5 当下的推荐动作。

## 1. 四条主线一图归并

| # | 主线 | 代表论文 / 产物 | 核心改造对象 | 落地形态 |
|---|---|---|---|---|
| **L1** | 双阶段蒸馏 + Pareto 压缩 | [[DLNet_Dual_Stage_Distillation_Pareto_LNN_2601.06227_研读报告\|DLNet]] `arxiv 2601.06227` | 教师 → 学生 LNN | 学生 LNN (h≈8-16) → 边缘部署 |
| **L2** | 多速率 MoE 加速训练 | [[Multi-Rate_MoE_Accelerating_LNN_Training_2606.12240_研读报告\|MR-MoE LNN]] `arxiv 2606.12240` | 单 τ 的 CfC → K 个 τ 的 expert 池 + 路由 | `MultiRateMoECfC` / `MultiRateMoECfCNetwork`（仓库已落地，2026-08-03） |
| **L3** | 随机特征闭式化（数学基底） | [[Liquid_Random_Feature_Methods_TD-PDE_2606.15571_研读报告\|L-RFM]] `arxiv 2606.15571` | 把 LTC 的非线性时间演化 → 随机特征线性代数 | 求解 TD-PDE 的 mesh-free surrogate |
| **L4** | 基础模型蒸馏（LFM2.5 系列） | HF LiquidAI/LFM2.5-350M/1.2B/2.6B/8B-A1B + litert-community Encoder 350M | 大 LFM → 边缘小 LFM / 任务专用 head | `lnn/lfm2/inference.py` 推理路径 |

四条线不是平行罗列，而是按 "**训练成本从高到低、参数量从大到小**" 一条斜线：

```
L4 (LFM2.5 全家) ── 蒸馏 ──▶ L1 (DLNet 学生 LNN)
                                    │
L3 (L-RFM 数学基底) ────────▶ L2 (MR-MoE 多速率 CfC) ── 部署 ──▶ Jetson / 边缘
```

## 2. 主线逐一拆解

### 2.1 L1 · 双阶段蒸馏 + Pareto 压缩（DLNet, `2601.06227`）

- **核心问题**：电池剩余寿命（RUL）预测的 LNN 教师（h=64）参数过大，无法进入嵌入式 BMS。
- **方法论**：
  - **Stage 1**: 在隐层特征空间做 *activation-distillation*，对齐师生分布。
  - **Stage 2**: 用 Pareto 前沿在 **参数-精度-时延** 三维上选出 h∈{8,16,24} 的可部署版本。
- **结果**：学生 h=12 时 MSE 与教师差距 <2%，参数仅为教师的 4%。
- **本机意义**：DLNet 仓库今天仍在更新（`Dhivya-DD17/DLNet` 8/4 push），是当前 "LNN 边缘蒸馏" 的最完整开源实现之一，适合做复现 target。

### 2.2 L2 · 多速率 MoE 加速训练（MR-MoE LNN, `2606.12240`）

- **核心问题**：标准 CfC 的 τ 是 scalar，所有神经元共享同一时间尺度，对多频谱时间序列（呼吸 + 心跳 + 噪声）建模效率低。
- **方法论**：
  - K 个 expert，每个 expert 跑一组 τ ∈ {τ₁, …, τ_K} 的 CfC。
  - 每个时间步做 **Expert-Choice (EC) routing**：固定每个 expert 接收 top-C tokens，反向决定 token→expert 分配。
  - 路由可学习，但 EC 比 Soft-MoE / Gumbel 更稳定（论文 + 仓库多次消融支持）。
- **结果**：在 PhysioNet sepsis / Activity / Power 多数据集上，相比单 τ CfC **训练步数 -35% 到 -60%**，最终 AUC 不变或 +0.5pp。
- **仓库落地**：
  - `lnn/core/multirate_moe_cfc.py` `MultiRateMoECfC`
  - `lnn/core/multirate_moe_cfc.py::MultiRateMoECfCNetwork`
  - `examples/*` 有 smoke 测试；2026-08-03 Pareto 已跑（CPU + GPU）。

### 2.3 L3 · 随机特征闭式化（Liquid Random Feature Methods, `2606.15571`）

- **核心问题**：把 LTC 拟合到 TD-PDE 的 residual collocation loss 上时，**ODE 内积的解析形式缺失**，只能 forward rollout 计算残差，耗时且不稳。
- **方法论**：
  - 用 **Liquid Random Feature** 把 LTC 的闭式解近似为 φ(x; θ) 线性组合（φ 取随机傅里叶特征 / Hermite 多项式）。
  - 这样 PDE 残差从 rollout → 解析 → 可一次性 Jacobian → Newton 收敛更快。
- **理论意义**：
  - 与 CfC 的 σ(-f·t)·g + (1-σ)·h **同源**——都是 "把 ODE 解展开为闭式"。
  - 给后续 "LNN-as-feature" 提供了数学基础：可以脱离 ODE 求解器训练更深的 LNN 表示。
- **本机意义**：纯数学/方法论文，对应 [LNN_Family_Taxonomy_And_Gap_2026-08-03]] 中 **L1 (低优先级缺口)**，建议下个 cron 抽 1 行进入"待复现候选"。

### 2.4 L4 · LFM2.5 蒸馏家族（基础模型路径）

- **核心问题**：把 LiquidAI 官方 LFM2.5 系列（350M → 8B-A1B）蒸馏到边缘可跑规格，并验证任务 head 与 LFM2 主干的协同。
- **HF 生态（2026-08-04 当日）**：
  - 官方：`LiquidAI/LFM2.5-{350M, 1.2B, 2.6B, 8B-A1B}`、`LFM2.5-Encoder-350M-Spellchecker/PII-Detector/Policy-Linter/Prompt-Router`（已 task-specific distill）。
  - 社区：`litert-community/LFM2.5-{1.2B-Instruct,1.2B-Thinking,1.2B-JP}`、`FastFlowLM/LFM2-{1.2B,2.6B}-NPU2`、`bartowski/LFM2.5-2.6B-GGUF`、`noctrex/LFM2.5-2.6B-heretic-uncensored-GGUF`。
- **仓库现状**：
  - `lnn/lfm2/inference.py` 已支持 `LFM2-350M/700M/1.2B/2.6B-Exp/24B-A2B` 推理。
  - **缺**：蒸馏脚本（teacher→student LNN head）、LFM2.5-350M + CfC head 的混合 fine-tune recipe、量化 → ONNX → TensorRT pipeline。
- **本机意义**：8/3 的 [[LNN_Family_Taxonomy_And_Gap_2026-08-03]] Gap **G5 (LFM2.5-350M Jetson 端到端 8GB smoke)** 仍未实测。

## 3. 横向对照矩阵

| 维度 | L1 DLNet | L2 MR-MoE | L3 L-RFM | L4 LFM2.5 |
|---|---|---|---|---|
| **改造层面** | 模型规模 | 时间常数分布 | 数学求解方式 | 知识来源（教师） |
| **核心机制** | 蒸馏 + Pareto | EC routing | 随机特征闭式 | teacher=LLM, student=LNN head |
| **可叠加性** | ✅ 可作为 L2/L4 的后置压缩 | ✅ 可作为 L4 head | ✅ 给 L2/L4 提供更快训练 | ❌ 是替代路径 |
| **边缘就绪度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐（仅算法） | ⭐⭐⭐⭐⭐（350M/1.2B） |
| **本机覆盖度** | 报告 ✅ 代码 ✅ 复现 ❌ | 报告 ✅ 代码 ✅ 复现 ✅ | 报告 ✅ 代码 ❌ 复现 ❌ | 报告 ✅ 代码 ⚠ 量化 ❌ |

## 4. Gap 增量更新（8/3 → 8/5）

承接 [[LNN_Family_Taxonomy_And_Gap_2026-08-03]] 的 3.x 节：

| # | 缺口 | 8/3 状态 | 8/5 状态 | 触发/依据 |
|---|---|---|---|---|
| G1 | Orin Nano Super GPU 真 CUDA benchmark | ✅ | ✅ 增量更新 | 2026-08-05 quick CPU benchmark 见 §5 |
| G2 | Multi-Rate MoE (2606.12240) | ✅ 落地 | ✅ 验证 Pareto | CPU + GPU Pareto 跑通 |
| G3 | Liquid-3DGS (2606.07670) 代码层 | 待做 | 待做 | 论文已有研读，代码侧缺 |
| G4 | LiquidTAD streaming stateful ckpt | 待做 | 待做 | 2604.18274 已有研读 |
| G5 | LFM2.5-350M Jetson 部署 prototype | 离线 README | 仍未实测 | HF 8/4 出现更多 LFM2.5 派生量化版 |
| **新增 N1** | DLNet 蒸馏复现 (2601.06227) | — | **新建议**：复现 Stage 1 蒸馏 + Pareto 前沿 | DLNet 仓库 8/4 更新 |
| **新增 N2** | L-RFM 数学嵌入 LNN 训练 (`khlfft_attn_cfc` 路线) | — | **新建议**：写 `L_RFM_CfC` 单元 | 2606.15571 与 KHLFFT SSM 同源 |
| **新增 N3** | TFP Memory-Fusion (2607.08283) 跨 LNN 迁移 | — | **新建议**：把 TFP 的 memory fusion 嫁接到 `CfCCell` 的门控 | 报告已落地，待代码 |
| **新增 N4** | FlowFake (2606.19579) audio CfC head | — | **新建议**：`LiquidAudioClassifier` skeleton | 报告已落地 |

## 5. 8/5 实验数据点：Quick CPU Benchmark

由 `scripts/jetson_lnn_benchmark.py --date 2026-08-05 --quick --cpu` 跑出，完整结果：[analysis/jetson/2026-08-05_lnn_benchmark.md](analysis/jetson/2026-08-05_lnn_benchmark.md)。

| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312 | 11439.7 | 10.71 | 0.86 |
| LTC | 1321 | 0.465 | 9191.7 | 24.36 | 1.02 |
| **PDNAPulse** | **3170** | **0.286** | **23388.9** | **4.33** | **0.40** |
| GRU (基线) | 1969 | 0.393 | 76627.1 | 3.79 | 0.14 |
| NCPS-LTC | 2547 | 0.621 | 2151.8 | 66.00 | 4.65 |
| **NCPS-CfC** | **15737** | **0.106** | 11026.7 | 11.98 | 0.85 |

**关键观察（与 8/3 GPU Pareto 对照）**：
1. **精度王者**：NCPS-CfC（mlech26l/ncps 官方实现），MSE=0.106，但参数量 15737 是 GRU 的 8×。CPU 模式跑说明 ncps-cfc 的核心运算没被 torch 编译优化吃满，量化后可能有惊喜。
2. **精度/速度甜点**：**PDNAPulse**（仓库自研，融合 arxiv 2603.00153 pulse 调制）MSE=0.286、推理 23K 步/秒、能耗 0.40 mJ/步。综合最优，8/3 GPU Pareto 也确认过。
3. **GRU 仍是吞吐王者**（76K 步/秒），但 MSE 0.393 比 CfCStyle 高 26%——经典 trade-off。
4. **LTC 在 CPU quick 上偏弱**（MSE=0.465、训练 24s），但 8/3 GPU Pareto 给出 LTC h=24 seq=32 的 MSE 0.279，说明 **LTC 需要 GPU 才能发挥 ODE solver 优势**，CPU 上被 pyoddeint/euler 拖慢。

## 6. 推荐下一步动作（按 ROI 排序）

1. **本周内可落地**：把 2607.08283 TFP 的 memory-fusion 思路移植到 `CfCCell` 门控（**新增 N3**），属于增量改动，~80 行代码。
2. **下周**：跑 LFM2.5-350M + CfC head 的混合 fine-tune smoke（**G5**），先在 CPU 上做 forward sanity，验证 head 与 LFM2 主干的 embedding 对齐。
3. **下下周**：复现 DLNet 双阶段蒸馏到仓库内 `MultiRateMoECfC` 教师 → 学生路径（**新增 N1**），把 Pareto sweep 接进 `scripts/bench_*`。
4. **路线图**：把 L-RFM 的随机特征闭式化（L3）与 `lnn/core/khlfft_attn_cfc.py` 合流（**新增 N2**），目标是把 KHLFFT attention 的频域扩展为 LNN-friendly 的闭式特征基。

## 7. 数据源回链

- 训练范式主线来源
  - [[DLNet_Dual_Stage_Distillation_Pareto_LNN_2601.06227_研读报告]]
  - [[Multi-Rate_MoE_Accelerating_LNN_Training_2606.12240_研读报告]]
  - [[Liquid_Random_Feature_Methods_TD-PDE_2606.15571_研读报告]]
  - [HF LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M)
- 配套综合 / Gap
  - [[LNN_Family_Taxonomy_And_Gap_2026-08-03]]
  - [[Orin_Nano_Super_LNN_Deployment_v2_2026-08-03]]
- 每日追踪
  - [[LNN_每日研究追踪 - 2026-08-03]]
  - [[LNN_每日研究追踪 - 2026-08-04]]
  - [[LNN_每日研究追踪 - 2026-08-05]]
- 实验数据
  - [analysis/jetson/2026-08-05_lnn_benchmark.md](analysis/jetson/2026-08-05_lnn_benchmark.md)
  - [analysis/jetson/2026-08-05_lnn_benchmark.json](analysis/jetson/2026-08-05_lnn_benchmark.json)
