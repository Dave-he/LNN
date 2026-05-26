---
title: LNN 训练方法与方向可行报告
date: 2026-05-26
tags: [LNN, training, dataset, architecture, feasibility]
---

# LNN 训练方法与方向可行报告

本文回答三个落地问题：LNN 如何构建数据集、如何搭建神经网络架构、如何训练和调参。结论按研究方向拆分，详细方案见各方向报告。

## 1. 方向总览

| 方向 | 适合任务 | 首选架构 | 数据集构建重点 | 可行性 |
|---|---|---|---|---|
| 核心架构与通用训练 | 任何序列任务的基线验证 | CfC -> LTC -> AutoNCP | 统一为 `(batch, time, feature)`，保留 `delta_t` | 高，仓库已具备基础代码 |
| 非平稳时间序列、金融、医疗 | 预测、分类、早期预警 | CfC/LTC，必要时加 CNN/Attention 编码器 | 时间切分、滑窗、缺失值 mask、外生变量 | 高，现有脚本可直接改造 |
| 机器人控制与模仿学习 | 自动驾驶、机械臂、导航 | CNN/MLP encoder + CfC/LTC/AutoNCP + MDN head | episode、观测、动作、时间戳、闭环验证 | 中高，需要新增机器人数据加载器 |
| 边缘部署与压缩 | Jetson、MCU、Loihi、低功耗设备 | 小型 CfC、Euler-LTC、蒸馏学生模型 | 校准集、延迟/能耗标注、硬件约束 | 中高，已有 Jetson smoke benchmark |
| 图时空与通信系统 | 交通、6G beamforming、多智能体 | LNN + GNN、RLSTG、LNN + 优化器 | 图快照、信道矩阵、场景级 OOD 切分 | 中，需要补图数据管线 |
| 长序列与视频理解 | 视频动作检测、长文本/音频 | Liquid-S4、并行 liquid relaxation | 预提取特征、长序列 chunk、边界标签 | 中，需要外部模型实现 |

## 2. 通用训练流程

1. **定义任务形态**
   - 回归：下一步预测、轨迹预测、健康状态预测，损失用 MSE/MAE/Huber。
   - 分类：活动识别、情绪识别、故障分类，损失用 CrossEntropy/Focal Loss。
   - 控制：行为克隆、模仿学习、导航策略，损失用 MSE、NLL、MDN NLL，最终必须闭环评估。
   - 检测/分割：视频边界检测、图像 mask refinement，损失用 BCE/Dice/IoU/mAP 相关指标。

2. **构建数据集**
   - 每条样本统一为 `x[t:t+seq_len] -> y[t+seq_len:t+seq_len+horizon]` 或 `episode -> action sequence`。
   - 只用训练集统计量做标准化，避免验证集和测试集信息泄漏。
   - 非平稳任务优先按时间切分，控制任务按场景/轨迹切分，医疗任务按 subject/patient 切分。
   - 对不规则采样数据保留 `delta_t` 和 `mask`。如果使用本仓库当前 from-scratch `CfCNetwork`，先重采样为等间隔；若要保留真实时间间隔，应扩展 `forward(..., dt)` 或使用官方 `ncps`/`raminmh/CfC` 实现。

3. **搭建架构**
   - 起步基线：`CfCNetwork(input_size, hidden_size=32/64, output_size)`，速度快，适合作为第一版。
   - 动力学更强：`LTCNetwork(..., ode_method="rk4")`，适合概念漂移、物理系统和可解释动力学分析。
   - 控制与可解释：`NCPSAutoNCP` 或 `AutoNCP(units, output_size)`，用稀疏接线减少参数并便于审计。
   - 多模态：图像/文本/传感器先经专用 encoder，再把上下文拼到每个时间步输入 LNN。
   - 长序列：优先看 Liquid-S4 或 LiquidTAD 的并行 liquid-inspired 算子，不建议直接用逐步 ODE 求解长视频。

4. **训练与验证**
   - 默认优化器：AdamW 或 Adam，`lr=1e-3` 起步，复杂编码器可降到 `3e-4`。
   - 必开：gradient clipping，建议 `max_norm=1.0`；early stopping；固定随机种子；多次重复。
   - 基线必须包含 LSTM/GRU，长序列方向再加 S4/Mamba/Transformer 类基线。
   - 评估同时看任务指标、参数量、训练时间、推理延迟、OOD/概念漂移退化率。

5. **优化与调整**
   - 优先搜索：`seq_len`、`horizon`、`hidden_size`、层数、学习率、batch size。
   - LNN 专属搜索：`delta_t` 表达、LTC ODE solver、`tau` 初值/约束、AutoNCP sparsity、CfC 是否 mixed-memory。
   - 鲁棒性增强：时间 jitter、随机缺失 mask、噪声增强、domain holdout、regime holdout。
   - 边缘部署：先用 CfC 或 Euler-LTC，再做蒸馏、剪枝、int8 量化和 Pareto 筛选。

## 3. 各方向报告

- [[docs/reports/LNN_训练方向_核心架构与通用流程_可行报告|核心架构与通用流程]]
- [[docs/reports/LNN_训练方向_非平稳时间序列与医疗金融_可行报告|非平稳时间序列、医疗与金融]]
- [[docs/reports/LNN_训练方向_机器人控制与模仿学习_可行报告|机器人控制与模仿学习]]
- [[docs/reports/LNN_训练方向_边缘部署与压缩_可行报告|边缘部署与压缩]]
- [[docs/reports/LNN_训练方向_图时空与通信系统_可行报告|图时空与通信系统]]
- [[docs/reports/LNN_训练方向_长序列与视频理解_可行报告|长序列与视频理解]]

## 4. 推荐落地路线

第一阶段：用本项目现有代码完成 `CfC/LTC vs LSTM/GRU` 的时间序列复现实验，重点验证窗口长度、OOD、概念漂移。入口：`scripts/benchmark_comparison.py`、`scripts/experiment_ood.py`、`scripts/experiment_concept_drift.py`。

第二阶段：引入官方 `ncps` 的 `AutoNCP` 和 `CfC/LTC`，补齐稀疏接线和不规则时间间隔能力。入口：`lnn/ncps_integration/ncps_models.py`、`scripts/experiment_autoncp.py`。

第三阶段：按应用方向新增数据加载器。时间序列优先接入 PhysioNet、UCI HAR 或能源价格数据；机器人方向优先接入 RoboMimic/PointMaze；边缘方向复用 Jetson benchmark 并加入量化。

第四阶段：形成可比较的实验表，统一记录 `RMSE/MAE/Accuracy/mAP`、参数量、训练秒、推理延迟、OOD 退化率和硬件能耗。

## 5. 关键论文与资料

- Ramin Hasani 等，*Liquid Time-constant Networks*，arXiv:2006.04439，https://arxiv.org/abs/2006.04439
- Ramin Hasani 等，*Closed-form continuous-time neural networks*，Nature Machine Intelligence 2022，https://www.nature.com/articles/s42256-022-00556-7
- Mathias Lechner 等，*Neural circuit policies enabling auditable autonomy*，Nature Machine Intelligence 2020，https://www.nature.com/articles/s42256-020-00237-3
- Ramin Hasani 等，*Liquid Structural State-Space Models*，arXiv:2209.12951，https://arxiv.org/abs/2209.12951
- `ncps` 官方文档，https://ncps.readthedocs.io/
- `raminmh/CfC` 官方实现，https://github.com/raminmh/CfC
- 本仓库每日追踪：[[docs/daily/2026-05-26_LNN_research_digest]]
