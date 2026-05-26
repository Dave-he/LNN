---
title: LNN 训练方向：核心架构与通用流程可行报告
date: 2026-05-26
tags: [LNN, CfC, LTC, NCP, training]
---

# LNN 训练方向：核心架构与通用流程可行报告

## 1. 方向定位

该方向解决“先把 LNN 正确训练起来”的问题。推荐把 CfC 作为第一基线，把 LTC 作为动力学验证模型，把 AutoNCP 作为稀疏可解释控制模型。

## 2. 代表论文

| 论文 | 价值 | 对训练的启发 |
|---|---|---|
| *Liquid Time-constant Networks* | 提出输入依赖的动态时间常数，用 ODE solver 计算输出 | 适合非平稳序列，但训练成本高，需要控制 solver 和梯度 |
| *Closed-form continuous-time neural networks* | 用闭式近似避免 ODE 数值求解 | 适合工程首选基线，速度和稳定性更好 |
| *Neural circuit policies enabling auditable autonomy* | 用 19 个控制神经元完成自动驾驶子任务 | 稀疏接线有利于解释、参数效率和边缘部署 |
| `ncps` 文档 | 提供 PyTorch/TensorFlow 的 LTC/CfC/AutoNCP | 可以直接作为项目官方实现路径 |

## 3. 数据集构建

通用输入格式：

```text
X: [batch, time, feature]
y: [batch, target_dim] 或 [batch, horizon, target_dim]
dt: [batch, time]，仅在保留不规则采样间隔时需要
mask: [batch, time, feature]，仅在存在缺失值时需要
```

最小可行数据集：

1. 合成序列：sine、Mackey-Glass、Lorenz，用于验证训练链路和动力学可视化。
2. 真实序列：PhysioNet 2012、UCI HAR、能源价格或传感器数据，用于验证缺失值、噪声和 OOD。
3. 控制轨迹：`episode = {obs_t, action_t, reward_t, done_t, timestamp_t}`，用于行为克隆或策略学习。

切分规则：

- 时间序列按时间前后切分，禁止随机打乱整段序列。
- 医疗/人体传感器按患者或被试切分，避免同一主体同时出现在训练和测试。
- 控制任务按场景、天气、地图或轨迹切分，保留 OOD holdout。

## 4. 架构搭建

### 4.1 CfC 基线

适合先跑通训练。输入经 CfC cell 逐时间步更新隐藏状态，再接 `Linear(hidden, output)`。

推荐配置：

```text
hidden_size: 32, 64, 128
num_layers: 1 起步，复杂任务再试 2
lr: 1e-3 或 3e-4
gradient_clip: 1.0
batch_size: 32 或 64
```

本项目入口：

- `lnn/core/cfc.py`
- `scripts/experiment_timeseries.py`
- `scripts/benchmark_comparison.py`

### 4.2 LTC 动力学模型

LTC 使用 ODE solver，更接近原始液态动力学。适合概念漂移、物理系统和可解释性分析。

推荐配置：

```text
ode_method: euler 用于快速 smoke test，rk4 用于正式实验
hidden_size: 16, 32, 64
tau_base: 保持正值约束
gradient_clip: 1.0
```

注意点：

- LTC 比 CfC 慢，先小模型验证，再扩大隐藏维度。
- `dopri5` 等自适应 solver 可能更准，但训练吞吐更低。
- 对长序列不要直接无脑堆 LTC，先缩短窗口或改用 CfC/Liquid-S4。

本项目入口：

- `lnn/core/ltc.py`
- `scripts/experiment_concept_drift.py`

### 4.3 AutoNCP 稀疏接线

AutoNCP 用总神经元数和输出维度生成 NCP 接线。官方 `ncps` 文档说明，NCP 是 sensory、inter、command、motor 四层 recurrent 连接原则，`AutoNCP(units, output_size)` 是最简单入口。

推荐配置：

```text
units: 16, 28, 64
output_size: 任务动作维度或分类维度
sparsity_level: 0.3 到 0.7
cell: CfC 起步，LTC 用于可解释动力学
```

本项目入口：

- `lnn/ncps_integration/ncps_models.py`
- `scripts/experiment_autoncp.py`

## 5. 训练方法

标准训练循环：

1. 构建 sliding window 或 episode batch。
2. 前向得到最后一步输出或全序列输出。
3. 计算任务损失。
4. 反向传播。
5. 梯度裁剪。
6. optimizer step。
7. 验证集 early stopping。

推荐损失：

| 任务 | 损失 | 指标 |
|---|---|---|
| 单步/多步预测 | MSE、MAE、Huber | RMSE、MAE、MAPE |
| 分类 | CrossEntropy、Focal Loss | Accuracy、F1、AUROC |
| 行为克隆 | MSE、NLL、MDN NLL | offline MSE/NLL、closed-loop success |
| 多标签/事件检测 | BCE、Focal Loss | mAP、F1、IoU |

## 6. 调参与优化

优先级最高的调参项：

1. `seq_len`：决定模型能看到的历史范围，常用 16、32、64、128。
2. `hidden_size`：LNN 不一定越大越好，先扫 16 到 128。
3. `lr`：从 `1e-3` 起步，若 loss 震荡降到 `3e-4` 或 `1e-4`。
4. `tau`/`dt`：不规则采样任务必须正确传入时间间隔。
5. `sparsity_level`：控制 AutoNCP 参数量和可解释性。
6. solver：LTC 中 `euler` 快，`rk4` 稳，正式报告需要记录 solver。

稳定训练建议：

- 使用 train-only 标准化。
- 开启 gradient clipping。
- 对 OOD 任务单独记录 ID 与 OOD 指标，不能只报平均测试集。
- 每个配置至少 3 个 seed。
- 和 LSTM/GRU 同预算对比，否则无法判断 LNN 是否真的有效。

## 7. 可行结论

本方向已具备较高可行性。仓库已有 CfC/LTC from-scratch 实现、`ncps` 集成、通用 trainer、时间序列数据和多个实验脚本。下一步应优先补齐两个能力：不规则时间间隔 `dt` 的端到端传递，以及官方 `ncps` 与本项目 trainer 的统一训练接口。

## 8. 参考资料

- *Liquid Time-constant Networks*：https://arxiv.org/abs/2006.04439
- *Closed-form continuous-time neural networks*：https://www.nature.com/articles/s42256-022-00556-7
- *Neural circuit policies enabling auditable autonomy*：https://www.nature.com/articles/s42256-020-00237-3
- `ncps` wirings 文档：https://ncps.readthedocs.io/en/latest/api/wirings.html
- `raminmh/CfC`：https://github.com/raminmh/CfC
