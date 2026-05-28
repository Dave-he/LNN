---
title: LNN 训练方向：非平稳时间序列与医疗金融可行报告
date: 2026-05-28
tags: [LNN, time-series, medical, finance, dataset]
---

# LNN 训练方向：非平稳时间序列与医疗金融可行报告

## 1. 方向定位

该方向是 LNN 最容易落地的方向。核心价值是处理非平稳、含噪、缺失、不规则采样或 regime change 的序列数据。金融价格、能源负载、医疗 ICU、EEG、生物信号和电池健康都属于这一类。

检索证据：本方向纳入/暂缓记录见 [[docs/LNN_训练论文检索矩阵_2026-05-28]]。

## 2. 代表论文与数据源

| 论文或数据源 | 任务 | 关键启发 |
|---|---|---|
| *Liquid Neural Network Models for Natural Gas Spot Price Time-Series Forecasting* | 天然气现货价格短期预测 | 重点处理价格波动和市场机制切换 |
| *Adaptive Temporal Dynamics for Personalized Emotion Recognition* | EEG 与生理信号情绪识别 | LNN 可学习快慢时间常数并与注意力/自编码器融合 |
| *Modeling Retinal Ganglion Cells with Neural Differential Equations* | 视网膜神经节细胞响应建模 | LTC/CfC 在小数据、频繁重训场景有优势 |
| *When Smaller Wins* 与 *EntroLnn* | 电池健康预测 | 适合在线更新、蒸馏、压缩和边缘部署 |
| PhysioNet 2012 | ICU 死亡率预测 | 官方数据含 12,000 ICU stays、48 小时内不规则时间序列 |
| UCI HAR | 人体活动识别 | 30 名被试、手机加速度计和陀螺仪、50Hz 采样 |

## 3. 数据集构建方案

### 3.1 单变量或多变量预测

适合天然气价格、负载、电池 SoH、传感器读数。

```text
raw rows:
timestamp, target, covariate_1, covariate_2, ...

sample:
X = rows[t : t + seq_len, features]
y = rows[t + seq_len : t + seq_len + horizon, target]
```

关键处理：

- 按时间排序，去重，统一时区。
- 只用训练段统计量标准化。
- 外生变量必须按预测时点可获得性做滞后处理，避免未来信息泄漏。
- 对缺失值同时保留 imputed value 和 missing mask。
- 对多步预测明确 `horizon=1/7/24/100 cycles`，不要混用指标。

### 3.2 不规则医疗时间序列

适合 PhysioNet、ICU、电子病历和穿戴设备。

```text
X_value: [T, F]
X_mask:  [T, F]
X_dt:    [T]
static:  age, gender, device, subject metadata
y:       mortality / diagnosis / emotion / event
```

可选处理：

- 方案 A：重采样到固定间隔，缺失填充加 mask。工程简单，适合本项目当前 CfC/LTC 实现。
- 方案 B：保留真实 `delta_t`，使用官方 CfC/LTC 或扩展本项目模型，适合正式论文复现。

### 3.3 OOD 与概念漂移构造

必须单独构造验证集：

- 时间 OOD：训练 2015-2022，验证 2023，测试 2024-2025。
- regime holdout：高波动日、政策事件日、极端天气日单独作为 OOD。
- subject holdout：某些患者、被试或设备完全不进训练集。
- noise stress：在测试阶段增加噪声、随机缺失、采样间隔变化。

本项目已有合成入口：

- `lnn/data/timeseries.py`
- `scripts/experiment_ood.py`
- `scripts/experiment_concept_drift.py`

## 4. 架构搭建方案

### 4.1 标准时序预测

```text
Input features -> CfC/LTC -> last hidden -> Linear -> target horizon
```

推荐：

- 初始模型：CfC，`hidden_size=32/64`。
- 漂移明显：LTC 或 CfC + mixed memory。
- 长预测窗：encoder + CfC + direct multi-horizon head。

### 4.2 医疗与生理信号分类

```text
per-channel normalization
-> CNN/MLP feature extractor
-> CfC/LTC temporal model
-> attention pooling or last hidden
-> classifier
```

推荐：

- EEG：先用 1D CNN 或频带特征编码，再输入 LNN。
- ICU：静态变量与动态变量分支编码，最后 concat。
- 情绪识别：按 subject 切分，报告 subject-dependent 与 subject-independent 两种结果。

### 4.3 电池与工业健康预测

```text
cycle features / temperature entropy / voltage curve
-> CfC/LTC teacher
-> distilled student CfC/Euler-LTC
-> SoH/EoL/CFT output
```

推荐：

- 先训练高容量 teacher。
- 再训练小型 student 模仿 teacher 的轨迹或隐藏状态。
- 用误差、模型大小、延迟做 Pareto 筛选。

## 5. 训练方法

基础配置：

```text
optimizer: AdamW
lr: 1e-3 起步，复杂多模态模型用 3e-4
batch_size: 32/64
seq_len: 32/64/128
gradient_clip: 1.0
early_stopping: patience 10 到 20
```

损失函数：

- 预测：MSE + MAE 报告，Huber 用于尖峰噪声。
- 分类：CrossEntropy，类别不平衡时加 class weight 或 Focal Loss。
- 电池轨迹：点预测 loss + 曲线形状 loss，必要时加 monotonicity penalty。
- 不确定性：分位数损失或高斯 NLL。

评估指标：

- 预测：RMSE、MAE、MAPE、sMAPE。
- 分类：Accuracy、Macro-F1、AUROC、AUPRC。
- OOD：`OOD_metric / ID_metric` 或退化率。
- 资源：参数量、训练秒、推理延迟、峰值显存。

## 6. 优化与调参

优先搜索：

| 参数 | 搜索范围 | 判断标准 |
|---|---|---|
| `seq_len` | 16, 32, 64, 128, 256 | 验证集与 OOD 同时改善才保留 |
| `hidden_size` | 16, 32, 64, 128 | 看参数效率，不只看最小 loss |
| `horizon` | 1, 3, 7, 24, 100 cycles | 每个 horizon 单独报告 |
| `lr` | 1e-3, 3e-4, 1e-4 | loss 震荡时降低 |
| `ode_method` | euler, rk4 | LTC 正式实验建议至少比较一次 |
| `dt` encoding | fixed, real delta, log delta | 不规则采样必须比较 |

增强策略：

- time masking：随机遮蔽部分时间步。
- jitter：轻微扰动时间间隔或输入噪声。
- regime-balanced sampling：高波动样本过少时增加采样权重。
- teacher-student：用于边缘预测或小数据场景。

## 7. 可行结论

该方向应作为项目第一优先级。仓库已经有 synthetic time-series、OOD、concept drift 和 benchmark 脚本，能够快速形成可复现实验。正式推进时建议先接入一个公开真实数据集，如 PhysioNet 2012 或 UCI HAR，再加入一个领域数据集，如天然气价格或电池健康。

## 8. 参考资料

- *Liquid Neural Network Models for Natural Gas Spot Price Time-Series Forecasting*：https://arxiv.org/abs/2604.24788
- *Adaptive Temporal Dynamics for Personalized Emotion Recognition*：https://arxiv.org/abs/2602.06997
- *Modeling Retinal Ganglion Cells with Neural Differential Equations*：https://arxiv.org/abs/2511.18014
- *When Smaller Wins*：https://arxiv.org/abs/2601.06227
- *EntroLnn*：https://arxiv.org/abs/2601.06195
- PhysioNet 2012：https://www.physionet.org/content/challenge-2012/1.0.0/
- UCI HAR：https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
