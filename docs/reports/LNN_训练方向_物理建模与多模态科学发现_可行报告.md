---
title: LNN 训练方向：物理建模与多模态科学发现可行报告
date: 2026-05-28
tags: [LNN, physics-informed, multimodal, scientific-discovery, LTC]
---

# LNN 训练方向：物理建模与多模态科学发现可行报告

## 1. 方向定位

该方向关注从视频、音频、传感器、图表或实验记录中学习连续时间物理动态，并反推出可解释参数。LNN 在这里不是单纯做分类器，而是作为 latent dynamics model：用 LTC/CfC 学系统状态演化，再用物理方程残差、多任务回归或 rollout 一致性约束训练。

检索证据：本方向纳入/暂缓记录见 [[docs/LNN_训练论文检索矩阵_2026-05-28]]。

## 2. 代表论文

| 论文 | 任务 | 关键启发 |
|---|---|---|
| *EMMA: Extracting Multiple physical parameters from Multimodal Data* | 从 video/audio/image time-series 中恢复多个物理参数 | LTC 学 latent dynamics，physics-constrained loss 约束显式参数、隐式动态和校准不变量 |
| *Recovering implicit physics model under real-world constraints* | 从真实约束数据恢复隐式物理模型 | LTC-NN 可作为 SINDy/PINN 类方法之外的动态参数恢复器 |
| *Physics-Modeled Neural Networks* | ODE layer 与 CfC/Neural ODE 对比 | 将物理可解释 dynamical system 嵌入网络层，训练权重和动力学参数 |
| *LSS-LTCNet* | 医学图像 mask refinement | 把 LTC 当作连续时间迭代 refinement 模块，适合边界/形状随时间演化的问题 |

## 3. 数据集构建方案

### 3.1 多模态物理观测格式

```text
sample_id
time: [T]
modalities:
  video_features: [T, F_v] 或 raw_frames: [T, C, H, W]
  audio_features: [T, F_a] 可选
  sensor_features: [T, F_s] 可选
  chart_features: [T, F_c] 可选
targets:
  explicit_params: mass, length, damping, stiffness, friction, ...
  hidden_inputs: force, actuation, disturbance 可选
  rollout_state: position, velocity, angle, ...
physics:
  equation_id
  known_constants
  constraint_mask
metadata:
  scene, camera_calibration, sampling_rate, occlusion_level, noise_level
```

关键处理：

- 所有模态必须对齐到统一时间轴；不能对齐时要保留 `dt` 和 missing mask。
- 如果真实物理参数可得，做 supervised parameter regression；如果不可得，用 rollout consistency 和 physics residual 做弱监督。
- 按物理系统或场景切分，避免同一系统的近似重复轨迹同时进入训练和测试。
- 对遮挡、噪声、采样率变化、外力输入缺失做 OOD 切分。

### 3.2 合成到真实的最小路线

1. 合成 pendulum / spring-mass / Lorenz / damped oscillator，参数随机采样。
2. 生成状态序列、渲染简化视频或提取 chart features。
3. 训练 LNN 从观测恢复参数，并预测未来 rollout。
4. 再迁移到真实 rover、quadrotor、设备振动或医学边界演化数据。

## 4. 架构搭建方案

### 4.1 Encoder + LTC latent dynamics

```text
video/audio/sensor encoder
-> aligned feature sequence z_t
-> LTC/CfC latent dynamics
-> parameter head + rollout head
```

推荐：

- 第一版用预提取特征，避免端到端训练视觉 backbone。
- 需要高吞吐时用 CfC；需要更强物理可解释时用 LTC。
- 显式参数用 MLP head，隐式动态用 rollout head。

### 4.2 Physics-constrained training

```text
loss = parameter_loss
     + alpha * rollout_loss
     + beta * physics_residual_loss
     + gamma * calibration_or_invariant_loss
```

其中：

- `parameter_loss`：已知物理参数的 MSE/Huber。
- `rollout_loss`：未来状态预测误差。
- `physics_residual_loss`：将预测状态代入已知 ODE/PDE 后的残差。
- `calibration_or_invariant_loss`：长度、能量、守恒量或单位约束。

### 4.3 Continuous refinement module

适合图像分割、边界跟踪或轨迹 refine：

```text
initial estimate
-> local/self-similarity or image features
-> LTC refinement steps
-> refined mask / trajectory / state
```

这里 LNN 不必建模完整系统，只承担可解释迭代修正器角色。

## 5. 训练方法

推荐配置：

```text
optimizer: AdamW
lr: 3e-4 起步
batch_size: 8 到 32，取决于视觉 encoder
seq_len: 16, 32, 64
gradient_clip: 1.0
ode_method: CfC 无 solver；LTC 先 euler，再用 rk4 做精度验证
loss_weights: alpha/beta/gamma 先从 1.0/0.1/0.01 扫描
```

训练顺序：

1. 只训练 encoder + parameter head，确认参数监督链路可学。
2. 加 LNN latent dynamics 和 rollout loss。
3. 加 physics residual，观察是否提升 OOD 参数恢复，而不是只降低训练集 loss。
4. 冻结或半冻结 encoder，重点调 LNN hidden size、`dt` 和 loss weights。
5. 加 OOD 测试：未见参数区间、噪声、遮挡、采样率变化、外力缺失。

## 6. 优化与调参

重点：

- `dt` 表达：真实采样间隔、归一化时间、log-dt 至少比较一种。
- loss 权重：physics residual 太大可能压制数据拟合，太小则无约束效果。
- hidden size：从 16/32 起步，先看参数误差和 rollout 稳定性。
- 参数范围：训练集参数采样必须覆盖目标场景；额外报告 extrapolation。
- 模态 dropout：训练时随机丢弃 audio/video/sensor，提高鲁棒性。
- 单位一致性：所有物理量必须统一单位，否则 physics loss 会误导训练。

评估指标：

- 参数恢复：MAE、relative error、置信区间。
- 动态预测：rollout RMSE、长期漂移、稳定性。
- 物理一致性：equation residual、能量/守恒量误差。
- 鲁棒性：噪声、遮挡、缺失模态、未见参数区间下的退化率。
- 工程：训练秒、推理延迟、模型参数量。

## 7. 本项目落地建议

短期：

- 新增 `lnn/data/physics.py`，先生成 pendulum/spring-mass/Lorenz 参数化数据。
- 新增 `scripts/experiment_physics_lnn.py`，比较 CfC/LTC/GRU 在参数恢复和 rollout 上的表现。
- 输出到 `analysis/physics/`，记录参数误差、rollout 图和 OOD 退化率。

中期：

- 加入简化视频或图表特征，将 `lnn/core/multimodal.py` 的融合方式复用到物理任务。
- 实现 physics residual loss，并把 loss weights 写入 `configs/physics_lnn.yaml`。
- 选 EMMA 或 implicit physics paper 做一篇独立研读报告。

## 8. 可行结论

该方向研究价值高，但第一版应从合成物理系统开始，不应直接做完整 CVPR 级多模态系统。最务实路线是 `synthetic dynamics -> CfC/LTC parameter recovery -> physics residual -> OOD test`。一旦该链路稳定，再接真实视频、音频或设备传感器数据。

## 9. 参考资料

- *EMMA: Extracting Multiple physical parameters from Multimodal Data*：https://arxiv.org/abs/2605.24047v1
- *Recovering implicit physics model under real-world constraints*：https://arxiv.org/abs/2412.02215
- *Physics-Modeled Neural Networks*：https://arxiv.org/abs/2605.08176
- *Explainable Continuous-Time Mask Refinement with Local Self-Similarity Priors for Medical Image Segmentation*：https://arxiv.org/abs/2603.00459
- *Physics Informed Deep Learning (Part I)*：https://arxiv.org/abs/1711.10561
