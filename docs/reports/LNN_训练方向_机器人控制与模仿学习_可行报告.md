---
title: LNN 训练方向：机器人控制与模仿学习可行报告
date: 2026-05-28
tags: [LNN, robotics, imitation-learning, NCP, control]
---

# LNN 训练方向：机器人控制与模仿学习可行报告

## 1. 方向定位

机器人控制是 LNN 的经典应用方向。优势在于低参数、连续时间、可解释和对环境扰动的潜在鲁棒性。典型任务包括自动驾驶 lane keeping、无人机导航、机械臂操作、移动机器人避障和 PointMaze 导航。

检索证据：本方向纳入/暂缓记录见 [[docs/LNN_训练论文检索矩阵_2026-05-28]]。

## 2. 代表论文与资料

| 论文或资料 | 任务 | 关键启发 |
|---|---|---|
| *Neural circuit policies enabling auditable autonomy* | 自动驾驶子任务 | 19 个控制神经元将输入特征映射到转向命令，可解释且参数极少 |
| *Liquid Networks with Mixture Density Heads for Efficient Imitation Learning* | Push-T、RoboMimic Can、PointMaze | LNN + MDN head 相比 diffusion policy 更小、更快，在低数据下更稳 |
| `ncps` AutoNCP | 稀疏神经电路接线 | 用少量参数构建 sensory/inter/command/motor 结构 |
| RoboMimic | 机器人模仿学习数据与工具 | HDF5 demonstrations，含低维状态、图像、动作等 |
| PointMaze/Minari | 离线导航轨迹 | 连续轨迹、随机目标、PD controller 数据生成 |

## 3. 数据集构建方案

### 3.1 episode 格式

建议统一为：

```text
episode_id
obs:
  proprio: [T, P]
  image: [T, C, H, W] 可选
  lidar/depth: [T, ...] 可选
actions: [T, A]
rewards: [T]
dones: [T]
timestamps: [T]
metadata:
  scene, weather, object_id, demonstrator_id, success
```

行为克隆样本：

```text
X = obs[t : t + context_len]
y = actions[t : t + action_horizon]
```

### 3.2 数据收集与清洗

- 轨迹必须包含成功和失败标记，闭环评估时只看成功轨迹会高估效果。
- 图像和动作需要时间对齐，延迟最好显式记录。
- 对动作做 clipping 和单位统一，避免不同控制器尺度混乱。
- 按场景或对象切分 OOD，例如未见过的地图、物体、光照或动态障碍。
- 对 demonstration 质量分层，保留 expert、mixed-human、machine-generated 标签。

### 3.3 可直接复用的数据源

- RoboMimic：适合机械臂操作，官方提供下载脚本和 HDF5 数据。
- PointMaze/Minari：适合离线导航和低维控制。
- Push-T：适合 2D 操作与多模态动作分布验证。

## 4. 架构搭建方案

### 4.1 低维状态控制

```text
proprio/state sequence
-> CfC/LTC/AutoNCP
-> action head
```

推荐：

- 第一版：CfC + MSE action head。
- 稀疏可解释：AutoNCP-CfC。
- 动力学更强：LTC，但要控制训练速度。

### 4.2 图像控制

```text
image_t -> CNN encoder -> z_img_t
proprio_t -> MLP encoder -> z_prop_t
concat(z_img_t, z_prop_t)
-> CfC/LTC/AutoNCP
-> action head
```

注意：

- CNN 只做感知特征提取，LNN 负责时间记忆和控制输出。
- 图像模型训练初期可以冻结 encoder，先让 LNN 学控制。
- 若数据少，优先使用预训练视觉 encoder 或低维状态版本。

### 4.3 多模态动作分布

模仿学习常有多解动作，单一 MSE 会平均多个可行动作。可使用 MDN head：

```text
LNN hidden -> mixture weights, means, variances
loss = negative log likelihood(action | mixture)
```

适合：

- Push-T 类多峰策略。
- RoboMimic 中有多示范者的任务。
- 需要采样多个候选动作的闭环控制。

## 5. 训练方法

基础行为克隆：

```text
loss = MSE(pred_action, expert_action)
```

多峰行为克隆：

```text
loss = -log sum_k pi_k * Normal(action; mu_k, sigma_k)
```

推荐配置：

```text
context_len: 8, 16, 32
action_horizon: 1, 4, 8, 16
hidden_size: 32, 64, 128
lr: 3e-4
batch_size: 64
gradient_clip: 1.0
```

闭环评估必须包含：

- success rate。
- collision rate 或 failure type。
- trajectory length。
- control smoothness。
- inference latency。

## 6. 优化与调参

关键调参：

- `context_len`：太短看不到动力学，太长训练慢。
- `action_horizon`：短 horizon 稳定，长 horizon 可减少闭环误差累积。
- MDN mixture 数：从 5 或 10 起步。
- AutoNCP `units` 和 `sparsity_level`：先小网络，再扩大。
- 数据分层采样：低频失败场景要提高采样权重。

增强策略：

- action noise augmentation：提高闭环鲁棒性。
- observation dropout：模拟传感器异常。
- DAgger 或 failure mining：把闭环失败重新加入训练集。
- controller gain 记录：position-controlled 机器人中，动作误差会被底层控制器放大或抑制。

## 7. 本项目落地建议

短期：

- 用 `scripts/experiment_autoncp.py` 跑通 AutoNCP 与 Dense-CfC 对比。
- 新增 `lnn/data/robotics.py`，支持 RoboMimic HDF5 的 `obs/actions/dones`。
- 写 `scripts/experiment_imitation_lnn.py`，先支持低维状态行为克隆。

中期：

- 加 MDN action head。
- 加闭环评估 wrapper。
- 在 Jetson 上测试低维控制模型延迟。

## 8. 可行结论

该方向可行，但不能只看 offline loss。LNN 在低参数和低数据场景有优势，最终价值必须通过闭环 success rate、延迟和 OOD 场景来证明。

## 9. 参考资料

- *Neural circuit policies enabling auditable autonomy*：https://www.nature.com/articles/s42256-020-00237-3
- *Liquid Networks with Mixture Density Heads for Efficient Imitation Learning*：https://arxiv.org/abs/2603.27058
- RoboMimic 文档：https://robomimic.github.io/docs/
- RoboMimic 数据下载说明：https://github.com/ARISE-Initiative/robomimic/blob/master/docs/datasets/robomimic_v0.1.md
- PointMaze/Minari：https://minari.farama.org/v0.3.1/datasets/pointmaze/
- `ncps` 文档：https://ncps.readthedocs.io/
