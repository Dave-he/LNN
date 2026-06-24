---
title: "Fast Human Attention Prediction for Fixation-guided Active Perception in Autonomous Navigation"
arxiv_id: "2606.20491v1"
date: "2026-06-18"
authors: "Fatma Youssef Mohammed, Grzegorz Malczyk, Kostas Alexis"
tags: [LNN, CfC, robotics, active-perception, gaze-prediction, scanpath, Jetson, autonomous-navigation]
primary_anchor: "https://arxiv.org/abs/2606.20491v1"
pdf: "https://arxiv.org/pdf/2606.20491"
local_pdf: "papers/daily/2026-06-24/2026-06-18_GazeLNN_Fast_Human_Attention_Prediction_2606.20491.pdf"
report_date: "2026-06-24"
analyst: "LNN Daily Researcher (paper-analyzer SOP, arXiv PDF)"
---

# GazeLNN — 研读报告

> 本文把 CfC/LNN 用作轻量 scanpath prediction 的 recurrent engine，再把预测到的人类注视热力图接入强化学习主动相机策略，完成无人机真机导航验证。它是今日最贴近"边缘 LNN + 机器人主动感知"路线的论文。

## 1. 元数据

| 字段 | 值 |
|---|---|
| 标题 | Fast Human Attention Prediction for Fixation-guided Active Perception in Autonomous Navigation |
| 作者 | Fatma Youssef Mohammed, Grzegorz Malczyk, Kostas Alexis |
| 单位 | Norwegian University of Science and Technology (NTNU) |
| 时间 | 2026-06-18（arXiv:2606.20491v1） |
| 链接 | https://arxiv.org/abs/2606.20491v1 |
| PDF | https://arxiv.org/pdf/2606.20491 |
| 本地归档 | `papers/daily/2026-06-24/2026-06-18_GazeLNN_Fast_Human_Attention_Prediction_2606.20491.pdf` |
| 标签 | Liquid Neural Networks, CfC, scanpath prediction, active perception, aerial robot, Jetson Orin NX |

## 2. 核心问题

自主机器人处理全视野高分辨率视觉数据成本很高，尤其是无人机等边缘平台。人类视觉通过 saccade / fixation scanpath 选择性关注显著区域，用低成本获得关键信息。已有 scanpath prediction 模型精度提高明显，但常依赖 Transformer 或重 ConvLSTM，计算成本和延迟不适合实时机器人。

论文要解决的问题是：

1. 能否用轻量 LNN/CfC 模块预测人类注视 scanpath，保持 SOTA 级指标但大幅降计算量？
2. 预测到的 fixation heatmap 是否能作为 active perception 信号，驱动无人机主动转动相机，提升导航时的空间感知覆盖？

## 3. 方法论与核心思路

### 3.1 GazeLNN 架构

GazeLNN 自回归预测 fixation scanpath。每一步用当前图像特征、上一 fixation heatmap 和 hidden state 预测下一 fixation heatmap。

核心流水线：

1. 输入图像 resize 到 $256 \times 384$。
2. MobileNetV3 提取视觉特征，选择原因是嵌入式友好且计算开销低。
3. fixation heatmap 经过 CoordConv 增加显式 $x/y$ 坐标通道。
4. 图像特征、fixation 表征和 hidden state 拼接后送入 CfC recurrent module。
5. 输出投影成下采样 fixation map，再上采样为原图大小的 heatmap。
6. heatmap 最大值位置作为下一 fixation，并反馈进入下一步。

论文选择 CfC 而不是 LTC，是因为 CfC 相比 LTC 训练和推理更快。

### 3.2 CfC 更新公式

CfC 模块使用共享全连接 backbone（1024 units + LeCunTanh），再接四个 512-unit head：$f_1,f_2,t_a,t_b$。hidden state 更新为：

$$
h_{i+1} = (1-\sigma(t_a\Delta t+t_b))\odot\tanh(f_1(x_t)) + \sigma(t_a\Delta t+t_b)\odot\tanh(f_2(x_t))
$$

其中：

- $\Delta t$ 是 fixation 之间的 elapsed time；训练时来自 ground-truth fixation duration；
- 部署到主动相机策略时没有真实 fixation duration，因此固定 $\Delta t=1$，对应 CfC PyTorch 默认设置；
- gate $\sigma(t_a\Delta t+t_b)$ 决定两个候选状态的插值比例。

### 3.3 RL 主动感知策略

为了验证 scanpath prediction 的机器人价值，作者将 GazeLNN 预测的 fixation heatmap 接入主动相机控制策略。策略输出：

$$
a_t = [v_t^r, \omega_{t,z}^r, c_t^r]
$$

其中 $v_t^r$ 是机体系线速度，$\omega_{t,z}^r$ 是 yaw rate，$c_t^r=\{\chi_t^r,\psi_t^r\}$ 是相机 pitch/yaw 指令。

奖励函数：

$$
R(s_t,a_t)=r_t+l_t+p_t+h_t
$$

各项含义：

- $r_t=w_r(d_{t-1}-d_t)$：接近目标 waypoint 的进度奖励；
- $l_t=-w_l\|a_t-a_{t-1}\|^2$：机器人和相机动作平滑惩罚；
- $p_t$：障碍物接近惩罚；
- $h_t$：fixation-attraction 奖励，鼓励相机中心对准 GazeLNN 预测的显著区域。

fixation-attraction 项定义为：

$$
h_t = w_h \frac{\sum_{u,v}H_t(u,v)\exp(-\alpha d(u,v)^2)}{\sum_{u,v}H_t(u,v)>0}
$$

它将热力图中越靠近图像中心的显著区域奖励越高，从而驱动 pan-tilt 相机主动追踪人类可能关注的位置。

## 4. 核心公式提取

| 公式 | 说明 |
|---|---|
| $h_{i+1}=(1-\sigma(t_a\Delta t+t_b))\odot\tanh(f_1(x_t))+\sigma(t_a\Delta t+t_b)\odot\tanh(f_2(x_t))$ | GazeLNN/CfC recurrent update |
| $a_t=[v_t^r,\omega_{t,z}^r,c_t^r]$ | 导航 + 主动相机联合 action |
| $R(s_t,a_t)=r_t+l_t+p_t+h_t$ | RL 主动感知总奖励 |
| $r_t=w_r(d_{t-1}-d_t)$ | waypoint progress reward |
| $l_t=-w_l\|a_t-a_{t-1}\|^2$ | motion smoothness penalty |
| $h_t=w_h\frac{\sum_{u,v}H_t(u,v)\exp(-\alpha d(u,v)^2)}{\sum_{u,v}H_t(u,v)>0}$ | fixation-attraction reward |

## 5. 关键成果与贡献

### 5.1 Scanpath prediction 指标

在 MIT Low Resolution dataset 上，GazeLNN 对比人类 baseline、Itti、LeMeur、IOR-ROI、Chen Model、tSPM-Net 等方法：

| 模型 | Levenshtein ↓ | ScanMatch ↑ | Hausdorff ↓ | Frechet ↓ | fast DTW ↓ | TDE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Human BL | 10.77 | 0.38 | 95.97 | 140.02 | 550.84 | 42.40 |
| tSPM-Net | 11.47 | 0.34 | 103.44 | 144.77 | 610.02 | 43.74 |
| **GazeLNN** | **11.22** | **0.47** | **98.17** | **133.31** | **537.72** | **27.20** |

论文摘要中给出的相对提升包括：ScanMatch +34.29%、Levenshtein Distance +2.18%、Hausdorff Distance +5.09%、Frechet Distance +7.92%、FastDTW +11.85%、Time Delay Embedding +37.81%。

### 5.2 计算效率

Backbone ablation 显示 MobileNetV3 在性能接近最优的同时成本最低：

| Backbone | ScanMatch ↑ | Time (ms) ↓ | Params (M) ↓ | GFLOPs ↓ |
|---|---:|---:|---:|---:|
| VGG19+DeepLabV3 | 0.47 | 17.43 | 195.41 | 99.81 |
| ResNet50 | 0.46 | 7.39 | 35.39 | 8.33 |
| **MobileNetV3** | **0.47** | **6.84** | **15.24** | **0.61** |
| DINOv3(ViT-S)* | 0.48 | 8.76 | 31.35 | 8.49 |

相对 tSPM-Net 级重模型，GazeLNN 降低 91.80% trainable parameters、99.40% GFLOPs，并获得 6.42x 推理加速。

### 5.3 Recurrent module 消融

在相同 VGG19+DeepLabV3 backbone 下比较 recurrent module：

| RNN Model | ScanMatch ↑ | Time (ms) ↓ | Params (M) ↓ | GFLOPs ↓ |
|---|---:|---:|---:|---:|
| Bayesian ConvLSTM | 0.34 | 43.90 | 185.92 | 102.51 |
| ConvLSTM | 0.34 | 21.76 | 185.76 | 102.51 |
| **CfC** | **0.47** | **17.43** | 195.41 | **99.81** |

这说明在该 scanpath 任务中，CfC 不只是省算力，也明显提高 ScanMatch。

### 5.4 真机主动感知验证

机器人平台：

- 自研 quadrotor；
- Intel RealSense D455 RGB-D；
- 两轴 pan-tilt，相机 yaw $\pm45^\circ$、pitch $\pm60^\circ$；
- NVIDIA Jetson Orin NX 16GB 运行高层导航；
- PX4 做底层姿态与电机控制；
- GazeLNN onboard real-time，主动相机策略 10 Hz，ROS 通信。

静态相机 vs GazeLNN 主动相机：

| 指标 | Static Camera | Active Camera |
|---|---:|---:|
| Full voxel grid total voxels | 37,067 | 55,524 |
| Fixation grid total voxels | 873 | 6,770 |
| Fixation grid max hit count | 537 | 756 |

结论：主动相机不是简单增加局部点云密度，而是显著扩展环境覆盖；fixation grid 中被观察到的显著体素接近 8 倍增加。

## 6. 与 LNN 主线的关系

本文是本仓最直接的"边缘 LNN 机器人感知"证据之一：

- **CfC 作为 recurrent engine**：证明 CfC 可在真实机器人 perception loop 中承担高频自回归预测，而不只是时序分类。
- **Jetson 可部署性**：论文明确使用 Jetson Orin NX onboard real-time，和本仓 `scripts/jetson_lnn_benchmark.py` 的边缘验证目标一致。
- **主动感知闭环**：LNN 输出不是离线指标，而是进入 RL reward / camera action 闭环，能影响地图覆盖和导航观测质量。

本仓可落地的实验路径：

| 方向 | 目标 |
|---|---|
| GazeLNN-lite smoke test | 用 MobileNetV3 + CfC 复现 8-step heatmap autoregression，先用合成 fixation heatmap 验证接口 |
| Jetson latency probe | 输出 `analysis/jetson/2026-06-24_gazelnn_latency.md`，测 256x384 输入下 CfC recurrent latency |
| Active perception toy | 将 fixation heatmap reward 简化为 2D gridworld camera control，不触发真实设备 |
| SNCP/CfC 对照 | 比较 CfC、SNCP-lite、ConvGRU 在 scanpath autoregression 的参数量/延迟/稳定性 |

## 7. 局限性与未来展望

作者明确提到的未来方向：

- 进一步研究 fixation scanpath、主动相机配置和运动动力学之间的相关性；
- 扩展 RL policy，使人类注视预测与主动相机控制的耦合更稳健。

本仓视角下的局限：

1. **scanpath 训练与机器人部署存在 domain gap**：OSIE / MIT Low Resolution 是静态自然图像，真机导航是 RGB-D + 运动视角。
2. **训练时 $\Delta t$ 来自真实 fixation duration，部署时固定为 1**：这会削弱 CfC 时间门的真实连续时间语义。
3. **真机验证规模有限**：论文展示单类室内飞行环境，尚不足以证明不同光照、速度、障碍密度下的鲁棒性。
4. **RL heatmap 在仿真中用 proxy 生成**：训练用分割 mesh face 采样的 proxy heatmap，部署用 GazeLNN 预测 heatmap，两者分布差异需要进一步量化。
5. **未给出完整开源复现路径**：摘要与 PDF 没有明确代码仓库；若无代码，复现需从架构描述手工实现。

## 8. 今日结论

**状态：read_now + experiment。**

GazeLNN 值得进入本仓实验队列。它提供了一个很清楚的工程靶点：`MobileNetV3 feature extractor + CfC autoregressive heatmap predictor + fixation reward`。短期不建议直接做真机闭环；应先在合成 heatmap / 2D gridworld 中验证 CfC scanpath 模块和 Jetson 延迟，再决定是否接入真实机器人平台。
