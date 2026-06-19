---
title: GazeLNN - LNN 驱动的轻量级注视扫描路径预测与主动感知机器人导航 研读报告
arxiv_id: 2606.20491v1
date: 2026-06-18 (arXiv v1) / 研读 2026-06-20
tags: [LNN, CfC, scanpath, saliency, active-perception, RL, drone, edge-ai, paper-report]
parent: [[LNN_深度研读报告]]
---

# 论文研读报告 — GazeLNN: Fast Human Attention Prediction for Fixation-guided Active Perception in Autonomous Navigation

> arXiv:2606.20491v1 (cs.RO, 2026-06-18)
> 来源: [[docs/daily/2026-06-20_LNN_research_digest.md|2026-06-20 每日追踪]]
> 候选评分: 2 (`select_papers_for_report.py --top 3` 仅 1 篇未覆盖, 见 digest)

## 1. 元数据
- **标题**: Fast Human Attention Prediction for Fixation-guided Active Perception in Autonomous Navigation
- **作者**: Fatma Youssef Mohammed, Grzegorz Malczyk, Kostas Alexis (挪威科技大学 NTNU 工程控制系)
- **发表**: arXiv:2606.20491v1, 2026-06-18
- **资助**: Horizon Europe Grant No. 101120732, Research Council of Norway NO-338694 / NO-357451
- **代码**: 文中给出 GitHub URL 但未在摘要中显式标记 (`this https URL` 风格)
- **PDF**: `papers/daily/GazeLNN_2606.20491.pdf` (8 页, 4.6 MB)
- **关键词**: Liquid Neural Networks, CfC, scanpath, saliency, fixation prediction, active perception, reinforcement learning, Aerial Gym, Jetson Orin NX
- **领域**: 机器人 / 视觉注意 / 边缘部署 / 强化学习

## 2. 核心问题

自主机器人(尤其是敏捷空中平台)在严格的车载资源约束下, 不可能对全分辨率视觉流做穷举处理。仿人视觉系统用 **active perception + 扫视 (saccade) + 注视 (fixation)** 的方式把算力集中在感兴趣区域, 但把这种行为蒸馏成机器人可用的模型存在两个根本痛点:

1. **计算成本**: 现代 scanpath 预测 SOTA 普遍依赖重型 Transformer 或深度 RNN, 对边缘机器人而言速度慢、能耗高、不实时。
2. **闭环缺失**: 即便存在 CNN-based saliency 模型被用于 SLAM / 避障, 但 **"saliency 模型 + 主动相机控制策略" 联合优化** 的工作极少 — saliency 通常以离线方式输出静态图, 与控制回路解耦。

论文的核心问题: 能否用一个 **轻量级液态神经细胞 (CfC)** 取代重型 RNN/Transformer 主干, 把 scanpath 预测做到 **< 1 GFLOPs / 实时**, 然后把这个预测器 **实时地接入** RL 训练的 active-camera 策略, 在真机四旋翼上演示 **human-fixation-guided navigation**?

## 3. 方法论与核心思路

### 3.1 总体架构 (Fig. 2)
- **特征提取**: MobileNetV3 (输入 256×384 → 下采样 S=8 → 32×48)
- **注视表征**: 当前 fixation 用 Gaussian heatmap + CoordConv 表示 (中心 + xy 通道)
- **循环核心**: **CfC (Closed-form Continuous-depth)** 变体 — 选 CfC 而非 LTC, 因为 CfC 训练/推理比 LTC 快 (cf. Hasani 2022)
- **联合输入**: `[image_features; fixation_heatmap; hidden_state]` → FC backbone (1024) + LeCunTanh → 4 个并行 FC 头 (f1, f2, ta, tb)
- **输出**: 下一个 fixation 的 H_S × W_S heatmap, 上采样到原图尺寸后归一化, 取 argmax 作为 next fixation

### 3.2 CfC 隐状态更新 (Eq. 1)

设 $\Delta t$ 为两次注视之间的真实持续时间 (训练时取自标注; 部署时 $\Delta t \equiv 1$):

$$h_{i+1} = \bigl(1 - \sigma(t_a \Delta t + t_b)\bigr) \odot \tanh\!\bigl(f_1(x_t)\bigr) \;+\; \sigma(t_a \Delta t + t_b) \odot \tanh\!\bigl(f_2(x_t)\bigr)$$

其中 $\sigma$ 是 sigmoid, $\odot$ 是逐元素乘积。直觉: $t_a, t_b$ 学出一个 **time-dependent gating signal**, 把两个并行分支的隐状态按"距上次注视多久"动态混合, 等价于 ODE-1 的闭式离散解, 但避免了 RNN 的逐步数值积分。

### 3.3 RL 主动相机策略 (Section IV)
- **仿真**: Aerial Gym simulator
- **观察**: 机器人状态 + 深度图 + fixation heatmap + 局部 3D 占用栅格 + 相机朝向
- **主干**: 2D ResNet (heatmap 编码) + 冻结的 DCE (深度编码) + MLP + GRU
- **动作**: 6 维向量 $[v_t^r, \omega_{t,z}^r \;|\; a_t^{nav} \;|\; c_t^r = (\chi_t^r, \psi_t^r)]$
- **奖励** (Eq. 3-6):
  $$R(s_t, a_t) = r_t + \ell_t + p_t + h_t$$
  其中:
  - $r_t = w_r (d_{t-1} - d_t)$ (导航进度)
  - $\ell_t = -w_\ell \|a_t - a_{t-1}\|_2$ (运动平滑)
  - $p_t$: 障碍物距离惩罚
  - $h_t = w_h \cdot \frac{\sum_{u,v} H_t(u,v) e^{-\alpha d(u,v)^2}}{\sum_{u,v} H_t(u,v) > 0}$ (**fixation-attraction** — 相机偏向高热区域)

### 3.4 训练细节
- **scanpath 数据**: OSIE (700 图 × 15 受试者; 80/10/10 split), padded 到长度 8
- **测试集**: MIT Low Resolution (168 图 × 8 受试者, 仅最高分辨率)
- **优化**: Adam lr=1e-4, 100 epoch, KL-DTW loss (含动态时间规整的 KL 散度, 兼顾空间与时间结构)
- **Early stopping**: patience=20 epoch
- **硬件**: RTX 3500 Ada (laptop GPU); 真机部署用 Jetson Orin NX 16GB + PX4 飞控

## 4. 核心公式 (LaTeX)

**CfC 隐状态更新** (Eq. 1):
$$
h_{i+1} = \bigl(1 - \sigma(t_a \Delta t + t_b)\bigr) \odot \tanh(f_1(x_t)) + \sigma(t_a \Delta t + t_b) \odot \tanh(f_2(x_t))
$$

**Fixation-attraction reward** (Eq. 6):
$$
h_t = w_h \cdot \frac{\sum_{u,v} H_t(u,v) \cdot \exp(-\alpha d(u,v)^2)}{\sum_{u,v} H_t(u,v) > 0}
$$

**Total RL reward** (Eq. 3):
$$
R(s_t, a_t) = r_t + \ell_t + p_t + h_t
$$

**Navigation progress** (Eq. 4):
$$
r_t = w_r (d_{t-1} - d_t)
$$

## 5. 关键成果与贡献

### 5.1 Scanpath 预测 (Table I, Table II)
在 MIT Low Resolution 上 (8-fixation scanpath), 6 个指标**全部 SOTA**:
- Levenshtein Dist.: **11.22** (vs tSPM-Net 11.47, -2.18%)
- ScanMatch: **0.47** (vs tSPM-Net 0.34, **+34.29%**)
- Hausdorff Dist.: **98.17** (vs 103.44, -5.09%)
- Fréchet Dist.: **133.31** (vs 144.77, -7.92%)
- Fast DTW: **537.72** (vs 610.02, -11.85%)
- Time Delay Embedding: **27.20** (vs 43.74, -37.81%)

**计算开销 (Table II)**:
- **0.61 GFLOPs** / 15.24 M params / **6.84 ms** inference
- 相比 tSPM-Net: -91.80% params, -99.40% GFLOPs, **6.42× 加速**

### 5.2 Backbone Ablation (Table II)
- VGG19+DeepLabV3: 99.81 GFLOPs / 195.41 M params / 17.43 ms
- ResNet50+DeepLabV3: 69.8 / 77.4 / 14.32 ms
- **MobileNetV3 (选)**: **0.61 / 15.24 / 6.84 ms** ← 最优
- DINOv3 (ViT-S)*: 8.49 / 31.35 / 8.76 ms (ScanMatch 0.48 但计算高 14×)

### 5.3 RNN Ablation (Table III, 同 backbone VGG19+DLv3)
- Bayesian ConvLSTM: 102.51 GFLOPs / 185.92 M params / 43.90 ms / ScanMatch 0.34
- ConvLSTM: 102.51 / 185.76 / 21.76 ms / 0.34
- **CfC**: 99.81 / 195.41 / 17.43 ms / **ScanMatch 0.47**

注意: CfC 的参数量并未减少 (backbone 占主导), 但**性能显著提升 + 推理加速**, 这是 LNN 替换 ConvLSTM 的直接收益。

### 5.4 真机部署 (Section VI, Table IV)
在 Quadrotor + Intel RealSense D455 (pan-tilt, ±45° yaw / ±60° pitch, 15 Hz) + Jetson Orin NX 16GB + PX4 上的对比:
- **Static camera**: 37 067 total voxels, fixation grid 仅 **873 voxels**
- **Active camera (GazeLNN 驱动)**: **55 524 total voxels (+50%)**, fixation grid **6 770 voxels (+675%, ~8×)**
- 最大 hit count: 537 → 756 (持续关注 salient 区域更久)
- GazeLNN 在 Jetson 上以 10 Hz 运行, ROS 中间件

## 6. 局限性与未来展望

### 6.1 论文自己承认的局限
- **训练数据规模**: 仅用 OSIE (700 图), 难以扩展到大规模场景; 跨数据集泛化未量化。
- **Scanpath 长度固定**: padded 到 8, 短序列 (< 4) 被丢弃; 真实注视行为可能有 12-20 次以上。
- **真机实验场景单一**: 室内非结构化环境, 未测试室外 / 强光 / 高速运动。
- **Fixation duration 在部署时硬编码为 1**: 与训练时 ground-truth $\Delta t$ 分布不匹配, 可能在长注视序列上引入系统偏差。

### 6.2 本仓库视角的局限
- **论文 vs 实验脱节**: Table III 的 RNN ablation 在 VGG19+DLv3 大 backbone 上做, 而最终模型却用 MobileNetV3, 因此"换 CfC 替代 ConvLSTM" 的 isolated 收益**没有被干净地测出来**; 应该固定 MobileNetV3 再做一次 RNN ablation 才有说服力。
- **active-camera 收益 vs 静态 camera 的混淆变量**: 真机实验只跑 1 次对比, 没有多 seed 的均值/方差报告, 难以判断 +50% voxels 是否统计显著。
- **未对比 CfC vs LTC**: 仅声称 CfC 推理更快, 没有在同样 backbone 下报两者的精度-速度 Pareto。
- **未与 Gazeformer / 近期 Transformer-free LNN 比**: 论文主对比都是 2024 年前的 ConvLSTM / tSPM-Net, 缺少 2025 的强 baseline。

### 6.3 未来方向
1. **Causal scanpath model**: 当前 fixation heatmap 编码不含时间偏移信息, 未来可加 positional encoding 让模型感知"已注视了多久"。
2. **联合训练 CfC + RL policy**: 现在 CfC 在 OSIE 上预训练, RL 在仿真中用 proxy heatmap 训练; end-to-end 联合训练可能让 scanpath 更适配 navigation 任务。
3. **LTC vs CfC on drone**: LTC 的 ODE-1 + 真实 $\Delta t$ 可能比 CfC 更适合处理不规则采样 (drone 抖动 / 帧率波动), 值得复现。
4. **迁移到其他 embodied agents**: ground robot / manipulator / autonomous underwater vehicle。

## 7. LNN 桥接与本仓相关性

- **与本仓 round 134 (LiquidTAD) 的关系**: LiquidTAD 把 ODE-1 闭式解蒸馏成**并行离散卷积** (PLR); GazeLNN 把同样的 ODE-1 闭式解蒸馏成**显式 time-gated hidden state** (CfC)。两者都是"用 LNN 思想压缩重型 RNN"的实例化, 但目标不同 (TAD vs scanpath), 都验证了 **LNN 的连续时间先验在序列建模任务上的参数/算力优势**。
- **与 Jetson 部署主题**: GazeLNN 在 Jetson Orin NX 上跑 10 Hz 是本仓 [[PRD_LNN_Edge_Research]] 中"边缘部署可行性"的硬证据: **CfC + MobileNetV3 在 16GB 显存 Jetson 上可实时运行**, 推理 ~10 ms 量级, 与本仓 `scripts/jetson_lnn_benchmark.py` 的硬件定位一致。
- **可能的复现路径**:
  1. 用 `lnn/core/liquid_cells.py` 中的 `CfCCell` (已实现) 替换本仓 `experiment_imitation_lnn.py` 中的 GRU 主干, 在 drone navigation 任务上做 isolated ablation;
  2. 把 RL 奖励中的 $h_t$ 移植到本仓 `projects/active_camera_rl/` (如有);
  3. 写一个 `bench_gazelnn_scanpath.py` 对比 CfC vs GRU 在 OSIE 子集上的 ScanMatch 与推理时间。

## 8. Verdict
**TARGET-DEPENDENT-WITH-NUANCE** — 对"边缘机器人 scanpath 预测"这是 **NEGATIVE-on-CfC-overall (cf. paper) but POSITIVE-on-deployability**: 论文的 6.42× 加速 + 99.40% GFLOPs 节省在 Jetson 上得到真实验证, 但**没有干净地把 CfC vs 其他 LNN 变体的 isolated 收益拆出来**, 且 RNN ablation 的 backbone 不一致削弱了论文主张。对本仓而言, 价值在于: (a) **又一个 CfC 在真实机器人上跑通的案例**, (b) **scanpath + active-camera 的 RL 奖励设计范式** ($h_t$ 项), 可迁移到本仓的 edge LNN 部署实验。
