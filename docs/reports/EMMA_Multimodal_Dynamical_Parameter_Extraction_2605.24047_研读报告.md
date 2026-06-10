---
title: "论文研读报告: EMMA — Extracting Multiple physical parameters from Multimodal Data"
date: 2026-06-11
tags: [LNN, LTC, multimodal, physics-informed, parameter-estimation, CVPR2026, digital-twin]
arxiv_id: "2605.24047"
status: full-report
---

# 论文研读报告: EMMA — Extracting Multiple physical parameters from Multimodal Data

## 元数据

- **论文标题**: EMMA: Extracting Multiple physical parameters from Multimodal Data
- **作者**: Farhat Shaikh, Ayan Banerjee, Sandeep Gupta
- **作者单位**: IMPACT Lab, School of Computing & Augmented Intelligence (SCAI), Arizona State University
- **发表时间/会议**: 2026 年 5 月 (v1, 2026-05-21); 已被 CVPR 2026 接收 (正文: https://github.com/ImpactLabASU/EMMA-CVPR2026)
- **标签**: `#LNN` `#LTC` `#Multimodal` `#Physics-Informed` `#Inverse-Modeling` `#Digital-Twin` `#CVPR2026` `#Audio-Visual` `#Parameter-Estimation`
- **arXiv 链接**: <https://arxiv.org/abs/2605.24047v1>
- **本地 PDF 归档**: `papers/daily/2026-05-30/2026-05-21_EMMA_Extracting_Multiple_physical_parameters_from_Multimodal_Data_2605.24047v1.pdf`
- **官方代码**: <https://github.com/ImpactLabASU/EMMA-CVPR2026>
- **致谢**: DARPA AMP (N6600120C4020), DARPA FIRE (P000050426), NSF FDT-Biotech (2436801), NIH R21 (1R21HL175632)

## 核心问题

从多模态被动感知 (视频、音频、图表) 反推**物理系统的可识别动力学参数** $\theta \in \mathbb{R}^K$ 是构建机器人 / 自动驾驶 / 无人机 / 行星车数字孪生的关键一步。但现有方法 (Delfys, NIRPI, PAIG, ϕ-SfT, RISP, gradSim 等) 普遍存在四个局限性 (Table 1):

1. **忽略外部强迫输入** (forcing inputs): 例如 rover 轮胎转速、无人机电机转速,这些信号往往**未在视频中直接可见**, 必须从其它模态 (音频) 推断;
2. **只能恢复单一 / 有限子集参数**: 真实系统通常含 5–7 个耦合参数 (长度、质量、惯量、阻尼等), 同时识别能力薄弱;
3. **无法处理隐式动力学** (implicit dynamics): 例如摩擦阻尼、空气阻力等**不直接出现在任何传感器读数中**但通过非线性交互影响系统行为的项;
4. **依赖已知不变量**: 需要预先已知坐标系原点、初始条件、世界↔相机变换,这在野外机载部署中往往不成立。

> **EMMA 核心定位**: 把"从多模态被动感知反推 ODE 参数"问题形式化为一个**无监督、物理约束的连续时间潜变量推断问题**, 首次**同时**实现 (i) 多模态融合, (ii) 强迫输入重建, (iii) 隐式动力学推断, (iv) 不变量自动标定。

## 方法论与核心思路

### 整体架构 (Figure 2, Section 3.2)

EMMA 由三个串联阶段构成:

```
[视频] ─┐                    ┌─→ [LTC-NN] ─→ [Dense head] ─→ θ̄
[音频] ─┼→ 多模态特征管线 x(t) ┤    64 单元      (sigmoid 读出)
[图像] ─┘    (3.3 节)          └─→  h(t) 隐状态  + ReLU 不变量单元
                                                  ↓
                                       可微 ODE 仿真器 + physics loss
```

1. **统一多模态特征抽取 (Unified feature extraction)**: 对视频用 YOLOv11 + Kalman + 像素↔物理坐标变换;对音频用 STFT (FFT 2048, hop 512) + RMS / 频谱质心 / 主峰频率;对图像用 PIL + OpenCV 切出曲线 → 时序点。三路输出 $\mathbf{p}(t) \in \mathbb{R}^{d_v}$、$\mathbf{w}(t) \in \mathbb{R}^{d_a}$、$\mathbf{m}(t) \in \mathbb{R}^{d_m}$ 在统一时间网格上拼接为
   $$\mathbf{x}(t) = [\mathbf{p}(t);\, \mathbf{w}(t);\, \mathbf{m}(t)] \in \mathbb{R}^{D_{\text{in}}}, \quad D_{\text{in}} = d_v + d_a + d_m \tag{3}$$
   缺失模态用零填充 / 学到的嵌入填补。空间编码采用 $N_{\text{spatial}}=100$ 采样,产出 $\mathbf{x}_{\text{in}}(t) \in \mathbb{R}^{100}$。
2. **LTC 网络建模连续时间潜动力学 (Section 3.4)**: 用 [ncps](https://github.com/mlech26l/ncps) 库实现 **64 单元** LTC 神经网络,作为 EMMA 核心。每个 cell 实现:
   $$\frac{dh_i}{dt} = \underbrace{-\frac{h_i}{\tau_i / (1 + \tau_i f_{NN}(h_i, u, t, w_{NN}))}}_{\text{models forcing inputs}} + \underbrace{f_{NN}(h_i, u, t, w_{NN})A}_{\text{models physics-consistent dynamics}} \tag{4}$$
   这正是 **input-dependent time constant** 的关键 — $\tau_i(t)$ 由网络自身根据输入 $u(t)$ 调节, 使 LTC 天然能跟随**强迫输入**。作者在补充材料 Table S7 中对比 LTC vs Neural ODE vs CT-GRU, 在强迫输入下 LTC 平均参数误差比 Neural ODE **低 25%**, 比 CT-GRU 低 5%。
3. **稠密层读出 + 标定 (Section 3.4)**: LTC 隐状态 $h(t)$ 通过一个 **sigmoid 激活的稠密头**非线性映射到 $\bar{\theta}_k \in (0,1)$。这一读出被解释为**数据驱动的模态分解** (类 Dynamic Mode Decomposition / DMDc / Koopman 思想)。**额外加几个 ReLU 单元**用于学习不变量 $\gamma_i$ (坐标原点、初始位姿等)。最后**反归一化**到物理尺度:
   $$\theta_k = \Big(1 + (0.5 - \bar{\theta}_k)\cdot \tfrac{95}{100}\Big) \cdot \theta_k^{\text{nom}} \tag{5}$$

> **为什么用 LTC 而不是 Neural ODE?**: 强迫输入下, Neural ODE 的常时间常数无法适应快速变化的激励;LTC 的 $\tau_i(t)$ 自适应让隐状态能"快速重置"以跟随外部命令。这是 EMMA 在 drone / rover 这类有控制器驱动的系统上能稳定工作的**核心机制**。

### 物理约束训练 (Section 3.5)

EMMA 对参数**完全无监督**: 不使用任何 ground-truth 参数值, 训练信号全部来自物理一致性损失。**总损失**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}^{\text{cal}}_{\text{traj}} + \lambda_{\text{param}}\, \mathcal{L}_{\text{param}} \tag{6}$$

**校准后的轨迹损失** (只对测量到的状态分量求差):
$$\mathcal{L}^{\text{cal}}_{\text{traj}} = \sum_{i=1}^{n} M_{ii} \cdot \frac{1}{T_{\text{sim}}} \sum_{t=1}^{T_{\text{sim}}} \| x_i(t) - \gamma_i - x_{i,\text{sim}}(t) \|^2 \tag{7}$$
其中 $\gamma_i$ 是稠密层 ReLU 输出 (不变量), $M$ 是测量掩码对角阵 ($M_{ii}=1$ 表示该状态可测)。

**参数约束损失** (ReLU 软夹紧):
$$\mathcal{L}_{\text{param}} = \sum_{i=1}^{K} w_p(i)\, \text{ReLU}(-\theta_i) + w_l(i)\, \text{ReLU}(\theta_i - l_i) + w_{up}(i)\, \text{ReLU}(\theta_i - \text{up}_i) \tag{8}$$

**训练细节**: AdamW + cosine annealing, 6 步 ODE unfolding, PyTorch + ncps, YOLOv11, librosa, MoviePy, OpenCV, PIL。

### 与 SOTA 对比维度 (Table 1, Section 3.1)

| 方法 | 强迫输入 | 多参数 (≥3) | 隐式动力学 | 多模态 | 不变量学习 |
|---|---|---|---|---|---|
| Delfys [10] / NIRPI [18] | ✗ | ✗ | ✗ | ✗ | ✗ |
| PAIG [21] | ✗ | ✗ | ✗ | ✗ | ✗ |
| gradSim [22] / ϕ-SfT [26] | ✓ | ✓ | ✗ | ✗ | ✗ |
| Vid2Param [1] / Kandukuri [30] | ✗ / ✓ | ✗ | ✗ | ✗ | ✗ |
| **EMMA (本文)** | **✓** | **✓** | **✓** | **✓** | **✓** |

## 核心公式 (LaTeX)

> 以下公式按论文出现顺序整理, 数字与原论文 equation 编号一致。

**物理动力学模型** (方程 1):
$$\frac{d\mathbf{x}(t)}{dt} = f\!\big(\mathbf{x}(t),\, \mathbf{u}(t);\, \boldsymbol{\theta}\big) \tag{1}$$

**反演目标函数** (方程 2, 同时学不变量 $\psi$ 与参数 $\theta$):
$$\min_{\boldsymbol{\theta},\, \psi,\, \mathbf{x}_0}\; \frac{1}{T}\sum_{t=1}^{T} \big\|\mathbf{y}_t - \mathbf{y}^{\text{sim}}_t\big\|_2^2 + \mathcal{R}(\boldsymbol{\theta}_{\text{est}}) \tag{2}$$

**多模态拼接** (方程 3, 已在上面给出):
$$\mathbf{x}(t) = [\mathbf{p}(t); \mathbf{w}(t); \mathbf{m}(t)] \in \mathbb{R}^{D_{\text{in}}} \tag{3}$$

**LTC-NN 微分方程** (方程 4, EMMA 的"灵魂公式"):
$$\frac{dh_i}{dt} = \underbrace{-\frac{h_i}{\tau_i/(1+\tau_i f_{NN}(h_i,u,t,w_{NN}))}}_{\text{input-dependent decay}} + \underbrace{f_{NN}(h_i,u,t,w_{NN})A}_{\text{nonlinear physics}} \tag{4}$$

**反归一化** (方程 5):
$$\theta_k = \Big(1 + (0.5 - \bar{\theta}_k)\cdot \tfrac{95}{100}\Big) \cdot \theta_k^{\text{nom}} \tag{5}$$

**总损失** (方程 6, 已在上面给出):
$$\mathcal{L}_{\text{total}} = \mathcal{L}^{\text{cal}}_{\text{traj}} + \lambda_{\text{param}}\, \mathcal{L}_{\text{param}} \tag{6}$$

**校准轨迹损失** (方程 7, 含不变量 $\gamma_i$):
$$\mathcal{L}^{\text{cal}}_{\text{traj}} = \sum_{i=1}^{n} M_{ii}\, \frac{1}{T_{\text{sim}}} \sum_{t=1}^{T_{\text{sim}}} \big\| x_i(t) - \gamma_i - x_{i,\text{sim}}(t) \big\|^2 \tag{7}$$

**参数约束 ReLU 软夹紧** (方程 8):
$$\mathcal{L}_{\text{param}} = \sum_{i=1}^{K}\Big[ w_p(i)\, \text{ReLU}(-\theta_i) + w_l(i)\, \text{ReLU}(\theta_i - l_i) + w_{up}(i)\, \text{ReLU}(\theta_i - \text{up}_i) \Big] \tag{8}$$

**音频-转速线性先验** (Section 3.3 b):
$$f_{\text{tone}}(t) \approx \alpha\, v(t) + \beta$$
其中 $\alpha, \beta$ 是需要随 LTC 一起学到的**标定不变量**。

## 关键成果与贡献

### 实验设置

- **5 个标准动力学基准**: 单摆 (45/90/150 cm)、Torricelli 容器、滑动块 (低/中/高倾角)、LED、自由落体 — 共 75 个 Delfys 视频;
- **2 个真实机器人系统**: 真实 rover (5 维参数) + 真实四旋翼 (7 维参数, 含隐式动力学参数);
- **6 个混沌 / 生物系统图表 (Experiment C)**: Lotka-Volterra, Lorenz, F8 Crusader 战机, HIV therapy, AID therapy;
- **Baseline**: PAIG, NIRPI, Delfys, PySINDy, SINDy-PI, PINN。

### 核心定量结果

**Experiment A — 经典系统 (Table 2, 单摆/滑动块)**:
- EMMA 单摆长度恢复: 45 cm 视频 → 0.507±0.039 m (GT 0.45), 90 cm → 0.859±0.073 (GT 0.9), 150 cm → **1.501±0.004** (GT 1.5); PySINDy 在 150 cm 长摆上 $\tau$ 参数无法恢复 (0.00);
- EMMA 滑动块倾角: 低/中/高坡度分别 19.92° / 24.72° / 29.81° (GT 20°/25°/30°),**误差 < 1°**; PySINDy 在中/高坡度上严重偏离 (27° 估计);
- 摩擦系数 $\mu$ 恢复: 0.208 / 0.205 / 0.204 vs GT 0.20 — **相对误差 < 4%**。

**Experiment B — 真实机器人 (Table 4c/4d)**:
- **Rover (5 参数)**: EMMA 估计的 X/Y 臂长、轮半径、质量、CoM 高度与 ground truth 高度一致, **平均相对误差 8.8% ± 1.7%**;
- **Drone (7 参数, 含 4 个隐式动力学)**: 推力系数 $k_{Th}$ 1.017 (GT 1.1, 误差 7.5%)、电机增益 $k_p$ 1.007 (GT 0.91, 误差 10.7%)、电机时间常数 $\tau_2$ 0.015 (GT 0.012, 误差 25%)、机械臂长 $d_{xm}/d_{ym}/d_{zm}$ 误差 8–27%;**所有 7 个参数平均相对误差 15.9% ± 7.4%** — 注意这是**含 4 个隐式动力学参数**的难度;
- EMMA 不需要 idle wheel power 或四旋翼 idle 转速作为先验输入, 真正学到"最合适"的不变量。

**Experiment C — 图表反演 + 隐式动力学鲁棒性 (Table 5)**:
- 在 Lotka / Lorenz / HIV / AID 四个含混沌 / 多状态耦合的系统上, EMMA 的 $\theta_{\text{rmse}}$ 比 PySINDy 低 **2–10 倍**, 在 Lorenz 上 $\theta_{\text{rmse}}$ 1.7 vs PySINDy 37.4 (~22 倍);x_rmse 同样优势明显;
- 引入"隐式动力学" (只有一个状态可测) 时, PySINDy 性能崩塌 (Lorenz x_rmse 1.68 → 3.66), EMMA 几乎不退化 (1.7 vs 1.68);
- 这一项证明 LTC 的**超定线性化隐变量**比 SINDy 的稀疏回归对隐式项更鲁棒。

**音频噪声鲁棒性 (supplementary Table S9)**: 在 SNR = 5 dB 下, rover 全部参数恢复的变化 < **1.1%**, 表现出极强的工业可用性。

**执行效率 (Table 6)**:
- EMMA 模型仅 **53.2K 参数**, Delfys 是 **5.7M** — **EMMA 小 107×**;
- EMMA 单 epoch 0.37 s (RTX Ada 6000), Delfys 0.19 s — EMMA 慢 1.4×, 这是 ODE 求解的固有开销;但配合 107× 的参数压缩, 仍非常适合**边缘部署** (论文 [46] Xu et al. 2025 已展示 FPGA 上 11× 内存压缩)。

**鲁棒性 (supplementary Table S8)**: 初始化区间放宽到 ±200% 时, EMMA 在 5/6 配置下仍能正确收敛 — 对先验不敏感。

### 贡献总结 (按重要度)

1. **多模态物理参数反演统一框架**: 视频 + 音频 + 图表的端到端融合在统一连续时间 ODE 中;
2. **强迫输入下隐式动力学的 LTC 求解**: 解决了 Neural ODE / SINDy 在快速激励下退化的问题 (Table S7 证明);
3. **自动不变量标定**: 用稠密层 ReLU 单元学坐标原点, 摆脱"已知初始条件"假设;
4. **极致参数效率**: 53.2K 参数即可解 7 维参数反演, 边缘部署友好;
5. **开源代码** + CVPR 2026 接收, 完整评测套件 100+ 场景, 极具工程参考价值。

## 局限性与未来展望

### 作者明确承认的局限 (Section 5)

1. **依赖至少一个时变模态**: 若所有输入都是静态图像, LTC 的时间动力学无从学习;
2. **音频线性频率-转速先验**: $f_{\text{tone}}(t) \approx \alpha v(t) + \beta$ 在湍流 / 强谐波 / 多转子干涉下会失效, 需要扩展到非线性或数据驱动的 $\alpha(v, t)$;
3. **相机剧烈抖动敏感**: YOLOv11 检测在严重抖动的视频上漏检率上升, 需要引入 IMU 联合滤波或光流补偿;
4. **LTC 推理时间偏高**: ODE 求解器每步都要调用 nn.odeint, 在极端边缘 (微控制器) 上仍偏慢, 需量化 + 编译优化。

### 个人补充的局限 / 思考

- **物理 ODE 必须已知**: EMMA 假定系统动力学形式 $f(\mathbf{x}, \mathbf{u}; \theta)$ 是先验给出的, 仅反演参数 $\theta$。对**完全未知**的物理系统 (如新设计的机械结构) 仍需配合 SINDy 这类方程发现方法;
- **音频-转速线性先验 $\alpha, \beta$ 需要标定**: 在跨平台迁移时 (从实验室 rover 到野外 rover), 需重新标定;
- **6 步 ODE unfolding**: 训练时 ODE solver 的精度对反演结果影响敏感, 实际部署需要步数自适应;
- **缺乏对 LNN 范式之外的横向比较**: 比如论文没有把 EMMA 与直接用 Neural ODE / S4 / Mamba 作为潜变量的等价管线做 head-to-head, 难以判断 LTC 的优势是"input-dependent $\tau$" 还是"小模型 + 物理约束"的整体设计;
- **数据集未开源声明**: CVPR 2026 仓库仅开源代码, 真实 rover / drone 视频需要申请, 这限制可复现性。

### 未来方向 (基于论文 + 个人推演)

- **结合 SINDy-PI**: 把方程发现嵌入 EMMA, 让 $f$ 形式本身可学, 扩展到完全未知物理系统;
- **迁移到 LLM 物理 agent**: EMMA 输出的"可解释、可执行"参数 $\theta$ 天然适合作为 LLM 的"物理工具", 在物理 AI agent 中调用 (与本仓库 `analysis/control/2026-06-09_064318_imitation_lnn.json` 机器人控制方向高度契合);
- **音频非线性先验**: 用小型 NN 学 $f_{\text{tone}}(v, t)$, 而非线性性的 affine $\alpha,\beta$;
- **Jetson 部署**: EMMA 仅 53.2K 参数, 加上 [46] 提到的 FPGA 11× 压缩, 完全可以在 Jetson Orin Nano 上实时跑 drone 在线参数反演;
- **跨域泛化**: 把 EMMA 扩展到医疗 (从超声 + ECG 反推心脏动力学)、气候 (从卫星图像 + 站点数据反推大气模型) 等多模态物理反演场景。

---

**研读时间**: 2026-06-11 06:35 UTC  
**研读执行**: LNN 每日研读机器人 (Cron Job)  
**研读方法**: 全文 PDF 解析 (pymupdf, 14 页) + 公式重构 + 关键表格数据摘录  
**置信度**: 高 (论文结构清晰、公式与图表一一对应、CVPR 2026 接收, 实验可重现)
