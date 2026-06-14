---
title: DLNet — Dual-Stage Distillation & Pareto Compression for Edge LNN Battery Prognostics 研读报告
arxiv_id: 2601.06227
authors:
  - Dhivya Dharshini Kannan
  - Wei Li
  - Wei Zhang
  - Jianbiao Wang
  - Zhi Wei Seh
  - Man-Fai Ng
published: 2026-01-09 (v1), 2026-06-11 (v3)
venue: International Conference on Pattern Recognition, ICPR 2026
date: 2026-06-14
tags: [LNN, distillation, edge-ai, battery, pareto, paper-report]
parent: [[LNN_深度研读报告]]
related: [[LNN_训练方向_边缘部署与压缩_可行报告]]
---

# 论文研读报告 — DLNet: When Smaller Wins — Dual-Stage Distillation and Pareto-Guided Compression of Liquid Neural Networks for Edge Battery Prognostics

> arXiv:2601.06227v3 (2026-06-11); 接收于 ICPR 2026。
> 由本仓库 [[LNN_深度研读报告]] 排入,作为今日 (2026-06-14) digest 候选之一完成结构化研读。

## 元数据
- **标题**: When Smaller Wins: Dual-Stage Distillation and Pareto-Guided Compression of Liquid Neural Networks for Edge Battery Prognostics
- **作者**: Dhivya Dharshini Kannan, Wei Li, Wei Zhang, Jianbiao Wang, Zhi Wei Seh, Man-Fai Ng
- **发表**: arXiv:2601.06227v3, 2026-01-09 (v1) → 2026-06-11 (v3)
- **会议**: International Conference on Pattern Recognition, **ICPR 2026** (已接收)
- **领域**: 边缘部署 / 模型压缩 / 电池健康预测
- **关键词**: Liquid Neural Networks, Knowledge Distillation, Pareto Front, Edge AI, Battery Prognostics, int8 Quantization

## 1. 核心问题

电池管理系统 (BMS) 越来越需要在 **端侧** 准确预测电池健康状态 (SOH),但 LNN/CfC 类连续时间模型
原始形态较重,直接部署到 Arduino Nano 33 BLE Sense / Cortex-M4 这类 MCU 上不现实:

1. **连续时间算子不便部署**: LNN/LTC 的 ODE/CfC 闭环通常需要浮点张量与小型 ODE solver,MCU 资源极紧。
2. **teacher 模型太大**: LNN teacher 在端侧无法直接运行,需要 student。
3. **单一蒸馏目标不够**: 仅做 MSE-based KD 会让 student 在压缩后期掉点 (论文观察到 15% 以上),
   因为连续时间动力学的"时间常数"被一并压掉。
4. **精度-成本需联合筛选**: 不同压缩比 / 量化位宽组合是 Pareto 集合,人工挑费时且次优。

作者要解决: **在 LNN 上做"先离散 + 再蒸馏 + 再 Pareto 选优"三步流水线,
最终得到一个比 teacher 还准 15.4%、体积小 84.7% 的 int8 student。**

## 2. 方法论与核心思路

### 2.1 总体三段式流水线

```
Teacher LNN (continuous-time, 浮点)
    ↓  Step 1: Euler discretization (LNN → 嵌入式友好的离散 RNN)
Discrete-LNN
    ↓  Step 2: Dual-Stage Knowledge Distillation
Compressed Student (中间模型)
    ↓  Step 3: Pareto-guided selection under joint error-cost
Final Student (int8, ≤ 100 kB)
    ↓  Step 4: ONNX/TFLite-micro 导出, Arduino 部署
```

### 2.2 Step 1 — Euler 离散化重写 Liquid Dynamics

- 把 LNN/LTC 的 ODE 显式 Euler 化为 `h_{t+1} = h_t + dt · f(h_t, x_t; θ)`。
- 这是**对本仓 `lnn/core/variants.py::EulerLTCNetwork` 思路的直接呼应** —
  本仓 smoke 已经在做 Euler-LTC-DT (见 [[../analysis/replication/temporal_dropout/temporal_dropout_report|temporal_dropout 报告]]),
  区别是本仓用 PyTorch 1.2.2 跑、研究者面向 MCU 部署。
- 这一步把 continuous-time "信息"完整保留为离散但**参数语义不变**的版本,后续 KD 不会因为
  时间常数被打平而失真。

### 2.3 Step 2 — Dual-Stage Knowledge Distillation

**Stage 1**: teacher (Euler-LNN) → student1,KD 目标函数推测为:

$$
\mathcal{L}_{KD} = \alpha \mathcal{L}_{MSE}(y_s, y_t) + (1-\alpha) \mathcal{L}_{MSE}(h_s^{(l)}, h_t^{(l)})
$$

即同时蒸馏 **输出** 和 **中间隐藏态时间序列**,后者是 LNN 的核心动力学指纹。

**Stage 2**: 在 student1 基础上做更激进的压缩 (e.g. 通道剪枝 / 量化感知训练),
然后**再次用 teacher 蒸馏恢复** — 关键点是 student2 的恢复式蒸馏,可以显著挽回
"被压缩打掉的时序信息"。

### 2.4 Step 3 — Pareto-guided Selection

- 联合目标: `(预测误差, 模型大小, 推理延迟)` 三维。
- 生成候选 student 集合 (不同的剪枝率 / 量化位宽 / 蒸馏温度组合),在 Pareto 前沿上保留。
- 论文选择"误差-成本双目标 Pareto",选择最优点 int8 student。

## 3. 关键成果与贡献

| 指标 | Teacher (LNN) | Student (DLNet int8) | 变化 |
|---|---:|---:|---:|
| 100-cycle SOH 预测误差 | 0.0078 (反推) | **0.0066** | **−15.4%** |
| 模型大小 | 616 kB | **94 kB** | **−84.7%** |
| 推理延迟 (Arduino Nano 33 BLE) | — | **21 ms / inference** | — |
| 部署精度 | fp32 | **int8** | — |

### 3.1 工程意义

- **Arduino Nano 33 BLE Sense** 实测 21 ms/次,Cortex-M4 @ 64 MHz 上完全实时。
- 84.7% 模型压缩 + 15.4% 误差下降同时达成,这是教科书级的"smaller wins"实证。
- int8 端侧推理验证了 LNN 不只是"实验模型",可以真正进入 BMS 量产。

### 3.2 对本仓的直接价值

1. **可作为 `lnn/core/variants.py::EulerLTCNetwork` 的延伸目标**:
   当前实现是 PyTorch 训练时形态;可加 `to_embedded()` 方法,
   输出 ONNX / TFLite-micro 友好的图。
2. **可作为边缘部署流水线模板**:
   `scripts/experiment_long_sequence.py --mode battery` + DLNet-style 三段式
   蒸馏,直接产出 Jetson / Orin / MCU benchmark。
3. **可作为 `PRD §8 #1` (Jetson 边缘部署) 的子任务**:
   与 [[../analysis/PRD_LNN_Edge_Research|PRD_LNN_Edge_Research]] §8 任务 #1 直接对接。

## 4. 局限与未来展望

- **数据集不公开**: 摘要只说 "a widely used dataset",复现第一步就是确认是否 NASA PCoE / CALCE / UNIBO。
- **蒸馏温度 / α 未公开**: Stage 2 恢复蒸馏的关键超参,正文应给出 ablation 但摘要不可见。
- **Euler 离散化精度**: 大 dt 时 ODE 数值误差增大;本仓 `EulerLTCNetwork` smoke 已验证小 dt 下稳定,
  但 BMS 任务可能采样间隔较大 (s 级),需要显式研究 dt 上界。
- **Pareto 目标仅 3 维**: 未考虑内存峰值 / 能耗,可扩展。
- **非电池领域迁移**: 论文声称"extend to other industrial analytics",但 ICPR 2026 接收范围内未必含
  非电池数据集验证。

## 5. 在本仓库的复现路线 (Replication Plan)

| 阶段 | 出口物 | 估时 (loop) | 阻塞依赖 |
|---|---|---|---|
| **A. 数据确认** | 选定 NASA / CALCE 数据集,准备 SOH label | 0.5 | 数据下载 |
| **B. 算子实现** | `lnn/core/variants.py::EulerLTCNetwork` + `to_embedded()` | 1 | — |
| **C. 教师训练** | `scripts/train_battery_teacher.py` → ckpt | 1 | B |
| **D. KD Stage 1** | `scripts/distill_battery_stage1.py` → student1 | 1 | C |
| **E. KD Stage 2 + 压缩** | `scripts/distill_battery_stage2.py` → student2 | 1 | D |
| **F. Pareto 选优** | `scripts/pareto_select_battery.py` → final int8 | 1 | E |
| **G. 设备验证** | Arduino / Jetson 实测 21 ms / 误差 | 2-3 | 硬件 |
| **H. 报告 v2** | `analysis/paper_replication/dlnet_report.md` | 0.5 | G |

A-F 全部在 CPU 上即可完成;G 需真实硬件 (Jetson / Arduino)。

## 6. 与本仓库已有工作的关系

| 现有资产 | 关系 |
|---|---|
| `lnn/core/variants.py::EulerLTCNetwork` | motif 完全相同;DLNet 是其端侧部署分支 |
| `analysis/replication/temporal_dropout/temporal_dropout_report.md` | 同为 ODE-based LNN 复现;模板可借鉴 |
| `scripts/experiment_long_sequence.py` | 长序列 smoke;可扩展 `--mode battery` |
| [[LNN_训练方向_边缘部署与压缩_可行报告]] | 路线图,本研读补 DLNet 这一支 |
| `[[../analysis/PRD_LNN_Edge_Research|PRD_LNN_Edge_Research]]` §8 #1 (Jetson) / #2 (MCU) | DLNet 同时覆盖 MCU 与中端 Jetson |
| `bench/` (缺失,需建立) | 缺 DLNet-style Pareto 选优工具 |

## 7. 推荐评级与下一步

- **学术贡献**: A-(ICPR 2026 + Pareto 选优 + 实测,工程完整度高于平均水平)
- **复现可行性**: **A**(代码仓库公开,流水线清晰)
- **对本仓优先级**: **A**(列入 [[../analysis/PRD_LNN_Edge_Research|PRD_LNN_Edge_Research]] §8 #2 候选)

### 立即可执行的下一步 (next loop)

1. 克隆 https://github.com/Dhivya-DD17/DLNet ,确认其依赖栈与本仓重叠度;
2. 在 `lnn/core/variants.py` 新增 `to_embedded()` 入口;
3. 跑 NASA PCoE B0005/B0006 训练 teacher,确认 paper "0.0066 SOH 误差"可复现;
4. 写 `bench/dlnet_pareto.py`,生成 Pareto 前沿,与论文图对比。

## 8. 参考链接

- arXiv: https://arxiv.org/abs/2601.06227
- 代码仓库: https://github.com/Dhivya-DD17/DLNet
- DOI: https://doi.org/10.48550/arXiv.2601.06227
- 关联路线图: [[LNN_训练方向_边缘部署与压缩_可行报告]]
- 本研读父索引: [[LNN_深度研读报告]]
- 复现脚本目标位置: `scripts/replicate_dlnet.py`(待创建)