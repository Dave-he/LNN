---
title: Exploring Liquid Neural Networks on Loihi-2
date: 2024-07-30
tags: [LNN, Neuromorphic-Computing, Loihi-2, Spiking-Neural-Networks, Image-Classification]
---

# 研读报告：探索 Loihi-2 上的液态神经网络

## 1. 元数据
- **论文标题**：Exploring Liquid Neural Networks on Loihi-2
- **作者**：Wiktoria Agata Pawlak (Tilburg University), Murat Isik (Stanford University), Dexter Le (Drexel University), Ismail Can Dikmen (TEMSA R&D Center)
- **发表时间**：2024 年 7 月 30 日
- **来源**：arXiv:2407.20590v1 [cs.ET]

## 2. 核心问题
生物脑在具备高能效和灵活性的同时，采用的是连续时间（Continuous-time）的动力学机制。然而，传统的深度学习模型大多基于离散的张量运算。
- 液态神经网络（LNN）作为受生物（如秀丽隐杆线虫）启发的连续时间计算模型，已展现出极高的参数效率和强大的分布外（OOD）泛化性能。
- 然而，如何将基于常微分方程（ODE）的连续 LNN 算法高效部署在硬件架构极其特殊的神经拟态（Neuromorphic）芯片（如 Intel Loihi-2）上，克服其离散时间步、有限存储和计算能力的限制，是一个关键的软硬件协同设计（Co-design）难题。

## 3. 方法论与核心思路
论文提出了一种将 LNN 部署在 Intel Loihi-2 类脑神经拟态芯片上的系统性工程方案：
- **前向特征抽取**：利用级联卷积层（Convolutional layers）对 CIFAR-10 图像进行空间特征提取，逐步降低数据维度，获得抽象表征。
- **神经回路策略集成**：将提取的空间特征集成到神经回路策略（NCP - Neural Circuit Policy）的决策决策层中，引导信息进入 LNN 动态液态处理层。
- **训练、量化与部署**：
  1. 使用 Adam 优化器与反向传播在 GPU（RTX 3060）上进行离线循环训练。
  2. 对训练后的模型参数进行**量化（Quantization）**，选择性地降低参数敏感性，以最小化芯片内存开销与计算延迟。
  3. 基于 Intel **LAVA 框架** 将模型编译为可执行文件，最终部署在 Loihi-2 类脑芯片上以高并发、超低功耗运行。

**上下文关系**：
本方案将 LNN 在连续时间建模上的信息提取能力（通过微分演化算子），与 Loihi-2 拟态芯片基于 7nm 制程、128 个核心的片上低功耗异步事件驱动处理能力相融合。避免了传统处理器在求解 LNN 常微分方程时的巨大延迟。

## 4. 核心公式提取
为了定量评估拟态硬件加速器上的乘累加（MAC）操作数量，论文构建了 LNN 计算密度的组件分解模型：

1. **LNN 嵌入层 MAC 消耗 (LNN Embedding)**
   $$ \text{MAC}_{\text{Embedding}} = D \times S $$
   *(注：$D$ 为输入特征维度，$S$ 为嵌入维度)*

2. **动态自适应层 MAC 消耗 (Dynamic Adaptation Layer)**
   $$ \text{MAC}_{\text{Adaptation}} = A \times t $$
   *(注：$A$ 为自适应调整开销，$t$ 为当前时间)*

3. **LNN 核心处理层 MAC 消耗 (LNN Processing Layer)**
   $$ \text{MAC}_{\text{Processing}} = N \times C \times t $$
   *(注：$N$ 为活跃神经元个数，$C$ 为单个神经元的平均连接数)*

4. **总计算消耗 (Total LNN MAC Operations)**
   $$ \text{MAC}_{\text{Total}} = \text{MAC}_{\text{Embedding}} + \text{MAC}_{\text{Adaptation}} + \text{MAC}_{\text{Processing}} $$

5. **吞吐量与延迟关系 (Throughput and Latency Analysis)**
   $$ \text{Latency} = \frac{\text{Total Inference Time}}{\text{Total Inference Samples}} $$
   $$ \text{Throughput} = \frac{\#\text{MACs}}{\text{Latency}} $$

## 5. 关键成果与贡献
论文在 CIFAR-10 数据集上开展了横向模型与硬件对比，取得了以下突破性成果：

- **更高的分类准确率**：基于 Loihi-2 的 LNN 架构达到了 **91.3% 的分类准确率**，超越了常规深度网络（DNN: 85.1%，CNN: 89.0%）和脉冲神经网络（SNN: 82.5%）。
- **极佳的超低能效**：在 Loihi-2 ASIC 7nm 架构上运行，单帧能耗仅为 **213 $\mu$J**。在同等精度水平下，远远优于基于 FPGA 或常规 CMOS 工艺的 ASIC 硬件。
- **最低的端到端延迟**：在所有对比模型中，LNN 实现了最低的端到端推理延迟（**15.2 ms**，而 SNN 延迟为 35.0 ms）。
- **超高的计算密度效率**：电力能效达到 **25.3 GOP/s/W**（SNN 仅为 8.2 GOP/s/W），并且总 MAC 运算量降至 **0.85 GOP**。

## 6. 局限性与未来展望
- **状态维持与有功功耗**：由于 LNN 内部状态是随时间连续流动的，为了保持“液态”动力学，硬件在持续接收信息时需要频繁更新状态，这可能带来额外的有功静态功耗。
- **映射兼容性限制**：在片上内存和算力极其受限的类脑核中，LNN 的连续状态转移微分方程需要进行更高程度的离散化和近似。这可能导致在更大、更深的模型结构中出现精度衰减。
- **应用泛化探讨**：目前的物理落地局限于 CIFAR-10 图像分类，未来应探索其在移动机器人自动导航、工业实时异常检测等高度依赖时间序列的动态环境中的实际表现。
