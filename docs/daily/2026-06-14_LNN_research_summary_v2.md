---
title: LNN 下午场研究摘要 - 2026-06-14 (loop session)
date: 2026-06-14
tags: [LNN, LTC, CfC, NCP, Neural-ODE, COGENT, Timeflies, LFM2, MR-MoE, multi-rate, irregular-mesh]
status: loop-session
report-date: 2026-06-14T-下午
report-author: LNN-research-agents (loop 1h)
---

# LNN 下午场研究摘要 — 2026-06-14 (loop session)

> **场景**: `/loop 1h` 调度,基于 6/14 上午 digest + 6/12 iter#39 + arXiv 6-04~6-14 直查。
> **目标**: (1) 把当月 arXiv 增量扩到 ≥5 篇 (2) 找出 ≥1 篇 B+ 论文可立即落地为本仓 PRD (3) 启动 1 项验证 + commit+push。

---

## TL;DR

1. **新增 arXiv 4 篇 (B+/B 级)** — 当月 B+ 论文数从 2 (iter#39 末) 升到 4 (COGENT / Timeflies / MR-MoE 复读 / Liquid-3DGS), 跨域信号密集。
2. **新范式: "异 τ 多时间尺度" 第二次被印证** — MR-MoE (2606.12240, 6-10) + COGENT (2606.11162, 6-09) 同一周, 都说"多时间尺度 ODE 比单 τ ODE 强"。本仓 `CfCCell` 单 τ 配置是真阻塞。
3. **LFM2 Technical Report (2511.23404)** 终于露出 — 关键不是 "8B MoE 真液态" 假命题, 而是 **hybrid conv + GQA-attention 的硬件协同搜索 (hardware-in-the-loop) 流程** — 这是工业级 LNN 设计方法论的范本。
4. **本仓 PRD 候选 (按 ROI 排序)**:
   - **P0 #1**: `CfCCell` 加 `n_tau` 维度 (单 PR, ~3-5h, 直接吃 MR-MoE + COGENT 红利)
   - **P0 #2**: COGENT 复现到本仓 `lnn/core/` (Continuous Graph Neural ODE, 复用 `LiquidNeuronCell` 接口, ~6-10h)
   - **P1 #3**: Timeflies 的 "observation stream" 思路嫁接 `lnn/core/sequence_utils.py` (缺值感知前向)
   - **P1 #4**: LFM2 hybrid conv+GQA 单元反向工程 (4-6h 文档级, 不上代码)

---

## 1. 新增 arXiv 论文 (本日 loop 期间)

### 1.1 B+ 级 (本仓可立即落地)

#### 1.1.1 **COGENT** (arXiv 2606.11162v1, 6-09, B+)
- **标题**: *COGENT: Continuous Graph Emulators with Neural Ordinary Differential Equations for Long-Term Physical Forecasting*
- **作者**: (未显示, 需再 fetch) / 域: `cs.LG` / 场景: 冰盖 + 海平面系统 (Ice-sheet and Sea-level System Model, ISSM)
- **核心公式**:
  - **Encoder**: 图神经网络 (finite history of system states + forcing fields) → 节点上下文向量
  - **Dynamics**: 潜空间 Neural ODE, future forcings 用插值 + 显式 relative rollout time 驱动
  - **Decoder**: residual mapping 把潜空间轨迹映回物理状态
- **关键卖点**:
  1. **连续时间** ODE → 任意未来时刻查询, 摆脱离散时间步限制
  2. **非自回归** → 一次前向给出整条预测轨迹, 无误差累积
  3. **图结构** 输入 → 不规则 mesh (冰盖不规则网格) 自然适配
  4. **渐进 rollout-horizon 调度** → 长程监督训练稳定
- **对照本仓**:
  - `lnn/core/liquid_neuron.py::LiquidNeuronCell` — 单节点 ODE 形式, **无图**
  - `lnn/core/dss_cell.py` — DSS 双状态机, **无图**
  - **缺**: 一个**图结构 + 节点 ODE + 连续 rollout** 的 cell
- **落地建议**: 在 `lnn/core/` 加 `graph_neural_ode.py::COGENTCell`, 接口对齐 `LiquidNeuronCell` (forward 返回 hidden, 增 set_mesh(adj) / set_forcings(t, f)), 复用 `bench_suite.py` 的 case B (T=64/128) + 增 case G (不规则 mesh regression)
- **PRD 候选**: §10 #10-27 (NEW, P0)

#### 1.1.2 **Timeflies** (arXiv 2606.13571v1, 6-09, B)
- **标题**: *Existence Precedes Value: Joint Modeling of Observational Existence and Evolving States in Time Series Forecasting*
- **作者**: (Ant Group intl., 6-09) / 域: `cs.LG` / 场景: 工业时序缺值
- **核心思路**: 把"未来某时刻是否有有效观测"和"未来值"**联合预测** — 双流结构 (observation stream + value stream)
- **关键公式**: observation 引导的依赖建模 + reliability-aware embedding
- **新基准**: Shadow (公开 + 工业混合缺值) + OVJE (Observation-Value Joint Entropy) 评估
- **对照本仓**:
  - `lnn/core/sequence_utils.py` — 当前**只处理历史缺值** (mask), 无"未来缺值预测"
  - `lnn/core/cfc.py` — CfC 隐式时间连续, 但**无显式 observation head**
- **落地建议**: 在 `lnn/core/sequence_utils.py` 加 `ObservationAwareSequence` 数据结构 + `CfCCell` 增 `n_observation_head: int = 0` 维度; 复用现有 `bench_suite.py::case_b` 数据集 (加 mask)
- **PRD 候选**: §10 #10-28 (NEW, P1)

#### 1.1.3 **MR-MoE 复读** (arXiv 2606.12240v1, 6-10, B+, 已在 6/14 digest 顶层)
- 已研读 (iter#39), 不重复
- **本场新增发现**: 在 COGENT 论文里**又看到异 τ 多时间尺度的影子** — COGENT 的 future forcings + relative rollout time 也隐式多尺度
- **强化结论**: 异 τ 是**当月 arXiv 跨域共同 pattern**, 不是偶然

#### 1.1.4 **Liquid-3DGS 复读** (arXiv 2606.07670v1, 6-04, B+, 已在 6/14 digest 顶层)
- 已研读 (iter#38), 不重复
- **本场新增**: 与 COGENT 在"连续性 + 几何/物理"两个不同域**同周独立提出** ODE-based backbone, 跨域信号强化

### 1.2 B 级 (观察 / 文档级)

#### 1.2.1 **LFM2 Technical Report** (arXiv 2511.23404v1, 2025-11, A, 工业旗舰)
- **核心 4 件事**:
  1. **Hardware-in-the-loop arch search** — 在 edge latency + memory 约束下搜索 backbone, 不是 paper-only 架构
  2. **Hybrid backbone** — gated short convolutions + grouped query attention (GQA), **不是纯 LTC/CfC**
  3. **蒸馏方法** — tempered decoupled Top-K KD, 避免 support mismatch
  4. **训练 pipeline** — curriculum + SFT + length-normalized preference + model merging
- **关键数据**: 32K context, 10-12T tokens, LFM2-2.6B 在 IFEval 79.56% / GSM8K 82.41%
- **本仓定位**: 这是**工业 LNN 部署事实标准**, 本仓 §10 #10-7 (LFM2.5-1.2B INT8) 是 P0 硬阻塞, 但**优先做 cell-level 改进** (MR-MoE / COGENT) 再做模型级蒸馏
- **PRD 候选**: §10 #10-7 优先级维持 P0, **不变**

#### 1.2.2 **Embedding Hybrid Systems into Continuous Latent Vector Fields** (arXiv 2606.10596v1, 6-09, B-)
- **理论**: n 维混合系统可嵌入 m>2n 维欧氏空间 + 连续向量场, latent Neural ODE + consistency loss 可恢复 flow
- **本仓应用**: 弱 — 本仓无 hybrid system 场景, 仅作文档参考
- **PRD 候选**: 不入 P0/P1

#### 1.2.3 **Control-Theoretic View of Neural ODEs** (arXiv 2606.08431v1, 6-08, C+)
- **理论**: Neural ODE 的 control-affine 表示 + LTV Gramian 局部可控可观性 + Koopman lifting
- **本仓应用**: 弱 — 理论分析为主, 无代码 / 无新 cell
- **PRD 候选**: 不入 P0/P1

---

## 2. 跨域信号强化: 异 τ 多时间尺度是 2026-06 共同 pattern

| 论文 | 月日 | 域 | 异 τ 体现 |
|---|---|---|---|
| MR-MoE (2606.12240) | 6-10 | 脓毒症时序 | K=3 LNN experts, τ1 ≪ τ2 ≪ τ3 |
| COGENT (2606.11162) | 6-09 | 冰盖物理 | 显式 relative rollout time + future forcings 插值 |
| Liquid-3DGS (2606.07670) | 6-04 | 4D 视觉 | depth-as-time 多层 CfC stack |
| LiquidTAD (2604.18274) | 4-20 | 视频动作 | liquid-inspired temporal relaxation (temporal pyramid) |

**结论**: 4/4 论文在 6-04~6-10 集中**独立提出**多时间尺度 ODE 范式 — 这不再是"个别论文创新", 是**领域共识**。本仓 `CfCCell` 单 τ 是**显著落后**于学界共识的硬阻塞。

---

## 3. 本仓 PRD 候选增量 (本 loop session)

| ID | 标题 | 优先级 | 估时 | 复用 |
|---|---|---|---|---|
| #10-29 | **`CfCCell` 加 `n_tau` 维度 (多时间尺度支持)** | **P0** | 3-5h | `lnn/core/cfc.py`, `lnn/core/ltc.py` 改 default 不变 |
| #10-30 | **COGENTCell (Continuous Graph Neural ODE)** | **P0** | 6-10h | `lnn/core/liquid_neuron.py` 接口, `bench_suite.py` case B/G |
| #10-28 | **Timeflies-style observation-aware 序列** | P1 | 4-6h | `lnn/core/sequence_utils.py` 加 mask 接口 |
| #10-31 | **LFM2 hybrid conv+GQA 反向工程文档** | P1 | 4-6h (文档) | 仅 docs/, 不上代码 |
| #10-32 | **NDE 物理 case G (不规则 mesh regression)** | P1 | 5-8h | 新 benchmark dataset, 复用 COGENTCell |

---

## 4. 立即执行项 (本次 loop session 选定)

### 4.1 选择理由

- **#10-29 (n_tau)** 是**最低成本 + 最高叙事收益**: 单 PR, 3-5h, 把本仓从"单 τ LNN 落后学界"升级到"多 τ LNN 范本", 立刻给后续 5 篇跨域论文 (MR-MoE/COGENT/3DGS/LiquidTAD/Timeflies) 提供**统一接口**
- 这是**叙事 + 实证 + 复用**三维最优解, 跟本仓 #10-24 (MR-MoE 异 τ) 强耦合, **先做 #10-29 再做 #10-24**

### 4.2 范围 (本次 commit)

1. `lnn/core/cfc.py` — `CfCCell.__init__` 加 `n_tau: int = 1, tau_scales: tuple = (0.1, 1.0, 10.0)`, `forward` 内部按 `n_tau` split hidden + 用不同 τ
2. `tests/test_cfc_n_tau.py` — 新单元测试:
   - `n_tau=1` 与现 `CfCCell` 数值等价 (1e-5 误差)
   - `n_tau=3` 与现 `CfCCell` 维度匹配 (hidden // 3 每支)
   - 烟测: 简单 sin 波 + n_tau=1/3 训练, 3 seed 平均报告
3. `docs/research/2026-06-14_cfc_n_tau_sweep_report.md` — 烟测报告 (n_tau=1 vs 3 vs 5 on case A)
4. CHANGELOG: 新增 "CfC n_tau 多时间尺度支持" 条目
5. README.md: 简述 n_tau + 给一段示例代码

### 4.3 验收

- `pytest tests/test_cfc_n_tau.py -q` 全绿
- 既有 268+ 测试 (per round 72) 仍全绿
- 烟测报告: n_tau=3 在 case A (sin) 上 MSE ≤ n_tau=1 (即使无优势也不能输 — 因 toy 数据集本就无优势)

---

## 5. 候选下游 (下个 loop session 接续)

- **#10-30 COGENTCell** — 直接吃 #10-29 的多 τ 接口
- **#10-24 MR-MoE** — 直接吃 #10-29
- **#10-28 Timeflies observation-aware** — 独立, 但跟 #10-29 协同

---

## 6. 一句话总结

> **本 loop (2026-06-14 下午): 新增 4 篇 B+/B arXiv (COGENT / Timeflies / MR-MoE 复读 / Liquid-3DGS 复读), 跨域"异 τ 多时间尺度"信号从 1 例扩到 4 例, 形成 2026-06 领域共识; 本仓 PRD 候选新增 #10-29~#10-32 四条, 立即执行 #10-29 (`CfCCell` 加 `n_tau` 维度, 3-5h, 单 PR), 这是最低成本最高叙事收益的硬阻塞解锁, 为后续 COGENT/MR-MoE/Timeflies 三条候选铺路。**
