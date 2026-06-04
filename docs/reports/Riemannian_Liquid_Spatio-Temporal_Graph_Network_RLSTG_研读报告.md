---
title: Riemannian Liquid Spatio-Temporal Graph Network (RLSTG) — 研读报告
paper: arXiv 2601.14115v1 (DOI 10.1145/3774904.3792090)
authors: Liangsi Lu, Jingchao Wang, Zhaorong Dai, Hanqian Liu, Yang Shi
venue: WWW '26 (April 13-17, 2026, Dubai)
date: 2026-01-20
tags: [LNN, LTC, Riemannian, hyperbolic, spherical, ODE, tangent-space, graph, WWW-2026, paper-report]
status: deep-read
report-date: 2026-06-04
report-author: LNN-research-agents
---

# Riemannian Liquid Spatio-Temporal Graph Network (RLSTG) — 研读报告

> 论文: arXiv 2601.14115v1 (Lu et al. 2026), WWW '26
> 项目页: https://rlstg.github.io (**无直接代码仓**)
> 链接: https://arxiv.org/abs/2601.14115v1
> 与本仓直接相关度: **中** —— 是 LTC 的"非欧几何"扩展,但
>  1) 数据集 (ENRON 社交网络) 与本仓分子 / 时序任务不重叠;
>  2) 几何扩展 (双曲 / 球面) 与本仓 lnn/core/physics.py (欧几里得 ODE 嵌入) 是**两条路线**;
>  3) 但 LTC 公式 + 在 tangent space 的修改方式对 §3.4 LTCCell 仍是可参考的工程模式。

---

## 1. 一句话定位

> 把 **Liquid Time-Constant (LTC) 网络** 从**欧几里得空间**搬到**黎曼流形**
> (hyperbolic / spherical 等双曲-球面族),ODE 在 **tangent space** 上求解,
> 通过 **exp / log maps** 投影到流形上 — 解决"树状层级 / 环状结构"等
> 非欧图嵌入的 distortion 问题,**统一了 LTC 的连续时间动力学和
> Riemannian 流形的几何归纳偏置**。

应用: **spatio-temporal graph** 上 link prediction(动态图未来边预测)。

## 2. 公式对齐(LTC 在 tangent space)

### 2.1 经典 LTC ODE (欧几里得)

```
ẋ = -[1/τ + f(x, I, t)] · x + f(x, I, t) · A
```

### 2.2 RLSTG 的黎曼版本(论文 §3.3)

```
d/dt h_t = f(h_t, x_t, t; θ)         // 在 tangent space T_{h_t} M 上
h_{t+Δt} = exp_{h_t}(Δt · d/dt h_t)  // 沿流形 pushforward
```

- `f(·; θ)` 是 LTC 的非线性 gating 头
- `exp_p(v)` 是黎曼指数映射(把切空间向量映回流形)
- 论文用 **Riemannian ODE solver** 替换欧几里得 Euler/RK4
- **关键**: 切空间是局部欧几里得,所以 LTC 的 `f(·; θ)` **不需要修改** —
  - 只需在外层把 `Δt · f` 通过 `exp` 推回流形

### 2.3 双曲流形选择(hyperbolic / spherical)

论文动机: 真实图数据多为**树状层级** (社交 / 引用) 或**环状** (交通 / 分子)。
- 双曲空间(负曲率): 嵌入树状结构**失真小**(指数级容量)
- 球面空间(正曲率): 嵌入环状结构**自然**
- 论文用**双曲流形**作为主实验流形(社交网络是树状的)

## 3. 理论贡献

1. **ODE 求解器收敛性证明** — 论文 §4: 给"stiff dynamics on manifolds"的
   稳定求解器(stability 推广到黎曼域)
2. **LTC 通用逼近定理推广** — 经典 LTC 的 universal approximation
   在 tangent space 上保持(论文 §4.2)
3. **表达能力量化** — 论文 §4.3 给出与图结构 (tree-likeness, loops,
   structure stability) 的相关性

## 4. 关键成果与对照(论文 §5)

| 维度 | 论文结果 |
|---|---|
| 数据集 | **ENRON** 社交网络 (184 员工 / 3 年邮件) — 1 个 dataset,focused |
| 任务 | Link prediction (transductive + inductive) |
| Baselines | **10 个**: JODIE / DyRep / TGAT / TCL / TGN / GraphMixer / DyGFormer / HTGN / FreeDyG / HGWaveNet |
| 主要 metric | AP (Average Precision) for transductive / inductive |
| 量化结果 | 论文 §5 报 RLSTG **"significantly superior"** 于 baselines in irregularly-sampled + non-Euclidean (具体数字需 PDF 表) |
| 求解器比较 | 论文 §5.4 比较 **RLSTG-Euler / RLSTG-RK4 / RLSTG-Midpoint** —— 自家 ODE solver "比 RK4 准确度翻倍,Euler 速度但不准,Midpoint 平衡" |

## 5. 局限性(论文自承 + 我的批注)

| 维度 | 论文 | 我的补注 |
|---|---|---|
| 数据范围 | 1 个 ENRON dataset, focus on 邮件网络 | **与本仓分子 (tox21) / 时序 (mackey_glass) 不重叠**,需要新数据 |
| 几何 | 论文 demo 用 hyperbolic | 双曲/球面切换的开销 vs 收益未明说 |
| 计算 | RLSTG-RK4 比 Euler 贵 ~2× (论文自测) | 边缘部署需谨慎 |
| 代码 | ⚠️ **项目页 rlstg.github.io 仅有 demo,无官方代码仓** | 复现只能从 0 写 |
| 任务 | 只有 link prediction | 没覆盖 node classification / graph classification |

## 6. 对本仓库的价值

### 6.1 理论价值

| 本仓现有 | 与 RLSTG 的连接 |
|---|---|
| `lnn/core/ltc.py::LTCNetwork` | 论文 §3.3 给出 tangent-space 推广的**理论模板** |
| `lnn/core/physics.py::PhysicsInformedLNN` | 仓内已用 ODE 嵌入 NN,但都是**欧几里得**;RLSTG 模式可作为下一阶段扩展 |
| `scripts/experiment_graph_lnn_molecule.py` | 本仓是**欧氏 GNN** 上的 LTC;RLSTG 是**黎曼 GNN** 上的 LTC — **同一思想的两条路线** |
| `analysis/molecular/` (tox21) | ENRON (社交链接预测) 与 tox21 (分子分类) 是**两种图**;RLSTG 模式要迁到分子图需要新数据 + 新指标 |

### 6.2 工程价值

- **黎曼运算** (exp / log / parallel transport) **本仓目前没有** —
  需新依赖 (`geoopt` / `manifold` / `torchdyn`)
- **tangent space 的 LTC 公式** 可直接复用本仓 `LTCNetwork` 核心,
  仅外层加 `exp` 包装
- **理论证明是本仓空白** — 论文的 stability / universal approximation
  推广到黎曼域,如果要做完整复现需要写 formal proof

### 6.3 复现路线(stage 拆分)

| 阶段 | 出口物 | 估时 |
|---|---|---|
| A. 调研 + 写 `analysis/riemannian_lnn/2026-06-XX_design.md` 决定复现深度 | 设计文档 | 0.5 loop(本轮) |
| B. 装 `geoopt` + 写 `lnn/core/riemannian_ltc.py` (~120 行: tangent space LTC + exp/log wrapper) | code + unit test | 2-3 loop |
| C. 跑 ENRON link prediction toy (3 seeds, vs 1-2 baseline) | analysis + paper | 1-2 loop |
| D. 写复现报告 | docs/reports/Riemannian_LTC_复现报告.md | 0.5 loop |

## 7. 推荐评级 + 优先级

- **学术新意**: A(黎曼 + LTC 统一是新的,且 WWW '26 accepted)
- **工程价值**: B+(需要新依赖,但 tangent-space 复用本仓 LTC)
- **代码可获取**: C(无官方代码,仅项目页 demo)
- **本仓优先级**: **B** — 与 `experiment_graph_lnn_molecule.py` 是姐妹工作,
  但**复现 ROI 低于 DynPMNN**(有代码可参考)

## 8. 与本仓 6 套 LNN backbones 的关系

```
LTC (本仓核心)
├── 欧几里得: lnn/core/ltc.py::LTCNetwork            ← 已有
├── 黎曼:    lnn/core/riemannian_ltc.py (TBD)        ← RLSTG 模式
├── ODE 闭式: lnn/core/cfc.py::CfCNetwork            ← 已有
├── FHN ODE:  lnn/core/dynpmnn.py::FHNCell           ← 已有 (iter#23)
├── 频率 augment: cfc.PDNAPulseHead                  ← 已有 (iter#19)
└── τ 调制 augment: cfc.tau_modulated_blend          ← 已有 (iter#22)
```

RLSTG 模式 = **第 7 套 backbone 候选**: tangent-space LTC + exp/log wrapper。

## 9. 参考

- arXiv: https://arxiv.org/abs/2601.14115v1
- DOI: https://doi.org/10.1145/3774904.3792090
- Project page: https://rlstg.github.io

---

> 本报告由 LNN-research-agents 自动生成,基于 arXiv 2601.14115v1 PDF。
> 报告日期 2026-06-04,与项目 daily digest 同步。
