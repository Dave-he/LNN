# LNN 论文研读报告 — 2026-07-17

> 1h 循环研究第 1 轮。数据源:`papers/daily/2026-07-13_lnn_research.json`(25 篇 LNN 论文)+ `analysis/repo_watchlist/2026-07-17_lnn_open_source_watchlist.md`(开发生态观察)。
>
> 本轮目标:基于 6 月底至 7 月初最新 LNN 文献,挑出 **1 个本仓库尚未复现/扩展的机制** 落地为本轮 round。

---

## 一、文献全景(2026-06 → 2026-07,与本仓库对齐)

### 已覆盖/已实现(本仓库含相应代码或已有 round)

| ArXiv ID | 题目 | 本仓现状 |
|---|---|---|
| **2606.07670** | LNN as Drop-in Continuous-Time Deformation Field for Dynamic 3DGS | round 91(cfc temporal smoothness),已读但 1D 不复现 |
| **2606.12240** | Multi-Rate Mixture of Experts for LNN | **round 77 MR-MoE** ✅(K experts + softmax router) |
| **2606.19109** | Locally Stable Neural ODE via Lyapunov | 局部稳定架构,作为基准 |
| **2604.10815** | MeloTune on-device CfC agent | 类似控制/emotion 域 |
| **2603.00153** | **PDNA: Pulse-Driven Neural Architecture** | **round 135 PDNAPulseHead** ✅(在 `lnn/core/cfc.py:233`) |
| **2602.12139** | Oscillators Are All You Need(damped harmonic) | **round 128 OscillatorCfC** ✅ |
| **2601.14115** | Riemannian Liquid Spatio-Temporal Graph | 与 riemannian_lnn/ 域相关 |
| **2604.02149** | AEGIS thermodynamic SSM for zero-day | 与 control/ 域相关 |
| **2603.00459** | LSS-LTCNet continuous-time segmentation | 视觉域,参考 |
| **2604.24788** | Natural gas forecasting with LNN | 与 long_sequence/ 域相关 |
| **2604.18274** | LiquidTAD efficient temporal action detection | 视频域 |
| **2604.07219** | Liquid crystal antennas + LNN(6G) | 通信/天线域 |
| **2604.03955** | Symbolic-Vector Attention Fusion (SVAF) | 与 multimodal_physreg 域相关 |
| **2603.27058** | Liquid nets + MDN heads imitation learning | 类似 liquid_tad_imitate/ |
| **2602.06997** | Adaptive temporal dynamics for EEG emotion | emotion 域 |
| **2601.06227** | DLNet dual-stage distillation for LNN edge | 与 lfm25/ 蒸馏相关 |
| **2605.08176** | DynPMNN physics-modeled continuous-time | 与 physics.py 相关 |
| **2605.24047** | EMMA multimodal physics recovery | 已有 emma_drone/ + emma_rover/ |

### 未充分覆盖/仍可扩展

| ArXiv ID | 题目 | 洞察 |
|---|---|---|
| **2606.21295v6** | **Topological Neural Dynamics (TND): Neuron-wise Framework** | **每神经元独立 ODE + 显式 graph topology**(对比 layer-wise dense matmul)。论文在单玩家 Pong 上把 catch rate 拉到基线 3×。**本仓尚未实现 per-neuron dynamics**。本轮目标。 |
| **2607.08283** | TFP: Temporally Conditioned Memory-Fusion for Visuomotor | 在 VLA 政策上加 "stage memory";**memory phase fusion 概念** 与本仓 round 99 segment-reliability 类似。备选思路。 |
| **2607.01986** | Liquid Latent State Dynamics for Turbofan Degradation | LNN 作 latent dynamics model,用 C-MAPSS benchmark;**强调 interpretability through state**,与 round 80-95 interpretability 路径共振。诊断而非新模型。 |
| **2606.26849** | Liquid Fusion of Heterogeneous Representations (SOD) | LNN 子模块做显著性检测;**特征级 fusion**,与现有 SVAF/quasi-reverse 不同。 |
| **2606.19579** | FlowFake: Liquid Networks for Audio Deepfake Detection | 音频 + 跨数据集泛化;**与 audio_snr_threshold_scan/ 正交**,提供新 benchmark。 |
| **2606.15807** | Memory-Augmented Graph Liquid Time-Constant | 把 memory 显式注入 LTC 的 graph 结构;与 dynamic_tmoe 共振但更轻量。 |
| **2606.15571** | Liquid Random Feature Methods for time-dependent PDE | 半解析方法,工具性高于模型性。 |
| **2606.20491** | GazeLNN fixation-guided active perception | 注意力预测 + LNN;saccade + LNN 的小尺寸模型。 |

### 开源生态(2026-07-15 ~ 2026-07-17,来自 Hugging Face watchlist)

- **Liquid AI LFM2.5**:VL-450M / VL-1.6B 已有 GGUF 蒸馏版(7861+56 likes / 4059+44 likes)
- **Synaptics/SL2619**:Torq 平台上跑了 `LFM2p5-230M-LLM`——`onnx` + `lfm2` + `synaptics` + `npu` + `edge` ——**LFM2 在 NPU/SoC 上量化部署就绪**(1093 downloads)
- **社区蒸馏持续**:reaperdoesntknow / simaai / hauser458original 都出了 LFM2.5 GGUF/autoround 蒸馏版
- **LFM2.5-8B-A1B-Opus-Distil**:`lfm2_moe` + `instruction-tuning` + `reasoning` —— 8B A1B MoE + reasoning
- 含义:**LFM2.x 边缘推理生态已经成熟**(Liquid + Synaptics + Llama.cpp + ONNX);继续发论文级液体模型的同时,**真正的瓶颈是 substrate / NPU 适配 / 蒸馏配方**

---

## 二、本轮研究思路选择与依据

### 候选机制(已逐一排除)

1. ~~PDNA (2603.00153)~~ — 已实现(round 135)
2. ~~Damped harmonic oscillator (2602.12139)~~ — 已实现(round 128)
3. ~~Multi-Rate MoE (2606.12240)~~ — 已实现(round 77)
4. ~~Curvature routing (2606.10x)~~ — 已实现(round 101 ORC)
5. ~~Riemannian graph LNN (2601.14115)~~ — 在 riemannian_lnn/
6. ~~Liquid Random Feature PDEs (2606.15571)~~ — 工具性,不直接对应本仓栈
7. ~~TFP visuomotor stage memory (2607.08283)~~ — 域外,需要 visuomotor 数据集

### ★ 本轮新方向:Topological Neural Dynamics on CfC(2606.21295v6)

**核心思想:** 当前主流 sequence model(RNN/LSTM/CfC/Transformer)都是 **layer-wise dynamics**——同一层所有神经元共享同一个参数算子(W·h + b),各神经元无独立自由度。TND 提出 **neuron-wise dynamics**——每个神经元是一个独立的连续时间单元,通过**显式学习的有向 graph** 相互耦合。

**为何本仓尚未实现:**
- 我们现有 MoE 路线在 **expert 粒度** 做 routing/多样性的文章(round 76-104),但 expert 粒度的多样性 vs **neuron 粒度** 的独立性 是正交问题
- 我们所有 CfC 变体(osci/PDNA/pulse/tau/blend/etc.)都还是 **layer-wise matmul**,只是加 gating 或 auxiliary regularization
- TND 的 "per-neuron ODE + learned sparse adjacency" 是**结构性替代**,不是 regularization

**可验证假设:**
- H1: TopologicalCfC 在玩具 sin 任务上保持 CfC 的 msev 水平(结构兼容)
- H2: 稀疏拓扑(每神经元仅连 k<<H 个邻居)可以学出非平凡结构(不同神经元在 vs 时呈现差异化的局部动力学)
- H3: 在 smaller hidden_size 下,sparse-neuron-graph 比 dense-matmul 参数效率高(每神经元仅需 2-3 个 graph 邻居权重而非 H 个)
- H4: 与 FAME MoE 正交(后者 expert-粒度,本机制 neuron-粒度) — 二者可叠加

**潜在负面**(诚实预期):
- 91-101 audit 已证明大部分"正交多样性"机制在 1D 上是 honest-negative(round 100 SNNL +22% regression, round 101 ORC +89% regression on toy_sin)
- TND 的 per-neuron graph 与 CfC 的 layer-wise matmul 互为替代品,在 1D 不太可能胜出 baseline
- **预期结果:** H1 ✓ 形状兼容;H2 ✓ 学出非平凡图;H3 ✗ 参数效率优势在玩具不显现;H4 留作 honest-positive 候选

---

## 三、本轮实现计划(Round 299 — TopologicalCfC)

### 代码
- `lnn/core/topological_cfc.py`(新文件,~250 行)
  - `TopologicalCfCCell(input_size, hidden_size, n_subunits=hidden_size, graph_k=8)`
  - `forward(x, hx)` 返回 `(h_new, h_new)`
  - 每个 "subunit"(神经元)跑独立 ODE,邻居通过 `topo_adj` (B×subunits×subunits) 传递
  - 默认 `topo_adj` 由一个 `(H, k)` 的 (neuron, neighbor_idx) embedding 学得,forward 时 broadcast 成 dense
  - 关闭: `use_graph=False` 即退化为"独立神经元"无耦合,可在测试中对比

### 测试
- `tests/test_topological_cfc.py`(≥15 unit tests,模仿 `test_pdna_pulse.py` 风格)
  - shape determinism、 backward flow、 sparse vs dense 对比、 graph 稀疏度可学、non-trivial邻居选择
- 玩具三件套 toy_sin/structured/random × {CfC baseline, Topological-CfC k=4, k=8, k=H/2}

### bench
- `scripts/bench_topological_cfc.py`(可选,如果时间允许)
- bench_suite.py 一致接口,3 seeds × 100 epochs,记录 test_mse/forward latency/params

### 文档化
- `analysis/research/2026-07-17/2026-07-17_round299_topological_cfc.md`(本轮 round 报告)
- 更新 MEMORY.md(round 299 摘要)
- 由于本轮 git push 阻塞,本轮 commit 仅留本地;round 报告标注 push blocker

---

## 四、参考文献

1. Cai B., Zhao Y. (2026). **Topological Neural Dynamics: A Neuron-wise Framework for Sequence Modeling**. arXiv:2606.21295v6. https://arxiv.org/abs/2606.21295
2. Liquid AI (2025-12). **LFM2**: Hybrid Convolution-Attention for the Edge. Blog + HF release.
3. Sharma P. (2026). PDNA — already implemented (round 135).
4. Hasani R. et al. (2022). **Closed-form continuous-depth networks**. Nature MI. (Foundational CfC paper.)
5. Mem 引用: rounds 76-104 (MoE stack), 128 (oscillator), 135 (PDNA), 297 (decorrelation default).

---

**研读结论:** 7 月份 LNN 文献主线已转向 **应用层**(医疗/通信/边缘/电池/气体/音频),模型层创新集中在 (a) 多模态融合 (b) per-neuron 结构(TND) (c) interpretability through state(Turbofan)。**结构层创新——per-neuron ODE + 显式拓扑——是本仓未覆盖的 axis**,本轮实现。

— 2026-07-17, 1h /loop cycle #1
