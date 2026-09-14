---
title: MT-LNN (AwareLiquid/M1) — 研读报告
repo: https://github.com/AwareLiquid/M1
mirror: https://huggingface.co/EverestAn/MT-LNN
paper_pdf_en: https://huggingface.co/EverestAn/MT-LNN/resolve/main/mt_lnn_arxiv.pdf
paper_pdf_zh: https://huggingface.co/EverestAn/MT-LNN/resolve/main/mt_lnn_arxiv_zh.pdf
date: 2026-09-14
tags: [LNN, LTC, microtubule, bio-inspired, LLM, attention-free, O1-memory, continual-learning, edge-ai, retraction]
status: deep-read
report-author: LNN-research-agents
source: analysis/repo_watchlist/2026-09-14_lnn_open_source_watchlist.md GitHub 候选 (AwareLiquid/M1, 9/11 更新)
---

# MT-LNN (AwareLiquid/M1) — 研读报告

> 仓库: https://github.com/AwareLiquid/M1
> 源: 9/14 daily digest GitHub 候选 AwareLiquid/M1 (9/11 更新, 1★, Python)
> 报告日期: 2026-09-14 | 工具: `skills/paper-analyzer` (GitHub-repo 模式)
> 注意: 仓库 README 已自我撤回多条早期 claim, 研读时严格区分 **validated** / **retracted** 两栏

---

## 1. 一句话定位

> MT-LNN = **Microtubule-inspired Liquid Neural Network**, 借鉴生物微管的 13 根 protofilament 结构 + GTP-cap 动力学 + LAVI 节奏, 把每个 transformer block 替换成 13 路并行 Multi-Scale Resonance 单元, 实现 **attention-free 路径下 O(1) 推理记忆** (与上下文长度无关的固定状态, 0.381 MB @ hidden=832) + **跨 session bit-exact 状态快照/恢复** + **continual learning 零新增参数** 的 bounded-replay 实现.

仓库里同时维护两条产品线:
- **O-series (ARR)**: attention-free, 边缘 / 流式推理, 严格 O(1) 记忆
- **M-series**: 在 transformer block 之上挂 MT-adapter, 云端 GPU 推理, 全 base 质量

> ⚠️ **本仓库 1★** = 仍是单人早期项目, **不要当作生产级 LNN 库引用**. 适合做 **架构思想 + 概念验证** 来源, 不可作为 benchmark baseline.

---

## 2. 数据流与架构 (来自 README + 仓内 ARCHITECTURE.md)

```
Token Embedding + RoPE
  → n_layers × MTLNNBlock
      ├── MicrotubuleAttention (GQA + KV cache + SDPA/Flash-Attn)
      │     ├── 标量 polarity bias
      │     ├── GTP-cap ALiBi log-bias
      │     └── 可选 low-rank bilinear polarity σ(xWa)(xWb)ᵀ
      ├── MTLNNLayer (recurrent h_prev cache, parallel scan)
      │     ├── d_model → 13 protofilaments (d_proto = d_model/13, 832/13 = 64)
      │     ├── 13 × 5-scale MultiScaleResonance (几何 τ 扫, softmax blend)
      │     ├── κ-gate
      │     ├── 可选 LAVI rhythm (EEG 启发: cos(h, h_prev) 切 slow/fast τ)
      │     ├── LateralCoupling (static 13×13 + NN roll + RMC content-aware)
      │     └── MAPGate (fc2_bias=+2, 初始化近 open)
      └── 可选 GWTBLayer / CompetitiveGWTBLayer
  → GlobalRhythmController
  → GWTBLayer 或 CompetitiveGWTBLayer
  → GlobalCoherenceLayer (含 Orch-OR collapse gate, **已撤回**)
  → [PredictiveStateHead]
  → LayerNorm → lm_head
```

### 2.1 微管启发的具体对应

| 生物微管特征 | MT-LNN 实现 | 注释 |
|---|---|---|
| 13 根 protofilament | 13 路并行 MultiScaleResonance | 显式写死 (d_model 需 13 整除, hidden=832 严格满足) |
| 时间常数 τ 分布 | 几何 τ 扫 (5-scale) | 初始化自生物先验; 冻结-τ ablation: **0.285 vs 0.621 trained recall** |
| GTP-cap 切换 | 耦合 gating `exp(-γ·(t mod T_period))` | 周期性解耦, 引入"事件边界" |
| EEG 振荡 | LAVI rhythm (cos 相似度切慢/快 τ blend) | 与本仓库 [[lnn.edge]] 可结合验证 |
| Orch-OR collapse | GlobalCoherenceLayer 内的 Φ̂ 门 | **已撤回** (AVP failed, Φ̂ sign inverted, 训练路径 inert) |

### 2.2 O(1) 工作记忆的来源

- 完全 attention-free 的 **O-series**: recurrent 状态 `(F, z)` 维度固定, **0.381 MB @ hidden=832**
  - 128k context: 1008× 比 fp16 GQA=1 小, 1070.9× 比 2-bit + GQA=8 小
  - 1M context: 8063× / 8567×
  - crossover: 17–976 tokens (低于此区间 Transformer 仍占优)
- **M-series 不是 O(1)** — 它在 transformer block 上叠 MT-adapter, 训练记忆 > vanilla transformer
- 流式 state-only 解码: 1000-token decode footprint `~1020 KB → 4.1 KB` (丢 KV, 仅留 recurrent h_prev)
- 测试钉子: `tests/test_long_context_memory.py` 验证 state 在 T = 20× RoPE window 内保持平坦

### 2.3 Continual Learning 主张

- `examples/demo_streaming_continual.py` 给出对照:
  - Naive sequential → catastrophic forgetting (~1/n_tasks acc, forgetting ≈ 1.0)
  - Bounded reservoir-replay (via `replay.py`) → forgetting ≈ 0, acc ≈ 1.0, **零新增模型参数**
- 评估: `continual_eval.py`

---

## 3. Validated vs Retracted 严格分栏

### 3.1 ✅ Validated (README + 仓内测试支持)

| 主张 | 数值 / 证据 |
|---|---|
| 125M LM PPL @ WikiText-103, 20K 步, 3 seeds, fp32 | modern Transformer **78.86 ± 0.25** < MT-LNN **88.93 ± 0.33** < simple Transformer **94.14 ± 0.78** — 即: MT-LNN 比 modern Tx 仍**差 ~10 PPL**, 比 simple Tx 略好 |
| 跨窗口 associative recall | MT-LNN fast-weight **0.56** vs attention/LoRA **0.000** (结构性零) — 删 fast-weight 塌缩到 0.008 |
| 跨 session 快照/恢复 | `(F, z)` bit-exact round-trip |
| 测试套件 | 967 pass (`pytest tests/`), 快路径排除 12 slow (955 in ~99 s) |
| O(1) state | `tests/test_long_context_memory.py` 直接钉住 |
| 隐维 832 = 13×64 | 严格整除, 隐藏维度与 protofilament 数耦合 |

### 3.2 ❌ Retracted (README 自撤回)

| 撤回主张 | 替代说明 |
|---|---|
| "MT-adapter −28.5% / −27.7% / −34.4% PPL" | 那些 run 实际只 train LoRA, 数值不归功于 MT-adapter |
| "原生 125M −31% vs Transformer @ 2K 步" | 欠训练 + 弱 baseline, 撤稿 |
| "irregular-sampling edge vs GRU-D baseline" | 撤回 |
| Orch-OR / Φ̂ / AVP "consciousness" 结果 | 训练路径 inert, AVP failed, Φ̂ sign inverted — **不应在任何 consciousness/AGI 论点中引用** |
| frozen-MT + LoRA-only ≈ LoRA-only | PPL 7.98 vs 7.92 — 早期 "MT 显著增益" 论断不成立 |

> 评估原则: **引用 MT-LNN 数据时只引 §3.1; 涉及 consciousness/AGI/通用 intelligence 含义的声称直接判否.**

---

## 4. 依赖栈与本仓库对接路径

### 4.1 MT-LNN 依赖栈

- 主要 `requirements.txt` + 钉版本 `requirements.lock{,.cpu,.gpu}` + `pyproject.toml`
- 目标 HF Transformers (recipes wrap `AutoModelForCausalLM`)
- CPU / GPU Docker 双镜像 (`Dockerfile`, `Dockerfile.gpu`)
- 部署目标: Modal / Kaggle / Colab / 本地 Docker
- 可选 `pyphi` 跑精确 IIT 4.0 Φ (`mt_lnn.phi_iit`) — **不建议在生产环境启用** (Φ̂ sign 撤回)

### 4.2 本仓库 `lnn/` / `scripts/` 可对接点

| 本仓库资产 | MT-LNN 增量 |
|---|---|
| `lnn/data/timeseries.py` + `lnn/data/real_data.py` | 可用 **O-series recurrent mixer** 替换当前 LTC/CfC, 跑不规则采样 + 跨 session 状态恢复对比 |
| `lnn/data/emma_rover_*.py` | EMMA-Rover 序列本身就是好的 continual-learning benchmark 场景 (任务漂移), 可直接套 `replay.py` 验证 bounded-replay zero-param 主张 |
| `scripts/bench_irregular_dt.py` | 已有不规则采样 benchmark, 可挂 MT-LNN 的 13-protofilament vs 单 LTC baseline |
| `scripts/bench_hierarchical_multitau_cfc.py` | 已有 5-scale τ 扫描基建, MT-LNN 的 geometric τ ladder 思路可直接对位 (只是 13 路并行) |
| `scripts/experiment_concept_drift.py` | 概念漂移实验 + MT-LNN continual learning zero-param 主张 = 强对照 |
| `lnn/edge/tegrastats.py` + `scripts/export_lnn_tensorrt.py` | O-series 0.381 MB state + 4.1 KB decode footprint 直接适合 Jetson Orin Nano 边缘, 可补入 `docs/reports/Orin_Nano_Super_LNN_Deployment_v2_2026-08-03.md` 横向对比 |
| `analysis/sncp_ppo_lite` + 9/14 digest `heimdilon/sncp-ppo-crowdnav` | SNCP-PPO-Lite 用 LTC 做 crowd nav, MT-LNN O-series 是更激进的 LTC 替代, 可作下代架构 |
| `analysis/multimodal_physreg` | MT-LNN 的 MicrotubuleAttention polarity bias + LAVI rhythm 是多模态融合的天然 feature, 可探索 |

---

## 5. 与其他 LNN 工作的横向定位

| 项目 | 关注点 | MT-LNN 差异 |
|---|---|---|
| LiquidAI/LFM2 / LFM2.5 | production edge LLM, 1.2B-24B | MT-LNN 是 research-only O-series + MT-adapter 混合; LFM 系列已 production |
| `aygp-dr/liquid-neural-networks` (22★) | C. elegans 启发的 19-302 神经元小模型 | MT-LNN 走 microtubule 路径, **不**是 C. elegans 路径 |
| `kds1123001/liquid-time-constant` (2★) | Mojo-native LTC for edge robotics | MT-LNN Python + HF Transformers; Mojo 版走 SIMD RK4 不同方向 |
| `The-Silly-Glitch/cfc-async-fusion` (8/31) | CfC for 接触丰富操作的异步融合 | CfC 仍是 backbone, MT-LNN 用 microtubule 拓展 CfC |
| `Dmelon666/PhysLTCNet` (8/30) | physics-aware ODE-guided LTC | 单变量生产预测, MT-LNN 走 LLM scale |
| PDNA (arXiv:2603.00153, 2 月) | CfC + pulse + self-attend | MT-LNN 与 PDNA 同属 "在 CfC 上加结构", 但 PDNA 走振荡, MT-LNN 走 microtubule |
| EntroLnn | entropy-guided transformable LNN | 概念层相近, 实现路径不同 |

**MT-LNN 的独特价值**:
1. **明确给出 attention-free O(1) 边界** (17–976 token crossover), 给"液态 RNN vs Transformer"的混合架构决策提供了**精确拐点**, 不再是经验主义
2. **跨 session bit-exact snapshot/restore** 在 LNN 领域罕见, 对 long-running agent / streaming continual learning 极有价值
3. 自我撤回机制透明 — README 明确列出 retracted claims, **研读时可放心引用 §3.1**

**MT-LNN 的局限**:
1. 仓库单人 1★, 代码尚未成熟到可生产
2. O-series 125M PPL 88.93 vs modern Tx 78.86, **绝对质量仍输**
3. 13-protofilament 强耦合隐藏维度, 模型迁移到非 13 倍数 hidden 时需重排
4. retractions 多, 任何 "LNN = new path to AGI" 论调都站不住

---

## 6. 建议本仓库后续动作

### 6.1 短期 (本 cron 周期, 1-2 周内)

1. **不动架构, 只做引用**: 把 MT-LNN 的 §3.1 validated 数据加入 `analysis/sncp_ppo_lite` / `analysis/backbone_matrix` 的横向对比表
2. 写一份 "O(1) 边界 + crossover" 备忘到 `docs/notes/`, 与 PDNA / CfC 一起归档
3. 把 AwareLiquid/M1 标为 **research-only source, not baseline** — 在 `docs/LNN_深度研读报告.md` 引用表加一行警示

### 6.2 中期 (1-3 月)

1. 用 `scripts/experiment_concept_drift.py` 复现 MT-LNN continual learning 的 bounded-replay zero-param 主张, 跑 EMMA-Rover 漂移场景
2. 在 `scripts/bench_irregular_dt.py` 加 MT-LNN-style 13-protofilament 模型 (不依赖 HF Transformers, 仅做序列模块), 与单 LTC baseline 对照
3. 评估 Jetson Orin Nano 上 O-series 0.381 MB state + 4.1 KB decode footprint 是否可跑 LFM2.5-350M 推理

### 6.3 长期 (3 月+)

1. 若 O-series 在 Jetson 上验证可跑, 考虑把 `analysis/sncp_ppo_lite` 的 actor 网络升级到 MT-LNN-style 13-protofilament mixer (优于单 LTC)
2. 把 microtubule 13 路并行的思路写进 [[lnn-paper-survey]] 的横向合成, 作为 "LNN 拓扑结构的演化方向" 一节
3. **不做**: 不要把 MT-LNN 的 Orch-OR / consciousness 主张整合进本仓库任何文档 — 已被 README 自己撤回

---

## 7. 一句话总结

> MT-LNN 是 **架构思想优秀、claims 透明、工程未成熟** 的研究型 LNN. **唯一安全引用** 是 validated 的 125M PPL / fast-weight 0.56 / O(1) 0.381 MB / cross-session bit-exact / continual learning zero-param 五条; **Orch-OR/consciousness/AGI 路径全部撤稿**. 本仓库短期把它当 research-only reference, 中期用其 continual learning + O(1) 边界两条做对照实验, 长期才考虑把 13-protofilament mixer 整合到 SNCP-PPO-Lite 的 actor 升级路径.

---

## 引用

- 仓库: https://github.com/AwareLiquid/M1 (commit `main`, 739 commits, 1★, MIT)
- 镜像 + 论文 PDF: https://huggingface.co/EverestAn/MT-LNN
- 仓内 LaTeX: `mt_lnn_v2_reliable_long_pretraining_arxiv.tex`, `mt_lnn_operator_algebra_whitepaper.tex`
- 本仓库横向定位参考:
  - `docs/reports/Pulse-Driven_Neural_Architecture_PDNA_研读报告.md` (CfC + 振荡)
  - `docs/reports/LFM2.5-1.2B-Instruct-GGUF_Jetson_Orin_Nano_研读报告.md` (production edge LNN-LLM)
  - `docs/reports/Comparative_Analysis_of_LNN_and_LSTM_研读报告.md` (LNN vs RNN 横向)
  - `analysis/sncp_ppo_lite` (下游 SNCP-PPO 应用)
- 数据源: `analysis/repo_watchlist/2026-09-14_lnn_open_source_watchlist.md`
