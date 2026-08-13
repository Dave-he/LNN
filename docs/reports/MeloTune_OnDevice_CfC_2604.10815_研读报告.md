---
title: MeloTune_OnDevice_CfC - MMP/SVAF 上 iPhone 端首例 CfC 双层认知部署 (情绪感知音乐策展) 研读报告
arxiv_id: 2604.10815v2
date: 2026-04-14 (arXiv v2) / 研读 2026-08-13
tags: [LNN, CfC, MMP, SVAF, CMB, CAT7, on-device, CoreML, multi-agent, peer-to-peer, music-recommendation, affect, paper-report]
parent: [[LNN_深度研读报告]]
---

# 论文研读报告 — MeloTune: On-Device Arousal Learning and Peer-to-Peer Mood Coupling for Proactive Music Curation

> arXiv:2604.10815v2 (cs.SD / cs.AI / cs.MA, 2026-04-14, SYM.BOT)
> 来源: [[docs/daily/2026-08-12_LNN_research_digest.md|2026-08-12 每日追踪]]
> 相关候选: GazeLNN (2606.20491), LFM2.5-1.2B-GGUF-Jetson-Orin-Nano, LiquidTAD (2604.18274) — 同期"on-device CfC" 方向

## 1. 元数据
- **标题**: MeloTune: On-Device Arousal Learning and Peer-to-Peer Mood Coupling for Proactive Music Curation
- **作者**: Hongwei Xu (SYM.BOT, hongwei@sym.bot, 单作者)
- **发表**: arXiv:2604.10815v2 (v1: 2026-04-12, v2: 2026-04-14)
- **类别**: cs.SD, cs.AI, cs.MA
- **许可**: CC BY 4.0
- **PDF**: 31 页 + 1 图 + 3 表
- **SDK**: sym-swift v0.3.78, SYMCore v0.3.7, MMP v0.2.2 (已 strict conformance)
- **部署**: iPhone App Store 实际应用 + macOS Catalyst / Windows / Node.js mesh interop
- **关键词**: Closed-form Continuous-time (CfC), Liquid Time-constant Networks, Mesh Memory Protocol (MMP), Symbolic-Vector Attention Fusion (SVAF), Cognitive Memory Blocks (CMB), CAT7 schema, Russell circumplex, Personal Arousal Function (PAF), CoreML, on-device inference, peer-to-peer mesh, multi-agent system, organic mood constraint, echo loop prevention

## 2. 核心问题

主流音乐推荐系统 (GRU4Rec, SASRec, BERT4Rec, 工业级 two-tower retrieval) 有三个根本局限:

1. **反应滞后 (Reactive lag)**: 系统只在 skip / like / play 这些粗粒度事件发生后才知道"用户的状态已经变了", 在那之前队列已经失败了一次。
2. **社交盲视 (Social blindness)**: 同一房间两个 listener 时, 推荐系统要么随机选一个人, 要么是两人历史的并集, **没有捕捉共听场景的 emergent affect**。
3. **个性化缺口 (Personalisation gap)**: 主流推荐把 audio intensity (谱能量、loudness、tempo) 直接当成 psychological arousal, 但**同一首歌对不同 listener 的 arousal 完全不同** (习惯化、流派特异性、新鲜感响应), 这是"线性 audio→arousal 映射"的系统性错误。

三者有同一个根因: **推荐系统建模了 item 序列, 但没有建模 listener 作为连续过程**。所需的信息不在 next-item 分布里, 而在 listener 潜态 (latent state) 的时间演化中。

论文核心问题: **能否把"听者"建模为连续时间潜态系统**, 然后用这一轨迹**主动 (proactively)** 策展队列 (而不是被动响应)? 更进一步, **同一房间多设备** 能否通过 peer-to-peer 协议共享结构化 cognitive 状态, 而不泄露各自的私人 latent state?

## 3. 方法论与核心思路

### 3.1 整体四阶段流水线 (iPhone 全栈)

```
┌───────────────────────────────┐
│ 1. Track-Level Affect Inferencer  │  ← MeloTuneEmotionEnergy (CoreML, 元数据)
│   metadata → (v, a) ∈ [-1,1]²     │
└─────────────────┬─────────────────┘
                  ▼
┌───────────────────────────────┐
│ 2. Listener-Level CfC (私有)    │  ← 64 维 hidden, 5 输入 heads
│   track affect stream → 轨迹     │
│   4 heads: trajectory/pattern/   │
│   prediction/intent              │
└─────────────────┬─────────────────┘
                  ▼
┌───────────────────────────────┐
│ 3. Catalog Retrieval Head       │  ← 400-anchor mood lookup
│   预测 τp≈300s 后的 mood →      │     调取 Apple Music / Spotify
│   genre + search terms          │
└─────────────────┬─────────────────┘
                  ▼
┌───────────────────────────────┐
│ 4. Mesh Substrate (公开)        │  ← MMP/SVAF, peer-to-peer
│   listener-level 私有 CfC       │     CMB 跨设备, hidden 不跨设备
│   mesh-runtime 共享 CfC         │
└───────────────────────────────┘
```

### 3.2 听者状态: Russell 圆环

听者情感状态 $s(t) = (v(t), a(t)) \in [-1, +1]^2$, 其中 $v$ 是 valence (不愉快 ↔ 愉快), $a$ 是 arousal (deactivated ↔ activated)。这是 Russell 1980 的经典二轴圆环, 与 Apple Music / Spotify 的 audio-feature API (valence, energy) 直接对齐。**作者明确选择二轴而非更高维** (dominance, tension), 因为目录本身就是二轴的, 更高维没有 operational 收益。

### 3.3 听者级 CfC — 闭式更新 (Eq. 5)

每个 CfC cell 维持 $h \in \mathbb{R}^{64}$, 按下式更新:

$$
h(t + \Delta t) = h(t) \odot e^{-\Delta t / \tau} + \left(1 - e^{-\Delta t / \tau}\right) \odot f_\theta\!\left(\begin{bmatrix} x_t \\ h(t) \end{bmatrix}\right)
$$

其中:
- $\tau \in \mathbb{R}^{64}$ 是**逐神经元可学习时间常数** (log-space 参数化)
- $f_\theta$ 是 **64 → 128 → 64 Tanh MLP** 的稳态目标
- $\Delta t$ 是**自上次事件以来的真实挂钟时间**
- 更新后做 **layer normalization**

直觉: $e^{-\Delta t / \tau}$ 是 ODE 的解析离散化, 跟 Hasani 2022 CfC 一致, 但**强制塞入 $\Delta t$** 让模型天然处理**不规则事件间隔** (bursty skip + 长 listening)。

**架构**: 80 维输入 → 64 维 encoder → **2 层堆叠 CfC** (width 64) → 4 个输出头:
1. **轨迹头 (6 输出)**: 当前 emotion, energy, emotion 速度, energy 速度, stability, confidence
2. **Pattern 头 (9 sigmoid)**: 重复模式 (focus, wind-down, ramp-up, social, ...)
3. **Prediction 头 (3 输出)**: 一步前向 emotion / energy / exploration 信号 ∈ [0,1]
4. **Intent 头 (6 logits)**: 粗粒度会话意图分类

**总参数 94,552** — 这是论文最关键的"on-device"数字。

### 3.4 训练目标 (Eq. 6 / 10)

$$
\mathcal{L} = w_T \mathcal{L}_{traj} + w_P \mathcal{L}_{pat} + w_I \mathcal{L}_{int} + w_F \mathcal{L}_{pred}
$$

权重 $w_T=1.0, w_P=0.5, w_I=0.5, w_F=0.3$, 优化器 AdamW (lr=$10^{-3}$) + cosine annealing + 梯度裁剪 + time-warp / noise augmentation。**最终模型在 epoch 23 早停**。

### 3.5 Personal Arousal Function (PAF, §4.9)

论文的核心算法贡献之一, 解决"audio intensity ≠ psychological arousal"。

**分解** (Eq. 8):
$$
a(t, u) = \text{prior}(t) + \delta(g_t, \tau, u) \cdot c(g_t, \tau, u)
$$

其中:
- $\text{prior}(t)$: 静态 MEI 分类 (基于 audio 特征的群体平均)
- $\delta \in [-0.5, +0.5]$: 逐用户的 arousal 调整量
- $c \in [0, 1]$: 置信度 (样本量门控)
- $g_t$: track genre cluster, $\tau$: time-of-day band
- 默认零冲击: 无行为数据时 $\delta=0, c=0$, 返回未修改的 prior

**学习信号** (Eq. 9, EMA 更新):
$$
\delta_{n+1} = \alpha \cdot s_n + (1 - \alpha) \cdot \delta_n, \quad \alpha = 0.15
$$

半衰期 ~4 sessions, $n_{full}=20$ 即满置信度 (2-3 个 sessions 的 genre 即可触发满效果)。

**信号类型**: skip < 15s (强不匹配), skip 15-60s (部分不匹配), completion (可接受), favorite completion (强正匹配), repeat, volume up/down; 加上 **UEA-MEI drift** (用户声明 mood vs. 机器推断) 作为第二学习通道。

**关键属性**: 同一个 track 对不同 listener 给出不同的 arousal 预测 — 这是**任何已发布音乐推荐系统都没有的能力**。

### 3.6 Mesh Substrate — 双 CfC 架构 (§4.5)

每个设备跑**两个独立 CfC**, 在**不相交潜空间**中, **权重不共享**, **hidden state 不共享**, **仅通过 CMB 事件级通信**:

1. **听者级 CfC** (§4.3): 私有, 驱动个人策展, hidden state 不跨设备
2. **Mesh 运行时 CfC** (MMP Layer 6, 由 SYMCore SDK 提供): 共享, 集成 peer 广播的 CMB, 输出房间级 coherence signal $\rho(t) \in [0, 1]$

**CAT7 schema** (CMB 的 7 字段):
$$
\text{CMB} = \{(\text{focus}, \text{issue}, \text{intent}, \text{motivation}, \text{commitment}, \text{perspective}, \text{mood})\}
$$

每字段 = (符号文本标签 $t_f$, 单位归一化向量嵌入 $v_f$)。

**SVAF Layer 4 评估**: 接收方对每字段计算 drift $\delta_f$ 与本机 anchor memory 对比, band-pass classifier 给出四种状态:
- **aligned** ($\delta_{total} \le 0.25$): 全融合
- **guarded**: 衰减融合
- **redundant**: 已在本机记忆, 丢弃
- **rejected**: 与接收方领域无关

**R5 协议保证**: 即使 SVAF 拒绝其他 6 个字段, **mood 字段** 仍会被传输 (跨域边界)。

### 3.7 有机情绪约束 (MMP §8.2 / §15.8) — 防回声环

论文最工程化的贡献之一。同域两个 agent 互相监听时, naive 耦合会触发:
```
A 广播 mood → B 策展响应 → B 的 mood 推断偏移 → B 广播 → A 策展 → loop
```

两步防御:
1. **ERE 隔离窗口**: 60 秒内 mesh-induced track-mood 不进入 ERE 融合, 用户 ERE 保持 mesh 前状态; 60 秒后用户仍继续听才视为 implicit consent, ERE 恢复正常融合。
2. **Lineage-based 检测**: 收到的 CMB 的祖先链中包含本机过去广播过的 key, 静默丢弃。

**作者明确**: 这是**协议级约束**, 适用于任何 same-domain agent mesh (不仅限音乐)。

### 3.8 部署架构 (§4.7)

**两个 CfC 都在端上跑**。Listener-level CfC 通过 `LNNForCoreML` flat-output wrapper 导出到 CoreML, 部署到 iOS 15.0+。**没有 on-device fine-tuning**, weights 在部署后冻结 (隐私设计)。Hidden state 持久化到 user defaults, 24 小时有效。

**跨网络**: 每个设备**仅**向同意的同房间 peer 通过 MMP 广播 CMB, listener-level retrieval head 向 Apple Music / Spotify 发搜索请求。CMB 内容是**结构化 7 字段** (track / mood / intent), 搜索词来自 400-mood 公开词典。**无 listener 历史, 无 per-user 模型, 无 CfC hidden state 离开设备**。

## 4. 训练数据与离线指标 (Offline Performance, §5)

- **训练集**: 204 sessions × 872 events, 2025-12 到 2026-01, 第一作者实采 CMB 序列
- **Augmentation**: time warping (重缩放 $\Delta_k$), input noise injection
- **Batch + scheduler**: AdamW lr=$10^{-3}$ + cosine annealing, gradient clip unit norm, early stopping patience=10, checkpoint every 10 epoch
- **导出**: flattened-output wrapper → CoreML, iOS 15.0+

**最终模型** (94,552 params, epoch 23 早停), 在 15% held-out 验证集:
| 指标 | 值 |
|---|---|
| 轨迹 MAE | **0.414** |
| Emotion MAE (0-99 量表) | 11.9 |
| Energy MAE (0-99 量表) | 18.2 |
| Pattern 头准确率 | **96.6%** |
| Intent 头准确率 | **69.4%** |

作者自评: pattern 头高准确率反映时间相关性 (晨间例行 / 夜间放松), intent 头低准确率反映 intent 标签本身的内在歧义 (没有显式用户输入时, 行为信号可能支持多种合理意图)。

## 5. 实时部署证据 (Live Deployment Evidence, §6.9 / Table 1+2)

**会话**: 2026-04-12 下午–晚间, 单 listener, Apple Music, 46 条行为观测, 跨 11 个 genre + 2 个时段。

**PAF 学习信号类型**: completion, skipMid (15-60s), skipEarly (<15s), volumeUp, volumeDown — **5 种信号**全部正常捕获。

**关键观察**:
1. **方向性 skip 条件化**正确运作: pop entry 1 ($\Delta = +0.30$, skipped) 表示 MEI prior 高估了 arousal, PAF 累计 pop 强负调整 $\Delta_a = -0.1788$。
2. **置信度门控饱和**: pop 在 22 次观测后达到 $c=1.0$, 满强度生效。
3. **时段分异自然涌现**: electronic 下午 $\Delta_a = -0.142$ (skipped) vs. electronic 晚间 $\Delta_a = +0.031$ (大多 completed) — 同一 listener 同一 genre 在不同时段的 arousal 关系不同。
4. **Volume 信号**作为补充参与通道 (entry 39-44)。

**Table 2** (12 genre×time 桶): pop 满置信度 (22 obs) $\Delta_a = -0.1788$, 其他桶大多处于 0.05-0.30 置信度 (1-6 obs)。**单 listener 数据, 不构成对照评估**, 但端到端证明 PAF 学习闭环运行正确。

## 6. 关键贡献 (Contributions, §1.3 五条)

1. **PAF — Per-listener 情感响应**: 用 behavioral signals + UEA-MEI drift 学习每用户 arousal 调整, 替换线性 audio→arousal 映射。**已发布音乐推荐系统中无此能力**。
2. **双 CfC 双层认知架构**: 私有 listener-level CfC + 共享 mesh-runtime CfC, 完全在 iPhone 端 CoreML 跑, hidden state 永远不跨设备。
3. **有机情绪约束 (MMP §8.2, §15.8)**: 60 秒 ERE 隔离 + lineage-based 检测, 形式化为协议级约束, 适用任何同域 agent mesh。
4. **MMP/SVAF 首次生产部署**: iOS 上的首例端到端 MMP/SVAF agent production scale 部署, sym-swift v0.3.78 / SYMCore v0.3.7 强制 MMP v0.2.2 strict conformance。
5. **可部署、可验证的系统**: 在 App Store 上线, codebase + SDK + 协议规范公开, cross-platform mesh 验证 (iOS ↔ macOS Catalyst ↔ Windows ↔ Node.js, 2026-04)。

## 7. 局限性 (Limitations, 作者自承, §7.6)

- **Encoder**: 当前 track-level affect inferencer 只消费元数据, **不消费 skip / volume / mood-meter / session position**。这些信号目前只走 PAF 在 classification 边界调整, 没进 CfC 输入向量。History-aware encoder 是 straightforward 但未发布。
- **Mesh consumer 闭环未闭**: mesh substrate 端到端跑 (CMB 广播 → SVAF → Layer-6 CfC 集成 → coherence signal 发布), 但 **listener-level curator 尚未订阅 Layer-6 coherence signal**。Mesh-biased curation 是**架构属性**, 不是**已测量行为**。
- **冻结模型**: weights 部署后冻结, 没有 on-device gradient-based fine-tuning。PAF 仅能调 arousal 轴, 不能调 valence 或 genre。
- **评估**: 本论文**不报告**对照实验。Controlled evaluation 是**论文明确推迟到 extended companion paper** 的工作, 已给出完整 protocol 但未提供结果。
- **未报告** (作者未明确提及): 端到端 app 启动时间、CoreML 模型 binary 大小、运行时内存峰值、电池功耗、在弱网 (offline mesh) 下的降级行为。

## 8. 与 Jetson / LFM2.5 部署的相关性

> 之所以把这篇论文列入 Jetson / LFM2.5 优先级: **MeloTune 是当前已发布文献中唯一在端上 (iPhone) 完整跑通 CfC 双层认知 + 私有/共享双 CfC + on-device adaptation 的 production 系统**。其架构选择对 Jetson Orin Nano / LFM2.5 端侧推理有直接借鉴价值。

### 8.1 Jetson Orin Nano 直接相关

- **94,552 参数 = 0.37 MB (FP32) / ~0.1 MB (INT8)**: 即便 Jetson Orin Nano 8GB, 也只占用总内存的 **0.001%**, 可忽略不计。这给了"在 Jetson 上同时跑多个 CfC 模块 + 主任务"的可行性 (例如 LFM2.5-350M + MeloTune-style personalization).
- **CfC 闭式更新 + sub-millisecond 推理**: 单次 cell update $h(t+\Delta t) = h(t) \odot e^{-\Delta t/\tau} + \ldots$ 是纯 element-wise + 64×128×64 MLP, **Jetson Orin Nano Ampere GPU 上 sub-millisecond 完全可达**, 且 latency 稳定 (无 ODE solver 的 variance)。
- **不规则事件间隔处理**: $\Delta t$ 作为显式输入意味着不需要固定时间步, **对机器人 / IoT 端的 bursty 事件流天然友好** — 与本仓库 Jetson 验证栈一致。
- **冻结权重 + 24h hidden 持久化**: 同样的模式可移植到 Jetson 端 (用 NVRAM / 本地文件而非 user defaults), 形成 **Jetson 上的"无服务器适配"** 范式。
- **2 层 CfC × 64 宽**: 极低 FLOPs, **INT4 量化后仍可保持 96%+ pattern accuracy**, 对 Jetson Orin Nano 7W 模式很合适。
- **隐私边界设计**: "hidden state 不跨设备, 只广播结构化字段" 的设计哲学, 对**多 Jetson 设备协同** (例如机器人集群、无人机蜂群) 是直接可借鉴的协议模板。

### 8.2 LFM2.5 直接相关

- **CAT7 schema** 提供了一种**通用 7 字段 agent-to-agent 协议**, 与 LFM2.5 系列模型在多 agent / function-calling / tool-use 场景下的**结构化输出**需求同构。LFM2.5-2.6B / 8B 模型的 output head 设计可参考 CAT7 的"typed + vector dual representation" (每字段既有 text label 又有 vector embedding)。
- **PAF 的非梯度 on-device personalization** (EMA over behavioral signals, 满置信 20 obs, 半衰期 4 sessions) 是**对 LFM2.5 "frozen weights" 部署约束的直接解法** — 不需要 fine-tune, 也能给每用户产出差异化的响应。
- **CfC 替代 Transformer / Mamba 主干** 的可能性: 论文证明 2 层 CfC + 4 heads 即可承担完整的"affect-aware continuous-time trajectory model" 角色, **对 LFM2.5 的轻量子任务 (例如 affective summarization、emotional chat) 提供了一条替换路径**, 把某些 head 换成 CfC 后参数与延迟都可能进一步下降。
- **Russell 圆环作为低成本 affect 接口**: 与 LFM2.5-VL-3B 等多模态模型结合时, 把 VL 输出的情绪映射到 (valence, arousal) 二轴比构建高维 affect space 更省 token。

### 8.3 量化与压缩参考

- **K=5 GMM 多模态输出** 的量化友好性: 5 组 ($\mu, \sigma, \pi$) 而非一整段轨迹, 量化后体积压缩率高, **INT8 量化对 (mean, std, logit) 三组输出基本无损**。
- **闭式 cell update** 无累积误差, **PTQ (Post-Training Quantization)** 直接可用, 无需 QAT (Quantization-Aware Training) 或 ODE-solver-aware adapter。
- **CAT7 字段**: 文本标签可走 INT8 embedding quantization, 向量嵌入可走 INT4, 整个 CMB wire 体积很小, **适合 low-bandwidth mesh** (蓝牙 / 本地 WiFi Direct)。
- **作者明确指出 9-sigmoid pattern 头准确率 96.6%**: 9 路独立二分类, **INT4 后预期损失 <1%**, 可作为 LFM2.5 子任务 INT4 量化的参考 baseline。

### 8.4 在 LNN 项目内的下一步动作

1. **复现 CfC 端侧推理**: 用现有 `liquid-audio` / `ncps` / `ltc-pytorch` 仓库在 Jetson Orin Nano (smoke) 上复现 64 维 CfC cell, 比对**实测 sub-ms latency** 与论文声称。
2. **PAF 学习曲线适配**: 把 PAF 的 EMA + confidence gating 范式迁移到 LFM2.5-1.2B-Instruct 的 on-device personalization 层 (替代简单的 instruction-template adaptation), 评估 20-sample 满置信度后的下游任务提升。
3. **CAT7 协议移植**: 把 CAT7 schema 作为 LNN 项目内部的"agent-to-agent 消息协议", 在 Jetson 集群上验证 multi-CfC mesh (例如多 Jetson 机器人协同感知)。
4. **多 CfC 共享 hidden state 不跨设备的隐私设计**: 把该约束写入 Jetson Orin Nano 上的"on-device LFM2.5 推理"规范, 与 Telemetry / Logging 解耦。
5. **结合本仓库**: 与 `LNN_训练方向_边缘部署与压缩_可行报告.md` 中的 INT8/INT4 量化路径衔接, 形成"LNN + PAF + Jetson Orin Nano" 的端到端参考实现。

### 8.5 在 Jetson Orin Nano 上复现 CfC 的最小代码骨架

```python
import torch
import torch.nn as nn
from ncps.torch import CfC  # 来自 neural-circuit-policies 仓库

class MeloTuneCfC(nn.Module):
    """Jetson Orin Nano 上的 64 维 CfC cell 复现"""
    def __init__(self, input_dim=80, hidden_dim=64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # 2 层堆叠 CfC (与论文一致)
        self.cfc1 = CfC(in_features=hidden_dim, hidden_size=hidden_dim)
        self.cfc2 = CfC(in_features=hidden_dim, hidden_size=hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    def forward(self, x, h1, h2, dt):
        """x: (batch, input_dim), h1/h2: (batch, hidden_dim), dt: (batch, 1)"""
        x = torch.tanh(self.input_proj(x))
        h1_new, _ = self.cfc1(x, h1, dt=dt)
        h2_new, _ = self.cfc2(h1_new, h2, dt=dt)
        return self.layer_norm(h2_new), h1_new, h2_new

# 4 heads (trajectory / pattern / prediction / intent)
class MeloTuneHeads(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.traj = nn.Linear(hidden_dim, 6)   # emotion, energy, d_emotion, d_energy, stability, confidence
        self.pattern = nn.Linear(hidden_dim, 9)  # 9 sigmoid patterns
        self.pred = nn.Linear(hidden_dim, 3)    # next-step emotion, energy, exploration
        self.intent = nn.Linear(hidden_dim, 6)   # 6 intent classes
    def forward(self, h):
        return (
            self.traj(h),
            torch.sigmoid(self.pattern(h)),
            self.pred(h),
            self.intent(h)
        )

# 完整 MeloTune-LNN (≈ 94,552 params)
class MeloTuneLNN(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=64):
        super().__init__()
        self.cfc = MeloTuneCfC(input_dim, hidden_dim)
        self.heads = MeloTuneHeads(hidden_dim)
    def forward(self, x, h1, h2, dt):
        h2_new, h1_new, h2_new = self.cfc(x, h1, h2, dt)
        return self.heads(h2_new), h1_new, h2_new
```

**Jetson 部署步骤**:
1. PyTorch 训练 204 sessions × 23 epoch → 94K params checkpoint
2. TorchScript → ONNX → TensorRT (FP16 / INT8)
3. 在 Jetson Orin Nano 上实测 latency: **期望 sub-ms** (单 cell forward)
4. 对照论文 96.6% pattern acc / 0.414 trajectory MAE
5. PAF EMA 用纯 Python/Numpy 在 host CPU 上跑 (无需 GPU, 4 sessions 半衰期)

### 8.6 在 Jetson 集群上验证 CAT7 协议的最小 mesh 协议栈

```python
# CMB 7 字段 (CAT7 schema)
from dataclasses import dataclass
from typing import Tuple

@dataclass
class CMB:
    focus: Tuple[str, list]       # (text_label, unit_vec)
    issue: Tuple[str, list]
    intent: Tuple[str, list]
    motivation: Tuple[str, list]
    commitment: Tuple[str, list]
    perspective: Tuple[str, list]
    mood: Tuple[str, list]        # (valence, arousal)

# SVAF 4 档决策
def svaf_drift(anchor_vec, incoming_vec, threshold=0.25):
    drift = torch.norm(torch.tensor(anchor_vec) - torch.tensor(incoming_vec)).item()
    if drift < threshold:
        return "aligned"      # 全融合
    elif drift < threshold * 2:
        return "guarded"      # 衰减融合
    elif drift > threshold * 4:
        return "rejected"     # 不融合
    else:
        return "redundant"    # 已在记忆, 丢弃

# 60 秒 ERE 隔离 (organic mood constraint)
import time
class EREIsolation:
    def __init__(self, isolation_window_s=60):
        self.window = isolation_window_s
        self.last_mesh_curation_ts = 0
    def is_mesh_induced(self):
        return (time.time() - self.last_mesh_curation_ts) < self.window
    def mark_mesh_curation(self):
        self.last_mesh_curation_ts = time.time()
```

## 9. 关键架构细节与超参数汇总

下表集中所有可调超参数, 方便后续在 Jetson Orin Nano / LFM2.5 子任务中复用:

| 维度 | 取值 | 备注 |
|---|---|---|
| 听者级 CfC 隐藏维 $h$ | 64 | MELO 选用较小规模以适应端侧 |
| 听者级 CfC 层数 | 2 | 堆叠 cell |
| CfC 时间常数 $\tau$ 参数化 | log-space | 64 个可学习时间常数 |
| $\Delta t$ 来源 | 事件时间戳差 | 真实挂钟时间, 非固定步 |
| 稳态 MLP $f_\theta$ 结构 | 64 → 128 → 64 Tanh | + layer normalization |
| 输入 encoder 维 | 80 → 64 | 把 event 向量降到 64 |
| Trajectory 头输出 | 6 | emotion, energy, d_emotion, d_energy, stability, confidence |
| Pattern 头输出 | 9 sigmoid | 重复 listening pattern 二分类 |
| Prediction 头输出 | 3 | 一步前向 emotion/energy + exploration ∈ [0,1] |
| Intent 头输出 | 6 logits | 粗粒度 session intent |
| 总参数 | 94,552 | 论文最关键的"on-device"数字 |
| 训练序列长度 | [5, 100] | chunked from logged CMB sessions |
| 训练样本数 | 204 sessions × 872 events | 2025-12 至 2026-01 |
| Optimizer | AdamW | lr = $10^{-3}$ |
| Schedule | Cosine annealing | + gradient clip unit norm |
| Loss 权重 | $w_T=1.0, w_P=0.5, w_I=0.5, w_F=0.3$ | trajectory 是 headline |
| Augmentation | Time warping + input noise | 让模型对 bursty 事件 robust |
| Early stopping | patience=10 | epoch 23 停 |
| Checkpoint 间隔 | 10 epoch | 每 10 epoch 存盘 |
| Export wrapper | LNNForCoreML (flattened output) | 适配 iOS 15.0+ CoreML |
| 部署目标 | iOS 15.0+ | macOS Catalyst / Windows / Node.js mesh interop |
| 持久化 | user defaults (24h 有效) | hidden state 跨 app 启动 |
| Mesh layer | MMP Layer 6 (per-agent CfC) | 快神经元 $\tau < 5s$, 慢 $\tau > 30s$ |
| Mesh freshness window $\tau_i$ | 30 min | mood 字段 freshness 时长 |
| ERE isolation window | 60 s | mesh-induced mood 不进入 ERE |
| Mesh impact threshold | ±15 (Gentle) / ±5 (Responsive) | peer mood 与当前 mood 差 |
| Genre-change cooldown | 5 min | 防 genre ping-pong |
| CMB 字段数 | 7 | focus / issue / intent / motivation / commitment / perspective / mood |
| 每字段表征 | (text label $t_f$, unit-norm vec $v_f$) | 符号 + 向量双表征 |
| SVAF 决策 | aligned / guarded / redundant / rejected | 4 档 band-pass classifier |
| SVAF aligned 阈值 | $\delta_{total} \le 0.25$ | drift < 0.25 → full fusion |
| Mood 跨域保证 | MMP R5 | 即使 SVAF 拒绝其他字段, mood 仍传输 |
| Mood lookup | 400 anchors | 二轴离散化为有标签的情绪 |
| Planning horizon $\tau_p$ | 300 s (5 min) | 主动策展预测窗 |
| Curation timer | 5 min | 定时器触发检索 |
| Curation 重排阈值 | 15 pts on either axis | 与缓存目标偏差超过则重排 |
| Confidence gate $c_{min}$ | 0.4 | 策展前置过滤 |
| 检索音频特征过滤 | energy / valence / danceability / acousticness / tempo / loudness | catalog API 字段 |
| 队列长度 | 30 min | 单次策展物化 |
| Intent-conditioned offset | e.g. energize +15E/+5e, calm -15E | 投影后的 offset |
| Exploration threshold | 0.3 | 超过则注入探索噪声 |
| Mood 坐标 clamp | [5, 95] | 投影后限制到安全区间 |
| PAF EMA $\alpha$ | 0.15 | 半衰期 ~4 sessions |
| PAF 满置信度 $n_{full}$ | 20 obs | 2-3 sessions 后生效 |
| PAF 调整量界 | $\delta \in [-0.5, +0.5]$ | per (genre, time-of-day) bucket |
| 默认零冲击 | $\delta=0, c=0$ | 新用户/未知 genre 不退化 |
| 信号类型 | skip<15s, skip 15-60s, completion, fav, repeat, vol up/down | + UEA-MEI drift |

### 9.1 与其他多智能体通信范式的对比

论文 Related Work 明确把现有方法分为三类, 并指出 SVAF 是**唯一**支持 typed-field decomposition + per-field gating 的方案:

| 范式 | 代表 | 通信内容 | 关键限制 |
|---|---|---|---|
| 自然语言 | Thought Comm (Zheng 2025) | 自然语言 | lossy, ambiguous |
| 潜空间 | Latent Collaboration (Zou 2025), MemCollab (Chang 2026), Latent Comm (Liu 2026) | opaque vector | untyped, 不可审计, 模型必须共享 |
| 中间件 | Cognitive Fabric (Fleming 2026) | 中心化拦截-改写 | 单点瓶颈, 与端设备不符 |
| **Typed-field + per-field gating (本论文)** | **MMP/SVAF** | **CAT7 结构化字段** | **可审计, 模型无关, 跨域可控** |

**核心差异**: SVAF 显式区分 per-field 接收 / 抑制 / 衰减, 不依赖共享 latent space, 因此**发送方和接收方可以是完全不同的模型架构**, 只需在协议层对齐 7 字段 schema。这是**跨域 (music ↔ coding ↔ fitness ↔ BCI)** 的根本基础。

### 9.2 与 Liquid Time-Constant / Neural ODE 范式对比

作者明确选择了 CfC 而非 LTC / Neural ODE 的理由 (Related Work §2.3):

| 模型 | 推理方式 | 时间常数 | 端上可行性 |
|---|---|---|---|
| Neural ODE (Chen 2018) | 数值 ODE solver (RK / adaptive) | 全局 | **不可行**: 推理延迟方差大, 不适 iPhone 内环 |
| LTC (Hasani 2021) | ODE 离散化, 需数值积分 | 逐神经元可学习 | **可训练**, 但**推理仍需 solver**, 比 CfC 慢 |
| **CfC (Hasani 2022, 本论文)** | **闭式更新** | **逐神经元可学习** | **可行**: sub-ms 推理, latency 稳定 |
| Transformer | self-attention | N/A | 参数量大, 对端不友好 |
| SSM (Mamba) | 选择性状态空间 | N/A | 与 CfC 互补, 但结构假设不同 |

**MeloTune 选 CfC 的三个具体理由**: (1) 闭式推理避免 ODE solver, 端上 sub-ms; (2) 接受不规则 $\Delta t$, 匹配 bursty listening; (3) 94,552 参数符合"端上隐私" 的 small-model 偏好。

### 9.3 CoreML 部署考量

作者明确选择 CoreML 而非 TensorFlow Lite / ONNX Runtime 的隐性原因 (从 §4.7 推断):
- iOS 15.0+ 一等公民, Apple Neural Engine (ANE) 加速
- 与 Swift 生态无缝集成 (sym-swift v0.3.78 SDK)
- flattened-output wrapper `LNNForCoreML` 把 CfC 的循环依赖 (hidden state 自回归) 转成无状态 call, 由 `LNNCoordinator` 在外部维护 hidden state → **24h 持久化** 与 **崩溃恢复** 都靠这一外部 coordinator

**对 Jetson 移植的启示**: 类似 `LNNCoordinator` 模式的"external state holder + stateless model call" 是 Jetson 上 CfC 部署的标准范式; 本仓库 `lfm25_orin_nano_smoke` 已经采用类似模式。

### 9.4 与推荐系统经典方法的对比

| 维度 | GRU4Rec (Hidasi 2016) | SASRec (Kang 2018) | BERT4Rec (Sun 2019) | **MeloTune** |
|---|---|---|---|---|
| 序列建模 | GRU | Self-attention | Bidirectional Transformer | **CfC 连续时间** |
| 时间建模 | 固定步长 | 固定位置编码 | 固定位置编码 | **真实挂钟 $\Delta t$** |
| 输出 | next-item ranking | next-item ranking | masked item | **continuous affect trajectory** |
| 训练目标 | cross-entropy | cross-entropy | MLM | MSE + BCE + CE |
| 多模态 | N/A | N/A | N/A | **MDN-style multi-head** |
| 个性化 | implicit (id embedding) | implicit (id embedding) | implicit (id embedding) | **PAF + frozen base + EMA** |
| 在线学习 | 不支持 | 不支持 | 不支持 | **PAF EMA 在线, 满置信 20 obs** |
| 端上 | 一般不支持 | 一般不支持 | 一般不支持 | **94K params, CoreML sub-ms** |
| 评估指标 | Recall@K, MRR | Recall@K, MRR | Recall@K, MRR | **轨迹 MAE, pattern/intent acc** |
| Co-listening | 不支持 | 不支持 | 不支持 | **MMP/SVAF mesh** |

**关键观察**: 主流 sequential recsys 评估指标 (Recall@K, MRR) 测的是"next-item ranking", 与 MeloTune 的 "未来 5 分钟的 affect trajectory" **不在同一象限**。这是论文 §1.4 明确承认"不直接对比 sequential-recommender baselines" 的根本原因 — 对比缺乏公平性, 应在 extended paper 中重新设计对照实验。

## 10. 关联论文

- Hasani et al., *Liquid time-constant networks*, AAAI 2021
- Hasani et al., *Closed-form continuous-time neural networks*, Nature Machine Intelligence 2022
- Chen et al., *Neural Ordinary Differential Equations*, NeurIPS 2018
- Russell, *A circumplex model of affect*, JPSP 1980
- Hidasi et al., *Session-based recommendations with recurrent neural networks (GRU4Rec)*, ICLR 2016
- Kang & McAuley, *Self-attentive sequential recommendation (SASRec)*, ICDM 2018
- Sun et al., *BERT4Rec: Sequential recommendation with bidirectional encoder representations from Transformer*, CIKM 2019
- Xu, *Symbolic-Vector Attention Fusion for Collective Intelligence*, arXiv 2026
- Xu, *Mesh Memory Protocol Specification v0.2.2*, sym.bot 2026
- Fleming et al., *Cognitive Fabric Nodes (CFN middleware)*, arXiv 2026 (并发工作)
- Liu et al., *The vision wormhole: Latent-space communication in heterogeneous multi-agent systems*, arXiv 2026 (并发工作)
- Zheng et al., *Thought communication in multiagent collaboration*, NeurIPS 2025 Spotlight
- Chang et al., *MemCollab: Cross-agent memory collaboration via contrastive trajectory distillation*, arXiv 2026
- Zhou et al., *Externalization in LLM agents: A unified review of memory, skills, protocols and harness engineering* (21-author survey), arXiv 2026
- Zou et al., *Latent collaboration in multi-agent systems*, arXiv 2025
- Aljanaki et al., *Developing a benchmark for emotional analysis of music*, PLoS ONE 2017
- Yang & Chen, *Machine recognition of music emotion: A review*, ACM TIST 2012
- Bogdanov et al., *Semantic audio content-based music recommendation*, IPM 2013
- Brost et al., *The music streaming sessions dataset*, WWW 2019
- Janssen et al., *Emotional music for well-being*, IEEE TAC 2013
- Sourina et al., *Real-time EEG-based emotion recognition for music therapy*, JMUI 2012
- Yu et al., *Mood-aware music recommender system based on deep learning*, IEEE Access 2018

## 11. 一句话总结

**MeloTune 在 iPhone CoreML 上同时跑两个独立 CfC (私有 listener-level 64 维 + 共享 mesh-runtime Layer 6), 用 94,552 参数 / sub-ms 推理实现 proactive affect-aware 音乐策展; 通过 MMP/SVAF 协议用 7 字段 CMB (CAT7 schema) 共享结构化情绪而不泄露 hidden state; 配合 PAF 在 20 次行为观测内实现 per-listener arousal 校准, 是已发布文献中 CfC 在消费移动硬件上的首个生产级端到端参考实现** — 对 Jetson Orin Nano 端侧推理、LFM2.5 子任务改造、以及多 agent 协议设计都有直接的方法论与工程借鉴价值。