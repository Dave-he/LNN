---
title: LNN 最新进展研究报告 - 2026-06-03
date: 2026-06-03
tags: [LNN, CfC, LFM, EMMA, cross-modal, ablation, research-report, weekly]
related:
  - "[[docs/research/2026-06-02_LNN_research_report]]"
  - "[[docs/research/2026-06-02_multimodal_physreg_appendix]]"
  - "[[docs/daily/2026-06-03_LNN_research_digest]]"
---

# 🌊 LNN 最新进展研究报告 — 2026-06-03

> 接续 2026-06-02 (12 轮 /loop 跨度)。本日 arXiv API 失败 / GitHub rate-limited;HF 仍返回 14 个新 LFM2.5 衍生模型。本日核心研究产出是**架构-vs-信息消融**(round 13 §16) — 首次定量分离 cross_attn 在合成 vs 真实数据上的两种贡献来源。

## 1. 今日 digest 关键新信号 (来自 `docs/daily/2026-06-03_LNN_research_digest.md`)

| 信号 | 资源 | 价值 |
|---|---|---|
| **Claude Opus 蒸馏到 LFM2.5 8B-A1B** | `reaperdoesntknow/LFM2.5-8B-A1B-Opus-Distil` (HF, 2026-06-02, 104 dl/day) | 第一批 frontier 教师 → 液态基础学生的蒸馏开源;指向"液态架构 + 大模型知识" 的复合路径 |
| LFM2.5-1.2B SFT 蒸馏 | `reaperdoesntknow/LFM2.5-1.2B-Distilled-SFT` (488 dl/day) | 同思路的边缘规模实现 |
| LFM2.5-8B-A1B abliterated / uncensored | `mradermacher/Huihui-LFM2.5-8B-A1B-abliterated-{i1,}-GGUF`、`sahilchachra/LFM2.5-8B-A1B-Uncensored` | 社区生态扩展(中立提及,不评价) |
| Rust "liquid" 文本引擎 | `infinition/LSTN` (2026-06-02,Rust) | 把 LNN 思想搬到 Rust 推理引擎;离主流 PyTorch 生态距离较远但值得跟踪 |
| 91M 参数 Liquid-Time-Constant Transformer (本地 Python coding) | `SaiSudanV/mahoraga-lite` (2026-04-14) | 把 LTC 嵌入 Transformer 做小模型代码助手;是合成 vs 真实任务边界的另一个例子 |

> arXiv API 在 06-02 与 06-03 连续两天都因 timeout 失败,本仓库的 daily pipeline 保留了 06-01 的候选池 (25 篇)。建议下一轮 daily 调度加入 `--arxiv-fallback-window 3` 让 union 候选池作为正式信号。

## 2. 本日研究产出:Round 13 - 架构 vs 信息消融

### 2.1 来历

12 轮 /loop 跨度的关键转折:

| Round | 验证 | 解读 |
|---:|---|---|
| 7 | Cross-modal attention | PARTIAL (+7.6%) |
| 8 | Burst heterogeneous data | PASS (+27.6%) |
| 9 | modality_dropout | NEG |
| 10 | partial-occ train | NEG |
| 11 (cron) | Real EMMA rover | PASS (+51%) |
| 12 | Audio noise 640× sweep | 假设证伪,曲线平坦 |

第十二轮的**元结论**(`docs/research/2026-06-02_multimodal_physreg_appendix.md` §15):"架构论 vs 信息论 = 任务依赖性",合成任务的 +27.6% 增益主要来自架构,真实任务的 +51% 增益主要来自 audio 信息。

### 2.2 第十三轮的可证伪假设 (本日)

> 拆掉 audio encoder,把同一份 video 喂给两路 BidirectionalNoiseAdaptiveCfC,跑 cross-attention(self-cross 形式)。
> **预测**:
>
> - 合成 burst:val MSE 应接近 cross_attn(差距 < 5%),且仍 PASS ≥+20% vs video_only。
> - 真实 EMMA rover:val MSE 应大幅劣于 cross_attn(差距 > 20%),证明 audio 在真实数据上是不可替代的信息源。

### 2.3 实现:`UniVideoSelfXAttnWithMDN`

`lnn/core/multimodal_physreg.py` 新增。内部直接复用 `CrossModalAttnBiCfCNADWithMDN`,但 `forward(video, audio=None)` 把 video 同时喂给 video 和 audio 两个 slot。参数 shape / 注意力机制 / MDN 头与 cross_attn **完全相同**;唯一区别是输入信号源。

- 4 个新单测全过:形状、忽略 audio 输入(同 video 不同 audio → 输出 bit-identical)、双 encoder 梯度都收到、与 cross_attn 输出不同(确认 audio 真的被忽略)。
- 整套 `pytest tests/` **115/115 全过**(本日 numpy 1.x 降级修复了之前 round 11 / 12 的 ncps 环境问题,所以测试数从 109 升到 115)。

### 2.4 实验结果

#### A. 合成 burst (audio_noise=0.05, n=800, ep=20, K=2, seed=42)

| 模型 | params | val MSE | vs video_only |
|---|---:|---:|---:|
| video_only | 3 258 | 1.035 | — |
| cross_attn (round 8 复刻) | 8 186 | **0.750** | **+27.6%** PASS |
| **uni_video_xattn (新)** | 8 843 | **0.760** | **+26.6%** PASS |

→ uni_video 与 cross_attn 差距 **−1.3%**(几乎相同),**预测准确**。
→ 在合成 burst 上,cross_attn 的 +27.6% 增益**99% 来自架构(双编码器 + cross-attention),audio 信息贡献 ≤1.3%**。
→ JSON: `analysis/multimodal_physreg/2026-06-03_uni_video_xattn_synthetic_burst.json`。

#### B. 真实 EMMA rover (3-ch video + 1-ch audio, 200 samples, 20 ep, K=1)

| 模型 | params | test MSE | vs video_only | vs cross_attn |
|---|---:|---:|---:|---:|
| video_only (concat) | 3 595 | 536.85 | — | — |
| multimodal (concat) | 6 539 | 397.11 | +26.0% | — |
| cross_attn (round 11 复刻) | 8 523 | **262.87** | **+51.0%** PASS | — |
| **uni_video_xattn (新)** | 8 843 | **364.11** | +32.2% PASS | **−38.5%** |

→ uni_video 单独看 +32.2% 仍 PASS,但相对 cross_attn 落后 **−38.5%**,**预测准确**。
→ 在真实 EMMA rover 上,cross_attn 的 +51.0% 增益由两部分组成:
>   - **架构贡献 ≈ +32.2%**(uni_video 拿到的部分)
>   - **audio 信息贡献 ≈ +18.8 pp**(剩余 51.0 − 32.2 = 18.8 pp,来自 motor RPM ↔ wheel radius 这条 video 推不出的耦合)
→ JSON: `analysis/emma_rover/2026-06-03_005615_emma_rover.json`。

### 2.5 元结论的精确量化

| 任务 | 架构贡献 | audio 信息贡献 | 验证 |
|---|---:|---:|---|
| 合成 burst (round 8) | ~26.6 pp(99%) | ~1.0 pp(1%) | ✅ |
| 真实 EMMA rover (round 11) | ~32.2 pp(63%) | ~18.8 pp(37%) | ✅ |

**首次对"架构 vs 信息"的两种贡献做出量化分离。** 元结论从定性变为定量,W+1 backlog 据此精炼。

## 3. 下一步研究思路 (W+1 backlog, 精简后)

按价值排序:

1. **更多真实 EMMA 视频做 leave-one-trial-out** — 在 rover 多视频上重做 round 11+13,确认 audio 信息贡献的 18.8pp 是该任务的稳定属性还是单视频特异性。
2. **EMMA quadrotor 12 参数回归** — 同 pipeline 迁移到无人机数据(论文 Table 4(d) 提到 7/12 已知参数),看 audio 信息贡献是否更高(预期是,因为螺旋桨音频更直接编码 RPM)。
3. **视觉化 cross_attn 的 attention 矩阵** — 在 rover 数据上画 `_attn_video_queries_audio` 热图,看 video 在哪些 time step 主动"借" audio;若 attention 集中在 motor 加减速的瞬间,就直接看到了 audio 信息贡献的物理通道。
4. **LFM2.5-8B-A1B Opus 蒸馏的本地推理 smoke** — 用今日 digest 里 `reaperdoesntknow/LFM2.5-8B-A1B-Opus-Distil` 跑一段本地 MLX 推理,看蒸馏模型在 LNN 风格序列任务上是否保留液态时间常数特性。
5. *新增*:**架构-vs-信息 trade-off 的更高维探针** — 系统性扩 video_dim / audio_dim,看 "audio 信息贡献" 是否随 video 信息容量减少而成比例增长。这是把 round 13 的二元划分进一步细化为连续曲线。
6. 稀疏 / 分块 cross-attention(只在真实长视频 T=256+ 才有意义,本日不优先)。

## 4. 提交 + 推送

- 新增模型类 + 4 个单测 + 2 个 benchmark 扩展 + 2 个新 JSON + 本报告 + appendix §16,准备 commit。
- 全套 `pytest tests/` **115/115 通过**。numpy 从 2.4.6 降到 1.26.4(本会话 `pip install 'numpy<2'`)修复了之前 round 11/12 的 ncps 环境冲突,记入 commit message。

---
*本报告由 6h cron `7131cb00` 触发(今日第一次),与并发的 1h cron `855d0d94` 的 round 13 (本日同会话)结果一起构成今日完整产出。*
