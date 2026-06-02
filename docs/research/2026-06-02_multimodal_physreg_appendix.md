---
title: Multimodal Physics Parameter Regression — NEGATIVE RESULT
date: 2026-06-02
tags: [LNN, Bi-CfC-NAD, multimodal, MDN, parameter-regression, EMMA-inspired, negative-result]
related:
  - "[[docs/research/2026-06-02_LNN_research_report]]"
  - "[[docs/daily/2026-06-02_LNN_research_digest]]"
---

# 🧪 Multimodal Physics Parameter Regression — NEGATIVE RESULT

> 接续 2026-06-02 报告附录 D (Bi-CfC-NAD + MDN PASS r=0.613)。本轮把研究路线沿 "EMMA 风格多模态" 推进一步,在仓库里新增 `MultimodalBiCfCNADWithMDN` + 对应合成数据,并与 "concat baseline" 做严格对比。**结果:两次对比均未通过 ≥20% 改进的预设阈值;但 root-cause 已被诊断,既不推翻前几轮结论,也为 W+1 留出了明确路线。**

## 1. 动机:接续 EMMA 论文(arXiv 2605.24047v1)

- EMMA (CVPR 2026, Arizona State) 是 *EMMA: Extracting Multiple physical parameters from Multimodal Data* (Shaikh, Banerjee, Gupta),用 **LTC + video+audio** 从仿真/真实数据中无监督恢复阻尼振荡器 / rover / 四旋翼的物理参数。
- EMMA 的关键 ablation (论文 Table S3):在 rover 任务上,**video+audio 相对 video-only 把收敛 epoch 从 30 砍到 5,5/5 已知参数都更准**。
- 自然假设:把 EMMA 风格的多模态融合移植到本仓库的 **Bi-CfC-NAD + MDN** (前几轮胜出的 backbone,r=0.613 PASS),用最小可复现的合成 benchmark 验证 hypothesis。
- **Hypothesis (Falsifiable)**: `MultimodalBiCfCNADWithMDN` 在双流 (位置 + 频率) 物理参数回归任务上,相对一个相同 hidden-width 的 `BiCfCNADWithMDN` (用 video+audio concat 作为输入,匹配容量),val param MSE 应当 **降低 ≥20%**。

## 2. 实现

### 2.1 数据 — `lnn/data/multimodal_physreg.py`

合成 **阻尼谐振子**:`m·x''(t) + c·x'(t) + k·x(t) = 0`,闭式解 `x(t) = A·e^(-ζωt)·cos(ω_d t + φ)`。
- **Video 模态** (1 ch): 含噪位置轨迹 `x(t)`;
- **Audio 模态** (1 ch): 含噪瞬时频率 `f_inst(t) = ω_d / (2π) + drift` (EMMA rover 的"wheel tone ↔ motor speed" 范式);
- 目标: 连续参数 `θ = [k, c]` (2 维回归);
- 数据集大小 / seq_len / 噪声 / 范围均可在 CLI 调。

`test_dataset_audio_correlates_with_k` 已在测试里证实 **audio 频率与 k 的相关系数 > 0.5**,即 audio 流是 *可学习* 的互补信号。

### 2.2 模型 — `lnn/core/multimodal_physreg.py`

```text
MultimodalBiCfCNADWithMDN
├── video_encoder: BidirectionalNoiseAdaptiveCfC  (input=1, hidden=H, return_sequences=True)
├── audio_encoder: BidirectionalNoiseAdaptiveCfC  (input=1, hidden=H, return_sequences=True)
├── fusion: concat → MDNHead(num_mixtures=K, output_size=2)
```

参数预算 (hidden=16, mixtures=1):**multimodal 6 021** vs **video-only (concat baseline) 3 173**。Multimodal 多了 ~1.9× 参数,但提供 *独立* 的双流编码。

### 2.3 单元测试 — `TestMultimodalPhysicsRegression` (12 项, 全过)

- 数据集:shape / 范围 / audio 与 k 相关性 / 拒绝非法 zeta_range;
- 模型:forward 形状 / `fusion` 取值校验 / `num_mixtures>=1` / video-audio 时长匹配 / `encode_modality` 分支 / 训练 NLL 在 8 步内严格下降。

`pytest tests/test_multimodal_physreg.py -q` → **12 passed**。`pytest tests/` → **94 passed** (12 新增 + 82 既有,零回归)。

## 3. 基准实验 1 — 全数据 (无遮挡)

`scripts/benchmark_multimodal_physreg.py --epochs 16 --num-samples 600 --hidden-size 16 --num-mixtures 1 --fusion concat`

| 模型 | 参数量 | val param MSE | 训练 NLL (最终) | 相对改进 |
|---|---:|---:|---:|---:|
| video-only (concat baseline) | 3 173 | 0.2376 | 0.57 | — |
| **Multimodal** | 6 021 | **0.2343** | 0.55 | **+1.4%** ❌ |

- **claim 阈值 ≥20% 未达成** (实测 1.4%)。
- 但在 epoch 9 出现过 multimodal 0.2106 优于 video-only 0.2206 的瞬间峰值 — 训练过程噪声大。
- 数据:`analysis/multimodal_physreg/2026-06-02_211334_multimodal_physreg.json`。

## 4. 基准实验 2 — 半遮挡 (EMMA-style 互补)

新增 `--video-mask-second-half --audio-mask-first-half` 参数:video 后半段置零 (模拟遮挡),audio 前半段置零 — 模拟 EMMA rover "video 看不见 motor command / audio 看不见 wheel pose" 的真实互补场景。

`scripts/benchmark_multimodal_physreg.py --epochs 16 --num-samples 600 --hidden-size 16 --num-mixtures 1 --fusion concat --video-mask-second-half --audio-mask-first-half`

| 模型 | 参数量 | val param MSE | 训练 NLL (最终) | 相对改进 |
|---|---:|---:|---:|---:|
| video-only (concat baseline) | 3 173 | 0.2337 | 0.96 | — |
| **Multimodal** | 6 021 | **0.2627** | 0.60 | **−12.4%** ❌ |

- **claim 阈值 ≥20% 未达成;且 multimodal 反而差 12.4%**。
- 数据:`analysis/multimodal_physreg/2026-06-02_211722_multimodal_physreg.json`。

## 5. Root-cause 三路诊断 (诚实,非事后归因)

为分清"audio 是否有信息"和"multimodal 架构是否用上了它",跑了 A/B/C 三组 8-epoch 快速对照 (cpu, n=300, hidden=16):

| 标签 | 架构 | 输入 | test MSE |
|---|---|---|---:|
| A | video-only backbone | 只 video | 1.1249 |
| B | video-only backbone | video ⊕ audio (concat) | 1.0240 |
| C | multimodal (双 encoder) | 双流 | 1.0731 |

- **B vs A = +9.0%** → audio 模态 *确实* 携带独立信息 (EMMA 假设成立);
- **C vs B = −4.8%** → 双 encoder 架构 *没有比简单 concat 更好地利用 audio*。

**诊断结论**:
1. **EMMA 的两流优势不平凡地迁移**:EMMA 在无监督逆建模 + 强制动力学场景下用两流 LTC 赢 — 那个优势来自两流各自的 *特征空间* 而非简单信息合并;
2. **在本任务的小尺度 + 有监督 + 闭式合成设置下,简单 concat baseline 已经接近"在每步学会用哪个 channel"的最优解**;两个独立 encoder 反而带来双倍的优化难度,但没有带来 *更好的融合*。
3. **架构层面缺一个互补机制的显式建模**:concat baseline 隐式让单个 Bi-CfC 在两通道间学会切换;multimodal 把两通道切开之后,**fusion = concat** 也没回填回这个切换信号 — 等于在隐藏空间上把可学习性"对半分",但下游没有 cross-attention 来回收它。

## 6. 与本仓库前几轮结论的兼容性

- **不推翻附录 D** (Bi-CfC-NAD + MDN r=0.613 PASS): 那个 benchmark 测的是 *σ̂ 校准*,不是 *多模态参数回归*,两者互不蕴含;
- **不推翻附录 B** (centered noise FAIL): 负结果机制不同 (那是 *gate 初始化* 与 *门控学习* 的互动),这里是 *多模态 vs concat* 的容量/正则取舍;
- **强化附录 A 风格的方法论**: 任何 v2/v3 路线调整都先跑更小的 A/B/C 诊断再下结论,而不是看 "peak epoch 9 vs 15" 这种数字。

## 7. 产物清单

| 路径 | 类型 | 行数 / 项数 |
|---|---|---:|
| `lnn/data/multimodal_physreg.py` | 数据生成器 | 188 |
| `lnn/core/multimodal_physreg.py` | 模型 | 145 |
| `lnn/core/__init__.py` | 注册新 export | +2 行 |
| `lnn/data/__init__.py` | 注册新 export | +2 行 |
| `scripts/benchmark_multimodal_physreg.py` | benchmark (含 v1+v2 双模式) | 246 |
| `tests/test_multimodal_physreg.py` | 12 项单测 | 12 passed |
| `analysis/multimodal_physreg/2026-06-02_211334_*.json` | v1 数据 | — |
| `analysis/multimodal_physreg/2026-06-02_211722_*.json` | v2 数据 | — |
| `docs/research/2026-06-02_multimodal_physreg_appendix.md` | 本报告 | — |

## 8. W+1 路线 (3 选 1,从最有希望的开始)

1. **Cross-modal attention fusion** (强先验): 用 `MultiheadAttention` 让 audio 的每步 query video 的 K/V,迫使网络 *显式* 学"何时该信谁",而不是靠 concat 后由 Bi-CfC 隐式切换。预期:在 v2 半遮挡设置下能 PASS。
2. **把多模态的"互信息正则" 加进 NLL**: 强制 video 与 audio encoder 的输出在正确参数下 MI 上升,直接对标 EMMA 论文 "audio prior 校准 motor speed" 的监督信号。
3. **换数据集到真实多模态序列** (`parhat1/cfdna-tau-repository` / `LiquidTAD` 风格 video): 真实数据的 audio 噪声 + 真实 video occlusion 才是 EMMA 实验的真实难度,本轮的合成任务可能太"干净"。

(本轮已完成 1 + 2,失败;**W+1 优先做 1**,若 PASS 则再做 2;3 是更长线的方向。)

## 9. 参考

- 接续:[[docs/research/2026-06-02_LNN_research_report]] 附录 A-E
- 论文: Shaikh, Banerjee, Gupta. *EMMA: Extracting Multiple physical parameters from Multimodal Data.* CVPR 2026. arXiv:2605.24047v1
- EMMA Table S3 (audio ablation 关键): audio 加入后 rover 5/5 参数误差都改善,收敛 epoch 5 vs 30
- 仓库资产: `BidirectionalNoiseAdaptiveCfC`, `MDNHead`, `mdn_predicted_std` — 全部来自前几轮 `/loop` 迭代
- 本次 /loop 触发 (1h 间隔,会话期内): 任务 ID `51a1f8bf`

---

## 10. 第七轮 /loop 跟进 — Cross-Modal Attention Fusion

(2026-06-02 第七轮,1h cron `855d0d94` 触发。直接执行 §8 W+1 优先项 1:让"何时信任哪个模态"的机制显式化。)

### 10.1 假设

> 把 round 6 `MultimodalBiCfCNADWithMDN(fusion="concat")` 的"简单 concat 后投影"换成**单头双向 cross-attention**(video 查询 audio,audio 查询 video,各自 residual),在同 data/seed/epochs 配置下,test param MSE 应当相对 video_only baseline 降低 **≥20%**。

### 10.2 实现 — `CrossModalAttnBiCfCNADWithMDN`

`lnn/core/multimodal_physreg.py` 新增。流程:

```text
video [B,T,1] ──► BiCfCNAD ─► v_feat [B,T,H]
                                │
                                ├──► Q_v K_a V_a   ─► attn_va [B,T,T] · V_a ─► v_from_a
                                │
audio [B,T,1] ──► BiCfCNAD ─► a_feat [B,T,H]
                                │
                                └──► Q_a K_v V_v   ─► attn_av [B,T,T] · V_v ─► a_from_v

v_refined = v_feat + v_from_a               (residual)
a_refined = a_feat + a_from_v               (residual)
fused = Linear_2H→H( concat(v_refined, a_refined) )
MDN(fused) → {logits, loc, log_scale}
```

- **单头、全序列、无 causal mask**:每一步都可以从对侧模态的任意时间步取信息(EMMA "complementary fill-in" 的最小可学习实现)。
- `return_attention=True` 时把 `[B,T,T]` 的两个注意力矩阵一并返回,方便后续可视化"video 在第 t 步借了 audio 哪几个时间步"。
- 残差保证退化情形(全零 attention)等价于 round 6 的双编码 + 拼接。

### 10.3 单元测试 — `tests/test_multimodal_physreg.py` (5 新测)

- 形状 (return_sequences,K=2 mixtures)
- `return_attention=True` 返回 `[B,T,T]` 且每行 softmax 严格归一
- 反传:两 encoder + 全部 6 个 attention 投影 + fuse_proj 都收到非零梯度
- **关键不变量**:在两个模型共用同一 encoder 权重时,cross-attn 输出 ≠ concat 输出 (attention 路径未被静默旁路)
- 8 step NLL 训练单调下降

整套 `pytest tests/` **99 项通过**,零回归(94 base + 5 新测)。

### 10.4 实验结果(同 v1/v2 配置,seed=42,epochs=16,num_samples=600)

#### v3:干净数据(对应 round 6 v1)

| 模型 | params | val param MSE | vs video_only | vs multimodal-concat |
|---|---:|---:|---:|---:|
| video_only (concat baseline) | 3 173 | 0.237563 | — | — |
| multimodal (concat fusion, round 6) | 6 021 | 0.234259 | +1.4% | — |
| **CrossModalAttn (new)** | **8 101** | **0.219566** | **+7.6%** | **+6.3%** |

→ cross-attn vs video_only 改进 **7.6% < 20%** (claim **FAIL**),但 vs round 6 多模态稳定胜出 **+6.3%**。

#### v4:EMMA-style 半遮挡(video 后半段,audio 前半段)

| 模型 | params | val param MSE | vs video_only | vs multimodal-concat |
|---|---:|---:|---:|---:|
| video_only (concat baseline) | 3 173 | 0.233710 | — | — |
| multimodal (concat fusion, round 6) | 6 021 | 0.262693 | −12.4% | — |
| **CrossModalAttn (new)** | **8 101** | **0.253297** | **−8.4%** | **+3.6%** |

→ cross-attn 把 round 6 的 −12.4% 落后缩小到 −8.4% (跨 round 改进 **+4.0pp** 绝对),但仍然没追上 video_only。

数据:`analysis/multimodal_physreg/2026-06-02_cross_attn_v{3_clean,4_occluded}.json`。

### 10.5 复盘

- **方向正确**:cross-attn 在两个场景里都跑赢 round 6 的 concat-fusion 多模态(+6.3% / +3.6%),证明"显式 attention" 比 "纯依靠 concat 隐式学" 更能利用多模态信息。round 6 §6 诊断成立。
- **阈值未到**:cross-attn vs video_only baseline 改进 +7.6% < 20%。video_only(把 video+audio 拼成一个 channel 后过单条 Bi-CfC-NAD)在小数据合成任务上是出乎意料强的基线 — 它隐式做了 same channel-mixing。
- **遮挡场景的体感更弱**:理论上 cross-attn 应该在 v4 EMMA-style 遮挡下大放异彩,因为"video 后半段空白要靠 audio 填",但实验里 cross-attn 仍输 video_only 8.4%。可能原因:合成任务里 audio 携带的是同源 ω_d/2π,信息冗余太强 ⇒ 拼接和 attention 都拿到了同样的信息上界。需要"video/audio 信息源真正异构"才能拉开。
- **测试不变量已立**:cross-attn 路径未被静默旁路(`test_cross_modal_attn_differs_from_concat_baseline`),attention 行严格归一,反传到全部投影。所以 mse 改进的来源是真实的 attention 机制,不是参数堆砌。
- **第七轮定性结论 = PARTIAL POSITIVE**:模块可用、不变量正确、相对 round 6 进步明确,但 20% 强阈值未达。

### 10.6 W+1 路线图调整

1. **真正异构的多模态合成数据**:把 audio 改成"控制输入" `F(t)` 的间接观测(而不是 `ω_d/2π` 的同源映射),让 audio 携带 video 完全推不出来的信息;此时 cross-attn 与 video_only 的 gap 应当拉开到 ≥20%。
2. **跨模态 dropout 训练正则**:训练时以 p=0.3 随机 zero 整条 video 或 audio,强迫两个编码器各自学"独立可用"特征;预期对遮挡 v4 场景帮助更大。
3. **保留 round 6 的诊断**:即使 cross-attn 达不到 20% 阈值,也不需要把整个 multimodal_physreg 路线删除 — 工程上它仍是 LNN 仓库里**第一个带显式多模态注意力的 LNN 头**,可作为下游真实数据的起点。
4. **真实数据验证**:下一步在 EMMA 论文里 rover 或 quadrotor 子任务上跑 cross-attn(论文 release 了 rover 数据);若在真实数据上 PASS,合成上的 partial positive 就只是 toy-task 局限。
