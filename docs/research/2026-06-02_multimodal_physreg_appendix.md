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
