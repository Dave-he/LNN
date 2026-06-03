# LiquidTAD 3-way Head Ablation — 2026-06-03_233348

## 环境
- device: cpu
- torch: 2.11.0+cu130 (cuda available=False)
- python: 3.14.4

## 配置
- samples / seq_len / feature_size: 192 / 96 / 6
- num_classes (foreground) / num_blocks / hidden_size: 3 / 3 / 32
- epochs / batch_size / lr / seed: 8 / 16 / 0.0003 / 42
- decay init / growth (for hierarchical heads): 0.8 / 1.05

## 结果
| Head | 参数量 | Test loss | Test frame acc | Test boundary MAE | 训练秒 |
|---|---:|---:|---:|---:|---:|
| `data_dependent` | 22,730 | 0.4142 | 92.01% | 0.2315 | 15.62 |
| `hierarchical_decay` | 19,526 | 0.2224 | 99.34% | 0.1944 | 14.15 |
| `hierarchical_shared` | 19,462 | 0.2192 | 99.13% | 0.1933 | 14.72 |

## 相对 baseline (`data_dependent`) 的变化
| Head | Δparams | Δtest_loss | Δframe_acc (pp) |
|---|---:|---:|---:|
| `hierarchical_decay` | -14.10% | -46.32% | +7.33pp |
| `hierarchical_shared` | -14.38% | -47.07% | +7.12pp |

## 解读模板
- params 减少而 acc 不显著掉 → hierarchical prior 在该规模下生效;
- params 减少且 acc 明显掉 → 容量上限,推大 hidden_size/epochs 再看;
- params 减少且 acc 反而涨 → 论文 sharing prior 直接验证,记录为强证据。

产出 JSON: `analysis/long_sequence/2026-06-03_233348_liquid_tad_head_ablation.json`
