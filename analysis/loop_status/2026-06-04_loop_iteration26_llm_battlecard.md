---
title: 2026-06-04 Loop iteration 26 — LFM/LNN active≤3B vs 30B+ battlecard
date: 2026-06-04
tags: [LNN, LFM2.5, LLM, benchmark, battlecard, active-3B, 30B-plus, claim-audit]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 26 — LFM/LNN active≤3B vs 30B+ battlecard

> 面向用户目标“本机部署和训练 LNN 相关模型,看 3B 是否能吊打 30B+ LLM”。
> 本轮没有声称训练完成,而是先补一层可审计的打榜证据卡,把本地 LFM2.5
> 推理证据、公开 30B+ 基线、胜负统计和预测边界固定下来。

## 1. 新增脚本

`scripts/build_llm_battlecard.py`

- 默认候选: `LiquidAI/LFM2.5-8B-A1B`。
- 默认基线: `Qwen/Qwen3-30B-A3B` / `Qwen3-30B-A3B-Thinking-2507`。
- 明确区分:
  - exact 3B dense: **未证明**;
  - active≤3B MoE: `8.3B total / 1.5B active` 可作为当前 LFM-family 最接近目标。
- 汇总本仓已有本地验证:
  - `analysis/lfm25/2026-06-01_lfm25_local_validation.json`
  - GGUF `ok`, generation `19.00 tok/s`;
  - DPO `ok`, generation `1.75 tok/s`。

## 2. 打榜证据卡

输出:

- `analysis/llm_battlecard/2026-06-04_llm_battlecard.json`
- `analysis/llm_battlecard/2026-06-04_llm_battlecard.md`

核心结果:

| 对照 | 结果 |
|---|---:|
| shared public metrics | 13 |
| LFM2.5-8B-A1B wins | 7 |
| Qwen3-30B-A3B wins | 6 |
| win rate | 53.8% |
| knowledge/instruction split | 5 win / 1 loss |
| math/agentic split | 2 win / 5 loss |

结论: 支持 **active≤3B MoE 在 instruction / non-hallucination / 部分
agentic 场景可赢 30B+ 基线**;不支持“3B dense 全面吊打 30B+”。

## 3. PRD 更新

- §4 新增 `LFM/LLM 打榜证据卡` 功能模块。
- §10 新增 `10-11` 任务并标记本轮完成。

## 4. 验证

```bash
python scripts/build_llm_battlecard.py --no-write --json
python -m pytest tests/test_llm_battlecard.py tests/test_loop_status_prd.py tests/test_backbone_matrix_dedup.py
python scripts/verify_all_models.py
python scripts/loop_status.py --prd-status --no-write --json
```

结果:

- `build_llm_battlecard.py --no-write --json`: OK
- touched/regression tests: 18 passed
- `verify_all_models.py`: 9/9 passed
- `loop_status.py --prd-status`: OK, §10 新增 `10-11` 被识别为 completed

Full-suite note:

- `python -m pytest`: 208 passed / 2 skipped / 5 failed
- 5 个失败均来自 EMMA rover 资源缺失: `/tmp/RoverVideo.mp4: No such file or directory`
- battlecard 相关测试在 full-suite 与 targeted-suite 中均通过

## 5. 下一步

1. 下载/挂载 `LFM2.5-8B-A1B-GGUF`,在 Jetson/desktop 跑同一脚本的本地 inference 补证。
2. 接 `lm-eval-harness` 或 OpenCompass 子集,把 public snapshot 升级为本仓可复现 benchmark。
3. 若坚持 exact 3B,需先选一个 LNN/LFM-family 3B 目标;当前官方 Liquid 公开主线是 1.2B 与 8B/A1B,不是 exact 3B dense。

## 6. Sources

- <https://huggingface.co/LiquidAI/LFM2.5-8B-A1B>
- <https://www.liquid.ai/blog/lfm2-5-8b-a1b>
- <https://huggingface.co/Qwen/Qwen3-30B-A3B>
- [[LFM2.5 本地推理与 1.2B DPO 量化验证]]
