---
title: 2026-06-04 Loop iteration 27 — local LFM2.5 GGUF micro-eval harness
date: 2026-06-04
tags: [LNN, LFM2.5, LLM, local-eval, llama.cpp, micro-benchmark, 3B-vs-30B]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 27 — local LFM2.5 GGUF micro-eval harness

> 继续推进用户目标“本机部署/训练 LNN 相关模型,评估 3B 是否能吊打 30B+ LLM”。
> iter#26 已有 public battlecard;本轮把它向本机可复现实测推进一步。

## 1. 新增脚本

`scripts/run_llm_micro_eval.py`

- 通过本仓 `projects/llama.cpp/build/bin/llama-cli` 调用 GGUF 模型。
- 默认模型: `models/lfm25/LFM2.5-1.2B-Instruct-Q4_0.gguf`。
- 内置 7 个 deterministic sanity tasks:
  - arithmetic: 3
  - instruction following: 2
  - JSON structured output: 1
  - abstention / unknown: 1
- 输出 JSON + Markdown 到 `analysis/llm_micro_eval/`。
- 可通过 `--model` / `--model-name` 换成 `LFM2.5-8B-A1B-GGUF` 或 30B+ GGUF。

## 2. 本机实测结果

命令:

```bash
python scripts/run_llm_micro_eval.py --json
```

结果:

| Model | Accuracy | Passed | Mean generation |
|---|---:|---:|---:|
| `LFM2.5-1.2B-Instruct-Q4_0.gguf` | 100.0% | 7/7 | 26.2 tok/s |

输出:

- `analysis/llm_micro_eval/2026-06-04_lfm25_1_2b_instruct_q4_micro_eval.json`
- `analysis/llm_micro_eval/2026-06-04_lfm25_1_2b_instruct_q4_micro_eval.md`

## 3. Battlecard 集成

`scripts/build_llm_battlecard.py` 现在会自动读取最新
`analysis/llm_micro_eval/*_micro_eval.json`,并在 Local Evidence 中显示:

- micro-eval file path
- accuracy
- passed / total
- mean generation speed

重建后的 `analysis/llm_battlecard/2026-06-04_llm_battlecard.md` 已包含:

```text
Micro-eval: 100.0% (7/7), mean generation 26.157 tok/s
```

## 4. Fixes

- 修复 `Path.with_suffix()` 遇到 `model_name=lfm25_1.2b...` 时输出文件名被截断的问题。
- 新增 run_id sanitize,避免模型名里的 `.`, `/`, `:` 进入文件名。

## 5. 验证

```bash
python -m pytest tests/test_llm_battlecard.py tests/test_llm_micro_eval.py
python scripts/build_llm_battlecard.py --json
python scripts/run_llm_micro_eval.py --json
```

结果:

- battlecard + micro-eval tests: 13 passed
- `run_llm_micro_eval.py`: 7/7, mean generation 26.2 tok/s
- `build_llm_battlecard.py`: OK, local micro-eval 已接入

## 6. 结论边界

这证明了本机 LFM2.5 1.2B Q4 部署链路能完成最小 deterministic sanity
eval,但**仍不证明** 3B/active≤3B 可以吊打 30B+ LLM。下一道证据门槛:

1. 获取 `LFM2.5-8B-A1B-GGUF`,用同一 micro-eval 跑本机速度和正确率。
2. 接 `lm-eval-harness` 或 OpenCompass 子集,对公开 benchmark 做可复现子集。
3. 至少接一个 30B+ 本地 GGUF 或 OpenAI-compatible endpoint,同脚本同 prompt 对比。
