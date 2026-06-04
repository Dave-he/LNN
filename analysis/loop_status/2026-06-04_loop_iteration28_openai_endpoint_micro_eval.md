---
title: 2026-06-04 Loop iteration 28 — OpenAI-compatible endpoint backend for LLM micro-eval
date: 2026-06-04
tags: [LNN, LFM2.5, LLM, openai-compatible, endpoint-eval, 30B-plus, micro-benchmark]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 28 — endpoint backend for 30B+ micro-eval

> iter#27 已经能用本机 `llama-cli` 跑 LFM2.5 1.2B GGUF micro-eval。
> 本轮补上同一套题对 30B+ 模型服务的入口: OpenAI-compatible
> `/v1/chat/completions` backend。

## 1. 新能力

`scripts/run_llm_micro_eval.py` 新增:

```bash
--backend openai-chat
--openai-base-url http://127.0.0.1:8000/v1
--openai-model <model-id>
--openai-api-key <optional>
```

适配目标:

- 本机 `llama-server` OpenAI-compatible API;
- 本机/局域网 `vLLM` 或 `SGLang` 30B+ 服务;
- 远程 OpenAI-compatible API。

同一输出 schema:

- `summary.accuracy`
- `summary.generation_tps_mean`
- `results[].grade`
- `results[].usage`
- `results[].endpoint`

## 2. 本机 LFM2.5 GGUF 回归

命令:

```bash
python scripts/run_llm_micro_eval.py --json
```

结果:

- `LFM2.5-1.2B-Instruct-Q4_0.gguf`: 7/7
- mean generation: 16.843 tok/s
- 输出已刷新:
  - `analysis/llm_micro_eval/2026-06-04_lfm25_1_2b_instruct_q4_micro_eval.json`
  - `analysis/llm_micro_eval/2026-06-04_lfm25_1_2b_instruct_q4_micro_eval.md`

注: 该吞吐是当前系统负载下的瞬时值,同日此前 run 在 26-31 tok/s 区间。

## 3. Battlecard 集成

`analysis/llm_battlecard/2026-06-04_llm_battlecard.md` 已重建,Local Evidence
现在显示:

```text
Micro-eval: 100.0% (7/7), mean generation 16.843 tok/s
```

## 4. 验证

```bash
python -m pytest tests/test_llm_micro_eval.py
python scripts/run_llm_micro_eval.py --backend openai-chat \
  --model-name fake_30b_endpoint \
  --openai-base-url http://127.0.0.1:9/v1 \
  --dry-run --no-write
```

结果:

- `tests/test_llm_micro_eval.py`: 9 passed
- fake OpenAI server 单测覆盖 `/v1/chat/completions` request/response 解析
- dry-run endpoint payload OK

## 5. 仍未完成

这只证明“同 prompt 同 grader 可接 30B+ endpoint”。真实 30B+ 对照仍需要:

1. 启动本机/远程 30B+ OpenAI-compatible 服务,例如 Qwen3-30B-A3B；
2. 运行 `run_llm_micro_eval.py --backend openai-chat --openai-model ...`；
3. 把输出和本机 LFM2.5 1.2B/8B-A1B 输出合并成 micro leaderboard；
4. 再接 `lm-eval-harness` 或 OpenCompass 子集做公开榜可复现验证。
