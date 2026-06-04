---
title: 2026-06-04 Loop iteration 30 - Live local OpenAI-compatible endpoint micro-eval
date: 2026-06-04
tags: [LNN, LFM2.5, LLM, llama-server, openai-compatible, endpoint-eval, leaderboard]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 30 - live local endpoint micro-eval

> iter#29 已经把 micro-eval JSON 合成 leaderboard。
> 本轮用真实本机 `llama-server` OpenAI-compatible endpoint 跑同一套 7 题,
> 证明后续替换为 30B+ endpoint 时同一路径可直接复用。

## 1. 运行环境

已有服务:

```text
projects/llama.cpp/build/bin/llama-server \
  -m models/lfm25/LFM2.5-1.2B-Instruct-Q4_0.gguf \
  --host 0.0.0.0 --port 18080 -c 2048 -t 4 --parallel 1 \
  --n-gpu-layers 0 --fit off --no-webui
```

本轮没有启动 30B+ 权重。本地可用模型仍只有:

- `models/lfm25/LFM2.5-1.2B-Instruct-Q4_0.gguf`
- `models/lfm25-dpo-s1/LFM25-DPO-Q4_0.gguf`
- `models/lfm25-dpo-s1/LFM25-DPO-F16.gguf`
- `models/lfm25-dpo-s1/model.safetensors`

## 2. 命令

```bash
python scripts/run_llm_micro_eval.py \
  --backend openai-chat \
  --model-name lfm25_1.2b_instruct_q4_http \
  --openai-base-url http://127.0.0.1:18080/v1 \
  --openai-model lfm25-1.2b-instruct-q4 \
  --timeout 120 \
  --json
```

## 3. 结果

输出:

- `analysis/llm_micro_eval/2026-06-04_lfm25_1_2b_instruct_q4_http_micro_eval.json`
- `analysis/llm_micro_eval/2026-06-04_lfm25_1_2b_instruct_q4_http_micro_eval.md`

结果:

- accuracy: 7/7 = 100.0%
- arithmetic: 3/3
- instruction: 2/2
- structured_output: 1/1
- abstention: 1/1
- mean generation: 5.707 tok/s
- median generation: 5.587 tok/s

该速度低于直跑 `llama-cli` 的 16.843 tok/s,主要反映服务路径、HTTP 开销和短输出任务计时方式。

## 4. Leaderboard / Battlecard 集成

重建:

```bash
python scripts/build_llm_micro_leaderboard.py --json
python scripts/build_llm_battlecard.py --json
```

当前 micro leaderboard:

- entries: 2
- rankable: 2
- roles: `under_3b_candidate=2`
- rank 1: `lfm25_1.2b_instruct_q4` / `llama-cli` / 7/7 / 16.843 tok/s
- rank 2: `lfm25_1.2b_instruct_q4_http` / `openai-chat` / 7/7 / 5.707 tok/s

`analysis/llm_battlecard/2026-06-04_llm_battlecard.md` 现在会读取最新
`analysis/llm_micro_eval/2026-06-04_llm_micro_leaderboard.json`,并在 Local Evidence 中显示:

```text
Micro leaderboard: 2 entries, roles: under_3b_candidate=2; leader `lfm25_1.2b_instruct_q4` (100.0%, 16.843 tok/s)
```

## 5. 限制

- 这不是 30B+ baseline;只是本机 1.2B 模型经 OpenAI-compatible endpoint 的 live run;
- 当前 leaderboard 仍没有 `30b_plus_baseline` 行;
- 因此仍不能声称 3B/active≤3B 已在本机实测中击败 30B+。

## 6. 验证

```bash
python -m pytest tests/test_llm_battlecard.py tests/test_llm_micro_leaderboard.py tests/test_llm_micro_eval.py
python scripts/build_llm_micro_leaderboard.py --json
python scripts/build_llm_battlecard.py --json
```

结果:

- targeted tests: 20 passed
- leaderboard rebuild: OK,2 entries
- battlecard rebuild: OK,leaderboard summary 已接入
