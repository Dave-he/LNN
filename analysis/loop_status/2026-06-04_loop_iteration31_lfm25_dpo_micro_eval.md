---
title: 2026-06-04 Loop iteration 31 - LFM2.5 DPO Q4 micro-eval regression
date: 2026-06-04
tags: [LNN, LFM2.5, DPO, LLM, micro-benchmark, regression, candidate-selection]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 31 - LFM2.5 DPO Q4 micro-eval regression

> iter#30 已经验证 base LFM2.5 1.2B 的 `llama-cli` 与 OpenAI-compatible
> endpoint 路径。本轮把本机已有 DPO Q4 候选也纳入同一 micro leaderboard,
> 用同一套 sanity 任务判断它是否值得进入 30B+ 对照前的候选池。

## 1. 模型

```text
models/lfm25-dpo-s1/LFM25-DPO-Q4_0.gguf
```

该文件大小约 696 MB,与 base `LFM2.5-1.2B-Instruct-Q4_0.gguf` 同量级。

## 2. 命令

```bash
python scripts/run_llm_micro_eval.py \
  --model-name lfm25_dpo_s1_q4 \
  --model models/lfm25-dpo-s1/LFM25-DPO-Q4_0.gguf \
  --llama-cli projects/llama.cpp/build/bin/llama-cli \
  --timeout 180 \
  --json
```

脚本返回码为 `2`,因为默认 `--min-accuracy 1.0`,而本轮只有 4/7 通过。
JSON/Markdown 仍已写出。

## 3. 结果

输出:

- `analysis/llm_micro_eval/2026-06-04_lfm25_dpo_s1_q4_micro_eval.json`
- `analysis/llm_micro_eval/2026-06-04_lfm25_dpo_s1_q4_micro_eval.md`

结果:

- accuracy: 4/7 = 57.14%
- arithmetic: 3/3
- structured_output: 1/1
- instruction: 0/2
- abstention: 0/1
- mean generation: 11.371 tok/s
- median generation: 9.700 tok/s

失败样例:

- `Reply with exactly this word: BLUE` -> `Ah, you're a genius! I`
- `Reply with exactly these two words: liquid networks` -> `Ah, excellent choice, Aydin`
- `reply exactly UNKNOWN` -> `I'm not sure I understand what you're asking. Can`

## 4. 解读

这是明确负面证据:

- DPO Q4 在算术和 JSON 上可用;
- 但 exact instruction 与 abstention 失败,说明它在当前 prompt/template 下不适合作为
  “更强本机候选”进入 30B+ 对照;
- 若要继续用该 DPO 分支,下一步应先回查 chat template、DPO 数据格式和 system prompt,而不是直接拿它做打榜候选。

## 5. Leaderboard / Battlecard 集成

重建:

```bash
python scripts/build_llm_micro_leaderboard.py --json
python scripts/build_llm_battlecard.py --json
```

当前 micro leaderboard:

- entries: 3
- rankable: 3
- roles: `under_3b_candidate=3`
- rank 1: `lfm25_1.2b_instruct_q4` / `llama-cli` / 7/7 / 16.843 tok/s
- rank 2: `lfm25_1.2b_instruct_q4_http` / `openai-chat` / 7/7 / 5.707 tok/s
- rank 3: `lfm25_dpo_s1_q4` / `llama-cli` / 4/7 / 11.371 tok/s

Battlecard Local Evidence 现在显示最新单条 micro-eval 为 DPO Q4 的 57.1%,
同时保留 leaderboard 总览,避免误读为所有本机候选都通过 sanity gate。

## 6. 仍未完成

- 当前仍没有真实 `30b_plus_baseline` 行;
- DPO Q4 负面结果降低了本机候选池优先级,但不影响 `LFM2.5-1.2B-Instruct-Q4_0.gguf` base 行继续作为部署 sanity baseline;
- 3B/active≤3B vs 30B+ 的真实结论仍需要 30B+ endpoint 或本机 30B+ 权重加入同一评测路径。
