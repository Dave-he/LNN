---
title: LFM/LNN-related 3B-vs-30B battlecard
date: 2026-06-04
tags: [LFM2.5, LNN, LLM, benchmark, battlecard, active-3B, 30B-plus]
parent: [[PRD_LNN_Edge_Research]]
---

# LFM/LNN-related battlecard - 2026-06-04

## Verdict

- Verdict: **active_under_3b_not_dense_3b**
- Readout: LFM2.5-8B-A1B beats Qwen3-30B-A3B-Thinking-2507 on 7 shared metrics and loses on 6, but it is 8.3B total / 1.5B active, so this supports only the active<=3B MoE thesis, not an exact 3B dense claim.
- Scope: candidate total<=3B = False; active<=3B = True; baseline total>=30B = True.

## Models

| Role | Model | Total params | Active params | Architecture | Source |
|---|---|---:|---:|---|---|
| Candidate | LFM2.5-8B-A1B | 8.30B | 1.50B | MoE + double-gated LIV convolution + GQA | [link](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) |
| 30B+ baseline | Qwen3-30B-A3B-Thinking-2507 | 30.50B | 3.30B | Transformer MoE | [link](https://huggingface.co/Qwen/Qwen3-30B-A3B) |

## Shared Public Benchmark Snapshot

| Metric | Candidate | Baseline | Delta | Winner |
|---|---:|---:|---:|---|
| AA-Omniscience Accuracy | 8.67 | 18.80 | -10.13 | baseline |
| AA-Omniscience Index | -24.70 | -51.31 | +26.61 | candidate |
| AA-Omniscience Non-Hallucination | 63.47 | 13.87 | +49.60 | candidate |
| AIME25 | 42.53 | 71.67 | -29.14 | baseline |
| AIME26 | 50.00 | 66.67 | -16.67 | baseline |
| BFCLv3 | 64.79 | 73.39 | -8.60 | baseline |
| BFCLv4 | 49.73 | 50.53 | -0.80 | baseline |
| IFBench | 56.47 | 51.11 | +5.36 | candidate |
| IFEval | 91.84 | 90.82 | +1.02 | candidate |
| MATH500 | 88.76 | 86.48 | +2.28 | candidate |
| Multi-IF | 79.93 | 79.04 | +0.89 | candidate |
| Tau2 Retail | 39.82 | 56.14 | -16.32 | baseline |
| Tau2 Telecom | 88.07 | 21.93 | +66.14 | candidate |

Shared metric tally: **7 win / 6 loss / 0 tie** (win rate 53.8%).

## Domain Split

| Group | Wins | Losses | Ties | Metrics |
|---|---:|---:|---:|---:|
| knowledge_instruction | 5 | 1 | 0 | 6 |
| math_agentic | 2 | 5 | 0 | 7 |

## Local Evidence

- Local validation file: `analysis/lfm25/2026-06-01_lfm25_local_validation.json`
- GGUF: status `ok`, generation 19.00 tok/s
- DPO: status `ok`, generation 1.75 tok/s
- Local micro-eval file: `analysis/llm_micro_eval/2026-06-04_lfm25_1_2b_instruct_q4_micro_eval.json`
- Micro-eval: 100.0% (7/7), mean generation 16.843 tok/s

## Prediction

- Near-term target should be **agentic/RAG/tool-use and instruction-following**, where the current LFM-family public data is strongest.
- Do **not** claim a general 3B model can beat 30B+ models yet: the strongest evidence here is active<=3B MoE, with clear losses on AIME and parts of BFCL/Tau Retail.
- Next evidence gate: run local LFM2.5-8B-A1B GGUF on Jetson/desktop, then add a reproducible lm-eval or OpenCompass subset before any public leaderboard claim.

## Sources

- [LiquidAI LFM2.5-8B-A1B model card](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) - LFM2.5-8B-A1B metrics and architecture snapshot
- [LiquidAI LFM2.5-8B-A1B blog](https://www.liquid.ai/blog/lfm2-5-8b-a1b) - release date, benchmark interpretation, inference support
- [Qwen3-30B-A3B model card](https://huggingface.co/Qwen/Qwen3-30B-A3B) - 30.5B total / 3.3B active parameter metadata
- [LFM2.5-1.2B local validation](analysis/lfm25/2026-06-01_lfm25_local_inference_quantization.md) - local GGUF/DPO smoke inference evidence already in this repo
- [Local LLM micro-eval](analysis/llm_micro_eval/2026-06-04_lfm25_1_2b_instruct_q4_micro_eval.md) - deterministic local deployment sanity check
