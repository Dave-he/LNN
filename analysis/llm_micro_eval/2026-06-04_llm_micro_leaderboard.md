---
title: LLM micro-eval leaderboard
date: 2026-06-04
tags: [LFM2.5, LNN, LLM, local-eval, micro-benchmark, leaderboard]
parent: [[PRD_LNN_Edge_Research]]
---

# LLM micro-eval leaderboard - 2026-06-04

## Summary

- Scanned: `analysis/llm_micro_eval/*_micro_eval.json`
- Entries: **4** total, **4** rankable
- Current leader: `lfm25_1.2b_instruct_q4` (100.0%, 16.843 tok/s)

## Leaderboard

| Rank | Model | Backend | Role | Accuracy | Tasks | Mean tok/s | Source |
|---:|---|---|---|---:|---:|---:|---|
| 1 | `lfm25_1.2b_instruct_q4` | `llama-cli` | `under_3b_candidate` | 100.0% | 7/7 | 16.843 | [md](2026-06-04_lfm25_1_2b_instruct_q4_micro_eval.md) |
| 2 | `lfm25_1.2b_instruct_q4_http` | `openai-chat` | `under_3b_candidate` | 100.0% | 7/7 | 5.707 | [md](2026-06-04_lfm25_1_2b_instruct_q4_http_micro_eval.md) |
| 3 | `lfm25_dpo_s1_q4` | `llama-cli` | `under_3b_candidate` | 57.1% | 4/7 | 11.371 | [md](2026-06-04_lfm25_dpo_s1_q4_micro_eval.md) |
| 4 | `lfm25_dpo_s1_f16` | `llama-cli` | `under_3b_candidate` | 14.3% | 1/7 | 4.257 | [md](2026-06-04_lfm25_dpo_s1_f16_micro_eval.md) |

## Category Split

| Model | arithmetic | instruction | structured_output | abstention |
|---|---:|---:|---:|---:|
| `lfm25_1.2b_instruct_q4` | 3/3 | 2/2 | 1/1 | 1/1 |
| `lfm25_1.2b_instruct_q4_http` | 3/3 | 2/2 | 1/1 | 1/1 |
| `lfm25_dpo_s1_q4` | 3/3 | 0/2 | 1/1 | 0/1 |
| `lfm25_dpo_s1_f16` | 1/3 | 0/2 | 0/1 | 0/1 |

## Interpretation

- Ranking order is accuracy, then task coverage, then mean generation speed.
- This is a local deployment sanity leaderboard, not a public benchmark.
- Rows with different task signatures are useful for smoke checks but should not be used for dominance claims.
- A real 30B+ comparison requires at least one `30b_plus_baseline` row from a live endpoint or local 30B+ runtime.
