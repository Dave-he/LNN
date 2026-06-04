---
title: LLM micro-eval - lfm25_1.2b_instruct_q4_http
date: 2026-06-04
tags: [LFM2.5, LNN, LLM, local-eval, llama.cpp, micro-benchmark]
parent: [[PRD_LNN_Edge_Research]]
---

# LLM micro-eval - lfm25_1.2b_instruct_q4_http

## Summary

- Backend: `openai-chat`
- OpenAI model: `lfm25-1.2b-instruct-q4`
- Endpoint: `http://127.0.0.1:18080/v1`
- Accuracy: **100.0%** (7/7)
- Mean generation speed: `5.707` tok/s

## Category Split

| Category | Passed | Total | Accuracy |
|---|---:|---:|---:|
| abstention | 1 | 1 | 100.0% |
| arithmetic | 3 | 3 | 100.0% |
| instruction | 2 | 2 | 100.0% |
| structured_output | 1 | 1 | 100.0% |

## Tasks

| Task | Category | Pass | Expected | Response |
|---|---|---:|---|---|
| arith_2_plus_3 | arithmetic | yes | `5` | `5` |
| arith_17_minus_9 | arithmetic | yes | `8` | `8` |
| arith_12_times_4 | arithmetic | yes | `48` | `48` |
| instr_exact_word | instruction | yes | `BLUE` | `BLUE` |
| instr_exact_two_words | instruction | yes | `liquid networks` | `liquid networks` |
| json_color_blue | structured_output | yes | `{"color":"blue"}` | `{"color":"blue"}` |
| abstain_unknown | abstention | yes | `UNKNOWN` | `UNKNOWN` |

## Interpretation

- This is a deployment sanity check. Passing it does not prove public leaderboard strength; failing it blocks any serious 3B-vs-30B claim until prompt/template/runtime issues are fixed.
- Next gate: run the same script on `LFM2.5-8B-A1B-GGUF`, then run a public harness subset such as lm-eval or OpenCompass.
