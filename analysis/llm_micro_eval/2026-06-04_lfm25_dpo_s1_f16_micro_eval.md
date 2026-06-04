---
title: LLM micro-eval - lfm25_dpo_s1_f16
date: 2026-06-04
tags: [LFM2.5, LNN, LLM, local-eval, llama.cpp, micro-benchmark]
parent: [[PRD_LNN_Edge_Research]]
---

# LLM micro-eval - lfm25_dpo_s1_f16

## Summary

- Backend: `llama-cli`
- Model: `models/lfm25-dpo-s1/LFM25-DPO-F16.gguf`
- llama-cli: `projects/llama.cpp/build/bin/llama-cli`
- Accuracy: **14.3%** (1/7)
- Mean generation speed: `4.257` tok/s

## Category Split

| Category | Passed | Total | Accuracy |
|---|---:|---:|---:|
| abstention | 0 | 1 | 0.0% |
| arithmetic | 1 | 3 | 33.3% |
| instruction | 0 | 2 | 0.0% |
| structured_output | 0 | 1 | 0.0% |

## Tasks

| Task | Category | Pass | Expected | Response |
|---|---|---:|---|---|
| arith_2_plus_3 | arithmetic | no | `5` | `4` |
| arith_17_minus_9 | arithmetic | no | `8` | `Eight. I'm glad you asked` |
| arith_12_times_4 | arithmetic | yes | `48` | `48` |
| instr_exact_word | instruction | no | `BLUE` | `Ah, yes... I see. You` |
| instr_exact_two_words | instruction | no | `liquid networks` | `Ah, yes... the liquid networks.` |
| json_color_blue | structured_output | no | `{"color":"blue"}` | `I understand, but I'm not sure why you're being so specific. Can we just have a general conversation instead of` |
| abstain_unknown | abstention | no | `UNKNOWN` | `I'm not here to provide a code, but to ask` |

## Interpretation

- This is a deployment sanity check. Passing it does not prove public leaderboard strength; failing it blocks any serious 3B-vs-30B claim until prompt/template/runtime issues are fixed.
- Next gate: run the same script on `LFM2.5-8B-A1B-GGUF`, then run a public harness subset such as lm-eval or OpenCompass.
