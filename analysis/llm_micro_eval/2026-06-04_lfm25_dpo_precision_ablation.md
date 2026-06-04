---
title: LFM2.5 DPO precision ablation
date: 2026-06-04
tags: [LFM2.5, DPO, GGUF, quantization, micro-benchmark, candidate-selection]
parent: [[PRD_LNN_Edge_Research]]
---

# LFM2.5 DPO precision ablation - 2026-06-04

## Question

`LFM25-DPO-Q4_0.gguf` failed the 7-task micro-eval at 4/7. This check asks
whether that failure is caused by Q4 quantization or by the DPO branch /
prompt-template behavior itself.

## Inputs

| Model | File | Size | Backend |
|---|---|---:|---|
| DPO Q4 | `models/lfm25-dpo-s1/LFM25-DPO-Q4_0.gguf` | 695,750,560 bytes | `llama-cli` |
| DPO F16 | `models/lfm25-dpo-s1/LFM25-DPO-F16.gguf` | 2,343,325,600 bytes | `llama-cli` |

## Results

| Model | Accuracy | Arithmetic | Instruction | JSON | Abstention | Mean tok/s | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| DPO Q4 | 57.1% | 3/3 | 0/2 | 1/1 | 0/1 | 11.371 | [micro-eval](2026-06-04_lfm25_dpo_s1_q4_micro_eval.md) |
| DPO F16 | 14.3% | 1/3 | 0/2 | 0/1 | 0/1 | 4.257 | [micro-eval](2026-06-04_lfm25_dpo_s1_f16_micro_eval.md) |

## Readout

- F16 does **not** recover the Q4 failures; it performs worse on this sanity set.
- The DPO branch fails exact instruction following in both precision variants.
- The failure is unlikely to be explained by Q4 quantization alone.
- The DPO branch should not be used as the current 3B/under-3B candidate for 30B+ comparison until chat template, prompt format, and DPO data target are audited.

## Candidate Decision

Use `LFM2.5-1.2B-Instruct-Q4_0.gguf` as the current local sanity baseline.
Treat both DPO GGUF variants as regression evidence, not as leaderboard
candidates for a serious 30B+ comparison.
