---
title: 2026-06-04 Loop iteration 29 - LLM micro-eval leaderboard
date: 2026-06-04
tags: [LNN, LFM2.5, LLM, leaderboard, micro-benchmark, 30B-plus, claim-audit]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 29 - LLM micro-eval leaderboard

> iter#28 已经让同一套 micro-eval 可以接 OpenAI-compatible 30B+ endpoint。
> 本轮把多个 micro-eval JSON 合并成可排序榜单,为后续真实 30B+ 对照留出统一数据面。

## 1. 新能力

`scripts/build_llm_micro_leaderboard.py`

功能:

- 扫描 `analysis/llm_micro_eval/*_micro_eval.json`;
- 读取 `run_llm_micro_eval.py` 生成的统一 JSON schema;
- 自动标记模型角色:
  - `under_3b_candidate`;
  - `active_under_3b_moe_candidate`;
  - `30b_plus_baseline`;
  - `unknown`;
- 按 accuracy -> task coverage -> mean generation tok/s 排序;
- 输出 JSON + Markdown 榜单。

## 2. 当前榜单

输出:

- `analysis/llm_micro_eval/2026-06-04_llm_micro_leaderboard.json`
- `analysis/llm_micro_eval/2026-06-04_llm_micro_leaderboard.md`

当前扫描结果:

- entries: 1
- rankable: 1
- current leader: `lfm25_1.2b_instruct_q4`
- role: `under_3b_candidate`
- accuracy: 7/7 = 100.0%
- mean generation: 16.843 tok/s

重要限制:

- 当前榜单还没有真实 `30b_plus_baseline` 行;
- 因此它是“本机小模型部署 sanity 榜单”,不是 3B 已吊打 30B+ 的证据;
- 真正对照需要运行 `run_llm_micro_eval.py --backend openai-chat --openai-model <30B+>` 后再重建榜单。

## 3. 文档接入

- `README.md` Quick Paths 新增 micro leaderboard 入口;
- `docs/PRD_LNN_Edge_Research.md` §4 将 leaderboard 纳入 LFM/LLM 打榜证据卡模块;
- `docs/PRD_LNN_Edge_Research.md` §10 新增 `10-14` 完成项。

## 4. 验证

```bash
python -m pytest tests/test_llm_micro_leaderboard.py
python scripts/build_llm_micro_leaderboard.py --no-write --json
python scripts/build_llm_micro_leaderboard.py --json
```

结果:

- `tests/test_llm_micro_leaderboard.py`: 4 passed
- `build_llm_micro_leaderboard.py --no-write --json`: OK
- `build_llm_micro_leaderboard.py --json`: OK,输出已写入 `analysis/llm_micro_eval/`

## 5. 下一步

1. 启动真实 30B+ endpoint,例如 Qwen3-30B-A3B 的 vLLM/SGLang/llama-server 服务;
2. 运行同一套 `run_llm_micro_eval.py --backend openai-chat`;
3. 重新运行 `build_llm_micro_leaderboard.py`;
4. 再接 lm-eval/OpenCompass 子集,把 smoke leaderboard 升级成公开榜可审计证据。
