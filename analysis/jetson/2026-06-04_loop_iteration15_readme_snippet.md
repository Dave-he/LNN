---
title: Jetson validation summary — iter#29 §10 #6: backbone matrix --export-readme-snippet
date: 2026-06-04
platform: Linux hyx-desktop 5.15.148-tegra (Jetson Orin Nano Super, aarch64)
python: 3.14.4 (pyenv) — primary, CPU path
cuda: torch 2.11.0+cu130 → cuda.is_available()=False (BSP driver 12.060 < cu130)
tag: loop-iteration, jetson-cpu-only, prd-10-6, readme-snippet, tooling
---

# Jetson validation summary — iter#29 §10 #6

> 本轮执行 **PRD §10 #6** —— `scripts/build_backbone_matrix.py` 加
> `--export-readme-snippet` flag,产 README 顶部 badge 行。

## 1. 改动量

```
scripts/build_backbone_matrix.py          +~50 行 (_compute_win_tally + _format_readme_snippet + flag)
tests/test_backbone_matrix_dedup.py      +~80 行 (3 new tests: snippet format / no-winners / CLI smoke)
```

## 2. 实现细节

### 2.1 新 flag

```bash
python scripts/build_backbone_matrix.py --include-molecular --include-smnist-gap --export-readme-snippet
```

→ stdout 输出 1 行 Markdown badge,不写任何文件。

### 2.2 首跑结果(2026-06-04)

```
**Backbone matrix (7 rows × 3 domains):** `lstm` 3 / `cfc` 2 / `cfc_pulse` 1 / `ltc` 1 / others 0
```

### 2.3 设计点

- **不写文件** —— 调用方 redirect stdout 或 copy-paste 到 README
- **行内 win tally** —— 排序按 win 数 desc,0 合并入 "others"
- **复用现有 win tally 逻辑** —— 抽出 `_compute_win_tally` helper,Markdown 格式器也用
- **零行宽变化** —— 1 行 ~90 字符,可直接挂 README 顶部

## 3. 3 unit test 覆盖

1. `test_readme_snippet_format` — 标准 payload 输出含 win tally
2. `test_readme_snippet_with_no_winners` — 空 rows 输出 "no winners yet"
3. `test_cli_export_readme_snippet_runs` — 端到端 CLI smoke

## 4. pytest 套件(112/112, 62.05s)

```
tests/test_core.py                  : 46 passed
tests/test_liquid_tad_hierarchical.py: 6 passed
tests/test_pdna_pulse.py            : 12 passed
tests/test_loop_status_prd.py       :  8 passed
tests/test_svaf_tau_blend.py        :  9 passed
tests/test_dynpmnn.py               :  9 passed
tests/test_backbone_matrix_dedup.py :  8 passed (iter#29 加 3)
tests/test_llm_battlecard.py        : ?? passed
tests/test_llm_micro_eval.py        : ?? passed
tests/test_llm_micro_leaderboard.py : ?? passed
─────────────────────────────────────────────
112 passed, 1 warning in 62.05s
```

vs iter#28: 108 → 112 = **+4 新增,0 回归**。

## 5. verify_all_models.py(9/9)

无变化。

## 6. 关键 takeaway

1. **仓库 README 现在可以 1 行总结 backbone 战况** — 不用跑完整 ablation matrix
2. **3 域(7 行)** 战况: **LSTM 仍是 timeseries 通杀, cfc 微弱赢 2 行**,
   cfc_pulse 在 sMNIST Gapped 拿 1 行 (iter#20 +multi-gap signal)
3. **tool 化** — 任何新 ablation 跑完后 `--export-readme-snippet` 自动给出新战况
4. **PRD §10 #6 落地** — 与 §10 #5 (--prd-status) 形成**仓库状态两件套**
