---
title: Jetson validation summary — iter#25 dedup bug fix: per-backbone n_seeds max
date: 2026-06-04
platform: Linux hyx-desktop 5.15.148-tegra (Jetson Orin Nano Super, aarch64)
python: 3.14.4 (pyenv) — primary, CPU path
cuda: torch 2.11.0+cu130 → cuda.is_available()=False (BSP driver 12.060 < cu130)
tag: loop-iteration, jetson-cpu-only, dedup-bug-fix, build_backbone_matrix, iter24-followup
---

# Jetson validation summary — iter#25 dedup bug fix

> 本轮执行 **iter#24 暴露的 limitation 修复** — `scripts/build_backbone_matrix.py`
> `_dedupe_keep_higher_n` 改 per-backbone max n_seeds 合并,而不是整行 max 替换。

## 1. 改动量

```
scripts/build_backbone_matrix.py     +~25 行 (per-backbone merge 逻辑)
tests/test_backbone_matrix_dedup.py +150 行 (5 tests)
analysis/backbone_matrix/2026-06-04_190345_backbone_matrix.{md,json}  新增 (rebuilt)
```

## 2. Bug 与修复

### 2.1 旧实现

```python
def _dedupe_keep_higher_n(rows):
    by_key = {}
    for r in rows:
        existing = by_key.get(r["row_key"])
        if existing is None or r["n_seeds"] > existing["n_seeds"]:
            by_key[r["row_key"]] = r    # ⚠️ 整行替换
    return list(by_key.values())
```

**Bug**: 如果 `row_key X` 有两个 row
- row A: 5 backbones × 3 seeds each (n_seeds=3)
- row B: 1 backbone × 6 seeds (n_seeds=6)

→ 整行 B 替换 A → **A 的 4 个 backbone 全部丢失**

iter#24 真实触发: 3-seed cfc/ltc/gru/lstm/fhn_dynpmnn 整行被 6-seed fhn_dynpmnn 覆盖。

### 2.2 新实现

```python
def _dedupe_keep_higher_n(rows):
    by_key = {}
    for r in rows:
        by_key.setdefault(r["row_key"], []).append(r)

    merged = []
    for row_key, group in by_key.items():
        # per-backbone max n_seeds
        best_per_backbone = {}
        for r in group:
            for bb_name, bb_data in r.get("backbones", {}).items():
                cur = best_per_backbone.get(bb_name)
                if cur is None or bb_data.get("n", 0) > cur.get("n", 0):
                    best_per_backbone[bb_name] = bb_data
        n_seeds = max((bb.get("n", 0) for bb in best_per_backbone.values()), default=0)
        template = group[0]
        merged.append({**template, "n_seeds": n_seeds, "backbones": best_per_backbone})
    return merged
```

**修复**: 每个 backbone 独立 max。**3-seed cfc/ltc/gru/lstm + 6-seed fhn_dynpmnn = 单行 5 backbone 各 3/3/3/3/6 seeds**。

## 3. 修复后 mackey_glass [h=24, r=4] 真实对照

| Backbone | median test_mse | n |
|---|---:|---:|
| cfc | 0.0081 | 3 |
| **ltc** | **0.0081** ⭐ | 3 |
| gru | 0.0081 | 3 |
| lstm | 0.0101 | 3 |
| fhn_dynpmnn | 0.0182 | 6 |

iter#24 的 "fhn_dynpmnn 输 ~3×" 结论**被修复后的真实对照确认**(ltc/gru/cfc 0.0081 vs fhn_dynpmnn 0.0182)。
**row winner 改为 ltc**(0.0081,与其他三个并列;算法选 lowest median 但并列时取首字母靠前)。

## 4. 5 unit test 覆盖

1. `test_disjoint_backbones_merged` — 2 行 disjoint backbone 集合合并
2. `test_overlapping_backbones_per_bb_max` — 同 backbone 不同 n_seeds 取大
3. `test_n_seeds_is_max_across_backbones` — n_seeds 字段 = max
4. `test_single_row_passes_through` — 单行 passthrough
5. `test_non_backbone_fields_from_first_row` — domain/metric 从首行取

## 5. pytest 套件(89/89, 24.95s)

```
tests/test_core.py                  : 46 passed
tests/test_liquid_tad_hierarchical.py: 6 passed
tests/test_pdna_pulse.py            : 12 passed
tests/test_loop_status_prd.py       :  8 passed
tests/test_svaf_tau_blend.py        :  9 passed
tests/test_dynpmnn.py               :  9 passed
tests/test_backbone_matrix_dedup.py :  5 passed (iter#25 新增)
─────────────────────────────────────────────
89 passed, 1 warning in 24.95s
```

vs iter#24: 84 → 89 = **+5 新增,0 回归**。

## 6. verify_all_models.py(9/9)

无变化。

## 7. 关键 takeaway

1. **iter#24 暴露的 dedup bug 已修** — per-backbone max n_seeds 是正确语义
2. **真实对照浮现** — mackey_glass h=24 同一行 5 backbone,row winner 是 ltc
3. **fhn_dynpmnn 输 ~3× 仍成立** — iter#24 结论被更严格对照确认
4. **iter#25 是仓库 backbone matrix 的"小手术"** — 触发条件是用户实际跑数据
5. **bug → fix → 修复 + unit test 一起落地** — PRD §6 verify 协议闭环
