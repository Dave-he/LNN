# 2026-10-12 LNN 自动化执行情况审计

## 结论摘要

| 检查项 | 状态 | 证据 |
|---|---|---|
| systemd timer 调度 | ✅ 健康 | `enabled` / `active (waiting)`，下次触发 2026-10-13 04:37:45 CST |
| crontab 双调度残留 | ✅ 已禁用 | `crontab -l` 仅剩 2026-09-01 的禁用注释行 |
| digest 产出连续性 | ✅ 10/10 天 | 10-03 ~ 10-12 每日 `arXiv 候选论文：24 篇`，无缺口 |
| daily service 退出码 | ✅ exit 0 | 10-12 04:33:21 `Finished ...`，push `cf54df2..58bc3d6` 成功 |
| Jetson benchmark 产物 | ❌ 连续 4 天缺失 | 10-09 / 10-10 / 10-11 / 10-12 无 `analysis/jetson/` 产物 |
| 独立研读报告 | ⚠️ 主题饱和（预期） | selector 输出 `candidates: []`，`n_skipped_reported: 12/12` |

## 缺陷 1：Jetson benchmark 静默失败 4 天（已修）

**现象**：10-09 至 10-12 每天 04:33 日志都出现同一 traceback，且**没有**任何 `analysis/jetson/<date>_lnn_benchmark.*` 产物。

```
ImportError: libcudss.so.0: cannot open shared object file: No such file or directory
  File "scripts/jetson_lnn_benchmark.py", line 571, in run_benchmark
    import torch
```

**根因**（不是 torch 缺失，也不是显存不足）：

1. `run_benchmark()` 第一行 `import torch` 抛的是 `ImportError`，**不是** `ModuleNotFoundError`，
   因此 `main()` 里已有的 "skipped" 分支接不住它。
2. 落到通用 `except Exception`，但 `looks_like_cuda_runtime_error()`（`jetson_lnn_benchmark.py:883`）
   要求文本同时含 `"cuda"` 与 `"cudnn"/"cublas"/...`。异常文本是 `libcudss.so.0: cannot open
   shared object file` —— 含 `cudss` 但**不含 `cuda` 子串**（`libcudss` ≠ `cuda`），
   谓词返回 `False` → 直接 `raise`。
3. `write_report()` 在 `raise` 之后，永远不执行 → 当天没有任何可审计产物。

**为什么 CPU fallback 也没救回来**：`run_daily_lnn_task.sh` 的 fallback 是用**同一个**
`PYTHON_BIN` 加 `--cpu` 重跑，同一个解释器 `import torch` 会再次以同样方式失败。

**修复**：新增 `_is_torch_load_failure()`（`scripts/jetson_lnn_benchmark.py`），
识别 dlopen 类 `ImportError`（`shared object file` / `cannot load` / `undefined symbol`）
且 torch 确实已安装（用 `importlib.util.find_spec("torch")`，因为异常消息本身不含 "torch"，
只有 traceback 含），产出 `status: "skipped"` 报告而不是崩溃。
回归测试：`tests/test_jetson_benchmark_failure_handling.py`（7 项，含一条**故意断言**
`looks_like_cuda_runtime_error("libcudss.so.0")` 为 False 的用例，把本缺陷的成因钉住）。

**遗留 / 根因定位（本次审计已完成）**：`libcudss.so.0` 并非偶发，也**不是** 2026-09-29
记录的"系统 RAM 抖动"。实测两个解释器：

```
/usr/bin/python3.10                          → ImportError: libcudss.so.0 (每次必现)
/home/hyx/.pyenv/versions/3.14.4/bin/python3 → torch 2.11.0+cu130 可 import, cuda False
                                               (driver 12060 vs torch cu130 → 太旧)
```

`libcudss.so.0` 不在 `ldconfig -p` 缓存中。timer 的 `PYTHON_BIN=/usr/bin/python3.10`
其 torch wheel 期望一个本机 R36 未提供的 cuDSS 库 —— 是 **wheel 与系统库不匹配**，
所以 10-09~10-12 每天必然失败，而非时好时坏。

**修复 3（解释器选择）**：`run_daily_lnn_task.sh` 新增 `resolve_bench_python()`，
为 benchmark 单独挑选第一个 `import torch` 能成功的解释器（digest 仍用 `PYTHON_BIN`，
它只需标准库，一直正常）。在 unit / `install_daily_lnn_timer.sh` 中显式固定
`PYTHON_BENCH_BIN=/home/hyx/.pyenv/versions/3.14.4/bin/python3`，并已 `daemon-reload`。

**修复 4（诚实 status）**：新增 `mark_cpu_only_run()` —— CUDA 不可用时不再报裸 `"ok"`
（今天的产物原本就是 `status: "ok"` + `CUDA: False`，极易被当成 GPU 证据），改为
`"ok_cpu"` 并附 `status_note` 明示"非 GPU/边缘性能证据"。已有的 `"ok_cpu_fallback"`
保持原样（它已携带 `cuda_fallback_reason`）。回归测试扩到 10 项。

**边界（不变）**：CPU 下的步/秒数据**不得**当作 Jetson 边缘性能证据。修复 3 恢复的是
"每天都有一份可审计产物"，**不是**恢复 GPU 测量 —— 这台机器目前两个 torch 都没有可用
CUDA（一个缺库，一个 driver 太旧）。

## 缺陷 2：`run_lnn_research_pipeline.sh` 漏 add `analysis/`（已修）

10-12 告警项 (2) 的根因已定位：`run_daily_lnn_task.sh:162` 是 `git add docs papers analysis`（正确），
但**另一个入口** `scripts/run_lnn_research_pipeline.sh:133` 是 `git add docs papers`（漏 `analysis`），
而该脚本第 129 行的注释本就写着 "docs/ papers/ analysis/ 的当日变更" —— 代码与注释自相矛盾。

当日两次提交的差异印证了双入口：

| 时间 (CST) | 入口 | commit message | `git add` |
|---|---|---|---|
| 04:33 | systemd timer → `run_daily_lnn_task.sh` | `chore(daily): update LNN research digest 2026-10-12` | `docs papers analysis` ✅ |
| 06:32 | 手工 → `run_lnn_research_pipeline.sh` | `chore(daily): LNN digest + 研读报告 2026-10-12` | `docs papers` ❌ |

06:32 那次把 `analysis/repo_watchlist/2026-10-12_lnn_open_source_watchlist.md` 落在工作区未提交，
只能手工补 `7a923b1`。已把 `run_lnn_research_pipeline.sh:132-133` 改为与注释一致。

## 今日研究产出

- digest：`docs/daily/2026-10-12_LNN_research_digest.md`（24 篇 / 41 仓库 / 18 模型，`errors: []`）
- 研读报告：**0 篇**，selector 判定 `n_skipped_reported: 12/12` —— LNN / CfC / LTC / NCP 主题
  持续饱和，最新一篇为 ContiLNN (arXiv:2610.12337, 2026-10-08)，已覆盖。
- Jetson benchmark：`analysis/jetson/2026-10-12_lnn_benchmark.{json,md}`（本次审计补齐，
  `status: "ok_cpu"`）。
- 两份 digest 快照的 HF 模型列表不同（`Rcarvalo/lfm2a-fr-v3-checkpoints` vs
  `mradermacher/LFM2.5-VL-1.6B-...`），系不同时刻抓取所致，非抓取失败。

## 未解决 / 下次会话接手

1. **本机无可用 CUDA**：需要为 Jetson R36 装匹配的 torch wheel（cu126/cu124 匹配 JetPack 6）
   或升级驱动，才能拿到真正的 GPU 步/秒数据。在此之前 benchmark 只会报 `ok_cpu`。
2. 手工跑 `run_lnn_research_pipeline.sh` 会与 systemd timer 形成当日第二次抓取与提交。
   建议：同日已跑过 timer 时不再手工重跑，或给 pipeline 脚本加"当日 digest 已存在则跳过抓取"守卫。
3. `analysis/pipeline_notes/` 未被 `git add docs papers analysis` 覆盖时会遗漏 —— 当前两个入口
   都已含 `analysis`，但若新增其它产物目录需同步检查。
