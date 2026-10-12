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

## 缺陷 3：`write_report` 状态白名单漏 `ok_cpu`，结果表被吞（已修）

缺陷 1 的修复（`mark_cpu_only_run()`）在 04:33 产出 `status: "ok_cpu"` 的当天**没有被自己覆盖到**：
`write_report()` 用 `payload["status"] not in {"ok", "ok_cpu_fallback"}` 决定要不要渲染结果段，
而新引入的 `ok_cpu` 不在这个集合里。于是 10-12 报告只剩：

```text
## 状态
- ok_cpu: None
```

6 个模型的结果表、Pareto 解读、以及已经写到磁盘的 `2026-10-12_lnn_benchmark.png` 链接**全部消失**，
而同目录的 `.json` 里 `results` 有完整 6 条。这是典型的"状态枚举加了一处、判消费方没跟"的回归 ——
JSON 完好所以从数据层面看不出来，只有读 md 才会发现。

**修复**：抽出 `SUCCESS_STATUSES = frozenset({"ok", "ok_cpu", "ok_cpu_fallback"})` 常量与
`mark_cpu_only_run()` 并排放置，`write_report` 改为引用它；并给 `ok_cpu` 补一段
"⚠️ CPU-only 运行"警告，把 `status_note` 原文写进 md（此前只在 JSON 里）。回归测试 +5 项
（`TestSuccessStatusesCoverEveryProducibleStatus` 钉住状态集合、`TestOkCpuReportRendersResults`
端到端断言 md 里有结果行且没有 `## 状态`）。

**方法论教训**：诚实状态重命名这类改动必须同时审计**消费该状态的判据**。
`status` 从 `"ok"` 变成 `"ok_cpu"` 是给 JSON 的诚实性改进，但 `write_report` 用同一个字段做分支，
于是同一个 commit 同时"让 JSON 更诚实"和"让 md 少信息"。加新枚举值时要 grep 出所有比较点。

**旁证**：`analysis/jetson/2026-06-09_test_{quick,single}_lnn_benchmark.md` 曾显示为未提交修改，
症状相同（`ok_cpu: None` + 结果表消失）。它们是 `tests/test_jetson_lnn_benchmark.py` 的副作用产物
（run_date 硬编码 `2026-06-09_test_*`），不是真实跑损坏的存档；已 `git checkout` 回滚并保持原样。

## 环境噪音：本机 pytest 全面收集失败（非本次改动引入）

`python -m pytest tests/` 报 **229 errors during collection**，两类根因都是环境问题：

1. `ImportError: libcudss.so.0` —— 与缺陷 1 同源，`/usr/bin/python3.10` 的 torch cu130 wheel 缺库。
   凡是在模块顶层 `import torch` 的测试文件（如 `tests/test_core.py`）都无法收集。
2. `ModuleNotFoundError: No module named 'lnn'` —— 本机未做 `pip install -e .`。

另有一个**全局插件**在启动期就炸：用户级 site-packages 的 `deepeval` pytest 插件
经 `eventlet` → `pyOpenSSL`（`AttributeError: module 'lib' has no attribute 'GEN_EMAIL'`）
导入失败，导致 `python3 -m pytest` **任何文件都收集 0 条**，且**退出码为 0**。
加 `-p no:deepeval` 可绕过。

**风险**：这会让 CI 式的"跑一遍 pytest 看是否全绿"在本机静默变成 no-op。
本次验证 Jetson 相关测试时用 `python3 -m unittest`（不加载插件）与
`pytest -p no:deepeval` 双路跑：unittest 15/15、pytest 7/7 通过。
`test_jetson_lnn_benchmark.py` / `test_jetson_benchmark_failure_handling.py` 不在
`.github/workflows/lnn_weekly_verify.yml` 的 pytest 列表里（该 CI 只跑 core + paper_models +
liquid_tad_hierarchical），CI 用 Python 3.11 且装 CPU torch，所以本地收集失败不影响 CI。

## 今日研究产出

- digest：`docs/daily/2026-10-12_LNN_research_digest.md`（24 篇 / 41 仓库 / 18 模型，`errors: []`）
- 研读报告：**0 篇**，selector 判定 `n_skipped_reported: 12/12` —— LNN / CfC / LTC / NCP 主题
  持续饱和，最新一篇为 ContiLNN (arXiv:2610.12337, 2026-10-08)，已覆盖。
- Jetson benchmark：`analysis/jetson/2026-10-12_lnn_benchmark.{json,md,png}`（本次审计补齐，
  `status: "ok_cpu"`，md 结果表经缺陷 3 修复后已回填 6 个模型）。
- 两份 digest 快照的 HF 模型列表不同（`Rcarvalo/lfm2a-fr-v3-checkpoints` vs
  `mradermacher/LFM2.5-VL-1.6B-...`），系不同时刻抓取所致，非抓取失败。

## 未解决 / 下次会话接手

1. **本机无可用 CUDA**：需要为 Jetson R36 装匹配的 torch wheel（cu126/cu124 匹配 JetPack 6）
   或升级驱动，才能拿到真正的 GPU 步/秒数据。在此之前 benchmark 只会报 `ok_cpu`。
2. 手工跑 `run_lnn_research_pipeline.sh` 会与 systemd timer 形成当日第二次抓取与提交。
   建议：同日已跑过 timer 时不再手工重跑，或给 pipeline 脚本加"当日 digest 已存在则跳过抓取"守卫。
3. `analysis/pipeline_notes/` 未被 `git add docs papers analysis` 覆盖时会遗漏 —— 当前两个入口
   都已含 `analysis`，但若新增其它产物目录需同步检查。
4. **本机 `python3 -m pytest` 静默 no-op**：全局 `deepeval` 插件导入失败导致收集 0 条但退出码为 0。
   任何"跑 pytest 验证"的本地检查都必须加 `-p no:deepeval`（或改用 `python3 -m unittest`），
   否则会把"没跑"误读成"全绿"。另需 `pip install -e .` 才能收集依赖 `lnn` 包的测试。
