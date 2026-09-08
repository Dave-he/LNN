# 2026-09-02 LNN pipeline run notes

## Summary

| 阶段 | 状态 | 备注 |
|---|---|---|
| 0. git fetch | FAILED | LAN proxy `192.168.6.25:7890` 间歇性吐 `Ncat: Error reading proxy response Status-Line`；DNS 也被劫持到 `198.18.0.x`（非路由段）→ 5 次 retry 全失败 |
| 1. digest | OK | 25 arXiv / 40 GitHub / 22 HF；`docs/daily/2026-09-02_LNN_research_digest.md` 生成 + 本地 commit `a12e3d6` |
| 2. select | 1 candidate | 12 篇 arXiv 中 10 篇已被 `docs/reports/` 覆盖；剩 2 篇中 1 篇 score=0 跳过（`2607.08283v3` TFP 已在历史报告中），最终剩 `2608.28702v1` (Stochastic Liquid Deformation Fields, score=21) |
| 3. report | OK | `paper-analyzer` 技能仍缺失，按 `2026-07-11` / `2026-06-24` 兜底路径：arXiv API 拿摘要 + curl 拉 PDF 2.1MB + `uv run --no-project --with pypdf` 解出 465 行文本 → 按 AGENTS.md SOP 生成独立报告 `Stochastic_Liquid_Deformation_Fields_SDE_CfC_2608.28702_研读报告.md`（含元数据/核心问题/方法论/3 条核心公式/D-NeRF+NeRF-DS 完整结果表/noise-level sweep/局限/复现建议），同步 append 到 `docs/LNN_深度研读报告.md` |
| 4. commit | partial | 本地 3 个 commit（`a12e3d6` digest / `2cec384` 研读报告 / `64c3dfb` 全局索引），**push 全部被 LAN proxy 拦下** |
| 5. reproduce | failed | 命中 `2606.26849v1` (Liquid Fusion for Salient Object Detection) → `scripts/experiment_imitation_lnn.py` → `ModuleNotFoundError: No module named 'torch'`（与昨日相同问题：`replicate_paper_dispatch.py:77` 硬编码 `python3` 而非 `.venv312/bin/python`）。失败已写 `logs/pipeline/2026-09-02_reproduce.log` |
| 6. archive | n/a | `analysis/replication/` 无新产物 |

## SDE-LNN (arXiv 2608.28702) 摘要

- 承接 [[Liquid_Neural_Networks_3DGS_Deformation_Field_2606.07670_研读报告|2606.07670]] (deterministic CfC drop-in)，把 CfC 的 closed-form gate pre-activation 视为 one-step Itô 过程，加 Gaussian noise $\lambda \varepsilon$；$\lambda = 0$ 时严格退化为 standard CfC，**完全共享参数 / FLOPs / 推理路径**
- D-NeRF 8 场景 6/8 最佳 PSNR；Hook +1.24 dB 单点最大；NeRF-DS 7 场景均值持平（23.73 vs 23.82 dB），其中 Bell/Press 略好
- Noise-level sweep：λ=0.05 是 "D-NeRF 上 noise 仍可接受的上界"，NeRF-DS 上 λ=0 已是最佳
- 立场：**机制清晰化 + 诚实负结果**——区别于 2606.07670 的 "新 SOTA" 立场

## Recommended fix (when proxy restored)

1. `git push origin HEAD` 把 `64c3dfb` / `2cec384` / `a12e3d6` 推上去
2. 修 `scripts/replicate_paper_dispatch.py:77` 改用 `.venv312/bin/python`（详情见 `analysis/pipeline_notes/2026-09-01_proxy_down.md`），下次 reproduce 不会因 torch 缺失失败
3. 如需 LAN proxy 修复：Clash/Xray 进程或 `~/.ssh/config` 的 `ProxyCommand ncat --proxy 192.168.6.25:7890 %h %p` 应能让 ncat 透传 SSH 到 github.com:22

## Files touched today

- `docs/daily/2026-09-02_LNN_research_digest.md` (new, committed)
- `papers/daily/2026-09-02_lnn_research.json` (new, committed)
- `papers/daily/2608.28702v1.pdf` (new, committed, 2.1MB)
- `analysis/repo_watchlist/2026-09-02_lnn_open_source_watchlist.md` (new, untracked)
- `docs/reports/Stochastic_Liquid_Deformation_Fields_SDE_CfC_2608.28702_研读报告.md` (new, committed)
- `docs/LNN_深度研读报告.md` (modified, committed)
- `logs/pipeline/2026-09-02_pipeline.log` (new)
- `logs/pipeline/2026-09-02_reproduce.log` (new)
- `analysis/pipeline_notes/2026-09-02_proxy_down.md` (this file, new)