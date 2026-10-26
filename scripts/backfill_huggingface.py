#!/usr/bin/env python3
"""补抓某一天的 Hugging Face 候选模型, 修复抓取失败留下的降级产物.

为什么需要这个工具
------------------
`daily_lnn_research.py` 早期对 HF 查询**没有重试**, 单次
`Connection reset by peer` 就让整轮 5 条查询全灭, 候选池清零。
更糟的是渲染层把 "0 个" 写成"本次未发现可记录的 Hugging Face 模型"——
**把抓取失败伪装成领域事实** (违反 AGENTS.md 约束 #7 的 (a)/(b) 二分)。
实测中招的有 2026-10-24 (已带 models=0 进了 origin) 与 2026-10-26 04:38
(当日 06:31 重跑才修好)。

为什么不能直接重跑 `--date` 那一天
---------------------------------
重跑整轮会**重新抓 arXiv / GitHub**。arXiv 的 `/api/query` 是按当前时间窗
返回的, 今天重跑 10-24 会把 10-25/10-26 的新论文灌进 10-24 的历史池 ——
历史 archive 会漂移。所以本工具**只补 HF**: 其余字段原样保留, 并用脚本里
同一套渲染函数重生成 digest / watchlist, 保证三者一致。

用法
----
    python3 scripts/backfill_huggingface.py --date 2026-10-24
    python3 scripts/backfill_huggingface.py --date 2026-10-24 --dry-run

退出码: 0=补抓成功并已写回; 1=补抓仍失败 (JSON/MD 保留失败痕迹, 并标注降级);
        2=参数或文件问题。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import daily_lnn_research as dlr  # noqa: E402


def is_hf_error(error: str) -> bool:
    lowered = error.lower()
    return "hugging face" in lowered or lowered.startswith("hf ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Run date in YYYY-MM-DD format.")
    parser.add_argument(
        "--per-query",
        type=int,
        default=8,
        help="Maximum Hugging Face results per query (matches the daily run default).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Fetch and report, but do not write files.")
    args = parser.parse_args()

    json_path = dlr.ROOT / "papers" / "daily" / f"{args.date}_lnn_research.json"
    if not json_path.exists():
        print(f"[error] no such daily payload: {json_path}", file=sys.stderr)
        return 2

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    errors: list[str] = list(payload.get("errors") or [])
    hf_errors_before = [e for e in errors if is_hf_error(e)]
    other_errors = [e for e in errors if not is_hf_error(e)]
    models_before = payload.get("huggingface_models") or []

    print(
        f"[info] {args.date}: HF models before={len(models_before)}, "
        f"HF errors before={len(hf_errors_before)}, other errors={len(other_errors)}"
    )
    if not hf_errors_before and models_before:
        print("[info] payload has no HF failures and a non-empty pool; nothing to repair.")
        return 0

    fetch_errors: list[str] = []
    models = dlr.fetch_huggingface_models(args.per_query, fetch_errors)

    if fetch_errors or not models:
        print(
            f"[error] HF backfill still failing ({len(fetch_errors)} query errors, "
            f"{len(models)} models). Keeping the degraded payload.",
            file=sys.stderr,
        )
        for error in fetch_errors:
            print(f"        {error}", file=sys.stderr)
        # 仍然用新的渲染层重生成 MD —— 让"抓取失败"在摘要行里显式可见,
        # 而不是继续写着"本次未发现可记录的 Hugging Face 模型"。
        if not args.dry_run:
            digest_path = dlr.ROOT / "docs" / "daily" / f"{args.date}_LNN_research_digest.md"
            digest_path.write_text(
                dlr.render_daily_digest(
                    args.date,
                    payload.get("papers") or [],
                    payload.get("github_repos") or [],
                    models_before,
                    payload.get("downloaded_pdfs") or [],
                    other_errors + hf_errors_before,
                ),
                encoding="utf-8",
            )
            print(f"[info] re-rendered degraded digest with explicit failure markers: {digest_path}")
        return 1

    print(f"[info] HF backfill recovered {len(models)} models (was {len(models_before)}).")
    if args.dry_run:
        print("[info] --dry-run: not writing any file.")
        return 0

    payload["huggingface_models"] = models
    payload["errors"] = other_errors
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    digest_path = dlr.ROOT / "docs" / "daily" / f"{args.date}_LNN_research_digest.md"
    digest_path.write_text(
        dlr.render_daily_digest(
            args.date,
            payload.get("papers") or [],
            payload.get("github_repos") or [],
            models,
            payload.get("downloaded_pdfs") or [],
            other_errors,
        ),
        encoding="utf-8",
    )

    watchlist_path = dlr.ROOT / "analysis" / "repo_watchlist" / f"{args.date}_lnn_open_source_watchlist.md"
    watchlist_path.write_text(
        dlr.render_repo_watchlist(args.date, payload.get("github_repos") or [], models),
        encoding="utf-8",
    )

    # 索引按日期去重, 同一天重跑是幂等的 —— 计数会被这一轮的数字覆盖。
    dlr.update_indexes(args.date, digest_path, payload.get("papers") or [], payload.get("github_repos") or [], models)

    print(f"[info] updated: {json_path.relative_to(dlr.ROOT)}")
    print(f"[info] updated: {digest_path.relative_to(dlr.ROOT)}")
    print(f"[info] updated: {watchlist_path.relative_to(dlr.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
