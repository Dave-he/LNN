#!/usr/bin/env python3
"""Cross-task / cross-backbone Pareto matrix builder — PRD §9 #7.

Scans ``analysis/timeseries_ablation/*_lnn_vs_lstm.json`` (produced by
``scripts/ablation_lnn_vs_lstm_timeseries.py``) and pivots them into a
single ``task × backbone`` matrix.  This makes the "no universal
backbone winner" thesis (iter#6/7/9/10/11) mechanically reproducible
without re-running 32 trials by hand.

Rules:

* Each input JSON contributes one row keyed by its ``config.dataset``
  + a short config tag (warmup / N seeds / hidden) so two runs on the
  same dataset with different protocols don't overwrite each other.
* When two JSONs share the same key, the one with the larger ``n_seeds``
  wins (statistically more trustworthy per iter#11 lesson).
* The output Markdown reports median MSE per backbone (iter#11 lesson)
  and marks the per-row winner with ⭐ ; mean MSE shown beside for
  reference.
* When ``--include-molecular`` is set, it also ingests
  ``analysis/molecular/*_tox21_styled_graph_lnn.json`` and merges them
  into the same matrix under a "graph_tox21" task — gives a true
  cross-domain (graph + timeseries) view.

Usage::

    python scripts/build_backbone_matrix.py          # default: timeseries only
    python scripts/build_backbone_matrix.py --include-molecular
    python scripts/build_backbone_matrix.py --json   # stdout JSON
    python scripts/build_backbone_matrix.py --no-write
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _config_tag(cfg: dict) -> str:
    parts = []
    if cfg.get("warmup_frac", 0.0) > 0.0:
        parts.append(f"warmup={cfg['warmup_frac']}")
    if cfg.get("hidden_size"):
        parts.append(f"h={cfg['hidden_size']}")
    if cfg.get("num_regimes"):
        parts.append(f"r={cfg['num_regimes']}")
    return ",".join(parts) if parts else "default"


def _ingest_timeseries(path: pathlib.Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cfg = payload.get("config", {})
    summary = payload.get("summary") or {}
    if not summary:
        return None
    dataset = cfg.get("dataset", "unknown")
    tag = _config_tag(cfg)
    row_key = f"{dataset} [{tag}]"
    n_seeds = max((len(payload.get("per_run", [])), 1))
    # Build per-backbone summary; recompute median from per_run if missing.
    by_backbone: dict[str, list[float]] = {}
    for r in payload.get("per_run", []):
        by_backbone.setdefault(r["backbone"], []).append(float(r["test_mse"]))
    backbones = {}
    for name, mse_list in by_backbone.items():
        backbones[name] = {
            "params": summary.get(name, {}).get("parameters"),
            "median_mse": statistics.median(mse_list) if mse_list else None,
            "mean_mse": statistics.fmean(mse_list) if mse_list else None,
            "std_mse": statistics.stdev(mse_list) if len(mse_list) > 1 else 0.0,
            "n": len(mse_list),
        }
    seeds_per_backbone = max((len(v) for v in by_backbone.values()), default=0)
    return {
        "row_key": row_key,
        "domain": "timeseries",
        "metric": "test_mse",
        "metric_direction": "lower_is_better",
        "n_seeds": seeds_per_backbone,
        "backbones": backbones,
        "source_path": str(path.relative_to(ROOT)),
        "run_id": payload.get("run_id", "?"),
    }


def _ingest_molecular(path: pathlib.Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cfg = payload.get("config", {})
    results = payload.get("results", [])
    if not results:
        return None
    # graph_lnn_molecule runs are single-seed per JSON; aggregate across
    # multiple input files outside this function.
    backbones = {}
    for r in results:
        name = r.get("backbone")
        if not name:
            continue
        # auc is "higher is better"; we invert for matrix consistency.
        backbones[name] = {
            "params": r.get("parameters"),
            # For a single trial there's no median; report acc directly.
            "single_acc": r.get("val_accuracy"),
            "single_auc": r.get("val_auc_roc"),
        }
    seed = cfg.get("seed", 42)
    return {
        "row_key": f"graph_tox21 [seed={seed}]",
        "domain": "molecular",
        "metric": "val_auc_roc",
        "metric_direction": "higher_is_better",
        "n_seeds": 1,
        "backbones": backbones,
        "source_path": str(path.relative_to(ROOT)),
    }


def _merge_molecular_per_seed(rows: list[dict]) -> list[dict]:
    """Aggregate single-seed molecular rows into one row per backbone."""
    if not rows:
        return []
    import math
    by_backbone: dict[str, list[float]] = {}
    params_by_backbone: dict[str, int] = {}
    for r in rows:
        for name, b in r["backbones"].items():
            auc = b.get("single_auc")
            if auc is None:
                continue
            try:
                auc_f = float(auc)
            except (TypeError, ValueError):
                continue
            if math.isnan(auc_f) or math.isinf(auc_f):
                continue  # iter#6 first run had unbalanced labels → AUC NaN
            by_backbone.setdefault(name, []).append(auc_f)
            if name not in params_by_backbone and b.get("params") is not None:
                params_by_backbone[name] = b["params"]
    backbones = {}
    for name, aucs in by_backbone.items():
        if not aucs:
            continue
        backbones[name] = {
            "params": params_by_backbone.get(name),
            "median_metric": statistics.median(aucs),
            "mean_metric": statistics.fmean(aucs),
            "n": len(aucs),
        }
    if not backbones:
        return []
    return [{
        "row_key": "graph_tox21 [seeds:%d]" % max((b["n"] for b in backbones.values()), default=1),
        "domain": "molecular",
        "metric": "val_auc_roc",
        "metric_direction": "higher_is_better",
        "n_seeds": max((b["n"] for b in backbones.values()), default=1),
        "backbones": backbones,
        "source_path": "analysis/molecular/*_tox21_styled_graph_lnn.json (merged)",
    }]


def _dedupe_keep_higher_n(rows: list[dict]) -> list[dict]:
    by_key: dict[str, dict] = {}
    for r in rows:
        existing = by_key.get(r["row_key"])
        if existing is None or r["n_seeds"] > existing["n_seeds"]:
            by_key[r["row_key"]] = r
    return list(by_key.values())


def _format_markdown(payload: dict) -> str:
    rows = payload["rows"]
    backbones = payload["backbones_seen"]
    lines = [
        f"# Backbone × Task Matrix — {payload['run_id']}",
        "",
        "## Source",
        f"- Scanned: {payload['scan_summary']}",
        f"- Generator: `scripts/build_backbone_matrix.py` (PRD §9 #7)",
        f"- iter#11 lesson: median MSE used for timeseries (outlier-resistant);"
        " ⭐ marks per-row winner.",
        "",
        "## Timeseries (lower test_mse better)",
        "",
        "| Task / config | n | " + " | ".join(f"`{b}`" for b in backbones) + " |",
        "|---" + " |---:" * (len(backbones) + 1) + " |",
    ]
    for row in rows:
        if row["domain"] != "timeseries":
            continue
        # Find winner column (lowest median).
        values = {b: row["backbones"].get(b, {}).get("median_mse") for b in backbones}
        valid = {k: v for k, v in values.items() if v is not None}
        winner = min(valid, key=valid.get) if valid else None
        cells = []
        for b in backbones:
            cell = row["backbones"].get(b)
            if not cell or cell.get("median_mse") is None:
                cells.append("—")
                continue
            mark = " ⭐" if b == winner else ""
            cells.append(f"{cell['median_mse']:.4f}{mark}")
        lines.append(
            f"| {row['row_key']} | {row['n_seeds']} | " + " | ".join(cells) + " |"
        )

    mol_rows = [r for r in rows if r["domain"] == "molecular"]
    if mol_rows:
        lines.extend([
            "",
            "## Molecular (higher val_auc_roc better)",
            "",
            "| Task / config | n | " + " | ".join(f"`{b}`" for b in backbones) + " |",
            "|---" + " |---:" * (len(backbones) + 1) + " |",
        ])
        for row in mol_rows:
            values = {b: row["backbones"].get(b, {}).get("median_metric") for b in backbones}
            valid = {k: v for k, v in values.items() if v is not None}
            winner = max(valid, key=valid.get) if valid else None
            cells = []
            for b in backbones:
                cell = row["backbones"].get(b)
                if not cell or cell.get("median_metric") is None:
                    cells.append("—")
                    continue
                mark = " ⭐" if b == winner else ""
                cells.append(f"{cell['median_metric']:.4f}{mark}")
            lines.append(
                f"| {row['row_key']} | {row['n_seeds']} | " + " | ".join(cells) + " |"
            )

    # Per-backbone win tally.
    tally: dict[str, int] = {b: 0 for b in backbones}
    for row in rows:
        if row["domain"] == "timeseries":
            valid = {b: row["backbones"].get(b, {}).get("median_mse")
                     for b in backbones if row["backbones"].get(b, {}).get("median_mse") is not None}
            if valid:
                tally[min(valid, key=valid.get)] += 1
        elif row["domain"] == "molecular":
            valid = {b: row["backbones"].get(b, {}).get("median_metric")
                     for b in backbones if row["backbones"].get(b, {}).get("median_metric") is not None}
            if valid:
                tally[max(valid, key=valid.get)] += 1

    lines.extend([
        "",
        "## Win tally",
        "| Backbone | wins | comment |",
        "|---|---:|---|",
    ])
    for b in backbones:
        lines.append(f"| `{b}` | {tally[b]} | {'⭐ overall lead' if tally[b] == max(tally.values()) and tally[b] > 0 else ''} |")

    lines.extend([
        "",
        "## 解读",
        "- 矩阵把仓库 11 轮 loop 的 ablation JSON pivot 成单表;",
        "- 列 = backbone, 行 = (dataset + 关键 config) — iter#10 (warmup_frac=0.1 N=3) "
        "和 iter#11 (warmup_frac=0.1 N=8) 因 dedup-keep-higher-n 规则,后者覆盖前者;",
        "- ⭐ 标记 per-row 的最佳 backbone(median 视角,outlier-resistant);",
        "- win tally 给出'通杀 backbone'是否存在的最终答案。",
        "",
        f"JSON 原数据: `{payload['json_path']}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-molecular", action="store_true",
                        help="Also ingest analysis/molecular/*_tox21_styled_graph_lnn.json")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output-dir", default="analysis/backbone_matrix")
    args = parser.parse_args()

    rows: list[dict] = []
    timeseries_dir = ROOT / "analysis" / "timeseries_ablation"
    if timeseries_dir.exists():
        for path in sorted(timeseries_dir.glob("*_lnn_vs_lstm.json")):
            row = _ingest_timeseries(path)
            if row:
                rows.append(row)
    rows = _dedupe_keep_higher_n(rows)

    if args.include_molecular:
        molecular_dir = ROOT / "analysis" / "molecular"
        if molecular_dir.exists():
            mol_rows = []
            for path in sorted(molecular_dir.glob("*_tox21_styled_graph_lnn.json")):
                r = _ingest_molecular(path)
                if r:
                    mol_rows.append(r)
            rows.extend(_merge_molecular_per_seed(mol_rows))

    # Discover full backbone set across all rows.
    backbones_seen: list[str] = []
    for r in rows:
        for b in r["backbones"]:
            if b not in backbones_seen:
                backbones_seen.append(b)
    # Canonical order.
    canonical = ["cfc", "ltc", "gru", "lstm"]
    backbones_seen = [b for b in canonical if b in backbones_seen] + \
                     [b for b in backbones_seen if b not in canonical]

    now = dt.datetime.now()
    run_id = now.strftime("%Y-%m-%d_%H%M%S")
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    json_path = output_dir / f"{run_id}_backbone_matrix.json"
    md_path = output_dir / f"{run_id}_backbone_matrix.md"
    rel_json = json_path.relative_to(ROOT) if json_path.is_relative_to(ROOT) else json_path

    payload = {
        "run_id": run_id,
        "generated_at": now.isoformat(),
        "scan_summary": (
            f"{len(rows)} rows; timeseries from analysis/timeseries_ablation/"
            + (", molecular from analysis/molecular/" if args.include_molecular else "")
        ),
        "rows": rows,
        "backbones_seen": backbones_seen,
        "json_path": str(rel_json),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not args.no_write:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(_format_markdown(payload), encoding="utf-8")

    print(f"=== Backbone matrix ({len(rows)} rows, {len(backbones_seen)} backbones) ===")
    for row in rows:
        ranks = {b: row["backbones"].get(b, {}).get("median_mse")
                 or row["backbones"].get(b, {}).get("median_metric")
                 for b in backbones_seen}
        ranks = {k: v for k, v in ranks.items() if v is not None}
        if not ranks:
            continue
        if row["domain"] == "timeseries":
            winner = min(ranks, key=ranks.get)
        else:
            winner = max(ranks, key=ranks.get)
        print(f"  {row['row_key']:40s} (n={row['n_seeds']:>2})  winner: {winner}")
    if not args.no_write:
        print(f"  wrote JSON: {json_path}")
        print(f"  wrote MD:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
