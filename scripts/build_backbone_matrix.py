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


def _ingest_smnist_gap(path: pathlib.Path) -> dict | None:
    """Ingest a PDNA stage B summary JSON (5 variants × N seeds, sMNIST Gapped).

    Each variant name is treated as a separate "backbone" for the matrix;
    primary reported metric is multi-gap accuracy (higher better), per the
    PDNA paper's headline number (+4.62 pp multi-gap, Cohen's d=0.87).
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    variants = payload.get("variants") or []
    if not variants:
        return None
    backbones: dict = {}
    for v in variants:
        name = v.get("name", "?")
        agg = v.get("aggregate", {}) or {}
        if not agg:
            continue
        backbones[name] = {
            "params": agg.get("n_params_mean"),
            "median_metric": agg.get("multi20_mean"),
            "mean_metric": agg.get("multi20_mean"),
            "std_metric": agg.get("multi20_std", 0.0),
            "median_metric_gap5": agg.get("5%_mean"),
            "median_metric_gap0": agg.get("0%_mean"),
            "n": agg.get("n_seeds", 0),
        }
    if not backbones:
        return None
    n_seeds = max((b["n"] for b in backbones.values()), default=0)
    return {
        "row_key": f"smnist_gap [n={n_seeds},h={payload.get('hidden_size',64)}]",
        "domain": "smnist_gap",
        "metric": "multi_gap_acc",
        "metric_direction": "higher_is_better",
        "n_seeds": n_seeds,
        "backbones": backbones,
        "source_path": str(path.relative_to(ROOT)),
        "run_id": payload.get("run_id", "?"),
    }


def _ingest_lra_pathfinder(path: pathlib.Path) -> dict | None:
    """Ingest a PDNA stage C summary JSON (N variants × seeds, synthetic Pathfinder).

    Reported metric is binary classification test accuracy (higher better).
    Pathfinder is an LRA-style long-range task: 32x32 grid → seq_len=1024,
    and the model must integrate information across the full sequence to
    decide whether two endpoint markers are connected by a path.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    variants = payload.get("variants") or []
    if not variants:
        return None
    backbones: dict = {}
    for v in variants:
        name = v.get("name", "?")
        agg = v.get("aggregate", {}) or {}
        if not agg:
            continue
        backbones[name] = {
            "params": agg.get("n_params_mean"),
            "median_metric": agg.get("test_acc_mean"),
            "mean_metric": agg.get("test_acc_mean"),
            "std_metric": agg.get("test_acc_std", 0.0),
            "n": agg.get("n_seeds", 0),
        }
    if not backbones:
        return None
    n_seeds = max((b["n"] for b in backbones.values()), default=0)
    return {
        "row_key": f"lra_pathfinder [n={n_seeds},h={payload.get('hidden_size',32)},seq={payload.get('seq_len',1024)}]",
        "domain": "lra_pathfinder",
        "metric": "test_acc",
        "metric_direction": "higher_is_better",
        "n_seeds": n_seeds,
        "backbones": backbones,
        "source_path": str(path.relative_to(ROOT)),
        "run_id": payload.get("run_id", "?"),
    }


def _compute_win_tally(rows: list[dict], backbones: list[str]) -> dict[str, int]:
    """Per-backbone win tally (used by both Markdown formatter and README snippet).

    Each row declares a winner (lowest median_mse for timeseries, highest
    median_metric for molecular / smnist_gap). Returns a dict {bb: wins}.
    """
    tally: dict[str, int] = {b: 0 for b in backbones}
    for row in rows:
        if row["domain"] == "timeseries":
            valid = {b: row["backbones"].get(b, {}).get("median_mse")
                     for b in backbones if row["backbones"].get(b, {}).get("median_mse") is not None}
            if valid:
                tally[min(valid, key=valid.get)] += 1
        elif row["domain"] in ("molecular", "smnist_gap", "lra_pathfinder"):
            valid = {b: row["backbones"].get(b, {}).get("median_metric")
                     for b in backbones if row["backbones"].get(b, {}).get("median_metric") is not None}
            if valid:
                tally[max(valid, key=valid.get)] += 1
    return tally


def _format_readme_snippet(payload: dict) -> str:
    """PRD §10 #6: 1-line Markdown badge for README.md embedding.

    Example output (PRD §10 #6 / iter#29):
        **Backbone matrix:** LSTM 3 / cfc 2 / cfc_pulse 1 / others 0 (5 rows × 4 domains)
    """
    rows = payload["rows"]
    backbones = payload["backbones_seen"]
    tally = _compute_win_tally(rows, backbones)
    # Count distinct domains represented
    domains = sorted({r["domain"] for r in rows})
    domain_count = len(domains)
    row_count = len(rows)
    # Build the "name wins" string, sorted by win count desc
    parts = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
    nonzero = [(name, n) for name, n in parts if n > 0]
    if not nonzero:
        return f"**Backbone matrix:** {row_count} rows × {domain_count} domains (no winners yet)"
    pieces = [f"`{name}` {n}" for name, n in nonzero]
    pieces.append(f"others {sum(n for _, n in parts) - sum(n for _, n in nonzero)}")
    return (
        f"**Backbone matrix ({row_count} rows × {domain_count} domains):** "
        + " / ".join(pieces)
    )


def _dedupe_keep_higher_n(rows: list[dict]) -> list[dict]:
    """Merge rows with the same row_key by per-backbone max n_seeds (iter#25).

    The original implementation replaced the whole row with the higher-n_seeds
    one, which had a real failure mode: a 3-seed cfc/ltc/gru/lstm/fhn_dynpmnn
    row and a 6-seed fhn_dynpmnn-only row (same row_key) would keep the
    6-seed row and silently drop the 3-seed cfc/ltc/gru/lstm data.

    The new logic: for each (row_key, backbone) pair, keep the row where
    that backbone has the highest n_seeds. This is per-backbone max.
    """
    by_key: dict[str, list[dict]] = {}
    for r in rows:
        by_key.setdefault(r["row_key"], []).append(r)

    merged: list[dict] = []
    for row_key, group in by_key.items():
        # For each backbone across all rows in this group, keep the
        # entry with the highest n_seeds for that backbone.
        best_per_backbone: dict[str, dict] = {}
        for r in group:
            for bb_name, bb_data in r.get("backbones", {}).items():
                cur = best_per_backbone.get(bb_name)
                if cur is None or bb_data.get("n", 0) > cur.get("n", 0):
                    best_per_backbone[bb_name] = bb_data
        # Total n_seeds for the row is the max across backbones (matches
        # the original semantics so the n column in the Markdown table
        # still reflects "the strongest backbone had this many seeds").
        n_seeds = max((bb.get("n", 0) for bb in best_per_backbone.values()), default=0)
        # Use the first row as a template for non-backbone fields.
        template = group[0]
        merged.append({
            **template,
            "n_seeds": n_seeds,
            "backbones": best_per_backbone,
        })
    return merged


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

    smnist_gap_rows = [r for r in rows if r["domain"] == "smnist_gap"]
    if smnist_gap_rows:
        lines.extend([
            "",
            "## sMNIST Gapped (higher multi_gap_acc better, PDNA stage B iter#20+)",
            "",
            "| Task / config | n | " + " | ".join(f"`{b}`" for b in backbones) + " |",
            "|---" + " |---:" * (len(backbones) + 1) + " |",
        ])
        for row in smnist_gap_rows:
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
                cells.append(f"{cell['median_metric']:.2f}{mark}")
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
    tally = _compute_win_tally(rows, backbones)

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
    parser.add_argument("--include-smnist-gap", action="store_true",
                        help="Also ingest analysis/pdna/2026-06-04_pdna_stage_b_summary.json "
                             "(PDNA stage B, iter#20+)")
    parser.add_argument("--include-lra", action="store_true",
                        help="Also ingest analysis/pdna_lra/*_pdna_lra_summary.json "
                             "(PDNA stage C, iter#28+ LRA Pathfinder smoke)")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--export-readme-snippet", action="store_true",
                        help=(
                            "PRD §10 #6 / iter#29: emit a 1-line Markdown badge summarising "
                            "the cross-task backbone matrix (e.g. 'LSTM 3 / cfc 2 / ...'). "
                            "Designed to be embedded in README.md. Does not write any file."
                        ))
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

    if args.include_smnist_gap:
        pdna_dir = ROOT / "analysis" / "pdna"
        # Match the stage-B summary file produced by scripts/experiment_pdna_smoke.py
        for path in sorted(pdna_dir.glob("*_pdna_stage_b_summary.json")):
            r = _ingest_smnist_gap(path)
            if r:
                rows.append(r)

    if args.include_lra:
        pdna_lra_dir = ROOT / "analysis" / "pdna_lra"
        # Match the stage-C summary file produced by scripts/experiment_pdna_lra.py
        for path in sorted(pdna_lra_dir.glob("*_pdna_lra_summary.json")):
            r = _ingest_lra_pathfinder(path)
            if r:
                rows.append(r)

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
            + (", smnist_gap from analysis/pdna/" if args.include_smnist_gap else "")
            + (", lra_pathfinder from analysis/pdna_lra/" if args.include_lra else "")
        ),
        "rows": rows,
        "backbones_seen": backbones_seen,
        "json_path": str(rel_json),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.export_readme_snippet:
        # PRD §10 #6: emit a 1-line Markdown badge for README.md embedding.
        # Does NOT write any file — caller is expected to redirect stdout
        # or copy-paste the result into README.md.
        print(_format_readme_snippet(payload))
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
