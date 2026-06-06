#!/usr/bin/env python3
"""Head-to-head benchmarking CLI for LNN/SSM backbones (round 73).

Trains a sweep of (backbone × dataset × seed) and reports:
  * A markdown table on stdout (one row per (backbone, dataset) with
    the mean ± std of test_metric_value over seeds).
  * A machine-readable JSON dump to analysis/head_to_head/<timestamp>.json
    with all per-seed rows.

Usage::

    python scripts/bench_cfc_vs_ssm.py                                  # default sweep
    python scripts/bench_cfc_vs_ssm.py --backbones cfc,gru --datasets mackey_glass
    python scripts/bench_cfc_vs_ssm.py --seeds 0,1,2 --hidden 32 --epochs 5
    python scripts/bench_cfc_vs_ssm.py --quick                           # 1 epoch, hidden 16, 1 seed
    python scripts/bench_cfc_vs_ssm.py --json-only                      # suppress markdown
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lnn.core.bench_suite import (  # noqa: E402
    list_backbones,
    list_datasets,
    run_suite,
)


def _parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_int_csv(s: str) -> list[int]:
    return [int(x) for x in _parse_csv(s)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--backbones",
        type=str,
        default=",".join(list_backbones()),
        help=f"Comma-separated backbone names (default: {','.join(list_backbones())})",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(list_datasets()),
        help=f"Comma-separated dataset names (default: {','.join(list_datasets())})",
    )
    parser.add_argument(
        "--seeds", type=str, default="0,1,2", help="Comma-separated seeds (default: 0,1,2)"
    )
    parser.add_argument("--hidden", type=int, default=32, help="Hidden size (default: 32)")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per run (default: 5)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--quick", action="store_true", help="Quick smoke: 1 seed, 16 hidden, 1 epoch"
    )
    parser.add_argument("--json-only", action="store_true", help="Only emit the JSON report")
    args = parser.parse_args()

    if args.quick:
        seeds = [0]
        hidden = 16
        epochs = 1
    else:
        seeds = _parse_int_csv(args.seeds)
        hidden = args.hidden
        epochs = args.epochs

    backbones = _parse_csv(args.backbones)
    datasets = _parse_csv(args.datasets)

    # Validate
    bad_b = [b for b in backbones if b not in list_backbones()]
    bad_d = [d for d in datasets if d not in list_datasets()]
    if bad_b:
        print(f"ERROR: unknown backbones {bad_b}. Valid: {list_backbones()}", file=sys.stderr)
        return 1
    if bad_d:
        print(f"ERROR: unknown datasets {bad_d}. Valid: {list_datasets()}", file=sys.stderr)
        return 1

    # Sweep
    all_rows: list[dict] = []
    n_total = len(backbones) * len(datasets) * len(seeds)
    n_done = 0
    t_start = time.time()

    if not args.json_only:
        print(
            f"[bench] sweep: {len(backbones)} backbones × {len(datasets)} datasets × "
            f"{len(seeds)} seeds = {n_total} runs"
        )
        print(
            f"[bench] hidden={hidden}  epochs={epochs}  batch={args.batch_size}  lr={args.lr}"
        )
        print()

    for backbone in backbones:
        for dataset in datasets:
            for seed in seeds:
                n_done += 1
                if not args.json_only:
                    print(
                        f"[{n_done:3d}/{n_total}] backbone={backbone:5s}  "
                        f"dataset={dataset:14s}  seed={seed} ...",
                        end=" ",
                        flush=True,
                    )
                t0 = time.time()
                r = run_suite(
                    backbone,
                    dataset,
                    seed=seed,
                    hidden=hidden,
                    epochs=epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                )
                if not args.json_only:
                    print(
                        f"{r.test_metric_name}={r.test_metric_value:.4f}  "
                        f"params={r.n_params}  wall={r.wall_clock_s:.2f}s"
                    )
                d = r.to_dict()
                d["sweep_wall_s"] = round(time.time() - t0, 3)
                all_rows.append(d)

    elapsed = time.time() - t_start

    # Aggregate: per (backbone, dataset) mean / std over seeds
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in all_rows:
        grouped.setdefault((row["backbone"], row["dataset"]), []).append(row)

    if not args.json_only:
        print()
        print(f"[bench] done in {elapsed:.1f}s. Aggregating ...")
        print()
        print(
            "| Backbone | Dataset | N | "
            f"{'MSE ± std' if all(r['test_metric_name'] == 'mse' for r in all_rows) else 'Metric ± std'} | "
            "Params | Wall (s/run) |"
        )
        print("|---|---|---:|---:|---:|---:|")
        for (backbone, dataset), rows in sorted(grouped.items()):
            metric_name = rows[0]["test_metric_name"]
            vals = [r["test_metric_value"] for r in rows]
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            params = rows[0]["n_params"]
            wall = sum(r["wall_clock_s"] for r in rows) / len(rows)
            print(
                f"| {backbone} | {dataset} | {len(rows)} | "
                f"{mean:.4f} ± {std:.4f} ({metric_name}) | "
                f"{params} | {wall:.2f} |"
            )

    # Write JSON
    out_dir = ROOT / "analysis" / "head_to_head"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = out_dir / f"bench_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config": {
                    "backbones": backbones,
                    "datasets": datasets,
                    "seeds": seeds,
                    "hidden": hidden,
                    "epochs": epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                },
                "elapsed_s": round(elapsed, 2),
                "rows": all_rows,
                "grouped": {
                    f"{b}__{d}": {
                        "metric_name": rows[0]["test_metric_name"],
                        "mean": statistics.mean(r["test_metric_value"] for r in rows),
                        "std": (
                            statistics.stdev(r["test_metric_value"] for r in rows)
                            if len(rows) > 1
                            else 0.0
                        ),
                        "n": len(rows),
                        "params": rows[0]["n_params"],
                    }
                    for (b, d), rows in grouped.items()
                },
            },
            indent=2,
        )
    )
    if not args.json_only:
        print()
        print(f"[bench] wrote {out_path.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
