#!/usr/bin/env python3
"""从 fallback json 中挑选前 N 个 30 天内 + score>=6 的候选, 输出研读清单."""
import argparse, json, pathlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", required=True)
    ap.add_argument("--date", required=True, help="yyyy-mm-dd")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--within-days", type=int, default=30)
    ap.add_argument("--min-score", type=int, default=4)
    args = ap.parse_args()

    rows = json.loads(pathlib.Path(args.in_json).read_text(encoding="utf-8"))
    today = pathlib.Path("/Users/hyx/workspace/LNN").joinpath("docs/daily").exists()  # unused marker
    # parse date
    import datetime
    run = datetime.datetime.strptime(args.date, "%Y-%m-%d").date()

    def days(d):
        try:
            return abs((run - datetime.datetime.strptime(d, "%Y-%m-%d").date()).days)
        except Exception:
            return 9999

    kept = [r for r in rows if days(r["date"]) <= args.within_days and r["score"] >= args.min_score]
    kept.sort(key=lambda r: (r["score"], -days(r["date"])), reverse=True)
    top = kept[: args.top]
    print(f"[select] within {args.within_days}d, score>={args.min_score}: {len(kept)} papers; top {args.top}:")
    for r in top:
        print(f"  score={r['score']:>2} days={days(r['date']):>3} {r['date']} {r['id']} :: {r['title'][:90]}")
    out = pathlib.Path("/Users/hyx/workspace/LNN/papers/daily/_selected_2026-09-27.json")
    out.write_text(json.dumps(top, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[select] written -> {out}")

if __name__ == "__main__":
    main()
