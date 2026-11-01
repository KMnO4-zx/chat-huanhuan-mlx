"""Pull per-run training metrics from SwanLab and print a summary table.

Uses ``swanlab api`` so it works without importing the SDK's run state.

Usage:
    uv run python scripts/analyze_swanlab.py
    uv run python scripts/analyze_swanlab.py --json outputs/swanlab_analysis.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "kmno4/huanhuan-mlx"
KEYS = "train_loss,val_loss,iterations_per_second,tokens_per_second,peak_memory"


def api(*args: str) -> dict:
    out = subprocess.run(
        ["uv", "run", "swanlab", "api", *args],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out[out.find("{"):])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json", default=None)
    args = p.parse_args()

    listing = api("run", "list", PROJECT)
    runs = listing["data"]["list"]
    summary = []
    for r in runs:
        rid = r["run_id"]
        name = r["name"]
        try:
            metrics = api("run", "metrics", f"{PROJECT}/{rid}", "--keys", KEYS)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: metrics fetch failed: {e}")
            continue
        entry = {"name": name, "run_id": rid, "finished": bool(r.get("finished_at"))}
        for series in metrics["data"]["list"]:
            pts = series["metrics"]
            if not pts:
                continue
            key = series["key"]
            vals = [x["data"] for x in pts]
            entry[key] = {
                "n": len(pts),
                "first": vals[0],
                "last": vals[-1],
                "min": min(vals),
                "max": max(vals),
                "last_index": pts[-1]["index"],
            }
        summary.append(entry)

    print(f"{'run':10s} {'iters':>6s} {'train_loss':>12s} {'val_loss':>10s} "
          f"{'tok/s':>8s} {'peak_GB':>8s} {'finished':>8s}")
    for e in summary:
        tl = e.get("train_loss", {})
        vl = e.get("val_loss", {})
        ts = e.get("tokens_per_second", {})
        pm = e.get("peak_memory", {})
        print(f"{e['name']:10s} {tl.get('last_index','-'):>6} "
              f"{tl.get('last', float('nan')):>12.3f} "
              f"{vl.get('last', float('nan')):>10.3f} "
              f"{ts.get('last', float('nan')):>8.0f} "
              f"{pm.get('max', float('nan')):>8.2f} "
              f"{str(e['finished']):>8s}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
