"""Compare two benchmark result JSONs (e.g. base vs final).

Usage:
    uv run python scripts/compare.py --a base --b run3
    uv run python scripts/compare.py --a base --b run3 --markdown
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs" / "eval"


def load(tag: str) -> dict:
    with open(EVAL / f"{tag}.json", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args()

    a, b = load(args.a), load(args.b)
    dims = list(a["dim_avg"].keys())
    rows = []
    for d in dims:
        va, vb = a["dim_avg"][d], b["dim_avg"][d]
        rows.append((d, va, vb, vb - va))
    cats = sorted(set(a["by_category"]) | set(b["by_category"]))
    cat_rows = []
    for c in cats:
        va = a["by_category"].get(c, {}).get("total")
        vb = b["by_category"].get(c, {}).get("total")
        cat_rows.append((c, va, vb, None if va is None or vb is None else vb - va))

    if args.markdown:
        print(f"| 维度 | {args.a} | {args.b} | Δ |")
        print("|---|---:|---:|---:|")
        for d, va, vb, dv in rows:
            print(f"| {d} | {va:.3f} | {vb:.3f} | {dv:+.3f} |")
        print(f"| **总分** | **{a['total']:.3f}** | **{b['total']:.3f}** | "
              f"**{b['total']-a['total']:+.3f}** |")
        print()
        print(f"| 分类 | {args.a} | {args.b} | Δ |")
        print("|---|---:|---:|---:|")
        for c, va, vb, dv in cat_rows:
            va_s = "-" if va is None else f"{va:.3f}"
            vb_s = "-" if vb is None else f"{vb:.3f}"
            dv_s = "-" if dv is None else f"{dv:+.3f}"
            print(f"| {c} | {va_s} | {vb_s} | {dv_s} |")
    else:
        print(f"{'dimension':16s} {args.a:>8s} {args.b:>8s} {'delta':>8s}")
        for d, va, vb, dv in rows:
            print(f"{d:16s} {va:8.3f} {vb:8.3f} {dv:+8.3f}")
        print(f"{'TOTAL':16s} {a['total']:8.3f} {b['total']:8.3f} "
              f"{b['total']-a['total']:+8.3f}")

    # per-case changes
    am = {r["id"]: r for r in a["results"]}
    bm = {r["id"]: r for r in b["results"]}
    diffs = []
    for cid in am:
        if cid not in bm:
            continue
        ta = sum(am[cid]["scores"].values()) / len(am[cid]["scores"])
        tb = sum(bm[cid]["scores"].values()) / len(bm[cid]["scores"])
        diffs.append((tb - ta, cid, am[cid]["output"], bm[cid]["output"]))
    diffs.sort()
    print("\n-- biggest regressions --")
    for dv, cid, oa, ob in diffs[:5]:
        print(f"[{cid}] {dv:+.2f}\n  before: {oa[:70]}\n  after : {ob[:70]}")
    print("\n-- biggest improvements --")
    for dv, cid, oa, ob in diffs[-5:][::-1]:
        print(f"[{cid}] {dv:+.2f}\n  before: {oa[:70]}\n  after : {ob[:70]}")


if __name__ == "__main__":
    main()
