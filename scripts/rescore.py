"""Re-apply the current scorer to stored benchmark outputs.

Why this exists:
    Evaluation outputs depend only on (model, adapter, sampler, seed), none of
    which change when scoring rules are revised (benchmark v1.0 -> v1.1,
    scorer bug fixes). Regenerating would be byte-identical but costs ~3 min
    per model; this script re-runs just the scoring over the stored outputs
    and is exactly equivalent.

Usage:
    uv run python scripts/rescore.py --tag base
    uv run python scripts/rescore.py --all        # every eval JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate import score_case, weighted_total

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs" / "eval"


def rescore_tag(tag: str) -> None:
    path = EVAL / f"{tag}.json"
    if not path.exists():
        raise SystemExit(f"missing eval: {path}")
    out = json.load(open(path, encoding="utf-8"))
    bench = json.load(open(ROOT / "benchmark" / "benchmark.json", encoding="utf-8"))
    cases = {c["id"]: c for c in bench["cases"]}
    dims = bench["dimensions"]

    for r in out["results"]:
        case = cases.get(r["id"])
        if case is None:
            raise SystemExit(f"{tag}: unknown case id {r['id']}")
        scores, checks = score_case(r["output"], case, dims)
        r["scores"] = scores
        r["checks"] = checks

    results = out["results"]
    dim_avg = {
        d: round(sum(r["scores"][d] for r in results) / len(results), 3)
        for d in dims
    }
    out["dim_avg"] = dim_avg
    out["total"] = weighted_total(dim_avg, dims)
    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
    out["by_category"] = {
        c: {
            "n": len(rs),
            "total": weighted_total(
                {d: sum(x["scores"][d] for x in rs) / len(rs) for d in dims},
                dims,
            ),
        }
        for c, rs in sorted(by_cat.items())
    }
    out["n_cases"] = len(results)
    out["benchmark_version"] = bench["version"]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"{tag:28s} total={out['total']:.3f}  (n={len(results)}, "
          f"bench v{bench['version']})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=None)
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    if not args.tag and not args.all:
        raise SystemExit("specify --tag NAME or --all")
    tags = [q.stem for q in sorted(EVAL.glob("*.json"))] if args.all else [args.tag]
    for t in tags:
        rescore_tag(t)


if __name__ == "__main__":
    main()
