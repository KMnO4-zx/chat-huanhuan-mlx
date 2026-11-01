"""Finalize delivery:
  1. pick the best evaluated adapter by benchmark total score
  2. copy it to outputs/adapters/final/
  3. fuse LoRA into a standalone model at models/Qwen3-0.6B-huanhuan-mlx/
  4. write REPORT.md and a delivery manifest

Usage:
    uv run python scripts/finalize.py [--best run2]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs" / "eval"
ADAPTERS = ROOT / "outputs" / "adapters"
MERGED = ROOT / "models" / "Qwen3-0.6B-huanhuan-mlx"


def best_tag() -> str:
    """Pick the best *full-benchmark* tuned eval.

    Diagnostic probes (``probe_*``, 10 cases, greedy) are excluded and the
    case count must match the base run; otherwise a 10-case probe could be
    crowned final over a 40-case evaluation.
    """
    base_path = EVAL / "base.json"
    base_n = json.load(open(base_path, encoding="utf-8"))["n_cases"] \
        if base_path.exists() else None
    scores = {}
    for p in EVAL.glob("*.json"):
        if p.stem.startswith("probe_"):
            continue
        d = json.load(open(p, encoding="utf-8"))
        if not d.get("adapter"):
            continue
        # Only the primary protocol (system prompt on) may be selected;
        # --no-system robustness evals are a different protocol.
        if not d.get("generation", {}).get("with_system", True):
            continue
        if base_n is not None and d.get("n_cases") != base_n:
            continue
        scores[p.stem] = d["total"]
    if not scores:
        raise SystemExit("no evaluated adapters found")
    return max(scores, key=scores.get)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--best", default=None, help="override auto-picked tag")
    p.add_argument("--skip-fuse", action="store_true")
    args = p.parse_args()

    tag = args.best or best_tag()
    src = ADAPTERS / tag
    if not (src / "adapters.safetensors").exists():
        raise SystemExit(f"missing adapter weights: {src}")

    final_dir = ADAPTERS / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True)
    # Deliver the final adapter + its config only. Iteration checkpoints
    # (NNN_adapters.safetensors) and the swanlog dir stay in the run dir.
    keep = {"adapters.safetensors", "adapter_config.json"}
    for f in src.iterdir():
        if f.is_file() and (f.name in keep or f.suffix in {".yaml", ".txt", ".json"}):
            shutil.copy2(f, final_dir / f.name)
    print(f"copied {src} -> {final_dir} "
          f"({', '.join(sorted(x.name for x in final_dir.iterdir()))})")

    if not args.skip_fuse:
        if MERGED.exists():
            shutil.rmtree(MERGED)
        cmd = [
            "uv", "run", "mlx_lm.fuse",
            "--model", str(ROOT / "models" / "Qwen3-0.6B-mlx"),
            "--adapter-path", str(final_dir),
            "--save-path", str(MERGED),
        ]
        print("Fusing:", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)
        print(f"merged standalone model -> {MERGED}")

    # Snapshot SwanLab metrics for REPORT.md (best-effort; report.py falls
    # back gracefully if the API is unreachable).
    subprocess.run(
        ["uv", "run", "python", "scripts/analyze_swanlab.py",
         "--json", "outputs/swanlab_analysis.json"],
        cwd=ROOT, check=False,
    )
    subprocess.run(["uv", "run", "python", "scripts/report.py"], cwd=ROOT,
                   check=True)

    def _score(t: str) -> dict | None:
        p = EVAL / f"{t}.json"
        if not p.exists():
            return None
        d = json.load(open(p, encoding="utf-8"))
        return {
            "total": d.get("total"),
            "n_cases": d.get("n_cases"),
            "benchmark_version": d.get("benchmark_version"),
            "enable_thinking": d.get("generation", {}).get("enable_thinking"),
            "eval_file": str(p.relative_to(ROOT)),
        }

    base_s, final_s = _score("base"), _score(tag)
    manifest = {
        "selected_tag": tag,
        "adapter": str(final_dir.relative_to(ROOT)),
        "standalone_model": None if args.skip_fuse else str(MERGED.relative_to(ROOT)),
        "base_model": "models/Qwen3-0.6B-mlx",
        "benchmark": "benchmark/benchmark.json",
        "base_eval": base_s,
        "final_eval": final_s,
        "delta": round(final_s["total"] - base_s["total"], 3)
        if base_s and final_s and base_s.get("total") is not None
        and final_s.get("total") is not None else None,
        "swanlab_project": "https://swanlab.cn/@kmno4/huanhuan-mlx",
        "improvements": "IMPROVEMENTS.md",
        "report": "REPORT.md",
    }
    with open(ROOT / "DELIVERY.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
