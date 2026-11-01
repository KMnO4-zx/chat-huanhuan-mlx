"""Verify that every program.md requirement is satisfied.

Run at the end:
    uv run python scripts/check_deliverables.py
Exits non-zero if anything required is missing.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    CHECKS.append((name, cond, detail))
    print(("PASS" if cond else "FAIL"), name, ("— " + detail) if detail else "")


def main() -> None:
    runs = json.load(open(ROOT / "logs" / "training_runs.json", encoding="utf-8"))
    finished = [r for r in runs if r.get("duration_seconds") is not None]

    # --- program.md: 任务与目标 ---
    check("dataset/huanhuan.json 存在", (ROOT / "dataset/huanhuan.json").exists())
    check("基础模型为 Qwen3-0.6B（ModelScope 下载至 models/）",
          (ROOT / "models/Qwen3-0.6B/config.json").exists())
    mlc = json.load(open(ROOT / "models/Qwen3-0.6B/config.json", encoding="utf-8"))
    check("config model_type == qwen3", mlc.get("model_type") == "qwen3",
          str(mlc.get("model_type")))
    check("MLX 权重存在", (ROOT / "models/Qwen3-0.6B-mlx/model.safetensors").exists())
    check("自建 benchmark 存在且 >=30 条",
          (lambda b: len(b["cases"]) >= 30)(
              json.load(open(ROOT / "benchmark/benchmark.json", encoding="utf-8"))))

    # 非思考：训练目标与推理均不产生思维链
    bench = json.load(open(ROOT / "benchmark/benchmark.json", encoding="utf-8"))
    check("评测配置 enable_thinking=False",
          all(
              not json.load(open(p, encoding="utf-8"))
              ["generation"].get("enable_thinking", True)
              for p in (ROOT / "outputs/eval").glob("*.json")
          ) if (ROOT / "outputs/eval").exists() else False)

    # --- program.md: 约束 ---
    check("uv 环境：uv.lock 存在", (ROOT / "uv.lock").exists())
    check("ModelScope 下载的模型在 models/", (ROOT / "models/Qwen3-0.6B").is_dir())
    swan_runs = subprocess.run(
        ["uv", "run", "swanlab", "api", "run", "list", "kmno4/huanhuan-mlx"],
        cwd=ROOT, capture_output=True, text=True,
    )
    n_swan = 0
    try:
        n_swan = json.loads(swan_runs.stdout[swan_runs.stdout.find("{"):])["data"]["total"]
    except Exception:  # noqa: BLE001
        pass
    check("SwanLab 项目有运行记录", n_swan >= 1, f"{n_swan} runs")
    check("训练次数 <= 10", len(runs) <= 10, f"{len(runs)} records")
    check("每次训练 <= 20 分钟",
          all(r["duration_seconds"] <= 1200 + 2 for r in finished),
          ", ".join(f"{r['name']}={r['duration_seconds']}s" for r in finished))

    # --- program.md: 评估与交付 ---
    evals = list((ROOT / "outputs/eval").glob("*.json")) if (ROOT / "outputs/eval").exists() else []
    tags = {p.stem for p in evals}

    def full(tag: str) -> bool:
        if tag not in tags:
            return False
        d = json.load(open(ROOT / "outputs/eval" / f"{tag}.json", encoding="utf-8"))
        return d["n_cases"] == len(bench["cases"])

    full_tuned = [
        t for t in tags
        if not t.startswith("probe_") and full(t)
        and json.load(open(ROOT / "outputs/eval" / f"{t}.json",
                           encoding="utf-8")).get("adapter")
    ]
    check("训练前 base 评测完成", full("base"))
    check("至少一个训练后完整评测", len(full_tuned) >= 1, str(sorted(full_tuned)))
    check("IMPROVEMENTS.md 存在", (ROOT / "IMPROVEMENTS.md").exists())
    check("REPORT.md 存在", (ROOT / "REPORT.md").exists())
    check("最终 adapter 存在", (
        ROOT / "outputs/adapters/final/adapters.safetensors").exists()
        or any((ROOT / "outputs/adapters" / t / "adapters.safetensors").exists()
               for t in full_tuned),
        str(sorted(full_tuned)))
    check("program.md 未被修改", _program_md_untouched())

    n_fail = sum(1 for _, c, _ in CHECKS if not c)
    print(f"\n{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed")
    if n_fail:
        raise SystemExit(1)
    print("\n全部 program.md 要求已满足。")


def _program_md_untouched() -> bool:
    """program.md is read-only per the goal; compare against the recorded MD5
    taken before any work began."""
    EXPECTED_MD5 = "4b5402f13758ce173d17c2c1ad716d94"
    try:
        out = subprocess.run(["md5", "-q", str(ROOT / "program.md")],
                             capture_output=True, text=True, check=True).stdout.strip()
        return out == EXPECTED_MD5
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    main()
