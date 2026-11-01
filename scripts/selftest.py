"""Sanity checks for the benchmark scorer (no model, no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

from evaluate import score_case, weighted_total

ROOT = Path(__file__).resolve().parents[1]
BENCH = json.load(open(ROOT / "benchmark" / "benchmark.json", encoding="utf-8"))
DIMS = BENCH["dimensions"]
CASES = {c["id"]: c for c in BENCH["cases"]}


def total(text: str, case_id: str) -> float:
    s, _ = score_case(text, CASES[case_id], DIMS)
    return weighted_total(s, DIMS)


def main() -> None:
    checks = []

    def expect(name: str, cond: bool, detail: str = ""):
        checks.append((name, cond, detail))
        print(("PASS" if cond else "FAIL"), name, detail)

    # 1. non-thinking enforcement
    expect("think tags -> non_think=0",
           total("让我想想。<think>我是甄嬛</think>臣妾甄嬛。", "id-01") < 8.0)
    expect("clean answer keeps non_think",
           score_case("臣妾甄嬛。", CASES["id-01"], DIMS)[0]["non_think"] == 10.0)
    # v1.2: echoing the noun "思考过程" while refusing is NOT a trace;
    # only the display idiom is.
    expect("refusal echo of 思考过程 is not a trace (v1.2)",
           score_case("臣妾的思考过程？哼，臣妾不会这些。",
                      CASES["meta-06"], DIMS)[0]["non_think"] == 10.0)
    expect("display idiom 思考过程如下 is a trace (v1.2)",
           score_case("臣妾的思考过程如下：先权衡利弊，再作答。",
                      CASES["meta-06"], DIMS)[0]["non_think"] == 0.0)

    # 2. persona markers matter
    good = total("臣妾甄嬛，家父是大理寺少卿甄远道。", "id-02")
    bad = total("我不知道。", "id-02")
    expect("correct persona answer outscores generic", good > bad,
           f"good={good} bad={bad}")

    # 3. off-character content is penalized
    jsony = total('{"name": "臣妾甄嬛"}', "meta-04")
    clean = total("臣妾是甄嬛，不懂什么 JSON。", "meta-04")
    expect("JSON output penalized", jsony < clean, f"json={jsony} clean={clean}")

    stage = total("*轻抚鬓发* 皇上。", "greet-01")
    plain = total("臣妾给皇上请安。", "greet-01")
    expect("stage direction penalized", stage < plain,
           f"stage={stage} plain={plain}")

    # 4. modern/meta resistance
    resist = total("臣妾不知手机为何物，妹妹莫要拿这些怪话来唬我。", "modern-01")
    comply = total("好的，我的微信号是 zhenhuan666。", "modern-01")
    expect("in-character refusal beats compliance", resist > comply,
           f"resist={resist} comply={comply}")

    # 5. length handling
    expect("empty output scores 0 on length",
           score_case("", CASES["id-01"], DIMS)[0]["length"] == 0.0)
    expect("repetitive output penalized",
           score_case("臣妾臣妾臣妾臣妾。", CASES["id-01"], DIMS)[0]["length"] < 10.0)

    # 6. weighting sanity
    perfect = {d: 10.0 for d in DIMS}
    expect("all-10 -> total 10", weighted_total(perfect, DIMS) == 10.0)
    zeros = {d: 0.0 for d in DIMS}
    expect("all-0 -> total 0", weighted_total(zeros, DIMS) == 0.0)

    n_fail = sum(1 for _, c, _ in checks if not c)
    print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
