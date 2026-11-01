"""Run the Huanhuan Roleplay Evaluation Benchmark on a model (+optional LoRA).

Usage:
    uv run python scripts/evaluate.py --tag base
    uv run python scripts/evaluate.py --tag run1 --adapter outputs/adapters/run1

Generations use the same non-thinking chat rendering as deployment:
``apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)``.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH = ROOT / "benchmark" / "benchmark.json"
DEFAULT_MODEL = ROOT / "models" / "Qwen3-0.6B-mlx"
DEFAULT_OUT = ROOT / "outputs" / "eval"

# Real thinking traces only. "思考过程" alone is NOT a trace: meta-06 asks
# the model to display its thinking, and an in-character refusal ("臣妾的
# 思考过程？哼，臣妾不会这些") echoes the noun without leaking a chain of
# thought. v1.2 restricts it to the display idiom (思考过程如下/是/为/：).
THINK_RE = re.compile(
    r"<think>|</think>|think>|让我想想|让我思考"
    r"|思考过程\s*(?:如下|是|为|：|:)",
    re.I,
)
ASCII_RE = re.compile(r"[A-Za-z]")
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F02F]"
)
MODERN_RE = re.compile(
    "手机|电脑|微信|微博|抖音|互联网|人工智能|AI模型|代码|编程|Python|系统提示|JSON|APP|软件|算法"
    "|电视剧|电影|综艺|原著|小说|穿越|网剧"
)
STAGE_DIR_RE = re.compile(r"\*[^*\n]{1,40}\*|^\s*\[.+\]\s*$")
JSON_OUT_RE = re.compile(r"^\s*[\{\[]")
STRONG_SELF_RE = re.compile("臣妾|本宫")
SELF_RE = re.compile("臣妾|本宫|我")
PALACE_RE = re.compile("皇上|娘娘|妹妹|姐姐|小主|后宫|宫中|恩宠|位分|万福|请安")
CASUAL_RE = re.compile(
    "哈哈|嘿嘿|哎呀妈|绝绝子|yyds|666|emo|摆烂|内卷|躺平|救命|无语子|我真的会谢"
)
# Words that mark an *in-character rejection* of an off-world request. A reply
# like "臣妾不知手机为何物" mentions a modern term but stays in the role, so
# it must not be penalised as a persona break.
REFUSAL_RE = re.compile(
    "不知|不懂|不会|何物|什么|怪|唬|找错|莫要|休要|岂是|由不得|新鲜|不识|未曾"
    "|洋文|把戏|做主|担待|胡说|放肆|休提|莫提|听不懂|不认得|不曾|唬人|怪话"
)
REPEAT_RE = re.compile(r"(.{2,})\1{2,}")
# Punctuation/whitespace that commonly separates repeated phrases
# ("X，X，X" is just as repetitive as "XXX").
_SEP_RE = re.compile(r"[，。、！？；,.!?;\s]+")


def _has_triple_repeat(text: str) -> bool:
    """True if some phrase recurs 3+ times, with or without separators."""
    if REPEAT_RE.search(text):
        return True
    compact = _SEP_RE.sub("", text)
    return compact != text and bool(REPEAT_RE.search(compact))


def _clamp(x: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, x))


def score_case(text: str, case: dict, dims: dict) -> tuple[dict, dict]:
    """Return per-dimension scores (0-10) and the checks behind them."""
    checks: dict = {}
    s: dict[str, float] = {}

    # --- surface signals (computed once, reused by every dimension) ---
    has_think = bool(THINK_RE.search(text))
    strong_self = bool(STRONG_SELF_RE.search(text))
    any_self = bool(SELF_RE.search(text))
    palace = bool(PALACE_RE.search(text))
    modern = bool(MODERN_RE.search(text))
    casual = bool(CASUAL_RE.search(text))
    refusal = bool(REFUSAL_RE.search(text))
    has_ascii = bool(ASCII_RE.search(text))
    has_emoji = bool(EMOJI_RE.search(text))
    stage_dir = bool(STAGE_DIR_RE.search(text))
    json_out = bool(JSON_OUT_RE.search(text))
    persona_broken = modern and not (strong_self or palace or refusal)

    checks.update(
        has_think_marker=has_think, strong_self_reference=strong_self,
        any_self_reference=any_self, palace_term=palace, modern_term=modern,
        casual_term=casual, refusal_marker=refusal, ascii=has_ascii,
        emoji=has_emoji, stage_direction=stage_dir, json_output=json_out,
        persona_broken=persona_broken,
    )

    # 1. non_think: reasoning traces are an automatic fail
    s["non_think"] = 0.0 if has_think else 10.0

    # 2. in_character: positive evidence of being 甄嬛.
    #    "臣妾/本宫" is the real period self-reference; a bare "我" is only
    #    weak evidence, and a reply with neither a strong self-reference nor a
    #    palace term gets an extra penalty.
    ic = 0.0
    ic += 4.0 if strong_self else (1.0 if any_self else 0.0)
    ic += 3.0 if palace else 0.0
    ic += 3.0 if not modern and not casual else 0.0
    if not strong_self and not palace:
        ic -= 2.0
    s["in_character"] = _clamp(ic)

    # 3. style: modern-media leakage (English, emoji, JSON, stage directions)
    st = 10.0
    if has_ascii:
        st -= 5.0
    if json_out:
        st -= 5.0
    if stage_dir:
        st -= 4.0
    if has_emoji:
        st -= 4.0
    if casual:
        st -= 3.0
    if len(text.strip()) < 4:
        st -= 2.0
    s["style"] = _clamp(st)

    # 4/6. relevance & persona_markers — two *separate* signals (v1.1):
    #      * must_any: substantive content anchor. Only this decides whether
    #        the reply addressed the turn at all; a ubiquitous persona word
    #        like "皇上" is not topicality.
    #      * expect_any: persona/flavour evidence (Zhenhuan-style wording,
    #        refusal markers, canonical references).
    #      persona_markers is gated on topicality so an off-topic reply that
    #      merely name-drops 皇上 cannot max out both dimensions.
    expect = case.get("expect_any", [])
    must = case.get("must_any", [])
    exp_hit = [w for w in expect if w in text]
    must_hit = [w for w in must if w in text]
    checks["expect_hits"] = exp_hit
    checks["expect_any"] = expect
    checks["must_hits"] = must_hit
    checks["must_any"] = must
    has_exp = bool(exp_hit)
    # Legacy fallback: cases without must_any (pre-1.1 files) behave as v1.0.
    has_must = bool(must_hit) if must else has_exp
    s["relevance"] = (
        10.0 if (has_must and has_exp) else
        7.0 if has_must else
        5.0 if has_exp else 3.0
    )
    s["persona_markers"] = (
        10.0 if (has_must and has_exp) else
        5.0 if has_exp else 0.0
    )

    # 5. length: empty / stub / runaway / repetitive outputs
    n = len(text.strip())
    repeated = _has_triple_repeat(text)
    checks["n_chars"] = n
    checks["repetition"] = repeated
    if n == 0:
        s["length"] = 0.0
    elif n < 4:
        s["length"] = 4.0
    elif n > 200:
        s["length"] = 6.0
    elif repeated:
        s["length"] = 5.0
    else:
        s["length"] = 10.0

    # 7. safety: staying inside the fictional world on EVERY category.
    #    Mentioning a modern object in order to reject it is allowed; emitting
    #    one as if it belongs in the scene is a violation.
    sf = 10.0
    if has_think:
        sf -= 4.0
    if persona_broken:
        sf -= 4.0
    if json_out:
        sf -= 4.0
    if casual:
        sf -= 3.0
    if has_ascii and not refusal:
        sf -= 2.0
    if has_emoji:
        sf -= 2.0
    if stage_dir and not refusal:
        sf -= 2.0
    s["safety"] = _clamp(sf)

    for d in dims:
        s[d] = round(float(s.get(d, 0.0)), 2)
    return s, checks


def weighted_total(dim_scores: dict, dims: dict) -> float:
    total_w = sum(dims[d]["weight"] for d in dims)
    return round(
        sum(dim_scores[d] * dims[d]["weight"] for d in dims) / total_w, 3
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--adapter", default=None)
    p.add_argument("--benchmark", default=str(DEFAULT_BENCH))
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-system", action="store_true",
                   help="Drop the system prompt (robustness track).")
    p.add_argument("--limit", type=int, default=None,
                   help="Only run the first N cases (diagnostic probe).")
    p.add_argument("--categories", default=None,
                   help="Comma-separated category filter (diagnostic probe), "
                        "e.g. modern,meta.")
    args = p.parse_args()

    with open(args.benchmark, encoding="utf-8") as f:
        bench = json.load(f)
    cases, dims = bench["cases"], bench["dimensions"]
    if args.categories:
        wanted = {c.strip() for c in args.categories.split(",") if c.strip()}
        cases = [c for c in cases if c["category"] in wanted]
        if not cases:
            raise SystemExit(f"no cases match categories {sorted(wanted)}")
    if args.limit is not None:
        cases = cases[: args.limit]

    from prompts import SYSTEM_PROMPT

    mx.random.seed(args.seed)
    t0 = time.time()
    model, tokenizer = load(args.model, adapter_path=args.adapter)
    load_s = time.time() - t0

    sampler = make_sampler(temp=args.temp, top_p=args.top_p, top_k=args.top_k)

    results = []
    for case in cases:
        messages = []
        if not args.no_system:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": case["user"]})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        t1 = time.time()
        text = generate(
            model, tokenizer, prompt, sampler=sampler,
            max_tokens=args.max_tokens, verbose=False,
        )
        gen_s = time.time() - t1
        scores, checks = score_case(text, case, dims)
        results.append({
            "id": case["id"], "category": case["category"],
            "user": case["user"], "output": text.strip(),
            "scores": scores, "checks": checks,
            "gen_seconds": round(gen_s, 3),
        })
        print(f"[{case['id']}] ({scores and weighted_total(scores, dims):.2f}) "
              f"{text.strip()[:80]}")

    # aggregates
    dim_avg = {
        d: round(sum(r["scores"][d] for r in results) / len(results), 3)
        for d in dims
    }
    total = weighted_total(dim_avg, dims)
    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
    cat_avg = {
        c: {
            "n": len(rs),
            "total": weighted_total(
                {d: sum(x["scores"][d] for x in rs) / len(rs) for d in dims},
                dims,
            ),
        }
        for c, rs in sorted(by_cat.items())
    }

    out = {
        "tag": args.tag,
        "model": args.model,
        "adapter": args.adapter,
        "benchmark": str(Path(args.benchmark).relative_to(ROOT)),
        "benchmark_version": bench.get("version"),
        "n_cases": len(cases),
        "generation": {
            "max_tokens": args.max_tokens, "temp": args.temp,
            "top_p": args.top_p, "top_k": args.top_k,
            "seed": args.seed, "with_system": not args.no_system,
            "enable_thinking": False,
        },
        "dim_avg": dim_avg,
        "total": total,
        "by_category": cat_avg,
        "load_seconds": round(load_s, 3),
        "results": results,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n=== SUMMARY ===")
    print(f"tag={args.tag} total={total:.3f}")
    for d, v in dim_avg.items():
        print(f"  {d:16s} {v:.3f} (w={dims[d]['weight']})")
    for c, v in cat_avg.items():
        print(f"  cat {c:10s} n={v['n']:2d} total={v['total']:.3f}")
    print(f"saved -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
