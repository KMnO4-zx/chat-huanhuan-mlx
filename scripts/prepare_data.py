"""Build mlx-lm chat-format SFT data from dataset/huanhuan.json.

Usage:
    uv run python scripts/prepare_data.py --out-dir data/v1
    uv run python scripts/prepare_data.py --out-dir data/v2 --dedup-instruction \\
        --augment dataset/adversarial_aug.json
    uv run python scripts/prepare_data.py --out-dir data/v3 --quality-filter \\
        --min-output-chars 24 --dedup-instruction --augment \\
        dataset/adversarial_aug.json dataset/persona_v3.json

Each output line is {"messages": [...]} which mlx_lm's ChatDataset consumes.
The system prompt comes from scripts/prompts.py so train and eval stay in sync.

``--quality-filter`` (v3) keeps only raw samples that are (a) long enough to
carry content, (b) anchored to 甄嬛's persona/world, and (c) not speaking AS
another character. It applies to the raw corpus only; hand-written augment
files are already curated.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from prompts import SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "dataset" / "huanhuan.json"

# --- v3 quality filter -------------------------------------------------
# Strong consort self-reference (almost certainly a 后妃, overwhelmingly the
# protagonist's register in this corpus).
PERSONA_ANCHOR = re.compile("臣妾|本宫|嫔妾|妾身|嬛儿|嬛嬛")
# 甄嬛's world: people/places/events that only appear in her storyline. A
# reply mentioning these is anchored to her even without a self-reference.
ZHENHUAN_WORLD = re.compile(
    "浣碧|槿汐|果郡王|允礼|温实初|温太医|碎玉轩|甘露寺|凌云峰|玉娆|甄远道"
    "|眉庄|梅庄|安陵容|杏花微雨|惊鸿舞|纯元|先皇后|长街|莞嫔|莞贵人|莞常在"
)
# Self-identification as *another* palace role: "我是/我叫 X 的侧福晋" etc.
OTHER_SELF = re.compile(
    r"我(?:是|叫)[^，。！？\n]{1,10}(?:的)?"
    r"(?:福晋|侧福晋|答应|常在|贵人|格格|奴才|奴婢|太医|小主|娘娘|妃|嫔)"
)


def quality_ok(item: dict, min_chars: int) -> bool:
    out = item["output"]
    if len(out) < min_chars:
        return False
    if OTHER_SELF.search(out):
        return False
    return bool(PERSONA_ANCHOR.search(out) or ZHENHUAN_WORLD.search(out))


def load_raw() -> list[dict]:
    with open(RAW_PATH, encoding="utf-8") as f:
        data = json.load(f)
    cleaned = []
    for item in data:
        instruction = (item.get("instruction") or "").strip()
        output = (item.get("output") or "").strip()
        if not instruction or not output:
            continue
        # Guard against leaked reasoning traces in the corpus.
        if "<think" in output.lower() or "</think" in output.lower():
            continue
        cleaned.append({"instruction": instruction, "output": output})
    return cleaned


def dedup(data: list[dict], by_instruction: bool) -> list[dict]:
    seen: set[str] = set()
    out = []
    for item in data:
        key = item["instruction"] if by_instruction else (
            item["instruction"] + "\x00" + item["output"]
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--valid-ratio", type=float, default=0.06)
    p.add_argument("--min-output-chars", type=int, default=1)
    p.add_argument("--quality-filter", action="store_true",
                   help="Raw-only: keep replies anchored to 甄嬛's persona/world, "
                        "long enough to carry content, and not spoken AS another "
                        "character (see PERSONA_ANCHOR/ZHENHUAN_WORLD/OTHER_SELF).")
    p.add_argument("--dedup-instruction", action="store_true",
                   help="Keep only the first output per instruction "
                        "(removes ~238 duplicate/ambiguous prompts).")
    p.add_argument("--no-dedup", action="store_true",
                   help="Skip dedup entirely. Needed when an augment file is "
                        "passed multiple times for intentional oversampling "
                        "(exact-pair dedup would collapse the repeats).")
    p.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    p.add_argument("--augment", nargs="+", default=None,
                   help="One or more {instruction, output} JSON lists merged "
                        "into the corpus (e.g. adversarial + persona Q&A).")
    args = p.parse_args()

    data = load_raw()
    n_raw = len(data)

    n_quality_in = len(data)
    if args.quality_filter:
        data = [d for d in data if quality_ok(d, args.min_output_chars)]
    n_after_quality = len(data)

    n_aug = 0
    if args.augment:
        for aug_name in args.augment:
            aug_path = ROOT / aug_name
            with open(aug_path, encoding="utf-8") as f:
                aug = json.load(f)
            aug = [
                {"instruction": a["instruction"].strip(),
                 "output": a["output"].strip()}
                for a in aug
                if a.get("instruction", "").strip() and a.get("output", "").strip()
            ]
            data.extend(aug)
            n_aug += len(aug)

    if args.min_output_chars > 1:
        data = [d for d in data if len(d["output"]) >= args.min_output_chars]
    n_after_len = len(data)

    if args.no_dedup:
        n_after_dedup = len(data)
    else:
        data = dedup(data, by_instruction=args.dedup_instruction)
        n_after_dedup = len(data)

    rng = random.Random(args.seed)
    rng.shuffle(data)

    n_valid = max(1, int(len(data) * args.valid_ratio))
    valid, train = data[:n_valid], data[n_valid:]

    def render(item: dict) -> dict:
        return {
            "messages": [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["output"]},
            ]
        }

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("valid", valid)):
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for item in split:
                f.write(json.dumps(render(item), ensure_ascii=False) + "\n")

    out_counts = Counter(d["output"] for d in data)
    manifest = {
        "source": str(RAW_PATH.relative_to(ROOT)),
        "n_raw": n_raw,
        "quality_filter": args.quality_filter,
        "n_before_quality_filter": n_quality_in,
        "n_after_quality_filter": n_after_quality,
        "n_augment": n_aug,
        "augment_path": args.augment,
        "n_after_len_filter": n_after_len,
        "n_after_dedup": n_after_dedup,
        "n_train": len(train),
        "n_valid": len(valid),
        "seed": args.seed,
        "valid_ratio": args.valid_ratio,
        "min_output_chars": args.min_output_chars,
        "dedup_instruction": args.dedup_instruction,
        "no_dedup": args.no_dedup,
        "system_prompt_sha": __import__("hashlib").sha256(
            args.system_prompt.encode()
        ).hexdigest()[:16],
        "top_outputs": out_counts.most_common(5),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
