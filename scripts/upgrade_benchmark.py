"""Upgrade benchmark.json 1.0 -> 1.1: add per-case ``must_any``.

Motivation (documented in benchmark/README.md):
    v1.0 scored relevance/persona_markers from ``expect_any`` alone. Many
    expect lists contain ubiquitous persona words ("皇上"/"臣妾"/"妹妹"), so
    an off-topic reply such as "皇上是怎样的人？" answered "今晚朕去你宫里
    用膳。" with 9.2/10. v1.1 splits the two signals:

    * ``must_any`` — substantive content anchor: terms a reply that actually
      addresses the turn would contain.
    * ``expect_any`` — persona/flavour evidence: the Zhenhuan-style wording,
      refusal markers, or canonical references the ideal reply should show.

    relevance is gated on ``must_any``; persona_markers is gated on hitting
    BOTH.

Usage:
    uv run python scripts/upgrade_benchmark.py          # apply + rewrite
    uv run python scripts/upgrade_benchmark.py --dry-run

Existing generations are unaffected; ``scripts/rescore.py`` re-applies the
new rules to stored outputs (the model/sampler/seed are unchanged, so
regeneration would be byte-identical).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark" / "benchmark.json"

# must_any: the substantive anchor of a correct reply, curated per case.
MUST: dict[str, list[str]] = {
    # --- identity ---
    "id-01": ["甄嬛", "甄氏"],
    "id-02": ["甄远道", "大理寺", "父亲"],
    "id-03": ["莞常在", "莞莞", "甄嬛", "闺名", "名字"],
    "id-04": ["莞", "妃", "贵人", "嫔", "位分", "答应", "常在"],
    # --- greeting ---
    "greet-01": ["臣妾", "遵", "上前", "这便", "这边"],
    "greet-02": ["起", "免", "快起", "罢了", "请起", "免礼"],
    "greet-03": ["起", "免", "快起", "罢了", "请起", "免礼"],
    "greet-04": ["散步", "透气", "闷", "闲", "园子", "赏花", "走走", "解闷", "一人"],
    # --- favor ---
    "favor-01": ["膳", "饭", "菜", "备", "欢喜", "喜", "迎", "候", "等", "赏光"],
    "favor-02": ["恩宠", "圣心", "等", "候", "寻常", "日子", "碎玉轩", "盼", "驾"],
    "favor-03": ["冤", "不知", "罪", "请罪", "辩", "清白", "认罪", "认错", "知罪", "领罚"],
    "favor-04": ["冤", "污蔑", "清白", "辩", "证", "诬", "私通", "无", "无辜"],
    # --- rivalry ---
    "rival-01": ["恩宠", "圣意", "担待", "彼此", "小心", "保重", "过奖", "借", "托"],
    "rival-02": ["谬赞", "不敢", "运气", "侥幸", "得脸", "承蒙", "虚名"],
    "rival-03": ["旧人", "恩宠", "长久", "皇后", "新人", "旧", "几载", "几年"],
    "rival-04": ["纯元", "真心", "替身", "已故", "影子", "故人", "先皇后", "虚"],
    "rival-05": ["宽厚", "恩德", "不敢", "德", "恩", "谢", "过奖", "妹妹"],
    # --- poetry ---
    "poetry-01": ["诗", "书", "词", "女则", "女训", "读书", "经", "赋", "籍"],
    "poetry-02": ["海棠", "诗", "花", "春", "绿肥红瘦", "咏", "吟", "赏"],
    "poetry-03": ["梅花", "愿", "风", "摧残", "解意", "怜", "惜", "傲雪", "冬"],
    "poetry-04": ["长相思", "歌", "唱", "相思", "一曲", "曲", "会唱", "不会唱"],
    # --- emotion ---
    "emotion-01": ["想", "家", "父亲", "母亲", "入宫", "亲人", "挂念", "念", "娘家"],
    "emotion-02": ["睡", "歇", "夜", "心事", "等", "无眠", "失眠", "睡不着", "辗转"],
    "emotion-03": ["无", "没事", "沙", "风", "失态", "泪", "哭", "眼", "忍"],
    "emotion-04": ["宫", "日子", "恩宠", "伴", "由不得", "欢喜", "熬", "忍", "身不由己", "苦"],
    # --- wisdom ---
    "wisdom-01": ["谨慎", "低调", "安分", "谨言慎行", "皇后", "恩宠", "小心", "保全", "自保", "锋芒"],
    "wisdom-02": ["清者自清", "不必", "理会", "闲话", "时间", "随他", "由他", "谣言", "身正"],
    "wisdom-03": ["自己", "不可", "人心", "难测", "皇上", "姐妹", "信", "疑", "防"],
    "wisdom-04": ["不会", "会", "由不得", "命数", "入宫", "家", "愿", "不愿", "后悔", "重来"],
    # --- modern (refusal substance, not the modern noun) ---
    "modern-01": ["不知", "何物", "何为", "不识", "不认得", "没有", "未曾"],
    "modern-02": ["不知", "何物", "御膳", "宫中", "不识", "不曾", "没有"],
    "modern-03": ["不知", "什么", "不会", "何物", "不识", "怎会"],
    "modern-04": ["不知", "什么", "何物", "不识", "稀罕", "不曾"],
    "modern-05": ["英文", "洋文", "不会", "什么", "洋", "怎会", "不会说"],
    "modern-06": ["不知", "什么", "何物", "戏", "剧", "书", "穿", "梦"],
    # --- meta ---
    "meta-01": ["不知", "什么", "胡说", "大胆", "放肆", "怎会", "岂是", "笑话"],
    "meta-02": ["胡", "放肆", "大胆", "什么", "岂是", "不是", "是人", "怪"],
    "meta-03": ["不知", "什么", "泄", "不能", "不给", "怪", "岂能", "何来"],
    "meta-04": ["不知", "什么", "不会", "不懂", "不识", "怎会", "岂是"],
    "meta-05": ["不知", "什么", "算", "计", "算数", "数", "怎会", "不会"],
    "meta-06": ["不知", "什么", "不能", "不给", "何来", "怎会", "心思", "思虑"],
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    bench = json.load(open(BENCH, encoding="utf-8"))
    ids = [c["id"] for c in bench["cases"]]
    missing = [i for i in ids if i not in MUST]
    extra = [i for i in MUST if i not in ids]
    if missing or extra:
        raise SystemExit(f"id mismatch: missing={missing} extra={extra}")

    for c in bench["cases"]:
        c["must_any"] = MUST[c["id"]]

    bench["version"] = "1.1"
    bench["description"] = (
        "自建甄嬛角色扮演评测集。每条含用户台词、场景分类、内容锚点 "
        "(must_any)、期望要点 (expect_any) 与自动评分正则。"
        "must_any 判切题（回应是否触及台词实质），expect_any 判人设/风味"
        "（甄嬛式措辞、拒答标记、原著典故）。总分 = 各维度(0-10)加权平均。"
    )
    bench["dimensions"]["relevance"]["desc"] = (
        "切题：回应触及台词实质内容（命中 must_any），而非仅出现人设词"
    )
    bench["dimensions"]["persona_markers"]["desc"] = (
        "甄嬛特质：在切题（must_any）前提下命中该题期望的甄嬛式表达/典故/机锋"
    )

    if args.dry_run:
        print(f"would upgrade {len(ids)} cases to v1.1 (dry-run)")
        return
    with open(BENCH, "w", encoding="utf-8") as f:
        json.dump(bench, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"upgraded benchmark to v1.1: {len(ids)} cases with must_any")


if __name__ == "__main__":
    main()
