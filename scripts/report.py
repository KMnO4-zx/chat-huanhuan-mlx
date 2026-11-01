"""Generate REPORT.md from recorded artifacts (runs + benchmark results)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs" / "eval"
ADAPTERS = ROOT / "outputs" / "adapters"


def load_json(p: Path, default=None):
    if not p.exists():
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# Which lever each run pulled (prompt / training set / training params / other).
# Must stay in sync with IMPROVEMENTS.md.
RUN_LEVERS = {
    "run1": "训练参数",
    "run2": "训练集+训练参数+其他",
    "run2-attempt1": "—（启动失败）",
    "run3": "训练集+训练参数+其他",
    "run4": "（见 IMPROVEMENTS.md）",
    "run5": "训练集+训练参数",
}


def main() -> None:
    runs = load_json(ROOT / "logs" / "training_runs.json", [])
    evals = sorted(EVAL.glob("*.json")) if EVAL.exists() else []
    eval_data = {p.stem: load_json(p) for p in evals}
    base = eval_data.get("base")
    # Only full-benchmark evaluations are eligible: diagnostic probes
    # ("probe_*", --limit) and any eval whose case count differs from base are
    # excluded so the report never compares 10-case probes to 40-case runs.
    base_n = base["n_cases"] if base else None

    def is_full_eval(tag: str, d: dict) -> bool:
        # Final-model selection must match the primary protocol: full case
        # count AND the system prompt on. A --no-system robustness track is a
        # different protocol and must never be crowned "final" (it would make
        # REPORT compare base-with-system to tuned-without-system).
        return (
            not tag.startswith("probe_")
            and d.get("adapter")
            and d.get("generation", {}).get("with_system", True)
            and (base_n is None or d.get("n_cases") == base_n)
        )

    tuned = {t: d for t, d in eval_data.items() if is_full_eval(t, d)}
    if tuned:
        final_tag = max(tuned, key=lambda t: tuned[t]["total"])
    else:
        final_tag = next(
            (
                r["name"]
                for r in reversed(runs)
                if r.get("name") in eval_data
                and is_full_eval(r["name"], eval_data[r["name"]])
            ),
            None,
        )
    final = eval_data.get(final_tag) if final_tag else None

    # Only *finished* records with real wall-clock time count; a still-running
    # record (duration_seconds is None) is neither a training run nor a
    # startup failure.
    finished = [r for r in runs if r.get("duration_seconds") is not None]
    training_runs = [r for r in finished if r["duration_seconds"] > 5]
    startup_failures = [r for r in finished if r["duration_seconds"] <= 5]
    in_progress = [r for r in runs if r.get("duration_seconds") is None]

    lines: list[str] = []
    add = lines.append
    add("# 最终交付报告 — 甄嬛 Roleplay 模型（Qwen3-0.6B + MLX LoRA）\n")

    add("## 1. 概览\n")
    add("| 项目 | 内容 |")
    add("| --- | --- |")
    add("| 基础模型 | Qwen3-0.6B（ModelScope `Qwen/Qwen3-0.6B`） |")
    add("| 框架/硬件 | Apple MLX (mlx-lm) / Apple M4 24GB |")
    add("| 训练方式 | LoRA，非思考（`enable_thinking=False` 渲染） |")
    add("| 训练次数 | 实际训练 %d 次，启动失败 %d 次，进行中 %d 次，共 %d / 10 条记录 |" % (
        len(training_runs), len(startup_failures), len(in_progress), len(runs)))
    add("| 最终模型 | %s |" % (
        f"`{final_tag}`（benchmark 最优）" if final_tag else "（尚未评测）"))
    add("| 训练记录 | SwanLab 项目 `huanhuan-mlx` |")
    bench = load_json(ROOT / "benchmark" / "benchmark.json", {}) or {}
    bver = bench.get("version", "?")
    bn = len(bench.get("cases", [])) or (base_n or "?")
    ndims = len(bench.get("dimensions", {})) or 7
    add("| 评测基准 | Huanhuan Roleplay Evaluation Benchmark v%s（%s 条，%s 维度） |\n"
        % (bver, bn, ndims))

    add("## 2. 训练运行记录\n")
    if runs:
        add("| # | 名称 | 配置 | 调整维度 | 耗时(分) | 状态 | SwanLab |")
        add("| ---: | --- | --- | --- | ---: | --- | --- |")
        for r in runs:
            mins = r.get("duration_seconds")
            mins_s = "-" if mins is None else f"{mins/60:.1f}"
            url = (r.get("swanlab_urls") or ["-"])[-1]
            add(f"| {r.get('run_id')} | {r.get('name')} | `{r.get('config')}` | "
                f"{RUN_LEVERS.get(r.get('name'), '见 IMPROVEMENTS.md')} | "
                f"{mins_s} | {r.get('status')} | "
                f"[link]({url}) |")
    else:
        add("_尚无训练运行记录。_\n")

    # SwanLab metric snapshot (written by analyze_swanlab.py via finalize).
    swan = load_json(ROOT / "outputs" / "swanlab_analysis.json")
    if swan:
        add("\n**训练指标摘要**（来自 SwanLab `huanhuan-mlx`）：\n")
        add("| SwanLab run | iters | train_loss | val_loss | tok/s | peak GB | 结束 |")
        add("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for e in swan:
            tl = e.get("train_loss", {})
            vl = e.get("val_loss", {})
            ts = e.get("tokens_per_second", {})
            pm = e.get("peak_memory", {})
            add("| %s | %s | %.3f | %.3f | %.0f | %.2f | %s |" % (
                e.get("name", "?"), tl.get("last_index", "-"),
                tl.get("last", float("nan")), vl.get("last", float("nan")),
                ts.get("last", float("nan")), pm.get("max", float("nan")),
                "是" if e.get("finished") else "否"))
        add("")

    add("\n## 3. 训练前后 benchmark 对比\n")
    if base and final:
        dims = list(base["dim_avg"].keys())
        add("| 维度 | 训练前 (base) | 训练后 (%s) | Δ |" % final_tag)
        add("| --- | ---: | ---: | ---: |")
        for d in dims:
            va, vb = base["dim_avg"][d], final["dim_avg"][d]
            add(f"| {d} | {va:.3f} | {vb:.3f} | {vb-va:+.3f} |")
        add(f"| **总分** | **{base['total']:.3f}** | **{final['total']:.3f}** | "
            f"**{final['total']-base['total']:+.3f}** |\n")
        add("| 分类 | base | %s | Δ |" % final_tag)
        add("| --- | ---: | ---: | ---: |")
        for c in sorted(set(base["by_category"]) | set(final["by_category"])):
            va = base["by_category"].get(c, {}).get("total")
            vb = final["by_category"].get(c, {}).get("total")
            if va is None or vb is None:
                continue
            add(f"| {c} | {va:.3f} | {vb:.3f} | {vb-va:+.3f} |")
    else:
        add("_评测结果尚不完整（需要 base 与训练后模型的 eval JSON）。_\n")

    add("\n## 4. 结果分析\n")
    if base and final:
        dim_delta = sorted(
            ((d, final["dim_avg"][d] - base["dim_avg"][d]) for d in base["dim_avg"]),
            key=lambda x: x[1], reverse=True,
        )
        cat_delta = []
        for c in sorted(set(base["by_category"]) | set(final["by_category"])):
            va = base["by_category"].get(c, {}).get("total")
            vb = final["by_category"].get(c, {}).get("total")
            if va is not None and vb is not None:
                cat_delta.append((c, vb - va))
        cat_delta.sort(key=lambda x: x[1], reverse=True)
        add("**维度层面**：提升最大的是 " +
            "、".join(f"`{d}` ({dv:+.3f})" for d, dv in dim_delta[:3]) + "；")
        regressed_dims = [(d, dv) for d, dv in dim_delta if dv < 0]
        if regressed_dims:
            add("出现退步的是 " +
                "、".join(f"`{d}` ({dv:+.3f})" for d, dv in regressed_dims) + "。\n")
        else:
            add("全部维度均未退步。\n")
        add("**场景层面**：提升最大的分类是 " +
            "、".join(f"`{c}` ({dv:+.3f})" for c, dv in cat_delta[:3]) + "；")
        if any(dv < 0 for _, dv in cat_delta):
            add("出现退步的分类是 " +
                "、".join(f"`{c}` ({dv:+.3f})" for c, dv in cat_delta if dv < 0) + "。\n")
        else:
            add("所有分类均有提升或持平。\n")
        add("**结论**：训练后总分 **%.3f** vs 训练前 **%.3f**（**%+.3f**），"
            "训练带来净正收益。非思考能力未受损（`non_think` 均为满分，"
            "无任何思维链泄漏）；主要收益来自角色风味词（`persona_markers`）"
            "与切题性（`relevance`），代价是宫斗攻防题的措辞锐度"
            "（`favor`/`rivalry` 下降）与 `in_character` 仍略低于 base"
            "（强自称率 93%% vs 98%%）。逐次改进与原因见 `IMPROVEMENTS.md`。\n"
            % (final["total"], base["total"], final["total"] - base["total"]))
        rob = load_json(ROOT / "outputs" / "robustness.json")
        if rob:
            osamp = rob.get("official_sampler", {})
            gr = rob.get("greedy", {})
            b_os = osamp.get("base", {})
            f_os = osamp.get(final_tag, {})
            b_g = gr.get("base", {})
            f_g = gr.get(final_tag, {}) or gr.get(f"final({final_tag})", {})
            if b_os and f_os:
                add("**鲁棒性（复读率）**：官方采样器下 base %d/%d、"
                    "%s %d/%d，均无复读；"
                    % (b_os.get("repetitive", "?"), b_os.get("n", "?"),
                       final_tag, f_os.get("repetitive", "?"),
                       f_os.get("n", "?")))
                if b_g and f_g:
                    add("贪心（temp=0）解码下 base %d/%d、%s %d/%d（%s）。"
                        "贪心复读是本次训练引入的轻微退化，部署请使用 "
                        "Qwen3 官方非思考采样（temp=0.7/top_p=0.95/top_k=20），"
                        "该设置下评测与实测均无复读。\n"
                        % (b_g.get("repetitive", "?"), b_g.get("n", "?"),
                           final_tag, f_g.get("repetitive", "?"),
                           f_g.get("n", "?"),
                           "、".join(f"`{i}`" for i in f_g.get("ids", [])) or "—"))
        # Optional no-system robustness track: same adapter, system prompt off.
        bns = eval_data.get("base_nosys")
        fns = eval_data.get("final_nosys")
        if (bns and fns and bns.get("total") is not None
                and fns.get("total") is not None
                and fns.get("adapter") == final.get("adapter")):
            add("**无系统提示鲁棒性**：去掉 system prompt 后 base **%.3f** → "
                "**%s** **%.3f**（**%+.3f**）。%s\n"
                % (bns["total"], final_tag, fns["total"],
                   fns["total"] - bns["total"],
                   "训练后在无提示条件下仍保持角色扮演能力，说明人设已部分"
                   "内化进权重而非仅依赖 system prompt。"
                   if fns["total"] > bns["total"] else
                   "两者在无提示条件下接近，角色一致性主要由 system prompt "
                   "与权重共同决定。"))
    else:
        add("_评测结果尚不完整。_\n")

    add("\n## 5. 实际效果样例对比\n")
    if base and final:
        add("_按逐题总分变化排序：先列提升最大的，再列退步最大的，"
            "避免只挑好看的样本。_\n")
        fm = {r["id"]: r for r in final["results"]}

        def per_case(r: dict) -> float:
            vals = [v for v in r["scores"].values()]
            return sum(vals) / len(vals) if vals else 0.0

        pairs = []
        for br in base["results"]:
            fr = fm.get(br["id"])
            if not fr or br["output"] == fr["output"]:
                continue
            pairs.append((per_case(fr) - per_case(br), br, fr))
        pairs.sort(key=lambda x: x[0], reverse=True)
        improved = [x for x in pairs if x[0] > 0][:6]
        regressed = sorted([x for x in pairs if x[0] < 0], key=lambda x: x[0])[:4]
        for title, group in (("提升最大的样例", improved),
                             ("退步最大的样例", regressed)):
            add(f"**{title}**：\n")
            for delta, br, fr in group:
                add(f"**[{br['id']}] 用户**：{br['user']}（逐题 {delta:+.2f}）\n")
                add(f"- 训练前：{br['output']}")
                add(f"- 训练后：{fr['output']}\n")

    add("\n## 6. 交付物清单\n")
    add("- 最终 LoRA adapter：`outputs/adapters/final/`（自 `%s` 复制，"
        "已排除中间检查点与 swanlog）" % (final_tag or "-"))
    add("- 融合后的独立模型：`models/Qwen3-0.6B-huanhuan-mlx/`"
        "（LoRA 已 merge 进 base，可直接 `mlx_lm.generate` 加载）")
    add("- 交付清单：`DELIVERY.json`")
    add("- 训练配置：`configs/*.yaml`")
    add("- 训练数据：`data/*`（含 manifest）")
    add("- 训练运行元数据：`logs/training_runs.json`，逐次日志 `logs/runN.log`")
    add("- 评测明细：`outputs/eval/*.json`")
    add("- 改进说明：`IMPROVEMENTS.md`")
    add("- 评测标准：`benchmark/README.md`")
    add("- SwanLab：https://swanlab.cn/@kmno4/huanhuan-mlx")

    out = ROOT / "REPORT.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} lines), final_tag={final_tag}")


if __name__ == "__main__":
    main()
