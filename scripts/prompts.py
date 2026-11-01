"""Shared prompts and rendering helpers for Zhenhuan roleplay training/eval.

The exact same system prompt is used for both training and evaluation so that
the model is calibrated to the prompt it will see at deployment.
"""

from __future__ import annotations

# NOTE: This prompt is baked into every training sample AND every benchmark
# request. Keep it stable within an experiment; if you change it, rebuild the
# training data and re-run the benchmark.
SYSTEM_PROMPT = (
    "你是电视剧《甄嬛传》中的甄嬛。请全程以甄嬛的身份、口吻和心境与对方对话，"
    "自称为“臣妾”（对皇上）或“我”，依情境称呼对方为“皇上”“娘娘”“妹妹”等。"
    "你的语言含蓄典雅、绵里藏针，符合清代后宫语境，可化用诗词典故；"
    "不使用现代白话、网络流行语、英文或表情符号。"
    "直接给出甄嬛的回应，不作任何解释、旁白、动作描写或剧情复述，不输出思考过程。"
)

# Shorter prompt used for the "no system prompt" robustness track.
SYSTEM_PROMPT_SHORT = "你是《甄嬛传》中的甄嬛，请以甄嬛的身份回应。"

# We only train on assistant turns; the loss over system/user tokens (and the
# leading ``<think>\\n\\n</think>\\n\\n`` emitted by Qwen3's template when
# enable_thinking=False) is masked out.
MASK_PROMPT = True


def build_messages(user_text: str, *, with_system: bool = True) -> list[dict]:
    messages: list[dict] = []
    if with_system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_text})
    return messages
