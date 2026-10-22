###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Constants and helpers for GLM-5.3 thinking-budget injection.

Kept free of ``sglang`` / ``torch`` imports so the Anthropic Messages shim can
use these helpers without blocking the API event loop on first request.
"""

from __future__ import annotations

import os
from typing import Any, Literal

# dill STACK_GLOBAL for
# ``infera.engine.sglang.thinking_budget.Glm53ThinkingBudgetLogitProcessor``.
# Keep in sync with the Rust anthropic shim and
# ``test_processor_str_targets_glm53_class``.
GLM53_THINKING_BUDGET_PROCESSOR_STR = (
    '{"callable":"'
    "8004954e000000000000008c24696e666572612e656e67696e652e73676c616e672e7468"
    "696e6b696e675f627564676574948c21476c6d35335468696e6b696e674275646765744c"
    "6f67697450726f636573736f729493942e"
    '"}'
)

# GLM-5.3 tokenizer ids for <think>, </think>, and newline.
GLM53_THINKING_START_TOKEN_ID = 154841
GLM53_THINKING_END_TOKEN_ID = 154842
GLM53_NEW_LINE_TOKEN_ID = 198

# Router/API opt-in: inject custom_logit_processor only when set. Workers must
# also run with ``--enable-custom-logit-processor`` or SGLang rejects the field.
THINKING_BUDGET_ENV = "INFERA_ENABLE_THINKING_BUDGET"

ThinkingBudgetAction = Literal["newline", "end"]


def thinking_budget_injection_enabled() -> bool:
    """Return True when hard-cap injection is explicitly enabled."""
    raw = os.environ.get(THINKING_BUDGET_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def is_glm53_model(model: Any) -> bool:
    """Return True when ``model`` names a GLM-5.3 served id."""
    if not isinstance(model, str):
        return False
    m = model.lower().replace("_", "-")
    return "glm-5-3" in m or "glm-5.3" in m or "glm53" in m


# Tokens forced after the budget is reached: newline, then ``</think>``.
THINKING_FORCED_TOKENS = 2
# Upper bound of completion tokens kept for the answer after thinking ends.
MAX_ANSWER_RESERVE = 1024
# Smallest ``max_tokens`` that fits one thinking token, the forced tokens and one answer token.
MIN_MAX_TOKENS_FOR_BUDGET = 4


def thinking_budget_ceiling(max_tokens: int) -> int:
    """Return the largest budget that still leaves room to close thinking and answer."""
    answer_reserve = max(1, min(MAX_ANSWER_RESERVE, int(max_tokens) // 4))
    return int(max_tokens) - THINKING_FORCED_TOKENS - answer_reserve


def resolve_thinking_budget(budget_tokens: int, max_tokens: int) -> int:
    """Clamp ``budget_tokens`` to the ceiling derived from ``max_tokens``."""
    return max(1, min(int(budget_tokens), thinking_budget_ceiling(max_tokens)))


def apply_glm53_thinking_budget(out: dict[str, Any], budget_tokens: int) -> None:
    """Attach SGLang custom logit processor fields for a thinking budget."""
    out["custom_logit_processor"] = GLM53_THINKING_BUDGET_PROCESSOR_STR
    out["custom_params"] = {"thinking_budget": int(budget_tokens)}


def _last_open_in_prompt(origin_input_ids: list[int], start_id: int, end_id: int) -> int:
    """Return the index of an unclosed start token in the prompt, or -1.

    The last start/end marker decides: scanning backwards stops at it.
    """
    for i in range(len(origin_input_ids) - 1, -1, -1):
        tok = origin_input_ids[i]
        if tok == start_id:
            return i
        if tok == end_id:
            return -1
    return -1


def resolve_thinking_budget_action(
    origin_input_ids: list[int],
    output_ids: list[int],
    thinking_budget: int,
    start_id: int,
    end_id: int,
    newline_id: int,
    state: dict[str, int] | None = None,
) -> ThinkingBudgetAction | None:
    """Decide whether to force newline or thinking-end for the next token.

    Uses the last unclosed ``<think>`` across prompt+output so a template that
    opens thinking in the prompt still counts. Token count toward the budget
    only includes generated ``output_ids`` (or the suffix of output after a
    start token that appears in the output), so closed history ``</think>``
    blocks and prompt-side literals do not exhaust or skip the budget.

    ``state`` caches the prompt scan and the scan position in ``output_ids``
    so each call only inspects newly generated tokens. It is reset when
    ``output_ids`` is shorter than the cached position.
    """
    if thinking_budget < 0:
        return None
    if state is None:
        state = {}
    out_len = len(output_ids)
    if "origin_open" not in state or state["out_scanned"] > out_len:
        state["origin_open"] = _last_open_in_prompt(origin_input_ids, start_id, end_id)
        state["out_open"] = -1
        state["out_scanned"] = 0
    for i in range(state["out_scanned"], out_len):
        tok = output_ids[i]
        if tok == start_id:
            state["out_open"] = i
        elif tok == end_id:
            state["out_open"] = -1
            state["origin_open"] = -1
    state["out_scanned"] = out_len
    if state["out_open"] >= 0:
        num_tokens_after_start = out_len - state["out_open"] - 1
    elif state["origin_open"] >= 0:
        num_tokens_after_start = out_len
    else:
        return None
    if num_tokens_after_start < thinking_budget:
        return None
    if not output_ids or output_ids[-1] != newline_id:
        return "newline"
    return "end"


# Placeholder for committed tokens whose ids are not visible to the processor yet.
_UNKNOWN_TOKEN_ID = -1


def resolve_thinking_budget_spec_actions(
    origin_input_ids: list[int],
    output_ids: list[int],
    committed_output_len: int,
    candidates: list[int],
    thinking_budget: int,
    start_id: int,
    end_id: int,
    newline_id: int,
) -> list[ThinkingBudgetAction | None]:
    """Return the forced action for each row of an EAGLE chain verify block.

    ``candidates`` is the draft chain of one request: ``candidates[0]`` is the
    bonus token (the last committed token) and row ``k`` predicts the token that
    follows ``candidates[k]``. The context of row ``k`` is the committed outputs
    plus ``candidates[1..k]``. When ``committed_output_len`` exceeds
    ``len(output_ids)``, the tokens not yet written to ``output_ids`` count toward
    the budget; only the bonus token among them has a known id.
    """
    n = len(candidates)
    if thinking_budget < 0 or n == 0:
        return [None] * n
    lag = committed_output_len - len(output_ids)
    base = list(output_ids)
    if lag >= 1:
        base.extend([_UNKNOWN_TOKEN_ID] * (lag - 1))
        base.append(candidates[0])
    return [
        resolve_thinking_budget_action(
            origin_input_ids,
            base + list(candidates[1 : k + 1]),
            thinking_budget,
            start_id,
            end_id,
            newline_id,
        )
        for k in range(n)
    ]
