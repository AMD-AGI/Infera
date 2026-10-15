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


def resolve_thinking_budget(budget_tokens: int, max_tokens: int) -> int:
    """Clamp ``budget_tokens`` so at least one completion token remains."""
    return max(1, min(int(budget_tokens), int(max_tokens) - 1))


def apply_glm53_thinking_budget(out: dict[str, Any], budget_tokens: int) -> None:
    """Attach SGLang custom logit processor fields for a thinking budget."""
    out["custom_logit_processor"] = GLM53_THINKING_BUDGET_PROCESSOR_STR
    out["custom_params"] = {"thinking_budget": int(budget_tokens)}


def last_open_thinking_start(token_ids: list[int], start_id: int, end_id: int) -> int:
    """Return index of the last unclosed thinking-start token, or -1."""
    start_index = -1
    for i, tok in enumerate(token_ids):
        if tok == start_id:
            start_index = i
        elif tok == end_id:
            start_index = -1
    return start_index


def resolve_thinking_budget_action(
    origin_input_ids: list[int],
    output_ids: list[int],
    thinking_budget: int,
    start_id: int,
    end_id: int,
    newline_id: int,
) -> ThinkingBudgetAction | None:
    """Decide whether to force newline or thinking-end for the next token.

    Uses the last unclosed ``<think>`` across prompt+output so a template that
    opens thinking in the prompt still counts. Token count toward the budget
    only includes generated ``output_ids`` (or the suffix of output after a
    start token that appears in the output), so closed history ``</think>``
    blocks and prompt-side literals do not exhaust or skip the budget.
    """
    if thinking_budget < 0:
        return None
    cur_ids = [*origin_input_ids, *output_ids]
    start_index = last_open_thinking_start(cur_ids, start_id, end_id)
    if start_index < 0:
        return None
    origin_len = len(origin_input_ids)
    if start_index < origin_len:
        num_tokens_after_start = len(output_ids)
    else:
        num_tokens_after_start = len(cur_ids) - start_index - 1
    if num_tokens_after_start < thinking_budget:
        return None
    if not output_ids or output_ids[-1] != newline_id:
        return "newline"
    return "end"
