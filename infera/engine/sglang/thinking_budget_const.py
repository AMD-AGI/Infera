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

from typing import Any

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
