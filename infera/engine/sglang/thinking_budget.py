###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""GLM-5.3 thinking-budget logit processor for SGLang.

Anthropic ``thinking.budget_tokens`` is mapped by the Messages shim onto
SGLang ``custom_logit_processor`` + ``custom_params.thinking_budget``.
GLM-5.3 uses think token ids that differ from the stock
``Glm4MoeThinkingBudgetLogitProcessor`` (GLM-4.5/4.6), so this module
ships the correct ids.

The serialized form is a stable dill STACK_GLOBAL reference to this
class. Workers must run with ``--enable-custom-logit-processor``.
"""

from __future__ import annotations

from typing import Any

# dill.dumps of ``infera.engine.sglang.thinking_budget.Glm53ThinkingBudgetLogitProcessor``
# (protocol 4 STACK_GLOBAL). Keep in sync with the Rust anthropic shim.
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


try:
    from sglang.srt.sampling.custom_logit_processor import (  # type: ignore
        ThinkingBudgetLogitProcessor,
    )
except ImportError:  # pragma: no cover - unit tests without sglang
    ThinkingBudgetLogitProcessor = object  # type: ignore[misc, assignment]


class Glm53ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """Hard-cap thinking tokens for GLM-5.3."""

    THINKING_START_TOKEN_ID = GLM53_THINKING_START_TOKEN_ID
    THINKING_END_TOKEN_ID = GLM53_THINKING_END_TOKEN_ID
    NEW_LINE_TOKEN_ID = GLM53_NEW_LINE_TOKEN_ID


def apply_glm53_thinking_budget(out: dict[str, Any], budget_tokens: int) -> None:
    """Attach SGLang custom logit processor fields for a thinking budget."""
    out["custom_logit_processor"] = GLM53_THINKING_BUDGET_PROCESSOR_STR
    out["custom_params"] = {"thinking_budget": int(budget_tokens)}
