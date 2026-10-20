###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""GLM-5.3 thinking-budget logit processor for SGLang workers.

Workers deserialize this class via dill when a request carries
``custom_logit_processor``. Import fails hard if SGLang does not provide
``ThinkingBudgetLogitProcessor`` so a missing base class cannot silently
become an uncallable ``object`` during sampling.

Request-path helpers live in ``thinking_budget_const`` (no SGLang import).
"""

from __future__ import annotations

from typing import Any

from sglang.srt.sampling.custom_logit_processor import ThinkingBudgetLogitProcessor

from infera.engine.sglang.thinking_budget_const import (
    GLM53_NEW_LINE_TOKEN_ID,
    GLM53_THINKING_BUDGET_PROCESSOR_STR,
    GLM53_THINKING_END_TOKEN_ID,
    GLM53_THINKING_START_TOKEN_ID,
    apply_glm53_thinking_budget,
    is_glm53_model,
    resolve_thinking_budget,
    resolve_thinking_budget_action,
)

__all__ = [
    "GLM53_NEW_LINE_TOKEN_ID",
    "GLM53_THINKING_BUDGET_PROCESSOR_STR",
    "GLM53_THINKING_END_TOKEN_ID",
    "GLM53_THINKING_START_TOKEN_ID",
    "Glm53ThinkingBudgetLogitProcessor",
    "apply_glm53_thinking_budget",
    "is_glm53_model",
    "resolve_thinking_budget",
]


class Glm53ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """Hard-cap thinking tokens for GLM-5.3.

    Overrides the upstream ``ThinkingBudgetLogitProcessor`` so multi-turn
    history with closed ``</think>`` blocks does not disable the budget, and
    prompt-side ``<think>`` literals do not count toward the budget.
    """

    THINKING_START_TOKEN_ID = GLM53_THINKING_START_TOKEN_ID
    THINKING_END_TOKEN_ID = GLM53_THINKING_END_TOKEN_ID
    NEW_LINE_TOKEN_ID = GLM53_NEW_LINE_TOKEN_ID

    def __call__(self, logits: Any, custom_param_list: list[dict[str, Any]] | None = None) -> Any:
        if custom_param_list is None or not custom_param_list:
            return logits
        for i, param_dict in enumerate(custom_param_list):
            if param_dict is None:
                continue
            thinking_budget = param_dict.get("thinking_budget")
            if (
                thinking_budget is None
                or not isinstance(thinking_budget, int)
                or isinstance(thinking_budget, bool)
                or thinking_budget < 0
            ):
                continue
            req = param_dict.get("__req__")
            if req is None:
                continue
            # Per-request scan cache; param_dict lives as long as the request.
            action = resolve_thinking_budget_action(
                origin_input_ids=getattr(req, "origin_input_ids", None) or [],
                output_ids=getattr(req, "output_ids", None) or [],
                thinking_budget=thinking_budget,
                start_id=self.THINKING_START_TOKEN_ID,
                end_id=self.THINKING_END_TOKEN_ID,
                newline_id=self.NEW_LINE_TOKEN_ID,
                state=param_dict.setdefault("__thinking_scan__", {}),
            )
            if action is None:
                continue
            token_id = self.NEW_LINE_TOKEN_ID if action == "newline" else self.THINKING_END_TOKEN_ID
            logits[i, :] = -float("inf")
            logits[i, token_id] = 0.0
        return logits
