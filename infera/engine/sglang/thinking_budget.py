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

from sglang.srt.sampling.custom_logit_processor import ThinkingBudgetLogitProcessor

from infera.engine.sglang.thinking_budget_const import (
    GLM53_NEW_LINE_TOKEN_ID,
    GLM53_THINKING_BUDGET_PROCESSOR_STR,
    GLM53_THINKING_END_TOKEN_ID,
    GLM53_THINKING_START_TOKEN_ID,
    apply_glm53_thinking_budget,
    is_glm53_model,
    resolve_thinking_budget,
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
    """Hard-cap thinking tokens for GLM-5.3."""

    THINKING_START_TOKEN_ID = GLM53_THINKING_START_TOKEN_ID
    THINKING_END_TOKEN_ID = GLM53_THINKING_END_TOKEN_ID
    NEW_LINE_TOKEN_ID = GLM53_NEW_LINE_TOKEN_ID
