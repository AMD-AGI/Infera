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
    resolve_thinking_budget_spec_actions,
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
            num_rows = int(param_dict.get("__spec_num_tokens__", 1))
            if num_rows > 1 and logits.shape[0] == len(custom_param_list) * num_rows:
                self._apply_spec_block(logits, i, num_rows, param_dict, req, thinking_budget)
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
            self._force(logits, i, action)
        return logits

    def _force(self, logits: Any, row: int, action: str) -> None:
        token_id = self.NEW_LINE_TOKEN_ID if action == "newline" else self.THINKING_END_TOKEN_ID
        logits[row, :] = -float("inf")
        logits[row, token_id] = 0.0

    def _apply_spec_block(
        self,
        logits: Any,
        index: int,
        num_rows: int,
        param_dict: dict[str, Any],
        req: Any,
        thinking_budget: int,
    ) -> None:
        """Force tokens on the rows of one request in an EAGLE chain verify block."""
        origin = getattr(req, "origin_input_ids", None) or []
        output = getattr(req, "output_ids", None) or []
        # Cheap gate on the cached scan: skip the draft readback unless the budget
        # can be reached within this block, allowing for the unwritten committed tokens.
        gate = resolve_thinking_budget_action(
            origin_input_ids=origin,
            output_ids=output,
            thinking_budget=max(0, thinking_budget - 2 * num_rows),
            start_id=self.THINKING_START_TOKEN_ID,
            end_id=self.THINKING_END_TOKEN_ID,
            newline_id=self.NEW_LINE_TOKEN_ID,
            state=param_dict.setdefault("__thinking_scan__", {}),
        )
        if gate is None:
            return
        candidates = param_dict["__spec_candidates__"].tolist()
        # seq_len counts the prompt and every committed output except the bonus token.
        committed = int(param_dict["__spec_seq_len__"]) - len(origin) + 1
        actions = resolve_thinking_budget_spec_actions(
            origin_input_ids=origin,
            output_ids=output,
            committed_output_len=committed,
            candidates=candidates,
            thinking_budget=thinking_budget,
            start_id=self.THINKING_START_TOKEN_ID,
            end_id=self.THINKING_END_TOKEN_ID,
            newline_id=self.NEW_LINE_TOKEN_ID,
        )
        for k, action in enumerate(actions):
            if action is not None:
                self._force(logits, index * num_rows + k, action)
