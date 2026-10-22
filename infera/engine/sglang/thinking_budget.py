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
    # Read by the EAGLE verify patch: this processor handles several rows per request.
    SUPPORTS_SPEC_VERIFY_ROWS = True

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
            num_rows = param_dict.get("__spec_num_tokens__")
            if num_rows is not None:
                # The direct form has one param per request. SGLang's sampler repeats
                # each request's params once per verify row before this call.
                num_rows = int(num_rows)
                n_params = len(custom_param_list)
                n_logits = logits.shape[0]
                if num_rows > 0 and n_logits == n_params * num_rows:
                    request_index = i
                elif (
                    num_rows > 0
                    and n_logits == n_params
                    and n_params % num_rows == 0
                    and i % num_rows == 0
                ):
                    request_index = i // num_rows
                else:
                    continue
                self._apply_spec_block(
                    logits, request_index, num_rows, param_dict, req, thinking_budget
                )
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
        history = param_dict.get("__spec_history__") or ()
        # Tokens not visible in output_ids (committed history plus the draft chain) span
        # at most ``reach`` positions. The draft chain is read back from the device only
        # when the budget can be reached inside that span: either it is not larger than
        # the span, or the visible thinking block is already within the span of it.
        reach = num_rows * (len(history) + 2)
        near = resolve_thinking_budget_action(
            origin_input_ids=origin,
            output_ids=output,
            thinking_budget=max(0, thinking_budget - reach),
            start_id=self.THINKING_START_TOKEN_ID,
            end_id=self.THINKING_END_TOKEN_ID,
            newline_id=self.NEW_LINE_TOKEN_ID,
            state=param_dict.setdefault("__thinking_scan__", {}),
        )
        if thinking_budget > reach and near is None:
            return
        # Host copies are shared by every request of a step and made at most once.
        chains, seqs = param_dict["__spec_step__"].cpu()
        row = param_dict["__spec_row__"]
        past = []
        for step, step_row in history:
            past_chains, past_seqs = step.cpu()
            past.append((past_chains[step_row], int(past_seqs[step_row])))
        actions = resolve_thinking_budget_spec_actions(
            origin_input_ids=origin,
            output_ids=output,
            candidates=chains[row],
            seq_len=int(seqs[row]),
            history=past,
            thinking_budget=thinking_budget,
            start_id=self.THINKING_START_TOKEN_ID,
            end_id=self.THINKING_END_TOKEN_ID,
            newline_id=self.NEW_LINE_TOKEN_ID,
            state=param_dict["__thinking_scan__"],
        )
        for k, action in enumerate(actions):
            if action is not None:
                self._force(logits, index * num_rows + k, action)
