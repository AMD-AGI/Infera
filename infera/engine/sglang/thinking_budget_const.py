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

# The processor is injected into the request whenever ``thinking.budget_tokens``
# is present. Workers run with ``--enable-custom-logit-processor`` (added by the
# SGLang worker launcher) so SGLang accepts the field.
ThinkingBudgetAction = Literal["newline", "end"]


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


# Budget used when a GLM-5.3 request does not carry ``budget_tokens``.
DEFAULT_THINKING_BUDGET = 1024
# ``budget_tokens`` value that leaves thinking uncapped.
UNLIMITED_THINKING_BUDGET = -1


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def resolve_request_thinking_budget(thinking: Any, max_tokens: Any) -> int | None:
    """Return the thinking budget to enforce for a GLM-5.3 request, or None for no cap.

    ``thinking`` is the request's ``thinking`` object (``type`` and optional
    ``budget_tokens``). ``type=disabled`` and ``budget_tokens=-1`` leave thinking
    uncapped. An omitted budget uses ``DEFAULT_THINKING_BUDGET``. The budget is
    clamped so the forced newline and ``</think>`` plus an answer reserve fit in
    ``max_tokens``; a missing ``max_tokens`` needs no clamp. An omitted budget is
    skipped when ``max_tokens`` cannot fit one, an explicit one is rejected.
    Raises ``ValueError`` for an invalid explicit ``budget_tokens`` or ``max_tokens``.
    """
    thinking = thinking if isinstance(thinking, dict) else {}
    if thinking.get("type") == "disabled":
        return None
    max_ok = max_tokens is None or (_is_int(max_tokens) and max_tokens >= MIN_MAX_TOKENS_FOR_BUDGET)
    if "budget_tokens" in thinking:
        budget = thinking["budget_tokens"]
        if _is_int(budget) and budget == UNLIMITED_THINKING_BUDGET:
            return None
        if not _is_int(budget) or budget < 1:
            raise ValueError("thinking.budget_tokens must be a positive integer or -1")
        if not max_ok:
            raise ValueError(
                f"max_tokens must be >= {MIN_MAX_TOKENS_FOR_BUDGET} when thinking.budget_tokens is set"
            )
    else:
        budget = DEFAULT_THINKING_BUDGET
        if not max_ok:
            return None
    return budget if max_tokens is None else resolve_thinking_budget(budget, max_tokens)


def apply_openai_thinking_budget(body: dict[str, Any]) -> None:
    """Apply the thinking controls of an OpenAI chat body in place (GLM-5.3 only).

    The top-level ``thinking`` object maps to ``chat_template_kwargs`` without
    overriding keys the client set, and is removed from the forwarded body. The
    budget comes from ``resolve_request_thinking_budget`` with
    ``max_completion_tokens`` taking precedence over ``max_tokens``. Thinking
    turned off through ``chat_template_kwargs`` is not capped.
    """
    if not is_glm53_model(body.get("model")):
        return
    thinking = body.pop("thinking", None) if isinstance(body.get("thinking"), dict) else None
    kwargs = body.get("chat_template_kwargs")
    if isinstance(thinking, dict) and thinking.get("type") in ("enabled", "disabled"):
        flag = thinking["type"] == "enabled"
        if kwargs is None:
            kwargs = body["chat_template_kwargs"] = {}
        if isinstance(kwargs, dict):
            kwargs.setdefault("enable_thinking", flag)
            kwargs.setdefault("thinking", flag)
    if isinstance(kwargs, dict) and (
        kwargs.get("enable_thinking") is False or kwargs.get("thinking") is False
    ):
        return
    max_tokens = body.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_tokens")
    budget = resolve_request_thinking_budget(thinking, max_tokens)
    if budget is not None:
        apply_glm53_thinking_budget(body, budget)


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


# Placeholder for committed tokens whose ids cannot be recovered.
_UNKNOWN_TOKEN_ID = -1


def _committed_tail(
    history: list[tuple[list[int], int]],
    candidates: list[int],
    seq_len: int,
    lag: int,
) -> list[int]:
    """Return the last ``lag`` committed tokens, rebuilt from earlier verify steps.

    ``history`` holds ``(chain, seq_len)`` of previous verify steps, oldest first.
    A step that grew the sequence by ``c`` committed the drafts ``chain[1:c]`` plus
    the bonus token, which is the first entry of the next step's chain. The last
    committed token is always ``candidates[0]``. Tokens older than the history are
    reported as ``_UNKNOWN_TOKEN_ID``.
    """
    tail: list[int] = []
    for j, (chain, seq) in enumerate(history):
        if j + 1 < len(history):
            next_chain, next_seq = history[j + 1]
            next_bonus = next_chain[0]
        else:
            next_seq, next_bonus = seq_len, candidates[0]
        committed = next_seq - seq
        if committed < 1 or committed > len(chain):
            tail = []
            continue
        tail.extend(chain[1:committed])
        tail.append(next_bonus)
    if not tail:
        tail = [candidates[0]]
    if len(tail) < lag:
        return [_UNKNOWN_TOKEN_ID] * (lag - len(tail)) + tail
    return tail[len(tail) - lag :]


def resolve_thinking_budget_spec_actions(
    origin_input_ids: list[int],
    output_ids: list[int],
    candidates: list[int],
    seq_len: int,
    history: list[tuple[list[int], int]],
    thinking_budget: int,
    start_id: int,
    end_id: int,
    newline_id: int,
    state: dict[str, int] | None = None,
) -> list[ThinkingBudgetAction | None]:
    """Return the forced action for each row of an EAGLE chain verify block.

    ``candidates`` is the draft chain of one request: ``candidates[0]`` is the
    bonus token (the last committed token) and row ``k`` predicts the token that
    follows ``candidates[k]``. The context of row ``k`` is the committed outputs
    plus ``candidates[1..k]``. ``seq_len`` counts the prompt and every committed
    output except the bonus token. Committed tokens that are not yet in
    ``output_ids`` are rebuilt from ``history`` (see ``_committed_tail``). No row is
    forced when a committed token cannot be recovered, since it may be a marker.
    ``state`` is a scan state of ``output_ids`` as kept for
    ``resolve_thinking_budget_action``; it is copied and never modified.
    """
    n = len(candidates)
    if thinking_budget < 0 or n == 0:
        return [None] * n
    lag = seq_len - len(origin_input_ids) + 1 - len(output_ids)
    context = list(output_ids)
    if lag >= 1:
        tail = _committed_tail(history, candidates, seq_len, lag)
        if _UNKNOWN_TOKEN_ID in tail:
            return [None] * n
        context.extend(tail)
    scan = dict(state) if state else {}
    actions: list[ThinkingBudgetAction | None] = []
    for k in range(n):
        if k:
            context.append(candidates[k])
        actions.append(
            resolve_thinking_budget_action(
                origin_input_ids, context, thinking_budget, start_id, end_id, newline_id, scan
            )
        )
    return actions


def eagle_topk_disables_thinking_budget(algorithm: Any, topk: Any) -> bool:
    """Return True when EAGLE tree drafts (topk > 1) make the thinking budget inert."""
    if not isinstance(algorithm, str) or not algorithm.upper().startswith(("EAGLE", "NEXTN")):
        return False
    return isinstance(topk, int) and not isinstance(topk, bool) and topk > 1
