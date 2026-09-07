#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Flush unstreamed tool-call arguments on the /v1/responses stream.

WHAT: the streaming function-call detectors are not contracted to emit
byte-complete JSON. ``serving_chat.py`` knows this and reconciles when
generation finishes, in ``_check_for_unstreamed_tool_args``: it diffs the
accumulated argument deltas against the detector's fully parsed
``prev_tool_call_arr`` and sends the remainder. ``serving_responses.py`` has
no such step -- it treats ``state["arguments"] += call.parameters`` as
authoritative and reuses that buffer for both
``response.function_call_arguments.done`` and ``response.output_item.done``.

WHY IT MATTERS HERE: clients that rebuild tool input by concatenating
``response.function_call_arguments.delta`` (Codex, Claude Code) then receive
JSON the detector never finished. Measured on GLM-5.3 with ``glm47``: a tool
whose last argument is an object loses the outer ``}``, and the agent loop
reports the whole call as unparsed tool input. The same request over
``/v1/chat/completions`` is complete, because that path reconciles.

FIX: mirror the chat reconciliation inside ``_close_tool_call_state``. Emit
the missing suffix as one final delta, then close the item with the completed
arguments. Detector-agnostic, so any parser that under-emits is covered
rather than one model family.

DROP THIS SCRIPT once base sglang reconciles tool arguments on the Responses
path. ``prev_tool_call_arr`` appearing in serving_responses.py is treated as
"already fixed" and the script exits 0 without writing.

Self-locating and idempotent. Marker: ``_infera_responses_unstreamed_tool_args``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TAG = "[responses-unstreamed-tool-args]"
_MARKER = "_infera_responses_unstreamed_tool_args"
_REL = "entrypoints/openai/serving_responses.py"

# Set by upstream when it grows its own reconciliation. Its presence means the
# Responses path already consults the parsed arguments, so this patch is moot.
_UPSTREAM_FIXED_HINT = "prev_tool_call_arr"

_OLD_CLOSE = """            arguments = state["arguments"]
            completed_item = ResponseFunctionToolCall(
"""

_NEW_CLOSE = """            arguments = state["arguments"]
            # Marker: _infera_responses_unstreamed_tool_args
            pending_events = []
            pending = _infera_responses_unstreamed_tool_args(
                tool_parser, tool_index, arguments
            )
            if pending:
                arguments += pending
                state["arguments"] = arguments
                pending_events.append(
                    _send_event(
                        openai_responses_types.ResponseFunctionCallArgumentsDeltaEvent(
                            type="response.function_call_arguments.delta",
                            sequence_number=-1,
                            item_id=state["item_id"],
                            output_index=state["output_index"],
                            delta=pending,
                        )
                    )
                )
            completed_item = ResponseFunctionToolCall(
"""

_OLD_EVENTS = """            events = [
                _send_event(
                    openai_responses_types.ResponseFunctionCallArgumentsDoneEvent(
"""

_NEW_EVENTS = """            events = pending_events + [
                _send_event(
                    openai_responses_types.ResponseFunctionCallArgumentsDoneEvent(
"""

_HELPER = '''

def _infera_responses_unstreamed_tool_args(parser, tool_index, streamed):
    """Return the tool-argument suffix the streaming detector never emitted.

    Mirrors ``serving_chat._check_for_unstreamed_tool_args``. The detector's
    ``prev_tool_call_arr`` holds the fully parsed call, so serialising it and
    dropping the already-streamed prefix yields whatever is still owed.
    Returns an empty string when the stream is already complete, when the
    detector exposes nothing to compare against, or when the serialisation
    does not extend the streamed bytes -- in that last case the delta stream
    is not a prefix of the parsed form and appending would corrupt it.
    """
    detector = getattr(parser, "detector", parser)
    parsed_calls = getattr(detector, "prev_tool_call_arr", None)
    if not parsed_calls or not isinstance(tool_index, int):
        return ""
    if tool_index < 0 or tool_index >= len(parsed_calls):
        return ""
    entry = parsed_calls[tool_index]
    if not isinstance(entry, dict):
        return ""
    expected = entry.get("arguments", {})
    if not isinstance(expected, str):
        try:
            expected = json.dumps(expected, ensure_ascii=False)
        except (TypeError, ValueError):
            return ""
    if not expected.startswith(streamed):
        return ""
    return expected[len(streamed) :]
'''

_HELPER_ANCHOR = "logger = logging.getLogger(__name__)\n"


def _srt_dir() -> Path | None:
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        return None
    d = Path(spec.origin).parent / "srt"
    return d if d.is_dir() else None


def apply_to_source(src: str) -> tuple[str | None, str]:
    """Return patched source, or ``(None, reason)`` when nothing is written."""
    if _MARKER in src:
        return None, "already present"
    if _UPSTREAM_FIXED_HINT in src:
        return None, "upstream already reconciles tool args"
    for name, anchor in (("close state", _OLD_CLOSE), ("done events", _OLD_EVENTS)):
        if anchor not in src:
            return None, f"{name} anchor is gone"
        if src.count(anchor) != 1:
            return None, f"{name} anchor is ambiguous"
    if src.count(_HELPER_ANCHOR) != 1:
        return None, "logger anchor missing or ambiguous"
    patched = src.replace(_HELPER_ANCHOR, _HELPER_ANCHOR + _HELPER, 1)
    patched = patched.replace(_OLD_CLOSE, _NEW_CLOSE, 1)
    patched = patched.replace(_OLD_EVENTS, _NEW_EVENTS, 1)
    return patched, "helper and close reconciliation inserted"


# Reasons that mean the tree needs no change; anything else is a real mismatch
# and must fail the build rather than ship an unreconciled Responses stream.
_BENIGN = ("already present", "upstream already reconciles tool args")


def main() -> int:
    srt = _srt_dir()
    if srt is None:
        print(f"{_TAG} sglang not importable — skipping")
        return 0

    path = srt / _REL
    if not path.is_file():
        print(f"{_TAG} {path} is missing — skipping")
        return 0

    src = path.read_text()
    patched, reason = apply_to_source(src)
    if patched is None:
        print(f"{_TAG} {reason} — skipping")
        return 0 if reason in _BENIGN else 1

    path.write_text(patched)
    print(f"{_TAG} patched {path} ({reason})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
