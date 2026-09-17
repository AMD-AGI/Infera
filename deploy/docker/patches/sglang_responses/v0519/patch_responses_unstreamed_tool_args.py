#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Flush unstreamed tool-call arguments on the /v1/responses stream (v0.5.19).

Same defect, same fix, and the same helper as the sibling script one directory
up; only the splice differs. See that script's header for the full argument.

WHY A SEPARATE FILE. v0.5.19 restructured ``_close_tool_call_state`` to handle
custom tool calls: ``events`` is now initialised empty before a
custom/function branch instead of being built as one list literal after it. The
pre-v0.5.19 shape therefore needs two splices (introduce ``pending_events``,
then prepend them to the literal) while this one needs a single insertion. One
script accepting both shapes would have to carry two anchor sets and two splice
counts, and its error reporting could no longer name which shape drifted. The
bases are isolated instead: this directory is applied only by the image whose
base is v0.5.19, and it is deleted when no supported base predates that.

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

# Reconcile before the custom/function branch, where `events` is already an
# empty list, but act only on the function-call arm: the custom arm decodes
# `arguments` through decode_custom_tool_input and diffs its own remainder
# against state["payload"], so it neither needs nor wants a second delta here.
_OLD_CLOSE = """            arguments = state["arguments"]
            events: list = []
            if state["custom"]:
"""

_NEW_CLOSE = """            arguments = state["arguments"]
            events: list = []
            if not state["custom"]:
                pending = _infera_responses_unstreamed_tool_args(
                    tool_parser, tool_index, arguments
                )
                if pending:
                    arguments += pending
                    state["arguments"] = arguments
                    events.append(
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
            if state["custom"]:
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
    if _OLD_CLOSE not in src:
        return None, "close state anchor is gone"
    if src.count(_OLD_CLOSE) != 1:
        return None, "close state anchor is ambiguous"
    if src.count(_HELPER_ANCHOR) != 1:
        return None, "logger anchor missing or ambiguous"
    patched = src.replace(_HELPER_ANCHOR, _HELPER_ANCHOR + _HELPER, 1)
    patched = patched.replace(_OLD_CLOSE, _NEW_CLOSE, 1)
    return patched, "helper and close reconciliation inserted"


# Reasons that mean the tree needs no change; anything else is a real mismatch
# and must fail the build rather than ship an unreconciled Responses stream.
_BENIGN = ("already present", "upstream already reconciles tool args")


def main() -> int:
    srt = _srt_dir()
    if srt is None:
        print(f"{_TAG} sglang not importable — skipping")
        return 0
    target = srt / _REL
    if not target.is_file():
        print(f"{_TAG} {target} is missing — sglang layout changed")
        return 1

    src = target.read_text()
    patched, reason = apply_to_source(src)
    if patched is None:
        print(f"{_TAG} {reason}")
        return 0 if reason in _BENIGN else 1

    target.write_text(patched)
    print(f"{_TAG} {reason}: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
