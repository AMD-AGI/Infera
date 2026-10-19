#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Keep interleaved system messages in place on /v1/responses.

WHAT: ``_construct_input_messages`` coalesces ``instructions`` and every
``system``/``developer`` input item into one leading system message, wherever
the item sat in the conversation.

WHY IT MATTERS HERE: Claude Code appends a system message to the conversation
on every turn (a ``<total_tokens>`` budget reminder). Hoisting it to the top
changes the leading system message on each turn, so a turn's prompt is no
longer a prefix of the next one and the prefix cache reuses nothing past the
tool definitions. Measured on GLM-5.3 through LiteLLM: every Claude Code turn
reused ~15-21K tokens of a 20-160K prompt and re-prefilled the rest.

FIX: coalesce only the leading run of system messages. A system message that
appears after the conversation starts stays where the client put it, so an
append-only conversation renders to an append-only prompt. The GLM-5.3 chat
template renders a system message at any position.

SCOPE: GLM images only. Some chat templates reject a system message that is
not first, so this is not applied to the generic sglang images.

Self-locating and idempotent. Marker: ``_infera_responses_keep_interleaved_system``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TAG = "[responses-keep-interleaved-system]"
_MARKER = "_infera_responses_keep_interleaved_system"
_REL = "entrypoints/openai/serving_responses.py"

_OLD = """        system_chunks: list[str] = []
        other_msgs: list = []
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "system":
"""

_NEW = """        system_chunks: list[str] = []
        other_msgs: list = []
        # Marker: _infera_responses_keep_interleaved_system
        # Only the leading run is coalesced; a system message after the
        # conversation starts stays in place so the prompt stays append-only.
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "system" and not other_msgs:
"""


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
    if _OLD not in src:
        return None, "coalescing anchor is gone"
    if src.count(_OLD) != 1:
        return None, "coalescing anchor is ambiguous"
    return src.replace(_OLD, _NEW, 1), "interleaved system messages kept in place"


# Reasons that mean the tree needs no change; anything else is a real mismatch
# and must fail the build rather than ship a cache-defeating prompt layout.
_BENIGN = ("already present",)


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
