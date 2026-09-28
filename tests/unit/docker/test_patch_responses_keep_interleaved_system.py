###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Unit tests for the Responses interleaved-system-message patch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATCH = (
    Path(__file__).resolve().parents[3]
    / "deploy"
    / "docker"
    / "patches"
    / "sglang_responses_glm"
    / "patch_responses_keep_interleaved_system.py"
)
_MARKER = "_infera_responses_keep_interleaved_system"

# The coalescing tail of upstream ``_construct_input_messages``, verbatim at
# SGLANG_GLM53_REF, wrapped in a standalone function so the tests can run it.
_UPSTREAM = """
def coalesce(messages):
    # Most chat templates expect a single leading ``system`` message;
    # coalesce any ``instructions`` + interleaved ``developer`` entries.
    system_chunks: list[str] = []
    other_msgs: list = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            content = m.get("content")
            if isinstance(content, str):
                system_chunks.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text")
                        if isinstance(text, str):
                            system_chunks.append(text)
        else:
            other_msgs.append(m)
    if system_chunks:
        return [
            {"role": "system", "content": "\\n\\n".join(system_chunks)}
        ] + other_msgs
    return other_msgs
"""


def _load_patch():
    spec = importlib.util.spec_from_file_location("patch_responses_keep_interleaved_system", _PATCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coalesce(src: str):
    namespace: dict = {}
    exec(src, namespace)
    return namespace["coalesce"]


def _dedent(block: str) -> str:
    return "\n".join(line[4:] if line.startswith("    ") else line for line in block.split("\n"))


def _patched(mod) -> str:
    fixture = _UPSTREAM
    old, new = _dedent(mod._OLD), _dedent(mod._NEW)
    assert old in fixture
    return fixture.replace(old, new, 1)


def _turn(n_reminders: int) -> list[dict]:
    """A Claude Code shaped history: one reminder system message appended per turn."""
    msgs = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "review the PR"},
        {"role": "system", "content": "# Environment"},
    ]
    for i in range(n_reminders):
        msgs += [
            {"role": "assistant", "content": f"call {i}"},
            {"role": "tool", "content": f"result {i}"},
            {"role": "system", "content": f"<total_tokens>{i}</total_tokens>"},
        ]
    return msgs


def _rendered(msgs: list[dict]) -> str:
    return "".join(f"<|{m['role']}|>{m['content']}" for m in msgs)


def test_upstream_coalescing_breaks_the_prompt_prefix() -> None:
    # The regression this patch exists for: each turn's appended system message
    # is hoisted to the top, so turn N is no longer a prefix of turn N+1 and the
    # prefix cache can reuse nothing past the first system message.
    coalesce = _coalesce(_UPSTREAM)
    first, second = _rendered(coalesce(_turn(1))), _rendered(coalesce(_turn(2)))
    assert not second.startswith(first)


def test_interleaved_system_messages_keep_the_prompt_append_only() -> None:
    coalesce = _coalesce(_patched(_load_patch()))
    turns = [_rendered(coalesce(_turn(n))) for n in range(4)]
    for earlier, later in zip(turns, turns[1:]):
        assert later.startswith(earlier)


def test_leading_system_messages_are_still_coalesced() -> None:
    coalesce = _coalesce(_patched(_load_patch()))
    out = coalesce(
        [
            {"role": "system", "content": "instructions"},
            {"role": "system", "content": [{"type": "text", "text": "developer"}]},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "later"},
        ]
    )
    assert out[0] == {"role": "system", "content": "instructions\n\ndeveloper"}
    assert out[1] == {"role": "user", "content": "hi"}
    assert out[2] == {"role": "system", "content": "later"}
    assert len(out) == 3


def test_apply_is_idempotent_and_rejects_a_drifted_anchor() -> None:
    mod = _load_patch()
    src = "class S:\n    def f(self, messages):\n" + mod._OLD + "\n"
    patched, reason = mod.apply_to_source(src)
    assert patched is not None and _MARKER in patched, reason
    again, reason = mod.apply_to_source(patched)
    assert again is None and reason in mod._BENIGN
    drifted, reason = mod.apply_to_source("class S:\n    pass\n")
    assert drifted is None and reason not in mod._BENIGN
