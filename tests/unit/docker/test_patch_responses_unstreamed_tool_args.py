###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Unit tests for the Responses unstreamed tool-args reconciliation patch."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_PATCH = (
    Path(__file__).resolve().parents[3]
    / "deploy"
    / "docker"
    / "patches"
    / "sglang_responses"
    / "patch_responses_unstreamed_tool_args.py"
)
_MARKER = "_infera_responses_unstreamed_tool_args"


def _load_patch():
    spec = importlib.util.spec_from_file_location(
        "patch_responses_unstreamed_tool_args", _PATCH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_helper(mod):
    """Exec the injected helper the way serving_responses.py would see it."""
    namespace: dict = {"json": json}
    exec(mod._HELPER, namespace)
    return namespace[_MARKER]


class _Detector:
    def __init__(self, parsed):
        self.prev_tool_call_arr = parsed


class _Parser:
    def __init__(self, parsed):
        self.detector = _Detector(parsed)


def _fixture(mod, close: str | None = None, events: str | None = None) -> str:
    return (
        "import json\n"
        "import logging\n\n"
        "logger = logging.getLogger(__name__)\n"
        "class ServingResponses:\n"
        "    def _stream(self):\n"
        "        def _close_tool_call_state(tool_index: int):\n"
        "            state = tool_call_states.get(tool_index)\n"
        '            if state is None or state.get("done"):\n'
        "                return []\n"
        f"{mod._OLD_CLOSE if close is None else close}"
        "                arguments=arguments,\n"
        "            )\n"
        f"{mod._OLD_EVENTS if events is None else events}"
        "                    )\n"
        "                ),\n"
        "            ]\n"
        "            return events\n"
    )


def test_suffix_completes_truncated_nested_object() -> None:
    helper = _load_helper(_load_patch())
    parsed = [
        {
            "name": "emit_intent",
            "arguments": {
                "intent_type": "propose_action",
                "payload": {"action_name": "baseline", "predicted_gain_pct": 0},
            },
        }
    ]
    streamed = (
        '{"intent_type": "propose_action", '
        '"payload": {"action_name": "baseline", "predicted_gain_pct": 0}'
    )
    suffix = helper(_Parser(parsed), 0, streamed)
    assert suffix == "}"
    assert json.loads(streamed + suffix) == parsed[0]["arguments"]


def test_suffix_empty_when_stream_already_complete() -> None:
    helper = _load_helper(_load_patch())
    args = {"city": "Beijing", "date": "2024-06-27"}
    parsed = [{"name": "get_weather", "arguments": args}]
    streamed = json.dumps(args, ensure_ascii=False)
    assert helper(_Parser(parsed), 0, streamed) == ""


def test_suffix_empty_when_stream_is_not_a_prefix() -> None:
    """A detector whose deltas diverge from the parsed form is left alone."""
    helper = _load_helper(_load_patch())
    parsed = [{"name": "t", "arguments": {"a": 1}}]
    assert helper(_Parser(parsed), 0, '{"b": 2') == ""


def test_suffix_empty_without_detector_state() -> None:
    helper = _load_helper(_load_patch())
    assert helper(_Parser([]), 0, "{") == ""
    assert helper(_Parser([{"arguments": {"a": 1}}]), 3, "{") == ""
    assert helper(_Parser([{"arguments": {"a": 1}}]), -1, "{") == ""


def test_suffix_handles_string_arguments() -> None:
    helper = _load_helper(_load_patch())
    parsed = [{"name": "t", "arguments": '{"a": 1}'}]
    assert helper(_Parser(parsed), 0, '{"a": 1') == "}"


def test_suffix_empty_for_unserialisable_arguments() -> None:
    helper = _load_helper(_load_patch())
    parsed = [{"name": "t", "arguments": {"a": object()}}]
    assert helper(_Parser(parsed), 0, "{") == ""


def test_apply_inserts_helper_and_is_idempotent() -> None:
    mod = _load_patch()
    src = _fixture(mod)
    patched, reason = mod.apply_to_source(src)
    assert patched is not None
    assert "helper" in reason
    assert _MARKER in patched
    assert "events = pending_events + [" in patched
    assert mod._OLD_CLOSE not in patched
    again, reason2 = mod.apply_to_source(patched)
    assert again is None
    assert reason2 == "already present"


def test_patched_fixture_still_compiles() -> None:
    mod = _load_patch()
    patched, _ = mod.apply_to_source(_fixture(mod))
    assert patched is not None
    compile(patched, "serving_responses.py", "exec")


def test_apply_skips_when_upstream_reconciles() -> None:
    mod = _load_patch()
    src = _fixture(mod) + "\n# uses prev_tool_call_arr now\n"
    patched, reason = mod.apply_to_source(src)
    assert patched is None
    assert reason == "upstream already reconciles tool args"


def test_apply_reports_missing_close_anchor() -> None:
    mod = _load_patch()
    src = _fixture(mod, close="            arguments = state.pop('arguments')\n")
    patched, reason = mod.apply_to_source(src)
    assert patched is None
    assert reason == "close state anchor is gone"
    assert reason not in mod._BENIGN


def test_apply_reports_missing_events_anchor() -> None:
    mod = _load_patch()
    src = _fixture(mod, events="            events = list(_done_events())\n")
    patched, reason = mod.apply_to_source(src)
    assert patched is None
    assert reason == "done events anchor is gone"
    assert reason not in mod._BENIGN
