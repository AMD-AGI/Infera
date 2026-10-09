"""Criterion 16 — design §12, and the projection §8.5 makes a MUST."""

from __future__ import annotations

import json
from typing import Any

from agent.backend import AgentResult
from monitor import event
from monitor.protocols import EventKind
from task_graph.ids import TaskId

PROMPT = "write me a haiku about the load-bearing wall"
REASONING = "the user probably wants five-seven-five, so I will"


class FakeResultMessage:
    """What `ClaudeSDKClient` hands back. **Exactly one of its fields is
    annotated "safe to log (no message content)"** — `api_error_status` — and
    the rest of these carry prompt-derived text."""

    subtype = "success"
    is_error = False
    duration_ms = 1200
    num_turns = 3
    total_cost_usd = 0.042
    usage = {"input_tokens": 900}
    model_usage = {"sonnet": {"input_tokens": 900}}
    api_error_status = None
    result = PROMPT + " — here it is"
    structured_output = {"draft": REASONING}
    permission_denials = [{"tool_name": "Bash", "input": {"command": PROMPT}}]


def test_records_hold_no_prompt_text() -> None:
    """**The adapter projects a named subset. It never stores the
    `ResultMessage`.** Persisting the whole message would put prompt-derived
    text in the system's record."""
    projected = _project(FakeResultMessage())
    serialised = json.dumps(projected.model_dump(mode="json"))
    assert PROMPT not in serialised
    assert REASONING not in serialised
    assert set(projected.usage) >= {"duration_ms", "num_turns", "total_cost_usd"}


def test_the_event_record_carries_no_prompt_either() -> None:
    """The other thing this package persists is the event stream, and the same
    rule holds for it."""
    record = event(
        EventKind.PHASE_DONE,
        TaskId.new(),
        attempt=0,
        reported_by="agent.Runner",
        attributes={"message": "TaskStatus.RUNNING finished"},
    )
    serialised = json.dumps(record.model_dump(mode="json"), default=str)
    assert PROMPT not in serialised
    assert REASONING not in serialised


def test_the_agent_result_has_three_fields_and_no_content_one() -> None:
    """Structural: there is nowhere in `AgentResult` for a transcript to go."""
    assert set(AgentResult.model_fields) == {"status", "usage", "detail"}


def test_this_package_writes_no_agent_record() -> None:
    """Design §10: every recorded field already belongs to `task_graph`'s
    `Agent`, and this module writes none of them. There is no second agent
    object here."""
    import agent

    assert not [name for name in agent.__all__ if name.endswith("Record")]


def _project(message: Any) -> AgentResult:
    """`ClaudeSdkBackend._project`, reached without importing the extra."""
    from agent.backends.claude_sdk import ClaudeSdkBackend

    return ClaudeSdkBackend._project(_Unconstructed(), message, 1.2)


class _Unconstructed:
    """`_project` reads nothing off `self`, which is what makes it a pure
    projection — and this is how the test says so."""
