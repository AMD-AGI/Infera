"""Criteria 9, 10 and 11 — design §12.

**Backends are exercised against a fake transport, not a live harness.**
Nothing here needs a credential, a network, or the 376 MB extra: the adapter
takes its client from `config["client"]`, which is the same seam a third party
uses to pin a pre-configured handle.

`test_the_extra_is_not_imported_at_module_scope` is the one that guards §8.1.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from agent.backend import AgentStatus, Assignment, BackendUnsupported
from agent.backends.claude_sdk import ClaudeSdkBackend, pre_tool_use


class Message:
    def __init__(self, **fields: Any) -> None:
        self.subtype = "success"
        self.is_error = False
        self.terminal_reason = None
        self.result = ""
        # Every message type in the SDK declares `session_id`, and it is the
        # only place the id is available: `ClaudeSDKClient` has no such
        # attribute. Measured against 0.2.148.
        self.session_id = "sess-abc"
        self.__dict__.update(fields)


class FakeClient:
    """The SDK's durable handle, as much of it as the adapter touches."""

    def __init__(self) -> None:
        # **No `session_id` and no `get_session_messages`.** The real
        # `ClaudeSDKClient` has neither, and an earlier version of this double
        # defined both — so the adapter's guesses at the SDK's surface were
        # ratified by the fake instead of checked, and `session_ref` returned
        # `None` and `query()` raised `AttributeError` on every real run while
        # this file stayed green.
        self.connected = False
        self.queries: list[str] = []
        self.interrupts = 0
        self.responses: list[list[Message]] = []
        self.buffered: list[Message] = []

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def query(self, prompt: str) -> None:
        self.queries.append(prompt)

    async def interrupt(self) -> None:
        self.interrupts += 1

    async def receive_response(self):
        for message in self.responses.pop(0) if self.responses else [Message()]:
            yield message

    async def receive_messages(self):
        while self.buffered:
            yield self.buffered.pop(0)


def _backend(client: FakeClient | None = None, **config: Any) -> ClaudeSdkBackend:
    return ClaudeSdkBackend(
        "claude_sdk",
        {"client": client or FakeClient(), **config},
        Assignment(goal="do the thing", readme="R", zone="/tmp/z"),
    )


# --------------------------------------------------------------------------- #
# §8.1 — the extra, and the rule that keeps it out of every entry point


def test_the_extra_is_not_imported_at_module_scope() -> None:
    """The SDK is 376 MB and ~1.3 s to import. A hard dependency makes every
    `agent_sys` entry point pay both, including `env-mgr check`, which has
    nothing to do with agents. **Never at module scope, anywhere in the
    package.**"""
    root = Path(__file__).resolve().parents[2] / "agent"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:  # module scope only
            names = _imported(node)
            assert "claude_agent_sdk" not in names, path
            if path.name != "claude_sdk.py":
                assert "agent.backends.claude_sdk" not in names, path


def test_a_missing_extra_is_a_backend_unsupported_naming_it(monkeypatch) -> None:
    """**Not an `ImportError` at start-up** in a process that was never going
    to use it — which is the per-entry error message §6.3 exists for."""
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    with pytest.raises(BackendUnsupported) as caught:
        ClaudeSdkBackend("claude_sdk", {}, Assignment())
    assert "claude" in str(caught.value)


def _imported(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name for alias in node.names}
    if isinstance(node, ast.ImportFrom):
        return {node.module or ""}
    return set()


# --------------------------------------------------------------------------- #
# Criterion 9


def test_interrupt_drains_before_next_query() -> None:
    """**The drain is not "consume N messages".** `terminal_reason` is
    `aborted_streaming` or `aborted_tools` for an interrupted turn, so the
    interrupted submission's own result is self-identifying — and the test
    asserts on that rather than on a count, because a count-based assertion
    passes against a backend that drained the wrong number of messages for the
    wrong reason."""
    client = FakeClient()
    client.buffered = [
        Message(result="stale tool output"),
        Message(terminal_reason="aborted_streaming", result="the abandoned turn"),
        Message(result="the NEW query's response"),
    ]
    backend = _backend(client)
    backend._deploy()
    backend.interrupt()

    assert client.interrupts == 1
    assert backend.status is AgentStatus.INTERRUPTED
    assert len(client.buffered) == 1
    assert client.buffered[0].result == "the NEW query's response"


def test_the_drain_is_bounded() -> None:
    """**The bound is load-bearing.** `terminal_reason` is `None` on CLIs
    predating the field and on a synthesized result after a fatal session
    failure — a drain that waits for an `aborted_*` value and nothing else
    hangs in exactly the case where the session has already died."""
    client = FakeClient()
    client.buffered = [Message(result=str(n)) for n in range(50)]
    backend = _backend(client)
    backend.interrupt()  # every message has terminal_reason None
    assert client.buffered == []


# --------------------------------------------------------------------------- #
# Criterion 10


def test_instruct_does_not_end_run() -> None:
    """`instruct()` reaches a running agent and affects its behaviour without
    ending the run. It is a queue operation: the loop delivers it, and the
    caller never touches the harness."""
    client = FakeClient()
    backend = _backend(client)
    backend.start_async(lambda: None)
    backend.instruct("also check the error path")
    backend.mainloop()

    assert "also check the error path" in client.queries
    assert client.connected
    assert backend.status is AgentStatus.FINISHED


# --------------------------------------------------------------------------- #
# Criterion 11


def test_the_session_ref_is_learned_from_messages_not_from_the_client() -> None:
    """`ClaudeSDKClient` has **no** `session_id` attribute — measured against
    0.2.148, `hasattr` is `False`. The id is on the messages, so the adapter
    records it as they stream past and there is exactly one writer.

    The regression this pins: reading it off the client returned `None` on every
    real run, and the old double's invented attribute kept that green.
    """
    client = FakeClient()
    backend = _backend(client)
    assert not hasattr(client, "session_id")  # the double may not re-invent it

    assert backend.session_ref is None  # nothing has run; nothing is known
    client.responses = [[Message(session_id="sess-xyz")]]
    backend.start()

    assert backend.session_ref == "sess-xyz"


def test_query_history_session_matches_agent_id(monkeypatch) -> None:
    """`query()` returns the agent's history, and the session corresponds to
    the recorded `AgentId` through `session_ref` — the adapter records the
    correspondence there and `task_graph`'s `Agent` gains no field.

    **`get_session_messages` is a module-level function of the SDK**, not a
    client method, and it is synchronous. Stubbing the module keeps the test
    free of the 376 MB extra while still pinning the call shape, which is the
    part that was wrong.
    """
    calls: list[tuple[str, str | None]] = []
    entries = [{"type": "user"}, {"type": "assistant"}]

    def fake_get_session_messages(session_id, directory=None, limit=None, offset=0):
        calls.append((session_id, directory))
        return entries

    monkeypatch.setitem(
        sys.modules,
        "claude_agent_sdk",
        types.SimpleNamespace(get_session_messages=fake_get_session_messages),
    )

    client = FakeClient()
    backend = _backend(client)
    client.responses = [[Message(session_id="sess-abc")]]
    backend.start()
    history = backend.query()

    assert history.entries == entries
    assert history.session_ref == "sess-abc"
    # The zone is the `cwd` the CLI was given, so it is the project directory
    # the transcript is filed under.
    assert calls == [("sess-abc", "/tmp/z")]
    assert "entries" in type(history).model_fields  # untyped on purpose


def test_query_before_anything_ran_is_empty_not_an_error() -> None:
    """No submission means no session. The SDK's own contract for an unknown id
    is an empty list, so the adapter answers the same way rather than inventing
    a second convention — and never reaches the import."""
    backend = _backend(FakeClient())

    history = backend.query()

    assert history.entries == []
    assert history.session_ref is None


# --------------------------------------------------------------------------- #
# §8.5 — the projection, and what `subtype` is not


def test_a_failed_result_does_not_report_detail_success() -> None:
    """**`subtype` is not a verdict.** Measured against 0.2.148: a run that
    failed to authenticate came back `subtype='success'`, `is_error=True`,
    `terminal_reason='api_error'`, `api_error_status=None`, `cost=0.0`.

    The old projection was `api_error_status or subtype`, so that record read
    `status=FAILED, detail='success'` — two fields contradicting each other,
    with the more readable one wrong.
    """
    client = FakeClient()
    client.responses = [
        [
            Message(
                subtype="success",
                is_error=True,
                terminal_reason="api_error",
                api_error_status=None,
                result="Not logged in · Please run /login",
            )
        ]
    ]
    backend = _backend(client)

    result = backend.start()

    assert result.status is AgentStatus.FAILED
    assert result.detail == "api_error"
    # The model-facing text never crosses, whatever else does.
    assert "Not logged in" not in result.detail


def test_a_success_still_reads_success() -> None:
    """A successful run carries `terminal_reason='completed'`. Letting that
    through would rewrite every success's record to fix a failure-only bug."""
    client = FakeClient()
    client.responses = [[Message(terminal_reason="completed")]]

    result = _backend(client).start()

    assert result.status is AgentStatus.FINISHED
    assert result.detail == "success"


# --------------------------------------------------------------------------- #
# The outputs section — the only channel that tells the agent where its work goes


def _with_outputs(**assignment: Any) -> ClaudeSdkBackend:
    return ClaudeSdkBackend(
        "claude_sdk",
        {"client": FakeClient(), **assignment.pop("config", {})},
        Assignment(goal="g", **assignment),
    )


def test_the_outputs_section_reaches_the_model_with_the_brief() -> None:
    """It has to arrive in the same channel as the brief, because an
    environment variable cannot instruct a conversation — which is exactly how
    the first real model call finished `success` having written nothing."""
    backend = _with_outputs(readme="THE BRIEF", outputs_brief="OUTPUTS: summary -> /s/v0/content")

    prompt = backend._options()["system_prompt"]

    assert "THE BRIEF" in prompt
    assert "OUTPUTS: summary -> /s/v0/content" in prompt


def test_the_outputs_section_survives_a_caller_supplied_system_prompt() -> None:
    """**The judgement, and it is the one that matters.**

    A caller-supplied `system_prompt` wins over `readme`, and that is right: a
    caller pinning a brief is replacing something the *package* could have
    written. **The outputs section is not that.** It names each declared output
    and its resolved path — allocated at dispatch, under `<store>/<hid>/v<N>/`
    — so a caller **cannot** have authored it.

    Dropping it with the brief would remove the only channel by which the agent
    learns where its work goes, which is the defect measured today one layer
    up: the brief named `AGENT_SYS_DEMO_OUTSIDE` twice, named no output path,
    and the agent finished cleanly having written nothing.

    > An override may replace what the caller could have written; it may not
    > remove what only the runner knows.
    """
    backend = _with_outputs(
        readme="THE BRIEF",
        outputs_brief="OUTPUTS: summary -> /s/v0/content",
        config={"options": {"system_prompt": "A CALLER'S OWN PROMPT"}},
    )

    prompt = backend._options()["system_prompt"]

    assert prompt.startswith("A CALLER'S OWN PROMPT")
    assert "THE BRIEF" not in prompt  # the override did replace the brief
    assert "OUTPUTS: summary -> /s/v0/content" in prompt  # and did not remove the destination


def test_no_outputs_section_leaves_the_prompt_exactly_as_it_was() -> None:
    """A task with no declared outputs must not gain a trailing blank section,
    and a `program` executor's assignment carries none at all."""
    backend = _with_outputs(readme="THE BRIEF")

    assert backend._options()["system_prompt"] == "THE BRIEF"


def test_the_outputs_section_arrives_even_with_no_brief() -> None:
    """`readme` is empty for a task whose body declares none. The destination
    is still the runner's to state, so it must not be conditional on the brief
    existing — the `strip()` is what keeps it from arriving with a leading
    blank line."""
    backend = _with_outputs(outputs_brief="OUTPUTS: summary -> unresolved")

    assert backend._options()["system_prompt"] == "OUTPUTS: summary -> unresolved"


# --------------------------------------------------------------------------- #
# The harness's own permission layer, and the switch that stands it down


def test_the_harness_layer_stays_on_while_we_enforce() -> None:
    """**The dangerous direction, and this is the assertion that says so.**

    A `bypassPermissions` that fires while our own enforcement is on is worse
    than the bug it fixes: it is silent, and it is in the one direction that
    matters.

    **This used to read the default and now states the mode**, because on
    2026-08-30 the default flipped to unenforced (`interfaces.md` §4.22f) and
    `_with_outputs()` with no argument now means *off*. Reading a default would
    have made this test change its meaning without anyone editing it — it would
    still pass, against the other mode. So the mode is named.
    """
    backend = _with_outputs(readme="R", permissions_enforced=True)

    assert "permission_mode" not in backend._options()


def test_the_harness_layer_stands_down_by_default() -> None:
    """The other half of the flip, and the reason it is not merely cosmetic.

    `Assignment.permissions_enforced` defaults to `False` now, so a caller that
    has not heard of the switch gets a harness that can actually run a tool.
    The old default — `True` — would leave the SDK in ask-for-approval mode with
    no approval channel while our own sandbox is down, which is **more**
    restricted than enforcing, not less.
    """
    backend = _with_outputs(readme="R")

    assert backend._options()["permission_mode"] == "bypassPermissions"


def test_the_switch_stands_the_harness_layer_down_too() -> None:
    """**The live wall, measured.** Under `AGENT_SYS_NO_PERMISSIONS=1` the agent
    knew its output path, tried `echo`, `grep` and `printenv` to read it — all
    three blocked — then tried to `Write` **inside its own zone** and got
    *"requested permissions … but you haven't granted it yet"*.

    The CLI was in its default ask-for-approval mode with no approval channel,
    so nothing could ever be approved: **the run was more restricted with
    permission management off than with it on.** We switched off our sandbox
    and left the SDK's in the way.
    """
    backend = _with_outputs(readme="R", permissions_enforced=False)

    assert backend._options()["permission_mode"] == "bypassPermissions"


def test_a_caller_may_still_pin_its_own_permission_mode() -> None:
    """`setdefault`, so an explicit `config["options"]["permission_mode"]` wins.

    Unlike the outputs section, this **is** something a caller can author, and
    a caller that names a mode has said something the switch has not.
    """
    backend = _with_outputs(
        readme="R",
        permissions_enforced=False,
        config={"options": {"permission_mode": "plan"}},
    )

    assert backend._options()["permission_mode"] == "plan"


def test_the_switch_does_not_touch_the_hook_s_deny_path() -> None:
    """`permission_mode` is step 4 of the SDK's six-step evaluation, so the
    `PreToolUse` hook is untouched by it.

    Routing the bypass through the hook would have meant returning `allow`,
    and `pre_tool_use`'s own docstring records that an allow decision **also
    skips `can_use_tool`** — the one thing that hook must never do. This pins
    that the fix went to the option and not to the hook.
    """
    backend = _with_outputs(readme="R", permissions_enforced=False)
    assert backend._options()["permission_mode"] == "bypassPermissions"

    decision = asyncio.run(pre_tool_use(lambda payload: "no")({"tool_name": "Bash"}, "id", None))

    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"


# --------------------------------------------------------------------------- #
# O2 / ROADMAP §9.2 — whose CLI is *the* CLI


def test_the_backend_runs_the_cli_env_mgr_reported() -> None:
    """Measured on this machine, 0.2.148: `_find_cli()` returns the SDK's own
    bundled executable before it ever consults `PATH`, and the two are different
    binaries at different versions (2.1.251 bundled, 2.1.246 on `PATH`). So the
    backend must be told which one `env_mgr` installed plugins into."""
    backend = ClaudeSdkBackend(
        "claude_sdk",
        {"client": FakeClient()},
        Assignment(
            goal="g",
            environment={"CLAUDE_CONFIG_DIR": "/tmp/z/.claude"},
            agent_cli="/usr/bin/claude",
        ),
    )

    assert backend._options()["cli_path"] == "/usr/bin/claude"


def test_a_prepared_run_with_no_reported_cli_refuses() -> None:
    """**A silent fallback is the wrong failure** — `interfaces.md` §4.11. Both
    binaries work, so choosing the wrong one produces a successful run whose
    agent is missing its own recipe's plugins; nothing anywhere goes red."""
    backend = ClaudeSdkBackend(
        "claude_sdk", {"client": FakeClient()}, Assignment(goal="g", environment={})
    )
    backend.assignment.environment = {"CLAUDE_CONFIG_DIR": "/tmp/z/.claude"}

    with pytest.raises(BackendUnsupported) as caught:
        backend._options()

    assert "agent_cli" in str(caught.value)


def test_an_unprepared_backend_leaves_the_choice_to_the_sdk() -> None:
    """No `env_mgr` claim, nothing to contradict. A test or a third party
    pinning its own handle must not be made to satisfy a report it never had."""
    backend = _backend(FakeClient())

    assert "cli_path" not in backend._options()


def test_an_explicit_cli_path_outranks_the_report() -> None:
    """`config["cli_path"]` is a caller pinning a binary deliberately."""
    backend = ClaudeSdkBackend(
        "claude_sdk",
        {"client": FakeClient(), "cli_path": "/pinned/claude"},
        Assignment(
            goal="g",
            environment={"CLAUDE_CONFIG_DIR": "/tmp/z/.claude"},
            agent_cli="/usr/bin/claude",
        ),
    )

    assert backend._options()["cli_path"] == "/pinned/claude"


# --------------------------------------------------------------------------- #
# §8.2 / D4 — the hook rule


def test_the_pre_tool_use_hook_never_returns_allow() -> None:
    """**The single easiest thing to get wrong here**, and the spec does not
    mention it: a `PreToolUse` hook returning `allow` also skips
    `can_use_tool`, so the natural phrasing — *return allow when the check
    passes* — silently disables every downstream check for that call."""
    passing = asyncio.run(pre_tool_use()({"tool_name": "Bash"}, "id", None))
    refusing = asyncio.run(
        pre_tool_use(lambda payload: "outside the zone")({"tool_name": "Bash"}, "id", None)
    )
    assert passing == {}
    assert refusing["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "allow" not in _string_literals(pre_tool_use)


def _string_literals(function: Any) -> set[str]:
    """Every string the function can *emit*, docstrings excluded.

    Over the AST rather than the text, for the reason
    `tests/interfaces/test_import_rules.py` gives: the docstring explains at
    length what returning `allow` would do, so a substring check would fail on
    the explanation rather than on the code.
    """
    tree = ast.parse(inspect.getsource(function))
    docstrings = {
        ast.get_docstring(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module))
    }
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value not in docstrings
    }


# --------------------------------------------------------------------------- #
# §8.3 — on_started


def test_on_started_fires_when_connect_returns() -> None:
    """`connect()` performs an `initialize` control-protocol handshake, which
    is a real signal for "the agent really started" rather than "was asked
    to"."""
    client = FakeClient()
    backend = _backend(client)
    seen: list[bool] = []
    backend.start_async(lambda: seen.append(client.connected))
    backend.mainloop()
    assert seen == [True]


# ------------------------------------------------- spec §5.5's remote tool surface


class _ToolDef:
    """`env_mgr.remote.tools.ToolDef`'s shape, without importing `env_mgr`.

    Deliberately a local stand-in: `agent` may not import `env_mgr`, and a test
    that reached for the real class would be asserting an import edge the
    package does not have.
    """

    def __init__(self, name, description, schema, call):  # noqa: ANN001, ANN204
        self.name = name
        self.description = description
        self.schema = schema
        self.call = call


def _defs(call=None):  # noqa: ANN001, ANN202
    return (
        _ToolDef(
            "env_remote_run",
            "Run a command on the remote side.",
            {
                "type": "object",
                "properties": {"command": {"type": "array", "items": {"type": "string"}}},
                "required": ["command"],
                "additionalProperties": False,
            },
            call or (lambda **kw: {"returncode": 0, "stdout": "", "stderr": ""}),
        ),
    )


def test_tools_become_an_in_process_server_with_the_names_the_cli_uses() -> None:
    """`mcp__<server>__<tool>` is **measured**, not read: that spelling appears
    nowhere in the SDK, because it is the CLI's
    (`scratch/single-real-task-2026-08/c_probe_sdk_tool_reachable.py`).

    A wrong name here is a tool the model cannot call, which is indistinguishable
    from the tool not existing — so it is pinned.
    """
    pytest.importorskip("claude_agent_sdk")
    from agent.backends.claude_sdk import _tool_server

    server, names = _tool_server(_defs())
    assert names == ["mcp__env_mgr__env_remote_run"]
    assert server["type"] == "sdk"
    assert server["name"] == "env_mgr"


def test_the_handler_runs_the_blocking_call_off_the_event_loop() -> None:
    """`conn.run` is a blocking `subprocess` and a bring-up runs for minutes.
    Awaiting it inline would stall every other SDK message for that whole time,
    so it goes through `asyncio.to_thread`.

    Asserted by running **the real handler** and observing which thread the
    blocking function landed on. A test that rebuilt the wrapper inline would be
    checking a copy of the code against itself.
    """
    pytest.importorskip("claude_agent_sdk")
    import asyncio
    import json
    import threading

    from agent.backends.claude_sdk import _adapt_tool

    seen: dict[str, int] = {}

    def blocking(**kwargs: object) -> dict[str, object]:
        seen["call"] = threading.get_ident()
        return {"returncode": 0, "stdout": "hi", "stderr": ""}

    adapted = _adapt_tool(_defs(blocking)[0])

    async def drive() -> dict:
        seen["loop"] = threading.get_ident()
        return await adapted.handler({"command": ["true"]})

    out = asyncio.run(drive())
    assert json.loads(out["content"][0]["text"])["stdout"] == "hi"
    assert seen["call"] != seen["loop"], "the blocking call ran on the event loop's thread"


def test_a_refusing_tool_propagates_its_message(monkeypatch) -> None:
    """`remote.tools._inside` raises `PermissionError`, and the SDK turns a
    raising handler into an `isError` result carrying `str(e)` — measured end to
    end in `scratch/single-real-task-2026-08/c_probe_tool_refusal_visible.py`,
    where the text reached the model verbatim and it kept working.

    So the adapter deliberately does **not** catch. This pins that: the
    exception must escape the handler with its message intact, because a
    well-meant `except` here would replace a refusal the agent can act on with
    one it cannot.
    """
    pytest.importorskip("claude_agent_sdk")
    import asyncio

    from agent.backends.claude_sdk import _adapt_tool

    def refuse(**kwargs: object) -> dict[str, object]:
        raise PermissionError("'../x' resolves outside your zone '/z'.")

    adapted = _adapt_tool(_defs(refuse)[0])
    with pytest.raises(PermissionError, match="outside your zone"):
        asyncio.run(adapted.handler({"command": ["true"]}))


def test_the_tools_reach_the_options_the_sdk_is_constructed_with() -> None:
    """**The wiring, and it is the half criterion 18 never had.**

    `remote/tools.py` has been complete since it was written and unreachable by
    any agent, because nothing carried a `ToolDef` across the seam. Testing the
    adapter alone would repeat that mistake one layer up — so this asserts on
    what `_options()` actually hands `ClaudeAgentOptions`.

    Both gates are checked: `mcp_servers` makes a tool *available* and
    `allowed_tools` makes it *permitted*, and they are separate.
    """
    pytest.importorskip("claude_agent_sdk")
    backend = _with_outputs(zone="/z", tools=_defs())

    options = backend._options()

    assert "env_mgr" in options["mcp_servers"]
    assert options["mcp_servers"]["env_mgr"]["type"] == "sdk"
    assert "mcp__env_mgr__env_remote_run" in options["allowed_tools"]


def test_no_tools_means_no_mcp_server_at_all() -> None:
    """**The control**, and it is the configuration every run without a mapping
    uses. An agent with no far side must see *no tool*, rather than a tool that
    fails when called — so nothing is added to the options at all.

    Without this, the test above would pass equally well against an adapter that
    published an empty server unconditionally.
    """
    backend = _with_outputs(zone="/z")

    options = backend._options()

    assert "mcp_servers" not in options
    assert not options.get("allowed_tools")
