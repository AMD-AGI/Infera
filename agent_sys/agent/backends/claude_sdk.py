"""The `claude-agent-sdk` adapter — design §8. Verified against 0.2.144.

**This module is never imported at module scope by anything in `agent/`.** The
SDK is a 376 MB extra costing ~1.3 s to import, of which 328 MB is a single
bundled executable, so a hard dependency makes every `agent_sys` entry point —
including `env-mgr check`, which has nothing to do with agents — pay both. A
missing extra is therefore a `BackendUnsupported` naming the extra, and not an
`ImportError` at start-up in a process that was never going to use it.

The import itself happens inside the constructor, which is the probe (§6.4).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

from agent.backend import (
    AgentHistory,
    AgentResult,
    AgentStatus,
    Assignment,
    BackendUnsupported,
    ExecutorBase,
    history_of,
)

__all__ = ["DRAIN_SECONDS", "USAGE_KEYS", "ClaudeSdkBackend", "pre_tool_use"]

#: What crosses into `AgentResult`, and nothing else — design §8.5.
#:
#: `ResultMessage` also carries `result` (the final response text),
#: `structured_output` and `permission_denials`, and exactly one of its fields
#: is annotated "safe to log". **The adapter projects a named subset. It never
#: stores the `ResultMessage`.** Criterion 16 is about the *system's* record,
#: and persisting the whole message would put prompt-derived text in it.
USAGE_KEYS: tuple[str, ...] = ("duration_ms", "num_turns", "total_cost_usd")

#: The bound on the interrupt drain, and it is load-bearing.
#: `terminal_reason` is documented as `None` on CLI versions predating the
#: field, on results that bypassed the query loop, and on synthesized error
#: results after a fatal failure. A drain that waits for an `aborted_*` value
#: and nothing else hangs in exactly the case where the session has already
#: died.
DRAIN_SECONDS = 30.0

_ABORTED = ("aborted_streaming", "aborted_tools")

#: The marker that this run went through `env_mgr.material.deploy` at all —
#: always set when it did (`Assignment.environment`). Without it there is no
#: preparation to contradict, and the SDK choosing its own CLI is not a defect.
_PREPARED_MARKER = "CLAUDE_CONFIG_DIR"

#: The in-process MCP server the remote tools are published under. It becomes
#: part of the name the model calls — `mcp__env_mgr__env_remote_run` — so it is
#: a compatibility surface, not a label.
_TOOL_SERVER = "env_mgr"


def _tool_server(tools: Sequence[Any]) -> tuple[Any, list[str]]:
    """`env_mgr`'s `ToolDef`s as an in-process MCP server, and their full names.

    Three adaptations, each measured rather than assumed:

    **sync to async.** `ToolDef.call` is a plain function taking keyword
    arguments; an `SdkMcpTool` handler is a coroutine taking one `args` dict.
    `conn.run` is a blocking `subprocess`, so it goes through
    `asyncio.to_thread` — awaiting it inline would stall the SDK's event loop
    for the whole duration of a remote command, which for a bring-up is minutes.

    **The return shape.** `ToolDef.call` returns a plain dict; MCP wants
    `{"content": [...]}`. Serialised as JSON text, because these are
    `returncode`/`stdout`/`stderr` and a `SyncReport`, all of which a model
    reads better as one block than as prose.

    **Refusals propagate, and that is measured, not assumed.**
    `remote.tools._inside` raises `PermissionError`, and the SDK catches a
    raising handler and returns `isError` with `str(e)`
    (`claude_agent_sdk/__init__.py:595-615`). That is the SDK speaking for its
    own layer, so it was checked end to end
    (`scratch/single-real-task-2026-08/c_probe_tool_refusal_visible.py`): the
    text reaches the model verbatim **and the model keeps working afterwards**
    rather than treating the tool as broken. So there is no catch-and-re-wrap
    here — it would only hide the message that already arrives.

    Argument validation needs nothing either: the SDK runs `jsonschema.validate`
    against `input_schema` *before* the handler, and `ToolDef.schema` is already
    JSON Schema with `additionalProperties: False`.
    """
    from claude_agent_sdk import create_sdk_mcp_server  # noqa: PLC0415

    adapted = [_adapt_tool(defn) for defn in tools]
    server = create_sdk_mcp_server(name=_TOOL_SERVER, version="1.0.0", tools=adapted)
    return server, [f"mcp__{_TOOL_SERVER}__{defn.name}" for defn in tools]


def _adapt_tool(defn: Any) -> Any:
    """One `ToolDef` as an `SdkMcpTool`. **Module level so it can be driven.**

    `create_sdk_mcp_server` keeps its tools privately, so a test that only had
    the server could not call the handler — and a test that rebuilt the same
    wrapper inline would be asserting against a copy of the code rather than
    against the code. Splitting it here is what lets the handler itself be run.
    """
    from claude_agent_sdk import SdkMcpTool  # noqa: PLC0415

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(lambda: defn.call(**args))
        return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}

    return SdkMcpTool(
        name=defn.name,
        description=defn.description,
        input_schema=defn.schema,
        handler=handler,
    )


def pre_tool_use(refuse: Any = None) -> Any:
    """Our `PreToolUse` hook — **and it never returns `allow`**. Design §8.2, D4.

    The single easiest thing to get wrong here, and the spec does not mention
    it. The SDK's permission evaluation is six steps — hooks, deny rules, ask
    rules, permission mode, allow rules, `can_use_tool` — and `can_use_tool`'s
    own docstring records the coupling: *"a `PreToolUse` hook returning an
    allow decision also skips this callback."* `permissionDecision` is
    `NotRequired`, and omitting it lets the call flow through the normal
    evaluation.

    So the natural phrasing — *return allow when the check passes* — silently
    disables every downstream check for that call.

    > **MUST: this hook returns `deny`, or omits `permissionDecision`.**

    **The hook is still not the boundary.** It sees
    `{"tool_name": "Bash", "command": "python3 x.py"}` with no file path in it;
    `env_mgr` spec §4 owns confinement, and the SDK's own bash sandbox is a
    third layer rather than a replacement for either.

    `refuse(payload) -> str | None` returns a reason to deny, or `None` to let
    the call flow through. The default refuses nothing.
    """

    async def hook(payload: Any, tool_use_id: Any, context: Any) -> dict[str, Any]:
        reason = refuse(payload) if refuse is not None else None
        if reason is None:
            return {}  # flow through; NEVER {"permissionDecision": "allow"}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    return hook


def _as_mapping(entry: Any) -> dict[str, Any]:
    """A transcript entry as plain data. `AgentHistory.entries` is untyped on
    purpose (`backend.py`), so this flattens rather than models: 0.2.148 returns
    `SessionMessage` dataclasses, and a mapping already suits."""
    if isinstance(entry, Mapping):
        return dict(entry)
    if dataclasses.is_dataclass(entry) and not isinstance(entry, type):
        return dataclasses.asdict(entry)
    return {"value": str(entry)}


def _detail_of(message: Any, subtype: str, failed: bool) -> str:
    """What went wrong, in the SDK's own vocabulary and none of the prompt's.

    **`subtype` is not a verdict.** Measured against 0.2.148: a run that failed
    to authenticate came back `subtype='success'`, `is_error=True`,
    `terminal_reason='api_error'`, `api_error_status=None`, `duration_ms=89`,
    `total_cost_usd=0.0`. The previous projection was `api_error_status or
    subtype`, so that result reported `status=FAILED, detail='success'` — a
    record whose two fields contradict each other, and whose detail is the more
    readable of the two.

    `terminal_reason` is what actually distinguishes the cases (`completed`,
    `api_error`, `aborted_streaming`, `aborted_tools`). It is a categorical
    value rather than content, which is the property that let `api_error_status`
    cross in the first place — so it crosses on the same terms, and `result`
    (which held `'Not logged in · Please run /login'`, model-facing text) still
    does not.

    **Only a failure consults it.** A successful run carries
    `terminal_reason='completed'`, which is true and is not what this field is
    for — letting it through would rewrite every success's record from
    `'success'` to `'completed'` to fix a bug that only appears on failures.

    **Module-level, not a method**, because `_project` reads nothing off `self`
    and `test_records.py` asserts exactly that by calling it on a stand-in.
    """
    if not failed:
        return str(subtype)
    for field in ("api_error_status", "terminal_reason"):
        value = getattr(message, field, None)
        if value:
            return str(value)
    return str(subtype)


class ClaudeSdkBackend(ExecutorBase):
    """Level 2. `ClaudeSDKClient` is the durable handle; this is a projection
    and a drain."""

    def __init__(
        self,
        key: str = "claude_sdk",
        config: Mapping[str, Any] | None = None,
        assignment: Assignment | None = None,
    ) -> None:
        super().__init__(key, assignment)
        self.config = dict(config or {})
        self._loop = asyncio.new_event_loop()
        self._loop_lock = threading.Lock()
        # Initialised here rather than read with a `getattr` default: "have I
        # connected?" is a question about my own state and there is exactly one
        # place that can answer it truthfully.
        self._connected = False
        #: The SDK's session id, learned from the messages that carry it. See
        #: `session_ref` for why it cannot be read off the client.
        self._session_ref: str | None = None
        self._client = self._build_client()

    # ---- the probe is the constructor ------------------------------------ #

    def _build_client(self) -> Any:
        """Construct the SDK client, importing the extra here and nowhere else.

        A test supplies `config["client"]`, which is also how a third party
        pins a pre-configured handle. Nothing in `pytest agent_sys` reaches the
        import path.
        """
        supplied = self.config.get("client")
        if supplied is not None:
            return supplied
        try:
            from claude_agent_sdk import (  # noqa: PLC0415 — see the module docstring
                ClaudeAgentOptions,
                ClaudeSDKClient,
            )
        except ImportError as exc:
            raise BackendUnsupported(
                self.key,
                "run here",
                "the `claude` extra is not installed: pip install 'agent-sys-helper[claude]'",
            ) from exc
        return ClaudeSDKClient(ClaudeAgentOptions(**self._options()))

    def _options(self) -> dict[str, Any]:
        """What the prepared environment decides, and the one hook rule.

        **`cli_path` comes from the prepared environment when there is one.**
        `_find_cli()` prefers the SDK's bundled 328 MB executable and falls back
        to `shutil.which("claude")`, while `env_mgr/installers/claude.py`
        installs plugins into whatever is on `PATH` — so by default the backend
        runs the CLI `env_mgr` never touched, and an agent would not see the
        plugins its own recipe installed.

        **This is a defect, not a division of labour.** An earlier revision of
        this docstring closed with *"O2 records that the decision is really
        `env_mgr`'s"*, which reads as an assignment and is not one: nothing was
        assigned, nothing is owed, and the silent-wrong-CLI behaviour ships. The
        two ends disagree about which binary is *the* CLI, and the disagreement
        is invisible — a recipe installs plugins, the run succeeds, and the
        agent simply does not have them.

        **What it needs:** `env_mgr` reports the CLI it installed into, and this
        backend uses that one or **refuses**. A silent fallback to a different
        binary is the wrong failure — it is `interfaces.md` §4.11's family, a
        plausible value consumed as if it were the right one. `ROADMAP.md` §9.2
        carries it with the confinement work, because both turn on the same
        question of who owns the CLI process.
        """
        # **`dict()` is one level and the write below is two.**
        # `options.setdefault("mcp_servers", {})[_TOOL_SERVER] = …` inserts a
        # fresh dict when the key is absent — the ordinary case, and why this was
        # never seen. When the caller's config *does* carry an `mcp_servers` map,
        # `setdefault` returns that nested dict **by reference** and the write
        # lands in it. `selection.py` passes `decl.config`: the agent-spec
        # declaration's own object, shared by every attempt of every task using
        # that spec. So the escape is into configuration, not into one run, and
        # a user-configured server named `env_mgr` is overwritten with it.
        #
        # The nested copy is made where the write happens, below, rather than
        # here — `copy.deepcopy` would traverse every value an operator put in
        # `options` (hooks, clients, anything holding a lock) to protect the one
        # key this method touches.
        #
        # A review reported this as unconditional and it is not: measured, the
        # default configuration never reaches the aliasing branch. Fixed anyway,
        # because the condition is an operator's key and not an invariant.
        options = dict(self.config.get("options") or {})
        if self.assignment.zone:
            options.setdefault("cwd", self.assignment.zone)
        if self.assignment.readme:
            options.setdefault("system_prompt", self.assignment.readme)
        if self.assignment.outputs_brief:
            # **Appended, and it survives a caller-supplied `system_prompt`.**
            #
            # `setdefault` above lets `config["options"]["system_prompt"]` win,
            # which is right for the *brief*: a caller pinning one is replacing
            # something the package could have written. **The outputs section is
            # not that.** It names each declared output and its resolved path —
            # `<store>/<hid>/v<N>/content`, allocated at dispatch — so a caller
            # **cannot** have authored it, and dropping it with the brief would
            # remove the only channel by which the agent learns where its work
            # goes.
            #
            # That is today's failure one layer up, and it is measured: the
            # first real model call finished `success` having written nothing,
            # because the brief named `AGENT_SYS_DEMO_OUTSIDE` twice and named
            # no output path at all. An override that silently removes the
            # destination reproduces it for every caller that sets one.
            #
            # **So the rule is: an override may replace what the caller could
            # have written; it may not remove what only the runner knows.**
            base = str(options.get("system_prompt") or "")
            options["system_prompt"] = f"{base}\n\n{self.assignment.outputs_brief}".strip()
        if not self.assignment.permissions_enforced:
            # **The harness's own permission layer, switched off with ours.**
            #
            # `interfaces.md:1560` records it as an aside — *"in practice a
            # harness runs with `bypassPermissions` on"* — and nothing set it.
            # Measured live under `AGENT_SYS_NO_PERMISSIONS=1`: the agent knew
            # its output path, tried `echo`, `grep` and `printenv` to read it,
            # and all three were blocked; a `Write` **inside its own zone** came
            # back *"requested permissions … but you haven't granted it yet"*.
            # The CLI was in its default ask-for-approval mode with no approval
            # channel, so nothing could ever be approved. **The run was more
            # restricted with permission management off than with it on.**
            #
            # **Gated, never unconditional.** The switch is one fact with one
            # reader; a harness left permissive while our own enforcement is on
            # is the reverse defect, and it is the silent one. `Assignment`
            # defaults this to `True`, so anything that has not heard of the
            # switch keeps the harness's layer.
            #
            # **This removes no boundary we rely on.** `pre_tool_use` above
            # says the SDK's layer is a third layer rather than a replacement,
            # and `env_mgr`'s confinement is the real one — which the same
            # switch has already turned off for this run.
            #
            # `permission_mode` is step 4 of the SDK's six-step evaluation, so
            # **the hook's deny path is untouched**: routing this through the
            # hook instead would have meant returning `allow`, which also skips
            # `can_use_tool` — the one thing `pre_tool_use` is documented never
            # to do.
            options.setdefault("permission_mode", "bypassPermissions")
        if self.assignment.environment:
            options.setdefault("env", dict(self.assignment.environment))
        if self.assignment.mcp_servers:
            # **The per-agent components' external servers, under the tool
            # server's own collision policy.** Not a second policy: the reason a
            # name may not be taken twice is that `mcp__<server>__<tool>` is what
            # the model calls, and that is true of a component's server exactly
            # as it is of `env_mgr`'s. So the refusal below is the same refusal,
            # said about a different name.
            #
            # **Refused rather than merged, and the direction matters.** The
            # caller's `options["mcp_servers"]` is an operator's configuration;
            # `assignment.mcp_servers` is what a package declared. Letting either
            # win silently means one of the two gets different tools than the
            # ones they wrote, with nothing said — the defect the `env_mgr` key
            # already carries a comment about, one collision wider.
            servers = dict(options.get("mcp_servers") or {})
            clash = sorted(set(servers) & set(self.assignment.mcp_servers))
            if clash:
                raise BackendUnsupported(
                    self.key,
                    "mcp_servers",
                    f"this config and this agent's components both declare MCP "
                    f"server(s) {clash}. The model addresses these as "
                    f"mcp__<server>__<tool>, so two servers cannot share a name — "
                    f"rename one side rather than letting the other's tools "
                    f"disappear.",
                )
            servers.update(self.assignment.mcp_servers)
            options["mcp_servers"] = servers
        if self.assignment.tools:
            # **Spec §5.5's remote surface, and the only place that knows the
            # SDK.** `env_mgr` may not import the SDK and `agent/backend.py` is
            # backend-agnostic, so the `ToolDef` -> `SdkMcpTool` adapter belongs
            # here beside every other option this file assembles.
            server, names = _tool_server(self.assignment.tools)
            # **A collision is named, not resolved.** This was
            # `options.setdefault("mcp_servers", {})[_TOOL_SERVER] = server`,
            # which silently replaced a caller's own server of the same name —
            # the only line in this method that overwrites rather than
            # `setdefault`s, and the one whose key is a compatibility surface
            # (`mcp__env_mgr__…` is what the model calls). An operator who lost
            # their tools that way would get no message and no failure, just
            # different tools than the ones they configured.
            servers = dict(options.get("mcp_servers") or {})
            if _TOOL_SERVER in servers:
                raise BackendUnsupported(
                    self.key,
                    "mcp_servers",
                    f"this config declares an MCP server named {_TOOL_SERVER!r}, which is "
                    f"the name env_mgr publishes its remote tools under. Rename yours: "
                    f"the model addresses these as mcp__{_TOOL_SERVER}__<tool>, so two "
                    f"servers cannot share the name.",
                )
            servers[_TOOL_SERVER] = server
            options["mcp_servers"] = servers
            # `mcp_servers` makes them *available*; `allowed_tools` makes them
            # *permitted*, and they are separate gates. Measured 2026-09-01, SDK
            # 0.2.148: the CLI addresses an in-process tool as
            # `mcp__<server>__<tool>` -- a spelling that appears nowhere in the
            # SDK, because it is the CLI's. See
            # `scratch/single-real-task-2026-08/c_probe_sdk_tool_reachable.py`.
            options["allowed_tools"] = [*options.get("allowed_tools", []), *names]
        cli_path = self.config.get("cli_path") or self._prepared_cli()
        if cli_path:
            options.setdefault("cli_path", cli_path)
        options.setdefault("hooks", self.config.get("hooks") or {})
        return options

    def _prepared_cli(self) -> str | None:
        """The CLI `env_mgr` installed into, **or a refusal** — O2's other half.

        Measured, 0.2.148 on this machine: `_find_cli()` returns the SDK's own
        bundled executable *before* it ever calls `shutil.which`, and the two are
        different binaries at different versions —
        `.../claude_agent_sdk/_bundled/claude` is 2.1.251 while the `claude` on
        `PATH` that `env_mgr` installs plugins into is 2.1.246. Left to itself the
        backend runs the one nobody configured, and the agent silently lacks its
        own recipe's plugins.

        Both binaries work, so nothing fails — which is the whole problem, and
        why the missing report is a refusal rather than a fallback
        (`interfaces.md` §4.11).

        **Only when this run was prepared.** A backend constructed directly — a
        test, a third party pinning its own handle — has no `env_mgr` claim to
        contradict, so the SDK's choice stands.

        **`Assignment.agent_cli`, not an environment variable.** The first
        version read `environment["AGENT_SYS_CLAUDE_CLI"]`, a name `env_mgr`
        never published and now never will: the report is `Prepared.agent_cli`,
        *declared* by the `Context` rather than discovered, because `PATH` can
        differ between the process that ran the recipe and the process that runs
        the agent. The unit test that was meant to pin the seam asserted the
        literal against itself, so it stayed green while every prepared AI run
        refused. `tests/interfaces/test_agent_cli_seam.py` pins the field
        instead, from both sides.
        """
        reported = self.assignment.agent_cli
        if reported:
            return reported
        if _PREPARED_MARKER in (self.assignment.environment or {}):
            raise BackendUnsupported(
                self.key,
                "run an AI task in this prepared environment",
                "env_mgr prepared this run and reported no `claude` CLI in "
                "`Prepared.agent_cli`. Letting the SDK choose would run its "
                "bundled executable rather than the binary env_mgr installed "
                "plugins into, and the run would succeed without them",
            )
        return None

    # ---- the asynchronous form ------------------------------------------- #

    def _deploy(self) -> None:
        """`on_started` fires when `connect()` returns — design §8.3.

        `connect()` performs an `initialize` control-protocol handshake with a
        timeout and stores `_initialization_result`, which `get_server_info()`
        returns. That is a real signal for "the agent really started", which is
        what spec §4.3 asks the callback to mean.
        """
        if not self._connected:
            self._await(self._client.connect())
            self._connected = True

    def _run(self) -> AgentResult:
        started = time.monotonic()
        prompt = self.assignment.goal or self.assignment.readme
        self._await(self._client.query(prompt))
        message = self._await(self._final_message())
        return self._project(message, time.monotonic() - started)

    def _deliver(self, message: str) -> None:
        """Streaming input on the same session. `instruct` reaches a running
        agent and affects its behaviour **without ending the run**."""
        self._await(self._client.query(message))

    def _terminate(self) -> None:
        if self._connected:
            self._await(self._client.disconnect())
            self._connected = False

    # ---- level 2 ---------------------------------------------------------- #

    def interrupt(self) -> None:
        """Interrupt, then drain to the aborted `ResultMessage`.

        `interrupt()` sends a **control request** on a channel separate from the
        message stream, so nothing touches the buffer: the messages the
        interrupted task already produced, including its `ResultMessage`, stay
        there and must be drained before a new query's response can be read.

        **The drain is not "consume N messages".** `terminal_reason` is
        `aborted_streaming` or `aborted_tools` for an interrupted turn, so the
        interrupted submission's own result is self-identifying — and a
        count-based drain would consume the wrong number for the wrong reason.
        """
        self._await(self._client.interrupt())
        self._await(self._drain())
        self.status = AgentStatus.INTERRUPTED

    def instruct(self, message: str) -> None:
        """Queue it; `mainloop` delivers it. The caller does not touch the
        harness, which is what keeps one writer for the session."""
        self._enqueue_instruction(message)

    def query(self) -> AgentHistory:
        """The agent's history, fetched on demand and never stored.

        **`get_session_messages` is a module-level function, not a client
        method** — measured against 0.2.148, where `ClaudeSDKClient` exposes
        fifteen public names and that is not one of them. An earlier revision
        called `self._client.get_session_messages()`, which raises
        `AttributeError` against the real SDK on every call; the suite stayed
        green because the test double defined the method, so the fake ratified
        the guess instead of checking it (`test_claude_sdk.py`'s `FakeClient`).

        It reads the session's JSONL transcript, so it is **synchronous** and
        takes the session id plus the project directory — which is the `cwd` we
        gave the CLI, i.e. the zone.

        **Only the main agent is interactable** (criterion 12). This does not
        call `list_subagents`, and no method here takes a subagent id.

        Before the first submission there is no session, and the SDK's own
        contract for an unknown id is an empty list rather than an error — so
        this answers the same way instead of inventing a second convention.
        """
        session = self.session_ref
        if session is None:
            return history_of([], None)
        from claude_agent_sdk import get_session_messages  # noqa: PLC0415 — see the docstring

        entries = get_session_messages(session, directory=self.assignment.zone or None)
        return history_of([_as_mapping(entry) for entry in entries or []], session)

    @property
    def session_ref(self) -> str | None:
        """`AgentId` and the SDK's session id are different things; the adapter
        records the correspondence here and `task_graph`'s `Agent` gains no
        field.

        **The id is on the messages, not on the client.** `ClaudeSDKClient` has
        no `session_id` attribute at all — measured, `hasattr` is `False` — so
        the previous `getattr(self._client, "session_id", None)` returned `None`
        on every real run while the test double's invented attribute kept the
        assertion green. `ResultMessage.session_id` carries it, and so does
        every other message, which is why the streams record it as they go.
        """
        return self._session_ref

    # ---- internals -------------------------------------------------------- #

    def _note(self, message: Any) -> Any:
        """Record the session id off any message that carries one.

        Every message type in the SDK declares `session_id`, so the first one
        through either stream settles it. One writer, and it is this.
        """
        session = getattr(message, "session_id", None)
        if session:
            self._session_ref = str(session)
        return message

    async def _final_message(self) -> Any:
        last: Any = None
        async for message in self._client.receive_response():
            last = self._note(message)
        return last

    async def _drain(self) -> None:
        """Drain until the interrupted turn's own result arrives, or the bound
        expires. See `DRAIN_SECONDS` for why the bound is not optional."""
        deadline = time.monotonic() + DRAIN_SECONDS
        stream = self._client.receive_messages()
        try:
            async for message in stream:
                self._note(message)
                if getattr(message, "terminal_reason", None) in _ABORTED:
                    return
                if time.monotonic() > deadline:
                    return
        finally:
            # Leaving an `async for` early abandons the generator, and the loop
            # this ran on is gone by the time the collector reaches it.
            closer = getattr(stream, "aclose", None)
            if closer is not None:
                await closer()

    def _project(self, message: Any, seconds: float) -> AgentResult:
        usage: dict[str, float] = {"seconds": seconds}
        for key in USAGE_KEYS:
            value = getattr(message, key, None)
            if isinstance(value, (int, float)):
                usage[key] = float(value)
        usage.setdefault("turns", usage.get("num_turns", 0.0))
        subtype = getattr(message, "subtype", "success")
        failed = bool(getattr(message, "is_error", False)) or subtype != "success"
        return AgentResult(
            status=AgentStatus.FAILED if failed else AgentStatus.FINISHED,
            usage=usage,
            detail=_detail_of(message, subtype, failed),
        )

    def _await(self, awaitable: Any) -> Any:
        """Run one coroutine on this adapter's loop.

        The loop is driven from whichever thread called, which for the main
        phase is the attempt's. `interrupt` arrives from the monitor's thread
        instead, and the lock is what keeps the two off the loop at once.
        """
        if not asyncio.iscoroutine(awaitable) and not asyncio.isfuture(awaitable):
            return awaitable
        with self._loop_lock:
            return self._loop.run_until_complete(awaitable)
