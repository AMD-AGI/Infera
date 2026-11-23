"""stdio MCP server `o11y`: the agent door over `o11y_door.api` (four tools).

Rules from the experiments (research/mcp.md, live runs with Claude Code 2.1.325):
- No top-level allOf/oneOf/anyOf in a tool schema (Claude Code drops such a tool); inputs are re-validated here
  against the full protocol schema, because Claude Code does not enforce inputSchema.
- Tools are listed only to fleet members (spec §10.4): `AGENT_SYS_O11Y` ∈ {true, child} at start, or the fleet
  registry lists the parent session `active`. When membership appears later, the change reaches the client
  only through a `subscriptions/listen` stream on the 2026-07-28 wire (and `listChanged` is advertised only if
  that method is served), so the server serves it and publishes ToolsListChanged; the 2025-era unsolicited
  notification is sent too.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import anyio
import mcp_types as types
from mcp.server.lowlevel.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.server.subscriptions import InMemorySubscriptionBus, ListenHandler, ToolsListChanged
from o11y_protocol import fleet

from . import __version__, api, claude_sessions

# Seconds between membership checks; the env override exists for tests only.
MEMBERSHIP_POLL_S = float(os.environ.get("AGENT_SYS_O11Y_MEMBERSHIP_POLL_S") or 30)

# One line per kind: required fields and the enums a model most often gets wrong. A refusal returns the
# kind's full schema, so nothing else needs to be listed here (measured budget: spec §10.4).
LEDGER_DOC = """Record one fact in the o11y run ledger (never edit runs.jsonl). Required fields of `record` per kind:
task: task_id, title, metric{name,unit,direction:higher|lower}, baseline{value}
decision: decision_id, by:user|leader|child|external, text
run_start: run_id, theme, point:explore|validate|replicate|baseline|repro|screen|diagnostic|profile|calibration; optional machines[{node}], hypothesis
run_end: run_id, status:kept|neutral|rejected|failed|aborted|cancelled|unscored|inconclusive|done; kept|neutral|rejected need value, evidence{path}; others need conclusion
best: value, run_ids[]
plan: next[{theme}]
idea: idea_id, title, status
campaign: campaign_root (absolute), operator_ids[], started_at (RFC 3339), machine{node}
hotops: profile_id, role, ops[{name,share_pct}]
kernel: op_id, stage
incident: incident_id, severity:info|warning|error, category, summary
finding: finding_id, text, evidence_rank:first_hand|second_hand|inference
Unknown fields are refused; a refusal lists the problems and the kind's full schema. Fix and retry."""

TOOLS = [
    types.Tool(
        name="ledger_append",
        description=LEDGER_DOC,
        input_schema={
            "type": "object",
            "required": ["kind", "record"],
            "properties": {"kind": {"enum": api.ledger_kinds()}, "record": {"type": "object"}},
        },
    ),
    types.Tool(
        name="ledger_query",
        description="Read ledger records (latest version of each fact, newest last). since: RFC 3339.",
        input_schema={
            "type": "object",
            "properties": {
                "kind": {"enum": api.ledger_kinds()},
                "run_id": {"type": "string"},
                "task_id": {"type": "string"},
                "since": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
        annotations=types.ToolAnnotations(read_only_hint=True),
    ),
    types.Tool(
        name="session_register",
        description=(
            "decision child: a child you are about to launch (returns the env and the tmux launch argv to use) "
            "or, with session_id, a running session you adopt. not_in_fleet: declare session_id not part of "
            "your fleet. Both need a purpose; a child needs a name or tmux_target."
        ),
        input_schema={
            "type": "object",
            "required": ["decision"],
            "properties": {
                "decision": {"enum": ["child", "not_in_fleet"]},
                "name": {"type": "string"},
                "purpose": {"type": "string"},
                "session_id": {"type": "string"},
                "tmux_target": {"type": "string"},
                "cwd": {"type": "string"},
            },
        },
    ),
    types.Tool(
        name="fleet_status",
        description="Fleet summary: sessions, task, best, open runs, next plan, recent warnings.",
        input_schema={"type": "object", "properties": {}},
        annotations=types.ToolAnnotations(read_only_hint=True),
    ),
]
TOOL_SCHEMAS = {t.name: t.input_schema for t in TOOLS}


def current_session() -> str | None:
    return claude_sessions.session_of_pid(os.getppid(), fleet.claude_config_dir())


def context(root: Path | None) -> api.Context:
    return api.Context(root=root, session_id=current_session())


def is_member(root: Path | None = None) -> bool:
    """Launched as main or child, or listed active in the fleet registry (review A3)."""
    if os.environ.get(fleet.ACTIVATION_ENV_VAR) in ("true", "child"):
        return True
    sid = current_session()
    if not sid:
        return False
    sessions, _ = api.read_registry(Path(root) if root else fleet.fleet_dir(session_id=sid))
    return any(s.get("session_id") == sid and s.get("active") for s in sessions)


def _result(obj: dict) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(obj, ensure_ascii=False))],
        is_error=not obj.get("ok", True),
    )


def _input_problems(name: str, args: dict) -> list[str]:
    import jsonschema

    v = jsonschema.Draft202012Validator(TOOL_SCHEMAS[name])
    return [f"/{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in v.iter_errors(args)]


def call(ctx: api.Context, name: str, args: dict) -> dict:
    if name not in TOOL_SCHEMAS:
        return {"ok": False, "problems": [f"unknown tool {name}"]}
    if name == "ledger_append" and not isinstance(args.get("record"), dict):
        # tolerate the flat form {kind, field: ...}; the record is validated against runs.v1 either way
        args = {"kind": args.get("kind"), "record": {k: v for k, v in args.items() if k != "kind"}}
    probs = _input_problems(name, args)
    if probs:
        return {"ok": False, "problems": probs, "hint": f"fix the arguments and call {name} again"}
    if name == "ledger_append":
        return api.ledger_append(ctx, {**args["record"], "kind": args["kind"]})
    if name == "ledger_query":
        return api.ledger_query(
            ctx,
            args.get("kind"),
            args.get("run_id"),
            args.get("task_id"),
            args.get("since"),
            args.get("limit", 20),
        )
    if name == "session_register":
        return api.session_register(
            ctx,
            args["decision"],
            args.get("name"),
            args.get("purpose"),
            args.get("session_id"),
            args.get("tmux_target"),
            args.get("cwd"),
        )
    return api.fleet_status(ctx)


def run(root: Path | None = None) -> None:
    state = {"session": None, "member": is_member(root)}
    bus = InMemorySubscriptionBus()

    async def on_list_tools(ctx, _params):
        state["session"] = ctx.session  # connection-scoped outbound, for the 2025-era notification
        state["member"] = state["member"] or is_member(root)
        return types.ListToolsResult(tools=TOOLS if state["member"] else [])

    async def on_call_tool(_ctx, params):
        if not (state["member"] or is_member(root)):
            return _result(
                {"ok": False, "problems": ["this session is not part of an agent_sys o11y fleet"]}
            )
        try:  # identity and root are re-resolved per call: they follow /clear
            return _result(call(context(root), params.name, dict(params.arguments or {})))
        except Exception as exc:  # never kill the server on one bad call
            return _result(
                {"ok": False, "problems": [f"internal error: {exc.__class__.__name__}: {exc}"]}
            )

    async def watch_membership():
        while not state["member"]:
            await anyio.sleep(MEMBERSHIP_POLL_S)
            if is_member(root):
                state["member"] = True
                await bus.publish(ToolsListChanged())
                if state["session"] is not None:
                    try:
                        await state["session"].send_tool_list_changed()
                    except Exception:  # noqa: BLE001 - best effort; the next list sees the tools
                        pass

    async def main():
        server = Server(
            "o11y",
            version=__version__,
            on_list_tools=on_list_tools,
            on_call_tool=on_call_tool,
            on_subscriptions_listen=ListenHandler(bus),
        )
        opts = server.create_initialization_options(NotificationOptions(tools_changed=True))
        async with stdio_server() as (r, w), anyio.create_task_group() as tg:
            tg.start_soon(watch_membership)
            await server.run(r, w, opts)
            tg.cancel_scope.cancel()

    anyio.run(main)


if __name__ == "__main__":  # bin/py -m o11y_door.mcp_server
    run()
