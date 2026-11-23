"""stdio MCP server `o11y` (official python-sdk, low-level Server) over o11y_core.api.

Tool-schema rules from research/mcp.md: no top-level allOf/oneOf/anyOf (Claude Code silently drops such a
tool), $refs inlined, inputs re-validated server-side against the full protocol schema (Claude Code does not
enforce inputSchema), v/id/ts/writer set by the server.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import anyio
import mcp_types as types
from mcp.server.lowlevel.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.server.subscriptions import InMemorySubscriptionBus, ListenHandler, ToolsListChanged

from . import __version__, api, fleet, schemas

SERVER_SET = {"v", "id", "ts"}


def _inline(node, defs):
    if isinstance(node, dict):
        if "$ref" in node:
            target = copy.deepcopy(defs[node["$ref"].split("/")[-1]])
            rest = {k: v for k, v in node.items() if k != "$ref"}
            return _inline({**target, **rest}, defs)
        return {
            k: _inline(v, defs)
            for k, v in node.items()
            if k not in ("allOf", "anyOf", "oneOf", "if", "then")
        }
    if isinstance(node, list):
        return [_inline(x, defs) for x in node]
    return node


KIND_DEF = {
    "task": "task",
    "decision": "decision",
    "run_start": "run_start",
    "run_end": "run_end",
    "best": "best",
    "plan": "plan",
    "idea": "idea",
    "machine": "machine_event",
    "campaign": "campaign",
    "hotops": "hotops",
    "kernel": "kernel",
    "incident": "incident",
    "finding": "finding",
}

# One line per kind: required fields with the enums a model most often gets wrong. Full doc: ledger_schema(kind).
LEDGER_DOC = """Record one fact in the o11y run ledger (never edit runs.jsonl). Required fields of `record` per kind:
task: task_id, title, metric{name,unit,direction:higher|lower}, baseline{value}
decision: decision_id, by:user|leader|child|external, text
run_start: run_id, theme, point:explore|validate|replicate|baseline|repro|screen|diagnostic|profile|calibration; optional machines[{node}], hypothesis
run_end: run_id, status:kept|neutral|rejected|failed|aborted|cancelled|unscored|inconclusive|done; kept|neutral|rejected need value, evidence{path}; others need conclusion
best: value, run_ids[]
plan: next[{theme}]
idea: idea_id, title, status
machine: event:granted|assigned|released|lost|idle_released|bad|access_lost, node
campaign: campaign_root (absolute), operator_ids[], started_at (RFC 3339), machine{node}
hotops: profile_id, role, ops[{name,share_pct}]
kernel: op_id, stage
incident: incident_id, severity:info|warning|error, category, summary
finding: finding_id, text, evidence_rank:first_hand|second_hand|inference
Unknown fields are rejected; all fields: ledger_schema(kind). Problems come back one per line; fix and retry."""


def kind_schema(kind: str) -> dict:
    """Full field schema of one ledger kind (refs inlined, per-kind rules kept) for ledger_schema."""
    full = schemas.load("runs.v1")
    defs = full["$defs"]
    common = {
        k: v for k, v in full["properties"].items() if k not in SERVER_SET | {"kind", "writer"}
    }
    body = copy.deepcopy(defs[KIND_DEF[kind]])
    rules = [r for r in body.pop("allOf", [])]
    out = {
        "kind": kind,
        "required": body.get("required", []),
        "properties": {**_inline(common, defs), **_inline(body.get("properties", {}), defs)},
    }
    if rules:
        out["conditional_rules"] = rules
    out["note"] = "v, id, ts and writer are set by the server; pass the fields above in `record`."
    return out


TOOLS = [
    types.Tool(
        name="ledger_append",
        description=LEDGER_DOC,
        input_schema={
            "type": "object",
            "required": ["kind", "record"],
            "properties": {"kind": {"enum": list(KIND_DEF)}, "record": {"type": "object"}},
        },
    ),
    types.Tool(
        name="ledger_schema",
        description="Full field list (types, enums, optional fields) of one ledger kind.",
        input_schema={
            "type": "object",
            "required": ["kind"],
            "properties": {"kind": {"enum": list(KIND_DEF)}},
        },
        annotations=types.ToolAnnotations(read_only_hint=True),
    ),
    types.Tool(
        name="ledger_query",
        description="Read ledger records (latest version of each fact, newest last). since: RFC 3339.",
        input_schema={
            "type": "object",
            "properties": {
                "kind": {"enum": list(KIND_DEF)},
                "run_id": {"type": "string"},
                "task_id": {"type": "string"},
                "since": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
        annotations=types.ToolAnnotations(read_only_hint=True),
    ),
    types.Tool(
        name="event_report",
        description=(
            "Log a notable event (service up/down, conclusion, incident, anything unplanned). "
            "event_name: <area>.<what> (recorded as o11y.agent.<area>.<what>); attributes e.g. run_id, "
            "node, evidence[{path,line}]."
        ),
        input_schema={
            "type": "object",
            "required": ["severity", "event_name", "body"],
            "properties": {
                "severity": {"enum": ["INFO", "WARN", "ERROR"]},
                "event_name": {"type": "string"},
                "body": {"type": "string"},
                "attributes": {"type": "object"},
            },
        },
    ),
    types.Tool(
        name="session_register",
        description=(
            "Register this session (main), a child you are about to launch (child: returns the env and a "
            "tmux launch command to use), a running session as your child (child + session_id), or "
            "declare a session not part of the fleet (declined + session_id)."
        ),
        input_schema={
            "type": "object",
            "required": ["name", "role"],
            "properties": {
                "name": {"type": "string"},
                "role": {"enum": ["main", "child", "declined"]},
                "session_id": {"type": "string"},
                "tmux_target": {"type": "string"},
                "purpose": {"type": "string"},
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


def _result(obj: dict) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(obj, ensure_ascii=False))],
        is_error=not obj.get("ok", True),
    )


def _input_problems(name: str, args: dict) -> list[str]:
    import jsonschema

    if name == "ledger_append":
        return []  # the record is validated against the full runs.v1 schema; the flat form is tolerated
    v = jsonschema.Draft202012Validator(TOOL_SCHEMAS[name])
    return [f"/{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in v.iter_errors(args)]


def agent_event_name(name: str) -> str:
    """Agent-reported events live under o11y.agent.* (design §8.4); detector and model-pass names stay theirs."""
    rest = name.removeprefix("o11y.").removeprefix("agent.")
    return f"o11y.agent.{rest}"


def call(ctx: api.Context, name: str, args: dict) -> dict:
    if name not in TOOL_SCHEMAS:
        return {"ok": False, "problems": [f"unknown tool {name}"]}
    probs = _input_problems(name, args)
    if probs:
        return {"ok": False, "problems": probs, "hint": f"fix the arguments and call {name} again"}
    if name == "ledger_append":
        record = args.get("record")
        if not isinstance(record, dict):  # tolerate the flat form {kind, field: ...}
            record = {k: v for k, v in args.items() if k != "kind"}
        return api.ledger_append(ctx, {**record, "kind": args.get("kind")})
    if name == "ledger_schema":
        return {"ok": True, **kind_schema(args["kind"])}
    if name == "ledger_query":
        return api.ledger_query(
            ctx,
            args.get("kind"),
            args.get("run_id"),
            args.get("task_id"),
            args.get("since"),
            args.get("limit", 20),
        )
    if name == "event_report":
        return api.event_report(
            ctx,
            args["severity"],
            agent_event_name(args["event_name"]),
            args["body"],
            args.get("attributes"),
        )
    if name == "session_register":
        return api.session_register(
            ctx,
            args["name"],
            args["role"],
            args.get("tmux_target"),
            args.get("purpose"),
            args.get("cwd"),
            args.get("session_id"),
        )
    return api.fleet_status(ctx)


# Seconds between membership checks; the env override exists for tests only.
MEMBERSHIP_POLL_S = float(os.environ.get("AGENT_SYS_O11Y_MEMBERSHIP_POLL_S") or 30)


def is_member(root: Path | None = None) -> bool:
    """Fleet member (spec §10.4): launched as main or child, or activated on skill use — the trigger mod then
    writes `state/active/<sid>.json` for this (parent) session under the fleet root."""
    if os.environ.get(fleet.ACTIVATION_ENV_VAR) in ("true", "child"):
        return True
    ctx = api.Context(root=root, role="other")
    return (ctx.root / "state" / "active" / f"{ctx.session_id}.json").is_file()


def run(root: Path | None = None) -> None:
    state = {"session": None, "member": is_member(root)}
    bus = InMemorySubscriptionBus()

    async def on_list_tools(ctx, _params):
        state["session"] = ctx.session  # connection-scoped outbound, used for list_changed later
        state["member"] = state["member"] or is_member(root)
        return types.ListToolsResult(tools=TOOLS if state["member"] else [])

    async def on_call_tool(_ctx, params):
        if not (state["member"] or is_member(root)):
            return _result(
                {"ok": False, "problems": ["this session is not part of an agent_sys o11y fleet"]}
            )
        ctx = api.Context(root=root)  # re-detect per call: follows /clear and env changes
        try:
            return _result(call(ctx, params.name, dict(params.arguments or {})))
        except Exception as exc:  # never kill the server on one bad call
            return _result(
                {"ok": False, "problems": [f"internal error: {exc.__class__.__name__}: {exc}"]}
            )

    async def watch_membership():
        """Once membership appears (skill activation mid-session), tell the client to re-list the tools."""
        while not state["member"]:
            await anyio.sleep(MEMBERSHIP_POLL_S)
            if is_member(root):
                state["member"] = True
                # 2026-07-28 wire: clients learn of changes only through a subscriptions/listen stream;
                # 2025 wire: an unsolicited notification. Send both; either may have no listener.
                await bus.publish(ToolsListChanged())
                if state["session"] is not None:
                    try:
                        await state["session"].send_tool_list_changed()
                    except Exception:  # noqa: BLE001 - delivery is best effort; the next list sees the tools
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


if __name__ == "__main__":  # py.sh -m o11y_core.mcp_server  (fleet root per design §4)
    run()
