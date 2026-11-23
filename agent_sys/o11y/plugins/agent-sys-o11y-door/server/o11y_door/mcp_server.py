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
from .tools import (  # noqa: F401  (LEDGER_DOC, TOOL_SCHEMAS: tests)
    LEDGER_DOC,
    SPECS,
    TOOL_SCHEMAS,
    call,
)

# Seconds between membership checks; the env override exists for tests only.
MEMBERSHIP_POLL_S = float(os.environ.get("AGENT_SYS_O11Y_MEMBERSHIP_POLL_S") or 30)

TOOLS = [
    types.Tool(
        name=t["name"],
        description=t["description"],
        input_schema=t["input_schema"],
        annotations=types.ToolAnnotations(read_only_hint=True) if t.get("read_only") else None,
    )
    for t in SPECS
]


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
