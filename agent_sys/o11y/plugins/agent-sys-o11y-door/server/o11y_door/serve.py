"""Entry of the door's MCP server (plugin.json: `bin/py -m o11y_door.serve`).

In a non-interactive session (`claude -p`, CLAUDE_CODE_ENTRYPOINT != "cli") that is not a fleet member at start,
answer as an MCP server with no tools, without importing the MCP SDK: `-p` waits for every pending server before
the first turn, and the SDK import cost +2.1 s per `-p` run on this host (measured, median of 5). Exiting instead
is not an option: Claude Code then records the server in <config dir>/mcp-needs-auth-cache.json and skips it in
the sessions that start after (a member `-p` run saw no door tools until the entry was removed; measured).
`-p` fleets are out of scope (spec §1.2), so such a session cannot gain membership later. Interactive sessions
keep the full server (activation can come later). Membership at start: AGENT_SYS_O11Y in {true, child}, or the
trigger's pin for this session exists.
"""

import os
import sys


def _member_at_start() -> bool:
    if os.environ.get("AGENT_SYS_O11Y") in ("true", "child"):
        return True
    sid = os.environ.get("CLAUDE_CODE_SESSION_ID")
    prefix = os.environ.get("AGENT_SYS_HOME") or os.path.join(
        os.path.expanduser("~"), ".infera_agent_sys"
    )
    return bool(sid) and os.path.exists(
        os.path.join(prefix, "state", "o11y", "sessions", f"{sid}.json")
    )


def serve_empty(stdin=sys.stdin, stdout=sys.stdout) -> None:
    """Newline-delimited JSON-RPC on stdio: initialize, ping and tools/list (empty); notifications are ignored."""
    import json

    for line in stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if not isinstance(msg, dict) or "id" not in msg:
            continue
        method, params = msg.get("method"), msg.get("params") or {}
        reply = {"jsonrpc": "2.0", "id": msg["id"]}
        if method == "initialize":
            reply["result"] = {
                "protocolVersion": params.get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "o11y", "version": "0"},
            }
        elif method == "tools/list":
            reply["result"] = {"tools": []}
        elif method == "ping":
            reply["result"] = {}
        else:
            reply["error"] = {"code": -32601, "message": f"method not found: {method}"}
        stdout.write(json.dumps(reply) + "\n")
        stdout.flush()


if __name__ == "__main__":
    if os.environ.get("CLAUDE_CODE_ENTRYPOINT", "cli") != "cli" and not _member_at_start():
        serve_empty()
        sys.exit(0)
    from o11y_door.mcp_server import run

    run()
