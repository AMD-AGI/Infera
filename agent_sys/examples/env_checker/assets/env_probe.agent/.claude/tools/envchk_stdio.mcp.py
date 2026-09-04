#!/usr/bin/env python3
"""`envchk_stdio` — a **bundled** stdio MCP server. L3.

The difference between this file and
`agent_sys/env_mgr/addons/envchk-baseline/.claude/servers/envchk_baseline_server.py`
is not what it does — both speak the same four JSON-RPC methods and both return
a nonce-derived token — it is **how it is declared**:

| | declaration | level |
|---|---|---|
| `envchk-baseline` | a `command`/`args` entry in `.claude/.mcp.json` | L2, a component this repo ships |
| this file | its own **location and suffix**, `.claude/tools/*.mcp.py` | L3, carried by one task package for one agent |

Nothing names this file. `env_mgr` finds it because of where it is and what it
is called, registers it as `envchk_stdio` (the stem, less `.mcp`), and the model
calls `mcp__envchk_stdio__envchk_report`. That auto-registration is the seventh
capability this package exists to prove, and proving it needs a server that is
declared *nowhere* — so the near-duplication with the component is deliberate
and must stay: a shared import would give this file a declaration.

Standard library only, for the reason the component states — a server that
cannot start reports as a server with **no tools**, not as an error.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys

#: ENVCHK_SALT: a65f7dfdb2fffd8e47bd0dd3c76a3548
SALT = "a65f7dfdb2fffd8e47bd0dd3c76a3548"
LABEL = "mcp_stdio"
LEVEL = "L3"

SERVER_NAME = "envchk_stdio"
TOOL_NAME = "envchk_report"
DEFAULT_PROTOCOL = "2025-06-18"


def token(nonce: str) -> str:
    """The one derivation every capability in this package shares.

    `sha256(f"{salt}:{label}:{nonce}")[:12]`, and the salt above is the only
    copy of itself in the repository.
    """
    digest = hashlib.sha256(f"{SALT}:{LABEL}:{nonce}".encode()).hexdigest()[:12]
    return f"ENVCHK-{LABEL.upper()}-{digest}"


def report() -> dict[str, str | int]:
    """Token plus liveness. `pid` and `at` are what separate a live call from a
    transcribed one; the token alone cannot."""
    return {
        "token": token(os.environ.get("ENVCHK_NONCE", "")),
        "label": LABEL,
        "level": LEVEL,
        "pid": os.getpid(),
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


TOOL_DESCRIPTION = (
    "Return the bundled stdio MCP server's capability token, with the pid and "
    "timestamp of the process that produced it. Takes no arguments."
)

TOOL_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


def handle(request: dict) -> dict | None:
    """One request to one response; `None` for a notification, which must draw
    no response at all."""
    method = request.get("method")
    request_id = request.get("id")

    if method == "initialize":
        protocol = request.get("params", {}).get("protocolVersion") or DEFAULT_PROTOCOL
        return _ok(
            request_id,
            {
                "protocolVersion": protocol,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": "1.0.0"},
            },
        )
    if method is not None and method.startswith("notifications/"):
        return None
    if method == "tools/list":
        return _ok(
            request_id,
            {
                "tools": [
                    {
                        "name": TOOL_NAME,
                        "description": TOOL_DESCRIPTION,
                        "inputSchema": TOOL_SCHEMA,
                    }
                ]
            },
        )
    if method == "tools/call":
        called = request.get("params", {}).get("name")
        if called != TOOL_NAME:
            return _err(request_id, -32602, f"unknown tool {called!r}; have {TOOL_NAME!r}")
        return _ok(request_id, {"content": [{"type": "text", "text": json.dumps(report(), indent=2)}]})
    if request_id is None:
        return None
    return _err(request_id, -32601, f"unknown method {method!r}")


def _ok(request_id: object, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _err(request_id: object, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def main() -> int:
    """Line-delimited JSON both ways. **Nothing but protocol on stdout** — a
    stray `print` corrupts the client's frame and the server vanishes instead of
    failing."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"{SERVER_NAME}: undecodable frame: {exc}", file=sys.stderr)
            continue
        response = handle(request)
        if response is None:
            continue
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
