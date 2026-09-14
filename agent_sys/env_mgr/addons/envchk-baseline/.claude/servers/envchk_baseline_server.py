#!/usr/bin/env python3
"""`envchk_baseline` -- one MCP tool over stdio, standard library only.

Declared by `../.mcp.json` (a config entry naming a command), as opposed to
`.claude/tools/*.mcp.py`, where the file's location is the declaration.

No third-party import, on purpose: a server that cannot start reports as a
server with no tools, not as an error, and the protocol here is four
JSON-RPC methods the standard library covers.

Returns a token derived from this component's salt and the per-run nonce;
`check_capabilities_genuine` re-runs this exact file and compares. See
`../../README.md`.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys

#: This component's salt. It exists nowhere else in this repository, and that is
#: what makes the token evidence rather than a format.
#:
#: ENVCHK_SALT: 48d7f4c12e751bebb631ff42ffe54656
SALT = "48d7f4c12e751bebb631ff42ffe54656"

#: Which of `examples/env_checker`'s six capabilities this one is.
LABEL = "mcp_external"

#: Which of the two install routes delivered it. This file is not copied
#: with a `.claude/` tree: a package recipe copies it into
#: `$CLAUDE_CONFIG_DIR/servers/`. Reported as `installed_by`.
INSTALLED_BY = "recipe"

SERVER_NAME = "envchk_baseline"
TOOL_NAME = "envchk_report"

#: Echoed back to the client when it does not state one. Any client that does
#: state one gets its own value back, which is what the specification asks of a
#: server that supports the requested version.
DEFAULT_PROTOCOL = "2025-06-18"


def token(nonce: str) -> str:
    """`ENVCHK-<LABEL>-<12 hex>`, the one derivation, in one place.

    Unset `ENVCHK_NONCE` reads as `""` rather than raising, so the failure
    surfaces as a validator mismatch rather than a server with no tools.
    """
    digest = hashlib.sha256(f"{SALT}:{LABEL}:{nonce}".encode()).hexdigest()[:12]
    return f"ENVCHK-{LABEL.upper()}-{digest}"


def report() -> dict[str, str | int]:
    """The tool's whole result.

    `pid` and `at` are liveness, kept separate from the token, so a validator
    can tell a live call from a transcribed one.
    """
    return {
        "token": token(os.environ.get("ENVCHK_NONCE", "")),
        "label": LABEL,
        "installed_by": INSTALLED_BY,
        "pid": os.getpid(),
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


TOOL_DESCRIPTION = (
    "Return the envchk-baseline add-on's capability token, with the pid "
    "and timestamp of the process that produced it. Takes no arguments."
)

TOOL_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


def handle(request: dict) -> dict | None:
    """One JSON-RPC request to one response, or `None` for a notification.

    An unknown method gets `-32601` rather than silence, so a client that
    would otherwise retry until timeout gets a name instead.
    """
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

    # Notifications carry no `id` and must draw no response at all; replying to
    # one is a protocol error the client is entitled to close the pipe over.
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
        text = json.dumps(report(), indent=2)
        return _ok(request_id, {"content": [{"type": "text", "text": text}]})

    if request_id is None:
        return None
    return _err(request_id, -32601, f"unknown method {method!r}")


def _ok(request_id: object, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _err(request_id: object, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def main() -> int:
    """Line-delimited JSON on stdin, line-delimited JSON on stdout.

    Nothing but protocol goes to stdout; anything to say goes to stderr,
    since a stray `print` corrupts the client's frame.
    """
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
