"""The writer's session id: the `sessionId` in Claude Code's per-process record `<config>/sessions/<ppid>.json`.

Why not an environment variable (measured with 2.1.325, interactive tmux session, plugin MCP server):
- `CLAUDE_CODE_SESSION_ID` in an MCP server's environment is the id it was spawned with; after `/clear` the
  same server process still had the old id while `sessions/<ppid>.json` already held the new one.
- `CLAUDE_PID` is not set for MCP servers; the value seen there was inherited from an enclosing session.
The door runs as a direct child of claude (py.sh execs python), so its ppid is the claude pid; every launch
wrapper must `exec` to keep that true (tested against `sessions/<pid>.json`).
This module reads one file and makes no liveness judgement: the trigger's registry owns that (review A4).
"""

from __future__ import annotations

import json
from pathlib import Path


def session_of_pid(pid: int, config_dir: Path) -> str | None:
    """`sessionId` of the claude process `pid`, or None when its record is absent, empty or torn."""
    try:
        doc = json.loads((config_dir / "sessions" / f"{pid}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    sid = doc.get("sessionId") if isinstance(doc, dict) else None
    return sid if isinstance(sid, str) and sid else None
