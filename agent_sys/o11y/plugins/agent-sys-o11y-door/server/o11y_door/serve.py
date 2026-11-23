"""Entry of the door's MCP server (plugin.json: `bin/py -m o11y_door.serve`).

In a non-interactive session (`claude -p`, CLAUDE_CODE_ENTRYPOINT != "cli") that is not a fleet member at start,
exit before importing the MCP SDK: `-p` waits for every pending server before the first turn, and starting this
one cost +2.1 s per `-p` run on this host (measured, median of 5). `-p` fleets are out of scope (spec §1.2), so
such a session cannot gain membership later. Interactive sessions keep the server (activation can come later).
Membership at start: AGENT_SYS_O11Y in {true, child}, or the trigger's pin for this session exists.
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


if __name__ == "__main__":
    if os.environ.get("CLAUDE_CODE_ENTRYPOINT", "cli") != "cli" and not _member_at_start():
        sys.exit(0)
    from o11y_door.mcp_server import run

    run()
