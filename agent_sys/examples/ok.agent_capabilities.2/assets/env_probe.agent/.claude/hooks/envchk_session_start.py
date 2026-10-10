#!/usr/bin/env python3
"""The `SessionStart` hook. Installed by the copy route.

Declared by `../settings.json`, the only file Claude Code reads hooks from.
Writes a token plus the hook payload (`session_id`, `transcript_path`, `cwd`,
`hook_event_name`) copied from stdin to `$AGENT_SYS_MY_LOGS/envchk-hook.json`,
falling back to `$CLAUDE_CONFIG_DIR`. The payload proves the harness invoked
this file at SessionStart; the token alone only proves the salt was
reachable. Prints only the output path, never the token, since a
`SessionStart` hook's stdout is added to the session's context.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

#: ENVCHK_SALT: 6ea6f6c74db34d32fc6deeb468877508
SALT = "6ea6f6c74db34d32fc6deeb468877508"
LABEL = "hook"
INSTALLED_BY = "copied"

#: The file the brief tells the agent to read. Named for the capability rather
#: than for the event, because a second hook on a second event would be a second
#: file and not a second writer of this one.
OUT_NAME = "envchk-hook.json"

#: The fields copied out of Claude Code's hook input. Copied by name rather than
#: wholesale: the payload is documented to grow, and a hook that echoes whatever
#: it was handed into a file that a validator reads is a channel nobody designed.
PAYLOAD_KEYS = ("session_id", "transcript_path", "cwd", "hook_event_name")


def token(nonce: str) -> str:
    """`sha256(f"{salt}:{label}:{nonce}")[:12]` — the derivation shared by all
    six capabilities in this package."""
    digest = hashlib.sha256(f"{SALT}:{LABEL}:{nonce}".encode()).hexdigest()[:12]
    return f"ENVCHK-{LABEL.upper()}-{digest}"


def read_payload() -> dict[str, object]:
    """Claude Code's hook input, or `{}`.

    Never raises, so an unexpected stdin does not prevent the output file
    from being written.
    """
    try:
        raw = sys.stdin.read()
    except OSError:
        return {}
    if not raw.strip():
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {key: loaded[key] for key in PAYLOAD_KEYS if key in loaded}


def out_path() -> Path:
    root = os.environ.get("AGENT_SYS_MY_LOGS") or os.environ.get("CLAUDE_CONFIG_DIR") or "."
    return Path(root) / OUT_NAME


def main() -> int:
    record = {
        "token": token(os.environ.get("ENVCHK_NONCE", "")),
        "label": LABEL,
        "installed_by": INSTALLED_BY,
        "pid": os.getpid(),
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "payload": read_payload(),
    }
    target = out_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"envchk hook: cannot write {target}: {exc}", file=sys.stderr)
        return 1
    print(f"envchk SessionStart hook wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
