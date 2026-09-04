#!/usr/bin/env python3
"""The `SessionStart` hook. Installed by the copy route.

Declared by `../settings.json`, which is the **only** file Claude Code reads
hooks from. `agent`'s `hooks:` key places a file at `<zone>/config/hooks/<name>`
and nothing reads it (`env_mgr/material.py` writes no `settings.json`), so a
hook that is *declared* and a hook that *fires* are two different things and
this package exists to tell them apart.

## What it writes, and why the payload matters more than the token

The hook receives Claude Code's hook input as JSON on **stdin** —
`session_id`, `transcript_path`, `cwd`, `hook_event_name` — and copies those
four straight through into its output file beside the token. The token says
*this salt was reachable*. The payload says *the harness invoked this file at
the SessionStart event*, and there is no way to obtain a `session_id` matching
the running session's own transcript by reading a file.

Stated exactly, because it is the strongest claim any of the six capabilities
can make and it is still not a proof of honesty: an agent that ran this script
itself, by hand, would get no stdin and therefore no `session_id` — the fields
would be absent and `check_capabilities_genuine` reports that. An agent that
went looking for its own session id and forged the file is not caught by
anything here, and `check_capabilities_genuine`'s readme says so in its
*What it cannot catch* section rather than leaving it implicit.

## Where it writes

`$AGENT_SYS_MY_LOGS`, the zone's own logs directory
(`env_mgr/paths.py:LOGS_ENV_VAR`) — granted, inside the zone, and it survives
the session, which `$TMPDIR` does not. Falling back to `$CLAUDE_CONFIG_DIR`
rather than to `/tmp`: an unwritable target must not become a file somewhere
nobody looks.

## What it prints

The path, and nothing else. A `SessionStart` hook's stdout is added to the
session's context, so printing the token would hand it to the agent without the
agent having read the file — and reading the file is the half of this that
proves the write succeeded. A hook that cannot write says so on stderr and exits
non-zero; a `SessionStart` hook that fails does not block the session, so the
absence must be visible in the report rather than as a missing session.
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

    Never raises. A hook that dies on an unexpected stdin produces no file at
    all, and the absence of the file is then indistinguishable from the hook
    never having been installed — which is the one distinction this package is
    built to make.
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
