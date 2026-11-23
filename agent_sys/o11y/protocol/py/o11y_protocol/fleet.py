"""Where a fleet's files live (design §4).

Duplicated facts, drift-tested (engineer_principle §5.1):
- the prefix rule of `env_mgr/prefix.py` (`AGENT_SYS_HOME`, else `$HOME/.infera_agent_sys`);
- Claude Code's project-directory slug (every non-alphanumeric character becomes `-`), shared with the
  TS layer through `protocol/test_vectors/slug.json`.

Fleet root resolution, in order (design §4, review E6): the session's pin
`<prefix>/state/o11y/sessions/<sid>.json` (written once by the trigger at activation, so the root does not
follow `/cd` or worktree moves), else `AGENT_SYS_O11Y_FLEET_DIR`, else `<prefix>/state/o11y/fleets/<slug>` of
the launch directory (`CLAUDE_PROJECT_DIR`, else the cwd).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

DIRNAME = ".infera_agent_sys"
HOME_ENV_VAR = "AGENT_SYS_HOME"
FLEET_DIR_ENV_VAR = "AGENT_SYS_O11Y_FLEET_DIR"
PARENT_ENV_VAR = "AGENT_SYS_O11Y_PARENT_SESSION_ID"
ACTIVATION_ENV_VAR = "AGENT_SYS_O11Y"


def _home(environ: Mapping[str, str]) -> Path:
    home = environ.get("HOME")
    if home:
        return Path(home).expanduser()
    try:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:  # noqa: BLE001 - total, like env_mgr's resolve
        return Path(tempfile.gettempdir()) / f"infera-agent-sys-{os.getuid()}"


def prefix(environ: Mapping[str, str] | None = None) -> Path:
    """env_mgr's prefix root: `$AGENT_SYS_HOME` (expanded, resolved), else `$HOME/.infera_agent_sys`."""
    environ = os.environ if environ is None else environ
    override = environ.get(HOME_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return _home(environ) / DIRNAME


SLUG_MAX = 200


def slug(directory: str) -> str:
    """Claude Code's project-directory name: every non-alphanumeric character becomes `-`. A name longer than
    200 characters keeps its first 200 and appends `-` + 12 hex of the directory's SHA-256 (a path component
    must stay under the file-system limit; Claude Code also truncates at 200 and appends a hash)."""
    name = re.sub(r"[^A-Za-z0-9]", "-", directory)
    if len(name) <= SLUG_MAX:
        return name
    return f"{name[:SLUG_MAX]}-{hashlib.sha256(directory.encode()).hexdigest()[:12]}"


def fleets_root(environ: Mapping[str, str] | None = None) -> Path:
    return prefix(environ) / "state" / "o11y" / "fleets"


def launch_dir(environ: Mapping[str, str] | None = None) -> str:
    """`CLAUDE_PROJECT_DIR` (the session's project dir; it can follow `/cd` and worktree moves, which is why
    the pin comes first), else the cwd."""
    environ = os.environ if environ is None else environ
    return environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()


def pin_path(session_id: str, environ: Mapping[str, str] | None = None) -> Path:
    return prefix(environ) / "state" / "o11y" / "sessions" / f"{session_id}.json"


def pin(session_id: str | None, environ: Mapping[str, str] | None = None) -> dict | None:
    """The session's pin document (session_pin.v1), or None. Stdlib only: the backstop reads it."""
    if not session_id:
        return None
    try:
        doc = json.loads(pin_path(session_id, environ).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def pinned_fleet_dir(
    session_id: str | None, environ: Mapping[str, str] | None = None
) -> Path | None:
    """The fleet root recorded in the session's pin, or None."""
    root = (pin(session_id, environ) or {}).get("fleet_dir")
    return Path(root) if isinstance(root, str) and root.startswith("/") else None


def pinned_role(session_id: str | None, environ: Mapping[str, str] | None = None) -> str | None:
    """`main` or `child` as recorded in the session's pin, or None."""
    role = (pin(session_id, environ) or {}).get("role")
    return role if role in ("main", "child") else None


def fleet_dir(
    environ: Mapping[str, str] | None = None,
    launch: str | None = None,
    session_id: str | None = None,
) -> Path:
    """The pin of `session_id`, else a valid `AGENT_SYS_O11Y_FLEET_DIR`, else `<prefix>/state/o11y/fleets/<slug>`."""
    environ = os.environ if environ is None else environ
    pinned = pinned_fleet_dir(session_id, environ)
    if pinned is not None:
        return pinned
    explicit = explicit_fleet_dir(environ)
    if explicit is not None:
        return explicit
    return fleets_root(environ) / slug(launch or launch_dir(environ))


def explicit_fleet_dir(environ: Mapping[str, str]) -> Path | None:
    """`AGENT_SYS_O11Y_FLEET_DIR` when absolute or `~/`-prefixed (expanded with this environ's HOME); a relative
    value is ignored, never resolved against the cwd (protocol/test_vectors/fleet_dir.json)."""
    value = environ.get(FLEET_DIR_ENV_VAR) or ""
    if value.startswith("~/"):
        return _home(environ) / value[2:]
    return Path(value) if value.startswith("/") else None


def claude_config_dir(environ: Mapping[str, str] | None = None) -> Path:
    """The session's Claude configuration directory: `$CLAUDE_CONFIG_DIR`, else `~/.claude`."""
    environ = os.environ if environ is None else environ
    return Path(os.path.expanduser(environ.get("CLAUDE_CONFIG_DIR") or "~/.claude"))
