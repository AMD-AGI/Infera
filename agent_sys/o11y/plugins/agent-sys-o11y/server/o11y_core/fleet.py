"""Where a fleet's files live (design §4).

Duplicated facts, drift-tested (engineer_principle §5.1):
- the prefix rule of `env_mgr/prefix.py` (`AGENT_SYS_HOME`, else `$HOME/.infera_agent_sys`);
- Claude Code's project-directory slug (every non-alphanumeric character becomes `-`), shared with the
  TS layer through `protocol/test_vectors/slug.json`.
"""

from __future__ import annotations

import hashlib
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
    """The directory claude was started in: `CLAUDE_PROJECT_DIR` (cwd can change), else the cwd."""
    environ = os.environ if environ is None else environ
    return environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()


def fleet_dir(environ: Mapping[str, str] | None = None, launch: str | None = None) -> Path:
    """`AGENT_SYS_O11Y_FLEET_DIR`, else `<prefix>/state/o11y/fleets/<slug(launch dir)>`."""
    environ = os.environ if environ is None else environ
    explicit = environ.get(FLEET_DIR_ENV_VAR)
    if explicit:
        return Path(explicit).expanduser()
    return fleets_root(environ) / slug(launch or launch_dir(environ))


def claude_config_dir(environ: Mapping[str, str] | None = None) -> Path:
    """The session's Claude configuration directory: `$CLAUDE_CONFIG_DIR`, else `~/.claude`."""
    environ = os.environ if environ is None else environ
    return Path(os.path.expanduser(environ.get("CLAUDE_CONFIG_DIR") or "~/.claude"))
