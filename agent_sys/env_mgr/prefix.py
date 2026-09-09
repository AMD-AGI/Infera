# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`~/.infera_agent_sys` — agent_sys's own `~/.local`.

**Why a prefix at all.** Things `agent_sys` installs have to live somewhere, and
the two obvious somewheres are both wrong: `/usr/local/bin` is host state we
promised not to touch, and `~/.local/bin` is the user's. A prefix we own is the
only place where "install" and "uninstall" are both a directory operation.

**`resolve` takes its environment as an argument** and never reads `os.environ`.
A component deciding *where the user's Claude transcripts go* must be testable
without a process-global, and the same discipline keeps it out of the ambient
environment at runtime. `o11y` is the first consumer, not the owner.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = ["CLAUDE_CONFIG_ENV_VAR", "Prefix", "agent_environment"]

#: The directory name. Fixed, and deliberately dotted: it is machine state, not
#: something a user browses.
DIRNAME = ".infera_agent_sys"

HOME_ENV_VAR = "AGENT_SYS_HOME"
BIN_ENV_VAR = "AGENT_SYS_BIN"
SHARE_ENV_VAR = "AGENT_SYS_SHARE"
STATE_ENV_VAR = "AGENT_SYS_STATE"
RUN_ENV_VAR = "AGENT_SYS_RUN"
CLAUDE_HOME_ENV_VAR = "AGENT_SYS_CLAUDE_HOME"

#: AgentsView's own two names. Not ours to rename — they are the published
#: interface of an external dependency whose code we do not modify.
AGENTSVIEW_DATA_ENV_VAR = "AGENTSVIEW_DATA_DIR"
CLAUDE_PROJECTS_ENV_VAR = "CLAUDE_PROJECTS_DIR"

#: Claude Code's own name for "where my config, credentials and transcripts
#: live". Also not ours to rename.
CLAUDE_CONFIG_ENV_VAR = "CLAUDE_CONFIG_DIR"


def _home(environ: Mapping[str, str]) -> Path:
    """`$HOME`, then the passwd entry, then a per-uid directory in `$TMPDIR`.

    The last step is real, not a formality: `resolve` has to be total, and the
    cwd would put machine state inside whichever repository is checked out. An
    archive that does not survive a reboot is the right degradation here.
    """
    home = environ.get("HOME")
    if home:
        return Path(home).expanduser()
    try:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:  # noqa: BLE001 - see the module docstring; this cannot fail
        return Path(tempfile.gettempdir()) / f"infera-agent-sys-{os.getuid()}"


@dataclass(frozen=True)
class Prefix:
    """One resolved prefix. Every path is derived; none is stored twice."""

    root: Path

    @classmethod
    def resolve(cls, environ: Mapping[str, str]) -> Prefix:
        """**Total: it always answers, and never raises.**

        It was `environ["HOME"]` — a `KeyError` under `env -i`, caught at one of
        three call sites and fatal at the other two. `expanduser`/`resolve`
        because `~/foo` is a literal to `Path` and a relative override moves
        with a cwd that changes between zones.
        """
        override = environ.get(HOME_ENV_VAR)
        if override:
            return cls(Path(override).expanduser().resolve())
        return cls(_home(environ) / DIRNAME)

    @property
    def bin(self) -> Path:
        return self.root / "bin"

    @property
    def share(self) -> Path:
        return self.root / "share"

    @property
    def state(self) -> Path:
        return self.root / "state"

    @property
    def run(self) -> Path:
        return self.root / "run"

    @property
    def claude_home(self) -> Path:
        """`CLAUDE_CONFIG_DIR` for agent children.

        **Not under a run root**: a reader of these transcripts outlives any
        one run, so `runs/<id>/` would name a directory that stops existing.
        """
        return self.state / "claude"

    @property
    def agentsview_data(self) -> Path:
        return self.state / "agentsview"

    def environment(self) -> dict[str, str]:
        """Every directory, by name, ready to merge into a child's `env`.

        Returned rather than exported: the caller decides whose environment
        this joins, and the answer is never this process's.
        """
        return {
            HOME_ENV_VAR: str(self.root),
            BIN_ENV_VAR: str(self.bin),
            SHARE_ENV_VAR: str(self.share),
            STATE_ENV_VAR: str(self.state),
            RUN_ENV_VAR: str(self.run),
            CLAUDE_HOME_ENV_VAR: str(self.claude_home),
            AGENTSVIEW_DATA_ENV_VAR: str(self.agentsview_data),
            CLAUDE_PROJECTS_ENV_VAR: str(self.claude_home / "projects"),
        }

    def create(self) -> None:
        """Idempotent. Creates only inside `root` — never a parent."""
        for d in (
            self.bin,
            self.share,
            self.state,
            self.run,
            self.claude_home / "projects",
            self.agentsview_data,
        ):
            d.mkdir(parents=True, exist_ok=True)


def agent_environment(
    prefix: Prefix, base: Mapping[str, str], *, bin_on_path: bool = True
) -> dict[str, str]:
    """`base`, plus the prefix, plus the one variable that scopes the panel.

    **`CLAUDE_CONFIG_DIR` goes in the returned dict, never into `os.environ`** —
    the whole promise to the user, guarded by
    `test_agent_environment_does_not_touch_this_process`.
    **`bin_on_path=False` under a policy:** `executable_path` derives `PATH`
    from the granted set, which excludes `$HOME`. `AGENT_SYS_BIN` still names it.
    """
    env = dict(base)
    env.update(prefix.environment())
    env[CLAUDE_CONFIG_ENV_VAR] = str(prefix.claude_home)
    if bin_on_path:
        env["PATH"] = ":".join([str(prefix.bin), base.get("PATH", "")]).rstrip(":")
    return env
