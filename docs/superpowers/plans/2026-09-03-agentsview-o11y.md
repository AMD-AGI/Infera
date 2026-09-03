# AgentsView o11y Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploying `agent_sys` automatically brings up an AgentsView web panel on port 18888 that shows `agent_sys`'s own agent sessions and nothing else, and that can never fail the deployment.

**Architecture:** AgentsView is installed as an unmodified external binary into a dedicated user prefix `~/.infera_agent_sys` owned by `env_mgr`. Agent child processes get `CLAUDE_CONFIG_DIR` pointed into that prefix, so their transcripts land there instead of `~/.claude`; AgentsView is pointed at that one directory and every other session provider is switched off. A small supervisor module starts the daemon once per deployment and degrades to a single log warning on every failure path.

**Tech Stack:** Python 3 (stdlib `argparse`, `socket`, `subprocess`, `pathlib`, `shutil`), pytest, the existing `env_mgr` recipe/installer machinery, AgentsView (Go binary, external).

**Spec:** `docs/superpowers/specs/2026-09-03-agentsview-o11y-design.md`. Read it first.

**Workspace for all throwaway activity:** `/home/yihou/ws.agentsview_o11y/`. The repo receives deliverables only.

---

## File Structure

| Path | Responsibility | New? |
|---|---|---|
| `agent_sys/env_mgr/o11y/__init__.py` | package marker, re-exports the four public functions | create |
| `agent_sys/env_mgr/o11y/prefix.py` | the `~/.infera_agent_sys` layout: where each directory is, what its env var is called, and how to create them | create |
| `agent_sys/env_mgr/o11y/agentsview.py` | the supervisor: `resolve_port`, `port_is_free`, `ensure_running`, `health`. Never raises. | create |
| `agent_sys/env_mgr/recipes/agentsview.o11y.yaml` | the one `bin` recipe item that installs the binary | create |
| `agent_sys/env_mgr/paths.py` | add the `AGENT_SYS_*` prefix env-var name constants next to the existing family | modify |
| `agent_sys/cli/main.py` | `--agentsview-port` / `--no-agentsview`, and the one call to `ensure_running` | modify |
| `agent_sys/tests/env_mgr/test_o11y_prefix.py` | prefix layout + env-var names | create |
| `agent_sys/tests/env_mgr/test_o11y_agentsview.py` | port resolution, warn-and-skip on every failure mode | create |
| `agent_sys/tests/cli/test_agentsview_flags.py` | CLI flags and the `os.environ`-untouched guard | create |

**Why the supervisor is called from `cli/main.py` and not from `EnvManager`:**
`prepare.py:683`'s docstring records that `test_env_manager_exposes_exactly_these`
pins that class's method set, so a third method is a deliberate decision and not
a drive-by. `prepare()` also runs **once per task**, and the daemon must start
once per deployment. The module still *lives* in `env_mgr` — it is env_mgr's
prefix and env_mgr's concern — but the single call site is the CLI's startup.

---

## Phase 0 — Settle the two unknowns before writing dependent code

Both are experiments. Neither writes repo code. Record results in
`/home/yihou/ws.agentsview_o11y/recon/PHASE0.md`.

### Task 0.1: Does the official installer accept a prefix?

**Files:**
- Create: `/home/yihou/ws.agentsview_o11y/recon/PHASE0.md`

- [ ] **Step 1: Read the installer before running it**

```bash
mkdir -p /home/yihou/ws.agentsview_o11y/recon
cd /home/yihou/ws.agentsview_o11y/recon
curl -fsSL https://agentsview.io/install.sh -o install.sh
grep -nE 'INSTALL_DIR|PREFIX|BIN_DIR|HOME|mkdir|mv |cp ' install.sh
```

Expected: the grep names whichever variable decides the destination. Do not
pipe this script to `sh` before reading it.

- [ ] **Step 2: Install into the prefix and confirm nothing landed elsewhere**

```bash
export AGENT_SYS_HOME="$HOME/.infera_agent_sys"
mkdir -p "$AGENT_SYS_HOME/bin"
# substitute the variable Step 1 found, e.g. AGENTSVIEW_INSTALL_DIR
AGENTSVIEW_INSTALL_DIR="$AGENT_SYS_HOME/bin" sh ./install.sh
ls -l "$AGENT_SYS_HOME/bin/agentsview" && "$AGENT_SYS_HOME/bin/agentsview" --version
ls -l "$HOME/.local/bin/agentsview" 2>&1 | tail -1   # must be "No such file"
```

Expected: a versioned binary inside the prefix, and **no** copy in `~/.local/bin`.

- [ ] **Step 3: If the script has no prefix knob, use the release tarball instead**

```bash
curl -fsSL https://api.github.com/repos/kenn-io/agentsview/releases/latest \
  | grep -oE '"browser_download_url": "[^"]*linux[^"]*amd64[^"]*"' | cut -d'"' -f4
# then: curl -fsSL <url> -o av.tgz && tar -xzf av.tgz -C "$AGENT_SYS_HOME/bin" --strip-components=1 agentsview
```

- [ ] **Step 4: Write the verdict**

Record in `PHASE0.md`, in this exact shape, the literal one-line command that
Task 3.1 will paste into the recipe's `install:` field:

```
## 0.1 install command
verdict: <install.sh honours PREFIX | install.sh has no prefix knob>
install: <the exact one-line command>
check_cmd: agentsview --version
measured version: <output of --version>
```

### Task 0.2: Does `claude-agent-sdk` honour `CLAUDE_CONFIG_DIR`?

**Files:**
- Modify: `/home/yihou/ws.agentsview_o11y/recon/PHASE0.md`

- [ ] **Step 1: Snapshot the user's real Claude directory**

```bash
find "$HOME/.claude/projects" -name '*.jsonl' | sort > /home/yihou/ws.agentsview_o11y/recon/before.txt
wc -l < /home/yihou/ws.agentsview_o11y/recon/before.txt
```

- [ ] **Step 2: Run one trivial agent turn with the variable redirected**

```bash
export AGENT_SYS_CLAUDE_HOME="$HOME/.infera_agent_sys/state/claude"
mkdir -p "$AGENT_SYS_CLAUDE_HOME"
cd /home/yihou/ws.agentsview_o11y/scratch
env CLAUDE_CONFIG_DIR="$AGENT_SYS_CLAUDE_HOME" claude -p 'reply with the single word ok' 2>&1 | tail -5
```

- [ ] **Step 3: Read the artefact, not the exit code**

```bash
find "$AGENT_SYS_CLAUDE_HOME/projects" -name '*.jsonl' | head
find "$HOME/.claude/projects" -name '*.jsonl' | sort > /home/yihou/ws.agentsview_o11y/recon/after.txt
diff /home/yihou/ws.agentsview_o11y/recon/before.txt /home/yihou/ws.agentsview_o11y/recon/after.txt && echo "USER DIR UNTOUCHED"
```

Expected, and all three must hold: at least one JSONL under the prefix; `diff`
reports no change; the turn actually answered (credentials resolved).

- [ ] **Step 4: If credentials did NOT resolve, find what the child needs**

```bash
ls -a "$HOME/.claude" | head -20
```

Symlink each credential/settings file (never copy — no second token on disk)
into `$AGENT_SYS_CLAUDE_HOME` and re-run Step 2:

```bash
for f in .credentials.json settings.json; do
  [ -e "$HOME/.claude/$f" ] && ln -sfn "$HOME/.claude/$f" "$AGENT_SYS_CLAUDE_HOME/$f"
done
```

- [ ] **Step 5: Write the verdict**

```
## 0.2 CLAUDE_CONFIG_DIR
transcripts redirected: <yes|no>
user ~/.claude/projects unchanged: <yes|no>
credentials resolved: <yes, natively | yes, after symlinking: <list> | no>
```

`no` on the first line stops the plan — bring it to the user rather than
inventing a workaround.

- [ ] **Step 6: Commit the recon note into the workspace only (not the repo)**

No repo commit for Phase 0.

---

## Phase 1 — The prefix and its environment variables

### Task 1.1: Prefix layout module

**Files:**
- Create: `agent_sys/env_mgr/o11y/__init__.py`
- Create: `agent_sys/env_mgr/o11y/prefix.py`
- Test: `agent_sys/tests/env_mgr/test_o11y_prefix.py`

- [ ] **Step 1: Write the failing test**

```python
# agent_sys/tests/env_mgr/test_o11y_prefix.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The `~/.infera_agent_sys` prefix: where it is, and what names it publishes."""

from __future__ import annotations

from pathlib import Path

from env_mgr.o11y.prefix import Prefix


def test_default_root_is_infera_agent_sys_under_home(tmp_path: Path) -> None:
    p = Prefix.resolve({"HOME": str(tmp_path)})
    assert p.root == tmp_path / ".infera_agent_sys"


def test_env_var_overrides_home(tmp_path: Path) -> None:
    p = Prefix.resolve({"HOME": "/nowhere", "AGENT_SYS_HOME": str(tmp_path / "elsewhere")})
    assert p.root == tmp_path / "elsewhere"


def test_the_layout_is_local_shaped(tmp_path: Path) -> None:
    p = Prefix.resolve({"HOME": str(tmp_path)})
    assert p.bin == p.root / "bin"
    assert p.share == p.root / "share"
    assert p.state == p.root / "state"
    assert p.run == p.root / "run"
    assert p.claude_home == p.state / "claude"
    assert p.agentsview_data == p.state / "agentsview"


def test_environment_names_every_directory(tmp_path: Path) -> None:
    p = Prefix.resolve({"HOME": str(tmp_path)})
    env = p.environment()
    assert env["AGENT_SYS_HOME"] == str(p.root)
    assert env["AGENT_SYS_BIN"] == str(p.bin)
    assert env["AGENT_SYS_SHARE"] == str(p.share)
    assert env["AGENT_SYS_STATE"] == str(p.state)
    assert env["AGENT_SYS_RUN"] == str(p.run)
    assert env["AGENT_SYS_CLAUDE_HOME"] == str(p.claude_home)
    assert env["AGENTSVIEW_DATA_DIR"] == str(p.agentsview_data)
    assert env["CLAUDE_PROJECTS_DIR"] == str(p.claude_home / "projects")


def test_create_is_idempotent(tmp_path: Path) -> None:
    p = Prefix.resolve({"HOME": str(tmp_path)})
    p.create()
    p.create()
    for d in (p.bin, p.share, p.state, p.run, p.claude_home, p.agentsview_data):
        assert d.is_dir()


def test_resolve_does_not_read_the_ambient_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_SYS_HOME", "/should/be/ignored")
    p = Prefix.resolve({"HOME": str(tmp_path)})
    assert p.root == tmp_path / ".infera_agent_sys"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cd agent_sys && python -m pytest tests/env_mgr/test_o11y_prefix.py -v`
Expected: `ModuleNotFoundError: No module named 'env_mgr.o11y'`

- [ ] **Step 3: Write the implementation**

```python
# agent_sys/env_mgr/o11y/__init__.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""o11y side-cars: things that watch a run and may never fail one."""

from .prefix import Prefix

__all__ = ["Prefix"]
```

```python
# agent_sys/env_mgr/o11y/prefix.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`~/.infera_agent_sys` — agent_sys's own `~/.local`.

**Why a prefix at all.** The o11y binary has to live somewhere, and the two
obvious somewheres are both wrong: `/usr/local/bin` is host state we promised
not to touch, and `~/.local/bin` is the user's, shared with everything else they
installed. A prefix we own is the only place where "install" and "uninstall"
are both a directory operation.

**`resolve` takes its environment as an argument** and never reads
`os.environ` itself. A component that decides *where the user's Claude
transcripts go* must be testable without a process-global, and the same
discipline is what keeps this out of the ambient environment at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

__all__ = ["Prefix"]

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


@dataclass(frozen=True)
class Prefix:
    """One resolved prefix. Every path is derived; none is stored twice."""

    root: Path

    @classmethod
    def resolve(cls, environ: Mapping[str, str]) -> "Prefix":
        override = environ.get(HOME_ENV_VAR)
        if override:
            return cls(Path(override))
        return cls(Path(environ["HOME"]) / DIRNAME)

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

        **Not under a run root**, deliberately. The daemon outlives any single
        run, so the directory it reads has to be stable; putting this under
        `runs/<id>/` would point the panel at a directory that stops existing.
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
```

- [ ] **Step 4: Run the tests**

Run: `cd agent_sys && python -m pytest tests/env_mgr/test_o11y_prefix.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add agent_sys/env_mgr/o11y/ agent_sys/tests/env_mgr/test_o11y_prefix.py
git commit -s -m "feat(env_mgr): the ~/.infera_agent_sys prefix and its env var names"
```

### Task 1.2: Publish the names next to the existing `AGENT_SYS_*` family

**Files:**
- Modify: `agent_sys/env_mgr/paths.py` (append to `__all__` and to the constant block ending at `LOGS_ENV_VAR`, around line 138)
- Test: `agent_sys/tests/env_mgr/test_o11y_prefix.py`

- [ ] **Step 1: Add the failing test**

Append to `test_o11y_prefix.py`:

```python
def test_prefix_names_are_reachable_from_the_paths_family() -> None:
    """`paths` is where a reader looks for an `AGENT_SYS_*` name. All of them."""
    from env_mgr import paths

    assert paths.HOME_ENV_VAR == "AGENT_SYS_HOME"
    assert paths.BIN_ENV_VAR == "AGENT_SYS_BIN"
    assert paths.CLAUDE_HOME_ENV_VAR == "AGENT_SYS_CLAUDE_HOME"
    for name in ("HOME_ENV_VAR", "BIN_ENV_VAR", "CLAUDE_HOME_ENV_VAR"):
        assert name in paths.__all__
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cd agent_sys && python -m pytest tests/env_mgr/test_o11y_prefix.py::test_prefix_names_are_reachable_from_the_paths_family -v`
Expected: `AttributeError: module 'env_mgr.paths' has no attribute 'HOME_ENV_VAR'`

- [ ] **Step 3: Re-export in `paths.py`**

After the `LOGS_ENV_VAR` block (~line 138) add:

```python
#: **The prefix family, re-exported rather than redefined.** `o11y.prefix` owns
#: these because it owns the layout; they are visible here because `paths` is
#: where a reader looks for an ``AGENT_SYS_*`` name, and a name with two
#: definitions is the drift this module exists to prevent.
from .o11y.prefix import (  # noqa: E402
    BIN_ENV_VAR,
    CLAUDE_HOME_ENV_VAR,
    HOME_ENV_VAR,
    RUN_ENV_VAR,
    SHARE_ENV_VAR,
    STATE_ENV_VAR,
)
```

and add those six names to `__all__`.

- [ ] **Step 4: Run the full env_mgr suite for import cycles**

Run: `cd agent_sys && python -m pytest tests/env_mgr/ -q`
Expected: all pass. A circular-import error here means `o11y.prefix` grew an
`env_mgr` import it must not have — `prefix.py` depends on stdlib only.

- [ ] **Step 5: Commit**

```bash
git add agent_sys/env_mgr/paths.py agent_sys/tests/env_mgr/test_o11y_prefix.py
git commit -s -m "feat(env_mgr): publish the prefix env var names from paths"
```

---

## Phase 2 — The supervisor that can never fail a run

### Task 2.1: Port resolution

**Files:**
- Create: `agent_sys/env_mgr/o11y/agentsview.py`
- Test: `agent_sys/tests/env_mgr/test_o11y_agentsview.py`

- [ ] **Step 1: Write the failing test**

```python
# agent_sys/tests/env_mgr/test_o11y_agentsview.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The AgentsView side-car: which port, and every way it is allowed to fail."""

from __future__ import annotations

import socket

from env_mgr.o11y import agentsview


def test_the_default_port_is_18888() -> None:
    assert agentsview.DEFAULT_PORT == 18888
    assert agentsview.resolve_port(None, {}) == 18888


def test_the_environment_beats_the_default() -> None:
    assert agentsview.resolve_port(None, {"AGENTSVIEW_PORT": "9001"}) == 9001


def test_the_flag_beats_the_environment() -> None:
    assert agentsview.resolve_port(9002, {"AGENTSVIEW_PORT": "9001"}) == 9002


def test_an_unparseable_environment_value_falls_back_to_the_default() -> None:
    assert agentsview.resolve_port(None, {"AGENTSVIEW_PORT": "not-a-port"}) == 18888


def test_port_is_free_says_no_when_something_is_listening() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        taken = s.getsockname()[1]
        assert agentsview.port_is_free(taken) is False


def test_port_is_free_says_yes_when_nothing_is() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    assert agentsview.port_is_free(free) is True
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cd agent_sys && python -m pytest tests/env_mgr/test_o11y_agentsview.py -v`
Expected: `ImportError: cannot import name 'agentsview'`

- [ ] **Step 3: Write the implementation**

```python
# agent_sys/env_mgr/o11y/agentsview.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""AgentsView, started as a side-car and never allowed to fail a run.

**The one rule this module exists to enforce:** an observability panel that can
break the thing it observes is worse than no panel. Every function here returns
a `Status` and raises nothing. There is a test per failure mode holding that
line, because the failure mode of a warning-only component is that someone
later "improves" it into a raise.

**Why we decide the port instead of letting AgentsView decide.** `agentsview
serve` auto-discovers a free port when the requested one is busy. That is a
sensible default for a human at a terminal and the wrong one here: the mission
asks for a *warning and a skip* on a taken port, and a daemon that quietly
moved to 18889 is a panel nobody knows the address of. So the bind probe
happens here, before launch.
"""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass
from typing import Mapping

__all__ = ["DEFAULT_PORT", "Status", "port_is_free", "resolve_port"]

log = logging.getLogger("env_mgr.o11y.agentsview")

#: The mission's number.
DEFAULT_PORT = 18888

PORT_ENV_VAR = "AGENTSVIEW_PORT"


@dataclass(frozen=True)
class Status:
    """What happened. `url` is set only when `running` is true."""

    running: bool
    reason: str
    url: str | None = None


def resolve_port(flag: int | None, environ: Mapping[str, str]) -> int:
    """Flag, then environment, then 18888.

    **An unparseable environment value is the default, not an error.** This is
    a side-car; refusing to start the whole deployment over a typo in a
    variable nobody needed would invert the priority the module is built on.
    """
    if flag is not None:
        return int(flag)
    raw = environ.get(PORT_ENV_VAR)
    if raw is None:
        return DEFAULT_PORT
    try:
        return int(raw)
    except ValueError:
        log.warning(
            "%s=%r is not a port number; using the default %d",
            PORT_ENV_VAR,
            raw,
            DEFAULT_PORT,
        )
        return DEFAULT_PORT


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    """A real bind, not a connect.

    Connecting answers "is someone accepting", which is a different question:
    a socket bound and not listening still makes our own bind fail. We ask the
    question we actually need answered.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True
```

- [ ] **Step 4: Run the tests**

Run: `cd agent_sys && python -m pytest tests/env_mgr/test_o11y_agentsview.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add agent_sys/env_mgr/o11y/agentsview.py agent_sys/tests/env_mgr/test_o11y_agentsview.py
git commit -s -m "feat(env_mgr): agentsview side-car port resolution and bind probe"
```

### Task 2.2: `ensure_running` — warn and skip on every failure path

**Files:**
- Modify: `agent_sys/env_mgr/o11y/agentsview.py`
- Test: `agent_sys/tests/env_mgr/test_o11y_agentsview.py`

- [ ] **Step 1: Write the failing tests**

Append to `test_o11y_agentsview.py`:

```python
import subprocess
from pathlib import Path

import pytest

from env_mgr.o11y.prefix import Prefix


@pytest.fixture()
def prefix(tmp_path: Path) -> Prefix:
    p = Prefix.resolve({"HOME": str(tmp_path)})
    p.create()
    return p


def _fake_binary(prefix: Prefix, body: str) -> None:
    exe = prefix.bin / "agentsview"
    exe.write_text("#!/bin/sh\n" + body)
    exe.chmod(0o755)


def test_a_taken_port_is_one_warning_and_a_skip(prefix, caplog) -> None:
    _fake_binary(prefix, "exit 0\n")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        taken = s.getsockname()[1]
        with caplog.at_level("WARNING"):
            status = agentsview.ensure_running(prefix, port=taken)
    assert status.running is False
    assert "port" in status.reason
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_a_missing_binary_is_a_warning_and_a_skip(prefix, caplog) -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    with caplog.at_level("WARNING"):
        status = agentsview.ensure_running(prefix, port=free)
    assert status.running is False
    assert "not installed" in status.reason
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_a_daemon_that_exits_nonzero_is_a_warning_and_a_skip(prefix, caplog) -> None:
    _fake_binary(prefix, "echo boom >&2\nexit 3\n")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    with caplog.at_level("WARNING"):
        status = agentsview.ensure_running(prefix, port=free)
    assert status.running is False
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_a_launch_that_times_out_is_a_warning_and_a_skip(prefix, caplog, monkeypatch) -> None:
    _fake_binary(prefix, "sleep 30\n")
    monkeypatch.setattr(agentsview, "LAUNCH_TIMEOUT_S", 0.2)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    with caplog.at_level("WARNING"):
        status = agentsview.ensure_running(prefix, port=free)
    assert status.running is False
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_no_failure_mode_raises(prefix) -> None:
    """The whole point of the module, asserted directly."""
    for body in ("exit 3\n", "sleep 30\n"):
        _fake_binary(prefix, body)
        agentsview.ensure_running(prefix, port=1)  # privileged port: bind fails
        agentsview.ensure_running(prefix, port=0)


def test_a_successful_launch_reports_the_url(prefix, monkeypatch) -> None:
    _fake_binary(prefix, "exit 0\n")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    monkeypatch.setattr(agentsview, "_wait_for_health", lambda url, timeout: True)
    status = agentsview.ensure_running(prefix, port=free)
    assert status.running is True
    assert status.url == f"http://127.0.0.1:{free}"


def test_the_child_gets_the_prefix_environment_and_os_environ_is_untouched(
    prefix, monkeypatch
) -> None:
    """`AGENTSVIEW_DATA_DIR` reaches the child; this process never learns it."""
    seen: dict[str, str] = {}

    def spy(cmd, env=None, **kw):  # noqa: ANN001
        seen.update(env or {})
        return subprocess.CompletedProcess(cmd, 0, "", "")

    _fake_binary(prefix, "exit 0\n")
    monkeypatch.setattr(subprocess, "run", spy)
    monkeypatch.setattr(agentsview, "_wait_for_health", lambda url, timeout: True)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    agentsview.ensure_running(prefix, port=free)
    assert seen["AGENTSVIEW_DATA_DIR"] == str(prefix.agentsview_data)
    assert seen["CLAUDE_PROJECTS_DIR"] == str(prefix.claude_home / "projects")
    assert "AGENTSVIEW_DATA_DIR" not in __import__("os").environ
```

- [ ] **Step 2: Run and watch them fail**

Run: `cd agent_sys && python -m pytest tests/env_mgr/test_o11y_agentsview.py -v`
Expected: `AttributeError: module 'env_mgr.o11y.agentsview' has no attribute 'ensure_running'`

- [ ] **Step 3: Implement**

Append to `agentsview.py` (and extend `__all__` with `"ensure_running"`, `"write_config"`):

```python
import json
import subprocess
import time
import urllib.error
import urllib.request

#: How long the launch subprocess itself may take. `serve --background`
#: daemonises and returns immediately, so anything slower is a hung binary.
LAUNCH_TIMEOUT_S = 20.0

#: How long we then wait for the daemon to answer. Cold-start reads the whole
#: session archive, so this is generous.
HEALTH_TIMEOUT_S = 30.0

#: Every provider AgentsView can scan, minus Claude Code. Written into the
#: prefix's own `config.toml` so the panel physically cannot read a directory
#: belonging to some other tool the user happens to have installed.
OTHER_PROVIDERS = (
    "aider", "amp", "antigravity", "antigravity-cli", "claude-cowork",
    "codebuff", "codex", "command-code", "copilot-cli", "cortex-code",
    "cursor", "cursor-ide", "deepseek-tui", "deepseek-harness", "devin",
    "forge", "gemini-cli", "goose", "gptme", "kilo", "kimi-work", "kiro",
    "openclaude", "opencode", "poolside", "positron", "roocode", "trae",
    "vscode-copilot", "windsurf", "zed",
)


def write_config(prefix: "Prefix") -> None:
    """The prefix's `config.toml`. Idempotent, and ours alone.

    Written into `AGENTSVIEW_DATA_DIR`, never `~/.agentsview`: a user who
    already runs AgentsView keeps their own archive and settings untouched.
    """
    cfg = prefix.agentsview_data / "config.toml"
    disabled = ", ".join(json.dumps(name) for name in OTHER_PROVIDERS)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        "# Written by agent_sys. AgentsView itself is unmodified.\n"
        f"disabled_agents = [{disabled}]\n"
        'host = "127.0.0.1"\n'
        "disable_update_check = true\n"
    )


def _wait_for_health(url: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:  # noqa: S310
                if 200 <= r.status < 400:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    return False


def ensure_running(prefix: "Prefix", port: int) -> Status:
    """Start the panel, or say in one line why there is none.

    **Every return is a `Status` and every failure logs exactly one warning.**
    One, not two: a caller that sees the same problem reported twice starts
    hunting for two problems.
    """
    url = f"http://127.0.0.1:{port}"
    exe = prefix.bin / "agentsview"

    if not port_is_free(port):
        if _wait_for_health(url, timeout=2.0):
            return Status(True, "already running", url)
        log.warning(
            "agentsview: port %d is in use by something else; skipping the o11y "
            "panel. Pass --agentsview-port to choose another.",
            port,
        )
        return Status(False, f"port {port} in use")

    if not exe.is_file():
        log.warning(
            "agentsview: not installed at %s; skipping the o11y panel. "
            "Run `env-mgr install env_mgr/recipes/agentsview.o11y.yaml`.",
            exe,
        )
        return Status(False, "not installed")

    try:
        prefix.create()
        write_config(prefix)
        env = {**prefix.environment(), "PATH": str(prefix.bin), "HOME": str(prefix.root)}
        proc = subprocess.run(  # noqa: S603
            [str(exe), "serve", "--background", "--no-browser",
             "--host", "127.0.0.1", "--port", str(port)],
            env=env,
            capture_output=True,
            text=True,
            timeout=LAUNCH_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("agentsview: could not launch (%s); skipping the o11y panel.", e)
        return Status(False, f"launch failed: {e}")

    if proc.returncode != 0:
        log.warning(
            "agentsview: `serve --background` exited %d; skipping the o11y panel. stderr: %s",
            proc.returncode,
            (proc.stderr or "").strip()[:400],
        )
        return Status(False, f"exit {proc.returncode}")

    if not _wait_for_health(url, HEALTH_TIMEOUT_S):
        log.warning(
            "agentsview: started but did not answer %s within %.0fs; skipping the o11y panel.",
            url,
            HEALTH_TIMEOUT_S,
        )
        return Status(False, "health check timed out")

    (prefix.run / "agentsview.port").write_text(str(port))
    return Status(True, "started", url)
```

Add `from .prefix import Prefix` under `TYPE_CHECKING` for the annotation.

- [ ] **Step 4: Run the tests**

Run: `cd agent_sys && python -m pytest tests/env_mgr/test_o11y_agentsview.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add agent_sys/env_mgr/o11y/agentsview.py agent_sys/tests/env_mgr/test_o11y_agentsview.py
git commit -s -m "feat(env_mgr): ensure_running warns and skips on every failure path"
```

---

## Phase 3 — Installing the binary through the existing recipe machinery

### Task 3.1: The recipe item

**Files:**
- Create: `agent_sys/env_mgr/recipes/agentsview.o11y.yaml`
- Test: `agent_sys/tests/env_mgr/test_o11y_agentsview.py`

- [ ] **Step 1: Write the failing test**

Append to `test_o11y_agentsview.py`:

```python
def test_the_recipe_installs_agentsview_as_an_optional_bin_item() -> None:
    """`suggested`, not `required`: install failure must stay a warning."""
    from env_mgr.recipe import load_recipe

    _target, items = load_recipe("env_mgr/recipes/agentsview.o11y.yaml")
    (item,) = [i for i in items if i.spec.get("name") == "agentsview"]
    assert item.installer == "bin"
    assert item.importance == "suggested"
    assert item.spec["check_cmd"] == "agentsview --version"
    assert "o11y" in item.tags
```

- [ ] **Step 2: Run and watch it fail**

Run: `cd agent_sys && python -m pytest tests/env_mgr/test_o11y_agentsview.py -k recipe -v`
Expected: `RecipeError` / file-not-found.

- [ ] **Step 3: Write the recipe, pasting the verdict from Task 0.1 Step 4**

```yaml
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# AgentsView, agent_sys's o11y panel. An external dependency, installed
# unmodified into agent_sys's own prefix.
#
# `importance: suggested` is load-bearing, not decoration: `level_for_missing`
# turns a failed install of a non-required item into a warning, which is
# exactly the "o11y may never fail a deployment" rule, using the mechanism
# env_mgr already has rather than a second one.
version: 1

target:
  kind: prefix
  name: infera_agent_sys
  path: ${AGENT_SYS_HOME}

items:
  - installer: bin
    importance: suggested
    layer: system
    name: agentsview
    check_cmd: "agentsview --version"
    install: "<PASTE THE LINE FROM recon/PHASE0.md § 0.1>"
    tags: [o11y]
```

- [ ] **Step 4: Run the tests, then really install once**

Run: `cd agent_sys && python -m pytest tests/env_mgr/test_o11y_agentsview.py -k recipe -v`
Expected: PASS.

```bash
cd agent_sys
AGENT_SYS_HOME="$HOME/.infera_agent_sys" python -m env_mgr install env_mgr/recipes/agentsview.o11y.yaml
"$HOME/.infera_agent_sys/bin/agentsview" --version
```

Expected: a version string. Re-running the install must report
`already present (skip)`.

- [ ] **Step 5: Commit**

```bash
git add agent_sys/env_mgr/recipes/agentsview.o11y.yaml agent_sys/tests/env_mgr/test_o11y_agentsview.py
git commit -s -m "feat(env_mgr): agentsview o11y recipe, installed into the prefix"
```

---

## Phase 4 — Pointing agent children at the prefix

### Task 4.1: `CLAUDE_CONFIG_DIR` into the child environment, and nowhere else

**Files:**
- Modify: `agent_sys/env_mgr/prepare.py` (the `environment = {"PATH": executable_path(policy)}` line, ~468)
- Test: `agent_sys/tests/env_mgr/test_o11y_prefix.py`

- [ ] **Step 1: Write the failing test**

Append to `test_o11y_prefix.py`:

```python
def test_agent_environment_carries_claude_config_dir(tmp_path: Path) -> None:
    from env_mgr.o11y.prefix import Prefix, agent_environment

    p = Prefix.resolve({"HOME": str(tmp_path)})
    env = agent_environment(p, base={"PATH": "/usr/bin"})
    assert env["CLAUDE_CONFIG_DIR"] == str(p.claude_home)
    assert env["PATH"].startswith(str(p.bin) + ":")
    assert "/usr/bin" in env["PATH"]


def test_agent_environment_does_not_touch_this_process(tmp_path: Path) -> None:
    """The guard on 'the user's own Claude Code is unaffected'."""
    import os

    from env_mgr.o11y.prefix import Prefix, agent_environment

    before = dict(os.environ)
    agent_environment(Prefix.resolve({"HOME": str(tmp_path)}), base={"PATH": "/usr/bin"})
    assert dict(os.environ) == before
    assert "CLAUDE_CONFIG_DIR" not in os.environ
```

- [ ] **Step 2: Run and watch it fail**

Run: `cd agent_sys && python -m pytest tests/env_mgr/test_o11y_prefix.py -k agent_environment -v`
Expected: `ImportError: cannot import name 'agent_environment'`

- [ ] **Step 3: Add `agent_environment` to `prefix.py`**

```python
CLAUDE_CONFIG_ENV_VAR = "CLAUDE_CONFIG_DIR"


def agent_environment(prefix: Prefix, base: Mapping[str, str]) -> dict[str, str]:
    """`base`, plus the prefix, plus the one variable that scopes the panel.

    **`CLAUDE_CONFIG_DIR` goes in the returned dict and never into
    `os.environ`.** That distinction is the whole promise to the user: a Claude
    Code they start in their own terminal inherits nothing from us and keeps
    reading `~/.claude`. `test_agent_environment_does_not_touch_this_process`
    is the guard, and it is not a formality — a single `os.environ[...] = ...`
    added here for convenience would silently redirect the user's own agent.
    """
    env = dict(base)
    env.update(prefix.environment())
    env[CLAUDE_CONFIG_ENV_VAR] = str(prefix.claude_home)
    env["PATH"] = ":".join([str(prefix.bin), base.get("PATH", "")]).rstrip(":")
    return env
```

Add `"agent_environment"` and `"CLAUDE_CONFIG_ENV_VAR"` to `__all__`.

- [ ] **Step 4: Wire it into `prepare.py`**

Replace `environment = {"PATH": executable_path(policy)}` (~line 468) with:

```python
    # `PATH` is derived from the policy, never chosen (see below), and the o11y
    # prefix joins that derivation rather than being appended afterwards: an
    # entry naming a directory the kernel will refuse is the failure mode this
    # line exists to prevent, and the prefix is under $HOME, which is granted.
    #
    # `CLAUDE_CONFIG_DIR` rides along here because this dict is the *child's*
    # environment. Setting it in ours would redirect a Claude Code the user
    # started themselves, which is the one thing this integration promised not
    # to do.
    environment = agent_environment(
        Prefix.resolve(os.environ), base={"PATH": executable_path(policy)}
    )
```

Add at the top of `prepare.py`:

```python
from .o11y.prefix import Prefix, agent_environment
```

(`os` is already imported; confirm with `grep -n '^import os' env_mgr/prepare.py`
and add it if absent.)

- [ ] **Step 5: Run the tests**

Run: `cd agent_sys && python -m pytest tests/env_mgr/ tests/agent/ -q`
Expected: all pass. Any test asserting `environment == {"PATH": ...}` exactly
now needs updating — update the assertion, do not weaken the feature.

- [ ] **Step 6: Commit**

```bash
git add agent_sys/env_mgr/o11y/prefix.py agent_sys/env_mgr/prepare.py agent_sys/tests/env_mgr/test_o11y_prefix.py
git commit -s -m "feat(env_mgr): scope agent transcripts to the prefix via CLAUDE_CONFIG_DIR"
```

### Task 4.2: Credentials reachable from the redirected config dir

**Files:**
- Modify: `agent_sys/env_mgr/o11y/prefix.py`
- Test: `agent_sys/tests/env_mgr/test_o11y_prefix.py`

Implement only what Task 0.2 Step 5 measured as necessary. If it recorded
`credentials resolved: yes, natively`, **skip this task entirely** and note the
skip in the checkpoint file.

- [ ] **Step 1: Write the failing test**

```python
def test_credentials_are_linked_not_copied(tmp_path: Path) -> None:
    """A token must not exist twice on disk."""
    from env_mgr.o11y.prefix import Prefix, link_credentials

    user = tmp_path / ".claude"
    user.mkdir()
    (user / ".credentials.json").write_text('{"token": "secret"}')
    p = Prefix.resolve({"HOME": str(tmp_path)})
    p.create()
    link_credentials(p, user_claude=user)
    linked = p.claude_home / ".credentials.json"
    assert linked.is_symlink()
    assert linked.resolve() == (user / ".credentials.json").resolve()


def test_linking_is_idempotent_and_skips_absent_files(tmp_path: Path) -> None:
    from env_mgr.o11y.prefix import Prefix, link_credentials

    user = tmp_path / ".claude"
    user.mkdir()
    p = Prefix.resolve({"HOME": str(tmp_path)})
    p.create()
    link_credentials(p, user_claude=user)
    link_credentials(p, user_claude=user)
    assert not (p.claude_home / ".credentials.json").exists()
```

- [ ] **Step 2: Run and watch it fail**

Run: `cd agent_sys && python -m pytest tests/env_mgr/test_o11y_prefix.py -k credentials -v`
Expected: `ImportError: cannot import name 'link_credentials'`

- [ ] **Step 3: Implement**

```python
#: Exactly the files Task 0.2 measured as needed. Not a wildcard: a glob over
#: `~/.claude` would sweep the user's history and projects into our prefix.
LINKED_FROM_USER = (".credentials.json", "settings.json")


def link_credentials(prefix: Prefix, user_claude: Path) -> None:
    """Symlink, never copy — a second copy of a token is a second thing to leak.

    Absent files are skipped rather than created. Idempotent.
    """
    for name in LINKED_FROM_USER:
        src = user_claude / name
        if not src.exists():
            continue
        dst = prefix.claude_home / name
        if dst.is_symlink() and dst.resolve() == src.resolve():
            continue
        dst.symlink_to(src)
```

Call it from `agent_environment`'s caller in `prepare.py`, guarded by
`if (Path.home() / ".claude").is_dir():`.

- [ ] **Step 4: Run the tests**

Run: `cd agent_sys && python -m pytest tests/env_mgr/test_o11y_prefix.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add agent_sys/env_mgr/o11y/prefix.py agent_sys/env_mgr/prepare.py agent_sys/tests/env_mgr/test_o11y_prefix.py
git commit -s -m "feat(env_mgr): link (never copy) claude credentials into the prefix"
```

---

## Phase 5 — The CLI surface and the single call site

### Task 5.1: `--agentsview-port` and `--no-agentsview`

**Files:**
- Modify: `agent_sys/cli/main.py`
- Test: `agent_sys/tests/cli/test_agentsview_flags.py`

- [ ] **Step 1: Write the failing test**

```python
# agent_sys/tests/cli/test_agentsview_flags.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The two flags, and the promise that o11y cannot fail a run."""

from __future__ import annotations

import pytest

from cli import main as cli_main


def test_the_port_flag_parses() -> None:
    args = cli_main._parser().parse_args(["run", "pkg", "--agentsview-port", "9001"])
    assert args.agentsview_port == 9001


def test_the_disable_flag_parses() -> None:
    args = cli_main._parser().parse_args(["run", "pkg", "--no-agentsview"])
    assert args.no_agentsview is True


def test_the_default_is_enabled_and_unset() -> None:
    args = cli_main._parser().parse_args(["run", "pkg"])
    assert args.no_agentsview is False
    assert args.agentsview_port is None


def test_disabled_makes_no_external_call(monkeypatch) -> None:
    called = []
    monkeypatch.setattr(
        cli_main, "ensure_running", lambda *a, **k: called.append(1)
    )
    cli_main._start_o11y(port_flag=None, disabled=True)
    assert called == []


def test_a_raising_side_car_does_not_reach_the_caller(monkeypatch) -> None:
    """Belt and braces: even a bug inside ensure_running cannot fail a run."""

    def boom(*a, **k):
        raise RuntimeError("this must never escape")

    monkeypatch.setattr(cli_main, "ensure_running", boom)
    assert cli_main._start_o11y(port_flag=None, disabled=False) is None
```

- [ ] **Step 2: Run and watch it fail**

Run: `cd agent_sys && python -m pytest tests/cli/test_agentsview_flags.py -v`
Expected: `AttributeError: module 'cli.main' has no attribute '_parser'`

- [ ] **Step 3: Implement**

If `main.py` builds its parser inline, extract it into `_parser()` returning the
`ArgumentParser` unchanged, and have `main` call it — a pure refactor, no
behaviour change. Then add to the `run` sub-parser:

```python
    run_p.add_argument(
        "--agentsview-port",
        type=int,
        default=None,
        metavar="N",
        help="port for the AgentsView o11y panel (default 18888; "
             "a port already in use is a warning and a skip)",
    )
    run_p.add_argument(
        "--no-agentsview",
        action="store_true",
        help="do not start the AgentsView o11y panel",
    )
```

and add the call site:

```python
from env_mgr.o11y.agentsview import ensure_running, resolve_port
from env_mgr.o11y.prefix import Prefix


def _start_o11y(port_flag: int | None, disabled: bool) -> str | None:
    """The one call site. Returns the panel URL, or None, and never raises.

    **The bare `except Exception` is deliberate and is the point.** Everything
    inside `ensure_running` already degrades to a warning; this catches the
    case that module has not thought of. An observability side-car that can
    abort a run is a worse bug than a missing panel, and this is the line that
    makes that structurally impossible rather than merely intended.
    """
    if disabled:
        return None
    try:
        prefix = Prefix.resolve(os.environ)
        status = ensure_running(prefix, port=resolve_port(port_flag, os.environ))
        if status.running:
            log.info("agentsview: o11y panel at %s", status.url)
        return status.url
    except Exception as e:  # noqa: BLE001
        log.warning("agentsview: o11y start-up failed (%s); continuing without a panel.", e)
        return None
```

Call `_start_o11y(args.agentsview_port, args.no_agentsview)` once in `main`,
after argument parsing and before the graph runs.

- [ ] **Step 4: Run the tests**

Run: `cd agent_sys && python -m pytest tests/cli/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add agent_sys/cli/main.py agent_sys/tests/cli/test_agentsview_flags.py
git commit -s -m "feat(cli): --agentsview-port / --no-agentsview and the o11y call site"
```

---

## Phase 6 — Acceptance on `examples/demo2`

### Task 6.1: Run the six checks from the spec

**Files:**
- Create: `/home/yihou/ws.agentsview_o11y/recon/ACCEPTANCE.md`

Each check names a file to open and a condition that fails. **A green exit code
proves none of them.**

- [ ] **Step 1: Snapshot the user's Claude directory**

```bash
find "$HOME/.claude/projects" -name '*.jsonl' | sort > /home/yihou/ws.agentsview_o11y/recon/acc_before.txt
```

- [ ] **Step 2: Run demo2 with the panel enabled**

```bash
cd agent_sys
python -m cli.main run examples/demo2 2>&1 | tee /home/yihou/ws.agentsview_o11y/logs/demo2.log
echo "exit=$?"
```

- [ ] **Step 3: Check 1 — the panel answers**

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:18888/
```

Expected: `200`. Anything else fails check 1.

- [ ] **Step 4: Check 2 — this run's transcript is in the prefix**

```bash
find "$HOME/.infera_agent_sys/state/claude/projects" -name '*.jsonl' -newer /home/yihou/ws.agentsview_o11y/recon/acc_before.txt
```

Expected: at least one path. Zero fails check 2.

- [ ] **Step 5: Check 3 — the panel lists that session and no others**

```bash
curl -s 'http://127.0.0.1:18888/api/v1/sessions?limit=50' | python -m json.tool | head -60
```

Read the list. Every entry must be a demo2 session. **One entry from any other
project fails check 3.** Paste the list into `ACCEPTANCE.md`.

- [ ] **Step 6: Check 4 — the user's directory is untouched**

```bash
find "$HOME/.claude/projects" -name '*.jsonl' | sort > /home/yihou/ws.agentsview_o11y/recon/acc_after.txt
diff /home/yihou/ws.agentsview_o11y/recon/acc_{before,after}.txt && echo "CHECK 4 PASS"
```

- [ ] **Step 7: Check 5 — the exit code is unaffected**

```bash
cd agent_sys
python -m cli.main run examples/demo2 --no-agentsview >/dev/null 2>&1; echo "without=$?"
python -m cli.main run examples/demo2 >/dev/null 2>&1; echo "with=$?"
```

Expected: identical. Differing codes fail check 5.

- [ ] **Step 8: Check 6 — a taken port warns once and does not fail**

```bash
python - <<'PY' &
import socket, time
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", 18888)); s.listen(1); time.sleep(180)
PY
sleep 1
cd agent_sys
python -m cli.main run examples/demo2 2>&1 | tee /home/yihou/ws.agentsview_o11y/logs/demo2_portbusy.log
echo "exit=$?"
grep -c 'port 18888 is in use' /home/yihou/ws.agentsview_o11y/logs/demo2_portbusy.log
```

Expected: same exit code as check 5, and the grep counts exactly `1`.
Kill the blocker afterwards with `kill %1` — **only the job this shell started**.

- [ ] **Step 9: Write ACCEPTANCE.md**

One row per check: the command, the observed output, PASS or FAIL. A FAIL goes
back to the responsible phase; it is not written up as a caveat.

- [ ] **Step 10: Commit the documentation of the result**

```bash
git add docs/superpowers/plans/2026-09-03-agentsview-o11y.md
git commit -s -m "docs(agentsview-o11y): acceptance run recorded"
```

### Task 6.2: Document the component

**Files:**
- Modify: `agent_sys/env_mgr/README.md`

- [ ] **Step 1: Add an `o11y` section**

Cover, in this order: what the panel is and where it lives
(`http://127.0.0.1:18888`), the prefix layout and its env vars, the two flags,
what "warning and skip" means and every case that triggers it, and the one
sentence that matters to a reader who is nervous about their own setup — *the
user's `~/.claude` is never read, written, or reconfigured*.

- [ ] **Step 2: Commit**

```bash
git add agent_sys/env_mgr/README.md
git commit -s -m "docs(env_mgr): document the agentsview o11y side-car"
```

---

## Self-review against the spec

| Spec section | Covered by |
|---|---|
| §2 prefix + env vars | Tasks 1.1, 1.2 |
| §3 gate 1 (`CLAUDE_CONFIG_DIR` into the child only) | Task 4.1 |
| §3 gate 2 (`CLAUDE_PROJECTS_DIR`) | Task 1.1 `environment()`, passed to the child in Task 2.2 |
| §3 gate 3 (`disabled_agents`) | Task 2.2 `write_config` |
| §3 gate 4 (separate `AGENTSVIEW_DATA_DIR`) | Task 1.1, asserted in Task 2.2 |
| §3 credential symlinks | Task 4.2 (conditional on Task 0.2) |
| §3 open question on `CLAUDE_CONFIG_DIR` | Task 0.2 |
| §4 trigger + resident daemon | Task 5.1 |
| §4 port resolution order | Task 2.1 |
| §4 taken port warns and skips | Task 2.2, re-checked live in Task 6.1 Step 8 |
| §4 every failure warning-level | Task 2.2, plus the outer guard in Task 5.1 |
| §4 loopback binding | Task 2.2 `--host 127.0.0.1` and `config.toml` |
| §5 no new installer class | Task 3.1 |
| §5 `install.sh` prefix unknown | Task 0.1 |
| §6 five unit tests | Tasks 1.1, 2.1, 2.2, 4.1, 5.1 |
| §7 six acceptance checks | Task 6.1 |
