# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One smoke test against the **real installed binary**.

**Why this file exists, and it is not redundant with `test_o11y_agentsview.py`.**
Every test in that file substitutes a fake `agentsview` shell script which
ignores its config file entirely. That approach is structurally blind to any bug
in what the binary does with what we hand it, and two real bugs proved it:

1. `disabled_agents` naming providers this version does not know — the daemon
   exits 1 on the first one, and 626 green tests coexisted with a panel that had
   never once come up (`recon/ACCEPTANCE.md`, check 1).
2. AgentsView hiding one-shot sessions by default — the panel opens **completely
   empty and looks correct**.

Both are invisible to a fake. So this test writes the config we really produce,
starts the real daemon, and asks it for the sessions we really planted.

**The assertion that matters is a non-zero session count, never HTTP 200.** A
200 with `{"sessions":[],"total":0}` is precisely bug 2, and a health check
would have called it green.

**Nothing here touches the operator's state.** Temporary prefix, temporary
`AGENTSVIEW_DATA_DIR`, ephemeral port — never 18888, never `~/.agentsview`,
never the real `~/.infera_agent_sys/state`. Only the binary is shared, which is
the whole point. Two copies of this test running at once cannot collide, and the
daemon is stopped through its own `serve stop` against *our* data directory,
never a pattern-matched kill: a user's own instance may be on this box.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from env_mgr.o11y import agentsview
from env_mgr.o11y.prefix import Prefix

#: How long the daemon may take to answer after `serve --background` returns.
#: Cold start builds the SQLite archive from the session root; ours holds one
#: file, so this is generous rather than tight.
READY_TIMEOUT_S = 60.0

#: `serve --background` daemonises and returns at once.
LAUNCH_TIMEOUT_S = 30.0

#: Measured against v0.42.0: the default `/api/v1/sessions` view **excludes**
#: one-shot sessions, and a two-turn transcript is one. The parameter that
#: includes them is snake_case; `includeOneShot` and `exclude_one_shot=false`
#: were both measured to do nothing. See `zonelink/ENFORCED_MODE.md`'s sibling
#: notes and the message to `recon` for the four spellings tried.
#:
#: This test deliberately does **not** assert that the *default* view is
#: non-empty. That is the behaviour a user meets and it is the one worth
#: fixing, but the fix is a config decision owned elsewhere; asserting it here
#: today would only add a red test. When that lands, the assertion belongs in
#: this file, because it is the one that stops the empty-panel bug shipping.
INCLUDE_ONE_SHOT = "include_one_shot=true"

_PID_RE = re.compile(r"pid (\d+)")


def _binary() -> str | None:
    """The installed binary, or `None`. Prefix first, then `PATH`."""
    try:
        candidate = Prefix.resolve(os.environ).bin / "agentsview"
    except KeyError:  # no $HOME and no AGENT_SYS_HOME
        candidate = None
    if candidate is not None and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which("agentsview")


#: **Skip, never fail.** A fresh checkout has no binary and its test run must
#: stay green; this file's job is to catch a bug in the binary we ship with, not
#: to make the absence of one an error.
requires_binary = pytest.mark.skipif(
    _binary() is None,
    reason="the agentsview binary is not installed; this smoke test needs the real one",
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _plant_session(prefix: Prefix, session_id: str, first_message: str) -> None:
    """One Claude Code transcript, in Claude Code's own shape.

    Two turns, because that is the smallest thing AgentsView will call a
    session, and a slug subdirectory because that is how the real CLI names
    them — one per working directory.
    """
    slug = prefix.claude_home / "projects" / "-tmp-agent-sys-smoke"
    slug.mkdir(parents=True, exist_ok=True)
    turns = [
        {
            "parentUuid": None,
            "isSidechain": False,
            "type": "user",
            "message": {"role": "user", "content": first_message},
            "uuid": f"{session_id[:8]}-0000-0000-0000-000000000001",
            "timestamp": "2026-09-03T12:00:00.000Z",
            "cwd": "/tmp/agent-sys/smoke",
            "sessionId": session_id,
            "version": "1.0.0",
            "userType": "external",
        },
        {
            "parentUuid": f"{session_id[:8]}-0000-0000-0000-000000000001",
            "isSidechain": False,
            "type": "assistant",
            "message": {
                "id": "msg_1",
                "role": "assistant",
                "model": "claude-opus-5",
                "content": [{"type": "text", "text": "acknowledged"}],
                "usage": {"input_tokens": 10, "output_tokens": 3},
            },
            "uuid": f"{session_id[:8]}-0000-0000-0000-000000000002",
            "timestamp": "2026-09-03T12:00:01.000Z",
            "cwd": "/tmp/agent-sys/smoke",
            "sessionId": session_id,
            "version": "1.0.0",
        },
    ]
    (slug / f"{session_id}.jsonl").write_text("".join(json.dumps(t) + "\n" for t in turns))


def _get(url: str, timeout: float = 10.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310
            return int(r.status), r.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()


class Panel:
    """A live daemon and the one session we planted in it."""

    def __init__(self, port: int, session_id: str, first_message: str) -> None:
        self.port = port
        self.session_id = session_id
        self.first_message = first_message

    def sessions(self, query: str = INCLUDE_ONE_SHOT) -> list[dict]:
        status, body = _get(f"http://127.0.0.1:{self.port}/api/v1/sessions?{query}")
        assert status == 200, f"the sessions API answered {status}"
        return list(json.loads(body).get("sessions", []))


@pytest.fixture()
def panel(tmp_path: Path) -> Iterator[Panel]:
    binary = _binary()
    assert binary is not None  # guarded by `requires_binary`

    prefix = Prefix(tmp_path / "prefix")
    prefix.create()
    session_id = "5m0ke7e5-0000-4000-8000-00000000000a"
    first_message = "planted by the agent_sys smoke test"
    _plant_session(prefix, session_id, first_message)

    # **The config we really ship**, produced by the module that ships it — so a
    # change to `write_config` or to `OTHER_PROVIDERS` is exercised here rather
    # than mirrored into a copy that can drift out of agreement with it.
    agentsview.write_config(prefix, agentsview.OTHER_PROVIDERS)

    # `HOME` too: the binary falls back to `~/.agentsview` for anything the
    # explicit variables do not cover, and the operator's must stay untouched.
    env = {
        **prefix.environment(),
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(prefix.root),
    }
    port = _free_port()
    started = subprocess.run(  # noqa: S603 — `binary` is a resolved path
        [
            binary,
            "serve",
            "--background",
            "--no-browser",
            "--no-update-check",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=LAUNCH_TIMEOUT_S,
        check=False,
    )
    if started.returncode != 0:
        pytest.fail(
            f"`agentsview serve --background` exited {started.returncode}. "
            f"This is the failure mode a fake binary cannot reproduce.\n"
            f"stdout: {started.stdout}\nstderr: {started.stderr}"
        )
    # Recorded now, so teardown can stop **this** process even if `serve stop`
    # cannot find its own state. One pid we printed ourselves; never a pattern.
    match = _PID_RE.search(started.stdout or "")
    pid = int(match.group(1)) if match else None

    try:
        deadline = time.monotonic() + READY_TIMEOUT_S
        while time.monotonic() < deadline:
            status, _ = _get(f"http://127.0.0.1:{port}/", timeout=3.0)
            if status == 200:
                break
            time.sleep(0.5)
        else:
            pytest.fail(f"the daemon never answered on 127.0.0.1:{port}")
        yield Panel(port, session_id, first_message)
    finally:
        subprocess.run(  # noqa: S603
            [binary, "serve", "stop"],
            env=env,
            capture_output=True,
            timeout=LAUNCH_TIMEOUT_S,
            check=False,
        )
        if pid is not None and not agentsview.port_is_free(port):
            # Still up: signal the **recorded** pid and nothing else. An
            # `os.kill` on a pid we read from our own launch output cannot
            # reach a stranger's daemon the way a name match could.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, 15)


@requires_binary
def test_the_real_daemon_serves_the_session_we_planted(panel: Panel) -> None:
    """**The assertion this whole file exists for: a non-zero count.**

    Not 200, and not "the process is alive". A daemon that starts, answers
    every health check and returns an empty list is the exact shape of the
    empty-panel bug, and it is what shipped.
    """
    sessions = panel.sessions()

    assert sessions, (
        "the daemon answered 200 with zero sessions. Either the config we write "
        "stopped it reading our session root, or the default view is filtering "
        "them out again — both have happened, and both look healthy from "
        "outside."
    )
    assert any(s.get("id") == panel.session_id for s in sessions), (
        f"sessions came back but none is ours ({panel.session_id}); "
        f"got {[s.get('id') for s in sessions]}"
    )


@requires_binary
def test_it_is_our_prefix_being_read_and_not_some_other_root(panel: Panel) -> None:
    """The count could be non-zero for the wrong reason — a stray archive, or a
    provider we failed to disable. Reading the content settles it."""
    ours = [s for s in panel.sessions() if s.get("id") == panel.session_id]

    assert len(ours) == 1
    assert ours[0].get("first_message") == panel.first_message
    assert ours[0].get("agent") == "claude"
