"""tmux -L $AGENT_SYS_TMUX (default agent_sys), session agent_sys: the server's window and the runs' windows. A test or
demo server uses a tmux of its own. Also the environment agent_sys gives the processes it starts."""

from __future__ import annotations

import os
import re
import subprocess

from .i18n import _
from .model import AgentSysError

TMUX = os.environ.get("AGENT_SYS_TMUX", "agent_sys")
SESSION = "agent_sys"
# Set inside a Claude Code session (CLAUDE*, and ANTHROPIC_* from its settings).
# Inherited by the server, they would reach the runners it starts and override the gateway settings in
# ~/.claude/settings.json.
PARENT_SESSION = re.compile(r"(CLAUDE\w*|ANTHROPIC\w*|IS_SANDBOX)$")


def clean_env():
    return {k: v for k, v in os.environ.items() if not PARENT_SESSION.match(k)}


def clean_own_env():
    """Drop the parent session's variables from this process's environment (and so its children's)."""
    env = clean_env()  # before clear(): it reads os.environ
    os.environ.clear()
    os.environ.update(env)


def run(*args):
    return subprocess.run(
        ["tmux", "-L", TMUX, *args], env=clean_env(), capture_output=True, text=True
    )


def window_names():
    r = run("list-windows", "-t", SESSION, "-F", "#W")
    return set(r.stdout.split()) if r.returncode == 0 else set()


def window_name(*names):
    """The first of names that no window of the session has, else the last one numbered -2, -3, … (attach -t
    agent_sys:<name> must find one). Only letters, digits, _ and -: tmux reads a "." in a target as a pane index."""
    taken = window_names()
    names = [re.sub(r"[^A-Za-z0-9_-]", "_", n) for n in names]
    name = next((n for n in names if n not in taken), names[-1])
    stem, n = name, 2
    while name in taken:
        name, n = f"{stem}-{n}", n + 1
    return name


def open_window(name, cmd, cwd):
    """A new window `name` running the shell command cmd in cwd (the session too, if there is none); its window id."""
    where = (
        ("new-session", "-d", "-s", SESSION)
        if run("has-session", "-t", SESSION).returncode
        else ("new-window", "-d", "-t", f"{SESSION}:")
    )
    r = run(*where, "-n", name, "-c", str(cwd), "-P", "-F", "#{window_id}", cmd)
    if r.returncode:
        raise AgentSysError(_("无法在 tmux 中启动运行：{error}", error=r.stderr.strip()), 500)
    return r.stdout.strip()


def attach(window):
    return f"tmux -L {TMUX} attach -t {SESSION}:{window}"
