"""The processes of a run, found in /proc: its runner, the claude processes of its session, its acceptance checks.
A pid may be reused after a reboot: each is checked to still be what it was before it is trusted or killed."""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

TERM_WAIT = 10  # claude processes of the session get TERM, then KILL after this long
# in the environment of a check and of what it starts: the run directory, symlinks resolved
CHECK_MARK = "AGENT_SYS_CHECK_RUN"


def cmdline(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace").split("\0")[:-1]
    except OSError:
        return None


def other_pids():
    pids = (int(e.name) for e in Path("/proc").iterdir() if e.name.isdigit())
    return [pid for pid in pids if pid != os.getpid()]


def is_runner(pid, run_dir):
    """pid is still the runner of run_dir."""
    args = cmdline(pid) if pid else None
    return bool(args) and "agent_sys_v2.runner" in args and str(run_dir) in args


def session_pids(session_id):
    """claude processes of a session: their command line has --session-id / --resume with the sid."""
    marks = {f"--session-id={session_id}", f"--resume={session_id}"}
    pids = []
    for pid in other_pids():
        args = cmdline(pid) or []
        pairs = {f"{a}={b}" for a, b in zip(args, args[1:])}
        if marks & (set(args) | pairs):
            pids.append(pid)
    return pids


def check_pids(run_dir):
    """The checks of run_dir still running, and whatever they started: their environment has CHECK_MARK (however
    run_dir is reached: the runner resolves it)."""
    mark = f"{CHECK_MARK}={Path(run_dir).resolve()}".encode()
    pids = []
    for pid in other_pids():
        try:
            if mark in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
                pids.append(pid)
        except OSError:  # another user's process, or gone
            pass
    return pids


def children(pid):
    out = []
    for entry in Path("/proc").iterdir():
        try:
            stat = (entry / "stat").read_text() if entry.name.isdigit() else ""
            if stat and stat.rsplit(")", 1)[1].split()[1] == str(pid):
                out.append(int(entry.name))
        except (OSError, IndexError):
            pass
    return out


def signal_all(pids, sig):
    for pid in pids:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def zombie(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return True


def wait_gone(pids, seconds):
    end = time.time() + seconds
    while time.time() < end:
        pids = [p for p in pids if Path(f"/proc/{p}").exists() and not zombie(p)]
        if not pids:
            return []
        time.sleep(0.2)
    return pids


def end_session(session_id, also=()):
    """TERM the session's claude processes (and `also`), KILL what is left after TERM_WAIT."""
    pids = sorted(set(session_pids(session_id)) | set(also))
    signal_all(pids, signal.SIGTERM)
    signal_all(wait_gone(pids, TERM_WAIT), signal.SIGKILL)


def end_checks(run_dir):
    """KILL the checks of a run whose runner is gone: nobody would time them out, and they may be using the task's
    machines after it let them go."""
    signal_all(check_pids(run_dir), signal.SIGKILL)
