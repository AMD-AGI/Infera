"""The one-shot invocation shared by `o11y_log` and `o11y_facts` (design §8.1).

The two never import each other; both run the same skeleton with their own kind, lock, state dir and pass
function: non-blocking lock for the kind → claim queue items of the kind → pass → re-scan the queue and loop
while new items arrived, within the budget → exit. It stops early when the main session is gone. This is
the only module of `o11y_sources` that touches the fleet root, and only its `locks/`, `queue/` and the
caller's own state directory.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from .config import PLUGIN_DIR

LAUNCHER = PLUGIN_DIR / "bin" / "o11y-launch"


@contextmanager
def lock(root: Path, name: str, blocking: bool = False, timeout_s: float = 60):
    p = root / "locks" / f"{name}.lock"
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(p, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    t0 = time.time()
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if not blocking or time.time() - t0 > timeout_s:
                os.close(fd)
                yield False
                return
            time.sleep(0.2)
    try:
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()} {time.time():.0f}\n".encode())
        yield True
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def claim_queue(root: Path, kind: str) -> list[Path]:
    """Move this kind's queue items to queue/claimed/ (an atomic claim no parallel run can repeat)."""
    q = root / "queue"
    claimed: list[Path] = []
    if not q.is_dir():
        return claimed
    dst = q / "claimed"
    for f in sorted(q.glob("*.json")):
        k = None
        try:
            k = json.loads(f.read_text(encoding="utf-8")).get("kind")
        except (OSError, ValueError, AttributeError):
            pass
        if k is None:  # file-name convention <ts>-<kind>.json
            k = f.stem.rsplit("-", 1)[-1]
        if k == kind:
            dst.mkdir(exist_ok=True, mode=0o700)
            target = dst / f"{os.getpid()}-{f.name}"
            try:
                os.replace(f, target)
                claimed.append(target)
            except FileNotFoundError:
                pass  # another run claimed it first
    return claimed


def drop_claimed(root: Path, claimed: list[Path]) -> int:
    """Remove drained items — only files whose resolved path lies inside the fleet root's queue/."""
    qdir = (root / "queue").resolve()
    n = 0
    for p in claimed:
        try:
            rp = p.resolve(strict=True)
        except FileNotFoundError:
            continue
        if rp.is_file() and qdir in rp.parents:
            rp.unlink()
            n += 1
    return n


def sweep() -> None:
    """Design §7: the collector sweep also runs at every analyzer start (covers a SIGKILLed last session)."""
    if LAUNCHER.is_file() and os.access(LAUNCHER, os.X_OK):
        try:
            subprocess.Popen(
                [str(LAUNCHER), "--sweep"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            pass


def main_launch_dir(root: Path) -> str | None:
    """The main session's launch directory (relative task-book paths resolve against it)."""
    from o11y_core import api, store

    doc = store.read_json(root / store.SESSIONS) or {}
    for s in doc.get("sessions") or []:
        if isinstance(s, dict) and s.get("role") == "main":
            return s.get("launch_dir") or s.get("cwd")
    try:
        launches = api.launches(root)
    except Exception:  # noqa: BLE001 - a missing or invalid launch log must not stop analysis
        launches = []
    return next((r["cwd"] for r in launches if r.get("role") == "main" and r.get("cwd")), None)


def run(
    root: Path,
    kind: str,
    budget_s: float,
    one_pass: Callable[[float, dict], bool],
    do_sweep: bool = True,
) -> dict:
    """Drive `one_pass(deadline, report) -> keep_going` under the kind's lock until the queue is drained."""
    report: dict = {"kind": kind, "pid": os.getpid(), "fleet_dir": str(root), "passes": 0}
    t0 = time.time()
    deadline = t0 + budget_s
    if do_sweep:
        sweep()
    with lock(root, f"analyze-{kind}") as got:
        if not got:
            report["skipped"] = f"another {kind} run is active"
            return report
        claimed: list[Path] = []
        while True:
            batch = claim_queue(root, kind)
            if report["passes"] and not batch:
                break  # served every request that arrived while we ran (spec §4.5)
            claimed += batch
            report["passes"] += 1
            keep_going = one_pass(deadline, report)
            drop_claimed(root, batch)
            if not keep_going:
                break
            if time.time() >= deadline:
                report.setdefault("notes", []).append("budget reached")
                break
        report["claimed"] = [p.name for p in claimed]
    report["duration_s"] = round(time.time() - t0, 2)
    return report


def append_run_log(state_dir: Path, report: dict) -> None:
    """The consumer's private run log (one line per invocation: cost, passes, notes)."""
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(
        state_dir / "runs.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600
    )
    try:
        os.write(fd, (json.dumps(report, default=str) + "\n").encode())
    finally:
        os.close(fd)
