"""o11y_job — the mechanics every analysis job repeats (design §8.1).

`run_consumer(kind, handler)`: non-blocking fcntl lock per kind (busy → exit 0) → target =
`requests/analysis.json[kind].seq` → handler → record `served` in `state/<kind>/served.json` → loop while
`seq > served`, within the budget → exit 0. Stops early when the fleet's main session is gone (the log job
first makes one pass: the crash is its to report). Nothing in the fleet root is deleted. `CursorStore` keeps a job's history cursors
in its own state directory; `llm` is the structured model call.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from o11y_sources.cursor import Cursor

from .door import api, fleet

BUDGET_S = 300  # one invocation's runtime budget (design §8.1; a code constant per review F4)


@dataclass
class Job:
    root: Path
    kind: str
    state_dir: Path
    deadline: float
    report: dict


def write_private(path: Path, doc) -> None:
    """Whole-file JSON, owner-only, never through a symlink: fresh tmp (O_EXCL|O_NOFOLLOW) + fsync + rename."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{os.urandom(4).hex()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_private(path: Path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


class CursorStore:
    """A job's own history cursors (`state/<kind>/cursors/`): never read by another job."""

    def __init__(self, state_dir: Path):
        self.dir = Path(state_dir) / "cursors"

    def _file(self, key: str) -> Path:
        return self.dir / (hashlib.sha1(key.encode()).hexdigest()[:20] + ".json")

    def get(self, key: str, path: str, follow_rename: bool = False) -> Cursor:
        d = read_private(self._file(key), None)
        if not isinstance(d, dict):
            return Cursor(path=path)
        c = Cursor(
            **{
                k: d[k]
                for k in ("path", "inode", "offset", "last_start", "last_sha", "extra")
                if k in d
            }
        )
        if c.path != path:  # a different file, unless the key names an inode (rotated OTel files)
            return (
                Cursor(
                    path=path,
                    inode=c.inode,
                    offset=c.offset,
                    last_start=c.last_start,
                    last_sha=c.last_sha,
                    extra=c.extra,
                )
                if follow_rename
                else Cursor(path=path, extra=c.extra)
            )
        return c

    def put(self, key: str, cur: Cursor) -> None:
        write_private(self._file(key), {**asdict(cur), "key": key})


def _requested_seq(root: Path, kind: str) -> int:
    doc = read_private(root / "requests" / "analysis.json", {})
    try:
        return int((doc.get(kind) or {}).get("seq") or 0)
    except (AttributeError, TypeError, ValueError):
        return 0


def main_gone(root: Path) -> bool:
    """True when the registry names a main and it is not alive (T is the only judge, review A4)."""
    sessions, _ = api.read_registry(root)
    mains = [s for s in sessions if s.get("role") == "main"]
    return bool(mains) and not any(s.get("alive") for s in mains)


def run_consumer(
    kind: str, handler, budget_s: float = BUDGET_S, root: Path | None = None, force: bool = False
) -> dict:
    """Serve `requests/analysis.json[kind]` until drained or out of budget. `handler(job) -> None`.
    `force` serves one pass even without a pending request (manual runs and tests)."""
    root = Path(root) if root else fleet.fleet_dir()
    state_dir = root / "state" / kind
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    report: dict = {"kind": kind, "pid": os.getpid(), "fleet_dir": str(root), "passes": 0}
    t0 = time.time()
    lock_path = root / "locks" / f"{kind}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            report["skipped"] = "lock busy"
            return report
        served_path = state_dir / "served.json"
        while True:
            served = int(read_private(served_path, {}).get("seq", 0))
            target = _requested_seq(root, kind)
            if target <= served and not (force and report["passes"] == 0):
                break
            if main_gone(root):  # the log job still makes one pass: the crash is its to report
                report["main_gone"] = True
                if kind != "log" or report["passes"]:
                    break
            handler(Job(root, kind, state_dir, t0 + budget_s, report))
            report["passes"] += 1
            write_private(served_path, {"seq": max(target, served), "ts": time.time()})
            if time.time() - t0 >= budget_s:
                report["budget_reached"] = True
                break
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    report["duration_s"] = round(time.time() - t0, 2)
    return report


def main(kind: str, handler, argv: list[str] | None = None) -> int:
    """Entry point of `python -m o11y_<kind>` (launched by o11y-run). Exit 0 = drained or lock busy."""
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[argv.index("--fleet-dir") + 1]).expanduser() if "--fleet-dir" in argv else None
    try:
        report = run_consumer(kind, handler, root=root, force="--force" in argv)
    except Exception as exc:  # noqa: BLE001 - one log line, non-zero exit (design §8.1)
        print(f"o11y-run {kind}: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 1
    log = Path(report["fleet_dir"]) / "state" / kind / "runs.jsonl"
    fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        os.write(fd, (json.dumps(report, default=str) + "\n").encode())
    finally:
        os.close(fd)
    print(json.dumps(report, default=str))
    return 0
