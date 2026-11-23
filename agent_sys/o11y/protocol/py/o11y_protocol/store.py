"""Files under the fleet root: paths, locked JSONL append, tolerant JSONL read, atomic whole-file write."""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import secrets
from collections.abc import Iterator
from pathlib import Path
from typing import Any

MAX_LINE = 64 * 1024

LEDGER = "ledger/runs.jsonl"
EVENTS = "logs/events.jsonl"
LAUNCHES = "registry/launches.jsonl"
SESSIONS = "registry/sessions.json"
CONFIG = {
    "trigger": "config/trigger.json",
    "analysis": "config/analysis.json",
    "display": "config/display.json",
}
FACTS = {t: f"facts/{t}.json" for t in ("general", "detail", "kernelforge")}
REQUESTS = "requests/analysis.json"
OTEL_HEALTH = "otel/health.json"


class LineTooLong(ValueError):
    pass


def ensure_dir(path: Path) -> None:
    """Create `path` and its missing parents with mode 0700 (spec §8.6)."""
    missing = []
    p = path
    while not p.exists():
        missing.append(p)
        p = p.parent
    for d in reversed(missing):
        try:
            d.mkdir(mode=0o700)
        except FileExistsError:
            pass


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def new_id(prefix: str) -> str:
    return f"{prefix}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(3)}"


def append(path: Path, record: dict) -> int:
    """Append one record as one line under an exclusive flock; returns the byte offset of the line."""
    line = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    if len(line) > MAX_LINE:
        raise LineTooLong(f"record is {len(line)} bytes, limit {MAX_LINE}")
    ensure_dir(path.parent)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            offset = os.lseek(fd, 0, os.SEEK_END)
            # A previous writer killed mid-line leaves no '\n'; start ours on a fresh line so it stays parseable.
            if offset > 0:
                with open(path, "rb") as r:
                    r.seek(offset - 1)
                    if r.read(1) != b"\n":
                        line = b"\n" + line
                        offset += 1
            written = os.write(fd, line)
            if written != len(line):
                raise OSError(f"short write {written}/{len(line)} to {path}")
            os.fsync(fd)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    return offset


def read_jsonl(path: Path, start: int = 0) -> Iterator[tuple[int, Any | None, bytes]]:
    """Yield (offset, record or None if invalid JSON, raw line) for complete lines only."""
    try:
        with open(path, "rb") as f:
            f.seek(start)
            offset = start
            for raw in f:
                if not raw.endswith(b"\n"):
                    return  # partial last line: a writer is mid-append or died; skip it
                try:
                    rec = json.loads(raw)
                except ValueError:
                    rec = None
                yield offset, rec, raw
                offset += len(raw)
    except FileNotFoundError:
        return


def write_json(path: Path, doc: Any) -> None:
    """Whole-file write: tmp in the same dir, fsync, rename."""
    ensure_dir(path.parent)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(
        os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8"
    ) as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    """Whole-file text write with the same tmp + fsync + rename discipline as write_json."""
    ensure_dir(path.parent)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(
        os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8"
    ) as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> Any | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return None
