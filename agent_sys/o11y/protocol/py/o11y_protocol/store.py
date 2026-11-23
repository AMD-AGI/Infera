"""Files under the fleet root: paths, locked JSONL append, tolerant JSONL read, atomic owner-only whole-file
write. Standard library only (bin/o11y-launch imports it)."""

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
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            offset = os.lseek(fd, 0, os.SEEK_END)
            # A previous writer killed mid-line leaves no '\n'; start ours on a fresh line so it stays parseable.
            if offset > 0 and os.pread(fd, 1, offset - 1) != b"\n":
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


def write_bytes(path: Path, data: bytes) -> None:
    """Whole-file write, owner-only, never through a symlink: a fresh tmp (O_EXCL|O_NOFOLLOW, 0600) in the same
    directory, fsync, rename over the target. Readers never see a torn document."""
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(4)}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view) :]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def write_json(path: Path, doc: Any) -> None:
    write_bytes(path, (json.dumps(doc, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))


def write_text(path: Path, text: str) -> None:
    write_bytes(path, text.encode("utf-8"))


def read_json(path: Path, default: Any = None) -> Any:
    """The document, or `default` when the file is missing, unreadable or not valid JSON."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default
