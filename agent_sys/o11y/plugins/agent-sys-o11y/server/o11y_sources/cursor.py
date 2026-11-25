"""Incremental JSONL reading with a durable cursor.

Cursor = (path, inode, offset of the byte after the last consumed '\\n', start offset and sha256 of the
last consumed line). On the next read the cursor is verified: a different inode, a file shorter than the
offset, or a changed last line means the file was replaced or rewritten, and reading restarts at 0
(`reset=True`). Only complete lines are returned. The caller commits the new cursor only after its
outputs are durably written, so a crash re-reads instead of skipping.

Stateless: persistence is each job's own (o11y_job.CursorStore in `state/<kind>/cursors`).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field


@dataclass
class Cursor:
    path: str
    inode: int = 0
    offset: int = 0
    last_start: int = 0
    last_sha: str = ""
    extra: dict = field(default_factory=dict)  # per-source accumulated state (token totals, ...)


@dataclass
class Chunk:
    lines: list[tuple[int, bytes]]  # (byte offset of line start, raw line without '\n')
    cursor: Cursor  # cursor to commit after processing
    reset: bool = False  # file was rewritten/replaced; caller must rebuild derived state
    missing: bool = False
    problem: str | None = None  # the opener refused the file


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_new(cur: Cursor, max_bytes: int | None = None, opener=None) -> Chunk:
    """New complete lines since `cur`, all read from ONE open file. `opener(path) -> (file, problem)` (e.g.
    `owned_open`) checks the opened file itself; a refused file comes back as `problem`, nothing read."""
    p = cur.path
    if opener is None:
        try:
            f = open(p, "rb")
        except FileNotFoundError:
            return Chunk([], cur, missing=True)
    else:
        f, why = opener(p)
        if f is None:
            return Chunk([], cur, missing=not os.path.lexists(p), problem=why)
    with f:
        st = os.fstat(f.fileno())
        start = cur.offset
        reset = bool(cur.inode and st.st_ino != cur.inode) or st.st_size < cur.offset
        if not reset and cur.offset and cur.last_sha:
            f.seek(cur.last_start)
            reset = _sha(f.read(cur.offset - cur.last_start).rstrip(b"\n")) != cur.last_sha
        if reset:
            start = 0
        new = Cursor(
            path=p,
            inode=st.st_ino,
            offset=start,
            last_start=0 if reset else cur.last_start,
            last_sha="" if reset else cur.last_sha,
            extra={} if reset else dict(cur.extra),
        )
        end = st.st_size if max_bytes is None else min(st.st_size, start + max_bytes)
        if end <= start:
            return Chunk([], new, reset=reset)
        f.seek(start)
        data = f.read(end - start)
    last_nl = data.rfind(b"\n")
    if last_nl < 0:
        return Chunk([], new, reset=reset)  # no complete line yet
    data = data[: last_nl + 1]
    lines = []
    pos = start
    for raw in data.split(b"\n")[:-1]:
        lines.append((pos, raw))
        pos += len(raw) + 1
    off_last, raw_last = lines[-1]
    new.offset = start + len(data)
    new.last_start = off_last
    new.last_sha = _sha(raw_last)
    return Chunk(lines, new, reset=reset)


def tail_start(path: str, nbytes: int) -> int:
    """Offset of the first line boundary at or after size-nbytes (for lookback-limited first reads)."""
    size = os.path.getsize(path)
    if size <= nbytes:
        return 0
    with open(path, "rb") as f:
        f.seek(size - nbytes)
        f.readline()
        return f.tell()


def parse(raw: bytes):
    try:
        d = json.loads(raw)
    except ValueError:
        return None
    return d if isinstance(d, dict) else None
