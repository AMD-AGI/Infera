"""Incremental JSONL reading with a durable cursor.

Cursor = (path, inode, offset of the byte after the last consumed '\\n', start offset and sha256 of the
last consumed line). On the next read the cursor is verified: a different inode, a file shorter than the
offset, or a changed last line means the file was replaced or rewritten, and reading restarts at 0
(`reset=True`). Only complete lines are returned. The caller commits the new cursor only after its
outputs are durably written, so a crash re-reads instead of skipping.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import core


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


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_new(cur: Cursor, max_bytes: int | None = None) -> Chunk:
    p = cur.path
    try:
        st = os.stat(p)
    except FileNotFoundError:
        return Chunk([], cur, missing=True)
    reset = False
    start = cur.offset
    if cur.inode and st.st_ino != cur.inode:
        reset = True
    elif st.st_size < cur.offset:
        reset = True
    elif cur.offset and cur.last_sha:
        with open(p, "rb") as f:
            f.seek(cur.last_start)
            prev = f.read(cur.offset - cur.last_start)
        if _sha(prev.rstrip(b"\n")) != cur.last_sha:
            reset = True
    if reset:
        start = 0
    new = Cursor(
        path=p,
        inode=st.st_ino,
        offset=start,
        last_start=cur.last_start if not reset else 0,
        last_sha=cur.last_sha if not reset else "",
        extra={} if reset else dict(cur.extra),
    )
    end = st.st_size if max_bytes is None else min(st.st_size, start + max_bytes)
    if end <= start:
        return Chunk([], new, reset=reset)
    with open(p, "rb") as f:
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


class CursorStore:
    def __init__(self, root: Path):
        self.dir = Path(root) / "state" / "analyze" / "cursors"

    def _file(self, key: str) -> Path:
        return self.dir / (hashlib.sha1(key.encode()).hexdigest()[:20] + ".json")

    def get(self, key: str, path: str, follow_rename: bool = False) -> Cursor:
        """The stored cursor for `key`. A different path normally means a different file (fresh cursor, extra
        kept); with `follow_rename` (keys that name an inode) the cursor follows the renamed file."""
        try:
            d = json.loads(self._file(key).read_text(encoding="utf-8"))
            c = Cursor(
                **{
                    k: d[k]
                    for k in ("path", "inode", "offset", "last_start", "last_sha", "extra")
                    if k in d
                }
            )
            if c.path != path:
                c = (
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
        except (FileNotFoundError, ValueError, TypeError, KeyError):
            return Cursor(path=path)

    def put(self, key: str, cur: Cursor) -> None:
        d = asdict(cur)
        d["key"] = key
        core.write_json_atomic(self._file(key), d)


def parse(raw: bytes):
    try:
        d = json.loads(raw)
    except ValueError:
        return None
    return d if isinstance(d, dict) else None
