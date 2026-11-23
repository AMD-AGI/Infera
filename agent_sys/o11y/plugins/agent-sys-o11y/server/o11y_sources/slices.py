"""History slices for the model passes (shared by o11y_log and o11y_facts; read-only).

A slice is the new part of one session's main history since the caller's model cursor, condensed to text
with `@<offset>` markers so the model can cite evidence. First sight reads only the last `lookback_bytes`;
a history that was replaced or rewritten is not re-read: the cursor jumps to its end (design §8.2). The
caller advances its cursors only after its outputs are written.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import transcripts as tr_src
from .cursor import Cursor, CursorStore, _sha, read_new, tail_start


def line_uuid(path: str, offset: int) -> str | None:
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            d = json.loads(f.readline())
        return d.get("uuid") if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


class Slice:
    def __init__(self, session, path: str, text: str, lo: int, hi: int, cursor: Cursor):
        self.session, self.path, self.text, self.lo, self.hi, self.cursor = (
            session,
            path,
            text,
            lo,
            hi,
            cursor,
        )

    def evidence(self, offsets) -> list[dict]:
        ev = []
        for o in offsets or []:
            if isinstance(o, int) and self.lo <= o < self.hi:
                item = {"path": self.path, "offset": o}
                u = line_uuid(self.path, o)
                if u:
                    item["uuid"] = u
                ev.append(item)
        return ev


def _end_cursor(path: str) -> Cursor:
    """A cursor at the end of the last complete line: the model pass resumes from here (design §8.2)."""
    st = os.stat(path)
    return Cursor(path=path, inode=st.st_ino, offset=tail_start(path, 0))


def collect_slices(
    store: CursorStore, sessions, prefix: str, budget_chars: int, lookback_bytes: int
) -> list[Slice]:
    """New history of each session for one model pass. First sight reads only the last `lookback_bytes`;
    a history that was replaced or rewritten is not re-read: the cursor jumps to its end."""
    out = []
    live = [s for s in sessions if s.transcript_path and Path(s.transcript_path).exists()]
    if not live:
        return out
    per = max(2000, budget_chars // len(live))
    for s in live:
        path = s.transcript_path
        key = f"{prefix}:{path}"
        cur = store.get(key, path)
        if cur.offset == 0 and not cur.inode:  # first sight: only the recent tail
            st = os.stat(path)
            cur = Cursor(path=path, inode=st.st_ino, offset=tail_start(path, lookback_bytes))
        chunk = read_new(cur)
        if chunk.reset:
            store.put(key, _end_cursor(path))
            continue
        if not chunk.lines:
            if chunk.cursor.offset != cur.offset:
                store.put(key, chunk.cursor)
            continue
        text, end_off = tr_src.condense(chunk.lines, per)
        used = [ln for ln in chunk.lines if ln[0] < end_off] or chunk.lines[
            :1
        ]  # budget may stop early
        last_off, last_raw = used[-1]
        nc = Cursor(
            path=path,
            inode=chunk.cursor.inode,
            offset=last_off + len(last_raw) + 1,
            last_start=last_off,
            last_sha=_sha(last_raw),
        )
        if not text.strip():
            store.put(key, nc)  # nothing worth reading; skip ahead
            continue
        out.append(Slice(s, path, text, used[0][0], nc.offset, nc))
    return out


def sessions_block(sessions) -> str:
    return "\n".join(
        f"- {s.session_id} name={s.name or '-'} role={s.role} status={s.status}" for s in sessions
    )
