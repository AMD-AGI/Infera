"""History slices for the model passes (o11y_extract and o11y_log; read-only).

A slice is the new part of one session's main history since the caller's model cursor, condensed to text
with `@<offset>` markers so the model can cite evidence. First sight reads only the last `lookback_bytes`;
a history that was replaced or rewritten is not re-read: the cursor jumps to its end (design §8.2). The
caller advances its cursors only after its outputs are written.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from . import transcripts as tr_src
from .cursor import Cursor, _sha, read_new, tail_start


def line_fields(path: str, offset: int) -> dict:
    """`uuid` and `timestamp` of the history line at `offset` (evidence for a model output), else {}."""
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            d = json.loads(f.readline())
    except (OSError, ValueError):
        return {}
    return {
        k: d[k] for k in ("uuid", "timestamp") if isinstance(d, dict) and isinstance(d.get(k), str)
    }


@dataclass
class Slice:
    session: object  # registry.Session
    path: str
    text: str
    lo: int  # byte offsets [lo, hi) of the lines given to the model
    hi: int
    cursor: Cursor  # committed by the caller after its outputs are written

    def evidence(self, offsets) -> list[dict]:
        ev = []
        for o in offsets or []:
            if isinstance(o, int) and self.lo <= o < self.hi:
                ev.append({"path": self.path, "offset": o, **line_fields(self.path, o)})
        return ev


def _end_cursor(path: str) -> Cursor:
    """A cursor at the end of the last complete line: the model pass resumes from here (design §8.2)."""
    st = os.stat(path)
    return Cursor(path=path, inode=st.st_ino, offset=tail_start(path, 0))


def collect_slices(
    store, sessions, prefix: str, budget_chars: int, lookback_bytes: int
) -> list[Slice]:
    """New history of each session for one model pass. `store` is the job's cursor store (get/put)."""
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


def prompt_sections(slices: list[Slice]) -> list[str]:
    """The model passes' "new slices" part of the prompt, one section per session."""
    return ["\n## New slices"] + [
        f"\n### session {sl.session.session_id} ({sl.session.name or '-'}, {sl.session.role})\n{sl.text}"
        for sl in slices
    ]
