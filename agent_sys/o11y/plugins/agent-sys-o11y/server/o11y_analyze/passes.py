"""LLM passes: `log` (events + missing ledger records) and `display` (fill dashboard gaps).

Each pass keeps its own transcript cursors (`llm-log:*`, `llm-display:*`), separate from the
deterministic `ingest:*` cursors, and advances them only after its outputs are written. A failed or
timed-out model call leaves the cursors where they were, so the slice is retried next time (bounded by
`max_slice_bytes`).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import core, llm
from .cursor import Cursor, CursorStore, _sha, read_new, tail_start
from .llm_schemas import DISPLAY_OUT, LOG_OUT
from .sources import transcripts as tr_src

identity = core.identity


def _ledger_tail(root: Path, n: int) -> list[dict]:
    recs = core.query_ledger(root)
    out = []
    for r in recs[-n:]:
        out.append({k: v for k, v in r.items() if k not in ("writer", "v", "id")})
    return out


def _line_uuid(path: str, offset: int) -> str | None:
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
                u = _line_uuid(self.path, o)
                if u:
                    item["uuid"] = u
                ev.append(item)
        return ev


def _end_cursor(path: str) -> Cursor:
    """A cursor at the end of the last complete line: the model pass resumes from here (design §8.2)."""
    st = os.stat(path)
    return Cursor(path=path, inode=st.st_ino, offset=tail_start(path, 0))


def collect_slices(
    root: Path, sessions, prefix: str, budget_chars: int, lookback_bytes: int
) -> list[Slice]:
    """New history of each session for one model pass. First sight reads only the last `lookback_bytes`;
    a history that was replaced or rewritten is not re-read: the cursor jumps to its end."""
    store = CursorStore(root)
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


def _sessions_block(sessions) -> str:
    return "\n".join(
        f"- {s.session_id} name={s.name or '-'} role={s.role} status={s.status}" for s in sessions
    )


def log_pass(root: Path, cfg: dict, ctx: dict, deadline: float) -> dict:
    lc = cfg["llm"]["log"]
    slices = collect_slices(
        root, ctx["fleet"], "llm-log", int(lc["max_slice_bytes"]), int(lc["lookback_bytes"])
    )
    report = {
        "slices": len(slices),
        "llm": None,
        "events": 0,
        "ledger_added": 0,
        "ledger_rejected": 0,
    }
    if not slices:
        return report
    recent = core.query_events(root)[-30:]
    prompt = "\n".join(
        [
            lc["prompt_text"],
            "\n## Sessions\n" + _sessions_block(ctx["fleet"]),
            "\n## Run ledger tail\n"
            + "\n".join(
                json.dumps(r, ensure_ascii=False)
                for r in _ledger_tail(root, int(lc["lookback_ledger_records"]))
            ),
            "\n## Already reported\n"
            + "\n".join(f"- [{e.get('severity_text')}] {e.get('body')}" for e in recent),
            "\n## New slices",
            *[
                f"\n### session {sl.session.session_id} ({sl.session.name or '-'}, {sl.session.role})\n{sl.text}"
                for sl in slices
            ],
        ]
    )
    timeout = max(30, min(float(lc["timeout_s"]), deadline - time.time()))
    res = llm.run(
        prompt,
        LOG_OUT,
        model=lc["model"],
        effort=lc.get("effort"),
        claude_bin=cfg["claude_bin"],
        timeout_s=timeout,
        disable_thinking=cfg["llm"].get("disable_thinking", True),
        cwd=str(_llm_cwd(root)),
    )
    report["llm"] = {
        "ok": res.ok,
        "cost_usd": res.cost_usd,
        "duration_s": round(res.duration_s, 1),
        "model": res.model,
        "error": res.error,
        "prompt_chars": len(prompt),
    }
    if not res.ok:
        core.append_event(
            root,
            severity="WARN",
            event_name="o11y.analyzer.llm_failed",
            body=f"log pass model call failed: {res.error}"[:300],
            attributes={},
        )
        return report
    by_sid = {sl.session.session_id: sl for sl in slices}
    for e in res.data.get("events", []):
        sl = by_sid.get(e.get("session_id")) or slices[0]
        s = sl.session
        attrs = {
            "session.id": s.session_id,
            "session.role": s.role,
            "evidence": sl.evidence(e.get("evidence_offsets")),
        }
        if s.name:
            attrs["session.name"] = s.name
        try:
            core.append_event(
                root,
                severity=e["severity"],
                event_name="o11y." + e["event_name"],
                body=e["body"],
                attributes=attrs,
                source="llm",
            )
            report["events"] += 1
        except core.ValidationError as err:
            report.setdefault("event_rejections", []).append(err.problems[:2])
    existing = {identity(r) for r in core.query_ledger(root)}
    for rec in res.data.get("ledger", []):
        sl = by_sid.get(rec.get("session_id")) or slices[0]
        body = {
            k: v
            for k, v in rec.items()
            if k not in ("session_id", "evidence_offsets", "writer", "ts", "v")
        }
        ident = identity(body)
        if ident and ident in existing:
            continue
        ev = sl.evidence(rec.get("evidence_offsets"))
        origin = [{k: v for k, v in x.items() if k in ("path", "offset", "uuid")} for x in ev] or [
            {"path": sl.path}
        ]
        try:
            core.append_ledger(
                root,
                body,
                session_id=sl.session.session_id,
                session_name=sl.session.name,
                origin=origin,
            )
            report["ledger_added"] += 1
            existing.add(ident)
        except core.ValidationError as err:
            report["ledger_rejected"] += 1
            report.setdefault("rejections", []).append(
                {"kind": body.get("kind"), "problems": err.problems[:3]}
            )
    store = CursorStore(root)
    for sl in slices:
        store.put(f"llm-log:{sl.path}", sl.cursor)
    return report


def display_pass(root: Path, cfg: dict, ctx: dict, deadline: float, need: bool) -> dict:
    """Cheap extraction of task/current run/plan from the MAIN session when the ledger lacks them."""
    dc = cfg["llm"]["display"]
    report = {"llm": None}
    mains = [s for s in ctx["fleet"] if s.role == "main"]
    if not need or not mains:
        return report
    slices = collect_slices(
        root, mains[:1], "llm-display", int(dc["max_slice_bytes"]), int(dc["lookback_bytes"])
    )
    if not slices:
        return report
    sl = slices[0]
    prompt = "\n".join(
        [
            dc["prompt_text"],
            "\n## Run ledger tail\n"
            + "\n".join(json.dumps(r, ensure_ascii=False) for r in _ledger_tail(root, 15)),
            "\n## Task book excerpt\n" + _task_book_excerpt(cfg),
            f"\n## Main session {sl.session.name or sl.session.session_id} (newest part)\n{sl.text}",
        ]
    )
    timeout = max(30, min(float(dc["timeout_s"]), deadline - time.time()))
    res = llm.run(
        prompt,
        DISPLAY_OUT,
        model=dc["model"],
        effort=dc.get("effort"),
        claude_bin=cfg["claude_bin"],
        timeout_s=timeout,
        disable_thinking=cfg["llm"].get("disable_thinking", True),
        cwd=str(_llm_cwd(root)),
    )
    report["llm"] = {
        "ok": res.ok,
        "cost_usd": res.cost_usd,
        "duration_s": round(res.duration_s, 1),
        "model": res.model,
        "error": res.error,
        "prompt_chars": len(prompt),
    }
    if not res.ok:
        return report
    out = dict(res.data)
    out["evidence"] = sl.evidence(out.pop("evidence_offsets", []))
    out["generated_at"] = core.now_iso()
    out["session_id"] = sl.session.session_id
    core.write_json_atomic(Path(root) / "state" / "analyze" / "display_extract.json", out)
    CursorStore(root).put(f"llm-display:{sl.path}", sl.cursor)
    return report


def _task_book_excerpt(cfg: dict, limit: int = 6000) -> str:
    p = cfg["sources"].get("task_book")
    try:
        text = Path(p).read_text(encoding="utf-8", errors="replace") if p else ""
    except OSError:
        text = ""
    return text[:limit] if text else "(none)"


def _llm_cwd(root: Path) -> Path:
    p = Path(root) / "state" / "analyze" / "llm_cwd"  # neutral dir: no project CLAUDE.md, no hooks
    p.mkdir(parents=True, exist_ok=True)
    return p
