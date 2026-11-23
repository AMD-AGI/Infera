"""The log model pass: new slices of main and child histories + the ledger tail → task-level events with
evidence and missing ledger records (`writer.source=extracted`), written through o11y_core (design §8.5).

Its model cursors (`llm-log:*`) live in `state/log/cursors` and advance only after the outputs are written;
a failed call leaves them, so the same slice is retried (bounded by `max_slice_bytes`). Model output is
untrusted: per-kind output schema, o11y_core validation, and a filesystem check of extracted campaign roots.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from o11y_sources import forge as forge_src
from o11y_sources import model
from o11y_sources.config import prompt_text
from o11y_sources.cursor import CursorStore
from o11y_sources.slices import collect_slices, sessions_block

from . import core
from .llm_schemas import LOG_OUT

PROMPT = Path(__file__).resolve().parent / "prompts" / "log.md"


def _ledger_tail(root: Path, n: int) -> list[dict]:
    return [
        {k: v for k, v in r.items() if k not in ("writer", "v", "id")}
        for r in core.query_ledger(root)[-n:]
    ]


def campaign_allowlist(root: Path, cfg: dict) -> list[str]:
    """Where extracted campaign roots may point: configured campaign globs, and roots that sessions
    themselves (`writer.source == written`) recorded in the ledger."""
    allowed = list(cfg["sources"].get("campaign_globs") or [])
    for r in core.query_ledger(root, kind="campaign"):
        if (r.get("writer") or {}).get("source") == "written" and r.get("campaign_root"):
            allowed.append(r["campaign_root"])
    return allowed


def log_pass(root: Path, cfg: dict, state_dir: Path, fleet, deadline: float) -> dict:
    lc = cfg["llm"]["log"]
    store = CursorStore(state_dir)
    slices = collect_slices(
        store, fleet, "llm-log", int(lc["max_slice_bytes"]), int(lc["lookback_bytes"])
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
            prompt_text(lc["prompt"], PROMPT),
            "\n## Sessions\n" + sessions_block(fleet),
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
    cwd = state_dir / "model_cwd"  # neutral dir: no project CLAUDE.md
    cwd.mkdir(parents=True, exist_ok=True, mode=0o700)
    res = model.run(
        prompt,
        LOG_OUT,
        model=lc["model"],
        effort=lc.get("effort"),
        claude_bin=cfg["claude_bin"],
        timeout_s=max(30, min(float(lc["timeout_s"]), deadline - time.time())),
        disable_thinking=cfg["llm"].get("disable_thinking", True),
        cwd=str(cwd),
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
            attributes={"o11y.dedup_key": f"llm_failed:{time.time():.0f}"},
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
            "o11y.dedup_key": f"llm:{s.session_id}:{e['event_name']}:{sl.lo}:{sl.hi}:{e['body'][:60]}",
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
    existing = {core.identity(r) for r in core.query_ledger(root)}
    allow = None
    for rec in res.data.get("ledger", []):
        sl = by_sid.get(rec.get("session_id")) or slices[0]
        body = {k: v for k, v in rec.items() if k not in ("session_id", "evidence_offsets")}
        if body.get("kind") == "campaign":
            allow = campaign_allowlist(root, cfg) if allow is None else allow
            why = forge_src.campaign_root_problem(body.get("campaign_root", ""), allow)
            if why:
                report["ledger_rejected"] += 1
                report.setdefault("rejections", []).append(
                    {"kind": "campaign", "problems": [f"campaign_root {why}"]}
                )
                continue
        ident = core.identity(body)
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
    for sl in slices:
        store.put(f"llm-log:{sl.path}", sl.cursor)
    return report
