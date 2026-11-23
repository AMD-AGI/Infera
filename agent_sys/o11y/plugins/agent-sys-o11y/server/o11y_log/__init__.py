"""o11y_log — events only (design §8.3, §8.4): the deterministic detectors, then one model pass over new
history slices for task-level meaning (runs, conclusions, services, blocks, incidents, files removed by
someone else). The model pass never writes ledger records (that is o11y_extract's). Model output is untrusted:
catalogue names only, evidence offsets checked against the slice, and `files.removed_externally` gated by
code (the path is gone now, and the fleet's own commands in the slice do not remove or move it).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from o11y_job import CursorStore, config, door, llm
from o11y_sources.slices import collect_slices, sessions_block

from .detectors import Detectors

PROMPT = Path(__file__).resolve().parent / "prompts" / "log.md"
NAMES = [
    "run.started",
    "run.finished",
    "task.conclusion",
    "task.service",
    "task.blocked",
    "incident.reported",
    "files.removed_externally",
]
_OFFSETS = {"type": "array", "items": {"type": "integer", "minimum": 0}, "maxItems": 5}
OUT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["events"],
    "properties": {
        "events": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["severity", "event_name", "body", "session_id", "evidence_offsets"],
                "properties": {
                    "severity": {"enum": ["INFO", "WARN", "ERROR"]},
                    "event_name": {"enum": NAMES},
                    "body": {"type": "string", "minLength": 1, "maxLength": 300},
                    "session_id": {"type": "string", "minLength": 1, "maxLength": 200},
                    "evidence_offsets": _OFFSETS,
                    "path": {"type": "string", "minLength": 2, "maxLength": 500},
                },
            },
        }
    },
}
_REMOVE = re.compile(r"\b(rm|rmdir|unlink|mv|shred|truncate|git\s+(rm|mv|clean))\b")


def removal_problem(path: str | None, slice_text: str) -> str | None:
    """Why a model-reported `files.removed_externally` must not be written (None = keep it)."""
    if not path or not os.path.isabs(path):
        return "no absolute path"
    if os.path.lexists(path):
        return "the path exists"
    for line in slice_text.splitlines():
        if " TOOL " in line and path in line and _REMOVE.search(line):
            return "the fleet's own command removes or moves it"
    return None


def model_pass(job, cfg: dict, sessions: list) -> None:
    store = CursorStore(job.state_dir)
    lc = cfg["log"]
    slices = collect_slices(
        store, sessions, "log", int(cfg["max_slice_bytes"]["log"]), int(lc["lookback_bytes"])
    )
    rep = job.report.setdefault("log_pass", {"slices": 0, "events": 0, "rejected": []})
    rep["slices"] += len(slices)
    if not slices:
        return
    recent = [
        e for _, e, _ in door.store.read_jsonl(job.root / door.store.EVENTS) if isinstance(e, dict)
    ][-30:]
    prompt = "\n".join(
        [
            config.prompt_text(lc.get("prompt"), PROMPT),
            "\n## Sessions\n" + sessions_block(sessions),
            "\n## Already reported\n"
            + "\n".join(f"- [{e.get('severity_text')}] {e.get('body')}" for e in recent),
            "\n## New slices",
            *[
                f"\n### session {sl.session.session_id} ({sl.session.name or '-'}, {sl.session.role})\n{sl.text}"
                for sl in slices
            ],
        ]
    )
    data = llm.ask(job, cfg, prompt, OUT)
    if data is None:
        return
    by_sid = {sl.session.session_id: sl for sl in slices}
    for e in data["events"]:
        sl = by_sid.get(e["session_id"]) or slices[0]
        ev = sl.evidence(e["evidence_offsets"])
        attrs = {
            "evidence": ev,
            "o11y.dedup_key": f"llm:{sl.session.session_id}:{e['event_name']}:{sl.lo}:{sl.hi}:{e['body'][:60]}",
        }
        sev, name = e["severity"], e["event_name"]
        if name == "files.removed_externally":
            why = "no evidence in the slice" if not ev else removal_problem(e.get("path"), sl.text)
            if why:
                rep["rejected"].append({"event_name": name, "problem": why})
                continue
            sev, attrs["o11y.path"] = "ERROR", e["path"]
        cat = door.schemas.events()["o11y." + name]
        sev = sev if sev in cat.get("levels", [cat["level"]]) else cat["level"]
        rep["events"] += door.event(
            job.root, sev, "o11y." + name, e["body"], attrs, sl.session, source="llm"
        )
    for sl in slices:
        store.put(f"log:{sl.path}", sl.cursor)


def handle(job) -> None:
    cfg, problems, new = config.load(job.root, job.state_dir)
    sessions, reg_problem = config.fleet(job.root, cfg)
    det = Detectors(job.root, cfg, job.state_dir, sessions)
    det.run(reg_problem, problems, new)
    job.report["detector_events"] = job.report.get("detector_events", 0) + det.emitted
    det.commit()
    mains = [s for s in sessions if s.role == "main"]
    if mains and any(s.alive for s in mains):
        model_pass(job, cfg, [s for s in sessions if s.alive])
