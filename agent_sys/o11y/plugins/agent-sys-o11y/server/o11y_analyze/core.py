"""The analyzer's only door to protocol files: o11y_core (validate + flock append + atomic JSON)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from o11y_core import api, schemas, store

ANALYZER_SID = "o11y-analyzer"  # writer.session_id for fleet-level facts not tied to one session


class ValidationError(Exception):
    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def now_iso() -> str:
    return store.now()


def _ctx(root: Path, session_id: str | None, session_name: str | None = None) -> api.Context:
    return api.Context(
        Path(root),
        session_id=session_id or ANALYZER_SID,
        role="analyzer",
        session_name=session_name or None,
    )


def append_event(
    root: Path,
    *,
    severity: str,
    event_name: str,
    body: str,
    attributes: dict,
    source: str = "detector",
    evidence_origin: list | None = None,
) -> str:
    """source: detector|llm|otel. writer.source = detected (deterministic) or extracted (LLM)."""
    attrs = dict(attributes)
    attrs["o11y.source"] = source
    sid = attrs.get("session.id")
    ctx = _ctx(root, sid, attrs.get("session.name"))
    res = api.event_report(
        ctx,
        severity,
        event_name,
        body[:2000] or "-",
        attrs,
        source=source,
        origin=evidence_origin,
        writer_source="extracted" if source == "llm" else "detected",
    )
    if not res.get("ok"):
        raise ValidationError(res.get("problems") or ["event rejected"])
    return res["id"]


def append_ledger(
    root: Path, record: dict, *, session_id: str, session_name: str | None, origin: list | None
) -> str:
    rec = dict(record)
    rec["writer"] = {"source": "extracted", "from": origin or []}
    if not rec["writer"]["from"]:
        del rec["writer"]["from"]
    res = api.ledger_append(_ctx(root, session_id, session_name), rec)
    if not res.get("ok"):
        raise ValidationError(res.get("problems") or ["record rejected"])
    return res["id"]


def identity(rec: dict):
    """runs.v1 x-identity key (o11y_core's supersede rule)."""
    fn = getattr(api, "identity", None) or api._identity  # public name from core v1.1
    return fn(rec)


def query_ledger(root: Path, kind: str | None = None, latest_only: bool = True) -> list[dict]:
    recs, _ = api.read_ledger(Path(root), latest_only=latest_only)
    return [r for r in recs if kind is None or r.get("kind") == kind]


def ledger_invalid_lines(root: Path) -> int:
    return api.read_ledger(Path(root), latest_only=False)[1]


def query_events(root: Path) -> list[dict]:
    return [e for _, e, _ in store.read_jsonl(Path(root) / store.EVENTS) if isinstance(e, dict)]


def events_invalid_lines(root: Path) -> int:
    return sum(
        1 for _, e, _ in store.read_jsonl(Path(root) / store.EVENTS) if not isinstance(e, dict)
    )


def problems(schema_name: str, obj: Any) -> list[str]:
    return schemas.problems(schema_name, obj)


def write_json_atomic(path: Path | str, obj: Any) -> None:
    store.write_json(Path(path), obj)
