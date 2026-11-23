"""The one place the analysis jobs import the protocol layer from (design §2.1: the jobs reach the ledger and
the events only through the door's api). A rename of the protocol package changes this file only."""

from __future__ import annotations

import sys

from o11y_door import api  # noqa: F401  (re-exported for the jobs)
from o11y_protocol import fleet, schemas, store  # noqa: F401

ANALYZER_SID = "o11y-analyzer"  # writer.session_id for facts not tied to one observed session


def context(root, session_id: str | None = None, session_name: str | None = None) -> api.Context:
    return api.Context(
        root,
        session_id=session_id or ANALYZER_SID,
        role="analyzer",
        session_name=session_name or None,
    )


def event(
    root, severity: str, name: str, body: str, attrs: dict, sess=None, source: str = "detector"
) -> bool:
    """Append one event through the door; `sess` (a registry Session) names the observed session. A refusal is
    a stderr diagnostic (review C4), never an event about events."""
    a = {k: v for k, v in attrs.items() if v is not None}
    if sess is not None:
        a.update({"session.id": sess.session_id, "session.role": sess.role})
        if sess.name:
            a["session.name"] = sess.name
    ctx = context(root, a.get("session.id"), a.get("session.name"))
    res = api.event_report(
        ctx,
        severity,
        name,
        (body or "-")[:2000],
        a,
        source=source,
        writer_source="extracted" if source == "llm" else "detected",
    )
    if not res.get("ok"):
        print(f"o11y: event {name} refused: {res.get('problems')}", file=sys.stderr)
    return bool(res.get("ok"))


def ledger_extracted(root, record: dict, sess, origin: list) -> dict:
    """Append one model-extracted ledger record (writer.source=extracted) on behalf of `sess`."""
    rec = {**record, "writer": {"source": "extracted", **({"from": origin} if origin else {})}}
    return api.ledger_append(context(root, sess.session_id, sess.name), rec)
