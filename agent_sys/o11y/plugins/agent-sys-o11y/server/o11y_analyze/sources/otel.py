"""Optional: Claude Code native OTel log events from a local collector's JSONL (otel.md).

Accepted line shapes:
- flattened (vector remap): {"time", "event_name", "attributes": {...}, "resource": {...}}
- vector native decoding: {"message": "claude_code.<event>", "attributes": {...}, "timestamp"}
- OTel Collector file exporter (OTLP-JSON): {"resourceLogs": [{"scopeLogs": [{"logRecords": [...]}]}]}
Used events: api_request (tokens/cost per session.id), compaction, api_error, api_retries_exhausted.
"""

from __future__ import annotations

import json

from . import Problems

USED = {"api_request", "compaction", "api_error", "api_retries_exhausted"}


def _anyvalue(v: dict):
    if not isinstance(v, dict):
        return v
    for k in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if k in v:
            return int(v[k]) if k == "intValue" else v[k]
    if "arrayValue" in v:
        return [_anyvalue(x) for x in v["arrayValue"].get("values", [])]
    return None


def records(path: str, lines, problems: Problems) -> list[dict]:
    """Normalise to {event, attrs} for the events we use."""
    out = []
    for off, raw in lines:
        try:
            d = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(d, dict):
            continue
        batch = []
        if "resourceLogs" in d:
            for rl in d.get("resourceLogs") or []:
                for sl in rl.get("scopeLogs") or []:
                    for r in sl.get("logRecords") or []:
                        batch.append(
                            {
                                a["key"]: _anyvalue(a.get("value"))
                                for a in r.get("attributes") or []
                                if isinstance(a, dict) and "key" in a
                            }
                        )
        elif isinstance(d.get("attributes"), dict):
            batch.append(d["attributes"])
        else:
            problems.add("otel", path, "attributes", "unrecognised line shape")
            continue
        for a in batch:
            name = a.get("event.name")
            if name in USED:
                if "session.id" not in a:
                    problems.add("otel", path, "session.id", "missing required attribute")
                    continue
                out.append({"event": name, "attrs": a, "offset": off})
    return out
