"""Output contracts between the analyzer and the model (A-private; not protocol files).
Kept flat and permissive: protocol validation happens in o11y_core when records are appended."""

LOG_OUT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["events", "ledger"],
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
                    "event_name": {
                        "type": "string",
                        "pattern": "^(run|task|incident|machine|campaign|session)\\.[a-z][a-z0-9_]{1,30}$",
                    },
                    "body": {"type": "string", "maxLength": 300},
                    "session_id": {"type": "string"},
                    "evidence_offsets": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "maxItems": 5,
                    },
                },
            },
        },
        "ledger": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "required": ["kind", "session_id", "evidence_offsets"],
                "properties": {
                    "kind": {
                        "enum": [
                            "run_start",
                            "run_end",
                            "plan",
                            "campaign",
                            "kernel",
                            "hotops",
                            "incident",
                        ]
                    },
                    "session_id": {"type": "string"},
                    "evidence_offsets": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "maxItems": 5,
                    },
                },
            },
        },
    },
}

_S = {"type": ["string", "null"], "maxLength": 120}
_N = {"type": ["number", "null"]}

DISPLAY_OUT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["task", "current_run", "recent_runs", "next_plan", "evidence_offsets"],
    "properties": {
        "task": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "properties": {
                "title": _S,
                "metric_name": _S,
                "unit": _S,
                "direction": {"enum": ["higher", "lower", None]},
                "baseline": _N,
                "target": _N,
            },
        },
        "current_run": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "properties": {
                "theme": _S,
                "point": _S,
                "machines": {"type": "array", "items": {"type": "string"}},
                "started_at": _S,
            },
        },
        "recent_runs": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"run_id": _S, "theme": _S, "result": _S, "status": _S},
            },
        },
        "next_plan": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"theme": _S, "why": _S},
            },
        },
        "evidence_offsets": {"type": "array", "items": {"type": "integer"}, "maxItems": 8},
    },
}
