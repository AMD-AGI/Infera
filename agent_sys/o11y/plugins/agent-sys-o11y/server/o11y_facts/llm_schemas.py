"""Output contract between the display pass and the model (o11y_facts-private; not a protocol file)."""

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
