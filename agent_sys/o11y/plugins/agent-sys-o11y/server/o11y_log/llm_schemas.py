"""Output contract between the log pass and the model (o11y_log-private; not a protocol file).

Model output is untrusted (history text can carry instructions), so every ledger item has a per-kind shape
with `additionalProperties: false`; o11y_core's full `runs.v1` validation is the second gate, and campaign
roots get a filesystem check before they are written (passes.py).
"""

_STR = {"type": "string", "minLength": 1, "maxLength": 200}
_TXT = {"type": "string", "minLength": 1, "maxLength": 500}
_OFFSETS = {"type": "array", "items": {"type": "integer", "minimum": 0}, "maxItems": 5}
POINT = [
    "explore",
    "validate",
    "replicate",
    "baseline",
    "repro",
    "screen",
    "diagnostic",
    "calibration",
    "profile",
]
STAGE = [
    "hot",
    "selected",
    "priced",
    "workset",
    "optimizing",
    "candidate",
    "handed_off",
    "e2e_testing",
    "e2e_accepted",
    "e2e_rejected",
    "e2e_inconclusive",
    "dropped",
]
INCIDENT_CATEGORY = [
    "preemption",
    "machine_lost",
    "access_lost",
    "crash",
    "hang",
    "oom",
    "session_stall",
    "permission_wait",
    "api_error",
    "infra",
    "config_bug",
    "correctness",
    "data_loss",
    "own_mistake",
    "other",
]


def _kind(name: str, required: list[str], props: dict) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind", "session_id", "evidence_offsets", *required],
        "properties": {
            "kind": {"const": name},
            "session_id": _STR,
            "evidence_offsets": _OFFSETS,
            **props,
        },
    }


_MACHINES = {
    "type": "array",
    "maxItems": 8,
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["node"],
        "properties": {"node": _STR},
    },
}

LEDGER_ITEM = {
    "anyOf": [
        _kind(
            "run_start",
            ["run_id", "theme", "point"],
            {
                "run_id": _STR,
                "theme": _STR,
                "point": {"enum": POINT},
                "hypothesis": _TXT,
                "machines": _MACHINES,
            },
        ),
        _kind(
            "run_end",
            ["run_id", "status"],
            {
                "run_id": _STR,
                "status": {
                    "enum": [
                        "kept",
                        "neutral",
                        "rejected",
                        "failed",
                        "aborted",
                        "cancelled",
                        "unscored",
                        "inconclusive",
                        "done",
                    ]
                },
                "value": {"type": "number"},
                "theme": _STR,
                "conclusion": _TXT,
                "evidence": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["path"],
                    "properties": {"path": _STR},
                },
            },
        ),
        _kind(
            "plan",
            ["next"],
            {
                "task_id": _STR,
                "next": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["theme"],
                        "properties": {"theme": _STR, "why": _TXT},
                    },
                },
            },
        ),
        _kind(
            "campaign",
            ["campaign_root", "operator_ids", "started_at", "machine"],
            {
                "campaign_root": _STR,
                "operator_ids": {"type": "array", "minItems": 1, "maxItems": 16, "items": _STR},
                "started_at": _STR,
                "machine": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["node"],
                    "properties": {"node": _STR},
                },
            },
        ),
        _kind(
            "kernel",
            ["op_id", "stage"],
            {
                "op_id": _STR,
                "stage": {"enum": STAGE},
                "e2e": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["verdict"],
                    "properties": {
                        "verdict": {"enum": ["accepted", "rejected", "inconclusive", "pending"]},
                        "delta_pct": {"type": "number"},
                    },
                },
            },
        ),
        _kind(
            "hotops",
            ["profile_id", "role", "ops"],
            {
                "profile_id": _STR,
                "role": _STR,
                "ops": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 30,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["name", "share_pct"],
                        "properties": {
                            "name": _STR,
                            "share_pct": {"type": "number", "minimum": 0, "maximum": 100},
                        },
                    },
                },
            },
        ),
        _kind(
            "incident",
            ["incident_id", "severity", "category", "summary"],
            {
                "incident_id": _STR,
                "severity": {"enum": ["info", "warning", "error"]},
                "category": {"enum": INCIDENT_CATEGORY},
                "summary": _TXT,
            },
        ),
    ]
}

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
                    "body": {"type": "string", "minLength": 1, "maxLength": 300},
                    "session_id": _STR,
                    "evidence_offsets": _OFFSETS,
                },
            },
        },
        "ledger": {"type": "array", "maxItems": 12, "items": LEDGER_ITEM},
    },
}
