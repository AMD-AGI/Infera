"""o11y_extract — the one model extractor (design §8.3): new slices of main and child histories + the ledger
tail → the ledger records the sessions did not write (task, run_start with machines, run_end, plan, campaign,
kernel, hotops), `writer.source=extracted`, deduplicated against the ledger, through the door.

Model output is untrusted (history text can carry instructions): a per-kind output schema with
`additionalProperties: false`, the door's full runs.v1 validation, and a filesystem check of campaign roots.
The model cursors advance only after the records are written; a failed call leaves them.
"""

from __future__ import annotations

import json
from pathlib import Path

from o11y_job import CursorStore, config, door, llm
from o11y_sources import forge as forge_src
from o11y_sources.slices import collect_slices, sessions_block

PROMPT = Path(__file__).resolve().parent / "prompts" / "extract.md"

_S = {"type": "string", "minLength": 1, "maxLength": 200}
_T = {"type": "string", "minLength": 1, "maxLength": 500}
_N = {"type": "number"}
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
END = [
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


def _obj(required: list[str], props: dict) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": props,
    }


_MACHINE = _obj(["node"], {"node": _S, "cluster": _S, "jobid": _S})


def _kind(name: str, required: list[str], props: dict) -> dict:
    return _obj(
        ["kind", "session_id", "evidence_offsets", *required],
        {
            "kind": {"const": name},
            "session_id": _S,
            "evidence_offsets": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "maxItems": 5,
            },
            **props,
        },
    )


LEDGER_ITEM = {
    "anyOf": [
        _kind(
            "task",
            ["task_id", "title", "metric", "baseline"],
            {
                "task_id": _S,
                "title": _T,
                "metric": _obj(
                    ["name", "unit", "direction"],
                    {"name": _S, "unit": _S, "direction": {"enum": ["higher", "lower"]}},
                ),
                "baseline": _obj(["value"], {"value": _N, "label": _S}),
                "target": _obj(["value"], {"value": _N, "rule": _S}),
            },
        ),
        _kind(
            "run_start",
            ["run_id", "theme", "point"],
            {
                "run_id": _S,
                "theme": _S,
                "point": {"enum": POINT},
                "hypothesis": _T,
                "machines": {"type": "array", "maxItems": 8, "items": _MACHINE},
            },
        ),
        _kind(
            "run_end",
            ["run_id", "status"],
            {
                "run_id": _S,
                "status": {"enum": END},
                "value": _N,
                "theme": _S,
                "conclusion": _T,
                "evidence": _obj(["path"], {"path": _S}),
            },
        ),
        _kind(
            "plan",
            ["next"],
            {
                "task_id": _S,
                "next": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": _obj(["theme"], {"theme": _S, "why": _T}),
                },
            },
        ),
        _kind(
            "campaign",
            ["campaign_root", "operator_ids", "started_at", "machine"],
            {
                "campaign_root": _S,
                "operator_ids": {"type": "array", "minItems": 1, "maxItems": 16, "items": _S},
                "started_at": _S,
                "machine": _MACHINE,
            },
        ),
        _kind(
            "kernel",
            ["op_id", "stage"],
            {
                "op_id": _S,
                "stage": {"enum": STAGE},
                "e2e": _obj(
                    ["verdict"],
                    {
                        "verdict": {"enum": ["accepted", "rejected", "inconclusive", "pending"]},
                        "delta_pct": _N,
                    },
                ),
            },
        ),
        _kind(
            "hotops",
            ["profile_id", "role", "ops"],
            {
                "profile_id": _S,
                "role": {"enum": ["prefill", "decode", "colocated", "verify", "extend", "mixed"]},
                "ops": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 30,
                    "items": _obj(
                        ["name", "share_pct"],
                        {"name": _S, "share_pct": {"type": "number", "minimum": 0, "maximum": 100}},
                    ),
                },
            },
        ),
    ]
}
OUT = _obj(["ledger"], {"ledger": {"type": "array", "maxItems": 12, "items": LEDGER_ITEM}})


def _same_plan(a: dict | None, b: dict) -> bool:
    """plan has no stable identity (task_id + time): an unchanged `next` list is the same plan."""
    return a is not None and [x.get("theme") for x in a.get("next") or []] == [
        x["theme"] for x in b["next"]
    ]


def handle(job) -> None:
    cfg, _, _ = config.load(job.root, job.state_dir)
    sessions, _ = config.fleet(job.root, cfg)
    store = CursorStore(job.state_dir)
    xc = cfg["extract"]
    slices = collect_slices(
        store,
        sessions,
        "extract",
        int(cfg["max_slice_bytes"]["extract"]),
        int(xc["lookback_bytes"]),
    )
    rep = job.report.setdefault("extract", {"slices": 0, "added": 0, "rejected": []})
    rep["slices"] += len(slices)
    if not slices:
        return
    ledger, _ = door.api.read_ledger(job.root)
    tail = [
        {k: v for k, v in r.items() if k not in ("writer", "v", "id")}
        for r in ledger[-cfg["ledger_tail"] :]
    ]
    prompt = "\n".join(
        [
            config.prompt_text(xc.get("prompt"), PROMPT),
            "\n## Sessions\n" + sessions_block(sessions),
            "\n## Run ledger tail\n" + "\n".join(json.dumps(r, ensure_ascii=False) for r in tail),
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
    known = {door.api.identity(r) for r in ledger}
    last_plan = next((r for r in reversed(ledger) if r["kind"] == "plan"), None)
    for item in data["ledger"]:
        sl = by_sid.get(item["session_id"]) or slices[0]
        rec = {k: v for k, v in item.items() if k not in ("session_id", "evidence_offsets")}
        if rec["kind"] == "campaign":
            why = forge_src.campaign_root_problem(rec["campaign_root"])
            if why:
                rep["rejected"].append({"kind": "campaign", "problems": [f"campaign_root {why}"]})
                continue
        ident = door.api.identity(rec)
        if (rec["kind"] != "plan" and ident in known) or (
            rec["kind"] == "plan" and _same_plan(last_plan, rec)
        ):
            continue
        origin = [
            {k: v for k, v in e.items() if k in ("path", "offset", "uuid")}
            for e in sl.evidence(item["evidence_offsets"])
        ] or [{"path": sl.path}]
        res = door.ledger_extracted(job.root, rec, sl.session, origin)
        if res.get("ok"):
            rep["added"] += 1
            known.add(ident)
            last_plan = rec if rec["kind"] == "plan" else last_plan
        else:
            rep["rejected"].append(
                {"kind": rec["kind"], "problems": (res.get("problems") or [])[:3]}
            )
    for sl in slices:
        store.put(f"extract:{sl.path}", sl.cursor)
