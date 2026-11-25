"""Fleet-level facts come from the main session only; a child's plan is its own; sessions declared
not_in_fleet are ignored. Replays the shape of the live dsv41 ledger of 2026-11-25 (ids shortened)."""

import json

import analysis_paths  # noqa: F401  (sys.path as o11y-run arranges)
import pytest
from o11y_door import api
from o11y_facts import handle as facts_handle
from o11y_job import run_consumer
from o11y_protocol import store

MAIN, MAIN_OLD, KF, FORK = "s-main", "s-main-0", "s-kf", "s-fork"


def _rec(kind, ts, sid, role, source="written", **fields):
    w = {"session_id": sid, "role": role, "source": source}
    return {"v": 1, "kind": kind, "id": f"rec-{ts}", "ts": ts, "writer": w, **fields}


def _task(task_id, ts, sid, role, source="written"):
    return _rec(
        "task", ts, sid, role, source, task_id=task_id, title=task_id,
        metric={"name": "tput", "unit": "tok/s", "direction": "higher"}, baseline={"value": 100},
    )  # fmt: skip


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    root = tmp_path / "fleet"
    monkeypatch.setenv("AGENT_SYS_O11Y_FLEET_DIR", str(root))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("AGENT_SYS_O11Y_OTEL_DIR", str(tmp_path / "otel"))
    monkeypatch.setenv("AGENT_SYS_HOME", str(tmp_path / "prefix"))
    monkeypatch.delenv("AGENT_SYS_O11Y", raising=False)
    rows = [
        {"session_id": MAIN, "pid": 1, "alive": True, "role": "main", "status": "idle", "active": True,
         "name": "main", "lineage": [MAIN_OLD]},
        {"session_id": KF, "pid": 2, "alive": True, "role": "child", "status": "busy", "name": "infera-kf",
         "parent_session_id": MAIN},
    ]  # fmt: skip
    store.write_json(
        root / store.SESSIONS,
        {"v": 1, "generated_at": store.now(), "main_session_id": MAIN, "sessions": rows},
    )
    for decision, sid, name in (("child", KF, "infera-kf"), ("not_in_fleet", FORK, None)):
        store.append(
            root / store.LAUNCHES,
            {"v": 1, "id": f"l-{sid}", "ts": store.now(), "decision": decision, "session_id": sid,
             "purpose": "replay", "parent_session_id": MAIN, **({"name": name} if name else {})},
        )  # fmt: skip
    ledger = [
        _task("fleet-task", "2026-11-25T08:49:55Z", MAIN, "main"),
        _rec(
            "best",
            "2026-11-25T08:51:38Z",
            MAIN,
            "main",
            task_id="fleet-task",
            value=489954,
            run_ids=["r1"],
        ),
        _rec(
            "plan",
            "2026-11-25T08:51:52Z",
            MAIN,
            "main",
            next=[{"theme": "fleet: vLLM PD baseline"}],
        ),
        _rec(
            "plan",
            "2026-11-25T09:06:31Z",
            FORK,
            "analyzer",
            "extracted",
            next=[{"theme": "fork: cron jobs"}],
        ),
        _rec("plan", "2026-11-25T09:14:19Z", KF, "child", next=[{"theme": "kf: Forge round 3"}]),
        _task("kf-task", "2026-11-25T09:15:00Z", KF, "child"),
        _rec(
            "best",
            "2026-11-25T09:16:00Z",
            KF,
            "child",
            task_id="kf-task",
            value=1.7,
            run_ids=["kf1"],
        ),
    ]
    for r in ledger:
        store.append(root / store.LEDGER, r)
    return root


def _facts(root):
    assert run_consumer("facts", facts_handle, root=root, force=True)["passes"] == 1
    return {k: store.read_json(root / store.FACTS[k]) for k in ("general", "detail")}


def test_fleet_plan_task_and_best_come_from_the_main(fleet):
    f = _facts(fleet)
    assert [p["theme"] for p in f["detail"]["next"]] == ["fleet: vLLM PD baseline"]
    assert f["general"]["task"]["task_id"] == "fleet-task"
    assert f["general"]["best"]["value"] == 489954
    # the child's plan is its own, shown under its session; the not_in_fleet fork's plan nowhere
    kids = [
        (c["session_id"], c["name"], [p["theme"] for p in c["next"]])
        for c in f["detail"]["child_next"]
    ]
    assert kids == [(KF, "infera-kf", ["kf: Forge round 3"])]
    assert "fork: cron jobs" not in json.dumps(f)


def test_written_beats_extracted_and_a_cleared_main_is_the_main(fleet):
    late = {
        **_rec(
            "plan", "2026-11-25T10:20:00Z", MAIN, "analyzer", "extracted", next=[{"theme": "late"}]
        ),
        "writer": {
            "session_id": MAIN,
            "role": "analyzer",
            "source": "extracted",
            "session_role": "main",
        },
    }
    store.append(
        fleet / store.LEDGER, late
    )  # a later extraction never displaces what the main wrote
    assert _facts(fleet)["detail"]["next"][0]["theme"] == "fleet: vLLM PD baseline"
    store.append(
        fleet / store.LEDGER,
        _rec(
            "plan", "2026-11-25T10:14:46Z", MAIN, "main", next=[{"theme": "fleet: written later"}]
        ),
    )
    assert _facts(fleet)["detail"]["next"][0]["theme"] == "fleet: written later"
    rows = api.read_registry(fleet)[
        0
    ]  # an older ledger: extracted from the main before /clear, no session_role
    old = _rec(
        "plan", "2026-11-25T08:00:00Z", MAIN_OLD, "analyzer", "extracted", next=[{"theme": "o"}]
    )
    assert api.record_role(old, rows) == "main"
    assert api.record_role(_rec("plan", "t", FORK, "analyzer", "extracted"), rows) is None


def test_the_door_refuses_a_session_declared_not_in_fleet(fleet):
    res = api.ledger_append(
        api.Context(root=fleet, session_id=FORK, role="child"),
        {"kind": "plan", "next": [{"theme": "x"}]},
    )
    assert not res["ok"] and "not_in_fleet" in res["problems"][0]
    ok = api.ledger_append(
        api.Context(root=fleet, session_id=KF), {"kind": "plan", "next": [{"theme": "y"}]}
    )
    assert ok["ok"]


def test_extracted_records_carry_the_observed_sessions_role(fleet):
    from o11y_job import door

    class Sess:
        session_id, name, role = KF, "infera-kf", "child"

    res = door.ledger_extracted(fleet, {"kind": "plan", "next": [{"theme": "z"}]}, Sess(), [])
    assert res["ok"]
    rec = [r for r in api.read_ledger(fleet)[0] if r["id"] == res["id"]][0]
    assert rec["writer"]["role"] == "analyzer" and rec["writer"]["session_role"] == "child"


def test_log_ignores_records_of_a_session_declared_not_in_fleet(fleet):
    from o11y_log import handle as log_handle

    run_consumer(
        "log", log_handle, root=fleet, force=True
    )  # first pass: what exists is the baseline
    for sid, role in ((FORK, "child"), (KF, "child")):
        store.append(
            fleet / store.LEDGER,
            _rec("finding", f"2026-11-25T09:3{len(sid)}:00Z", sid, role, finding_id=f"f-{sid}",
                 text=f"finding by {sid}", evidence_rank="first_hand"),
        )  # fmt: skip
    run_consumer("log", log_handle, root=fleet, force=True)
    bodies = [e["body"] for _, e, _ in store.read_jsonl(fleet / store.EVENTS) if e]
    assert any(f"finding by {KF}" in b for b in bodies)
    assert not any(f"finding by {FORK}" in b for b in bodies)
