import copy
import json
import multiprocessing as mp
import subprocess
import sys

import pytest
from door_paths import EXAMPLES, PY
from o11y_door import api
from o11y_protocol import schemas, store


def _examples():
    out = []
    for p in sorted(EXAMPLES.glob("*.jsonl")):
        for n, line in enumerate(p.read_text().splitlines(), 1):
            out.append(pytest.param(json.loads(line), id=f"{p.stem}:{n}"))
    return out


EXAMPLES_LIST = _examples()


def test_examples_present():
    assert (
        len(EXAMPLES_LIST) == 38
    )  # 42 converted, 4 of kind machine dropped with the kind (review C3)


@pytest.mark.parametrize("rec", EXAMPLES_LIST)
def test_real_history_examples_validate(rec):
    assert schemas.problems("runs.v1", rec) == []


def _first(kind):
    return copy.deepcopy(next(p.values[0] for p in EXAMPLES_LIST if p.values[0]["kind"] == kind))


MUTANTS = [
    ("task", lambda r: r.pop("baseline"), "missing required field 'baseline'"),
    ("task", lambda r: r["metric"].update(direction="up"), "is not one of"),
    ("run_start", lambda r: r.pop("point"), "missing required field 'point'"),
    ("run_end", lambda r: r.update(status="kept", value=None), "must be of type number"),
    (
        "run_end",
        lambda r: [r.update(status="failed"), r.pop("failure", None), r.pop("conclusion", None)],
        "needs conclusion or failure",
    ),
    ("run_end", lambda r: r.update(status="great"), "is not one of"),
    ("run_end", lambda r: r["writer"].pop("role"), "missing required field 'role'"),
    ("run_end", lambda r: r["writer"].update(source="guessed"), "is not one of"),
    ("run_end", lambda r: r.update(ts="yesterday"), "is not an RFC 3339 time"),
    ("best", lambda r: r.update(run_ids=[]), "should be non-empty"),
    ("plan", lambda r: r.update(next=[]), "should be non-empty"),
    ("campaign", lambda r: r.update(campaign_root="relative/path"), "does not match"),
    ("hotops", lambda r: r["ops"][0].update(share_pct=130), "greater than the maximum"),
    ("kernel", lambda r: r.update(stage="almost"), "is not one of"),
    ("incident", lambda r: r.pop("severity"), "missing required field 'severity'"),
    ("idea", lambda r: r.update(status="maybe"), "is not one of"),
    ("decision", lambda r: r.update(by="someone"), "is not one of"),
]


@pytest.mark.parametrize(
    "kind,mutate,expect", MUTANTS, ids=[m[2][:30] + "|" + m[0] for m in MUTANTS]
)
def test_mutants_rejected_with_one_line_hint(kind, mutate, expect):
    rec = _first(kind)
    mutate(rec)
    probs = schemas.problems("runs.v1", rec)
    assert probs, "mutant accepted"
    assert any(expect in p for p in probs), probs
    assert all("\n" not in p and len(p) < 400 for p in probs), probs  # one line, no instance dump


def test_ledger_append_sets_server_fields_and_keeps_extracted_source(ctx):
    rec = _first("run_end")
    rec["v"], rec["ts"], rec["id"] = 99, "bogus", "mine"
    res = api.ledger_append(ctx, rec)
    assert res["ok"], res
    stored = api.ledger_query(ctx, kind="run_end")["records"][-1]
    assert stored["v"] == 1 and stored["id"] == res["id"] and stored["ts"].endswith("Z")
    assert stored["writer"]["session_id"] == "sess-test-0001" and stored["writer"]["role"] == "main"
    assert stored["writer"]["source"] == "extracted"  # caller's source and from are kept
    assert stored["writer"]["from"] == rec["writer"]["from"]


def test_ledger_append_rejects_and_writes_nothing(ctx, root):
    res = api.ledger_append(ctx, {"kind": "run_end", "run_id": "r1", "status": "failed"})
    assert not res["ok"] and res["problems"] == [
        "kind=run_end status=failed: (record): needs conclusion or failure"
    ]
    assert not (root / store.LEDGER).exists()


def test_supersede_by_identity(ctx):
    base = {"kind": "run_end", "run_id": "r1", "theme": "t"}
    assert api.ledger_append(
        ctx, {**base, "status": "kept", "value": 1, "evidence": {"path": "a"}}
    )["ok"]
    assert api.ledger_append(
        ctx, {"kind": "run_end", "run_id": "r2", "status": "failed", "conclusion": "oom"}
    )["ok"]
    assert api.ledger_append(
        ctx, {**base, "status": "kept", "value": 2, "evidence": {"path": "b"}}
    )["ok"]
    latest = api.ledger_query(ctx, kind="run_end")["records"]
    assert [(r["run_id"], r.get("value")) for r in latest] == [("r2", None), ("r1", 2)]
    assert len(api.ledger_query(ctx, kind="run_end", latest_only=False)["records"]) == 3


def test_query_skips_invalid_and_partial_lines(ctx, root):
    assert api.ledger_append(ctx, _first("task"))["ok"]
    with open(root / store.LEDGER, "ab") as f:
        f.write(b'{"kind":"task"}\nnot json\n{"kind":"run_end","partial":')
    res = api.ledger_query(ctx)
    assert res["count"] == 1 and res["invalid_lines"] == 2
    # the next append starts on a fresh line, so the partial line does not corrupt it
    assert api.ledger_append(ctx, _first("plan"))["ok"]
    assert api.ledger_query(ctx)["count"] == 2


def test_line_cap(ctx):
    rec = _first("finding")
    rec["text"] = "x" * (70 * 1024)
    res = api.ledger_append(ctx, rec)
    assert not res["ok"] and "limit" in res["problems"][0]


def _worker(args):
    root, n, wid = args
    from o11y_door import api as a

    c = a.Context(root=root, session_id=f"w{wid}", role="analyzer")
    for i in range(n):
        r = a.ledger_append(
            c,
            {
                "kind": "finding",
                "finding_id": f"f-{wid}-{i}",
                "text": "y" * 4000,
                "evidence_rank": "inference",
            },
        )
        assert r["ok"], r
    return n


def _paths(parent_path):
    sys.path[:0] = [p for p in parent_path if p not in sys.path]


def test_concurrent_appends_are_whole_lines(root):
    with mp.get_context("spawn").Pool(4, initializer=_paths, initargs=(sys.path[:],)) as pool:
        assert sum(pool.map(_worker, [(str(root), 150, w) for w in range(4)])) == 600
    lines = (root / store.LEDGER).read_bytes().split(b"\n")
    assert lines[-1] == b"" and len(lines) == 601
    ids = {json.loads(line)["finding_id"] for line in lines[:-1]}
    assert len(ids) == 600


def test_event_report_catalogue(ctx, root):
    assert api.event_report(ctx, "warning", "o11y.machine.lost", "node lost", {"node": "node-01"})[
        "ok"
    ]
    assert not api.event_report(ctx, "ERROR", "o11y.made.up", "x")[
        "ok"
    ]  # not in protocol/events.json
    wrong_level = api.event_report(ctx, "INFO", "o11y.session.crashed", "x")
    assert not wrong_level["ok"] and "not reported at INFO" in wrong_level["problems"][0]
    assert api.event_report(ctx, "ERROR", "o11y.incident.reported", "x")["ok"]  # one of its levels
    ev = json.loads((root / store.EVENTS).read_text().splitlines()[0])
    assert ev["severity_text"] == "WARN" and ev["severity_number"] == 13
    assert ev["attributes"]["o11y.source"] == "detector" and ev["writer"]["source"] == "detected"
    assert schemas.problems("event.v1", ev) == []


def test_session_register_child_env(ctx, root):
    res = api.session_register(ctx, "child", name="proj-kf", purpose="KernelForge")
    assert res["env"] == {
        "AGENT_SYS_O11Y": "child",
        "AGENT_SYS_O11Y_FLEET_DIR": str(root),
        "AGENT_SYS_O11Y_PARENT_SESSION_ID": "sess-test-0001",
    }
    assert "claude --name proj-kf" in res["launch_example"]
    assert "-e AGENT_SYS_O11Y=child" in res["launch_example"]
    assert not api.session_register(ctx, "main")[
        "ok"
    ]  # the trigger registers the main, not the door
    no_key = api.session_register(ctx, "child", purpose="KF")  # neither name nor tmux target
    assert not no_key["ok"] and "needs name or tmux_target" in no_key["problems"][0]
    no_purpose = api.session_register(ctx, "not_in_fleet", session_id="s-x")
    assert not no_purpose["ok"] and "purpose" in no_purpose["problems"][0]
    recs = api.launches(root)
    assert [r["decision"] for r in recs] == ["child"] and recs[0][
        "parent_session_id"
    ] == "sess-test-0001"
    assert res["argv"][:4] == ["tmux", "new-session", "-d", "-s"] and res["argv"][-3:] == [
        "claude",
        "--name",
        "proj-kf",
    ]


def test_config_defaults_invalid_and_last_valid_kept(root):
    cfg, probs, new = api.load_config(root, "trigger")
    assert not probs and cfg["kick"]["interval_s"] == 1800 and cfg["packup"]["every_n_kicks"] == 3
    assert "{children}" in cfg["kick"]["template"]
    store.write_json(root / store.CONFIG["trigger"], {"v": 1, "kick": {"interval_s": 120}})
    cfg, probs, new = api.load_config(root, "trigger")
    assert not probs and cfg["kick"]["interval_s"] == 120
    store.write_json(root / store.CONFIG["trigger"], {"v": 1, "kick": {"intervall_s": 60}})
    cfg, probs, new = api.load_config(root, "trigger")
    assert new and "unknown key 'intervall_s' (did you mean 'interval_s'?)" in probs[0]
    assert cfg["kick"]["interval_s"] == 120  # the last valid document stays in effect (review F5)
    cfg, probs, new = api.load_config(root, "trigger")
    assert probs and not new  # the same invalid document is reported once
    store.write_json(
        root / store.CONFIG["trigger"], {"v": 1, "max_pending": 2}
    )  # a key that became a constant
    _, probs, new = api.load_config(root, "trigger")
    assert new and "unknown key 'max_pending'" in probs[0]
    acfg, _, _ = api.load_config(root, "analysis")
    assert set(acfg) == {"v", "log", "extract", "sources"}
    assert acfg["log"]["lookback_bytes"] == 800000 and acfg["extract"]["lookback_bytes"] == 800000


def test_fleet_status_reads_only_the_registry(ctx, root):
    api.ledger_append(ctx, _first("task"))
    api.ledger_append(
        ctx, {"kind": "run_start", "run_id": "r9", "theme": "decode mem 0.95", "point": "explore"}
    )
    api.event_report(ctx, "ERROR", "o11y.session.crashed", "child gone")
    st = api.fleet_status(ctx)
    assert st["task"]["task_id"] == "dsv41-pd" and st["open_runs"][0]["run_id"] == "r9"
    assert st["recent_alerts"][0]["severity"] == "ERROR"
    assert (
        st["sessions"] == [] and "absent" in st["registry_problem"]
    )  # no recomputation (review A4)
    store.write_json(
        root / store.SESSIONS,
        {
            "v": 1,
            "generated_at": store.now(),
            "sessions": [
                {
                    "session_id": "s-main",
                    "pid": 1,
                    "alive": True,
                    "role": "main",
                    "status": "idle",
                    "active": True,
                }
            ],
        },
    )
    st = api.fleet_status(ctx)
    assert [s["session_id"] for s in st["sessions"]] == ["s-main"] and st[
        "registry_problem"
    ] is None


def test_cli_exit_codes(root):
    run = lambda *a, **kw: subprocess.run(  # noqa: E731
        [str(PY), "-m", "o11y_door", *a], capture_output=True, text=True, **kw
    )
    ok = run("ledger", "append", json.dumps({"kind": "plan", "next": [{"theme": "x"}]}))
    assert ok.returncode == 0, ok.stderr
    bad = run("ledger", "append", "--file", "-", input=json.dumps({"kind": "plan", "next": []}))
    assert bad.returncode == 1 and "non-empty" in bad.stdout
    vf = run("validate-file", str(root / store.LEDGER))
    assert vf.returncode == 0 and json.loads(vf.stdout)["schema"] == "runs.v1"


def test_unknown_field_rejected_with_suggestion(ctx):
    res = api.ledger_append(
        ctx,
        {"kind": "run_start", "run_id": "r", "theme": "t", "point": "explore", "node": "m2m-135"},
    )
    assert not res["ok"]
    assert res["problems"] == [
        "kind=run_start: (record): unknown field(s) node for this kind; did you mean node -> machines[].node?"
    ]
