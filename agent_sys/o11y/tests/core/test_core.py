import copy
import json
import multiprocessing as mp
import os
import subprocess
import sys

import pytest
from core_paths import EXAMPLES, SERVER
from o11y_core import api, claude_sessions, schemas, store


def _examples():
    out = []
    for p in sorted(EXAMPLES.glob("*.jsonl")):
        for n, line in enumerate(p.read_text().splitlines(), 1):
            out.append(pytest.param(json.loads(line), id=f"{p.stem}:{n}"))
    return out


EXAMPLES_LIST = _examples()


def test_examples_present():
    assert len(EXAMPLES_LIST) == 42


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
    ("machine", lambda r: r.pop("event"), "missing required field 'event'"),
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
    from o11y_core import api as a

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


def test_concurrent_appends_are_whole_lines(root):
    with mp.get_context("spawn").Pool(
        4, initializer=sys.path.insert, initargs=(0, str(SERVER))
    ) as pool:
        assert sum(pool.map(_worker, [(str(root), 150, w) for w in range(4)])) == 600
    lines = (root / store.LEDGER).read_bytes().split(b"\n")
    assert lines[-1] == b"" and len(lines) == 601
    ids = {json.loads(line)["finding_id"] for line in lines[:-1]}
    assert len(ids) == 600


def test_event_report(ctx, root):
    assert api.event_report(ctx, "warning", "incident.reported", "node lost", {"node": "m2m-135"})[
        "ok"
    ]
    bad = api.event_report(ctx, "fatal", "o11y.incident.reported", "x")
    assert not bad["ok"] and any("severity_text" in p for p in bad["problems"])
    bad2 = api.event_report(ctx, "INFO", "Bad Name", "x")
    assert not bad2["ok"]
    ev = json.loads((root / store.EVENTS).read_text().splitlines()[0])
    assert ev["severity_text"] == "WARN" and ev["severity_number"] == 13
    assert (
        ev["event_name"] == "o11y.incident.reported" and ev["attributes"]["o11y.source"] == "agent"
    )
    assert schemas.problems("event.v1", ev) == []


def test_session_register_child_env(ctx, root):
    res = api.session_register(ctx, "proj-kf", "child", purpose="KernelForge")
    assert res["env"] == {
        "AGENT_SYS_O11Y": "child",
        "AGENT_SYS_O11Y_FLEET_DIR": str(root),
        "AGENT_SYS_O11Y_PARENT_SESSION_ID": "sess-test-0001",
    }
    assert (
        "claude --name proj-kf" in res["launch_example"]
        and "-e AGENT_SYS_O11Y=child" in res["launch_example"]
    )
    main = api.session_register(ctx, "proj-main", "main")
    assert main["ok"] and "env" not in main
    recs = api.launches(root)
    assert [r["role"] for r in recs] == ["child", "main"] and recs[1][
        "session_id"
    ] == "sess-test-0001"


def test_config_defaults_and_invalid(root):
    cfg, probs = api.load_config(root, "trigger")
    assert not probs and cfg["kick"]["interval_s"] == 1800 and cfg["packup"]["every_n_kicks"] == 3
    assert "{children}" in cfg["kick"]["template"]
    store.write_json(root / store.CONFIG["trigger"], {"v": 1, "kick": {"interval_s": 120}})
    cfg, probs = api.load_config(root, "trigger")
    assert not probs and cfg["kick"]["interval_s"] == 120 and cfg["kick"]["enabled"] is True
    store.write_json(root / store.CONFIG["trigger"], {"v": 1, "kick": {"intervall_s": 120}})
    cfg, probs = api.load_config(root, "trigger")
    assert probs and "intervall_s" in probs[0] and cfg["kick"]["interval_s"] == 1800
    acfg, _ = api.load_config(root, "analysis")
    assert (
        acfg["detectors"]["permission_wait_warn_s"] == 600
        and acfg["detectors"]["token_step"] == 1000000
    )


def test_claude_sessions_tolerant(tmp_path):
    sdir = tmp_path / "sessions"
    sdir.mkdir()
    me = os.getpid()
    (sdir / f"{me}.json").write_text(
        json.dumps(
            {
                "pid": me,
                "sessionId": "s-live",
                "cwd": "/x/y",
                "procStart": claude_sessions.proc_start(me),
                "status": "waiting",
                "waitingFor": "permission prompt",
                "statusUpdatedAt": 1795429523421,
                "tmux": "0:@1.%2",
            }
        )
    )
    (sdir / "999999.json").write_text(
        json.dumps({"pid": 999999, "sessionId": "s-dead", "status": "busy"})
    )
    (sdir / "123.json").write_text("")  # observed: empty file
    (sdir / "124.json").write_text('{"pid": 124, "sess')  # torn
    (sdir / "125.abc.key").write_text("k")  # not a session file
    sessions, probs = claude_sessions.read_all(tmp_path)
    by = {s["session_id"]: s for s in sessions}
    assert by["s-live"]["alive"] and by["s-live"]["status"] == "waiting"
    assert by["s-live"]["status_since"].startswith("2026-11-23T10:25:23")
    assert by["s-live"]["transcript_path"] == str(tmp_path / "projects" / "-x-y" / "s-live.jsonl")
    assert by["s-dead"]["crashed"] and not by["s-dead"]["alive"]
    assert len(probs) == 2
    assert claude_sessions.session_of_pid(me, tmp_path) == "s-live"


def test_proc_start_mismatch_counts_as_dead(tmp_path):
    (tmp_path / "sessions").mkdir()
    me = os.getpid()
    (tmp_path / "sessions" / f"{me}.json").write_text(
        json.dumps({"pid": me, "sessionId": "s", "procStart": "1"})
    )
    s, _ = claude_sessions.read_all(tmp_path)
    assert s[0]["alive"] is False  # pid reused by another process


def test_fleet_status(ctx, root, tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
    (tmp_path / "cfg" / "sessions").mkdir(parents=True)
    api.session_register(ctx, "proj-main", "main")
    api.ledger_append(ctx, _first("task"))
    api.ledger_append(
        ctx, {"kind": "run_start", "run_id": "r9", "theme": "decode mem 0.95", "point": "explore"}
    )
    api.event_report(ctx, "ERROR", "o11y.session.crashed", "child gone")
    st = api.fleet_status(ctx)
    assert st["task"]["task_id"] == "dsv41-pd" and st["open_runs"][0]["run_id"] == "r9"
    assert st["recent_alerts"][0]["severity"] == "ERROR" and st["sessions"] == []


def test_cli_exit_codes(root):
    env = dict(os.environ, PYTHONPATH=str(SERVER))
    run = lambda *a, **kw: subprocess.run(
        [sys.executable, "-m", "o11y_core", *a], env=env, capture_output=True, text=True, **kw
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
