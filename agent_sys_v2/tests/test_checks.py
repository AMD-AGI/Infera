"""Acceptance checks (checks.py): parsed from a task's frontmatter, run by the harness when the agent says it is
done, a failure sent back into the session, the task done once they pass."""

import json

import pytest
from conftest import DONE, new, ok, replay, run_home, wait, ws
from fastapi.testclient import TestClient

from agent_sys_v2 import checks
from agent_sys_v2.server import create_app

# fails the first time it runs in a workspace, passes after: an agent that fixed what the first attempt showed
FLAKY = (
    'if [ -f "$AGENT_SYS_WORKSPACE/.tried" ]; then echo fixed; '
    'else touch "$AGENT_SYS_WORKSPACE/.tried"; echo "not yet: results missing"; exit 1; fi'
)


def test_checks_parse_and_refuse_what_cannot_run():
    assert checks.parse(None) == [] and checks.parse([]) == []
    assert checks.parse("test -s report.md") == [
        {"name": "test -s report.md", "run": "test -s report.md", "timeout": checks.DEFAULT_TIMEOUT}
    ]
    assert checks.parse([{"name": "r", "run": "true", "timeout": 5}]) == [
        {"name": "r", "run": "true", "timeout": 5}
    ]
    for bad in (
        [""],
        [{"name": "no run"}],
        [{"run": "x", "timeout": 0}],
        [{"run": "x", "timeout": True}],
        [3],
    ):
        with pytest.raises(ValueError):
            checks.parse(bad)
    assert (
        checks.retries({}) == checks.DEFAULT_RETRIES and checks.retries({"check_retries": 0}) == 0
    )


def test_run_all_runs_every_check_and_keeps_the_end_of_the_output(tmp_path):
    seen = []
    attempt = checks.run_all(
        checks.parse(
            [
                'echo "$AGENT_SYS_TASK"',
                "python3 -c \"print('x' * 9000); raise SystemExit(3)\"",
                {"name": "slow", "run": "sleep 5", "timeout": 0.5},
            ]
        ),
        tmp_path,
        {"AGENT_SYS_TASK": "develop/a"},
        on_start=lambda n, c: seen.append(("start", n)),
        on_end=lambda n, r: seen.append(("end", n, r["ok"])),
    )
    a, b, c = attempt["results"]
    assert not attempt["ok"] and a["ok"] and a["output"] == "develop/a\n"
    assert (b["exit"], len(b["output"])) == (3, checks.TAIL + 1) and b["output"].startswith("…")
    # the whole group is killed at its timeout
    assert c["timed_out"] and not c["ok"] and c["seconds"] < 3
    assert seen == [
        ("start", 1),
        ("end", 1, True),
        ("start", 2),
        ("end", 2, False),
        ("start", 3),
        ("end", 3, False),
    ]
    assert checks.summary(tmp_path) is None
    assert (
        checks.record(tmp_path, attempt) == 1
        and checks.record(tmp_path, {**attempt, "ok": True}) == 2
    )
    s = checks.summary(tmp_path)
    assert (s["ok"], s["attempts"], s["passed"], s["total"]) == (True, 2, 1, 3)
    text = checks.feedback(attempt, 2)
    assert (
        "2/3 条失败" in text and "exit 3" in text and "超时" in text and "还会再检查 2 次" in text
    )
    assert "did not pass (2 of 3 failed)" in checks.feedback(attempt, 1, "en")


def test_a_failed_check_goes_back_to_the_agent_and_the_task_is_done_once_it_passes(tmp_path):
    home = run_home(tmp_path)
    with TestClient(create_app(home)) as c:
        new(
            c,
            "a",
            agent=replay("message_mid_command", speed=40, writes={"progress.md": DONE}),
            checks=[FLAKY],
        )
        t = wait(c, "a", lambda t: t["status"] == "done")
        assert t["checks"]["ok"] and t["checks"]["attempts"] == 2
        attempts = t["check_runs"]["attempts"]
        assert [a["ok"] for a in attempts] == [False, True]
        assert "not yet: results missing" in attempts[0]["results"][0]["output"]
        run = t["runs"][0]["id"]
        events = [
            json.loads(line)
            for line in (ws(home, "a") / f".agent_sys/runs/{run}/events.jsonl")
            .read_text()
            .splitlines()
        ]
        sent = [e["text"] for e in events if e.get("subtype") == "send_user"]
        # the recording's own messages, then the failed attempt sent back into the session, once
        assert (
            sent[-1].startswith("验收检查没有通过（1/1 条失败）")
            and "not yet: results missing" in sent[-1]
        )
        assert sum(s.startswith("验收检查没有通过") for s in sent) == 1
        steps = [
            (e["subtype"], e.get("phase"), e.get("ok"))
            for e in events
            if e.get("subtype") in ("check", "checks")
        ]
        assert steps[:4] == [
            ("checks", "start", None),
            ("check", "start", None),
            ("check", "end", False),
            ("checks", "end", False),
        ]
        assert steps[-1] == ("checks", "end", True)


def test_checks_that_keep_failing_leave_the_task_needing_you(tmp_path):
    home = run_home(tmp_path)
    with TestClient(create_app(home)) as c:
        new(
            c,
            "a",
            agent=replay("message_mid_command", speed=40, writes={"progress.md": DONE}),
            checks=["false"],
        )
        t = wait(c, "a", lambda t: t["status"] == "needs_you")
        assert t["detail"] == {
            "kind": "checks_failed",
            "failed": ["false"],
            "passed": 0,
            "total": 1,
        }
        assert t["checks"]["attempts"] == 1 + checks.DEFAULT_RETRIES and not t["checks"]["ok"]
        home_page = ok(c.get("/api/home"))
        assert [x["id"] for x in home_page["needs_you"]] == ["develop/a"]


def test_a_task_without_checks_is_done_as_progress_md_says(tmp_path):
    home = run_home(tmp_path)
    with TestClient(create_app(home)) as c:
        new(c, "a", agent=replay("message_mid_command", speed=40, writes={"progress.md": DONE}))
        t = wait(c, "a", lambda t: t["status"] == "done")
        assert t["checks"] is None and t["check_runs"] is None


def test_a_check_tool_changed_during_the_run_fails_the_attempt(tmp_path):
    """The run records the checks' files when it starts; an attempt after one of them changed fails first, naming it,
    whatever the check itself says."""
    from agent_sys_v2.runner import Events, run_checks

    pdir = tmp_path / "p"
    gate = pdir / "skills/own/scripts/gate.sh"
    gate.parent.mkdir(parents=True)
    gate.write_text('test -s "$AGENT_SYS_WORKSPACE/report.md"\n')
    workspace = pdir / "workspaces/experiment/x"
    workspace.mkdir(parents=True)
    (workspace / "report.md").write_text("## 结论\n")
    cfg = {
        "cwd": str(pdir),
        "checks": checks.parse(["bash skills/own/scripts/gate.sh"]),
        "env": {"AGENT_SYS_WORKSPACE": str(workspace), "AGENT_SYS_PYTHON": "python3"},
    }
    cfg["toolchain"] = checks.toolchain(pdir, cfg["checks"], cfg["env"])
    assert str(gate.resolve()) in cfg["toolchain"]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    attempt = run_checks(cfg, run_dir, Events(run_dir / "events.jsonl"))
    assert attempt["ok"] and len(attempt["results"]) == 1
    gate.write_text("exit 0\n")  # loosened by the agent
    attempt = run_checks(cfg, run_dir, Events(run_dir / "events.jsonl"))
    assert not attempt["ok"] and [r["ok"] for r in attempt["results"]] == [False, True]
    assert attempt["results"][0]["name"] == checks.TOOLCHAIN_CHECK
    text = checks.feedback(attempt, 1, "en")
    assert f"changed: {gate.resolve()}" in text and "1 of 2 failed" in text
    steps = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
        if '"check"' in line
    ]
    assert [(s["phase"], s["n"], s.get("ok")) for s in steps[-3:]] == [
        ("end", 1, False),
        ("start", 2, None),
        ("end", 2, True),
    ]


def test_a_check_gets_the_runs_rules_from_the_runner(tmp_path):
    """The rules come from the runner's memory (launch.json is the agent's to edit), and `python -m` in the project
    directory does not import a package lying there (PYTHONSAFEPATH)."""
    from agent_sys_v2.runner import Events, run_checks

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = {
        "cwd": str(tmp_path),
        "checks": checks.parse(['echo "safe=$PYTHONSAFEPATH rules=$AGENT_SYS_RULES"']),
        "env": {},
        "ledger": {"noise_pct": 2.0, "accept_pct": 3.0},
    }
    out = run_checks(cfg, run_dir, Events(run_dir / "events.jsonl"))["results"][0]["output"]
    assert 'safe=1 rules={"noise_pct": 2.0, "accept_pct": 3.0}' in out
