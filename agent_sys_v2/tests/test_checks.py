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
