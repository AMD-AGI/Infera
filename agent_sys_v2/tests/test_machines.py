"""Machines (machines.py): declared by a project, named by its tasks, held by one task at a time across projects."""

import pytest
from conftest import T, get, manage, new, ok, run_home, wait
from fastapi.testclient import TestClient

from agent_sys_v2 import machines, runs, sched
from agent_sys_v2.model import AgentSysError, new_state, write_state
from agent_sys_v2.scan import load_project
from agent_sys_v2.server import create_app


def test_machines_are_declared_and_named_by_tasks():
    assert machines.declared(
        {"machines": ["a", {"name": "b", "gpus": 8, "note": "MI355X"}, {"name": "a"}, {}, 3]}
    ) == [
        {"name": "a", "gpus": None, "note": ""},
        {"name": "b", "gpus": 8, "note": "MI355X"},
    ]
    assert machines.task_machines({"machines": "n1"}) == ["n1"]
    assert machines.task_machines({"machines": ["n1", " n2 ", "n1", ""]}) == ["n1", "n2"]
    assert machines.task_machines({}) == []


def test_a_queued_task_waits_for_its_machines_across_projects(make_project):
    one = make_project(
        {"develop/a": "---\nmachines: [n1]\n---\n跑 A\n"},
        name="one",
        config="title: one\nmachines: [{name: n1, gpus: 8}]\n",
    )
    two = make_project(
        {
            "develop/b": "---\nmachines: [n1, n2]\n---\n跑 B\n",
            "develop/c": "---\nmachines: n2\n---\n跑 C\n",
        },
        name="two",
        config="title: two\n",
    )
    # it waits for you: still holds
    manage(one, "develop/a", status="needs_you", detail={"kind": "asked"})
    manage(two, "develop/b", "develop/c")
    projects = [load_project(one), load_project(two)]
    go = sched.plan(projects, 4)
    b, c = projects[1].tasks["develop/b"], projects[1].tasks["develop/c"]
    assert b.detail == {
        "kind": "machines",
        "machines": [{"name": "n1", "project": "one", "task": "develop/a", "title": "a"}],
    }
    assert [t.id for _p, t in go] == ["develop/c"] and c.detail == {}
    with pytest.raises(AgentSysError) as e:
        machines.check_free(projects, projects[1], b)
    assert e.value.status == 409 and "n1" in str(e.value) and "one/develop/a" in str(e.value)
    rows = {m["name"]: m for m in machines.table(projects)}
    [h] = rows["n1"]["holders"]
    # it has only just started waiting
    assert rows["n1"]["gpus"] == 8 and h["task"] == "develop/a" and h["idle_minutes"] == 0
    # two declares none, holds none yet
    assert [m["name"] for m in machines.table(projects, only=projects[1])] == []
    # once a's wait has gone on for an hour, it holds n1 idle
    write_state(
        one / "workspaces/develop/a",
        {**new_state("needs_you", {"kind": "asked"}), "since": "2026-01-01T00:00:00Z"},
    )
    projects = [load_project(one), load_project(two)]
    assert machines.table(projects)[0]["holders"][0]["idle_minutes"] >= machines.IDLE_MINUTES
    # done, it lets go: b starts and takes n1 and n2, so c waits for b now
    write_state(one / "workspaces/develop/a", new_state("done"))
    projects = [load_project(one), load_project(two)]
    assert [t.id for _p, t in sched.plan(projects, 4)] == ["develop/b"]
    assert projects[1].tasks["develop/c"].detail["machines"][0]["task"] == "develop/b"


def test_a_machine_held_twice_shows_both_and_neither_runs_again(make_project):
    """Two holders only if the rule was broken (a machine added to a task that waits for you): both are listed, and
    neither may run again on it while the other holds it."""
    pdir = make_project({f"develop/{n}": "---\nmachines: [n1]\n---\n跑\n" for n in "ab"})
    manage(pdir, "develop/a", status="needs_you", detail={"kind": "asked"})
    manage(pdir, "develop/b", status="running")
    project = load_project(pdir)
    [row] = machines.table([project])
    assert [h["task"] for h in row["holders"]] == ["develop/a", "develop/b"]
    for tid, other in [("develop/a", "develop/b"), ("develop/b", "develop/a")]:
        with pytest.raises(AgentSysError, match=other):
            runs.check_machines(pdir.parent.parent, project, project.tasks[tid])


def test_a_project_that_cannot_be_read_does_not_stop_the_others(make_project):
    """Any error reading another project (here its project.yaml unreadable: an OSError, not an AgentSysError) skips
    that project; it does not refuse every run that needs a machine."""
    pdir = make_project({"develop/a": "---\nmachines: [n1]\n---\n跑\n"})
    broken = make_project({}, name="broken")
    (broken / "project.yaml").chmod(0)
    project = load_project(pdir)
    runs.check_machines(pdir.parent.parent, project, project.tasks["develop/a"])


def test_machines_through_the_server(tmp_path):
    home = run_home(tmp_path, "machines: [{name: n1, gpus: 8, note: MI300X}]\n")
    with TestClient(create_app(home)) as c:
        # plays message_mid_command and stops without being done: waits for you, holds n1
        new(c, "a", machines=["n1"])
        wait(c, "a", lambda t: t["status"] == "needs_you")
        new(c, "b", machines=["n1"])
        b = wait(c, "b", lambda t: t["detail"].get("kind") == "machines", 10)
        assert b["detail"]["machines"][0]["task"] == "develop/a" and b["machines"] == ["n1"]
        assert "develop/a" in c.post(f"{T}/b/actions/start").json()["error"]
        board = ok(c.get("/api/projects/r/board"))
        assert board["machines"] == [
            {
                "name": "n1",
                "gpus": 8,
                "note": "MI300X",
                "projects": ["r"],
                "holders": board["machines"][0]["holders"],
            }
        ]
        assert [h["task"] for h in board["machines"][0]["holders"]] == ["develop/a"]
        assert [m["name"] for m in ok(c.get("/api/home"))["machines"]] == ["n1"]
        assert get(c, "b")["project_machines"] == [{"name": "n1", "gpus": 8, "note": "MI300X"}]
        ok(c.post(f"{T}/a/actions/cancel"))
        wait(c, "b", lambda t: t["status"] in ("running", "needs_you"), 10)
