"""Test data.

- `home`: a copy of tests/fixtures/home, a AGENT_SYS_HOME with one made-up project `llama70b` (fixture `proj`):
  unmanaged tasks with and without frontmatter, done with review: true, in progress, blocked, a draft, a file at
  the root of tasks/. Tests write only to the copy.
- `make_project`: small made-up projects for the rules one at a time.
- `run_home`: a AGENT_SYS_HOME with project `r` whose tasks replay recorded sessions (tests/fixtures/events) instead of
  calling a model; the helpers below (new, get, wait, ws, …) work on it through the server's test client.
- `client`: the server on `home` (a test module may give a `home` of its own) through FastAPI's test client. Its
  loops poll every 0.2 s (not 2 s and 0.5 s); a test that checks what one round of it does calls `store.poll()` itself.
Every test gets a tmux server of its own for the runs it starts (by hand or by the scheduler), never the user's
`tmux -L agent_sys`; it is killed when the test ends, with whatever still runs in it. A run that is not a replay gets
/bin/false for claude: no test calls a model, even if the scheduler starts a task nobody meant to run.
agent_sys speaks Chinese in every test (the messages they check were written in it), whatever agent_sys.yaml says; runs get it
through launch.json. tests/test_i18n.py checks English.
"""

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_sys_v2 import i18n, launch, server, tmux
from agent_sys_v2.model import new_state, read_json, read_state, write_state
from agent_sys_v2.ops import new_project
from agent_sys_v2.server import create_app

FIXTURE_HOME = Path(__file__).parent / "fixtures" / "home"
EVENTS = Path(__file__).parent / "fixtures" / "events"


@pytest.fixture(autouse=True)
def own_tmux(monkeypatch):
    name = f"agent_sys-test-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(tmux, "TMUX", name)
    monkeypatch.setattr(launch, "bundled_cli", lambda: "/bin/false")
    yield name
    subprocess.run(["tmux", "-L", name, "kill-server"], capture_output=True)
    (Path(os.environ.get("TMUX_TMPDIR") or "/tmp") / f"tmux-{os.getuid()}" / name).unlink(
        missing_ok=True
    )


@pytest.fixture(autouse=True)
def chinese(monkeypatch):
    monkeypatch.setattr(i18n, "LANG", "zh")
    monkeypatch.setattr(i18n, "use", lambda home: i18n.LANG)


@pytest.fixture
def home(tmp_path):
    shutil.copytree(FIXTURE_HOME, tmp_path / "home")
    # however long ago the repository was checked out (reminders depend on age)
    recent = time.time() - 60
    for path in (tmp_path / "home").rglob("*"):
        os.utime(path, (recent, recent))
    return tmp_path / "home"


@pytest.fixture(autouse=True)
def quick_poll(monkeypatch):
    monkeypatch.setattr(server, "POLL_SECONDS", 0.2)
    monkeypatch.setattr(server, "RUN_POLL", 0.2)


@pytest.fixture
def client(home):
    with TestClient(create_app(home)) as c:
        yield c


@pytest.fixture
def proj(home):
    return home / "projects" / "llama70b"


@pytest.fixture
def make_project(tmp_path):
    """make_project({'develop/a': '<task file text>', ...}, progress={'develop/a': '状态: 已完成\\n…'})
    -> project directory under a temporary AGENT_SYS_HOME."""

    def make(tasks, progress=None, name="p", config="title: 测试项目\n", types=()):
        pdir = tmp_path / "home" / "projects" / name
        pdir.mkdir(parents=True)
        (pdir / "project.yaml").write_text(config, encoding="utf-8")
        for t in types:
            (pdir / "tasks" / t).mkdir(parents=True, exist_ok=True)
        for tid, text in tasks.items():
            path = pdir / "tasks" / f"{tid}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        for tid, text in (progress or {}).items():
            path = pdir / "workspaces" / tid / "progress.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return pdir

    return make


DONE = "状态: 已完成\n下一步: 无\n完成时间: 2026-10-21 17:10\n"
RUNNING = "状态: 进行中\n下一步: 跑 C64\n"
BLOCKED = "状态: 阻塞\n下一步: 等你决定用哪个镜像\n"


# ---------------------------------------------------------------- project r: tasks that replay recorded sessions

T = "/api/projects/r/tasks/develop"


def run_home(tmp_path, config=""):
    """AGENT_SYS_HOME with project `r`, whose tasks replay message_mid_command (under a second) unless they say otherwise. config:
    more lines of its project.yaml."""
    home = tmp_path / "home"
    pdir = new_project(home, "r", "blank")
    (pdir / "project.yaml").write_text(
        f"title: r\n{config}agent:\n  model: claude-haiku-4-5\n  budget_usd: 1\n"
        f"  replay: {{events: {EVENTS}/message_mid_command.jsonl, speed: 40}}\n",
        encoding="utf-8",
    )
    return home


def ok(response, status=200):
    assert response.status_code == status, response.text
    return response.json()


def error(response, status):
    """The error message of a refused request."""
    assert response.status_code == status, response.text
    return response.json()["error"]


def wait_for(cond, seconds=20):
    end = time.time() + seconds
    while not cond():
        assert time.time() < end, "timed out"
        time.sleep(0.1)


def new(client, name, prompt="做事", **fields):
    return ok(
        client.post(
            "/api/projects/r/tasks",
            json={"type": "develop", "name": name, "prompt": prompt, "mode": "queue", **fields},
        )
    )


def replay(recording, **opts):
    """agent.replay for a task: one recording, or a list (run n plays the n-th)."""
    path = lambda name: str(EVENTS / f"{name}.jsonl")  # noqa: E731
    return {
        "replay": {
            "events": [path(r) for r in recording]
            if isinstance(recording, list)
            else path(recording),
            **opts,
        }
    }


def get(client, name):
    return ok(client.get(f"{T}/{name}"))


def wait(client, name, cond, seconds=30):
    end = time.time() + seconds
    while time.time() < end:
        t = get(client, name)
        if cond(t):
            return t
        time.sleep(0.2)
    raise AssertionError(f"timed out: {t['status']} {t['detail']}")


def ws(home, name):
    return home / "projects/r/workspaces/develop" / name


def run_dir_of(home, t):
    """The run directory of task t's (as the API gives it) last run."""
    return home / "projects/r/workspaces" / t["id"] / ".agent_sys/runs" / t["runs"][0]["id"]


def launch_json(home, t):
    return read_json(run_dir_of(home, t) / "launch.json")


def manage(pdir, *tids, status="queued", **fields):
    """Managed tasks: a new state.json with this status (and fields) in each one's workspace."""
    for tid in tids:
        write_state(pdir / "workspaces" / tid, {**new_state(status), **fields})


def set_status(pdir, tid, status):
    w = pdir / "workspaces" / tid
    write_state(w, {**read_state(w), "status": status})


@pytest.fixture(autouse=True)
def test_client_host(monkeypatch):
    """FastAPI's TestClient sends Host: testserver; the server answers only to the names it is given."""
    monkeypatch.setenv("AGENT_SYS_HOSTS", "testserver")


def write_result(pdir, tid, name, doc):
    """results/<name>.json in a task's workspace: doc as JSON, with the schema and its name as the label unless it says
    them; text as it is."""
    path = pdir / "workspaces" / tid / "results" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        doc
        if isinstance(doc, str)
        else json.dumps({"schema": "infera.result/v1", "label": name, **doc})
    )
    path.write_text(text, encoding="utf-8")
    return path
