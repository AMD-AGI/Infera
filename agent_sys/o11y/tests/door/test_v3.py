"""Design v3 behaviour of the door: writer role, file modes, registration, config, ppid identity, selective
tool listing."""

import json
import os
import stat
import subprocess

import anyio
import pytest
from door_paths import PY
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from o11y_door import api
from o11y_protocol import schemas, store


@pytest.mark.parametrize(
    "flag,listed,expect",
    [
        ("true", None, "main"),
        ("child", None, "child"),
        (None, None, "other"),
        ("off", None, "other"),
        (None, "child", "child"),
        ("true", "child", "child"),
        (None, "other", "other"),
    ],
)
def test_writer_role(root, monkeypatch, flag, listed, expect):
    if flag is None:
        monkeypatch.delenv("AGENT_SYS_O11Y", raising=False)
    else:
        monkeypatch.setenv("AGENT_SYS_O11Y", flag)
    if listed:
        store.write_json(
            root / store.SESSIONS,
            {
                "v": 1,
                "generated_at": store.now(),
                "sessions": [
                    {
                        "session_id": "sess-x",
                        "pid": 1,
                        "alive": True,
                        "role": listed,
                        "status": "idle",
                    }
                ],
            },
        )
    assert api.Context(root=root, session_id="sess-x").role == expect


def test_public_identity_and_alias():
    rec = {"kind": "run_end", "run_id": "r1"}
    assert api.identity(rec) == ("run_end", "r1")
    assert api.identity({"kind": "nope"}) is None


def test_file_and_dir_modes(ctx, root):
    assert api.ledger_append(ctx, {"kind": "plan", "next": [{"theme": "x"}]})["ok"]
    store.write_json(root / "facts" / "general.json", {"v": 1})
    for d in (root, root / "ledger", root / "facts"):
        assert stat.S_IMODE(d.stat().st_mode) == 0o700, d
    for f in (root / store.LEDGER, root / "facts" / "general.json"):
        assert stat.S_IMODE(f.stat().st_mode) == 0o600, f


def test_event_dedup_key_filled_and_required(ctx, root):
    assert api.event_report(ctx, "INFO", "o11y.compaction.done", "compacted")["ok"]
    ev = json.loads((root / store.EVENTS).read_text())
    assert ev["attributes"]["o11y.dedup_key"].startswith("auto:")
    ev["attributes"].pop("o11y.dedup_key")
    assert any("o11y.dedup_key" in p for p in schemas.problems("event.v1", ev))


def test_register_not_in_fleet_and_adopted_child(ctx, root):
    d = api.session_register(
        ctx, "not_in_fleet", purpose="someone else's notebook", session_id="s-stray"
    )
    assert d["ok"] and "env" not in d
    c = api.session_register(ctx, "child", name="hand-started", purpose="KF", session_id="s-run")
    assert c["ok"] and "env" not in c  # already running: nothing to launch
    recs = api.launches(root)
    assert [(r["decision"], r.get("session_id")) for r in recs] == [
        ("not_in_fleet", "s-stray"),
        ("child", "s-run"),
    ]


def test_config_v3_keys(root):
    cfg, probs, _ = api.load_config(root, "analysis")
    assert not probs
    assert cfg["sources"] == {"machine_pool_dir": None, "e2e_flow_globs": [], "otel_dir": None}
    tcfg, _, _ = api.load_config(root, "trigger")
    assert set(tcfg) == {"v", "task_book", "kick", "packup", "refresh", "log"}
    dcfg, _, _ = api.load_config(root, "display")
    assert set(dcfg) == {"v", "mode", "poll_ms", "recent_runs_n", "log_lines"}


def test_ppid_is_the_launcher(tmp_path):
    """py.sh execs python, so the door's ppid is the process that started it — the claude pid whose
    sessions/<pid>.json names the session (review E10)."""
    out = subprocess.run(
        [str(PY), "-c", "import os; print(os.getppid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert int(out.stdout) == os.getpid()
    from o11y_door import claude_sessions

    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / f"{os.getpid()}.json").write_text(
        json.dumps({"pid": os.getpid(), "sessionId": "s-me"})
    )
    assert claude_sessions.session_of_pid(os.getpid(), tmp_path) == "s-me"
    (tmp_path / "sessions" / f"{os.getpid()}.json").write_text("")  # observed: empty records exist
    assert claude_sessions.session_of_pid(os.getpid(), tmp_path) is None


def test_selective_listing_and_list_changed(tmp_path):
    """Not a member: no tools, calls refused. When the registry lists the parent session active, the server
    publishes the change and lists the four tools (spec §10.4, review A3)."""
    fleet_root = tmp_path / "fleet"
    cfg = tmp_path / "cfg"
    (cfg / "sessions").mkdir(parents=True)
    # the server's ppid is this test process (py.sh execs python): make it a Claude session record
    (cfg / "sessions" / f"{os.getpid()}.json").write_text(
        json.dumps({"pid": os.getpid(), "sessionId": "sess-act"})
    )
    env = {k: v for k, v in os.environ.items() if k != "AGENT_SYS_O11Y"}
    env.update(
        AGENT_SYS_O11Y_FLEET_DIR=str(fleet_root),
        CLAUDE_CONFIG_DIR=str(cfg),
        AGENT_SYS_O11Y_MEMBERSHIP_POLL_S="0.3",
        AGENT_SYS_HOME=str(tmp_path / "prefix"),
    )
    seen = []

    async def handler(msg):
        seen.append(type(getattr(msg, "root", msg)).__name__)

    async def go():
        params = StdioServerParameters(
            command=str(PY), args=["-m", "o11y_door.mcp_server"], env=env
        )
        async with (
            stdio_client(params) as (r, w),
            ClientSession(r, w, message_handler=handler) as s,
        ):
            await s.initialize()
            before = [t.name for t in (await s.list_tools()).tools]
            refused = await s.call_tool("fleet_status", {})
            store.write_json(
                fleet_root / store.SESSIONS,
                {
                    "v": 1,
                    "generated_at": store.now(),
                    "sessions": [
                        {
                            "session_id": "sess-act",
                            "pid": os.getpid(),
                            "alive": True,
                            "role": "main",
                            "status": "busy",
                            "active": True,
                        }
                    ],
                },
            )
            with anyio.fail_after(10):
                while not any("ToolListChanged" in n for n in seen):
                    await anyio.sleep(0.1)
            after = sorted(t.name for t in (await s.list_tools()).tools)
            return before, refused, after

    before, refused, after = anyio.run(go)
    assert before == []
    assert refused.is_error and "not part of" in refused.content[0].text
    assert after == ["fleet_status", "ledger_append", "ledger_query", "session_register"]
