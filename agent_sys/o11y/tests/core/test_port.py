"""Design v3 behaviour of the protocol library and the door: prefix/slug drift, fleet root and pin, writer
role, file modes, registration, schemas and catalogue, selective tool listing, ppid identity."""

import json
import os
import stat
import subprocess
import sys

import anyio
import pytest
from core_paths import AGENT_SYS, PROTOCOL, SERVER, SLUG_VECTORS
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from o11y_core import api, fleet, schemas, store


# ------------------------------------------------------------------------------------------ drift tests
def _env_mgr_prefix():
    sys.path.insert(0, str(AGENT_SYS))
    try:
        from env_mgr.prefix import Prefix
    finally:
        sys.path.remove(str(AGENT_SYS))
    return Prefix


@pytest.mark.parametrize("case", ["home", "override", "tilde", "relative", "no_home", "symlink"])
def test_prefix_matches_env_mgr(case, tmp_path, monkeypatch):
    Prefix = _env_mgr_prefix()
    real = tmp_path / "real_prefix"
    real.mkdir()
    link = tmp_path / "link_prefix"
    link.symlink_to(real)
    monkeypatch.chdir(tmp_path)
    environ = {
        "home": {"HOME": str(tmp_path / "h")},
        "override": {"HOME": "/x", "AGENT_SYS_HOME": str(real)},
        "tilde": {"HOME": str(tmp_path), "AGENT_SYS_HOME": "~/real_prefix"},
        "relative": {"HOME": "/x", "AGENT_SYS_HOME": "real_prefix"},
        "no_home": {},
        "symlink": {"HOME": "/x", "AGENT_SYS_HOME": str(link)},
    }[case]
    if case == "tilde":
        monkeypatch.setenv("HOME", str(tmp_path))  # expanduser reads the process env
    assert fleet.prefix(environ) == Prefix.resolve(environ).root
    if case == "symlink":
        assert (
            fleet.prefix(environ) == real
        )  # canonical: the mods cannot resolve symlinks (design §4)


def test_slug_vectors_shared_with_ts():
    doc = json.loads(SLUG_VECTORS.read_text())
    assert len(doc["cases"]) >= 8
    for c in doc["cases"]:
        assert fleet.slug(c["in"]) == c["out"], c["in"]
        assert len(fleet.slug(c["in"])) <= 213


def test_fleet_dir_resolution(tmp_path):
    env = {"HOME": "/h", "AGENT_SYS_HOME": str(tmp_path / "p"), "CLAUDE_PROJECT_DIR": "/w/proj.x"}
    assert fleet.fleet_dir(env) == tmp_path / "p" / "state" / "o11y" / "fleets" / "-w-proj-x"
    env["AGENT_SYS_O11Y_FLEET_DIR"] = str(tmp_path / "explicit")
    assert fleet.fleet_dir(env) == tmp_path / "explicit"


# ------------------------------------------------------------------------------------------ identity and role
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


def test_fleet_pin_comes_first(tmp_path):
    env = {
        "HOME": "/h",
        "AGENT_SYS_HOME": str(tmp_path / "p"),
        "CLAUDE_PROJECT_DIR": "/w/a",
        "AGENT_SYS_O11Y_FLEET_DIR": str(tmp_path / "explicit"),
    }
    assert fleet.fleet_dir(env, session_id="s1") == tmp_path / "explicit"  # no pin yet
    pin = fleet.pin_path("s1", env)
    store.write_json(
        pin,
        {
            "v": 1,
            "session_id": "s1",
            "fleet_dir": str(tmp_path / "pinned"),
            "pinned_at": store.now(),
        },
    )
    assert schemas.problems("session_pin.v1", store.read_json(pin)) == []
    assert fleet.fleet_dir(env, session_id="s1") == tmp_path / "pinned"
    pin.write_text("{torn")
    assert fleet.fleet_dir(env, session_id="s1") == tmp_path / "explicit"  # a torn pin is ignored




def test_schemas_share_common_definitions():
    for path in sorted((PROTOCOL / "schemas").glob("*.schema.json")):
        doc = json.loads(path.read_text())
        assert doc["$id"] == path.name
        if path.name != "common.v1.schema.json":  # one definition of time/id/writer (review C5)
            assert not {"time", "id", "writer"} & set(doc.get("$defs", {})), path.name
    good = {"v": 1, "log": {"seq": 3, "ts": store.now(), "reason": "timer"}}
    assert schemas.problems("requests.v1", good) == []
    assert schemas.problems("requests.v1", {"v": 1, "display": {"seq": 1, "ts": store.now()}})
    assert (
        schemas.problems(
            "otel_health.v1", {"v": 1, "ts": store.now(), "healthy": False, "outages": 2}
        )
        == []
    )
    assert "machine" not in api.ledger_kinds()


def test_event_catalogue_is_well_formed():
    cat = schemas.events()
    pattern = schemas.load("event.v1")["properties"]["event_name"]["pattern"]
    import re

    for name, entry in cat.items():
        assert re.fullmatch(pattern, name), name
        assert entry["level"] in ("INFO", "WARN", "ERROR") and entry["level"] in entry.get(
            "levels", [entry["level"]]
        )
    for gone in ("o11y.session.started", "o11y.session.ended", "o11y.protocol.invalid_line"):
        assert gone not in cat  # review C4


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
        [str(SERVER / "py.sh"), "-c", "import os; print(os.getppid())"],
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


# ------------------------------------------------------------------------------------------ selective listing
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
            command=str(SERVER / "py.sh"), args=["-m", "o11y_door.mcp_server"], env=env
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
