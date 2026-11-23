"""Design v2 behaviour of o11y_core: prefix/slug drift, fleet root, writer role, file modes, registration,
selective tool listing."""

import json
import os
import stat
import sys

import anyio
import pytest
from core_paths import AGENT_SYS, SERVER, SLUG_VECTORS
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
    assert api.identity(rec) == ("run_end", "r1") and api._identity is api.identity
    assert api.identity({"kind": "nope"}) is None


def test_file_and_dir_modes(ctx, root):
    assert api.ledger_append(ctx, {"kind": "plan", "next": [{"theme": "x"}]})["ok"]
    store.write_json(root / "facts" / "general.json", {"v": 1})
    for d in (root, root / "ledger", root / "facts"):
        assert stat.S_IMODE(d.stat().st_mode) == 0o700, d
    for f in (root / store.LEDGER, root / "facts" / "general.json"):
        assert stat.S_IMODE(f.stat().st_mode) == 0o600, f


def test_event_dedup_key_filled_for_agents_and_required(ctx, root):
    assert api.event_report(ctx, "INFO", "o11y.agent.note", "service up")["ok"]
    ev = json.loads((root / store.EVENTS).read_text())
    assert ev["attributes"]["o11y.dedup_key"].startswith("agent:")
    ev["attributes"].pop("o11y.dedup_key")
    assert any("o11y.dedup_key" in p for p in schemas.problems("event.v1", ev))


def test_register_declined_and_running_child(ctx, root, tmp_path):
    (tmp_path / "claude-config" / "sessions").mkdir(parents=True)
    me = os.getpid()
    from o11y_core import claude_sessions

    (tmp_path / "claude-config" / "sessions" / f"{me}.json").write_text(
        json.dumps(
            {
                "pid": me,
                "sessionId": "s-run",
                "procStart": claude_sessions.proc_start(me),
                "name": "hand-started",
            }
        )
    )
    d = api.session_register(ctx, "stray", "declined", session_id="s-stray")
    assert d["ok"] and "env" not in d
    c = api.session_register(ctx, "hand-started", "child", purpose="KF", session_id="s-run")
    assert c["ok"] and "env" not in c  # already running: nothing to launch
    recs = api.launches(root)
    assert [(r["role"], r.get("session_id")) for r in recs] == [
        ("declined", "s-stray"),
        ("child", "s-run"),
    ]
    fleet_ids = [s["session_id"] for s in api.fleet_sessions(root)]
    assert fleet_ids == ["s-run"]


def test_config_v2_keys(root):
    cfg, probs = api.load_config(root, "analysis")
    assert not probs
    assert cfg["sources"]["e2e_flow_globs"] == [] and cfg["sources"]["task_book"] is None
    assert cfg["log"]["lookback_bytes"] == 800000 and cfg["display"]["lookback_bytes"] == 200000
    assert cfg["budget_s"] == 300 and cfg["detectors"]["token_step"] == 1000000
    tcfg, _ = api.load_config(root, "trigger")
    assert tcfg["question"]["enabled"] and "{name}" in tcfg["question"]["template"]


# ------------------------------------------------------------------------------------------ selective listing
def test_selective_listing_and_list_changed(tmp_path):
    """Not a member: no tools, calls refused. When state/active/<sid>.json appears, the server sends
    notifications/tools/list_changed and lists the six tools (spec §10.4)."""
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
    )
    seen = []

    async def handler(msg):
        seen.append(type(getattr(msg, "root", msg)).__name__)

    async def go():
        params = StdioServerParameters(
            command=str(SERVER / "py.sh"), args=["-m", "o11y_core.mcp_server"], env=env
        )
        async with (
            stdio_client(params) as (r, w),
            ClientSession(r, w, message_handler=handler) as s,
        ):
            await s.initialize()
            before = [t.name for t in (await s.list_tools()).tools]
            refused = await s.call_tool("fleet_status", {})
            store.write_json(fleet_root / "state" / "active" / "sess-act.json", {"v": 1})
            with anyio.fail_after(10):
                while not any("ToolListChanged" in n for n in seen):
                    await anyio.sleep(0.1)
            after = sorted(t.name for t in (await s.list_tools()).tools)
            return before, refused, after

    before, refused, after = anyio.run(go)
    assert before == []
    assert refused.is_error and "not part of" in refused.content[0].text
    assert after == [
        "event_report",
        "fleet_status",
        "ledger_append",
        "ledger_query",
        "ledger_schema",
        "session_register",
    ]
