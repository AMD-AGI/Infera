import json
import os
import subprocess

import anyio
from door_paths import DOOR, PY
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from o11y_door import api, mcp_server
from o11y_protocol import store

HOOK = DOOR / "hooks" / "backstop.sh"  # hooks.json command: sh membership precheck, then bin/py


def test_tool_schemas_have_no_top_level_combinators_or_refs():
    assert [t.name for t in mcp_server.TOOLS] == [
        "ledger_append",
        "ledger_query",
        "session_register",
        "fleet_status",
    ]
    for t in mcp_server.TOOLS:
        s = t.input_schema
        assert s["type"] == "object"
        assert not {"allOf", "anyOf", "oneOf", "if", "then", "$defs"} & set(s), t.name
        assert "$ref" not in json.dumps(s), t.name
        assert not (t.meta or {}).get("anthropic/alwaysLoad"), t.name  # review E3
    assert mcp_server.TOOL_SCHEMAS["ledger_append"]["required"] == ["kind", "record"]
    assert len(mcp_server.LEDGER_DOC) <= 1200
    assert "machine:" not in mcp_server.LEDGER_DOC


def _call_server(root, calls):
    async def go():
        # the same invocation as the door's plugin.json; a session launched as main
        params = StdioServerParameters(
            command=str(PY),
            args=["-m", "o11y_door.serve"],
            env={
                **os.environ,
                "AGENT_SYS_O11Y_FLEET_DIR": str(root),
                "AGENT_SYS_O11Y": "true",
                "CLAUDE_CODE_ENTRYPOINT": "sdk-cli",  # a member's -p run still gets the server
            },
        )
        out = []
        async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            out.append(sorted(t.name for t in (await s.list_tools()).tools))
            for name, args in calls:
                res = await s.call_tool(name, args)
                out.append((res.is_error, json.loads(res.content[0].text)))
        return out

    return anyio.run(go)


def test_mcp_stdio_roundtrip(root):
    run_end = {
        "run_id": "r9",
        "status": "kept",
        "value": 48204,
        "evidence": {"path": "results/r9.json"},
    }
    out = _call_server(
        root,
        [
            ("ledger_append", {"kind": "run_end", "record": {"run_id": "r9", "status": "kept"}}),
            ("ledger_append", {"kind": "run_end", "record": run_end}),
            ("ledger_query", {"kind": "run_end"}),
            (
                "session_register",
                {"decision": "child", "name": "proj-kf", "purpose": "KernelForge"},
            ),
            ("session_register", {"decision": "maybe"}),
            ("fleet_status", {}),
            ("ledger_append", {"kind": "plan", "next": [{"theme": "flat form still accepted"}]}),
        ],
    )
    assert out[0] == ["fleet_status", "ledger_append", "ledger_query", "session_register"]
    err, bad = out[1]
    assert err and bad["problems"] == [
        "kind=run_end status=kept: (record): missing required field 'value'"
    ]
    assert (
        bad["schema"]["kind"] == "run_end" and "evidence" in bad["schema"]["properties"]
    )  # review C1
    assert out[2][1]["ok"] and not out[2][0]
    assert out[3][1]["records"][0]["value"] == 48204
    assert out[4][1]["env"]["AGENT_SYS_O11Y"] == "child"
    assert out[5][0] and "decision" in out[5][1]["problems"][0]  # tool input enforced server-side
    assert out[6][1]["open_runs"] == [] and out[6][1]["registry_problem"]
    assert out[7][1]["ok"]


def test_serve_exits_for_non_member_print_mode(tmp_path):
    """`claude -p` outside every fleet: the server exits before importing the MCP SDK (spec 13.2 start cost)."""
    drop = ("AGENT_SYS_O11Y", "AGENT_SYS_O11Y_FLEET_DIR", "CLAUDE_CODE_SESSION_ID")
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(AGENT_SYS_HOME=str(tmp_path / "prefix"), CLAUDE_CODE_ENTRYPOINT="sdk-cli")
    run = [str(PY), "-X", "importtime", "-m", "o11y_door.serve"]
    p = subprocess.run(run, input="", capture_output=True, text=True, env=env, timeout=30)
    assert p.returncode == 0 and "o11y_door.mcp_server" not in p.stderr
    pin = tmp_path / "prefix" / "state" / "o11y" / "sessions" / "s-p.json"
    pin.parent.mkdir(parents=True)
    pin.write_text("{}")
    env["CLAUDE_CODE_SESSION_ID"] = "s-p"  # a pinned main session's -p run keeps the server
    p = subprocess.run(run, input="", capture_output=True, text=True, env=env, timeout=30)
    assert "o11y_door.mcp_server" in p.stderr


def _hook(root, payload, member=True, env_extra=None):
    drop = (
        "CLAUDE_PLUGIN_ROOT",
        "AGENT_SYS_O11Y",
        "CLAUDE_CODE_SESSION_ID",
        "AGENT_SYS_O11Y_FLEET_DIR",
    )
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env["AGENT_SYS_HOME"] = str(
        root.parent / "prefix"
    )  # no venv there: bin/py falls back to python3
    if member:
        env.update(AGENT_SYS_O11Y="true", AGENT_SYS_O11Y_FLEET_DIR=str(root))
    env.update(env_extra or {})
    p = subprocess.run(
        [str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    return p.returncode, p.stderr


def test_backstop_hook(ctx, root):
    ledger = root / store.LEDGER
    bash = {"tool_name": "Bash", "tool_input": {"command": "true"}, "cwd": str(root)}
    assert _hook(root, bash) == (0, "")  # no ledger yet
    assert api.ledger_append(ctx, {"kind": "plan", "next": [{"theme": "x"}]})["ok"]
    assert _hook(root, bash)[0] == 0  # door-written lines are fine
    with open(ledger, "a") as f:  # echo >> with a bad record
        f.write(
            '{"v":1,"kind":"run_end","ts":"2026-11-23T11:00:00Z","run_id":"r7","status":"kept"}\n'
        )
    rc, err = _hook(root, bash)
    assert (
        rc == 2
        and "line 2" in err
        and "missing required field 'writer'" in err
        and "ledger_append" in err
    )
    assert _hook(root, bash) == (0, "")  # reported once
    with open(ledger, "a") as f:  # a non-member session's hook stops at the membership precheck
        f.write("not json\n")
    assert _hook(root, bash, member=False) == (0, "")
    ledger.write_text(ledger.read_text().replace('"theme":"x"', '"theme":"y"'))  # in-place edit
    rc, err = _hook(
        root, {"tool_name": "Edit", "tool_input": {"file_path": str(ledger)}, "cwd": str(root)}
    )
    assert rc == 2 and "modified in place" in err and "Edit was used on the ledger" in err


def test_backstop_pinned_main_session(ctx, root):
    """A main session has no env marker: the sh precheck finds its pin, the hook resolves the fleet from it."""
    from o11y_protocol import fleet

    prefix = root.parent / "prefix"
    env = {"AGENT_SYS_HOME": str(prefix), "CLAUDE_CODE_SESSION_ID": "s-pinned"}
    bash = {
        "tool_name": "Bash",
        "tool_input": {"command": "true"},
        "session_id": "s-pinned",
        "cwd": str(root),
    }
    assert api.ledger_append(ctx, {"kind": "plan", "next": [{"theme": "x"}]})["ok"]
    with open(root / store.LEDGER, "a") as f:
        f.write("not json\n")
    assert _hook(root, bash, member=False, env_extra=env) == (0, "")  # no pin: not a member
    pin = fleet.pin_path("s-pinned", env)
    pin.parent.mkdir(parents=True, exist_ok=True)
    store.write_json(
        pin, {"v": 1, "session_id": "s-pinned", "fleet_dir": str(root), "pinned_at": store.now()}
    )
    rc, err = _hook(root, bash, member=False, env_extra=env)
    assert rc == 2 and "not valid JSON" in err
