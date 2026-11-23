import json
import os
import subprocess
import sys

import anyio
from door_paths import DOOR, PY
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from o11y_door import api, mcp_server
from o11y_protocol import store

HOOK = DOOR / "hooks" / "ledger_backstop.py"


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
            args=["-m", "o11y_door.mcp_server"],
            env={**os.environ, "AGENT_SYS_O11Y_FLEET_DIR": str(root), "AGENT_SYS_O11Y": "true"},
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


def _hook(root, payload, member=True):
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDE_PLUGIN_ROOT", "AGENT_SYS_O11Y")}
    env["AGENT_SYS_O11Y_FLEET_DIR"] = str(root)
    if member:
        env["AGENT_SYS_O11Y"] = "true"
    p = subprocess.run(
        [sys.executable, "-I", str(HOOK)],
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
