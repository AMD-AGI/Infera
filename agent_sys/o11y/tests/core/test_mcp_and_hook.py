import json
import os
import subprocess
import sys

import anyio
from core_paths import SERVER
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from o11y_core import mcp_server, store

HOOK = SERVER.parent / "hooks" / "ledger_backstop.py"
PY_SH = SERVER / "py.sh"


def test_tool_schemas_have_no_top_level_combinators_or_refs():
    for t in mcp_server.TOOLS:
        s = t.input_schema
        assert s["type"] == "object"
        assert not {"allOf", "anyOf", "oneOf", "if", "then", "$defs"} & set(s), t.name
        assert "$ref" not in json.dumps(s), t.name
    assert mcp_server.TOOL_SCHEMAS["ledger_append"]["required"] == ["kind", "record"]
    assert len(mcp_server.LEDGER_DOC) <= 1200


def _call_server(root, calls):
    async def go():
        # the same invocation as plugin.json: py.sh -m o11y_core.mcp_server; a session launched as main
        params = StdioServerParameters(
            command=str(PY_SH),
            args=["-m", "o11y_core.mcp_server"],
            env={**os.environ, "AGENT_SYS_O11Y_FLEET_DIR": str(root), "AGENT_SYS_O11Y": "true"},
        )
        out = []
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                tools = await s.list_tools()
                out.append(sorted(t.name for t in tools.tools))
                for name, args in calls:
                    res = await s.call_tool(name, args)
                    out.append((res.is_error, json.loads(res.content[0].text)))
        return out

    return anyio.run(go)


def test_mcp_stdio_roundtrip(root):
    out = _call_server(
        root,
        [
            ("ledger_append", {"kind": "run_end", "record": {"run_id": "r9", "status": "kept"}}),
            (
                "ledger_append",
                {
                    "kind": "run_end",
                    "record": {
                        "run_id": "r9",
                        "status": "kept",
                        "value": 48204,
                        "evidence": {"path": "results/r9.json"},
                    },
                },
            ),
            ("ledger_query", {"kind": "run_end"}),
            (
                "event_report",
                {"severity": "WARN", "event_name": "o11y.incident.reported", "body": "node lost"},
            ),
            ("event_report", {"severity": "critical", "event_name": "x.y", "body": "b"}),
            ("session_register", {"name": "proj-kf", "role": "child"}),
            ("fleet_status", {}),
            ("ledger_schema", {"kind": "campaign"}),
            ("ledger_append", {"kind": "plan", "next": [{"theme": "flat form still accepted"}]}),
        ],
    )
    assert out[0] == [
        "event_report",
        "fleet_status",
        "ledger_append",
        "ledger_query",
        "ledger_schema",
        "session_register",
    ]
    err, bad = out[1]
    assert err and bad["problems"] == [
        "kind=run_end status=kept: (record): missing required field 'value'"
    ]
    assert out[2][1]["ok"] and not out[2][0]
    assert out[3][1]["records"][0]["value"] == 48204
    assert out[4][1]["ok"]
    assert out[5][0] and "severity" in out[5][1]["problems"][0]  # tool input enforced server-side
    assert out[6][1]["env"]["AGENT_SYS_O11Y"] == "child"
    assert out[7][1]["open_runs"] == [] and out[7][1]["recent_alerts"][0]["severity"] == "WARN"
    assert out[8][1]["required"] == ["campaign_root", "operator_ids", "started_at", "machine"]
    assert "node" in out[8][1]["properties"]["machine"]["properties"]
    assert out[9][1]["ok"]


def _hook(root, payload):
    env = {k: v for k, v in os.environ.items() if k != "CLAUDE_PLUGIN_ROOT"}
    env["AGENT_SYS_O11Y_FLEET_DIR"] = str(root)
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
    from o11y_core import api

    assert api.ledger_append(ctx, {"kind": "plan", "next": [{"theme": "x"}]})["ok"]
    assert _hook(root, bash)[0] == 0  # core-written lines are fine
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
    text = ledger.read_text().replace(
        '"theme":"x"', '"theme":"y"'
    )  # in-place edit of an earlier line
    ledger.write_text(text)
    rc, err = _hook(
        root, {"tool_name": "Edit", "tool_input": {"file_path": str(ledger)}, "cwd": str(root)}
    )
    assert rc == 2 and "modified in place" in err and "Edit was used on the ledger" in err
