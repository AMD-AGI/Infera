"""Spec §12 acceptance criteria 13, 14, 15 and 29, door side.

Offline parts always run. Parts marked `live` start `claude -p` (isolated config dir under the test's tmp dir,
the plugins by --plugin-dir) and call the model; they run only with O11Y_LIVE=1 and credentials in the
environment, else they are skipped. The claude binary: $O11Y_CLAUDE_BIN, else the newest under
~/.local/share/claude/versions, else `claude` on PATH.
Criterion 15 runs on the prefix's file system ($AGENT_SYS_HOME, else ~/.infera_agent_sys), in
<prefix>/tmp/yihou_o11y_c15 (override: O11Y_C15_DIR); it truncates its own ledger there and deletes nothing.
"""

import json
import multiprocessing as mp
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from door_paths import DOOR, O11Y
from o11y_door import api
from o11y_protocol import fleet, schemas, store

TOOL = "mcp__plugin_agent-sys-o11y-door_o11y__ledger_append"
HOOK = DOOR / "hooks" / "backstop.sh"
HINT_MARK = "o11y run ledger"


def _claude_bin() -> str | None:
    if os.environ.get("O11Y_CLAUDE_BIN"):
        return os.environ["O11Y_CLAUDE_BIN"]
    versions = Path.home() / ".local" / "share" / "claude" / "versions"
    found = sorted(
        (p for p in versions.glob("*") if p.is_file()),
        key=lambda p: [int(x) if x.isdigit() else 0 for x in p.name.split(".")],
    )
    return str(found[-1]) if found else shutil.which("claude")


CLAUDE = _claude_bin()
live = pytest.mark.skipif(
    os.environ.get("O11Y_LIVE") != "1"
    or not CLAUDE
    or not any(k.startswith(("ANTHROPIC_", "CLAUDE_CODE_USE_")) for k in os.environ),
    reason="live: set O11Y_LIVE=1, with model credentials in the environment",
)


def _run_claude(tmp: Path, prompt: str, plugin_dirs, env_extra=None, model=None, tools=(TOOL,)):
    """One `claude -p` run with stream-json output; returns (events, raw stdout)."""
    proj = tmp / "proj"
    proj.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AGENT_SYS_O11Y", "CLAUDE_"))}
    env.update(
        CLAUDE_CONFIG_DIR=str(tmp / "cfg"),
        DISABLE_AUTOUPDATER="1",
        AGENT_SYS_HOME=str(tmp / "prefix"),
        **(env_extra or {}),
    )
    argv = [
        CLAUDE,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--allowedTools",
        ",".join(tools),
    ]
    for d in plugin_dirs:
        argv += ["--plugin-dir", str(d)]
    if model:
        argv += ["--model", model]
    p = subprocess.run(
        argv + ["--", prompt],
        cwd=proj,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        stdin=subprocess.DEVNULL,
    )
    events = []
    for line in p.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    assert events, f"no stream-json output (rc={p.returncode}): {p.stderr[-500:]}"
    return events, p.stdout


def _blocks(events, kind):
    for ev in events:
        content = (ev.get("message") or {}).get("content")
        if isinstance(content, list):
            yield from (b for b in content if isinstance(b, dict) and b.get("type") == kind)


def _member_env(fleet_root: Path) -> dict:
    return {"AGENT_SYS_O11Y": "true", "AGENT_SYS_O11Y_FLEET_DIR": str(fleet_root)}


# --- 13: a refusal names the field; the agent fixes it on the next call ---------------------------------------


def test_c13_misspelt_required_key_names_both(ctx):
    res = api.ledger_append(
        ctx, {"kind": "finding", "finding_id": "f1", "text": "t", "evidance_rank": "inference"}
    )
    assert not res["ok"] and any(
        "'evidence_rank'" in p and "'evidance_rank' looks like a misspelling" in p
        for p in res["problems"]
    ), res["problems"]


@pytest.mark.parametrize(
    "record, field",
    [
        ({"finding_id": "f1", "text": "t"}, "evidence_rank"),  # missing required
        (
            {"finding_id": "f1", "text": "t", "evidence_rank": "inference", "evidance": "x"},
            "evidance",
        ),  # unknown
    ],
)
def test_c13_refusal_names_the_field(ctx, root, record, field):
    res = api.ledger_append(ctx, {"kind": "finding", **record})
    assert not res["ok"] and any(field in p for p in res["problems"]), res["problems"]
    assert not (root / store.LEDGER).exists()  # nothing written


@live
def test_c13_live_default_model_fixes_on_next_call(tmp_path):
    fleet_root = tmp_path / "fleet"
    prompt = (
        'Call the ledger_append tool with kind "finding" and this record, copied verbatim with the misspelt key '
        'exactly as written: {"finding_id": "f-c13", "text": "c13 probe", "evidance_rank": "inference"}. '
        "If the call is refused, correct the record as the refusal says and call ledger_append again. "
        "Then reply DONE."
    )
    events, _ = _run_claude(tmp_path, prompt, [DOOR], _member_env(fleet_root))
    uses = [b for b in _blocks(events, "tool_use") if b["name"] == TOOL]
    results = {b["tool_use_id"]: b for b in _blocks(events, "tool_result")}
    assert uses, "the model made no ledger_append call"
    first = results[uses[0]["id"]]
    if not first.get("is_error"):
        pytest.skip(
            "the model corrected the record before its first call; the refusal path was not exercised"
        )
    assert "evidance_rank" in json.dumps(first["content"])  # the refusal names the misspelt field
    assert len(uses) >= 2 and not results[uses[1]["id"]].get("is_error"), (
        "not fixed on the next call"
    )
    recs, _ = api.read_ledger(fleet_root)
    assert [r["finding_id"] for r in recs if r["kind"] == "finding"] == ["f-c13"]


# --- 14: an invalid out-of-door append is reported at the next tool call; a valid one is accepted -------------


def _hook(env_extra, payload=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AGENT_SYS_O11Y", "CLAUDE_"))}
    env.update(env_extra)
    payload = payload or {"tool_name": "Bash", "tool_input": {"command": "true"}}
    p = subprocess.run(
        [str(HOOK)], input=json.dumps(payload), capture_output=True, text=True, env=env, timeout=30
    )
    return p.returncode, p.stderr


def _valid_line(tmp: Path) -> str:
    """A runs.v1 line exactly as the door writes it, made in a scratch fleet."""
    scratch = api.Context(root=tmp / "scratch-fleet", session_id="s-writer", role="main")
    assert api.ledger_append(scratch, {"kind": "plan", "next": [{"theme": "c14"}]})["ok"]
    return (tmp / "scratch-fleet" / store.LEDGER).read_text()


def test_c14_out_of_door_append(ctx, root, tmp_path):
    env = {**_member_env(root), "AGENT_SYS_HOME": str(tmp_path / "prefix")}
    assert api.ledger_append(ctx, {"kind": "plan", "next": [{"theme": "x"}]})["ok"]
    assert _hook(env) == (0, "")
    with open(root / store.LEDGER, "a") as f:  # e.g. `echo ... >> runs.jsonl`
        f.write('{"v":1,"kind":"finding","finding_id":"f9","text":"t"}\n')
    rc, err = _hook(env)  # the writing session's next tool call
    assert rc == 2 and HINT_MARK in err and "line 2" in err and "ledger_append" in err
    assert _hook(env) == (0, "")  # reported once
    with open(root / store.LEDGER, "a") as f:
        f.write(_valid_line(tmp_path))
    assert _hook(env) == (0, "")  # a valid record appended outside the door is accepted
    assert [r["kind"] for r in api.read_ledger(root)[0]] == ["plan", "plan"]


@live
def test_c14_live_reported_to_the_writing_session(tmp_path):
    fleet_root = tmp_path / "fleet"
    ledger = fleet_root / store.LEDGER
    ctx = api.Context(root=fleet_root, session_id="s-seed", role="main")
    assert api.ledger_append(ctx, {"kind": "plan", "next": [{"theme": "seed"}]})["ok"]
    env = _member_env(fleet_root)
    bad = '{"v":1,"kind":"finding","finding_id":"f-c14","text":"t"}'
    prompt = (
        f"Run exactly this Bash command and nothing else: printf '%s\\n' '{bad}' >> {ledger}\n"
        "Then reply with any feedback you received after the command (hook or system messages), verbatim, "
        "or NONE."
    )
    events, raw = _run_claude(tmp_path / "s1", prompt, [DOOR], env, model="haiku", tools=("Bash",))
    assert HINT_MARK in raw and "evidence_rank" in raw, raw[-2000:]
    good = _valid_line(tmp_path).strip()
    prompt = (
        f"Run exactly this Bash command and nothing else: printf '%s\\n' '{good}' >> {ledger}\n"
        "Then reply with any feedback you received after the command, verbatim, or NONE."
    )
    events, raw = _run_claude(tmp_path / "s2", prompt, [DOOR], env, model="haiku", tools=("Bash",))
    assert [b for b in _blocks(events, "tool_use") if b["name"] == "Bash"], "no Bash call"
    assert HINT_MARK not in raw


# --- 15: concurrent appends on the prefix's file system never tear or interleave ------------------------------


def _c15_dir() -> Path:
    if os.environ.get("O11Y_C15_DIR"):
        return Path(os.environ["O11Y_C15_DIR"])
    return fleet.prefix() / "tmp" / "yihou_o11y_c15"


def _c15_worker(args):
    root, n, wid, paths = args
    sys.path[:0] = [p for p in paths if p not in sys.path]
    from o11y_door import api as a

    ctx = a.Context(root=Path(root), session_id=f"s-c15-{wid}", role="child")
    for i in range(n):
        size = 200 + (i * 7919 + wid * 104729) % 30000  # lines from 0.3 KiB to 30 KiB
        r = a.ledger_append(
            ctx,
            {
                "kind": "finding",
                "finding_id": f"f-{wid}-{i}",
                "text": "y" * size,
                "evidence_rank": "inference",
            },
        )
        assert r["ok"], r
    return n


def test_c15_concurrent_appends_on_prefix_fs():
    root = _c15_dir() / "fleet"
    ledger = root / store.LEDGER
    if ledger.exists():
        ledger.write_bytes(b"")  # this test's own file: truncate, never delete
    procs, n = 4, 100
    with mp.get_context("spawn").Pool(procs) as pool:
        args = [(str(root), n, w, sys.path[:]) for w in range(procs)]
        assert sum(pool.map(_c15_worker, args)) == procs * n
    data = ledger.read_bytes()
    assert data.endswith(b"\n")
    lines = data.split(b"\n")[:-1]
    assert len(lines) == procs * n
    seen = set()
    for line in lines:
        rec = json.loads(line)  # a torn or interleaved line does not parse
        assert schemas.problems("runs.v1", rec) == []
        seen.add(rec["finding_id"])
    assert len(seen) == procs * n
    fs = subprocess.run(
        ["stat", "-f", "-c", "%T", str(root)], capture_output=True, text=True
    ).stdout.strip()
    print(f"c15: {procs}x{n} appends on {root} (fs type {fs})")


# --- 29: owner-only files; development plugins override installed ones; installer settings ------------------


def test_c29_protocol_files_owner_only(ctx, root, tmp_path):
    env = {**_member_env(root), "AGENT_SYS_HOME": str(tmp_path / "prefix")}
    assert api.ledger_append(ctx, {"kind": "plan", "next": [{"theme": "x"}]})["ok"]
    assert api.event_report(ctx, "ERROR", "o11y.incident.reported", "x")["ok"]
    assert api.session_register(ctx, "child", name="c29", purpose="modes")["ok"]
    with open(root / store.LEDGER, "a") as f:
        f.write("not json\n")
    assert _hook(env)[0] == 2  # the backstop writes its cursor
    pin = fleet.pin_path("s-c29", {"AGENT_SYS_HOME": str(tmp_path / "prefix")})
    store.write_json(
        pin, {"v": 1, "session_id": "s-c29", "fleet_dir": str(root), "pinned_at": store.now()}
    )
    checked = 0
    for base in (root, pin.parent):
        for d, dirs, files in os.walk(base):
            assert stat.S_IMODE(os.stat(d).st_mode) == 0o700, d
            for n in files:
                assert stat.S_IMODE(os.stat(os.path.join(d, n)).st_mode) == 0o600, os.path.join(
                    d, n
                )
                checked += 1
    assert checked >= 5


def test_c29_readme_lists_every_user_side_setting():
    """Registration changes the user's Claude configuration, so it is a README step (env_mgr recipe o11y.yaml
    installs only the venv); check the README names each one."""
    text = (O11Y / "README.md").read_text()
    assert "claude plugin marketplace add agent_sys/o11y" in text
    for name in ("agent-sys-o11y-door", "agent-sys-o11y", "agent-sys-o11y-dash"):
        assert name in text
    assert "mcp__plugin_agent-sys-o11y-door_o11y__*" in text  # permissions.allow
    assert "keybindings" in text and "--plugin-dir" in text


@live
def test_c29_live_dev_plugin_overrides_installed(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_")}
    env.update(CLAUDE_CONFIG_DIR=str(tmp_path / "cfg"), DISABLE_AUTOUPDATER="1")
    for argv in (
        ["marketplace", "add", str(O11Y)],
        ["install", "agent-sys-o11y-door@agent-sys-local"],
    ):
        subprocess.run(
            [CLAUDE, "plugin", *argv], env=env, check=True, capture_output=True, timeout=120
        )
    dev = tmp_path / "dev-door"
    shutil.copytree(
        DOOR, dev, ignore=shutil.ignore_patterns("__pycache__")
    )  # protocol symlink dereferenced
    events, _ = _run_claude(tmp_path, "Reply with just OK.", [dev], model="haiku", tools=("Bash",))
    init = next(e for e in events if e.get("type") == "system" and e.get("subtype") == "init")
    doors = [p for p in init.get("plugins", []) if p.get("name") == "agent-sys-o11y-door"]
    assert [Path(p["path"]).resolve() for p in doors] == [dev.resolve()], init.get("plugins")
