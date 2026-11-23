"""o11y_sources / o11y_log / o11y_facts: cursors, adapters, detectors, facts, model passes (fake claude),
one-shot locking and queue, security rules.
Fixtures are copies of real sources; every test works in tmp_path and never modifies them."""

import fcntl
import json
import os
import shutil
import stat
import time
from pathlib import Path

import pytest
from analyze_paths import FIX
from o11y_core import api, claude_sessions, schemas, store
from o11y_facts import __main__ as facts_cli
from o11y_log import __main__ as log_cli
from o11y_sources import Problems
from o11y_sources import cursor as cur_mod
from o11y_sources import transcripts as tr

MAIN_SID = "sess-main-0001"


def _w(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for d in lines:
            f.write(json.dumps(d) + "\n")


def _w_private(path: Path, lines):
    """Like the collector writes them: dir 0700, files not group/world-writable."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    _w(path, lines)
    os.chmod(path, 0o600)


def _assistant(mid, inp=10, cc=20, cr=30, out=5, uuid=None, model="claude-opus-5-5"):
    return {
        "type": "assistant",
        "uuid": uuid or f"u-{mid}-{time.time_ns()}",
        "timestamp": "2026-11-23T10:00:00.000Z",
        "message": {
            "id": mid,
            "model": model,
            "role": "assistant",
            "content": [{"type": "text", "text": "x"}],
            "usage": {
                "input_tokens": inp,
                "cache_creation_input_tokens": cc,
                "cache_read_input_tokens": cr,
                "output_tokens": out,
            },
        },
    }


def _api_error(i):
    return {
        "type": "assistant",
        "uuid": f"err-{i}-{time.time_ns()}",
        "timestamp": "2026-11-23T10:01:00.000Z",
        "isApiErrorMessage": True,
        "error": "server_error",
        "apiErrorStatus": 529,
        "message": {
            "id": f"e{i}",
            "model": "<synthetic>",
            "content": [{"type": "text", "text": "API Error: 529 overloaded"}],
        },
    }


class Env:
    def __init__(self, tmp: Path, monkeypatch):
        self.tmp = tmp
        self.root = tmp / "o11y_run"
        self.claude = tmp / "claude"
        self.pool = tmp / "pool"
        self.forge = tmp / "forge"
        monkeypatch.setenv("AGENT_SYS_O11Y_FLEET_DIR", str(self.root))
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(self.claude))
        monkeypatch.setenv(
            "AGENT_SYS_O11Y_OTEL_DIR", str(tmp / "otel")
        )  # never the host's real collector dir
        monkeypatch.delenv("AGENT_SYS_O11Y", raising=False)
        self.cwd = "/work/proj"
        self.transcript = (
            self.claude
            / "projects"
            / claude_sessions.project_dir_name(self.cwd)
            / f"{MAIN_SID}.jsonl"
        )
        self.transcript.parent.mkdir(parents=True)
        shutil.copy(FIX / "transcript_small.jsonl", self.transcript)
        self.set_session(MAIN_SID, os.getpid(), status="busy")
        shutil.copytree(FIX / "pool", self.pool)
        shutil.copytree(FIX / "forge", self.forge)
        api.session_register(
            api.Context(self.root, MAIN_SID, "main"), name="proj-main", role="main"
        )
        self.config({})

    def set_session(
        self, sid, pid, status="idle", since_ms=None, waiting_for=None, name=None, cwd=None
    ):
        rec = {
            "pid": pid,
            "sessionId": sid,
            "cwd": cwd or self.cwd,
            "name": name or "proj-main",
            "status": status,
            "waitingFor": waiting_for,
            "statusUpdatedAt": since_ms or int(time.time() * 1000),
            "procStart": claude_sessions.proc_start(pid) or "123",
            "kind": "interactive",
            "tmux": "s:@1.%1",
            "messagingSocketPath": f"/run/user/{os.getuid()}/cc-socks/{pid}.sock",
        }
        p = self.claude / "sessions" / f"{pid}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec))

    def config(self, over):
        doc = {
            "v": 1,
            "sources": {"machine_pool_dir": str(self.pool), "campaign_globs": [str(self.forge)]},
        }
        for k, v in over.items():
            doc.setdefault(k, {}).update(v) if isinstance(v, dict) else doc.__setitem__(k, v)
        assert schemas.problems("config.analysis.v1", doc) == []
        store.write_json(self.root / "config" / "analysis.json", doc)

    def run(self, kind="all", llm=False, now=None):
        if kind == "log":
            return log_cli.run(self.root, use_llm=llm, now=now, sweep=False)
        if kind == "display":
            return facts_cli.run(self.root, use_llm=llm, now=now, sweep=False)
        rl = log_cli.run(self.root, use_llm=llm, now=now, sweep=False)
        rf = facts_cli.run(self.root, use_llm=llm, now=now, sweep=False)
        return {
            "fleet": rl.get("fleet"),
            "main_gone": rl.get("main_gone") or rf.get("main_gone"),
            "log": rl,
            "display": rf,
        }

    def events(self):
        return [e for _, e, _ in store.read_jsonl(self.root / store.EVENTS) if e]

    def names(self):
        return [e["event_name"] for e in self.events()]


@pytest.fixture
def env(tmp_path, monkeypatch):
    return Env(tmp_path, monkeypatch)


# ------------------------------------------------------------------------------------------ cursor
def test_cursor_incremental_partial_and_reset(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_bytes(b'{"a":1}\n{"a":2}\n{"a":3')  # last line incomplete
    c = cur_mod.read_new(cur_mod.Cursor(path=str(p)))
    assert [json.loads(r)["a"] for _, r in c.lines] == [1, 2] and not c.reset
    with open(p, "ab") as f:
        f.write(b'}\n{"a":4}\n')
    c2 = cur_mod.read_new(c.cursor)
    assert [json.loads(r)["a"] for _, r in c2.lines] == [3, 4] and not c2.reset
    # rewrite of an earlier line with the same size: sha mismatch -> reset, full rescan
    data = p.read_bytes().replace(b'{"a":4}', b'{"a":5}')
    p.write_bytes(data)
    c3 = cur_mod.read_new(c2.cursor)
    assert c3.reset and len(c3.lines) == 4
    # truncation -> reset
    p.write_bytes(b'{"a":9}\n')
    c4 = cur_mod.read_new(c3.cursor)
    assert c4.reset and [json.loads(r)["a"] for _, r in c4.lines] == [9]
    # replaced file (new inode) -> reset
    q = tmp_path / "g.jsonl"
    q.write_bytes(b'{"a":7}\n{"a":8}\n')
    os.replace(q, p)
    c5 = cur_mod.read_new(c4.cursor)
    assert c5.reset and len(c5.lines) == 2
    # missing file
    assert cur_mod.read_new(cur_mod.Cursor(path=str(tmp_path / "nope"))).missing


def test_cursor_store_roundtrip(tmp_path):
    s = cur_mod.CursorStore(tmp_path)
    c = cur_mod.Cursor(
        path="/x", inode=5, offset=10, last_start=3, last_sha="ab", extra={"usage": {"input": 1}}
    )
    s.put("k", c)
    assert s.get("k", "/x") == c
    assert s.get("k", "/moved").offset == 0  # path changed -> fresh cursor, keeps extra
    assert s.get("other", "/x").offset == 0


# ------------------------------------------------------------------------------------------ transcripts
def test_usage_dedup_by_message_id_and_synthetic_excluded(tmp_path):
    lines = [
        _assistant("m1"),
        _assistant("m1"),
        _assistant("m2", inp=1, cc=0, cr=0, out=1),
        _api_error(1),
        {"type": "user", "message": {"content": "hi"}},
    ]
    raw = [(i, json.dumps(d).encode()) for i, d in enumerate(lines)]
    extra = {}
    probs = Problems()
    ing = tr.ingest_lines("t.jsonl", raw, extra, probs)
    assert extra["usage"] == {"input": 11, "cache_creation": 20, "cache_read": 30, "output": 6}
    assert extra["messages"] == 2 and ing.api_error_streak == 1 and not probs.items
    # the same message id continuing in the next chunk is still deduplicated
    ing2 = tr.ingest_lines(
        "t.jsonl",
        [(99, json.dumps(_assistant("m2", inp=1, cc=0, cr=0, out=1)).encode())],
        extra,
        probs,
    )
    assert extra["usage"]["input"] == 11 and ing2.usage_delta["input"] == 0


def test_transcript_missing_fields_reported_once(tmp_path):
    bad = {"type": "assistant", "message": {"model": "claude-x", "usage": {"input_tokens": 1}}}
    raw = [(i, json.dumps(bad).encode()) for i in range(3)] + [(9, b'{"no_type": 1}')]
    probs = Problems()
    tr.ingest_lines("t.jsonl", raw, {}, probs)
    fields = sorted(p.field for p in probs.items)
    assert fields == ["message.id", "type"]


def test_real_transcript_fixture_totals_match_independent_count():
    raw = [
        (i, ln)
        for i, ln in enumerate((FIX / "transcript_small.jsonl").read_bytes().split(b"\n"))
        if ln
    ]
    extra = {}
    tr.ingest_lines("t", raw, extra, Problems())
    seen, tot = set(), 0
    for _, line in raw:
        d = json.loads(line)
        m = d.get("message") if d.get("type") == "assistant" else None
        if not m or m.get("model") == "<synthetic>" or m.get("id") in seen or not m.get("usage"):
            continue
        seen.add(m["id"])
        tot += sum(m["usage"].get(k, 0) for k in tr.USAGE_KEYS)
    assert tr.total(extra["usage"]) == tot > 0


def test_condense_offsets_and_budget():
    raw = [
        (i * 100, json.dumps(d).encode())
        for i, d in enumerate(
            [
                {
                    "type": "user",
                    "message": {"content": "start the bench"},
                    "timestamp": "2026-11-23T10:00:00Z",
                },
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "tool_use", "name": "Bash", "input": {"command": "run.sh"}}
                        ]
                    },
                },
                {"type": "system", "subtype": "away_summary", "content": "recap text"},
                {"type": "queue-operation"},
            ]
        )
    ]
    text, end = tr.condense(raw, 10_000)
    assert "@0 " in text and "TOOL Bash: run.sh" in text and "RECAP: recap text" in text
    text2, end2 = tr.condense(raw, 40)
    assert text2.count("\n") == 0 and end2 <= 100 + len(raw[0][1]) + 1


# ------------------------------------------------------------------------------------------ end-to-end, no LLM
def _valid_facts(root):
    for t in ("general", "detail", "kernelforge", "usage"):
        doc = store.read_json(root / "facts" / f"{t}.json")
        assert doc is not None, t
        assert schemas.problems(f"facts.{t}.v1", doc) == [], (
            t,
            schemas.problems(f"facts.{t}.v1", doc),
        )


def test_first_run_baseline_writes_valid_facts_and_no_history_events(env):
    rep = env.run()
    assert rep["fleet"] == [MAIN_SID], rep
    _valid_facts(env.root)
    names = env.names()
    assert "o11y.session.started" not in names and "o11y.tokens.crossed" not in names
    # the fixture transcript contains real compactions -> reported (they are facts, not history noise)
    assert names.count("o11y.compaction.done") >= 1
    for e in env.events():
        assert schemas.problems("event.v1", e) == []
        assert e["writer"]["role"] == "analyzer"
    u = store.read_json(env.root / "facts" / "usage.json")
    assert u["fleet"]["tokens"]["new"] > 0 and u["fleet"]["tokens"]["cache_read"] > 0
    assert u["sessions"][0]["session_id"] == MAIN_SID
    k = store.read_json(env.root / "facts" / "kernelforge.json")
    assert k["campaigns"] and k["campaigns"][0]["operator_ids"] == ["mhc_seam_bf16_decode"]


def test_detectors_compaction_api_burst_tokens_idempotent(env):
    env.config({"detectors": {"token_step": 1000, "api_error_repeat": 3}})
    env.run()
    n0 = len(env.events())
    _w(
        env.transcript,
        [
            {
                "type": "system",
                "subtype": "compact_boundary",
                "uuid": "cb-new",
                "timestamp": "2026-11-23T10:00:00Z",
                "compactMetadata": {"trigger": "manual", "preTokens": 900, "postTokens": 100},
            },
            _assistant("big1", inp=5000, cc=0, cr=0, out=0),
            _api_error(1),
            _api_error(2),
            _api_error(3),
        ],
    )
    env.run()
    new = env.events()[n0:]
    names = [e["event_name"] for e in new]
    assert names.count("o11y.compaction.done") == 1
    assert "o11y.api.error_repeated" in names
    assert names.count("o11y.tokens.crossed") == 2  # session + fleet
    burst = next(e for e in new if e["event_name"] == "o11y.api.error_repeated")
    assert burst["severity_text"] == "ERROR" and burst["attributes"]["evidence"][0]["path"] == str(
        env.transcript
    )
    n1 = len(env.events())
    env.run()
    assert len(env.events()) == n1  # nothing new -> nothing reported


def test_cursor_reset_does_not_duplicate(env):
    env.run()
    n0 = len(env.events())
    data = env.transcript.read_bytes()
    env.transcript.write_bytes(data)  # same content, new inode via rewrite below
    tmp = env.transcript.with_suffix(".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, env.transcript)
    env.run()
    new = env.events()[n0:]
    assert [e["event_name"] for e in new] == [
        "o11y.protocol.history_reset"
    ]  # recorded once, nothing re-reported
    env.run()
    assert len(env.events()) == n0 + 1


def test_permission_wait_and_idle(env):
    old = int((time.time() - 1500) * 1000)
    env.set_session(
        MAIN_SID, os.getpid(), status="waiting", waiting_for="permission prompt", since_ms=old
    )
    env.run()
    ev = [e for e in env.events() if e["event_name"] == "o11y.session.permission_wait"]
    assert len(ev) == 1 and ev[0]["severity_text"] == "WARN" and "25 min" in ev[0]["body"]
    env.run()
    assert len([e for e in env.events() if e["event_name"] == "o11y.session.permission_wait"]) == 1
    env.set_session(MAIN_SID, os.getpid(), status="idle", since_ms=int((time.time() - 3600) * 1000))
    env.run()
    assert "o11y.session.idle" in env.names()


def test_child_crash_and_start(env):
    env.run()
    child_sid, dead_pid = "sess-child-0002", 4_000_000  # above pid_max: never alive
    api.session_register(api.Context(env.root, MAIN_SID, "main"), name="proj-kf", role="child")
    env.set_session(child_sid, dead_pid, name="proj-kf", since_ms=int((time.time() - 600) * 1000))
    env.run()
    crash = [e for e in env.events() if e["event_name"] == "o11y.session.crashed"]
    assert (
        len(crash) == 1
        and crash[0]["attributes"]["session.id"] == child_sid
        and crash[0]["severity_text"] == "ERROR"
    )
    assert (
        "o11y.session.started" not in env.names()
    )  # a session already dead is reported only as crashed
    env.run()
    assert len([e for e in env.events() if e["event_name"] == "o11y.session.crashed"]) == 1


def test_pool_events_only_for_fleet(env):
    env.run()  # first sight of the outbox = baseline
    me = f"uds:/run/user/{os.getuid()}/cc-socks/{os.getpid()}.sock"
    _w(
        env.pool / "outbox.jsonl",
        [
            {
                "ts": "2026-11-23T12:00:00Z",
                "req": "r-1",
                "type": "granted",
                "msg": "job 1 RUNNING on node-a",
                "notify": me,
                "cluster": "c",
            },
            {
                "ts": "2026-11-23T12:05:00Z",
                "req": "r-1",
                "type": "lost",
                "msg": "job 1 preempted",
                "notify": me,
                "cluster": "c",
            },
            {
                "ts": "2026-11-23T12:05:00Z",
                "req": "r-9",
                "type": "lost",
                "msg": "someone else",
                "notify": "uds:/x/1.sock",
                "cluster": "c",
            },
            {"ts": "2026-11-23T12:06:00Z", "type": "lost"},
        ],
    )  # missing req -> schema mismatch
    env.run()
    ev = env.events()
    assert [e["event_name"] for e in ev if e["event_name"].startswith("o11y.machine.")] == [
        "o11y.machine.granted",
        "o11y.machine.lost",
    ]
    lost = next(e for e in ev if e["event_name"] == "o11y.machine.lost")
    assert lost["severity_text"] == "WARN"
    mm = [e for e in ev if e["event_name"] == "o11y.source.schema_mismatch"]
    assert any(
        e["attributes"]["o11y.field"] == "req"
        and e["attributes"]["o11y.path"].endswith("outbox.jsonl")
        for e in mm
    )


def test_forge_stale_and_schema_mismatch(env):
    ev_file = env.forge / "decode" / "forge_experiments" / "events.jsonl"
    old = time.time() - 7200
    os.utime(ev_file, (old, old))
    env.run()
    stale = [e for e in env.events() if e["event_name"] == "o11y.campaign.stale"]
    assert len(stale) == 1
    k = store.read_json(env.root / "facts" / "kernelforge.json")
    assert k["campaigns"][0]["status"] == "stale" and k["campaigns"][0]["problem"]
    rs = env.forge / "decode" / "forge_experiments" / "run_state.json"
    d = json.loads(rs.read_text())
    del d["round_costs"]
    rs.write_text(json.dumps(d))
    env.run()
    mm = [e for e in env.events() if e["event_name"] == "o11y.source.schema_mismatch"]
    assert any(
        e["attributes"]["o11y.field"] == "round_costs.campaign_sec"
        and e["attributes"]["o11y.path"] == str(rs)
        for e in mm
    )
    assert all(e["severity_text"] == "ERROR" for e in mm)


def test_ledger_driven_facts(env):
    ctx = api.Context(env.root, MAIN_SID, "main", session_name="proj-main")
    assert api.ledger_append(
        ctx,
        {
            "kind": "task",
            "task_id": "t1",
            "title": "maximise tput",
            "metric": {"name": "tput", "unit": "tok/s/gpu", "direction": "higher"},
            "baseline": {"value": 1000},
            "target": {"value": 1500},
        },
    )["ok"]
    assert api.ledger_append(
        ctx,
        {
            "kind": "run_end",
            "run_id": "r1",
            "status": "kept",
            "value": 1200,
            "theme": "mem 0.9",
            "evidence": {"path": "/x/r1.json"},
        },
    )["ok"]
    assert api.ledger_append(
        ctx,
        {
            "kind": "run_start",
            "run_id": "r2",
            "theme": "decode tp8",
            "point": "explore",
            "machines": [{"node": "node-a"}],
        },
    )["ok"]
    assert api.ledger_append(
        ctx, {"kind": "plan", "task_id": "t1", "next": [{"theme": "try mtp", "why": "accept 0.7"}]}
    )["ok"]
    env.run()
    _valid_facts(env.root)
    g = store.read_json(env.root / "facts" / "general.json")
    assert (
        g["task"]["task_id"] == "t1"
        and g["best"]["value"] == 1200
        and g["best"]["of_target_pct"] == 40.0
    )
    assert g["progress"]["gain_abs"] == 200 and g["progress"]["gain_pct"] == 20.0
    d = store.read_json(env.root / "facts" / "detail.json")
    assert d["current_runs"][0]["run_id"] == "r2" and d["next"][0]["theme"] == "try mtp"
    assert d["recent_runs"][0]["run_id"] == "r1"
    assert any(m["node"] == "node-a" and m["source"] == "ledger" for m in d["machines"])


# ------------------------------------------------------------------------------------------ CLI: lock + queue
def _queue_item(root, kind):
    item = {
        "v": 1,
        "id": f"q-{kind}-{time.time_ns()}",
        "ts": store.now(),
        "kind": kind,
        "reason": "manual",
        "writer": {"session_id": MAIN_SID, "role": "main", "source": "written"},
    }
    assert schemas.problems("queue.v1", item) == []
    p = root / "queue" / f"{time.time_ns()}-{kind}.json"
    store.write_json(p, item)
    return p


def test_queue_claimed_per_kind_drained_and_lock_coalesces(env):
    pl, pd = _queue_item(env.root, "log"), _queue_item(env.root, "display")
    rep = env.run(kind="display")
    assert len(rep["claimed"]) == 1 and pl.exists() and not pd.exists()
    assert not list((env.root / "queue" / "claimed").glob("*display.json"))  # drained items removed
    lockf = env.root / "locks" / "analyze-log.lock"
    with open(lockf, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        rep2 = env.run(kind="log")
        assert rep2.get("skipped")
    assert pl.exists()


def test_cli_main_writes_report(env, tmp_path, capsys):
    rc = facts_cli.main(
        [
            "--fleet-dir",
            str(env.root),
            "--no-sweep",
            "--no-llm",
            "--report",
            str(tmp_path / "r.json"),
        ]
    )
    assert rc == 0
    rep = json.loads((tmp_path / "r.json").read_text())
    assert rep["kind"] == "display" and rep["cost_usd"] == 0
    assert (env.root / "state" / "facts" / "runs.jsonl").exists()


# ------------------------------------------------------------------------------------------ LLM pass (fake claude)
FAKE = r"""#!/usr/bin/env python3
import json, os, sys
prompt = sys.stdin.read()
here = os.path.dirname(os.path.abspath(__file__))
open(os.path.join(here, "fake.log"), "a").write(json.dumps({"argv": sys.argv[1:], "len": len(prompt),
    "thinking_off": os.environ.get("CLAUDE_CODE_DISABLE_THINKING"), "env": sorted(os.environ)}) + "\n")
mode = open(os.path.join(here, "fake.mode")).read().strip() if os.path.exists(os.path.join(here, "fake.mode")) else "ok"
if mode == "error":
    print(json.dumps([{"type": "result", "subtype": "error_during_execution", "is_error": True, "result": "API Error: 529",
                       "total_cost_usd": 0.001}]))
    sys.exit(1)
import re
offs = [int(x) for x in re.findall(r"^@(\d+) ", prompt, re.M)]
m = re.search(r"### session (\S+)", prompt)
sid = m.group(1) if m else None
schema = json.loads(sys.argv[sys.argv.index("--json-schema") + 1])
if "events" in schema["properties"]:
    out = {"events": [{"severity": "INFO", "event_name": "run.finished", "body": "run r9 finished: 1234 tok/s",
                       "session_id": sid, "evidence_offsets": offs[-1:] + [10**12]}],
           "ledger": [{"kind": "run_end", "run_id": "r9", "status": "kept", "value": 1234, "theme": "fake",
                       "evidence": {"path": "/x/r9.json"}, "session_id": sid, "evidence_offsets": offs[-1:]},
                      {"kind": "run_end", "run_id": "r10", "status": "kept", "session_id": sid, "evidence_offsets": []}]}
else:
    out = {"task": {"title": "fake task", "metric_name": "tput", "unit": "tok/s", "direction": "higher",
                    "baseline": 1000, "target": 2000},
           "current_run": {"theme": "exploring fake", "point": "explore", "machines": ["n1"], "started_at": None},
           "recent_runs": [], "next_plan": [{"theme": "next fake", "why": "because"}], "evidence_offsets": offs[:1]}
print(json.dumps([{"type": "system"}, {"type": "result", "subtype": "success", "is_error": False,
                   "structured_output": out, "total_cost_usd": 0.0021, "modelUsage": {"Claude-Haiku-4.5": {}}}]))
"""


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    p = tmp_path / "fake_claude.py"
    p.write_text(FAKE)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "fake.log"
    monkeypatch.setenv("O11Y_TEST_SECRET_NOT_FOR_CHILD", "x")  # must not reach the model process
    return p, log


def test_log_pass_events_ledger_and_cursor(env, fake_claude, monkeypatch):
    fake, log = fake_claude
    env.config({"claude_bin": str(fake)})
    rep = env.run(kind="log", llm=True)
    lp = rep["log_pass"][0]
    assert (
        lp["llm"]["ok"]
        and lp["events"] == 1
        and lp["ledger_added"] == 1
        and lp["ledger_rejected"] == 1
    ), lp
    assert rep["cost_usd"] == 0.0021
    ev = [e for e in env.events() if e["event_name"] == "o11y.run.finished"][0]
    assert ev["attributes"]["o11y.source"] == "llm" and ev["writer"]["source"] == "extracted"
    assert len(ev["attributes"]["evidence"]) == 1  # the out-of-slice offset was dropped
    recs, _ = api.read_ledger(env.root)
    r9 = next(r for r in recs if r.get("run_id") == "r9")
    assert (
        r9["writer"]["role"] == "analyzer"
        and r9["writer"]["source"] == "extracted"
        and r9["writer"]["session_id"] == MAIN_SID
    )
    call = json.loads(log.read_text().splitlines()[0])
    assert call["thinking_off"] == "1" and "--bare" in call["argv"]
    i = call["argv"].index("--tools")
    assert call["argv"][i + 1] == "" and "dontAsk" in call["argv"]  # no tools at all
    assert "O11Y_TEST_SECRET_NOT_FOR_CHILD" not in call["env"]  # stripped child env
    assert "AGENT_SYS_O11Y_FLEET_DIR" not in call["env"]
    # nothing new -> no second model call
    env.run(kind="log", llm=True)
    assert len(log.read_text().splitlines()) == 1
    # new content -> another call; r9 already in the ledger is not appended twice
    _w(
        env.transcript,
        [
            {
                "type": "user",
                "message": {"content": "next step"},
                "timestamp": "2026-11-23T11:00:00Z",
            }
        ],
    )
    rep3 = env.run(kind="log", llm=True)
    assert rep3["log_pass"][0]["ledger_added"] == 0
    assert (
        len([r for r in api.read_ledger(env.root, latest_only=False)[0] if r.get("run_id") == "r9"])
        == 1
    )


def test_log_pass_failure_keeps_cursor(env, fake_claude, monkeypatch):
    fake, log = fake_claude
    env.config({"claude_bin": str(fake)})
    (fake.parent / "fake.mode").write_text("error")
    rep = env.run(kind="log", llm=True)
    assert rep["log_pass"][0]["llm"]["ok"] is False and "o11y.analyzer.llm_failed" in env.names()
    (fake.parent / "fake.mode").write_text("ok")
    rep2 = env.run(kind="log", llm=True)
    assert (
        rep2["log_pass"][0]["slices"] == 1 and rep2["log_pass"][0]["llm"]["ok"]
    )  # same slice retried


def test_display_pass_fills_gaps_from_extraction(env, fake_claude):
    fake, _ = fake_claude
    env.config({"claude_bin": str(fake)})
    rep = env.run(kind="display", llm=True)
    assert rep["display_pass"][0]["llm"]["ok"], rep
    _valid_facts(env.root)
    g = store.read_json(env.root / "facts" / "general.json")
    d = store.read_json(env.root / "facts" / "detail.json")
    assert g["task"]["title"] == "fake task" and "extracted" in g["sources"]
    assert (
        d["current_runs"][0]["theme"] == "exploring fake" and d["next"][0]["theme"] == "next fake"
    )
    # once the ledger has task + open run + plan, the display LLM is not called again
    ctx = api.Context(env.root, MAIN_SID, "main")
    api.ledger_append(
        ctx,
        {
            "kind": "task",
            "task_id": "t1",
            "title": "real",
            "metric": {"name": "m", "unit": "u", "direction": "higher"},
            "baseline": {"value": 1},
        },
    )
    api.ledger_append(ctx, {"kind": "run_start", "run_id": "a", "theme": "t", "point": "explore"})
    api.ledger_append(ctx, {"kind": "plan", "task_id": "t1", "next": [{"theme": "p"}]})
    rep2 = env.run(kind="display", llm=True)
    assert rep2["display_pass"][0]["llm"] is None
    assert store.read_json(env.root / "facts" / "general.json")["task"]["title"] == "real"


# ------------------------------------------------------------------------------------------ OTel adapter (optional source)
@pytest.mark.parametrize("name", ["vector_native", "vector_flat", "otelcol_otlp"])
def test_otel_records_all_shapes(name):
    from o11y_sources import otel

    p = FIX / "otel" / f"{name}.jsonl"
    raw = [(i, ln) for i, ln in enumerate(p.read_bytes().split(b"\n")) if ln]
    probs = Problems()
    recs = otel.records(str(p), raw, probs)
    assert not probs.items
    assert recs and all("session.id" in r["attrs"] for r in recs)
    if name != "vector_flat":
        assert any(r["event"] == "api_request" and float(r["attrs"]["cost_usd"]) > 0 for r in recs)


def test_otel_cost_and_retries_into_facts(env):
    sid_lines = []
    for line in (FIX / "otel" / "vector_native.jsonl").read_text().splitlines():
        d = json.loads(line)
        d["attributes"]["session.id"] = MAIN_SID
        sid_lines.append(d)
    sid_lines.append(
        {
            "message": "claude_code.api_retries_exhausted",
            "attributes": {
                "event.name": "api_retries_exhausted",
                "session.id": MAIN_SID,
                "total_attempts": 3,
                "status_code": 503,
                "error": "fake 503",
            },
        }
    )
    _w_private(env.tmp / "otel" / "logs.jsonl", sid_lines)
    env.config({"sources": {"otel_dir": str(env.tmp / "otel")}})
    env.run()
    u = store.read_json(env.root / "facts" / "usage.json")
    assert u["cost_usd"][MAIN_SID] > 0 and "otel" in u["sources"]
    ex = [e for e in env.events() if e["event_name"] == "o11y.api.retries_exhausted"]
    assert len(ex) == 1 and ex[0]["attributes"]["o11y.source"] == "otel"


# ------------------------------------------------------------------------------------------ design v2 behaviour
def _named(env, name):
    return [e for e in env.events() if e["event_name"] == name]


def test_token_milestone_counts_new_tokens_not_cache_reads(env):
    env.config({"detectors": {"token_step": 1000}})
    env.run()
    _w(env.transcript, [_assistant("cr-only", inp=0, cc=0, cr=50_000, out=0)])  # cache reads only
    env.run()
    assert _named(env, "o11y.tokens.crossed") == []
    _w(env.transcript, [_assistant("big", inp=2500, cc=0, cr=0, out=0)])  # 2.5 steps of new tokens
    env.run()
    crossed = _named(env, "o11y.tokens.crossed")
    assert len(crossed) == 2  # one session + one fleet record
    assert {e["attributes"]["o11y.scope"] for e in crossed} == {"session", "fleet"}
    u = store.read_json(env.root / "facts" / "usage.json")
    s = u["sessions"][0]["tokens"]
    assert s["new"] == s["input"] + s["cache_creation"] + s["output"]
    g = store.read_json(env.root / "facts" / "general.json")
    assert g["spend"]["tokens_new"] == u["fleet"]["tokens"]["new"]
    assert g["spend"]["tokens_cache_read"] == u["fleet"]["tokens"]["cache_read"]


def test_fatal_api_error_on_first_occurrence(env):
    env.run()
    fatal = _api_error(1)
    fatal.update(error="billing_error", apiErrorStatus=400)
    quota = _api_error(2)
    quota.update(error="rate_limit", apiErrorStatus=429)
    quota["message"]["content"][0]["text"] = "API Error: 429 monthly usage limit reached"
    transient = _api_error(3)
    transient.update(error="rate_limit", apiErrorStatus=429)
    _w(
        env.transcript,
        [fatal, _assistant("ok1"), quota, _assistant("ok2"), transient, _assistant("ok3")],
    )
    env.run()
    errs = _named(env, "o11y.api.error")
    assert [e["attributes"]["o11y.error_kind"] for e in errs] == ["billing_error", "rate_limit"]
    assert all(e["severity_text"] == "ERROR" for e in errs)
    assert _named(env, "o11y.api.error_repeated") == []  # never 3 in a row
    env.run()
    assert len(_named(env, "o11y.api.error")) == 2


def test_pool_ended_reported_only_without_released(env):
    env.run()
    me = f"uds:/run/user/{os.getuid()}/cc-socks/{os.getpid()}.sock"
    _w(
        env.pool / "outbox.jsonl",
        [
            {
                "ts": "2026-11-23T12:00:00Z",
                "req": "r-a",
                "type": "ended",
                "msg": "job ended",
                "notify": me,
            },
            {
                "ts": "2026-11-23T12:00:05Z",
                "req": "r-a",
                "type": "released",
                "msg": "released",
                "notify": me,
            },
            {
                "ts": "2026-11-23T12:00:00Z",
                "req": "r-b",
                "type": "ended",
                "msg": "job ended",
                "notify": me,
            },
            {
                "ts": "2026-11-23T12:00:09Z",
                "req": "r-b",
                "type": "done",
                "msg": "done",
                "notify": me,
            },
        ],
    )
    env.run()
    rel = _named(env, "o11y.machine.released")
    assert [e["attributes"]["o11y.request_id"] for e in rel] == ["r-a"]  # r-b held back
    env.run(now=time.time() + 700)  # grace elapsed, no `released`
    rel = _named(env, "o11y.machine.released")
    assert [e["attributes"]["o11y.request_id"] for e in rel] == ["r-a", "r-b"]
    env.run(now=time.time() + 1400)
    assert len(_named(env, "o11y.machine.released")) == 2


def test_schema_mismatch_once_until_the_file_changes(env):
    rs = env.forge / "decode" / "forge_experiments" / "run_state.json"
    d = json.loads(rs.read_text())
    del d["best"]
    rs.write_text(json.dumps(d))
    env.run()
    env.run()
    mm = [
        e
        for e in _named(env, "o11y.source.schema_mismatch")
        if e["attributes"]["o11y.path"] == str(rs)
    ]
    assert len(mm) == 1 and mm[0]["attributes"]["o11y.field"] == "best.mean_case_speedup"
    d["iteration"] = "three"  # still broken, and the file changed -> reported again
    rs.write_text(json.dumps(d))
    env.run()
    mm = [
        e
        for e in _named(env, "o11y.source.schema_mismatch")
        if e["attributes"]["o11y.path"] == str(rs)
    ]
    assert len(mm) == 3 and {e["attributes"]["o11y.field"] for e in mm[1:]} == {
        "best.mean_case_speedup",
        "iteration",
    }
    k = store.read_json(env.root / "facts" / "kernelforge.json")
    assert k["campaigns"] == []  # the source is skipped; the rest of the facts still built


def test_invalid_protocol_lines_reported_when_they_appear(env):
    env.run()
    (env.root / store.LEDGER).parent.mkdir(parents=True, exist_ok=True)
    with open(env.root / store.LEDGER, "a") as f:
        f.write('{"kind": "run_end"}\n')  # valid JSON, invalid record
    env.run()
    env.run()
    inv = _named(env, "o11y.protocol.invalid_line")
    assert len(inv) == 1 and inv[0]["attributes"]["o11y.count"] == 1


def test_waiting_on_sandbox_request_counts_as_permission_wait(env):
    old = int((time.time() - 900) * 1000)
    env.set_session(
        MAIN_SID, os.getpid(), status="waiting", waiting_for="sandbox request", since_ms=old
    )
    env.run()
    w = _named(env, "o11y.session.permission_wait")
    assert len(w) == 1 and w[0]["attributes"]["o11y.waiting_for"] == "sandbox request"
    env.set_session(
        MAIN_SID, os.getpid(), status="waiting", waiting_for="dialog open", since_ms=old - 1000
    )
    env.run()
    assert len(_named(env, "o11y.session.permission_wait")) == 1


def test_dead_main_reported_by_next_run_and_no_model_call(env, fake_claude):
    fake, log = fake_claude
    env.config({"claude_bin": str(fake)})
    env.run()
    dead = 4_000_001
    (env.claude / "sessions" / f"{os.getpid()}.json").rename(
        env.claude / "sessions" / f"{dead}.json"
    )
    rec = json.loads((env.claude / "sessions" / f"{dead}.json").read_text())
    rec.update(pid=dead, statusUpdatedAt=int((time.time() - 600) * 1000))
    (env.claude / "sessions" / f"{dead}.json").write_text(json.dumps(rec))
    rep = env.run(kind="all", llm=True)
    assert rep.get("main_gone") and not log.exists()  # exits early: no model call for a dead main
    crash = _named(env, "o11y.session.crashed")
    assert len(crash) == 1 and crash[0]["attributes"]["session.role"] == "main"


def test_queue_items_arriving_during_a_run_are_served(env, monkeypatch):
    _queue_item(env.root, "display")
    from o11y_facts import facts as facts_mod

    real = facts_mod.build
    calls = []

    def build_and_enqueue(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            _queue_item(env.root, "display")  # a request made while this analyzer runs
        return real(*a, **k)

    monkeypatch.setattr(facts_mod, "build", build_and_enqueue)
    rep = env.run(kind="display")
    assert rep["passes"] == 2 and len(rep["claimed"]) == 2
    assert not list((env.root / "queue").glob("*.json"))


def test_model_cursor_first_look_is_bounded_and_reset_jumps_to_end(env, fake_claude):
    fake, log = fake_claude
    env.config(
        {"claude_bin": str(fake), "log": {"lookback_bytes": 3000, "max_slice_bytes": 200000}}
    )
    env.run(kind="log", llm=True)
    first = json.loads(log.read_text().splitlines()[0])
    size = env.transcript.stat().st_size
    assert first["len"] < 3000 + 20000 and size > 100_000  # only the tail was read
    data = env.transcript.read_bytes()
    tmp = env.transcript.with_suffix(".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, env.transcript)  # same content, new inode
    env.run(kind="log", llm=True)
    assert len(log.read_text().splitlines()) == 1  # not re-read by the model
    _w(
        env.transcript,
        [
            {
                "type": "user",
                "message": {"content": "new work after the reset"},
                "timestamp": "2026-11-23T12:00:00Z",
            }
        ],
    )
    env.run(kind="log", llm=True)
    calls = log.read_text().splitlines()
    assert len(calls) == 2 and json.loads(calls[1])["len"] < 20000


def test_otel_rotated_files_are_not_reread(env):
    odir = env.tmp / "otel"
    lines = []
    for line in (FIX / "otel" / "vector_native.jsonl").read_text().splitlines():
        d = json.loads(line)
        d["attributes"]["session.id"] = MAIN_SID
        lines.append(d)
    _w_private(odir / "logs.jsonl", lines)
    env.run()
    c1 = store.read_json(env.root / "facts" / "usage.json")["cost_usd"][MAIN_SID]
    os.rename(
        odir / "logs.jsonl", odir / "logs-2026-11-23T12-00-00.000.jsonl"
    )  # collector rotation
    _w_private(odir / "logs.jsonl", lines[:1])
    env.run()
    c2 = store.read_json(env.root / "facts" / "usage.json")["cost_usd"][MAIN_SID]
    first = float(lines[0]["attributes"].get("cost_usd") or 0)
    assert abs(c2 - (c1 + first)) < 1e-3  # facts round to 4 decimals; a re-read would double c1


# ------------------------------------------------------------------------------------------ security rules
def test_pool_attribution_rule():
    from o11y_sources import pool

    class S:
        def __init__(self, name, sock):
            self.name, self.messaging_socket = name, sock

    me = os.getuid()
    a = S("proj-main", f"/run/user/{me}/cc-socks/111.sock")
    fleet = [a]
    assert pool.attribute(f"uds:/run/user/{me}/cc-socks/111.sock", None, fleet) is a
    assert (
        pool.attribute(f"uds:/run/user/{me + 1}/cc-socks/111.sock", None, fleet) is None
    )  # other uid
    assert (
        pool.attribute(f"uds:/run/user/{me}/other/111.sock", None, fleet) is None
    )  # trailing pid only
    user = pool.current_user()
    assert pool.attribute("proj-main", user, fleet) is a
    assert pool.attribute("proj-main", "someone-else", fleet) is None  # another user's request
    assert pool.attribute("proj-main", None, fleet) is None


def test_extracted_campaign_root_must_be_allowed_real_and_deep(env, tmp_path):
    from o11y_sources import forge

    good = str(env.forge / "decode")
    assert forge.campaign_root_problem(good, [str(env.forge)]) is None
    assert forge.campaign_root_problem("/", [str(env.forge)]) == "too shallow"
    assert forge.campaign_root_problem("relative/dir", [str(env.forge)]) == "not an absolute path"
    assert (
        forge.campaign_root_problem(good, [str(tmp_path / "elsewhere")])
        == "outside the allowed campaign roots"
    )
    empty = tmp_path / "a" / "b" / "c"
    empty.mkdir(parents=True)
    assert (
        forge.campaign_root_problem(str(empty), [str(tmp_path)])
        == "no forge_experiments/run_state.json under it"
    )


def test_log_pass_rejects_extracted_campaign_outside_allowlist(env, fake_claude, tmp_path):
    fake, log = fake_claude
    script = fake.read_text().replace(
        '"ledger": [{"kind": "run_end"',
        '"ledger": [{"kind": "campaign", "campaign_root": "/etc/ssl/certs", "operator_ids": ["x"], '
        '"started_at": "2026-11-23T10:00:00Z", "machine": {"node": "n"}, "session_id": sid, '
        '"evidence_offsets": []}, {"kind": "run_end"',
    )
    fake.write_text(script)
    env.config({"claude_bin": str(fake)})
    rep = env.run(kind="log", llm=True)
    rej = rep["log_pass"][0].get("rejections") or []
    assert any(r["kind"] == "campaign" for r in rej)
    assert not [r for r in api.read_ledger(env.root)[0] if r.get("kind") == "campaign"]


def test_otel_dir_not_private_is_reported_and_skipped(env):
    odir = env.tmp / "otel"
    _w(
        odir / "logs.jsonl",
        [{"attributes": {"event.name": "api_request", "session.id": MAIN_SID, "cost_usd": 1}}],
    )
    os.chmod(odir, 0o777)
    env.config({"sources": {"otel_dir": str(odir)}})
    env.run()
    mm = [
        e
        for e in env.events()
        if e["event_name"] == "o11y.source.schema_mismatch"
        and e["attributes"]["o11y.source_kind"] == "otel"
    ]
    assert mm and "not trusted" in mm[0]["body"]
    u = store.read_json(env.root / "facts" / "usage.json")
    assert "cost_usd" not in u


def test_otel_metric_and_trace_files_are_not_log_shape_errors(tmp_path):
    from o11y_sources import otel

    raw = [(0, b'{"resourceMetrics": []}'), (30, b'{"resourceSpans": []}')]
    probs = Problems()
    assert otel.records("x.jsonl", raw, probs) == [] and not probs.items
