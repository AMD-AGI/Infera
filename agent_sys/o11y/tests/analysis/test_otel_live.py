"""Criteria 21 and 30 end to end, live: real `claude` (haiku, a few calls), real collector container, ONLY local
stand-ins upstream (nothing leaves the host). Skipped unless O11Y_LIVE=1. Needs docker with the
otel/opentelemetry-collector-contrib image already present (`--pull never`) and the gateway credentials in
~/.claude/settings.json (read, never written). Every container this module starts has a test-only name and is
stopped and removed at the end; nothing outside the pytest temp dir is written.

- 21: forwarded content == what the session sent (decoded); collector down at launch → direct export; a
  sender without the bearer token is refused.
- 30: a non-fleet session's export is untouched; privacy switches unchanged; a killed collector is restarted
  by its restart policy and the outage reaches `o11y.otel.collector_down` via otel/health.json + the log job.
- The analysis consumes the OTel copy in a live fleet: `api_retries_exhausted` from a session pointed at a
  local 503 server becomes `o11y.api.retries_exhausted`.
"""

import http.server
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest
from analysis_paths import PLUGIN
from o11y_job import run_consumer
from o11y_log import handle as log_handle
from o11y_protocol import store

pytestmark = pytest.mark.skipif(os.environ.get("O11Y_LIVE") != "1", reason="live: set O11Y_LIVE=1")

LAUNCH = PLUGIN / "bin" / "o11y-launch"
IMAGE = "otel/opentelemetry-collector-contrib:latest"
UID = os.getuid()
SUFFIX = "live-test"
CONTAINER = f"infera-agent-sys-o11y-otelcol-{UID}-{SUFFIX}"
DECODER = f"infera-agent-sys-o11y-otelcol-{UID}-live-decoder"
BASE = 36100 + (UID % 100) * 8  # collector ports BASE..BASE+3; stand-ins above
SECRET_PROMPT = "zebra-4471 say only the word ok"
GATEWAY_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
)


class Upstream:
    """Stand-in for the user's OTLP endpoint: saves each request body, its path and its headers; replies 200."""

    def __init__(self, port: int):
        self.posts: list[dict] = []
        up = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("content-length") or 0))
                up.posts.append(
                    {
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": body,
                        "t": time.time(),
                    }
                )
                self.send_response(200)
                self.send_header("content-type", "application/x-protobuf")
                self.end_headers()

            def log_message(self, *a):
                pass

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


class Fake503:
    """A model endpoint that always answers 503 (to produce `api_retries_exhausted`)."""

    def __init__(self, port: int):
        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length") or 0))
                body = (
                    b'{"type":"error","error":{"type":"overloaded_error","message":"stand-in 503"}}'
                )
                self.send_response(503)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


def _gateway() -> dict:
    env = json.loads((Path.home() / ".claude" / "settings.json").read_text()).get("env", {})
    return {k: env[k] for k in GATEWAY_KEYS if env.get(k)}


def _docker(*args, check=False):
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=120, check=check
    )


def _remove(name: str):
    """Stop and remove a container this module started (exact test-only name)."""
    assert name.startswith(f"infera-agent-sys-o11y-otelcol-{UID}-live")
    _docker("stop", name)
    _docker("rm", name)


class Live:
    def __init__(self, root: Path):
        self.root = root
        self.cc = root / "cc"  # isolated Claude config dir: its settings point OTel at the stand-in
        self.work = root / "work"
        self.odir = root / "otel"
        self.fleet = root / "fleet"
        for d in (self.cc, self.work):
            d.mkdir(parents=True, exist_ok=True)
        self.up_port, self.dec_port, self.fake_port = BASE + 4, BASE + 5, BASE + 6
        (self.cc / "settings.json").write_text(
            json.dumps(
                {
                    "env": {
                        "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
                        "OTEL_METRICS_EXPORTER": "otlp",
                        "OTEL_LOGS_EXPORTER": "otlp",
                        "OTEL_TRACES_EXPORTER": "otlp",
                        "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1",
                        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
                        "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{self.up_port}",
                        "OTEL_EXPORTER_OTLP_HEADERS": "x-test-token=abc123",
                        "OTEL_METRIC_EXPORT_INTERVAL": "2000",
                        "OTEL_LOGS_EXPORT_INTERVAL": "1000",
                        "OTEL_TRACES_EXPORT_INTERVAL": "1000",
                        "OTEL_LOG_USER_PROMPTS": "0",
                        "OTEL_LOG_TOOL_DETAILS": "0",
                    }
                }
            )
        )
        self.up = Upstream(self.up_port)
        store.write_json(
            self.fleet / "config" / "analysis.json",
            {"v": 1, "sources": {"otel_dir": str(self.odir)}},
        )

    def env(self, **extra) -> dict:
        return {
            "HOME": str(Path.home()),
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            **_gateway(),
            "CLAUDE_CONFIG_DIR": str(self.cc),
            "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "AGENT_SYS_O11Y_OTEL_DIR": str(self.odir),
            "AGENT_SYS_O11Y_OTEL_PORT_BASE": str(BASE),
            "AGENT_SYS_O11Y_OTEL_CONTAINER": SUFFIX,
            "AGENT_SYS_O11Y_OTEL_IMAGE": IMAGE,
            "AGENT_SYS_O11Y_CLAUDE_BIN": str(Path.home() / ".local" / "bin" / "claude"),
            **extra,
        }

    def claude(self, sid: str, launched: bool, **extra) -> subprocess.CompletedProcess:
        args = ["-p", SECRET_PROMPT, "--model", "haiku", "--session-id", sid]
        argv = (
            [str(LAUNCH), "--otel", "--", *args]
            if launched
            else [str(Path.home() / ".local/bin/claude"), *args]
        )
        p = subprocess.run(
            argv, cwd=self.work, env=self.env(**extra), capture_output=True, text=True, timeout=300
        )
        time.sleep(6)  # exporters flush at exit; the collector forwards as it receives
        return p

    def ours(self) -> list[dict]:
        """OTLP-JSON lines our collector's file exporters wrote (= what the sessions sent to it)."""
        out = []
        for f in sorted(self.odir.glob("*.jsonl")):
            out += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        return out

    def decode(self, posts: list[dict]) -> list[dict]:
        """Decode upstream request bodies with a throwaway collector (the reference decoder)."""
        out = self.root / "decoded"
        out.mkdir(mode=0o700, exist_ok=True)
        cfg = {
            "receivers": {
                "otlp": {"protocols": {"http": {"endpoint": f"127.0.0.1:{self.dec_port}"}}}
            },
            "exporters": {"file": {"path": "/out/decoded.jsonl"}},
            "service": {
                "telemetry": {"metrics": {"level": "none"}},
                "pipelines": {
                    s: {"receivers": ["otlp"], "exporters": ["file"]}
                    for s in ("logs", "metrics", "traces")
                },
            },
        }
        (out / "decoder.json").write_text(json.dumps(cfg))
        _docker(
            "run",
            "-d",
            "--name",
            DECODER,
            "--pull",
            "never",
            "--network",
            "host",
            "--user",
            f"{UID}:{os.getgid()}",
            "-v",
            f"{out}:/out",
            IMAGE,
            "--config",
            "/out/decoder.json",
            check=True,
        )
        try:
            for _ in range(60):
                try:
                    urllib.request.urlopen(
                        urllib.request.Request(
                            f"http://127.0.0.1:{self.dec_port}/v1/logs",
                            data=b"",
                            method="POST",
                            headers={"content-type": "application/x-protobuf"},
                        ),
                        timeout=1,
                    )
                    break
                except OSError:
                    time.sleep(0.5)
            for p in posts:
                urllib.request.urlopen(
                    urllib.request.Request(
                        f"http://127.0.0.1:{self.dec_port}{p['path']}",
                        data=p["body"],
                        method="POST",
                        headers={
                            "content-type": p["headers"].get(
                                "content-type", "application/x-protobuf"
                            )
                        },
                    ),
                    timeout=10,
                )
            time.sleep(3)
            f = out / "decoded.jsonl"
            return (
                [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
                if f.exists()
                else []
            )
        finally:
            _remove(DECODER)


def _attrs(lst) -> dict:
    return {a["key"]: next(iter(a.get("value", {}).values()), None) for a in lst or []}


def items(batches: list[dict], sid: str) -> list[str]:
    """Canonical (resource, scope, record) strings of one session's logs, spans and metric points."""
    out = []
    for b in batches:
        for kind, scopes_key, recs_key in (
            ("resourceLogs", "scopeLogs", "logRecords"),
            ("resourceSpans", "scopeSpans", "spans"),
        ):
            for r in b.get(kind, []):
                res = _attrs(r.get("resource", {}).get("attributes"))
                for sc in r.get(scopes_key, []):
                    for rec in sc.get(recs_key, []):
                        if _attrs(rec.get("attributes")).get("session.id") == sid:
                            out.append(
                                json.dumps(
                                    [res, sc.get("scope", {}).get("name"), rec], sort_keys=True
                                )
                            )
        for r in b.get("resourceMetrics", []):
            res = _attrs(r.get("resource", {}).get("attributes"))
            for sc in r.get("scopeMetrics", []):
                for m in sc.get("metrics", []):
                    data = next(
                        (v for k, v in m.items() if isinstance(v, dict) and "dataPoints" in v), {}
                    )
                    for dp in data.get("dataPoints", []):
                        if _attrs(dp.get("attributes")).get("session.id") == sid:
                            out.append(json.dumps([res, m["name"], dp], sort_keys=True))
    return sorted(out)


def _post_receiver(token: str | None) -> int:
    req = urllib.request.Request(
        f"http://127.0.0.1:{BASE}/v1/logs",
        data=b"",
        method="POST",
        headers={
            "content-type": "application/x-protobuf",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
    )
    try:
        return urllib.request.urlopen(req, timeout=5).status
    except urllib.error.HTTPError as e:
        return e.code


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    lv = Live(tmp_path_factory.mktemp("o11y_live"))
    yield lv
    lv.up.close()
    _remove(CONTAINER)


def test_c21_forwarded_equals_sent_and_no_token_upstream(live):
    sid = str(uuid.uuid4())
    p = live.claude(sid, launched=True)
    assert p.returncode == 0, p.stderr[-800:]
    assert "OTel copy off" not in p.stderr, p.stderr[-800:]
    sent = items(live.ours(), sid)  # what the session sent to our collector
    forwarded = items(live.decode(live.up.posts), sid)  # what the stand-in upstream received
    assert sent and forwarded == sent, (len(sent), len(forwarded))
    print(f"c21: {len(sent)} records sent == forwarded, {len(live.up.posts)} upstream posts")
    hdrs = [x["headers"] for x in live.up.posts]
    assert all(h.get("x-test-token") == "abc123" for h in hdrs)  # the user's own header is mirrored
    assert not any("authorization" in h for h in hdrs)  # our token never leaves


def test_c21_sender_without_token_refused(live):
    token = (live.odir / "collector.token").read_text().strip()
    assert _post_receiver(None) == 401
    assert _post_receiver("wrong-" + token) == 401
    assert _post_receiver(token) == 200


def test_c30_non_fleet_session_export_untouched_and_privacy_unchanged(live):
    sid = str(uuid.uuid4())
    n0 = len(live.up.posts)
    p = live.claude(sid, launched=False)
    assert p.returncode == 0, p.stderr[-800:]
    direct = live.up.posts[n0:]
    assert direct and items(live.decode(direct), sid)  # reached the user's endpoint directly
    assert items(live.ours(), sid) == []  # and never went through our collector
    settings = json.loads((live.odir / "claude-settings.json").read_text())["env"]
    assert all(
        k.startswith("OTEL_EXPORTER_OTLP_") for k in settings
    )  # endpoints/protocol/headers only
    assert not any("LOG_USER_PROMPTS" in k or "LOG_TOOL" in k or "TELEMETRY" in k for k in settings)
    both = json.dumps(live.ours()) + json.dumps(live.decode(live.up.posts))
    assert (
        "zebra-4471" not in both
    )  # the user's OTEL_LOG_USER_PROMPTS=0 still holds through the copy


def test_c21_collector_down_at_launch_exports_directly(live):
    sid = str(uuid.uuid4())
    n0 = len(live.up.posts)
    p = live.claude(
        sid,
        launched=True,
        AGENT_SYS_O11Y_OTEL_IMAGE="otel/opentelemetry-collector-contrib:absent-tag",
        AGENT_SYS_O11Y_OTEL_CONTAINER="live-down",
    )
    assert p.returncode == 0 and "OTel copy off" in p.stderr, p.stderr[-800:]
    assert items(live.decode(live.up.posts[n0:]), sid)  # direct export, as without o11y
    assert (
        _docker("inspect", f"infera-agent-sys-o11y-otelcol-{UID}-live-down").returncode != 0
    )  # nothing created


def _health_sampler(live, stop: threading.Event):
    """Stand-in for the trigger's sampler (design §7): poll the health URL, write outage edges to
    otel/health.json (otel_health.v1). The trigger's own sampler is tested on the mod side."""
    url, healthy, outages = f"http://127.0.0.1:{BASE + 2}/", True, 0
    while not stop.is_set():
        try:
            ok = urllib.request.urlopen(url, timeout=1).status == 200
        except OSError:
            ok = False
        if ok != healthy:
            healthy = ok
            outages += 0 if ok else 1
            doc = {
                "v": 1,
                "ts": store.now(),
                "healthy": ok,
                "outages": outages,
                "last_change": store.now(),
            }
            if not ok:
                doc["down_since"] = store.now()
            store.write_json(live.fleet / "otel" / "health.json", doc)
        time.sleep(0.2)


def test_c30_killed_collector_restarted_and_outage_reported(live):
    store.write_json(
        live.fleet / "otel" / "health.json",
        {"v": 1, "ts": store.now(), "healthy": True, "outages": 0},
    )
    assert run_consumer("log", log_handle, root=live.fleet, force=True)["passes"] == 1  # baseline
    st0 = json.loads(_docker("inspect", CONTAINER).stdout)[0]["State"]  # started by the c21 launch
    assert st0["Running"]
    stop = threading.Event()
    t = threading.Thread(target=_health_sampler, args=(live, stop), daemon=True)
    t.start()
    os.kill(st0["Pid"], 9)  # the collector runs as our uid: a crash, not `docker stop`
    for _ in range(120):
        st = json.loads(_docker("inspect", CONTAINER).stdout)[0]
        if st["RestartCount"] >= 1 and st["State"]["Running"]:
            break
        time.sleep(0.5)
    time.sleep(3)
    stop.set()
    t.join()
    st = json.loads(_docker("inspect", CONTAINER).stdout)[0]
    assert (
        st["State"]["Running"]
        and st["RestartCount"] >= 1
        and st["HostConfig"]["RestartPolicy"]["Name"] == "unless-stopped"
    )
    assert (
        _post_receiver((live.odir / "collector.token").read_text().strip()) == 200
    )  # serving again
    assert run_consumer("log", log_handle, root=live.fleet, force=True)["passes"] == 1
    down = [
        e
        for _, e, _ in store.read_jsonl(live.fleet / store.EVENTS)
        if e and e["event_name"] == "o11y.otel.collector_down"
    ]
    assert len(down) == 1 and down[0]["severity_text"] == "ERROR"


def test_analysis_consumes_the_otel_copy_live(live):
    """api_retries_exhausted reaches the log job only through OTel (the transcript has no such record)."""
    fake = Fake503(live.fake_port)
    sid = str(uuid.uuid4())
    store.write_json(
        live.fleet / store.SESSIONS,
        {
            "v": 1,
            "generated_at": store.now(),
            "main_session_id": sid,
            "sessions": [
                {
                    "session_id": sid,
                    "pid": os.getpid(),
                    "alive": True,
                    "role": "main",
                    "status": "busy",
                    "status_since": store.now(),
                    "name": "live-otel",
                }
            ],
        },
    )
    try:
        run_consumer(
            "log", log_handle, root=live.fleet, force=True
        )  # baseline for this OTel window
        p = live.claude(
            sid,
            launched=True,
            ANTHROPIC_BASE_URL=f"http://127.0.0.1:{live.fake_port}",
            CLAUDE_CODE_MAX_RETRIES="2",
        )
        assert "OTel copy off" not in p.stderr, p.stderr[-800:]
    finally:
        fake.close()
    run_consumer("log", log_handle, root=live.fleet, force=True)
    ex = [
        e
        for _, e, _ in store.read_jsonl(live.fleet / store.EVENTS)
        if e and e["event_name"] == "o11y.api.retries_exhausted"
    ]
    assert (
        len(ex) == 1
        and ex[0]["attributes"]["session.id"] == sid
        and ex[0]["attributes"]["o11y.source"] == "otel"
    )
