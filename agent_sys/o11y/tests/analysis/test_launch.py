"""bin/o11y-launch: mirroring the user's effective OTel settings, the collector config, the `--settings`
file, fallbacks, and client bookkeeping. No docker and no claude here (the live checks are in the report)."""

import importlib.machinery
import importlib.util
import json
import os

import pytest
from analysis_paths import PLUGIN

LAUNCH = PLUGIN / "bin" / "o11y-launch"


@pytest.fixture
def ol(tmp_path, monkeypatch):
    loader = importlib.machinery.SourceFileLoader("o11y_launch", str(LAUNCH))
    spec = importlib.util.spec_from_loader("o11y_launch", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    monkeypatch.setattr(mod, "MANAGED", [tmp_path / "managed-settings.json"])
    monkeypatch.setattr(mod, "MANAGED_DROPINS", tmp_path / "managed-settings.d")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cc"))
    monkeypatch.setenv("AGENT_SYS_O11Y_OTEL_DIR", str(tmp_path / "otel"))
    monkeypatch.setenv("AGENT_SYS_O11Y_OTEL_PORT_BASE", "36000")
    for k in list(os.environ):
        if (
            k.startswith(("OTEL_", "CLAUDE_CODE_", "BETA_", "ENABLE_BETA"))
            or k == "NODE_EXTRA_CA_CERTS"
        ):
            monkeypatch.delenv(k, raising=False)
    (tmp_path / "cc").mkdir()
    mod.tmp = tmp_path
    return mod


def user_env(ol, env, **top):
    (ol.tmp / "cc" / "settings.json").write_text(json.dumps({"env": env, **top}))


BASE = {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "OTEL_METRICS_EXPORTER": "otlp",
    "OTEL_LOGS_EXPORTER": "otlp",
    "OTEL_TRACES_EXPORTER": "otlp",
    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "https://collector.example.com",
    "OTEL_LOG_TOOL_DETAILS": "0",
}


def test_mirrors_generic_endpoint_and_headers(ol):
    user_env(ol, {**BASE, "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Bearer%20t1,x-team=a"})
    plan, why = ol.plan_copy(*ol.effective_settings())
    assert why == "ok" and set(plan["signals"]) == {"logs", "metrics", "traces"}
    assert plan["signals"]["logs"]["endpoint"] == "https://collector.example.com/v1/logs"
    assert plan["signals"]["metrics"]["headers"] == {"Authorization": "Bearer t1", "x-team": "a"}
    cfg = ol.collector_config(plan, "tok" * 12)
    assert "processors" not in cfg and "processors" not in json.dumps(cfg["service"])
    assert (
        cfg["exporters"]["otlp_http/traces"]["traces_endpoint"]
        == "https://collector.example.com/v1/traces"
    )
    assert cfg["service"]["pipelines"]["logs"] == {
        "receivers": ["otlp"],
        "exporters": ["otlp_http/logs", "file/logs"],
    }
    assert cfg["receivers"]["otlp"]["protocols"]["http"]["endpoint"] == "127.0.0.1:36000"
    st = ol.claude_settings(plan, "tok" * 12)["env"]
    assert st["OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"] == "http://127.0.0.1:36000/v1/logs"
    assert not any("LOG_TOOL" in k or "USER_PROMPTS" in k for k in st)  # privacy switches untouched


def test_per_signal_endpoint_and_shell_layering(ol, monkeypatch):
    # user settings beat the shell; a per-signal endpoint beats the generic one for that signal
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://shell-loses:4318")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_HEADERS", "x-logs=1")
    user_env(ol, {**BASE, "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT": "https://logs.example.com/ingest"})
    plan, _ = ol.plan_copy(*ol.effective_settings())
    assert plan["signals"]["logs"]["endpoint"] == "https://logs.example.com/ingest"
    assert plan["signals"]["logs"]["headers"] == {"x-logs": "1"}
    assert plan["signals"]["metrics"]["endpoint"] == "https://collector.example.com/v1/metrics"


def test_grpc_and_non_otlp_signals(ol):
    user_env(
        ol,
        {
            **BASE,
            "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://grpc.example:4317",
            "OTEL_METRICS_EXPORTER": "prometheus",
            "OTEL_TRACES_EXPORTER": "none",
        },
    )
    plan, _ = ol.plan_copy(*ol.effective_settings())
    assert set(plan["signals"]) == {"logs"}
    cfg = ol.collector_config(plan, "tok" * 12)
    assert cfg["exporters"]["otlp/logs"]["endpoint"] == "grpc.example:4317"
    assert cfg["exporters"]["otlp/logs"]["tls"] == {"insecure": True}
    assert set(cfg["service"]["pipelines"]) == {"logs"}


@pytest.mark.parametrize(
    "env,top,managed,reason",
    [
        ({**BASE, "CLAUDE_CODE_ENABLE_TELEMETRY": "0"}, {}, {}, "not enabled"),
        ({**BASE, "OTEL_EXPORTER_OTLP_CLIENT_KEY": "/k.pem"}, {}, {}, "cannot mirror"),
        (BASE, {"otelHeadersHelper": "/bin/helper"}, {}, "otelHeadersHelper"),
        (BASE, {}, {"env": {"OTEL_EXPORTER_OTLP_ENDPOINT": "https://org"}}, "managed settings pin"),
        (
            {
                **BASE,
                "OTEL_LOGS_EXPORTER": "console",
                "OTEL_METRICS_EXPORTER": "none",
                "OTEL_TRACES_EXPORTER": "",
            },
            {},
            {},
            "no signal",
        ),
        (
            {k: v for k, v in BASE.items() if k != "OTEL_EXPORTER_OTLP_ENDPOINT"},
            {},
            {},
            "no endpoint",
        ),
    ],
)
def test_unmirrorable_settings_fall_back(ol, env, top, managed, reason):
    user_env(ol, env, **top)
    if managed:
        (ol.tmp / "managed-settings.json").write_text(json.dumps(managed))
    plan, why = ol.plan_copy(*ol.effective_settings())
    assert plan is None and reason in why


def test_dry_run_with_callers_settings_is_plain(ol, capsys):
    user_env(ol, BASE)
    assert ol.launch(["--settings", "x.json", "-p", "hi"], otel=True, dry_run=True) == 0
    out = json.loads(capsys.readouterr().out)
    assert out == {"otel": False, "reason": "the caller passes its own --settings"}


def test_dry_run_redacts_headers(ol, capsys):
    user_env(ol, {**BASE, "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=secret"})
    ol.launch([], otel=True, dry_run=True)
    out = capsys.readouterr().out
    assert "secret" not in out and "<redacted>" in out


def test_live_clients_prunes_only_dead_entries(ol):
    odir = ol.otel_dir()
    (odir / "clients").mkdir(parents=True)
    me = {"pid": os.getpid(), "proc_start": ol.proc_start(os.getpid())}
    (odir / "clients" / f"{os.getpid()}.json").write_text(json.dumps(me))
    (odir / "clients" / "4000002.json").write_text(json.dumps({"pid": 4000002, "proc_start": "1"}))
    for f in (odir / "clients").iterdir():
        os.chmod(f, 0o600)  # as write_private leaves them
    live = ol.live_clients(odir, prune=True)
    assert [c["pid"] for c in live] == [os.getpid()]
    assert sorted(p.name for p in (odir / "clients").iterdir()) == [f"{os.getpid()}.json"]


def test_template_fills_ports_as_integers(ol):
    cfg = ol._fill(json.loads(ol.TEMPLATE.read_text()), ol.ports())
    reader = cfg["service"]["telemetry"]["metrics"]["readers"][0]["pull"]["exporter"]["prometheus"]
    assert (
        reader["port"] == 36003
        and cfg["extensions"]["health_check"]["endpoint"] == "127.0.0.1:36002"
    )
    assert "_comment" not in cfg


def test_receiver_requires_bearer_token_and_session_sends_it(ol):
    user_env(ol, BASE)
    plan, _ = ol.plan_copy(*ol.effective_settings())
    tok = "t" * 43
    cfg = ol.collector_config(plan, tok)
    assert cfg["extensions"]["bearertokenauth"] == {"token": tok}
    assert "bearertokenauth" in cfg["service"]["extensions"]
    for proto in cfg["receivers"]["otlp"]["protocols"].values():
        assert proto["auth"] == {"authenticator": "bearertokenauth"}
    env = ol.claude_settings(plan, tok)["env"]
    assert env["OTEL_EXPORTER_OTLP_HEADERS"] == f"Authorization=Bearer {tok}"
    assert all(
        env[f"OTEL_EXPORTER_OTLP_{s.upper()}_HEADERS"] == f"Authorization=Bearer {tok}"
        for s in plan["signals"]
    )


def test_token_file_private_and_reused(ol):
    odir = ol.otel_dir()
    assert ol.otel_src.ensure_private_tree(str(odir)) is None
    t1 = ol.load_or_create_token(odir)
    assert ol.load_or_create_token(odir) == t1 and len(t1) >= 32
    st = os.stat(odir / "collector.token")
    assert st.st_mode & 0o777 == 0o600


def test_private_tree_refuses_symlink_and_open_modes(ol, tmp_path):
    base = tmp_path / "priv"
    assert ol.otel_src.ensure_private_tree(str(base / "a" / "b")) is None
    os.chmod(base / "a", 0o755)
    assert "needs 0700" in ol.otel_src.tree_problem(str(base / "a" / "b"))
    os.chmod(base / "a", 0o700)
    (tmp_path / "link").symlink_to(base)
    assert "symlink" in ol.otel_src.tree_problem(str(tmp_path / "link" / "a"))


def test_launch_falls_back_when_dir_not_private(ol, tmp_path, monkeypatch):
    user_env(ol, BASE)
    bad = tmp_path / "shared"
    bad.mkdir(mode=0o777)
    os.chmod(bad, 0o777)
    monkeypatch.setenv("AGENT_SYS_O11Y_OTEL_DIR", str(bad / "otel"))
    calls = []
    monkeypatch.setattr(
        ol,
        "exec_claude",
        lambda args, settings: calls.append(settings) or (_ for _ in ()).throw(SystemExit(0)),
    )
    with pytest.raises(SystemExit):
        ol.launch(["-p", "hi"], otel=True, dry_run=False)
    assert calls == [None]  # plain claude, no --settings


def test_exec_exports_health_url_only_with_the_copy(ol, monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        ol.os,
        "execvp",
        lambda f, argv: seen.append((argv, os.environ.get("AGENT_SYS_O11Y_OTEL_HEALTH_URL"))),
    )
    monkeypatch.setenv("AGENT_SYS_O11Y_OTEL_HEALTH_URL", "http://stale/")
    ol.exec_claude(["-p", "x"], tmp_path / "s.json")
    assert seen[-1][0][1:3] == ["--settings", str(tmp_path / "s.json")]
    assert seen[-1][1] == f"http://127.0.0.1:{ol.ports()['HEALTH_PORT']}/"
    ol.exec_claude(["-p", "x"], None)
    assert seen[-1][1] is None  # plain claude: no collector to sample
