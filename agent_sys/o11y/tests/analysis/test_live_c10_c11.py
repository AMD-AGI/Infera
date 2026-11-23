"""Criteria 10 and 11, live and timed: an isolated fleet (real interactive `claude` main with the three plugins
from the source tree, the trigger running with a 60 s slow timer) and two real children launched through the
door. A child waiting on a permission prompt for more than 10 minutes must produce one WARN, and a child killed
with SIGKILL one ERROR, each within one slow-timer period (plus one registry tick and the job's own runtime).

Skipped unless O11Y_LIVE=1, O11Y_LIVE_DIR (an existing absolute directory you own, outside any project and
not directly under $HOME: use local disk, e.g. /var/tmp/<user>/o11y_live), tmux,
and the gateway credentials in ~/.claude/settings.json (read, never written). Runs on its own tmux server and
CLAUDE_CONFIG_DIR inside a new subdirectory of O11Y_LIVE_DIR; at the end only that tmux server is killed.
Takes about 13 minutes (the 10-minute threshold is a code constant and is not shortened).
"""

import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pytest
from analysis_paths import O11Y

LIVE = (
    os.environ.get("O11Y_LIVE_DIR") or f"/var/tmp/{os.environ.get('USER', os.getuid())}/o11y_live"
)
BIN = str(Path.home() / ".local" / "bin" / "claude")
PLUGINS = O11Y / "plugins"
GATEWAY_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
)
SLOW_S, TICK_S, JOB_S = (
    60,
    30,
    45,
)  # slow timer (config minimum), registry tick (T constant), job runtime margin
PERM_WAIT_S = 600  # o11y_job.config CONSTANTS["perm_wait_s"]


def _skip() -> str | None:
    if os.environ.get("O11Y_LIVE") != "1":
        return "live: set O11Y_LIVE=1"
    if not LIVE.startswith("/") or not os.path.isdir(LIVE):
        return "O11Y_LIVE_DIR (an existing absolute directory) not set"
    if not shutil.which("tmux") or not os.access(BIN, os.X_OK):
        return "tmux or claude missing"
    try:
        env = json.loads((Path.home() / ".claude" / "settings.json").read_text()).get("env", {})
    except (OSError, ValueError):
        env = {}
    return (
        None
        if env.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        else "no gateway credentials"
    )


pytestmark = pytest.mark.skipif(_skip() is not None, reason=_skip() or "")


def _t(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _wait(what: str, fn, timeout_s: float, step_s: float = 2.0):
    end = time.time() + timeout_s
    while True:
        v = fn()
        if v:
            return v
        if time.time() > end:
            raise AssertionError(f"timed out after {timeout_s:.0f}s waiting for {what}")
        time.sleep(step_s)


class Run:
    def __init__(self):
        self.id = f"c10c11-{time.strftime('%Y%m%d%H%M%S')}-{os.getpid()}"
        self.dir = Path(LIVE) / self.id
        self.cfg, self.proj, self.prefix, self.fleet = (
            self.dir / x for x in ("cfg", "proj", "prefix", "fleet")
        )
        for d in (self.cfg, self.proj, self.prefix, self.fleet / "config"):
            d.mkdir(parents=True)
        gw = json.loads((Path.home() / ".claude" / "settings.json").read_text()).get("env", {})
        gw = {
            k: os.environ.get(k) or gw.get(k)
            for k in GATEWAY_KEYS
            if os.environ.get(k) or gw.get(k)
        }
        key = gw.get("ANTHROPIC_API_KEY", "")
        (self.cfg / ".claude.json").write_text(
            json.dumps(
                {
                    "hasCompletedOnboarding": True,
                    "projects": {str(self.proj): {"hasTrustDialogAccepted": True}},
                    "customApiKeyResponses": {
                        "approved": [key[-20:]] if key else [],
                        "rejected": [],
                    },
                }
            )
        )
        never = {"every_n_tool_calls": 1_000_000, "every_n_turns": 1_000_000}
        (self.fleet / "config" / "trigger.json").write_text(
            json.dumps(
                {
                    "v": 1,
                    "kick": {"interval_s": 86_400},
                    "refresh": never,
                    "log": {**never, "slow_interval_s": SLOW_S},
                }
            )
        )
        (self.proj / "mission.md").write_text(
            "Live o11y test. Do exactly what each prompt says; nothing else.\n"
        )
        self.env = {
            "HOME": str(Path.home()),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C.UTF-8",
            "TERM": "xterm-256color",
            **gw,
            "O11Y_LIVE_RUN": self.id,
            "CLAUDE_CONFIG_DIR": str(self.cfg),
            "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
            "AGENT_SYS_HOME": str(self.prefix),
            "AGENT_SYS_O11Y_FLEET_DIR": str(self.fleet),
        }

    def tmux(self, *a) -> str:
        return subprocess.run(
            ["tmux", "-S", str(self.dir / "tmux.sock"), *a],
            env=self.env,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    def launch(self, name: str, extra_env: dict, *claude_args: str) -> int:
        env_args = [x for k, v in extra_env.items() for x in ("-e", f"{k}={v}")]
        plugins = [
            x
            for p in ("agent-sys-o11y-door", "agent-sys-o11y", "agent-sys-o11y-dash")
            for x in ("--plugin-dir", str(PLUGINS / p))
        ]
        self.tmux(
            "new-session",
            "-d",
            "-s",
            name,
            "-x",
            "200",
            "-y",
            "50",
            "-c",
            str(self.proj),
            *env_args,
            BIN,
            "--model",
            "haiku",
            "--name",
            name,
            *plugins,
            *claude_args,
        )
        return int(self.tmux("display", "-p", "-t", name, "#{pane_pid}").strip())

    def registry(self) -> list[dict]:
        try:
            return json.loads((self.fleet / "registry" / "sessions.json").read_text())["sessions"]
        except (OSError, ValueError, KeyError):
            return []

    def row(self, name: str) -> dict | None:
        return next((s for s in self.registry() if s.get("name") == name), None)

    def events(self, name: str) -> list[dict]:
        try:
            lines = (self.fleet / "logs" / "events.jsonl").read_text().splitlines()
        except OSError:
            return []
        return [e for e in map(json.loads, filter(None, lines)) if e.get("event_name") == name]

    def door(self, *args: str, sid: str) -> dict:
        env = {**self.env, "CLAUDE_CODE_SESSION_ID": sid}
        p = subprocess.run(
            [str(PLUGINS / "agent-sys-o11y-door" / "bin" / "py"), "-m", "o11y_door", *args],
            env=env,
            cwd=self.proj,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert p.returncode == 0, p.stdout + p.stderr
        return json.loads(p.stdout)


@pytest.fixture(scope="module")
def run():
    r = Run()
    yield r
    subprocess.run(["tmux", "-S", str(r.dir / "tmux.sock"), "kill-server"], capture_output=True)


def test_c10_c11_permission_wait_and_killed_child_within_one_slow_period(run):
    run.launch("main", {"AGENT_SYS_O11Y": "true"})
    main = _wait(
        "the main in the registry", lambda: (m := run.row("main")) and m.get("alive") and m, 120
    )
    for name in ("kid-wait", "kid-dead"):
        run.door(
            "session",
            "register",
            "--decision",
            "child",
            "--name",
            name,
            "--purpose",
            "live c10/c11",
            sid=main["session_id"],
        )
    child = {
        "AGENT_SYS_O11Y": "child",
        "AGENT_SYS_O11Y_FLEET_DIR": str(run.fleet),
        "AGENT_SYS_O11Y_PARENT_SESSION_ID": main["session_id"],
    }
    probe = run.dir / "perm_probe"
    run.launch(
        "kid-wait",
        child,
        "--permission-mode",
        "default",
        f"Use the Bash tool to run exactly this command and nothing else: touch {probe}",
    )
    dead_pid = run.launch(
        "kid-dead", child, "--permission-mode", "dontAsk", "Reply with the single word OK."
    )
    _wait(
        "kid-dead registered as an alive child",
        lambda: (r := run.row("kid-dead")) and r.get("alive") and r.get("role") == "child",
        180,
    )
    os.kill(dead_pid, 9)
    t_kill = time.time()
    crash = _wait(
        "one session.crashed for kid-dead",
        lambda: run.events("o11y.session.crashed"),
        SLOW_S + TICK_S + JOB_S + 30,
    )
    crash_latency = _t(crash[0]["ts"]) - t_kill
    assert (
        crash[0]["attributes"]["session.name"] == "kid-dead"
        and crash[0]["severity_text"] == "ERROR"
    )
    assert crash_latency <= SLOW_S + TICK_S + JOB_S, crash_latency
    w = _wait(
        "kid-wait waiting on a permission prompt",
        lambda: (
            (r := run.row("kid-wait"))
            and r.get("status") == "waiting"
            and r.get("waiting_for") == "permission prompt"
            and r
        ),
        180,
    )
    since = _t(w["status_since"])
    warn = _wait(
        "one session.permission_wait for kid-wait",
        lambda: run.events("o11y.session.permission_wait"),
        since + PERM_WAIT_S + SLOW_S + TICK_S + JOB_S + 60 - time.time(),
        step_s=5,
    )
    late = _t(warn[0]["ts"]) - (since + PERM_WAIT_S)
    assert (
        warn[0]["attributes"]["session.name"] == "kid-wait" and warn[0]["severity_text"] == "WARN"
    )
    assert 0 <= late <= SLOW_S + TICK_S + JOB_S, late
    assert not probe.exists()  # the prompt was never answered
    time.sleep(SLOW_S + TICK_S)  # one more slow period: still exactly one of each
    assert (
        len(run.events("o11y.session.crashed")) == 1
        and len(run.events("o11y.session.permission_wait")) == 1
    )
    print(
        f"c11 crash latency {crash_latency:.0f}s, c10 warn {late:.0f}s after the 10 minutes; run dir {run.dir}"
    )
