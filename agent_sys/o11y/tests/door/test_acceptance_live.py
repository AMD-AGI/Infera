"""Spec §12 criteria 20, 23 and 32, live (they call the model; skipped without O11Y_LIVE=1 and credentials).

Every run uses its own config dirs under the test's tmp dir: A = no plugins, B = the same base plus
`claude plugin install` of the three plugins from the agent_sys/o11y directory marketplace (what a user does).
Criterion 32 is a latency bound on the reference host: it also needs O11Y_REFERENCE_HOST=1 and takes ~10 min.
"""

import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import pytest
from door_paths import DOOR, O11Y
from o11y_door import api
from test_acceptance import CLAUDE, TOOL, live

PLUGINS = ("agent-sys-o11y-door", "agent-sys-o11y", "agent-sys-o11y-dash")
O11Y_RUN = O11Y / "plugins" / "agent-sys-o11y" / "server" / "o11y-run"
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
reference_host = pytest.mark.skipif(
    os.environ.get("O11Y_REFERENCE_HOST") != "1", reason="latency bounds: set O11Y_REFERENCE_HOST=1"
)


class Setup:
    """Config dir A (no plugins) and B (plugins installed), one project dir each, a PATH whose `claude` is the
    pinned binary (the analyzer calls `claude` from PATH)."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        (tmp / "bin").mkdir()
        (tmp / "bin" / "claude").symlink_to(CLAUDE)
        version = subprocess.run(
            [CLAUDE, "--version"], capture_output=True, text=True
        ).stdout.split()[0]
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        for c in "AB":
            self.proj(c).mkdir(parents=True)
            (self.proj(c) / ".claude").mkdir()
            (self.proj(c) / "README.md").write_text("scratch project\n")
            self.cfg(c).mkdir()
            seed = {  # what a configured user has: onboarding done, key approved, project trusted
                "hasCompletedOnboarding": True,
                "lastReleaseNotesSeen": version,
                "customApiKeyResponses": {"approved": [key[-20:]] if key else [], "rejected": []},
                "projects": {str(self.proj(c)): {"hasTrustDialogAccepted": True}},
            }
            (self.cfg(c) / ".claude.json").write_text(json.dumps(seed))
        env = self.env("B")
        for argv in (
            ["marketplace", "add", str(O11Y)],
            *(["install", f"{p}@agent-sys-local"] for p in PLUGINS),
        ):
            subprocess.run(
                [CLAUDE, "plugin", *argv], env=env, check=True, capture_output=True, timeout=120
            )

    def cfg(self, c):
        return self.tmp / f"cfg{c}"

    def proj(self, c):
        return self.tmp / f"proj{c}"

    def fleet(self, c):
        return self.tmp / f"fleet{c}"

    def env(self, c, member=False):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("AGENT_SYS", "CLAUDE_"))}
        env.update(
            CLAUDE_CONFIG_DIR=str(self.cfg(c)),
            DISABLE_AUTOUPDATER="1",
            AGENT_SYS_HOME=str(self.tmp / f"prefix{c}"),
            PATH=f"{self.tmp / 'bin'}:{os.environ['PATH']}",
        )
        if member:
            env.update(AGENT_SYS_O11Y="true", AGENT_SYS_O11Y_FLEET_DIR=str(self.fleet(c)))
        return env

    def claude_p(self, c, prompt, member=False, tools=("Bash",), strace=None, extra=()):
        argv = [CLAUDE, "-p", "--model", "haiku", "--output-format", "stream-json", "--verbose"]
        argv += ["--allowedTools", ",".join(tools), *extra, "--", prompt]
        if strace:
            argv = [
                "strace",
                "-f",
                "-qq",
                "-s",
                "400",
                "-e",
                "trace=%file,%process",
                "-o",
                str(strace),
            ] + argv
        p = subprocess.run(
            argv,
            cwd=self.proj(c),
            env=self.env(c, member),
            capture_output=True,
            text=True,
            timeout=600,
            stdin=subprocess.DEVNULL,
        )
        events = [json.loads(line) for line in p.stdout.splitlines() if line.startswith("{")]
        assert events, f"no output (rc={p.returncode}): {p.stderr[-500:]}"
        return events

    def snapshot(self, c) -> dict:
        out = {}
        for base, tag in ((self.cfg(c), "cfg"), (self.proj(c) / ".claude", "proj/.claude")):
            for d, _, files in os.walk(base):
                for n in files:
                    f = Path(d) / n
                    rel = f"{tag}/{f.relative_to(base)}"
                    out[rel] = (
                        "link" if f.is_symlink() else hashlib.sha256(f.read_bytes()).hexdigest()
                    )
        return out

    def norm(self, path: str, c: str) -> str:
        path = path.replace(re.sub(r"[^A-Za-z0-9]", "-", str(self.proj(c))), "<proj>")
        return re.sub(r"\d{10,}", "<n>", UUID.sub("<uuid>", path))


def _changed(a: dict, b: dict, s: Setup, c: str) -> set[str]:
    return {s.norm(k, c) for k in set(a) | set(b) if a.get(k) != b.get(k)}


def _result(events) -> dict:
    return next(e for e in events if e.get("type") == "result")


# --- 20: no settings/plugin/skill/CLAUDE.md change; config dir differs only in Claude's own files -------------

# Claude Code's own per-session files (seen changing without the plugins too) and the marketplace list it
# refreshes (B registered the o11y marketplace at install, A gets the official one added on first start).
ALLOWED_B_ONLY = {"cfg/plugins/known_marketplaces.json"}
NEVER = re.compile(
    r"(^|/)(settings(\.local)?\.json|CLAUDE\.md|installed_plugins\.json|keybindings\.json)$|/skills/"
)
PACKAGE_PROC = re.compile(r"o11y|backstop|agent-sys-o11y")
CLAUDE_PROC = re.compile(
    r"^\S*claude\S* claude "
)  # the analyzer's `claude -p` child is Claude Code itself
LINE = re.compile(r"^(\d+)\s+(\w+)\((.*)$")
WRITE_FLAGS = ("O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND")
MUTATING = {
    "rename",
    "renameat",
    "renameat2",
    "unlink",
    "unlinkat",
    "mkdir",
    "mkdirat",
    "rmdir",
    "link",
    "linkat",
}
MUTATING |= {"symlink", "symlinkat", "truncate", "chmod", "fchmodat", "creat"}


def _package_writes(trace: Path) -> list[tuple[str, str, str]]:
    """(process, call, path) for successful file writes by this package's processes and their descendants,
    except those of Claude Code itself (a process whose nearest exec is the claude binary)."""
    exe, parent, pending, out = {}, {}, {}, []
    for line in trace.read_text(errors="replace").splitlines():
        head = re.match(r"^(\d+) (.*) <unfinished \.\.\.>$", line)
        if head:
            pending[int(head[1])] = head[2]
            continue
        tail = re.match(r"^(\d+) <\.\.\. \w+ resumed>(.*)$", line)
        if tail:
            line = f"{tail[1]} {pending.pop(int(tail[1]), '')}{tail[2]}"
        m = LINE.match(line)
        if not m:
            continue
        pid, call, rest = int(m[1]), m[2], m[3]
        if call == "execve" and rest.rstrip().endswith("= 0"):
            exe[pid] = " ".join(re.findall(r'"([^"]*)"', rest.split("], ")[0])[:4])
        elif call in ("clone", "clone3", "vfork", "fork") and re.search(r"= (\d+)$", rest):
            parent[int(re.search(r"= (\d+)$", rest)[1])] = pid
        elif re.search(r"= -1 ", rest) is None and (
            call in MUTATING or (call in ("open", "openat") and any(f in rest for f in WRITE_FLAGS))
        ):
            paths = re.findall(r'"(/[^"]*)"', rest)
            chain, p = [], pid
            while p is not None:
                chain.append(exe.get(p, ""))
                p = parent.get(p)
            nearest = next((e for e in chain if e), "")
            if (
                paths
                and any(PACKAGE_PROC.search(e) for e in chain)
                and not CLAUDE_PROC.match(nearest)
            ):
                out.append((nearest, call, paths[-1]))
    return out


@live
def test_c20_live_config_dirs_differ_only_in_claudes_own_files(tmp_path):
    s = Setup(tmp_path)
    if not shutil.which("strace"):
        pytest.skip("strace not found")
    prompt = (
        f"Run exactly these two Bash commands, one after the other: `echo c20` then "
        f"`{O11Y_RUN} --kind extract --force`. Then reply with exactly DONE."
    )
    changes, extract = {}, None
    for c in "AB":
        before = s.snapshot(c)
        s.claude_p(
            c, "Run `echo c20` with the Bash tool, then reply with exactly DONE."
        )  # outside every fleet
        trace = tmp_path / f"{c}.strace"
        events = s.claude_p(
            c, prompt, member=True, strace=trace
        )  # fleet member; the analysis pass inside it
        changes[c] = _changed(before, s.snapshot(c), s, c)
        if c == "B":
            out = [
                json.dumps(b.get("content"))
                for e in events
                for b in (e.get("message") or {}).get("content") or []
                if isinstance(b, dict) and b.get("type") == "tool_result"
            ]
            extract = next((o for o in out if '\\"kind\\": \\"extract\\"' in o), None)
            writes = _package_writes(trace)
    assert extract and '\\"ok\\": true' in extract, (
        f"the analysis pass made no successful model call: {extract}"
    )
    bad = sorted(p for p in changes["B"] if NEVER.search(p)) + sorted(
        p for p in changes["A"] if NEVER.search(p)
    )
    assert not bad, bad
    assert not any(p.startswith("proj/.claude") for c in "AB" for p in changes[c])
    assert changes["B"] - changes["A"] <= ALLOWED_B_ONLY, sorted(changes["B"] - changes["A"])
    # Claude Code's own processes (incl. the analyzer's `claude -p`) are judged by the snapshot diff above;
    # no other process of this package writes under the config dir or the project's .claude/
    cfg, proj = str(s.cfg("B")), str(s.proj("B") / ".claude")
    offending = [w for w in writes if w[2].startswith((cfg + "/", proj))]
    assert any(w[2].startswith(str(s.fleet("B"))) for w in writes), (
        "the trace saw none of our writes"
    )
    assert not offending, offending


# --- 23: door tool definitions <= 1,000 input tokens per request in a fleet session, none elsewhere ----------


def _input_tokens(events) -> int:
    u = _result(events)["usage"]
    return (
        u.get("input_tokens", 0)
        + u.get("cache_creation_input_tokens", 0)
        + u.get("cache_read_input_tokens", 0)
    )


@live
def test_c23_live_tool_definition_tokens(tmp_path):
    s = Setup(tmp_path)
    ask = "Reply with just OK."
    none = _input_tokens(s.claude_p("A", ask))
    outside = _input_tokens(s.claude_p("B", ask))
    member = _input_tokens(s.claude_p("B", ask, member=True, tools=(TOOL,)))
    print(
        f"c23 input tokens: no plugins {none}, installed outside a fleet {outside}, fleet member {member}"
    )
    assert outside == none
    assert 0 < member - none <= 1000


# --- 32: added latency on the reference host --------------------------------------------------------------


def _p_overhead(s: Setup, c: str) -> float:
    t0 = time.perf_counter()
    events = s.claude_p(c, "Reply with just OK.")
    return time.perf_counter() - t0 - _result(events)["duration_api_ms"] / 1000.0


def _interactive_ready(s: Setup, c: str, sock: str) -> float:
    env = s.env(c)
    exports = " ".join(
        f"{k}='{env[k]}'" for k in ("CLAUDE_CONFIG_DIR", "DISABLE_AUTOUPDATER", "AGENT_SYS_HOME")
    )
    name = f"c32{c}{os.getpid()}"
    t0 = time.perf_counter()
    cmd = f"env -u AGENT_SYS_O11Y -u AGENT_SYS_O11Y_FLEET_DIR {exports} '{CLAUDE}' --model haiku"
    subprocess.run(
        [
            "tmux",
            "-L",
            sock,
            "new-session",
            "-d",
            "-s",
            name,
            "-x",
            "200",
            "-y",
            "50",
            "-c",
            str(s.proj(c)),
            cmd,
        ],
        check=True,
    )
    try:
        while time.perf_counter() - t0 < 60:
            pane = subprocess.run(
                ["tmux", "-L", sock, "capture-pane", "-p", "-t", name],
                capture_output=True,
                text=True,
            ).stdout
            if "❯" in pane and "for shortcuts" in pane:
                return time.perf_counter() - t0
            time.sleep(0.02)
        raise AssertionError(f"interactive session not ready in 60 s:\n{pane}")
    finally:
        subprocess.run(["tmux", "-L", sock, "kill-session", "-t", name], capture_output=True)
        time.sleep(1.0)


def _backstop_ms(s: Setup, n: int) -> dict:
    hook = str(DOOR / "hooks" / "backstop.sh")
    payload = json.dumps(
        {"tool_name": "Bash", "tool_input": {"command": "true"}, "session_id": "s-c32"}
    )
    base = {**s.env("B"), "CLAUDE_CODE_SESSION_ID": "s-c32"}
    member = {**base, "AGENT_SYS_O11Y": "true", "AGENT_SYS_O11Y_FLEET_DIR": str(s.fleet("B"))}

    def call(env):
        t0 = time.perf_counter()
        r = subprocess.run([hook], input=payload, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return 1000 * (time.perf_counter() - t0)

    ctx = api.Context(root=s.fleet("B"), session_id="s-c32", role="main")
    assert api.ledger_append(ctx, {"kind": "plan", "next": [{"theme": "c32"}]})["ok"]
    call(member)  # first sight: the cursor becomes current
    out = {
        "outside": [call(base) for _ in range(n)],
        "unchanged": [call(member) for _ in range(n)],
        "changed": [],
    }
    for i in range(n):
        rec = {
            "kind": "finding",
            "finding_id": f"f{i}",
            "text": "x" * 300,
            "evidence_rank": "inference",
        }
        assert api.ledger_append(ctx, rec)["ok"]
        out["changed"].append(call(member))
    return {k: statistics.median(v) for k, v in out.items()}


@live
@reference_host
def test_c32_live_added_latency_on_the_reference_host(tmp_path):
    s = Setup(tmp_path)
    n = int(os.environ.get("O11Y_C32_N", "10"))
    p = {"A": [], "B": []}
    i = {"A": [], "B": []}
    sock = f"yihou_o11y_c32_{os.getpid()}"
    try:
        for _ in range(n):  # interleaved, so drift affects both
            for c in "AB":
                p[c].append(_p_overhead(s, c))
            for c in "AB":
                i[c].append(_interactive_ready(s, c, sock))
    finally:
        subprocess.run(["tmux", "-L", sock, "kill-server"], capture_output=True)
    dp = statistics.median(p["B"]) - statistics.median(p["A"])
    di = statistics.median(i["B"]) - statistics.median(i["A"])
    bs = _backstop_ms(s, max(100, 10 * n))
    print(f"c32: -p +{dp:.3f} s, interactive +{di:.3f} s (N={n}); backstop medians ms {bs}")
    assert dp <= 0.4 and di <= 0.25
    assert bs["outside"] <= 10 and bs["unchanged"] <= 80 and bs["changed"] <= 600
