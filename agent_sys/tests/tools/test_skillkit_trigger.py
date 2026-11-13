"""When the entry skills engage, and how the installer renders and hooks them."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

AGENT_SYS = Path(__file__).resolve().parents[2]
SKILL = AGENT_SYS / "tools" / "skill"
KERNEL_SKILLS = AGENT_SYS / "kernel_registry" / "skills"
ASTK = SKILL / "lib" / "astk.py"
sys.path.insert(0, str(SKILL / "lib"))

from skillkit import settings  # noqa: E402

#: In the domain: the entry skill engages in auto mode.
ENGAGE = """
优化 sglang
tune qwen3.6 on mi300x
PD分离部署
帮我优化 sglang 在 MI300X 上的 decode 吞吐
首字延迟太高怎么办
首token时间太长
TTFT is too high on our vllm server
算子优化有什么思路
融合算子的性能怎么测
prefill/decode 分离怎么配置
预填充阶段太慢了
chunked prefill makes ttft worse
prefill throughput dropped after the upgrade
set up pd disaggregation with mooncake
帮我调一下 GEMM
ATOM 起不来
attention kernel 的 occupancy 很低
profile 一下 agentic workload
deploy an llm with fp8 quantization
llm inference latency is bad
glm decode throughput regression
rdma bandwidth between gpu nodes
mtp 和 eagle 哪个投机效果好
大模型推理服务怎么压测
fp4 gemm kernel on gfx950
写一个 triton kernel
kv cache 命中率低
deepseek 在 mi355x 上的部署
显卡 推理 性能 不行
量化后 解码 变慢
gpu kernel latency regressed
预填充 吞吐 掉了
优化一下 GLM-5.2 在 AgentX 上的 e2e 吞吐
新模型 PD 上线，提升 tok/s/chip
run a Hyperloom Forge campaign
optimize an AKLR schema
resume forge-loop
""".strip().splitlines()
#: Outside it: quiet, including every false positive the review found.
QUIET = """
what is the capital of France
优化一下网站的性能和延迟
优化一下我的网页
make my web page faster
an eagle flew
the atomic counter overflows
把每个单词首字母大写
C++ 运算子重载
Sobel算子边缘检测
表单预填充怎么做
prefill the form fields with the user's saved address
disaggregation of survey data by region
look at this stack trace from the linux kernel
decode this JWT and trace the request
set up e2e tests and deploy the web app
前后端分离项目怎么部署
k8s集群部署一个nginx
base64解码性能怎么样
benchmark this sorting function
my user profile page has high latency
逻辑推理题怎么做
deploy the django app to kubernetes
git checkout dev.yihou.agentsys.skills
表单预填充怎么优化性能
表单预填充之后怎么优化
prefill the form fields and optimize the page
prefill the form fields then deploy the web app
optimize this GPU shader in unity
deploy my GPU-accelerated video encoder with docker
set up an LLM chatbot and deploy it on vercel
""".strip().splitlines()
#: Explicit mode: these name agent_sys (also as the head of a module or skill name) ...
NAMED = """
用 agent_sys 优化 sglang
用agentsys优化
run the agent-sys task
use agent_sys.
ask infera-agent-sys first
/infera-agent-sys
/agent-sys-prior-work
请用 agent_sys-task-runner 跑一下
run the agent-sys-task-runner skill on this
how does agent_sys.closure work
""".strip().splitlines()
#: ... these do not (an identifier's tail, a path component, the domain alone).
NOT_NAMED = """
git checkout dev.yihou.agentsys.skills
my-agentsys-thing
look at /x/agent_sys/foo.py
优化 sglang
optimize an AKLR schema
resume Hyperloom forge-loop
""".strip().splitlines()


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    env = {
        "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
        "INFERA_AGENT_SYS_SETTINGS": str(tmp_path / "s.json"),
        "INFERA_AGENT_SYS_CONFIG": str(tmp_path / "none.json"),
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return {**os.environ, **env}


def _run(cmd: list, env: dict, stdin: str = "") -> subprocess.CompletedProcess:
    return subprocess.run(cmd, input=stdin, env=env, capture_output=True, text=True)


def _mode(env: dict, prompt: str, *flags: str, python: tuple = (sys.executable,)) -> int:
    return _run([*python, str(ASTK), "mode", *flags, "--prompt-file", "-"], env, prompt).returncode


@pytest.mark.parametrize("prompt", ENGAGE)
def test_engages_in_domain(prompt, isolated):
    assert settings.matches(prompt) and settings.engaged(prompt)


@pytest.mark.parametrize("prompt", QUIET)
def test_quiet_outside_domain(prompt, isolated):
    assert settings.matches(prompt) == [] and not settings.engaged(prompt)


@pytest.mark.parametrize(
    "prompt,named", [(p, True) for p in NAMED] + [(p, False) for p in NOT_NAMED]
)
def test_explicit_mode_needs_agent_sys_named(isolated, prompt, named):
    settings.save(trigger="off")  # alias of explicit
    assert settings.load()["trigger"] == "explicit" and settings.engaged(prompt) == named


def test_prior_work_gate(isolated):
    for prompt in (
        "is there a task or handoff for this",
        "知识库里有没有这个 handoff",
        "prior work?",
    ):
        assert settings.engaged(prompt, prior_work=True), prompt
    assert _mode(isolated, "有没有人做过这个? 之前的结果呢", "--prior-work") == 0
    assert _mode(isolated, "有没有人做过这个? 之前的结果呢") == 1  # the entry gate does not widen
    assert _mode(isolated, "what is the capital of France", "--prior-work") == 1
    settings.save(trigger="explicit")  # explicit mode still needs agent_sys named
    assert _mode(isolated, "is there a task or handoff for this", "--prior-work") == 1
    assert _mode(isolated, "agent_sys 知识库里有没有这个 handoff", "--prior-work") == 0


def test_mode_runs_on_the_standard_library_and_crashes_are_not_step_aside(isolated, tmp_path):
    bare = (sys.executable, "-S")  # no site-packages: no pydantic, yaml, ...
    assert _run([*bare, "-c", "import pydantic"], isolated).returncode != 0
    assert _mode(isolated, "用 agent_sys 优化 sglang", python=bare) == 0
    assert _mode(isolated, "what is the capital of France", python=bare) == 1
    other = _run([*bare, str(ASTK), "remote", "status"], isolated)
    assert other.returncode == 4 and "dependencies" in other.stderr
    (tmp_path / "s.json").write_text("[1, 2]")  # ignored, not a crash
    assert _mode(isolated, "优化 sglang") == 0
    crash = _run(
        [sys.executable, str(ASTK), "mode", "--prompt-file", str(tmp_path / "x")], isolated
    )
    assert crash.returncode == 2 and crash.stderr.startswith("astk: ")


def _checkout_copy(tmp_path: Path, name: str) -> Path:
    """A second checkout's tools/skill (enough for rendering and the installer)."""
    dst = tmp_path / name / "agent_sys" / "tools" / "skill"
    shutil.copytree(SKILL, dst, ignore=shutil.ignore_patterns("__pycache__"), symlinks=True)
    return dst


def _description(p: Path) -> str:
    return next(ln for ln in p.read_text().splitlines() if ln.startswith("description:"))


def test_rendered_per_checkout_and_mode_rerenders_all(isolated, tmp_path):
    other, gone = _checkout_copy(tmp_path, "wtB"), _checkout_copy(tmp_path, "wtC")
    a, b, _c = (settings.render_gated(root) for root in (SKILL, other, gone))
    assert settings.rendered_dir(SKILL) != settings.rendered_dir(other)
    assert str(SKILL) in a[0].read_text() and "${CLAUDE_SKILL_DIR}" not in a[0].read_text()
    assert str(other) not in a[0].read_text() and str(other) in b[0].read_text()
    assert "ONLY" not in _description(a[0])
    assert settings.render_status(SKILL, settings.GATED[0]) is None
    settings.save(trigger="explicit")
    assert settings.render_status(SKILL, settings.GATED[0]).startswith("rendered for auto")
    shutil.rmtree(tmp_path / "wtC")
    redone = settings.rerender_all("explicit")
    assert len(redone) == 2 * len(settings.GATED)  # A and B; the deleted C is skipped
    assert all("ONLY" in _description(p) for p in redone)


def _install(env: dict, root: Path, *args: str) -> subprocess.CompletedProcess:
    return _run(["sh", str(root / "install.sh"), *args], env)


def test_installer_links_skills_and_adds_hook_idempotently(isolated, tmp_path):
    claude = tmp_path / "claude"
    claude.mkdir()
    (claude / "settings.json").write_text(json.dumps({"env": {"X": "1"}}))
    env = {**isolated, "CLAUDE_CONFIG_DIR": str(claude)}
    for _ in range(2):
        assert _install(env, SKILL, "--hook").returncode == 0
    names = sorted(p.name for p in (claude / "skills").iterdir())
    sources = {
        p.parent.name: p.parent
        for p in [*SKILL.glob("*/SKILL.md"), *KERNEL_SKILLS.glob("*/SKILL.md")]
    }
    found = [p.parent.name for p in [*SKILL.glob("*/SKILL.md"), *KERNEL_SKILLS.glob("*/SKILL.md")]]
    assert len(found) == len(set(found)), "two skills share a name; the installer would skip one"
    assert "kernel-artifacts" in sources and names == sorted(sources)
    for n in names:
        target = (claude / "skills" / n).resolve()
        assert target == (settings.rendered_dir(SKILL) / n if n in settings.GATED else sources[n])
    assert "  ok       infera-agent-sys" in _install(env, SKILL, "--check").stdout
    assert _install(env, SKILL, "--mode", "explicit").returncode == 0
    assert "ONLY" in _description(claude / "skills" / "infera-agent-sys" / "SKILL.md")
    data = json.loads((claude / "settings.json").read_text())
    assert data["env"] == {"X": "1"} and len(data["hooks"]["UserPromptSubmit"]) == 1
    assert len(list(claude.glob("settings.json.bak-astk-*"))) == 1
    hook_cmd = data["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
    probe = _run(["sh", "-c", hook_cmd.replace("prompt_hint.py", "gone.py")], env, "{}")
    assert probe.returncode == 0  # a moved checkout never blocks a prompt


UPSTREAM = {"source": "github", "repo": "uppinote20/claude-dashboard"}


def _bin_dir(tmp_path: Path) -> Path:
    """A PATH dir with what install.sh runs, but no `claude` CLI."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for tool in (
        "python3",
        "git",
        "sh",
        "sed",
        "tail",
        "dirname",
        "basename",
        "readlink",
        "ln",
        "mkdir",
    ):
        found = shutil.which(tool)
        if found and not (bin_dir / tool).exists():
            (bin_dir / tool).symlink_to(found)
    return bin_dir


#: A stand-in `claude` CLI: logs its arguments; `marketplace add` exits
#: $FAKE_ADD_RC and, on success, registers the upstream marketplace; `install`
#: records a user-scope install in a fresh cache dir.
FAKE_CLAUDE = r"""#!/bin/sh
echo "$*" >> "$FAKE_LOG"
case "$2 $3" in
  "marketplace add")
    [ "${FAKE_ADD_RC:-0}" = 0 ] || exit "$FAKE_ADD_RC"
    python3 -c 'import json, sys, pathlib; p = pathlib.Path(sys.argv[1]); d = json.loads(p.read_text()) if p.exists() else {}; d.setdefault("claude-dashboard", {"source": {"source": "github", "repo": "uppinote20/claude-dashboard"}}); p.write_text(json.dumps(d))' "$CLAUDE_CONFIG_DIR/plugins/known_marketplaces.json" ;;
  "install claude-dashboard@claude-dashboard")
    python3 -c 'import json, sys, pathlib; p = pathlib.Path(sys.argv[1]); d = json.loads(p.read_text()) if p.exists() else {"version": 2, "plugins": {}}; path = pathlib.Path(sys.argv[2]); (path / "dist").mkdir(parents=True, exist_ok=True); d["plugins"].setdefault("claude-dashboard@claude-dashboard", []).append({"scope": "user", "installPath": str(path)}); p.write_text(json.dumps(d))' "$CLAUDE_CONFIG_DIR/plugins/installed_plugins.json" "$CLAUDE_CONFIG_DIR/plugins/cache/user-install" ;;
esac
"""


def _env(
    isolated: dict, claude: Path, tmp_path: Path, *, fake_claude: bool = False, files: str = ""
) -> dict:
    (claude / "plugins").mkdir(parents=True, exist_ok=True)
    bin_dir = _bin_dir(tmp_path)
    if fake_claude:
        (bin_dir / "claude").write_text(FAKE_CLAUDE)
        (bin_dir / "claude").chmod(0o755)
    return {
        **isolated,
        "CLAUDE_CONFIG_DIR": str(claude),
        "PATH": str(bin_dir),
        "FAKE_LOG": str(tmp_path / "claude.log"),
        # unreachable unless a test serves the files: proves which paths stay offline
        "ASTK_DASHBOARD_FILES": files or "http://127.0.0.1:9/x",
    }


def _register(
    claude: Path,
    tmp_path: Path,
    *,
    scope: str = "user",
    with_feature: bool = False,
    source=UPSTREAM,
) -> Path:
    """Record a claude-dashboard install the way Claude Code does."""
    path = tmp_path / "cache" / scope / "claude-dashboard" / "1.36.0"
    (path / "dist").mkdir(parents=True)
    if with_feature:
        (path / "commands").mkdir()
        (path / "dist" / "calibrate-pricing.js").write_text("// shipped\n")
        (path / "commands" / "calibrate-pricing.md").write_text("shipped\n")
    plugins = claude / "plugins"
    plugins.mkdir(parents=True, exist_ok=True)
    reg = plugins / "installed_plugins.json"
    data = json.loads(reg.read_text()) if reg.exists() else {"version": 2, "plugins": {}}
    row = {"scope": scope, "installPath": str(path)}
    if scope != "user":
        row["projectPath"] = str(tmp_path / "some-project")
    data["plugins"].setdefault("claude-dashboard@claude-dashboard", []).append(row)
    reg.write_text(json.dumps(data))
    (plugins / "known_marketplaces.json").write_text(
        json.dumps({"claude-dashboard": {"source": source}})
    )
    return path


def _pr_files(tmp_path: Path) -> str:
    """A local stand-in for #118's raw files at its pinned commit."""
    root = tmp_path / "pr118"
    (root / "dist").mkdir(parents=True)
    (root / "commands").mkdir()
    (root / "dist" / "calibrate-pricing.js").write_text("#!/usr/bin/env node\nmain();")
    (root / "commands" / "calibrate-pricing.md").write_text("---\ndescription: calibrate\n")
    return root.as_uri()


def test_installer_never_installs_dashboard_unasked(isolated, tmp_path):
    claude = tmp_path / "claude"
    env = _env(isolated, claude, tmp_path)  # no claude CLI: asking and declining must not need it
    out = _install(env, SKILL).stdout  # no tty: no prompt, no install
    assert "skipped  not a terminal; sh install.sh --dashboard yes installs it" in out
    out = _install(env, SKILL, "--dashboard", "no").stdout
    assert "skipped  (sh install.sh --dashboard yes installs it later)" in out
    assert "  off      (sh install.sh --dashboard yes" in _install(env, SKILL, "--check").stdout
    assert "skipped  claude CLI not on PATH" in _install(env, SKILL, "--dashboard", "yes").stdout
    assert not list((claude / "plugins").iterdir())
    assert _install(env, SKILL, "--dashboard", "maybe").returncode == 2
    assert _install(env, SKILL, "--dashboard").returncode == 2


def test_installer_reports_an_existing_dashboard_install(isolated, tmp_path):
    claude = tmp_path / "claude"
    path = _register(claude, tmp_path)
    env = _env(isolated, claude, tmp_path)
    out = _install(env, SKILL, "--check").stdout  # offline: the files URL is unreachable
    assert f"  ok       installed: {path}" in out
    assert (
        "  missing  calibrate-pricing (sh install.sh adds it from uppinote20/claude-dashboard#118)"
        in out
    )
    assert "/claude-dashboard:calibrate-pricing" in out
    out = _install(env, SKILL, "--repo", "--check").stdout  # plugins are per user, not per repo
    assert f"  ok       installed: {path}" in out
    (path / "dist" / "calibrate-pricing.js").write_text("// shipped\n")
    (path / "commands").mkdir()
    (path / "commands" / "calibrate-pricing.md").write_text("shipped\n")
    assert "  ok       calibrate-pricing available" in _install(env, SKILL, "--check").stdout


def test_installer_adds_the_pr_files_to_an_upstream_install(isolated, tmp_path):
    claude = tmp_path / "claude"
    path = _register(claude, tmp_path)
    (path / "commands").mkdir()
    (path / "commands" / "calibrate-pricing.md").write_text("mine\n")
    env = _env(isolated, claude, tmp_path, files=_pr_files(tmp_path))
    out = _install(env, SKILL).stdout
    assert "  added    calibrate-pricing from uppinote20/claude-dashboard#118" in out
    assert (path / "dist" / "calibrate-pricing.js").read_text() == "#!/usr/bin/env node\nmain();"
    assert (path / "commands" / "calibrate-pricing.md").read_text() == "mine\n"  # never overwritten
    # a failed download is reported, never fatal
    (path / "dist" / "calibrate-pricing.js").unlink()
    env["ASTK_DASHBOARD_FILES"] = (tmp_path / "gone").as_uri()
    res = _install(env, SKILL)
    assert res.returncode == 0 and "  FAILED   adding calibrate-pricing" in res.stdout
    assert "== settings" in res.stdout


def test_installer_leaves_an_install_from_another_source_alone(isolated, tmp_path):
    claude = tmp_path / "claude"
    fork = {"source": "git", "url": "https://github.com/someone/claude-dashboard.git", "ref": "dev"}
    path = _register(claude, tmp_path, source=fork)
    env = _env(isolated, claude, tmp_path, files=_pr_files(tmp_path))
    out = _install(env, SKILL).stdout
    assert (
        "not added: marketplace claude-dashboard is https://github.com/someone/claude-dashboard.git#dev"
        in out
    )
    assert not (path / "dist" / "calibrate-pricing.js").exists()


def test_installer_will_not_install_from_a_same_name_marketplace_elsewhere(isolated, tmp_path):
    claude = tmp_path / "claude"
    (claude / "plugins").mkdir(parents=True)
    fork = {"source": "github", "repo": "someone/claude-dashboard"}
    (claude / "plugins" / "known_marketplaces.json").write_text(
        json.dumps({"claude-dashboard": {"source": fork}})
    )
    env = {**_env(isolated, claude, tmp_path, fake_claude=True), "FAKE_ADD_RC": "1"}
    out = _install(env, SKILL, "--dashboard", "yes").stdout
    assert (
        "skipped  marketplace claude-dashboard is someone/claude-dashboard, not uppinote20/claude-dashboard"
        in out
    )
    assert "installed claude-dashboard" not in out
    log = (tmp_path / "claude.log").read_text()
    assert (
        "plugin marketplace add uppinote20/claude-dashboard" in log and "plugin install" not in log
    )


def test_installer_installs_for_the_user_despite_another_projects_install(isolated, tmp_path):
    claude = tmp_path / "claude"
    project = _register(claude, tmp_path, scope="project")
    env = _env(isolated, claude, tmp_path, fake_claude=True, files=_pr_files(tmp_path))
    assert "  off      (sh install.sh --dashboard yes" in _install(env, SKILL, "--check").stdout
    out = _install(env, SKILL, "--dashboard", "yes").stdout
    assert "installed claude-dashboard from uppinote20/claude-dashboard" in out
    user = claude / "plugins" / "cache" / "user-install"
    assert (user / "dist" / "calibrate-pricing.js").exists()
    assert not (project / "dist" / "calibrate-pricing.js").exists()


def test_installer_survives_an_unreadable_plugin_registry(isolated, tmp_path):
    claude = tmp_path / "claude"
    (claude / "plugins").mkdir(parents=True)
    (claude / "plugins" / "installed_plugins.json").write_text('{"plugins": "not a map"}')
    (claude / "plugins" / "known_marketplaces.json").write_text(
        '{"claude-dashboard": {"source": "a string"}}'
    )
    env = {**isolated, "CLAUDE_CONFIG_DIR": str(claude)}
    res = _install(env, SKILL, "--check")
    assert res.returncode == 0 and "  off      (sh install.sh --dashboard yes" in res.stdout


def test_hook_is_found_by_marker_repointed_and_deduplicated(isolated, tmp_path):
    other = _checkout_copy(tmp_path, "wtB")
    cc = tmp_path / "cc"
    cc.mkdir()
    foreign = {"command": "echo mine"}
    stale = {"command": "sh -c x _ /old/agent_sys/tools/skill/hooks/prompt_hint.py"}
    groups = [{"hooks": [foreign, stale]}, {"hooks": [dict(stale)]}]
    (cc / "settings.json").write_text(json.dumps({"hooks": {"UserPromptSubmit": groups}}))
    env = {**isolated, "CLAUDE_CONFIG_DIR": str(cc)}

    def commands() -> list[list[str]]:
        groups = json.loads((cc / "settings.json").read_text())["hooks"]["UserPromptSubmit"]
        return [[h["command"] for h in g["hooks"]] for g in groups]

    assert "not this checkout's" in _install(env, SKILL, "--check").stdout
    for root in (SKILL, other):
        assert _install(env, root, "--hook").returncode == 0
    mine = commands()[0][1]
    assert commands() == [["echo mine", mine]] and str(other / "hooks") in mine  # theirs kept
    data = json.loads((cc / "settings.json").read_text())
    data["hooks"]["UserPromptSubmit"].append({"hooks": [{"type": "command", "command": mine}]})
    (cc / "settings.json").write_text(json.dumps(data))
    assert "installed 2 times" in _install(env, other, "--check").stdout
    assert _install(env, other, "--hook").returncode == 0 and commands() == [["echo mine", mine]]


@pytest.mark.parametrize("text", ["{not json", "[]"])
def test_malformed_settings_json_is_reported_and_left_alone(isolated, tmp_path, text):
    cc = tmp_path / "cc"
    cc.mkdir()
    (cc / "settings.json").write_text(text)
    env = {**isolated, "CLAUDE_CONFIG_DIR": str(cc)}
    check = _install(env, SKILL, "--check")
    assert check.returncode == 0 and "ERROR" in check.stdout
    hook = _install(env, SKILL, "--hook")
    assert hook.returncode == 1 and "left untouched" in hook.stdout
    assert "== settings" in hook.stdout  # the rest of the install still ran
    assert (cc / "settings.json").read_text() == text and not list(cc.glob("*.bak-astk-*"))


def test_hint_hook(isolated):
    def hook(prompt: str) -> str:
        script = SKILL / "hooks" / "prompt_hint.py"
        return _run([sys.executable, str(script)], isolated, json.dumps({"prompt": prompt})).stdout

    out = json.loads(hook("帮我优化 sglang 在 MI300X 上的 decode 吞吐"))["hookSpecificOutput"]
    assert out["hookEventName"] == "UserPromptSubmit"
    assert "infera-agent-sys" in out["additionalContext"]
    assert hook("优化一下我的网页") == ""
    settings.save(trigger="explicit")
    assert hook("帮我优化 sglang") == "" and hook("用 agent_sys 优化 sglang") != ""
