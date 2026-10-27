"""The agent's Claude Code configuration (claude_config.py): its own directory with what it needs from yours, or yours
(claude_config: inherit). conftest's operator_claude is the made-up ~/.claude."""

import json
import os
import stat

import pytest
from conftest import T, run_home, ws
from fastapi.testclient import TestClient

from agent_sys_v2 import claude_config, launch, ops
from agent_sys_v2.model import AgentSysError
from agent_sys_v2.ops import new_task
from agent_sys_v2.scan import load_project
from agent_sys_v2.server import create_app

KEY = "sk-test-0123456789abcdefghij"


def test_env_forwarded_by_name_never_the_plugins():
    block = {
        "ANTHROPIC_BASE_URL": "u",
        "ANTHROPIC_API_KEY": "k",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "o",
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "AWS_REGION": "us-east-1",
        "HTTPS_PROXY": "http://proxy",
        "no_proxy": "localhost",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "IS_SANDBOX": "1",
        # never: plugins, the configuration directory, PATH and temp dirs
        "CLAUDE_CODE_PLUGIN_DIRS": "/home/me/mods",
        "CLAUDE_CONFIG_DIR": "/home/me/.claude",
        "CLAUDE_CODE_TMPDIR": "/t",
        "PATH": "/x",
        "TMPDIR": "/t",
        # not on the list
        "CLAUDE_CODE_DANGEROUSLY_SKIP_PERMISSIONS": "1",
        "MY_TOKEN": "secret",
    }
    write, forwarded, dropped = claude_config.forward_env(block, {"HTTPS_PROXY": "http://env"})
    assert forwarded == sorted(
        [
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "CLAUDE_CODE_USE_BEDROCK",
            "AWS_REGION",
            "HTTPS_PROXY",
            "no_proxy",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
            "IS_SANDBOX",
        ]
    )
    assert dropped == sorted(
        [
            "CLAUDE_CODE_PLUGIN_DIRS",
            "CLAUDE_CONFIG_DIR",
            "CLAUDE_CODE_TMPDIR",
            "PATH",
            "TMPDIR",
            "CLAUDE_CODE_DANGEROUSLY_SKIP_PERMISSIONS",
            "MY_TOKEN",
        ]
    )
    # the environment's value holds: the key is not written, the agent inherits it
    assert "HTTPS_PROXY" not in write and write["ANTHROPIC_API_KEY"] == "k"
    assert set(write) == set(forwarded) - {"HTTPS_PROXY"}
    assert claude_config.forward_env(None, {}) == ({}, [], [])


def test_settings_keys_carried_and_left_behind():
    operator = {
        "env": {"ANTHROPIC_MODEL": "m", "CLAUDE_CODE_PLUGIN_DIRS": "/mods"},
        "modelOverrides": {"claude-opus-5-5": "gw-opus"},
        "modelSettings": {"claude-opus-5-5": {"effortLevel": "high"}},
        "skipDangerousModePermissionPrompt": False,
        "skipWorkflowUsageWarning": True,
        "autoCompactEnabled": False,
        "permissions": {"defaultMode": "auto", "allow": ["Bash(*)"], "deny": ["Bash(rm -rf /:*)"]},
        "enabledPlugins": {"frontend-design@official": True},
        "hooks": {"Stop": []},
        "theme": "dark",
        "editorMode": "vim",
        "tui": "x",
        "verbose": True,
        "askUserQuestionTimeout": "5m",
        "cleanupPeriodDays": 30,
    }
    settings, info = claude_config.agent_settings(operator, {})
    assert settings == {
        "env": {"ANTHROPIC_MODEL": "m"},
        "modelOverrides": {"claude-opus-5-5": "gw-opus"},
        "modelSettings": {"claude-opus-5-5": {"effortLevel": "high"}},
        "skipWorkflowUsageWarning": True,
        "autoCompactEnabled": False,
        "permissions": {"deny": ["Bash(rm -rf /:*)"]},  # deny only: the runner sets the mode
        "skipDangerousModePermissionPrompt": True,  # whatever yours says
    }
    assert info == {
        "env_forwarded": ["ANTHROPIC_MODEL"],
        "env_dropped": ["CLAUDE_CODE_PLUGIN_DIRS"],
        "settings": [
            "modelOverrides",
            "modelSettings",
            "skipWorkflowUsageWarning",
            "autoCompactEnabled",
            "permissions",
        ],
        "settings_dropped": [
            "enabledPlugins",
            "hooks",
            "theme",
            "editorMode",
            "tui",
            "verbose",
            "askUserQuestionTimeout",
            "cleanupPeriodDays",
        ],
    }
    # permissions without deny rules are not carried at all
    settings, info = claude_config.agent_settings({"permissions": {"defaultMode": "auto"}}, {})
    assert settings == {"skipDangerousModePermissionPrompt": True}
    assert info["settings"] == [] and info["settings_dropped"] == ["permissions"]


def test_projects_linked_once_never_deleted(tmp_path):
    cdir, target = tmp_path / "c", tmp_path / "op" / "projects"
    cdir.mkdir()
    claude_config.link_projects(cdir, target)  # target made where missing
    link = cdir / "projects"
    assert link.is_symlink() and link.resolve() == target.resolve() and target.is_dir()
    claude_config.link_projects(cdir, target)  # again: nothing changes
    assert os.readlink(link) == str(target)

    # an empty directory is replaced
    link.unlink()
    link.mkdir()
    claude_config.link_projects(cdir, target)
    assert link.is_symlink()

    # a directory with session records is not touched, and the start is refused
    link.unlink()
    (link / "-p").mkdir(parents=True)
    (link / "-p" / "s.jsonl").write_text("{}\n")
    with pytest.raises(AgentSysError, match="非空目录") as e:
        claude_config.link_projects(cdir, target)
    assert e.value.status == 409 and (link / "-p" / "s.jsonl").read_text() == "{}\n"

    # a symlink to somewhere else, a file: reported, left as they are
    (link / "-p" / "s.jsonl").unlink()
    (link / "-p").rmdir()
    link.rmdir()
    link.symlink_to(tmp_path)
    with pytest.raises(AgentSysError, match="符号链接"):
        claude_config.link_projects(cdir, target)
    assert os.readlink(link) == str(tmp_path)
    link.unlink()
    link.write_text("x")
    with pytest.raises(AgentSysError, match="文件"):
        claude_config.link_projects(cdir, target)
    assert link.read_text() == "x"


def test_state_seeded_where_missing(tmp_path):
    cdir = tmp_path / "c"
    cdir.mkdir()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    claude_config.seed_state(cdir, cwd, KEY)
    path = cdir / ".claude.json"
    state = json.loads(path.read_text())
    assert state["hasCompletedOnboarding"] is True
    assert state["customApiKeyResponses"] == {"approved": [KEY[-20:]], "rejected": []}
    assert state["projects"][str(cwd)]["hasTrustDialogAccepted"] is True
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    # claude's own state is kept; a key it rejected is approved; nothing missing -> not written again
    state["numStartups"] = 7
    state["customApiKeyResponses"] = {"approved": [], "rejected": [KEY[-20:]]}
    state["projects"][str(cwd)]["lastCost"] = 1.5
    path.write_text(json.dumps(state))
    claude_config.seed_state(cdir, cwd, KEY)
    again = json.loads(path.read_text())
    assert again["numStartups"] == 7 and again["projects"][str(cwd)]["lastCost"] == 1.5
    assert again["customApiKeyResponses"] == {"approved": [KEY[-20:]], "rejected": []}
    before = path.stat().st_mtime_ns
    os.utime(path, ns=(before - 10**9, before - 10**9))
    claude_config.seed_state(cdir, cwd, KEY)
    assert path.stat().st_mtime_ns == before - 10**9

    # a file claude is in the middle of writing is left alone
    path.write_text('{"numStartups": ')
    claude_config.seed_state(cdir, tmp_path / "other", KEY)
    assert path.read_text() == '{"numStartups": '


def test_prepare_isolated_and_inherit(tmp_path, operator_claude):
    home, cwd = tmp_path / "home", tmp_path / "proj"
    home.mkdir()
    cwd.mkdir()
    c = claude_config.prepare(home, cwd, environ={})
    cdir = home / "claude"
    assert c["mode"] == "isolated" and c["dir"] == str(cdir)
    assert c["env"] == {"CLAUDE_CONFIG_DIR": str(cdir)}  # no OAuth login file
    assert c["env_dropped"] == ["CLAUDE_CODE_PLUGIN_DIRS"]
    assert c["settings"] == ["modelOverrides"] and c["settings_dropped"] == [
        "enabledPlugins",
        "theme",
    ]
    assert stat.S_IMODE(cdir.stat().st_mode) == 0o700
    settings = json.loads((cdir / "settings.json").read_text())
    assert stat.S_IMODE((cdir / "settings.json").stat().st_mode) == 0o600
    assert (
        settings["env"]["ANTHROPIC_API_KEY"] == KEY
        and "CLAUDE_CODE_PLUGIN_DIRS" not in settings["env"]
    )
    assert (
        "enabledPlugins" not in settings and settings["skipDangerousModePermissionPrompt"] is True
    )
    assert (cdir / "projects").resolve() == (operator_claude / "projects").resolve()
    assert json.loads((cdir / ".claude.json").read_text())["customApiKeyResponses"]["approved"] == [
        KEY[-20:]
    ]
    # no value in what launch.json keeps
    assert KEY not in json.dumps(c) and "gateway.example" not in json.dumps(c)

    # made again from yours on every start: a change there reaches the next run
    op = json.loads((operator_claude / "settings.json").read_text())
    op["modelOverrides"]["claude-opus-5-5"] = "gw-opus"
    (operator_claude / "settings.json").write_text(json.dumps(op))
    # MCP servers' OAuth tokens only: not handed over
    (operator_claude / ".credentials.json").write_text('{"mcpOAuth": {"atlassian|1": {}}}')
    assert "CLAUDE_SECURESTORAGE_CONFIG_DIR" not in claude_config.prepare(home, cwd, {})["env"]
    (operator_claude / ".credentials.json").write_text('{"claudeAiOauth": {"accessToken": "t"}}')
    c = claude_config.prepare(home, cwd, environ={"ANTHROPIC_BASE_URL": "https://env"})
    settings = json.loads((cdir / "settings.json").read_text())
    assert settings["modelOverrides"]["claude-opus-5-5"] == "gw-opus"
    assert "ANTHROPIC_BASE_URL" not in settings["env"]  # the environment's value holds
    # an OAuth login is used where it is
    assert c["env"]["CLAUDE_SECURESTORAGE_CONFIG_DIR"] == str(operator_claude)
    assert not (cdir / ".credentials.json").exists()

    # the way back: your ~/.claude, nothing made
    (home / "agent_sys.yaml").write_text("claude_config: inherit\n")
    assert claude_config.prepare(tmp_path / "home", cwd) == {"mode": "inherit"}
    home2 = tmp_path / "home2"
    home2.mkdir()
    (home2 / "agent_sys.yaml").write_text("claude_config: inherit\n")
    claude_config.prepare(home2, cwd)
    assert not (home2 / "claude").exists()
    # an unknown value: isolated
    (home2 / "agent_sys.yaml").write_text("claude_config: maybe\n")
    assert claude_config.mode(home2) == "isolated"


def test_models_read_what_the_agent_gets(tmp_path, operator_claude, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv("ANTHROPIC_DEFAULT_HAIKU_MODEL", raising=False)
    by_id = {m["id"]: m["resolves"] for m in launch.models(home)}
    assert by_id["haiku"] == "gw-haiku" and by_id["claude-haiku-5-5"] == "gw-haiku-5-5"
    monkeypatch.setenv(
        "ANTHROPIC_DEFAULT_HAIKU_MODEL", "env-haiku"
    )  # the environment's value first
    assert {m["id"]: m["resolves"] for m in launch.models(home)}["haiku"] == "env-haiku"
    (home / "agent_sys.yaml").write_text("claude_config: inherit\n")
    by_id = {m["id"]: m["resolves"] for m in launch.models(home)}  # yours as it is
    assert by_id["haiku"] == "gw-haiku" and by_id["claude-haiku-5-5"] == "gw-haiku-5-5"


def prepared(home, extra_args=()):
    """launch.prepare of a new task in project r, with extra_args."""
    project = load_project(home / "projects/r")
    name = f"t{len(list((home / 'projects/r/tasks/develop').glob('*.md')))}"
    new_task(
        project.path, "develop", name, "做", queue=True, agent={"extra_args": list(extra_args)}
    )
    project = load_project(project.path)
    t = project.tasks[f"develop/{name}"]
    parts, text = launch.system(home, project, t)
    return launch.prepare(
        home,
        project,
        t,
        run_dir=t.workspace / ".agent_sys/runs/20261027-100000",
        kind="terminal",
        parts=parts,
        system_text=text,
        message="m",
        resume=False,
        max_budget_usd=1,
        cost_start_usd=0.0,
        window=name,
        cli_path="/x/claude",
    )


def test_command_lines_and_launch_md(tmp_path, operator_claude):
    home = run_home(tmp_path, "autorun: false\n")
    cfg = prepared(
        home,
        [
            "--mcp-config /p/mcp.json",
            "--plugin-dir /p/a",
            "--plugin-dir=/p/b",
            "--strict-mcp-config",
        ],
    )
    cdir = str(home / "claude")
    assert cfg["env"]["CLAUDE_CONFIG_DIR"] == cdir and cfg["claude"]["mode"] == "isolated"
    cmd = launch.sdk_command(cfg)
    assert cmd.count("--strict-mcp-config") == 1 and "--setting-sources=user" in cmd
    assert cmd[cmd.index("--mcp-config") + 1] == "/p/mcp.json"
    assert [cmd[i + 1] for i, a in enumerate(cmd) if a == "--plugin-dir"] == ["/p/a", "/p/b"]
    assert launch.sdk_options(cfg).env["CLAUDE_CONFIG_DIR"] == cdir
    term = launch.terminal_argv(cfg)
    assert term.count("--strict-mcp-config") == 1
    assert term[term.index("--setting-sources") + 1] == "user"
    assert [term[i + 1] for i, a in enumerate(term) if a == "--plugin-dir"] == ["/p/a", "/p/b"]
    md = (launch.Path(cfg["system_md"]).parent / "launch.md").read_text()
    assert "## Claude Code 配置" in md and f"`{cdir}`" in md
    assert "- 不转发的 env：`CLAUDE_CODE_PLUGIN_DIRS`" in md and "`ANTHROPIC_API_KEY`" in md
    assert "- 不带的 settings 键：`enabledPlugins`、`theme`" in md
    assert "`--mcp-config /p/mcp.json`、`--plugin-dir /p/a`、`--plugin-dir /p/b`" in md
    # key names only: no value of yours in launch.md or launch.json
    for secret in (KEY, "gateway.example", "/home/me/mods", "gw-haiku"):
        assert secret not in md and secret not in json.dumps(cfg)

    # inherit: your ~/.claude, as before
    (home / "agent_sys.yaml").write_text("claude_config: inherit\n")
    cfg = prepared(home)
    assert cfg["claude"] == {"mode": "inherit"} and "CLAUDE_CONFIG_DIR" not in cfg["env"]
    assert "--strict-mcp-config" not in launch.sdk_command(cfg)
    assert "--strict-mcp-config" not in launch.terminal_argv(cfg)
    md = (launch.Path(cfg["system_md"]).parent / "launch.md").read_text()
    assert "claude_config: inherit" in md

    # a launch.json from before isolation (no claude key) starts as it did
    old = {k: v for k, v in cfg.items() if k != "claude"}
    assert "--strict-mcp-config" not in launch.sdk_command(old)


def test_extra_args_repeated_plugin_dirs_one_mcp_config():
    assert launch.parse_extra_args(["--plugin-dir a", "--plugin-dir=b", "--max-turns 3"]) == {
        "plugin-dir": ["a", "b"],
        "max-turns": "3",
    }
    with pytest.raises(AgentSysError, match="--mcp-config"):
        launch.parse_extra_args(["--mcp-config a.json", "--mcp-config b.json"])


def test_projects_that_cannot_be_linked_stop_the_start(tmp_path):
    home = run_home(tmp_path, "autorun: false\n")
    new_task(home / "projects/r", "develop", "a", "做 A", queue=True)
    with TestClient(create_app(home)) as c:
        (home / "claude/projects/-x").mkdir(parents=True)
        (home / "claude/projects/-x/s.jsonl").write_text("{}\n")
        r = c.post(f"{T}/a/actions/start")
        assert r.status_code == 409 and "非空目录" in r.json()["error"]
        assert not list((ws(home, "a") / ".agent_sys/runs").glob("*"))
        assert (home / "claude/projects/-x/s.jsonl").is_file()


def test_the_switch_is_a_setting(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "agent_sys.yaml").write_text(ops.AGENT_SYS_YAML)
    assert claude_config.mode(home) == "isolated"
    ops.save_settings(home, "global", {"claude_config": "inherit"})
    assert claude_config.mode(home) == "inherit"
    assert (
        "# isolated: the agent's own Claude Code directory" in (home / "agent_sys.yaml").read_text()
    )
    with pytest.raises(AgentSysError, match="claude_config"):
        ops.save_settings(home, "global", {"claude_config": "none"})
