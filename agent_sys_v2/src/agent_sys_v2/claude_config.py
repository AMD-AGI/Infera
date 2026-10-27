"""The Claude Code configuration an agent runs with: agent_sys.yaml `claude_config`.

isolated (the default): a configuration directory of its own, $AGENT_SYS_HOME/claude, given as CLAUDE_CONFIG_DIR.
Your ~/.claude holds your MCP servers (~/.claude.json), your plugins (enabledPlugins, CLAUDE_CODE_PLUGIN_DIRS), your
CLAUDE.md, skills and hooks: a session in bypassPermissions should get none of them. What it needs to log in and reach
its models is taken from ~/.claude on every start (prepare), into the agent's directory:

  settings.json   ~/.claude/settings.json reduced to what the agent needs: the env keys FORWARD_ENV matches (a key
                  set in agent_sys's own environment is left out, so that the environment's value holds), the keys of
                  CARRIED, and skipDangerousModePermissionPrompt. The values go nowhere else: launch.md and launch.json
                  name the keys only.
  .claude.json    Claude Code's own state, where it is missing: onboarding done, the forwarded API key approved, the
                  project directory trusted. Without them an interactive claude (a terminal takeover) opens on the
                  theme picker, the API key question, the login and the trust question.
  projects/       a symlink to ~/.claude/projects: session records stay where resume, a terminal takeover and
                  session_cost look for them, those of sessions started before isolation included.

A Claude login (claudeAiOauth in ~/.claude/.credentials.json) is used where it is: CLAUDE_SECURESTORAGE_CONFIG_DIR
points there. Not a symlink: Claude Code writes the file without following one (O_NOFOLLOW), and a copy would go stale
at the first token refresh. A credentials file without a Claude login (only MCP servers' OAuth tokens, say) is not
handed to the agent.
The runner adds --strict-mcp-config; MCP servers and plugins a project needs go in agent.extra_args
(--mcp-config <file>, --plugin-dir <dir>), and launch.md lists them.

inherit: your ~/.claude as it is, everything above loaded (how runs worked before isolation).
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

from .i18n import _
from .model import AgentSysError, load_config, read_json, read_text

# Your Claude Code directory: its settings, its login and the session records. Not $CLAUDE_CONFIG_DIR: the server and
# the runner drop CLAUDE* from their environment (tmux.clean_own_env).
OPERATOR_DIR = Path.home() / ".claude"
MODES = ("isolated", "inherit")

# The env keys of ~/.claude/settings.json the agent gets: login and model routing (ANTHROPIC_*, the cloud providers'
# switches and credentials, client certificates, an OAuth token), the network (proxies, CA certificates) and what the
# gateway or the model needs (no nonessential traffic, no adaptive thinking, output and thinking limits, timeouts).
# IS_SANDBOX lets bypassPermissions run as root. Any other key stays behind; launch.md names it.
FORWARD_ENV = re.compile(
    r"ANTHROPIC_\w+|AWS_\w+|GOOGLE_\w+|CLOUD_ML_REGION|VERTEX_REGION_\w+"
    r"|CLAUDE_CODE_(USE|SKIP)_(BEDROCK|VERTEX|FOUNDRY)(_AUTH)?|CLAUDE_CODE_OAUTH_TOKEN"
    r"|CLAUDE_CODE_API_KEY_HELPER_TTL_MS|CLAUDE_CODE_CLIENT_(CERT|KEY|KEY_PASSPHRASE)"
    r"|CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC|CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"
    r"|CLAUDE_CODE_MAX_OUTPUT_TOKENS|CLAUDE_CODE_SUBAGENT_MODEL|MAX_THINKING_TOKENS|API_TIMEOUT_MS"
    r"|DISABLE_(TELEMETRY|ERROR_REPORTING|AUTOUPDATER|NON_ESSENTIAL_MODEL_CALLS|PROMPT_CACHING\w*)"
    r"|(HTTP|HTTPS|NO|http|https|no)_(PROXY|proxy)|NODE_EXTRA_CA_CERTS|SSL_CERT_FILE|IS_SANDBOX"
)
# Never forwarded, whatever FORWARD_ENV says: your plugins, your configuration directory, your PATH and temp dirs.
NEVER_ENV = frozenset(
    {"CLAUDE_CODE_PLUGIN_DIRS", "CLAUDE_CONFIG_DIR", "CLAUDE_CODE_TMPDIR", "PATH", "TMPDIR"}
)
# The settings keys besides env the agent gets, and why. Any other key stays behind (launch.md names it): plugins
# (enabledPlugins, extraKnownMarketplaces), hooks, what only your own screen shows (theme, editorMode, tui, verbose,
# statusLine, outputStyle, askUserQuestionTimeout), cleanupPeriodDays (the runner sets its own).
CARRIED = {
    "model": "the model of a task that names none",
    "modelOverrides": "model ids (claude-opus-5-5 …) to the names your gateway knows",
    "modelSettings": "per-model settings such as the effort level",
    "apiKeyHelper": "login: the command that prints the API key",
    "awsAuthRefresh": "login to Bedrock",
    "awsCredentialExport": "login to Bedrock",
    "autoCompactEnabled": "how a long session goes on when its context is full: as in your own sessions",
    "skipWorkflowUsageWarning": "else an ultracode terminal takeover opens on the usage warning",
    "permissions": "only its deny rules: they narrow what the agent may do; the runner sets the permission mode",
}
# Set whatever ~/.claude says: a terminal takeover runs claude in bypassPermissions, which otherwise asks first.
FORCED = {"skipDangerousModePermissionPrompt": True}
# Claude Code's state (.claude.json) a new directory gets: no onboarding. (An interactive claude still clones the
# official plugin marketplace's catalog into the directory; no plugin of it is enabled.)
STATE = {"hasCompletedOnboarding": True}

_lock = threading.Lock()  # two runs starting at once write the same files


def mode(home):
    """agent_sys.yaml claude_config: isolated (default) or inherit."""
    value = load_config(home).get("claude_config")
    return value if value in MODES else MODES[0]


def config_dir(home):
    return Path(home) / "claude"


def operator_settings():
    return read_json(OPERATOR_DIR / "settings.json")


def forward_env(block, environ):
    """The env block of your settings.json -> (what the agent's settings.json gets, the keys forwarded, the keys
    left behind). A forwarded key set in environ is not written: the agent inherits the environment's value."""
    write, forwarded, dropped = {}, [], []
    for key, value in (block if isinstance(block, dict) else {}).items():
        if key in NEVER_ENV or not FORWARD_ENV.fullmatch(key):
            dropped.append(key)
            continue
        forwarded.append(key)
        if key not in environ:
            write[key] = str(value)
    return write, sorted(forwarded), sorted(dropped)


def agent_settings(operator, environ):
    """settings.json of the agent's directory from yours -> (settings, {env_forwarded, env_dropped, settings,
    settings_dropped}: key names only)."""
    env, forwarded, env_dropped = forward_env(operator.get("env"), environ)
    out = {"env": env} if env else {}
    for key in CARRIED:
        if key not in operator:
            continue
        value = operator[key]
        if key == "permissions":
            deny = value.get("deny") if isinstance(value, dict) else None
            if not deny:
                continue
            value = {"deny": deny}
        out[key] = value
    # what was written: permissions without deny rules is not
    carried = [k for k in operator if k in out and k != "env"]
    dropped = [k for k in operator if k not in out and k not in FORCED and k != "env"]
    out.update(FORCED)
    info = {
        "env_forwarded": forwarded,
        "env_dropped": env_dropped,
        "settings": carried,
        "settings_dropped": dropped,
    }
    return out, info


def effective_settings(home, environ=None):
    """The settings the agent's claude reads (what an alias or a model id resolves to, launch.models): yours under
    inherit; else your settings reduced as the agent gets them, env values from environ first."""
    operator = operator_settings()
    if mode(home) == "inherit":
        return operator
    environ = os.environ if environ is None else environ
    settings, info = agent_settings(operator, environ)
    env = dict(operator.get("env") or {}) if isinstance(operator.get("env"), dict) else {}
    settings["env"] = {k: str(environ.get(k, env.get(k))) for k in info["env_forwarded"]}
    return settings


def write_private(path, data):
    """data as JSON, readable by you only, through a temporary file and a rename."""
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}.{threading.get_ident()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def link_projects(cdir, target):
    """cdir/projects -> target (made if missing), once. A symlink there already is kept if it points at target; a
    real directory is replaced only when empty: anything else is reported, nothing is deleted."""
    link = cdir / "projects"
    target.mkdir(parents=True, exist_ok=True)
    for _attempt in range(2):
        if link.is_symlink():
            if link.resolve() == target.resolve():
                return
            raise AgentSysError(
                _(
                    "{link} 是指向 {other} 的符号链接，应指向 {target}：没有改动它，请检查后删掉它再开始",
                    link=link,
                    other=os.readlink(link),
                    target=target,
                ),
                409,
            )
        if link.is_dir():
            if any(link.iterdir()):
                raise AgentSysError(
                    _(
                        "{link} 是一个非空目录，应是指向 {target} 的符号链接：没有删除它，"
                        "请把里面的会话记录移到 {target} 后删掉它再开始",
                        link=link,
                        target=target,
                    ),
                    409,
                )
            link.rmdir()
        elif link.exists():
            raise AgentSysError(
                _("{link} 应是指向 {target} 的符号链接，现在是一个文件", link=link, target=target),
                409,
            )
        try:
            link.symlink_to(target, target_is_directory=True)
            return
        except FileExistsError:  # made meanwhile by another start: look again
            continue


def seed_state(cdir, cwd, api_key=""):
    """Claude Code's state in cdir/.claude.json where it is missing: STATE, the API key approved (Claude Code keeps
    its last 20 characters), the project directory trusted. Written only when something was missing; a file that
    cannot be read now (claude writing it) is left alone."""
    path = cdir / ".claude.json"
    if path.exists():
        try:
            state = json.loads(read_text(path))
        except json.JSONDecodeError:
            return
        if not isinstance(state, dict):
            return
    else:
        state = {}
    new = json.loads(json.dumps(state))
    for key, value in STATE.items():
        new.setdefault(key, value)
    if api_key:
        responses = new.get("customApiKeyResponses")
        responses = responses if isinstance(responses, dict) else {}
        approved = [k for k in responses.get("approved") or [] if isinstance(k, str)]
        if api_key[-20:] not in approved:
            approved.append(api_key[-20:])
        rejected = [k for k in responses.get("rejected") or [] if k != api_key[-20:]]
        new["customApiKeyResponses"] = {**responses, "approved": approved, "rejected": rejected}
    projects = new.get("projects") if isinstance(new.get("projects"), dict) else {}
    for p in dict.fromkeys([str(cwd), os.path.realpath(cwd)]):
        entry = projects.get(p) if isinstance(projects.get(p), dict) else {}
        projects[p] = {**entry, "hasTrustDialogAccepted": True}
    new["projects"] = projects
    if new != state:
        write_private(path, new)


def prepare(home, cwd, environ=None):
    """The agent's Claude Code configuration for a run in cwd, made now -> what launch.json keeps of it: {mode} under
    inherit; else {mode, dir, env (the variables the agent gets), and the key names of agent_settings}. Raises
    AgentSysError when projects/ cannot be linked."""
    if mode(home) == "inherit":
        return {"mode": "inherit"}
    environ = os.environ if environ is None else environ
    cdir = config_dir(home)
    operator = operator_settings()
    settings, info = agent_settings(operator, environ)
    env_block = operator.get("env") if isinstance(operator.get("env"), dict) else {}
    api_key = environ.get("ANTHROPIC_API_KEY") or str(env_block.get("ANTHROPIC_API_KEY") or "")
    with _lock:
        cdir.mkdir(parents=True, exist_ok=True)
        cdir.chmod(0o700)
        link_projects(cdir, OPERATOR_DIR / "projects")
        write_private(cdir / "settings.json", settings)
        seed_state(cdir, cwd, api_key if "ANTHROPIC_API_KEY" in info["env_forwarded"] else "")
    env = {"CLAUDE_CONFIG_DIR": str(cdir)}
    if read_json(OPERATOR_DIR / ".credentials.json").get("claudeAiOauth"):
        env["CLAUDE_SECURESTORAGE_CONFIG_DIR"] = str(OPERATOR_DIR)
    return {"mode": "isolated", "dir": str(cdir), "env": env, **info}
