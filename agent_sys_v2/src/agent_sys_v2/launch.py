"""What one run of the agent is given, written into its run directory by the server:

  .agent_sys/runs/<run_id>/system.md     agent_sys protocol + global rules + project rules + skills index +
                                     knowledge indexes, appended to Claude Code's own system prompt
                                     (--append-system-prompt-file)
  .agent_sys/runs/<run_id>/launch.json   what the runner reads: claude options, the message to send, the cost baseline
  .agent_sys/runs/<run_id>/launch.md     for people: where each injected part came from and its size, the knowledge
                                     indexes, the whole system.md (folded), the message, the equivalent claude command
                                     and the environment
  .agent_sys/runs/<run_id>/inbox.jsonl   empty; the server appends your messages and stop commands to it

The same options make the interactive claude of a terminal takeover (terminal_argv).
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

from . import checks, i18n, machines, skills, tmux
from .i18n import _, translate
from .model import (
    GLOBAL_DEFAULTS,
    AgentSysError,
    iso,
    load_config,
    mtime,
    now_iso,
    read_json,
    read_text,
    runs_dir,
    write_json,
    write_text,
)
from .scan import main_report, project_dir

# Claude Code's settings and session records. Not $CLAUDE_CONFIG_DIR: the server and the runner drop CLAUDE* from their
# environment (tmux.clean_own_env).
CLAUDE_DIR = Path.home() / ".claude"
# The agent_sys protocol in each language of agent_sys.yaml protocol_language (zh by default). The agent works in the
# language of the task's prompt whichever it gets (the protocol says so).
PROTOCOL_DIR = Path(__file__).resolve().parent / "templates" / "_common" / "rules"
PROTOCOLS = {"zh": PROTOCOL_DIR / "protocol.md", "en": PROTOCOL_DIR / "protocol.en.md"}
# The worker's environment on top of the server's cleaned one: background commands may run 24 h and are
# not reaped under memory pressure, the CLI reports session_state_changed (how a round's end is seen), and the
# worker does not write into the user's auto memory.
ENV = {
    "BASH_MAX_TIMEOUT_MS": "86400000",
    "CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP": "1",
    "CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
}
SETTINGS = '{"cleanupPeriodDays": 365}'  # keep session records for a year: a task may wait long before a resume


def settings_json(cfg):
    """--settings of a run: SETTINGS, plus ultracode (agent.ultracode: true). Ultracode is Claude Code's switch for
    multi-agent workflows (the Workflow tool on every substantial task, at a much higher cost); in a headless run it is
    turned on by this setting, with any effort."""
    return (
        json.dumps({**json.loads(SETTINGS), "ultracode": True})
        if cfg.get("ultracode")
        else SETTINGS
    )


# An upstream's dependency kind in the first message (translated where it is used). What agent_sys tells the agent
# (system.md, the first message, messages into its session) is in protocol_language: translate(text, lang), or
# say_for(home, text) where only the home is at hand.
KIND_ZH = {
    "input": "产出依赖",
    "code": "代码基线",
    "children": "等它和它的子任务完成",
    "review": "评审覆盖",
    "subtask": "父任务",
}


def agent_language(home):
    """agent_sys.yaml protocol_language: the language of the protocol and of what agent_sys tells the agent (zh or en)."""
    lang = load_config(home).get("protocol_language")
    return lang if lang in PROTOCOLS else GLOBAL_DEFAULTS["protocol_language"]


def say_for(home, text, /, **params):
    """text (Chinese) in the language of what agent_sys tells the agents of this AGENT_SYS_HOME."""
    return translate(text, agent_language(home), **params)


def protocol(home):
    """The protocol file a run of this AGENT_SYS_HOME gets."""
    return PROTOCOLS[agent_language(home)]


def rules_dir(home, pid=None):
    return project_dir(home, pid) / "rules" if pid else Path(home) / "rules"


def knowledge_dir(home, pid=None):
    return project_dir(home, pid) / "knowledge" if pid else Path(home) / "knowledge"


def bundled_cli():
    """The claude executable that comes with the pinned claude-agent-sdk."""
    import claude_agent_sdk

    return str(Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude")


# The models offered where one is chosen (agent of a task, project defaults), [(id, label)]. An alias
# (opus, sonnet, haiku) is resolved by Claude Code: ANTHROPIC_DEFAULT_<X>_MODEL in ~/.claude/settings.json (the
# environment's is dropped, tmux.clean_own_env), else the CLI's own default for it; any other name can be typed in.
MODELS = [
    ("opus", "Opus"),
    ("sonnet", "Sonnet"),
    ("haiku", "Haiku"),
    ("claude-opus-5-5", "Opus 5.5"),
    ("claude-sonnet-5-5", "Sonnet 5.5"),
    ("claude-haiku-5-5", "Haiku 5.5"),
    ("claude-haiku-4-5", "Haiku 4.5"),
]


def user_settings():
    return read_json(CLAUDE_DIR / "settings.json")


def models():
    """MODELS as [{id, label, resolves}]: resolves is the model an alias stands for, or the name modelOverrides
    sends a model id to, as far as ~/.claude/settings.json says ('' when it does not)."""
    settings = user_settings()
    env = {k: str(v) for k, v in (settings.get("env") or {}).items()}
    overrides = (
        settings.get("modelOverrides") if isinstance(settings.get("modelOverrides"), dict) else {}
    )
    out = []
    for mid, label in MODELS:
        resolves = env.get(f"ANTHROPIC_DEFAULT_{mid.upper()}_MODEL", "") if "-" not in mid else ""
        resolves = str(overrides.get(resolves or mid) or resolves)
        out.append({"id": mid, "label": label, "resolves": "" if resolves == mid else resolves})
    return out


# ---------------------------------------------------------------- system.md


def part_title(kind, lang):
    """The title of a part of system.md, in lang."""
    return {
        "protocol": translate("agent_sys 约定", lang),
        "rules": translate("全局规则", lang),
        "project_rules": translate("项目规则", lang),
        "skills": translate("skills 索引", lang),
        "project_knowledge": translate("项目经验索引", lang),
        "knowledge": translate("全局经验索引", lang),
    }[kind]


KNOWLEDGE = ("project_knowledge", "knowledge")  # the parts that are knowledge indexes


def system_parts(home, project):
    """The parts of system.md in order, [{kind, title, path, text}] with the title in the UI's language; missing or
    empty files are left out. project None: only what every project gets (the global rules page)."""
    files = [("protocol", protocol(home))]
    files += [("rules", p) for p in sorted(rules_dir(home).glob("*.md"))]
    if project:
        files += [("project_rules", p) for p in sorted(rules_dir(home, project.id).glob("*.md"))]
        files.append(("skills", None))
        files.append(("project_knowledge", knowledge_dir(home, project.id) / "INDEX.md"))
    files.append(("knowledge", knowledge_dir(home) / "INDEX.md"))
    parts = []
    for kind, path in files:
        text = skills.index(project) if kind == "skills" else read_text(path).strip()
        if text:
            path = skills.source(project) if kind == "skills" else str(path)
            parts.append(
                {"kind": kind, "title": part_title(kind, i18n.LANG), "path": path, "text": text}
            )
    return parts


def system_md(home, project, task, parts, lang="zh"):
    """system.md: a head that says where things are (rules written with relative paths do not hold for a worker
    whose cwd is the project root), then each part with its source; agent_sys's own words in lang."""
    head = (
        translate(
            "# 这个项目的工作约定\n\n下面是我给这个项目的所有任务定的约定和规则，每次运行都会附上。\n\n",
            lang,
        )
        + translate("- 项目目录（也是当前目录）：`{project}`\n", lang, project=project.path)
        + translate(
            "- AGENT_SYS_HOME：`{home}`（全局规则在其中的 `rules/`，全局经验库在 `knowledge/`）\n",
            lang,
            home=home,
        )
        + translate(
            "- 本任务：prompt 是 `tasks/{task}.md`，工作区是 `workspaces/{task}/`\n",
            lang,
            task=task.id,
        )
        + translate(
            "- 规则里写的相对路径先按项目目录找；找不到时，按该规则文件所在的目录（下面每段开头注明了来源）理解。\n",
            lang,
        )
    )
    out = [head]
    colon = translate("：", lang)
    for p in parts:
        title, intro = part_title(p["kind"], lang), ""
        if p["kind"] == "skills":
            intro = translate(
                "{title}：一行一个 skill（名字、用途、SKILL.md 的路径）；用得上时先读它的 SKILL.md，里面有步骤和脚本。\n\n",
                lang,
                title=title,
            )
        if p["kind"] in KNOWLEDGE:
            intro = translate(
                "{title}：一行一条经验，正文在 `{dir}/entries/<name>.md`，用得上时再读。\n\n",
                lang,
                title=title,
                dir=Path(p["path"]).parent,
            )
        out.append(f"<!-- {title}{colon}{p['path']} -->\n{intro}{p['text']}\n")
    return "\n---\n\n".join(out)


def system(home, project, task):
    """(parts, system.md text) for a run of task."""
    parts = system_parts(home, project)
    return parts, system_md(home, project, task, parts, agent_language(home))


def known_system_md(workspace):
    """The system.md the session last got to know, '' if none: Claude Code keeps the system prompt a session was
    created with, on every resume, so it is the one of the run that created the session, or of a later run
    whose message told the agent to read it."""
    for run in sorted(runs_dir(workspace).glob("*/launch.json"), reverse=True):
        if read_json(run).get("system_known"):
            return read_text(run.parent / "system.md")
    return ""


# ---------------------------------------------------------------- the message

HANDOFF_LINES = 15  # items of an upstream's handoff list in the first message


def report_ref(t, lang):
    report = main_report(t.workspace)
    if not report:
        return translate("还没有 report.md", lang)
    return translate("报告 `workspaces/{task}/{report}`", lang, task=t.id, report=report)


def upstream_lines(project, task, lang="zh"):
    """One line per upstream task: title, kind, workspace, report and the 下一步 of its progress; then the handoff
    list of its report, when it has one, and for a children dependency its subtasks with their reports."""
    lines = []
    for e in project.upstream(task.id):
        up = project.tasks[e.up]
        line = translate(
            "- {title}（`{task}`，{kind}）：工作区 `workspaces/{task}/`，{report}",
            lang,
            title=up.title,
            task=up.id,
            kind=translate(KIND_ZH.get(e.kind, e.kind), lang),
            report=report_ref(up, lang),
        )
        if up.progress and up.progress.next:
            line += translate("；progress 的下一步：{next}", lang, next=up.progress.next)
        lines.append(line)
        # the upstream's own list of what it leaves for you: read before its workspace
        if up.handoff:
            lines.append("  " + translate("交接清单：", lang))
            lines += [f"  - {h['text']}" for h in up.handoff[:HANDOFF_LINES]]
            if len(up.handoff) > HANDOFF_LINES:
                lines.append(
                    "  - " + translate("……（共 {n} 条，其余见报告）", lang, n=len(up.handoff))
                )
        # what a round's close task integrates: the plan's subtasks, each with its report
        if e.kind == "children" and (
            kids := [project.tasks[c] for c in project.children(up.id) if c != task.id]
        ):
            lines.append(translate("  它的子任务：", lang))
            lines += [
                translate(
                    "  - {title}（`{task}`，{status}）：{report}",
                    lang,
                    title=c.title,
                    task=c.id,
                    status=c.status,
                    report=report_ref(c, lang),
                )
                for c in kids
            ]
    return lines


def first_message(project, task, reply="", lang="zh"):
    """The message a task's session starts with: which task, its upstream tasks, its machines and checks, the prompt.
    A task joined with work already done (纳入托管) is told to read its progress.md first, with your reply."""
    out = [
        translate(
            "请完成任务 `tasks/{task}.md`，工作区是 `workspaces/{task}/`。", lang, task=task.id
        )
    ]
    ups = upstream_lines(project, task, lang)
    if ups:
        out.append(
            translate("上游任务（需要时读它们的 report.md 和 progress.md）：", lang)
            + "\n"
            + "\n".join(ups)
        )
    if task.machines:
        gpus = {m["name"]: m.get("gpus") for m in machines.declared(project.config)}
        names = [
            name + (translate("（{gpus} GPU）", lang, gpus=gpus[name]) if gpus.get(name) else "")
            for name in task.machines
        ]
        out.append(
            translate(
                "本任务可以使用的机器：{machines}。任务运行期间和等我回复期间，其他任务不会被安排到这些机器上；不要使用别的机器。",
                lang,
                machines=", ".join(names),
            )
        )
    if task.checks:
        out.append(
            translate(
                "本任务的验收检查：你把 progress.md 标为已完成后，系统在项目目录下依次运行它们，全部通过才算完成；"
                "没通过时会把输出发给你。",
                lang,
            )
            + "\n"
            + "\n".join(
                f"- `{c['run']}`"
                if c["name"] == c["run"]
                else translate("- {name}：`{run}`", lang, name=c["name"], run=c["run"])
                for c in task.checks
            )
        )
    out.append(
        translate("任务说明（`tasks/{task}.md` 的正文）：", lang, task=task.id)
        + "\n\n"
        + (task.body.strip() or translate("（空）", lang))
    )
    if task.progress is not None:
        out.append(
            translate(
                "工作区里已有进展，请先读 `workspaces/{task}/progress.md`，接着做。",
                lang,
                task=task.id,
            )
        )
    if reply.strip():
        out.append(translate("我的回复：{reply}", lang, reply=reply.strip()))
    return "\n\n".join(out)


def resume_message(text, rules_updated="", wait_interrupted=False, moved=None, lang="zh"):
    """Your message to an existing session, after what agent_sys has to tell first. moved: (old id, new id) when the task
    was renamed or changed type since the session last ran."""
    out = []
    if moved:
        out.append(
            translate(
                "（这个任务已从 `tasks/{old}.md` 移到 `tasks/{new}.md`，工作区现在是 `workspaces/{new}/`，"
                "旧路径已失效。）",
                lang,
                old=moved[0],
                new=moved[1],
            )
        )
    if rules_updated:
        out.append(
            translate(
                "（agent_sys 规则有更新，请先读 `{path}`，之后按新规则做。）",
                lang,
                path=rules_updated,
            )
        )
    if wait_interrupted:
        out.append(
            translate(
                "（上次运行被中断，后台等待的命令已经结束。请先检查在等的条件，需要的话重新起后台等待。）",
                lang,
            )
        )
    out.append(text.strip())
    return "\n\n".join(out)


# ---------------------------------------------------------------- claude options


def extra_args(value):
    """agent.extra_args as a list of strings: a single string is one item."""
    if value in (None, ""):
        return []
    return [str(a) for a in (value if isinstance(value, list) else [value])]


def parse_extra_args(args):
    """The task's extra_args (each item as written: '--strict-mcp-config', '--max-turns 5', '--x=y') -> the SDK's
    {flag: value or None}."""
    try:
        tokens = [tok for a in extra_args(args) for tok in shlex.split(a)]
    except ValueError as e:
        raise AgentSysError(_("agent.extra_args 无法解析：{error}", error=e)) from None
    out, i = {}, 0
    while i < len(tokens):
        tok = tokens[i].lstrip("-")
        if "=" in tok:
            key, value = tok.split("=", 1)
            out[key] = value
        elif i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
            out[tok] = tokens[i + 1]
            i += 1
        else:
            out[tok] = None
        i += 1
    return out


def sdk_options(cfg, **extra):
    """ClaudeAgentOptions for a run from its launch.json (the preset system prompt, bypassPermissions on
    every start since a resume does not keep it, user settings only, --session-id first and --resume after)."""
    from claude_agent_sdk import ClaudeAgentOptions

    sid = cfg["session_id"]
    return ClaudeAgentOptions(
        cli_path=cfg["cli_path"],
        model=cfg.get("model") or None,
        effort=cfg.get("effort") or None,
        cwd=cfg["cwd"],
        system_prompt={"type": "preset", "preset": "claude_code"},
        permission_mode="bypassPermissions",
        setting_sources=["user"],
        session_id=None if cfg["resume"] else sid,
        resume=sid if cfg["resume"] else None,
        max_budget_usd=cfg["max_budget_usd"],
        settings=settings_json(cfg),
        env=dict(cfg["env"]),
        extra_args={
            "append-system-prompt-file": cfg["system_md"],
            **parse_extra_args(cfg.get("extra_args")),
        },
        **extra,
    )


def sdk_command(cfg):
    """The claude command line the SDK runs for these options (the equivalent command in launch.md)."""
    from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

    return SubprocessCLITransport(prompt="", options=sdk_options(cfg))._build_command()


def terminal_argv(cfg):
    """Interactive claude for a terminal takeover: the runner's permission mode, setting sources and system.md; an
    existing session is resumed, a new one starts with the first message. --max-budget-usd only works
    with --print, so the budget does not hold in the terminal."""
    sid = cfg["session_id"]
    argv = [cfg["cli_path"], *(["--resume", sid] if cfg["resume"] else ["--session-id", sid])]
    for flag, value in (("--model", cfg.get("model")), ("--effort", cfg.get("effort"))):
        if value:
            argv += [flag, str(value)]
    argv += [
        "--permission-mode",
        "bypassPermissions",
        "--setting-sources",
        "user",
        "--settings",
        settings_json(cfg),
        "--append-system-prompt-file",
        cfg["system_md"],
    ]
    for flag, value in parse_extra_args(cfg.get("extra_args")).items():
        argv += [f"--{flag}"] + ([value] if value is not None else [])
    if not cfg["resume"] and cfg.get("message"):
        argv.append(cfg["message"])
    return argv


def shell_command(cwd, env, argv):
    lines = [
        f"cd {shlex.quote(str(cwd))}",
        "env " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " \\",
    ]
    return "\n".join(lines) + "\n  " + shlex.join(argv)


# ---------------------------------------------------------------- the run directory


def kb(n):
    return f"{n / 1024:.1f} KB"


def fenced(text, info=""):
    """text in a code fence longer than any run of backticks inside it."""
    fence = "`" * max([3, *(len(m) + 1 for m in re.findall(r"`+", text))])
    return f"{fence}{info}\n{text.rstrip(chr(10))}\n{fence}"


def launch_md(cfg, task, parts, system_text, unknown_skills=()):
    """launch.md, for people (shown in the run's 启动信息): in the language of agent_sys.yaml; the system.md and the
    message it quotes are what the agent got."""
    kind = {
        "sdk": _("首次运行") if not cfg["resume"] else "resume",
        "replay": _("回放（不调用模型）"),
        "terminal": _("在终端中接管"),
    }[cfg["kind"]]
    size = kb(len(system_text.encode()))
    out = [
        _("# 运行 {id}", id=cfg["id"]),
        "",
        _("- 任务：`{task}`（{title}）", task=task.id, title=task.title),
        _(
            "- 类型：{kind}；会话 `{session}`（{flag}）",
            kind=kind,
            session=cfg["session_id"],
            flag="--resume" if cfg["resume"] else "--session-id",
        ),
        _("- 开始：{started}；cwd：`{cwd}`", started=cfg["started"], cwd=cfg["cwd"]),
        _(
            "- model：{model}；effort：{effort}；本次预算上限 ${budget}（任务预算减去已花费）",
            model=cfg.get("model") or _("默认"),
            effort=(cfg.get("effort") or _("默认"))
            + (" + ultracode" if cfg.get("ultracode") else ""),
            budget=f"{cfg['max_budget_usd']:.2f}",
        ),
        "",
        _("## 注入的 system.md（{size}）", size=size),
        "",
        _(
            "文件：`{path}`。Claude Code 在会话创建时保存 system prompt，resume 时沿用那一份；"
            "规则或经验索引更新后，resume 的消息开头会提示 agent 读新的 system.md。",
            path=cfg["system_md"],
        ),
        "",
        _("| # | 内容 | 来源 | 大小 |"),
        "|---|---|---|---|",
    ]
    out += [
        f"| {i} | {p['title']} | `{p['path']}` | {kb(len(p['text'].encode()))} |"
        for i, p in enumerate(parts, 1)
    ]
    if unknown_skills:
        out += [
            "",
            _(
                "project.yaml 的 skills 中这些名字不存在：{names}",
                names=_("、").join(unknown_skills),
            ),
        ]
    for p in parts:
        if p["kind"] in KNOWLEDGE:
            n = len(p["text"].splitlines())
            out += ["", _("### {title}（{n} 行）", title=p["title"], n=n), "", fenced(p["text"])]
    out += [
        "",
        _("<details><summary>system.md 全文（规则和索引，{size}）</summary>", size=size),
        "",
        fenced(system_text, "markdown"),
        "",
        "</details>",
    ]
    if cfg["kind"] == "replay":
        r = cfg["replay"]
        gap = _("，间隔最长 {gap} 秒", gap=r["max_gap"]) if r.get("max_gap") else ""
        out += [
            "",
            _("## 回放"),
            "",
            _(
                "回放 `{events}`（速度 ×{speed}{gap}），不调用模型；发出的消息只记录，"
                "由回放给出一条说明作为回复。",
                events=r["events"],
                speed=r.get("speed") or 1,
                gap=gap,
            ),
        ]
    out += [
        "",
        _("## 本次发送的消息") if cfg["resume"] else _("## 首条消息"),
        "",
        fenced(cfg["message"], "markdown") if cfg["message"] else _("（无）"),
    ]
    if cfg["kind"] == "terminal":
        out += [
            "",
            _("## 终端中的命令"),
            "",
            fenced(shell_command(cfg["cwd"], cfg["env"], terminal_argv(cfg))),
            "",
            _(
                "在 tmux 窗口 `{window}` 中运行：`{attach}`。"
                "退出 claude 后，agent_sys 读回会话累计花费，并按 progress.md 判断任务状态。",
                window=cfg["window"],
                attach=tmux.attach(cfg["window"]),
            ),
        ]
    else:
        how = (_("回放没有启动 claude；真实运行时，") if cfg["kind"] == "replay" else "") + _(
            "runner 用 Claude Agent SDK 以下面的参数启动 claude，消息从标准输入以 stream-json 发送："
        )
        out += [
            "",
            _("## 等价命令"),
            "",
            how,
            "",
            fenced(shell_command(cfg["cwd"], cfg["env"], sdk_command(cfg))),
        ]
    out += [
        "",
        _("## 环境变量"),
        "",
        _("在 agent-sys-v2 serve 清理过的环境（去掉 `CLAUDE*`、`ANTHROPIC*`、`IS_SANDBOX`）上加："),
        "",
    ]
    out += [f"- `{k}={v}`" for k, v in cfg["env"].items()]
    return "\n".join(out) + "\n"


def write_run(run_dir, cfg, system_text, md):
    """Create the run directory: system.md, launch.json, launch.md and an empty inbox.jsonl."""
    run_dir.mkdir(parents=True)
    write_text(run_dir / "system.md", system_text)
    write_json(run_dir / "launch.json", cfg)
    write_text(run_dir / "launch.md", md)
    (run_dir / "inbox.jsonl").touch()


def session_records(session_id):
    """Claude Code's records of the session, ~/.claude/projects/<cwd>/<sid>.jsonl."""
    return list((CLAUDE_DIR / "projects").glob(f"*/{session_id}.jsonl"))


def session_exists(session_id):
    """Claude Code has a record of this session: a first run that died before claude started left none, and must
    start the session again rather than resume it."""
    return bool(session_records(session_id))


def session_cost(session_id):
    """The session's total_cost_usd from the last cost-state line of its record, None if there is none."""
    cost = None
    for path in session_records(session_id):
        for line in read_text(path).splitlines():
            if '"cost-state"' in line:
                try:
                    cost = json.loads(line).get("totalCostUSD", cost)
                except json.JSONDecodeError:
                    pass
    return cost


def prepare(
    home,
    project,
    task,
    *,
    run_dir,
    kind,
    parts,
    system_text,
    message,
    resume,
    max_budget_usd,
    cost_start_usd,
    window,
    replay=None,
    cli_path=None,
    system_known=None,
):
    """Everything a run needs, written into run_dir; returns launch.json as a dict. parts / system_text from system(),
    message as it is to be sent (first_message / resume_message make it). system_known: this run's system.md reaches
    the agent (a new session; or a resume whose message says the rules changed)."""
    agent = task.agent
    report = main_report(task.workspace)
    cfg = {
        "id": run_dir.name,
        "kind": kind,
        "task": task.id,
        "project": project.id,
        "cwd": str(project.path),
        "session_id": task.state["session_id"],
        "resume": resume,
        "model": agent.get("model") or "",
        "effort": agent.get("effort") or "",
        "ultracode": agent.get("ultracode") is True,
        "max_budget_usd": round(max_budget_usd, 4),
        "extra_args": extra_args(agent.get("extra_args")),
        "cli_path": cli_path or bundled_cli(),
        "system_md": str(run_dir / "system.md"),
        "env": {
            **ENV,
            **checks.env(project.path, task.id, task.workspace, task.machines, skills.SHIPPED),
        },
        "checks": task.checks,
        "check_retries": checks.retries(task.fm),
        "message": message,
        "window": window,
        "started": now_iso(),
        "cost_start_usd": cost_start_usd,
        "replay": replay,
        "language": i18n.LANG,
        "agent_language": agent_language(home),
        "system_known": (not resume) if system_known is None else system_known,
        # what the task was before this run: a done task that only answered a question stays done
        "before": {
            "status": task.status,
            "detail": task.detail,
            "report": report,
            "report_mtime": iso(mtime(task.workspace / report)) if report else "",
        },
    }
    # made before the run directory: bad extra_args stop the start here, with nothing written
    md = launch_md(cfg, task, parts, system_text, skills.unknown(project))
    write_run(run_dir, cfg, system_text, md)
    return cfg
