"""The language of what agent_sys says: to you in agent_sys.yaml `language` (en, the default, or zh), to the agent in
`protocol_language` (launch.agent_language).

Texts are written in Chinese, the language agent_sys was first written in, and looked up in EN for English:
_('任务「{tid}」不存在', tid=tid) in the UI's language, translate(text, lang) in another one. A text missing from EN
is shown in Chinese (tests/test_i18n.py checks that none is missing and that every entry is used). {name} is filled
from the keyword arguments; format the values first ({x:.2f} is not read).

The server sets the UI's language when it starts and when the setting is saved (and then scans every project again: the
reminders on the cards are made at scan time); the CLI sets it from the AGENT_SYS_HOME it works on, the runner from its
launch.json.
"""

from __future__ import annotations

import re
from pathlib import Path

LANGUAGES = ("en", "zh")
DEFAULT = "en"
LANG = DEFAULT
PARAM = re.compile(r"\{(\w+)\}")


def normalize(value):
    """'zh', 'zh-CN', '中文' -> zh; anything else -> en."""
    return "zh" if str(value or "").strip().lower().startswith(("zh", "中")) else "en"


def set_language(value):
    global LANG
    LANG = normalize(value)
    return LANG


def use(home):
    """The language of AGENT_SYS_HOME's agent_sys.yaml (en if it has none or cannot be read)."""
    from .model import load_config  # model imports this module

    try:
        return set_language(load_config(Path(home)).get("language"))
    except Exception:  # noqa: BLE001 - a broken agent_sys.yaml: the settings page says so; speak the default meanwhile
        return set_language(DEFAULT)


def translate(text, lang, /, **params):
    """text (Chinese) in lang, {name} filled from params (a name without a value stays as written)."""
    s = EN.get(text, text) if lang == "en" else text
    if params:
        s = PARAM.sub(lambda m: str(params[m[1]]) if m[1] in params else m[0], s)
    return s


def _(text, /, **params):
    """text in the UI's language."""
    return translate(text, LANG, **params)


STATUS = {
    "draft": ("草稿", "draft"),
    "queued": ("待开始", "queued"),
    "running": ("运行中", "running"),
    "needs_you": ("待确认", "needs you"),
    "in_review": ("待审核", "in review"),
    "done": ("已完成", "done"),
    "cancelled": ("已取消", "cancelled"),
    "unknown": ("其他", "other"),
}


def status_word(status):
    """A task status as the UI names it."""
    zh, en = STATUS.get(status, (status, status))
    return en if LANG == "en" else zh


# Chinese message -> English. Grouped by the module that says it; shared messages first.
EN = {
    # ---- shared
    "、": ", ",
    "；": "; ",
    "：": ": ",
    "名称": "name",
    "未知": "unknown",
    "默认": "default",
    "（未托管）": " (unmanaged)",
    "退出码 {code}": "exit code {code}",
    "{path} 在你编辑期间已被修改，请载入新版本后再保存": "{path} was changed while you were editing it; load the new version, then save",
    "文件「{path}」不存在": 'File "{path}" does not exist',
    "项目「{project}」不存在": 'Project "{project}" does not exist',
    "项目「{project}」已归档，请先取消归档": 'Project "{project}" is archived; unarchive it first',
    "scope 应为 global，或 project 并给出 project": "scope must be global, or project with a project given",
    "frontmatter 无法解析：{error}": "Cannot parse the frontmatter: {error}",
    "会形成循环依赖：「{up}」已经（间接）依赖这个任务": 'This would make a dependency cycle: "{up}" already depends on this task (maybe indirectly)',
    # ---- model
    "{what}「{name}」不合法：只能包含字母、数字、_ 和 -，且须以字母或数字开头": 'Invalid {what} "{name}": use only letters, digits, _ and -, starting with a letter or digit',
    "不是键值表": "not a mapping",
    "state.json 无法解析：{error}": "Cannot parse state.json: {error}",
    "state.json 的状态「{status}」不认识": 'Unknown status "{status}" in state.json',
    "任务「{task}」不存在": 'Task "{task}" does not exist',
    # ---- scan: reminders on the cards, project.yaml
    "state.json 无法解析": "Cannot parse state.json",
    "名称只能包含字母、数字、_ 和 -：其他任务的 prompt 中写它的路径不会被识别为依赖": "Names may only contain letters, digits, _ and -: other tasks that write its path in their prompt will not get it as a dependency",
    "depends_on 中的「{entry}」看不懂：应写成 类型/任务名 或 {task, kind, label}": 'Cannot read "{entry}" in depends_on: write type/task or {task, kind, label}',
    "depends_on 中的任务「{task}」不存在": 'Task "{task}" in depends_on does not exist',
    "depends_on 中「{task}」的 kind「{kind}」不认识（可用 {kinds}）": 'Unknown kind "{kind}" for "{task}" in depends_on (use {kinds})',
    "parent「{parent}」看不懂：应写成 类型/任务名": 'Cannot read parent "{parent}": write type/task',
    "parent「{parent}」不存在": 'Parent "{parent}" does not exist',
    "progress.md 缺少「状态:」行": 'progress.md has no "状态:" (Status:) line',
    "已完成，但 progress.md 缺少「完成时间:」行": 'Done, but progress.md has no "完成时间:" (Finished:) line',
    "progress.md 已 {hours} 小时未更新": "progress.md has not been updated for {hours} hours",
    "已完成，但工作区中没有 report.md": "Done, but there is no report.md in the workspace",
    "没有 report.md，报告为 {report}": "No report.md; the report is {report}",
    "已完成，但 progress.md 仍有 {n} 个未勾选项": "Done, but progress.md still has {n} unchecked items",
    "子任务「{task}」的进展晚于本任务 progress.md，状态和下一步可能未同步": 'Subtask "{task}" has progress newer than this task\'s progress.md: its status and next step may be out of date',
    "{project}/project.yaml 无法解析：{error}": "Cannot parse {project}/project.yaml: {error}",
    "{project}/project.yaml 的 agent 应为键值表（例如 agent: {model: opus}）": "agent in {project}/project.yaml must be a mapping (e.g. agent: {model: opus})",
    "{project}/project.yaml 的 {key} 应为 true 或 false，而不是「{value}」": '{key} in {project}/project.yaml must be true or false, not "{value}"',
    # ---- ops
    "类型名": "type name",
    "任务名": "task name",
    "项目名": "project name",
    "任务文件 {file} 已不存在": "The task file {file} no longer exists",
    "{file} 的 frontmatter 无法解析：{error}": "Cannot parse the frontmatter of {file}: {error}",
    "{file} 无法解析：{error}": "Cannot parse {file}: {error}",
    "类型「{type}」已存在": 'Type "{type}" already exists',
    "类型「{type}」不存在": 'Type "{type}" does not exist',
    "任务「{task}」已存在": 'Task "{task}" already exists',
    "depends_on 应为列表": "depends_on should be a list",
    "泳道顺序应为类型名的列表": "The lane order should be a list of type names",
    "依赖类型「{kind}」不认识（可用 {kinds}）": 'Unknown dependency kind "{kind}" (use {kinds})',
    "会形成循环依赖：「{up}」已经（间接）依赖这个新任务": 'This would make a dependency cycle: "{up}" already depends on this new task (maybe indirectly)',
    "任务文件在你编辑期间已被修改，请载入新版本后再保存": "The task file was changed while you were editing it; load the new version, then save",
    "任务「{task}」已纳入托管": 'Task "{task}" is already managed',
    "任务当前为「{status}」，无法执行「审核通过」": "The task is {status}: it cannot be approved",
    "任务当前为「{status}」{unmanaged}，无法取消": "The task is {status}{unmanaged}: it cannot be cancelled",
    "改为「{task}」会形成循环依赖：别的任务的 prompt 里已经写了这个路径": 'Changing it to "{task}" would make a dependency cycle: another task\'s prompt already has this path',
    "任务不能依赖自己": "A task cannot depend on itself",
    "项目「{project}」中还有 {n} 个运行中的任务（含后台等待和终端接管），请先停止或等待其结束后再归档": 'Project "{project}" still has {n} running tasks (background waits and terminal takeovers included); stop them or let them finish before archiving',
    "不认识的设置「{key}」": 'Unknown setting "{key}"',
    "并发上限应为正整数": "The concurrency limit must be a positive integer",
    "语言应为 {languages}": "The language must be {languages}",
    "模板「{template}」不存在": 'Template "{template}" does not exist',
    "项目「{project}」已存在": 'Project "{project}" already exists',
    "agent_sys: 模板带来的 {n} 个任务等你审核：在看板上检查后点「开始运行」，或手动开始其中一个任务": (
        'agent_sys: the template brought {n} tasks for you to review: check them on the board, then click "Start '
        'project", or start one of them by hand'
    ),
    "模板预置的任务等你审核；在看板上点「开始运行」或手动开始其中一个任务后，这一行自动删除": (
        'the tasks the template brought wait for your review; this line goes once you click "Start project" on the '
        "board or start one of the tasks by hand"
    ),
    "没有待纳入托管的草稿（子任务或直接依赖它的任务）": "There are no drafts to join (subtasks, or tasks that depend on it directly)",
    "machines 应为机器的列表": "machines should be a list of machines",
    "ledger 应为键值表（primary、higher_is_better、baseline）": "ledger should be a mapping (primary, higher_is_better, baseline)",
    # ---- runs
    "无法在 tmux 中启动运行：{error}": "Cannot start the run in tmux: {error}",
    "agent.replay.writes 应为「工作区内的路径: 内容」的键值表，或者它们的列表": 'agent.replay.writes must be a mapping of "path in the workspace: content", or a list of them',
    "该任务未托管，agent_sys 没有它的会话：请在原终端中回复，或先将任务纳入托管": "This task is unmanaged and agent_sys has no session of it: reply in its own terminal, or manage the task first",
    "预算已用尽（${cost} / ${budget}），请先追加预算": "The budget is used up (${cost} / ${budget}); add budget first",
    "该任务在 {host} 上运行，请在那台机器上的 agent_sys 中操作": "This task runs on {host}; use the agent_sys on that machine",
    "任务当前为「{status}」，无法执行「开始」": "The task is {status}: it cannot be started",
    "消息内容不能为空": "The message is empty",
    "该任务正在终端中接管，请在终端中回复": "This task is taken over in a terminal; reply there",
    "runner 已退出，agent_sys 正在更新任务状态，请稍后再发送": "The runner has exited and agent_sys is updating the task; send again in a moment",
    "任务当前为「{status}」，尚未开始，暂时不能发送消息": "The task is {status} and has not started: messages cannot be sent yet",
    "任务当前为「{status}」，没有正在运行的 runner": "The task is {status}: no runner is running",
    "该任务正在终端中接管，请使用「结束接管」": "This task is taken over in a terminal; use End takeover",
    "该任务已在终端中接管": "This task is already taken over in a terminal",
    "任务当前为「{status}」，无法在终端中接管": "The task is {status}: it cannot be taken over in a terminal",
    "该任务当前未处于终端接管状态": "This task is not taken over in a terminal",
    "追加的金额应为正数": "The amount to add must be positive",
    # ---- runner (exit details, replays)
    "…[共 {n} 字符]": "…[{n} characters in all]",
    "回放文件不存在：{path}": "The replay file does not exist: {path}",
    "（回放模式，没有调用模型）收到你的消息：「{text}」。": '(Replay, no model called) Got your message: "{text}".',
    "claude 已退出：{error}": "claude exited: {error}",
    "claude 进程已退出": "The claude process exited",
    "（stderr：{stderr}）": " (stderr: {stderr})",
    "回放任务：这里是一个 shell，代替交互式 claude（不调用模型）。\n可以编辑 workspaces/{task}/progress.md 模拟 agent 的进展，然后 exit 结束接管。\n": "Replay task: this is a shell standing in for interactive claude (no model is called).\n"
    "Edit workspaces/{task}/progress.md to play the agent's progress, then exit to end the takeover.\n",
    "运行验收检查……": "Running the acceptance checks...",
    # ---- checks
    "checks 的第 {n} 项应为一条命令，或 {name, run, timeout}": "Item {n} of checks should be a command, or {name, run, timeout}",
    "checks 的第 {n} 项的 timeout 应为正数（秒）": "The timeout of item {n} of checks should be a positive number (seconds)",
    # ---- machines
    "机器 {machines} 正被 {tasks} 占用：等它完成或取消，或修改本任务的机器": "Machines {machines} are held by {tasks}: wait for it to finish or be cancelled, or change this task's machines",
    # ---- questions
    "questions.md 中没有「{rows}」这一行，请刷新后重新填写": 'questions.md has no row "{rows}"; refresh and fill it in again',
    "本轮没有 questions.md": "There is no questions.md in this round",
    "{path} 中没有含「{column}」列的表格，请查看原文": '{path} has no table with a "{column}" column; see the file itself',
    "answers 应为「行: 回答」的键值表": 'answers must be a mapping of "row: answer"',
    "任务当前为「{status}」，只有待确认的任务可以提交回答": "The task is {status}: only a task that needs you takes answers",
    "{path} 在你填写期间已被修改，请刷新后重新填写": "{path} was changed while you were filling it in; refresh and fill it in again",
    # ---- files
    "路径「{path}」不在任务工作区内": 'Path "{path}" is not in the task\'s workspace',
    "路径「{path}」经软链接指向工作区以外，不能修改": 'Path "{path}" leads outside the workspace through a symlink: it cannot be changed',
    ".agent_sys/ 由 agent_sys 维护，不能通过界面修改": ".agent_sys/ belongs to agent_sys and cannot be changed from the UI",
    "目录「{path}」不存在": 'Directory "{path}" does not exist',
    "请指定要保存的文件": "Say which file to save",
    "「{path}」是目录": '"{path}" is a directory',
    # ---- knowledge (rules, lessons)
    "经验名称": "lesson name",
    "规则文件名「{name}」不合法：应以 .md 结尾，只能包含字母、数字、_、- 和 .": 'Invalid rule file name "{name}": it must end in .md and contain only letters, digits, _, - and .',
    "规则文件 {name} 已存在": "Rule file {name} already exists",
    "缺少 frontmatter": "no frontmatter",
    "缺少 name": "no name",
    "name「{name}」不合法": 'invalid name "{name}"',
    "缺少 description": "no description",
    "type「{type}」不认识": 'unknown type "{type}"',
    "缺少 type": "no type",
    "请填写描述：经验索引里只有这一行": "Fill in the description: it is the only line the knowledge index has",
    "经验类型「{type}」不认识（可用 {types}）": 'Unknown lesson type "{type}" (use {types})',
    "经验「{name}」不存在": 'Lesson "{name}" does not exist',
    "经验「{name}」在你编辑期间已被修改，请载入新版本后再保存": 'Lesson "{name}" was changed while you were editing it; load the new version, then save',
    "经验「{name}」在你查看之后已被修改，请刷新后再删除": 'Lesson "{name}" was changed after you looked at it; refresh, then delete',
    "经验候选「{id}」不存在": 'Lesson candidate "{id}" does not exist',
    "该经验候选已处理或已不存在": "This lesson candidate was already handled or no longer exists",
    "经验库中已有「{name}」：换一个名字，或确认覆盖": 'The library already has "{name}": choose another name, or confirm replacing it',
    "请填写要写入规则的内容": "Fill in what to write into the rule",
    # ---- sched
    "自动开始失败：{error}": "Auto start failed: {error}",
    # ---- server
    "扫描项目失败：{error}": "Scanning the project failed: {error}",
    "接口不存在：{method} {path}": "No such API: {method} {path}",
    "请求格式不正确（{where}）": "Bad request ({where})",
    "运行记录「{run}」不存在": 'Run "{run}" does not exist',
    "运行记录「{run}」中没有第 {seq} 条事件": 'Run "{run}" has no event {seq}',
    "不认识的操作「{action}」": 'Unknown action "{action}"',
    # ---- launch
    "agent.extra_args 无法解析：{error}": "Cannot read agent.extra_args: {error}",
    # ---- launch.md (shown as a run's launch info; system.md and the message in it are what the agent got)
    "首次运行": "first run",
    "回放（不调用模型）": "replay (no model)",
    "在终端中接管": "terminal takeover",
    "# 运行 {id}": "# Run {id}",
    "- 任务：`{task}`（{title}）": "- Task: `{task}` ({title})",
    "- 类型：{kind}；会话 `{session}`（{flag}）": "- Kind: {kind}; session `{session}` ({flag})",
    "- 开始：{started}；cwd：`{cwd}`": "- Started: {started}; cwd: `{cwd}`",
    "- model：{model}；effort：{effort}；本次预算上限 ${budget}（任务预算减去已花费）": "- model: {model}; effort: {effort}; budget cap of this run ${budget} (the task's budget less what it has cost)",
    "## 注入的 system.md（{size}）": "## Injected system.md ({size})",
    "文件：`{path}`。Claude Code 在会话创建时保存 system prompt，resume 时沿用那一份；规则或经验索引更新后，resume 的消息开头会提示 agent 读新的 system.md。": "File: `{path}`. Claude Code keeps the system prompt of the session's creation and uses it again on resume; "
    "after the rules or knowledge indexes change, the resume message starts by telling the agent to read the new system.md.",
    "| # | 内容 | 来源 | 大小 |": "| # | Content | Source | Size |",
    "project.yaml 的 skills 中这些名字不存在：{names}": "Names in the skills of project.yaml that do not exist: {names}",
    "### {title}（{n} 行）": "### {title} ({n} lines)",
    "<details><summary>system.md 全文（规则和索引，{size}）</summary>": "<details><summary>All of system.md (rules and indexes, {size})</summary>",
    "## 回放": "## Replay",
    "，间隔最长 {gap} 秒": ", gaps of at most {gap} s",
    "回放 `{events}`（速度 ×{speed}{gap}），不调用模型；发出的消息只记录，由回放给出一条说明作为回复。": "Replays `{events}` (speed ×{speed}{gap}) without a model; messages sent are only recorded, and the replay "
    "answers each with a note.",
    "## 本次发送的消息": "## Message sent this time",
    "## 首条消息": "## First message",
    "（无）": "(none)",
    "## 终端中的命令": "## Command in the terminal",
    "在 tmux 窗口 `{window}` 中运行：`{attach}`。退出 claude 后，agent_sys 读回会话累计花费，并按 progress.md 判断任务状态。": "Runs in tmux window `{window}`: `{attach}`. When claude exits, agent_sys reads "
    "the session's total cost back and tells the task's status from progress.md.",
    "回放没有启动 claude；真实运行时，": "The replay did not start claude; in a real run, ",
    "runner 用 Claude Agent SDK 以下面的参数启动 claude，消息从标准输入以 stream-json 发送：": "the runner starts claude through the Claude Agent SDK with these arguments, and sends the messages as "
    "stream-json on standard input:",
    "## 等价命令": "## Equivalent command",
    "## 环境变量": "## Environment",
    "在 agent-sys-v2 serve 清理过的环境（去掉 `CLAUDE*`、`ANTHROPIC*`、`IS_SANDBOX`）上加：": "On top of the environment agent-sys-v2 serve cleaned (without `CLAUDE*`, `ANTHROPIC*`, `IS_SANDBOX`):",
    # ---- what agent_sys tells the agent, in protocol_language: system.md, the first message, messages into its
    # session (launch, runs, questions, server), what failed checks say (checks.feedback)
    "# 这个项目的工作约定\n\n下面是我给这个项目的所有任务定的约定和规则，每次运行都会附上。\n\n": "# This project's working conventions\n\nThe conventions and rules I set for every task of this project, "
    "attached to every run.\n\n",
    "- 项目目录（也是当前目录）：`{project}`\n": "- Project directory (also the current directory): `{project}`\n",
    "- AGENT_SYS_HOME：`{home}`（全局规则在其中的 `rules/`，全局经验库在 `knowledge/`）\n": "- AGENT_SYS_HOME: `{home}` (the global rules are in its `rules/`, the global knowledge library in `knowledge/`)\n",
    "- 本任务：prompt 是 `tasks/{task}.md`，工作区是 `workspaces/{task}/`\n": "- This task: the prompt is `tasks/{task}.md`, the workspace `workspaces/{task}/`\n",
    "- 规则里写的相对路径先按项目目录找；找不到时，按该规则文件所在的目录（下面每段开头注明了来源）理解。\n": "- A relative path in the rules is looked up from the project directory first; if it is not there, read it "
    "from the directory of the rule file it is in (each part below starts with its source).\n",
    "agent_sys 约定": "agent_sys protocol",
    "全局规则": "Global rules",
    "项目规则": "Project rules",
    "项目经验索引": "Project knowledge index",
    "skills 索引": "Skills index",
    "{title}：一行一个 skill（名字、用途、SKILL.md 的路径）；用得上时先读它的 SKILL.md，里面有步骤和脚本。\n\n": "{title}: one skill per line (name, what it is for, the path of its SKILL.md); when one is of use, read its "
    "SKILL.md first: it has the steps and the scripts.\n\n",
    "本任务可以使用的机器：{machines}。任务运行期间和等我回复期间，其他任务不会被安排到这些机器上；不要使用别的机器。": "The machines this task may use: {machines}. While it runs and while it waits for my reply, no other task is "
    "scheduled on them; do not use other machines.",
    "（{gpus} GPU）": " ({gpus} GPUs)",
    "全局经验索引": "Global knowledge index",
    "{title}：一行一条经验，正文在 `{dir}/entries/<name>.md`，用得上时再读。\n\n": "{title}: one lesson per line, the text in `{dir}/entries/<name>.md`; read it when it is of use.\n\n",
    "产出依赖": "output dependency",
    "代码基线": "code baseline",
    "评审覆盖": "review coverage",
    "父任务": "parent task",
    "等它和它的子任务完成": "after it and its subtasks",
    "  它的子任务：": "  Its subtasks:",
    "  - {title}（`{task}`，{status}）：{report}": "  - {title} (`{task}`, {status}): {report}",
    "报告 `workspaces/{task}/{report}`": "report `workspaces/{task}/{report}`",
    "还没有 report.md": "no report.md yet",
    "- {title}（`{task}`，{kind}）：工作区 `workspaces/{task}/`，{report}": "- {title} (`{task}`, {kind}): workspace `workspaces/{task}/`, {report}",
    "；progress 的下一步：{next}": "; Next in its progress: {next}",
    "交接清单：": "Handoff:",
    "……（共 {n} 条，其余见报告）": "... ({n} in all; the rest are in the report)",
    "请完成任务 `tasks/{task}.md`，工作区是 `workspaces/{task}/`。": "Please do the task `tasks/{task}.md`; its workspace is `workspaces/{task}/`.",
    "上游任务（需要时读它们的 report.md 和 progress.md）：": "Upstream tasks (read their report.md and progress.md when you need them):",
    "任务说明（`tasks/{task}.md` 的正文）：": "The task (the body of `tasks/{task}.md`):",
    "（空）": "(empty)",
    "工作区里已有进展，请先读 `workspaces/{task}/progress.md`，接着做。": "The workspace already has progress: read `workspaces/{task}/progress.md` first and carry on from there.",
    "我的回复：{reply}": "My reply: {reply}",
    "（agent_sys 规则有更新，请先读 `{path}`，之后按新规则做。）": "(The agent_sys rules have changed: read `{path}` first, then follow the new rules.)",
    "（上次运行被中断，后台等待的命令已经结束。请先检查在等的条件，需要的话重新起后台等待。）": "(The last run was interrupted and the background wait has ended. Check the condition you were waiting for "
    "first, and start the background wait again if needed.)",
    "（这个任务已从 `tasks/{old}.md` 移到 `tasks/{new}.md`，工作区现在是 `workspaces/{new}/`，旧路径已失效。）": "(This task has moved from `tasks/{old}.md` to `tasks/{new}.md`; its workspace is now `workspaces/{new}/`, and "
    "the old paths no longer hold.)",
    "继续": "Continue",
    "预算已追加 ${usd}，请继续。": "The budget has been raised by ${usd}; please go on.",
    "任务 prompt 已更新（`tasks/{task}.md`），请重新阅读并按新要求调整。": "The task prompt has changed (`tasks/{task}.md`): read it again and adjust to what it now asks.",
    "已在 {path} 中回答（已填写 {filled} / {total} 项），未填写的项请按你的建议处理。": "Answered in {path} ({filled} / {total} filled in); for the rows left empty, follow your suggestion.",
    "本任务的验收检查：你把 progress.md 标为已完成后，系统在项目目录下依次运行它们，全部通过才算完成；没通过时会把输出发给你。": "This task's acceptance checks: once you mark progress.md done, agent_sys runs them one after "
    "the other in the project directory, and the task is done only when all of them pass; when one fails, you get its "
    "output.",
    "- {name}：`{run}`": "- {name}: `{run}`",
    "验收检查没有通过（{failed}/{total} 条失败），任务还不能算完成。请修复后再把 progress.md 标为已完成；如果需要我来决定，把状态改为阻塞并写明原因。": "The acceptance checks did not pass ({failed} of {total} failed), so the task is not done yet. "
    "Fix it and mark progress.md done again; if it needs my decision, set the status to blocked and say why.",
    "超时": "timed out",
    "（没有输出）": "(no output)",
    "（{why}）：": " ({why}):",
    "（还会再检查 {left} 次。）": "({left} more check attempt(s) before I am asked.)",
    # ---- scan: the handoff list of a report
    '报告中没有交接清单（"## 交接清单"一节），下游任务只能自己翻工作区': 'The report has no handoff list (a "## Handoff" section): downstream tasks have to search the workspace',
    "交接清单中的路径不存在：{paths}": "Paths in the handoff list that are not there: {paths}",
    # ---- cli
    "agent_sys：Infera 性能工程的 agent 工作台": "agent_sys: an agent workbench for Infera performance engineering",
    "agent_sys: socket 所在的目录 {path} 其他用户也能进入，请换一个只有你能访问的目录（例如 {example}/）": "agent_sys: other users can enter {path}, the socket's directory; use one only you can enter (e.g. {example}/)",
    "agent_sys: {sock} 上已有 agent_sys server 在运行": "agent_sys: an agent_sys server is already running on {sock}",
    "agent_sys: {home} 已有 agent_sys server 在运行（{holder}）": "agent_sys: {home} already has an agent_sys server running ({holder})",
    "agent_sys server：AGENT_SYS_HOME={home}，socket {sock}": "agent_sys server: AGENT_SYS_HOME={home}, socket {sock}",
    "agent_sys server 已在运行：{sock}": "agent_sys server is already running: {sock}",
    "agent_sys server 已退出，按回车关闭窗口": "agent_sys server has exited; press Enter to close the window",
    "agent_sys server 已在 tmux -L {tmux} 的 {window} 窗口启动：{sock}": "agent_sys server started in window {window} of tmux -L {tmux}: {sock}",
    "agent_sys server 没有启动起来（tmux -L {tmux} 的 {window} 窗口），窗口里显示：": "agent_sys server did not start (window {window} of tmux -L {tmux}); the window shows:",
    "查看日志：{attach}": "Its log: {attach}",
    "在你的电脑上运行：ssh -N -L 8765:{sock} {host}\n然后用浏览器打开 http://localhost:8765": "On your computer run: ssh -N -L 8765:{sock} {host}\nthen open http://localhost:8765 in a browser",
    "AGENT_SYS_HOME：默认取环境变量 AGENT_SYS_HOME，没有则 {default}（当前 %(default)s）": "AGENT_SYS_HOME: default $AGENT_SYS_HOME, else {default} (now %(default)s)",
    "用模板新建项目": "make a new project from a template",
    "项目名，即 projects/ 下的目录名": "the project's name: its directory under projects/",
    "模板：{templates}（默认 blank）": "template: {templates} (default blank)",
    "项目标题（默认用项目名）": "the project's title (default: its name)",
    "启动 agent_sys server（默认放进 tmux）": "start the agent_sys server (in tmux by default)",
    "Unix socket（默认 %(default)s）": "Unix socket (default %(default)s)",
    "在当前终端运行，不进 tmux": "run in this terminal, not in tmux",
    "agent_sys: 已在 {home} 创建 {made}": "agent_sys: made {made} in {home}",
    # ---- exec
    "任务 prompt 有错误，改正后才能运行：{error}": "The task prompt has an error; fix it before the task can run: {error}",
    "agent_sys: {sock} 上运行的是另一个 AGENT_SYS_HOME 的 server，请用 --sock 换一个 socket": "agent_sys: the server on {sock} is another AGENT_SYS_HOME's; use another socket with --sock",
    # ---- api: server, ops, scan
    "阻塞类依赖（input / code / children）形成了循环，任务会一直等待": "The blocking dependencies (input / code / children) form a cycle: the task will wait forever",
    "progress.md 的「完成时间:」{value} 不是 YYYY-MM-DD HH:MM 格式的时间": 'The "完成时间:" (Finished:) of progress.md, {value}, is not a time written as YYYY-MM-DD HH:MM',
    "任务「{task}」正在运行，请先停止再{action}": 'Task "{task}" is running; stop it before you {action}',
    "任务「{task}」刚纳入托管，可能仍在终端中运行：请先确认终端会话已结束并回复它，再{action}": 'Task "{task}" was just made managed and may still run in a terminal: make sure that session has ended and '
    "reply to the task, then {action}",
    "改名或改类型": "rename it or change its type",
    "删除": "delete it",
    "不接受 Host「{host}」：请通过 localhost 访问，或将它加入环境变量 AGENT_SYS_HOSTS": 'Host "{host}" is not accepted: open agent_sys as localhost, or add the name to the environment variable AGENT_SYS_HOSTS',
    "拒绝来自其他网站的请求": "A request from another site is refused",
}
