"""Write operations on projects and tasks. The server calls these under its write lock; each one checks what it
needs on a fresh scan and writes files (the server scans again afterwards: what they return is not a new scan).
Refusals raise AgentSysError with a message for the user."""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

from ruamel.yaml.comments import CommentedMap, CommentedSeq

from . import claude_config, preflight
from .checks import parse as parse_checks
from .i18n import LANGUAGES, _, status_word
from .model import (
    EDGE_KINDS,
    GLOBAL_DEFAULTS,
    NAME,
    PROJECT_DEFAULTS,
    PROJECTS,
    AgentSysError,
    check_name,
    check_scope,
    cycle_tasks,
    dump_yaml,
    iso,
    load_config,
    load_yaml,
    mtime,
    new_state,
    now_iso,
    parse_dep,
    plain,
    positive,
    project_agent,
    project_path,
    read_task_file,
    read_text,
    split_frontmatter,
    write_state,
    write_task_file,
    write_text,
)
from .scan import (
    TASK_REF,
    check_config,
    depends_on,
    load_project,
    project_dir,
    project_ids,
    read_config,
)

TEMPLATES = Path(__file__).resolve().parent / "templates"
DEFAULT_SKELETON = '## 任务描述\n\n- \n\n## 交付\n\n- `report.md`：以"## 结论"开头。\n'
AGENT_SYS_YAML = f"""# agent_sys's global settings. A key left out takes its default; saving on the settings page keeps these comments.
language: {GLOBAL_DEFAULTS["language"]}                     # en or zh: the language of the web UI and of agent_sys's messages
protocol_language: {GLOBAL_DEFAULTS["protocol_language"]}            # zh or en: the agent_sys protocol and what agent_sys tells the agent (it works in the prompt's language)
max_running: {GLOBAL_DEFAULTS["max_running"]}                   # tasks running at once across all projects (background waits and terminal takeovers count)
claude_config: {GLOBAL_DEFAULTS["claude_config"]}          # isolated: the agent's own Claude Code directory ($AGENT_SYS_HOME/claude), without your MCP servers, plugins, CLAUDE.md and skills; inherit: your ~/.claude as it is
# cli_path: /path/to/claude      # default: the claude bundled with claude-agent-sdk in agent_sys's venv
"""


def read_frontmatter(path):
    """(frontmatter, body) of a task file; refused with 422 if the frontmatter cannot be parsed."""
    try:
        return read_task_file(path)
    except ValueError as e:
        raise AgentSysError(
            _("{file} 的 frontmatter 无法解析：{error}", file=path.name, error=e), 422
        ) from None


def check_checks(value):
    """Acceptance checks as frontmatter writes them; refused unless checks.parse reads them."""
    try:
        parse_checks(value)
    except ValueError as e:
        raise AgentSysError(str(e)) from None


def update_frontmatter(path, change):
    """Rewrite a task file's frontmatter with change(fm); the body and unknown fields are kept."""
    if not path.is_file():
        raise AgentSysError(_("任务文件 {file} 已不存在", file=path.name), 404)
    fm, body = read_frontmatter(path)
    change(fm)
    write_task_file(path, fm, body)


def edit_yaml(path, change):
    """Rewrite a YAML file (project.yaml, agent_sys.yaml) with change(data); comments and key order are kept.
    A missing file starts empty."""
    try:
        data = load_yaml(read_text(path))
    except ValueError as e:
        raise AgentSysError(_("{file} 无法解析：{error}", file=path.name, error=e), 422) from None
    change(data)
    write_text(path, dump_yaml(data))


def flow(mapping):
    """A mapping written on one line, like `agent: {budget_usd: 300}`."""
    m = CommentedMap(mapping)
    m.fa.set_flow_style()
    return m


def flow_list(items):
    """A list written on one line, like `lanes: [develop, bug_fix]`."""
    seq = CommentedSeq(items)
    seq.fa.set_flow_style()
    return seq


def dep_entry(up, kind="input", label=""):
    """depends_on entry: the short form for a plain input, else {task, kind, label} on one line."""
    if kind == "input" and not label:
        return up
    return flow({"task": up, "kind": kind, **({"label": label} if label else {})})


# project.yaml keys the settings page may write: the defaults, and what a project says about itself.
PROJECT_SETTINGS = PROJECT_DEFAULTS | {"title": "", "goal": "", "machines": [], "ledger": {}}


def check_not_archived(project):
    if project.archived:
        raise AgentSysError(_("项目「{project}」已归档，请先取消归档", project=project.title), 409)


def check_idle(t, action):
    """Rename and delete move the workspace: refused while an agent may still work in it, under agent_sys (running)
    or in a terminal (progress.md 进行中, or a task just joined whose terminal session may not have ended)."""
    if t.status == "running":
        raise AgentSysError(
            _("任务「{task}」正在运行，请先停止再{action}", task=t.title, action=action), 409
        )
    if t.detail.get("kind") == "joined":
        raise AgentSysError(
            _(
                "任务「{task}」刚纳入托管，可能仍在终端中运行：请先确认终端会话已结束并回复它，再{action}",
                task=t.title,
                action=action,
            ),
            409,
        )


def task_paths(project, type_, name):
    """(prompt file, workspace) of a new task type_/name; refused unless the name is valid, the type exists and
    neither the file nor the workspace does."""
    check_name(name, _("任务名"))
    if type_ not in {ty.id for ty in project.types}:
        raise AgentSysError(_("类型「{type}」不存在", type=type_), 404)
    path, ws = (
        project.path / "tasks" / type_ / f"{name}.md",
        project.path / "workspaces" / type_ / name,
    )
    if path.exists() or ws.exists():
        raise AgentSysError(_("任务「{task}」已存在", task=f"{type_}/{name}"), 409)
    return path, ws


def new_cycle(pdir, tid, before):
    """After a write that may add edges into tid: the upstream task that now closes a cycle through tid, if any
    (`before` = the tasks on cycles before the write)."""
    project = load_project(pdir)
    cyc = cycle_tasks(project.edges)
    if tid not in cyc or tid in before:
        return None
    return next(project.tasks[e.up] for e in project.upstream(tid) if e.blocking and e.up in cyc)


# ---------------------------------------------------------------- types and tasks


def new_type(pdir, type_id, review=False):
    """New type = tasks/<type>/; also task_types/<type>.md with a generic skeleton unless one exists."""
    check_name(type_id, _("类型名"))
    pdir = Path(pdir)
    if (pdir / "tasks" / type_id).exists():
        raise AgentSysError(_("类型「{type}」已存在", type=type_id), 409)
    skeleton = pdir / "task_types" / f"{type_id}.md"
    if not skeleton.exists():
        write_task_file(skeleton, CommentedMap(review=True) if review else {}, DEFAULT_SKELETON)
    (pdir / "tasks" / type_id).mkdir(parents=True)


def set_lanes(pdir, order):
    """The order of the board's lanes: project.yaml lanes, the type ids from left to right. Types left out follow
    by name; ids of types that are gone are dropped."""
    project = load_project(pdir)
    if not isinstance(order, list) or not all(isinstance(x, str) for x in order):
        raise AgentSysError(_("泳道顺序应为类型名的列表"))
    types = [ty.id for ty in project.types]
    lanes = list(dict.fromkeys(x for x in order if x in types))

    def change(config):
        if lanes:
            config["lanes"] = flow_list(lanes)
        else:
            config.pop("lanes", None)

    edit_yaml(project.path / "project.yaml", change)


def new_task(
    pdir,
    type_,
    name,
    prompt,
    *,
    title="",
    depends_on=(),
    review=False,
    agent=None,
    queue=True,
    machines=None,
    checks=None,
):
    """创建并排队 (queue=True): the task file and a state.json in 待开始, with a new session_id.
    保存为草稿 (queue=False): the task file only. depends_on: ids or {task, kind, label}. agent: only what differs
    from the project's defaults, so that later changes to project.yaml still reach the task. machines: names from
    project.yaml. checks: the acceptance checks as frontmatter writes them; None = the type skeleton's."""
    project = load_project(pdir)
    check_not_archived(project)
    path, ws = task_paths(project, type_, name)
    tid = f"{type_}/{name}"

    fm = CommentedMap()
    if title:
        fm["title"] = title
    if deps := dep_entries(project, tid, list(depends_on), {}):
        fm["depends_on"] = deps
    if review:
        fm["review"] = True
    if machines:
        fm["machines"] = flow_list([str(m) for m in machines])
    skeleton = next(ty for ty in project.types if ty.id == type_)
    if checks := skeleton.checks if checks is None else checks:
        check_checks(checks)
        fm["checks"] = checks
    if agent:
        fm["agent"] = flow(agent)
    write_task_file(path, fm, prompt)

    # Other prompts may already name this task's path, so check cycles on the scan that includes it.
    if up := new_cycle(pdir, tid, cycle_tasks(project.edges)):
        path.unlink()
        raise AgentSysError(
            _("会形成循环依赖：「{up}」已经（间接）依赖这个新任务", up=up.title), 409
        )
    if queue:
        write_state(ws, new_state("queued"))


def dep_entries(project, tid, deps, fm):
    """A depends_on list from the UI ([id or {task, kind, label}]) as it is written: short form for a plain input.
    An entry already in the file may name a task that is gone (it stays a record reminder); a new one may not."""
    if not isinstance(deps, list):
        raise AgentSysError(_("depends_on 应为列表"))
    known = {parse_dep(e)[0] for e in depends_on(fm)}
    out, seen = [], set()
    for d in deps:
        up, kind, label = parse_dep(d)
        if up is None:
            raise AgentSysError(
                _(
                    "depends_on 中的「{entry}」看不懂：应写成 类型/任务名 或 {task, kind, label}",
                    entry=d,
                )
            )
        if up == tid:
            raise AgentSysError(_("任务不能依赖自己"))
        if up not in known:
            project.task(up)
        if kind not in EDGE_KINDS:
            raise AgentSysError(
                _("依赖类型「{kind}」不认识（可用 {kinds}）", kind=kind, kinds="/".join(EDGE_KINDS))
            )
        if up not in seen:
            seen.add(up)
            out.append(dep_entry(up, kind, label))
    return out


def update_task(pdir, tid, frontmatter=None, prompt=None, base_mtime=""):
    """修改任务配置: set the frontmatter keys given ('' / None / {} / [] removes the key; a mapping is written on one
    line) and / or replace the prompt. base_mtime is the task file's mtime when you started editing: if the file
    changed since (an agent or an editor wrote it), nothing is written. A prompt that now names a task which
    (indirectly) depends on this one is refused, as in new_task."""
    project = load_project(pdir)
    t = project.task(tid)
    if base_mtime and base_mtime != iso(mtime(t.path)):
        raise AgentSysError(_("任务文件在你编辑期间已被修改，请载入新版本后再保存"), 409)
    old = read_text(t.path)
    fm, body = read_frontmatter(t.path)
    for key, value in (frontmatter or {}).items():
        if key == "depends_on" and value:
            value = dep_entries(project, tid, value, fm)
        if key == "checks" and value:
            check_checks(value)
        if value is None or value == "" or value == {} or value == []:
            fm.pop(key, None)
        else:
            fm[key] = flow(value) if isinstance(value, dict) else value
    write_task_file(t.path, fm, body if prompt is None else prompt)
    if up := new_cycle(pdir, tid, cycle_tasks(project.edges)):
        write_text(t.path, old)
        raise AgentSysError(_("会形成循环依赖：「{up}」已经（间接）依赖这个任务", up=up.title), 409)


def join(pdir, tid):
    """纳入托管: 草稿 -> 待开始; progress 已完成 -> 已完成 (review: true -> 待审核);
    any other progress -> 待确认 (detail joined: confirm the terminal session has ended first). A draft is refused while
    the check before a start finds an error in it.
    A draft without review or checks in its frontmatter gets its type's defaults, as a task made in the UI does:
    a draft an agent wrote (a plan's experiments) is held to the same acceptance checks."""
    project = load_project(pdir)
    check_not_archived(project)
    t = project.task(tid)
    if t.status == "draft":  # it is queued, and will start by itself: as 全部纳入托管
        preflight.refuse(project, preflight.preflight(project, [tid]), _("纳入托管"))
    _join(project, t)


def _join(project, t):
    if t.managed:
        raise AgentSysError(_("任务「{task}」已纳入托管", task=t.title), 409)
    p = t.progress
    if p is None:
        ty = next((ty for ty in project.types if ty.id == t.type), None)
        defaults = {"review": ty.review, "checks": ty.checks} if ty else {}
        if defaults := {k: v for k, v in defaults.items() if v and k not in t.fm}:
            try:
                update_frontmatter(t.path, lambda fm: fm.update(defaults))
            # a frontmatter that cannot be parsed: joined as it is (it shows why)
            except AgentSysError:
                pass
        state = new_state("queued", created=t.created)
    elif p.done:
        status = "in_review" if t.status == "in_review" else "done"
        state = new_state(
            status,
            created=t.created,
            started=t.started,
            finished=t.finished,
            seen_at=now_iso() if status == "done" else "",
        )
    else:
        state = new_state("needs_you", {"kind": "joined"}, created=t.created, started=t.started)
    write_state(t.workspace, state)


def drafts_of(project, tid):
    """The drafts tid wrote: those whose parent it is, and those that depend on it directly (a plan's close task,
    which waits for the plan's subtasks without being one). Review edges do not count."""
    direct = {e.down for e in project.downstream(tid) if e.kind != "review"}
    return [
        t.id
        for t in project.tasks.values()
        if t.status == "draft" and (t.parent == tid or t.id in direct)
    ]


def join_children(pdir, tid):
    """全部纳入托管: join every draft tid wrote (drafts_of: the plan an agent wrote as drafts, approved at once).
    Returns the ids joined. None is joined while the check before a start finds an error in them."""
    project = load_project(pdir)
    check_not_archived(project)
    project.task(tid)
    drafts = drafts_of(project, tid)
    if not drafts:
        raise AgentSysError(_("没有待纳入托管的草稿（子任务或直接依赖它的任务）"), 409)
    preflight.refuse(project, preflight.preflight(project, drafts), _("纳入托管"))
    for child in drafts:
        _join(project, project.tasks[child])
    return drafts


def approve(pdir, tid):
    """审核通过: 待审核 -> 已完成, which lets downstream tasks start. An unmanaged task (progress 已完成,
    review: true) stays unmanaged: approved_at goes into its frontmatter, and if progress.md later reports a
    newer finish (you went on in the terminal) it is back in 待审核."""
    project = load_project(pdir)
    t = project.task(tid)
    if t.status != "in_review":
        raise AgentSysError(
            _("任务当前为「{status}」，无法执行「审核通过」", status=status_word(t.status)), 409
        )
    now = now_iso()
    if t.managed:
        write_state(
            t.workspace, {**t.state, "status": "done", "detail": {}, "since": now, "seen_at": now}
        )
    else:

        def stamp(fm):
            fm["approved_at"] = max(now, t.finished)

        update_frontmatter(t.path, stamp)


def mark_read(pdir, *tids):
    """标为已查看: opening a finished task clears its unread mark (全部标为已查看: several at once, one scan).
    Returns the ids that were unread."""
    project = load_project(pdir)
    marked = [t for t in map(project.task, tids) if t.unread]
    for t in marked:
        write_state(t.workspace, {**t.state, "seen_at": now_iso()})
    return [t.id for t in marked]


def retarget_paths(text, pid, old, new):
    """Task paths and @ mentions in a prompt (tasks/<old>, workspaces/<old>, also under projects/<pid>/, @<old>) pointed
    at new."""

    def sub(m):
        if (m[1] and m[1] != pid) or f"{m[2]}/{m[3]}" != old:
            return m[0]
        return m[0][: m.start(2) - m.start()] + new

    return TASK_REF.sub(sub, text)


def _rewrite_refs(project, old, new, moved, backup):
    """Point every task at new instead of old: depends_on / parent in the frontmatter and the task paths in the
    prompt; and project.yaml's ledger.baseline. moved: where the renamed task's own file is now. backup gets
    {path: text before} of each file written. A file whose frontmatter cannot be parsed is left as it is (it shows a
    record reminder already)."""
    for t in project.tasks.values():
        path = moved if t.id == old else t.path
        text = read_text(path)
        try:
            fm, body = split_frontmatter(text)
        except ValueError:
            continue
        new_body = retarget_paths(body, project.id, old, new)
        deps = depends_on(fm)
        hits = [i for i, e in enumerate(deps) if parse_dep(e)[0] == old]
        if not hits and fm.get("parent") != old and new_body == body:
            continue
        for i in hits:  # in place: the comments of the other entries stay
            if isinstance(deps[i], dict):
                deps[i]["task"] = new
            else:
                deps[i] = new
        if hits and not isinstance(fm["depends_on"], list):  # one entry without the list
            fm["depends_on"] = deps[0]
        if fm.get("parent") == old:
            fm["parent"] = new
        backup[path] = text
        write_task_file(path, fm, new_body)

    # the ledger's baseline: the task, or a file of it (<id>/<file>, workspaces/<id>/<file>)
    ledger = project.config.get("ledger")
    base = ledger.get("baseline") if isinstance(ledger, dict) else None
    if m := re.fullmatch(rf"(workspaces/)?{re.escape(old)}(/.*)?", str(base or "")):
        path = project.path / "project.yaml"
        backup[path] = read_text(path)

        def change(config):
            config["ledger"]["baseline"] = f"{m[1] or ''}{new}{m[2] or ''}"

        edit_yaml(path, change)


def move(src, dst, moved):
    """shutil.move, noted in moved as (src, dst) so that undo() can put it back."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(src, dst)
    moved.append((src, dst))


def undo(moved, backup=None):
    """Put back what move() moved and the files backup holds {path: text before}."""
    for f, text in (backup or {}).items():
        write_text(f, text)
    for src, dst in reversed(moved):
        shutil.move(dst, src)


def rename(pdir, tid, new_type, new_name):
    """改名 / 改类型 (also 拖到另一条泳道): moves the prompt and the workspace (with .agent_sys/: state, runs, session) and
    points the other tasks at the new id, in their frontmatter and in the task paths of their prompts, so that every
    dependency stays (check_idle: nobody may be working in the workspace). The session goes on: its cwd is the
    project directory; the next message to it says where the task went. Returns the new id."""
    project = load_project(pdir)
    t = project.task(tid)
    check_idle(t, _("改名或改类型"))
    new_id = f"{new_type}/{new_name}"
    if new_id == tid:
        return tid
    path, ws = task_paths(project, new_type, new_name)
    # files rewritten in place: their text before; (from, to) of what was moved
    backup, moved = {}, []
    try:
        if t.workspace.exists():  # first: a workspace that cannot move leaves nothing half done
            move(t.workspace, ws, moved)
        move(t.path, path, moved)
        _rewrite_refs(project, tid, new_id, path, backup)
        # Other prompts may name the new path already, so check cycles on the scan that includes it (a cycle the
        # task was on before, under its old id, is not a new one).
        before = {new_id if x == tid else x for x in cycle_tasks(project.edges)}
        if new_cycle(pdir, new_id, before):
            raise AgentSysError(
                _(
                    "改为「{task}」会形成循环依赖：别的任务的 prompt 里已经写了这个路径",
                    task=new_id,
                ),
                409,
            )
    except BaseException:
        undo(moved, backup)
        raise
    if t.managed and t.state.get("runs"):
        # the session knows the old paths: the next message to it says where the task is now (runs.start_run)
        state = {**t.state, "moved_from": t.state.get("moved_from") or tid}
        if state["moved_from"] == new_id:
            del state["moved_from"]
        write_state(ws, state)
    return new_id


def delete(pdir, tid):
    """删除: the workspace and the prompt move to .trash/<time>-<type>-<name>/{workspaces,tasks}/… (check_idle:
    nobody may be working in the workspace); moving them back restores the task as it was. Other tasks are not
    touched: a depends_on / parent naming the deleted task makes no edge and shows as a record reminder. Returns the
    trash directory."""
    project = load_project(pdir)
    t = project.task(tid)
    check_idle(t, _("删除"))
    trash = project.path / ".trash" / f"{datetime.now():%Y%m%d-%H%M%S}-{t.type}-{t.name}"
    while trash.exists():
        trash = trash.with_name(trash.name + "_")
    moved = []
    try:
        if t.workspace.exists():  # first, as in rename
            move(t.workspace, trash / "workspaces" / t.type / t.name, moved)
        move(t.path, trash / "tasks" / t.type / t.path.name, moved)
    except BaseException:
        undo(moved)
        raise
    return trash


# ---------------------------------------------------------------- projects and settings


def set_archived(pdir, archived):
    """归档 / 取消归档: archived_at in project.yaml. Archiving is refused while managed tasks are running (including
    waiting in the background and terminal takeovers): what they ask afterwards would not show on the home page."""
    project = load_project(pdir)
    if archived == project.archived:
        return
    if archived and (n := project.running()):
        raise AgentSysError(
            _(
                "项目「{project}」中还有 {n} 个运行中的任务（含后台等待和终端接管），请先停止或等待其结束后再归档",
                project=project.title,
                n=n,
            ),
            409,
        )

    def change(config):
        if archived:
            config["archived_at"] = now_iso()
        else:
            config.pop("archived_at", None)

    edit_yaml(project.path / "project.yaml", change)


def start_problems(project):
    """What the check before a start finds in the tasks 开始运行 lets run: the managed ones not finished yet (a draft
    is checked when it is joined)."""
    return preflight.preflight(
        project,
        [t.id for t in project.tasks.values() if t.managed and t.status not in preflight.FINISHED],
    )


def start_project(pdir, checked=False):
    """开始运行: the template's tasks are reviewed, template_review leaves project.yaml and the project runs as any
    other (autorun on: the first task starts at once, the others as their upstream tasks are done). Starting one of
    its tasks by hand does the same (checked: it ran the check already). Refused while the check before a start finds
    an error (start_problems)."""
    project = load_project(pdir)
    check_not_archived(project)
    if project.template_review:
        if not checked:
            preflight.refuse(project, start_problems(project), _("开始运行"))
        edit_yaml(project.path / "project.yaml", lambda config: config.pop("template_review", None))


def settings(home, pid=None):
    """agent_sys.yaml and project.yaml with defaults filled in. Without a project, the
    defaults of every project are listed too (the global settings page shows them)."""

    def project_values(pdir):
        config = plain(read_config(pdir))
        return {
            **PROJECT_DEFAULTS,
            **config,
            "title": config.get("title") or pdir.name,
            "agent": project_agent(config),
        }

    out = {"global": {"path": "agent_sys.yaml", "values": load_config(home)}, "project": None}
    if pid:
        out["project"] = {
            "path": f"{project_path(pid)}/project.yaml",
            "values": project_values(project_dir(home, pid)),
        }
    else:
        out["projects"] = []
        for p in project_ids(home):
            try:
                values = project_values(project_dir(home, p))
            except AgentSysError:
                continue
            out["projects"].append(
                {
                    "id": p,
                    "path": f"{project_path(p)}/project.yaml",
                    "values": values,
                    "archived": bool(values.get("archived_at")),
                }
            )
    return out


def save_settings(home, scope, values, pid=None):
    """Write values into agent_sys.yaml (scope global) or a project's project.yaml (scope project), keeping the
    comments; a mapping value (agent) is merged key by key."""
    pid = check_scope(scope, pid)
    known = GLOBAL_DEFAULTS if scope == "global" else PROJECT_SETTINGS
    for key, value in values.items():
        if key not in known:
            raise AgentSysError(_("不认识的设置「{key}」", key=key))
        if key == "max_running" and positive(value, None) is None:
            raise AgentSysError(_("并发上限应为正整数"))
        if key in ("language", "protocol_language") and value not in LANGUAGES:
            raise AgentSysError(_("语言应为 {languages}", languages=" / ".join(LANGUAGES)))
        if key == "claude_config" and value not in claude_config.MODES:
            raise AgentSysError(
                _("claude_config 应为 {modes}", modes=" / ".join(claude_config.MODES))
            )
        if key == "machines" and not isinstance(value, list):
            raise AgentSysError(_("machines 应为机器的列表"))
        if key == "ledger" and not isinstance(value, dict):
            raise AgentSysError(_("ledger 应为键值表（primary、higher_is_better、baseline）"))
    path = project_dir(home, pid) / "project.yaml" if pid else Path(home) / "agent_sys.yaml"
    if pid and not path.is_file():
        raise AgentSysError(_("项目「{project}」不存在", project=pid), 404)

    def change(config):
        for key, value in values.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)  # in place: the comments inside agent: stay
            else:
                config[key] = value
        if pid:  # checked as load_project will read it, before it is written
            check_config(config, pid)

    edit_yaml(path, change)
    return settings(home, pid)[scope]


def init_home(home):
    """Make what a AGENT_SYS_HOME lacks: projects/, rules/, knowledge/entries/ (the server writes its
    INDEX.md) and agent_sys.yaml with the defaults written out. Nothing that exists is touched. Returns what was made."""
    home, made = Path(home), []
    for sub in (PROJECTS, "rules", "knowledge/entries"):
        if not (home / sub).is_dir():
            (home / sub).mkdir(parents=True)
            made.append(f"{sub}/")
    if not (home / "agent_sys.yaml").exists():
        write_text(home / "agent_sys.yaml", AGENT_SYS_YAML)
        made.append("agent_sys.yaml")
    return made


def template_tasks(src):
    """The tasks a template brings, tasks/<type>/<name>.md, as [type/name]."""
    return sorted(
        f"{f.parent.name}/{f.stem}"
        for f in (Path(src) / "tasks").glob("*/*.md")
        if not f.name.startswith(".") and NAME.fullmatch(f.parent.name) and NAME.fullmatch(f.stem)
    )


def list_templates(templates_dir=TEMPLATES):
    """[{id, title, description, types: [{id, review, skeleton}], tasks: [type/name]}] for templates/<id>/
    (templates/_common is shared, not a template)."""
    out = []
    for d in sorted(Path(templates_dir).iterdir()):
        if d.is_dir() and not d.name.startswith(("_", ".")):
            config = load_yaml(read_text(d / "project.yaml"))
            types = []
            for f in sorted((d / "task_types").glob("*.md")):
                fm, skeleton = read_task_file(f)
                types.append(
                    {"id": f.stem, "review": fm.get("review") is True, "skeleton": skeleton}
                )
            out.append(
                {
                    "id": d.name,
                    "title": config.get("title") or d.name,
                    "description": str(config.get("description") or ""),
                    "types": types,
                    "tasks": template_tasks(d),
                }
            )
    return out


def new_project(home, name, template="blank", title=None, templates_dir=TEMPLATES):
    """New project = copy the template's rules/, knowledge/, task_types/, skills/ + project.yaml with the
    title filled in + tasks/<type>/ for every type. The tasks the template brings (tasks/<type>/<name>.md, e.g. the
    first round of a project) are copied too and queued, each with a state.json of its own: their depends_on keeps
    all but the first waiting. They do not start by themselves yet: project.yaml gets template_review: true, so you
    review them (and the rules) first and start the project by hand (start_project). The agent_sys protocol is not
    copied: every run injects it from the code, so that it follows agent_sys's version.
    Built in a hidden directory and renamed into place, so a failure leaves nothing half made."""
    check_name(name, _("项目名"))
    templates_dir = Path(templates_dir)
    src = templates_dir / template
    if not NAME.fullmatch(template) or not src.is_dir():  # not _common, a path or ..
        raise AgentSysError(_("模板「{template}」不存在", template=template), 404)
    dst = project_dir(home, name)
    if dst.exists():
        raise AgentSysError(_("项目「{project}」已存在", project=name), 409)
    tmp = dst.with_name(f".{name}.new")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for sub in ("rules", "knowledge", "task_types", "skills"):
        if (src / sub).is_dir():
            shutil.copytree(src / sub, tmp / sub)
    (tmp / "rules").mkdir(exist_ok=True)
    (tmp / "knowledge" / "entries").mkdir(parents=True, exist_ok=True)
    if not (tmp / "knowledge" / "INDEX.md").exists():
        write_text(tmp / "knowledge" / "INDEX.md", "")
    config = load_yaml(read_text(src / "project.yaml"))
    config.pop("description", None)  # describes the template, not the project
    if "title" in config:
        config["title"] = title or name
    else:
        config.insert(0, "title", title or name)
    if "template" not in config:
        config.insert(1, "template", template)
    if template_tasks(src):
        config.insert(
            2,
            "template_review",
            True,
            comment=_(
                "模板预置的任务等你审核；在看板上点「开始运行」或手动开始其中一个任务后，这一行自动删除"
            ),
        )
    write_text(tmp / "project.yaml", dump_yaml(config))
    for skeleton in (tmp / "task_types").glob("*.md"):
        (tmp / "tasks" / skeleton.stem).mkdir(parents=True)
    (tmp / "workspaces").mkdir()
    made = now_iso()  # made at once: the board orders them by their dependencies alone
    for tid in template_tasks(src):
        (tmp / "tasks" / tid).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / "tasks" / f"{tid}.md", tmp / "tasks" / f"{tid}.md")
        write_state(tmp / "workspaces" / tid, new_state("queued", created=made))
    tmp.rename(dst)
    return dst
