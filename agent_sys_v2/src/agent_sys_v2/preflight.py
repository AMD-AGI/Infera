"""The check before a start: every problem a project's files show before anything runs, listed at once.

  preflight(project, tasks=None) -> [{task, level: error | warn, message, fix}]

What a run would only find out when it gets there: a dependency or a parent that does not exist, a cycle, a frontmatter
or checks that cannot be read, a machine project.yaml does not declare, a skill that does not exist, a check that runs a
module, a perfcheck kind, a ledger command or a script that is not there. An error stops the start: 开始运行
(ops.start_project, and the first 开始 of a template's task, which ends the review too), 全部纳入托管 (ops.join_children)
and `agent-sys-v2 check` all refuse with the whole list (refuse). A warning is shown and lets it start.

task is '' for the project's own problems (project.yaml). A task whose frontmatter or checks cannot be read gets that
error alone: what follows from it would only repeat it. Not checked: what intake fills in (rules/project.md's
placeholders, project.yaml with no machines yet): a template's tasks wait for review with them on purpose.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

from . import checks, ledger, machines, perfcheck, skills
from .i18n import _
from .model import AgentSysError, cycle_tasks
from .scan import closest, graph_problems, lane_order

FINISHED = ("done", "in_review", "cancelled")
PACKAGE = checks.PACKAGE
SUBCOMMANDS = {  # the modules whose first argument is a kind the check can read
    "perfcheck": lambda: tuple(perfcheck.CHECKS),
    "ledger": lambda: ledger.parser().commands,
}
OPERATOR = re.compile(
    r"[();<>|&]+"
)  # what shlex's punctuation_chars splits off: ; && || | ( ) and redirections
# words that may lead a simple command without being it
KEYWORDS = {
    "if",
    "elif",
    "while",
    "until",
    "then",
    "else",
    "do",
    "done",
    "fi",
    "!",
    "{",
    "}",
    "time",
}
INTERPRETERS = {"python", "python3", "bash", "sh", "$AGENT_SYS_PYTHON", "${AGENT_SYS_PYTHON}"}
ROOTS = {  # the variables a check's path may start with, and where they point before a run
    "AGENT_SYS_SKILLS": lambda project: skills.SHIPPED,
    "AGENT_SYS_PROJECT": lambda project: project.path,
}
DYNAMIC = re.compile(r"[$`*?\[~]")  # what a path needs the shell for: not checked


def preflight(project, tasks=None):
    """The problems of project, for tasks (ids; None: every task not done, in review or cancelled, drafts too),
    errors first."""
    ids = (
        [t.id for t in project.tasks.values() if t.status not in FINISHED]
        if tasks is None
        else [tid for tid in tasks if tid in project.tasks]
    )
    out = project_problems(project, ids)
    cyclic, cycles_told = cycle_tasks(project.edges), set()
    for tid in ids:
        out += task_problems(project, project.tasks[tid], cyclic, cycles_told)
    return sorted(out, key=lambda p: p["level"] != "error")


def problem(task, level, message, fix=""):
    return {"task": task, "level": level, "message": message, "fix": fix}


def errors(problems):
    return [p for p in problems if p["level"] == "error"]


def project_problems(project, ids):
    out = []
    have = sorted(
        {s["name"] for s in skills.found(skills.SHIPPED) + skills.found(project.path / "skills")}
    )
    for name in skills.unknown(project):
        near = closest(name, have)
        out.append(
            problem(
                "",
                "error",
                _(
                    "project.yaml 的 skills 中的「{name}」不存在（有：{have}）",
                    name=name,
                    have=", ".join(have) or "-",
                ),
                _("是不是「{near}」？", near=near)
                if near
                else _("删掉它，或在项目的 skills/{name}/SKILL.md 写一个", name=name),
            )
        )
    lanes = lane_order(project.config)
    if lanes:
        for type_ in dict.fromkeys(project.tasks[tid].type for tid in ids):
            if type_ not in lanes:
                out.append(
                    problem(
                        "",
                        "warn",
                        _(
                            "类型「{type}」不在 project.yaml 的 lanes 中：它的泳道排在最后",
                            type=type_,
                        ),
                        _("把它加进 lanes"),
                    )
                )
    return out


def task_problems(project, t, cyclic, cycles_told):
    if t.invalid:  # what follows would only repeat it
        return [
            problem(
                t.id,
                "error",
                t.invalid,
                _("改正任务文件开头的 frontmatter（--- 之间的 YAML）"),
            )
        ]
    out = []
    for p in graph_problems(project, t, cyclic):
        if p["kind"] == "cycle":  # once per cycle (its edges), on the first of its tasks checked
            ring = frozenset(zip(p["tasks"], p["tasks"][1:]))
            if ring in cycles_told:
                continue
            cycles_told.add(ring)
        out.append(problem(t.id, "error", p["message"], p["fix"]))
    out += machine_problems(project, t)
    try:
        own = draft_checks(project, t)
    except ValueError as e:  # the type's default checks cannot be read
        return [
            *out,
            problem(t.id, "error", str(e), _("改正 task_types/{type}.md 的 checks", type=t.type)),
        ]
    for check in own:
        out += check_problems(project, t, check)
    for e in project.upstream(t.id):
        if project.tasks[e.up].status == "cancelled":
            out.append(
                problem(
                    t.id,
                    "warn",
                    _("依赖的任务「{task}」已取消：本任务会一直等它", task=e.up)
                    if e.blocking
                    else _("依赖的任务「{task}」已取消", task=e.up),
                    _("去掉这条依赖，或改为依赖接替它的任务"),
                )
            )
    return out


def draft_checks(project, t):
    """The checks a task will run: a draft without checks of its own gets its type's when it is joined (ops._join)."""
    if t.status != "draft" or "checks" in t.fm:
        return t.checks
    ty = next((ty for ty in project.types if ty.id == t.type), None)
    return checks.parse(ty.checks) if ty else []


def machine_problems(project, t):
    declared = [m["name"] for m in machines.declared(project.config)]
    out = []
    for name in t.machines:
        if name in declared:
            continue
        near = closest(name, declared)
        out.append(
            problem(
                t.id,
                "error",
                _(
                    "machines 中的「{name}」不在 project.yaml 的 machines 中（有：{have}）",
                    name=name,
                    have=", ".join(declared) or _("还没有声明机器"),
                ),
                _("是不是「{near}」？", near=near)
                if near
                else _("先在 project.yaml 的 machines 中加上它，或改成已声明的机器"),
            )
        )
    return out


def modules():
    """The package's modules, as python -m names them after the package."""
    return sorted(
        [p.stem for p in PACKAGE.glob("*.py") if p.stem != "__init__"]
        + [p.parent.name for p in PACKAGE.glob("*/__init__.py")]
    )


def module_exists(name):
    base = PACKAGE.joinpath(*name.split("."))
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def commands(run):
    """A check's command line as its simple commands, [[word]]: split at the shell's operators (stuck to a word or
    not), without a keyword or ! that leads one, a redirection's target or a comment. Raises ValueError (a quote that
    does not close)."""
    lex = shlex.shlex(run, posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    out, cur, redirect = [], [], False
    for tok in lex:
        if OPERATOR.fullmatch(tok):
            if "<" in tok or ">" in tok:
                redirect = True  # what follows is written or read by the shell, not by the command
            elif cur:
                out.append(cur)
                cur = []
        elif redirect:
            redirect = False
        elif cur or tok not in KEYWORDS:
            cur.append(tok)
    return [*out, cur] if cur else out


def check_problems(project, t, check):
    """A check's module, its kind (perfcheck, ledger) and its paths."""
    name = check["name"]
    try:
        cmds = commands(check["run"])
    except ValueError as e:
        return [
            problem(
                t.id,
                "warn",
                _("验收检查「{check}」的命令无法静态分析：{error}", check=name, error=e),
                _("检查引号是否配对"),
            )
        ]
    out = []
    for words in cmds:
        out += module_problems(t, name, words) + path_problems(project, t, name, words)
    return out


def module_problems(t, name, words):
    out = []
    for i, word in enumerate(words):
        if word == "-m" and i + 1 < len(words) and words[i + 1].startswith(PACKAGE.name + "."):
            module = words[i + 1].split(".", 1)[1]
            if not module_exists(module):
                near = closest(module, modules())
                out.append(
                    problem(
                        t.id,
                        "error",
                        _(
                            "验收检查「{check}」运行的模块 {package}.{module} 不存在",
                            check=name,
                            package=PACKAGE.name,
                            module=module,
                        ),
                        _("是不是 {package}.{near}？", package=PACKAGE.name, near=near)
                        if near
                        else _("改成存在的模块，或去掉这条检查"),
                    )
                )
            elif module in SUBCOMMANDS and i + 2 < len(words):
                out += kind_problem(t, name, module, words[i + 2])
    return out


def kind_problem(t, name, module, kind):
    known = SUBCOMMANDS[module]()
    if kind in known or kind.startswith(("$", "-")):
        return []
    near = closest(kind, known)
    return [
        problem(
            t.id,
            "error",
            _(
                "验收检查「{check}」的 {module} 类型「{kind}」不认识（可用 {known}）",
                check=name,
                module=module,
                kind=kind,
                known="/".join(known),
            ),
            _("是不是「{near}」？", near=near) if near else _("改成上面列出的一种"),
        )
    ]


def path_problems(project, t, name, words):
    """The paths under $AGENT_SYS_SKILLS/ and $AGENT_SYS_PROJECT/ in one simple command that the shell would not
    change. A shipped skill's file must be there; so must a script the check runs from the project. What else it names
    in the project (outside workspaces/) may be written by an upstream task: a warning. What test or [ looks at is what
    the check is about: not checked. A path that cannot be looked at (too long, a directory you cannot read) is not
    either."""
    if not words or words[0] in ("test", "[", "[["):
        return []
    out = []
    for i, word in enumerate(words):
        m = re.fullmatch(r"\$(?:\{(\w+)\}|(\w+))/(.+)", word)
        var = m and (m[1] or m[2])
        if var not in ROOTS or DYNAMIC.search(m[3]):
            continue
        rel = os.path.normpath(m[3])
        if var == "AGENT_SYS_PROJECT" and rel.split(os.sep)[0] == "workspaces":
            continue
        path = Path(ROOTS[var](project)) / rel
        try:
            if path.exists():
                continue
            near = closest(path.name, os.listdir(path.parent)) if path.parent.is_dir() else None
        except (OSError, ValueError):
            continue
        runs = i == 0 or (i == 1 and words[0] in INTERPRETERS)
        level = "error" if var == "AGENT_SYS_SKILLS" or runs else "warn"
        message = _("验收检查「{check}」用到的 {path} 不存在", check=name, path=word)
        if level == "warn":
            message += _("（如果它由上游任务生成，可以忽略）")
        out.append(
            problem(
                t.id,
                level,
                message,
                _("是不是 {near}？", near=f"{word.rsplit('/', 1)[0]}/{near}")
                if near
                else _("改正路径，或去掉这条检查"),
            )
        )
    return out


def text(problems):
    """The list as lines: level, task, message, then the fix indented."""
    lines = []
    for p in problems:
        where = p["task"] or "project.yaml"
        lines.append(f"{p['level']:<5}  {where}: {p['message']}")
        if p["fix"]:
            lines.append(f"       → {p['fix']}")
    return lines


def refuse(project, problems, action):
    """Refuse an action (409) when problems has an error; the whole list goes with it (the reply's problems and
    project: the UI lists them with links to the tasks)."""
    if bad := errors(problems):
        raise AgentSysError(
            "\n".join(
                [
                    _("有 {n} 个问题要先改正，才能{action}：", n=len(bad), action=action),
                    *text(problems),
                ]
            ),
            409,
            problems=problems,
            project=project.id,
        )
