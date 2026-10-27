"""Acceptance checks: commands agent_sys runs when the agent says a task is done (design §5.4).

In a task's frontmatter (a type's skeleton can carry default checks):

  checks:
    - test -s "$AGENT_SYS_WORKSPACE/report.md"           # a command; its name is the command
    - {name: results, run: '"$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger check "$AGENT_SYS_WORKSPACE/results"', timeout: 60}
  check_retries: 2                                      # how many times a failure goes back to the agent

The runner runs them when a round ends with progress.md 已完成 (not when a done task only answered a follow-up
question: followup_only), in the project directory with `bash -c`, one after the other, each with its timeout and the
AGENT_SYS_* variables of env(). It writes every attempt into the run's checks.json and, while retries are left, sends
what failed back into the same session; the server reads the last attempt (summary) to decide between done and
待确认 · 验收未通过. After a terminal takeover the wrapper in the tmux window runs them once. A check is the harness's,
not the agent's: the agent cannot mark a task done past a failing one, and a task whose checks cannot be read does not
run (runs.start_run).
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from .i18n import _, translate
from .model import iso, mtime, read_json, write_json

PACKAGE = Path(__file__).resolve().parent
ENTRY_MODULES = (
    "ledger",
    "perfcheck",
)  # the package's modules the template's checks run (python -m agent_sys_v2.<m>)
TOOLCHAIN_CHECK = "the files the checks use are unchanged"
DEFAULT_TIMEOUT = 600  # seconds
DEFAULT_RETRIES = 2
TAIL = 4000  # characters of a check's output kept (its end: where the error usually is)
FILE = "checks.json"


def parse(value):
    """frontmatter checks -> [{name, run, timeout}]; [] for none. Raises ValueError when it is malformed."""
    if value in (None, "", []):
        return []
    items = value if isinstance(value, list) else [value]
    out = []
    for n, item in enumerate(items, 1):
        if isinstance(item, str) and item.strip():
            item = {"run": item}
        if not isinstance(item, dict) or not str(item.get("run") or "").strip():
            raise ValueError(_("checks 的第 {n} 项应为一条命令，或 {name, run, timeout}", n=n))
        timeout = item.get("timeout", DEFAULT_TIMEOUT)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(_("checks 的第 {n} 项的 timeout 应为正数（秒）", n=n))
        run = str(item["run"]).strip()
        out.append({"name": str(item.get("name") or run)[:120], "run": run, "timeout": timeout})
    return out


def retries(fm):
    value = fm.get("check_retries")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return DEFAULT_RETRIES


def env(project_dir, task_id, workspace, machines, skills_dir):
    """What a check (and the agent, through launch.ENV) gets to know where it is. AGENT_SYS_PYTHON runs agent_sys's
    own tools whatever is on PATH: "$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger check ..."""
    return {
        "AGENT_SYS_PYTHON": sys.executable,
        "AGENT_SYS_PROJECT": str(project_dir),
        "AGENT_SYS_TASK": task_id,
        "AGENT_SYS_WORKSPACE": str(workspace),
        "AGENT_SYS_MACHINES": ",".join(machines),
        "AGENT_SYS_SKILLS": str(skills_dir),
    }


def run_one(check, cwd, env):
    """Run one check. Its whole process group is killed when it runs out of time."""
    start = time.time()
    proc = subprocess.Popen(
        ["bash", "-c", check["run"]],
        cwd=cwd,
        env={**os.environ, **env},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    timed_out = False
    try:
        out, _err = proc.communicate(timeout=check["timeout"])
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        out, _err = proc.communicate()
    text = out.decode("utf-8", "replace")
    return {
        "name": check["name"],
        "run": check["run"],
        "ok": proc.returncode == 0 and not timed_out,
        "exit": proc.returncode,
        "timed_out": timed_out,
        "seconds": round(time.time() - start, 2),
        "output": text if len(text) <= TAIL else "…" + text[-TAIL:],
    }


def run_all(checks, cwd, env, on_start=None, on_end=None, halt=None, first=()):
    """Run the checks in order (all of them: one failure does not hide the next). Returns the attempt record, or None
    if halt (a threading.Event) was set: the attempt was stopped and does not count, and the check it killed gets
    no on_end. first: results that go before the checks' own (toolchain_check)."""
    results = list(first)
    for n, check in enumerate(checks, len(results) + 1):
        if on_start:
            on_start(n, check)
        result = run_one(check, cwd, env)
        if halt and halt.is_set():
            return None
        results.append(result)
        if on_end:
            on_end(n, result)
    return {"time": iso(time.time()), "ok": all(r["ok"] for r in results), "results": results}


def record(run_dir, attempt):
    """Append an attempt to the run's checks.json."""
    data = load(run_dir) or {"attempts": []}
    data["attempts"].append(attempt)
    write_json(Path(run_dir) / FILE, data)
    return len(data["attempts"])


def load(run_dir):
    data = read_json(Path(run_dir) / FILE)
    return data if isinstance(data.get("attempts"), list) else None


def summary(run_dir):
    """The last attempt of a run, short: {ok, passed, total, failed: [names], attempts, time}; None if none ran."""
    data = load(run_dir)
    if not data or not data["attempts"]:
        return None
    last = data["attempts"][-1]
    results = last.get("results") or []
    return {
        "ok": bool(last.get("ok")),
        "passed": sum(bool(r.get("ok")) for r in results),
        "total": len(results),
        "failed": [r.get("name") for r in results if not r.get("ok")],
        "attempts": len(data["attempts"]),
        "time": last.get("time") or "",
    }


def followup_only(before, workspace):
    """The run only answered a follow-up question: the task was done before it and its main report is unchanged
    (before: launch.json's, see launch.prepare). Its checks do not run again and it stays done."""
    from .scan import main_report  # scan imports this module

    report = main_report(workspace)
    now = (report, iso(mtime(workspace / report)) if report else "")
    was = (before.get("report"), before.get("report_mtime") or "")
    return before.get("status") == "done" and now == was


def feedback(attempt, left, lang="zh"):
    """The message that sends a failed attempt back to the agent, in its language."""
    failed = [r for r in attempt["results"] if not r["ok"]]
    lines = [
        translate(
            "验收检查没有通过（{failed}/{total} 条失败），任务还不能算完成。请修复后再把 progress.md 标为已完成；"
            "如果需要我来决定，把状态改为阻塞并写明原因。",
            lang,
            failed=len(failed),
            total=len(attempt["results"]),
        )
    ]
    for r in failed:
        why = translate("超时", lang) if r["timed_out"] else f"exit {r['exit']}"
        output = r["output"].strip() or translate("（没有输出）", lang)
        head = f"- `{r['name']}`" + translate("（{why}）：", lang, why=why)
        lines.append(f"\n{head}\n```\n{output}\n```")
    lines.append("\n" + translate("（还会再检查 {left} 次。）", lang, left=left))
    return "\n".join(lines)


# ---------------------------------------------------------------- the checks' own files


def imports(name):
    """The modules of this package one of its modules imports (from . import x, from .x import y, agent_sys_v2.x)."""
    try:
        tree = ast.parse((PACKAGE / f"{name}.py").read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module:
                out.add(node.module.split(".")[0])
            elif node.level == 1 or node.module == PACKAGE.name:
                out |= {a.name for a in node.names}
            elif node.module and node.module.startswith(PACKAGE.name + "."):
                out.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            out |= {
                a.name.split(".")[1] for a in node.names if a.name.startswith(PACKAGE.name + ".")
            }
    return {m for m in out if (PACKAGE / f"{m}.py").is_file()}


def module_closure(names):
    """names and every package module they import, transitively."""
    seen, todo = set(), [n for n in names if (PACKAGE / f"{n}.py").is_file()]
    while todo:
        name = todo.pop()
        if name not in seen:
            seen.add(name)
            todo += sorted(imports(name) - seen)
    return seen


def expand(word, env):
    """$NAME and ${NAME} in a check's word, from env (then the process's environment)."""
    return re.sub(
        r"\$\{(\w+)\}|\$(\w+)",
        lambda m: env.get(m[1] or m[2], os.environ.get(m[1] or m[2], "")),
        word,
    )


def named_files(check_list, project_dir, env):
    """The files a check's command names, outside the project's workspaces (what the agent writes there is what the
    checks judge, not how): a script, a rules file."""
    project_dir = Path(project_dir).resolve()
    out = set()
    for c in check_list:
        try:
            words = shlex.split(c["run"])
        except ValueError:
            words = c["run"].split()
        for word in words:
            text = expand(word, env)
            if not text or text == env.get("AGENT_SYS_PYTHON"):
                continue
            path = Path(text) if Path(text).is_absolute() else project_dir / text
            try:
                path = path.resolve()
                if not path.is_file() or (project_dir / "workspaces") in path.parents:
                    continue
            except (OSError, ValueError):
                continue
            out.add(path)
    return out


def toolchain(project_dir, check_list, env):
    """{file: sha256} of what the task's checks run, recorded when a run starts and compared before each attempt:
    this package's modules the checks import, the scripts of the shipped skills and of the project's skills
    (skills/**/scripts/), the files the check commands name, and anything in the project directory named like this
    package (it would shadow it)."""
    names = set(ENTRY_MODULES) | {
        m for c in check_list for m in re.findall(rf"{PACKAGE.name}\.(\w+)", c["run"])
    }
    files = {PACKAGE / f"{m}.py" for m in module_closure(names)}
    for root in (PACKAGE / "skills", Path(project_dir) / "skills"):
        files |= {
            p
            for p in root.glob("*/scripts/**/*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        }
    files |= named_files(check_list, project_dir, env)
    # a package of this name in the project directory would be imported instead of this one (Python before 3.11
    # puts the current directory first on sys.path whatever PYTHONSAFEPATH says)
    for shadow in Path(project_dir).glob(f"{PACKAGE.name}*"):
        files |= {shadow} if shadow.is_file() else {p for p in shadow.rglob("*") if p.is_file()}
    out = {}
    for p in sorted(files):
        try:
            out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            continue
    return out


def toolchain_changes(before, now):
    """['changed: f', 'added: f', 'removed: f'] between two toolchain() records."""
    return (
        [f"changed: {f}" for f in sorted(before) if f in now and now[f] != before[f]]
        + [f"added: {f}" for f in sorted(now) if f not in before]
        + [f"removed: {f}" for f in sorted(before) if f not in now]
    )


def toolchain_check(before, now):
    """A failed check result listing what changed since the run started; None when nothing did (or the run recorded
    nothing)."""
    if not isinstance(before, dict):
        return None
    changes = toolchain_changes(before, now)
    if not changes:
        return None
    text = "\n".join(
        [
            "The files the acceptance checks use changed since this run started, so no check can be trusted:",
            *(f"  {c}" for c in changes),
            "",
            "Put them back (git checkout -- <file>, or undo the edit): a check is the user's, not the task's to "
            "change. If the change is the user's (agent_sys_v2 was upgraded), say so and mark the task blocked in progress.md; "
            "the next run records the new files.",
            "If agent_sys_v2 is installed from the same checkout as the repository the project changes (an editable "
            "install of repos/Infera), a git checkout or pull there changes the checks too: install agent_sys_v2 from "
            "a checkout of its own, or a wheel.",
        ]
    )
    return {
        "name": TOOLCHAIN_CHECK,
        "run": "(agent_sys: compare the checks' files with the run's start)",
        "ok": False,
        "exit": None,
        "timed_out": False,
        "seconds": 0,
        "output": text if len(text) <= TAIL else "…" + text[-TAIL:],
    }
