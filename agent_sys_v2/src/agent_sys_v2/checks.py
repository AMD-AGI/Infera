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

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .i18n import _, translate
from .model import iso, mtime, read_json, write_json

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


def run_all(checks, cwd, env, on_start=None, on_end=None, halt=None):
    """Run the checks in order (all of them: one failure does not hide the next). Returns the attempt record, or None
    if halt (a threading.Event) was set: the attempt was stopped and does not count, and the check it killed gets
    no on_end."""
    results = []
    for n, check in enumerate(checks, 1):
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
