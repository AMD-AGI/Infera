"""Runs as the server handles them: start a runner or a terminal takeover in a tmux window,
pass it your messages and 停止 through its inbox, follow its events, and when a run is over decide the task's next
status from the run's end record and progress.md. The server never holds a runner: everything here
works from the files in the run directory, so a restarted server picks up where the last one was. Only runs on
this host are touched; a run on another host is only shown.

The functions that change a task take (home, project dir, task id), read the project fresh and are called under the
server's write lock; Monitor is the server's 0.5 s loop over the running tasks. The events themselves are read in
events.py, the processes found in procs.py.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import signal
import socket
import sys
import time
from pathlib import Path

from . import checks, files, launch, machines, questions, tmux
from .events import Follow
from .i18n import _, status_word
from .model import (
    AgentSysError,
    iso,
    load_config,
    mtime,
    now_iso,
    read_json,
    run_cost,
    runs_dir,
    write_state,
)
from .ops import check_not_archived, flow, start_project, update_frontmatter
from .procs import check_pids, children, end_checks, end_session, is_runner, signal_all
from .scan import load_project, project_dir, project_ids

log = logging.getLogger("agent_sys")
HOST = socket.gethostname()
# a run with no run.json this long after it was started: its runner never came up
START_SECONDS = 30
STOP_WAIT = 12  # how long a stop or a takeover waits for the runner to end its round and exit
# how long 结束接管 waits for the wrapper; one that runs the checks is left to finish them
END_TAKEOVER_WAIT = 5


def clean(t):
    """Before a resume or a takeover: KILL the last run's runner if it is still that process, and the checks it
    left, then end whatever claude still runs the session. Claude Code has no session lock: two processes on one
    session fork its record."""
    for run in sorted(runs_dir(t.workspace).glob("*/run.json")):
        rec = read_json(run)
        if rec.get("host") == HOST and not rec.get("ended"):
            if is_runner(rec.get("pid"), run.parent):
                os.kill(rec["pid"], signal.SIGKILL)
            end_checks(run.parent)
    end_session(t.state["session_id"])


# ---------------------------------------------------------------- run files


def new_run_dir(t):
    """Runs are named after their start, local time (today's cost groups them by day)."""
    base = runs_dir(t.workspace) / time.strftime("%Y%m%d-%H%M%S")
    path, n = base, 2
    while path.exists():
        path, n = base.with_name(f"{base.name}-{n}"), n + 1
    return path


def record(run_dir):
    """run.json; before the runner wrote it, what launch.json says (a run that is starting)."""
    rec = read_json(run_dir / "run.json")
    if rec:
        return rec
    cfg = read_json(run_dir / "launch.json")
    if not cfg:
        return {}
    return {
        "id": run_dir.name,
        "kind": cfg.get("kind"),
        "resume": cfg.get("resume"),
        "pid": None,
        "host": HOST,
        "session_id": cfg.get("session_id"),
        "model": cfg.get("model"),
        "window": cfg.get("window"),
        "started": cfg.get("started"),
        "ended": None,
        "cost_start_usd": cfg.get("cost_start_usd") or 0.0,
        "cost_end_usd": None,
        "exit": None,
        "starting": True,
    }


def is_live(run_dir, rec=None):
    """The run has not ended: it has no end record and its runner is still there (on another host: no end record)."""
    rec = rec if rec is not None else record(run_dir)
    if not rec or rec.get("ended"):
        return False
    if rec.get("host") != HOST:
        return True
    if rec.get("starting"):
        return time.time() - (mtime(run_dir / "launch.json") or 0) < START_SECONDS
    return is_runner(rec.get("pid"), run_dir)


_last_words = {}  # events.jsonl -> ((size, mtime), the agent's last words in it)


def last_words(path):
    """The agent's last words in an events.jsonl, "" if none; read from its last 256 KB, cached by its size and
    mtime."""
    try:
        st = path.stat()
    except OSError:
        return ""
    key = (st.st_size, st.st_mtime_ns)
    if _last_words.get(path, (None,))[0] != key:
        follow = Follow(path.parent)
        follow.pos = max(0, st.st_size - 262144)
        if follow.pos:  # start at a line boundary
            with open(path, "rb") as f:
                f.seek(follow.pos)
                follow.pos += len(f.readline())
        _last_words[path] = (key, follow.poll().last_text)
    return _last_words[path][1]


def last_message(t):
    """The agent's last words in the task's latest run that has any: a terminal takeover has none (its events, if
    any, are the checks')."""
    ws_runs = runs_dir(t.workspace)
    if not (t.state or {}).get("run_id") or not ws_runs.is_dir():
        return ""
    for run_dir in sorted(ws_runs.iterdir(), reverse=True):
        if text := last_words(run_dir / "events.jsonl"):
            return text
    return ""


def run_json(t, rec, follow=None):
    """Run from its record; a live sdk run adds what it is doing. follow: the Monitor's, for the current run."""
    run_dir = runs_dir(t.workspace) / rec["id"]
    live = rec.get("id") == (t.state or {}).get("run_id") and is_live(run_dir, rec)
    f = None if rec.get("ended") else follow or Follow(run_dir).poll()
    cost = run_cost(rec, f.cost if f else rec.get("cost_end_usd"))
    out = {
        "id": rec["id"],
        "kind": rec.get("kind"),
        "resume": rec.get("resume"),
        "started": rec.get("started"),
        "ended": rec.get("ended"),
        "exit": rec.get("exit") or (None if live else "interrupted"),
        "exit_detail": rec.get("exit_detail") or "",
        "model": rec.get("model"),
        "live": live,
        "cost_usd": None if cost is None else round(cost, 6),
    }
    if rec.get("host") and rec["host"] != HOST:
        out["host"] = rec["host"]
    if rec.get("kind") == "terminal" and rec.get("window"):
        out["attach"] = tmux.attach(rec["window"])
    if live and rec.get("kind") != "terminal":
        out.update(
            activity=f.activity,
            current=f.current,
            background=[b.get("description", "") for b in f.bg],
        )
    return out


def list_runs(t, follow=None):
    """The task's runs, newest first; the current one may still be starting (launch.json only)."""
    recs = (
        [record(d) for d in sorted(runs_dir(t.workspace).iterdir(), reverse=True)]
        if runs_dir(t.workspace).is_dir()
        else []
    )
    return [
        run_json(t, r, follow if r["id"] == (t.state or {}).get("run_id") else None)
        for r in recs
        if r
    ]


# ---------------------------------------------------------------- starting runs


def replay_spec(t, n):
    """agent.replay (project.yaml or the task's frontmatter): a recorded events.jsonl to play back instead of calling
    the model, or a list of them (run n plays the n-th, the last one after that), or {events, speed, max_gap, writes}.
    writes: the files run n writes into the workspace when its recording is over, as an agent would (progress.md,
    report.md, questions.md: {path relative to the workspace: text}); a list picked like events, or one mapping for
    every run. With them a chain of tasks, questions and reviews can be played without a model."""
    spec = t.agent.get("replay")
    if not spec:
        return None
    spec = dict(spec) if isinstance(spec, dict) else {"events": spec}

    def pick(items):
        if not isinstance(items, list):
            return items
        return items[min(n, len(items) - 1)] if items else None

    events = spec.get("events")
    path = Path(str(pick(events if isinstance(events, list) else [events])))
    if not path.is_absolute():
        path = t.workspace.parent.parent.parent / path  # relative to the project directory
    writes = pick(spec.get("writes")) or {}
    if not isinstance(writes, dict):
        raise AgentSysError(
            _("agent.replay.writes 应为「工作区内的路径: 内容」的键值表，或者它们的列表")
        )
    writes = {
        str(files.resolve(t.workspace, str(rel), write=True)): str(text)
        for rel, text in writes.items()
    }
    return {
        "events": str(path),
        "speed": spec.get("speed") or 1,
        "max_gap": spec.get("max_gap"),
        "writes": writes,
    }


def check_can_run(project, t):
    if not t.managed:
        raise AgentSysError(
            _("该任务未托管，agent_sys 没有它的会话：请在原终端中回复，或先将任务纳入托管"), 409
        )
    check_not_archived(project)


def budget(t):
    """The task's agent.budget_usd; 0 if it is not a number (the task stops on its budget until you add some)."""
    try:
        return float(t.agent.get("budget_usd") or 0)
    except (TypeError, ValueError):
        return 0.0


def budget_left(t):
    return budget(t) - float(t.state.get("cost_usd") or 0)


def check_can_start(home, project, t, kind="sdk"):
    """Refuse a new run of t, before anything is written: no budget left (a terminal has none to keep); a
    frontmatter or checks that cannot be read (scan: t.invalid; it would run without its checks, and with the default
    machines, budget and review); extra_args that cannot be parsed."""
    if kind != "terminal" and budget_left(t) <= 0:
        raise AgentSysError(
            _(
                "预算已用尽（${cost} / ${budget}），请先追加预算",
                cost=f"{t.state.get('cost_usd', 0):.2f}",
                budget=f"{budget(t):.2f}",
            ),
            409,
        )
    if t.invalid:
        raise AgentSysError(_("任务 prompt 有错误，改正后才能运行：{error}", error=t.invalid), 409)
    check_machines(home, project, t)
    launch.parse_extra_args(t.agent.get("extra_args"))


def check_machines(home, project, t):
    """Refuse a run of t while another task holds one of its machines. Every way a task runs again comes through
    start_run, under the write lock, so no two tasks hold a machine at once."""
    if not t.machines:
        return
    projects = [project]
    for pid in project_ids(home):
        if pid != project.id:
            try:
                projects.append(load_project(project_dir(home, pid)))
            # one that cannot be read, whatever the error: the scheduler does not see it either (the server's scan
            # reports why)
            except Exception:
                pass
    machines.check_free(projects, project, t)


def start_run(home, project, t, text="", kind="sdk"):
    """A new run of the task: write its run directory, mark the task running, open the tmux window. text is your
    message (the first run sends the task's first message, with text as your reply); messages that came too late
    for the last run (state unsent) go first."""
    state = t.state
    check_can_start(home, project, t, kind)
    if kind != "terminal":
        text = "\n\n".join(x for x in [*(state.get("unsent") or []), text] if x.strip())
    clean(t)
    replay = replay_spec(t, state.get("runs") or 0)
    resume = (state.get("runs") or 0) > 0 if replay else launch.session_exists(state["session_id"])
    parts, system_text = launch.system(home, project, t)
    run_dir, updated, lang = new_run_dir(t), "", launch.agent_language(home)
    if resume:
        known = launch.known_system_md(t.workspace)
        updated = (
            str(run_dir / "system.md") if known and known != system_text and text.strip() else ""
        )
        wait = t.status == "needs_you" and t.detail.get("kind") == "wait_interrupted"
        moved = (state["moved_from"], t.id) if state.get("moved_from") else None
        message = launch.resume_message(text, updated, wait, moved, lang) if text.strip() else ""
    else:
        message = launch.first_message(project, t, text, lang)
    kind = "replay" if replay and kind == "sdk" else kind
    window = tmux.window_name(t.name, f"{t.name}-{project.id}")
    cfg = launch.prepare(
        home,
        project,
        t,
        run_dir=run_dir,
        kind=kind,
        parts=parts,
        system_text=system_text,
        message=message,
        resume=resume,
        max_budget_usd=max(budget_left(t), 0.0),
        # a new session (its record was cleaned up) counts its cost from 0
        cost_start_usd=float(state.get("session_cost_usd") or 0) if resume else 0.0,
        window=window,
        replay=replay,
        cli_path=load_config(home).get("cli_path"),
        system_known=not resume or bool(updated),
    )
    now = now_iso()
    # the move is told once; the late messages were sent now
    told = {
        k: v
        for k, v in state.items()
        if (k != "moved_from" or (resume and not message)) and (k != "unsent" or kind == "terminal")
    }
    write_state(
        t.workspace,
        {
            **told,
            "status": "running",
            "detail": {},
            "since": now,
            "run_id": run_dir.name,
            "runs": (state.get("runs") or 0) + 1,
            "started": state.get("started") or now,
        },
    )
    argv = [sys.executable, "-m", "agent_sys_v2.runner", str(run_dir)] + (
        ["--terminal"] if kind == "terminal" else []
    )
    try:
        tmux.open_window(window, shlex.join(argv), project.path)
    except AgentSysError:
        # no run started: the task is as it was, and no run directory is left half made
        write_state(t.workspace, state)
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    # Until the runner wrote run.json there is no pid to stop or check; no other action may come in between (the
    # caller holds the write lock). A runner that never comes up is found by the Monitor after START_SECONDS.
    wait_for(lambda: (run_dir / "run.json").is_file(), 10)
    return cfg


def wait_for(cond, seconds):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.2)
    return cond()


def append_inbox(run_dir, cmd):
    """One line per command, ASCII only: U+2028 and the like in a pasted message would split the line for its reader
    (str.splitlines) and the message would be lost."""
    with open(run_dir / "inbox.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(cmd) + "\n")


def current_run(t):
    run_id = (t.state or {}).get("run_id")
    if not run_id:
        return None, {}
    run_dir = runs_dir(t.workspace) / run_id
    return run_dir, record(run_dir)


def check_local(rec):
    if rec.get("host") and rec["host"] != HOST:
        raise AgentSysError(
            _("该任务在 {host} 上运行，请在那台机器上的 agent_sys 中操作", host=rec["host"]), 409
        )


def stop_runner(t, reason="stopped"):
    """Ask the live runner to interrupt and exit, wait for it; KILL it if it does not go."""
    run_dir, rec = current_run(t)
    if run_dir and is_live(run_dir, rec):
        append_inbox(run_dir, {"type": "stop", "reason": reason})
        if not wait_for(lambda: not is_live(run_dir), STOP_WAIT):
            clean(t)
    return run_dir


# ---------------------------------------------------------------- actions


def fresh(pdir, tid):
    project = load_project(pdir)
    return project, project.task(tid)


def start(home, pdir, tid):
    """开始: a queued task's first run, whether or not its upstream tasks are done. The first start by hand ends
    the template's review (ops.start_project)."""
    project, t = fresh(pdir, tid)
    check_can_run(project, t)
    if t.status != "queued":
        raise AgentSysError(
            _("任务当前为「{status}」，无法执行「开始」", status=status_word(t.status)), 409
        )
    start_run(home, project, t)
    if project.template_review:
        start_project(pdir)


def message(home, pdir, tid, text):
    """Your message -> when it is read: after_tool / next_turn / now into the live run; resume starts a
    new round of the session."""
    if not str(text or "").strip():
        raise AgentSysError(_("消息内容不能为空"))
    project, t = fresh(pdir, tid)
    check_can_run(project, t)
    if t.status == "running":
        run_dir, rec = current_run(t)
        check_local(rec)
        if rec.get("kind") == "terminal":
            raise AgentSysError(_("该任务正在终端中接管，请在终端中回复"), 409)
        if not run_dir or not is_live(run_dir, rec):
            raise AgentSysError(_("runner 已退出，agent_sys 正在更新任务状态，请稍后再发送"), 409)
        delivery = Follow(run_dir).poll().delivery()
        append_inbox(run_dir, {"type": "user", "text": text})
        return delivery
    if t.status not in ("needs_you", "in_review", "done"):
        raise AgentSysError(
            _("任务当前为「{status}」，尚未开始，暂时不能发送消息", status=status_word(t.status)),
            409,
        )
    start_run(home, project, t, text)
    return "resume"


def resume_interrupted(home, pdir, tid):
    """继续, for 继续全部中断任务: only a task whose run was interrupted, its background wait too.
    False if the task is not one of those."""
    _p, t = fresh(pdir, tid)
    if t.status != "needs_you" or t.detail.get("kind") not in ("interrupted", "wait_interrupted"):
        return False
    message(home, pdir, tid, launch.say_for(home, "继续"))
    return True


def stop(home, pdir, tid):
    """停止 = interrupt(): the command in flight (or the check) is killed, the runner exits, the task goes to 待确认
    (stopped)."""
    project, t = fresh(pdir, tid)
    if not t.managed or t.status != "running":
        raise AgentSysError(
            _("任务当前为「{status}」，没有正在运行的 runner", status=status_word(t.status)), 409
        )
    run_dir, rec = current_run(t)
    check_local(rec)
    if rec.get("kind") == "terminal":
        raise AgentSysError(_("该任务正在终端中接管，请使用「结束接管」"), 409)
    stop_runner(t)
    finish(home, pdir, tid)


def takeover(home, pdir, tid):
    """在终端中接管: stop the runner (its round ends as stopped), clean the session's processes, start interactive
    claude in tmux. If the terminal does not come up the task stays stopped."""
    project, t = fresh(pdir, tid)
    check_can_run(project, t)
    if t.status == "running":
        run_dir, rec = current_run(t)
        check_local(rec)
        if rec.get("kind") == "terminal":
            raise AgentSysError(_("该任务已在终端中接管"), 409)
        stop_runner(t, "takeover")
        finish(home, pdir, tid)
        project, t = fresh(pdir, tid)
    elif t.status not in ("needs_you", "in_review", "done"):
        raise AgentSysError(
            _("任务当前为「{status}」，无法在终端中接管", status=status_word(t.status)), 409
        )
    cfg = start_run(home, project, t, kind="terminal")
    return {"window": f"{tmux.SESSION}:{cfg['window']}", "attach": tmux.attach(cfg["window"])}


def end_takeover(home, pdir, tid):
    """结束接管: end the interactive claude (the session's record stays); its wrapper runs the checks if
    progress.md says done, writes the end record, and the round ends as usual: progress.md decides. A wrapper still
    running the checks after END_TAKEOVER_WAIT is left to finish them; the Monitor moves the task on then."""
    project, t = fresh(pdir, tid)
    run_dir, rec = current_run(t)
    if t.status != "running" or rec.get("kind") != "terminal":
        raise AgentSysError(_("该任务当前未处于终端接管状态"), 409)
    check_local(rec)
    wrapper = rec.get("pid") if is_runner(rec.get("pid"), run_dir) else None
    inner = set(children(wrapper)) - set(check_pids(run_dir)) if wrapper else set()
    # an interactive shell (a replay task's stand-in for claude) ignores TERM
    signal_all(inner, signal.SIGHUP)
    end_session(t.state["session_id"], inner)
    if wrapper:
        wait_for(lambda: not is_live(run_dir), END_TAKEOVER_WAIT)
    finish(home, pdir, tid)


def add_budget(home, pdir, tid, usd, text=""):
    """追加预算: the task's budget (frontmatter agent.budget_usd, a setting of the task) grows by usd; a task that
    stopped on its budget goes on."""
    try:
        usd = float(usd)
    except (TypeError, ValueError):
        usd = 0
    if usd <= 0:
        raise AgentSysError(_("追加的金额应为正数"))
    project, t = fresh(pdir, tid)
    check_can_run(project, t)
    total = round(budget(t) + usd, 2)

    def change(fm):
        agent = fm.get("agent")
        if isinstance(agent, dict):
            agent["budget_usd"] = total
        else:
            fm["agent"] = flow({"budget_usd": total})

    update_frontmatter(t.path, change)
    project, t = fresh(pdir, tid)
    if t.status == "needs_you":
        text = text or launch.say_for(home, "预算已追加 ${usd}，请继续。", usd=f"{usd:g}")
        start_run(home, project, t, text)
    return total


def cancel(home, pdir, tid):
    """取消, also of a running task: its runner, its checks and the session's processes are ended first."""
    project, t = fresh(pdir, tid)
    if not t.managed or t.status in ("done", "cancelled"):
        raise AgentSysError(
            _(
                "任务当前为「{status}」{unmanaged}，无法取消",
                status=status_word(t.status),
                unmanaged="" if t.managed else _("（未托管）"),
            ),
            409,
        )
    state = t.state
    if t.status == "running":
        run_dir, rec = current_run(t)
        check_local(rec)
        clean(t)
        state = account(state, run_dir)
    write_state(t.workspace, {**state, "status": "cancelled", "detail": {}, "since": now_iso()})


# ---------------------------------------------------------------- the end of a run


def round_cost(t):
    """What the task's latest run cost (本轮花费, shown with a report to review); None while it is not known."""
    _run, rec = current_run(t)
    cost = run_cost(rec, rec.get("cost_end_usd") if rec.get("ended") else None)
    return None if cost is None else round(cost, 6)


def run_end_cost(run_dir, rec):
    """The session's total cost when the run ended: the end record; for a run that ended without one, the last
    result in its events (a terminal run: the session's record)."""
    if rec.get("ended") and rec.get("cost_end_usd") is not None:
        return rec["cost_end_usd"]
    if rec.get("kind") == "terminal":
        return launch.session_cost(rec.get("session_id"))
    return Follow(run_dir).poll().cost


def account(state, run_dir):
    """state with the run's cost added: the session total is cumulative, so a run costs end - start."""
    rec = record(run_dir)
    end = run_end_cost(run_dir, rec)
    cost = run_cost(rec, end)
    if cost is None:
        return state
    return {
        **state,
        "cost_usd": round(float(state.get("cost_usd") or 0) + cost, 6),
        "session_cost_usd": end,
    }


def round_end(t, cfg, run_dir):
    """A round ended normally. A done task that only answered a follow-up question (checks.followup_only) stays done
    and is not unread again. Else progress.md says done and the run's checks (if it had any) passed -> done (review:
    true -> in_review); done with checks that failed every attempt -> needs_you (checks_failed); else needs_you
    (asked, or questions when the round wrote questions.md)."""
    progress = t.progress
    if progress and progress.done:
        if checks.followup_only(cfg.get("before") or {}, t.workspace):
            return "done", {}, False
        if cfg.get("checks"):
            none = {"failed": [], "passed": 0, "total": len(cfg["checks"])}  # they never ran
            result = checks.summary(run_dir) or none
            if not result.get("ok"):
                detail = {k: result[k] for k in ("failed", "passed", "total")}
                return "needs_you", {"kind": "checks_failed", **detail}, False
        return ("in_review" if t.review else "done"), {}, True
    q = questions.round_questions(t)
    if q:
        return "needs_you", {"kind": "questions", "count": q["count"], "path": q["path"]}, False
    return "needs_you", {"kind": "asked"}, False


def unread_messages(run_dir, rec):
    """Your messages the runner had not read when it exited (sent in the moment it decided to)."""
    try:
        with open(run_dir / "inbox.jsonl", "rb") as f:
            f.seek(rec.get("inbox_read") or 0)
            lines = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            continue
        if cmd.get("type") == "user" and str(cmd.get("text") or "").strip():
            out.append(cmd["text"])
    return out


def over(t):
    """The task's current run is over and on this host: it has an end record, or its runner is gone (or never came
    up) without one."""
    run_dir, rec = current_run(t)
    return bool(run_dir) and rec.get("host", HOST) == HOST and not is_live(run_dir, rec)


def finish(home, pdir, tid):
    """If the task's current run is over, move the task on. Returns True if it did."""
    project, t = fresh(pdir, tid)
    if not t.managed or t.status != "running":
        return False
    if not over(t):
        return False
    run_dir, rec = current_run(t)
    if not rec.get("ended"):  # the runner died: nobody times its checks out any more
        end_checks(run_dir)
    cfg = read_json(run_dir / "launch.json")
    exit = rec.get("exit") if rec.get("ended") else None
    ended = rec.get("ended") or now_iso()  # a server that was down finds the run over later
    finished = t.state.get("finished") or ""
    if exit in ("idle", "terminal_exit") or (rec.get("kind") == "terminal" and not exit):
        status, detail, new = round_end(t, cfg, run_dir)
        finished = ended if new else finished
    elif exit in ("stopped", "takeover"):  # takeover: the terminal never came up
        status, detail = "needs_you", {"kind": "stopped"}
    elif exit == "budget":
        status, detail = "needs_you", {"kind": "budget"}
    elif exit == "error":
        status, detail = "needs_you", {"kind": "interrupted", "error": rec.get("exit_detail") or ""}
    else:  # the runner died: kill -9, a closed window, a reboot (or a stop that had to KILL it)
        status, detail, bg = "needs_you", {"kind": "interrupted"}, Follow(run_dir).poll().bg
        if bg:
            waiting_for = (t.progress.next if t.progress else "") or bg[0].get("description", "")
            detail = {"kind": "wait_interrupted", "waiting_for": waiting_for}
    state = account(t.state, run_dir)
    write_state(
        t.workspace,
        {**state, "status": status, "detail": detail, "since": ended, "finished": finished},
    )
    later = unread_messages(run_dir, rec) if exit == "idle" else []
    if later:  # sent while the runner was exiting: they start the next round
        project, t = fresh(pdir, tid)
        try:
            start_run(home, project, t, "\n\n".join(later))
        except AgentSysError:
            # e.g. the budget is used up: the task stays in 待确认, the messages go with its next run
            unsent = [*(t.state.get("unsent") or []), *later]
            write_state(t.workspace, {**t.state, "unsent": unsent})
    return True


# ---------------------------------------------------------------- the server's view of running tasks


class Monitor:
    """Every 0.5 s: the events of each running task's current run, read incrementally, give its
    sub-state (执行中 / 等待中 / 终端接管) and what it runs; a run that is over moves its task on (finish)."""

    def __init__(self, store):
        self.store = store
        self.follows = {}  # run dir -> Follow
        self.told = {}  # run dir -> the last seq the listeners were told of

    def follow(self, t):
        run_dir = runs_dir(t.workspace) / t.state["run_id"]
        f = self.follows.get(run_dir)
        if f is None:
            f = self.follows[run_dir] = Follow(run_dir)
        return run_dir, f.poll()

    def annotate(self, project):
        """Set the running tasks' detail (and since, while waiting) from their runs. Called after every rescan."""
        for t in project.tasks.values():
            if t.managed and t.status == "running" and t.state.get("run_id"):
                self.annotate_task(t)

    def annotate_task(self, t):
        run_dir, f = self.follow(t)
        rec = record(run_dir)
        cur = f.current
        if rec.get("kind") == "terminal":
            window = rec.get("window")
            detail = {"kind": "terminal", "attach": tmux.attach(window or ""), "window": window}
            cur = f.checking  # once claude exits, the wrapper runs the checks
        elif f.activity == "waiting":
            detail = {"kind": "waiting", "waiting_for": f.bg[0].get("description", "")}
            t.since = iso(f.idle_since) or t.since
        else:
            detail = {"kind": "working"}
        if cur and detail["kind"] != "waiting":
            detail["current"] = f"{cur['tool']} {cur['text']}"[:160]
        if rec.get("host") and rec["host"] != HOST:
            detail["host"] = rec["host"]
        t.detail = detail
        if f.cost is not None:  # the run's cost so far counts in the task's cost while it runs
            t.cost_running = run_cost(rec, f.cost)
        return run_dir, f

    def poll(self):
        """One round over the running tasks: finish the runs that are over, push sub-state changes."""
        finished, changed, grown = [], [], []
        with self.store.lock:
            running = [
                (p, t)
                for p in self.store.live()
                for t in p.tasks.values()
                if t.managed and t.status == "running" and t.state.get("run_id")
            ]
        for p, t in running:
            with self.store.lock:
                try:
                    if over(t):
                        if finish(self.store.home, p.path, t.id):
                            finished.append((p.id, t.id))
                        continue
                    before = (dict(t.detail), t.cost_running)
                    run_dir, f = self.annotate_task(t)
                    if (t.detail, t.cost_running) != before:
                        changed.append((p.id, t.id))
                    if f.seq != self.told.get(run_dir, -1):
                        self.told[run_dir] = f.seq
                        grown.append(
                            {"project": p.id, "task": t.id, "run": run_dir.name, "seq": f.seq}
                        )
                except AgentSysError:
                    pass
                # one task's files (NFS, permissions) must not hold up the other runs
                except Exception:
                    log.exception("following the run of %s/%s failed", p.id, t.id)
        live = {runs_dir(t.workspace) / t.state["run_id"] for _, t in running}
        self.follows = {k: v for k, v in self.follows.items() if k in live}
        self.told = {k: v for k, v in self.told.items() if k in live}
        if finished:
            self.store.sync(also=finished)
        if changed:
            self.store.notify(changed)
        # the run views of these runs fetch what is new (one SSE per page, not one per run)
        for run in grown:
            self.store.tell(run)
