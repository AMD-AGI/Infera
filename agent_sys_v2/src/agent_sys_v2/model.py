"""agent_sys data model: projects, task types, tasks, edges, and the files they live in.

  projects/<project>/project.yaml                  a project is a directory with project.yaml
  projects/<project>/tasks/<type>/<name>.md        a task; every subdirectory of tasks/ is a type (a swimlane)
  projects/<project>/task_types/<type>.md          the type's prompt skeleton; its frontmatter holds defaults (review, checks)
  projects/<project>/workspaces/<type>/<name>/     the task's workspace: progress.md, report.md, issue.md, ...
      .agent_sys/state.json                            present = the task is managed by agent_sys (schema below)
      .agent_sys/runs/<run_id>/run.json                one run of the agent (schema below; written by the runner)

Everything here is plain data that json.dumps can take through the to_json() methods.
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from ruamel.yaml import YAML, YAMLError
from ruamel.yaml.comments import CommentedMap

from . import i18n
from .i18n import _

# AGENT_SYS_HOME when neither --home nor $AGENT_SYS_HOME is given
DEFAULT_HOME = Path.home() / "agent_sys_home"

# What agent_sys.yaml (AGENT_SYS_HOME) and project.yaml mean when they leave a key out.
GLOBAL_DEFAULTS = {
    "max_running": 4,
    "language": i18n.DEFAULT,
    "protocol_language": "zh",
    "claude_config": "isolated",
    "stall_minutes": 15,
}
PROJECT_DEFAULTS = {
    "autorun": True,
    "max_running": 3,
    "agent": {
        "model": "opus",
        "effort": "high",
        "ultracode": False,
        "budget_usd": 200,
        "extra_args": [],
    },
}

# Task status. Unmanaged tasks get one from their progress.md (classify_progress); `unknown` is for an unmanaged
# progress.md whose 状态 line is none of the known words, and for a state.json that cannot be read.
STATUSES = ("draft", "queued", "running", "needs_you", "in_review", "done", "cancelled", "unknown")

# Edge kinds. Only input, code and children block the downstream task and take part in cycle checks. children: the
# upstream is done and so is every task whose parent it is (see Project.pending_children).
EDGE_KINDS = ("input", "code", "children", "review", "subtask")
BLOCKING = ("input", "code", "children")

# Project, type and task names. ASCII only, the same rule as the web UI and the prompt path rule (scan.TASK_REF).
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

PROJECTS = "projects"  # the directory under AGENT_SYS_HOME that holds one directory per project


def project_path(pid):
    """A project's directory relative to AGENT_SYS_HOME, the form paths take in the API."""
    return f"{PROJECTS}/{pid}"


class AgentSysError(Exception):
    """A request agent_sys refuses. The message, made with _() in the user's language, is shown as is; `status` is
    the HTTP code the server answers with (400 bad input, 404 not found, 409 not allowed now / cycle / already
    exists, 422 a file that cannot be read); `data` goes into the reply beside the message (preflight's problems)."""

    def __init__(self, message, status=400, **data):
        super().__init__(message)
        self.status = status
        self.data = data


def check_scope(scope, pid=None):
    """A settings, rules or knowledge write goes to global, or to a project (pid given). The project id, or None."""
    if scope not in ("global", "project") or (scope == "project" and not pid):
        raise AgentSysError(_("scope 应为 global，或 project 并给出 project"))
    return pid if scope == "project" else None


def project_agent(config):
    """A project's agent defaults: project.yaml agent over PROJECT_DEFAULTS."""
    return {**PROJECT_DEFAULTS["agent"], **plain(config.get("agent") or {})}


def check_name(name, what=None):
    """what: the name's kind for the message, e.g. _('任务名')."""
    if not NAME.fullmatch(name or ""):
        raise AgentSysError(
            _(
                "{what}「{name}」不合法：只能包含字母、数字、_ 和 -，且须以字母或数字开头",
                what=what or _("名称"),
                name=name,
            )
        )


# ---------------------------------------------------------------- time


def iso(ts):
    """Unix time -> '2026-10-22T04:10:28Z' (the API's format); None/0 -> ''."""
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if ts else ""


def now_iso():
    return iso(datetime.now(timezone.utc).timestamp())


def mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def local_time_iso(text):
    """progress.md's 完成时间 ('2026-10-21 17:10', local time) -> ISO UTC; '' if it is not such a time (2026-09-31
    and 24:00 are not: lint() reminds)."""
    m = re.match(r"(\d{4}-\d{2}-\d{2})[ T](\d{1,2}:\d{2})", text)
    try:
        return iso(datetime.strptime(f"{m[1]} {m[2]}", "%Y-%m-%d %H:%M").astimezone().timestamp())
    except (TypeError, ValueError):  # no match, or not a real date and time
        return ""


# ---------------------------------------------------------------- files


def read_text(path):
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def write_text(path, text):
    """Write through a temporary file and rename, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    """A JSON object from a file; {} if the file is missing, cannot be parsed or holds something else."""
    try:
        data = json.loads(read_text(path))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path, data):
    write_text(path, json.dumps(data, ensure_ascii=False, indent=1) + "\n")


_yaml = YAML()  # round trip: keeps comments, key order and flow style when a file is rewritten
_yaml.indent(mapping=2, sequence=4, offset=2)
_yaml.width = 4096
_yaml.preserve_quotes = True
# the instance keeps its reader and parser state: one document at a time
_yaml_lock = threading.Lock()
FRONT = re.compile(r"---\n(.*?\n)?---\n?", re.S)


def load_yaml(text):
    try:
        with _yaml_lock:
            data = _yaml.load(text) if text.strip() else None
    except YAMLError as e:
        raise ValueError(str(e).splitlines()[0]) from None
    if data is None:
        return CommentedMap()
    if not isinstance(data, dict):
        raise ValueError(_("不是键值表"))
    return data


def dump_yaml(data):
    buf = io.StringIO()
    with _yaml_lock:
        _yaml.dump(data, buf)
    return buf.getvalue()


def plain(value):
    """YAML data -> plain JSON-able data: dates and times (unquoted in YAML) become ISO strings."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, datetime):
        return iso(value.timestamp()) if value.tzinfo else value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def load_config(home):
    """agent_sys.yaml with the defaults filled in; a broken file counts as missing (the settings page shows it)."""
    try:
        config = plain(load_yaml(read_text(Path(home) / "agent_sys.yaml")))
    except ValueError:
        config = {}
    return {**GLOBAL_DEFAULTS, **config}


def positive(value, default):
    """A limit such as max_running from a file: a positive integer, else the default."""
    return (
        value if isinstance(value, int) and not isinstance(value, bool) and value >= 1 else default
    )


def non_negative(value, default):
    """A number of minutes such as stall_minutes from a file: a number >= 0 (0: off), else the default."""
    return (
        value
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
        else default
    )


def split_frontmatter(text):
    """'---\\n<yaml>---\\n<body>' -> (frontmatter, body). No frontmatter -> ({}, text). A byte order mark and
    CRLF line ends are dropped first (else a write would put a second frontmatter on top), and a block of comment
    lines only, such as a prompt opening with a --- rule, is prompt text, not an empty frontmatter.
    Raises ValueError when the YAML cannot be parsed."""
    text = text.removeprefix("\ufeff").replace("\r\n", "\n")
    m = FRONT.match(text)
    if not m:
        return CommentedMap(), text
    fm = load_yaml(m.group(1) or "")
    if not fm and (m.group(1) or "").strip():
        return CommentedMap(), text
    return fm, text[m.end() :]


def join_frontmatter(fm, body):
    return f"---\n{dump_yaml(fm)}---\n{body}" if fm else body


def read_task_file(path):
    return split_frontmatter(read_text(path))


def write_task_file(path, fm, body):
    write_text(path, join_frontmatter(fm, body))


def parse_dep(entry):
    """A depends_on entry, 'type/name' or {task, kind, label} -> (task, kind, label); a kind left out or null is input.
    task is None if the entry is neither."""
    if not isinstance(entry, dict):
        return (entry if isinstance(entry, str) else None), "input", ""
    tid = entry.get("task")
    return (
        tid if isinstance(tid, str) else None,
        entry.get("kind") or "input",
        str(entry.get("label") or ""),
    )


# ---------------------------------------------------------------- progress.md

CHECKBOX = re.compile(r"^\s*[-*] \[([ xX~])\]", re.M)


def field_value(text, keys):
    """Value of a '状态: …'-style line: first match, any case, list markers, full-width colons and bold (`**状态**:`,
    `**状态:**`, `**已完成**`) allowed; an empty 状态 line does not run on into the next line."""
    m = re.search(rf"^[ \t\-*]*\**(?:{keys})\**[ \t]*[:：]\**[ \t]*(.+)$", text, re.M | re.I)
    return m.group(1).strip().strip("*").strip() if m else ""


def is_done(status):
    return re.match(r"(已完成|完成|done)", status, re.I) is not None


@dataclass
class Progress:
    """The head of progress.md, as the agent wrote it."""

    status: str  # 状态 line, raw ('' when missing)
    next: str  # 下一步
    finished: str  # 完成时间, raw
    checks: dict  # {'done', 'doing', 'todo'}: counts of '- [x]', '- [~]', '- [ ]'
    mtime: float

    @classmethod
    def parse(cls, text, mtime):
        text = text.removeprefix("\ufeff")
        checks = CHECKBOX.findall(text)
        return cls(
            status=field_value(text, "状态|Status"),
            next=field_value(text, "下一步|Next"),
            finished=field_value(text, "完成时间|Finished"),
            checks={
                "done": sum(c in "xX" for c in checks),
                "doing": checks.count("~"),
                "todo": checks.count(" "),
            },
            mtime=mtime,
        )

    @property
    def done(self):
        return is_done(self.status)

    @property
    def finished_iso(self):
        """When the agent reported it done: the 完成时间 line, else the file's mtime."""
        return local_time_iso(self.finished) or iso(self.mtime)

    def to_json(self):
        return {
            "status": self.status,
            "next": self.next,
            "finished": self.finished,
            "checks": self.checks,
            "updated": iso(self.mtime),
            "done": self.done,
        }


def classify_progress(progress, review, approved_at=""):
    """The status an unmanaged task is counted as -> (status, detail). A done task with review: true
    stays in in_review until it is approved (approved_at in its frontmatter, ISO); if progress.md later reports a
    newer finish (you went on in the terminal), it is back in in_review."""
    if progress is None:
        return "draft", {}
    st = progress.status
    if re.match(r"(进行中|in progress|running)", st, re.I):
        return "running", {"kind": "unmanaged"}
    if re.match(r"(阻塞|blocked)", st, re.I):
        return "needs_you", {"kind": "unmanaged"}
    if progress.done:
        if review and not (approved_at and progress.finished_iso <= approved_at):
            return "in_review", {"kind": "review"}
        return "done", {}
    return "unknown", {"raw": st}


# ---------------------------------------------------------------- .agent_sys/state.json
#
# Written only by agent_sys, and only facts: what can be worked out from other files (why a queued task waits,
# whether a running one is working or waiting, that in_review means "read the report") is computed when the
# task is read (scan.derive_details), so it never goes stale. Schema version 1:
#
#   version               1
#   status                queued | running | needs_you | in_review | done | cancelled
#   detail                needs_you only: {"kind": ...} why the task stopped, one of:
#                           asked | questions | stopped | interrupted | wait_interrupted | budget | joined |
#                           checks_failed
#                         ({} for every other status)
#   since                 ISO time the task entered its status
#   session_id            Claude Code session, made when the state is created. The first run starts it with
#                         --session-id=<sid> (runs == 0), every later run uses --resume=<sid>
#   runs                  number of runs started (轮次)
#   run_id                the current run while running, else the last one; null before the first run
#   cost_usd              task cost so far = sum over runs of (cost at run end - cost at run start)
#   session_cost_usd      baseline: the session's cumulative total_cost_usd as of the last run's end. Claude
#                         Code reports the session total, across resumes, not the run's cost; the next run
#                         records it as its cost_start_usd and its cost = cost_end_usd - cost_start_usd
#   created               ISO, task created (for a joined task: its prompt file's mtime)
#   started               ISO, first run started ('' before; for a joined task: when work began in the terminal)
#   finished              ISO, last entered done / in_review ('' otherwise)
#   seen_at               ISO, the user last opened the task; done with seen_at < finished = unread (新完成)
#   moved_from            optional: the task id the session last knew, when the task was renamed or changed type
#                         after it ran; the next message to the session says where it went, then it is dropped

STATE_VERSION = 1
MANAGED_STATUSES = ("queued", "running", "needs_you", "in_review", "done", "cancelled")


def new_state(status, detail=None, *, created="", started="", finished="", seen_at=""):
    now = now_iso()
    return {
        "version": STATE_VERSION,
        "status": status,
        "detail": detail or {},
        "since": now,
        "session_id": str(uuid.uuid4()),
        "runs": 0,
        "run_id": None,
        "cost_usd": 0.0,
        "session_cost_usd": 0.0,
        "created": created or now,
        "started": started,
        "finished": finished,
        "seen_at": seen_at,
    }


def state_path(workspace):
    return workspace / ".agent_sys" / "state.json"


def runs_dir(workspace):
    return workspace / ".agent_sys" / "runs"


def read_state(workspace):
    """state.json as a dict, None if the task is not managed. Raises ValueError if it cannot be parsed."""
    path = state_path(workspace)
    if not path.is_file():
        return None
    try:
        state = json.loads(read_text(path))
    except json.JSONDecodeError as e:
        raise ValueError(_("state.json 无法解析：{error}", error=e)) from None
    if not isinstance(state, dict) or state.get("status") not in MANAGED_STATUSES:
        raise ValueError(
            _(
                "state.json 的状态「{status}」不认识",
                status=state.get("status") if isinstance(state, dict) else state,
            )
        )
    return state


def write_state(workspace, state):
    write_json(state_path(workspace), state)


# ---------------------------------------------------------------- .agent_sys/runs/<run_id>/run.json
#
# One run = one runner process (or one terminal takeover). The run directory is named after its start
# time, YYYYMMDD-HHMMSS (local time), and holds launch.json / system.md / launch.md / inbox.jsonl from the server
# (launch.py) and events.jsonl / run.json from the runner (runner.py). The runner writes run.json when it starts
# and when it ends. Schema version 1:
#
#   version           1
#   id                run directory name
#   kind              sdk (runner) | terminal (takeover in tmux) | replay (recorded events played back)
#   resume            false for the session's first run (--session-id), true after (--resume)
#   pid               the runner's pid (terminal: the wrapper script's). Before trusting or killing it, check
#                     /proc/<pid>/cmdline is still that process
#   host              hostname the run is on; a server on another host only shows it, never touches it
#   session_id        Claude Code session id
#   model             model the run was started with
#   window            tmux window, e.g. agent_sys:tp4 (terminal runs; attach with tmux -L $AGENT_SYS_TMUX attach -t <window>)
#   started, ended    ISO; ended is null while running
#   cost_start_usd    session total_cost_usd when the run started (state.json session_cost_usd)
#   cost_end_usd      session total_cost_usd at the end; null while running
#   exit              null while running | idle | stopped | budget | takeover | terminal_exit | error
#                     (a run that ended without an exit record is shown as interrupted)
#   exit_detail       free text, e.g. why the run ended in error
#   inbox_read        bytes of inbox.jsonl the runner had read when it ended (a message after that came too late)


def run_cost(rec, end):
    """What a run cost, given the session's total cost at its end (or as far as it went): Claude Code reports the
    session's total, so a run costs end - cost_start_usd. None if end is not known."""
    if not isinstance(end, (int, float)):
        return None
    return max(0.0, end - float(rec.get("cost_start_usd") or 0))


def cost_today(workspace):
    """Cost of the task's runs started today (the server's local time), from their end records."""
    runs = runs_dir(workspace).glob(datetime.now().strftime("%Y%m%d") + "-*/run.json")
    return sum(run_cost(rec, rec.get("cost_end_usd")) or 0.0 for rec in map(read_json, runs))


# ---------------------------------------------------------------- tasks, types, edges, projects


@dataclass
class Edge:
    up: str  # upstream task id (the one depended on)
    down: str  # downstream task id
    kind: str = "input"
    label: str = ""  # why (from depends_on's label)
    evidence: str = ""  # the prompt line naming the upstream's path
    source: str = "frontmatter"  # frontmatter (depends_on / parent of the downstream) | prompt (a path in its prompt)

    @property
    def blocking(self):
        return self.kind in BLOCKING

    def to_json(self):
        return {
            "from": self.up,
            "to": self.down,
            "kind": self.kind,
            "label": self.label,
            "evidence": self.evidence,
            "source": self.source,
            "blocking": self.blocking,
        }


@dataclass
class TaskType:
    id: str
    # defaults of its new tasks, from the skeleton's frontmatter: 完成后需要我审, and the checks as written
    review: bool = False
    checks: list = field(default_factory=list)
    skeleton: str = ""  # prompt skeleton (body of task_types/<type>.md)
    count: int = 0

    def to_json(self):
        return {
            "id": self.id,
            "review": self.review,
            "checks": plain(self.checks),
            "skeleton": self.skeleton,
            "count": self.count,
        }


@dataclass
class Task:
    project: str
    type: str
    name: str
    path: Path  # tasks/<type>/<name>.md
    workspace: Path  # workspaces/<type>/<name>/
    fm: dict  # frontmatter as read; unknown fields kept
    body: str  # prompt text after the frontmatter
    title: str = ""
    summary: str = ""
    review: bool = False
    parent: str | None = None  # frontmatter parent
    # effective: project.yaml agent defaults + frontmatter agent
    agent: dict = field(default_factory=dict)
    machines: list = field(default_factory=list)  # frontmatter machines: names from project.yaml
    checks: list = field(default_factory=list)  # frontmatter checks, as checks.parse gives them
    state: dict | None = None  # state.json; None = unmanaged
    progress: Progress | None = None
    status: str = "draft"
    detail: dict = field(default_factory=dict)
    since: str = ""
    created: str = ""
    started: str = ""
    updated: str = ""
    finished: str = ""
    report: str | None = None  # report file relative to the workspace, for done / in_review tasks
    handoff: list | None = None  # the report's handoff list (scan.handoff), None when it has none
    cost_today: float = 0.0  # managed: cost of the runs started today
    # running: what the current run has cost so far (not in state.json until it ends)
    cost_running: float = 0.0
    # why its frontmatter or checks cannot be read: it is not run (runs.check_can_run)
    invalid: str = ""
    # the latest run's checks.json (checks.summary), None if none ran
    check_result: dict | None = None
    # the task's result files in the ledger (ledger.task_briefs), None if none
    results: dict | None = None
    # done / in review: its input and code upstream tasks that changed after its last round started (upstream.changes)
    upstream_changed: list = field(default_factory=list)
    warnings: list = field(default_factory=list)  # 记录提醒 shown on the card
    hints: list = field(default_factory=list)  # shown only in the task detail

    @property
    def id(self):
        return f"{self.type}/{self.name}"

    @property
    def managed(self):
        return self.state is not None

    @property
    def unread(self):
        return (
            self.managed
            and self.status == "done"
            and (self.state.get("seen_at") or "") < (self.state.get("finished") or "")
        )

    def to_json(self):
        s = self.state or {}
        return {
            "id": self.id,
            "project": self.project,
            "type": self.type,
            "name": self.name,
            "title": self.title,
            "summary": self.summary,
            "managed": self.managed,
            "status": self.status,
            "detail": self.detail,
            "since": self.since,
            "unread": self.unread,
            "review": self.review,
            "parent": self.parent,
            "machines": self.machines,
            "checks": self.check_result,
            "results": self.results,
            "created": self.created,
            "started": self.started,
            "updated": self.updated,
            "finished": self.finished,
            "progress": self.progress and self.progress.to_json(),
            "report": self.report,
            "session_id": s.get("session_id"),
            "cost_usd": round((s.get("cost_usd") or 0) + self.cost_running, 6) if s else None,
            "budget_usd": plain(self.agent.get("budget_usd")) if s else None,
            "upstream_changed": self.upstream_changed,
            "warnings": self.warnings,
            "hints": self.hints,
        }

    def detail_json(self):
        """to_json() plus what the task detail needs from the files (the server adds upstream / downstream and
        runs). Paths are relative to AGENT_SYS_HOME, as everywhere in the API."""
        base = project_path(self.project)
        return {
            **self.to_json(),
            "frontmatter": plain(self.fm),
            "agent": plain(self.agent),
            "prompt": {
                "path": f"{base}/tasks/{self.id}.md",
                "body": self.body,
                "mtime": iso(mtime(self.path)),
            },
            "handoff": self.handoff,
            "workspace": f"{base}/workspaces/{self.id}",
        }


@dataclass
class Project:
    id: str
    path: Path
    config: dict  # project.yaml
    types: list[TaskType] = field(default_factory=list)
    tasks: dict[str, Task] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    # ledger.head: primary metric, baseline, best (the server fills it in)
    ledger: dict | None = None

    @property
    def title(self):
        return self.config.get("title") or self.id

    @property
    def archived(self):
        return bool(self.config.get("archived_at"))

    @property
    def template_review(self):
        """The tasks the template brought wait for you to review them: nothing starts by itself until you start the
        project or one of its tasks (ops.start_project)."""
        return self.config.get("template_review") is True

    @property
    def autorun(self):
        return self.config.get("autorun", True) is not False

    @property
    def max_running(self):
        return positive(self.config.get("max_running"), PROJECT_DEFAULTS["max_running"])

    def task(self, tid):
        if tid not in self.tasks:
            raise AgentSysError(_("任务「{task}」不存在", task=tid), 404)
        return self.tasks[tid]

    def upstream(self, tid):
        return [e for e in self.edges if e.down == tid]

    def downstream(self, tid):
        return [e for e in self.edges if e.up == tid]

    def satisfied(self, tid):
        """An upstream task lets its downstream start once it is done: managed by state.json; unmanaged when
        progress.md says 已完成, and with review: true only after it was approved."""
        return self.tasks[tid].status == "done"

    def children(self, tid):
        """The tasks whose parent is tid."""
        return [t.id for t in self.tasks.values() if t.parent == tid and t.id != tid]

    def counted_children(self, e):
        """For a children edge: the subtasks of its upstream it waits for. The waiting task itself and its own
        downstream do not count, or it would wait for itself."""
        after = reachable(blocking_graph(self.edges), e.down)
        return [c for c in self.children(e.up) if c not in after]

    def pending_children(self, e):
        """counted_children that are neither done nor cancelled (drafts too: a plan's draft is still part of it)."""
        return [
            c for c in self.counted_children(e) if self.tasks[c].status not in ("done", "cancelled")
        ]

    def edge_json(self, e):
        """Edge.to_json; a children edge adds {done, total} of the subtasks it waits for."""
        out = e.to_json()
        if e.kind == "children":
            total = len(self.counted_children(e))
            out["children"] = {"done": total - len(self.pending_children(e)), "total": total}
        return out

    def edge_satisfied(self, e):
        """Whether an edge lets its downstream start: its upstream is done, and for children all of its subtasks."""
        return self.satisfied(e.up) and not (e.kind == "children" and self.pending_children(e))

    def waiting_on(self, tid):
        """Blocking upstream tasks that are not done yet."""
        return [e.up for e in self.upstream(tid) if e.blocking and not self.satisfied(e.up)]

    def waiting_on_children(self, tid):
        """{upstream: [its pending subtasks]} of the children edges into tid whose upstream is done."""
        return {
            e.up: pending
            for e in self.upstream(tid)
            if e.kind == "children"
            and self.satisfied(e.up)
            and (pending := self.pending_children(e))
        }

    def counts(self):
        counts = dict.fromkeys(STATUSES, 0)
        for t in self.tasks.values():
            counts[t.status] += 1
        # running tasks that may be stuck (疑似卡住): they need you as much as 待确认
        stalled = sum(
            t.status == "running" and bool(t.detail.get("stall")) for t in self.tasks.values()
        )
        return {"total": len(self.tasks), **counts, "stalled": stalled}

    def running(self):
        """Managed tasks that are running: they hold a slot of max_running."""
        return sum(t.managed and t.status == "running" for t in self.tasks.values())

    def to_json(self):
        """ProjectSummary. The board adds types, tasks and edges."""
        return {
            "id": self.id,
            "title": self.title,
            "path": project_path(self.id),
            "template": self.config.get("template"),
            "autorun": self.autorun,
            "max_running": self.max_running,
            "archived": self.archived,
            "archived_at": plain(self.config.get("archived_at")),
            "template_review": self.template_review,
            "goal": str(plain(self.config.get("goal")) or ""),
            "ledger": self.ledger,
            "agent": project_agent(self.config),
            "counts": self.counts(),
            "running": self.running(),
            "cost_today_usd": round(sum(t.cost_today for t in self.tasks.values()), 4),
        }


# ---------------------------------------------------------------- blocking edges as a graph


def blocking_graph(edges):
    """upstream -> [downstream] along the blocking edges (input, code, children)."""
    graph = {}
    for e in edges:
        if e.blocking:
            graph.setdefault(e.up, []).append(e.down)
    return graph


def reachable(graph, start):
    """Every task reached from start in a blocking_graph(), start included."""
    seen, todo = {start}, [start]
    while todo:
        for nxt in graph.get(todo.pop(), ()):
            if nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return seen


def cycle_tasks(edges):
    """Tasks on a cycle of blocking edges: the strongly connected components with more than one task
    (Kosaraju, O(V + E)). Such tasks wait for each other forever."""
    graph, back = blocking_graph(edges), {}
    for up, downs in graph.items():
        for down in downs:
            back.setdefault(down, []).append(up)
    order, seen = [], set()  # tasks by DFS finish time
    for root in graph:
        if root in seen:
            continue
        seen.add(root)
        stack = [(root, iter(graph.get(root, ())))]
        while stack:
            tid, it = stack[-1]
            nxt = next((n for n in it if n not in seen), None)
            if nxt is None:
                order.append(stack.pop()[0])
            else:
                seen.add(nxt)
                stack.append((nxt, iter(graph.get(nxt, ()))))
    on_cycle, done = set(), set()
    for root in reversed(order):  # each walk back along the edges from here is one component
        if root in done:
            continue
        done.add(root)
        comp, todo = [root], [root]
        while todo:
            for n in back.get(todo.pop(), ()):
                if n not in done:
                    done.add(n)
                    comp.append(n)
                    todo.append(n)
        if len(comp) > 1:
            on_cycle.update(comp)
    return on_cycle
