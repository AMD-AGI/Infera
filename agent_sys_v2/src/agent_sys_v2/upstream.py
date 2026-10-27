"""上游已更新: what a task's blocking upstream tasks looked like when its last round started, and what changed since.

launch.json records, per upstream task of an input or code dependency (snapshot):

  {"develop/a": {"status": "done", "report": "report.md", "report_sha256": "…",
                 "files": {"results/x.json": [size, mtime_ns], "log/notes.md": [size, mtime_ns]}}}

files are results/'s own files and the files the report's handoff list names (not a directory: a code dependency's
checkout changes with whatever the downstream builds in it). The scan compares that with the upstream as it is now, for the
tasks done or in review (changes), and the card and the task detail say so, with a message to send the task to check
its conclusion against the new results. Nothing else follows: the task stays done, no downstream is reopened (v1
learned that a version kept per input is enough and invalidating is not).

Digests are cached by path, mtime and size, launch.json by path and mtime: a scan reads what changed only.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from . import i18n
from .i18n import status_word, translate
from .model import read_json, runs_dir

KINDS = ("input", "code")  # the blocking dependencies whose results the task worked from
CACHE_MAX = 10000
_sha_cache, _launch_cache = {}, {}


def _stat(path):
    try:
        st = os.stat(path)
        return st.st_size, st.st_mtime_ns
    except OSError:
        return None


def sha256(path):
    """A file's sha256, cached by (path, size, mtime); '' when it cannot be read."""
    st = _stat(path)
    if st is None:
        return ""
    key = (str(path), *st)
    if key not in _sha_cache:
        if len(_sha_cache) > CACHE_MAX:  # old versions of files pile up in a long-running server
            _sha_cache.clear()
        try:
            _sha_cache[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            return ""
    return _sha_cache[key]


def handoff_paths(up, report):
    """{path as written: path} of the report's handoff items that name a file that is there (a directory changes with
    whatever is built in it); results/ and the report itself are left to their own digests. scan.handoff found them
    (whether each is there, where in the workspace): nothing is resolved again (a resolve is a stat per directory,
    slow on NFS)."""
    from .scan import handoff  # scan imports this module

    ws = up.workspace
    items = up.handoff if up.report == report and up.handoff is not None else handoff(ws, report)
    out = {}
    for h in items or []:
        if not h["path"] or h["kind"] != "file":
            continue
        rel = h["rel"]
        if rel == report or rel.startswith("results/"):
            continue
        base = Path(h["path"]).expanduser()
        if rel:
            out[h["path"]] = ws / rel
        elif base.is_absolute():
            out[h["path"]] = base
        else:  # outside the workspace: as scan.handoff looked, the workspace first (a link out of it), then the project
            out[h["path"]] = ws / base if _stat(ws / base) else ws.parents[2] / base
    return out


def digest(up):
    """The upstream task as it is now: status, main report and its sha256, and its files' [size, mtime_ns]."""
    from .scan import main_report  # scan imports this module

    ws = up.workspace
    report = up.report or main_report(ws)
    files = {}
    try:
        entries = list(os.scandir(ws / "results"))
    except OSError:
        entries = []
    for e in entries:
        if e.name.startswith("."):  # an editor's or NFS's own file
            continue
        try:
            if e.is_file():
                st = e.stat()
                files[f"results/{e.name}"] = [st.st_size, st.st_mtime_ns]
        except OSError:  # gone since the listing
            pass
    if report:
        for name, path in handoff_paths(up, report).items():
            if st := _stat(path):
                files[name] = list(st)
    return {
        "status": up.status,
        "report": report or "",
        "report_sha256": sha256(ws / report) if report else "",
        "files": files,
    }


def snapshot(project, task):
    """What launch.json records: digest() of every upstream task of an input or code dependency."""
    return {e.up: digest(project.tasks[e.up]) for e in project.upstream(task.id) if e.kind in KINDS}


def recorded(task):
    """The snapshot of the task's last round: launch.json's upstream and started; None for a run from before."""
    run_id = (task.state or {}).get("run_id")
    if not run_id:
        return None
    path = runs_dir(task.workspace) / run_id / "launch.json"
    st = _stat(path)
    if st is None:
        return None
    key = str(path)
    if _launch_cache.get(key, (None,))[0] != st:
        try:
            cfg = read_json(path)
        except (OSError, ValueError):  # cannot be read: nothing to compare with
            cfg = {}
        up = cfg.get("upstream")
        _launch_cache[key] = (st, (up, cfg.get("started") or "") if isinstance(up, dict) else None)
    return _launch_cache[key][1]


def well_formed(then):
    """A recorded digest as digest() writes it (launch.json is in the workspace: anything may be in it)."""
    if not isinstance(then, dict):
        return False
    files = then.get("files")
    return (
        isinstance(then.get("status"), str)
        and isinstance(files, dict)
        and all(isinstance(v, list) and len(v) == 2 for v in files.values())
    )


def compare(then, now):
    """What changed between two digests: {status: [then, now] | None, report: bool, added, removed, changed}; None
    when nothing did."""
    a, b = then.get("files") or {}, now["files"]
    out = {
        "status": [then.get("status"), now["status"]]
        if then.get("status") != now["status"]
        else None,
        "report": (then.get("report"), then.get("report_sha256"))
        != (now["report"], now["report_sha256"]),
        "added": sorted(set(b) - set(a)),
        "removed": sorted(set(a) - set(b)),
        "changed": sorted(f for f in set(a) & set(b) if list(a[f]) != list(b[f])),
    }
    return (
        out
        if out["status"] or out["report"] or out["added"] or out["removed"] or out["changed"]
        else None
    )


def changes(project):
    """Set t.upstream_changed on every managed task done or in review whose blocking upstream changed after its last
    round started: [{task, title, since, status, report, added, removed, changed, text}], text in the UI's language
    (made at scan time, as the cards' reminders)."""
    digests = {}  # an upstream of several tasks is read once
    for t in project.tasks.values():
        if not t.managed or t.status not in ("done", "in_review"):
            continue
        rec = recorded(t)
        if not rec:
            continue
        ups, since = rec
        out = []
        for tid, then in ups.items():
            up = project.tasks.get(tid)
            if up is None or not well_formed(then):  # renamed or deleted since; or not ours
                continue
            if tid not in digests:
                digests[tid] = digest(up)
            if diff := compare(then, digests[tid]):
                change = {"task": tid, "title": up.title, "since": since, **diff}
                out.append({**change, "text": notice(change, i18n.LANG)})
        t.upstream_changed = out


LIST = 5  # file names shown per kind of change


def names(files, lang):
    shown = translate("、", lang).join(files[:LIST])
    return shown + (translate("等 {n} 个", lang, n=len(files)) if len(files) > LIST else "")


def describe(change, lang):
    """What changed in one upstream task, as one line."""
    parts = []
    if change["status"]:
        then, now = change["status"]
        parts.append(
            translate(
                "状态从「{then}」变为「{now}」",
                lang,
                then=status_word(then, lang),
                now=status_word(now, lang),
            )
        )
    if change["report"]:
        parts.append(translate("报告有改动", lang))
    if change["added"]:
        parts.append(translate("新增 {files}", lang, files=names(change["added"], lang)))
    if change["removed"]:
        parts.append(translate("删除了 {files}", lang, files=names(change["removed"], lang)))
    if change["changed"]:
        parts.append(translate("{files} 有改动", lang, files=names(change["changed"], lang)))
    return translate("；", lang).join(parts)


def notice(change, lang):
    """The card's and the detail's line: 上游 X 在本任务开始后更新了：……"""
    return translate(
        "上游「{title}」在本任务开始后更新了：{what}",
        lang,
        title=change["title"],
        what=describe(change, lang),
    )


def message(task, lang):
    """The message to send the task, in the agent's language: check the conclusion against the new results."""
    lines = [
        translate("你开始本任务之后，上游任务更新了：", lang),
        *(
            translate(
                "- {title}（`workspaces/{task}/`）：{what}",
                lang,
                title=c["title"],
                task=c["task"],
                what=describe(c, lang),
            )
            for c in task.upstream_changed
        ),
        "",
        translate(
            "请读它们新的报告和交接清单，按新结果复核本任务的结论。结论不变就简短说明理由；需要改就更新 report.md。",
            lang,
        ),
    ]
    return "\n".join(lines)
