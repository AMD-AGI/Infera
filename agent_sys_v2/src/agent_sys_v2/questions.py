"""questions.md answered as a form. When the agent has several questions it
writes them as a Markdown table, `# | 问题 | 建议 | 你的意见` (`# | Question | Suggestion | Your answer` under the
English protocol), the last column left for you. The first table with that column is read as rows; your answers go
into it (| escaped, line breaks written as <br>) and the agent is told to go on in the same session. Only this
round's questions.md counts (round_questions): once you answered it, the next round begins and the table is an
old one.
"""

from __future__ import annotations

import re

from . import launch, runs
from .i18n import _, status_word
from .model import AgentSysError, iso, mtime, read_text, runs_dir, write_text

ANSWER = "你的意见"
# the protocol's column in Chinese and in English
ANSWER_COLUMN = re.compile(r"你的意见|your answer", re.I)
ROW = re.compile(r"\s*\|")
RULE = re.compile(r"\s*\|?[\s:|-]*-[\s:|-]*")  # |---|:--| under the header
PIPE = re.compile(r"(?<!\\)\|")
BR = re.compile(r"<br\s*/?>", re.I)


def spans(line):
    """Where the cells of a table row are in it, [(start, end)]: between the pipes that are not escaped, the outer
    ones dropped."""
    body = line.rstrip()
    edges = [m.start() for m in PIPE.finditer(body)]
    start = len(body) - len(body.lstrip())
    if not edges or edges[0] != start:
        edges.insert(0, start - 1)
    if len(edges) == 1 or edges[-1] != len(body) - 1:
        edges.append(len(body))
    return [(a + 1, b) for a, b in zip(edges, edges[1:])]


def cells(line):
    return [line[a:b].strip() for a, b in spans(line)]


def unescape(cell):
    return BR.sub("\n", cell).replace("\\|", "|")


def escape(text):
    return "<br>".join(line.strip().replace("|", "\\|") for line in text.strip().splitlines())


def parse(text):
    """The first table that has a 你的意见 column, or None: {columns, answer_column, rows, before, after, lines}.
    A row is {key, cells, answer, line, span}; key is its first cell (Q1, 1, …), or #<n> where that is empty or
    repeated; span: where its answer is in the line (None if the row stops before it)."""
    lines = text.split("\n")
    for i in range(len(lines) - 1):
        if not ROW.match(lines[i]) or not RULE.fullmatch(lines[i + 1]):
            continue
        columns = cells(lines[i])
        col = next((j for j, c in enumerate(columns) if ANSWER_COLUMN.search(c)), None)
        if col is None:
            continue
        end = i + 2
        while end < len(lines) and ROW.match(lines[end]):
            end += 1
        rows, keys = [], set()
        for n, line in enumerate(lines[i + 2 : end]):
            where = spans(line)
            c = [line[a:b].strip() for a, b in where]
            span = where[col] if col < len(where) else None
            c += [""] * (len(columns) - len(c))
            if len(c) > len(columns) and 0 < col == len(columns) - 1:
                # a | left unescaped before the answer, as in `grep x | wc -l`: back into the cell before it (as
                # shown; the line keeps it as written)
                c[col - 1 :] = [" \\| ".join(c[col - 1 : -1]), c[-1]]
                span = where[-1]
            key = c[0] if c[0] and c[0] not in keys else f"#{n + 1}"
            keys.add(key)
            rows.append(
                {
                    "key": key,
                    "cells": c,
                    "answer": unescape(c[col]),
                    "line": i + 2 + n,
                    "span": span,
                }
            )
        return {
            "columns": columns,
            "answer_column": col,
            "rows": rows,
            "before": "\n".join(lines[:i]).strip("\n"),
            "after": "\n".join(lines[end:]).strip("\n"),
            "lines": lines,
        }
    return None


def fill(text, answers):
    """text with the answers written into its table -> (new text, rows answered, rows). answers: {key: text}; the
    rows not named keep what they had. Only the answers that changed are rewritten; the rest of the line stays as
    the agent wrote it."""
    table = parse(text)
    unknown = set(answers) - {r["key"] for r in table["rows"]}
    if unknown:
        raise AgentSysError(
            _(
                "questions.md 中没有「{rows}」这一行，请刷新后重新填写",
                rows=_("、").join(sorted(unknown)),
            )
        )
    lines, col = table["lines"], table["answer_column"]
    filled = 0
    for r in table["rows"]:
        answer = str(answers.get(r["key"], r["answer"]) or "")
        filled += bool(answer.strip())
        if answer.strip() != r["answer"].strip():
            line = lines[r["line"]]
            if r["span"]:
                a, b = r["span"]
                lines[r["line"]] = f"{line[:a]} {escape(answer)} {line[b:]}"
            else:  # the row stops before its answer cell
                c = list(r["cells"])
                c[col] = escape(answer)
                lines[r["line"]] = f"| {' | '.join(c)} |"
    return "\n".join(lines), filled, len(table["rows"])


def round_start(t):
    """When the task's current round began: the time its run's launch.json was written, to the sub-second. (By the
    second, a questions.md you answered right before a new run started would still count as this round's.) None
    before the first run."""
    run_id = (t.state or {}).get("run_id")
    return mtime(runs_dir(t.workspace) / run_id / "launch.json") if run_id else None


def round_questions(t):
    """questions.md written in this round (its mtime is after the round started): {path, count, mtime}, else None.
    count: the rows of its table, as the form shows them (0 if it has none). Once you answered it and the next round
    began, it is an old one."""
    path = t.workspace / "questions.md"
    m, since = mtime(path), round_start(t)
    if not m or not since or m <= since:
        return None
    table = parse(read_text(path))
    return {"path": "questions.md", "count": len(table["rows"]) if table else 0, "mtime": iso(m)}


def read(t):
    """This round's questions.md as a table (GET …/questions). 404 if the round wrote none, 422 if it has no table
    with a 你的意见 column."""
    q = round_questions(t)
    if not q:
        raise AgentSysError(_("本轮没有 questions.md"), 404)
    path = t.workspace / q["path"]
    table = parse(read_text(path))
    if table is None:
        raise AgentSysError(
            _("{path} 中没有含「{column}」列的表格，请查看原文", path=q["path"], column=ANSWER), 422
        )
    return {
        "path": q["path"],
        "mtime": iso(mtime(path)),
        "columns": table["columns"],
        "answer_column": table["answer_column"],
        "before": table["before"],
        "after": table["after"],
        "rows": [{k: r[k] for k in ("key", "cells", "answer")} for r in table["rows"]],
    }


def submit(home, pdir, tid, answers, base_mtime=""):
    """提交回答 (POST …/questions): write the answers into this round's questions.md and tell the agent, in the same
    session. base_mtime is the file's mtime when you started filling it in: if it changed since, nothing is written."""
    if not isinstance(answers, dict) or not all(isinstance(v, str) for v in answers.values()):
        raise AgentSysError(_("answers 应为「行: 回答」的键值表"))
    project, t = runs.fresh(pdir, tid)
    runs.check_can_run(project, t)
    if t.status != "needs_you":
        raise AgentSysError(
            _("任务当前为「{status}」，只有待确认的任务可以提交回答", status=status_word(t.status)),
            409,
        )
    # before the answers are written: a run that is refused would leave them unread
    runs.check_can_start(home, project, t)
    q = read(t)
    if base_mtime and base_mtime != q["mtime"]:
        raise AgentSysError(_("{path} 在你填写期间已被修改，请刷新后重新填写", path=q["path"]), 409)
    path = t.workspace / q["path"]
    text, filled, total = fill(read_text(path), answers)
    write_text(path, text)
    runs.message(
        home,
        pdir,
        tid,
        launch.say_for(
            home,
            "已在 {path} 中回答（已填写 {filled} / {total} 项），未填写的项请按你的建议处理。",
            path=q["path"],
            filled=filled,
            total=total,
        ),
    )
    return {"saved": q["path"], "delivery": "resume"}
