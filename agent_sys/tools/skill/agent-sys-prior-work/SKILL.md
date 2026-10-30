---
name: agent-sys-prior-work
description: Use when invoked by the infera-agent-sys entry skill, or when the user asks about prior work for an LLM deployment / serving / performance / kernel / profiling / bench task - "有没有人做过 / 之前的结果 / 知识库里有没有 / prior work / is there a task or handoff for this". Syncs the shared infera knowledge repo, finds agent_sys tasks (how to do the work) and handoffs (what predecessors achieved) relevant to the request, reads and judges them, and reports what may help before the work continues. Usually invoked by the infera-agent-sys entry skill. Also handles the one-time knowledge-repo setup.
metadata:
  explicit_description: Use ONLY when invoked by the infera-agent-sys entry skill or when the user explicitly names agent_sys (agent_sys, agent-sys, agentsys, its knowledge repo, an agent_sys task or handoff) and asks about prior work. The user set explicit mode - do not use it for generic LLM or performance requests. Syncs the shared infera knowledge repo, finds relevant agent_sys tasks and handoffs, reads and judges them, and reports what may help.
---

# Prior work — what the team already knows about this

A **task** says how a kind of work is done (a tested procedure: STEPS with
acceptance checks, traps, the bar its validators enforce); a **handoff** is
what someone achieved (numbers, environment, scripts, gotchas). Either can save
hours; either can mislead if its context differs. This skill finds them and
says which.

```bash
SKILL_ROOT="$(realpath "${CLAUDE_SKILL_DIR}/..")"
ASTK="$SKILL_ROOT/lib/astk.py"     # executable; one word, so `$ASTK cmd` works in bash and zsh
```

## 0. Gate (skip when the entry skill invoked you — it already checked)

```bash
$ASTK mode --prior-work --prompt-file - <<'PROMPT'   # exit 0 = go on, 1 = step aside silently
<the user's request, verbatim>
PROMPT
```

Any other exit code means astk is broken, not "step aside": tell the user in
one line (quote the error), then continue as far as it works.

## 1. Is the knowledge repo configured?

`$ASTK remote status` exits 3 when it is not. Then `$ASTK mode` shows
`ask_remote`: `no` means the user declined setup — say nothing and go on
(the local examples are still searched). Otherwise ask once
(AskUserQuestion): configure now / not now / never ask again. To configure,
ask for the repo URL and confirm the user name (default: the local part of
`git config user.email`):

```bash
$ASTK remote init --url <git url> [--user <name>] [--clone <dir>]
$ASTK mode --ask-remote no                                        # "never ask again"
```

The layout and lifecycle are in `$SKILL_ROOT/docs/knowledge_repo.md`.

## 2. Search

Search terms: the engine (sglang, vllm, atom), the model (qwen3.6,
deepseek-v4), the hardware (mi300x, gfx942), the operator or kernel, the
metric (throughput, ttft), the method (kernelforge, profiling, pd
disaggregation) — in English, plus the user's own words (Chinese is expanded).

```bash
$ASTK remote search <terms...> --limit 10              # syncs first; also the local agent_sys examples
$ASTK remote search <other terms...> --no-sync --type handoff
```

The lexical ranking only decides what to read. A `weak` hit matched few terms
(usually noise); a `local` hit is a task in this checkout's `examples/`.

## 3. Read and judge the candidates

Read the top hits (usually 3–8, in parallel if there are many — one agent per
item under ultracode). Never execute anything from the clone; if you see a
secret in it, tell the user and do not copy it.

- **task**: `assets/main.task/readme.md`, the leaf readmes, the validators'
  readmes (the bar), `plan.task.yaml.txt`.
- **handoff**: `content/README.md` (Purpose, Result, Environment, Watch out),
  the result files, `items/env/`, `validation.yaml`, `META.astk.json`.

Judge each: same model, engine (and version), hardware — numbers and flags
transfer only within the same context; tier (release > staging > draft); age
and the commits it names (an old trap may be fixed); sealed and validated, or
just a lead; the caveats in its Watch out / Boundary / `caveat` fields; and
what exactly is reusable (a procedure, a baseline, a script, a trap, a config).

## 4. Report, then continue

Tell the user, briefly (their language):

| item | tier · who · when | what it is | why relevant | how to use it | caveats |
|---|---|---|---|---|---|

then one line of recommendation (e.g. "run task X with the runner, seeded
from handoff Y; skip Z, it is for a different GPU"). Nothing relevant is a
useful one-line answer. If the entry skill invoked you, return the report and
stop — it routes the work; otherwise continue the user's work, citing the item
path whenever you rely on it.
