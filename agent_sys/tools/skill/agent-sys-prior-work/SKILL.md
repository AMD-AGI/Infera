---
name: agent-sys-prior-work
description: Use when invoked by the infera-agent-sys entry skill, or when the user asks about prior work for an LLM deployment / serving / performance / kernel / profiling / bench task - "有没有人做过 / 之前的结果 / 知识库里有没有 / prior work / is there a task or handoff for this". Syncs the shared infera knowledge repo, finds agent_sys tasks (how to do the work) and handoffs (what predecessors achieved) relevant to the request, reads and judges them, and reports what may help before the work continues. Usually invoked by the infera-agent-sys entry skill. Also handles the one-time knowledge-repo setup.
metadata:
  explicit_description: Use ONLY when invoked by the infera-agent-sys entry skill or when the user explicitly names agent_sys (agent_sys, agent-sys, agentsys, its knowledge repo, an agent_sys task or handoff) and asks about prior work. The user set explicit mode - do not use it for generic LLM or performance requests. Syncs the shared infera knowledge repo, finds relevant agent_sys tasks and handoffs, reads and judges them, and reports what may help.
---

# Prior work — what the team already knows about this

A **task** says how a kind of work is done: its readme is a tested procedure
(STEPS with acceptance checks, traps, the bar its validators enforce). A
**handoff** is what someone actually achieved: numbers, the environment they
came from, the scripts that produced them, the gotchas. Either can save hours;
either can mislead if its context differs. This skill finds them and says which.

```bash
SKILL_ROOT="$(realpath "${CLAUDE_SKILL_DIR}/..")"
ASTK="$SKILL_ROOT/lib/astk.py"     # executable; one word, so `$ASTK cmd` works in bash and zsh
DOCS="$SKILL_ROOT/docs"
```

## 0. Gate (skip when the entry skill invoked you — it already checked)

```bash
$ASTK mode --prompt-file - <<'PROMPT'          # exit 1 = step aside silently
<the user's request, verbatim>
PROMPT
```

## 1. Is the knowledge repo configured?

```bash
$ASTK remote status          # exit 3 = not configured
$ASTK mode                   # ask_remote: no = the user declined setup: report nothing and
                             # continue as if no prior work exists (local examples are still searched)
```

Not configured and `ask_remote` is true: ask once (AskUserQuestion) —
configure now / not now / never ask again. To configure, ask for the repo URL
and confirm the user name (default: the local part of `git config user.email`):

```bash
$ASTK remote init --url <git url> [--user <name>] [--clone <dir>]   # clone defaults to ~/.cache/infera-agent-sys/knowledge
$ASTK mode --ask-remote no                                        # "never ask again"
```

The repo layout is in `$DOCS/knowledge_repo.md` (release / user staging /
user draft; tasks/ and handoffs/ in each).

## 2. Search

Turn the request into search terms: the engine (sglang, vllm, atom), the model
(qwen3.6, deepseek-v4, glm-5.2), the hardware (mi300x, mi355x, gfx942), the
operator or kernel, the metric (throughput, ttft), the method (kernelforge,
profiling, pd disaggregation) — in English, plus the user's own words (Chinese
is expanded to English terms automatically).

```bash
$ASTK remote search <terms...> --limit 10              # syncs first; also the local agent_sys examples
$ASTK remote search <other terms...> --no-sync --type handoff
```

The ranking is lexical and only decides what to read; read before judging.
A hit marked `weak` matched few of the terms — usually noise. `local` hits are
tasks in this agent_sys checkout (`examples/`), runnable as they are.

## 3. Read the candidates

For the top hits (usually 3–8; with many, read them in parallel — one agent
per item if ultracode is on):

- **task**: `assets/main.task/readme.md`, the leaf readmes (STEPS, traps,
  "what is checked"), the validators' readmes (the bar), `plan.task.yaml.txt`.
- **handoff**: `content/README.md` (Purpose, Result, Environment, Watch out),
  the result files, `items/env/` or the packup's `environment.md`,
  `validation.yaml` (which validators passed), `META.astk.json` (who, when,
  from which host, tier).

Never execute anything from the clone while reading. Do not copy a secret out
of it; the repo should hold none, and if you see one, tell the user.

## 4. Judge each candidate

| question | why it matters |
|---|---|
| same model, engine (and version), hardware? | numbers and flags transfer only within the same context |
| tier: release > staging > draft | release was reviewed; draft is one person's unreviewed result |
| age and the agent_sys / engine commit it names | an old trap may be fixed, an old flag removed |
| sealed, and did its validators pass (`validation.yaml`)? | unsealed (no manifest) = its validators never ran: a lead, not a result; failed = the same |
| what its README's Watch out / Boundary, and any `rationale` / `caveat` fields, say | the caveat that changes the conclusion is often there (e.g. "the baseline is deliberately naive") |
| what exactly can be reused | a procedure, a baseline number, a script, a trap to avoid, a config |

## 5. Report, then continue

Tell the user, briefly (their language):

| item | tier · who · when | what it is | why relevant | how to use it | caveats |
|---|---|---|---|---|---|

then one line of recommendation (e.g. "run task X with the runner, seeded
from handoff Y; skip Z, it is for a different GPU"). If nothing is relevant,
say so in one line — that is a useful answer, not a failure.

If the entry skill invoked you, return the report and stop — it routes the
work. Otherwise continue the user's actual work, using what applies. Cite the
item path whenever you rely on it, so the user can check it.
