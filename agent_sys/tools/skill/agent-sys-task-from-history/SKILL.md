---
name: agent-sys-task-from-history
description: Use when the user wants Claude's past work (this session, earlier sessions, a date range, a topic in the history) turned into an agent_sys task or task package — "把这段工作总结成 task / 把历史沉淀成 agent_sys 任务 / make this reusable as a task / capture what we did as a task with handoff and validators". Produces as few tasks as possible (prefer one), each a large decoupled module, with handoff definitions and calibrated validators, in agent_sys's real format. NOT for packaging a result as a handoff (that is agent-sys-handoff-packup) or running a task (agent-sys-task-runner).
---

# Task from history

Turn work Claude already did into an agent_sys task package that an agent can
re-run. The deliverable is **few, large, decoupled tasks** — prefer exactly one.
Small steps stay inside a task's readme: an agent (especially with workflows)
handles those by itself; splitting them into tasks only adds handoffs to break.

## Setup

```bash
ASTK="$(realpath "${CLAUDE_SKILL_DIR}")/../lib/astk.py"   # executable; one word, so bash and zsh both run `$ASTK cmd`
DOCS="$(realpath "${CLAUDE_SKILL_DIR}")/../docs"
```

Each Bash call is a fresh shell: repeat these two lines (or use the full path).
A package with required `${VAR}`s needs the same `--var K=V` on **every**
command that loads it; `astk` names the missing ones.

Read `$DOCS/primer.md` (agent_sys in one page) and `$DOCS/calibration.md`
before writing any YAML. Use only fields they name — agent_sys's schemas
reject unknown keys.

## 1. Pin the scope

Ask (AskUserQuestion) only what the request leaves open: which work, which
sessions or dates, what the task must achieve when re-run. To show the options:

```bash
$ASTK history list [--since 2026-10-14]  # sessions of this project: time, size, title, first prompt
$ASTK history list --cwd <other dir>     # sessions that ran elsewhere (the dir Claude was started in)
```

## 2. Read the history — the digest, not the raw JSONL

```bash
$ASTK history digest --session <id-prefix|latest|all> [--since 2026-10-14T00] [--grep <topic>] \
    --out <workdir>/digest.md --split-chars 200000
```

The digest keeps human prompts (incl. corrections), compaction summaries,
assistant conclusions, every shell command and every file written; secrets are
redacted. Read all of it. If it is more than ~3 parts, summarise the parts in
parallel (Workflow if ultracode is on, else Agent) into the same five headings,
then merge:

1. **Goal and success criteria** as the user stated them (quote the user).
2. **What finally worked** — the commands, scripts, configs, in order.
3. **Traps** — what failed and why, and the fix (these become the readme's traps).
4. **Artefacts** — files produced, where they are, what consumes them.
5. **Environment** — machines, images, paths, versions, secrets *by name*.

Also read what the history points at: the repo diff (`git log`/`git diff`),
the project `CLAUDE.md`, memory files, and the final artefacts themselves.
A digest line is a pointer, not evidence — open the file it names. Two places
the digest does not cover: Workflow scripts are written beside `--out` as
`*.workflow-N.js` (the method, when the work was a workflow), and subagent /
workflow **results** live under `<session>/subagents/` (the digest names the
dir). Text produced by subagents may start with a `[harness: …]` annotation —
strip it before quoting or packing that text.

## 3. Decide the tasks (prefer ONE)

Split only where a boundary is real. A boundary is real when **one** holds:

- the artefact on it is consumed on its own (by a person, another flow, a later run);
- the two sides need different executors (a deterministic script vs. judgement);
- the left side is expensive (GPU-hours) and worth checkpointing so the right
  side can be retried alone.

Otherwise it is a step, and it goes in the readme's STEPS. For each task fix:
inputs, outputs (handoff kinds), executor (`ai` or `program`), and the
content type of each kind (`$DOCS/primer.md` table): a runnable result with
numbers → `reproducible`; code/a kit/a packup tree → `code`; a JSON/YAML record
→ `structured_text`; prose → `text`. A scalar input (a topic, a model path,
a node name) is a **variable** (`${name}` in the task's `env:`), not a kind; it
becomes a kind only when a person or a later run consumes it as an artefact.

Show the proposal as a table (task, executor, inputs → outputs, why this
boundary exists) and get the user's yes before writing.

## 4. Ask where it goes — three suggestions

Ask with AskUserQuestion, three options plus "Other", each with its reason:

1. **Beside the work** — `<repo of the work>/agent_tasks/<name>/`: versioned
   with the code it drives. (`agent-sys run` needs the package inside a git repo.)
2. **As an agent_sys example/test** — `<infera>/agent_sys/examples/<NN_name>/`
   or `agent_sys/tests/e2e_test/<case>/`: when it teaches or guards agent_sys.
3. **Personal scratch** — `<scratch disk>/<user>/agent_sys_tasks/<name>/`
   (e.g. `/var/tmp/$USER/...`): a draft not yet worth a commit. Runnable with
   `agent-sys-task-runner`; `agent-sys run` needs it inside a git repo.

Never write into a directory that holds other content without saying so.

## 5. Write the plan, then scaffold

Write `<dest>.plan.yaml` (format and a full example:
`references/plan_format.md`), then:

```bash
$ASTK scaffold <dest>.plan.yaml <dest> [--var K=V ...]   # writes the package, then runs agent_sys `show`
```

Re-run with `--force` after editing the plan; it refuses (writing nothing)
when a renamed or removed task left generated files behind — remove those
yourself, scaffold never deletes. The plan is the source of truth: calibration decisions go in its `calibration:` fields (rendered into
the validator readmes), not into generated files, which `--force` rewrites.

`scaffold` derives what agent_sys requires and the plan need not repeat: a
non-leaf `main`, grants, subgraph `froms`, one generic validator per kind
(`assets/lib/handoff_check.py`, configured by `args`), and a `publish_inputs`
leaf for any input no task produces (given a `fixture`). It refuses a plan that
would not load and prints `show`'s verdict. Exit 0 = agent_sys accepts it.

**The readme is the task.** For an AI task it is the whole prompt; write it
from the history using `references/readme_template.md`: what you are given,
success stated so it can fail, STEPS (ordered commands, each with its
acceptance), traps (from §2.3, with the symptom and the fix), what is checked,
where to write each output. Copy scripts that worked into
`assets/<task>.task/` and call them from STEPS by
`$AGENT_SYS_TASK_PACKAGE/assets/<task>.task/<script>`. Program tasks only where
a script already does the whole job deterministically.

## 6. Calibrate the validators — prove it both ways

Rules: `$DOCS/calibration.md`. Build every control in a scratch dir, never
inside the repo or the history's own result dirs.

Write the package's `--var K=V` flags out in each command (`[--var ...]`
below); a shell variable holding several flags does not split in zsh.
Exit 1 is a refusal (FAIL); exit 2 is a usage or load error — not a FAIL.

```bash
# positive: the real result from the history, packed as a content dir
$ASTK template --package <dest> --kind <kind> --out <tmp>/positive [--var ...]   # skeleton + "what to write" hint
cp -a <the real result files> <tmp>/positive/items/...                          # verbatim; fill README sections
$ASTK check --package <dest> --kind <kind> --content <tmp>/positive [--var ...]  # must PASS (exit 0)
# negatives
$ASTK template --package <dest> --kind <kind> --out <tmp>/skeleton [--var ...]
$ASTK check --package <dest> --kind <kind> --content <tmp>/skeleton [--var ...]  # must FAIL (exit 1)
mkdir -p <tmp>/empty && $ASTK check --package <dest> --kind <kind> --content <tmp>/empty [--var ...]   # must FAIL (exit 1)
```

Add a negative that targets the bar itself when you can (a fabricated or
truncated result) — a validator that only refuses empty dirs is not checking
the work. If the history left no result to pack, say so and run only the
negatives. When the positive control fails, decide whether the bar or the
result is wrong, fix that one, and write the decision into the plan's
`calibration:` field. Add a hand-written validator (same body contract,
`references/plan_format.md` §custom) only for a check the generic one cannot
express — e.g. re-running a command and comparing within a stated tolerance.
Label it `weak` unless it can state in advance the number that decides it
(`$DOCS/calibration.md` § strong vs weak).

## 7. Report

Package path; the task table; each kind with its validators and the control
results (verbatim PASS/FAIL lines); the `show` verdict; how to run it
(`agent-sys-task-runner`, or `agent-sys run --package <dest>` when it is inside
a git repo); and what the history did not settle (open questions), stated as open.

## Rules

- agent_sys's formats only; when unsure of a field, read `spec_loader/schemas/`.
- No secret values anywhere — names and where they come from.
- Do not delete or move anything the history produced; copy.
- Ask before copying any file over 4 MB into the package.
