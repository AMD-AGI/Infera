---
name: agent-sys-task-runner
description: Use when the user wants to run, try, debug or learn an agent_sys task or task package directly in Claude Code without starting agent_sys — "在 claude code 里直接跑这个 task / 不启动 agent_sys 执行 task package / run this agent_sys task here / debug this task / 用 workflow 跑 e2e-flow". Reads the package, explains its graph, executes its leaves as a Workflow mirroring the task graph, and holds every output to the task's own validators so the handoffs it writes are reusable by agent_sys. NOT for writing a task (agent-sys-task-from-history) or packing finished work (agent-sys-handoff-packup).
---

# Task runner — an agent_sys package, executed by Claude Code

The package is the contract (readmes = instructions, kinds = outputs,
validators = the gate); only the runtime is replaced, so a handoff published
here passed the seal test and validators agent_sys would have run.

## Setup — and recommend ultracode

```bash
SKILL_ROOT="$(realpath "${CLAUDE_SKILL_DIR}/..")"
ASTK="$SKILL_ROOT/lib/astk.py"     # executable; one word, so `$ASTK cmd` works in bash and zsh
DOCS="$SKILL_ROOT/docs"
```

Each Bash call is a fresh shell: repeat these lines. `--var K=V` goes to
`inspect`, `show` and `run-init`; `--run` commands take none (the run stores them).

Tell the user once, in their language: this skill works best with **ultracode**
on (`ultracode`, or `/effort ultracode`) — the graph runs as a Workflow, each
leaf in its own agent, independent ones in parallel; without it, §4b. Invoking
this skill is the user's request to run the package's workflow.

## 1. Understand before running

```bash
$ASTK inspect <package> [--var K=V ...]     # leaves in order, waits-for, kinds, validators (cost/strength)
$ASTK show <package> [--var K=V ...]        # agent_sys's own static check
```

Read `assets/main.task/readme.md`, the package README/CONTRACT if any, every
leaf's readme and every validator's readme (`$DOCS/primer.md` says where they
live). Then tell the user, briefly: the graph (who produces what, who waits for
whom); each leaf's executor — **program** (`entry.sh` runs as is) or **ai**
(Claude follows its readme); what it needs (required `--var`s, external inputs
= kinds no leaf produces, GPUs/containers/nodes); the bar (each kind's
validators, which are `gpu_hours`); and what cannot run here — now, not after
an hour. A long AI leaf (e2e-flow's `deploy_and_prove`, `optimize_kernel`) is
worth splitting into stages (`references/workflow_patterns.md`).

## 2. Run dir, variables, external inputs

Propose a run dir (default `/var/tmp/$USER/agent_sys_runs/<package>-<YYYYMMDD-HHMM>/`)
and the `--var` values; confirm with the user.

```bash
$ASTK run-init --run <run> --package <package> --var K=V ...
$ASTK seed --run <run> --kind <kind> --content <dir>    # README.md + items/; validated like a producer's output
```

Seed from a package fixture (`assets/fixture/<kind>/`), an earlier run
(`<old run>/handoffs/handoff.<kind>.*/v*/content`) or `agent-sys-handoff-packup`.

## 3. The per-leaf protocol

```bash
$ASTK prepare     --run <run> --leaf <leaf>   # zone, staged inputs, output slots, env.sh, brief.md
$ASTK run-program --run <run> --leaf <leaf>   # program leaves only
$ASTK finish      --run <run> --leaf <leaf>   # THE GATE: seal test + every validator; publish on a full pass
```

- `prepare` refuses a leaf whose inputs are unpublished or whose upstream leaf
  started and has not succeeded, and names what is missing. Each `prepare` is
  a new attempt: new zone, new output slots.
- `brief.md` is the leaf's prompt as agent_sys assembles it, plus how to source
  `env.sh`. An AI executor reads it whole and works in the zone.
- `finish` judges all of the current attempt's outputs as one output phase:
  they are published together only if the whole phase passes. On NOT PASSED,
  fix the **handoff** and run `finish` again — at most 3 calls per leaf. Never
  edit a validator, its args or the package to get a pass; if a bar looks
  wrong, stop that leaf and report which bar and why.

## 4a. With ultracode: the Workflow

```bash
$ASTK workflow --run <run> --out <run>/workflow.js
```

One agent per leaf, started once its upstream leaves succeeded (their `finish`
exit and the run state), then a closure agent; leaves already done are left out.
Read it, refine stages if needed, run it with the Workflow tool (`scriptPath`).
After a failure, fix the cause and **regenerate**: `resumeFromRunId` replays
cached results, the failure included.

## 4b. Without ultracode: inline

Walk the leaves in `inspect` order, §3 for each (also when the Workflow tool is
unavailable, e.g. you are a subagent).

## 5. Closure and report

```bash
$ASTK closure --run <run>      # every leaf's status, every non-leaf's exported kinds; exit 0 = complete
$ASTK status  --run <run>      # per leaf: status, prepares=N finishes=M
$ASTK export  --run <run> --to <run>/fixtures     # fixtures + the publish line for agent_sys
```

Report the leaf table (status, finish attempts), each published handoff's path,
every non-PASS verdict verbatim, and the divergences from a real agent_sys run
(`references/workflow_patterns.md`). Then offer, in one question, to back the
published handoffs up to the knowledge repo (`$ASTK remote backup
<run>/handoffs/handoff.<kind>.<hid>/v<N> --summary ... --tag ...`;
`$DOCS/knowledge_repo.md`). Copying fixtures into a package is the user's call;
fixtures of `external_dynamic` validators may only pass where they were made.

## Rules

- The package is read-only: write only in the zone and the output slots `brief.md` names.
- Shared machines: check GPU use first; never kill a process or remove a container you did not start.
- No secret value in any handoff, kept log or report.
- A green exit is not a result: open the published handoff before calling a leaf done.
