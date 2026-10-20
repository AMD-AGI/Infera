---
name: agent-sys-task-runner
description: Use when the user wants to run, try, debug or learn an agent_sys task or task package directly in Claude Code without starting agent_sys — "在 claude code 里直接跑这个 task / 不启动 agent_sys 执行 task package / run this agent_sys task here / debug this task / 用 workflow 跑 e2e-flow". Reads the package, explains its graph, executes its leaves as a Workflow mirroring the task graph, and holds every output to the task's own validators so the handoffs it writes are reusable by agent_sys. NOT for writing a task (agent-sys-task-from-history) or packing finished work (agent-sys-handoff-packup).
---

# Task runner — an agent_sys package, executed by Claude Code

The package's definitions are the contract: its readmes are the instructions,
its handoff kinds are the outputs, its validators are the quality gate.
Only the runtime is replaced. A handoff this skill publishes passed the same
seal test and the same validators agent_sys would have run on it.

## Setup — and recommend ultracode

```bash
SKILL_ROOT="$(realpath "${CLAUDE_SKILL_DIR}/..")"
ASTK="$SKILL_ROOT/lib/astk.py"     # executable; one word, so `$ASTK cmd` works in bash and zsh
DOCS="$SKILL_ROOT/docs"
```

Each Bash call is a fresh shell: repeat these two lines (or use the full path).
`--var K=V` goes to `inspect`, `show` and `run-init`; the run stores the
values, so the `--run` commands take none (re-run `run-init --var` to change one).

Tell the user once, in their language: this skill works best with
**ultracode** on (type `ultracode` in a message, or `/effort ultracode`) — the
task graph then runs as a Workflow, independent leaves in parallel, each leaf
in its own agent context. Without it, run the same steps leaf by leaf inline
(§4b). Invoking this skill is the user's request to run the package's workflow.

## 1. Understand before running

```bash
$ASTK inspect <package> [--var K=V ...]     # leaves in order, waits-for, kinds, validators (cost/strength)
$ASTK show <package> [--var K=V ...]        # agent_sys's own static check
```

Read, in this order: `assets/main.task/readme.md`, the package README/CONTRACT
if any, every leaf's readme, every validator's readme (`$DOCS/primer.md` says
where they live). Then explain to the user, briefly:

- the graph: leaves, what each one produces, who waits for whom;
- each leaf's executor: **program** (its `entry.sh` runs as is) or **ai**
  (Claude follows its readme);
- what it needs: required `--var`s (a `${VAR}` without default fails the load),
  external inputs (kinds no leaf produces), GPUs/containers/cluster nodes;
- the bar: each kind's validators, and which ones are `gpu_hours`;
- what cannot run here (a machine that is not this one, a missing image) — say
  it now, not after an hour.

Large AI leaves: read their STEPS. If a leaf is long (e2e-flow's
`deploy_and_prove`, `optimize_kernel`), plan to split it into stages
(`references/workflow_patterns.md` § refining a leaf).

## 2. Choose the run dir and variables

Propose a run dir (default `/var/tmp/$USER/agent_sys_runs/<package>-<YYYYMMDD-HHMM>/`,
or a scratch disk the user works on) and the `--var` values; confirm with the user.

```bash
$ASTK run-init --run <run> --package <package> --var K=V ...
```

External inputs are seeded, and seeding validates them like any producer's output:

```bash
$ASTK seed --run <run> --kind <kind> --content <content dir>    # README.md + items/
```

Sources for a seed: a fixture in the package (`assets/fixture/<kind>/`), a
handoff from an earlier run (`<old run>/handoffs/handoff.<kind>.*/v*/content`),
or one made with `agent-sys-handoff-packup`.

## 3. The per-leaf protocol (what every executor does)

```bash
$ASTK prepare     --run <run> --leaf <leaf>   # zone, staged inputs, output slots, env.sh, brief.md
$ASTK run-program --run <run> --leaf <leaf>   # program leaves only
$ASTK finish      --run <run> --leaf <leaf>   # THE GATE: seal test + every validator; publish on pass
```

- `prepare` refuses a leaf whose inputs are not published yet (the scheduler's
  readiness rule) and says which leaf or seed is missing.
- `brief.md` is the leaf's prompt as agent_sys assembles it (readme + "What this
  task must deliver" with exact paths and sections) plus how to source `env.sh`.
  An AI executor reads it whole and works in the zone.
- `finish` prints every verdict with the validator's own reasons. On NOT PASSED:
  fix the **handoff**, run `finish` again — at most 3 calls per leaf. Never edit
  a validator, its args, or the package to get a pass; if a bar looks wrong,
  stop that leaf and report which bar and why.

## 4a. With ultracode: the Workflow

```bash
$ASTK workflow --run <run> --out <run>/workflow.js
```

The generated script starts one agent per leaf as soon as the leaves it waits
for have succeeded (no stage barriers), gates on each leaf's `finish` exit
code, skips a leaf whose upstream failed, and ends with a closure agent. Leaves
the run already has (succeeded, or seeded) are left out. Read it, refine it if
a leaf needs stages, then run it with the Workflow tool
(`scriptPath: <run>/workflow.js`).

After a failure: fix the cause, **regenerate** (`astk workflow` again — finished
leaves drop out) and run the new script. `resumeFromRunId` replays cached
agent results, including a cached failure, so it re-runs nothing unless the
failed leaf's prompt changed.

## 4b. Without ultracode: inline

Walk the leaves in `inspect` order; for each, §3. Independent leaves can still
go to background Agents if the user agreed to subagents. Use this path too when
the Workflow tool is not available to you (e.g. you are a subagent);
`astk workflow` is still worth generating to review the graph.

## 5. Closure and report

```bash
$ASTK closure --run <run>        # every leaf's status, every non-leaf's exported kinds; exit 0 = complete
$ASTK status  --run <run>      # per leaf: status, prepares=N finishes=M
```

Report: the leaf table (status, finish attempts), each published handoff's
path, every non-PASS verdict verbatim, and what diverged from a real agent_sys
run (`references/workflow_patterns.md` § divergences). Then reuse:

```bash
$ASTK export --run <run> --to <run>/fixtures     # fixtures + the publish line for agent_sys
```

Then offer, in one question, to back the run's published handoffs up to the
knowledge repo (`$ASTK remote backup <run>/handoffs/handoff.<kind>.<hid>/v<N>
--summary ... --tag ...`; see `$DOCS/knowledge_repo.md`) — a handoff from a
real run is what the next person's prior-work search will find.

Copying fixtures into a package is the user's decision (the package is
read-only to this skill). `export` notes kinds whose validators re-measure the
world (`external_dynamic`): those fixtures may only pass on the machine that
made them.

## Rules

- The package is read-only. Each zone has its own package copy; write only in
  the zone and the output slots `brief.md` names.
- Shared machines: check GPU use before taking GPUs; never kill a process or
  remove a container you did not start.
- No secret value in any handoff, log you keep, or report.
- A green exit is not a result: open the handoff `finish` published before
  calling a leaf done.
