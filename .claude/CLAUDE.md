# Task — Reproduce, debug and accept the LLM e2e opt chain on THIS cluster

Authority: repo-root **`mission.md`** + **`mission.verify.e2e.md`**. Everything
below traces to one of them. The previous round's file is
`.claude/CLAUDE.handoff-refine.20260906-0627.md.bak` — **it is about a different
cluster**; read it for mechanism, never for values (nodes, jobids, `/shared_nfs`
paths, `spur`, 8-hour holds — none of that is true here).

## The job

`agent_sys/examples/llm_e2e_performance_optimization/e2e-flow/` — one package
whose graph runs module 1 → 5 in **one `agent-sys run`**. No `agent_sys` core
changes.

1. **Reproduce** the first cluster's board here (modules 1, 2, 5 real).
2. **Complete all five stages in one run, reaching `packup`.**
3. Run module 4 for real if compute allows. **It is not this hold's goal.**
4. Debug what breaks and record it.

### The user's two decisions, 2026-09-06 06:2x

- **Two phases on the model.** Phase A: `Qwen3-32B` + the stock image already on
  this host — get the chain green with the cheapest possible bring-up. Phase B:
  swap `image`/`model_path` to GLM-5.3-Flash and re-run once for acceptance.
- **Priority is 1→2→3→5 + `packup`.** Module 4 rides on a replayed or
  self-declared-degraded artefact this hold. A degraded artefact **must say so in
  itself**; a `speedup: 1.0` from an empty-change baseline is a correct value,
  not a guess.

---

## Context — this host (every row measured 2026-09-06 06:2x; re-measure, do not inherit)

| | |
|---|---|
| where I am | **on the compute node itself**, `smci355-ccs-aus-n04-25`. Not a login node. `docker` talks to the local daemon. |
| GPUs | 8 × MI355X (gfx950), **all idle, VRAM 0 %** at 06:25 |
| the hold | slurm job **29184**, `TimeLimit=16:00:00`, **`EndTime=2026-09-06T14:00:01`**. Partition timelimit is `infinite` but this job's is not. **Reckon time remaining, and re-read `scontrol show job 29184` — do not trust this row.** |
| transport | **`--var transport=local`, and it must be asked for.** `assets/lib/remote.sh` has a `local` branch that the probe **never selects** ("no transport binary present" ≠ "I am on the node"). `spur` is absent, `srun` is present, so the probe would pick `srun` and step into an allocation we are already inside. |
| scratch | **`/data/yihou/e2e_verify_20260906/`** — local `/dev/md0`, 41 T free, writable. All temporary activity here. |
| run root | **`/data/yihou/agent_sys_runroot`** |
| `work_root` | must be **local disk**: `/data/yihou/e2e_flow`. `/home` is autofs NFS; `/apps` is NFS and 99 % full. The package warns a root-squashed network home makes the engine fail to write logs *silently*. |
| model (phase A) | `/apps/data/models/Qwen3-32B` (~65 GB, plain directory) |
| model (phase B) | `/apps/data/models/GLM-5.3-Flash` (306 GB) + an image built from `/apps/yihou/packups/glm53flash.mix.packup_20260830/Dockerfile.sglang.glm53` |
| image (phase A) | `lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260903` — already loaded |
| etcd image | `quay.io/coreos/etcd:v3.5.14` — already loaded |
| network | outbound HTTPS to quay.io and github.com works (200) |
| **no corpus** | `/shared_nfs` exists and is **empty**. The first cluster's 25 sealed handoffs are NOT here. **So nothing can be mocked until we have produced our own.** The ladder is bottom-up: run a stage real → seal → `assets/lib/replay_root.py` materialises a `mock_root` from it (`SKIP-AHEAD.md` is the manual). |
| foreign tenants | containers `rc_26_7_902` and `xiaoming-dev` are running and **hold no GPU**. Leave them. Re-check before every bring-up. |
| fast loop | `PYTHONPATH=$PWD/agent_sys python3 -m cli.main show --package … --var …` type-checks every yaml in < 1 s. **`PYTHONPATH` is mandatory** — a bare `agent-sys` resolves to the `infera.aiopt.all` worktree, not this one. |

---

## Key references, in the order to read them

1. **`e2e-flow/RUN-PLAN.md` § `## 2. The command — CANONICAL LAUNCH BLOCK`**
   (~line 2299). **This file holds SEVEN launch blocks and they disagree.** Diff
   against that one and no other; its heading is unique on purpose.
2. **`e2e-flow/WHAT-GREEN-ESTABLISHES.md`** — which validators have ever refused
   anything, and what a pass from each is worth.
3. `e2e-flow/CONTRACT.md` — the frozen fifteen-kind cross-module contract.
4. `e2e-flow/SKIP-AHEAD.md` — the replay mechanism, **and its first page says it
   is a debugging accelerator and never an acceptance path.**
5. `/apps/yihou/packups/glm53flash.mix.packup_20260830/` — how GLM-5.3-Flash was
   brought up on an 8×MI355X box, for phase B.

---

## Core principles

1. **Read the artefact, not the exit code.** Every acceptance claim names a file
   to open and a condition that fails.
2. **Before attributing ANY refusal**, run
   `bash assets/lib/refusal_saw_something.sh <run dir>` — **`bash`, not `sh`**.
   A validation zone is sometimes handed **zero files** (~1 in 11–13 zones);
   a refusal from a zero-file zone says nothing about the artefact, and a PASS
   carries no reason at all, so the file count is the only retrospective check.
3. **Record your launch line beside the run.** The staged package keeps
   `${var:-default}` unrendered, so which `--var` a run used **cannot be read
   back from the artefact**. Four separate incidents trace to this.
4. **Audit the whole variable table against your `mock_stages` every launch** —
   not the rows you remember. `expect_ranks`, `adhoc_cases`, `bench_rounds` each
   have one value when a stage is mocked and another when it is real; three
   launches were lost fixing them one at a time.
5. **`E2E_KIT_PORT_BASE` moves the whole port band.** `port_router` moves one
   port and leaves the band behind. A collision makes the body `exit 1` with no
   escalation recipient, so the task sits at `running` and the log reads healthy
   for the full stall timer.
6. **The `run_with_long_stall.py` wrapper is not optional.** `stall_after` is not
   on the CLI; a bare launch gets the 20-second default and a real rung cannot
   survive it. The wrapper prints `stall_after 20s -> 900s` when it takes — **if
   that line is absent, kill and re-issue.**
7. **Module 5 brings its arms up through a node-wide GPU reset** that kills every
   GPU process whose command name matches the engine's. It protects no
   co-tenant. Its arms are pinned to the first TP devices and two of its
   coordination ports are literals, **so a run with a real module 5 owns the
   host.** Check the cards and the container list before every bring-up.
8. **Killing an orchestrator does not kill its agents**, and an agent will
   rebuild a container about a minute after you stop it. **Order: agents first,
   then containers, then verify.** Before any irreversible action against a run,
   open the transcript or the event's `attributes.detail` — not the phase line.
9. **Pre-register.** Write down what each outcome would mean, and what the
   current state is, *before* the result arrives.
10. **Never let a computed value reach a destructive command unchecked**
    (`[ -n "$X" ] || exit 1`). **Deletion: nothing whose path lacks `yihou` or
    `/tmp`.** Stop containers with `docker stop -t 10`; never force-remove one
    you did not create. Never `--clean` a shared run root.
11. **Do not infer ownership from a name.** Ownership is a label, an
    auto-remove flag and a process list.
12. **A tool that returns a well-formed answer on bad input is the dangerous
    kind.** Before trusting a new instrument, run it on a case whose answer you
    already know.
13. **Do not request machines.** Query and use what is held; if there is not
    enough, say so.

## Working rules

- **Work in English. Report to the user in Chinese.**
- **Every commit is `git commit -s`.** CI rejects a commit without the trailer.
  Branch `dev.yihou.aiopt.task_package.concat`. The repository receives only
  `agent_sys/examples/llm_e2e_performance_optimization/` and the root `*.md`
  notes.
- **Agent team.** Leader polls every 10 min; a problem seen once is recorded, and
  intervened on at the second sighting **unless it is burning GPU or hold time,
  in which case intervene at the first.** An owner with nothing assigned is the
  leader's failure.
- **A checkpoint writer appends to `work.checkpoint.summary.md` every 30 min**:
  estimated completion %, elapsed and projected time, reliability; current
  state; code problems fixed/unfixed; non-code problems; open questions; new
  commits one line each.
- Framework bugs → `agent_sys/examples/llm_e2e_performance_optimization/bug.record.2026-09-06.md`;
  validator failures → `validator.failures.2026-09-06.md`, same directory.
- **Do not change the host** (root filesystem, system state) without asking.
