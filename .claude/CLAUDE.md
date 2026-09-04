# Task — Handoff refine: chain the five LLM e2e optimisation stages into one graph

Build **one** task package,
`agent_sys/examples/llm_e2e_performance_optimization/e2e-flow/`, whose graph runs
stage 1 → 5 in a **single `agent-sys run`**, to the rules in the repo-root
`mission.md` (rewritten 2026-09-03).

Three deliverables, in order:

1. **Definitions rewritten** to mission.md's general rules + per-module list.
2. **Mock e2e green** — mock agents fed by the 25 real sealed handoffs in
   `/shared_nfs/yihou/agent_sys/cheat_for_mock/`; one run walks all five stages
   and every validator passes.
3. **Real e2e** — staged, one stage promoted from mock to real at a time.

Decided with the user 2026-09-03: **new package, the five demos stay untouched**;
**all three items this round**; **team = 5 module owners + 1 checkpoint writer**
after a solo contract freeze.

## Background

The five stages exist today as five *separate* packages (`deploy-demo`,
`profiling-demo`, `analyze-demo`, `kernel-opt-demo`, `integration-demo`), each
driven to a real cluster run on 2026-09-02: **45 sealed handoffs, 50 validator
PASS, 2 documented refusals**, ~3.5k lines of YAML and **~20k lines of proven
`.py`/`.sh`**.

They are not a flow. **A handoff only travels inside one run's graph**, so five
packages are five runs and nothing chains. The seams recorded in
`handoff.analysis.md` — one `kernel_table` name over two `content_type`s,
"workset" spanning two kinds with different required files, two same-named
`check_service_live`s — are exactly what side-by-side authoring produces.

**This is a refine of definitions, not a rewrite of bodies.** The 20k lines of
assets are the only thing here that has ever produced a number; they move and
adapt, they do not get re-derived.

## Context — this environment, measured 2026-09-03

| | |
|---|---|
| where I am | login node, **no GPU**, no direct docker daemon |
| GPU holds | `106250`→`crsuse2-m2m-061` (10.245.159.129), `106253`→`crsuse2-m2m-031` (10.245.144.239); 8 h each from ~12:40 UTC; both reachable, both carrying **other tenants' containers** |
| reaching them | `spur exec <jobid> bash -c '...'` — exec namespace; docker talks to the **host** daemon, but the **filesystem identity is you, not root** (measured 2026-09-04: `id -u` → `50112975`, writes land `-rw-r--r-- yihou ubuntu`). Matters because `/home` is `sec=sys` NFS, where a root-squashed write would map to `nobody` and leave a tree nobody can clean up. |
| shared FS | `/shared_nfs` 360 T, **46 T free**, shared by every spur node — this is how "remote" works |
| scratch | `/shared_nfs/yihou/agent_sys/ws_handoff_refine/` — all temp activity lives here |
| mock inputs | `/shared_nfs/yihou/agent_sys/cheat_for_mock/` — 25 real sealed handoffs, one folder per kind |
| fast loop | `python3 -m agent_sys.cli.main show --package <dir> --var …` loads and type-checks every yaml in **< 1 s** |

## Key references

- **`mission.md`** (repo root) — the authority. Every requirement below traces to
  a numbered item there.
- `handoff.analysis.md` (repo root) — the 26 kinds, their validators, and the
  three cross-stage seams.
- `/shared_nfs/yihou/agent_sys/cheat_for_mock/README.md` — **four things that
  will mislead you**, including a `kernel_table` that is a 34-row synthetic seed
  and an `integration_report` carrying a *refused* verdict.
- `/shared_nfs/yihou/agent_sys/debugging/integration/DELIVERY-NOTE-FROM-LEADER.md`
  — why that refusal was the validator working, and why the 5 % / 10 % bars must
  **not** be widened.
- `/shared_nfs/yihou/agent_sys/temp/leader/repair_modes.py` — restores the file
  modes a past `chmod -R 777` erased from sealed handoffs.
- `agent_sys/docs/design.md` §13 (WIP) — the runtime tree.
- `agent_sys/spec_loader/validate.py:34-56` — the `jsonschema` idiom to copy.

## Core principles

1. **Read the artefact, not the exit code.** Every acceptance claim names a file
   to open and a condition that fails.
2. **`items_schema` is not a schema layer.** Measured: for a file/tree item
   `handoff/content.py:184-197` validates the *filename string*, never the
   contents, and the schema is never exported to a body. Mission rule G2 needs
   real schemas under `assets/schemas/`, loaded by producer **and** validator.
3. **Every identifier bound on a shared host is a parameter.** Container names,
   ports, workdir, served model name. `: "${VAR:=…}"`, never `export VAR=`.
4. **Never delete anything on `/shared_nfs` whose path lacks the substring
   `yihou`.** Never `docker rm -f` a container you did not create. Never
   `agent-sys run --clean` on a shared root — it removes *every* run.
5. **Research → gather → analyse → plan → work.** Temp activity in
   `ws_handoff_refine/`; the repo receives only `e2e-flow/` and `todo.md`.
6. Bugs in `agent_sys` are recorded under
   `agent_sys/examples/llm_e2e_performance_optimization/temp/bugs/` first, then
   worked around; fixed only when the evidence is unambiguous.
7. Work in English; report to the user in Chinese.

## Other notable details

### Two mission requirements that read as traps

- **Rule 7, "all tasks share one docker container".** Module 5 needs **two** by
  construction — a container holds one state for its life, which is the whole
  reason the two-arm design exists. G5.1's *"如果不行，再考虑…启动不同的 docker
  container"* grants the exception: modules 1–4 share one, module 5 brings up
  its own two arms from the same image and the same `environment` record.
- **M5.1.3, Python runtime hijack.** M5.3 says *"首先记入 todo … 但现在就这样吧"*
  — record the disagreement in `todo.md`, **keep `overlay_files`**.

### DCO sign-off is required on every commit

CI blocks any PR containing a commit without a `Signed-off-by:` trailer.

```bash
git commit -s -m "..."
git config user.name && git config user.email
```

Sign off **as yourself** — never a colleague's line, never a bot identity.

### Branch / PR

Branch `dev.yihou.aiopt.task_package.concat`. Activity is limited to
`agent_sys/examples/llm_e2e_performance_optimization/` plus the root-level
`*.md` notes this effort writes.
