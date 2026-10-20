---
name: agent-sys-handoff-packup
description: Use when Claude's work on a task should become an agent_sys handoff — "把结果打包成 handoff / 交付成 agent_sys 能用的 handoff / package this as a handoff / make this result reusable by agent_sys / 整理成 handoff". Packs the work into a complete, reproducible, right-sized handoff in agent_sys's on-disk format and proves it against the task's own validators. Builds on experiment-result-packup (its reproduction-kit rules are carried inline). If no task/handoff definition exists yet, first runs agent-sys-task-from-history. NOT for running a task (agent-sys-task-runner).
---

# Handoff pack-up

A handoff is agent_sys's unit of trust: the next task — or the next person —
starts from it cold. Two properties, in this order:

1. **Complete and reproducible** — a stranger can re-run it and get the result
   without asking you anything (experiment-result-packup's test).
2. **Not absurdly large** — every byte is there because someone needs it;
   big artefacts are referenced by absolute path + sha256, not copied.

## Setup

```bash
ASTK="$(realpath "${CLAUDE_SKILL_DIR}")/../lib/astk.py"   # executable; one word, so bash and zsh both run `$ASTK cmd`
DOCS="$(realpath "${CLAUDE_SKILL_DIR}")/../docs"
```

Each Bash call is a fresh shell: repeat these two lines (or use the full path).
A package with required `${VAR}`s needs the same `--var K=V` on **every**
command that loads it; `astk` names the missing ones.

Read `$DOCS/primer.md` § A handoff and `references/layouts.md` first.

## 1. Find the definition

You need a task package and the **kind** to produce:

```bash
$ASTK inspect <package> [--var K=V ...]   # kinds, content types, producers, validators
```

- No package, or no kind that fits the work → invoke the
  `agent-sys-task-from-history` skill first (Skill tool), scoped to this work,
  and come back with its package. If the user wants no task, derive a
  one-kind package with it anyway — a handoff without a validator is not one
  agent_sys can admit.
- Read the kind's `description`, its `items_schema`, and the readme of every
  validator bound to it (`assets/<validator>.validator/readme.md`) — that is
  the bar. Read the producing task's readme § "What you must write".
- `items_schema` with `additionalProperties: false` admits **only** the items
  it lists: then everything else below (environment, commands, notes) goes
  into README sections, not extra items. Otherwise extra items are legal.
- Only judging, and the package wants `--var`s that feed the producer alone?
  `--allow-unset-vars` loads it; a validator whose `args` need one is reported
  UNREACHED rather than run on a wrong value.

## 2. Ask where it goes — three suggestions

AskUserQuestion, three options plus "Other", each with its reason:

1. **Beside the work** — `<experiment dir>/<kind>.handoff_<YYYYMMDD>/` (the
   packup convention: results stay co-located with their source).
2. **As an agent_sys fixture** — `<package>/assets/fixture/<kind>/`. agent_sys
   consumes it through a program leaf that publishes it (`astk scaffold`
   generates one, `publish_inputs`, for a plan kind with `fixture:`), or
   `agent-sys-task-runner` seeds it (`astk seed`). A copy anywhere else is
   just a copy nothing consumes.
3. **Into an agent_sys-shaped store** — `<runs dir>/handoffs/` via `astk seal`
   (`handoff.<kind>.<uuid>/v0/{content,manifest.yaml,validation.yaml}`), e.g. the
   run dir of `agent-sys-task-runner` so a downstream leaf can start from it.

The content is built once in a staging dir (scratch, not the repo) and copied
to the destination(s).

## 3. Gather — then build the content

Scale this to the kind: a hello-world-sized record needs its inputs, the
commands and the environment; an experiment needs everything below.
Gather as experiment-result-packup Phase 1 does (spec, working log, result
dirs, memory, conversation corrections). If that skill is in your skill list,
use it for a `code` kind's packup tree (`references/layouts.md` says per
content type whether a packup tree is used); otherwise follow its rules,
which are inline below.

```bash
$ASTK template --package <pkg> --kind <kind> --out <staging>/content [--var ...]   # skeleton + the task's "what to write"
$ASTK env --repo <the repo the work ran in>                                        # machine + software snapshot
```

Put the `env` snapshot where the layout says (an `env` item, or the README's
Environment section when the kind admits no extra item), and state the gap
between capture time and when the work ran.

**Copy the work verbatim.** Where it falls short of the kind's contract,
say so in the README; do not rewrite the work to fit. If you did the
producing task's work yourself, record each input you read: its path, its
handoff id and its digest (`$ASTK digest <dir>`).

Fill every README section with real text (the skeleton's bodies are the
placeholder the store rejects). Fill the items per `references/layouts.md`.
The seven things reproduction needs — each present, or `N/A, because …`:

1. **Time** — absolute dates of the runs.
2. **Purpose + spec** — the goal and the success criteria, quoted; the result
   stated against each criterion (pass/fail + margin).
3. **Environment** — hardware (GPU model/count, driver/ROCm), software (image
   tag **and** digest, git branch **and** commit), the `astk env` snapshot.
4. **Exact commands/scripts** — copied verbatim; the reproduce command is an
   executable script (`chmod +x`), ordered from zero to result. For a report,
   reproduction = the commit it describes + how to re-check it + how to re-run
   the task that wrote it.
5. **Patches / fixes** — each with what / why / how / the symptom it cured.
6. **External dependencies** — absolute paths (weights, datasets, injected
   libs) and how to obtain them; **secrets by name and source, never value**.
7. **Notes** — traps and wrong turns as what / why / how / context.

Text produced by subagents may start with a `[harness: …]` annotation; strip it.

## 4. Size and hygiene

```bash
$ASTK audit <staging>/content
```

- `BLOCK` lines (a literal secret value, unfilled template text in prose) must be fixed.
- `ASK` lines go to the user: a file over 4 MB → include / reference by absolute
  path + sha256 / drop; logs → keep / trim / gzip. Record a referenced file in
  the README (path, size, sha256, how to fetch).
- Never move or delete an original — copy (`cp -a`, modes kept: the digest
  counts the executable bit).

## 5. Prove it against the task's own validators

```bash
$ASTK check --package <pkg> --kind <kind> --content <staging>/content [--var K=V ...]
```

This is the store's seal test, the gate's executable-bit test, then **every
validator bound to the kind** run the way agent_sys runs them (fresh zone under
`$TMPDIR/astk-<uid>/validation`, on copies of the content, `inputs.json`/`args.json`/`materials.json`, no
inherited environment). Exit 0 = agent_sys would accept it. On a refusal: read
the reasons, fix the handoff, run again. Never edit a validator or its args to
get a pass; if a bar looks wrong, stop and tell the user which bar and why.

- `cost: gpu_hours` validators re-run real work — say so before running them.
- `logic_source: external_dynamic` validators measure the world (the host, a
  server): run them where the work ran, whatever their cost.
- A pass from a `weak` validator is low-confidence: open the artefact and check
  the claim it could not.

Then deliver to the chosen destination:

```bash
mkdir -p <dest> && cp -a <staging>/content/. <dest>/       # beside the work (<dest> = the dated dir) / fixture
$ASTK seal --package <pkg> --kind <kind> --content <staging>/content \
    --store <runs>/handoffs --task-id <producing task>#<n> [--var ...]
```

`seal` re-validates in place and writes `manifest.yaml` (sha256 tree digest,
agent_sys's algorithm) only on a full pass; sealing identical content again is
a no-op that names the existing version.

## 6. Report

Destination path(s); the digest (`$ASTK digest <content>`); every validator
verdict, verbatim; what was left out and why (big files referenced, logs
excluded); gaps you could not fill (e.g. "image digest not captured at run
time"), stated as gaps; any bar that looked too weak or too strong while
packing (for `agent-sys-task-from-history` to fix in the plan); and how
agent_sys consumes it (fixture + publish leaf, or the store path for
`agent-sys-task-runner`'s `seed`).
