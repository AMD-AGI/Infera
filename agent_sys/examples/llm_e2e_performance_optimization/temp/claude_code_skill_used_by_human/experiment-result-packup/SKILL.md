---
name: experiment-result-packup
description: Use when an experiment, benchmark, or debugging effort has reached a milestone and the user asks to package / tidy / archive / write up the results, produce a deliverable, make the work reproducible, or "整理实验成果 / 打包成果 / 交付". Gathers the spec, working-process log, environment (hardware + software), exact reproduction commands, patches/fixes, dependency paths, and required secrets into one clean, self-contained folder next to the experiment so anyone can reproduce it cold. Trigger even when the user only says "整理一下" / "pack this up" in an experiment context.
---

# Experiment result pack-up

## Purpose

An experiment is only valuable if someone else (or future-you, six months from
now, on a fresh machine) can **reproduce it cold**. This skill turns a pile of
scratch — a spec markdown, a `working_process.md` log, timestamped result dirs,
half-remembered debugging fixes — into one clean, self-contained deliverable
folder that captures *everything* needed to re-run the experiment and trust the
numbers.

The deliverable is not a summary. It is a reproduction kit. The test is simple:
**hand the folder to a stranger with cluster access and they should be able to
reproduce the result without asking you a single question.**

## When this triggers

The user has done real work (benchmark / perf sweep / debug session / ablation)
and signals a milestone: "整理成果", "pack this up", "write this up", "make it
reproducible", "archive the results", "produce a deliverable". Even a terse
"整理一下" in an experiment context counts — err toward triggering.

## Two hard rules (read before touching anything)

1. **Never delete original experiment files** — not local, not remote, not the
   workspace, not the containers, not the logs — unless the user *explicitly*
   says so. You are *copying and synthesizing*, not moving. The original mess
   stays exactly where it is. When in doubt, copy; never `mv`, never `rm`.

2. **Ask before committing heavy or sensitive artifacts.** The deliverable may
   be committed to git. Before including any file **> 4 MB**, ask the user. Logs
   can be huge and are often the crux of a debug story — **ask the user whether
   they want log files included** (and if so, whether trimmed/gzipped). Never
   pack real secret *values*; record only *which* secrets are needed and where
   they come from (see the secrets section).

## Workflow

Work through these phases in order. Use a task list to track them — a pack-up
touches many sources and it's easy to drop one.

### Phase 1 — Gather the sources

Find and read every input that describes what was done. Typical sources in this
kind of repo:

- **The spec / task file** the user handed down — usually a `*.md` (e.g.
  `dsv4_bench_sglang_pd_align.md`). This is the *purpose* of the experiment.
  Also check `CLAUDE.md` for the "current task" pointer.
- **The working-process log** — `*.working_process.md`. This is the *narrative*:
  what was tried, what broke, what fixed it, the wrong turns. The debugging gold
  lives here. Read it fully — do not skim.
- **Result directories** — timestamped dirs like `bench_results/<name>_<ts>/`,
  each typically holding `scripts/`, `out/`, `logs/`. These are the *evidence*.
- **Your memory** for this project (`memory/MEMORY.md` and linked files) — for
  feedback, gotchas, and decisions not written elsewhere.
- **Conversation history** — corrections and constraints the user gave live-only.

If a source is ambiguous or missing (e.g. no working-process log exists), note
the gap explicitly rather than inventing content. A reproduction kit that says
"env driver version not captured — see host at the time" is more honest and more
useful than a confident guess.

### Phase 2 — Build the deliverable directory

Create the deliverable **in the same folder as the experiment** (rule: results
stay co-located with their source), with a clear, dated name:

```
<experiment-name>.packup_<YYYYMMDD>/
```

Use the canonical layout below. Read `references/deliverable_layout.md` for the
full annotated structure and a copy-paste `REPRODUCE.md` template. The skeleton:

```
<experiment-name>.packup_YYYYMMDD/
├── README.md            # entry point: what/why/result at a glance + how to navigate
├── REPRODUCE.md         # THE reproduction kit — exact ordered steps, copy-pasteable
├── environment.md       # hardware + software env, pinned versions, SHAs
├── scripts/             # every script needed to reproduce (copied verbatim)
├── patches/             # fixes/patches applied, each with a what/why/how note
├── results/             # the numbers: CSV/PNG/summary tables (evidence)
├── notes.md             # gotchas, wrong turns, error analysis (what/why/how/context)
└── logs/                # OPTIONAL — only if user opts in (may be trimmed/gzipped)
```

### Phase 3 — Fill each file (the seven things reproduction needs)

The `references/checklist.md` file is the authoritative field-by-field checklist.
The seven essentials, each of which maps to a place in the layout:

1. **Time** — when the experiment ran (date range, and per-run timestamps). Put
   in README + environment. Convert relative dates ("yesterday") to absolute.

2. **Purpose + the spec** — *what* the experiment set out to prove and its
   success criteria. Copy the originating spec markdown into the packup (or link
   it by exact path if too large / still evolving). README states the goal in
   2-3 sentences; the spec gives the full definition.

3. **Environment** — the single biggest reproducibility trap. Capture both:
   - **Hardware**: exact machines/hostnames, GPU model + count, CPU, RAM, and —
     critically for this repo — **driver versions and the RDMA fabric**
     (InfiniBand / ionic / RoCE config + driver versions, active rails). Use
     `scripts/collect_env.sh` (bundled) to snapshot this on each node.
   - **Software**: docker image tag **and** the base-image `sha256` digest,
     the Dockerfile used, and the **git branch + commit SHA** of the repo the
     run used. Pin everything — a floating `:nightly` tag is not reproducible.

4. **Exact reproduction commands / scripts** — the heart of the kit. Copy the
   real scripts that ran (server launch, router, sweep, parse) **verbatim** into
   `scripts/`. `REPRODUCE.md` then lists the ordered, copy-pasteable command
   sequence that invokes them. Someone should be able to run it top-to-bottom.
   - Include the **debug/fix outcomes**: if the experiment only worked after a
     fix, the fix is *part of* the reproduction. Put patches in `patches/` each
     with a **what / why / how / context** note (what changed, why it was
     needed, how it was applied, what symptom it cured). A patch with no "why"
     rots into a mystery.

5. **Dependency absolute paths / uncommitted files** — list every external
   input the run depends on that is NOT in the repo: model weights on shared
   storage (absolute path, e.g. `/mnt/vast/...`), datasets, uncommitted local
   config, injected host libraries (e.g. `libionic`). If a path is a shared
   mount, say which filesystem and roughly how to obtain/mount it.

6. **Required secrets** — enumerate *which* credentials the run needs (registry
   login, cluster SSH, API keys, S3/etcd creds) and *where they come from* —
   never the values themselves. E.g. "Docker registry: `inferaimage` creds from
   team vault", "cluster: SSH via configured ProxyJump". This lets a reproducer
   know what to arrange without leaking anything.

7. **Explanatory doc (notes)** — `notes.md`: the caveats, the gotchas, the
   traps, the wrong turns and error analysis, written as **what / why / how /
   context** so a reader learns *why* a step matters, not just that it exists.
   This is where the working-process narrative gets distilled into durable
   lessons. This is often the most re-read file in the whole packup.

### Phase 4 — Verify the kit is self-contained

Before declaring done, do a cold-read pass: pretend you know nothing about the
experiment and read only the packup. Can you reproduce it? Specifically check:

- `REPRODUCE.md` references only files inside the packup or clearly-marked
  external absolute paths (Phase 3 items 5-6) — no dangling "see the other file"
  pointers into the scratch workspace.
- Every script referenced actually exists in `scripts/`.
- Every version/SHA/path is concrete, not "the latest" or "the usual place".
- The success criteria from the spec are restated and the actual result is
  stated against them (did it pass? by how much?).

Then report to the user: the deliverable path, a one-paragraph summary of what's
inside, and any gaps you had to flag (missing env capture, logs you left out
pending their decision, files > 4 MB awaiting approval).

## Bundled resources

- `scripts/collect_env.sh` — snapshots hardware + software environment on a node
  (GPU/driver via `rocm-smi`/`nvidia-smi`, RDMA rails via `ibv_devices`/`ib_*`,
  docker image digest, git branch+SHA, CPU/RAM). Run it on each machine involved
  and drop the output into `environment.md`. Read the script header for usage.
- `references/deliverable_layout.md` — the full annotated directory layout plus
  ready-to-fill `README.md` and `REPRODUCE.md` templates.
- `references/checklist.md` — the authoritative field-by-field completeness
  checklist; walk it before Phase 4 to make sure nothing is missing.
