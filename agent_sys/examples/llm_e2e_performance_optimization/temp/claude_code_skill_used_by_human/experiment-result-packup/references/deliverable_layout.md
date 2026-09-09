# Deliverable layout — annotated + templates

The packup folder lives **beside the experiment it documents** and is named
`<experiment-name>.packup_<YYYYMMDD>/`. Below is the full structure with the
purpose of each file, followed by copy-paste templates for the two files that do
the heavy lifting: `README.md` and `REPRODUCE.md`.

## Annotated structure

```
<experiment-name>.packup_YYYYMMDD/
├── README.md          # Entry point. Read-first. What/why/result in a glance,
│                      #   plus a map of the folder. Keep it short — details go
│                      #   in the other files. Answers: "what is this, did it
│                      #   work, where do I go next?"
├── REPRODUCE.md       # The reproduction kit. Ordered, copy-pasteable commands
│                      #   from zero to result. This is the file a reproducer
│                      #   actually executes. Must be self-contained.
├── environment.md     # Hardware + software environment. Pinned versions, image
│                      #   digests, git SHA, driver + RDMA fabric. The #1 repro
│                      #   trap — be exhaustive here.
├── scripts/           # Every script needed to reproduce, copied VERBATIM from
│                      #   the run (server launch, router, sweep, parse, inject,
│                      #   container start). Do not paraphrase — copy the real
│                      #   thing so byte-level flags survive.
├── patches/           # Fixes/patches that were required for the run to work or
│                      #   to be correct. One note per patch: what/why/how/
│                      #   context. A .patch/.diff file + a short .md beside it.
├── results/           # The evidence: CSV/PNG/summary tables/JSON. The numbers
│                      #   that back the claim. Keep the machine-readable form,
│                      #   not just a screenshot.
├── notes.md           # Gotchas, wrong turns, error analysis. what/why/how/
│                      #   context. The durable lessons distilled from the
│                      #   working-process log.
└── logs/              # OPTIONAL. Only if the user opts in. Often large — offer
                       #   to trim or gzip. Raw logs are sometimes the crux of a
                       #   debug story, sometimes noise; let the user decide.
```

Not every experiment needs every folder. A pure perf bench may have no
`patches/`; a debug session may have a fat `patches/` and thin `results/`. Omit
empty folders rather than shipping hollow scaffolding — but never omit
`README.md`, `REPRODUCE.md`, or `environment.md`.

## README.md template

```markdown
# <Experiment name>

**Ran:** <YYYY-MM-DD> (to <YYYY-MM-DD> if multi-day)
**Author:** <who>
**Status:** <PASS / PARTIAL / FAILED> against the success criteria below.

## Goal

<2-3 sentences: what this experiment set out to prove and why. Reference the
originating spec.>

**Spec:** `<path or ./spec/....md>`
**Success criteria:** <the concrete bar, e.g. "throughput > single-node AND
TTFT ≥ 40% of single-node" — copied from the spec so it's unmissable.>

## Result

<The headline number(s), stated against the criteria. Did it pass? By how much?
One small table beats a paragraph.>

| Config | Metric | Target | Actual | Verdict |
|--------|--------|--------|--------|---------|
| ...    | ...    | ...    | ...    | ✅ / ❌  |

## How to reproduce

See `REPRODUCE.md`. TL;DR: <one line, e.g. "start container on both nodes, launch
1P1D, run sweep">.

## Folder map

- `REPRODUCE.md` — step-by-step reproduction
- `environment.md` — exact HW/SW the numbers came from
- `scripts/` — the scripts that ran
- `patches/` — fixes required for the run (with rationale)
- `results/` — raw numbers (CSV/PNG)
- `notes.md` — gotchas & wrong turns worth knowing
```

## REPRODUCE.md template

```markdown
# Reproduction kit — <Experiment name>

Goal: reproduce <the result> from a clean machine with cluster access.
Estimated time: <e.g. ~45 min, of which ~30 min is CUDA-graph capture>.

## 0. Prerequisites (arrange before you start)

- **Machines:** <exact hostnames + roles, e.g. "chi2811 (prefill) ↔ chi2866
  (decode)", data-plane IPs>. How they're obtained: <slurm hold / etc>.
- **Secrets needed** (values NOT included — source them yourself):
  - <e.g. Docker registry `inferaimage` login — from team vault>
  - <e.g. cluster SSH — ProxyJump preconfigured in ~/.ssh/config>
- **External dependencies (absolute paths, not in this repo):**
  - Model: `<e.g. /mnt/vast/yihou/models/DeepSeek-V4-Pro-fixed>`
  - <injected host libs / datasets / uncommitted config>
- **Repo state:** git `<branch>` @ `<commit SHA>`.
- **Image:** `<tag>` (base digest `sha256:...` — see environment.md).

## 1. <First step — e.g. pull image + start container on both nodes>

    <exact commands, copy-pasteable, referencing ./scripts/...>

## 2. <Next step — e.g. launch prefill + decode servers>

    <exact commands>

## 3. <Run the benchmark / sweep>

    <exact commands>

## 4. <Parse + read the result>

    <exact commands; where the output CSV lands; what number to look at>

## Expected output

<What success looks like: the metric, the file it appears in, the value to expect.
Tie it back to the success criteria in README.md.>

## If it doesn't reproduce

See `notes.md` for known gotchas (GPU mem must return to baseline between runs,
CUDA-graph capture is slow, etc).
```
